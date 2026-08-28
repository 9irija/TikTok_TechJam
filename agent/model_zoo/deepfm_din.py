"""DeepFM + DIN-style attention over user history (P2, the next lever after
multi-task learning -- the starter kit README's own #2-ranked untested
headroom item, and the last genuinely different, structurally untested one
after this pass's other four levers all came back negative).

Standalone-scored first (see `tools/check_sequence_model.py`), same
methodology already used for LightGBM this pass: verify the real signal
before investing in full `agent/experiment.py` pipeline integration -- not
built into the Model Zoo registry / `run_experiment`'s training loop yet.

Architecture: the same DeepFM backbone (shared 5-field embedding table ->
FM linear + 2nd-order interaction, deep MLP over concatenated field
embeddings) as `agent/model_zoo/deepfm.py`/`deepfm_mtl.py`, plus a DIN
attention block (Zhou et al. 2018, "Deep Interest Network for
Click-Through Rate Prediction") reading a DEDICATED, separate video-only
embedding table (`agent/sequences.py`'s own vocabulary -- deliberately not
sharing the 5-field table's internal offset scheme, to avoid coupling this
newer, higher-risk module to that already-validated internal layout). The
candidate video's own embedding is the attention query; the row's
historical video embeddings are the keys/values; scaled dot-product scores
measure "how relevant is each past video to this candidate," with
padding/UNK positions masked out of the softmax so a brand-new user (no
real history) safely falls back to attending over the shared PAD
embedding, never NaN.

Parent in the Research Map (if/when this gets promoted to a real node):
`deepfm_regularized`, NOT `deepfm_mtl_v1` -- isolating the sequence-
modeling effect as the only variable, same discipline `deepfm_mtl_v1`
itself followed against `deepfm_regularized`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Net(nn.Module):
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int, video_vocab_size: int):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))

        self.Vid = nn.Embedding(video_vocab_size, k)  # dedicated, shared between query and history keys
        nn.init.normal_(self.Vid.weight, 0.0, 0.01)
        self.attn_scale = k ** 0.5

        sizes = [n_fields * k + k] + list(hidden)  # +k for the attention summary vector
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(sizes[-1], 1)

    def _attend(self, cur_video: torch.Tensor, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        query = self.Vid(cur_video)                       # (B, k)
        keys = self.Vid(seq)                               # (B, L, k)
        logits = (query.unsqueeze(1) * keys).sum(-1) / self.attn_scale  # (B, L)
        mask = seq != pad_idx                               # (B, L) -- True where a real history item exists
        logits = logits.masked_fill(~mask, -1e9)  # not -inf: an all-padding row (new user) still
        # softmaxes to a stable, uniform distribution over the (also learned) PAD embedding, not NaN.
        weights = F.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * keys).sum(1)        # (B, k)

    def forward(self, X: torch.Tensor, seq: torch.Tensor, cur_video: torch.Tensor, pad_idx: int):
        E = self.V(X)                                       # (B, F, k)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, Fd, K = E.shape
        attn_out = self._attend(cur_video, seq, pad_idx)    # (B, k)
        concat = torch.cat([E.reshape(B, Fd * K), attn_out], dim=1)
        h = self.deep(concat)
        return self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)


class DeepFM_DIN:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 video_vocab_size: int = 1, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.pad_idx = video_vocab_size - 1  # sequences.py's UNK/PAD slot is always the last index
        self.net = _Net(dim, k, hidden, n_fields, video_vocab_size)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def seq_step(self, X: np.ndarray, y: np.ndarray, seq: np.ndarray, cur_video: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        seqt = torch.from_numpy(seq).long()
        curt = torch.from_numpy(cur_video).long()

        self.opt.zero_grad()
        logit = self.net(Xt, seqt, curt, self.pad_idx)
        loss = self.loss_fn(logit, yt)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def predict_seq(self, X: np.ndarray, seq: np.ndarray, cur_video: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                seqt = torch.from_numpy(seq[i:i + bs]).long()
                curt = torch.from_numpy(cur_video[i:i + bs]).long()
                logit = self.net(Xt, seqt, curt, self.pad_idx)
                out.append(logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_DIN:
    keys = {"k", "hidden", "lr", "l2", "video_vocab_size", "seed"}
    return DeepFM_DIN(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
