"""DeepFM + DIN attention + Multi-Task auxiliary heads, combined -- P2,
the "combine two independently-partial results" move (same reasoning
`agent/model_zoo/deepfm_bpr.py` already used to combine P1's BPR loss
with DeepFM's architecture, per `docs/P2_FEATURES_AND_RESULTS.md` §8).

`deepfm_din_v1` alone (agent/model_zoo/deepfm_din.py) was a
`ranking_tradeoff` on top of `deepfm_regularized`: nDCG@5 improved
(+0.0005) but GAUC dropped (-0.0003). `deepfm_mtl_v1`
(agent/model_zoo/deepfm_mtl.py) alone was a clean `clear_improvement` on
the same parent. Neither is obviously redundant with the other -- DIN
changes what the model can SEE (attention over recent history, richer
input representation), MTL changes what the model is trained to PREDICT
(denser multi-signal gradient into the shared embeddings). Hypothesis:
MTL's regularizing effect might correct DIN's GAUC regression while
keeping DIN's nDCG@5 gain, since they act on different parts of the
architecture/loss rather than competing for the same capacity.

Architecture: `deepfm_din.py`'s exact attention block (dedicated
video-only embedding table, masked scaled dot-product attention over
recent history, output concatenated with the 5-field embeddings before
the deep MLP), PLUS `deepfm_mtl.py`'s 4 binary auxiliary sigmoid heads
(is_like/is_follow/is_comment/is_forward) reading the same post-attention
deep trunk.

Standalone check first (tools/check_din_mtl.py), same discipline as every
other new lever this pass -- not wired into the Model Zoo registry /
agent/experiment.py. Parent in the Research Map (if/when promoted):
`deepfm_regularized`, matching `deepfm_bpr_v1`'s own precedent for "this
combines two things as one joint change from the common ancestor," not an
incremental tweak to either standalone result individually.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_AUX_TASKS = 4  # is_like, is_follow, is_comment, is_forward -- same as deepfm_mtl.py


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
        self.aux_heads = nn.Linear(sizes[-1], N_AUX_TASKS)  # shares the post-attention deep trunk

    def _attend(self, cur_video: torch.Tensor, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        query = self.Vid(cur_video)                       # (B, k)
        keys = self.Vid(seq)                               # (B, L, k)
        logits = (query.unsqueeze(1) * keys).sum(-1) / self.attn_scale  # (B, L)
        mask = seq != pad_idx
        logits = logits.masked_fill(~mask, -1e9)
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
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)                       # (B, 4)
        return main_logit, aux_logits


class DeepFM_DIN_MTL:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 aux_weight: float = 0.2, video_vocab_size: int = 1, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight = aux_weight
        self.pad_idx = video_vocab_size - 1  # sequences.py's UNK/PAD slot is always the last index
        self.net = _Net(dim, k, hidden, n_fields, video_vocab_size)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.main_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

    def seq_mtl_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray,
                      seq: np.ndarray, cur_video: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()
        seqt = torch.from_numpy(seq).long()
        curt = torch.from_numpy(cur_video).long()

        self.opt.zero_grad()
        main_logit, aux_logits = self.net(Xt, seqt, curt, self.pad_idx)
        main_loss = self.main_loss_fn(main_logit, yt)
        aux_loss = self.aux_loss_fn(aux_logits, auxt)
        loss = main_loss + self.aux_weight * aux_loss
        loss.backward()
        self.opt.step()
        return float(main_loss.item())  # matches deepfm_mtl.py's convention -- the logged/plotted
        # loss curve tracks the main task only, the aux loss is a training-time regularizer.

    def predict_seq(self, X: np.ndarray, seq: np.ndarray, cur_video: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                seqt = torch.from_numpy(seq[i:i + bs]).long()
                curt = torch.from_numpy(cur_video[i:i + bs]).long()
                main_logit, _ = self.net(Xt, seqt, curt, self.pad_idx)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_DIN_MTL:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "video_vocab_size", "seed"}
    return DeepFM_DIN_MTL(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
