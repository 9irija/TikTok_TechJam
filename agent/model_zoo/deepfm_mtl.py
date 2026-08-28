"""Multi-Task DeepFM -- P2 Extended Model Zoo, first genuine PyTorch use in
this project.

Same architecture as `agent/model_zoo/deepfm.py` (shared field-embedding
table -> FM linear + 2nd-order interaction, plus a parallel deep MLP over
the concatenated embeddings, logits summed before the sigmoid), but with
four small auxiliary sigmoid heads reading the same pooled embedding
representation, trained jointly on TikTok's other logged engagement signals
(`is_like`, `is_follow`, `is_comment`, `is_forward` -- see
`agent/features.py`'s `AUX_LABEL_FIELDS`/`load_aux_labels`). This is a
shared-bottom multi-task setup, the same family as the ESMM/MMoE line of
public recsys work this project is allowed to draw on (problem statement:
"Any papers, public solutions... Changes to any pipeline stage") -- the
concrete hypothesis is that `long_view` shares useful structure with
`is_like`/`is_follow`/etc. (all downstream of "did this user actually enjoy
this video"), so gradients from the denser auxiliary signals can regularize
the shared embedding table even though only the main task's logit is ever
scored (GAUC/nDCG@5 read `long_view` alone -- the aux heads exist purely to
shape training, never touch evaluation).

Why torch here and nowhere else in the Model Zoo: FM/DeepFM/FM_BPR each have
one clean, well-understood backward pass that was worth hand-deriving to
keep the numpy-only philosophy the starter kit itself uses (see those
files' own docstrings). A 5-headed shared-bottom network's backward pass
(one shared trunk, five different loss gradients merging back into it) is
exactly the case hand-rolled backprop stops paying for itself -- autograd
is the correct tool, not a shortcut.

Same object-shaped interface as every other Model Zoo entry (`predict`,
`get_state`/`set_state`, `num_params`) so `agent/experiment.py`'s training
loop treats it identically to the numpy models, plus one new method,
`mtl_step`, following the exact precedent `fm_bpr.py`'s `bpr_step` already
set for "this model needs a different per-batch training signal than plain
`(X, y)`" (see the `is_bpr`/`is_mtl` branches in `agent/experiment.py`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_AUX_TASKS = 4  # is_like, is_follow, is_comment, is_forward -- agent/features.py's AUX_LABEL_FIELDS


class _Net(nn.Module):
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))

        sizes = [n_fields * k] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(sizes[-1], 1)
        self.aux_heads = nn.Linear(sizes[-1], N_AUX_TASKS)  # shares the deep trunk's last hidden layer

    def forward(self, X: torch.Tensor):
        E = self.V(X)                              # (B, F, k)
        S = E.sum(1)                                # (B, k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, F, K = E.shape
        h = self.deep(E.reshape(B, F * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)              # (B, 4)
        return main_logit, aux_logits


class DeepFM_MTL:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 aux_weight: float = 0.2, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight = aux_weight
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.main_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

    def mtl_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()

        self.opt.zero_grad()
        main_logit, aux_logits = self.net(Xt)
        main_loss = self.main_loss_fn(main_logit, yt)
        aux_loss = self.aux_loss_fn(aux_logits, auxt)
        loss = main_loss + self.aux_weight * aux_loss
        loss.backward()
        self.opt.step()
        return float(main_loss.item())  # logged/plotted loss curve tracks the *main* task only,
        # matching every other model's loss_curve semantics -- the aux loss is a training-time
        # regularizer, not part of what "loss went down" is supposed to mean to a human reading it.

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                main_logit, _ = self.net(Xt)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "seed"}
    return DeepFM_MTL(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
