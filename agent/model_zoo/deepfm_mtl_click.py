"""Multi-Task DeepFM + `is_click` as a 5th auxiliary head -- a reasoned
candidate, not another blind architecture bet, built after checking real
evidence first (per explicit direction: find what would actually work
before implementing it).

Why this one, not another architecture variant: this project's own
evidence at this point is unusually clear. Every architecture bet in P2
(DIN attention, BPR loss, listwise loss, PDAOM hard-mining) came back
negative or tied, independently reconfirming the organizers' own finding
that capacity/architecture isn't this benchmark's bottleneck. The ONE
thing that worked, `deepfm_mtl_v1`, didn't change the architecture at all
-- it added a genuinely NEW TRAINING SIGNAL (auxiliary supervision on
is_like/is_follow/is_comment/is_forward). Both attempts to refine THAT
mechanism further (uncertainty-weighting the existing 4 signals'
magnitudes, PCGrad on their gradient directions) failed to improve on the
original recipe. The remaining lever this reasoning points to isn't a new
mechanism or a refinement of the existing one -- it's MORE of what already
worked: is there another logged signal, not yet used anywhere in this
project, dense/informative enough to plausibly help the way the first 4
auxiliary signals did?

`is_click` qualifies on a real, checked basis, not a guess: sampled
directly from the raw log before writing any of this code, it has a
~45.9% positive rate (500K-row sample) vs. `AUX_LABEL_FIELDS`'s own
existing four signals at 0.1-1.8% each. Every auxiliary signal
`deepfm_mtl_v1` already uses is a RARE positive-engagement action;
`is_click` is a dense, near-balanced signal, qualitatively different in
character, not just a 5th flavor of the same sparse-engagement idea.
Two other candidates from the same log (`is_hate`, `is_profile_enter`)
were checked and set aside for this round: `is_hate` is even rarer than
the existing four (~0.04%, too sparse to plausibly move gradients
meaningfully); `is_profile_enter` (~2.5%) is a reasonable second
candidate but adding it alongside `is_click` in the same experiment would
confound which one drove any effect -- isolated to a single variable,
same discipline as every other candidate in this project.

`organizer's video_features_statistic_pure.csv` was also investigated as
a candidate feature source and explicitly DECLINED: verified via the
KuaiRand paper's own description (not assumed) that it's averaged over
the full one-month collection window (2022-04-08 to 2022-05-08),
overlapping both the valid and test date ranges -- using it as an input
feature would leak future engagement into training, exactly the kind of
subtle mistake this project's leakage discipline exists to catch before
it happens, not after. `user_features_pure.csv` (confirmed via the same
source to be a static, time-independent profile snapshot) is safe but
untested here -- a real, disclosed next candidate, not chased in this
same pass to keep this experiment isolated to one variable.

Architecture: identical to `deepfm_mtl.py` except `aux_heads` outputs 5
logits instead of 4 (adds `is_click`). `aux_fields` (a new class-level
declaration `agent/experiment.py`'s `is_mtl` branch reads via
`getattr(model, "aux_fields", ...)`) points it at
`agent/features.py`'s `AUX_LABEL_FIELDS_WITH_CLICK` instead of the
default `AUX_LABEL_FIELDS` -- `deepfm_mtl.py` and every other MTL variant
built so far declare no such attribute and keep using the original 4,
unaffected.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..features import AUX_LABEL_FIELDS_WITH_CLICK

N_AUX_TASKS = 5  # is_like, is_follow, is_comment, is_forward, is_click -- AUX_LABEL_FIELDS_WITH_CLICK


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
        B, Fd, K = E.shape
        h = self.deep(E.reshape(B, Fd * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)              # (B, 5)
        return main_logit, aux_logits


class DeepFM_MTL_Click:
    aux_fields = AUX_LABEL_FIELDS_WITH_CLICK  # read by agent/experiment.py's is_mtl branch

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
        return float(main_loss.item())  # loss curve tracks the main task only, same convention as deepfm_mtl.py

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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_Click:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "seed"}
    return DeepFM_MTL_Click(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
