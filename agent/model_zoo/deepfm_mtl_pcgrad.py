"""DeepFM multi-task with PCGrad gradient surgery (Yu et al. 2020, "Gradient
Surgery for Multi-Task Learning", arXiv:2001.06782) -- refines the lever
that's already this project's one real win, rather than trying yet another
architecture. Every architecture variant tried in this pass (DIN, BPR,
listwise, PDAOM) came back negative or tied; the organizers' own finding
("capacity/architecture isn't the bottleneck") has now been independently
reconfirmed enough times that another architecture bet is low expected
value. `deepfm_mtl_v1`'s multi-task setup is the one mechanism that
demonstrably works -- this asks whether ITS OWN training dynamics can be
improved, a different question than "try a new model."

`deepfm_mtl_uncertainty_v1` (already tried) addressed one specific
multi-task pathology: fixed loss *magnitude* weighting is arbitrary,
replaced with a learned per-task weight. It came back essentially tied
(noise_floor at 8 seeds). PCGrad addresses a DIFFERENT, genuinely separate
pathology: even with magnitudes perfectly balanced, two tasks' gradients
can point in conflicting DIRECTIONS on the shared parameters -- one task's
update step actively undoes progress the other task just made. Kendall/
Gal/Cipolla's uncertainty weighting (used by deepfm_mtl_uncertainty.py)
cannot fix this even in principle -- it only rescales each task's gradient
by a scalar, which changes magnitude, never direction. PCGrad is the first
lever in this project that actually touches gradient *direction*.

Simplified to a 2-task formulation (main `long_view` loss vs. combined
auxiliary loss, not full 5-way pairwise surgery across the 4 individual
aux heads): `deepfm_mtl.py`'s single `aux_heads` layer produces all 4
auxiliary logits from one shared weight matrix, so the 4 auxiliary
"tasks" aren't cleanly separable parameters the way main-vs-aux already
is -- disentangling them would mean redesigning the architecture, not
just the training step. The two-task version still captures PCGrad's core
mechanism (resolving conflict between the primary objective and
everything else) and is the more defensible, lower-risk implementation to
get right on the first attempt.

Algorithm (`mtl_step`, replacing the single `loss.backward()` every other
model in this Model Zoo uses): compute each task's gradient on the SHARED
parameters (the embedding table, linear term, and deep MLP trunk) via
`torch.autograd.grad` rather than `.backward()`, so neither call
accumulates into the other. If the two tasks' shared-parameter gradients
have negative cosine similarity (they conflict), project each onto the
other's normal plane before summing -- removing exactly the conflicting
component, keeping everything else. Each task's PRIVATE parameters (the
main-only output layer, the aux-only output layer) never conflict with
anything else by construction, so their gradients are computed and
applied normally, no surgery needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Net(nn.Module):
    """Identical architecture to deepfm_mtl.py's _Net -- PCGrad is the only
    variable this model tests, so the network itself must not differ."""
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
        self.deep_out = nn.Linear(sizes[-1], 1)          # main-only, private
        self.aux_heads = nn.Linear(sizes[-1], 4)          # aux-only, private

    def shared_parameters(self) -> list[nn.Parameter]:
        """Only V (the embedding table) and the deep trunk actually appear
        in BOTH main_loss's and aux_loss's computational graph -- both
        losses read `h = self.deep(E.reshape(...))`, and `E = self.V(X)`
        feeds that reshape directly. `self.W` (the FM linear term) and
        `self.b` (bias) feed `main_logit` alone; `h` never touches them.
        Passing them to `torch.autograd.grad(aux_loss, ...)` throws
        ("not used in the graph") for exactly this reason -- caught by
        `test_deepfm_mtl_pcgrad_forward_backward_reduces_loss`, not
        theorized in advance. They're genuinely main-task-private, so
        `mtl_step` below groups them with `deep_out`, not with `V`/`deep`."""
        return list(self.V.parameters()) + list(self.deep.parameters())

    def main_private_parameters(self) -> list[nn.Parameter]:
        """W and b feed main_logit only (see shared_parameters' docstring);
        deep_out is main_logit's own output layer. All three are
        main-task-private -- no conflict resolution needed, since aux_loss
        has zero graph dependency on any of them."""
        return list(self.W.parameters()) + [self.b] + list(self.deep_out.parameters())

    def forward(self, X: torch.Tensor):
        E = self.V(X)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, Fd, K = E.shape
        h = self.deep(E.reshape(B, Fd * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)
        return main_logit, aux_logits, h


def _pcgrad_resolve(g_main: list[torch.Tensor], g_aux: list[torch.Tensor],
                     rng: np.random.Generator) -> list[torch.Tensor]:
    """Core PCGrad step for exactly two tasks over a shared parameter list.
    Each g_* is a list of per-parameter gradient tensors (same shapes as
    `shared_parameters()`); flattened into one vector per task to compute a
    single cosine/projection across the whole shared parameter set (PCGrad
    treats "the shared parameters" as one space, not per-tensor), then
    un-flattened back to per-parameter shapes before returning the summed,
    de-conflicted gradient.
    """
    shapes = [g.shape for g in g_main]
    flat_main = torch.cat([g.reshape(-1) for g in g_main])
    flat_aux = torch.cat([g.reshape(-1) for g in g_aux])

    # Randomize which task's gradient gets projected first -- PCGrad's own
    # prescription for >2 tasks; with exactly 2 tasks this just means each
    # step has a 50/50 chance of resolving from main's or aux's perspective
    # first, which matters once one of the two projections below runs.
    order = ["main", "aux"] if rng.random() < 0.5 else ["aux", "main"]
    vecs = {"main": flat_main.clone(), "aux": flat_aux.clone()}
    others = {"main": flat_aux, "aux": flat_main}

    for task in order:
        v, o = vecs[task], others[task]
        dot = torch.dot(v, o)
        if dot < 0:  # conflicting gradients -- project v onto the plane orthogonal to o
            vecs[task] = v - (dot / (o.norm() ** 2 + 1e-12)) * o

    combined_flat = vecs["main"] + vecs["aux"]
    out, offset = [], 0
    for shape in shapes:
        n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
        out.append(combined_flat[offset:offset + n].reshape(shape))
        offset += n
    return out


class DeepFM_MTL_PCGrad:
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
        self._rng = np.random.default_rng(seed)

    def mtl_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()

        main_logit, aux_logits, _h = self.net(Xt)
        main_loss = self.main_loss_fn(main_logit, yt)
        aux_loss = self.aux_loss_fn(aux_logits, auxt) * self.aux_weight

        shared = self.net.shared_parameters()
        g_main_shared = torch.autograd.grad(main_loss, shared, retain_graph=True, allow_unused=False)
        g_aux_shared = torch.autograd.grad(aux_loss, shared, retain_graph=True, allow_unused=False)
        g_shared = _pcgrad_resolve(list(g_main_shared), list(g_aux_shared), self._rng)

        main_private = self.net.main_private_parameters()
        g_main_private = torch.autograd.grad(main_loss, main_private)
        g_aux_private = torch.autograd.grad(aux_loss, self.net.aux_heads.parameters())

        self.opt.zero_grad()
        for p, g in zip(shared, g_shared):
            p.grad = g.clone()
        for p, g in zip(main_private, g_main_private):
            p.grad = g.clone()
        for p, g in zip(self.net.aux_heads.parameters(), g_aux_private):
            p.grad = g.clone()
        self.opt.step()

        return float(main_loss.item())  # loss curve tracks the main task only, same convention as deepfm_mtl.py

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                main_logit, _, _ = self.net(Xt)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_PCGrad:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "seed"}
    return DeepFM_MTL_PCGrad(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
