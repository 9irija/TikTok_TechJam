"""Graph-propagated embeddings as a FROZEN auxiliary feature, not a
trainable initialization -- a direct, diagnosis-driven follow-up to
`tools/check_gnn_init.py` (P2 Sec.27). That check found initializing
`deepfm_mtl_v1`'s user_id/video_id embedding rows via LightGCN-style graph
propagation made no measurable difference (0.6045 +/- 0.0003 vs. the
parent's 0.6046 +/- 0.0003), and gave a specific, testable reason: with
only ~700K parameters and Adam's adaptive updates, ~10-12 epochs of
gradient descent is enough to move the trainable embeddings well past
whatever the initial condition encoded -- the graph signal doesn't
survive training, so of course it doesn't matter.

This tests that diagnosis directly: instead of writing the graph
embedding into the model's own TRAINABLE embedding table (where gradient
descent can and does erase it), keep it as a separate, FROZEN buffer
(`requires_grad=False`, never touched by the optimizer) and feed it to the
deep tower as an extra, persistent input alongside the normal learned
embeddings -- so if the graph structure is useful, the model has access
to it at every single epoch, not just epoch 0. If this ALSO comes back
null, that's much stronger evidence the graph signal itself isn't useful
here (not just poorly delivered) -- since two structurally different
delivery mechanisms would have failed the same way.

Same FM 2nd-order term as `deepfm_mtl.py` (reads only the trainable
embedding `V`, unchanged) -- only the deep tower's input is different:
`concat(V(X).flatten(), graph_embed[user_id], graph_embed[video_id])`
instead of just `V(X).flatten()`. `graph_embed` is precomputed OUTSIDE
this file (by whichever script builds it, e.g. `tools/check_gnn_feature.py`,
reusing `tools/check_gnn_init.py`'s already-correctness-tested propagation
math) and passed into `build()` as a plain `(dim, k)` numpy array; this
file only registers it as a frozen buffer, it doesn't compute it -- one
place owns the graph-construction logic (`tools/check_gnn_init.py`'s
`build_lightgcn_init`), not two potentially-drifting copies.

Not registered in `agent/model_zoo/registry.py`: needs a precomputed
graph-embedding array at construction time, which doesn't fit the
standard `build(dim, n_fields, **hyperparams)` signature every other
registry entry uses (same reason `deepfm_din.py`'s sequence features stay
a standalone check rather than a registry entry).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_AUX_TASKS = 4  # is_like, is_follow, is_comment, is_forward -- agent/features.py's AUX_LABEL_FIELDS


class _Net(nn.Module):
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int, graph_embed: np.ndarray):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))

        # Frozen -- registered as a buffer (moves with .to()/state_dict() like a parameter,
        # but never appears in .parameters(), so the optimizer never touches it).
        assert graph_embed.shape == (dim, k), f"graph_embed must be (dim={dim}, k={k}), got {graph_embed.shape}"
        self.register_buffer("graph_embed", torch.from_numpy(graph_embed.astype(np.float32)))

        deep_in = n_fields * k + 2 * k  # + frozen user graph embed, + frozen video graph embed
        sizes = [deep_in] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(sizes[-1], 1)
        self.aux_heads = nn.Linear(sizes[-1], N_AUX_TASKS)

    def forward(self, X: torch.Tensor):
        E = self.V(X)                              # (B, F, k) -- trainable
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, F, K = E.shape

        user_graph = self.graph_embed[X[:, 0]]      # (B, k) -- frozen, column 0 = user_id
        video_graph = self.graph_embed[X[:, 1]]     # (B, k) -- frozen, column 1 = video_id
        deep_in = torch.cat([E.reshape(B, F * K), user_graph, video_graph], dim=1)

        h = self.deep(deep_in)
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)
        return main_logit, aux_logits


class DeepFM_MTL_GNNFeature:
    def __init__(self, dim: int, graph_embed: np.ndarray, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-4,
                 aux_weight: float = 0.2, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight = aux_weight
        self.net = _Net(dim, k, hidden, n_fields, graph_embed)
        # Only trainable parameters go to the optimizer -- graph_embed is a buffer, never included.
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
        return float(main_loss.item())

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
        # Trainable only -- the frozen graph_embed buffer is real memory but not a
        # trainable parameter, matching what every other model's num_params() reports.
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))


def build(dim: int, graph_embed: np.ndarray, n_fields: int = 5, **hp) -> DeepFM_MTL_GNNFeature:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "seed"}
    return DeepFM_MTL_GNNFeature(dim, graph_embed, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
