"""Graph-propagated embedding initialization (LightGCN, He et al. 2020) as
a pre-training step for `deepfm_mtl_v1`'s user_id/video_id embedding rows
-- a genuinely different MECHANISM from everything else tried this
project. Every prior lever changed the loss function, architecture,
training signal, or combined already-trained models; all of them still
treat embeddings as a pure lookup table learned from scratch via
gradient descent alone. This instead injects collaborative-filtering
STRUCTURE before training even starts: propagate signal through the
user-video interaction graph (multi-hop "users who engaged with what I
engaged with also engaged with X") to get an initialization that already
encodes neighborhood structure, then fine-tune normally with the exact
same objective/hyperparameters as `deepfm_mtl_v1`. Isolates embedding
INITIALIZATION as the only variable -- architecture, loss, training loop
all unchanged, reusing `agent/model_zoo/deepfm_mtl.py` completely as-is
(no new model file: this script builds a normal DeepFM_MTL, then
overwrites its embedding table's user_id/video_id rows before training).

Graph construction, leakage-safety:
  - Nodes: every user_id and video_id value that appears in `Xtr` (the
    TRAIN split's own already-encoded, already-offset indices from the
    organizer's own `data.encode()` -- reused directly, not re-derived,
    so this can never drift from the real embedding-table index space).
  - Edges: TRAIN rows where `is_click == 1` (agent/features.py's
    `load_aux_labels`, already used elsewhere in this project, e.g.
    `deepfm_mtl_click.py`) -- a genuine "user engaged with this video"
    signal, denser than `long_view` (~46% vs ~31-34%) and, deliberately,
    NOT the exact target label being predicted: using is_click as the
    graph's edge criterion keeps the graph a distinct auxiliary signal
    (same spirit as the 4 MTL aux heads) rather than circularly building
    the propagation structure directly out of the scored label. Train
    split only -- valid/test rows never touch graph construction, same
    discipline as every train-only-aggregate feature in this project.

LightGCN propagation (numpy + scipy.sparse, no torch needed for this
part -- a normalized-adjacency matrix-vector propagation is exactly the
kind of thing worth hand-deriving directly, matching this project's own
numpy-first philosophy for anything that doesn't need autograd):
symmetric normalized adjacency A_hat = D^-1/2 (A + A^T) D^-1/2 over the
touched user/video nodes only; `n_layers` propagation steps
E^(l+1) = A_hat @ E^(l) from a small random E^(0); final embedding is the
mean over all layers (LightGCN's own "layer combination" -- averaging
smooths out over-smoothing from any single deep layer). Rescaled to match
`deepfm_mtl.py`'s own init std (repeated normalized averaging shrinks
variance, a real, checked effect -- rescaling isolates GRAPH STRUCTURE as
the difference from random init, not scale) before overwriting the
model's embedding rows for exactly the touched user_id/video_id indices;
every other row (author_id/tab/dur_bucket, and any user/video with zero
is_click=1 train rows) keeps `deepfm_mtl.py`'s own random init untouched.

Usage: python tools/check_gnn_init.py [--data_dir ...] [--n_layers 2] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.features import AUX_LABEL_FIELDS, load_aux_labels  # noqa: E402
from agent.model_zoo.deepfm_mtl import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

HP = {"k": 16, "hidden": [128, 64], "lr": 0.001, "l2": 1e-4, "aux_weight": 0.2}
EPOCHS, PATIENCE, BATCH = 20, 5, 8192


def build_lightgcn_init(user_idx: np.ndarray, video_idx: np.ndarray, k: int,
                         n_layers: int, target_std: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (propagated_embeddings, touched_node_indices) -- the caller
    overwrites only `touched_node_indices` rows of the real embedding
    table with the corresponding rows of `propagated_embeddings` (indexed
    by the SAME global node id, not the compact local id used internally
    for the sparse matrix)."""
    nodes = np.union1d(user_idx, video_idx)
    node_to_local = {int(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    rows = np.fromiter((node_to_local[int(u)] for u in user_idx), dtype=np.int64, count=len(user_idx))
    cols = np.fromiter((node_to_local[int(v)] for v in video_idx), dtype=np.int64, count=len(video_idx))
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A = A + A.T
    A.data[:] = 1.0  # binarize -- a repeated click between the same user/video collapses to one edge

    deg = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    rng = np.random.default_rng(seed)
    E = rng.normal(0.0, 0.01, size=(n_nodes, k)).astype(np.float32)
    layers = [E]
    for _ in range(n_layers):
        E = A_hat @ E
        layers.append(E)
    E_final = np.mean(layers, axis=0)

    cur_std = float(E_final.std())
    if cur_std > 1e-9:
        E_final = E_final * (target_std / cur_std)

    return E_final, nodes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    aux = load_aux_labels(args.data_dir, fields=list(AUX_LABEL_FIELDS))
    aux_tr = aux["train"]

    click = load_aux_labels(args.data_dir, fields=["is_click"])["train"][:, 0]
    click_rows = click > 0.5
    print(f"{click_rows.sum():,} of {len(click_rows):,} train rows are is_click=1 -- these become graph edges")

    user_idx_all = Xtr[click_rows, 0]
    video_idx_all = Xtr[click_rows, 1]
    print(f"Building LightGCN-style propagated init ({args.n_layers} layers) over "
          f"{len(np.union1d(user_idx_all, video_idx_all)):,} touched user/video nodes...")
    propagated, touched_nodes = build_lightgcn_init(user_idx_all, video_idx_all, HP["k"],
                                                      args.n_layers, target_std=0.01, seed=args.seed)

    model = build(dim=dim, n_fields=Xtr.shape[1], seed=args.seed, **HP)
    with torch.no_grad():
        for local_i, node in enumerate(touched_nodes):
            model.net.V.weight[int(node)] = torch.from_numpy(propagated[local_i])
    print(f"Overwrote {len(touched_nodes):,} of {dim:,} embedding rows with graph-propagated init "
          f"(the rest -- author_id/tab/dur_bucket, and any never-clicked user/video -- keep deepfm_mtl.py's "
          f"own random init).")

    rng = np.random.default_rng(args.seed)
    best_primary, best_epoch, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            model.mtl_step(Xtr[b], ytr[b], aux_tr[b])
        va = score(uva, yva, model.predict(Xva))
        print(f"  epoch {ep:2d} | valid primary {va.primary:.4f}")
        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, best_state, bad = va.primary, ep, model.get_state(), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.set_state(best_state)
    valid_final = score(uva, yva, model.predict(Xva))
    test_final = score(ute, yte, model.predict(Xte))
    print(f"\nBest epoch: {best_epoch}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_mtl_v1 (3-seed verified, for comparison): valid primary=0.6046 +/- 0.0003, "
          f"single-seed-0 primary=0.6049")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
