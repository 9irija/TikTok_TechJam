"""Real-data check for `agent/model_zoo/deepfm_mtl_gnn_feature.py` -- the
direct, diagnosis-driven follow-up to `tools/check_gnn_init.py`
(P2 Sec.27). That check found graph-propagated embedding INITIALIZATION
made no measurable difference, with a specific diagnosis: gradient descent
moves the trainable embeddings past the initial condition within ~10-12
epochs, so of course an initialization-only intervention doesn't survive.
This tests that diagnosis directly by making the graph signal a FROZEN
input feature instead -- available to the model at every epoch, not just
epoch 0. Reuses `tools/check_gnn_init.py`'s already-correctness-tested
`build_lightgcn_init()` directly (one owner for the graph-construction
math, not a second potentially-drifting copy).

Usage: python tools/check_gnn_feature.py [--data_dir ...] [--n_layers 2] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.features import AUX_LABEL_FIELDS, load_aux_labels  # noqa: E402
from agent.model_zoo.deepfm_mtl_gnn_feature import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools"))
from check_gnn_init import build_lightgcn_init  # noqa: E402

HP = {"k": 16, "hidden": [128, 64], "lr": 0.001, "l2": 1e-4, "aux_weight": 0.2}
EPOCHS, PATIENCE, BATCH = 20, 5, 8192


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
    user_idx_all = Xtr[click_rows, 0]
    video_idx_all = Xtr[click_rows, 1]
    print(f"Building LightGCN-style graph feature ({args.n_layers} layers) over "
          f"{len(np.union1d(user_idx_all, video_idx_all)):,} touched user/video nodes "
          f"(from {click_rows.sum():,} is_click=1 train rows)...")
    propagated, touched_nodes = build_lightgcn_init(user_idx_all, video_idx_all, HP["k"],
                                                      args.n_layers, target_std=0.01, seed=args.seed)
    graph_embed = np.zeros((dim, HP["k"]), dtype=np.float32)
    graph_embed[touched_nodes] = propagated
    print(f"Frozen graph_embed table: {len(touched_nodes):,} of {dim:,} rows populated, "
          f"the rest (author_id/tab/dur_bucket, never-clicked user/video) stay zero.")

    model = build(dim=dim, graph_embed=graph_embed, n_fields=Xtr.shape[1], seed=args.seed, **HP)
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
    print(f"\nBest epoch: {best_epoch}, trainable params: {model.num_params()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_mtl_v1 (3-seed verified, for comparison): valid primary=0.6046 +/- 0.0003, "
          f"single-seed-0 primary=0.6049")
    print(f"deepfm_mtl_gnn_init_v1 (trainable-init version, for comparison): "
          f"valid primary=0.6045 +/- 0.0003 (3-seed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
