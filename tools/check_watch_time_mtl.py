"""Standalone check for agent/model_zoo/deepfm_mtl_watch.py (DeepFM_MTL plus
a continuous watch-time auxiliary head) -- same verify-before-integrating
discipline already used for LightGBM and DIN: get a real, honest number
before deciding whether full agent/experiment.py pipeline integration is
worth the engineering investment. Not wired into the Model Zoo registry /
Research Map yet.

Trains with EXACTLY deepfm_mtl_v1's hyperparameters (k=16, hidden=[128,64],
lr=0.001, l2=1e-6, aux_weight=0.2, epochs=20, patience=5) plus one new
watch_weight for the 5th head, so the added watch-time signal is the only
variable -- same isolation discipline every other Model Zoo comparison in
this project follows.

Usage: python tools/check_watch_time_mtl.py [--watch_weight 0.2] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.features import load_aux_labels, load_watch_ratio  # noqa: E402
from agent.model_zoo.deepfm_mtl_watch import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--watch_weight", type=float, default=0.2)
    ap.add_argument("--aux_weight", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    aux = load_aux_labels(args.data_dir)
    watch = load_watch_ratio(args.data_dir)
    aux_tr, watch_tr = aux["train"], watch["train"]
    assert len(aux_tr) == len(ytr) and len(watch_tr) == len(ytr), \
        "aux/watch arrays must be row-aligned with the plain 5-field encoding"

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=0.001, l2=1e-6,
                  aux_weight=args.aux_weight, watch_weight=args.watch_weight, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    bs = 8192
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.process_time()
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [
            model.mtl_watch_step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]],
                                  aux_tr[idx[i:i + bs]], watch_tr[idx[i:i + bs]])
            for i in range(0, len(idx), bs)
        ]
        mean_loss = float(np.mean(losses))
        va = score(uva, yva, model.predict(Xva))
        print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va.primary:.4f}")
        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, bad = va.primary, ep, 0
            best_state = model.get_state()
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.set_state(best_state)
    valid_final = score(uva, yva, model.predict(Xva))
    test_final = score(ute, yte, model.predict(Xte))
    wall = time.process_time() - t0

    print(f"\nBest epoch: {best_epoch}, wall_time={wall:.1f}s, num_params={model.num_params()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_mtl_v1 (parent, for comparison): valid primary=0.6046, test primary=0.5974")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
