"""Standalone check for agent/model_zoo/deepfm_mtl_uncertainty.py
(deepfm_mtl_v1's architecture, but with learned per-task uncertainty
weighting instead of one fixed aux_weight) -- same verify-before-
integrating discipline used for every other new lever this pass.

Same base hyperparameters as deepfm_mtl_v1 (k=16, hidden=[128,64],
lr=0.001, l2=1e-6, epochs=20, patience=5) minus aux_weight (there isn't
one anymore -- the model learns its own per-task weights), so the only
variable is the weighting scheme.

Usage: python tools/check_mtl_uncertainty.py [--seed 0]
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
from agent.features import load_aux_labels  # noqa: E402
from agent.model_zoo.deepfm_mtl_uncertainty import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--lr", type=float, default=0.001)
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
    aux_tr = aux["train"]
    assert len(aux_tr) == len(ytr), "aux array must be row-aligned with the plain 5-field encoding"

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=args.lr, l2=1e-6, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    bs = 8192
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.process_time()
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [model.mtl_step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]], aux_tr[idx[i:i + bs]])
                  for i in range(0, len(idx), bs)]
        mean_loss = float(np.mean(losses))
        va = score(uva, yva, model.predict(Xva))
        print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va.primary:.4f} | "
              f"weights={ {k: round(v, 3) for k, v in model.learned_weights().items()} }")
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
    print(f"Learned task weights at best epoch: {model.learned_weights()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_mtl_v1 (current best, fixed aux_weight=0.2, for comparison): valid primary=0.6046, test primary=0.5974")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
