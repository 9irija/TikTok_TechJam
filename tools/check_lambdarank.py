"""Standalone check for agent/model_zoo/deepfm_lambdarank.py -- LambdaRank-
style, nDCG@5-weighted pairwise loss. The most structurally different loss
lever tried in this project: unlike BPR (every violated pair weighted
equally) or listwise softmax (no explicit connection to nDCG@5's
discounted, top-k shape), each pair's gradient here is scaled by the
actual |delta nDCG@5| that swapping it would cause -- a direct function of
the metric being scored, not just a proxy for "get more pairs right."

Reuses deepfm_listwise.py's per-user padded-batch infrastructure (same
architecture, same masking) -- ranks and IDCG are inherently per-user
quantities, so grouping by user is required regardless of pairwise vs.
listwise framing.

Same base hyperparameters as deepfm_regularized/deepfm_listwise (k=16,
hidden=[128,64], lr=0.001, l2=1e-4, epochs=20, patience=5) so the loss
function is the only variable versus that parent.

Usage: python tools/check_lambdarank.py [--max_len 64] [--batch_users 256]
                                         [--max_pairs_per_user 10] [--seed 0]
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
from agent.model_zoo.deepfm_lambdarank import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

# Reuse the exact same per-user batching helpers deepfm_listwise's own check already
# validated, rather than a second, potentially-divergent reimplementation.
sys.path.insert(0, str(REPO_ROOT / "tools"))
from check_listwise import build_user_groups, make_padded_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_users", type=int, default=256)
    ap.add_argument("--max_pairs_per_user", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    t0 = time.process_time()
    user_groups = build_user_groups(utr, ytr)
    eligible_users = list(user_groups.keys())
    print(f"{len(eligible_users)}/{len(set(utr))} train users have both a positive and negative "
          f"impression (LambdaRank-eligible), built in {time.process_time() - t0:.1f}s")

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=args.lr, l2=args.l2,
                  max_pairs_per_user=args.max_pairs_per_user, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t1 = time.process_time()
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(eligible_users)
        losses = []
        for i in range(0, len(order), args.batch_users):
            batch_ids = order[i:i + args.batch_users]
            Xb, yb, mask = make_padded_batch(list(batch_ids), user_groups, Xtr, ytr, args.max_len, rng)
            losses.append(model.lambdarank_step(Xb, yb, mask))
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
    wall = time.process_time() - t1

    print(f"\nBest epoch: {best_epoch}, wall_time={wall:.1f}s, num_params={model.num_params()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_regularized (parent, pointwise BCE, for comparison): valid primary=0.6035, test primary=0.5977")
    print(f"deepfm_listwise_v1 (per-user softmax, for comparison): valid primary=0.6033")
    print(f"deepfm_mtl_v1 (current best, for comparison): valid primary=0.6046")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
