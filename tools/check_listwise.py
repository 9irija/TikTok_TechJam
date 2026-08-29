"""Standalone check for agent/model_zoo/deepfm_listwise.py (DeepFM trained
with a per-user listwise softmax ranking loss instead of pointwise BCE) --
the untested half of the starter kit's own "pairwise BPR or listwise
per-user softmax" top guess for the loss-function lever (fm_bpr/deepfm_bpr
already tested the pairwise half, twice, both plateaued below baseline).

Batching is structurally different from every other model here: each
batch is a set of USERS (not rows). Every impression belonging to a
sampled user is gathered and padded to `--max_len` (masked out of the
softmax); users with fewer than 1 positive AND 1 negative impression in
train are excluded (zero listwise signal either way, same filtering
fm_bpr.py already uses). Users with more than `--max_len` impressions get
a fresh random subsample of that size EVERY epoch (not a one-time
truncation) -- over many epochs this still exposes the model to all of an
active user's history, just not all at once in a single batch.

Same base hyperparameters as deepfm_regularized (k=16, hidden=[128,64],
lr=0.001, l2=1e-4, epochs=20, patience=5) so the loss function is the only
variable versus that parent -- same isolation discipline every other
Model Zoo comparison in this project follows.

Usage: python tools/check_listwise.py [--max_len 64] [--batch_users 256] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.model_zoo.deepfm_listwise import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402


def build_user_groups(users: list[str], y: np.ndarray) -> dict[str, np.ndarray]:
    """user_id -> array of row indices, restricted to users with at least
    one positive AND one negative impression (same filtering precedent as
    agent/model_zoo/fm_bpr.py's build_user_pos_neg_index)."""
    by_user: dict[str, list[int]] = collections.defaultdict(list)
    for i, u in enumerate(users):
        by_user[u].append(i)
    out = {}
    for u, idxs in by_user.items():
        idxs = np.array(idxs)
        yu = y[idxs]
        if yu.max() > 0 and yu.min() < 1:  # has both a positive and a negative row
            out[u] = idxs
    return out


def make_padded_batch(user_ids: list[str], user_groups: dict[str, np.ndarray],
                       X: np.ndarray, y: np.ndarray, max_len: int, rng: np.random.Generator):
    B, Fd = len(user_ids), X.shape[1]
    Xb = np.zeros((B, max_len, Fd), dtype=np.int32)
    yb = np.zeros((B, max_len), dtype=np.float32)
    mask = np.zeros((B, max_len), dtype=bool)
    for b, u in enumerate(user_ids):
        idxs = user_groups[u]
        if len(idxs) > max_len:
            idxs = rng.choice(idxs, size=max_len, replace=False)  # fresh subsample every call
        n = len(idxs)
        Xb[b, :n] = X[idxs]
        yb[b, :n] = y[idxs]
        mask[b, :n] = True
    return Xb, yb, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_users", type=int, default=256)
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
          f"impression (listwise-eligible), built in {time.process_time() - t0:.1f}s")

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=args.lr, l2=args.l2, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t1 = time.process_time()
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(eligible_users)
        losses = []
        for i in range(0, len(order), args.batch_users):
            batch_ids = order[i:i + args.batch_users]
            Xb, yb, mask = make_padded_batch(list(batch_ids), user_groups, Xtr, ytr, args.max_len, rng)
            losses.append(model.listwise_step(Xb, yb, mask))
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
    print(f"deepfm_bpr_v1_regularized (pairwise BPR, for comparison): valid primary=0.5980")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
