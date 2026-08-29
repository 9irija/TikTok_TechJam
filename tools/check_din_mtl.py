"""Standalone check for agent/model_zoo/deepfm_din_mtl.py (DIN attention +
Multi-Task heads, combined) -- same verify-before-integrating discipline
already used for LightGBM, DIN alone, and the watch-time head.

Trains with `deepfm_din_v1`'s best setting (seq_len=20) and
`deepfm_mtl_v1`'s hyperparameters (k=16, hidden=[128,64], lr=0.001,
l2=1e-6, aux_weight=0.2, epochs=20, patience=5) -- both proven settings,
so the only variable is combining the two mechanisms into one model.

Usage: python tools/check_din_mtl.py [--seq_len 20] [--seed 0]
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
from agent.model_zoo.deepfm_din_mtl import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR  # noqa: E402
from agent.sequences import encode_with_history  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--aux_weight", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.process_time()
    print(f"Building sequences (seq_len={args.seq_len})...")
    enc, dim, seqs, cur_video, video_vocab_size = encode_with_history(args.data_dir, seq_len=args.seq_len)
    print(f"  video_vocab_size={video_vocab_size}, built in {time.process_time() - t0:.1f}s")

    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    seq_tr, seq_va, seq_te = seqs["train"], seqs["valid"], seqs["test"]
    cv_tr, cv_va, cv_te = cur_video["train"], cur_video["valid"], cur_video["test"]

    aux = load_aux_labels(args.data_dir)
    aux_tr = aux["train"]
    assert len(aux_tr) == len(ytr), "aux array must be row-aligned with the sequence encoding"

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=args.lr, l2=1e-6,
                  aux_weight=args.aux_weight, video_vocab_size=video_vocab_size, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    bs = 8192
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t1 = time.process_time()
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [
            model.seq_mtl_step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]], aux_tr[idx[i:i + bs]],
                                seq_tr[idx[i:i + bs]], cv_tr[idx[i:i + bs]])
            for i in range(0, len(idx), bs)
        ]
        mean_loss = float(np.mean(losses))
        va = score(uva, yva, model.predict_seq(Xva, seq_va, cv_va))
        print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va.primary:.4f}")
        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, bad = va.primary, ep, 0
            best_state = model.get_state()
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.set_state(best_state)
    valid_scores = model.predict_seq(Xva, seq_va, cv_va)
    test_scores = model.predict_seq(Xte, seq_te, cv_te)
    valid_final = score(uva, yva, valid_scores)
    test_final = score(ute, yte, test_scores)
    wall = time.process_time() - t1

    print(f"\nBest epoch: {best_epoch}, wall_time={wall:.1f}s, num_params={model.num_params()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_mtl_v1 (current best, for comparison): valid primary=0.6046, test primary=0.5974")
    print(f"deepfm_din_v1 (seq_len=20 alone, for comparison): valid primary=0.6036, test primary=0.5973")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
