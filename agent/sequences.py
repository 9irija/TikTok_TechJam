"""Per-user historical interaction sequences (P2, next lever after multi-task
learning) -- DIN-style (Zhou et al. 2018, "Deep Interest Network for
Click-Through Rate Prediction"): attend over a user's own past video
interactions, using the candidate video's embedding as the attention query,
so the model can weigh "how similar is this candidate to what this user
has actually watched" instead of treating every impression as independent.

Every model built before this one (FM, DeepFM, FM_BPR, DeepFM_MTL) scores
each row in total isolation -- none of them know what a user watched five
minutes ago. This is the starter kit README's own #2-ranked untested
headroom item (CLAUDE.md's "Unexplored headroom" list), and it's the last
genuinely different, structurally untested lever after this pass's other
four (features, LightGBM, hyperparameter search, ensembling) all came back
negative.

============================== LEAKAGE SAFETY ==============================
Read this before touching this file. Two separate, easy-to-get-wrong
questions:

1. "Can a row's OWN label leak into its own history?" No -- a row's
   history is built from OTHER rows (other impressions of the SAME user,
   at strictly earlier `time_ms`), never from the row's own play_time_ms/
   label. Same category of care as agent/features.py's train-only
   aggregates, just at row granularity instead of aggregate-statistic
   granularity.

2. "Is it OK for a VALID or TEST row's history to include TRAIN-split (or,
   for a test row, even earlier TEST-split) interactions?" Yes, and this
   is not a train/valid/test discipline violation -- it is *chronological*
   history, not split-based history. The splits are date ranges
   (train=4/8-4/21, valid=4/22-4/28, test=4/29-5/8); a user's real history
   at serving time for a valid-split impression genuinely includes
   whatever they did during the train-split date range, because that
   happened *earlier in time*. Restricting a row's history to only
   same-split rows would be an artificial, wrong simplification -- a real
   production recommender uses a user's full past interaction history
   regardless of which "split" that history happens to fall in under our
   evaluation date-partitioning. What's never allowed, and this module
   never does it: using any row with `time_ms` >= the target row's own
   `time_ms`, from ANY split, including the target's own. Enforced by
   `_recent_history` below with a strict `<` comparison, and tested by
   `test_sequences_no_future_leakage_and_vocab_consistent`.

The video-id VOCABULARY used to encode history items is built from the
TRAIN split only (same convention as every other field in this codebase)
-- a history item referencing a video never seen in training gets the same
UNK slot a same-situation *current-row* video_id would get. This is the
exact same vocab-building loop `agent/features.py`'s `encode_extended` uses
for the video_id field, duplicated here (not imported/refactored out of
that function) specifically so this new, higher-risk module can't
accidentally destabilize the already-validated `encode_extended` path by
sharing code with it.
"""
from __future__ import annotations

import bisect
import collections
from typing import Any

import numpy as np

from . import features as _features
from .paths import DEFAULT_DATA_DIR

SEQ_LEN_DEFAULT = 10


def _build_video_vocab(train_rows: list[dict[str, Any]]) -> dict[str, int]:
    """video_id -> integer code, train-split only, first-seen order --
    same construction `encode_extended` uses for its own video_id field,
    duplicated here on purpose (see module docstring)."""
    vocab: dict[str, int] = {}
    for x in train_rows:
        if x["video_id"] not in vocab:
            vocab[x["video_id"]] = len(vocab)
    return vocab


def _user_histories(all_rows: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[str]]]:
    """Every row (all splits, unfiltered) grouped by user_id and sorted by
    time_ms -- the full chronological log a real system would query against.
    Returns {user_id: (sorted_time_ms_list, corresponding_video_id_list)},
    parallel lists so `bisect` can binary-search the time axis directly."""
    by_user: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for x in all_rows:
        by_user[x["user_id"]].append((x["time_ms"], x["video_id"]))
    out: dict[str, tuple[list[int], list[str]]] = {}
    for u, events in by_user.items():
        events.sort(key=lambda e: e[0])
        out[u] = ([e[0] for e in events], [e[1] for e in events])
    return out


def _recent_history(histories: dict[str, tuple[list[int], list[str]]], user_id: str,
                     before_time_ms: int, seq_len: int) -> list[str]:
    """The `seq_len` most recent video_ids for `user_id` strictly before
    `before_time_ms` -- a strict `<`, never `<=`, so a row can never see
    itself (or any other row at the exact same timestamp) in its own
    history. Returns oldest-first, left-padded with "" (mapped to UNK by
    the caller) if fewer than `seq_len` exist."""
    times, vids = histories.get(user_id, ([], []))
    idx = bisect.bisect_left(times, before_time_ms)  # first index with time >= before_time_ms
    start = max(0, idx - seq_len)
    hist = vids[start:idx]
    if len(hist) < seq_len:
        hist = [""] * (seq_len - len(hist)) + hist
    return hist


def encode_with_history(data_dir: str | None = None, seq_len: int = SEQ_LEN_DEFAULT
                         ) -> tuple[dict[str, tuple], int, dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Same base-5-field encoding as `kuairand-starter-kit/data.py`'s own
    `encode()` (`{split: (X, y, users)}`, `dim`), PLUS two more per-split
    arrays in a SEPARATE, dedicated video-id-only vocabulary (deliberately
    NOT sharing agent/model_zoo/deepfm.py's offset-into-one-big-table
    scheme, to avoid coupling this newer, higher-risk module to that
    already-validated internal layout):

    - `seqs[split]`: `(N, seq_len)` int32, each row's historical video_ids
      (oldest-first, left-padded with the UNK/pad code).
    - `cur_video[split]`: `(N,)` int32, each row's OWN video_id in that same
      vocabulary -- the DIN attention query. A dedicated embedding table
      keyed by this vocabulary is what makes candidate-vs-history attention
      scores meaningful (comparing embeddings from the same learned space).

    Returns `(enc, dim, seqs, cur_video, video_vocab_size)`.
    """
    data_dir = str(data_dir or DEFAULT_DATA_DIR)
    splits = _features.load_splits(data_dir)
    enc, dim = _features.encode_extended(data_dir, _features.BASE_5)

    video_vocab = _build_video_vocab(splits["train"])
    video_unk = len(video_vocab)
    video_vocab_size = video_unk + 1

    all_rows = splits["train"] + splits["valid"] + splits["test"]
    histories = _user_histories(all_rows)

    def _code(vid: str) -> int:
        return video_vocab.get(vid, video_unk)

    seqs: dict[str, np.ndarray] = {}
    cur_video: dict[str, np.ndarray] = {}
    for name, rows in splits.items():
        S = np.empty((len(rows), seq_len), dtype=np.int32)
        C = np.empty(len(rows), dtype=np.int32)
        for n, x in enumerate(rows):
            hist = _recent_history(histories, x["user_id"], x["time_ms"], seq_len)
            S[n] = [_code(v) if v else video_unk for v in hist]  # "" (pad) also maps to UNK/pad slot
            C[n] = _code(x["video_id"])
        seqs[name] = S
        cur_video[name] = C

    return enc, dim, seqs, cur_video, video_vocab_size
