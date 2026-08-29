"""Engineered features beyond the starter kit's 5 base fields (P1 extension,
finally wired up in this pass -- `ExperimentConfig.fields` existed since
Phase 0 but was never actually connected to encoding; see `_load_encoded`
in agent/experiment.py and the note in that diff).

kuairand-starter-kit/data.py's own file comment explicitly invites this:
"加特征就往这里加，这是学生最该动的地方之一" -- "add features here, this is
one of the places students should most modify." This module doesn't modify
that file in place, though: doing so would retroactively change what every
already-validated Research Map node's baseline-reproduction comparison
means (`agent/model_zoo/fm.py`'s docstring requires staying numerically
identical to the organizer's own FM). Instead this reads the same raw CSVs
independently -- same date-based split boundaries, copied as a constant
(public, simple, not the kind of pinned SCORING logic evaluate.py
protects) -- and adds new fields on top. The original 5-field path through
`kuairand-starter-kit/data.py` stays exactly as validated; new experiments
opt into extended fields via `ExperimentConfig.fields`.

LEAKAGE WARNING, read before touching this file: `play_time_ms` is what
`long_view` (the label) is derived from -- empirically ~85% agreement with
a naive "played >=90% of duration" reconstruction, confirmed by directly
checking the raw log before writing any of this. TikTok has publicly
described completion rate / rewatch as strong signals (cited in the
brainstorm doc's P1 table), but `completion_rate = play_time_ms /
duration_ms` computed on THE SAME ROW being predicted would hand the model
the answer, not a signal. Every feature this file computes is instead a
TRAIN-SPLIT-ONLY AGGREGATE (a video's or author's *historical* average
across other impressions) -- the exact same pattern the organizer's own
item-popularity baseline already uses safely. Never add a field derived
from a row's own play_time_ms/duration_ms ratio without going through the
aggregation in `compute_train_only_aggregates` first.
"""
from __future__ import annotations

import collections
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from .paths import DEFAULT_DATA_DIR

LABEL = "long_view"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FAST_SKIP_MS = 1000.0  # near-instant skip -- TikTok's own disclosed treatment of skips as implicit negative signal

# The 5 fields kuairand-starter-kit/data.py always encodes, in its own order.
BASE_5 = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
# New fields this module can add, each a TRAIN-ONLY aggregate bucket -- never
# a same-row value.
EXTRA_FIELDS = ["video_completion_bucket", "video_rewatch_bucket", "video_fast_skip_bucket", "author_engagement_bucket"]


def _load_raw(data_dir: str) -> list[dict[str, Any]]:
    vid2author: dict[str, str] = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    rows: list[dict[str, Any]] = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "date": int(r["date"]), "time_ms": int(r["time_ms"]),
                    "user_id": r["user_id"], "video_id": r["video_id"],
                    "author_id": vid2author.get(r["video_id"], "UNK"), "tab": r["tab"],
                    "duration_ms": float(r["duration_ms"]), "play_time_ms": float(r["play_time_ms"]),
                    "label": 1 if r[LABEL] != "0" else 0,
                    "is_click": 1 if r["is_click"] != "0" else 0, "is_like": 1 if r["is_like"] != "0" else 0,
                    "is_follow": 1 if r["is_follow"] != "0" else 0, "is_comment": 1 if r["is_comment"] != "0" else 0,
                    "is_forward": 1 if r["is_forward"] != "0" else 0,
                })
    return rows


def load_splits(data_dir: str | None = None) -> dict[str, list[dict[str, Any]]]:
    data_dir = str(data_dir or DEFAULT_DATA_DIR)
    rows = _load_raw(data_dir)
    return {name: [x for x in rows if lo <= x["date"] <= hi] for name, (lo, hi) in SPLITS.items()}


AUX_LABEL_FIELDS = ["is_like", "is_follow", "is_comment", "is_forward"]


def load_aux_labels(data_dir: str | None = None) -> dict[str, np.ndarray]:
    """Per-split (N, 4) float32 arrays of `AUX_LABEL_FIELDS` -- TikTok's
    other logged engagement signals, used only as AUXILIARY training
    targets for a multi-task model (agent/model_zoo/deepfm_mtl.py), never
    as model *input* (that would be a different, much more direct leak than
    the play_time_ms one this file already guards against: is_like/
    is_follow/etc. are themselves outcomes of the same impression being
    scored, not history). Same row order as `load_splits`/`encode_extended`
    -- both iterate the identical two log files in the identical order with
    the identical date-range filter (kuairand-starter-kit/data.py does too;
    confirmed by direct comparison, not assumed), so an aux array from here
    lines up positionally with any encoder's X/y for the same split without
    needing to carry a shared row-id key through.
    """
    splits = load_splits(data_dir)
    return {name: np.array([[x[f] for f in AUX_LABEL_FIELDS] for x in rows], dtype=np.float32)
            for name, rows in splits.items()}


WATCH_RATIO_CLIP = 3.0  # cap rewatches (play_time_ms > duration_ms) at 3x duration before normalizing


def load_watch_ratio(data_dir: str | None = None, clip: float = WATCH_RATIO_CLIP) -> dict[str, np.ndarray]:
    """Per-split (N,) float32 arrays of a clipped, [0,1]-normalized
    play_time_ms/duration_ms completion ratio -- used ONLY as a continuous
    auxiliary training TARGET for `agent/model_zoo/deepfm_mtl_watch.py`
    (CLAUDE.md's "Unexplored headroom" #4, "Watch-time modeling... Still
    open"), never as model input.

    This is NOT the leakage case this file's own module docstring warns
    about: that warning is about using a row's own completion ratio as an
    INPUT feature, which would hand the model `long_view` directly (both
    are derived from the same play_time_ms/duration_ms pair). Using it as
    a training TARGET for an auxiliary head is a structurally different,
    already-established pattern in this codebase -- `load_aux_labels`'
    is_like/is_follow/etc. are exactly the same shape of thing (a same-row
    outcome of the impression being scored, used only to shape gradients
    into the shared embedding table). The main `long_view` logit -- the
    only thing GAUC/nDCG@5 ever score -- never sees this value at either
    train or inference time; `deepfm_mtl_watch.py`'s `predict()` only ever
    reads the main head.

    Same row order as `load_splits`/`load_aux_labels` (see that function's
    docstring for why that alignment is guaranteed, not assumed).
    """
    splits = load_splits(data_dir)
    out: dict[str, np.ndarray] = {}
    for name, rows in splits.items():
        ratios = [min(x["play_time_ms"] / x["duration_ms"], clip) if x["duration_ms"] > 0 else 0.0
                  for x in rows]
        out[name] = (np.array(ratios, dtype=np.float32) / clip)
    return out


def compute_train_only_aggregates(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-video and per-author aggregate stats from the TRAINING split
    only. `play_time_ms` is read here -- and ONLY here -- to compute a
    completion ratio that then gets averaged across many rows per video/
    author before ever becoming a feature; no single row's own ratio is
    ever exposed to the model."""
    video = collections.defaultdict(lambda: {"n": 0, "completion_sum": 0.0, "rewatch": 0, "fast_skip": 0})
    author = collections.defaultdict(lambda: {"n": 0, "like": 0, "follow": 0, "comment": 0, "forward": 0})

    for x in train_rows:
        cr = (x["play_time_ms"] / x["duration_ms"]) if x["duration_ms"] > 0 else 0.0
        v = video[x["video_id"]]
        v["n"] += 1
        v["completion_sum"] += cr
        v["rewatch"] += int(x["play_time_ms"] > x["duration_ms"])
        v["fast_skip"] += int(x["play_time_ms"] < FAST_SKIP_MS)

        a = author[x["author_id"]]
        a["n"] += 1
        a["like"] += x["is_like"]; a["follow"] += x["is_follow"]
        a["comment"] += x["is_comment"]; a["forward"] += x["is_forward"]

    return {
        "video_completion": {v_: s["completion_sum"] / s["n"] for v_, s in video.items()},
        "video_rewatch_rate": {v_: s["rewatch"] / s["n"] for v_, s in video.items()},
        "video_fast_skip_rate": {v_: s["fast_skip"] / s["n"] for v_, s in video.items()},
        "author_engagement": {a_: (s["like"] + s["follow"] + s["comment"] + s["forward"]) / s["n"]
                               for a_, s in author.items()},
    }


def _bucket_edges(values: list[float], n: int = 5) -> np.ndarray:
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1]) if values else np.array([])


def _bucket_of(value: float | None, edges: np.ndarray) -> str:
    if value is None:
        return "UNK"
    return str(int(np.searchsorted(edges, value)))


def encode_extended(data_dir: str, fields: list[str]) -> tuple[dict[str, tuple], int]:
    """Same return shape as kuairand-starter-kit/data.py's own `encode()`
    (`{split: (X, y, users)}`, `dim`) so `agent/model_zoo` never has to know
    which loader produced its input -- but supports any subset/superset of
    `BASE_5 + EXTRA_FIELDS`, not just the fixed 5.
    """
    splits = load_splits(data_dir)
    tr = splits["train"]
    dur_edges = _bucket_edges([x["duration_ms"] for x in tr])
    aggs = compute_train_only_aggregates(tr) if any(f in EXTRA_FIELDS for f in fields) else None

    edges = {}
    if aggs is not None:
        for key in ("video_completion", "video_rewatch_rate", "video_fast_skip_rate", "author_engagement"):
            edges[key] = _bucket_edges(list(aggs[key].values()))

    def raw(x: dict[str, Any]) -> list[str]:
        out = []
        for f in fields:
            if f == "user_id":
                out.append(x["user_id"])
            elif f == "video_id":
                out.append(x["video_id"])
            elif f == "author_id":
                out.append(x["author_id"])
            elif f == "tab":
                out.append(x["tab"])
            elif f == "dur_bucket":
                out.append(str(int(np.searchsorted(dur_edges, x["duration_ms"]))))
            elif f == "video_completion_bucket":
                out.append(_bucket_of(aggs["video_completion"].get(x["video_id"]), edges["video_completion"]))
            elif f == "video_rewatch_bucket":
                out.append(_bucket_of(aggs["video_rewatch_rate"].get(x["video_id"]), edges["video_rewatch_rate"]))
            elif f == "video_fast_skip_bucket":
                out.append(_bucket_of(aggs["video_fast_skip_rate"].get(x["video_id"]), edges["video_fast_skip_rate"]))
            elif f == "author_engagement_bucket":
                out.append(_bucket_of(aggs["author_engagement"].get(x["author_id"]), edges["author_engagement"]))
            else:
                raise ValueError(f"unknown field '{f}' -- must be one of {BASE_5 + EXTRA_FIELDS}")
        return out

    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x["label"]
            users.append(x["user_id"])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
