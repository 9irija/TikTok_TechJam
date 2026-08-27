"""Disk cache for encoded training data (P0 efficiency).

Every experiment runs in its own subprocess (agent/recovery.py's isolation
by design -- needed for a real timeout kill switch, see that file's
docstring). Each spawn starts a fresh interpreter and re-parses + re-encodes
the ~1.14M-row training CSVs via kuairand-starter-kit's pure-Python
`data.load()`/`data.encode()` -- measured on this machine, ~10.6s of pure
overhead (4.2s load + 6.4s encode) paid again on *every single experiment
attempt*, completely independent of model training time. With 6 subprocess
spawns in one Phase 0 run today (1 baseline + 3-seed variance + 2 DeepFM)
and more once P1 adds more experiments per run, that overhead compounds
directly into the wall-clock/GPU-hour totals the Feasibility score is
judged on.

This cache makes every spawn after the first pay ~0.1s instead of ~10.6s:
`data.encode()`'s output is memoized to disk, keyed by (data_dir, fields)
and invalidated automatically if any source CSV is newer than the cache
file (so re-downloading a corrected dataset, or pointing at a different
data_dir, can never silently serve stale encoded data).
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable

from .paths import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".cache"
_SOURCE_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
)


def _cache_key(data_dir: str, fields: list[str]) -> str:
    raw = f"{data_dir}|{','.join(sorted(fields))}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(data_dir: str, fields: list[str]) -> Path:
    return CACHE_DIR / f"encoded_{_cache_key(data_dir, fields)}.pkl"


def load_or_build(data_dir: str, fields: list[str], build_fn: Callable[[], tuple]):
    """Returns `build_fn()`'s (enc, dim), from disk if a fresh cache entry
    exists, else computes it once and writes the cache for next time.
    `build_fn` is only ever called on a cache miss or a stale cache."""
    path = _cache_path(data_dir, fields)
    d = Path(data_dir)
    existing_sources = [d / f for f in _SOURCE_FILES if (d / f).exists()]
    newest_source = max((f.stat().st_mtime for f in existing_sources), default=0.0)

    if path.exists() and path.stat().st_mtime > newest_source:
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # corrupt/partial cache file -- fall through and rebuild

    result = build_fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{id(result)}.tmp")  # unique tmp name -- concurrent workers won't collide
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on the same filesystem -- no half-written cache file if interrupted
    except Exception:
        tmp.unlink(missing_ok=True)  # best-effort cleanup; caller still gets a valid in-memory result
    return result
