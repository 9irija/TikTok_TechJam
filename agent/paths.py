"""Central path resolution for the agent package.

Everything downstream (evaluator wrapper, submission validator, model zoo)
needs to import the organizer-provided `evaluate.py` / `data.py` / `submit.py`
from `kuairand-starter-kit/` *unmodified*. This module is the one place that
knows where that directory lives and puts it on `sys.path`, so nothing else
has to duplicate the lookup or risk importing a stale copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTER_KIT_DIR = REPO_ROOT / "kuairand-starter-kit"
DEFAULT_DATA_DIR = STARTER_KIT_DIR / "KuaiRand-Pure" / "data"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
LOGS_DIR = REPO_ROOT / "logs"


def ensure_starter_kit_on_path() -> None:
    """Put kuairand-starter-kit/ on sys.path (idempotent) so `import evaluate`,
    `import data`, `import submit`, `import baseline` resolve to the organizer's
    files, not anything we write ourselves."""
    p = str(STARTER_KIT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def data_dir_available(data_dir: Path | None = None) -> bool:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return (d / "log_standard_4_08_to_4_21_pure.csv").exists()
