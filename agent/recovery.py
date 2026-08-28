"""Failure Recovery (P0).

"Robustness here is about how the agent handles difficulty... when a step
fails (a code error, a timeout, an unexpected input), the agent can recover,
retry, or route around it, and long iterative runs neither crash, stall, nor
diverge." -- Task Requirement 3.

We run every experiment in its own subprocess (not just a try/except in the
main loop) for two reasons a plain try/except can't cover:

  1. A real timeout with a real kill switch. Windows has no SIGALRM, and pure
     numpy code won't yield to a Python-level "check a flag" timeout while
     it's inside a C loop -- only `Process.terminate()` can actually stop it.
  2. A hard crash (segfault, OOM-killed process, a runaway allocation) that
     never raises a catchable Python exception still can't take down the
     Orchestrator's own process this way.

On failure: retry up to `max_retries` times, then fall back once to a cheap
degraded run (few epochs) so the iteration still produces *some* logged
result instead of silently vanishing, then give up on this one config and
let the Orchestrator move to the next -- a single bad experiment must never
stall or crash the whole run.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig
from .experiment import run_experiment


@dataclass
class RecoveryEvent:
    kind: str  # "error" | "timeout" | "retry" | "fallback" | "abandoned"
    message: str
    attempt: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "attempt": self.attempt,
                "timestamp": self.timestamp}


def _worker(config_dict: dict, data_dir: str, seed: int, run_kwargs: dict[str, Any], queue) -> None:
    """Top-level, picklable target for `multiprocessing` (required on
    Windows, which only supports the 'spawn' start method). `run_kwargs` is
    passed straight through to `run_experiment` (e.g. `max_epochs_override`,
    `save_predictions_to`, `train_fraction`) -- kept as one dict so adding a
    new run_experiment parameter never requires touching this plumbing again.
    """
    try:
        config = ExperimentConfig.from_dict(config_dict)
        result = run_experiment(config, data_dir, seed, **run_kwargs)
        queue.put(("ok", result.to_dict()))
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure must be caught and reported
        queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _try_once(config: ExperimentConfig, data_dir: str, seed: int, timeout_s: float,
              attempt: int, events: list[RecoveryEvent], run_kwargs: dict[str, Any]):
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(config.to_dict(), data_dir, seed, run_kwargs, queue))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        events.append(RecoveryEvent(kind="timeout", attempt=attempt,
                      message=f"Experiment '{config.id}' (seed={seed}) exceeded {timeout_s:.0f}s "
                              f"wall-clock budget; process terminated."))
        return None

    if queue.empty():
        events.append(RecoveryEvent(kind="error", attempt=attempt,
                      message=f"Worker process for '{config.id}' exited (code={proc.exitcode}) "
                              f"without a result -- likely a crash/OOM outside a catchable exception."))
        return None

    status, payload = queue.get()
    if status == "ok":
        if attempt > 1:
            events.append(RecoveryEvent(kind="retry", attempt=attempt,
                          message=f"Retry #{attempt - 1} of '{config.id}' succeeded."))
        return payload

    events.append(RecoveryEvent(kind="error", attempt=attempt, message=payload))
    return None


def run_with_recovery(config: ExperimentConfig, data_dir: str, seed: int = 0,
                       timeout_s: float = 300.0, max_retries: int = 2,
                       degrade_epochs: int = 5,
                       save_predictions_to: str | None = None,
                       train_fraction: float = 1.0,
                       max_epochs_override: int | None = None) -> tuple[dict | None, list[RecoveryEvent]]:
    """Returns (result_dict, events). result_dict is None only if every
    retry AND the degraded fallback also failed -- the caller (Orchestrator
    or the Multi-Fidelity Runner) logs that as a fully-failed iteration and
    proceeds; this function must never raise.

    `save_predictions_to`, when given, is only honored on a *successful*
    (non-degraded) attempt -- see agent/experiment.py's run_experiment
    docstring. A degraded fallback's predictions are intentionally not
    cached: they come from a crippled few-epoch run, not the config's real
    result, and must never be mistaken for it downstream.

    `train_fraction` / `max_epochs_override` (P1's Multi-Fidelity Runner):
    trains on a random subsample of the training split, capped at this many
    epochs, on every attempt/retry of *this* call -- escalating fidelity
    stages are separate `run_with_recovery` calls, each with its own
    recovery/retry (agent/multi_fidelity.py). `max_epochs_override` here is
    the *requested* cap for a normal attempt; `degrade_epochs` remains a
    separate, smaller, emergency-fallback-only budget used after every
    normal retry has already failed.
    """
    events: list[RecoveryEvent] = []
    base_kwargs = {"save_predictions_to": save_predictions_to, "train_fraction": train_fraction}

    for attempt in range(1, max_retries + 1):
        result = _try_once(config, data_dir, seed, timeout_s, attempt, events,
                            run_kwargs={**base_kwargs, "max_epochs_override": max_epochs_override})
        if result is not None:
            return result, events

    events.append(RecoveryEvent(
        kind="fallback", attempt=max_retries + 1,
        message=f"All {max_retries} attempt(s) of '{config.id}' failed; falling back to a "
                f"{degrade_epochs}-epoch degraded run so this iteration still logs a real result."))
    result = _try_once(config, data_dir, seed, timeout_s, max_retries + 1, events,
                        run_kwargs={"save_predictions_to": None, "train_fraction": train_fraction,
                                    "max_epochs_override": degrade_epochs})
    if result is None:
        events.append(RecoveryEvent(
            kind="abandoned", attempt=max_retries + 2,
            message=f"Degraded fallback for '{config.id}' also failed -- abandoning this experiment; "
                    f"orchestrator proceeds to the next config."))
    return result, events
