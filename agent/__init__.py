"""Autonomous ML Research Agent -- Phase 0 foundation.

Package layout (see /CLAUDE.md for the full architecture write-up):

    paths.py         path resolution into kuairand-starter-kit/
    config.py        Structured Experiment Interface (P0)
    evaluator.py      Evaluator Wrapper around the organizer's evaluate.py (P0)
    convergence.py    Convergence Detector, epsilon/N read from baseline_scores.json (P0)
    model_zoo/        Model Zoo (base): FM + DeepFM (P0)
    experiment.py      Turns a config into a trained model + metrics
    recovery.py         Failure Recovery: subprocess isolation, retry, timeout (P0)
    run_logger.py        Structured Run Log: experiments/<run_id>/*/ + logs/run_log.jsonl (P0)
    orchestrator.py       Orchestrator/Planner: predefined experiment list, no LLM yet (P0)
    submission.py          Submission Validator wrapper around submit.py --check (P0)
"""
import os as _os

# BLAS threading fix -- must run before numpy's first import anywhere in
# this process (OpenBLAS reads these when it initializes its thread pool).
# `agent/__init__.py` executes before any submodule, so this is the
# earliest hook available regardless of entry point (run.py,
# tests/test_foundation.py, or a subprocess worker spawned by
# agent/recovery.py, which inherits this via os.environ).
#
# Why: OpenBLAS's default multi-threaded matmul is *slower* than
# single-threaded here, because every experiment does many small matrix
# multiplies per batch (embedding-sized, e.g. 8192x80 @ 80x128) in a tight
# Python loop -- thread-pool spawn/sync overhead dominates the actual
# FLOPs. Measured on this machine (agent/model_zoo/deepfm.py,
# hidden=[128,64]): default OpenBLAS threading gave ~88s/epoch; pinning to
# 1 thread gave ~15s/epoch (~6x). This is what caused an early production
# run's deepfm_wider iteration to blow through its per-experiment timeout
# and get abandoned by Failure Recovery -- Recovery handled that correctly
# (no crash, no stall, logged and moved on), but fixing the root cause here
# is strictly better than just widening every timeout.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
