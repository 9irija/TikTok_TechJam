"""Metric-Aware Diagnosis Engine (P1).

Turns "the score went down, try something else" into an actual diagnosis by
reading the *pattern* across GAUC/nDCG@5, not just the primary scalar --
per the brainstorm doc: "NDCG up/Recall down -> ranking problem; train
up/valid down -> overfitting; both low -> feature/model mismatch." This is
deliberately deterministic, rule-based logic -- no LLM. Phase 4's LLM
Research Strategist reads this module's output as *input context*
("dataset summary + metric history + research map"); it doesn't replace it.

Everything here compares a node's aggregated metrics against either its
parent (a real "what did this change do") or, for a root node, against the
official baseline (the only reference point available). All thresholds are
read from the organizer's own epsilon (baseline_scores.json), not invented,
so "clear_improvement" vs. "noise_floor" uses the exact same bar the
Convergence Detector itself uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .evaluator import load_baseline_scores


@dataclass
class Diagnosis:
    tag: str
    insight: str


def _epsilon() -> float:
    return load_baseline_scores()["convergence_rule"]["epsilon"]


def _significance_bar(v: dict, pv: dict, n_v: int, n_pv: int, eps: float) -> float:
    """The bar |delta primary| must clear to be treated as a real signal
    rather than noise.

    "Is this different from zero" (statistical significance) and "is this
    different enough to matter for convergence" (the organizer's practical
    epsilon, ~2.5 sigma of the baseline's OWN 5-seed std) are two different
    questions -- conflating them (an earlier version of this function used
    max(eps, 2*sem), which can never return less than eps) is exactly what
    caused this engine to mislabel deepfm-vs-FM as "noise_floor" during P1's
    own Research-Map-seeding pass, despite that being a real, separately
    confirmed ~8.9-sigma effect (docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md
    Sec.4) that simply landed below the practical epsilon on point estimates
    alone. When real seed data exists (>=2 seeds on either side), the bar
    here is purely statistical: 2x the combined standard error of the mean,
    which can be smaller *or* larger than epsilon depending on how noisy the
    comparison actually is. Epsilon is used as a fallback only when there
    isn't enough seed data to assess confidence at all (a single-seed
    comparison, where no std is even computable) -- the previous, and only
    honest, behavior in that case.
    """
    if n_v < 2 and n_pv < 2:
        return eps
    sem_v = v.get("primary_std", 0.0) / math.sqrt(max(n_v, 1))
    sem_pv = pv.get("primary_std", 0.0) / math.sqrt(max(n_pv, 1))
    combined_sem = math.sqrt(sem_v ** 2 + sem_pv ** 2)
    return 2 * combined_sem if combined_sem > 0 else eps


def _valid(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    if not metrics:
        return None
    return metrics.get("valid")


def diagnose(metrics: dict[str, Any] | None, parent_metrics: dict[str, Any] | None = None) -> Diagnosis:
    """`metrics` / `parent_metrics` are the aggregated per-config dicts
    `agent.orchestrator._aggregate_seed_results` produces (keys: valid, test,
    per_seed, ...). Only ever reads `valid` -- test scores play no role in
    diagnosis, matching the train/valid/test discipline everywhere else in
    this codebase.
    """
    if metrics is None:
        return Diagnosis("no_result", "No successful result to diagnose (all attempts failed/timed out).")

    v = _valid(metrics)
    eps = _epsilon()
    n_v = metrics.get("n_seeds", 1)
    overfit_note = _overfitting_note(metrics)

    if parent_metrics is None:
        b_full = load_baseline_scores()["scores"]["fm_official"]
        b = b_full["valid"]
        # The organizer's own reported baseline std (5 seeds) stands in for
        # "parent" uncertainty here -- there's no actual parent node to pull
        # a std from for a root.
        bar = _significance_bar(v, {"primary_std": b_full["std_over_5_seeds"]["test_primary"]},
                                 n_v, 5, eps)
        delta = v["primary_mean"] - b["primary"]
        if delta > bar:
            tag, insight = "baseline_beat", f"Root node beats the official FM baseline by {delta:+.4f} valid primary (bar={bar:.4f})."
        elif delta < -bar:
            tag, insight = "baseline_miss", f"Root node underperforms the official FM baseline by {delta:+.4f} valid primary (bar={bar:.4f})."
        else:
            tag, insight = "baseline_match", f"Root node matches the official FM baseline within noise ({delta:+.4f}, bar={bar:.4f})."
        return Diagnosis(tag, insight + overfit_note)

    pv = _valid(parent_metrics)
    if pv is None:
        return Diagnosis("parent_no_result", "Parent node has no result to compare against." + overfit_note)

    n_pv = parent_metrics.get("n_seeds", 1)
    bar = _significance_bar(v, pv, n_v, n_pv, eps)
    d_gauc = v["GAUC_mean"] - pv["GAUC_mean"]
    d_ndcg = v["nDCG@5_mean"] - pv["nDCG@5_mean"]
    d_primary = v["primary_mean"] - pv["primary_mean"]

    if d_gauc > bar and d_ndcg > bar:
        tag = "clear_improvement"
        insight = f"Both GAUC ({d_gauc:+.4f}) and nDCG@5 ({d_ndcg:+.4f}) improved over the parent -- a clean win (bar={bar:.4f})."
    elif d_gauc < -bar and d_ndcg < -bar:
        tag = "regression"
        insight = f"Both GAUC ({d_gauc:+.4f}) and nDCG@5 ({d_ndcg:+.4f}) got worse -- the change likely hurt (bar={bar:.4f})."
    elif d_gauc > bar and d_ndcg < -bar:
        tag = "ranking_tradeoff"
        insight = (f"GAUC improved ({d_gauc:+.4f}) but nDCG@5 dropped ({d_ndcg:+.4f}): broader within-user "
                   f"ordering got better, but top-5 precision got worse -- a ranking-quality trade-off, "
                   f"not a clean win. Consider a listwise/top-k-aware loss instead of pointwise.")
    elif d_ndcg > bar and d_gauc < -bar:
        tag = "ranking_tradeoff"
        insight = (f"nDCG@5 improved ({d_ndcg:+.4f}) but GAUC dropped ({d_gauc:+.4f}): top-5 precision got "
                   f"better, but broader ordering got worse -- likely overfit to head items/short lists.")
    elif abs(d_primary) <= bar:
        tag = "noise_floor"
        insight = (f"Change vs. parent ({d_primary:+.4f} primary) is within the significance bar "
                   f"({bar:.4f}, {'seed-aware' if bar != eps else 'flat epsilon, single-seed'}) "
                   f"-- not distinguishable from run-to-run variance.")
    else:
        tag = "mixed"
        insight = f"Mixed/inconclusive signal (GAUC {d_gauc:+.4f}, nDCG@5 {d_ndcg:+.4f}, primary {d_primary:+.4f}, bar={bar:.4f})."

    return Diagnosis(tag, insight + overfit_note)


def _overfitting_note(metrics: dict[str, Any]) -> str:
    """Weak overfitting signal from the training curves already logged per
    seed: if the best validation epoch consistently lands early relative to
    how long training ran (patience kept giving it more epochs, none of
    which helped), later epochs were spent overfitting train loss without
    validation benefit."""
    per_seed = metrics.get("per_seed") or []
    if not per_seed:
        return ""
    ratios = []
    for r in per_seed:
        epochs_run = r.get("epochs_run") or 0
        best_epoch = r.get("best_epoch") or 0
        if epochs_run > 0:
            ratios.append(best_epoch / epochs_run)
    if not ratios:
        return ""
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio < 0.4:
        return (f" [overfitting_risk: best validation epoch landed at ~{avg_ratio:.0%} of training "
                f"length on average across seeds -- later epochs likely overfit train loss without "
                f"validation benefit; consider fewer epochs or stronger regularization.]")
    return ""
