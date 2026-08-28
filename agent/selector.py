"""Best-First Node Selector (P1).

"A dedicated selector role... greedily picks the most promising node to
expand next, scored by (expected gain x confidence x novelty / cost) --
can branch off an older strong node, not just the most recent result."

This is what replaces Phase 0's flat "run PREDEFINED_EXPERIMENTS in list
order" behavior with AIDE's best-first tree search. No LLM here (that's
Phase 4) -- this is a deterministic heuristic scorer, and *which ideas
exist* is still a hand-authored candidate pool (agent/p1_orchestrator.py);
what's new is that the ORDER they get tried in, and which existing node
each branches from, is a computed score informed by the Research Map's
accumulated history, not just source order or "always branch off the most
recent result."

Every score is stored with a human-readable `reasoning` string --
Innovation & Problem Insight is judged on *why* the agent tried something,
not just that it did (PS: "Judged on what the agent identified as worth
trying and why -- not on implementation").
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .config import ExperimentConfig
from .research_map import ResearchMap

# Rough wall-clock priors (seconds, single seed, full data) used only when
# the Research Map has no historical data yet for a given model family --
# replaced by the map's own measured average the moment any data exists.
_DEFAULT_COST_S = {"fm": 90.0, "deepfm": 100.0, "fm_bpr": 100.0, "deepfm_bpr": 110.0}


@dataclass
class CandidateScore:
    config: ExperimentConfig
    expected_gain: float
    confidence: float
    novelty: float
    cost_s: float
    score: float
    reasoning: str


def _family_key(config: ExperimentConfig) -> str:
    """Groups configs for cost/novelty estimation -- model + edge-relevant
    hyperparam shape (e.g. 'deepfm' with hidden=[64,32] vs [128,64] are the
    same family for cost purposes, different for novelty purposes)."""
    return config.model


def _historical_cost(map: ResearchMap, family: str) -> float:
    times = []
    for n in map.nodes.values():
        if n.config.model == family and n.metrics:
            per_seed = n.metrics.get("per_seed") or []
            times.extend(r["wall_time_s"] for r in per_seed if "wall_time_s" in r)
    return mean(times) if times else _DEFAULT_COST_S.get(family, 100.0)


def _expected_gain(map: ResearchMap, candidate: ExperimentConfig) -> tuple[float, str]:
    """Extrapolates from the parent's own improvement over ITS parent, if
    both exist and succeeded -- a simple 'trend continuation' prior. Falls
    back to a small flat optimistic default for genuinely new directions
    (a 'draft' edge with no comparable history), and applies a penalty if
    this exact model family has a 'regression' diagnosis anywhere in the map
    (repeating a direction that already failed shouldn't score well)."""
    parent = map.nodes.get(candidate.parent_id) if candidate.parent_id else None
    family = _family_key(candidate)
    family_diagnoses = [n.diagnosis_tag for n in map.nodes.values() if n.config.model == family]

    if family_diagnoses.count("regression") > family_diagnoses.count("clear_improvement"):
        return 0.0005, (f"'{family}' has more regressions than clear improvements in the Research Map so far "
                         f"({family_diagnoses.count('regression')} vs {family_diagnoses.count('clear_improvement')}) "
                         f"-- low expected gain until a different direction shows promise.")

    if parent is None or parent.metrics is None or parent.parent_id is None:
        return 0.0015, "No comparable parent-over-grandparent trend available -- flat optimistic prior for a fresh direction."

    grandparent = map.nodes.get(parent.parent_id)
    if grandparent is None or grandparent.metrics is None:
        return 0.0015, "Parent has no scored grandparent to extrapolate a trend from -- flat optimistic prior."

    parent_gain = (parent.metrics["valid"]["primary_mean"] - grandparent.metrics["valid"]["primary_mean"])
    # Diminishing-returns extrapolation: assume the next step in the same
    # direction captures ~half of the previous step's gain, not the same
    # amount again -- avoids naively projecting linear improvement forever.
    projected = max(parent_gain * 0.5, 0.0)
    return projected, (f"Extrapolated at half of parent's own gain over its parent "
                        f"({parent_gain:+.4f} -> projected {projected:+.4f}), diminishing-returns prior.")


def _novelty(map: ResearchMap, candidate: ExperimentConfig) -> tuple[float, str]:
    family = _family_key(candidate)
    same_family = [n for n in map.nodes.values() if n.config.model == family]
    if not same_family:
        return 1.0, f"'{family}' is an entirely new model family in the Research Map -- maximum novelty."
    # Novelty decays with how many nodes already explore this family --
    # cheap proxy for "how much of this region of the solution space is covered."
    novelty = max(0.15, 1.0 / (1 + len(same_family)))
    return novelty, f"'{family}' already has {len(same_family)} node(s) in the Research Map -- novelty discounted."


def score_candidate(map: ResearchMap, candidate: ExperimentConfig) -> CandidateScore:
    gain, gain_reason = _expected_gain(map, candidate)
    novelty, novelty_reason = _novelty(map, candidate)
    cost = _historical_cost(map, _family_key(candidate))
    n_seeds = max(1, len(candidate.seeds))
    cost *= n_seeds  # multi-seed candidates cost proportionally more -- must earn it via gain/confidence

    # Confidence: more historical data for this family -> more confidence in
    # the gain estimate; a first-ever attempt at a family is a bigger bet.
    same_family_done = sum(1 for n in map.nodes.values() if n.config.model == _family_key(candidate) and n.metrics)
    confidence = min(0.9, 0.3 + 0.15 * same_family_done)

    score = (gain * confidence * novelty) / max(cost, 1.0)
    reasoning = (f"gain={gain:+.4f} ({gain_reason}) | confidence={confidence:.2f} (based on "
                 f"{same_family_done} prior scored node(s) in this family) | novelty={novelty:.2f} "
                 f"({novelty_reason}) | cost~={cost:.0f}s ({n_seeds} seed(s))")
    return CandidateScore(config=candidate, expected_gain=gain, confidence=confidence,
                           novelty=novelty, cost_s=cost, score=score, reasoning=reasoning)


def select_next(map: ResearchMap, candidate_pool: list[ExperimentConfig]
                 ) -> tuple[CandidateScore | None, list[CandidateScore]]:
    """Scores every not-yet-in-the-map candidate, returns (best, all_ranked).
    `all_ranked` is what lets a run log "generated N candidates, rejected
    M due to poor gain/cost ratio" -- the demo narrative the brainstorm doc
    explicitly calls out as the strongest way to show all five judging
    criteria in one sequence.
    """
    unexplored = [c for c in candidate_pool if c.id not in map.nodes]
    scored = sorted((score_candidate(map, c) for c in unexplored), key=lambda s: s.score, reverse=True)
    return (scored[0] if scored else None), scored
