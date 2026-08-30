"""Research Strategist (Phase 4): LLM-driven hypothesis generation.

Replaces the hand-authored candidate pools (Phase 0's PREDEFINED_EXPERIMENTS;
P1's static `p1_candidate_pool()` plus its two hardcoded diagnosis-driven
follow-ups) with an LLM call that reads the dataset summary, the Research
Map's full history, and remaining budget, then proposes exactly ONE new
experiment: `{hypothesis, experiment config, reasoning,
expected_metric_effect, estimated_cost, priority}` -- the exact shape the
brainstorm doc's Phase 4 spec asks for.

Design principles carried over from P0/P1, not new to this file:
  - Structured Experiment Interface: the LLM only ever proposes a
    STRUCTURED CONFIG (a model name from a fixed registry + hyperparams),
    never free-form code -- matching every other experiment in this
    codebase. A bad generation can corrupt a config value; it can never
    corrupt the training pipeline itself.
  - Separation of "propose" from "write code": per AI-Scientist-v2's design
    (cited in CLAUDE.md), the call that decides what to try next is kept
    separate from any call that would write raw code for it -- there is no
    code-generation call anywhere in this file, by construction.
  - Validation Gate, not blind trust: every field the LLM returns is
    checked against the same registry/schema everything else in this
    codebase already respects (`agent.model_zoo.registry.MODELS`,
    `ExperimentConfig`'s own field types) before it is ever executed.
  - Failure Recovery philosophy, one level up: `agent/llm_client.py`
    already retries transport failures (network, rate limit, malformed
    JSON). A well-formed JSON response that fails *semantic* validation
    (unknown model, non-existent parent_id, wrong hyperparam types) is a
    different failure mode -- handled here by re-prompting with the
    specific validation error and retrying, before giving up and returning
    None so the caller can log it and move on rather than crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .evaluator import load_baseline_scores
from .llm_client import GeminiClient, LLMError, LLMResponse
from .model_zoo import MODELS
from .research_map import ResearchMap

_MODEL_HYPERPARAM_DOCS = {
    "fm": "hyperparams: k (int, embedding dim), lr (float), l2 (float), batch (int), epochs (int), patience (int).",
    "deepfm": "hyperparams: same as fm, plus hidden (list of exactly 2 ints, e.g. [64,32] -- the MLP deep component's layer sizes).",
    "fm_bpr": "hyperparams: same as fm. Trained with pairwise BPR ranking loss instead of pointwise logloss.",
    "deepfm_mtl": "hyperparams: same as deepfm, plus aux_weight (float, typically ~0.2). Shares deepfm's embedding "
                  "table with 4 auxiliary sigmoid heads (is_like/is_follow/is_comment/is_forward) trained jointly "
                  "-- only the main long_view logit is ever scored.",
    "deepfm_bpr": "hyperparams: same as deepfm. deepfm's architecture trained on fm_bpr's pairwise objective "
                  "instead of pointwise logloss.",
    "deepfm_mtl_pcgrad": "hyperparams: same as deepfm_mtl. Resolves gradient DIRECTION conflict between the main "
                         "and auxiliary tasks on their shared parameters (PCGrad, Yu et al. 2020) before combining "
                         "them, instead of a plain summed loss.",
    "deepfm_mtl_click": "hyperparams: same as deepfm_mtl. Adds is_click as a 5th auxiliary head (a dense, "
                        "~46%-positive-rate signal, vs. the other 4 signals' 0.1-1.8% each).",
    "deepfm_mtl_focal": "hyperparams: same as deepfm_mtl, plus focal_gamma (float, paper default 2.0) and "
                        "focal_alpha (float, 0.5 is class-balance-neutral). Replaces the main task's BCE with "
                        "focal loss (Lin et al. 2017) -- down-weights already-easy examples so gradient "
                        "concentrates on ones the model still gets wrong.",
}


def _dead_ends_section(map: ResearchMap) -> str:
    """Generates the "known non-starters" guidance from the Research Map's
    own diagnosis tags at prompt-build time, rather than a hand-maintained
    string baked into `_MODEL_HYPERPARAM_DOCS` (the exact gap
    `docs/PHASE4_RESULTS.md` Sec.6 flagged: a static list goes stale the
    moment a NEW dead end is confirmed after the file was last hand-edited
    -- this project hit that directly: the old hardcoded fm_bpr note was
    still telling the LLM "'deepfm_bpr' is NOT a valid model name" long
    after deepfm_bpr had actually been built, registered, and even
    confirmed as its own real dead end in P2). Groups every `regression`/
    `noise_floor`-tagged `done` node by model family so the LLM sees
    exactly what's already been tried and didn't work, current as of
    THIS call, not whenever this file was last edited.
    """
    by_model: dict[str, list[str]] = {}
    for n in map.done_nodes():
        if n.diagnosis_tag in ("regression", "noise_floor", "mixed", "ranking_tradeoff"):
            by_model.setdefault(n.config.model, []).append(f"{n.node_id} [{n.diagnosis_tag}]")
    if not by_model:
        return "(none confirmed yet)"
    lines = []
    for model, entries in sorted(by_model.items()):
        lines.append(f'- "{model}": {len(entries)} non-improving result(s) already in the history -- '
                     f'{", ".join(entries)}. Don\'t repeat these exact configs without a genuinely new angle.')
    return "\n".join(lines)


class ValidationError(Exception):
    """A well-formed JSON response that fails semantic validation --
    different from LLMError (transport failure), and handled with a
    re-prompt-and-retry rather than immediate failure."""


@dataclass
class StrategistProposal:
    config: ExperimentConfig
    reasoning: str
    expected_metric_effect: dict[str, str]
    estimated_cost_s: float
    priority: float


def _build_prompt(map: ResearchMap, budget: dict[str, Any], retry_note: str = "") -> str:
    summary = map.explored_summary()
    baseline = load_baseline_scores()["scores"]["fm_official"]["valid"]
    recent = sorted(map.nodes.values(), key=lambda n: n.updated_at, reverse=True)[:8]

    history_lines = []
    for n in recent:
        vp = (n.metrics or {}).get("valid", {}).get("primary_mean")
        vp_str = f"{vp:.4f}" if isinstance(vp, (int, float)) else "n/a"
        history_lines.append(
            f'- id="{n.node_id}" model={n.config.model} edge={n.edge_type} parent={n.parent_id or "none"} '
            f'status={n.status} valid_primary={vp_str}\n'
            f'  hypothesis: {n.config.hypothesis[:180]}\n'
            f'  diagnosis [{n.diagnosis_tag}]: {n.insight[:220]}'
        )

    model_docs = "\n".join(f'- "{m}": {_MODEL_HYPERPARAM_DOCS.get(m, "no documented hyperparams")}'
                            for m in sorted(MODELS))
    dead_ends = _dead_ends_section(map)

    return f"""You are the Research Strategist for an autonomous ML research agent working on the
KuaiRand-Pure recommendation benchmark (TikTok TechJam 2026 challenge).

TASK: within-user ranking of short-video impressions. Label = long_view (binary, 0/1).
Metrics = GAUC and nDCG@5; primary = mean(GAUC, nDCG@5). Decisions must be based ONLY on
the VALIDATION split -- the agent never sees the hidden test set during iteration.

OFFICIAL BASELINE (Factorization Machine): valid primary = {baseline['primary']:.4f}
(GAUC={baseline['GAUC']:.4f}, nDCG@5={baseline['nDCG@5']:.4f}).
CURRENT BEST in the Research Map: {summary['best_node_id']} at valid primary = {summary['best_valid_primary']}.

AVAILABLE MODELS (you may ONLY propose one of these model names -- never invent a new one):
{model_docs}

KNOWN DEAD ENDS (generated live from the Research Map's own diagnosis tags, not a hand-maintained
list -- always current as of this call):
{dead_ends}

RESEARCH MAP HISTORY (most recent {len(recent)} of {summary['total_nodes']} total nodes -- read the
diagnosis insights carefully, they explain WHY each result happened, not just what it was):
{chr(10).join(history_lines)}

BUDGET REMAINING: {json.dumps(budget)}

Propose exactly ONE new experiment to try next -- the one you believe is most likely to improve
on the current best, given everything above. Do not repeat a dead end already confirmed in the
history without a genuinely new angle. Respond with a SINGLE JSON object, no other text, matching
exactly this schema:

{{
  "id": "short_snake_case_unique_id_not_already_in_the_history_above",
  "hypothesis": "what to try and why, citing specific evidence from the research map history above",
  "model": "one of the exact model names listed above",
  "hyperparams": {{"k": <int>, "lr": <float>, "l2": <float>, "batch": <int>, "epochs": <int>, "patience": <int>}},
  "parent_id": "an existing node id from the history above this experiment builds on or fixes, or null for a fresh idea",
  "edge_type": "draft (fresh idea) | improve (make a working node better) | debug (fix what broke a failed node)",
  "reasoning": "why this specific choice over other options, referencing the history",
  "expected_metric_effect": {{"gauc": "up|down|flat", "ndcg": "up|down|flat"}},
  "estimated_cost_s": <number, your best guess at wall-clock training time>,
  "priority": <number 0-1, your own confidence this is worth running now>
}}
{retry_note}"""


def _validate_and_build(raw: dict[str, Any], map: ResearchMap) -> ExperimentConfig:
    required = {"id", "hypothesis", "model", "hyperparams", "edge_type", "reasoning",
                "expected_metric_effect", "estimated_cost_s", "priority"}
    missing = required - raw.keys()
    if missing:
        raise ValidationError(f"missing required field(s): {sorted(missing)}")

    model = raw["model"]
    if model not in MODELS:
        raise ValidationError(f"unknown model '{model}' -- must be exactly one of {sorted(MODELS)}")

    node_id = str(raw["id"]).strip()
    if not node_id:
        raise ValidationError("id must be a non-empty string")
    base_id, suffix = node_id, 1
    while node_id in map.nodes:  # de-duplicate against the map rather than reject outright
        suffix += 1
        node_id = f"{base_id}_{suffix}"

    parent_id = raw.get("parent_id")
    if parent_id is not None and parent_id not in map.nodes:
        raise ValidationError(f"parent_id '{parent_id}' does not exist in the Research Map")

    edge_type = raw["edge_type"]
    if edge_type not in ("draft", "improve", "debug"):
        raise ValidationError(f"edge_type must be 'draft', 'improve', or 'debug', got {edge_type!r}")

    hp = raw["hyperparams"]
    if not isinstance(hp, dict):
        raise ValidationError("hyperparams must be a JSON object")
    for numeric_key in ("k", "lr", "l2", "batch", "epochs", "patience"):
        if numeric_key in hp and not isinstance(hp[numeric_key], (int, float)):
            raise ValidationError(f"hyperparams.{numeric_key} must be numeric, got {type(hp[numeric_key]).__name__}")
    if model == "deepfm" and "hidden" in hp:
        h = hp["hidden"]
        if not (isinstance(h, list) and len(h) == 2 and all(isinstance(x, int) for x in h)):
            raise ValidationError("hyperparams.hidden must be a 2-element list of ints for deepfm")

    if not isinstance(raw["expected_metric_effect"], dict):
        raise ValidationError("expected_metric_effect must be a JSON object")

    return ExperimentConfig(
        id=node_id, model=model, hypothesis=str(raw["hypothesis"]),
        hyperparams=hp, parent_id=parent_id, edge_type=edge_type, seeds=[0],
        notes=f"LLM-proposed (Phase 4 Research Strategist). priority={raw.get('priority')}",
    )


def propose_next_experiment(map: ResearchMap, budget: dict[str, Any],
                             client: GeminiClient | None = None,
                             max_validation_retries: int = 2) -> tuple[StrategistProposal | None, dict[str, Any]]:
    """Returns (proposal_or_None, meta). meta always includes 'tokens' (int,
    total across all attempts -- transport retries inside llm_client don't
    duplicate this, but validation-retry re-prompts do call the API again
    and their tokens are counted) and 'events' (list of dicts describing
    what happened, for the Structured Run Log). Never raises -- an LLM or
    validation failure returns (None, meta) for the caller to log and
    handle, matching agent/recovery.py's contract.
    """
    client = client or GeminiClient()
    prompt = _build_prompt(map, budget)
    total_tokens = 0
    events: list[dict[str, Any]] = []

    for attempt in range(1, max_validation_retries + 2):
        try:
            raw, meta = client.generate_json(prompt)
        except LLMError as e:
            events.append({"kind": "llm_error", "attempt": attempt, "message": str(e)})
            return None, {"tokens": total_tokens, "events": events}

        total_tokens += meta.total_tokens
        try:
            config = _validate_and_build(raw, map)
        except ValidationError as e:
            events.append({"kind": "validation_error", "attempt": attempt, "message": str(e), "raw": raw})
            prompt = _build_prompt(
                map, budget,
                retry_note=f"\nYour previous response was rejected: {e}\nFix it and respond again with ONLY the corrected JSON object.",
            )
            continue

        events.append({"kind": "proposal_accepted", "attempt": attempt, "config_id": config.id})
        proposal = StrategistProposal(
            config=config, reasoning=str(raw.get("reasoning", "")),
            expected_metric_effect=raw.get("expected_metric_effect", {}),
            estimated_cost_s=float(raw.get("estimated_cost_s", 0) or 0),
            priority=float(raw.get("priority", 0) or 0),
        )
        return proposal, {"tokens": total_tokens, "events": events}

    events.append({"kind": "validation_exhausted", "attempt": max_validation_retries + 1,
                    "message": "every attempt failed validation"})
    return None, {"tokens": total_tokens, "events": events}
