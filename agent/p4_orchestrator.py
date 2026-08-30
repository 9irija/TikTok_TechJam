"""P4 Orchestrator: the LLM Research Strategist driving the same loop P1 built.

This is deliberately a thin layer -- everything except "where does the next
candidate come from" is reused unchanged from P1:

    P1: candidate <- p1_candidate_pool() (hand-authored) or
                      _diagnosis_driven_candidates() (hand-authored trigger)
    P4: candidate <- propose_next_experiment() (LLM call)

Both feed the exact same Multi-Fidelity Runner (1%->10%->100%, kill early
if broken) and Metric-Aware Diagnosis Engine, and write to the exact same
persistent Research Map and Structured Run Log. This is the payoff of
having kept "what happened" (research_map.py), "why" (diagnosis.py), and
"what next" (selector.py / research_strategist.py) as separate, swappable
modules since P1 -- swapping the candidate source didn't require touching
any of the execution machinery.

`llm_tokens_total` in the resulting run_summary-style report is now REAL
(summed from every Gemini call this run made), not structurally zero --
this is the first place in the whole project that number means something.

Budget-aware stopping (closes a real, previously-documented gap): the
`budget` dict shown to the LLM in every prompt used to be purely
informational -- nothing in this loop actually changed behavior as it
depleted, despite the brainstorm doc's own explicit user story ("As the
agent, I want to track my own cumulative token usage and GPU-hours per
iteration, so I can flag when I'm approaching budget and prioritize
higher-expected-value experiments"). Two real, checked stop conditions
now exist alongside `max_iterations`, mirroring `agent/convergence.py`'s
own MAX_ITERATIONS/MAX_WALL_CLOCK_S pattern for Phase 0's loop: a wall-
clock ceiling (`max_wall_time_s`) checked BEFORE spending an LLM call on
the next proposal, and a priority floor (`min_priority_to_run`) -- if the
LLM's own returned `priority` for a proposal is below this, the loop
declines to spend Multi-Fidelity Runner compute on it (logged as
`skipped_low_priority`, the same "declined to waste budget, didn't fail
to have the idea" pattern `critic_rejected` already established) and asks
for a fresh proposal on the next iteration instead. Both default to None/0
(no behavior change from before) so existing callers are unaffected.

Auto-triggered multi-seed verification (closes another real, previously-
documented gap -- README used to list "`tools/verify_multiseed.py` is a
manual step, not auto-triggered when a new best node appears"): the moment
an LLM proposal survives on a single seed AND becomes the new raw-
leaderboard best (`map.best_node()`), this loop immediately re-runs it on
2 more seeds via `tools.verify_multiseed.verify_node_multiseed()` and
re-diagnoses before the *next* prompt's Research Map context is built --
so the Strategist never reasons from an unverified single-seed number as
if it were trustworthy. `auto_verify_new_best=True` by default; the extra
wall-clock cost is folded into `wall_time_total_s` (and therefore into the
`max_wall_time_s` budget check above) rather than hidden.
"""
from __future__ import annotations

import json
from typing import Any

from .diagnosis import diagnose
from .llm_client import GeminiClient
from .multi_fidelity import run_multi_fidelity
from .p1_orchestrator import seed_from_phase0
from .paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR
from .research_critic import review as critic_review
from .research_map import ResearchMap
from .research_strategist import propose_next_experiment
from .run_logger import RunLogger

# tools/ is a plain sibling directory, resolvable as a namespace package as
# long as REPO_ROOT is on sys.path (true whenever this is reached via
# run_p4.py from the repo root -- see README's Quick start).
from tools.verify_multiseed import verify_node_multiseed

RESEARCH_MAP_PATH = LOGS_DIR / "research_map.json"


def run_p4(data_dir: str | None = None, timeout_s: float = 240.0,
           max_iterations: int = 3, client: GeminiClient | None = None,
           max_wall_time_s: float | None = None, min_priority_to_run: float = 0.0,
           auto_verify_new_best: bool = True) -> dict[str, Any]:
    data_dir = data_dir or str(DEFAULT_DATA_DIR)
    map = ResearchMap(RESEARCH_MAP_PATH)
    seeded = seed_from_phase0(map, LOGS_DIR / "run_log.jsonl")

    logger = RunLogger(LOGS_DIR, EXPERIMENTS_DIR)
    client = client or GeminiClient()
    total_llm_tokens = 0
    wall_time_total_s = 0.0

    report: dict[str, Any] = {
        "run_id": logger.run_id, "seeded_from_phase0": seeded,
        "max_iterations": max_iterations, "max_wall_time_s": max_wall_time_s,
        "min_priority_to_run": min_priority_to_run, "iterations": [],
    }

    for i in range(1, max_iterations + 1):
        if max_wall_time_s is not None and wall_time_total_s >= max_wall_time_s:
            report["iterations"].append({
                "iteration": i, "status": "budget_exhausted_wall_time",
                "wall_time_used_so_far_s": wall_time_total_s, "max_wall_time_s": max_wall_time_s,
            })
            break

        budget = {"iterations_remaining": max_iterations - i + 1,
                  "llm_tokens_used_so_far": total_llm_tokens,
                  "wall_time_used_so_far_s": wall_time_total_s,
                  "max_wall_time_s": max_wall_time_s}
        proposal, meta = propose_next_experiment(map, budget, client=client)
        total_llm_tokens += meta.get("tokens", 0)

        if proposal is None:
            # The Research Strategist itself failed (transport or
            # validation exhausted) -- log it and stop cleanly rather than
            # crash or loop forever on a broken LLM call, matching
            # agent/recovery.py's philosophy one level up the stack.
            report["iterations"].append({
                "iteration": i, "status": "strategist_failed",
                "llm_tokens_this_call": meta.get("tokens", 0), "events": meta.get("events", []),
            })
            break

        config = proposal.config

        if proposal.priority < min_priority_to_run:
            # The LLM itself signaled low confidence -- decline to spend compute on it rather than
            # run everything unconditionally, same "declined to waste budget" pattern as critic_rejected.
            report["iterations"].append({
                "iteration": i, "config_id": config.id, "status": "skipped_low_priority",
                "llm_reasoning": proposal.reasoning, "llm_priority": proposal.priority,
                "min_priority_to_run": min_priority_to_run, "llm_tokens_this_call": meta.get("tokens", 0),
                "final_stage": None, "killed_at": None, "kill_reason": None,
                "diagnosis": {"tag": "skipped_low_priority",
                              "insight": f"LLM's own priority ({proposal.priority}) was below the "
                                         f"{min_priority_to_run} floor -- declined to spend compute."},
                "wall_time_s": 0.0, "estimated_time_saved_s": None,
            })
            continue

        verdict = critic_review(map, config)
        if not verdict.approved:
            # The LLM's own prompt already lists confirmed dead ends
            # (agent/research_strategist.py's _MODEL_HYPERPARAM_DOCS), but
            # the critic is the hard backstop for when it proposes one
            # anyway -- rejected before spending any compute, and the
            # rejection itself is logged (audit trail: the agent declined
            # to waste budget, it didn't fail to have the idea).
            report["iterations"].append({
                "iteration": i, "config_id": config.id, "status": "critic_rejected",
                "llm_reasoning": proposal.reasoning, "llm_tokens_this_call": meta.get("tokens", 0),
                "final_stage": None, "killed_at": None, "kill_reason": None,
                "diagnosis": {"tag": "critic_rejected", "insight": verdict.reason},
                "wall_time_s": 0.0, "estimated_time_saved_s": None,
            })
            continue

        iteration_id = logger.next_iteration_id()
        predictions_path = str(logger.experiments_dir / iteration_id / "predictions.npz")

        mf_result = run_multi_fidelity(config, data_dir, timeout_s=timeout_s,
                                        save_predictions_on_survive=predictions_path)
        wall_time_total_s += mf_result.total_wall_time_s

        status = "ok" if mf_result.survived() else "failed"
        metrics = mf_result.final_metrics() if mf_result.survived() else None
        parent_metrics = map.nodes[config.parent_id].metrics if config.parent_id in map.nodes else None
        d = diagnose(metrics, parent_metrics) if status == "ok" else diagnose(None, None)

        map.add_node(config, edge_type=(config.edge_type or "draft"), parent_id=config.parent_id)
        map.update_node(config.id, status=("done" if status == "ok" else "killed"),
                         metrics=metrics, fidelity=mf_result.final_stage,
                         insight=d.insight, diagnosis_tag=d.tag)

        fidelity_info = mf_result.to_fidelity_info(config)
        logger.log_iteration(iteration_id, config, metrics=metrics, wall_time_s=mf_result.total_wall_time_s,
                              recovery_events=mf_result.all_events, convergence_point=None, status=status,
                              fidelity_info=fidelity_info)

        iter_wall_time_s = mf_result.total_wall_time_s
        report_diagnosis = {"tag": d.tag, "insight": d.insight}
        auto_verified = None

        best = map.best_node()
        if (auto_verify_new_best and status == "ok" and metrics.get("n_seeds", 1) == 1
                and best is not None and best.node_id == config.id):
            # The LLM's proposal just became the new raw-leaderboard best on
            # a single seed -- verify it on 2 more before the *next* prompt's
            # Research Map context reports it as trustworthy, same discipline
            # agent/p1_orchestrator.py now applies (closes the README's
            # former Limitations entry: "a human has to remember to run
            # tools/verify_multiseed.py").
            out = verify_node_multiseed(map, config.id, data_dir, seeds=[0, 1, 2], timeout_s=timeout_s)
            if out is not None:
                agg, d2, extra_wall_s, newly_run = out
                iter_wall_time_s += extra_wall_s
                report_diagnosis = {"tag": d2.tag, "insight": d2.insight}
                auto_verified = {
                    "seeds_newly_run": newly_run, "n_seeds": agg["n_seeds"],
                    "valid_primary_mean": agg["valid"]["primary_mean"], "valid_primary_std": agg["valid"]["primary_std"],
                }
        wall_time_total_s += (iter_wall_time_s - mf_result.total_wall_time_s)  # add only the auto-verify extra

        entry = {
            "iteration": i, "config_id": config.id, "status": status,
            "llm_reasoning": proposal.reasoning, "llm_expected_metric_effect": proposal.expected_metric_effect,
            "llm_priority": proposal.priority, "llm_estimated_cost_s": proposal.estimated_cost_s,
            "llm_tokens_this_call": meta.get("tokens", 0),
            "final_stage": mf_result.final_stage, "killed_at": mf_result.killed_at,
            "kill_reason": mf_result.kill_reason,
            "diagnosis": report_diagnosis,
            "wall_time_s": iter_wall_time_s,
            "estimated_time_saved_s": fidelity_info["estimated_time_saved_s"],
        }
        if auto_verified is not None:
            entry["auto_verified_multiseed"] = auto_verified
        report["iterations"].append(entry)

    total_time_saved = sum(it.get("estimated_time_saved_s") or 0.0 for it in report["iterations"])
    report["resource_totals"] = {
        "llm_tokens_total": total_llm_tokens,
        "wall_time_total_s": wall_time_total_s,
        "gpu_hours_total": 0.0,
        "estimated_time_saved_by_early_termination_s": total_time_saved,
    }
    report["research_map_summary"] = map.explored_summary()
    (LOGS_DIR / "p4_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
