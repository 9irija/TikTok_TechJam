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

RESEARCH_MAP_PATH = LOGS_DIR / "research_map.json"


def run_p4(data_dir: str | None = None, timeout_s: float = 240.0,
           max_iterations: int = 3, client: GeminiClient | None = None) -> dict[str, Any]:
    data_dir = data_dir or str(DEFAULT_DATA_DIR)
    map = ResearchMap(RESEARCH_MAP_PATH)
    seeded = seed_from_phase0(map, LOGS_DIR / "run_log.jsonl")

    logger = RunLogger(LOGS_DIR, EXPERIMENTS_DIR)
    client = client or GeminiClient()
    total_llm_tokens = 0
    wall_time_total_s = 0.0

    report: dict[str, Any] = {
        "run_id": logger.run_id, "seeded_from_phase0": seeded,
        "max_iterations": max_iterations, "iterations": [],
    }

    for i in range(1, max_iterations + 1):
        budget = {"iterations_remaining": max_iterations - i + 1,
                  "llm_tokens_used_so_far": total_llm_tokens,
                  "wall_time_used_so_far_s": wall_time_total_s}
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

        report["iterations"].append({
            "iteration": i, "config_id": config.id, "status": status,
            "llm_reasoning": proposal.reasoning, "llm_expected_metric_effect": proposal.expected_metric_effect,
            "llm_priority": proposal.priority, "llm_estimated_cost_s": proposal.estimated_cost_s,
            "llm_tokens_this_call": meta.get("tokens", 0),
            "final_stage": mf_result.final_stage, "killed_at": mf_result.killed_at,
            "kill_reason": mf_result.kill_reason,
            "diagnosis": {"tag": d.tag, "insight": d.insight},
            "wall_time_s": mf_result.total_wall_time_s,
            "estimated_time_saved_s": fidelity_info["estimated_time_saved_s"],
        })

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
