"""Entry point: `python run_p4.py` -- Phase 4's LLM-driven Research Strategist.

    load/seed persistent Research Map -> for each iteration: ask the LLM
    (Gemini) to propose ONE new experiment given the full map history and
    remaining budget -> validate the proposal against the model registry
    -> run it through the same Multi-Fidelity Runner P1 built -> diagnose
    the result -> record it into the Research Map, which the *next*
    iteration's prompt will see

Requires a Gemini API key: copy .env.example to .env and fill in
GEMINI_API_KEY (free at https://aistudio.google.com/apikey). Run `python
run.py` (Phase 0) at least once first -- this seeds the Research Map from
its logs/run_log.jsonl so the Strategist starts with real historical
context instead of nothing.

This is the first entry point in the project where LLM tokens are actually
spent -- `resource_totals.llm_tokens_total` in the resulting
logs/p4_run_report.json is real, not structurally zero.
"""
from __future__ import annotations

import argparse

from agent.p4_orchestrator import run_p4
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--timeout_s", type=float, default=240.0,
                     help="Per-fidelity-stage wall-clock budget before Failure Recovery kills and retries it")
    ap.add_argument("--max_iterations", type=int, default=3,
                     help="How many LLM-proposed experiments to run this invocation")
    args = ap.parse_args()

    print("=" * 78)
    print("P4 -- LLM Research Strategist (Gemini)")
    print("=" * 78)

    report = run_p4(data_dir=args.data_dir, timeout_s=args.timeout_s, max_iterations=args.max_iterations)

    print(f"\nSeeded {report['seeded_from_phase0']} node(s) from Phase 0's run_log.jsonl.")
    for it in report["iterations"]:
        if it["status"] == "strategist_failed":
            print(f"\n[{it['iteration']}] LLM Research Strategist failed to produce a valid proposal:")
            for e in it["events"]:
                print(f"    [{e['kind']}] {e.get('message', '')}")
            break
        print(f"\n[{it['iteration']}] {it['config_id']} -- status={it['status']}, "
              f"reached fidelity={it['final_stage']}, wall_time={it['wall_time_s']:.1f}s, "
              f"llm_tokens={it['llm_tokens_this_call']}")
        print(f"    LLM reasoning: {it['llm_reasoning']}")
        print(f"    LLM expected effect: {it['llm_expected_metric_effect']} "
              f"(priority={it['llm_priority']}, est. cost={it['llm_estimated_cost_s']}s)")
        if it["status"] == "failed":
            print(f"    KILLED at {it['killed_at']}: {it['kill_reason']}")
        print(f"    diagnosis [{it['diagnosis']['tag']}]: {it['diagnosis']['insight']}")

    rt = report["resource_totals"]
    s = report["research_map_summary"]
    print(f"\nResource usage this run -- LLM tokens: {rt['llm_tokens_total']} | "
          f"wall-clock: {rt['wall_time_total_s']:.1f}s | GPU-hours: {rt['gpu_hours_total']}")
    print(f"Research Map now has {s['total_nodes']} total node(s). Best known: "
          f"{s['best_node_id']} (valid primary={s['best_valid_primary']})")

    print("\n" + "=" * 78)
    print(f"Round report: {LOGS_DIR / 'p4_run_report.json'}")
    print(f"Research Map: {LOGS_DIR / 'research_map.json'} (shared with P1 -- carries into any future run)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
