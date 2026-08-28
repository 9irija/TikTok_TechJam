"""Entry point: `python run_p1.py` -- P1's best-first, multi-fidelity round.

    load/seed persistent Research Map -> score candidate pool (Best-First
    Node Selector) -> run each in best-first order through the
    Multi-Fidelity Runner (1%->10%->100%, kill early if broken) ->
    diagnose each result (Metric-Aware Diagnosis Engine) -> record into
    the Research Map, which now carries forward into the *next* invocation

Run `python run.py` first (Phase 0) at least once -- this seeds the
Research Map from its logs/run_log.jsonl so the selector starts with real
historical data instead of nothing. Safe to run without it too; the map
just starts empty.

Still no LLM in the loop (see CLAUDE.md roadmap Phase 4) -- the candidate
pool (agent/p1_orchestrator.py's `p1_candidate_pool`) is hand-authored;
what's new versus `run.py` is the scored, best-first *order* candidates
run in, the persistent cross-run memory, and the staged compute spend.
"""
from __future__ import annotations

import argparse

from agent.p1_orchestrator import run_p1
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--timeout_s", type=float, default=240.0,
                     help="Per-fidelity-stage wall-clock budget before Failure Recovery kills and retries it")
    args = ap.parse_args()

    print("=" * 78)
    print("P1 -- Research Map + Best-First Selection + Multi-Fidelity Runner")
    print("=" * 78)

    report = run_p1(data_dir=args.data_dir, timeout_s=args.timeout_s)

    print(f"\nSeeded {report['seeded_from_phase0']} node(s) from Phase 0's run_log.jsonl "
          f"(0 means the map already had history, or no Phase 0 run exists yet).")
    print(f"\nCandidate pool: {report['candidates_considered']} not-yet-tried candidate(s), "
          f"best-first ranked:")
    for r in report["ranking"]:
        print(f"  #{report['ranking'].index(r) + 1} score={r['score']:.6f}  {r['id']}")
        print(f"      {r['reasoning']}")

    print("\nResults, in the order actually run (best-first, not declaration order):")
    for r in report["results"]:
        print(f"\n  [{r['rank']}] {r['config_id']} -- status={r['status']}, "
              f"reached fidelity={r['final_stage']}, wall_time={r['wall_time_s']:.1f}s")
        if r["status"] == "failed":
            print(f"      KILLED at {r['killed_at']}: {r['kill_reason']}")
        print(f"      diagnosis [{r['diagnosis']['tag']}]: {r['diagnosis']['insight']}")

    s = report["research_map_summary"]
    print(f"\nResearch Map now has {s['total_nodes']} total node(s). Best known: "
          f"{s['best_node_id']} (valid primary={s['best_valid_primary']})")
    print(f"By model family: {s['by_model']}")

    print("\n" + "=" * 78)
    print(f"Round report: {LOGS_DIR / 'p1_round_report.json'}")
    print(f"Research Map: {LOGS_DIR / 'research_map.json'} (persistent -- carries into the next run_p1.py)")
    print(f"Structured logs (same format as Phase 0): {LOGS_DIR / 'run_log.jsonl'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
