"""Promotes a single-seed Research Map node to a 3-seed-validated result --
the same discipline Phase 0 applied before trusting deepfm_wider as real
(fm_seed_variance), now reusable for any promising P1/P4 candidate before
it gets trusted as "the new best." A single-seed win can be a lucky draw;
this closes that gap by actually re-running the same config on more seeds
and re-aggregating, rather than assuming.

Usage:
    python tools/verify_multiseed.py <node_id> [--seeds 0,1,2] [--data_dir ...]

Updates the node's `metrics` in logs/research_map.json in place (now a real
n_seeds>=2 aggregate) and re-diagnoses it against its parent using the
now-real seed-aware significance bar (agent/diagnosis.py), so every
downstream consumer (the Best-First Selector's cost/gain estimates, a
future LLM prompt) sees the validated number, not the original single-seed
one.

`verify_node_multiseed()` is the reusable core (this file's CLI is now a
thin wrapper around it) -- `agent/p1_orchestrator.py` and
`agent/p4_orchestrator.py` both call it directly the moment a fresh
single-seed node becomes the new raw-leaderboard best, closing the gap the
README used to list under Limitations ("a human has to remember to run
this"): promotion to 3-seed-verified is now automatic, not manual. Skips
re-running any seed already present in the node's own `per_seed` history
(e.g. seed 0, already trained when the node was first created) unless
`force=True` -- the original CLI used to blindly retrain seed 0 too, which
this fixes as a real, if minor, wasted-compute bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.diagnosis import Diagnosis, diagnose  # noqa: E402
from agent.orchestrator import _aggregate_seed_results  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR  # noqa: E402
from agent.recovery import run_with_recovery  # noqa: E402
from agent.research_map import ResearchMap  # noqa: E402


def verify_node_multiseed(map: ResearchMap, node_id: str, data_dir: str, seeds: list[int] = (0, 1, 2),
                           timeout_s: float = 240.0, force: bool = False,
                           log: "callable | None" = print) -> tuple[dict, Diagnosis, float, list[int]] | None:
    """Runs whichever of `seeds` aren't already in `node_id`'s recorded
    `per_seed` history, merges them with what's already there, re-aggregates,
    and writes the updated metrics + diagnosis back into `map`.

    Returns `(agg_metrics, diagnosis, wall_time_s_spent, seeds_newly_run)`,
    or `None` if `node_id` isn't in the map or a new seed's training failed
    (Failure Recovery already retried/degraded internally by this point --
    a `None` here means it genuinely couldn't get a result, not a transient
    blip). `wall_time_s_spent` is 0.0 and `seeds_newly_run` is `[]` when
    every requested seed was already cached -- callers can fold this
    straight into their own wall-clock budget accounting either way.
    """
    if node_id not in map.nodes:
        if log:
            log(f"'{node_id}' not found in the Research Map.")
        return None
    node = map.get(node_id)
    seeds = [int(s) for s in seeds]

    existing_by_seed = {}
    if not force:
        for r in (node.metrics or {}).get("per_seed", []):
            if "seed" in r:
                existing_by_seed[r["seed"]] = r

    per_seed_results = []
    newly_run = []
    wall_time_s = 0.0
    for seed in seeds:
        if seed in existing_by_seed:
            per_seed_results.append(existing_by_seed[seed])
            continue
        if log:
            log(f"  seed={seed}...")
        result, events = run_with_recovery(node.config, data_dir, seed=seed, timeout_s=timeout_s)
        if result is None:
            if log:
                log(f"  seed={seed} FAILED: {[e.message for e in events]}")
            return None
        result = dict(result, seed=seed)
        per_seed_results.append(result)
        newly_run.append(seed)
        # `result["wall_time_s"]` is measured INSIDE the training subprocess
        # (agent/experiment.py's own time.process_time() call) -- summing
        # that, not timing this function's own process_time() around the
        # run_with_recovery() call, which spawns a separate subprocess
        # (agent/recovery.py, required on Windows). A wrapper's own
        # process_time() only accounts for CPU time the WRAPPER itself
        # burns (mostly idle time.sleep-style waiting on proc.join()), not
        # the child's -- caught live: this returned ~0.03s total for two
        # real multi-minute training runs before being fixed, which would
        # have silently broken agent/p1_orchestrator.py's and
        # agent/p4_orchestrator.py's wall-clock budget accounting (the
        # auto-verification step's real cost would never show up in
        # wall_time_total_s, undermining --max_wall_time_s).
        wall_time_s += result["wall_time_s"]
        if log:
            log(f"    valid primary = {result['valid']['primary']:.4f}")

    agg = _aggregate_seed_results(per_seed_results)
    parent_metrics = map.nodes[node.parent_id].metrics if node.parent_id in map.nodes else None
    d = diagnose(agg, parent_metrics)
    map.update_node(node_id, metrics=agg, insight=d.insight, diagnosis_tag=d.tag)
    return agg, d, wall_time_s, newly_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("node_id")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--timeout_s", type=float, default=240.0)
    ap.add_argument("--force", action="store_true", help="Retrain every seed even if already cached on the node")
    args = ap.parse_args()

    map = ResearchMap(LOGS_DIR / "research_map.json")
    if args.node_id not in map.nodes:
        print(f"'{args.node_id}' not found in the Research Map.")
        return 1
    node = map.get(args.node_id)
    seeds = [int(s) for s in args.seeds.split(",")]
    was_primary = (node.metrics or {}).get("valid", {}).get("primary_mean")
    print(f"Verifying '{args.node_id}' ({node.config.model}) on seeds {seeds}...")

    out = verify_node_multiseed(map, args.node_id, args.data_dir, seeds=seeds,
                                 timeout_s=args.timeout_s, force=args.force)
    if out is None:
        return 1
    agg, d, wall_time_s, newly_run = out

    print(f"\n3-seed result: valid primary = {agg['valid']['primary_mean']:.4f} "
          f"+/- {agg['valid']['primary_std']:.4f} (n={agg['n_seeds']})")
    print(f"Was (single-seed): {was_primary}")
    print(f"Diagnosis vs parent ('{node.parent_id}'): [{d.tag}] {d.insight}")
    print(f"Newly trained seeds: {newly_run} ({wall_time_s:.1f}s CPU time; "
          f"{len(seeds) - len(newly_run)} seed(s) reused from cache)")
    print(f"\nUpdated in logs/research_map.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
