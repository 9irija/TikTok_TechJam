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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.diagnosis import diagnose  # noqa: E402
from agent.orchestrator import _aggregate_seed_results  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR  # noqa: E402
from agent.recovery import run_with_recovery  # noqa: E402
from agent.research_map import ResearchMap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("node_id")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--timeout_s", type=float, default=240.0)
    args = ap.parse_args()

    map = ResearchMap(LOGS_DIR / "research_map.json")
    if args.node_id not in map.nodes:
        print(f"'{args.node_id}' not found in the Research Map.")
        return 1
    node = map.get(args.node_id)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Re-running '{args.node_id}' ({node.config.model}) on seeds {seeds}...")

    per_seed_results = []
    for seed in seeds:
        print(f"  seed={seed}...")
        result, events = run_with_recovery(node.config, args.data_dir, seed=seed, timeout_s=args.timeout_s)
        if result is None:
            print(f"  seed={seed} FAILED: {[e.message for e in events]}")
            return 1
        per_seed_results.append(result)
        print(f"    valid primary = {result['valid']['primary']:.4f}")

    agg = _aggregate_seed_results(per_seed_results)
    parent_metrics = map.nodes[node.parent_id].metrics if node.parent_id in map.nodes else None
    d = diagnose(agg, parent_metrics)

    print(f"\n3-seed result: valid primary = {agg['valid']['primary_mean']:.4f} "
          f"+/- {agg['valid']['primary_std']:.4f} (n={agg['n_seeds']})")
    print(f"Was (single-seed): {node.metrics['valid']['primary_mean']:.4f}")
    print(f"Diagnosis vs parent ('{node.parent_id}'): [{d.tag}] {d.insight}")

    map.update_node(args.node_id, metrics=agg, insight=d.insight, diagnosis_tag=d.tag)
    print(f"\nUpdated in logs/research_map.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
