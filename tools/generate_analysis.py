"""Generates the human-readable analysis report required by Deliverables
item 3 (Run & Iteration Logs) from the machine-readable logs/run_log.jsonl +
logs/run_summary.json + logs/manual_interventions.jsonl that agent/run_logger.py
writes during a run.

Usage:
    python tools/generate_analysis.py                 # writes logs/analysis_report.md
    python tools/generate_analysis.py --stdout         # also prints it

This file only reads and formats what agent/run_logger.py already wrote --
it must never recompute a metric itself (that would risk drifting from the
Evaluator Wrapper's numbers), only present them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sparkline(values: list[float]) -> str:
    if not values:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return blocks[4] * len(values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)


def _fmt_delta(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.4f}"


def build_report(logs_dir: Path = LOGS_DIR) -> str:
    entries = _load_jsonl(logs_dir / "run_log.jsonl")
    summary = _load_json(logs_dir / "run_summary.json")
    manual = _load_jsonl(logs_dir / "manual_interventions.jsonl")

    lines: list[str] = []
    lines.append("# Run & Iteration Log -- Analysis Report")
    lines.append("")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from "
                  f"`{(logs_dir / 'run_log.jsonl').relative_to(REPO_ROOT)}`_")
    lines.append("")

    if not entries:
        lines.append("No iterations logged yet. Run `python run.py` first.")
        return "\n".join(lines)

    run_id = entries[0]["run_id"]
    ok_entries = [e for e in entries if e["status"] == "ok"]
    failed_entries = [e for e in entries if e["status"] == "failed"]

    # ---------------------------------------------------------------- summary
    lines.append("## Run overview")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Iterations:** {len(entries)} total -- {len(ok_entries)} succeeded, "
                  f"{len(failed_entries)} failed after recovery was exhausted")
    if summary:
        conv = summary["convergence"]
        lines.append(f"- **Converged:** {conv['converged']} "
                      f"(epsilon={conv['epsilon']}, N={conv['N']}, per organizer's baseline_scores.json)")
        if conv["best_iteration_id"]:
            lines.append(f"- **Validation-best:** `{conv['best_iteration_id']}` "
                          f"(valid primary = {conv['best_valid_primary']:.4f})")
        lines.append(f"- **Manual interventions:** {summary['manual_intervention_count']} "
                      "(lower is better -- this is what judges use to score Autonomy)")
        rt = summary["resource_totals"]
        lines.append(f"- **Resource usage:** wall-clock {rt['wall_time_total_s']:.1f}s total | "
                      f"LLM tokens {rt['llm_tokens_total']} | GPU-hours {rt['gpu_hours_total']}")
        if summary.get("best_submission_metrics"):
            bsm = summary["best_submission_metrics"]
            d = bsm["delta_vs_baseline_test"]
            lines.append(f"- **Final submission ({bsm['config_id']}), hidden-test-style split:** "
                          f"GAUC {bsm['test']['GAUC']:.4f} ({_fmt_delta(d['delta_GAUC'])}), "
                          f"nDCG@5 {bsm['test']['nDCG@5']:.4f} ({_fmt_delta(d['delta_nDCG@5'])}), "
                          f"primary-metric delta {_fmt_delta(d['delta_primary_mean'])}")
    lines.append("")

    # ------------------------------------------------------------ trajectory
    valid_primaries = [
        e["metrics"]["valid"]["primary_mean"] for e in ok_entries if e.get("metrics")
    ]
    if valid_primaries:
        lines.append("## Validation-primary trajectory")
        lines.append("")
        lines.append(f"```\n{_sparkline(valid_primaries)}\n"
                      f"min={min(valid_primaries):.4f}  max={max(valid_primaries):.4f}  "
                      f"n={len(valid_primaries)}\n```")
        try:
            from agent.evaluator import load_baseline_scores  # noqa: E402
            sys.path.insert(0, str(REPO_ROOT))
            baseline_valid_primary = load_baseline_scores()["scores"]["fm_official"]["valid"]["primary"]
            lines.append(f"\nOfficial FM baseline (validation): **{baseline_valid_primary:.4f}** "
                          f"-- reference line for the trajectory above.")
        except Exception:
            pass
        lines.append("")

    # ------------------------------------------------------------- iterations
    lines.append("## Per-iteration log")
    lines.append("")
    lines.append("| # | Config | Model | Status | Valid primary | Δ vs baseline (mean) | "
                  "Wall time | Recovery events | Hypothesis |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in entries:
        m = e.get("metrics")
        valid_p = f"{m['valid']['primary_mean']:.4f}" if m else "--"
        delta = f"{_fmt_delta(m['delta_vs_baseline_test']['delta_primary_mean'])}" if m else "--"
        n_events = len(e.get("recovery_events") or [])
        hyp = (e["hypothesis"][:100] + "...") if len(e["hypothesis"]) > 100 else e["hypothesis"]
        hyp = hyp.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {e['iteration_id']} | `{e['config_id']}` | {e['model']} | {e['status']} | "
                      f"{valid_p} | {delta} | {e['wall_time_s']:.1f}s | {n_events} | {hyp} |")
    lines.append("")
    lines.append("_Δ vs baseline is computed on the locally-held test split for tracking parity with "
                  "the organizer's own baseline.py; it is never used to pick a config -- only the "
                  "Valid primary column drives Convergence Detector / config-selection decisions, "
                  "per Task Requirement 2 (train+validation only)._")
    lines.append("")

    # -------------------------------------------------------- code diffs
    lines.append("## Code diffs (structured-config)")
    lines.append("")
    for e in entries:
        diff = e.get("code_diff") or {}
        if diff.get("__root__"):
            lines.append(f"- **{e['iteration_id']}** (`{e['config_id']}`): root config, no prior iteration to diff against.")
            continue
        if not diff:
            continue
        changes = ", ".join(f"`{k}`: {v['before']} → {v['after']}" for k, v in diff.items())
        lines.append(f"- **{e['iteration_id']}** (`{e['config_id']}`): {changes}")
    lines.append("")

    # --------------------------------------------------------- errors/recovery
    all_events = [(e["iteration_id"], ev) for e in entries for ev in (e.get("recovery_events") or [])]
    lines.append("## Error / recovery events")
    lines.append("")
    if not all_events:
        lines.append("None -- every iteration completed on its first attempt.")
    else:
        for iter_id, ev in all_events:
            lines.append(f"- **{iter_id}** [{ev['kind']}] attempt {ev['attempt']}: {ev['message']}")
    lines.append("")

    # ------------------------------------------------------------- manual
    lines.append("## Manual interventions")
    lines.append("")
    if not manual:
        lines.append("None recorded -- fully autonomous run.")
    else:
        for m in manual:
            ts = time.strftime("%H:%M:%S", time.localtime(m["timestamp"]))
            lines.append(f"- [{ts}] iteration `{m.get('iteration_id') or '(run-level)'}`: {m['reason']}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default=str(LOGS_DIR))
    ap.add_argument("--out", default=str(LOGS_DIR / "analysis_report.md"))
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    report = build_report(Path(args.logs_dir))
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.stdout:
        try:
            print("\n" + report)
        except UnicodeEncodeError:
            # Windows console default codepage (cp1252) can't render the
            # sparkline's unicode block characters or the delta sign -- the
            # file itself is UTF-8 and unaffected; degrade gracefully here
            # instead of crashing a report the user explicitly asked to see.
            print("\n" + report.encode("ascii", errors="replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
