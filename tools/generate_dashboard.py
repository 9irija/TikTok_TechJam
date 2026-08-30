"""Regenerate docs/dashboard.html's NODES array from logs/research_map.json.

Why this exists: the dashboard went stale once already (commit 0548327 --
"still showed deepfm_regularized as best" after deepfm_mtl_v1 had already
taken over) because every field in NODES was hand-transcribed from the
Research Map and nothing forced them to stay in sync. This script derives
everything that *is* recoverable from ResearchMap/diagnosis.py verbatim
(id, model, edge, parent, valid, delta, insight) so those fields can never
drift from logs/research_map.json again.

Two fields genuinely aren't in the ResearchMap schema and can't be derived:
`phase` (P0/P1/P4/P2 -- which build stage introduced the node) and `llm`
(whether the LLM Research Strategist proposed it, vs. hand-authored). Rather
than guess at those with a heuristic, PHASE_OVERRIDES below is an explicit,
required table -- a new node with no entry here fails loudly instead of
silently rendering wrong, matching this project's existing "no invented
thresholds, no silent behavior" discipline.

Usage: python tools/generate_dashboard.py [--check]
  --check   exit 1 if docs/dashboard.html's NODES array is out of date,
            without writing anything (for a pre-demo sanity check).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.research_map import ResearchMap  # noqa: E402

MAP_PATH = REPO_ROOT / "logs" / "research_map.json"
DASHBOARD_PATH = REPO_ROOT / "docs" / "dashboard.html"

# node_id -> (phase, llm). Required for every node -- see module docstring.
PHASE_OVERRIDES: dict[str, tuple[str, bool]] = {
    "fm_baseline_repro": ("P0", False),
    "fm_seed_variance": ("P0", False),
    "deepfm_default": ("P0", False),
    "deepfm_wider": ("P0", False),
    "fm_bpr_default": ("P1", False),
    "fm_wider_k32": ("P1", False),
    "fm_bpr_regularized": ("P1", False),
    "fm_bpr_slow_and_steady": ("P1", False),
    "deepfm_regularized": ("P4", True),
    "deepfm_higher_l2": ("P4", True),
    "lgbm_baseline": ("P2", False),
    "features_v1": ("P2", False),
    "deepfm_mtl_v1": ("P2", False),
    "deepfm_mtl_v1_hpo": ("P2", False),
    "deepfm_din_v1": ("P2", False),
    "deepfm_bpr_v1": ("P2", False),
    "deepfm_bpr_v1_regularized": ("P2", False),
    "deepfm_mtl_watch_v1": ("P2", False),
    "deepfm_din_mtl_v1": ("P2", False),
    "deepfm_mtl_uncertainty_v1": ("P2", False),
    "deepfm_listwise_v1": ("P2", False),
    "deepfm_pdaom_v1": ("P2", False),
    "deepfm_mtl_pcgrad_v1": ("P2", False),
    "deepfm_mtl_click_v1": ("P2", False),
    "deepfm_mtl_focal_v1": ("P2", False),
    "deepfm_lambdarank_v1": ("P2", False),
    "dcnv2_v1": ("P2", False),
    "deepfm_mtl_deep_heads_v1": ("P2", False),
    "deepfm_mtl_gnn_init_v1": ("P2", False),
    "deepfm_mtl_gnn_feature_v1": ("P2", False),
    "deepfm_mtl_aux_weight_tuning": ("P4", True),
    "deepfm_mtl_focal_soft_v1": ("P4", True),
    "deepfm_mtl_capacity_v1": ("P4", True),
}

# diagnosis_tag (agent/diagnosis.py) -> (dashboard visual tag, tagLabel)
TAG_MAP: dict[str, tuple[str, str]] = {
    "baseline_beat": ("good", "beats baseline"),
    "baseline_miss": ("critical", "misses baseline"),
    "baseline_match": ("info", "baseline match"),
    "clear_improvement": ("good", "clear improvement"),
    "regression": ("critical", "regression"),
    "noise_floor": ("neutral", "noise floor"),
    "mixed": ("warning", "mixed"),
    "ranking_tradeoff": ("warning", "ranking tradeoff"),
    "no_result": ("critical", "no result"),
    "parent_no_result": ("critical", "parent no result"),
}


def js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_nodes_js() -> tuple[str, str]:
    rmap = ResearchMap(MAP_PATH)
    nodes = sorted(rmap.nodes.values(), key=lambda n: n.created_at)

    missing = [n.node_id for n in nodes if n.node_id not in PHASE_OVERRIDES]
    if missing:
        raise SystemExit(
            f"tools/generate_dashboard.py: PHASE_OVERRIDES is missing an entry for: {missing}. "
            f"Add (phase, llm) for each before regenerating -- see module docstring."
        )

    valid_by_id = {
        n.node_id: (n.metrics or {}).get("valid", {}).get("primary_mean")
        for n in nodes
    }

    lines = []
    for n in nodes:
        valid = valid_by_id.get(n.node_id)
        parent_valid = valid_by_id.get(n.parent_id) if n.parent_id else None
        delta = round(valid - parent_valid, 4) if (valid is not None and parent_valid is not None) else None
        tag, tag_label = TAG_MAP.get(n.diagnosis_tag, ("warning", n.diagnosis_tag or "unknown"))
        phase, llm = PHASE_OVERRIDES[n.node_id]

        lines.append(
            "    { id:\"%s\", model:\"%s\", edge:\"%s\", parent:%s, phase:\"%s\", llm:%s,\n"
            "      valid:%s, delta:%s, tag:\"%s\", tagLabel:\"%s\",\n"
            "      hyp:\"%s\",\n"
            "      insight:\"%s\" }," % (
                n.node_id, n.config.model, n.edge_type,
                ("\"%s\"" % n.parent_id) if n.parent_id else "null",
                phase, "true" if llm else "false",
                (round(valid, 4) if valid is not None else "null"),
                (delta if delta is not None else "null"),
                tag, tag_label,
                js_escape(n.config.hypothesis),
                js_escape(n.insight),
            )
        )

    best = rmap.best_confirmed_node()
    best_id = best.node_id if best else ""

    # "Nodes explored" stat tile -- same staleness class as NODES itself,
    # and fully derivable from PHASE_OVERRIDES, so no reason to hand-edit it.
    phase_label = {"P0": "Foundation", "P1": "Differentiators", "P4": "LLM-proposed", "P2": "P2"}
    counts: dict[str, int] = {}
    for n in nodes:
        phase, llm = PHASE_OVERRIDES[n.node_id]
        label = phase_label["P4"] if (phase == "P4" and llm) else phase_label.get(phase, phase)
        counts[label] = counts.get(label, 0) + 1
    order = ["Foundation", "Differentiators", "LLM-proposed", "P2"]
    breakdown = " · ".join(f"{counts[k]} {k}" for k in order if k in counts)
    breakdown += "".join(f" · {v} {k}" for k, v in counts.items() if k not in order)

    return "\n".join(lines), best_id, len(nodes), breakdown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                     help="exit 1 if the dashboard is stale, write nothing")
    args = ap.parse_args()

    nodes_js, best_id, n_total, breakdown = build_nodes_js()
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    new_nodes_block = "  var NODES = [\n%s\n  ];" % nodes_js
    html_new = re.sub(r"  var NODES = \[.*?\n  \];", lambda _m: new_nodes_block, html, count=1, flags=re.S)
    html_new = re.sub(r'var BEST_ID = "[^"]*";', 'var BEST_ID = "%s";' % best_id, html_new, count=1)
    html_new = re.sub(
        r'(<div class="label">Nodes explored</div>\s*<div class="value tnum">)\d+(</div>\s*<div class="sub">)[^<]*(</div>)',
        lambda m: f"{m.group(1)}{n_total}{m.group(2)}{breakdown}{m.group(3)}",
        html_new, count=1,
    )
    # The two chart aria-labels each cite the node count in prose -- caught going stale
    # once already (still said "14" at 30 real nodes) since nothing kept them in sync.
    html_new = re.sub(
        r'(aria-label="Validation primary score for each of the )\d+( experiment nodes)',
        lambda m: f"{m.group(1)}{n_total}{m.group(2)}", html_new, count=1,
    )
    html_new = re.sub(
        r'(aria-label="Tree of )(?:\d+ )?(experiment nodes)',
        lambda m: f"{m.group(1)}{n_total} {m.group(2)}", html_new, count=1,
    )

    if args.check:
        if html_new != html:
            print("docs/dashboard.html is STALE relative to logs/research_map.json.")
            return 1
        print("docs/dashboard.html is up to date.")
        return 0

    if html_new == html:
        print("docs/dashboard.html already up to date -- no changes written.")
        return 0

    DASHBOARD_PATH.write_text(html_new, encoding="utf-8")
    print(f"Regenerated docs/dashboard.html ({n_total} nodes, BEST_ID={best_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
