"""Research Map / Experiment Tree (P1).

The real behavioral shift from Phase 0: `logs/run_log.jsonl` is a per-run
audit trail (append-only, but nothing reads it back *into* a running
agent's decisions) -- the Research Map is **persistent, cross-run
memory** that the Best-First Node Selector (agent/selector.py) reads to
decide what to try next. Run the agent five times over a week and it should
get five times smarter about this dataset, not run the same four
experiments five times.

Modeled directly on AIDE (arXiv:2502.13138): a node is a full pipeline
config (we reuse `ExperimentConfig` verbatim -- no new schema), and every
edge is one of three types, matching AIDE's Solution Generator modes:

    draft   -- a fresh idea, only loosely related to its parent (or no
               parent at all): explores a new region of the solution space.
    improve -- a modification of a *working* node, intended to make a good
               thing better (a hyperparameter tweak, a wider net, ...).
    debug   -- a modification of a *failed* node, intended to fix whatever
               broke it (not "try something else" -- specifically address
               the failure).

This module only stores and queries the tree. Turning "what should we try
next" into an actual pick is agent/selector.py's job; turning a raw result
into an `insight` string is agent/diagnosis.py's job. Keeping these three
concerns in separate files is deliberate -- AI-Scientist-v2's core lesson
(cited in CLAUDE.md) is that the LLM call that will eventually write code
for a candidate must not also be the one deciding the tree's strategic
direction; keeping "what happened" (this file), "why" (diagnosis.py), and
"what next" (selector.py) as separate, swappable modules is what makes that
possible later without a rewrite.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import ExperimentConfig

EdgeType = Literal["draft", "improve", "debug"]
NodeStatus = Literal["pending", "smoke_tested", "escalated_10pct", "done", "failed", "killed"]


@dataclass
class ExperimentNode:
    node_id: str  # == ExperimentConfig.id; one config, one node, no ambiguity
    config: ExperimentConfig
    edge_type: EdgeType
    parent_id: str | None
    status: NodeStatus = "pending"
    fidelity: str = "none"  # "none" | "1pct" | "10pct" | "100pct" -- highest stage reached
    metrics: dict[str, Any] | None = None  # whatever fidelity stage completed at
    insight: str = ""
    diagnosis_tag: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentNode":
        d = dict(d)
        d["config"] = ExperimentConfig.from_dict(d["config"])
        return cls(**d)


class ResearchMap:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.nodes: dict[str, ExperimentNode] = {}
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------- mutation
    def add_node(self, config: ExperimentConfig, edge_type: EdgeType,
                 parent_id: str | None = None) -> str:
        if config.id in self.nodes:
            raise ValueError(f"node '{config.id}' already exists -- ExperimentConfig.id must be unique "
                              f"across the whole Research Map, not just within one run")
        pid = parent_id if parent_id is not None else config.parent_id
        node = ExperimentNode(node_id=config.id, config=config, edge_type=edge_type, parent_id=pid)
        self.nodes[node.node_id] = node
        self.save()
        return node.node_id

    def update_node(self, node_id: str, **fields: Any) -> ExperimentNode:
        node = self.nodes[node_id]
        for k, v in fields.items():
            setattr(node, k, v)
        node.updated_at = time.time()
        self.save()
        return node

    # ---------------------------------------------------------------- query
    def get(self, node_id: str) -> ExperimentNode:
        return self.nodes[node_id]

    def children(self, node_id: str) -> list[ExperimentNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def roots(self) -> list[ExperimentNode]:
        return [n for n in self.nodes.values() if n.parent_id is None]

    def done_nodes(self) -> list[ExperimentNode]:
        return [n for n in self.nodes.values() if n.status == "done"]

    def failed_nodes(self) -> list[ExperimentNode]:
        return [n for n in self.nodes.values() if n.status == "failed"]

    def best_node(self, metric_path: tuple[str, ...] = ("valid", "primary_mean")) -> ExperimentNode | None:
        """Highest metric among `done` nodes (defaults to valid primary --
        the only metric any P0/P1 selection logic is allowed to read).

        Deliberately a *raw* numeric leader, nothing more: two nodes 0.0006
        apart on a single-digit-seed comparison are indistinguishable from
        noise (agent/diagnosis.py's own significance bar said so, in
        `features_v1`'s case, the exact scenario that surfaced this gap --
        it out-scored `deepfm_regularized` by +0.0002 valid primary, 3-seed,
        and was still tagged `noise_floor`), so treating this method's
        return value as "the model to ship" would silently launder a coin
        flip into a claimed improvement. Use `best_confirmed_node()` for
        anything downstream of that decision (final submission, what a new
        candidate's `parent_id` should be, what gets reported as "current
        best" to the LLM Research Strategist); this method stays raw/simple
        for exploration bookkeeping (e.g. the Best-First Selector's novelty
        scoring, which *should* see every point estimate, not just
        confirmed ones)."""
        best, best_score = None, float("-inf")
        for n in self.done_nodes():
            v: Any = n.metrics
            for k in metric_path:
                if v is None:
                    break
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and v > best_score:
                best, best_score = n, v
        return best

    _UNCONFIRMED_TAGS = frozenset({"noise_floor", "regression", "mixed", "ranking_tradeoff"})

    def best_confirmed_node(self, metric_path: tuple[str, ...] = ("valid", "primary_mean")) -> ExperimentNode | None:
        """`best_node()`, but refuses to hand back a node whose own
        diagnosis says it isn't a *confirmed* improvement over what it was
        actually compared against (its parent, or the official baseline for
        a root) -- walks up the parent chain past any run of
        noise_floor/regression/mixed/ranking_tradeoff nodes until it lands
        on one tagged `clear_improvement`/`baseline_beat` (or a node with no
        eligible parent to fall back to, e.g. a root itself tagged
        `baseline_match`). This is the "what should we actually build on/
        ship" answer; `best_node()` is the raw leaderboard.

        Note this deliberately does NOT re-rank by score after unwinding --
        it walks strictly up one lineage from the numeric leader, so it can
        return a node that isn't the highest-scoring *confirmed* node
        anywhere in the map if a different, unrelated lineage scored higher
        but was never numerically on top. That's an accepted simplification
        (the map has never actually had competing lineages this close in
        practice); a full best-first search over confirmed nodes only would
        replace this if that ever changes.
        """
        node = self.best_node(metric_path)
        seen: set[str] = set()
        while node is not None and node.node_id not in seen:
            seen.add(node.node_id)
            if node.diagnosis_tag in self._UNCONFIRMED_TAGS and node.parent_id in self.nodes:
                node = self.nodes[node.parent_id]
            else:
                break
        return node

    def explored_summary(self) -> dict[str, Any]:
        """What the map already knows -- the input a Best-First Selector
        (or, in Phase 4, an LLM) needs to answer "what haven't we tried yet"
        instead of "what's the last score" (a User Story from the
        brainstorm doc's Agent section)."""
        by_model: dict[str, list[str]] = {}
        by_edge: dict[str, int] = {}
        for n in self.nodes.values():
            by_model.setdefault(n.config.model, []).append(n.status)
            by_edge[n.edge_type] = by_edge.get(n.edge_type, 0) + 1
        best = self.best_confirmed_node()  # not best_node(): a numeric leader tagged
        # noise_floor/regression isn't a real "best" -- see best_confirmed_node's docstring.
        raw_best = self.best_node()
        return {
            "total_nodes": len(self.nodes),
            "by_model": {m: {"count": len(s), "done": s.count("done"), "failed": s.count("failed"),
                              "killed": s.count("killed")} for m, s in by_model.items()},
            "by_edge_type": by_edge,
            "best_node_id": best.node_id if best else None,
            "best_valid_primary": (best.metrics or {}).get("valid", {}).get("primary_mean") if best else None,
            "raw_leaderboard_node_id": raw_best.node_id if raw_best else None,
            "diagnosis_tags_seen": sorted({n.diagnosis_tag for n in self.nodes.values() if n.diagnosis_tag}),
        }

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        """Atomic write, but merges with whatever is currently on disk
        first -- NOT a plain overwrite. Two processes each holding their
        own in-memory snapshot (this project runs many background
        experiments concurrently by design -- multiple `run_p1.py`/
        `tools/verify_multiseed.py`/one-off scripts in flight at once) would
        otherwise silently clobber each other: whichever saves last
        overwrites the *entire* file with its own, possibly-stale,
        in-memory node dict, discarding any node a different process added
        or updated in the meantime.

        This happened for real during this project's own P2 pass:
        `lgbm_baseline` was added by a short-lived script while a
        long-running `tools/verify_multiseed.py` process (already holding
        an older in-memory snapshot without that node) was still running in
        the background; when it finished and called `update_node` ->
        `save()`, it silently erased `lgbm_baseline` from the persisted
        file. Caught by noticing the Research Map had fewer nodes than
        expected, not by any test -- see
        `test_research_map_save_merges_concurrent_writes` for the
        regression test this added.

        Fix: per-node, last-writer-wins by `updated_at`, not per-file.
        Before writing, re-read the current on-disk state (if any) and
        merge it with this instance's in-memory `self.nodes`: a node
        present on disk but not in memory is kept (someone else added it);
        a node present in both keeps whichever copy has the later
        `updated_at`. This isn't a true lock -- two saves at the exact same
        instant touching the exact same node_id could still race -- but it
        closes the actual failure mode above, which was one process adding
        a *new* node while another, unrelated update was in flight.
        """
        merged = dict(self.nodes)
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text(encoding="utf-8"))
                for nid, d in on_disk.get("nodes", {}).items():
                    disk_node = ExperimentNode.from_dict(d)
                    if nid not in merged or disk_node.updated_at > merged[nid].updated_at:
                        merged[nid] = disk_node
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # disk file unreadable/corrupt -- fall back to this instance's own in-memory state
        self.nodes = merged

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic-ish swap -- a crash mid-write can't corrupt the persisted map

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {nid: ExperimentNode.from_dict(d) for nid, d in payload["nodes"].items()}
