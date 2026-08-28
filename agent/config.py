"""Structured Experiment Interface (P0).

A config schema (model, features, hyperparams) that fills in ~80% of
experiments without any LLM writing code -- only genuinely novel ideas
should ever fall through to raw code generation (Phase 4+). Every field
here is deliberately restricted to *choices*, not code, so an experiment
can be fully described, diffed, hashed, and replayed from JSON alone.

Each config also carries the two fields the Run & Iteration Log deliverable
requires directly: `hypothesis` (what we intend to try and why) and
`parent_id` (which earlier experiment this one modifies -- the seed of the
Research Map / Experiment Tree that P1 builds out fully; even in Phase 0's
flat predefined list, recording lineage costs nothing and pre-wires the
schema so P1 doesn't need a migration).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any


# The 5 field-domains the starter kit ships (data.py FIELDS). An experiment
# may extend this list; anything beyond these 5 requires new columns to be
# joined in agent/features.py (not part of Phase 0 -- see CLAUDE.md P1 notes).
BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


@dataclasses.dataclass
class ExperimentConfig:
    id: str
    model: str                       # key into agent.model_zoo.registry.MODELS
    hypothesis: str                  # what we intend to try and why
    hyperparams: dict[str, Any] = dataclasses.field(default_factory=dict)
    fields: list[str] = dataclasses.field(default_factory=lambda: list(BASE_FIELDS))
    parent_id: str | None = None     # lineage -- None for a fresh/root config
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0])
    notes: str = ""                  # free-text: prior evidence, paper citation, etc.
    edge_type: str | None = None     # "draft"|"improve"|"debug" (AIDE); None -> P1 orchestrator infers
                                      # draft/improve from parent_id, matching pre-P1 behavior

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(**d)


def diff_configs(prev: ExperimentConfig | None, cur: ExperimentConfig) -> dict[str, Any]:
    """Structural diff between two configs -- this *is* "the code diff applied"
    for the ~80% of experiments that only change a config (Deliverables item 3
    asks for a code diff per iteration; for structured-config experiments the
    config diff is the faithful, complete record of what changed and why an
    LLM-authored source diff would be redundant with `hypothesis`).

    Returns {} if prev is None (nothing to diff against -- first iteration)
    or if prev == cur (should not happen but keeps callers simple).
    """
    if prev is None:
        return {"__root__": True}
    a, b = prev.to_dict(), cur.to_dict()
    changed: dict[str, Any] = {}
    keys = set(a) | set(b)
    for k in sorted(keys):
        if k in ("id", "hypothesis", "notes"):
            continue  # metadata, not "what changed algorithmically"
        if a.get(k) != b.get(k):
            changed[k] = {"before": a.get(k), "after": b.get(k)}
    return changed
