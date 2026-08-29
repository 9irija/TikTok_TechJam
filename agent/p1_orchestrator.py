"""P1 Orchestrator: Research Map + Best-First Selection + Multi-Fidelity execution.

Replaces Phase 0's flat "run PREDEFINED_EXPERIMENTS in declaration order"
with AIDE-style best-first tree search:

    1. Load (or, on a first run, seed from Phase 0's own logs/run_log.jsonl)
       a *persistent* Research Map -- unlike a per-run log, this survives
       across invocations, so a second `python run_p1.py` starts from more
       knowledge than the first, not from scratch.
    2. Score every not-yet-tried candidate in a hand-authored P1 candidate
       pool via the Best-First Node Selector (agent/selector.py) --
       expected_gain x confidence x novelty / cost, per the brainstorm doc.
    3. Run candidates in best-first (scored) order, not declaration order,
       each through the Multi-Fidelity Runner (1%->10%->100%, killing early
       if clearly broken/not-learning -- agent/multi_fidelity.py).
    4. Diagnose every completed candidate (agent/diagnosis.py) and record
       it as a new Research Map node.

Still no LLM reasoning -- per the numbered Phase roadmap in CLAUDE.md,
that's Phase 4. The "research" here is a fixed heuristic scorer over a
hand-written candidate pool, not an autonomous hypothesis generator. What
IS new relative to Phase 0: the tree structure, the scored/ranked
selection order, the persistent cross-run memory, and staged compute spend
instead of always training at full scale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BASE_FIELDS, ExperimentConfig
from .diagnosis import diagnose
from .multi_fidelity import run_multi_fidelity
from .paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR
from .research_critic import review as critic_review
from .research_map import ResearchMap
from .run_logger import RunLogger
from .selector import select_next

RESEARCH_MAP_PATH = LOGS_DIR / "research_map.json"


def seed_from_phase0(map: ResearchMap, run_log_path: Path) -> int:
    """One-time: if the map is empty and a Phase 0 run_log.jsonl exists,
    populate the map from it -- continuity with Phase 0's validated
    results, not a cold start. Returns the number of nodes seeded."""
    if map.nodes or not run_log_path.exists():
        return 0
    n = 0
    for line in run_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("status") != "ok" or not e.get("metrics"):
            continue
        cfg = ExperimentConfig(id=e["config_id"], model=e["model"], hypothesis=e["hypothesis"],
                                hyperparams=e["hyperparams"], fields=list(BASE_FIELDS),
                                parent_id=e.get("parent_id"))
        if cfg.id in map.nodes:
            continue
        parent_metrics = map.nodes[cfg.parent_id].metrics if cfg.parent_id in map.nodes else None
        edge = "draft" if cfg.parent_id is None else "improve"
        map.add_node(cfg, edge_type=edge, parent_id=cfg.parent_id)
        d = diagnose(e["metrics"], parent_metrics)
        map.update_node(cfg.id, status="done", metrics=e["metrics"], fidelity="100pct",
                         insight=d.insight, diagnosis_tag=d.tag)
        n += 1
    return n


def p1_candidate_pool(map: ResearchMap) -> list[ExperimentConfig]:
    """Hand-authored candidates -- no LLM yet. Two genuinely different
    directions on purpose: one new algorithmic family (BPR loss, the
    starter kit's own #1-ranked untested idea) and one direct replication
    check of a documented dead end (capacity/k), so the Best-First
    Selector has a real choice to reason about, not a single obvious pick.
    """
    candidates = [
        ExperimentConfig(
            id="fm_bpr_default", model="fm_bpr",
            hypothesis=(
                "Starter kit README ranks a pairwise/listwise loss change as its #1 guess for "
                "untested headroom -- GAUC/nDCG are ranking metrics, but every model trained so "
                "far (FM, DeepFM) uses pointwise logloss. BPR directly optimizes "
                "P(positive item ranked above a negative item for the same user), which is "
                "exactly what GAUC measures. Same k=16 embedding size as the FM baseline, so any "
                "effect is attributable to the loss/objective change alone, not a confounded "
                "architecture change."
            ),
            hyperparams={"k": 16, "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
            fields=list(BASE_FIELDS), parent_id="fm_baseline_repro", seeds=[0],
            notes="agent/model_zoo/fm_bpr.py -- P1's first new algorithmic direction "
                  "(a training-objective change, not just a model/hyperparameter tweak).",
        ),
        ExperimentConfig(
            id="fm_wider_k32", model="fm",
            hypothesis=(
                "Direct replication check of the starter kit's own ablation (README: k=8/16/32 -> "
                "0.5895/0.5902/0.5887 test primary; conclusion: 'capacity is not the bottleneck') "
                "against OUR harness, at k=32. If it also shows no gain here, that's independent "
                "confirmation of a documented dead end using our own pipeline -- worth recording "
                "in the Research Map so the Best-First Selector learns to deprioritize pure-capacity "
                "changes on future rounds, not just trusting the starter kit's separate ablation script."
            ),
            hyperparams={"k": 32, "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
            fields=list(BASE_FIELDS), parent_id="fm_baseline_repro", seeds=[0],
            notes="",
        ),
    ]
    candidates.extend(_engineered_feature_candidates(map))
    candidates.extend(_multi_task_candidates(map))
    candidates.extend(_deepfm_bpr_candidates(map))
    candidates.extend(_pcgrad_candidates(map))
    candidates.extend(_diagnosis_driven_candidates(map))
    return [c for c in candidates if c.id not in map.nodes]


def _engineered_feature_candidates(map: ResearchMap) -> list[ExperimentConfig]:
    """Tests the brainstorm doc's TikTok-disclosed feature ideas (completion
    rate, rewatch, fast-skip, creator-level engagement -- P1 table, "Data &
    Features") as TRAIN-ONLY AGGREGATE fields (agent/features.py) on top of
    whatever the current best architecture is, so any effect is attributable
    to the new fields alone, not a confounded architecture change.

    Deliberately NOT `completion_rate = play_time_ms / duration_ms` computed
    on the row being predicted -- that would leak `long_view` (empirically
    ~85% correlated with a naive reconstruction of it from play_time_ms,
    checked directly against the raw log before any of this was built, see
    agent/features.py's module docstring). Every field here is a video's or
    author's *historical* average from the training split only.
    """
    best = map.best_confirmed_node()  # not best_node(): don't build on a numeric leader
    # that isn't actually a confirmed win (see ResearchMap.best_confirmed_node docstring).
    if best is None or best.status != "done" or "features_v1" in map.nodes:
        return []
    from .features import BASE_5, EXTRA_FIELDS
    base_hp = dict(best.config.hyperparams)
    return [ExperimentConfig(
        id="features_v1", model=best.config.model,
        hypothesis=(
            f"Tests TikTok's own disclosed strong signals as train-only aggregate features on top "
            f"of the current best architecture ('{best.node_id}', valid primary "
            f"{(best.metrics or {}).get('valid', {}).get('primary_mean')}), so any effect is "
            f"attributable to the new fields alone: video_completion_bucket, video_rewatch_bucket, "
            f"video_fast_skip_bucket (all TRAIN-split video-level aggregates), and "
            f"author_engagement_bucket (creator-level, per the P1 table's 'creator quality is part "
            f"of TikTok's real ranking story'). Not the same-row play_time_ms ratio, which would "
            f"leak the long_view label directly."
        ),
        hyperparams=base_hp, fields=BASE_5 + EXTRA_FIELDS, parent_id=best.node_id, seeds=[0],
        edge_type="improve",
        notes="agent/features.py -- first use of ExperimentConfig.fields actually affecting "
              "encoding (previously vestigial, see agent/experiment.py's _load_encoded fix).",
    )]


def _multi_task_candidates(map: ResearchMap) -> list[ExperimentConfig]:
    """Tests a genuinely different lever than every other P1/P4 candidate so
    far: not a hyperparameter, not a new input field, but a training
    *objective* change -- agent/model_zoo/deepfm_mtl.py adds four auxiliary
    sigmoid heads (is_like/is_follow/is_comment/is_forward) sharing the same
    embedding table as the main long_view task, a shared-bottom multi-task
    setup in the public ESMM/MMoE line of recsys work (allowed per the
    problem statement: "papers, public solutions... changes to any pipeline
    stage"). Same fields, same k/hidden/lr/l2 as the current best DeepFM
    node -- parent_id isolates the multi-task objective as the only
    variable, same discipline as fm_bpr_default's loss-only comparison.

    First real use of torch in this project (see deepfm_mtl.py's own
    docstring for why: FM/DeepFM/FM_BPR's single-loss backward passes were
    worth hand-deriving to keep the starter kit's numpy-only philosophy;
    merging five different loss gradients back into one shared trunk is
    exactly the case autograd is for, not a shortcut around it).
    """
    best = map.best_confirmed_node()
    if best is None or best.status != "done" or best.config.model != "deepfm" or "deepfm_mtl_v1" in map.nodes:
        return []  # only meaningful on top of a DeepFM node -- the aux heads share DeepFM's embedding table
    try:
        import torch  # noqa: F401
    except ImportError:
        return []  # torch not installed in this environment -- skip, don't fail the whole pool

    base_hp = dict(best.config.hyperparams)
    base_hp["aux_weight"] = 0.2  # standard small weighting (public MTL convention: aux regularizes,
    # doesn't dominate -- the metric that's actually scored is long_view alone)
    return [ExperimentConfig(
        id="deepfm_mtl_v1", model="deepfm_mtl",
        hypothesis=(
            f"Multi-task learning, not a hyperparameter or input-feature change: adds auxiliary "
            f"is_like/is_follow/is_comment/is_forward sigmoid heads sharing '{best.node_id}'s "
            f"embedding table (valid primary {(best.metrics or {}).get('valid', {}).get('primary_mean')}), "
            f"a shared-bottom setup in the public ESMM/MMoE recsys line. Hypothesis: long_view and "
            f"these other logged engagement signals share structure (all downstream of genuine "
            f"enjoyment of the video), so their gradients can regularize the shared embeddings even "
            f"though only long_view is ever scored. Same fields/k/hidden/lr/l2 as the parent, so any "
            f"effect is attributable to the multi-task objective alone."
        ),
        hyperparams=base_hp, fields=list(best.config.fields), parent_id=best.node_id, seeds=[0],
        edge_type="improve",
        notes="agent/model_zoo/deepfm_mtl.py -- first torch model in the Model Zoo; agent/experiment.py's "
              "is_mtl branch feeds it agent/features.py's load_aux_labels() alongside (X, y).",
    )]


def _deepfm_bpr_candidates(map: ResearchMap) -> list[ExperimentConfig]:
    """Combines two independently-partial P1/P2 results instead of trying a
    third, unrelated idea: `fm_bpr` (P1) showed the loss/metric-alignment
    direction is real but plateaued ~0.002 below the FM baseline in plain
    FM form after 3 diagnosis-driven rounds; the deep component
    (`deepfm`/`deepfm_regularized`) independently, separately proved to
    help. Neither was ever combined with the other -- flagged in this
    project's own roadmap notes as "a natural, cheap extension of what's
    already built" since before the P1 rounds even concluded.

    parent_id is deliberately `deepfm_regularized`, NOT `best_confirmed_node()`
    (currently `deepfm_mtl_v1`): the comparison this candidate is actually
    designed to make is "does BPR's pairwise objective help the SAME
    DeepFM architecture deepfm_regularized already validated," holding
    everything else (fields, k, hidden, l2) fixed -- deepfm_mtl_v1 has a
    different hyperparameter shape (aux_weight, MTL heads) that would
    confound the comparison, not isolate it.
    """
    parent = map.nodes.get("deepfm_regularized")
    if parent is None or parent.status != "done" or "deepfm_bpr_v1" in map.nodes:
        return []
    try:
        import torch  # noqa: F401
    except ImportError:
        return []  # torch not installed in this environment -- skip, don't fail the whole pool

    base_hp = {k: v for k, v in parent.config.hyperparams.items() if k in ("k", "hidden", "l2")}
    base_hp["lr"] = 0.001
    return [ExperimentConfig(
        id="deepfm_bpr_v1", model="deepfm_bpr",
        hypothesis=(
            f"Combines two independently-partial results rather than a new, unrelated idea: fm_bpr's "
            f"pairwise loss (P1, real loss/metric-alignment signal, plateaued ~0.002 below the FM "
            f"baseline in plain FM form) with '{parent.node_id}'s deep architecture (valid primary "
            f"{(parent.metrics or {}).get('valid', {}).get('primary_mean')}), which independently proved "
            f"to help. Same fields/k/hidden/l2 as the parent, so any effect is attributable to the "
            f"pairwise-vs-pointwise objective change alone -- exactly the same isolation discipline "
            f"fm_bpr_default used against fm_baseline_repro."
        ),
        hyperparams=base_hp, fields=list(parent.config.fields), parent_id="deepfm_regularized", seeds=[0],
        edge_type="improve",
        notes="agent/model_zoo/deepfm_bpr.py -- first torch model wired directly into the real pipeline "
              "(registry.py + agent/experiment.py's existing is_bpr branch) rather than standalone-checked "
              "first, since the bpr_step/predict contract fm_bpr already validated in P1 carries over "
              "with zero training-loop changes needed.",
    )]


def _pcgrad_candidates(map: ResearchMap) -> list[ExperimentConfig]:
    """Refines the one mechanism that's actually worked (deepfm_mtl_v1),
    rather than trying yet another architecture -- every architecture bet
    this pass (DIN, BPR, listwise, PDAOM) came back negative or tied,
    independently reconfirming the organizers' own "capacity/architecture
    isn't the bottleneck" finding. deepfm_mtl_uncertainty_v1 (already
    tried) addressed loss-magnitude weighting and came back tied; PCGrad
    (Yu et al. 2020) addresses a genuinely different multi-task pathology
    -- gradient DIRECTION conflict on shared parameters, which magnitude
    reweighting cannot fix even in principle. Same fields/k/hidden/l2/
    aux_weight as deepfm_mtl_v1, so gradient surgery is the only variable.
    """
    parent = map.nodes.get("deepfm_mtl_v1")
    if parent is None or parent.status != "done" or "deepfm_mtl_pcgrad_v1" in map.nodes:
        return []
    try:
        import torch  # noqa: F401
    except ImportError:
        return []  # torch not installed in this environment -- skip, don't fail the whole pool

    base_hp = dict(parent.config.hyperparams)
    return [ExperimentConfig(
        id="deepfm_mtl_pcgrad_v1", model="deepfm_mtl_pcgrad",
        hypothesis=(
            f"Refines '{parent.node_id}''s own multi-task mechanism (valid primary "
            f"{(parent.metrics or {}).get('valid', {}).get('primary_mean')}) rather than trying a new "
            f"architecture: PCGrad (Yu et al. 2020, 'Gradient Surgery for Multi-Task Learning') resolves "
            f"conflicting gradient DIRECTIONS between the main task and the auxiliary tasks on their "
            f"shared parameters, before combining them -- a different pathology than "
            f"deepfm_mtl_uncertainty_v1's loss-magnitude weighting (already tried, tied), which cannot "
            f"address direction conflict even in principle. Simplified to a 2-task formulation (main vs. "
            f"combined aux loss, not full 5-way pairwise) since aux_heads' 4 outputs share one weight "
            f"matrix, not cleanly separable parameters. Same fields/k/hidden/l2/aux_weight as the parent, "
            f"so gradient surgery is the only variable."
        ),
        hyperparams=base_hp, fields=list(parent.config.fields), parent_id="deepfm_mtl_v1", seeds=[0],
        edge_type="improve",
        notes="agent/model_zoo/deepfm_mtl_pcgrad.py -- reuses agent/experiment.py's existing is_mtl branch "
              "(same mtl_step(X,y,aux) contract as deepfm_mtl.py), wired directly into the real pipeline.",
    )]


def _diagnosis_driven_candidates(map: ResearchMap) -> list[ExperimentConfig]:
    """Reacts to a completed node's own diagnosis instead of only offering
    fresh drafts -- this is what makes a second `python run_p1.py` round
    genuinely "reflect + revise" rather than re-running the same static
    pool. Still hand-authored (no LLM -- Phase 4), but the template fires
    *because* of what the Diagnosis Engine actually found, not on a fixed
    schedule: closing the loop the brainstorm doc's own demo narrative
    describes ("performance dropped... agent recorded insight and avoided
    similar experiments") one step further into "and tried the specific fix
    the insight pointed at."
    """
    out: list[ExperimentConfig] = []

    bpr_node = map.nodes.get("fm_bpr_default")
    if (bpr_node and bpr_node.status == "done" and "overfitting_risk" in bpr_node.insight
            and "fm_bpr_regularized" not in map.nodes):
        out.append(ExperimentConfig(
            id="fm_bpr_regularized", model="fm_bpr",
            hypothesis=(
                "Directly acts on fm_bpr_default's own diagnosis, rather than treating BPR as a "
                "dead end: its overfitting_risk flag showed the best validation epoch landing at "
                "~33% of training length, meaning later epochs kept improving pairwise training "
                "loss on sampled pairs with no validation benefit -- a regularization/schedule "
                "problem, not evidence the objective itself is wrong. Two changes that target "
                "exactly that: L2 raised 100x (1e-6 -> 1e-4) and early-stopping patience tightened "
                "(4 -> 2) to stop closer to where validation actually peaks instead of continuing "
                "to fit noise in the sampled positive/negative pairs."
            ),
            hyperparams={"k": 16, "lr": 0.001, "l2": 1e-4, "batch": 8192, "epochs": 40, "patience": 2},
            fields=list(BASE_FIELDS), parent_id="fm_bpr_default", seeds=[0], edge_type="debug",
            notes="Generated by _diagnosis_driven_candidates() because fm_bpr_default's own "
                  "insight flagged overfitting_risk, not on a fixed schedule.",
        ))

    reg_node = map.nodes.get("fm_bpr_regularized")
    if (reg_node and reg_node.status == "done" and "fm_bpr_slow_and_steady" not in map.nodes):
        out.append(ExperimentConfig(
            id="fm_bpr_slow_and_steady", model="fm_bpr",
            hypothesis=(
                "Manual curve-shape analysis, not yet something the Diagnosis Engine checks "
                "automatically (a real gap -- it compares best_epoch/epochs_run ratios, not full "
                "trajectory shape against a healthy reference): fm_baseline_repro's own valid-primary "
                "curve rises steadily for 7 epochs (0.5869 -> 0.6015) before declining -- gradual, "
                "sustained convergence. Both prior BPR attempts (default AND regularized) instead "
                "plateau by epoch 2-4 and never show that gradual climb -- fast convergence to a "
                "shallow optimum, a different failure mode than 'overfitting', which the L2/patience "
                "fix in fm_bpr_regularized correctly stabilized (flatter curve) without correctly "
                "diagnosing (the real ceiling didn't move). This candidate targets convergence speed "
                "directly: lr cut ~3x (0.001 -> 0.0003) for a more gradual trajectory, patience and "
                "epoch budget both raised to give that slower process room to actually get there."
            ),
            hyperparams={"k": 16, "lr": 0.0003, "l2": 1e-5, "batch": 8192, "epochs": 60, "patience": 6},
            fields=list(BASE_FIELDS), parent_id="fm_bpr_regularized", seeds=[0], edge_type="debug",
            notes="Hand-authored from direct comparison against fm_baseline_repro's own curve -- "
                  "not (yet) an automated Diagnosis Engine trigger; see P1_FEATURES_AND_RESULTS.md.",
        ))

    deepfm_bpr_node = map.nodes.get("deepfm_bpr_v1")
    if (deepfm_bpr_node and deepfm_bpr_node.status == "done" and "overfitting_risk" in deepfm_bpr_node.insight
            and "deepfm_bpr_v1_regularized" not in map.nodes):
        base_hp = dict(deepfm_bpr_node.config.hyperparams)
        out.append(ExperimentConfig(
            id="deepfm_bpr_v1_regularized", model="deepfm_bpr",
            hypothesis=(
                "Directly acts on deepfm_bpr_v1's own diagnosis, same pattern that worked for "
                "fm_bpr_default->fm_bpr_regularized and deepfm_default->deepfm_regularized: its "
                "overfitting_risk flag showed the best validation epoch landing at ~20% of training "
                "length -- DeepFM's extra capacity combined with BPR's pairwise sampling overfits even "
                "faster than plain FM_BPR did. L2 raised another 10x (1e-4 -> 1e-3, on top of the L2 "
                "already inherited from deepfm_regularized) and patience tightened (default 4 -> 2) to "
                "stop closer to where validation actually peaks."
            ),
            hyperparams={**base_hp, "l2": base_hp.get("l2", 1e-4) * 10, "patience": 2},
            fields=list(deepfm_bpr_node.config.fields), parent_id="deepfm_bpr_v1", seeds=[0], edge_type="debug",
            notes="Generated by _diagnosis_driven_candidates() because deepfm_bpr_v1's own insight "
                  "flagged overfitting_risk, same reactive pattern as fm_bpr_regularized.",
        ))

    return out


def run_p1(data_dir: str | None = None, timeout_s: float = 240.0) -> dict[str, Any]:
    data_dir = data_dir or str(DEFAULT_DATA_DIR)
    map = ResearchMap(RESEARCH_MAP_PATH)
    seeded = seed_from_phase0(map, LOGS_DIR / "run_log.jsonl")

    logger = RunLogger(LOGS_DIR, EXPERIMENTS_DIR)
    pool = p1_candidate_pool(map)
    _, ranked = select_next(map, pool)

    report: dict[str, Any] = {
        "run_id": logger.run_id,
        "seeded_from_phase0": seeded,
        "candidates_considered": len(ranked),
        "ranking": [{"id": s.config.id, "score": s.score, "reasoning": s.reasoning} for s in ranked],
        "results": [],
    }

    for rank_i, s in enumerate(ranked, start=1):
        config = s.config

        verdict = critic_review(map, config)
        if not verdict.approved:
            # Rejected before spending any compute -- logged as its own
            # status (not "failed") so the rejection itself is part of the
            # audit trail: the agent declined to waste budget, it didn't
            # fail to have the idea.
            report["results"].append({
                "rank": rank_i, "config_id": config.id, "status": "critic_rejected",
                "final_stage": None, "killed_at": None, "kill_reason": None,
                "diagnosis": {"tag": "critic_rejected", "insight": verdict.reason},
                "wall_time_s": 0.0, "estimated_time_saved_s": None,
            })
            continue

        iteration_id = logger.next_iteration_id()
        predictions_path = str(logger.experiments_dir / iteration_id / "predictions.npz")

        mf_result = run_multi_fidelity(config, data_dir, timeout_s=timeout_s,
                                        save_predictions_on_survive=predictions_path)

        status = "ok" if mf_result.survived() else "failed"
        metrics = mf_result.final_metrics() if mf_result.survived() else None
        parent_metrics = map.nodes[config.parent_id].metrics if config.parent_id in map.nodes else None
        d = diagnose(metrics, parent_metrics) if status == "ok" else diagnose(None, None)

        inferred_edge = config.edge_type or ("improve" if config.parent_id else "draft")
        map.add_node(config, edge_type=inferred_edge, parent_id=config.parent_id)
        map.update_node(config.id, status=("done" if status == "ok" else "killed"),
                         metrics=metrics, fidelity=mf_result.final_stage,
                         insight=d.insight, diagnosis_tag=d.tag)

        fidelity_info = mf_result.to_fidelity_info(config)
        logger.log_iteration(iteration_id, config, metrics=metrics, wall_time_s=mf_result.total_wall_time_s,
                              recovery_events=mf_result.all_events, convergence_point=None, status=status,
                              fidelity_info=fidelity_info)

        report["results"].append({
            "rank": rank_i, "config_id": config.id, "status": status,
            "final_stage": mf_result.final_stage, "killed_at": mf_result.killed_at,
            "kill_reason": mf_result.kill_reason, "diagnosis": {"tag": d.tag, "insight": d.insight},
            "wall_time_s": mf_result.total_wall_time_s,
            "estimated_time_saved_s": fidelity_info["estimated_time_saved_s"],
        })

    total_time_saved = sum(r["estimated_time_saved_s"] or 0.0 for r in report["results"])
    report["resource_totals"] = {
        "wall_time_total_s": sum(r["wall_time_s"] for r in report["results"]),
        "estimated_time_saved_by_early_termination_s": total_time_saved,
    }
    report["research_map_summary"] = map.explored_summary()
    (LOGS_DIR / "p1_round_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
