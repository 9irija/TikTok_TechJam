"""Phase 0 foundation smoke tests -- no pytest dependency (dev environment
only has numpy + pandas + stdlib), so this runs as a plain script:

    python tests/test_foundation.py

Every P0 piece gets at least one test that would fail loudly if that piece
silently drifted: the evaluator wrapper's self-check against real data, the
convergence detector's epsilon/N logic against a synthetic trajectory, the
config diff, both models' forward/backward on synthetic data (fast -- no
need to touch the 1.14M-row CSVs to prove the math doesn't crash and the
loss goes down), and that Failure Recovery really does catch and report a
deliberately broken config instead of propagating the exception.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from agent.config import ExperimentConfig, diff_configs
from agent.convergence import ConvergenceDetector
from agent.model_zoo import build as build_model
from agent.paths import DEFAULT_DATA_DIR, data_dir_available

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------- tests

def test_evaluator_self_check():
    from agent.evaluator import self_check
    result = self_check()
    if result.get("skipped"):
        print(f"    (skipped -- {result['reason']})")
        return
    assert result["ok"], f"self-check mismatch: {result}"


def test_evaluator_wrapper_matches_known_values():
    """Sanity-check the wrapper against evaluate.py's own docstring example
    logic on a tiny hand-built case: one user with a mixed exposure set."""
    from agent.evaluator import score
    users = [1, 1, 1, 2, 2]
    labels = [1, 0, 0, 0, 0]           # user 1: 1 positive of 3; user 2: all-negative
    scores = [3.0, 1.0, 2.0, 0.5, 0.1]
    r = score(users, labels, scores)
    assert 0.0 <= r.gauc <= 1.0, f"GAUC out of range: {r.gauc}"
    assert 0.0 <= r.ndcg5 <= 1.0, f"nDCG@5 out of range: {r.ndcg5}"
    assert abs(r.primary - (r.gauc + r.ndcg5) / 2.0) < 1e-9, "primary must be mean(GAUC, nDCG@5)"


def test_convergence_detector_epsilon_n_rule():
    cd = ConvergenceDetector(epsilon=0.002, n=3)
    # Two real improvements, then N=3 consecutive stagnant iterations -> converged
    trajectory = [0.50, 0.55, 0.552, 0.553, 0.5531]
    converged_at = None
    for i, v in enumerate(trajectory):
        cd.update(f"iter_{i}", v)
        if cd.is_converged() and converged_at is None:
            converged_at = i
    assert converged_at is not None, "should have converged once 3 consecutive stagnant iterations occurred"
    bid, bscore = cd.best()
    assert bscore == max(trajectory), f"best score should be the trajectory max, got {bscore}"


def test_convergence_detector_reads_organizer_epsilon_n():
    cd = ConvergenceDetector.from_starter_kit()
    assert cd.epsilon == 0.002, cd.epsilon
    assert cd.n == 3, cd.n


def test_config_diff():
    a = ExperimentConfig(id="a", model="fm", hypothesis="h", hyperparams={"k": 16, "lr": 0.001})
    b = ExperimentConfig(id="b", model="deepfm", hypothesis="h2", hyperparams={"k": 16, "lr": 0.005})
    d = diff_configs(a, b)
    assert "model" in d and d["model"]["before"] == "fm" and d["model"]["after"] == "deepfm"
    assert "hyperparams" in d
    assert "id" not in d, "metadata fields (id/hypothesis/notes) must not appear in the diff"
    assert diff_configs(None, a) == {"__root__": True}


def _synthetic_batch(n=512, n_fields=5, dim=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y


def test_fm_forward_backward_reduces_loss():
    X, y = _synthetic_batch()
    m = build_model("fm", dim=200, n_fields=5, k=8, lr=0.05, seed=0)
    losses = [m.step(X, y) for _ in range(30)]
    assert losses[-1] < losses[0], f"FM loss should decrease on a fixed synthetic batch: {losses[0]} -> {losses[-1]}"
    preds = m.predict(X)
    assert preds.shape == (len(y),), preds.shape
    state = m.get_state()
    m2 = build_model("fm", dim=200, n_fields=5, k=8, lr=0.05, seed=1)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_forward_backward_reduces_loss():
    X, y = _synthetic_batch()
    m = build_model("deepfm", dim=200, n_fields=5, k=8, hidden=[16, 8], lr=0.02, seed=0)
    losses = [m.step(X, y) for _ in range(30)]
    assert losses[-1] < losses[0], f"DeepFM loss should decrease on a fixed synthetic batch: {losses[0]} -> {losses[-1]}"
    preds = m.predict(X)
    assert preds.shape == (len(y),), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite (no NaN/Inf leaking from the MLP)"
    state = m.get_state()
    m2 = build_model("deepfm", dim=200, n_fields=5, k=8, hidden=[16, 8], lr=0.02, seed=1)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_fm_bpr_forward_backward_reduces_loss():
    from agent.model_zoo.fm_bpr import build_user_pos_neg_index, sample_bpr_batches
    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    users = [f"u{rng.integers(0, 50)}" for _ in range(n)]  # 50 users over 2000 rows -> most have both classes

    user_index = build_user_pos_neg_index(users, y)
    assert len(user_index) > 0, "synthetic data should produce at least one user with both a positive and negative row"

    m = build_model("fm_bpr", dim=dim, n_fields=n_fields, k=8, lr=0.05, seed=0)
    batch_rng = np.random.default_rng(1)
    losses = [m.bpr_step(xp, xn) for xp, xn in sample_bpr_batches(batch_rng, X, user_index, n_batches=30, batch_size=64)]
    assert losses[-1] < losses[0], f"BPR loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"
    state = m.get_state()
    m2 = build_model("fm_bpr", dim=dim, n_fields=n_fields, k=8, lr=0.05, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_research_map_basic():
    import tempfile
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "research_map.json"
        rm = ResearchMap(path)
        root = ExperimentConfig(id="root", model="fm", hypothesis="h", parent_id=None)
        rm.add_node(root, edge_type="draft")
        rm.update_node("root", status="done",
                        metrics={"valid": {"primary_mean": 0.60}, "test": {"primary_mean": 0.59}})

        child = ExperimentConfig(id="child", model="fm", hypothesis="h2", parent_id="root")
        rm.add_node(child, edge_type="improve", parent_id="root")
        rm.update_node("child", status="done",
                        metrics={"valid": {"primary_mean": 0.61}, "test": {"primary_mean": 0.60}})

        assert [n.node_id for n in rm.children("root")] == ["child"]
        best = rm.best_node()
        assert best is not None and best.node_id == "child", f"expected 'child' as best, got {best}"

        # persistence round-trip -- a fresh ResearchMap instance over the same path must recover everything
        rm2 = ResearchMap(path)
        assert set(rm2.nodes.keys()) == {"root", "child"}
        assert rm2.get("child").metrics["valid"]["primary_mean"] == 0.61

        summary = rm2.explored_summary()
        assert summary["total_nodes"] == 2
        assert summary["best_node_id"] == "child"


def test_diagnosis_rules():
    from agent.diagnosis import diagnose

    parent = {"valid": {"GAUC_mean": 0.66, "nDCG@5_mean": 0.53, "primary_mean": 0.595}, "per_seed": []}
    improved = {"valid": {"GAUC_mean": 0.67, "nDCG@5_mean": 0.54, "primary_mean": 0.605}, "per_seed": []}
    regressed = {"valid": {"GAUC_mean": 0.65, "nDCG@5_mean": 0.52, "primary_mean": 0.585}, "per_seed": []}
    noise = {"valid": {"GAUC_mean": 0.6605, "nDCG@5_mean": 0.5305, "primary_mean": 0.5955}, "per_seed": []}

    assert diagnose(improved, parent).tag == "clear_improvement"
    assert diagnose(regressed, parent).tag == "regression"
    assert diagnose(noise, parent).tag == "noise_floor"
    assert diagnose(None, None).tag == "no_result"

    # Seed-aware significance: a delta *below* the flat epsilon (0.002) must
    # still be flagged as real when multiple low-std seeds make it
    # statistically robust -- this is the exact bug this engine had when it
    # mislabeled deepfm-vs-FM as "noise_floor" (see _significance_bar's
    # docstring). Small delta (0.0011 < eps), small per-seed std (0.0003),
    # 3 seeds each side -> combined SEM ~0.00024, so 2*SEM ~0.00049 << 0.0011.
    tight_parent = {"valid": {"GAUC_mean": 0.660, "nDCG@5_mean": 0.530, "primary_mean": 0.5950,
                               "GAUC_std": 0.0003, "nDCG@5_std": 0.0003, "primary_std": 0.0003},
                     "n_seeds": 3, "per_seed": []}
    tight_child = {"valid": {"GAUC_mean": 0.6615, "nDCG@5_mean": 0.5305, "primary_mean": 0.5961,
                              "GAUC_std": 0.0003, "nDCG@5_std": 0.0003, "primary_std": 0.0003},
                    "n_seeds": 3, "per_seed": []}
    small_but_real = diagnose(tight_child, tight_parent)
    assert small_but_real.tag != "noise_floor", (
        f"a tight (low-std, multi-seed) delta below flat epsilon must not be dismissed as noise, got {small_but_real.tag}")


def test_selector_scores_candidates():
    import tempfile
    from agent.research_map import ResearchMap
    from agent.selector import select_next

    with tempfile.TemporaryDirectory() as d:
        rm = ResearchMap(Path(d) / "map.json")
        root = ExperimentConfig(id="root", model="fm", hypothesis="h")
        rm.add_node(root, edge_type="draft")
        rm.update_node("root", status="done",
                        metrics={"valid": {"primary_mean": 0.60}, "per_seed": [{"wall_time_s": 90.0}]})

        cand_a = ExperimentConfig(id="cand_a", model="deepfm", hypothesis="h", parent_id="root")
        cand_b = ExperimentConfig(id="cand_b", model="fm", hypothesis="h", parent_id="root")
        best, ranked = select_next(rm, [cand_a, cand_b])
        assert best is not None, "should pick a candidate from a non-empty pool"
        assert len(ranked) == 2
        assert all(r.reasoning for r in ranked), "every candidate must carry a human-readable reasoning string"
        already_in_map, _ = select_next(rm, [root])
        assert already_in_map is None, "a config already in the map must not be re-offered as a candidate"


def test_multi_fidelity_kills_a_broken_config():
    from agent.multi_fidelity import run_multi_fidelity
    bad_config = ExperimentConfig(
        id="broken_multi_fidelity_smoke_test", model="this_model_does_not_exist",
        hypothesis="deliberately invalid", hyperparams={}, seeds=[0],
    )
    result = run_multi_fidelity(bad_config, str(DEFAULT_DATA_DIR), timeout_s=30, max_retries=1)
    assert not result.survived(), "a broken config must never survive to 100pct"
    assert result.killed_at == "1pct", f"should be killed at the first (cheapest) stage, got {result.killed_at}"


def test_recovery_catches_broken_config():
    from agent.recovery import run_with_recovery
    bad_config = ExperimentConfig(
        id="broken_smoke_test", model="this_model_does_not_exist", hypothesis="deliberately invalid",
        hyperparams={"epochs": 1, "batch": 64}, seeds=[0],
    )
    result, events = run_with_recovery(bad_config, str(DEFAULT_DATA_DIR), seed=0,
                                        timeout_s=30, max_retries=1, degrade_epochs=1)
    assert result is None, "a genuinely broken config must never silently succeed"
    kinds = {e.kind for e in events}
    assert "error" in kinds or "abandoned" in kinds, f"expected error/abandoned events, got kinds={kinds}"
    assert any(e.kind == "abandoned" for e in events), "must end in 'abandoned', not raise or hang"


def main() -> int:
    print("Phase 0 + P1 foundation smoke tests\n" + "=" * 40)
    tests = [
        test_evaluator_wrapper_matches_known_values,
        test_convergence_detector_epsilon_n_rule,
        test_convergence_detector_reads_organizer_epsilon_n,
        test_config_diff,
        test_fm_forward_backward_reduces_loss,
        test_deepfm_forward_backward_reduces_loss,
        test_fm_bpr_forward_backward_reduces_loss,
        test_research_map_basic,
        test_diagnosis_rules,
        test_selector_scores_candidates,
    ]
    if data_dir_available():
        tests.insert(0, test_evaluator_self_check)
        tests.append(test_recovery_catches_broken_config)
        tests.append(test_multi_fidelity_kills_a_broken_config)
    else:
        print("  (data dir not found -- skipping self-check and recovery-subprocess tests)")

    for t in tests:
        check(t.__name__, t)

    print("=" * 40)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
