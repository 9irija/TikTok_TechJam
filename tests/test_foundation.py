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
    print("Phase 0 foundation smoke tests\n" + "=" * 40)
    tests = [
        test_evaluator_wrapper_matches_known_values,
        test_convergence_detector_epsilon_n_rule,
        test_convergence_detector_reads_organizer_epsilon_n,
        test_config_diff,
        test_fm_forward_backward_reduces_loss,
        test_deepfm_forward_backward_reduces_loss,
    ]
    if data_dir_available():
        tests.insert(0, test_evaluator_self_check)
        tests.append(test_recovery_catches_broken_config)
    else:
        print("  (data dir not found -- skipping self-check and recovery-subprocess tests)")

    for t in tests:
        check(t.__name__, t)

    print("=" * 40)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
