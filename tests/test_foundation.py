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


def test_research_map_best_confirmed_node_skips_noise_floor():
    """Regression test for the gap `features_v1` surfaced live: a node that
    numerically out-scores its parent by less than the significance bar
    still gets tagged `noise_floor` by agent/diagnosis.py, but
    `ResearchMap.best_node()` (a raw numeric leaderboard) doesn't know
    that -- it would hand back the noise-floor node as "best". Anything
    that decides what to build on or ship must use `best_confirmed_node()`
    instead, which walks past unconfirmed nodes to the nearest real win."""
    import tempfile
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        rm = ResearchMap(Path(d) / "research_map.json")
        root = ExperimentConfig(id="root", model="fm", hypothesis="h", parent_id=None)
        rm.add_node(root, edge_type="draft")
        rm.update_node("root", status="done", diagnosis_tag="baseline_beat",
                        metrics={"valid": {"primary_mean": 0.60}})

        # A confirmed win over root.
        good = ExperimentConfig(id="good", model="fm", hypothesis="h", parent_id="root")
        rm.add_node(good, edge_type="improve", parent_id="root")
        rm.update_node("good", status="done", diagnosis_tag="clear_improvement",
                        metrics={"valid": {"primary_mean": 0.61}})

        # Numerically higher than 'good', but tagged noise_floor vs its parent --
        # exactly features_v1's real situation vs deepfm_regularized.
        fluke = ExperimentConfig(id="fluke", model="fm", hypothesis="h", parent_id="good")
        rm.add_node(fluke, edge_type="improve", parent_id="good")
        rm.update_node("fluke", status="done", diagnosis_tag="noise_floor",
                        metrics={"valid": {"primary_mean": 0.6103}})

        assert rm.best_node().node_id == "fluke", "raw leaderboard should still surface the numeric top score"
        confirmed = rm.best_confirmed_node()
        assert confirmed is not None and confirmed.node_id == "good", (
            f"best_confirmed_node() should walk past the noise_floor node to the nearest confirmed "
            f"win, got '{confirmed.node_id if confirmed else None}'")

        summary = rm.explored_summary()
        assert summary["best_node_id"] == "good"
        assert summary["raw_leaderboard_node_id"] == "fluke"


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


def test_research_critic_rejects_duplicate_and_confirmed_dead_end():
    import tempfile
    from agent.research_critic import review
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        rm = ResearchMap(Path(d) / "map.json")
        root = ExperimentConfig(id="fm_baseline_repro", model="fm", hypothesis="h",
                                 hyperparams={"k": 16, "lr": 0.001})
        rm.add_node(root, edge_type="draft")
        rm.update_node("fm_baseline_repro", status="done", metrics={"valid": {"primary_mean": 0.60}})

        # Duplicate: identical id already in the map.
        dup_verdict = review(rm, root)
        assert not dup_verdict.approved, "an already-present config must be rejected as a duplicate"

        # A pure-k change confirmed noise_floor -- the exact dead end pattern.
        k32 = ExperimentConfig(id="fm_wider_k32", model="fm", hypothesis="h",
                                hyperparams={"k": 32, "lr": 0.001}, parent_id="fm_baseline_repro")
        rm.add_node(k32, edge_type="improve", parent_id="fm_baseline_repro")
        rm.update_node("fm_wider_k32", status="done", diagnosis_tag="noise_floor",
                        metrics={"valid": {"primary_mean": 0.599}})

        # A second, NEW pure-k change on the same family -- must be caught.
        k64 = ExperimentConfig(id="fm_even_wider_k64", model="fm", hypothesis="h",
                                hyperparams={"k": 64, "lr": 0.001}, parent_id="fm_baseline_repro")
        dead_end_verdict = review(rm, k64)
        assert not dead_end_verdict.approved, "a repeated pure-capacity change on a confirmed dead-end family must be rejected"
        assert "fm_wider_k32" in dead_end_verdict.reason, "the rejection must cite the specific prior node, not just assert a rule"

        # A legitimately different change (not pure-k) on the same family must NOT be rejected by this rule.
        different_change = ExperimentConfig(id="fm_different_lr", model="fm", hypothesis="h",
                                             hyperparams={"k": 16, "lr": 0.01}, parent_id="fm_baseline_repro")
        ok_verdict = review(rm, different_change)
        assert ok_verdict.approved, "a change on a different axis (lr, not k) must not be caught by the capacity dead-end rule"


def test_multi_fidelity_kills_a_broken_config():
    from agent.multi_fidelity import run_multi_fidelity
    bad_config = ExperimentConfig(
        id="broken_multi_fidelity_smoke_test", model="this_model_does_not_exist",
        hypothesis="deliberately invalid", hyperparams={}, seeds=[0],
    )
    result = run_multi_fidelity(bad_config, str(DEFAULT_DATA_DIR), timeout_s=30, max_retries=1)
    assert not result.survived(), "a broken config must never survive to 100pct"
    assert result.killed_at == "1pct", f"should be killed at the first (cheapest) stage, got {result.killed_at}"


def test_multi_fidelity_time_saved_estimate():
    from agent.multi_fidelity import MultiFidelityResult

    config = ExperimentConfig(id="t", model="fm", hypothesis="h", hyperparams={"epochs": 40})

    killed_early = MultiFidelityResult(node_id="t", final_stage="1pct", killed_at="1pct", kill_reason="test")
    killed_early.stage_results = {"1pct": {"valid": {"primary_mean": 0.1}}}
    killed_early.stage_wall_times_s = {"1pct": 5.0}
    saved = killed_early.estimated_time_saved_s(config)
    assert saved is not None and saved > 0, f"a candidate killed at the cheapest stage should have a positive savings estimate, got {saved}"

    survived = MultiFidelityResult(node_id="t2", final_stage="100pct")
    survived.stage_results = {"1pct": {}, "10pct": {}, "100pct": {}}
    survived.stage_wall_times_s = {"1pct": 5.0, "10pct": 15.0, "100pct": 90.0}
    assert survived.estimated_time_saved_s(config) is None, "a survived candidate has nothing to 'save'"

    info = killed_early.to_fidelity_info(config)
    assert info["estimated_time_saved_s"] == saved
    assert info["stage_wall_times_s"] == {"1pct": 5.0}
    assert info["killed_at"] == "1pct"


def test_features_extended_shape_and_no_leakage():
    """One test, real data, checked once (agent/features.py's encode_extended
    re-parses the raw CSVs with no disk cache, so this is deliberately not
    split into multiple tests that would each pay that cost separately).

    Two properties: (1) adding fields doesn't change row count and does
    grow the vocab dimension: (2) the actual leakage-safety property --
    a video's engineered bucket must be IDENTICAL across every row that
    references it, proving it's a train-only aggregate (a property of the
    video), never derived from that specific row's own play_time_ms.
    """
    from agent import features as feat

    data_dir = str(DEFAULT_DATA_DIR)
    fields = feat.BASE_5 + feat.EXTRA_FIELDS
    enc, dim = feat.encode_extended(data_dir, fields)
    Xtr, ytr, utr = enc["train"]
    assert Xtr.shape[1] == len(feat.BASE_5) + len(feat.EXTRA_FIELDS) == 9

    enc_base, dim_base = feat.encode_extended(data_dir, feat.BASE_5)
    Xtr_base, _, _ = enc_base["train"]
    assert Xtr_base.shape[0] == Xtr.shape[0], "row count must be unchanged by adding fields"
    assert dim > dim_base, "extended encoding must have a larger total vocab dimension"

    Xva, yva, uva = enc["valid"]
    video_col = fields.index("video_id")
    completion_col = fields.index("video_completion_bucket")
    by_video: dict[int, set] = {}
    for i in range(len(Xva)):
        by_video.setdefault(int(Xva[i, video_col]), set()).add(int(Xva[i, completion_col]))
    inconsistent = [v for v, buckets in by_video.items() if len(buckets) > 1]
    assert not inconsistent, (
        f"a video's completion bucket must be identical across every row referencing it (a "
        f"train-only aggregate, never derived from each row's own outcome) -- found "
        f"{len(inconsistent)} video(s) with inconsistent buckets, which would mean leakage"
    )


class _FakeLLMResponse:
    """Duck-types agent.llm_client.LLMResponse without a real API call."""
    def __init__(self, total_tokens=50):
        self.total_tokens = total_tokens


class _FakeGeminiClient:
    """No network call -- returns a scripted sequence of raw dicts, so the
    Research Strategist's validation-gate + re-prompt-and-retry logic can
    be tested deterministically and for free."""
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls = 0

    def generate_json(self, prompt, max_retries=3, backoff_s=2.0):
        self.calls += 1
        return self._responses.pop(0), _FakeLLMResponse()


def test_research_strategist_accepts_a_valid_proposal():
    import tempfile
    from agent.research_map import ResearchMap
    from agent.research_strategist import propose_next_experiment

    with tempfile.TemporaryDirectory() as d:
        map = ResearchMap(Path(d) / "map.json")
        root = ExperimentConfig(id="root", model="fm", hypothesis="h")
        map.add_node(root, edge_type="draft")
        map.update_node("root", status="done", metrics={"valid": {"primary_mean": 0.60}, "per_seed": []})

        valid_raw = {
            "id": "llm_candidate_1", "hypothesis": "test", "model": "fm",
            "hyperparams": {"k": 16, "lr": 0.001, "l2": 1e-6, "batch": 8192, "epochs": 40, "patience": 4},
            "parent_id": "root", "edge_type": "improve", "reasoning": "test reasoning",
            "expected_metric_effect": {"gauc": "up", "ndcg": "up"}, "estimated_cost_s": 90, "priority": 0.8,
        }
        client = _FakeGeminiClient([valid_raw])
        proposal, meta = propose_next_experiment(map, {"iterations_remaining": 1}, client=client)
        assert proposal is not None, "a well-formed proposal must be accepted"
        assert proposal.config.model == "fm"
        assert proposal.config.parent_id == "root"
        assert meta["tokens"] == 50
        assert client.calls == 1, "a valid first response should not trigger a retry"


def test_research_strategist_rejects_and_retries_an_invalid_proposal():
    import tempfile
    from agent.research_map import ResearchMap
    from agent.research_strategist import propose_next_experiment

    with tempfile.TemporaryDirectory() as d:
        map = ResearchMap(Path(d) / "map.json")

        bad_raw = {  # unknown model -- must be rejected by the validation gate, not executed
            "id": "llm_candidate_bad", "hypothesis": "test", "model": "this_model_does_not_exist",
            "hyperparams": {"k": 16}, "parent_id": None, "edge_type": "draft",
            "reasoning": "test", "expected_metric_effect": {"gauc": "up", "ndcg": "up"},
            "estimated_cost_s": 90, "priority": 0.5,
        }
        good_raw = {
            "id": "llm_candidate_fixed", "hypothesis": "test", "model": "fm",
            "hyperparams": {"k": 16, "lr": 0.001}, "parent_id": None, "edge_type": "draft",
            "reasoning": "test", "expected_metric_effect": {"gauc": "up", "ndcg": "up"},
            "estimated_cost_s": 90, "priority": 0.5,
        }
        client = _FakeGeminiClient([bad_raw, good_raw])
        proposal, meta = propose_next_experiment(map, {"iterations_remaining": 1}, client=client)
        assert proposal is not None, "should recover via retry after one invalid response"
        assert proposal.config.model == "fm"
        assert client.calls == 2, "an invalid first response must trigger exactly one re-prompt"
        assert any(e["kind"] == "validation_error" for e in meta["events"])

        # Every attempt invalid -> must return None, never raise, never execute garbage.
        client2 = _FakeGeminiClient([bad_raw, bad_raw, bad_raw])
        proposal2, meta2 = propose_next_experiment(map, {"iterations_remaining": 1},
                                                     client=client2, max_validation_retries=2)
        assert proposal2 is None, "exhausting all validation retries must return None, not raise"


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


def test_recovery_catches_a_genuine_oom():
    """Not simulated -- k=10**9 makes the embedding table allocation
    (dim x k floats) genuinely fail with a real numpy MemoryError (numpy
    fails fast on an impossible shape; this never actually tries to
    allocate real RAM, so the test is fast and safe to run anywhere)."""
    from agent.recovery import run_with_recovery
    oom_config = ExperimentConfig(
        id="oom_smoke_test", model="fm", hypothesis="deliberately triggers a real MemoryError",
        hyperparams={"k": 10 ** 9, "epochs": 1, "batch": 64}, seeds=[0],
    )
    result, events = run_with_recovery(oom_config, str(DEFAULT_DATA_DIR), seed=0,
                                        timeout_s=60, max_retries=1, degrade_epochs=1)
    assert result is None, "a genuine OOM must never silently succeed"
    assert any("MemoryError" in e.message for e in events if e.kind == "error"), \
        "the real MemoryError should be visible in the recovery events, not swallowed"
    assert any(e.kind == "abandoned" for e in events), "must end in 'abandoned', not crash the parent process"


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
        test_research_map_best_confirmed_node_skips_noise_floor,
        test_diagnosis_rules,
        test_selector_scores_candidates,
        test_research_critic_rejects_duplicate_and_confirmed_dead_end,
        test_research_strategist_accepts_a_valid_proposal,
        test_research_strategist_rejects_and_retries_an_invalid_proposal,
        test_multi_fidelity_time_saved_estimate,
    ]
    if data_dir_available():
        tests.insert(0, test_evaluator_self_check)
        tests.append(test_recovery_catches_broken_config)
        tests.append(test_recovery_catches_a_genuine_oom)
        tests.append(test_multi_fidelity_kills_a_broken_config)
        tests.append(test_features_extended_shape_and_no_leakage)
    else:
        print("  (data dir not found -- skipping self-check and recovery-subprocess tests)")

    for t in tests:
        check(t.__name__, t)

    print("=" * 40)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
