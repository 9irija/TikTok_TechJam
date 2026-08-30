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


def test_convergence_detector_hits_iteration_cap_before_epsilon_n_rule():
    """Problem Statement Sec 2.3 "Limits": 50-iteration hard cap, on top of
    the epsilon/N rule. Feed a trajectory that keeps improving by MORE than
    epsilon every single iteration (so the epsilon/N rule alone would never
    fire) for more than 50 iterations -- the cap must still stop the run."""
    from agent.convergence import MAX_ITERATIONS
    cd = ConvergenceDetector(epsilon=0.002, n=3)
    for i in range(MAX_ITERATIONS + 5):
        cd.update(f"iter_{i}", 0.50 + i * 0.01)  # +0.01 > epsilon every time -- never stagnant
        if cd.is_converged():
            break
    assert cd.is_converged(), "must stop once MAX_ITERATIONS is reached, even mid-improvement"
    assert cd.stop_reason() == "iteration_cap", cd.stop_reason()
    assert len(cd.history) == MAX_ITERATIONS, \
        f"should stop exactly at the cap, not run past it: got {len(cd.history)} iterations"


def test_convergence_detector_hits_wall_clock_cap():
    """Problem Statement Sec 2.3's "6h wall-clock ceiling per run as a
    backstop" -- simulated by backdating the detector's own start time
    rather than actually waiting 6 hours."""
    from agent.convergence import MAX_WALL_CLOCK_S
    cd = ConvergenceDetector(epsilon=0.002, n=3)
    cd._run_start_s -= (MAX_WALL_CLOCK_S + 1.0)  # pretend the run started 6h+ ago
    cd.update("iter_0", 0.50 + 0.5)  # a real, still-improving iteration -- epsilon/N wouldn't fire
    assert cd.is_converged(), "must stop once the wall-clock ceiling is exceeded"
    assert cd.stop_reason() == "wall_clock_cap", cd.stop_reason()


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


def test_deepfm_mtl_forward_backward_reduces_loss():
    """P2's first torch model: same shape of test as every numpy model above
    (loss goes down on synthetic data, predict/get_state/set_state all
    behave), plus the one thing unique to it -- mtl_step actually consumes
    the 4 auxiliary labels and doesn't crash/produce non-finite output."""
    from agent.model_zoo.deepfm_mtl import build

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, aux_weight=0.2, seed=0)
    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, aux_weight=0.2, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_din_forward_backward_reduces_loss():
    """P2's sequence model (agent/model_zoo/deepfm_din.py): loss decreases
    on synthetic data, predict/get_state/set_state behave, PLUS the two
    things unique to it -- a fully-padded history (a brand-new user) must
    never produce NaN (the whole reason the attention mask uses -1e9, not
    -inf), and a real (non-padded) history must actually change the
    prediction relative to an all-padding one (proving the attention path
    is wired into the forward pass, not a dead branch)."""
    from agent.model_zoo.deepfm_din import build

    rng = np.random.default_rng(0)
    n, n_fields, dim, seq_len, video_vocab_size = 2000, 5, 200, 10, 50
    pad_idx = video_vocab_size - 1
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    seq = rng.integers(0, video_vocab_size, size=(n, seq_len)).astype(np.int32)
    cur_video = rng.integers(0, video_vocab_size, size=n).astype(np.int32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, video_vocab_size=video_vocab_size, seed=0)
    losses = [m.seq_step(X[i:i + 200], y[i:i + 200], seq[i:i + 200], cur_video[i:i + 200])
              for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict_seq(X, seq, cur_video)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    # A fully-padded history (a brand-new user) must not produce NaN/inf.
    all_pad_seq = np.full((n, seq_len), pad_idx, dtype=np.int32)
    preds_no_hist = m.predict_seq(X, all_pad_seq, cur_video)
    assert np.isfinite(preds_no_hist).all(), "an all-padding history must not produce NaN/inf"

    # A real history must actually move the prediction vs. an all-padding one -- proves
    # attention is wired into the forward pass, not silently a no-op.
    assert not np.allclose(preds, preds_no_hist), \
        "predictions with real history should differ from an all-padding history -- attention may be a dead branch"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, video_vocab_size=video_vocab_size, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict_seq(X, seq, cur_video), m2.predict_seq(X, seq, cur_video)), \
        "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_mtl_watch_forward_backward_reduces_loss():
    """deepfm_mtl.py extended with a 5th, continuous watch-time auxiliary
    head (agent/model_zoo/deepfm_mtl_watch.py) -- same shape of test as
    deepfm_mtl's own, plus proving mtl_watch_step actually consumes the
    continuous `watch` target (not just the 4 binary ones) without
    crashing or producing non-finite output."""
    from agent.model_zoo.deepfm_mtl_watch import build

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)
    watch = rng.uniform(0, 1, size=n).astype(np.float32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
              aux_weight=0.2, watch_weight=0.2, seed=0)
    losses = [m.mtl_watch_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200], watch[i:i + 200])
              for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
               aux_weight=0.2, watch_weight=0.2, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_din_mtl_forward_backward_reduces_loss():
    """agent/model_zoo/deepfm_din_mtl.py: DIN attention + MTL auxiliary
    heads combined into one model. Same shape of test as deepfm_din's own
    (loss decreases, predict/get_state/set_state behave, an all-padding
    history never produces NaN, a real history changes the prediction),
    plus proving seq_mtl_step actually consumes the 4 auxiliary labels
    alongside the sequence inputs without crashing."""
    from agent.model_zoo.deepfm_din_mtl import build

    rng = np.random.default_rng(0)
    n, n_fields, dim, seq_len, video_vocab_size = 2000, 5, 200, 10, 50
    pad_idx = video_vocab_size - 1
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)
    seq = rng.integers(0, video_vocab_size, size=(n, seq_len)).astype(np.int32)
    cur_video = rng.integers(0, video_vocab_size, size=n).astype(np.int32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
              aux_weight=0.2, video_vocab_size=video_vocab_size, seed=0)
    # 8 reps per slice, not 3 (deepfm_mtl's/deepfm_din's own figure): this model has strictly more
    # moving parts than either alone (attention block AND aux heads sharing one optimizer), so it
    # needs a few more steps to visibly overfit each fixed synthetic slice -- confirmed empirically,
    # not an arbitrary bump.
    losses = [m.seq_mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200], seq[i:i + 200], cur_video[i:i + 200])
              for i in range(0, n, 200) for _ in range(8)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict_seq(X, seq, cur_video)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    all_pad_seq = np.full((n, seq_len), pad_idx, dtype=np.int32)
    preds_no_hist = m.predict_seq(X, all_pad_seq, cur_video)
    assert np.isfinite(preds_no_hist).all(), "an all-padding history must not produce NaN/inf"
    assert not np.allclose(preds, preds_no_hist), \
        "predictions with real history should differ from an all-padding history -- attention may be a dead branch"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
               aux_weight=0.2, video_vocab_size=video_vocab_size, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict_seq(X, seq, cur_video), m2.predict_seq(X, seq, cur_video)), \
        "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_mtl_uncertainty_forward_backward_reduces_loss():
    """deepfm_mtl.py's architecture with learned per-task uncertainty
    weighting instead of one fixed aux_weight (agent/model_zoo/
    deepfm_mtl_uncertainty.py, Kendall et al. 2018) -- same shape of test
    as deepfm_mtl's own, plus proving the 5 learned log_var parameters
    actually move away from their zero-init (the whole point of the
    method) and stay finite."""
    from agent.model_zoo.deepfm_mtl_uncertainty import build

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, seed=0)
    log_vars_before = m.log_vars.detach().clone()
    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    assert not np.allclose(m.log_vars.detach().numpy(), log_vars_before.numpy()), \
        "learned uncertainty weights must move away from their zero-init during training"
    weights = m.learned_weights()
    assert all(np.isfinite(v) and v > 0 for v in weights.values()), f"learned weights must be finite and positive: {weights}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_listwise_forward_backward_reduces_loss():
    """agent/model_zoo/deepfm_listwise.py: DeepFM trained with a per-user
    listwise softmax loss instead of pointwise BCE. Loss decreases on a
    synthetic padded batch, predict() still works row-wise (flat (N,
    n_fields) input, not the padded (B, L, n_fields) training shape --
    proves _Net.forward's reshape trick handles both), and padded
    positions never leak into the loss (mask correctness): zeroing out a
    padded position's label must not change the loss, since it's masked
    out of the softmax entirely."""
    from agent.model_zoo.deepfm_listwise import build

    rng = np.random.default_rng(0)
    B, L, n_fields, dim = 16, 10, 5, 200
    X = rng.integers(0, dim, size=(B, L, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=(B, L)).astype(np.float32)
    mask = np.ones((B, L), dtype=bool)
    mask[:, 7:] = False  # last 3 slots per user are padding
    y[~mask] = 0

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    losses = [m.listwise_step(X, y, mask) for _ in range(20)]
    assert losses[-1] < losses[0], f"loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X.reshape(-1, n_fields))
    assert preds.shape == (B * L,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    # Flipping a PADDED position's label must not change the loss at all -- it's masked
    # out of the softmax, so its "target" is never read.
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    y_flipped = y.copy()
    y_flipped[:, 7:] = 1 - y_flipped[:, 7:]  # flip only the padded (masked-out) labels
    l1 = m2.listwise_step(X, y, mask)
    m3 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    l2 = m3.listwise_step(X, y_flipped, mask)
    assert abs(l1 - l2) < 1e-6, "flipping a padded/masked-out label must not change the loss"

    state = m.get_state()
    m4 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=2)
    m4.set_state(state)
    assert np.allclose(m.predict(X.reshape(-1, n_fields)), m4.predict(X.reshape(-1, n_fields))), \
        "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_pdaom_forward_backward_reduces_loss():
    """agent/model_zoo/deepfm_pdaom.py: DeepFM trained with a pairwise
    exponential AUC loss + per-user hard-pair mining. Loss decreases on a
    synthetic padded pos/neg candidate batch, predict() works on flat
    row-wise input, and -- the thing unique to hard mining -- a padded
    candidate must never be selected as the "hardest" pair: masking
    (via masked_fill to +-1e9 before min/max) must make the loss
    identical whether or not padded slots contain wildly different field
    ids, since real min/max selection should only ever look at the
    masked, real candidates."""
    from agent.model_zoo.deepfm_pdaom import build

    rng = np.random.default_rng(0)
    B, Kp, Kn, n_fields, dim = 16, 5, 5, 5, 200
    Xp = rng.integers(0, dim, size=(B, Kp, n_fields)).astype(np.int32)
    Xn = rng.integers(0, dim, size=(B, Kn, n_fields)).astype(np.int32)
    mp = np.ones((B, Kp), dtype=bool); mp[:, 3:] = False
    mn = np.ones((B, Kn), dtype=bool); mn[:, 3:] = False

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    losses = [m.pdaom_step(Xp, mp, Xn, mn) for _ in range(20)]
    assert losses[-1] < losses[0], f"loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(Xp.reshape(-1, n_fields))
    assert preds.shape == (B * Kp,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    # Changing ONLY the padded candidates' field ids must not change the loss at all.
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    l1 = m2.pdaom_step(Xp, mp, Xn, mn)
    Xp_altered_pad, Xn_altered_pad = Xp.copy(), Xn.copy()
    Xp_altered_pad[:, 3:] = rng.integers(0, dim, size=(B, 2, n_fields))
    Xn_altered_pad[:, 3:] = rng.integers(0, dim, size=(B, 2, n_fields))
    m3 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    l2 = m3.pdaom_step(Xp_altered_pad, mp, Xn_altered_pad, mn)
    assert abs(l1 - l2) < 1e-6, "changing only padded/masked-out candidates must not change the loss"

    state = m.get_state()
    m4 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=2)
    m4.set_state(state)
    assert np.allclose(m.predict(Xp.reshape(-1, n_fields)), m4.predict(Xp.reshape(-1, n_fields))), \
        "get_state/set_state round-trip must reproduce predictions"


def test_load_watch_ratio_shape_and_range():
    """agent/features.py's load_watch_ratio: row-aligned with load_aux_labels
    (same split sizes), values normalized into [0, 1] by construction (clip
    then divide by clip), and no leakage into the main label -- this is a
    training TARGET only, never wired into any model's input fields."""
    from agent.features import load_aux_labels, load_watch_ratio
    from agent.paths import data_dir_available

    if not data_dir_available():
        print("    (skipped -- KuaiRand-Pure data not present in this environment)")
        return

    aux = load_aux_labels()
    watch = load_watch_ratio()
    for split in ("train", "valid", "test"):
        assert len(watch[split]) == len(aux[split]), f"{split}: watch/aux row counts must match"
        assert watch[split].min() >= 0.0 and watch[split].max() <= 1.0, \
            f"{split}: watch ratio must be normalized into [0, 1], got range " \
            f"[{watch[split].min()}, {watch[split].max()}]"


def test_deepfm_bpr_forward_backward_reduces_loss():
    """P2's third torch model (agent/model_zoo/deepfm_bpr.py): DeepFM's
    architecture trained on fm_bpr.py's pairwise objective instead of
    pointwise logloss. Goes through the REAL Model Zoo registry (build_model),
    not a direct import, since this is the first torch model integrated
    into the actual pipeline (agent/experiment.py's existing is_bpr branch)
    rather than standalone-checked first -- proving that integration path
    actually works end-to-end, same shape of test as fm_bpr's own."""
    from agent.model_zoo.fm_bpr import build_user_pos_neg_index, sample_bpr_batches

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    users = [f"u{rng.integers(0, 50)}" for _ in range(n)]

    user_index = build_user_pos_neg_index(users, y)
    assert len(user_index) > 0, "synthetic data should produce at least one user with both a positive and negative row"

    m = build_model("deepfm_bpr", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=0)
    batch_rng = np.random.default_rng(1)
    losses = [m.bpr_step(xp, xn) for xp, xn in sample_bpr_batches(batch_rng, X, user_index, n_batches=30, batch_size=64)]
    assert losses[-1] < losses[0], f"BPR loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"
    state = m.get_state()
    m2 = build_model("deepfm_bpr", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_pcgrad_resolve_math_on_hand_constructed_cases():
    """Targeted correctness check of agent/model_zoo/deepfm_mtl_pcgrad.py's
    core `_pcgrad_resolve` -- the actual gradient-surgery math, not just
    "loss goes down" -- on three hand-constructed cases with known,
    hand-derived expected outcomes. This is the highest-risk new code in
    this project (manual multi-loss gradient manipulation, not a single
    .backward() call), so it gets checked against real numbers, not just
    smoke-tested."""
    import torch
    from agent.model_zoo.deepfm_mtl_pcgrad import _pcgrad_resolve

    rng = np.random.default_rng(0)

    # Case 1: directly opposing gradients (dot < 0, maximal conflict) --
    # each fully cancels when projected onto the other's orthogonal plane,
    # so the combined result is exactly zero regardless of resolution order.
    g_main = [torch.tensor([1.0, 0.0])]
    g_aux = [torch.tensor([-1.0, 0.0])]
    combined = _pcgrad_resolve(g_main, g_aux, rng)
    assert torch.allclose(combined[0], torch.zeros(2), atol=1e-6), \
        f"directly opposing gradients should fully cancel, got {combined[0]}"

    # Case 2: orthogonal gradients (dot == 0, no conflict) -- PCGrad must be a
    # no-op here; the combined result is exactly the unmodified sum.
    g_main = [torch.tensor([1.0, 0.0])]
    g_aux = [torch.tensor([0.0, 1.0])]
    combined = _pcgrad_resolve(g_main, g_aux, rng)
    assert torch.allclose(combined[0], torch.tensor([1.0, 1.0]), atol=1e-6), \
        f"orthogonal gradients should sum unmodified (no conflict to resolve), got {combined[0]}"

    # Case 3: same-direction gradients (dot > 0, reinforcing) -- also a no-op;
    # PCGrad only ever removes conflicting components, never reinforcing ones.
    g_main = [torch.tensor([2.0, 1.0])]
    g_aux = [torch.tensor([1.0, 0.5])]
    combined = _pcgrad_resolve(g_main, g_aux, rng)
    assert torch.allclose(combined[0], torch.tensor([3.0, 1.5]), atol=1e-6), \
        f"reinforcing gradients should sum unmodified, got {combined[0]}"


def test_deepfm_mtl_pcgrad_forward_backward_reduces_loss():
    """P2's gradient-surgery multi-task model, same shape of test as every
    other torch model above, through the REAL Model Zoo registry (reuses
    agent/experiment.py's existing is_mtl branch -- same mtl_step(X,y,aux)
    contract deepfm_mtl.py already validated, so this is wired directly
    into the real pipeline, not standalone-checked first)."""
    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)

    m = build_model("deepfm_mtl_pcgrad", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8],
                     lr=0.01, aux_weight=0.2, seed=0)
    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build_model("deepfm_mtl_pcgrad", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8],
                      lr=0.01, aux_weight=0.2, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_mtl_click_forward_backward_reduces_loss():
    """P2's is_click-extended MTL model (agent/model_zoo/deepfm_mtl_click.py),
    through the REAL Model Zoo registry -- same shape of test as
    deepfm_mtl's own, plus proving the 5-output aux head (not 4) trains
    correctly and declares aux_fields so agent/experiment.py's is_mtl
    branch knows to load the click-extended label set."""
    from agent.features import AUX_LABEL_FIELDS_WITH_CLICK
    from agent.model_zoo.deepfm_mtl_click import DeepFM_MTL_Click

    assert DeepFM_MTL_Click.aux_fields == AUX_LABEL_FIELDS_WITH_CLICK
    assert len(AUX_LABEL_FIELDS_WITH_CLICK) == 5 and "is_click" in AUX_LABEL_FIELDS_WITH_CLICK

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 5)).astype(np.float32)

    m = build_model("deepfm_mtl_click", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
                     aux_weight=0.2, seed=0)
    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build_model("deepfm_mtl_click", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
                      aux_weight=0.2, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_focal_loss_downweights_easy_examples_relative_to_bce():
    """Targeted correctness check of agent/model_zoo/deepfm_mtl_focal.py's
    FocalLoss -- the actual modulating-factor math, not just "loss goes
    down". Three properties, all checked against hand-reasoned expected
    behavior: (1) for a confidently-CORRECT prediction, focal loss is much
    smaller than plain BCE on the identical input (the whole point of the
    (1-p_t)^gamma modulating factor); (2) for a confidently-WRONG
    prediction, focal and BCE stay close (the modulating factor is near 1
    when p_t is near 0 -- hard examples aren't suppressed); (3) gamma=0
    exactly recovers alpha-weighted BCE (the modulating factor becomes
    identically 1), a direct algebraic sanity check of the formula itself.

    Cases 1-2 use alpha=0.5 -- the model's own real default, not a value
    picked to dodge the confound below. `alpha_t = alpha` for positives,
    `(1-alpha)` for negatives is the standard convention (a weight PER
    CLASS, positive vs. negative) -- there is no alpha that makes both
    simultaneously 1 (an earlier version of this test tried alpha=1.0 to
    "disable" alpha entirely and instead zeroed the negative class's
    weight completely, `alpha_t=1*0 + (1-1)*1=0`, silently making case 2's
    focal loss exactly 0 regardless of gamma -- a real bug in the test,
    not the implementation, caught by the assertion firing with a
    suspicious focal=0.000000 rather than a plausible-looking wrong
    number). alpha=0.5 is genuinely symmetric (equal weight, 0.5, for
    both classes) but does uniformly scale every example -- easy and hard
    alike -- by that same 0.5 factor, confirmed by hand-derivation:
    BCE=5.0067, focal=2.4700, ratio=0.4933 = alpha(0.5) * modulating
    (0.9866). The thresholds below are written against that known 0.5x
    baseline, not against raw BCE, so what's actually being checked is
    still the modulating factor's behavior, not alpha's."""
    import torch
    import torch.nn.functional as F
    from agent.model_zoo.deepfm_mtl_focal import FocalLoss

    # Case 1: confidently correct (logit strongly positive, target=1) -- focal should be MUCH smaller
    # than the alpha=0.5 baseline (0.5 * BCE), since the modulating factor also crushes it toward 0.
    logits = torch.tensor([5.0])
    targets = torch.tensor([1.0])
    bce = F.binary_cross_entropy_with_logits(logits, targets).item()
    focal = FocalLoss(gamma=2.0, alpha=0.5).forward(logits, targets).item()
    assert focal < bce * 0.5 * 0.1, \
        f"a confidently-correct example should be heavily down-weighted below the 0.5x alpha baseline: BCE={bce:.6f}, focal={focal:.6f}"

    # Case 2: confidently WRONG (logit strongly positive, target=0) -- focal should land close to the
    # alpha=0.5 baseline (0.5 * BCE), since the modulating factor (1-p_t)^gamma -> 1 as p_t -> 0 --
    # i.e. hard examples are NOT further suppressed beyond alpha's own uniform scaling.
    logits_wrong = torch.tensor([5.0])
    targets_wrong = torch.tensor([0.0])
    bce_wrong = F.binary_cross_entropy_with_logits(logits_wrong, targets_wrong).item()
    focal_wrong = FocalLoss(gamma=2.0, alpha=0.5).forward(logits_wrong, targets_wrong).item()
    assert focal_wrong > bce_wrong * 0.5 * 0.9, \
        f"a confidently-wrong example should land close to the 0.5x alpha baseline, not be further " \
        f"suppressed by the modulating factor: BCE={bce_wrong:.6f}, focal={focal_wrong:.6f}"

    # Case 3: at gamma=0 the modulating factor (1-p_t)^0 is identically 1 for every example, so
    # FocalLoss must reduce to EXACTLY alpha-weighted BCE -- 0.5 * plain BCE at alpha=0.5 (alpha_t=0.5
    # for every example regardless of class, per the same alpha convention cases 1-2 already established:
    # not "unweighted", a uniform half-scaling). Direct algebra check of the gamma=0 edge case.
    rng_logits = torch.tensor([-2.0, -0.5, 0.3, 1.7, 4.0])
    rng_targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    bce_batch = F.binary_cross_entropy_with_logits(rng_logits, rng_targets).item()
    focal_gamma0 = FocalLoss(gamma=0.0, alpha=0.5).forward(rng_logits, rng_targets).item()
    assert abs(focal_gamma0 - bce_batch * 0.5) < 1e-6, \
        f"gamma=0, alpha=0.5 must exactly equal 0.5x BCE: got {focal_gamma0}, expected {bce_batch * 0.5}"


def test_deepfm_mtl_focal_forward_backward_reduces_loss():
    """P2's focal-loss MTL model, through the REAL Model Zoo registry.
    Same shape of test as deepfm_mtl's own -- loss decreases, predict/
    get_state/set_state behave -- confirming the full model (not just the
    isolated FocalLoss module above) trains correctly end to end."""
    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)

    m = build_model("deepfm_mtl_focal", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
                     aux_weight=0.2, focal_gamma=2.0, focal_alpha=0.5, seed=0)
    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build_model("deepfm_mtl_focal", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01,
                      aux_weight=0.2, focal_gamma=2.0, focal_alpha=0.5, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"


def test_lambdarank_delta_ndcg5_math_on_hand_derived_cases():
    """Targeted correctness check of agent/model_zoo/deepfm_lambdarank.py's
    delta-nDCG@5 weighting -- the highest-risk new math in this project
    (a real formula, not just "loss decreases on synthetic data"), checked
    against hand-derived values before ever training on real data.
    """
    from agent.model_zoo.deepfm_lambdarank import dcg_discount, delta_ndcg5_for_pair, idcg_at_5

    # dcg_discount: 1/log2(rank+1), rank 1-indexed.
    assert abs(dcg_discount(1) - 1.0) < 1e-9, dcg_discount(1)
    assert abs(dcg_discount(2) - 0.6309297535714575) < 1e-9, dcg_discount(2)
    assert abs(dcg_discount(5) - 0.38685280723454163) < 1e-9, dcg_discount(5)

    # idcg_at_5: capped at 5 positives regardless of how many more exist.
    assert idcg_at_5(0) == 0.0
    assert abs(idcg_at_5(3) - 2.1309297535714578) < 1e-9, idcg_at_5(3)
    assert idcg_at_5(10) == idcg_at_5(5), "idcg@5 must be capped at 5 positives, not grow past it"

    # delta_ndcg5_for_pair: a positive at rank 1 swapped with a negative at rank 2 (user has 2
    # positives total) -- hand-derived: |discount(1)-discount(2)| / idcg5(2) = 0.22629...
    delta = delta_ndcg5_for_pair(rank_pos=1, rank_neg=2, idcg5=idcg_at_5(2))
    assert abs(delta - 0.22629438553091677) < 1e-9, delta

    # A pair entirely below rank 5 contributes exactly 0 -- nDCG@5 is blind to swaps outside the top 5.
    delta_below = delta_ndcg5_for_pair(rank_pos=10, rank_neg=11, idcg5=idcg_at_5(2))
    assert delta_below == 0.0, f"a pair entirely below rank 5 must contribute zero weight, got {delta_below}"

    # A pair straddling the rank-5 cutoff (positive AT rank 5, negative just below it) is still
    # weighted -- the positive's own discount still counts even though the negative's doesn't.
    delta_straddle = delta_ndcg5_for_pair(rank_pos=5, rank_neg=6, idcg5=idcg_at_5(1))
    assert delta_straddle > 0.0, "a pair with one rank inside the top 5 must contribute nonzero weight"

    # idcg5=0 (a user with zero positives) must never divide by zero.
    assert delta_ndcg5_for_pair(rank_pos=1, rank_neg=2, idcg5=0.0) == 0.0


def test_deepfm_lambdarank_step_reduces_loss_and_handles_edge_cases():
    """Standard shape of test (loss decreases, predict/get_state/set_state
    behave) PLUS the two edge cases unique to LambdaRank's per-user pair
    sampling: a batch where every user is single-class (no positive/negative
    pair exists anywhere) must return 0.0 and not crash; a real mixed batch
    must produce a finite, decreasing loss."""
    from agent.model_zoo.deepfm_lambdarank import build

    rng = np.random.default_rng(0)
    n_fields, dim, L, B = 5, 200, 16, 8
    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, max_pairs_per_user=5, seed=0)

    # Edge case: every "user" (row in the batch) is single-class (all positive) -- zero eligible pairs.
    X_pad = rng.integers(0, dim, size=(B, L, n_fields)).astype(np.int32)
    y_all_pos = np.ones((B, L), dtype=np.float32)
    mask = np.ones((B, L), dtype=bool)
    loss_no_pairs = m.lambdarank_step(X_pad, y_all_pos, mask)
    assert loss_no_pairs == 0.0, "a batch with zero eligible pairs must return 0.0, not crash"

    # Real mixed batch: each user has both classes -- loss should be finite and should decrease
    # over repeated steps on the same fixed batch.
    y_mixed = (rng.uniform(size=(B, L)) < 0.4).astype(np.float32)
    y_mixed[:, 0] = 1.0  # guarantee at least one positive per user
    y_mixed[:, 1] = 0.0  # guarantee at least one negative per user
    losses = [m.lambdarank_step(X_pad, y_mixed, mask) for _ in range(20)]
    assert all(np.isfinite(losses)), f"all losses must be finite: {losses}"
    assert losses[-1] < losses[0], f"loss should decrease on a fixed synthetic batch: {losses[0]} -> {losses[-1]}"

    X_flat = X_pad.reshape(-1, n_fields)
    preds = m.predict(X_flat)
    assert preds.shape == (B * L,)
    assert np.isfinite(preds).all()

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.05, max_pairs_per_user=5, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X_flat), m2.predict(X_flat)), "get_state/set_state round-trip must reproduce predictions"


def test_deepfm_mtl_deep_heads_forward_backward_reduces_loss():
    """Same shape of test as deepfm_mtl's own (main-task loss decreases,
    mtl_step consumes the 4 aux labels, predict/get_state/set_state all
    behave), plus the one thing unique to this variant -- each of the 4
    auxiliary heads must be an INDEPENDENT small MLP (not one shared
    nn.Linear the way deepfm_mtl.py's own _Net is), checked directly by
    confirming they don't share parameter objects."""
    from agent.model_zoo.deepfm_mtl_deep_heads import build

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    aux = rng.integers(0, 2, size=(n, 4)).astype(np.float32)

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, aux_weight=0.2, head_hidden=4, seed=0)

    # Each aux head must be its own MLP with independent parameters, not a shared layer.
    head_param_ids = [id(p) for head in m.net.aux_heads for p in head.parameters()]
    assert len(head_param_ids) == len(set(head_param_ids)), "each auxiliary head must own independent parameters"
    assert len(m.net.aux_heads) == 4, "must have exactly 4 independent auxiliary heads"

    losses = [m.mtl_step(X[i:i + 200], y[i:i + 200], aux[i:i + 200]) for i in range(0, n, 200) for _ in range(3)]
    assert losses[-1] < losses[0], f"main-task loss should decrease on synthetic data: {losses[0]} -> {losses[-1]}"

    preds = m.predict(X)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], lr=0.01, aux_weight=0.2, head_hidden=4, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"

    from agent.model_zoo import build as build_registry
    m3 = build_registry("deepfm_mtl_deep_heads", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], seed=0)
    assert m3.predict(X).shape == (n,), "must also be reachable through the real registry, not just its own build()"


def test_dcnv2_forward_backward_reduces_loss():
    """Same shape of test as every other model (loss decreases, predict/
    get_state/set_state all behave), plus the one thing unique to DCNv2's
    own forward pass -- the cross network's per-layer matrix-vector product
    must produce finite output across several stacked layers (a common
    failure mode for a hand-built cross network: growing or vanishing
    magnitudes through repeated x0-anchored multiplication), checked at
    n_cross_layers=3, one more than the default of 2.

    Fixed-batch design (repeated steps on the SAME rows, like
    test_deepfm_lambdarank_step_reduces_loss_and_handles_edge_cases),
    not the shifting-batch design most other models' tests use: a first
    attempt at the shifting design genuinely failed here -- lr=0.01 with 3
    stacked cross layers is noisier run-to-run across different random
    batches than DeepFM's own gentler curve tolerates, even though the
    backward pass itself is correct (verified separately: the same lr on a
    FIXED batch drives loss from ~0.70 to ~0.0015 in 30 steps). A real bug
    would still fail this version too -- it isolates "does the gradient
    genuinely reduce the loss it was computed from" from "is 30 steps
    enough to beat batch-to-batch data variance," which is what actually
    broke the first attempt, not the model."""
    from agent.model_zoo.dcnv2 import build

    rng = np.random.default_rng(0)
    n, n_fields, dim = 2000, 5, 200
    X_all = rng.integers(0, dim, size=(n, n_fields)).astype(np.int32)
    y_all = rng.integers(0, 2, size=n).astype(np.float32)
    X, y = X_all[:200], y_all[:200]  # one fixed batch, reused every step

    m = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], n_cross_layers=3, lr=0.01, seed=0)
    losses = [m.step(X, y) for _ in range(30)]
    assert losses[-1] < losses[0], f"loss should decrease on a fixed synthetic batch: {losses[0]} -> {losses[-1]}"
    assert np.isfinite(losses).all(), f"all losses must be finite: {losses}"

    preds = m.predict(X_all)
    assert preds.shape == (n,), preds.shape
    assert np.isfinite(preds).all(), "predictions must be finite -- a cross network's repeated " \
        "x0-anchored matrix product is a real place for magnitudes to blow up or vanish"

    state = m.get_state()
    m2 = build(dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], n_cross_layers=3, lr=0.01, seed=2)
    m2.set_state(state)
    assert np.allclose(m.predict(X), m2.predict(X)), "get_state/set_state round-trip must reproduce predictions"

    from agent.model_zoo import build as build_registry
    m3 = build_registry("dcnv2", dim=dim, n_fields=n_fields, k=8, hidden=[16, 8], seed=0)
    assert m3.predict(X_all).shape == (n,), "must also be reachable through the real registry, not just its own build()"


def test_load_aux_labels_with_click_field_matches_default_on_shared_columns():
    """Real-data check: agent/features.py's load_aux_labels(fields=...)
    parameter must return the SAME values for the original 4 columns
    whether called with the default AUX_LABEL_FIELDS or the click-extended
    AUX_LABEL_FIELDS_WITH_CLICK -- proves the new `fields` parameter is a
    pure extension, not a behavior change for every already-validated
    deepfm_mtl_v1-family node."""
    from agent.features import AUX_LABEL_FIELDS, AUX_LABEL_FIELDS_WITH_CLICK, load_aux_labels

    default = load_aux_labels(str(DEFAULT_DATA_DIR))
    extended = load_aux_labels(str(DEFAULT_DATA_DIR), fields=AUX_LABEL_FIELDS_WITH_CLICK)
    for split in ("train", "valid", "test"):
        assert extended[split].shape == (default[split].shape[0], 5)
        for i, f in enumerate(AUX_LABEL_FIELDS):
            assert np.array_equal(default[split][:, i], extended[split][:, i]), \
                f"'{f}' column diverged between the default and click-extended aux label calls ({split})"


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


def test_research_map_best_confirmed_node_searches_every_branch_not_just_one_lineage():
    """Regression test for the gap the DIN sibling-branch scenario surfaced
    live: `best_confirmed_node()` used to walk up only the numeric leader's
    own parent chain, so a higher-scoring *confirmed* node on a completely
    different branch -- one that was never itself the raw numeric leader --
    could be silently missed. Two children of the same parent, one a lower-
    scoring confirmed win and one a higher-scoring but unconfirmed sibling,
    is exactly that shape (deepfm_mtl_v1 vs. deepfm_din_v1 under
    deepfm_regularized in the real Research Map, just not close enough in
    score yet to have actually triggered the bug there)."""
    import tempfile
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        rm = ResearchMap(Path(d) / "research_map.json")
        root = ExperimentConfig(id="root", model="fm", hypothesis="h", parent_id=None)
        rm.add_node(root, edge_type="draft")
        rm.update_node("root", status="done", diagnosis_tag="baseline_beat",
                        metrics={"valid": {"primary_mean": 0.60}})

        # Branch A: a confirmed win over root, but the lower score of the two siblings.
        branch_a = ExperimentConfig(id="branch_a", model="fm", hypothesis="h", parent_id="root")
        rm.add_node(branch_a, edge_type="improve", parent_id="root")
        rm.update_node("branch_a", status="done", diagnosis_tag="clear_improvement",
                        metrics={"valid": {"primary_mean": 0.615}})

        # Branch B: root's OTHER child -- higher raw score (the numeric leader),
        # but unconfirmed vs. root, same as deepfm_din_v1 vs. deepfm_regularized.
        branch_b = ExperimentConfig(id="branch_b", model="fm", hypothesis="h", parent_id="root")
        rm.add_node(branch_b, edge_type="improve", parent_id="root")
        rm.update_node("branch_b", status="done", diagnosis_tag="ranking_tradeoff",
                        metrics={"valid": {"primary_mean": 0.62}})

        assert rm.best_node().node_id == "branch_b", "raw leaderboard should surface the numeric top score"
        confirmed = rm.best_confirmed_node()
        # The old walk-one-lineage implementation would start at branch_b (the
        # leader), see it's unconfirmed, walk to ITS parent (root, 0.60), and stop
        # there -- never considering branch_a (0.615, confirmed, a HIGHER score
        # than root but on a different branch than the leader).
        assert confirmed is not None and confirmed.node_id == "branch_a", (
            f"best_confirmed_node() must search every branch, not just the numeric leader's own "
            f"lineage, got '{confirmed.node_id if confirmed else None}'")


def test_research_map_save_merges_concurrent_writes():
    """Regression test for a real bug caught live during P2: `lgbm_baseline`
    was silently erased from logs/research_map.json because a long-running
    background process (tools/verify_multiseed.py) held a stale in-memory
    snapshot from before that node was added, and its own later save()
    overwrote the whole file. Simulates the same scenario: two ResearchMap
    instances over the same path, one adds a node the other never saw,
    and confirms both nodes survive after both have saved."""
    import tempfile
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "research_map.json"

        # Two independent processes both start from an empty/nonexistent map.
        rm1 = ResearchMap(path)
        rm2 = ResearchMap(path)

        # rm1 (like the short-lived "add lgbm_baseline" script) adds a node and saves first.
        cfg_a = ExperimentConfig(id="node_a", model="fm", hypothesis="h", parent_id=None)
        rm1.add_node(cfg_a, edge_type="draft")
        rm1.update_node("node_a", status="done", metrics={"valid": {"primary_mean": 0.60}})

        # rm2 (like the long-running verify_multiseed.py that started earlier and knows
        # nothing about node_a) adds a *different* node and saves after rm1 did.
        cfg_b = ExperimentConfig(id="node_b", model="fm", hypothesis="h", parent_id=None)
        rm2.add_node(cfg_b, edge_type="draft")
        rm2.update_node("node_b", status="done", metrics={"valid": {"primary_mean": 0.61}})

        # Without the merge-on-save fix, rm2's save would have silently erased node_a.
        rm3 = ResearchMap(path)  # fresh instance, reads whatever actually ended up on disk
        assert set(rm3.nodes.keys()) == {"node_a", "node_b"}, (
            f"both concurrently-added nodes should survive, got {sorted(rm3.nodes.keys())}")


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

        # A legitimately different change (not pure-k) on the same family must NOT be rejected by this rule
        # -- no prior confirmed dead end exists on the 'lr' axis yet.
        different_change = ExperimentConfig(id="fm_different_lr", model="fm", hypothesis="h",
                                             hyperparams={"k": 16, "lr": 0.01}, parent_id="fm_baseline_repro")
        ok_verdict = review(rm, different_change)
        assert ok_verdict.approved, "a change on a different axis (lr, not k) must not be caught by the capacity dead-end rule"

        # The rule generalizes beyond k: a confirmed dead end on a DIFFERENT single axis (mirrors the real
        # deepfm_wider/deepfm_default precedent, where only 'hidden' differs and it's tagged noise_floor)
        # must also be caught, not just the original k-specific case.
        deep_root = ExperimentConfig(id="deepfm_default", model="deepfm", hypothesis="h",
                                      hyperparams={"k": 16, "hidden": [32, 16]})
        rm.add_node(deep_root, edge_type="draft")
        rm.update_node("deepfm_default", status="done", metrics={"valid": {"primary_mean": 0.60}})
        deep_wider = ExperimentConfig(id="deepfm_wider", model="deepfm", hypothesis="h",
                                       hyperparams={"k": 16, "hidden": [64, 32]}, parent_id="deepfm_default")
        rm.add_node(deep_wider, edge_type="improve", parent_id="deepfm_default")
        rm.update_node("deepfm_wider", status="done", diagnosis_tag="noise_floor",
                        metrics={"valid": {"primary_mean": 0.601}})

        deep_even_wider = ExperimentConfig(id="deepfm_even_wider", model="deepfm", hypothesis="h",
                                            hyperparams={"k": 16, "hidden": [128, 64]}, parent_id="deepfm_default")
        generalized_verdict = review(rm, deep_even_wider)
        assert not generalized_verdict.approved, (
            "a repeated pure-'hidden' change on a confirmed dead-end family must be rejected too, "
            "not just the original pure-k case"
        )
        assert "deepfm_wider" in generalized_verdict.reason
        assert "hidden" in generalized_verdict.reason


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


def test_sequences_recent_history_never_leaks_the_future():
    """Synthetic, deterministic test of agent/sequences.py's core windowing
    logic -- no data dependency, always runs. Three properties: (1) a
    history never includes anything at or after the query timestamp (the
    single most important property in this file -- a strict `<`, not
    `<=`), (2) it returns exactly the `seq_len` most recent qualifying
    events, most-recent-last, (3) left-padding with "" when history is
    shorter than seq_len."""
    from agent.sequences import _recent_history, _user_histories

    rows = [
        {"user_id": "u1", "time_ms": 100, "video_id": "a"},
        {"user_id": "u1", "time_ms": 200, "video_id": "b"},
        {"user_id": "u1", "time_ms": 300, "video_id": "c"},
        {"user_id": "u1", "time_ms": 300, "video_id": "d"},  # same timestamp as "c" -- must never appear
        # in a query made exactly AT time_ms=300 (strict '<'), only in one made after it.
        {"user_id": "u2", "time_ms": 150, "video_id": "z"},
    ]
    hist = _user_histories(rows)

    # Query exactly at the "c"/"d" timestamp: neither may appear (strict '<', not '<=').
    got = _recent_history(hist, "u1", before_time_ms=300, seq_len=5)
    assert "c" not in got and "d" not in got, f"history at time_ms=300 leaked an event AT time_ms=300: {got}"
    assert got == ["", "", "", "a", "b"], f"unexpected history: {got}"

    # Query after everything: all 4 prior events for u1, most-recent-last, left-padded to seq_len.
    got_full = _recent_history(hist, "u1", before_time_ms=10_000, seq_len=6)
    assert got_full == ["", "", "a", "b", "c", "d"], f"unexpected full history: {got_full}"

    # seq_len shorter than available history: truncated to the most recent N, not the oldest N.
    got_trunc = _recent_history(hist, "u1", before_time_ms=10_000, seq_len=2)
    assert got_trunc == ["c", "d"], f"expected the 2 MOST RECENT events, got {got_trunc}"

    # A user with no history at all before the query time: fully padded.
    got_none = _recent_history(hist, "u1", before_time_ms=50, seq_len=3)
    assert got_none == ["", "", ""], f"expected all-padding for a query before any history, got {got_none}"

    # A different user's events never leak into u1's history.
    got_cross = _recent_history(hist, "u1", before_time_ms=10_000, seq_len=10)
    assert "z" not in got_cross, "u2's event leaked into u1's history"


def test_sequences_video_vocab_matches_encode_extended():
    """The real-data risk in agent/sequences.py: its own independently-built
    video_id vocab (`_build_video_vocab`) MUST assign the exact same
    integer code to the exact same video_id as agent/features.py's
    `encode_extended` does internally for the video_id field -- otherwise a
    shared embedding table would silently mix up unrelated videos. Verified
    by recovering the video_id field's offset from encode_extended's own
    output (its first train row is always assigned vocab code 0 by
    construction, in both implementations, since both build a fresh
    from-scratch dict over the identical row order) and checking every
    train row's recovered code against _build_video_vocab's independent
    computation.
    """
    from agent import features as feat
    from agent import sequences as seq

    data_dir = str(DEFAULT_DATA_DIR)
    splits = feat.load_splits(data_dir)
    enc, _dim = feat.encode_extended(data_dir, feat.BASE_5)
    Xtr, _, _ = enc["train"]
    video_col = feat.BASE_5.index("video_id")

    offset_video = int(Xtr[0, video_col])  # first train row's video_id -> vocab code 0 -> X value == offset
    my_vocab = seq._build_video_vocab(splits["train"])

    recovered = Xtr[:, video_col].astype(np.int64) - offset_video
    expected = np.array([my_vocab[x["video_id"]] for x in splits["train"]], dtype=np.int64)
    assert np.array_equal(recovered, expected), (
        "agent/sequences.py's video_id vocab diverges from encode_extended's own -- a shared "
        "embedding table would silently mix up unrelated videos between the candidate-item field "
        "and the history sequence."
    )


def test_sequences_encode_with_history_shape_and_no_future_leakage():
    """Real-data integration test of encode_with_history: correct shapes,
    and -- the property that matters most -- every row's history sequence
    contains only video_ids that genuinely appear somewhere in the full
    chronological log at a time_ms strictly before that row's own
    time_ms (checked directly against the raw per-user event lists, not
    re-derived from the same code path being tested)."""
    from agent import features as feat
    from agent import sequences as seq

    data_dir = str(DEFAULT_DATA_DIR)
    seq_len = 5
    enc, dim, seqs, cur_video, video_vocab_size = seq.encode_with_history(data_dir, seq_len=seq_len)
    Xva, yva, uva = enc["valid"]
    assert seqs["valid"].shape == (len(Xva), seq_len)
    assert cur_video["valid"].shape == (len(Xva),)
    assert seqs["train"].shape[0] == enc["train"][0].shape[0]
    assert seqs["valid"].max() < video_vocab_size and seqs["valid"].min() >= 0
    assert cur_video["valid"].max() < video_vocab_size and cur_video["valid"].min() >= 0

    splits = feat.load_splits(data_dir)
    all_rows = splits["train"] + splits["valid"] + splits["test"]
    histories = seq._user_histories(all_rows)
    my_vocab = seq._build_video_vocab(splits["train"])
    unk = len(my_vocab)
    code_to_video = {c: v for v, c in my_vocab.items()}

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(splits["valid"]), size=min(200, len(splits["valid"])), replace=False)
    for i in sample_idx:
        row = splits["valid"][i]
        expected_cur = my_vocab.get(row["video_id"], unk)
        assert cur_video["valid"][i] == expected_cur, (
            f"row {i}: cur_video code {cur_video['valid'][i]} doesn't match the row's own video_id "
            f"looked up through the same vocab ({expected_cur})"
        )
        times, vids = histories.get(row["user_id"], ([], []))
        for code in seqs["valid"][i]:
            if code == unk:
                continue  # UNK covers both "padding" and "a video not in the train vocab" -- can't
                          # distinguish those two cases from the code alone, so only check non-UNK codes
            video_id = code_to_video[code]
            # this exact video_id must appear in the user's real history at a time strictly
            # before the row's own time_ms
            matches = [t for t, v in zip(times, vids) if v == video_id and t < row["time_ms"]]
            assert matches, (
                f"row {i}: history contains video_id={video_id!r} (code={code}) but no matching "
                f"event exists in user {row['user_id']!r}'s log strictly before time_ms={row['time_ms']} "
                f"-- possible future leakage"
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


def test_dead_ends_section_generated_live_from_research_map_tags():
    """Regression test for the exact staleness gap docs/PHASE4_RESULTS.md
    Sec.6 flagged: the LLM prompt's dead-ends guidance used to be a
    hand-maintained string in _MODEL_HYPERPARAM_DOCS that went stale the
    moment a NEW dead end was confirmed (this project hit it for real --
    fm_bpr's old hardcoded note kept telling the LLM "'deepfm_bpr' is NOT
    a valid model name" long after deepfm_bpr had actually been built and
    registered). _dead_ends_section must instead reflect whatever's
    ACTUALLY in the Research Map at call time, with zero manual upkeep."""
    import tempfile
    from agent.research_map import ResearchMap
    from agent.research_strategist import _dead_ends_section

    with tempfile.TemporaryDirectory() as d:
        map = ResearchMap(Path(d) / "map.json")

        # A fresh map has no dead ends yet.
        assert "none confirmed" in _dead_ends_section(map)

        # Add a regression-tagged node for a NEW model family never mentioned in
        # _MODEL_HYPERPARAM_DOCS's old hand-written notes.
        cfg = ExperimentConfig(id="some_new_model_v1", model="some_new_model", hypothesis="h")
        map.add_node(cfg, edge_type="improve")
        map.update_node("some_new_model_v1", status="done", diagnosis_tag="regression",
                         metrics={"valid": {"primary_mean": 0.50}})

        section = _dead_ends_section(map)
        assert "some_new_model" in section and "some_new_model_v1" in section, (
            f"a brand-new model's confirmed regression must appear automatically, got: {section}"
        )
        assert "[regression]" in section


def test_run_p4_stops_immediately_when_wall_time_budget_already_exhausted():
    """Regression test for the real, previously-documented gap: the `budget`
    dict shown to the LLM used to be purely informational -- nothing in the
    loop actually stopped as it depleted. `max_wall_time_s=0` means the
    ceiling is exhausted before iteration 1 even starts, so the loop must
    stop WITHOUT ever calling the LLM (asserted via client.calls == 0, not
    just the reported status)."""
    import tempfile
    import agent.p4_orchestrator as p4mod

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        logs_dir, exp_dir = d / "logs", d / "experiments"
        logs_dir.mkdir()
        orig_map_path, orig_logs_dir, orig_exp_dir = p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR
        p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR = logs_dir / "research_map.json", logs_dir, exp_dir
        try:
            client = _FakeGeminiClient([])  # any generate_json call would raise (nothing scripted) -- proves it's unused
            report = p4mod.run_p4(max_iterations=3, client=client, max_wall_time_s=0.0)
        finally:
            p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR = orig_map_path, orig_logs_dir, orig_exp_dir

        assert client.calls == 0, "an already-exhausted wall-clock budget must stop before ever spending an LLM call"
        assert len(report["iterations"]) == 1
        assert report["iterations"][0]["status"] == "budget_exhausted_wall_time"


def test_run_p4_skips_low_priority_proposals_without_spending_compute():
    """A proposal whose OWN LLM-assigned priority is below `min_priority_to_run`
    must be declined before it ever reaches the Multi-Fidelity Runner -- same
    "logged as declined, not silently dropped" pattern as critic_rejected.
    Checked at three levels: the reported status, that the node never enters
    the Research Map (so a low-priority idea can't pollute best_node()/
    best_confirmed_node()), and that the loop asks the LLM again next
    iteration rather than stopping."""
    import tempfile
    import agent.p4_orchestrator as p4mod
    from agent.research_map import ResearchMap

    low_priority_raw = {
        "id": "llm_low_priority_candidate", "hypothesis": "test", "model": "fm",
        "hyperparams": {"k": 16, "lr": 0.001}, "parent_id": None, "edge_type": "draft",
        "reasoning": "a hunch, not a strong one", "expected_metric_effect": {"gauc": "up", "ndcg": "up"},
        "estimated_cost_s": 90, "priority": 0.1,
    }

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        logs_dir, exp_dir = d / "logs", d / "experiments"
        logs_dir.mkdir()
        orig_map_path, orig_logs_dir, orig_exp_dir = p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR
        p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR = logs_dir / "research_map.json", logs_dir, exp_dir
        try:
            # Two iterations, two scripted responses -- proves a skip doesn't stop the loop.
            client = _FakeGeminiClient([dict(low_priority_raw), dict(low_priority_raw)])
            report = p4mod.run_p4(max_iterations=2, client=client, min_priority_to_run=0.5)
        finally:
            p4mod.RESEARCH_MAP_PATH, p4mod.LOGS_DIR, p4mod.EXPERIMENTS_DIR = orig_map_path, orig_logs_dir, orig_exp_dir

        assert client.calls == 2, "a skip must still ask the LLM again next iteration, not stop the loop"
        assert len(report["iterations"]) == 2
        for it in report["iterations"]:
            assert it["status"] == "skipped_low_priority"
            assert it["diagnosis"]["tag"] == "skipped_low_priority"
        assert "llm_low_priority_candidate" not in ResearchMap(logs_dir / "research_map.json").nodes, (
            "a skipped-for-priority proposal must never be added to the Research Map"
        )


def test_verify_node_multiseed_reports_real_subprocess_wall_time():
    """Regression test for a real bug caught live: `verify_node_multiseed`
    used to time itself with `time.process_time()` wrapped around
    `run_with_recovery()`, which spawns a SEPARATE subprocess to do the
    actual training (agent/recovery.py, required on Windows) -- the
    wrapper's own process_time() only counts CPU time the wrapper itself
    burns (mostly idle time waiting on `proc.join()`), not the child's.
    Live symptom: a 3-seed DCNv2 verification that took several real
    minutes of subprocess training reported `wall_time_s=0.03`. Fixed by
    summing each per-seed result's own `wall_time_s` (measured INSIDE the
    subprocess by agent/experiment.py). This test trains a real, tiny,
    fast config (fm, 2 epochs) on two fresh seeds and checks the reported
    total is at least in the same neighborhood as genuine training time,
    not near-zero."""
    import tempfile
    from tools.verify_multiseed import verify_node_multiseed
    from agent.research_map import ResearchMap

    with tempfile.TemporaryDirectory() as d:
        map = ResearchMap(Path(d) / "map.json")
        cfg = ExperimentConfig(id="fm_wall_time_check", model="fm", hypothesis="h",
                                hyperparams={"k": 4, "lr": 0.01, "epochs": 2, "batch": 8192})
        map.add_node(cfg, edge_type="draft")
        map.update_node("fm_wall_time_check", status="done", metrics={"valid": {"primary_mean": 0.5}})

        out = verify_node_multiseed(map, "fm_wall_time_check", str(DEFAULT_DATA_DIR),
                                     seeds=[0, 1], timeout_s=120.0)
        assert out is not None, "a real, valid fm config must train successfully"
        agg, d, wall_time_s, newly_run = out
        assert newly_run == [0, 1], "neither seed was cached yet -- both must be freshly trained"
        assert wall_time_s > 0.5, (
            f"reported wall_time_s={wall_time_s} is implausibly small for two real subprocess training "
            f"runs -- this is exactly the bug this test guards against (the wrapper measuring its own "
            f"idle time instead of the subprocess's real training time)"
        )
        assert wall_time_s == sum(r["wall_time_s"] for r in agg["per_seed"]), (
            "must equal the sum of each per-seed result's own subprocess-measured wall_time_s exactly"
        )


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
        test_convergence_detector_hits_iteration_cap_before_epsilon_n_rule,
        test_convergence_detector_hits_wall_clock_cap,
        test_config_diff,
        test_fm_forward_backward_reduces_loss,
        test_deepfm_forward_backward_reduces_loss,
        test_fm_bpr_forward_backward_reduces_loss,
    ]
    try:
        import torch  # noqa: F401
        tests.append(test_deepfm_mtl_forward_backward_reduces_loss)
        tests.append(test_deepfm_din_forward_backward_reduces_loss)
        tests.append(test_deepfm_mtl_watch_forward_backward_reduces_loss)
        tests.append(test_deepfm_din_mtl_forward_backward_reduces_loss)
        tests.append(test_deepfm_mtl_uncertainty_forward_backward_reduces_loss)
        tests.append(test_deepfm_listwise_forward_backward_reduces_loss)
        tests.append(test_deepfm_pdaom_forward_backward_reduces_loss)
        tests.append(test_deepfm_bpr_forward_backward_reduces_loss)
        tests.append(test_pcgrad_resolve_math_on_hand_constructed_cases)
        tests.append(test_deepfm_mtl_pcgrad_forward_backward_reduces_loss)
        tests.append(test_deepfm_mtl_click_forward_backward_reduces_loss)
        tests.append(test_focal_loss_downweights_easy_examples_relative_to_bce)
        tests.append(test_deepfm_mtl_focal_forward_backward_reduces_loss)
        tests.append(test_lambdarank_delta_ndcg5_math_on_hand_derived_cases)
        tests.append(test_deepfm_lambdarank_step_reduces_loss_and_handles_edge_cases)
        tests.append(test_dcnv2_forward_backward_reduces_loss)
        tests.append(test_deepfm_mtl_deep_heads_forward_backward_reduces_loss)
    except ImportError:
        pass
    tests += [
        test_load_watch_ratio_shape_and_range,
        test_research_map_basic,
        test_research_map_best_confirmed_node_skips_noise_floor,
        test_research_map_best_confirmed_node_searches_every_branch_not_just_one_lineage,
        test_research_map_save_merges_concurrent_writes,
        test_diagnosis_rules,
        test_selector_scores_candidates,
        test_research_critic_rejects_duplicate_and_confirmed_dead_end,
        test_research_strategist_accepts_a_valid_proposal,
        test_research_strategist_rejects_and_retries_an_invalid_proposal,
        test_dead_ends_section_generated_live_from_research_map_tags,
        test_run_p4_stops_immediately_when_wall_time_budget_already_exhausted,
        test_run_p4_skips_low_priority_proposals_without_spending_compute,
        test_multi_fidelity_time_saved_estimate,
        test_sequences_recent_history_never_leaks_the_future,
    ]
    if data_dir_available():
        tests.insert(0, test_evaluator_self_check)
        tests.append(test_verify_node_multiseed_reports_real_subprocess_wall_time)
        tests.append(test_recovery_catches_broken_config)
        tests.append(test_recovery_catches_a_genuine_oom)
        tests.append(test_multi_fidelity_kills_a_broken_config)
        tests.append(test_features_extended_shape_and_no_leakage)
        tests.append(test_sequences_video_vocab_matches_encode_extended)
        tests.append(test_sequences_encode_with_history_shape_and_no_future_leakage)
        tests.append(test_load_aux_labels_with_click_field_matches_default_on_shared_columns)
    else:
        print("  (data dir not found -- skipping self-check and recovery-subprocess tests)")

    for t in tests:
        check(t.__name__, t)

    print("=" * 40)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
