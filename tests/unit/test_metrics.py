"""Unit tests for bvh_rssm.training.metrics.

Each test targets one invariant of one function.  All arithmetic is done with
tiny hand-computable arrays so expected values can be verified by inspection.
"""
import numpy as np
import pytest
from bvh_rssm.training.metrics import (
    mae_tau,
    c_index,
    brier_score,
    f1_switching,
    delta_return,
)


# ---------------------------------------------------------------------------
# mae_tau
# ---------------------------------------------------------------------------

class TestMaeTau:
    def test_perfect_prediction_is_zero(self):
        tau_pred = np.array([1.0, 5.0, 10.0])
        tau_star = np.array([1.0, 5.0, 10.0])
        assert mae_tau(tau_pred, tau_star) == pytest.approx(0.0)

    def test_scalar_error(self):
        # All predictions off by 2 — MAE must equal 2.0
        tau_pred = np.array([3.0, 7.0, 12.0])
        tau_star = np.array([1.0, 5.0, 10.0])
        assert mae_tau(tau_pred, tau_star) == pytest.approx(2.0)

    def test_asymmetric_errors(self):
        # errors: |4-1|=3, |2-5|=3, |10-10|=0 → mean=2.0
        tau_pred = np.array([4.0, 2.0, 10.0])
        tau_star = np.array([1.0, 5.0, 10.0])
        assert mae_tau(tau_pred, tau_star) == pytest.approx(2.0)

    def test_single_element(self):
        assert mae_tau(np.array([7.0]), np.array([3.0])) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# c_index
# ---------------------------------------------------------------------------

class TestCIndex:
    def test_perfect_predictor_returns_1(self):
        # pred ranks match oracle ranks exactly
        tau_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau_star = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert c_index(tau_pred, tau_star) == pytest.approx(1.0)

    def test_inverse_predictor_returns_0(self):
        # pred ranks perfectly reversed relative to oracle
        tau_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau_star = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert c_index(tau_pred, tau_star) == pytest.approx(0.0)

    def test_random_predictor_near_half(self):
        rng = np.random.default_rng(0)
        tau_pred = rng.uniform(0, 100, size=2000)
        tau_star = rng.uniform(0, 100, size=2000)
        score = c_index(tau_pred, tau_star)
        assert 0.45 < score < 0.55

    def test_range_is_0_to_1(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            tau_pred = rng.uniform(0, 10, 50)
            tau_star = rng.uniform(0, 10, 50)
            s = c_index(tau_pred, tau_star)
            assert 0.0 <= s <= 1.0

    def test_single_pair(self):
        # Only one valid pair (i=0, j=1): tau_star[0]<tau_star[1] and tau_pred[0]<tau_pred[1] → 1.0
        assert c_index(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_survival_function_is_zero(self):
        # S(t)[n, k] = 1 if tau*[n] > k else 0 — exactly the indicator I(tau* > t)
        # N=3 samples, K=5 time steps
        event_times = np.array([6, 6, 6])   # all survive past the window
        max_t = 5
        # Perfect: S(t)=1 for all t since event_times > max_t
        survival_curves = np.ones((3, 5))
        assert brier_score(survival_curves, event_times, max_t) == pytest.approx(0.0)

    def test_worst_survival_function(self):
        # S(t)=0 everywhere but oracle says everyone survives the window
        event_times = np.array([100, 100])
        max_t = 5
        survival_curves = np.zeros((2, 5))
        # Each cell error: (0 - 1)^2 = 1 → mean = 1.0
        assert brier_score(survival_curves, event_times, max_t) == pytest.approx(1.0)

    def test_early_event(self):
        # N=1, K=4, event at t=2 (0-indexed)
        # oracle indicator I(tau* > t) for t in 0..3: tau*=2 → [1,1,0,0]
        # S predicted perfectly: [1,1,0,0] → Brier = 0
        event_times = np.array([2])
        max_t = 4
        survival_curves = np.array([[1.0, 1.0, 0.0, 0.0]])
        assert brier_score(survival_curves, event_times, max_t) == pytest.approx(0.0)

    def test_output_is_scalar(self):
        rng = np.random.default_rng(2)
        curves = rng.uniform(0, 1, (10, 8))
        times = rng.integers(0, 20, 10)
        result = brier_score(curves, times, max_t=8)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# f1_switching
# ---------------------------------------------------------------------------

class TestF1Switching:
    def test_exact_match_f1_is_1(self):
        T = 20
        pred = np.zeros(T, dtype=bool)
        true = np.zeros(T, dtype=bool)
        pred[5] = True
        pred[15] = True
        true[5] = True
        true[15] = True
        precision, recall, f1 = f1_switching(pred, true, tolerance=0)
        assert f1 == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_no_overlap_f1_is_0(self):
        T = 20
        pred = np.zeros(T, dtype=bool)
        true = np.zeros(T, dtype=bool)
        pred[2] = True
        true[18] = True
        _, _, f1 = f1_switching(pred, true, tolerance=0)
        assert f1 == pytest.approx(0.0)

    def test_tolerance_allows_nearby_match(self):
        T = 20
        pred = np.zeros(T, dtype=bool)
        true = np.zeros(T, dtype=bool)
        pred[5] = True    # predicted at 5
        true[8] = True    # true at 8 — within tolerance=5
        precision, recall, f1 = f1_switching(pred, true, tolerance=5)
        assert f1 == pytest.approx(1.0)

    def test_tolerance_zero_rejects_nearby(self):
        T = 20
        pred = np.zeros(T, dtype=bool)
        true = np.zeros(T, dtype=bool)
        pred[5] = True
        true[8] = True   # distance 3, tolerance=0 → no match
        _, _, f1 = f1_switching(pred, true, tolerance=0)
        assert f1 == pytest.approx(0.0)

    def test_no_predictions_and_no_true_returns_1(self):
        # Edge case: nothing to detect, nothing predicted → F1=1
        pred = np.zeros(10, dtype=bool)
        true = np.zeros(10, dtype=bool)
        _, _, f1 = f1_switching(pred, true, tolerance=0)
        assert f1 == pytest.approx(1.0)

    def test_returns_three_floats(self):
        pred = np.array([True, False, False, True])
        true = np.array([False, True, False, True])
        result = f1_switching(pred, true, tolerance=1)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# delta_return
# ---------------------------------------------------------------------------

class TestDeltaReturn:
    def test_positive_advantage(self):
        bvh = np.array([10.0, 12.0, 8.0])
        baseline = np.array([5.0, 5.0, 5.0])
        assert delta_return(bvh, baseline) == pytest.approx(5.0)

    def test_negative_advantage(self):
        bvh = np.array([2.0, 2.0])
        baseline = np.array([5.0, 5.0])
        assert delta_return(bvh, baseline) == pytest.approx(-3.0)

    def test_zero_when_equal(self):
        vals = np.array([3.0, 7.0, 1.0])
        assert delta_return(vals, vals.copy()) == pytest.approx(0.0)

    def test_single_element(self):
        assert delta_return(np.array([9.0]), np.array([4.0])) == pytest.approx(5.0)
