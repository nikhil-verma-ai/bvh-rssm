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
    integrated_brier_score,
    time_dependent_auc,
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
# integrated_brier_score
# ---------------------------------------------------------------------------

class TestIntegratedBrierScore:
    def test_alias_matches_brier_score_exact(self):
        """IBS must return identical float to brier_score for same inputs."""
        rng = np.random.default_rng(42)
        curves = rng.uniform(0.0, 1.0, (10, 8))
        times = rng.integers(0, 20, 10)
        assert integrated_brier_score(curves, times, max_t=8) == brier_score(curves, times, max_t=8)

    def test_alias_matches_brier_score_perfect(self):
        """Perfect calibration: IBS == brier_score == 0."""
        event_times = np.array([100, 100, 100])
        max_t = 5
        survival_curves = np.ones((3, 5))
        assert integrated_brier_score(survival_curves, event_times, max_t) == pytest.approx(0.0)

    def test_alias_matches_brier_score_worst(self):
        """Worst calibration: IBS == brier_score == 1."""
        event_times = np.array([100, 100])
        max_t = 5
        survival_curves = np.zeros((2, 5))
        assert integrated_brier_score(survival_curves, event_times, max_t) == pytest.approx(1.0)

    def test_output_is_float(self):
        rng = np.random.default_rng(7)
        curves = rng.uniform(0, 1, (5, 4))
        times = rng.integers(0, 10, 5)
        result = integrated_brier_score(curves, times, max_t=4)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# time_dependent_auc
# ---------------------------------------------------------------------------

class TestTimeDependentAuc:
    def test_mean_auc_in_unit_interval_on_random_data(self):
        """mean_auc must lie in [0, 1] for arbitrary inputs."""
        rng = np.random.default_rng(13)
        N, K = 20, 10
        curves = rng.uniform(0.0, 1.0, (N, K))
        # event times span the full horizon to ensure both cases and controls exist
        times = rng.integers(0, K, N)
        _, mean_auc = time_dependent_auc(curves, times, max_t=K)
        assert 0.0 <= mean_auc <= 1.0

    def test_perfect_separation_gives_auc_near_1(self):
        """When S perfectly separates cases from controls, AUC(t) should be 1.0."""
        N, K = 8, 6
        # Cases: event_time = 0, so they are cases from t=0 onward. Give them S=0.
        # Controls: event_time = K (beyond window), S=1 everywhere.
        n_cases = 4
        n_ctrls = 4
        survival_curves = np.zeros((N, K))
        survival_curves[n_cases:, :] = 1.0     # controls have S=1
        event_times = np.array([0] * n_cases + [K] * n_ctrls)
        auc_per_t, mean_auc = time_dependent_auc(survival_curves, event_times, max_t=K)
        # At each t where both groups present, AUC should be 1.0
        valid = ~np.isnan(auc_per_t)
        assert valid.any(), "Expected at least one valid t with both cases and controls"
        assert np.allclose(auc_per_t[valid], 1.0)
        assert mean_auc == pytest.approx(1.0)

    def test_returns_correct_shapes(self):
        """auc_per_t must have length K; mean_auc must be a Python float."""
        N, K = 6, 5
        rng = np.random.default_rng(99)
        curves = rng.uniform(0, 1, (N, K))
        times = rng.integers(0, K, N)
        auc_per_t, mean_auc = time_dependent_auc(curves, times, max_t=K)
        assert auc_per_t.shape == (K,)
        assert isinstance(mean_auc, float)

    def test_nan_where_only_cases_or_controls(self):
        """At t=0, if no one has event_time <= 0, no cases exist → auc_per_t[0] is nan."""
        N, K = 4, 4
        # All event_times > 0, so at t=0 there are no cases
        survival_curves = np.ones((N, K)) * 0.5
        event_times = np.array([1, 2, 3, 4])
        auc_per_t, _ = time_dependent_auc(survival_curves, event_times, max_t=K)
        assert np.isnan(auc_per_t[0])

    def test_nan_when_fewer_than_two_valid_pairs(self):
        """Exactly 1 case × 1 control = 1 pair < 2 → must be nan (spec requirement)."""
        # At t=0: event_times[0]=0 is the only case; event_times[1..3] are controls.
        # n_cases=1, n_ctrls=3 → 3 pairs ≥ 2 → not nan.
        # At t=K-1: all 4 are cases; n_ctrls=0 → nan.
        # Construct so exactly one t has n_cases=1, n_ctrls=1 (1 pair < 2 → nan).
        N, K = 2, 3
        survival_curves = np.ones((N, K)) * 0.5
        # event_times=[0, 1]: at t=0 → case={0}, ctrl={1} → 1*1=1 pair → nan
        #                     at t=1 → case={0,1}, ctrl={} → 0 ctrls → nan
        #                     at t=2 → case={0,1}, ctrl={} → 0 ctrls → nan
        event_times = np.array([0, 1])
        auc_per_t, _ = time_dependent_auc(survival_curves, event_times, max_t=K)
        # t=0 has exactly 1 valid pair → spec requires nan
        assert np.isnan(auc_per_t[0])

    def test_inverse_separation_gives_auc_near_0(self):
        """When cases have S=1 and controls S=0, ranking is perfectly reversed → AUC=0."""
        N, K = 6, 4
        n_cases = 3
        n_ctrls = 3
        survival_curves = np.zeros((N, K))
        survival_curves[:n_cases, :] = 1.0     # cases have S=1 (wrong — higher than controls)
        # controls have S=0 (lower survival but event hasn't happened yet)
        event_times = np.array([0] * n_cases + [K] * n_ctrls)
        auc_per_t, mean_auc = time_dependent_auc(survival_curves, event_times, max_t=K)
        valid = ~np.isnan(auc_per_t)
        assert valid.any()
        assert np.allclose(auc_per_t[valid], 0.0)
        assert mean_auc == pytest.approx(0.0)


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
