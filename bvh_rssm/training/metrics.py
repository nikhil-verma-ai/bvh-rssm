"""
Pure metric functions for BVH-RSSM evaluation.

All functions are numpy-only — no torch, no gym imports.
All inputs are numpy arrays; all outputs are Python floats or tuples of floats.

Metrics:
    mae_tau                — Mean absolute error between predicted and oracle horizon.
    c_index                — Concordance index (pairwise ranking accuracy).
    brier_score            — Mean squared error between survival curve and survival indicator.
    integrated_brier_score — IBS alias for brier_score; makes IBS interpretation explicit.
    time_dependent_auc     — Time-dependent AUC(t) and mean AUC over all horizons.
    f1_switching           — F1 for switch-point detection with temporal tolerance.
    delta_return           — E[Return_BVH] - E[Return_Baseline].
"""
from __future__ import annotations

import numpy as np


def mae_tau(tau_pred: np.ndarray, tau_star: np.ndarray) -> float:
    """Mean absolute error between predicted and oracle horizon.

    Args:
        tau_pred: [N] predicted tau values (float).
        tau_star: [N] oracle tau values (float).

    Returns:
        Scalar MAE in the same units as tau.

    Complexity: O(N) time, O(1) extra space.
    """
    return float(np.mean(np.abs(tau_pred - tau_star)))


def c_index(tau_pred: np.ndarray, tau_star: np.ndarray) -> float:
    """Concordance index: P(τ̂_i < τ̂_j | τ*_i < τ*_j).

    Counts all concordant pairs (pred order matches oracle order) divided by
    all valid ordered pairs (i, j) where tau_star[i] < tau_star[j].
    Ties in tau_star are excluded. Ties in tau_pred count as 0.5.

    Range [0, 1]. Random predictor ≈ 0.5. Perfect predictor = 1.0.

    Args:
        tau_pred: [N] predicted tau values.
        tau_star: [N] oracle tau values.

    Returns:
        Scalar concordance index in [0, 1].

    Complexity: O(N^2) — acceptable for eval (N ≤ 10_000 typical).
    Side effects: none.
    """
    n = len(tau_pred)
    concordant = 0.0
    valid_pairs = 0

    for i in range(n):
        for j in range(n):
            if tau_star[i] >= tau_star[j]:
                # Only count pairs where oracle strictly orders i before j
                continue
            valid_pairs += 1
            if tau_pred[i] < tau_pred[j]:
                concordant += 1.0
            elif tau_pred[i] == tau_pred[j]:
                concordant += 0.5

    if valid_pairs == 0:
        # All oracle values are tied — concordance is undefined; return 0.5 by convention.
        return 0.5

    return float(concordant / valid_pairs)


def brier_score(
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    max_t: int,
) -> float:
    """Mean squared error between predicted S(t) and the survival indicator I(τ* > t).

    For each sample n and time step t ∈ {0, …, max_t-1}:
        error[n, t] = (S[n, t] - I(tau_star[n] > t))^2

    Returns the mean over all (n, t) pairs.

    Args:
        survival_curves: [N, K] — each row is S(0), S(1), …, S(K-1) for one sample.
                         K must equal max_t. Values in [0, 1].
        event_times:     [N] — oracle τ* per sample (integer steps).
        max_t:           Number of time steps K. Must match survival_curves.shape[1].

    Returns:
        Scalar Brier score in [0, 1].

    Raises:
        ValueError: if survival_curves.shape[1] != max_t.

    Complexity: O(N * K) time and space for indicator matrix construction.
    """
    n = survival_curves.shape[0]
    k = survival_curves.shape[1]
    if k != max_t:
        raise ValueError(
            f"survival_curves.shape[1]={k} does not match max_t={max_t}"
        )

    # Build indicator matrix: indicator[n, t] = 1 if tau_star[n] > t, else 0
    # t ranges over 0..max_t-1; event_times[n] is the oracle step count
    t_idx = np.arange(max_t)[np.newaxis, :]           # [1, K]
    # event_times[:, np.newaxis] broadcasts to [N, K]
    indicator = (event_times[:, np.newaxis] > t_idx).astype(np.float64)

    errors = (survival_curves - indicator) ** 2        # [N, K]
    return float(np.mean(errors))


def integrated_brier_score(
    survival_curves: np.ndarray,  # [N, K]
    event_times: np.ndarray,      # [N]
    max_t: int,
) -> float:
    """Integrated Brier Score (IBS) — mean squared calibration error over all time points.

    IBS = (1/K) * sum_{t=0}^{K-1} E[(S(t) - I(tau* > t))^2]

    For uniformly-spaced discrete time intervals this equals the mean of
    brier_score computed at each t — which is exactly what brier_score() returns
    (it averages over both n and t dimensions).

    Alias for brier_score() that makes the IBS interpretation explicit.
    Range [0, 0.25]. Lower is better. 0.0 = perfect calibration.

    Args:
        survival_curves: [N, K] — S(0)..S(K-1) per sample.
        event_times: [N] — oracle tau* per sample.
        max_t: Number of time steps K.

    Returns:
        Scalar IBS in [0, 0.25].
    """
    return brier_score(survival_curves, event_times, max_t)


def time_dependent_auc(
    survival_curves: np.ndarray,  # [N, K]
    event_times: np.ndarray,      # [N] oracle tau*
    max_t: int,
) -> tuple[np.ndarray, float]:
    """Time-dependent AUC at each horizon t.

    At each t, AUC(t) = P(S_i(t) < S_j(t) | tau_i* <= t < tau_j*)
    — among pairs where i is a "case" (event by t) and j is a "control"
    (event after t), what fraction did the model rank correctly?

    Cases: samples where tau_star[i] <= t  (shift already happened)
    Controls: samples where tau_star[j] > t  (shift not yet happened)
    Correct ranking: S_i(t) < S_j(t)  (lower survival for case)
    Ties count as 0.5.

    Args:
        survival_curves: [N, K] — S(0)..S(K-1) per sample.
        event_times: [N] — oracle tau* per sample (integer steps).
        max_t: Number of time steps K. Must equal survival_curves.shape[1].

    Returns:
        (auc_per_t, mean_auc): auc_per_t is [K] array of AUC at each horizon,
        nan where fewer than 2 valid pairs exist. mean_auc is the mean over
        non-nan entries.

    Complexity: O(N^2 * K) — use with N <= 5000.
    """
    N, K = survival_curves.shape
    if K != max_t:
        raise ValueError(
            f"survival_curves.shape[1]={K} does not match max_t={max_t}"
        )
    t_vals = event_times.astype(np.int64)
    auc_per_t = np.full(K, np.nan, dtype=np.float64)

    for t in range(K):
        case_mask = t_vals <= t      # [N] — event happened by t
        ctrl_mask = t_vals > t       # [N] — event hasn't happened yet
        n_cases = case_mask.sum()
        n_ctrls = ctrl_mask.sum()
        # Require at least 2 valid pairs (n_cases * n_ctrls >= 2) for a meaningful AUC
        if n_cases * n_ctrls < 2:
            continue

        S_cases = survival_curves[case_mask, t]   # [n_cases]
        S_ctrls = survival_curves[ctrl_mask, t]   # [n_ctrls]

        # Vectorized pairwise comparison: S_i(t) < S_j(t) for each (case, ctrl) pair
        # S_cases[:, None] < S_ctrls[None, :] → [n_cases, n_ctrls]
        concordant = (S_cases[:, None] < S_ctrls[None, :]).sum()
        tied = (S_cases[:, None] == S_ctrls[None, :]).sum()
        total = n_cases * n_ctrls
        auc_per_t[t] = float(concordant + 0.5 * tied) / total

    valid = ~np.isnan(auc_per_t)
    mean_auc = float(np.mean(auc_per_t[valid])) if valid.any() else float("nan")
    return auc_per_t, mean_auc


def f1_switching(
    predicted_switches: np.ndarray,   # [T] bool
    true_switches: np.ndarray,        # [T] bool
    tolerance: int = 5,
) -> tuple[float, float, float]:
    """F1 score for switch detection with temporal tolerance.

    A predicted switch within ±tolerance steps of a true switch counts as a
    True Positive (TP).  Each true switch can only be matched once (greedy,
    nearest-first).  Each predicted switch can only match one true switch.

    Args:
        predicted_switches: [T] boolean array — True at predicted switch steps.
        true_switches:      [T] boolean array — True at ground-truth switch steps.
        tolerance:          Maximum step distance for a match. Default 5.

    Returns:
        (precision, recall, f1) — all floats in [0, 1].

    Special cases:
        - No predictions and no true switches → (1.0, 1.0, 1.0).
        - No predictions but true switches exist → (1.0, 0.0, 0.0).
        - Predictions exist but no true switches → (0.0, 1.0, 0.0).

    Complexity: O(P * G) where P = |predicted|, G = |true|.
    Side effects: none.
    """
    pred_steps = np.where(predicted_switches)[0].tolist()
    true_steps = np.where(true_switches)[0].tolist()

    n_pred = len(pred_steps)
    n_true = len(true_steps)

    if n_pred == 0 and n_true == 0:
        return (1.0, 1.0, 1.0)
    if n_pred == 0:
        return (1.0, 0.0, 0.0)
    if n_true == 0:
        return (0.0, 1.0, 0.0)

    # Greedy matching: for each true switch, find nearest unmatched prediction within tolerance
    matched_pred = set()
    tp = 0

    for t in true_steps:
        best_dist = tolerance + 1
        best_p_idx = None
        for p_idx, p in enumerate(pred_steps):
            if p_idx in matched_pred:
                continue
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_p_idx = p_idx
        if best_p_idx is not None:
            matched_pred.add(best_p_idx)
            tp += 1

    precision = float(tp / n_pred)
    recall = float(tp / n_true)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    return (precision, recall, f1)


def delta_return(
    bvh_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> float:
    """E[Return_BVH] - E[Return_Baseline].

    Args:
        bvh_returns:      [N] episode returns for BVH agent.
        baseline_returns: [N] episode returns for baseline agent.

    Returns:
        Scalar advantage (positive = BVH better).

    Complexity: O(N).
    """
    return float(np.mean(bvh_returns) - np.mean(baseline_returns))
