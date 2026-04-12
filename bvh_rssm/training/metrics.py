"""
Pure metric functions for BVH-RSSM evaluation.

All functions are numpy-only — no torch, no gym imports.
All inputs are numpy arrays; all outputs are Python floats or tuples of floats.

Metrics:
    mae_tau          — Mean absolute error between predicted and oracle horizon.
    c_index          — Concordance index (pairwise ranking accuracy).
    brier_score      — Mean squared error between survival curve and survival indicator.
    f1_switching     — F1 for switch-point detection with temporal tolerance.
    delta_return     — E[Return_BVH] - E[Return_Baseline].
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
