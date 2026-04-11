"""
TradingRegime — Env 5 of FNSB.

A regime-switching stochastic process (Hidden Markov Model with 3 states:
trending up, trending down, mean-reverting). The agent observes a price
history window and places buy/sell/hold actions.

Shift mechanism: HMM transition probabilities change at shift epochs,
making regime transitions more or less likely.
Adversarial trigger: None (purely observational actions).

Purpose: directly relevant to quant trading domain; continuous hidden state,
discrete actions, hidden Markov structure — a natural POMDP.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bvh_rssm.envs.wrappers import ShiftWrapper

# HMM parameters for 3 regimes
_REGIME_DRIFT = np.array([0.005, -0.005, 0.000])   # per-step mean return
_REGIME_VOL   = np.array([0.010,  0.010, 0.005])   # per-step std dev
_N_REGIMES = 3
_PRICE_WINDOW = 20  # number of past prices in observation


class _TradingEnv(gym.Env):
    """Base trading environment with HMM-driven price dynamics."""

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(_PRICE_WINDOW,), dtype=np.float32
    )
    action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._price: float = 100.0
        self._regime: int = 0
        self._position: int = 0  # -1=short, 0=flat, 1=long
        self._price_history: list = [100.0] * _PRICE_WINDOW
        self._transition_matrix = self._default_transition()
        self._step_count: int = 0
        self._max_steps: int = 500

    @staticmethod
    def _default_transition() -> np.ndarray:
        """Default HMM transition matrix (tends to stay in current regime)."""
        return np.array([
            [0.95, 0.03, 0.02],
            [0.03, 0.95, 0.02],
            [0.02, 0.02, 0.96],
        ])

    def _step_price(self) -> float:
        """Advance price by one step under current regime."""
        drift = _REGIME_DRIFT[self._regime]
        vol   = _REGIME_VOL[self._regime]
        ret   = drift + vol * self._rng.standard_normal()
        self._price *= (1.0 + ret)
        return self._price

    def _step_regime(self) -> None:
        """Transition HMM regime according to transition matrix."""
        probs = self._transition_matrix[self._regime]
        self._regime = int(self._rng.choice(_N_REGIMES, p=probs))

    def _get_obs(self) -> np.ndarray:
        """Return log-returns from price history as observation.

        Price history holds between _PRICE_WINDOW and _PRICE_WINDOW+1 prices.
        np.diff gives _PRICE_WINDOW-1 or _PRICE_WINDOW elements; we always
        return exactly _PRICE_WINDOW float32 values.
        """
        prices = np.array(self._price_history, dtype=np.float32)
        log_returns = np.diff(np.log(prices + 1e-8))
        # Pad right with zeros when we have fewer than _PRICE_WINDOW diff values
        # (only occurs on reset when history has exactly _PRICE_WINDOW elements)
        if len(log_returns) < _PRICE_WINDOW:
            log_returns = np.pad(log_returns, (0, _PRICE_WINDOW - len(log_returns)))
        return log_returns[:_PRICE_WINDOW].astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        # super().reset() seeds self._np_random (Gymnasium contract) when seed is given
        super().reset(seed=seed)
        if seed is not None:
            # Keep our own RNG aligned with the provided seed
            self._rng = np.random.default_rng(seed)
        self._price = 100.0
        self._regime = 0
        self._position = 0
        self._price_history = [100.0] * _PRICE_WINDOW
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._step_regime()
        new_price = self._step_price()
        self._price_history.append(new_price)
        if len(self._price_history) > _PRICE_WINDOW + 1:
            self._price_history.pop(0)

        # Simple P&L: reward based on position and price change
        prev_price = self._price_history[-2]
        price_change = (new_price - prev_price) / (prev_price + 1e-8)

        if action == 1:    # buy
            self._position = 1
        elif action == 2:  # sell
            self._position = -1
        else:              # hold
            pass

        reward = float(self._position * price_change)
        self._step_count += 1
        terminated = self._step_count >= self._max_steps

        return self._get_obs(), reward, terminated, False, {}


class TradingRegime(ShiftWrapper):
    """TradingRegime with HMM transition matrix shifts.

    The hidden Markov structure changes at shift epochs — regimes become
    more volatile or more persistent. The agent cannot observe the regime
    directly and must infer it from price history.

    Args:
        shift_rate: Shifts per 1000 steps.
        seed: RNG seed.
        fast_mode: No effect (pure NumPy, already lightweight).
    """

    _TRANSITION_MATRICES = [
        # Default: moderate persistence
        np.array([[0.95, 0.03, 0.02],
                  [0.03, 0.95, 0.02],
                  [0.02, 0.02, 0.96]]),
        # High volatility: rapid regime switching
        np.array([[0.70, 0.20, 0.10],
                  [0.20, 0.70, 0.10],
                  [0.10, 0.10, 0.80]]),
        # Trending: mostly stays up or down
        np.array([[0.98, 0.01, 0.01],
                  [0.01, 0.98, 0.01],
                  [0.01, 0.01, 0.98]]),
    ]

    def __init__(
        self,
        shift_rate: float = 5.0,
        seed: int = 0,
        fast_mode: bool = False,
    ) -> None:
        base_env = _TradingEnv(seed=seed)
        super().__init__(base_env, shift_rate=shift_rate,
                         shift_type="abrupt", seed=seed)
        self._matrix_idx: int = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset with optional seed.

        When seed is provided, re-seeds the ShiftWrapper RNG deterministically
        so that gymnasium's check_env step-determinism check passes: two resets
        with the same seed produce identical shift schedules and thus identical
        oracle_tau values in info. The XOR offset avoids aliasing with the base
        env seed used by _TradingEnv.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed ^ 0xDEADBEEF)
        self._matrix_idx = 0
        # Restore the base HMM transition matrix to match _matrix_idx=0
        self.env.unwrapped._transition_matrix = self._TRANSITION_MATRICES[0].copy()
        return super().reset(seed=seed, options=options)

    def _apply_shift(self, progress: float = 1.0) -> None:
        """Cycle to the next HMM transition matrix."""
        self._matrix_idx = (self._matrix_idx + 1) % len(self._TRANSITION_MATRICES)
        self.env.unwrapped._transition_matrix = self._TRANSITION_MATRICES[self._matrix_idx].copy()

    def _is_interventionist(self, action: Any) -> bool:
        """No adversarial trigger — purely observational."""
        return False
