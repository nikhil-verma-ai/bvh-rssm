"""
ShiftWrapper — base class for FNSB (Fast Non-Stationary Benchmark) environments.

All non-stationarity logic lives here. Individual environments subclass and
implement only _apply_shift() and _is_interventionist().

Design invariants:
  - oracle_tau lives ONLY in info dict, never in observation_space
  - All three info keys always present: oracle_tau, is_interventionist, shift_occurred
  - Shift scheduling uses Poisson process (exponential inter-arrival times)
  - Abrupt: snap at boundary. Gradual: linear interp. Adversarial: action-triggered.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ShiftWrapper(gym.Wrapper):
    """Adds non-stationary dynamics to any Gymnasium environment.

    Subclasses must implement:
        _apply_shift(): Apply the parameter change to the underlying env.
        _is_interventionist(action): Return True if this action triggers a shift.

    The shift schedule uses a Poisson process: inter-shift intervals are drawn
    from an exponential distribution with mean = 1000 / shift_rate steps.
    """

    def __init__(
        self,
        env: gym.Env,
        shift_rate: float = 5.0,
        shift_type: str = "abrupt",
        gradual_window: int = 10,
        seed: int = 0,
    ) -> None:
        """
        Args:
            env: The base Gymnasium environment to wrap.
            shift_rate: Number of dynamics shifts per 1000 environment steps.
                        0.0 = never shift (useful for adversarial-only mode).
            shift_type: One of "abrupt", "gradual", "adversarial".
                        abrupt: parameter snaps at shift boundary.
                        gradual: parameter linearly interpolates over gradual_window.
                        adversarial: shift triggered by _is_interventionist() returning True.
            gradual_window: Steps over which to interpolate for gradual shifts.
            seed: RNG seed for shift schedule reproducibility.
        """
        super().__init__(env)
        self.shift_rate = shift_rate
        self.shift_type = shift_type
        self.gradual_window = gradual_window
        self._rng = np.random.default_rng(seed)
        self._step_counter: int = 0
        self._next_shift_step: int = self._sample_next_shift()
        self._shift_occurred: bool = False

    def _sample_next_shift(self) -> int:
        """Sample the step index of the next scheduled shift.

        Uses exponential distribution (Poisson process). When shift_rate=0,
        returns a very large value (effectively never).
        """
        if self.shift_rate <= 0.0:
            return int(1e9)
        mean_interval = 1000.0 / self.shift_rate
        return self._step_counter + max(1, int(self._rng.exponential(mean_interval)))

    @abc.abstractmethod
    def _apply_shift(self) -> None:
        """Apply a dynamics parameter shift to the underlying environment."""

    @abc.abstractmethod
    def _is_interventionist(self, action: Any) -> bool:
        """Return True if this action triggers or couples to a shift."""

    def oracle_tau(self) -> int:
        """Ground-truth steps until next shift from the current step."""
        return max(0, self._next_shift_step - self._step_counter)

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._shift_occurred = False
        is_interventionist = False

        # Adversarial: check if action triggers a shift BEFORE env step
        if self.shift_type == "adversarial" and self._is_interventionist(action):
            self._apply_shift()
            self._next_shift_step = self._sample_next_shift()
            self._shift_occurred = True
            is_interventionist = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_counter += 1

        # Schedule-based shift (abrupt or gradual)
        if not self._shift_occurred and self._step_counter >= self._next_shift_step:
            self._apply_shift()
            self._next_shift_step = self._sample_next_shift()
            self._shift_occurred = True

        info["oracle_tau"] = self.oracle_tau()
        info["is_interventionist"] = is_interventionist
        info["shift_occurred"] = self._shift_occurred

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_counter = 0
        self._next_shift_step = self._sample_next_shift()
        self._shift_occurred = False
        info["oracle_tau"] = self.oracle_tau()
        info["is_interventionist"] = False
        info["shift_occurred"] = False
        return obs, info
