"""
SensorDrift — Env 6 of FNSB.

Base: MuJoCo HalfCheetah-v4.
Shift: observation noise increases monotonically until a reset event.
drift_rate controls how fast noise_std grows per step.
Oracle τ*: steps until noise_std reaches _NOISE_RESET_THRESHOLD.
"""
from __future__ import annotations
from typing import Any, Optional, Tuple
import gymnasium as gym
import numpy as np
from bvh_rssm.envs.wrappers import ShiftWrapper

_NOISE_RESET_THRESHOLD = 0.5  # std dev at which noise resets


class SensorDrift(ShiftWrapper):
    def __init__(self, drift_rate=0.001, seed=0, fast_mode=False):
        base_env = gym.make("HalfCheetah-v4")
        # shift_rate=1.0 so ShiftWrapper schedules periodic reset events
        super().__init__(base_env, shift_rate=1.0, shift_type="abrupt", seed=seed)
        self.drift_rate = drift_rate
        self._noise_std = 0.0

    def _apply_shift(self, progress: float = 1.0) -> None:
        """The 'shift' event resets the noise level."""
        self._noise_std = 0.0

    def _is_interventionist(self, action: Any) -> bool:
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Apply drift AFTER ShiftWrapper's step (which may have reset noise via _apply_shift)
        self._noise_std += self.drift_rate
        noise = np.random.normal(0, self._noise_std, size=obs.shape).astype(obs.dtype)
        obs = obs + noise
        info["oracle_tau"] = self._oracle_tau_from_drift()
        return obs, reward, terminated, truncated, info

    def _oracle_tau_from_drift(self) -> int:
        if self.drift_rate <= 0:
            return int(1e9)
        remaining = (_NOISE_RESET_THRESHOLD - self._noise_std) / self.drift_rate
        return max(0, int(remaining))

    def reset(self, *, seed=None, options=None):
        self._noise_std = 0.0
        return super().reset(seed=seed, options=options)

    @property
    def current_noise_std(self) -> float:
        return self._noise_std
