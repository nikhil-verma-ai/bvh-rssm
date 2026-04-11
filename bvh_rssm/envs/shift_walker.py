"""
ShiftWalker — Env 2 of FNSB.

Base: MuJoCo Walker2d-v4.
Shift: ground friction coefficient cycles between low/medium/high.
Adversarial trigger: agent speed exceeds threshold (Source B coupling).
"""
from __future__ import annotations
from typing import Any
import gymnasium as gym
import numpy as np
from bvh_rssm.envs.wrappers import ShiftWrapper

_FRICTION_VALUES = (0.3, 0.8, 1.5)  # low, medium, high
_SPEED_THRESHOLD = 2.5  # m/s — above this, adversarial shift triggers


class ShiftWalker(ShiftWrapper):
    def __init__(self, shift_rate=5.0, shift_type="abrupt", seed=0, fast_mode=False):
        base_env = gym.make("Walker2d-v4")
        super().__init__(base_env, shift_rate=shift_rate, shift_type=shift_type, seed=seed)
        self._friction_idx = 0

    def _apply_shift(self, progress: float = 1.0) -> None:
        self._friction_idx = (self._friction_idx + 1) % len(_FRICTION_VALUES)
        friction = _FRICTION_VALUES[self._friction_idx]
        model = self.env.unwrapped.model
        # Set floor friction (geom index 0 is the floor in Walker2d)
        model.geom_friction[0, 0] = friction

    def _is_interventionist(self, action: Any) -> bool:
        try:
            vel = float(self.env.unwrapped.data.qvel[0])
            return vel > _SPEED_THRESHOLD
        except Exception:
            return False

    @property
    def current_friction(self) -> float:
        return float(_FRICTION_VALUES[self._friction_idx])
