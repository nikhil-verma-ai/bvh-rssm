"""
ShiftMaze — Env 3 of FNSB.

Base: MiniGrid FourRooms-v0.
Shift: layout index cycles (affects room connectivity on next reset).
Adversarial trigger: agent uses the "toggle" action (action=5, opens doors).
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from bvh_rssm.envs.wrappers import ShiftWrapper


class ShiftMaze(ShiftWrapper):
    def __init__(self, shift_rate=5.0, shift_type="abrupt", seed=0, fast_mode=False):
        import gymnasium as gym
        import minigrid  # noqa: F401
        base_env = gym.make("MiniGrid-FourRooms-v0")
        super().__init__(base_env, shift_rate=shift_rate, shift_type=shift_type, seed=seed)
        self._layout_idx = 0

    def _apply_shift(self, progress: float = 1.0) -> None:
        self._layout_idx = (self._layout_idx + 1) % 2

    def _is_interventionist(self, action: Any) -> bool:
        # Action 5 in MiniGrid is "toggle" (open door)
        return int(action) == 5
