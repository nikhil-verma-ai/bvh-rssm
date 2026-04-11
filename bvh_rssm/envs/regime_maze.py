"""
RegimeMaze — Env 4 of FNSB.

A custom 10×10 gridworld with 3 dynamics regimes. The agent can press a
"switch button" action that immediately triggers a regime change (Source B
pure isolation). Used for Experiment 3 causal attribution tests.

Observation: [agent_x/10, agent_y/10, current_regime/3, dist_to_goal/(2*GRID_SIZE)]
Actions: 0=up, 1=down, 2=left, 3=right, 4=SWITCH (triggers regime change)
Oracle τ*: from ShiftWrapper (next scheduled shift - t, or 0 if adversarial)
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bvh_rssm.envs.wrappers import ShiftWrapper

GRID_SIZE = 10
N_REGIMES = 3
ACTION_SWITCH = 4  # exported constant for tests

# Movement deltas per action (up, down, left, right)
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Goal position for each regime
_GOALS = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, GRID_SIZE - 1)]


class _MazeEnv(gym.Env):
    """10×10 gridworld with 3 goal-position regimes."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)  # 0-3: movement, 4: switch
        self._agent = (GRID_SIZE // 2, GRID_SIZE // 2)
        self._regime: int = 0
        self._step_count: int = 0
        self._max_steps: int = 200

    def _get_obs(self) -> np.ndarray:
        goal = _GOALS[self._regime]
        dist = abs(self._agent[0] - goal[0]) + abs(self._agent[1] - goal[1])
        return np.array([
            self._agent[0] / GRID_SIZE,
            self._agent[1] / GRID_SIZE,
            self._regime / N_REGIMES,
            dist / (2 * GRID_SIZE),
        ], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._agent = (GRID_SIZE // 2, GRID_SIZE // 2)
        self._regime = 0
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple:
        self._step_count += 1
        if action < 4:  # movement actions only; SWITCH handled by RegimeMaze
            dr, dc = _DELTAS[action]
            nr = int(np.clip(self._agent[0] + dr, 0, GRID_SIZE - 1))
            nc = int(np.clip(self._agent[1] + dc, 0, GRID_SIZE - 1))
            self._agent = (nr, nc)
        # action=4 (SWITCH) is a no-op at the inner env level

        goal = _GOALS[self._regime]
        reached = (self._agent == goal)
        reward = 1.0 if reached else -0.01
        terminated = bool(reached or self._step_count >= self._max_steps)

        return self._get_obs(), reward, terminated, False, {}


class RegimeMaze(ShiftWrapper):
    """RegimeMaze with switch-button intervention coupling.

    Pressing ACTION_SWITCH immediately triggers a regime change.
    Used to isolate Source B (intervention coupling) for causal attribution.
    """

    def __init__(
        self,
        shift_rate: float = 5.0,
        shift_type: str = "adversarial",
        seed: int = 0,
        fast_mode: bool = False,
    ) -> None:
        base_env = _MazeEnv()
        super().__init__(base_env, shift_rate=shift_rate,
                         shift_type=shift_type, seed=seed)
        self._fast_mode = fast_mode

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset environment state.

        When seed is provided, re-seeds the wrapper's shift-schedule RNG so
        that deterministic replay is possible. This satisfies gymnasium's
        seeding contract: reset(seed=X) must produce identical trajectories.
        ShiftWrapper intentionally does NOT re-seed on reset (to produce varied
        schedules across episodes), but when an explicit seed is given we must
        override that to ensure check_env passes.
        """
        if seed is not None:
            # Re-seed the shift-schedule RNG to honour gymnasium's determinism
            # contract. Without this, oracle_tau differs across two reset(seed=X)
            # calls because the exponential draw advances the RNG state.
            self._rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options=options)

    def _apply_shift(self, progress: float = 1.0) -> None:
        """Cycle to the next regime (wraps around at N_REGIMES).

        Args:
            progress: Interpolation progress in [0.0, 1.0]. RegimeMaze always
                      snaps regimes discretely, so progress is intentionally
                      ignored — there is no meaningful continuous interpolation
                      for a discrete goal position.
        """
        unwrapped = self.env.unwrapped
        unwrapped._regime = (unwrapped._regime + 1) % N_REGIMES

    def _is_interventionist(self, action: Any) -> bool:
        """Only ACTION_SWITCH triggers a regime change."""
        return int(action) == ACTION_SWITCH
