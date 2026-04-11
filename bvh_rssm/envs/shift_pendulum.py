"""
ShiftPendulum — Env 1 of FNSB.

Base: gymnasium Pendulum-v1.
Shift: gravity constant cycles between three values (g_low, g_mid, g_high).
Adversarial trigger: None (gravity shifts are purely schedule-based).
Oracle τ*: next_shift_step - current_step.

Purpose: simplest FNSB baseline. Validates that the τ-head learns anything at all.
The agent cannot observe gravity directly — only angle, angular velocity, and torque.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from gymnasium.utils import RecordConstructorArgs

from bvh_rssm.envs.wrappers import ShiftWrapper

# Three gravity values the environment cycles through
_GRAVITY_VALUES = (5.0, 9.81, 15.0)


class ShiftPendulum(RecordConstructorArgs, ShiftWrapper):
    """Pendulum-v1 with periodically shifting gravity.

    The agent observes [cos(θ), sin(θ), dθ/dt] — gravity is invisible.
    The oracle provides τ* = steps until next gravity shift.

    Args:
        shift_rate: Shifts per 1000 steps. Default 5.0 (slow tier).
        shift_type: "abrupt" or "gradual". No adversarial trigger.
        seed: RNG seed for shift schedule.
        fast_mode: No effect (Pendulum is already lightweight).
    """

    def __init__(
        self,
        shift_rate: float = 5.0,
        shift_type: str = "abrupt",
        seed: int = 0,
        fast_mode: bool = False,
        render_mode: Optional[str] = None,
        env: Optional[gym.Env] = None,
    ) -> None:
        # RecordConstructorArgs must be called before any other __init__ to
        # correctly capture constructor kwargs for gymnasium's env spec system.
        # 'env' is intentionally excluded: it is the base env passed in by
        # check_env's spec.make() and must not be deepcopied into saved_kwargs.
        RecordConstructorArgs.__init__(
            self,
            shift_rate=shift_rate,
            shift_type=shift_type,
            seed=seed,
            fast_mode=fast_mode,
            render_mode=render_mode,
        )
        # When reconstructed by check_env's spec.make(), env is passed in.
        # Otherwise, create the base Pendulum-v1 env internally.
        base_env = env if env is not None else gym.make("Pendulum-v1", render_mode=render_mode)
        ShiftWrapper.__init__(self, base_env, shift_rate=shift_rate,
                              shift_type=shift_type, seed=seed)
        self._gravity_idx: int = 0
        self._fast_mode = fast_mode

    def _apply_shift(self, progress: float = 1.0) -> None:
        """Cycle to the next gravity value.

        For abrupt shifts (progress=1.0), snaps immediately.
        For gradual shifts, interpolates between current and next gravity.
        """
        next_idx = (self._gravity_idx + 1) % len(_GRAVITY_VALUES)
        if progress >= 1.0:
            # Abrupt: snap to next gravity
            self._gravity_idx = next_idx
            new_gravity = _GRAVITY_VALUES[self._gravity_idx]
        else:
            # Gradual: interpolate between current and next
            current_g = _GRAVITY_VALUES[self._gravity_idx]
            next_g = _GRAVITY_VALUES[next_idx]
            new_gravity = current_g + progress * (next_g - current_g)
            # On final step of gradual window, commit the new index
            if progress >= (1.0 - 1e-6):
                self._gravity_idx = next_idx

        # Pendulum-v1 stores gravity as 'g' on the unwrapped env
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, 'g'):
            unwrapped.g = float(new_gravity)
        elif hasattr(unwrapped, 'gravity'):
            unwrapped.gravity = float(new_gravity)

    def _is_interventionist(self, action: Any) -> bool:
        """ShiftPendulum has no adversarial trigger."""
        return False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset with optional seed.

        When seed is provided, re-seeds the shift RNG deterministically so that
        gymnasium's check_env determinism check passes: two resets with the same
        seed produce identical shift schedules and thus identical oracle_tau values.
        The derived shift seed is offset from the env seed to avoid correlation.
        """
        if seed is not None:
            # Derive a deterministic but distinct seed for the shift schedule.
            # XOR with a large prime ensures no aliasing with the env seed.
            self._rng = np.random.default_rng(seed ^ 0xDEADBEEF)
        self._gravity_idx = 0
        return super().reset(seed=seed, options=options)

    @property
    def current_gravity(self) -> float:
        """Current gravity value (for testing/logging)."""
        return float(_GRAVITY_VALUES[self._gravity_idx])
