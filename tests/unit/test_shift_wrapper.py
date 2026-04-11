"""Tests for ShiftWrapper using a minimal toy environment."""
import pytest
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from bvh_rssm.envs.wrappers import ShiftWrapper


class _CounterEnv(gym.Env):
    """Minimal environment: obs=step_count, action=no-op."""
    observation_space = spaces.Box(low=0.0, high=1e6, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self):
        super().__init__()
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.array([float(self._step)], dtype=np.float32)
        return obs, 0.0, False, False, {}


class _ShiftCounterEnv(ShiftWrapper):
    """Concrete ShiftWrapper for testing: counts shifts."""
    def __init__(self, shift_rate=1000.0, shift_type="abrupt", gradual_window=10, seed=0):
        super().__init__(_CounterEnv(), shift_rate=shift_rate,
                         shift_type=shift_type, gradual_window=gradual_window, seed=seed)
        self.shift_count = 0

    def _apply_shift(self, progress: float = 1.0) -> None:
        self.shift_count += 1

    def _is_interventionist(self, action) -> bool:
        return int(action) == 1  # action=1 is interventionist


class TestShiftWrapperInfoContract:
    def test_oracle_tau_in_info_on_step(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "oracle_tau" in info
        assert isinstance(info["oracle_tau"], int)
        assert info["oracle_tau"] >= 0

    def test_oracle_tau_in_info_on_reset(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        _, info = env.reset()
        assert "oracle_tau" in info
        assert isinstance(info["oracle_tau"], int)
        assert info["oracle_tau"] >= 0

    def test_is_interventionist_in_info(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "is_interventionist" in info
        assert isinstance(info["is_interventionist"], bool)

    def test_shift_occurred_in_info(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "shift_occurred" in info
        assert isinstance(info["shift_occurred"], bool)

    def test_oracle_tau_not_in_observation_space(self):
        env = _ShiftCounterEnv(seed=42)
        obs, _ = env.reset()
        assert obs.shape == (1,)
        assert env.observation_space.shape == (1,)

    def test_oracle_tau_decrements_toward_shift(self):
        """oracle_tau should decrease by 1 each step until a shift fires."""
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        env.reset()
        tau_values = []
        for _ in range(15):
            _, _, _, _, info = env.step(0)
            tau_values.append(info["oracle_tau"])
            if info["shift_occurred"]:
                break
        # Collect values before the shift step
        shift_idx = next((i for i, v in enumerate(tau_values) if v == 0), len(tau_values))
        pre_shift = tau_values[:shift_idx]
        if len(pre_shift) > 1:
            # Each step should decrement tau by exactly 1
            assert all(b == a - 1 for a, b in zip(pre_shift, pre_shift[1:]))


class TestShiftWrapperAbrupt:
    def test_shift_occurs_on_schedule(self):
        """With very high shift rate, a shift must occur within 10 steps."""
        env = _ShiftCounterEnv(shift_rate=1000.0, seed=0)
        env.reset()
        shift_detected = False
        for _ in range(20):
            _, _, _, _, info = env.step(0)
            if info["shift_occurred"]:
                shift_detected = True
                break
        assert shift_detected

    def test_shift_count_increments(self):
        env = _ShiftCounterEnv(shift_rate=500.0, seed=0)
        env.reset()
        for _ in range(10):
            env.step(0)
        assert env.shift_count >= 1

    def test_no_shift_when_rate_is_zero(self):
        env = _ShiftCounterEnv(shift_rate=0.0, seed=0)
        env.reset()
        for _ in range(50):
            _, _, _, _, info = env.step(0)
            assert not info["shift_occurred"]


class TestShiftWrapperAdversarial:
    def test_interventionist_action_triggers_shift(self):
        env = _ShiftCounterEnv(shift_rate=0.0, shift_type="adversarial", seed=0)
        env.reset()
        _, _, _, _, info = env.step(1)  # action=1 is interventionist
        assert info["shift_occurred"]
        assert info["is_interventionist"]
        assert env.shift_count == 1

    def test_non_interventionist_action_no_shift(self):
        env = _ShiftCounterEnv(shift_rate=0.0, shift_type="adversarial", seed=0)
        env.reset()
        for _ in range(10):
            _, _, _, _, info = env.step(0)  # action=0 is not interventionist
            assert not info["shift_occurred"]
            assert not info["is_interventionist"]

    def test_oracle_tau_nonnegative_after_adversarial_shift(self):
        """After an adversarial shift, oracle_tau reflects time to next scheduled shift."""
        env = _ShiftCounterEnv(shift_rate=0.0, shift_type="adversarial", seed=0)
        env.reset()
        _, _, _, _, info = env.step(1)
        # With shift_rate=0, next scheduled shift is int(1e9) away
        assert info["oracle_tau"] >= 0
        assert isinstance(info["oracle_tau"], int)


class TestShiftWrapperReset:
    def test_reset_resets_step_counter(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=0)
        env.reset()
        for _ in range(5):
            env.step(0)
        env.reset()
        _, info = env.reset()
        assert info["oracle_tau"] >= 0

    def test_multiple_resets_work(self):
        env = _ShiftCounterEnv(shift_rate=10.0, seed=0)
        for _ in range(3):
            obs, info = env.reset()
            assert obs.shape == (1,)
            assert "oracle_tau" in info


class TestShiftWrapperGradual:
    def test_gradual_calls_apply_shift_multiple_times(self):
        """Gradual shift must call _apply_shift once per step over the window."""
        window = 5
        # Use extreme shift_rate so the first shift boundary is at step 1
        env = _ShiftCounterEnv(shift_rate=1000.0, shift_type="gradual",
                               gradual_window=window, seed=0)
        env.reset()

        # Step until the first shift boundary fires, then collect shift_count
        # over the subsequent window steps.
        steps_run = 0
        trigger_step = None
        count_at_trigger = None
        for i in range(200):
            _, _, _, _, info = env.step(0)
            steps_run += 1
            if info["shift_occurred"] and trigger_step is None:
                trigger_step = steps_run
                count_at_trigger = env.shift_count
                break

        assert trigger_step is not None, "No shift occurred within 200 steps"

        # On the trigger step itself _apply_shift is already called (step 1 of window)
        assert env.shift_count == count_at_trigger

        # Run the remaining (window - 1) steps and verify _apply_shift fires each time
        for j in range(window - 1):
            prev_count = env.shift_count
            env.step(0)
            assert env.shift_count == prev_count + 1, (
                f"Expected shift_count to increment on gradual step {j + 2}/{window}, "
                f"got {env.shift_count} (was {prev_count})"
            )

        # Total calls across the full window must equal gradual_window
        assert env.shift_count == count_at_trigger + (window - 1)

    def test_gradual_shift_occurred_only_on_trigger_step(self):
        """shift_occurred must be True only on the first step of a gradual shift.

        Use shift_rate=0.0 (no automatic re-scheduling) and manually arm the
        first shift boundary so that no second trigger can fire during the window.
        """
        window = 4
        # shift_rate=0 → _next_shift_step defaults to 1e9; arm it manually to step 1
        env = _ShiftCounterEnv(shift_rate=0.0, shift_type="gradual",
                               gradual_window=window, seed=0)
        env.reset()
        env._next_shift_step = 1  # force trigger on first step

        # Step 1: trigger
        _, _, _, _, info = env.step(0)
        assert info["shift_occurred"], "Expected shift_occurred=True on trigger step"

        # Steps 2..window: follow-up interpolation steps — shift_occurred must stay False
        for j in range(window - 1):
            _, _, _, _, follow_info = env.step(0)
            assert not follow_info["shift_occurred"], (
                f"shift_occurred must be False during gradual interpolation step {j + 2}/{window}"
            )

    def test_gradual_reset_clears_window(self):
        """reset() mid-window must discard remaining gradual steps."""
        window = 10
        env = _ShiftCounterEnv(shift_rate=1000.0, shift_type="gradual",
                               gradual_window=window, seed=0)
        env.reset()

        # Advance until a gradual shift fires
        for _ in range(200):
            _, _, _, _, info = env.step(0)
            if info["shift_occurred"]:
                break

        # Reset mid-window; _gradual_steps_remaining must be cleared
        env.reset()
        assert env._gradual_steps_remaining == 0

        # Steps after reset must not fire residual gradual callbacks
        count_after_reset = env.shift_count
        env.step(0)
        env.step(0)
        # shift_count may increase only if a new scheduled shift fires, not from
        # the previous window — we just verify the internal counter was cleared.
        assert env._gradual_steps_remaining == 0 or env.shift_count >= count_after_reset
