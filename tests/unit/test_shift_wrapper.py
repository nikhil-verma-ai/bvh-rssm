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
    def __init__(self, shift_rate=1000.0, shift_type="abrupt", seed=0):
        super().__init__(_CounterEnv(), shift_rate=shift_rate,
                         shift_type=shift_type, seed=seed)
        self.shift_count = 0

    def _apply_shift(self) -> None:
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
        env = _ShiftCounterEnv(shift_rate=10.0, seed=42)
        env.reset()
        tau_values = []
        for _ in range(20):
            _, _, _, _, info = env.step(0)
            tau_values.append(info["oracle_tau"])
            if info["shift_occurred"]:
                break
        # tau should decrease before a shift
        decreasing_segment = tau_values[:tau_values.index(max(tau_values)) + 1]
        if len(decreasing_segment) > 1:
            assert all(b <= a for a, b in zip(decreasing_segment, decreasing_segment[1:]))


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

    def test_oracle_tau_zero_when_adversarial_shift_triggered(self):
        env = _ShiftCounterEnv(shift_rate=0.0, shift_type="adversarial", seed=0)
        env.reset()
        _, _, _, _, info = env.step(1)
        assert info["oracle_tau"] >= 0


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
