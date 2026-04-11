"""Unit tests for _TradingEnv and TradingRegime."""
from __future__ import annotations

import collections

import numpy as np
import pytest

from bvh_rssm.envs.trading_regime import _TradingEnv, TradingRegime, _PRICE_WINDOW


class TestTradingEnvSpaces:
    """observation_space and action_space must be instance attributes."""

    def test_observation_space_is_instance_attribute(self):
        env = _TradingEnv(seed=0)
        # Must live on the instance dict, not only on the class
        assert "observation_space" in env.__dict__, (
            "observation_space must be an instance attribute, not a class attribute"
        )

    def test_action_space_is_instance_attribute(self):
        env = _TradingEnv(seed=0)
        assert "action_space" in env.__dict__, (
            "action_space must be an instance attribute, not a class attribute"
        )

    def test_observation_space_shape(self):
        env = _TradingEnv(seed=0)
        assert env.observation_space.shape == (_PRICE_WINDOW,)

    def test_action_space_size(self):
        env = _TradingEnv(seed=0)
        assert env.action_space.n == 3


class TestRewardUsePrevPosition:
    """Critical: reward must use the position held BEFORE the action is applied."""

    def test_reward_uses_prev_position_buy_from_flat(self):
        """Starting flat (position=0), buying should yield reward = 0 * price_change.

        The new position (1) must NOT be used to compute the reward for the
        step in which buy was chosen — that would be a look-ahead exploit.
        """
        env = _TradingEnv(seed=42)
        env.reset(seed=42)

        # Confirm initial position is flat
        assert env._position == 0, "Position must be 0 after reset"

        # Take buy action (action=1); prev_position was 0
        _, reward, _, _, _ = env.step(1)

        # Reward = prev_position * price_change = 0 * anything = 0
        assert reward == pytest.approx(0.0), (
            f"Reward should be 0.0 (prev_position=0), got {reward}. "
            "This indicates a look-ahead exploit: self._position was updated "
            "before reward computation."
        )

    def test_reward_uses_prev_position_sell_from_flat(self):
        """Starting flat (position=0), selling should yield reward = 0 * price_change."""
        env = _TradingEnv(seed=7)
        env.reset(seed=7)

        assert env._position == 0
        _, reward, _, _, _ = env.step(2)  # sell

        assert reward == pytest.approx(0.0), (
            f"Reward should be 0.0 (prev_position=0), got {reward}."
        )

    def test_reward_reflects_prior_long_position(self):
        """After buying (position=1), hold should earn/lose based on price move."""
        env = _TradingEnv(seed=0)
        env.reset(seed=0)

        # Step 1: buy (prev_pos=0 → reward=0, position becomes 1)
        env.step(1)
        assert env._position == 1

        # Step 2: hold (prev_pos=1, reward = 1 * price_change ≠ 0 in general)
        # We cannot predict sign, but we CAN assert reward == 1 * measured price_change.
        history_before = list(env._price_history).copy()
        _, reward, _, _, _ = env.step(0)  # hold

        # Reconstruct price_change from history
        history_after = list(env._price_history)
        new_price = history_after[-1]
        prev_price = history_after[-2]
        expected_price_change = (new_price - prev_price) / (prev_price + 1e-8)
        expected_reward = 1.0 * expected_price_change  # prev_position was 1

        assert reward == pytest.approx(expected_reward, rel=1e-5), (
            f"Reward {reward} != expected {expected_reward} for long position + hold"
        )

    def test_position_updated_after_reward_computed(self):
        """Verify new position is in effect for the NEXT step, not the current one."""
        env = _TradingEnv(seed=1)
        env.reset(seed=1)

        # From flat: buy. Reward for this step must be 0.
        _, r0, _, _, _ = env.step(1)
        assert r0 == pytest.approx(0.0), "Buy-from-flat reward must be 0 (prev_pos=0)"
        assert env._position == 1, "Position should be 1 after buy"

        # Next step: hold. Reward should reflect position=1 from previous step.
        history = list(env._price_history)
        _, r1, _, _, _ = env.step(0)
        history_after = list(env._price_history)
        pc = (history_after[-1] - history_after[-2]) / (history_after[-2] + 1e-8)
        assert r1 == pytest.approx(1.0 * pc, rel=1e-5)


class TestPriceHistoryDeque:
    """Price history should use a deque for O(1) append/eviction."""

    def test_price_history_is_deque(self):
        env = _TradingEnv(seed=0)
        assert isinstance(env._price_history, collections.deque), (
            "Price history must be a collections.deque"
        )

    def test_price_history_maxlen(self):
        env = _TradingEnv(seed=0)
        assert env._price_history.maxlen == _PRICE_WINDOW + 1, (
            f"deque maxlen must be {_PRICE_WINDOW + 1}"
        )

    def test_price_history_never_exceeds_maxlen(self):
        env = _TradingEnv(seed=0)
        env.reset(seed=0)
        for _ in range(50):
            env.step(0)
            assert len(env._price_history) <= _PRICE_WINDOW + 1

    def test_price_history_reset_is_deque(self):
        env = _TradingEnv(seed=0)
        env.reset(seed=0)
        assert isinstance(env._price_history, collections.deque)


class TestTransitionMatrixReset:
    """_transition_matrix must be reset to default on env.reset()."""

    def test_transition_matrix_reset_to_default(self):
        env = _TradingEnv(seed=0)
        default = env._default_transition()

        # Corrupt the transition matrix
        env._transition_matrix = np.zeros((3, 3))
        env._transition_matrix[0, 0] = 1.0
        env._transition_matrix[1, 1] = 1.0
        env._transition_matrix[2, 2] = 1.0

        env.reset()

        np.testing.assert_array_almost_equal(
            env._transition_matrix,
            default,
            err_msg="_transition_matrix was not restored to default on reset()",
        )

    def test_transition_matrix_reset_is_independent_copy(self):
        """Reset should give a fresh copy, not a reference to the static default."""
        env = _TradingEnv(seed=0)
        env.reset()
        env._transition_matrix[0, 0] = 0.0  # mutate
        env.reset()
        # Should be restored
        assert env._transition_matrix[0, 0] == pytest.approx(0.95)


class TestObservationContract:
    """Observation must match declared space and dtype."""

    def test_reset_obs_shape_and_dtype(self):
        env = _TradingEnv(seed=0)
        obs, info = env.reset()
        assert obs.shape == (_PRICE_WINDOW,)
        assert obs.dtype == np.float32

    def test_step_obs_shape_and_dtype(self):
        env = _TradingEnv(seed=0)
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (_PRICE_WINDOW,)
        assert obs.dtype == np.float32

    def test_episode_terminates_at_max_steps(self):
        env = _TradingEnv(seed=0)
        env.reset()
        terminated = False
        for i in range(env._max_steps):
            _, _, terminated, _, _ = env.step(0)
        assert terminated, "Episode must terminate after _max_steps steps"
