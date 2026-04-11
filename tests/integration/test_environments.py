"""Integration tests: check_env() compliance and oracle_tau contract for all FNSB envs."""
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env


def _run_n_steps(env, n=50, seed=0):
    """Run an env for n steps and collect all info dicts."""
    env.reset(seed=seed)
    infos = []
    for _ in range(n):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            env.reset()
    return infos


def _assert_info_contract(infos):
    """Assert oracle_tau, is_interventionist, shift_occurred always present."""
    for i, info in enumerate(infos):
        assert "oracle_tau" in info, f"Step {i}: oracle_tau missing"
        assert "is_interventionist" in info, f"Step {i}: is_interventionist missing"
        assert "shift_occurred" in info, f"Step {i}: shift_occurred missing"
        assert isinstance(info["oracle_tau"], int), f"Step {i}: oracle_tau not int"
        assert info["oracle_tau"] >= 0, f"Step {i}: oracle_tau negative"
        assert isinstance(info["is_interventionist"], bool)
        assert isinstance(info["shift_occurred"], bool)


class TestShiftPendulum:
    def setup_method(self):
        from bvh_rssm.envs.shift_pendulum import ShiftPendulum
        self.env = ShiftPendulum(shift_rate=50.0, seed=0)

    def teardown_method(self):
        self.env.close()

    def test_check_env(self):
        check_env(self.env, warn=True)

    def test_info_contract(self):
        infos = _run_n_steps(self.env)
        _assert_info_contract(infos)

    def test_observation_space_unchanged(self):
        from bvh_rssm.envs.shift_pendulum import ShiftPendulum
        import gymnasium as gym
        base_env = gym.make("Pendulum-v1")
        wrapped = ShiftPendulum(shift_rate=10.0)
        assert wrapped.observation_space == base_env.observation_space
        base_env.close()
        wrapped.close()

    def test_oracle_tau_not_in_observation(self):
        obs, info = self.env.reset()
        assert obs.shape == self.env.observation_space.shape
        # oracle_tau must not be part of observation
        assert obs.shape == (3,), f"Expected (3,), got {obs.shape}. oracle_tau must not be in obs."

    def test_shift_occurs_at_high_rate(self):
        infos = _run_n_steps(self.env, n=100)
        assert any(info["shift_occurred"] for info in infos)

    def test_gravity_changes_on_shift(self):
        from bvh_rssm.envs.shift_pendulum import ShiftPendulum
        env = ShiftPendulum(shift_rate=1000.0, seed=0)
        env.reset()
        gravities_seen = set()
        for _ in range(50):
            env.step(env.action_space.sample())
            # Try different attribute names used by different gymnasium versions
            unwrapped = env.env.unwrapped
            g = getattr(unwrapped, 'g', None)
            if g is None:
                g = getattr(unwrapped, 'gravity', None)
            if g is not None:
                gravities_seen.add(round(float(g), 4))
        env.close()
        # Should see at least 2 different gravity values
        assert len(gravities_seen) >= 2


class TestTradingRegime:
    def setup_method(self):
        from bvh_rssm.envs.trading_regime import TradingRegime
        self.env = TradingRegime(shift_rate=20.0, seed=0)

    def teardown_method(self):
        self.env.close()

    def test_check_env(self):
        check_env(self.env, warn=True)

    def test_info_contract(self):
        infos = _run_n_steps(self.env)
        _assert_info_contract(infos)

    def test_observation_space(self):
        obs, _ = self.env.reset()
        assert obs.shape == self.env.observation_space.shape
        assert obs.dtype == np.float32

    def test_action_space_is_discrete(self):
        import gymnasium as gym
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        assert self.env.action_space.n == 3  # buy, sell, hold

    def test_regime_shifts_occur(self):
        infos = _run_n_steps(self.env, n=200)
        assert any(info["shift_occurred"] for info in infos)

    def test_no_adversarial_trigger(self):
        infos = _run_n_steps(self.env, n=100)
        assert all(not info["is_interventionist"] for info in infos)
