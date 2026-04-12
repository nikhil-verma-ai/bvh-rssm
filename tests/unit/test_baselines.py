import numpy as np
import pytest
from bvh_rssm.training.baselines.base import BaselineAgent
from bvh_rssm.training.baselines.fixed_interval_switch import FixedIntervalSwitch
from bvh_rssm.training.baselines.random_switch import RandomSwitch


class TestFixedIntervalSwitch:
    def test_switches_at_interval(self):
        agent = FixedIntervalSwitch(switch_interval=10, action_dim=3)
        obs = np.zeros(8)
        state = agent.initial_state()
        switch_steps = []
        for t in range(50):
            action, state = agent.act(obs, state)
            if state.get("just_switched"):
                switch_steps.append(t)
        assert len(switch_steps) >= 4

    def test_act_returns_action_and_state(self):
        agent = FixedIntervalSwitch(switch_interval=5, action_dim=3)
        obs = np.zeros(8)
        state = agent.initial_state()
        action, new_state = agent.act(obs, state)
        assert action.shape == (3,)
        assert isinstance(new_state, dict)


class TestRandomSwitch:
    def test_act_returns_action(self):
        agent = RandomSwitch(switch_rate=0.1, action_dim=3, seed=0)
        obs = np.zeros(8)
        state = agent.initial_state()
        action, _ = agent.act(obs, state)
        assert action.shape == (3,)

    def test_switches_at_rate(self):
        agent = RandomSwitch(switch_rate=0.5, action_dim=3, seed=42)
        obs = np.zeros(8)
        state = agent.initial_state()
        switches = 0
        for _ in range(200):
            _, state = agent.act(obs, state)
            if state.get("just_switched"):
                switches += 1
        assert 50 < switches < 150
