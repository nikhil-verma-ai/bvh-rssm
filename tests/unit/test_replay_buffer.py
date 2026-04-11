import torch
import numpy as np
import pytest
from bvh_rssm.training.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def setup_method(self):
        self.obs_dim = 8
        self.action_dim = 3
        self.capacity = 100
        self.buf = ReplayBuffer(
            capacity=self.capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            seq_len=16,
        )

    def _push_transition(self, oracle_tau=10, is_interventionist=False):
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        action = np.random.randn(self.action_dim).astype(np.float32)
        reward = float(np.random.randn())
        terminated = False
        rng_state = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
        self.buf.push(obs, action, reward, terminated, oracle_tau, is_interventionist, rng_state)

    def test_push_and_len(self):
        assert len(self.buf) == 0
        self._push_transition()
        assert len(self.buf) == 1

    def test_push_fills_to_capacity(self):
        for _ in range(self.capacity + 10):
            self._push_transition()
        assert len(self.buf) == self.capacity

    def test_sample_returns_correct_keys(self):
        for _ in range(50):
            self._push_transition()
        batch = self.buf.sample(batch_size=4, seq_len=8)
        assert "obs" in batch
        assert "action" in batch
        assert "reward" in batch
        assert "oracle_tau" in batch
        assert "is_interventionist" in batch
        assert "rng_states" in batch

    def test_sample_shapes(self):
        for _ in range(50):
            self._push_transition()
        B, T = 4, 8
        batch = self.buf.sample(batch_size=B, seq_len=T)
        assert batch["obs"].shape == (B, T, self.obs_dim)
        assert batch["action"].shape == (B, T, self.action_dim)
        assert batch["reward"].shape == (B, T)
        assert batch["oracle_tau"].shape == (B, T)

    def test_oracle_tau_stored_correctly(self):
        for i in range(20):
            self._push_transition(oracle_tau=i)
        batch = self.buf.sample(batch_size=2, seq_len=5)
        assert batch["oracle_tau"].dtype == torch.int64

    def test_is_interventionist_bool(self):
        for _ in range(20):
            self._push_transition(is_interventionist=True)
        batch = self.buf.sample(batch_size=2, seq_len=5)
        assert batch["is_interventionist"].dtype == torch.bool

    def test_rng_states_accessible(self):
        for _ in range(20):
            self._push_transition()
        batch = self.buf.sample(batch_size=2, seq_len=5)
        # rng_states is a list of lists of dicts (B x T)
        assert len(batch["rng_states"]) == 2
        assert len(batch["rng_states"][0]) == 5

    def test_sample_temporal_contiguity_after_wrap(self):
        """After wrap, samples must contain temporally contiguous transitions."""
        # Push capacity + extra with unique oracle_tau = i for each transition
        n = self.capacity + 20
        for i in range(n):
            obs = np.random.randn(self.obs_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            rng_state = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
            self.buf.push(obs, action, 0.0, False, oracle_tau=i, is_interventionist=False, rng_state=rng_state)

        assert len(self.buf) == self.capacity  # confirm wrap happened

        # Sample many batches and verify all sequences are temporally contiguous
        for _ in range(20):
            batch = self.buf.sample(batch_size=4, seq_len=8)
            taus = batch["oracle_tau"]  # [4, 8] int64
            # Each row must be strictly increasing (unique oracle_tau = i)
            diffs = taus[:, 1:] - taus[:, :-1]
            assert (diffs == 1).all(), \
                f"Non-contiguous sequence detected. Diffs: {diffs}"
