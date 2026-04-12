"""
Replay buffer with BVH extensions.

Stores per-transition (beyond standard DreamerV3):
  oracle_tau: int — steps to next shift (from info["oracle_tau"])
  is_interventionist: bool — from info["is_interventionist"]
  rng_state: dict — torch_cpu + numpy RNG state (~600 bytes/transition)

RNG state enables Level 3 counterfactual replay (Plan 6): same z_t samples,
different action. Stored unconditionally — ~600MB overhead at 1M transitions.

Samples contiguous sequences of length seq_len for RSSM training (GRU needs
temporal context to build up latent state).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer for BVH-RSSM training.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        seq_len: Default sequence length for sampling (matches GRU unroll).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seq_len: int = 16,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._oracle_tau = np.zeros(capacity, dtype=np.int64)
        self._is_interventionist = np.zeros(capacity, dtype=bool)
        self._rng_states: List[Optional[Dict[str, Any]]] = [None] * capacity

        self._ptr = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        oracle_tau: int,
        is_interventionist: bool,
        rng_state: Dict[str, Any],
    ) -> None:
        """Store one transition."""
        idx = self._ptr
        self._obs[idx] = obs
        self._action[idx] = action
        self._reward[idx] = reward
        self._terminated[idx] = terminated
        self._oracle_tau[idx] = oracle_tau
        self._is_interventionist[idx] = is_interventionist
        self._rng_states[idx] = rng_state

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: Optional[int] = None) -> Dict[str, Any]:
        """Sample a batch of contiguous sequences.

        Args:
            batch_size: Number of sequences.
            seq_len: Sequence length. Defaults to self.seq_len.

        Returns:
            Dict with keys: obs [B,T,obs_dim], action [B,T,action_dim],
            reward [B,T], terminated [B,T], oracle_tau [B,T],
            is_interventionist [B,T], rng_states list[list[dict]].
        """
        T = seq_len or self.seq_len
        if self._size < T:
            raise ValueError(f"Buffer has {self._size} transitions, need at least {T}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Oldest valid slot: _ptr when buffer is full (just-overwritten slot is oldest),
        # else 0 (buffer hasn't wrapped yet).
        oldest = self._ptr if self._size == self.capacity else 0

        # Sample logical offsets within the valid window [0, _size - T]
        logical_starts = np.random.randint(0, self._size - T + 1, size=batch_size)

        # Convert to physical indices with circular wrap
        indices = np.array(
            [((oldest + s + t) % self.capacity) for s in logical_starts for t in range(T)]
        ).reshape(batch_size, T)

        rng_states: List[List[Optional[Dict[str, Any]]]] = [
            [self._rng_states[indices[b, t]] for t in range(T)]
            for b in range(batch_size)
        ]
        batch: Dict[str, Any] = {
            "obs": torch.from_numpy(self._obs[indices]),            # pyright: ignore[reportUnknownMemberType]
            "action": torch.from_numpy(self._action[indices]),      # pyright: ignore[reportUnknownMemberType]
            "reward": torch.from_numpy(self._reward[indices]),      # pyright: ignore[reportUnknownMemberType]
            "terminated": torch.from_numpy(self._terminated[indices]),  # pyright: ignore[reportUnknownMemberType]
            "oracle_tau": torch.from_numpy(self._oracle_tau[indices]),  # pyright: ignore[reportUnknownMemberType]
            "is_interventionist": torch.from_numpy(self._is_interventionist[indices]),  # pyright: ignore[reportUnknownMemberType]
            "rng_states": rng_states,
        }
        return batch

    def __len__(self) -> int:
        return self._size
