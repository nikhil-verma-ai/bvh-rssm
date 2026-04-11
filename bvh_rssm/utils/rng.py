"""
RNG state management for counterfactual trajectory replay.

Level 3 counterfactual attribution (Pearl's hierarchy) requires replaying
a trajectory with the same stochastic z_t samples but a different action.
This is "abduction" — inferring the latent noise from the factual trajectory
and holding it fixed under intervention.

The implementation stores PyTorch CPU RNG states at each sampling step.
On replay, we restore the stored state before drawing z_t samples, ensuring
the same noise realizations regardless of the action taken.

IMPORTANT: This operates on CPU RNG state only. If using CUDA:
  - z_t sampling must happen on CPU and be moved to GPU
  - Or: manage cuda_rng_state separately (extend RNGStateStore if needed)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List

import torch


class RNGStateStore:
    """Captures and restores PyTorch CPU RNG states for deterministic replay."""

    def __init__(self) -> None:
        self._states: List[torch.ByteTensor] = []

    def capture(self) -> torch.ByteTensor:
        """Capture current CPU RNG state and append to internal list.

        Returns:
            The captured ByteTensor state (also stored internally at len-1 index).
        """
        state = torch.get_rng_state()
        self._states.append(state)
        return state

    def restore(self, idx: int) -> None:
        """Restore CPU RNG state at the given index.

        Args:
            idx: Index into the internal states list (0-based).
        """
        torch.set_rng_state(self._states[idx])

    def get(self, idx: int) -> torch.ByteTensor:
        """Return the stored state at the given index without restoring it.

        Args:
            idx: Index into the internal states list (0-based).

        Returns:
            The ByteTensor state at that index.
        """
        return self._states[idx]

    def clear(self) -> None:
        """Remove all stored states."""
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)


@contextmanager
def restore_rng_states(
    states: List[torch.ByteTensor],
) -> Generator[List[torch.ByteTensor], None, None]:
    """Context manager that restores the caller's RNG state on exit.

    Saves the current RNG state on entry. Yields the provided states list
    for use inside the block. Restores the pre-entry RNG state on exit
    regardless of exceptions.

    Args:
        states: List of RNG state tensors (from RNGStateStore).

    Yields:
        The same states list passed in.
    """
    saved = torch.get_rng_state()
    try:
        yield states
    finally:
        torch.set_rng_state(saved)
