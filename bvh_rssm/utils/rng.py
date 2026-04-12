"""
RNG state save and restore utilities.

Used by the causal attribution module (Plan 6) to implement Level 3
counterfactual replay: replay a trajectory with the same stochastic z_t
samples but a different action sequence.

Critical invariant: restore_rng_states must produce byte-for-byte identical
torch.randn / np.random.randn calls after restoration.

Also provides RNGStateStore (Plan 1) for per-step CPU RNG capture during
trajectory rollout — used to hold z_t noise fixed under action intervention.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Plan 1 API: RNGStateStore — per-step CPU RNG capture for trajectory replay
# ---------------------------------------------------------------------------

class RNGStateStore:
    """Captures and restores PyTorch CPU RNG states for deterministic replay."""

    def __init__(self) -> None:
        self._states: List[Tensor] = []

    def capture(self) -> Tensor:
        """Capture current CPU RNG state and append to internal list.

        Returns:
            The captured state tensor (also stored internally at len-1 index).
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

    def get(self, idx: int) -> Tensor:
        """Return the stored state at the given index without restoring it.

        Args:
            idx: Index into the internal states list (0-based).

        Returns:
            The RNG state tensor at that index.
        """
        return self._states[idx]

    def clear(self) -> None:
        """Remove all stored states."""
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)


# ---------------------------------------------------------------------------
# Plan 4 API: dict-based global state snapshot (torch CPU + numpy + CUDA)
# ---------------------------------------------------------------------------

def save_rng_state() -> Dict[str, Any]:
    """Snapshot current global RNG state (CPU torch + numpy + CUDA if present).

    Returns:
        Dict with keys:
          - 'torch_cpu': ByteTensor from torch.get_rng_state()
          - 'numpy': tuple from np.random.get_state()
          - 'torch_cuda': ByteTensor from torch.cuda.get_rng_state() (only if CUDA
            is available; key absent otherwise)
    """
    state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_states(state: Dict[str, Any]) -> None:
    """Restore global RNG state from a previously saved snapshot.

    Restores torch CPU, numpy, and optionally CUDA RNG states so that
    subsequent calls to torch.randn / np.random.randn produce the same
    samples as they did immediately after save_rng_state() was called.

    Args:
        state: Dict returned by save_rng_state().
    """
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
        and state["torch_cuda"] is not None
    ):
        torch.cuda.set_rng_state(state["torch_cuda"])


@contextlib.contextmanager
def rng_snapshot():
    """Context manager: save RNG state on entry, restore on exit.

    Use to run a stochastic sub-computation and leave the global RNG
    unchanged afterward:

        with rng_snapshot():
            cf_latent = rssm.imagine(alt_action, state)  # uses stochastic z
        # RNG restored — subsequent calls are deterministic w.r.t. outer seed

    Yields:
        The saved state dict (from save_rng_state) for optional inspection.
    """
    state = save_rng_state()
    try:
        yield state
    finally:
        restore_rng_states(state)
