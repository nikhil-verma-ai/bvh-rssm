import torch
import numpy as np
import pytest
from bvh_rssm.utils.rng import save_rng_state, restore_rng_states


def test_save_returns_dict():
    state = save_rng_state()
    assert "torch_cpu" in state
    assert "numpy" in state


def test_restore_reproduces_torch_samples():
    state = save_rng_state()
    a = torch.randn(5)
    restore_rng_states(state)
    b = torch.randn(5)
    assert torch.allclose(a, b), "Restoring state must reproduce same samples"


def test_restore_reproduces_numpy_samples():
    state = save_rng_state()
    a = np.random.randn(5)
    restore_rng_states(state)
    b = np.random.randn(5)
    assert np.allclose(a, b), "Restoring numpy state must reproduce same samples"


def test_context_manager_restores_on_exit():
    from bvh_rssm.utils.rng import rng_snapshot
    state_before = save_rng_state()
    torch.randn(100)  # advance RNG
    with rng_snapshot():
        inside = torch.randn(5)
    outside = torch.randn(5)
    assert inside.shape == (5,)
    assert outside.shape == (5,)


def test_save_includes_cuda_key_if_available():
    state = save_rng_state()
    if torch.cuda.is_available():
        assert "torch_cuda" in state
    else:
        assert "torch_cuda" not in state or state["torch_cuda"] is None
