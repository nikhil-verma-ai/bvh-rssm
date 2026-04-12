import torch
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from bvh_rssm.training.experiment import set_seed, Checkpointer


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_set_seed_numpy():
    set_seed(7)
    a = np.random.randn(5)
    set_seed(7)
    b = np.random.randn(5)
    assert np.allclose(a, b)


def test_checkpointer_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(run_dir=tmpdir)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters())
        ckpt.save(model, optimizer, phase=1, step=100)
        state = ckpt.load(phase=1, step=100)
        assert "model" in state
        assert "optimizer" in state
        assert state["phase"] == 1
        assert state["step"] == 100


def test_checkpointer_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(run_dir=tmpdir)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters())
        ckpt.save(model, optimizer, phase=1, step=500)
        ckpt.save(model, optimizer, phase=1, step=1000)
        state = ckpt.load_latest(phase=1)
        assert state["step"] == 1000


def test_checkpointer_load_latest_nonexistent_phase():
    """load_latest returns None when phase directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(run_dir=tmpdir)
        result = ckpt.load_latest(phase=99)
        assert result is None


def test_checkpointer_load_does_not_create_phantom_dir():
    """load() must not create directories as a side effect."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(run_dir=tmpdir)
        try:
            ckpt.load(phase=1, step=0)
        except FileNotFoundError:
            pass  # expected — file doesn't exist
        # The phase directory must NOT have been created
        phase_dir = (ckpt.run_dir / "phase1")
        assert not phase_dir.exists(), "load() must not create phantom directories"
