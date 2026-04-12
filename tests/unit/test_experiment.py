import torch
import numpy as np
import pytest
import tempfile
import os
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
