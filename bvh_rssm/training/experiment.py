"""
Experiment management: seeding, checkpointing, wandb initialization.

set_seed() is called once at launch and logged to wandb config.
Checkpointer saves phase-aware checkpoints enabling crash recovery
without restarting from phase 1.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all global RNG seeds atomically.

    Sets Python random, NumPy, PyTorch CPU, and CUDA (if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Checkpointer:
    """Phase-aware checkpoint save/load.

    Format: {run_dir}/phase{n}/step{k}.pt

    Args:
        run_dir: Root directory for this run's checkpoints.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, phase: int, step: int) -> Path:
        phase_dir = self.run_dir / f"phase{phase}"
        phase_dir.mkdir(exist_ok=True)
        return phase_dir / f"step{step}.pt"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        phase: int,
        step: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint to {run_dir}/phase{phase}/step{step}.pt."""
        path = self._path(phase, step)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "phase": phase,
            "step": step,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load(self, phase: int, step: int) -> Dict[str, Any]:
        """Load checkpoint by phase and step."""
        path = self._path(phase, step)
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_latest(self, phase: int) -> Optional[Dict[str, Any]]:
        """Load highest-step checkpoint for a given phase. Returns None if none exist."""
        phase_dir = self.run_dir / f"phase{phase}"
        if not phase_dir.exists():
            return None
        steps = []
        for p in phase_dir.glob("step*.pt"):
            try:
                steps.append(int(p.stem.replace("step", "")))
            except ValueError:
                continue
        if not steps:
            return None
        return self.load(phase, max(steps))


def init_wandb(config: Dict[str, Any], run_name: str, project: str = "bvh-rssm") -> None:
    """Initialize wandb run. No-op if wandb is not installed."""
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
    except ImportError:
        pass


def log_metrics(metrics: Dict[str, float], step: int) -> None:
    """Log metrics to wandb. No-op if wandb not active.

    Always log all loss components individually, never just total.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
