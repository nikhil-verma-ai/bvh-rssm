"""DreamerV3 vanilla baseline — RSSM only, fixed 16-step horizon, no BVH heads."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from bvh_rssm.training.baselines.base import BaselineAgent


class DreamerV3Vanilla(BaselineAgent):
    """DreamerV3 without validity/hazard heads. Fixed imagination horizon.

    Args:
        model: Loaded BVH model (only RSSM + actor used, tau/hazard heads ignored).
        horizon: Fixed imagination horizon in steps.
        action_dim: Action space dimension.
    """

    def __init__(self, model: Any, horizon: int = 16, action_dim: int = 6) -> None:
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim

    def initial_state(self) -> Dict[str, Any]:
        return {"rssm_state": None, "step": 0, "just_switched": False}

    def act(self, obs: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Stub: full impl runs RSSM.observe + actor.forward with fixed horizon
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action, {**state, "step": state["step"] + 1, "just_switched": False}
