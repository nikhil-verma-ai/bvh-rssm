"""Fixed-interval switcher baseline — switches every K steps."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from bvh_rssm.training.baselines.base import BaselineAgent


class FixedIntervalSwitch(BaselineAgent):
    """Switches world model belief every `switch_interval` steps.

    Args:
        switch_interval: Steps between switches.
        action_dim: Action space dimension (outputs zero action).
    """

    def __init__(self, switch_interval: int, action_dim: int) -> None:
        self.switch_interval = switch_interval
        self.action_dim = action_dim

    def initial_state(self) -> Dict[str, Any]:
        return {"step": 0, "just_switched": False}

    def act(self, obs: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        step = state["step"] + 1
        just_switched = (step % self.switch_interval == 0)
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action, {"step": step, "just_switched": just_switched}
