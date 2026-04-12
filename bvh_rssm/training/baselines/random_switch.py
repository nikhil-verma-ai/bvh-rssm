"""Random switcher baseline — switches at Bernoulli(switch_rate) per step."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from bvh_rssm.training.baselines.base import BaselineAgent


class RandomSwitch(BaselineAgent):
    """Switches with probability `switch_rate` each step.

    Args:
        switch_rate: Probability of switching per step.
        action_dim: Action space dimension.
        seed: RNG seed for reproducibility.

    Note:
        The internal RNG (``_rng``) is instance state, not serialized into the
        ``state`` dict. Saving and restoring only the ``state`` dict will NOT
        reproduce the same random sequence. For checkpoint-resumable eval, capture
        the agent instance directly or re-seed with the same seed.
    """

    def __init__(self, switch_rate: float, action_dim: int, seed: int = 0) -> None:
        self.switch_rate = switch_rate
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def initial_state(self) -> Dict[str, Any]:
        return {"step": 0, "just_switched": False}

    def act(self, obs: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        just_switched = bool(self._rng.random() < self.switch_rate)
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action, {"step": state["step"] + 1, "just_switched": just_switched}
