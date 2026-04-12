"""Abstract baseline agent interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaselineAgent(ABC):
    """All baselines implement this interface. Called identically by eval loop."""

    @abstractmethod
    def act(self, obs: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action. Returns (action, new_state)."""

    @abstractmethod
    def initial_state(self) -> Dict[str, Any]:
        """Return initial agent state."""
