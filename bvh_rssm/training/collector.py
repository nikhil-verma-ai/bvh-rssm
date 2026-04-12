"""
Environment rollout collector for BVH-RSSM.

Collects transitions into a ReplayBuffer using either a random policy (Phase 0)
or the learned actor (Phase 3+). Captures per-step RNG state *before*
rssm.observe() so counterfactual replay (Level 3 causal attribution) can
reproduce identical z_t noise under an alternative action sequence.

Design invariants:
  - All model forward passes run under torch.no_grad() with eval() mode.
  - Episode resets clear RSSM state and prev_action to zeros.
  - oracle_tau is always non-negative (clamped via max(0, ...) on env side).
  - is_interventionist is always False for random_policy=True.
  - RNG state snapshot is taken BEFORE rssm.observe() so replay is exact.
  - Action storage: Box → float32 array of env action_space.shape;
    Discrete → np.array([int_action], dtype=np.float32) so buffer has
    uniform action_dim=1 regardless of n_actions.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState
from bvh_rssm.networks.rssm import State
from bvh_rssm.utils.rng import save_rng_state


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def _get_action_dim(action_space: gym.Space) -> int:
    """Return the buffer action dimension for an action space.

    For Box spaces, use the flat product of shape.
    For Discrete spaces, always return 1 (action is stored as a single int
    cast to float32 for buffer shape consistency).

    Args:
        action_space: A gymnasium action space.

    Returns:
        Integer buffer action dimension.
    """
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        return 1
    else:
        raise TypeError(f"Unsupported action space type: {type(action_space)}")


def _make_env(env_name: str, fast_mode: bool = False) -> gym.Env:
    """Construct a BVH-RSSM environment by name.

    Tries the FNSB registry in order. Falls back to gymnasium.make() for
    standard Gymnasium IDs. All FNSB environments accept fast_mode kwarg.

    Args:
        env_name: One of the FNSB environment class names (e.g. "ShiftPendulum",
                  "TradingRegime") or a standard Gymnasium ID.
        fast_mode: Passed to FNSB constructors; no-op for Gymnasium IDs.

    Returns:
        An initialised gymnasium.Env instance.

    Raises:
        ValueError: If the env_name is not found in FNSB registry or Gymnasium.
    """
    from bvh_rssm.envs import (
        ShiftPendulum, TradingRegime, RegimeMaze,
        ShiftWalker, ShiftMaze, SensorDrift,
    )

    _REGISTRY: Dict[str, type] = {
        "ShiftPendulum": ShiftPendulum,
        "TradingRegime": TradingRegime,
        "RegimeMaze":    RegimeMaze,
        "ShiftWalker":   ShiftWalker,
        "ShiftMaze":     ShiftMaze,
        "SensorDrift":   SensorDrift,
    }

    if env_name in _REGISTRY:
        cls = _REGISTRY[env_name]
        try:
            return cls(fast_mode=fast_mode)
        except TypeError:
            # Some envs may not accept fast_mode — fall back gracefully
            return cls()

    # Gymnasium fallback (e.g. "Pendulum-v1")
    try:
        return gym.make(env_name)
    except gym.error.Error as exc:
        raise ValueError(
            f"Unknown env '{env_name}'. Not in FNSB registry and Gymnasium "
            f"refused to make it: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Collector:
    """Environment rollout collector that populates a ReplayBuffer.

    Maintains a single environment instance and an RSSM world model. On each
    call to collect_steps(), it runs the environment for n_steps transitions,
    using either random actions (Phase 0 warm-up) or the learned actor network
    (Phase 3+).

    The AdaptivePolicyRouter is invoked at each step to classify the current
    validity state. When the router classifies STALE, the step is flagged as
    interventionist — the model recognised it needed a corrective action.

    RNG state is saved *before* rssm.observe() so counterfactual replay
    can feed the same stochastic z_t sample a different action and observe
    what would have happened (Level 3 causal attribution, Plan 6).

    Args:
        env_name: FNSB env class name or Gymnasium ID.
        model: Dict of nn.Module; must contain at minimum "encoder", "rssm",
               "tau_head", "hazard_head". "actor" is optional and only used
               when random_policy=False.
        replay_buffer: ReplayBuffer to push transitions into.
        device: Torch device for all tensor operations.
        fast_mode: Passed to the env constructor.
    """

    def __init__(
        self,
        env_name: str,
        model: Dict[str, nn.Module],
        replay_buffer: Any,
        device: torch.device,
        fast_mode: bool = False,
    ) -> None:
        self.env = _make_env(env_name, fast_mode=fast_mode)
        self.model = model
        self.buf = replay_buffer
        self.device = device
        self.fast_mode = fast_mode

        # Derive dimensions from the live env
        self._obs_dim: int = int(np.prod(self.env.observation_space.shape))
        self._is_discrete: bool = isinstance(self.env.action_space, gym.spaces.Discrete)
        self._action_dim: int = _get_action_dim(self.env.action_space)

        # RSSM action_dim must match the model's expectation.
        # For Discrete envs we embed the integer into a 1-d float for the RSSM.
        self._rssm_action_dim: int = self._action_dim

        # Router is stateless — one instance reused across all steps
        self._router = AdaptivePolicyRouter()

        # Persistent episode state (reset on episode boundary)
        self._rssm_state: State | None = None
        self._prev_action: torch.Tensor | None = None
        self._episode_obs: np.ndarray | None = None
        self._episode_done: bool = True  # trigger reset on first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_episode(self) -> np.ndarray:
        """Reset the environment and RSSM state for a new episode.

        Returns:
            Initial observation as a numpy array.
        """
        obs, _ = self.env.reset()
        rssm: nn.Module = self.model["rssm"]
        self._rssm_state = rssm.initial_state(1, device=self.device)
        self._prev_action = torch.zeros(
            1, self._rssm_action_dim, device=self.device, dtype=torch.float32
        )
        self._episode_done = False
        return np.asarray(obs, dtype=np.float32)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a [1, obs_dim] tensor on device."""
        return torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

    def _action_to_buffer(self, action: Any) -> np.ndarray:
        """Convert an env action to the buffer storage format.

        Box:      float32 array of shape (action_dim,)
        Discrete: float32 array of shape (1,) containing int(action)

        Args:
            action: Raw action returned by actor or action_space.sample().

        Returns:
            numpy float32 array of shape (self._action_dim,).
        """
        if self._is_discrete:
            return np.array([int(action)], dtype=np.float32)
        else:
            return np.asarray(action, dtype=np.float32).reshape(self._action_dim)

    def _action_to_rssm_tensor(self, action_buf: np.ndarray) -> torch.Tensor:
        """Convert buffer-format action to [1, action_dim] tensor for RSSM.

        Args:
            action_buf: float32 numpy array of shape (action_dim,).

        Returns:
            Tensor of shape [1, action_dim].
        """
        return torch.from_numpy(action_buf).float().unsqueeze(0).to(self.device)

    def _select_action_random(self) -> Any:
        """Sample a uniformly random action from the environment action space."""
        return self.env.action_space.sample()

    def _select_action_actor(self, latent: torch.Tensor) -> Any:
        """Select an action from the learned actor network.

        For Discrete envs: forward returns logits → deterministic argmax.
        For Box envs: forward returns (mean, log_std) → squashed tanh mean
        (no exploration noise; deterministic policy during collection).

        Args:
            latent: Tensor of shape [1, latent_dim].

        Returns:
            Action in the environment's native format (numpy for Box, int for Discrete).
        """
        actor: nn.Module = self.model["actor"]
        out = actor(latent)

        if self._is_discrete:
            # out: logits of shape [1, n_actions]
            action_int = int(out.argmax(dim=-1).item())
            return action_int
        else:
            # out: (mean, log_std) each [1, action_dim]
            mean, _ = out
            # Squash through tanh and scale to action range if needed
            action = torch.tanh(mean)
            # Scale from [-1, 1] to actual Box bounds
            low = torch.from_numpy(self.env.action_space.low).float().to(self.device)
            high = torch.from_numpy(self.env.action_space.high).float().to(self.device)
            action = low + (action + 1.0) * 0.5 * (high - low)
            return action.squeeze(0).cpu().numpy()

    def _classify_router(
        self,
        latent: torch.Tensor,
        prev_action: torch.Tensor,
        full_horizon: int,
    ) -> tuple[RouterState, int]:
        """Compute validity estimate and classify via AdaptivePolicyRouter.

        Args:
            latent: [1, latent_dim] concatenated RSSM latent.
            prev_action: [1, action_dim] action used to enter this state.
            full_horizon: Maximum imagination horizon for HIGH classification.

        Returns:
            (router_state, horizon): RouterState enum and integer horizon.
        """
        hazard_head: nn.Module = self.model["hazard_head"]
        tau_head: nn.Module = self.model["tau_head"]

        # Survival curve: [1, n_intervals] → squeeze to [n_intervals]
        S = hazard_head.survival(latent).squeeze(0)  # [K]

        # tau_hat: decode ValidityHead logits to scalar
        logits = tau_head(latent, prev_action, stop_grad=False)  # [1, n_bins]
        tau_hat = float(tau_head.decode(logits).item())           # scalar

        router_state = self._router.classify(tau_hat, S)
        horizon = self._router.imagination_horizon(router_state, tau_hat, full_horizon)
        return router_state, horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_steps(
        self,
        n_steps: int,
        random_policy: bool = False,
        full_horizon: int = 16,
    ) -> int:
        """Collect n_steps transitions into the replay buffer.

        Each transition is pushed as:
          (obs_t, action_t, reward_t, terminated_t, oracle_tau_t,
           is_interventionist_t, rng_state_before_observe)

        The rng_state is captured immediately before rssm.observe() so that
        counterfactual replay (Plan 6) can reproduce the same stochastic z_t
        sample under an alternative action.

        Random policy mode (Phase 0):
          - Action drawn uniformly from action_space.sample().
          - is_interventionist is always False.
          - RSSM is still updated to build a valid world-model representation.

        Actor policy mode (Phase 3+):
          - Actor network is used to select actions.
          - Router classifies each step; STALE → is_interventionist=True.
          - Falls back to random if "actor" not in model (with warning).

        Args:
            n_steps: Number of environment steps to collect.
            random_policy: If True, use uniform random policy regardless of actor.
            full_horizon: Passed to AdaptivePolicyRouter.imagination_horizon().

        Returns:
            Total number of steps collected (== n_steps unless episodes end
            unexpectedly, but in practice always == n_steps since we reset on done).
        """
        # Validate that required model components are present
        required_keys = {"encoder", "rssm", "tau_head", "hazard_head"}
        missing = required_keys - set(self.model.keys())
        if missing:
            raise KeyError(f"model dict is missing required keys: {missing}")

        use_actor = not random_policy
        if use_actor and "actor" not in self.model:
            warnings.warn(
                "random_policy=False but 'actor' not in model — "
                "falling back to random policy.",
                stacklevel=2,
            )
            use_actor = False

        # Put all model components into eval mode for deterministic z (argmax)
        for m in self.model.values():
            if isinstance(m, nn.Module):
                m.eval()

        steps_collected = 0

        # If mid-episode state is available from a previous call, continue it.
        # Otherwise force a reset.
        if self._episode_done or self._rssm_state is None:
            obs = self._reset_episode()
        else:
            obs = self._episode_obs  # type: ignore[assignment]

        with torch.no_grad():
            while steps_collected < n_steps:
                obs_t = self._obs_to_tensor(obs)

                # Encode observation
                encoder: nn.Module = self.model["encoder"]
                rssm: nn.Module = self.model["rssm"]

                embed = encoder(obs_t)  # [1, embed_dim]

                # Classify router state BEFORE observe (using current latent)
                # to decide is_interventionist and (optionally) imagination depth.
                current_latent = rssm.get_latent(self._rssm_state)

                is_interventionist = False
                if use_actor:
                    router_state, _ = self._classify_router(
                        current_latent, self._prev_action, full_horizon
                    )
                    is_interventionist = (router_state == RouterState.STALE)

                # Select action
                if use_actor:
                    action_native = self._select_action_actor(current_latent)
                else:
                    action_native = self._select_action_random()

                # Convert action to buffer format
                action_buf = self._action_to_buffer(action_native)
                action_tensor = self._action_to_rssm_tensor(action_buf)

                # ── Capture RNG state BEFORE rssm.observe() ──────────────
                # This allows Plan 6 counterfactual replay to reproduce the
                # exact z_t sample that was drawn during this transition.
                rng_state = save_rng_state()

                # Posterior RSSM update: incorporates current observation
                _, next_rssm_state = rssm.observe(embed, self._prev_action, self._rssm_state)

                # Step the environment with the selected action
                next_obs, reward, terminated, truncated, info = self.env.step(action_native)
                next_obs = np.asarray(next_obs, dtype=np.float32)

                episode_done = terminated or truncated
                oracle_tau = int(info.get("oracle_tau", 0))

                # Push to replay buffer
                self.buf.push(
                    obs=obs,
                    action=action_buf,
                    reward=float(reward),
                    terminated=bool(terminated),
                    oracle_tau=oracle_tau,
                    is_interventionist=is_interventionist,
                    rng_state=rng_state,
                )

                steps_collected += 1

                # Advance episode state
                self._rssm_state = next_rssm_state
                self._prev_action = action_tensor

                if episode_done:
                    obs = self._reset_episode()
                else:
                    obs = next_obs
                    self._episode_done = False

        # Persist current obs for resuming the episode on the next call
        self._episode_obs = obs

        return steps_collected
