#!/usr/bin/env python
"""
BVH-RSSM evaluation harness.

Usage:
    # Evaluate baselines (no checkpoint needed):
    python scripts/evaluate.py --fast

    # Evaluate with a trained checkpoint (BVH agent):
    python scripts/evaluate.py --checkpoint runs/seed42/phase3/step200000.pt

    # Custom seeds and environments:
    python scripts/evaluate.py --fast --seeds 4 --envs ShiftPendulum,TradingRegime

Output:
    results/eval_YYYY-MM-DD_HH-MM.json

JSON schema:
    {
      "date": "YYYY-MM-DD HH:MM",
      "fast_mode": bool,
      "checkpoint": str | null,
      "results": {
        "<agent_name>": {
          "<env_name>": {
            "mae_tau": float,
            "delta_return_vs_random": float,
            "mean_return": float,
            "seeds_run": int
          }
        }
      }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root on sys.path so imports work whether run as script or via pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bvh_rssm.envs import ShiftPendulum, TradingRegime, RegimeMaze, ShiftWalker, ShiftMaze, SensorDrift
from bvh_rssm.training.baselines.base import BaselineAgent
from bvh_rssm.training.baselines.fixed_interval_switch import FixedIntervalSwitch
from bvh_rssm.training.baselines.random_switch import RandomSwitch
from bvh_rssm.training.metrics import mae_tau, delta_return
from bvh_rssm.serving.predictor import Predictor


# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------

ALL_ENVS = ["ShiftPendulum", "TradingRegime", "RegimeMaze", "ShiftWalker", "ShiftMaze", "SensorDrift"]
FAST_ENVS = ["ShiftPendulum", "TradingRegime"]  # lightweight envs only

_ENV_CONSTRUCTORS = {
    "ShiftPendulum": lambda fast: ShiftPendulum(shift_rate=5.0, fast_mode=fast),
    "TradingRegime": lambda fast: TradingRegime(),
    "RegimeMaze":    lambda fast: RegimeMaze(),
    "ShiftWalker":   lambda fast: ShiftWalker(),
    "ShiftMaze":     lambda fast: ShiftMaze(),
    "SensorDrift":   lambda fast: SensorDrift(),
}


def make_env(name: str, fast_mode: bool):
    """Construct a named FNSB environment.

    Args:
        name: One of ALL_ENVS.
        fast_mode: Passed to lightweight constructors where supported.

    Returns:
        An instantiated gymnasium-compatible environment.

    Raises:
        ValueError: If name is not in _ENV_CONSTRUCTORS.
    """
    if name not in _ENV_CONSTRUCTORS:
        raise ValueError(f"Unknown env {name!r}. Valid: {list(_ENV_CONSTRUCTORS)}")
    return _ENV_CONSTRUCTORS[name](fast_mode)


def _get_action_dim(env) -> int:
    """Extract a scalar action dimension from any action space.

    Supports:
    - Box spaces: returns shape[0] (continuous actions).
    - Discrete spaces: returns 1 (single integer action, baselines output zeros).

    Args:
        env: A gymnasium environment with an ``action_space`` attribute.

    Returns:
        Integer action dimension for use with baseline agents.
    """
    import gymnasium as gym
    if isinstance(env.action_space, gym.spaces.Box):
        return int(env.action_space.shape[0])
    # Discrete (TradingRegime, RegimeMaze): baselines output np.zeros(1) — interpreted
    # as action index 0, which is valid for all Discrete envs.
    return 1


# ---------------------------------------------------------------------------
# BVH agent — real Predictor-backed agent
# ---------------------------------------------------------------------------

class BVHAgent(BaselineAgent):
    """BVH-RSSM agent backed by a Predictor instance.

    Uses the Predictor's /predict logic to run posterior RSSM inference at each
    step. The RSSM state is carried as serialised bytes in the agent state dict,
    making this agent compatible with the stateless episode runner.

    When router_signal == "STALE", just_switched is set True so the episode
    runner can record the predicted switch step for f1_switching metrics.

    The policy outputs zero actions — the agent's value is in tau prediction
    and router signalling, not in action quality (which requires a trained actor).
    A trained actor can be plugged in by subclassing and overriding _get_action().

    Args:
        predictor: Initialised Predictor instance.
        action_dim: Action space dimension (used for zero-action fallback).
        obs_dim: Expected observation dimension. If obs doesn't match, fall back
                 to stub behaviour (zero tau, no state update).
    """

    def __init__(self, predictor: Predictor, action_dim: int, obs_dim: int = -1) -> None:
        self.predictor = predictor
        self.action_dim = action_dim
        self.obs_dim = obs_dim  # -1 means no check

    def _get_action(self, obs: np.ndarray, action_dim: int) -> List[float]:
        """Return action to send to env. Override in subclass for trained policy."""
        return [0.0] * action_dim

    def initial_state(self) -> Dict[str, Any]:
        return {
            "step": 0,
            "just_switched": False,
            "tau_pred": 0.0,
            "rssm_state_bytes": None,   # None triggers cold start in Predictor
        }

    def act(
        self, obs: np.ndarray, state: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action_list = self._get_action(obs, self.action_dim)
        action = np.array(action_list, dtype=np.float32)

        # Obs_dim mismatch → degrade to stub (no RSSM update, tau=0)
        if self.obs_dim > 0 and obs.shape[0] != self.obs_dim:
            return action, {
                "step": state["step"] + 1,
                "just_switched": False,
                "tau_pred": 0.0,
                "rssm_state_bytes": state["rssm_state_bytes"],
            }

        result = self.predictor.predict(
            obs=obs.tolist(),
            action=action_list,
            state_bytes=state["rssm_state_bytes"],
        )
        tau = float(result["tau"])
        just_switched = result["router_signal"] == "STALE"
        return action, {
            "step": state["step"] + 1,
            "just_switched": just_switched,
            "tau_pred": tau,
            "rssm_state_bytes": result["state"],
        }


class BVHStub(BaselineAgent):
    """Stub BVH agent: predicts tau=0 always, acts with zeros.

    Kept for backward compatibility. Used when no checkpoint is provided and
    fast_mode is False (so we still run the metric pipeline end-to-end).

    Args:
        action_dim: Action space dimension.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def initial_state(self) -> Dict[str, Any]:
        return {"step": 0, "just_switched": False, "tau_pred": 0.0}

    def act(
        self, obs: np.ndarray, state: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action, {
            "step": state["step"] + 1,
            "just_switched": False,
            "tau_pred": 0.0,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Metrics collected from one episode."""
    tau_preds: List[float] = field(default_factory=list)
    tau_stars: List[float] = field(default_factory=list)
    total_return: float = 0.0
    switch_steps_pred: List[int] = field(default_factory=list)
    switch_steps_true: List[int] = field(default_factory=list)
    n_steps: int = 0


def run_episode(
    env,
    agent: BaselineAgent,
    max_steps: int,
    seed: int,
) -> EpisodeResult:
    """Roll out one episode. Collect tau predictions and oracle tau from info.

    Args:
        env: An FNSB environment (ShiftWrapper subclass).
        agent: Any BaselineAgent implementation.
        max_steps: Maximum steps before forced truncation.
        seed: Seed passed to env.reset().

    Returns:
        EpisodeResult with per-step tau predictions, oracle taus, and return.
    """
    obs, info = env.reset(seed=seed)
    agent_state = agent.initial_state()
    result = EpisodeResult()

    # oracle_tau at reset is the first tau* before any step
    result.tau_stars.append(float(info.get("oracle_tau", 0)))
    result.tau_preds.append(float(agent_state.get("tau_pred", 0.0)))

    for t in range(max_steps):
        action, agent_state = agent.act(obs, agent_state)

        # Discrete envs receive an integer action; Box envs receive a float array.
        # Baseline agents always output np.zeros(action_dim). For Discrete envs
        # (action_dim=1), cast the scalar to int to satisfy the env's type check.
        import gymnasium as gym
        if isinstance(env.action_space, gym.spaces.Discrete):
            stepped_action = int(action[0]) if hasattr(action, "__len__") else int(action)
        else:
            stepped_action = action

        obs, reward, terminated, truncated, info = env.step(stepped_action)

        result.total_return += float(reward)
        result.tau_stars.append(float(info.get("oracle_tau", 0)))
        result.tau_preds.append(float(agent_state.get("tau_pred", 0.0)))
        result.n_steps += 1

        # Track switch detections for future f1_switching use
        if info.get("shift_occurred", False):
            result.switch_steps_true.append(t)
        if agent_state.get("just_switched", False):
            result.switch_steps_pred.append(t)

        if terminated or truncated:
            break

    return result


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agents(
    action_dim: int,
    checkpoint: Optional[str],
    include_bvh_untrained: bool = False,
    device: str = "cpu",
    obs_dim: int = 8,
) -> Dict[str, BaselineAgent]:
    """Construct all agents for the evaluation run.

    Always includes FixedIntervalSwitch and RandomSwitch.

    If checkpoint is provided: loads a real BVHAgent from the checkpoint via
    Predictor.from_checkpoint(). The agent runs full RSSM posterior inference
    at every step.

    If include_bvh_untrained is True (--fast-bvh flag): instantiates a BVHAgent
    backed by Predictor.from_scratch(fast_mode=True). Useful for verifying the
    full eval pipeline end-to-end without a trained model.

    If neither: includes BVHStub (tau=0 always) as a pessimistic baseline that
    confirms the metric pipeline works.

    Args:
        action_dim: Action space dimension (used for zero-action policy).
        checkpoint: Path to a .pt checkpoint file, or None.
        include_bvh_untrained: If True, add an untrained BVHAgent (fast_mode).
        device: Device string for Predictor inference.

    Returns:
        Dict mapping agent name to agent instance.
    """
    agents: Dict[str, BaselineAgent] = {
        "FixedIntervalSwitch": FixedIntervalSwitch(switch_interval=20, action_dim=action_dim),
        "RandomSwitch": RandomSwitch(switch_rate=0.05, action_dim=action_dim, seed=0),
    }

    if checkpoint is not None:
        print(f"  Loading BVH checkpoint: {checkpoint}", flush=True)
        predictor = Predictor.from_checkpoint(checkpoint, device=device)
        agents["BVH"] = BVHAgent(predictor, action_dim=action_dim, obs_dim=predictor._obs_dim)
    elif include_bvh_untrained:
        print(f"  Using untrained BVH predictor (obs_dim={obs_dim}, action_dim={action_dim})", flush=True)
        # Build a custom from_scratch predictor matching env dims exactly
        import torch
        from bvh_rssm.networks.rssm import RSSM
        from bvh_rssm.networks.encoder import Encoder
        from bvh_rssm.networks.heads import ValidityHead, HazardHead
        h_dim, z_cats, z_classes, embed_dim, hidden_dim = 32, 4, 4, 16, 32
        latent_dim = h_dim + z_cats * z_classes
        predictor = Predictor(
            encoder=Encoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=2),
            rssm=RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes, obs_dim=embed_dim, action_dim=action_dim),
            tau_head=ValidityHead(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim),
            hazard_head=HazardHead(latent_dim=latent_dim, n_intervals=16, hidden_dim=hidden_dim),
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=torch.device(device),
        )
        agents["BVH_untrained"] = BVHAgent(predictor, action_dim=action_dim, obs_dim=obs_dim)
    else:
        agents["BVH_stub"] = BVHStub(action_dim=action_dim)

    return agents


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def evaluate(
    env_names: List[str],
    n_seeds: int,
    fast_mode: bool,
    checkpoint: Optional[str],
    max_steps_per_episode: int = 500,
    include_bvh_untrained: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run all agents on all environments across all seeds.

    Args:
        env_names: List of environment names to evaluate.
        n_seeds: Number of seeds to run per (agent, env) pair.
        fast_mode: If True, use lightweight constructors and fewer steps.
        checkpoint: Optional path to BVH checkpoint.
        max_steps_per_episode: Maximum steps before truncating an episode.
        include_bvh_untrained: If True, add untrained BVHAgent (fast_mode predictor).
        device: Inference device for BVHAgent.

    Returns:
        Nested dict: {agent_name: {env_name: {metric: value, ...}}}.
    """
    if fast_mode:
        max_steps_per_episode = 200

    # Discover action_dim and obs_dim from the first available env.
    action_dim = 1
    obs_dim = 8  # safe default matching Predictor.from_scratch
    for _probe_name in env_names:
        try:
            probe_env = make_env(_probe_name, fast_mode)
            action_dim = _get_action_dim(probe_env)
            obs, _ = probe_env.reset(seed=0)
            obs_dim = int(obs.shape[0])
            probe_env.close()
            break
        except Exception:
            continue

    agents = build_agents(
        action_dim,
        checkpoint,
        include_bvh_untrained=include_bvh_untrained,
        device=device,
        obs_dim=obs_dim,
    )
    results: Dict[str, Dict[str, Any]] = {name: {} for name in agents}

    for env_name in env_names:
        # Skip envs that require optional dependencies (mujoco, minigrid) not installed.
        try:
            probe = make_env(env_name, fast_mode)
            probe.close()
        except Exception as exc:
            print(f"  env={env_name} SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            continue
        print(f"  env={env_name}", flush=True)
        for agent_name, agent in agents.items():
            all_mae: List[float] = []
            all_returns: List[float] = []

            for seed in range(n_seeds):
                env = make_env(env_name, fast_mode)
                episode = run_episode(env, agent, max_steps_per_episode, seed=seed)
                env.close()

                tau_pred_arr = np.array(episode.tau_preds, dtype=np.float32)
                tau_star_arr = np.array(episode.tau_stars, dtype=np.float32)
                all_mae.append(mae_tau(tau_pred_arr, tau_star_arr))
                all_returns.append(episode.total_return)

            results[agent_name][env_name] = {
                "mae_tau": float(np.mean(all_mae)),
                "mean_return": float(np.mean(all_returns)),
                "seeds_run": n_seeds,
                "_raw_returns": all_returns,
            }

    # Compute delta_return for each agent vs RandomSwitch.
    # Snapshot RandomSwitch raw returns before the deletion loop: the loop
    # deletes _raw_returns as it processes each agent, and since random_results
    # is a reference into the same dict, deleting RandomSwitch's entry in-loop
    # would silently zero out delta_return for all subsequent agents.
    random_raw: Dict[str, list] = {}
    if "RandomSwitch" in results:
        for env_name_key in env_names:
            if env_name_key in results["RandomSwitch"]:
                random_raw[env_name_key] = results["RandomSwitch"][env_name_key]["_raw_returns"]

    for agent_name in results:
        for env_name in env_names:
            if env_name not in results[agent_name]:
                continue
            agent_returns = np.array(results[agent_name][env_name]["_raw_returns"])
            random_returns = np.array(random_raw.get(env_name, agent_returns))
            results[agent_name][env_name]["delta_return_vs_random"] = delta_return(
                agent_returns, random_returns
            )
            del results[agent_name][env_name]["_raw_returns"]

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BVH-RSSM agents across environments and seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to BVH .pt checkpoint. If omitted, BVHStub (tau=0) is used.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: 2 envs, 2 seeds, 200 steps/episode (~2 min on Mac).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds per (agent, env) pair. Overrides fast/full default.",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Comma-separated list of env names. Overrides fast/full default.",
    )
    parser.add_argument(
        "--fast-bvh",
        action="store_true",
        default=False,
        help="Include an untrained BVHAgent (from_scratch) to test the full eval pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device for BVHAgent (e.g. 'cpu', 'cuda:0', 'mps').",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fast_mode: bool = args.fast
    n_seeds: int = args.seeds if args.seeds is not None else (2 if fast_mode else 8)
    if args.envs is not None:
        env_names = [e.strip() for e in args.envs.split(",")]
    else:
        env_names = FAST_ENVS if fast_mode else ALL_ENVS

    print(f"BVH-RSSM Eval — fast={fast_mode}, envs={env_names}, seeds={n_seeds}")
    print(f"checkpoint={args.checkpoint!r}")

    results = evaluate(
        env_names=env_names,
        n_seeds=n_seeds,
        fast_mode=fast_mode,
        checkpoint=args.checkpoint,
        include_bvh_untrained=args.fast_bvh,
        device=args.device,
    )

    # Serialize
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = output_dir / f"eval_{timestamp}.json"

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fast_mode": fast_mode,
        "checkpoint": args.checkpoint,
        "n_seeds": n_seeds,
        "envs": env_names,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {output_path}")

    # Print summary table
    print(f"\n{'Agent':<28} {'Env':<20} {'MAE_tau':>10} {'mean_return':>13} {'delta_vs_rand':>14}")
    print("-" * 88)
    for agent_name, env_dict in results.items():
        for env_name, metrics in env_dict.items():
            print(
                f"{agent_name:<28} {env_name:<20} "
                f"{metrics['mae_tau']:>10.2f} "
                f"{metrics['mean_return']:>13.2f} "
                f"{metrics['delta_return_vs_random']:>14.2f}"
            )


if __name__ == "__main__":
    main()
