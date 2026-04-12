"""
3-phase BVH-RSSM trainer.

Phase 1: World model pretraining (encoder, decoder, RSSM, reward, continue).
Phase 2: BVH head training with frozen world model (tau_head, hazard_head).
Phase 3: Joint fine-tuning (all weights, full L_BVH loss). [stub in this plan]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from bvh_rssm.training.losses import world_model_loss, validity_loss
from bvh_rssm.training.experiment import Checkpointer, log_metrics


@dataclass
class TrainerConfig:
    """Trainer configuration (populated from Hydra config)."""
    phase1_steps: int = 100_000
    phase2_steps: int = 50_000
    phase3_steps: int = 200_000
    learning_rate: float = 1e-4
    grad_clip: float = 100.0
    batch_size: int = 16
    seq_len: int = 16
    lambda_tau: float = 1.0
    lambda_hazard: float = 1.0
    lambda_cf: float = 0.1
    cf_margin: float = 3.0
    log_every: int = 100
    checkpoint_every: int = 10_000
    device: str = "cpu"
    seed: int = 0
    run_dir: str = "runs/default"


class Trainer:
    """3-phase BVH-RSSM trainer.

    Args:
        model: Dict of network modules (encoder, decoder, rssm, reward_head,
               continue_head, tau_head, hazard_head, actor, critic).
        replay_buffer: Pre-filled ReplayBuffer.
        config: TrainerConfig.
    """

    def __init__(self, model: Dict[str, nn.Module], replay_buffer: Any,
                 config: TrainerConfig) -> None:
        self.model = model
        self.buf = replay_buffer
        self.cfg = config
        self.device = torch.device(config.device)
        self.checkpointer = Checkpointer(config.run_dir)
        self._global_step = 0

        for m in model.values():
            if isinstance(m, nn.Module):
                m.to(self.device)

    def _params_for_keys(self, keys):
        """Collect parameters from named modules in the model dict."""
        params = []
        for k in keys:
            if k in self.model and isinstance(self.model[k], nn.Module):
                params.extend(self.model[k].parameters())
        return params

    def _set_requires_grad(self, keys, value: bool) -> None:
        """Freeze or unfreeze parameters of the given module keys."""
        for k in keys:
            if k in self.model and isinstance(self.model[k], nn.Module):
                for p in self.model[k].parameters():
                    p.requires_grad_(value)

    def train_phase1(self) -> None:
        """Phase 1: world model pretraining.

        Trains encoder, decoder, RSSM, reward_head, continue_head.
        Freezes tau_head and hazard_head (not yet used).
        Loss: DreamerV3-style world model loss (prediction + dynamics KL + repr KL).
        """
        wm_keys = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
        self._set_requires_grad(wm_keys, True)
        self._set_requires_grad(["tau_head", "hazard_head"], False)

        params = self._params_for_keys(wm_keys)
        optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate)

        for step in range(self.cfg.phase1_steps):
            batch = self.buf.sample(self.cfg.batch_size, self.cfg.seq_len)
            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)
            rewards = batch["reward"].to(self.device)
            # continues = 1 - terminated: 1 means episode continues, 0 means done
            continues = (1.0 - batch["terminated"].float()).to(self.device)

            optimizer.zero_grad()
            result = world_model_loss(
                obs, actions, rewards, continues,
                self.model["encoder"], self.model["decoder"], self.model["rssm"],
                self.model["reward_head"], self.model["continue_head"],
            )
            result["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            self._global_step += 1

            if step % self.cfg.log_every == 0:
                log_metrics({
                    "phase1/loss_total": result["total"].item(),
                    "phase1/loss_prediction": result["prediction"].item(),
                    "phase1/loss_dynamics": result["dynamics"].item(),
                    "phase1/loss_representation": result["representation"].item(),
                }, step=self._global_step)

            if (
                self.cfg.checkpoint_every > 0
                and step % self.cfg.checkpoint_every == 0
                and step > 0
            ):
                self.checkpointer.save(
                    nn.ModuleDict(self.model), optimizer, phase=1, step=self._global_step
                )

    def train_phase2(self) -> None:
        """Phase 2: BVH head training with frozen world model.

        Freezes world model weights. Trains tau_head and hazard_head.
        Latents are computed under torch.no_grad() to respect the frozen boundary.
        Loss: validity_loss (cross-entropy over twohot oracle_tau distribution).
        """
        wm_keys = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
        self._set_requires_grad(wm_keys, False)
        self._set_requires_grad(["tau_head", "hazard_head"], True)

        head_keys = ["tau_head", "hazard_head"]
        params = self._params_for_keys(head_keys)
        optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate)

        for step in range(self.cfg.phase2_steps):
            batch = self.buf.sample(self.cfg.batch_size, self.cfg.seq_len)
            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)
            oracle_tau = batch["oracle_tau"].float().to(self.device)

            # Compute frozen latents: unroll RSSM over sequence under no_grad
            with torch.no_grad():
                rssm = self.model["rssm"]
                # initial_state accepts optional device kwarg — infers from parameters if None
                state = rssm.initial_state(self.cfg.batch_size, device=self.device)
                latents = []
                for t in range(obs.shape[1]):
                    emb = self.model["encoder"](obs[:, t])
                    _, state = rssm.observe(emb, actions[:, t], state)
                    latents.append(rssm.get_latent(state))
                # Stack to [B, T, latent_dim] then flatten to [B*T, latent_dim]
                latent = torch.stack(latents, dim=1).reshape(-1, latents[0].shape[-1])

            flat_actions = actions.reshape(-1, actions.shape[-1])
            flat_tau = oracle_tau.reshape(-1)

            optimizer.zero_grad()
            # stop_grad=True detaches latent inside ValidityHead.loss — belt-and-suspenders
            # since latent is already detached by no_grad above
            v_loss = validity_loss(
                self.model["tau_head"], latent, flat_actions, flat_tau, stop_grad=True
            )
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            self._global_step += 1

            if step % self.cfg.log_every == 0:
                log_metrics({"phase2/loss_validity": v_loss.item()}, step=self._global_step)

            if (
                self.cfg.checkpoint_every > 0
                and step % self.cfg.checkpoint_every == 0
                and step > 0
            ):
                self.checkpointer.save(
                    nn.ModuleDict(self.model), optimizer, phase=2, step=self._global_step
                )

    def train(self) -> None:
        """Run all three phases sequentially."""
        self.train_phase1()
        self.train_phase2()
        # Phase 3: joint fine-tuning — implemented in Plan 6
