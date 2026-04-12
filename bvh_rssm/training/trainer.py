"""
3-phase BVH-RSSM trainer.

Phase 1: World model pretraining (encoder, decoder, RSSM, reward, continue).
Phase 2: BVH head training with frozen world model (tau_head, hazard_head).
Phase 3: Joint fine-tuning (all weights, full L_BVH loss, actor-critic in imagination).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bvh_rssm.training.losses import world_model_loss, validity_loss, counterfactual_loss, survival_loss
from bvh_rssm.training.experiment import Checkpointer, log_metrics


@dataclass
class TrainerConfig:
    """Trainer configuration (populated from Hydra config)."""
    phase1_steps: int = 100_000
    phase2_steps: int = 50_000
    phase3_steps: int = 0
    learning_rate: float = 1e-4
    grad_clip: float = 100.0
    batch_size: int = 16
    seq_len: int = 16
    lambda_tau: float = 1.0
    lambda_hazard: float = 1.0
    lambda_cf: float = 0.1
    cf_margin: float = 3.0
    # Actor-critic hyperparameters (Phase 3)
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_coef: float = 3e-4
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

    def train_phase3(self) -> None:
        """Phase 3: joint fine-tuning with actor-critic in imagination space.

        All weights are unfrozen. Three separate optimizers:
          - wm_optimizer:          encoder + decoder + rssm + reward_head + continue_head
          - bvh_optimizer:         tau_head + hazard_head
          - actor_critic_optimizer: actor + critic

        Algorithm (DreamerV3 actor-critic in imagination):
        1. Sample a batch of real starting latents from the buffer (no_grad on WM).
        2. Unroll imagination for `full_horizon` steps using actor + rssm.imagine().
        3. Predict imagined rewards and continues at each step.
        4. Compute lambda-returns R^lambda_t from the imagined trajectory.
        5. Actor loss:  -mean(R^lambda) + entropy_coef * H[pi]
        6. Critic loss: twohot CE against R^lambda (stop-grad).
        7. World-model loss: same as Phase 1 on real buffer samples.
        8. BVH loss:    validity_loss + survival_loss on real buffer samples.
        9. Counterfactual loss: sample a random alt_action, compute tau under both.

        Lambda-return recurrence (reversed):
          R_H = V_H  (bootstrap from last critic value)
          R_t = r_t + gamma * cont_t * ((1-lambda)*V_{t+1} + lambda*R_{t+1})
        """
        wm_keys   = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
        head_keys = ["tau_head", "hazard_head"]
        ac_keys   = ["actor", "critic"]

        # Unfreeze ALL weights for joint fine-tuning
        self._set_requires_grad(wm_keys + head_keys + ac_keys, True)

        wm_params = self._params_for_keys(wm_keys)
        bvh_params = self._params_for_keys(head_keys)
        ac_params = self._params_for_keys(ac_keys)

        wm_optimizer = torch.optim.Adam(wm_params, lr=self.cfg.learning_rate)
        bvh_optimizer = torch.optim.Adam(bvh_params, lr=self.cfg.learning_rate)

        # Only create actor-critic optimizer if both modules exist
        has_actor_critic = "actor" in self.model and "critic" in self.model
        if has_actor_critic and len(ac_params) > 0:
            ac_optimizer = torch.optim.Adam(ac_params, lr=self.cfg.learning_rate)
        else:
            import warnings
            warnings.warn(
                "train_phase3: 'actor' or 'critic' not found in model dict. "
                "Actor-critic training will be skipped for this phase.",
                stacklevel=2,
            )
            has_actor_critic = False
            ac_optimizer = None

        full_horizon = 16
        gamma = self.cfg.gamma
        lambda_ = self.cfg.lambda_
        entropy_coef = self.cfg.entropy_coef

        rssm = self.model["rssm"]
        encoder = self.model["encoder"]
        reward_head = self.model["reward_head"]
        continue_head = self.model["continue_head"]
        tau_head = self.model["tau_head"]
        hazard_head = self.model["hazard_head"]

        for step in range(self.cfg.phase3_steps):
            # ------------------------------------------------------------------
            # 1. Sample a real batch for world-model and BVH losses
            # ------------------------------------------------------------------
            batch = self.buf.sample(self.cfg.batch_size, self.cfg.seq_len)
            obs        = batch["obs"].to(self.device)
            actions    = batch["action"].to(self.device)
            rewards    = batch["reward"].to(self.device)
            continues  = (1.0 - batch["terminated"].float()).to(self.device)
            oracle_tau = batch["oracle_tau"].float().to(self.device)

            # ------------------------------------------------------------------
            # 2. World model loss on real samples
            # ------------------------------------------------------------------
            wm_optimizer.zero_grad()
            wm_result = world_model_loss(
                obs, actions, rewards, continues,
                encoder, self.model["decoder"], rssm,
                reward_head, continue_head,
            )
            wm_result["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm_params, self.cfg.grad_clip)
            wm_optimizer.step()

            # ------------------------------------------------------------------
            # 3. Compute frozen latents for BVH loss (no_grad boundary respected)
            # ------------------------------------------------------------------
            with torch.no_grad():
                state = rssm.initial_state(self.cfg.batch_size, device=self.device)
                latents_list: List[torch.Tensor] = []
                for t in range(obs.shape[1]):
                    emb = encoder(obs[:, t])
                    _, state = rssm.observe(emb, actions[:, t], state)
                    latents_list.append(rssm.get_latent(state))
                # [B, T, latent_dim] → [B*T, latent_dim]
                latents_stacked = torch.stack(latents_list, dim=1)
                flat_latent = latents_stacked.reshape(-1, latents_list[0].shape[-1])

            flat_action    = actions.reshape(-1, actions.shape[-1])
            flat_oracle_tau = oracle_tau.reshape(-1)

            # Derive survival targets from oracle_tau
            K = hazard_head.n_intervals
            flat_event_times    = flat_oracle_tau.long().clamp(0, K - 1)
            flat_event_occurred = (flat_oracle_tau < K)

            # ------------------------------------------------------------------
            # 4. BVH loss: validity + survival + counterfactual
            # ------------------------------------------------------------------
            bvh_optimizer.zero_grad()

            v_loss = validity_loss(
                tau_head, flat_latent, flat_action, flat_oracle_tau, stop_grad=False
            )
            surv_loss = survival_loss(
                hazard_head, flat_latent, flat_event_times, flat_event_occurred,
                use_all_sources=True,
            )

            # Counterfactual: random interventional action as proxy for causal alt
            alt_action = torch.randn_like(flat_action)
            # tau_obs / tau_int are decoded scalar horizon estimates
            tau_obs = tau_head.decode(tau_head(flat_latent, flat_action.detach(), stop_grad=False))
            tau_int = tau_head.decode(tau_head(flat_latent, alt_action.detach(), stop_grad=False))
            cf_loss = counterfactual_loss(tau_int, tau_obs, margin=self.cfg.cf_margin)

            bvh_total = v_loss + surv_loss + self.cfg.lambda_cf * cf_loss
            bvh_total.backward()
            torch.nn.utils.clip_grad_norm_(bvh_params, self.cfg.grad_clip)
            bvh_optimizer.step()

            # ------------------------------------------------------------------
            # 5. Actor-critic loss in imagination space
            # ------------------------------------------------------------------
            actor_loss_val = 0.0
            critic_loss_val = 0.0

            if has_actor_critic:
                actor  = self.model["actor"]
                critic = self.model["critic"]

                # Use the last real latent as the starting state for imagination.
                # shape: [B, latent_dim] — last timestep of the real sequence
                start_latent = latents_stacked[:, -1, :].detach()  # [B, latent_dim]

                # Unroll H imagination steps
                imagined_latents:   List[torch.Tensor] = []
                imagined_rewards:   List[torch.Tensor] = []
                imagined_continues: List[torch.Tensor] = []
                imagined_values:    List[torch.Tensor] = []

                # Build State from the last observed recurrent state so the GRU
                # keeps the temporal context from the real observation sequence.
                # We need h and z separately — recover from concatenated latent.
                h_dim = rssm.h_dim
                img_h = start_latent[:, :h_dim]                         # [B, h_dim]
                img_z = start_latent[:, h_dim:]                         # [B, z_dim]
                from bvh_rssm.networks.rssm import State
                img_state = State(h=img_h, z=img_z)

                for _ in range(full_horizon):
                    lat = rssm.get_latent(img_state)                    # [B, latent_dim]
                    imagined_latents.append(lat)

                    # Actor produces action from imagined latent
                    actor_out = actor(lat)
                    if actor.discrete:
                        # Discrete: sample from logits via gumbel-softmax (straight-through)
                        logits = actor_out                               # [B, action_dim]
                        # Use one-hot relaxation — for imagination we treat the soft sample
                        # as the "action embedding" fed to the GRU
                        act = F.gumbel_softmax(logits, tau=1.0, hard=True)  # [B, action_dim]
                    else:
                        mean, log_std = actor_out
                        std = log_std.exp()
                        act = mean + std * torch.randn_like(mean)       # [B, action_dim]

                    # Imagine one step forward
                    _, img_state = rssm.imagine(act, img_state)

                    # Predict reward and continuation at this imagined latent
                    rew  = reward_head.decode(reward_head(lat))         # [B]
                    cont = continue_head.probability(lat)               # [B]
                    val  = critic(lat)                                  # [B, n_bins]

                    # Decode critic value to scalar for lambda-return computation
                    from bvh_rssm.utils import symexp, symlog_bins, twohot_decode
                    bins = symlog_bins(critic.n_bins).to(self.device)
                    val_scalar = symexp(twohot_decode(val.softmax(-1), bins))  # [B]

                    imagined_rewards.append(rew)
                    imagined_continues.append(cont)
                    imagined_values.append(val_scalar)

                # Bootstrap from the last imagined latent's value
                last_lat = rssm.get_latent(img_state)
                last_val_logits = critic(last_lat)
                bins = symlog_bins(critic.n_bins).to(self.device)
                last_val = symexp(twohot_decode(last_val_logits.softmax(-1), bins))  # [B]

                # Stack into [H, B]
                H = full_horizon
                rew_stack  = torch.stack(imagined_rewards, dim=0)       # [H, B]
                cont_stack = torch.stack(imagined_continues, dim=0)     # [H, B]
                val_stack  = torch.stack(imagined_values, dim=0)        # [H, B]

                # Lambda-return recurrence (backward pass over horizon)
                returns: List[torch.Tensor] = []
                R = last_val                                             # bootstrap [B]
                for t in reversed(range(H)):
                    R = (
                        rew_stack[t]
                        + gamma * cont_stack[t]
                        * ((1.0 - lambda_) * val_stack[t] + lambda_ * R)
                    )
                    returns.insert(0, R)
                returns_stack = torch.stack(returns, dim=0)             # [H, B]

                # --- Actor loss ---
                # Re-run actor on imagined latents to get log-prob / entropy
                lat_all = torch.stack(imagined_latents, dim=0).reshape(H * self.cfg.batch_size, -1)

                ac_optimizer.zero_grad()

                actor_out_all = actor(lat_all)
                if actor.discrete:
                    logits_all = actor_out_all                          # [H*B, action_dim]
                    # Entropy of categorical: -sum(p * log(p))
                    log_probs_all = F.log_softmax(logits_all, dim=-1)
                    probs_all     = log_probs_all.exp()
                    entropy = -(probs_all * log_probs_all).sum(-1)     # [H*B]
                else:
                    mean_all, log_std_all = actor_out_all
                    # Differential entropy of diagonal Gaussian:
                    # H = 0.5 * sum_d (1 + log(2*pi*e*sigma^2_d))
                    #   = 0.5 * sum_d (1 + 2*log_std_d + log(2*pi))
                    entropy = (
                        0.5 * (1.0 + 2.0 * log_std_all + math.log(2.0 * math.pi))
                    ).sum(-1)                                           # [H*B]

                actor_loss = (
                    -returns_stack.detach().reshape(-1).mean()
                    - entropy_coef * entropy.mean()
                )

                # --- Critic loss ---
                # Detach latents so critic gradients don't flow into the world model
                lat_all_detached = lat_all.detach()
                value_logits = critic(lat_all_detached)                # [H*B, n_bins]

                from bvh_rssm.utils import symlog, twohot
                target_symlog = symlog(
                    returns_stack.detach().reshape(-1).clamp(-1e3, 1e3)
                )
                bins_for_target = symlog_bins(critic.n_bins).to(self.device)
                target_th = twohot(target_symlog, bins_for_target)    # [H*B, n_bins]
                log_probs_critic = F.log_softmax(value_logits, dim=-1)
                critic_loss = -(target_th * log_probs_critic).sum(-1).mean()

                ac_loss_total = actor_loss + critic_loss
                ac_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(ac_params, self.cfg.grad_clip)
                ac_optimizer.step()

                actor_loss_val  = actor_loss.item()
                critic_loss_val = critic_loss.item()

            self._global_step += 1

            # ------------------------------------------------------------------
            # 6. Logging
            # ------------------------------------------------------------------
            if step % self.cfg.log_every == 0:
                log_metrics({
                    "phase3/wm_loss":       wm_result["total"].item(),
                    "phase3/actor_loss":    actor_loss_val,
                    "phase3/critic_loss":   critic_loss_val,
                    "phase3/bvh_loss":      bvh_total.item(),
                    "phase3/validity_loss": v_loss.item(),
                    "phase3/survival_loss": surv_loss.item(),
                    "phase3/cf_loss":       cf_loss.item(),
                }, step=self._global_step)

            if (
                self.cfg.checkpoint_every > 0
                and step % self.cfg.checkpoint_every == 0
                and step > 0
            ):
                self.checkpointer.save(
                    nn.ModuleDict(self.model), wm_optimizer, phase=3, step=self._global_step
                )

    def train(self) -> None:
        """Run all three phases sequentially."""
        self.train_phase1()
        self.train_phase2()
        if self.cfg.phase3_steps > 0:
            self.train_phase3()
