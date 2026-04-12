"""
Predictor — stateless single-step inference wrapper for BVH-RSSM.

Encapsulates Encoder + RSSM + ValidityHead + HazardHead into a single
callable. All state is serialised to/from bytes so the HTTP layer remains
fully stateless: the client owns the RSSM State blob and passes it back
each request.

Public API:
    Predictor.from_scratch(fast_mode)        — construct tiny random-weight model
    Predictor.from_checkpoint(path, device)  — load from training checkpoint
    predictor.predict(obs, action, state_bytes) -> dict
    predictor.refresh(obs_batch)             -> dict
    predictor._serialise_state(state)        -> bytes
    predictor._deserialise_state(state_bytes) -> State

Router signal thresholds (per spec):
    tau >= 20 → "HIGH"   (world model confident, trust predictions)
    5 <= tau < 20 → "DIM"   (moderate, staleness growing)
    tau < 5  → "STALE"  (stale, retrain)
"""
from __future__ import annotations

from io import BytesIO
from typing import List, Optional

import torch
import torch.nn as nn

from bvh_rssm.networks.rssm import RSSM, State
from bvh_rssm.networks.encoder import Encoder
from bvh_rssm.networks.heads import ValidityHead, HazardHead

# Retrain threshold: tau below this triggers retrain_needed=True
_RETRAIN_THRESHOLD: float = 5.0

# Router signal thresholds
_HIGH_THRESHOLD: float = 20.0
_DIM_THRESHOLD: float = 5.0


def _tau_to_router_signal(tau: float) -> str:
    """Convert scalar tau to router signal literal.

    Thresholds (per spec):
        tau >= 20 → "HIGH"
        5 <= tau < 20 → "DIM"
        tau < 5 → "STALE"

    Args:
        tau: Expected validity horizon in steps (>= 0.0).

    Returns:
        One of "HIGH", "DIM", "STALE".
    """
    if tau >= _HIGH_THRESHOLD:
        return "HIGH"
    elif tau >= _DIM_THRESHOLD:
        return "DIM"
    else:
        return "STALE"


class Predictor:
    """Stateless inference engine for BVH-RSSM.

    All four networks are set to eval mode on construction and never switched
    back to train mode by this class. Inference always runs under
    torch.no_grad() to ensure no gradient accumulation.

    Args:
        encoder: Observation encoder (raw obs → embed).
        rssm: Recurrent state space model.
        tau_head: ValidityHead — predicts validity horizon τ.
        hazard_head: HazardHead — predicts survival curve S(t) and λ(t).
        obs_dim: Raw observation dimensionality (encoder input size).
        action_dim: Action dimensionality.
        device: Torch device for inference.
    """

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        tau_head: ValidityHead,
        hazard_head: HazardHead,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self._encoder = encoder.to(device).eval()
        self._rssm = rssm.to(device).eval()
        self._tau_head = tau_head.to(device).eval()
        self._hazard_head = hazard_head.to(device).eval()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._device = device

    @classmethod
    def from_scratch(cls, fast_mode: bool = True, device: str = "cpu") -> "Predictor":
        """Construct a Predictor with randomly initialised weights.

        In fast_mode=True all hidden dims are shrunk to 32 and latent dims
        to tiny values so test suites run in milliseconds on CPU.

        Hyperparameters (fast_mode=True):
            obs_dim=8, action_dim=3
            embed_dim=16  (encoder output → RSSM obs_dim)
            h_dim=32, z_cats=4, z_classes=4  → z_dim=16, latent_dim=48
            n_intervals=16 (HazardHead)
            hidden_dim=32 (all MLPs)

        Args:
            fast_mode: If True, use tiny dimensions for fast testing.
            device: Device string for torch.device().

        Returns:
            Predictor ready for inference.
        """
        dev = torch.device(device)

        if fast_mode:
            obs_dim = 8
            action_dim = 3
            embed_dim = 16
            h_dim = 32
            z_cats = 4
            z_classes = 4
            hidden_dim = 32
            n_intervals = 16
        else:
            # Production defaults matching DreamerV3 scale
            obs_dim = 64
            action_dim = 6
            embed_dim = 256
            h_dim = 512
            z_cats = 32
            z_classes = 32
            hidden_dim = 512
            n_intervals = 16

        latent_dim = h_dim + z_cats * z_classes  # h + z concatenated

        encoder = Encoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
        rssm = RSSM(
            h_dim=h_dim,
            z_cats=z_cats,
            z_classes=z_classes,
            obs_dim=embed_dim,   # RSSM.obs_dim is encoder output = embed_dim
            action_dim=action_dim,
        )
        tau_head = ValidityHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        hazard_head = HazardHead(
            latent_dim=latent_dim,
            n_intervals=n_intervals,
            hidden_dim=hidden_dim,
        )

        return cls(
            encoder=encoder,
            rssm=rssm,
            tau_head=tau_head,
            hazard_head=hazard_head,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=dev,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "Predictor":
        """Load Predictor from a training checkpoint.

        Checkpoint format (saved by training loop):
            {
                "model": {
                    "encoder.*": ...,
                    "rssm.*": ...,
                    "tau_head.*": ...,
                    "hazard_head.*": ...,
                },
                "hparams": {
                    "obs_dim": int,
                    "action_dim": int,
                    "embed_dim": int,
                    "h_dim": int,
                    "z_cats": int,
                    "z_classes": int,
                    "hidden_dim": int,
                    "n_intervals": int,
                }
            }

        Hyperparameters are read from "hparams" if present; otherwise inferred
        from state dict shapes as a best-effort fallback.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Device string for torch.device().

        Returns:
            Predictor with loaded weights.
        """
        dev = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev)
        state_dict = ckpt["model"]

        if "hparams" in ckpt:
            hp = ckpt["hparams"]
            obs_dim = hp["obs_dim"]
            action_dim = hp["action_dim"]
            embed_dim = hp["embed_dim"]
            h_dim = hp["h_dim"]
            z_cats = hp["z_cats"]
            z_classes = hp["z_classes"]
            hidden_dim = hp.get("hidden_dim", 512)
            n_intervals = hp.get("n_intervals", 16)
        else:
            # Infer from state dict shapes — brittle but functional fallback
            # encoder.mlp.net.0.weight shape: [hidden_dim, obs_dim]
            enc_w = state_dict["encoder.mlp.net.0.weight"]
            hidden_dim = enc_w.shape[0]
            obs_dim = enc_w.shape[1]
            # encoder output (embed_dim) = last linear weight out_dim
            # Find the last linear in encoder by scanning keys
            enc_keys = [k for k in state_dict if k.startswith("encoder.")]
            embed_dim = max(
                state_dict[k].shape[0]
                for k in enc_keys
                if k.endswith(".weight") and len(state_dict[k].shape) == 2
            )
            # rssm.gru_cell.weight_ih shape: [3*h_dim, z_dim + action_dim]
            gru_ih = state_dict["rssm.gru_cell.weight_ih"]
            h_dim = gru_ih.shape[0] // 3
            # rssm.prior_head output: z_cats * z_classes
            prior_out = state_dict["rssm.prior_head.net.-1.weight"].shape[0]
            # z_cats inferred from posterior head input vs h_dim
            # posterior input = h_dim + embed_dim; output = z_cats * z_classes
            # We need z_cats: assume z_cats=z_classes=sqrt(prior_out) if square
            import math
            side = int(math.sqrt(prior_out))
            z_cats = side
            z_classes = side
            # hazard head source_b output
            hz_w = state_dict["hazard_head.source_b.net.-1.weight"]
            n_intervals = hz_w.shape[0]
            # action_dim from tau_head action_embed
            action_embed_w = state_dict["tau_head.action_embed.net.0.weight"]
            action_dim = action_embed_w.shape[1]

        latent_dim = h_dim + z_cats * z_classes

        encoder = Encoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=hidden_dim)
        rssm = RSSM(
            h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
            obs_dim=embed_dim, action_dim=action_dim,
        )
        tau_head = ValidityHead(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        hazard_head = HazardHead(
            latent_dim=latent_dim, n_intervals=n_intervals, hidden_dim=hidden_dim,
        )

        # Strip prefixes and load each sub-module
        def _strip_and_load(module: nn.Module, prefix: str) -> None:
            sub = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            module.load_state_dict(sub, strict=True)

        _strip_and_load(encoder, "encoder.")
        _strip_and_load(rssm, "rssm.")
        _strip_and_load(tau_head, "tau_head.")
        _strip_and_load(hazard_head, "hazard_head.")

        return cls(
            encoder=encoder,
            rssm=rssm,
            tau_head=tau_head,
            hazard_head=hazard_head,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=dev,
        )

    # ------------------------------------------------------------------
    # State serialisation: bytes ↔ State(h, z)
    # ------------------------------------------------------------------

    def _serialise_state(self, state: State) -> bytes:
        """Serialise RSSM State to bytes via torch.save.

        Both tensors are moved to CPU before saving so the bytes are
        device-agnostic. Deserialisation moves them back to self._device.

        Args:
            state: State namedtuple with h=[1, h_dim] and z=[1, z_dim].

        Returns:
            Raw bytes (pickle-based torch format).
        """
        buf = BytesIO()
        torch.save({"h": state.h.cpu(), "z": state.z.cpu()}, buf)
        return buf.getvalue()

    def _deserialise_state(self, state_bytes: bytes) -> State:
        """Deserialise bytes back to a State namedtuple on self._device.

        Args:
            state_bytes: Bytes produced by _serialise_state.

        Returns:
            State with h and z on self._device.
        """
        buf = BytesIO(state_bytes)
        # weights_only=False required: torch.save'd dict with tensors
        data = torch.load(buf, map_location=self._device, weights_only=True)
        return State(h=data["h"], z=data["z"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: List[float],
        action: List[float],
        state_bytes: Optional[bytes],
    ) -> dict:
        """Single-step posterior inference.

        Encodes the observation, performs a posterior RSSM update, then runs
        ValidityHead and HazardHead on the resulting latent. The updated State
        is serialised and returned so the caller can pass it back next call.

        All computation runs under torch.no_grad(); returned state tensors
        have requires_grad=False.

        Args:
            obs: Raw observation vector of length obs_dim.
            action: Action vector of length action_dim.
            state_bytes: Serialised State bytes from a prior response, or None
                         for episode start (a fresh zero State is created).

        Returns:
            dict with keys:
                tau (float): Expected validity horizon (>= 0.0).
                survival_curve (list[float]): S(t) for t=1..16.
                router_signal (str): "HIGH", "DIM", or "STALE".
                lambda_intervals (list[float]): Per-interval hazard h_total(i).
                state (bytes): Serialised next State.
        """
        with torch.no_grad():
            # --- Build tensors --------------------------------------------------
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)       # [1, obs_dim]
            act_t = torch.tensor(action, dtype=torch.float32, device=self._device).unsqueeze(0)    # [1, action_dim]

            # --- State initialisation ------------------------------------------
            if state_bytes is None:
                state = self._rssm.initial_state(batch_size=1, device=self._device)
            else:
                state = self._deserialise_state(state_bytes)

            # --- Encode observation --------------------------------------------
            obs_embed = self._encoder(obs_t)   # [1, embed_dim]

            # --- Posterior RSSM update -----------------------------------------
            # observe returns (posterior_logits, next_state)
            _logits, next_state = self._rssm.observe(obs_embed, act_t, state)

            # --- Full latent for heads -----------------------------------------
            latent = self._rssm.get_latent(next_state)   # [1, h_dim + z_dim]

            # --- ValidityHead: tau ---------------------------------------------
            tau_logits = self._tau_head(latent, act_t, stop_grad=False)  # [1, n_bins]
            tau = self._tau_head.decode(tau_logits).squeeze().item()      # scalar

            # --- HazardHead: survival + combined hazard -------------------------
            survival = self._hazard_head.survival(latent)         # [1, 16]
            lambda_h = self._hazard_head.combined_hazard(latent)  # [1, 16]

            survival_list = survival.squeeze(0).tolist()   # list[float] len 16
            lambda_list = lambda_h.squeeze(0).tolist()     # list[float] len 16

        return {
            "tau": float(tau),
            "survival_curve": survival_list,
            "router_signal": _tau_to_router_signal(float(tau)),
            "lambda_intervals": lambda_list,
            "state": self._serialise_state(next_state),
        }

    def refresh(
        self,
        obs_batch: List[List[float]],
    ) -> dict:
        """Sequential posterior update over a batch of recent observations.

        Rolls through obs_batch with zero actions, updating the RSSM state
        at each step. The final state and its associated tau are returned.
        Used by the /refresh endpoint after the client sends recent history.

        Args:
            obs_batch: List of T observation vectors, each of length obs_dim.

        Returns:
            dict with keys:
                new_tau (float): Tau estimated from the final state.
                retrain_needed (bool): True if new_tau < _RETRAIN_THRESHOLD.
                state (bytes): Serialised final State.
        """
        with torch.no_grad():
            # Zero action — no action information available during refresh
            zero_action = torch.zeros(
                1, self._action_dim, dtype=torch.float32, device=self._device
            )  # [1, action_dim]

            state = self._rssm.initial_state(batch_size=1, device=self._device)

            for obs in obs_batch:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self._device
                ).unsqueeze(0)   # [1, obs_dim]
                obs_embed = self._encoder(obs_t)   # [1, embed_dim]
                _logits, state = self._rssm.observe(obs_embed, zero_action, state)

            # Compute tau from final state
            latent = self._rssm.get_latent(state)  # [1, latent_dim]
            tau_logits = self._tau_head(latent, zero_action, stop_grad=False)
            tau = self._tau_head.decode(tau_logits).squeeze().item()

        return {
            "new_tau": float(tau),
            "retrain_needed": bool(float(tau) < _RETRAIN_THRESHOLD),
            "state": self._serialise_state(state),
        }
