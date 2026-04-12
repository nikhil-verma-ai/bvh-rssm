"""
Pydantic v2 request/response models for the BVH-RSSM serving API.

Wire format:
  POST /predict  — PredictRequest → PredictResponse
  POST /refresh  — RefreshRequest → RefreshResponse

All list fields with fixed length (survival_curve, lambda_intervals) are
validated with field_validator so a wrong-length payload is rejected at the
boundary with a clear 422, not a silent truncation.

RouterSignal literal values:
  HIGH  — tau is high; world model is valid, trust predictions
  DIM   — tau is moderate; predictions usable but staleness growing
  STALE — tau is low; world model likely stale, consider retraining
"""
from __future__ import annotations

import base64
from typing import Literal, Optional

from pydantic import BaseModel, field_serializer, field_validator

RouterSignal = Literal["HIGH", "DIM", "STALE"]

# Number of discrete survival intervals — must match HazardHead.n_intervals default.
# Validated explicitly so mismatched client payloads are rejected at the boundary.
_N_INTERVALS: int = 16


class PredictRequest(BaseModel):
    """Single-step inference request.

    Fields:
        obs: Raw observation vector (any length >= 1; must match model obs_dim at runtime).
        action: Action vector (any length >= 1; must match model action_dim at runtime).
        state: Opaque serialised RSSM State bytes from a prior response, or None for
               episode start (server will initialise a fresh zero State).
    """

    obs: list[float]
    action: list[float]
    state: Optional[bytes] = None

    @field_validator("obs")
    @classmethod
    def obs_nonempty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("obs must be non-empty")
        return v

    @field_validator("action")
    @classmethod
    def action_nonempty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("action must be non-empty")
        return v

    @field_validator("state", mode="before")
    @classmethod
    def decode_state_base64(cls, v: object) -> object:
        """Accept base64-encoded strings from JSON clients and decode to bytes.

        Pydantic v2 does not auto-decode base64 for bytes fields, so a client
        that echoes back the base64 string from PredictResponse.state must have
        it decoded here before Pydantic validates the type as bytes.
        """
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class PredictResponse(BaseModel):
    """Single-step inference response.

    Fields:
        tau: Expected validity horizon in steps (symexp-decoded from ValidityHead).
        survival_curve: S(t) for t=1..16 from HazardHead — monotonically non-increasing.
        router_signal: Categorical routing hint derived from tau.
        lambda_intervals: Per-interval combined hazard probabilities (16 values).
        state: Serialised RSSM State bytes. Pass back in next PredictRequest.state.
    """

    tau: float
    survival_curve: list[float]
    router_signal: RouterSignal
    lambda_intervals: list[float]
    state: bytes

    @field_validator("survival_curve")
    @classmethod
    def survival_curve_len(cls, v: list[float]) -> list[float]:
        if len(v) != _N_INTERVALS:
            raise ValueError(
                f"survival_curve must have exactly {_N_INTERVALS} elements, got {len(v)}"
            )
        return v

    @field_validator("lambda_intervals")
    @classmethod
    def lambda_intervals_len(cls, v: list[float]) -> list[float]:
        if len(v) != _N_INTERVALS:
            raise ValueError(
                f"lambda_intervals must have exactly {_N_INTERVALS} elements, got {len(v)}"
            )
        return v

    @field_serializer("state")
    def serialize_state(self, v: bytes) -> str:
        """Encode binary state bytes as base64 for JSON transport.

        Torch-serialised state blobs contain arbitrary binary data and cannot
        be embedded in JSON as raw strings. Base64 encoding is the standard
        approach; clients echo the string back verbatim and decode_state_base64
        in PredictRequest reconstructs the original bytes.
        """
        return base64.b64encode(v).decode("ascii")


class RefreshRequest(BaseModel):
    """Posterior batch update request.

    Fields:
        obs_batch: List of observation vectors (one per recent timestep).
                   Must have at least 1 element.
    """

    obs_batch: list[list[float]]

    @field_validator("obs_batch")
    @classmethod
    def batch_nonempty(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) == 0:
            raise ValueError("obs_batch must have at least 1 observation")
        return v


class RefreshResponse(BaseModel):
    """Posterior batch update response.

    Fields:
        new_tau: Updated tau estimate after observing the batch.
        retrain_needed: True if new_tau has fallen below retrain threshold (tau < 5.0).
        state: Updated serialised RSSM State bytes.
    """

    new_tau: float
    retrain_needed: bool
    state: bytes

    @field_serializer("state")
    def serialize_state(self, v: bytes) -> str:
        """Encode binary state bytes as base64 for JSON transport.

        Mirrors PredictResponse.serialize_state — see that docstring for rationale.
        """
        return base64.b64encode(v).decode("ascii")
