"""
FastAPI application for BVH-RSSM serving.

Routes:
  GET  /           — health check
  POST /predict    — single-step inference
  POST /refresh    — posterior batch update

A single Predictor instance is injected at app creation via create_app().
Route handlers receive it through FastAPI dependency injection — no global
state, no singleton pattern, fully testable.

Bytes fields (state) in request/response schemas are base64-encoded for JSON
transport. PredictResponse.state and RefreshResponse.state are serialised to
base64 strings by field_serializer on the Pydantic models. PredictRequest.state
accepts the base64 string verbatim and decodes it back to bytes via
field_validator(mode="before") before Pydantic validates the type.

Clients should pass the state field back verbatim (the base64 string received
in the response) in subsequent /predict calls — no manual base64 needed.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI

from bvh_rssm.serving.predictor import Predictor
from bvh_rssm.serving.schemas import (
    PredictRequest,
    PredictResponse,
    RefreshRequest,
    RefreshResponse,
)


def create_app(predictor: Predictor) -> FastAPI:
    """Construct and return the FastAPI application with a bound Predictor.

    The predictor is captured in a closure so route handlers can access it
    via a dependency function. This pattern avoids module-level globals and
    makes the app fully testable — tests call create_app(mock_predictor).

    Args:
        predictor: A fully-initialised Predictor instance (from checkpoint
                   or from_scratch).

    Returns:
        FastAPI application instance, ready to pass to uvicorn or TestClient.
    """
    app = FastAPI(
        title="BVH-RSSM Serving API",
        version="0.1.0",
        description="Stateless RSSM inference: /predict and /refresh endpoints.",
    )

    def get_predictor() -> Predictor:
        """FastAPI dependency that provides the shared Predictor instance."""
        return predictor

    # NOTE: Annotated type aliases for dependencies must NOT be stored as local
    # variables and referenced by name in function signatures — FastAPI resolves
    # Annotated metadata at decoration time via inspect.signature, and local
    # variable assignments are not visible to that machinery. Instead we use
    # Depends() directly as a default parameter value, which FastAPI always
    # recognises correctly regardless of scope.

    @app.get("/", summary="Health check")
    async def health() -> dict:
        """Return service liveness status."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse, summary="Single-step inference")
    async def predict(
        req: PredictRequest,
        pred: Predictor = Depends(get_predictor),
    ) -> PredictResponse:
        """Run one RSSM step.

        Pass state=null on episode start. Pass the state base64 string from the
        previous response on subsequent steps to maintain continuity.

        The state field is a base64-encoded string over the wire; PredictRequest
        decodes it back to bytes automatically before calling the predictor.

        Returns tau, survival curve, router signal, lambda intervals, and
        updated state as a base64-encoded string.
        """
        result = pred.predict(
            obs=req.obs,
            action=req.action,
            state_bytes=req.state,
        )
        return PredictResponse(**result)

    @app.post("/refresh", response_model=RefreshResponse, summary="Posterior batch update")
    async def refresh(
        req: RefreshRequest,
        pred: Predictor = Depends(get_predictor),
    ) -> RefreshResponse:
        """Run posterior update over a batch of observations.

        Useful after a distribution shift is detected — pass the most recent
        N observations to update the world model state and get a fresh tau.

        Sets retrain_needed=True if new_tau < 5.0 (STALE threshold).
        """
        result = pred.refresh(obs_batch=req.obs_batch)
        return RefreshResponse(**result)

    return app
