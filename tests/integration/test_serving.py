"""Integration smoke tests for the serving layer (no real server process needed)."""
import pytest
from pydantic import ValidationError


class TestSchemas:
    def test_predict_request_valid(self):
        from bvh_rssm.serving.schemas import PredictRequest
        req = PredictRequest(obs=[0.1, 0.2], action=[0.3], state=None)
        assert req.obs == [0.1, 0.2]
        assert req.action == [0.3]
        assert req.state is None

    def test_predict_request_with_state_bytes(self):
        from bvh_rssm.serving.schemas import PredictRequest
        req = PredictRequest(obs=[1.0], action=[2.0], state=b"\x00\x01\x02")
        assert req.state == b"\x00\x01\x02"

    def test_predict_request_rejects_non_float_obs(self):
        from bvh_rssm.serving.schemas import PredictRequest
        with pytest.raises(ValidationError):
            PredictRequest(obs=["not", "floats"], action=[1.0], state=None)

    def test_predict_response_valid(self):
        from bvh_rssm.serving.schemas import PredictResponse
        resp = PredictResponse(
            tau=3.5,
            survival_curve=[0.9] * 16,
            router_signal="HIGH",
            lambda_intervals=[0.1] * 16,
            state=b"abc",
        )
        assert resp.tau == 3.5
        assert len(resp.survival_curve) == 16

    def test_predict_response_rejects_wrong_survival_curve_length(self):
        from bvh_rssm.serving.schemas import PredictResponse
        with pytest.raises(ValidationError):
            PredictResponse(
                tau=1.0,
                survival_curve=[0.9] * 8,   # wrong length — must be 16
                router_signal="HIGH",
                lambda_intervals=[0.1] * 16,
                state=b"x",
            )

    def test_predict_response_rejects_wrong_lambda_intervals_length(self):
        from bvh_rssm.serving.schemas import PredictResponse
        with pytest.raises(ValidationError):
            PredictResponse(
                tau=1.0,
                survival_curve=[0.9] * 16,
                router_signal="HIGH",
                lambda_intervals=[0.1] * 5,  # wrong length — must be 16
                state=b"x",
            )

    def test_predict_response_rejects_invalid_router_signal(self):
        from bvh_rssm.serving.schemas import PredictResponse
        with pytest.raises(ValidationError):
            PredictResponse(
                tau=1.0,
                survival_curve=[0.9] * 16,
                router_signal="INVALID",      # not one of HIGH/DIM/STALE
                lambda_intervals=[0.1] * 16,
                state=b"x",
            )

    def test_refresh_request_valid(self):
        from bvh_rssm.serving.schemas import RefreshRequest
        req = RefreshRequest(obs_batch=[[0.1, 0.2], [0.3, 0.4]])
        assert len(req.obs_batch) == 2

    def test_refresh_request_rejects_empty_batch(self):
        from bvh_rssm.serving.schemas import RefreshRequest
        with pytest.raises(ValidationError):
            RefreshRequest(obs_batch=[])  # must have >= 1 observation

    def test_refresh_response_valid(self):
        from bvh_rssm.serving.schemas import RefreshResponse
        resp = RefreshResponse(new_tau=12.5, retrain_needed=False, state=b"bytes")
        assert resp.retrain_needed is False
