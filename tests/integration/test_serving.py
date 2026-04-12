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


class TestPredictor:
    def test_from_scratch_returns_predictor(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        assert p is not None

    def test_predict_returns_required_keys(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert "tau" in result
        assert "survival_curve" in result
        assert "router_signal" in result
        assert "lambda_intervals" in result
        assert "state" in result

    def test_predict_tau_is_nonneg_float(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert isinstance(result["tau"], float)
        assert result["tau"] >= 0.0

    def test_predict_survival_curve_length_16(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert len(result["survival_curve"]) == 16

    def test_predict_lambda_intervals_length_16(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert len(result["lambda_intervals"]) == 16

    def test_predict_router_signal_is_valid_literal(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert result["router_signal"] in ("HIGH", "DIM", "STALE")

    def test_predict_state_is_bytes(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.1] * 8, action=[0.0] * 3, state_bytes=None)
        assert isinstance(result["state"], bytes)
        assert len(result["state"]) > 0

    def test_state_roundtrip_preserves_tensor_values(self):
        """Serialise -> bytes -> deserialise must recover exact tensor values."""
        from bvh_rssm.serving.predictor import Predictor
        import torch
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[1.0] * 8, action=[1.0] * 3, state_bytes=None)
        state_bytes = result["state"]
        state = p._deserialise_state(state_bytes)
        re_bytes = p._serialise_state(state)
        state2 = p._deserialise_state(re_bytes)
        assert torch.allclose(state.h, state2.h)
        assert torch.allclose(state.z, state2.z)

    def test_predict_with_prior_state(self):
        """Second predict call using state from first must not crash."""
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        r1 = p.predict(obs=[0.5] * 8, action=[0.1] * 3, state_bytes=None)
        r2 = p.predict(obs=[0.5] * 8, action=[0.1] * 3, state_bytes=r1["state"])
        assert r2["tau"] >= 0.0

    def test_refresh_returns_required_keys(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.refresh(obs_batch=[[0.1] * 8, [0.2] * 8])
        assert "new_tau" in result
        assert "retrain_needed" in result
        assert "state" in result

    def test_refresh_retrain_needed_is_bool(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.refresh(obs_batch=[[0.0] * 8])
        assert isinstance(result["retrain_needed"], bool)

    def test_refresh_state_is_bytes(self):
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.refresh(obs_batch=[[0.0] * 8, [1.0] * 8])
        assert isinstance(result["state"], bytes)

    def test_no_grad_context_predict(self):
        """predict() must not accumulate gradients."""
        import torch
        from bvh_rssm.serving.predictor import Predictor
        p = Predictor.from_scratch(fast_mode=True)
        result = p.predict(obs=[0.0] * 8, action=[0.0] * 3, state_bytes=None)
        state = p._deserialise_state(result["state"])
        assert not state.h.requires_grad
        assert not state.z.requires_grad


class TestServer:
    @pytest.fixture
    def client(self):
        """TestClient wrapping the FastAPI app with a fast_mode predictor."""
        from bvh_rssm.serving.server import create_app
        from bvh_rssm.serving.predictor import Predictor
        predictor = Predictor.from_scratch(fast_mode=True)
        app = create_app(predictor)
        from starlette.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_predict_returns_200(self, client):
        resp = client.post(
            "/predict",
            json={"obs": [0.1] * 8, "action": [0.0] * 3, "state": None},
        )
        assert resp.status_code == 200

    def test_predict_response_shape(self, client):
        resp = client.post(
            "/predict",
            json={"obs": [0.1] * 8, "action": [0.0] * 3, "state": None},
        )
        body = resp.json()
        assert "tau" in body
        assert "survival_curve" in body
        assert len(body["survival_curve"]) == 16
        assert "router_signal" in body
        assert body["router_signal"] in ("HIGH", "DIM", "STALE")
        assert "lambda_intervals" in body
        assert len(body["lambda_intervals"]) == 16
        assert "state" in body  # base64 string

    def test_predict_chained_with_state(self, client):
        """Second request carries state from first — must not crash."""
        r1 = client.post(
            "/predict",
            json={"obs": [0.5] * 8, "action": [0.1] * 3, "state": None},
        )
        assert r1.status_code == 200
        state_b64: str = r1.json()["state"]
        # Pydantic v2 encodes bytes as base64 — pass back verbatim as string
        r2 = client.post(
            "/predict",
            json={"obs": [0.6] * 8, "action": [0.1] * 3, "state": state_b64},
        )
        assert r2.status_code == 200
        assert r2.json()["tau"] >= 0.0

    def test_predict_bad_request_422(self, client):
        """Malformed request must return 422 Unprocessable Entity."""
        resp = client.post(
            "/predict",
            json={"obs": "not_a_list", "action": [0.0], "state": None},
        )
        assert resp.status_code == 422

    def test_refresh_returns_200(self, client):
        resp = client.post(
            "/refresh",
            json={"obs_batch": [[0.1] * 8, [0.2] * 8]},
        )
        assert resp.status_code == 200

    def test_refresh_response_shape(self, client):
        resp = client.post(
            "/refresh",
            json={"obs_batch": [[0.3] * 8]},
        )
        body = resp.json()
        assert "new_tau" in body
        assert "retrain_needed" in body
        assert isinstance(body["retrain_needed"], bool)
        assert "state" in body

    def test_refresh_empty_batch_422(self, client):
        """Empty obs_batch must be rejected with 422."""
        resp = client.post(
            "/refresh",
            json={"obs_batch": []},
        )
        assert resp.status_code == 422

    def test_health_check(self, client):
        """GET / returns 200 with status ok."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
