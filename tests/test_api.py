"""Integration tests for the FastAPI application (api/main.py).

Uses TestClient with a mocked ONNX session (see conftest.py).
"""

import io
import pytest
from PIL import Image
from api.schemas import BatchResultItem


# ===================== Health check =====================

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


# ===================== Single predict =====================

class TestSinglePredict:
    def test_predict_valid_image(self, client, sample_image_rgb):
        resp = client.post("/predict", files={"file": ("test.png", sample_image_rgb, "image/png")})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "inference_time_ms" in data
        assert isinstance(data["prediction"], int)
        assert 0 <= data["prediction"] <= 9
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_no_file(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422  # FastAPI validation error

    def test_predict_empty_filename(self, client, sample_image_rgb):
        resp = client.post("/predict", files={"file": ("", sample_image_rgb, "image/png")})
        # FastAPI treats empty filename as validation error (422)
        assert resp.status_code == 422

    def test_predict_invalid_file(self, client):
        """Upload a text file instead of an image."""
        resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert resp.status_code == 500  # PIL will fail to open


# ===================== Batch predict =====================

class TestBatchPredict:
    def test_batch_predict_valid(self, client, sample_image_rgb, sample_image_l):
        """Upload 2 valid images."""
        files = [
            ("files", ("a.png", sample_image_rgb, "image/png")),
            ("files", ("b.png", sample_image_l, "image/png")),
        ]
        resp = client.post("/batch_predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert "filename" in r
            assert "prediction" in r
            assert "confidence" in r
            assert 0 <= r["prediction"] <= 9

    def test_batch_predict_empty(self, client):
        resp = client.post("/batch_predict")
        assert resp.status_code == 422

    def test_batch_predict_mixed(self, client, sample_image_rgb):
        """One valid image, one invalid file."""
        files = [
            ("files", ("valid.png", sample_image_rgb, "image/png")),
            ("files", ("invalid.txt", b"not an image", "text/plain")),
        ]
        resp = client.post("/batch_predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        # Valid image should have prediction
        assert data["results"][0]["prediction"] is not None
        # Invalid should have error
        assert "error" in data["results"][1]


# ===================== Save prediction =====================

class TestSavePrediction:
    def test_save_single(self, client, patch_db_path):
        from common.database import init_db
        init_db()
        resp = client.post("/api/save_prediction", json={
            "predicted_label": 5,
            "confidence": 0.95,
            "filename": "test.png",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["id"] > 0
        assert data["message"] == "Saved successfully"

    def test_save_single_missing_confidence(self, client):
        resp = client.post("/api/save_prediction", json={
            "predicted_label": 5,
        })
        assert resp.status_code == 422

    def test_save_batch(self, client, patch_db_path):
        from common.database import init_db
        init_db()
        resp = client.post("/api/save_batch_results", json={
            "batch_id": "batch_test_001",
            "results": [
                {"filename": "a.png", "predicted_label": 1, "confidence": 0.9},
                {"filename": "b.png", "predicted_label": 2, "confidence": 0.8},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "batch_test_001" in data["message"]


# ===================== Prediction history =====================

class TestPredictionHistory:
    def test_get_history(self, client, patch_db_path):
        from common.database import init_db, save_single_prediction
        init_db()
        save_single_prediction(predicted_label=5, confidence=0.95, filename="test.png")
        resp = client.get("/api/prediction_history")
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert "count" in data
        assert len(data["records"]) >= 1

    def test_get_history_with_params(self, client, patch_db_path):
        from common.database import init_db
        init_db()
        resp = client.get("/api/prediction_history?limit=5&offset=0")
        assert resp.status_code == 200


# ===================== CSV export =====================

class TestCsvExport:
    def test_export_single_csv(self, client):
        resp = client.post("/api/export_csv", json={
            "filename": "test.png",
            "predicted_label": 5,
            "confidence": 0.95,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers["content-disposition"]
        content = resp.text
        assert "置信度" in content
        assert "5" in content
        assert "0.95" in content

    def test_export_batch_csv(self, client):
        resp = client.post("/api/export_csv", json={
            "batch_id": "batch_001",
            "results": [
                {"filename": "a.png", "predicted_label": 1, "confidence": 0.9},
                {"filename": "b.png", "predicted_label": 2, "confidence": 0.8},
            ],
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        content = resp.text
        assert "a.png" in content
        assert "b.png" in content
        assert "0.9" in content
        assert "0.8" in content
