"""Tests for api/main.py — FastAPI endpoints. Uses httpx AsyncClient."""
import pytest
import httpx
import importlib

from httpx import ASGITransport
from tests.conftest import make_test_image_bytes


@pytest.fixture
def client(mock_model, tmp_path, monkeypatch):
    test_db = tmp_path / "test.db"
    monkeypatch.setattr("common.config.db_path", test_db)
    import api.main
    importlib.reload(api.main)
    transport = ASGITransport(app=api.main.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_predict_success(client):
    img_bytes = make_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = await client.post("/predict", files=files)
    assert resp.status_code == 200
    assert isinstance(resp.json()["prediction"], int)


@pytest.mark.asyncio
async def test_predict_no_file(client):
    resp = await client.post("/predict")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_filename(client):
    img_bytes = make_test_image_bytes()
    files = {"file": ("", img_bytes, "image/png")}
    resp = await client.post("/predict", files=files)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_with_corrupted_image(client):
    corrupted = b"not an image at all"
    files = {"file": ("bad.png", corrupted, "image/png")}
    resp = await client.post("/predict", files=files)
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_batch_predict_success(client):
    img_bytes = make_test_image_bytes()
    files = [
        ("files", ("a.png", img_bytes, "image/png")),
        ("files", ("b.png", img_bytes, "image/png")),
    ]
    resp = await client.post("/batch_predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_batch_predict_no_files(client):
    resp = await client.post("/batch_predict")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_too_many(client):
    from api.main import MAX_BATCH_SIZE
    img_bytes = make_test_image_bytes()
    files = [("files", (f"img_{i}.png", img_bytes, "image/png"))
             for i in range(MAX_BATCH_SIZE + 1)]
    resp = await client.post("/batch_predict", files=files)
    assert resp.status_code == 413
    assert "at most" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_batch_predict_with_corrupted_image(client):
    corrupted = b"garbage data"
    files = [
        ("files", ("good.png", make_test_image_bytes(), "image/png")),
        ("files", ("bad.png", corrupted, "image/png")),
    ]
    resp = await client.post("/batch_predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    good = [r for r in data["results"] if r["filename"] == "good.png"][0]
    bad = [r for r in data["results"] if r["filename"] == "bad.png"][0]
    assert good["prediction"] is not None
    assert good["error"] is None
    assert bad["prediction"] is None
    assert bad["error"] is not None


@pytest.mark.asyncio
async def test_save_prediction(client):
    resp = await client.post(
        "/api/save_prediction",
        json={"predicted_label": 5, "true_label": 5, "filename": "test.png"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] > 0


@pytest.mark.asyncio
async def test_save_batch_results(client):
    resp = await client.post(
        "/api/save_batch_results",
        json={
            "batch_id": "batch_test",
            "results": [
                {"filename": "a.png", "predicted_label": 1, "true_label": 1},
                {"filename": "b.png", "predicted_label": 2},
            ],
            "batch_accuracy": 50.0,
        },
    )
    assert resp.status_code == 200
    assert "saved successfully" in resp.json()["message"]


@pytest.mark.asyncio
async def test_prediction_history_returns_list(client):
    resp = await client.get("/api/prediction_history")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data
    assert isinstance(data["records"], list)
    assert "count" in data


@pytest.mark.asyncio
async def test_export_csv_single(client):
    resp = await client.post(
        "/api/export_csv",
        json={"filename": "test.png", "predicted_label": 5, "true_label": 5},
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_export_csv_batch(client):
    resp = await client.post(
        "/api/export_csv",
        json={
            "batch_id": "b1",
            "results": [
                {"filename": "a.png", "predicted_label": 1, "true_label": 1},
                {"filename": "b.png", "predicted_label": 2, "true_label": 9},
            ],
            "batch_accuracy": 50.0,
        },
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
