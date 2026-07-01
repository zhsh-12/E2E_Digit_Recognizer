"""Performance benchmarks for the FastAPI prediction endpoints.

Uses pytest-benchmark to measure response times.

Run with:
    pytest tests/test_performance.py --benchmark-only
    pytest tests/test_performance.py --benchmark-histogram=benchmark_results
"""

import io
import pytest
from PIL import Image


class TestPredictPerformance:
    """Benchmark single-image prediction."""

    @pytest.fixture(autouse=True)
    def _setup(self, client, sample_image_rgb):
        self.client = client
        self.sample_image_rgb = sample_image_rgb

    def test_single_predict(self, benchmark):
        """Measure single predict endpoint latency."""
        files = {"file": ("test.png", self.sample_image_rgb, "image/png")}

        result = benchmark(self.client.post, "/predict", files=files)
        assert result.status_code == 200

    def test_batch_predict_2(self, benchmark):
        """Measure batch predict with 2 images."""
        files = [
            ("files", ("a.png", self.sample_image_rgb, "image/png")),
            ("files", ("b.png", self.sample_image_rgb, "image/png")),
        ]
        result = benchmark(self.client.post, "/batch_predict", files=files)
        assert result.status_code == 200

    def test_batch_predict_10(self, benchmark):
        """Measure batch predict with 10 images."""
        files = [
            ("files", (f"img_{i}.png", self.sample_image_rgb, "image/png"))
            for i in range(10)
        ]
        result = benchmark(self.client.post, "/batch_predict", files=files)
        assert result.status_code == 200

    def test_batch_predict_50(self, benchmark):
        """Measure batch predict with 50 images."""
        files = [
            ("files", (f"img_{i}.png", self.sample_image_rgb, "image/png"))
            for i in range(50)
        ]
        result = benchmark(self.client.post, "/batch_predict", files=files)
        assert result.status_code == 200


class TestExportPerformance:
    """Benchmark CSV export."""

    def test_export_single_csv(self, benchmark, client):
        data = {
            "filename": "test.png",
            "predicted_label": 5,
            "confidence": 0.95,
        }
        result = benchmark(client.post, "/api/export_csv", json=data)
        assert result.status_code == 200

    def test_export_batch_csv_100(self, benchmark, client):
        """Export CSV with 100 results."""
        data = {
            "batch_id": "batch_001",
            "results": [
                {"filename": f"img_{i}.png", "predicted_label": i % 10, "confidence": 0.9}
                for i in range(100)
            ],
        }
        result = benchmark(client.post, "/api/export_csv", json=data)
        assert result.status_code == 200


class TestHistoryPerformance:
    """Benchmark history retrieval."""

    def test_get_history_default(self, benchmark, client, patch_db_path):
        from common.database import init_db, save_single_prediction
        init_db()
        # Insert some test data
        for i in range(20):
            save_single_prediction(predicted_label=i % 10, confidence=0.9, filename=f"img_{i}.png")

        result = benchmark(client.get, "/api/prediction_history")
        assert result.status_code == 200

    def test_get_history_large_offset(self, benchmark, client, patch_db_path):
        from common.database import init_db, save_single_prediction
        init_db()
        for i in range(50):
            save_single_prediction(predicted_label=i % 10, confidence=0.9, filename=f"img_{i}.png")

        result = benchmark(client.get, "/api/prediction_history?limit=10&offset=40")
        assert result.status_code == 200
