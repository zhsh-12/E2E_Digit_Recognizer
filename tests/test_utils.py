"""Tests for common/utils.py"""

import numpy as np
import pytest
from common.utils import cal_confidence, generate_csv_content
from api.schemas import ExportCsvRequest


# ===================== cal_confidence =====================

class TestCalConfidence:
    def test_confidence_highest_class(self, sample_logits):
        """Confidence should be the softmax probability of the max logit."""
        conf = cal_confidence(sample_logits)
        # logits[0,2] = 2.0 is the highest → softmax probability should be ~0.43
        assert 0.3 < conf < 0.5
        assert isinstance(conf, float)

    def test_confidence_uniform(self, uniform_logits):
        """When all logits are equal, confidence should be 0.1 (uniform)."""
        conf = cal_confidence(uniform_logits)
        assert conf == pytest.approx(0.1, abs=1e-6)

    def test_confidence_extreme_values(self):
        """One logit is very large → confidence close to 1.0."""
        logits = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                          dtype=np.float32)
        conf = cal_confidence(logits)
        assert conf > 0.9999


# ===================== generate_csv_content =====================

class TestGenerateCsvContent:
    def test_single_mode(self):
        """Single prediction CSV should contain header and one data row."""
        data = ExportCsvRequest(
            filename="test_img.png",
            predicted_label=5,
            confidence=0.95,
        )
        content, filename = generate_csv_content(data)

        assert "test_img" in filename
        assert filename.endswith(".csv")
        assert "置信度" in content
        assert "5" in content
        assert "0.95" in content

    def test_batch_mode(self):
        """Batch prediction CSV should contain multiple data rows."""
        from api.schemas import BatchResultItem
        data = ExportCsvRequest(
            batch_id="batch_test_001",
            results=[
                BatchResultItem(filename="a.png", predicted_label=1, confidence=0.9),
                BatchResultItem(filename="b.png", predicted_label=2, confidence=0.8),
                BatchResultItem(filename="c.png", predicted_label=3, confidence=0.7),
            ],
        )
        content, filename = generate_csv_content(data)

        assert "batch_test_001" in filename
        assert filename.endswith(".csv")
        assert "置信度" in content
        # Each row should appear
        assert "a.png" in content
        assert "b.png" in content
        assert "c.png" in content

    def test_batch_mode_empty_results(self):
        """When results is an empty list, it should fall back to single mode."""
        data = ExportCsvRequest(
            batch_id="batch_empty",
            results=[],
        )
        content, filename = generate_csv_content(data)
        # Falls back to single mode with no filename → "unknown"
        assert "unknown" in filename
