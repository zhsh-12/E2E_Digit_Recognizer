"""Tests for api/csv_export.py — CSV generation logic (pure function, no I/O)."""
import csv
import io

from api.schemas import ExportCsvRequest, BatchResultItem
from api.csv_export import generate_csv_content


class TestGenerateCsvContentSingle:
    """Single prediction CSV export scenarios."""

    def test_single_with_true_label(self):
        """Single prediction with true label → includes '结果' column with ✓/✗."""
        data = ExportCsvRequest(
            filename="img_1[2].png",
            predicted_label=2,
            true_label=2,
        )
        content, fname = generate_csv_content(data)
        assert "predict_img_1" in fname
        assert fname.endswith(".csv")
        # Parse CSV and verify
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row
        assert "预测值" in rows[0]
        assert "真实值" in rows[0]
        assert "结果" in rows[0]
        assert rows[1][2] == "2"  # predicted
        assert rows[1][3] == "2"  # true
        assert rows[1][4] == "✓"  # correct

    def test_single_wrong_prediction(self):
        """Prediction differs from true_label → ✗."""
        data = ExportCsvRequest(
            filename="img_1[2].png",
            predicted_label=5,
            true_label=2,
        )
        content, _ = generate_csv_content(data)
        rows = list(csv.reader(io.StringIO(content)))
        assert rows[1][4] == "✗"

    def test_single_without_true_label(self):
        """No true label → minimal CSV output (no 真实值/结果 column)."""
        data = ExportCsvRequest(
            filename="img_unknown.png",
            predicted_label=3,
            true_label=None,
        )
        content, fname = generate_csv_content(data)
        assert fname.startswith("predict_img_unknown")
        rows = list(csv.reader(io.StringIO(content)))
        assert "真实值" not in rows[0]
        # Only 5 columns: ID, 文件名, 预测值, 批次, 时间
        assert len(rows[1]) == 5


class TestGenerateCsvContentBatch:
    """Batch prediction CSV export scenarios."""

    def test_batch_with_labels(self):
        """Batch with true labels → accuracy column and ✓/✗ per row."""
        data = ExportCsvRequest(
            batch_id="batch_001",
            results=[
                BatchResultItem(filename="a.png", predicted_label=1, true_label=1),
                BatchResultItem(filename="b.png", predicted_label=2, true_label=9),
            ],
            batch_accuracy=50.0,
        )
        content, fname = generate_csv_content(data)
        assert fname.startswith("batch_batch_001")
        rows = list(csv.reader(io.StringIO(content)))
        assert len(rows) == 3  # header + 2 data rows
        assert rows[1][2] == "1"  # a.png predicted
        assert rows[1][4] == "✓"  # a.png correct
        assert rows[2][4] == "✗"  # b.png wrong

    def test_batch_without_labels(self):
        """Batch without true labels → minimal columns."""
        data = ExportCsvRequest(
            batch_id="batch_002",
            results=[
                BatchResultItem(filename="x.png", predicted_label=7),
                BatchResultItem(filename="y.png", predicted_label=3),
            ],
        )
        content, fname = generate_csv_content(data)
        assert fname.startswith("batch_batch_002")
        rows = list(csv.reader(io.StringIO(content)))
        assert len(rows) == 3
        assert "真实值" not in rows[0]

    def test_batch_empty_results(self):
        """Empty results list → treated as single mode."""
        data = ExportCsvRequest(
            filename="single.png",
            predicted_label=0,
            results=[],
        )
        content, fname = generate_csv_content(data)
        assert fname.startswith("predict_single")
        rows = list(csv.reader(io.StringIO(content)))
        assert len(rows) == 2
