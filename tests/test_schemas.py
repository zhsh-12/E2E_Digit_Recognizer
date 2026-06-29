"""Tests for api/schemas.py — Pydantic model validation."""
import pytest
from pydantic import ValidationError

from api.schemas import (
    SinglePredictionSave,
    BatchResultItem,
    BatchPredictionSave,
    ExportCsvRequest,
)


class TestSinglePredictionSave:
    def test_valid_minimal(self):
        s = SinglePredictionSave(predicted_label=3)
        assert s.predicted_label == 3
        assert s.true_label is None
        assert s.filename is None
        assert s.session_id is None

    def test_valid_full(self):
        s = SinglePredictionSave(
            predicted_label=7,
            true_label=7,
            filename="test.png",
            session_id="sess-001",
        )
        assert s.predicted_label == 7
        assert s.true_label == 7

    def test_negative_label_accepted(self):
        """Pydantic int field accepts negative values; no range validation."""
        s = SinglePredictionSave(predicted_label=-1)
        assert s.predicted_label == -1

    def test_invalid_missing_required(self):
        with pytest.raises(ValidationError):
            SinglePredictionSave()


class TestBatchResultItem:
    def test_valid(self):
        item = BatchResultItem(filename="a.png", predicted_label=5)
        assert item.predicted_label == 5

    def test_with_true_label(self):
        item = BatchResultItem(filename="a.png", predicted_label=5, true_label=5)
        assert item.true_label == 5

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            BatchResultItem(filename="a.png")


class TestBatchPredictionSave:
    def test_valid(self):
        data = BatchPredictionSave(
            batch_id="b1",
            results=[BatchResultItem(filename="a.png", predicted_label=1)],
        )
        assert data.batch_id == "b1"
        assert len(data.results) == 1

    def test_with_accuracy(self):
        data = BatchPredictionSave(
            batch_id="b1",
            results=[BatchResultItem(filename="a.png", predicted_label=1)],
            batch_accuracy=85.5,
        )
        assert data.batch_accuracy == 85.5


class TestExportCsvRequest:
    def test_single_mode(self):
        data = ExportCsvRequest(filename="test.png", predicted_label=3)
        assert data.filename == "test.png"
        assert data.results is None or len(data.results) == 0

    def test_batch_mode(self):
        data = ExportCsvRequest(
            batch_id="b1",
            results=[BatchResultItem(filename="a.png", predicted_label=1)],
        )
        assert data.batch_id == "b1"
