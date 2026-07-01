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
    def test_valid(self):
        data = SinglePredictionSave(predicted_label=5, confidence=0.95)
        assert data.predicted_label == 5
        assert data.confidence == 0.95
        assert data.filename is None
        assert data.batch_id is None

    def test_valid_with_optionals(self):
        data = SinglePredictionSave(
            predicted_label=3,
            confidence=0.88,
            filename="test.png",
            batch_id="batch_001",
        )
        assert data.filename == "test.png"
        assert data.batch_id == "batch_001"

    def test_missing_predicted_label(self):
        with pytest.raises(ValidationError):
            SinglePredictionSave(confidence=0.95)

    def test_missing_confidence(self):
        with pytest.raises(ValidationError):
            SinglePredictionSave(predicted_label=5)


class TestBatchResultItem:
    def test_valid(self):
        item = BatchResultItem(filename="a.png", predicted_label=1, confidence=0.9)
        assert item.filename == "a.png"
        assert item.predicted_label == 1
        assert item.confidence == 0.9

    def test_missing_confidence(self):
        with pytest.raises(ValidationError):
            BatchResultItem(filename="a.png", predicted_label=1)

    def test_missing_predicted_label(self):
        with pytest.raises(ValidationError):
            BatchResultItem(filename="a.png", confidence=0.9)


class TestBatchPredictionSave:
    def test_valid(self):
        data = BatchPredictionSave(
            batch_id="batch_001",
            results=[
                BatchResultItem(filename="a.png", predicted_label=1, confidence=0.9),
            ],
        )
        assert data.batch_id == "batch_001"
        assert len(data.results) == 1

    def test_empty_results_allowed(self):
        """Empty results list should be allowed by the schema."""
        data = BatchPredictionSave(batch_id="batch_empty", results=[])
        assert len(data.results) == 0

    def test_missing_batch_id(self):
        with pytest.raises(ValidationError):
            BatchPredictionSave(results=[])


class TestExportCsvRequest:
    def test_single_mode(self):
        data = ExportCsvRequest(
            filename="test.png",
            predicted_label=5,
            confidence=0.95,
        )
        assert data.filename == "test.png"
        assert data.predicted_label == 5
        assert data.confidence == 0.95
        assert data.batch_id is None
        assert data.results is None

    def test_batch_mode(self):
        data = ExportCsvRequest(
            batch_id="batch_001",
            results=[
                BatchResultItem(filename="a.png", predicted_label=1, confidence=0.9),
            ],
        )
        assert data.batch_id == "batch_001"
        assert len(data.results) == 1

    def test_all_fields_none(self):
        """All fields are optional in ExportCsvRequest."""
        data = ExportCsvRequest()
        assert data.filename is None
        assert data.predicted_label is None
        assert data.confidence is None
        assert data.batch_id is None
        assert data.results is None
