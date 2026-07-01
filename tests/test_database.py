"""Tests for common/database.py"""

import time
import pytest
from common.database import init_db, save_single_prediction, save_batch_predictions, get_history


class TestDatabase:
    """All database tests use a temporary database via patch_db_path fixture."""

    def test_init_db_creates_table(self, patch_db_path):
        """After init_db, the predictions table should exist."""
        import sqlite3
        from common.database import db_path
        init_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_db_version(self, patch_db_path):
        """After init_db, user_version should be 1."""
        import sqlite3
        from common.database import db_path
        init_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA user_version")
        version = cursor.fetchone()[0]
        assert version == 1
        conn.close()

    def test_save_single_prediction(self, patch_db_path):
        """Save a single prediction and verify it returns an id."""
        init_db()
        record_id = save_single_prediction(
            predicted_label=5,
            confidence=0.95,
            filename="test.png",
            batch_id=None,
        )
        assert record_id > 0

        # Verify the record
        records = get_history(limit=10)
        assert len(records) == 1
        assert records[0]["predicted_label"] == 5
        assert records[0]["confidence"] == 0.95
        assert records[0]["filename"] == "test.png"

    def test_save_single_prediction_with_batch(self, patch_db_path):
        """Save a single prediction with a batch_id."""
        init_db()
        save_single_prediction(
            predicted_label=3,
            confidence=0.88,
            filename="img.png",
            batch_id="batch_001",
        )
        records = get_history(limit=10)
        assert records[0]["batch_id"] == "batch_001"

    def test_save_batch_predictions(self, patch_db_path):
        """Save multiple predictions in batch."""
        init_db()
        results = [
            {"filename": "a.png", "predicted_label": 1, "confidence": 0.9},
            {"filename": "b.png", "predicted_label": 2, "confidence": 0.8},
            {"filename": "c.png", "predicted_label": 3, "confidence": 0.7},
        ]
        save_batch_predictions("batch_002", results)
        records = get_history(limit=10)
        assert len(records) == 3
        assert all(r["batch_id"] == "batch_002" for r in records)

    def test_get_history_empty(self, patch_db_path):
        """Querying history on an empty table returns []."""
        init_db()
        records = get_history(limit=10)
        assert records == []

    def test_get_history_ordering(self, patch_db_path):
        """Records should be ordered by created_at DESC (most recent first)."""
        init_db()
        save_single_prediction(predicted_label=1, confidence=0.5, filename="first.png")
        time.sleep(1.1)  # Ensure different second for ordering (SQLite second precision)
        save_single_prediction(predicted_label=2, confidence=0.6, filename="second.png")
        records = get_history(limit=10)
        assert records[0]["filename"] == "second.png"
        assert records[1]["filename"] == "first.png"

    def test_get_history_limit_offset(self, patch_db_path):
        """Limit and offset should work correctly."""
        init_db()
        for i in range(5):
            save_single_prediction(predicted_label=i, confidence=0.5, filename=f"{i}.png")
            time.sleep(1.1)  # Ensure different seconds for ordering
        # Get page 2 (offset=2, limit=2)
        records = get_history(limit=2, offset=2)
        assert len(records) == 2
        # Inserted: 0,1,2,3,4 → DESC: 4,3,2,1,0 → offset=2 → [2,1]
        assert records[0]["predicted_label"] == 2
        assert records[1]["predicted_label"] == 1
