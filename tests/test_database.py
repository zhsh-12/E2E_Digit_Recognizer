"""Tests for common/database.py — SQLite database operations."""
import pytest
import sqlite3
from pathlib import Path
import importlib


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Set db_path to a temp file for each test using monkeypatch + reload."""
    test_db = tmp_path / "test.db"
    monkeypatch.setattr("common.config.db_path", test_db)
    # Reload database module so it picks up the patched db_path
    import common.database
    importlib.reload(common.database)
    yield test_db
    if test_db.exists():
        test_db.unlink()


import common.database as db

# Grab references from the reloaded module (safest after monkeypatch + reload)
init_cli_db = db.init_cli_db
init_frontend_db = db.init_frontend_db
save_single_prediction = db.save_single_prediction
save_batch_frontend = db.save_batch_frontend
get_frontend_history = db.get_frontend_history
save_batch_results = db.save_batch_results


# ================================================================
# CLI database tests
# ================================================================

class TestInitCliDb:
    def test_init_creates_table(self, tmp_db):
        """init_cli_db should create the 'predictions' table."""
        init_cli_db()
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_db):
        """Calling init_cli_db multiple times should not raise."""
        init_cli_db()
        init_cli_db()  # second call should be fine

    def test_table_has_expected_columns(self, tmp_db):
        """Check that the predictions table has all required columns."""
        init_cli_db()
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"id", "batch_id", "filename", "predicted_label", "true_label",
                     "batch_accuracy", "created_at"}
        assert expected.issubset(columns)
        conn.close()


class TestSaveBatchResults:
    def test_save_and_query(self, tmp_db):
        """save_batch_results inserts rows; verify via direct SQL query."""
        init_cli_db()
        results = [
            ("a.png", 2, 2),
            ("b.png", 5, 3),
        ]
        save_batch_results("batch_001", results, batch_accuracy=50.0)

        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, predicted_label, true_label FROM predictions WHERE batch_id=? ORDER BY id", ("batch_001",))
        rows = cursor.fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0] == ("a.png", 2, 2)
        assert rows[1] == ("b.png", 5, 3)

    def test_save_unlabeled(self, tmp_db):
        """Unlabeled mode: true_label can be None."""
        init_cli_db()
        results = [
            ("x.png", 7, None),
            ("y.png", 3, None),
        ]
        save_batch_results("batch_u", results, batch_accuracy=None)

        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT true_label, batch_accuracy FROM predictions WHERE batch_id=?", ("batch_u",))
        rows = cursor.fetchall()
        conn.close()
        assert rows[0][0] is None
        assert rows[0][1] is None


# ================================================================
# Frontend database tests
# ================================================================

class TestInitFrontendDb:
    def test_init_creates_table(self, tmp_db):
        init_frontend_db()
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frontend_predictions'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_table_has_expected_columns(self, tmp_db):
        init_frontend_db()
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(frontend_predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"id", "session_id", "filename", "predicted_label", "true_label",
                     "is_correct", "batch_id", "batch_accuracy", "source", "created_at"}
        assert expected.issubset(columns)
        conn.close()


class TestSaveSinglePrediction:
    def test_save_with_true_label(self, tmp_db):
        """Saving with true_label should set is_correct."""
        init_frontend_db()
        record_id = save_single_prediction(
            predicted_label=5, true_label=5, filename="test.png", session_id="s1",
        )
        assert record_id is not None and record_id > 0
        history = get_frontend_history(limit=10)
        assert len(history) == 1
        assert history[0]["predicted_label"] == 5
        assert history[0]["true_label"] == 5
        assert history[0]["is_correct"] == 1  # SQLite stores bool as 0/1

    def test_save_without_true_label(self, tmp_db):
        """Without true_label, is_correct should be None."""
        init_frontend_db()
        save_single_prediction(predicted_label=3, filename="unknown.png")
        history = get_frontend_history(limit=10)
        assert history[0]["true_label"] is None
        assert history[0]["is_correct"] is None


class TestSaveBatchFrontend:
    def test_save_batch(self, tmp_db):
        """Save multiple frontend predictions."""
        init_frontend_db()
        results = [
            {"filename": "a.png", "predicted_label": 1, "true_label": 1},
            {"filename": "b.png", "predicted_label": 2, "true_label": 9},
        ]
        save_batch_frontend("batch_f1", results, batch_accuracy=50.0, session_id="s1")
        history = get_frontend_history(limit=10)
        assert len(history) == 2

    def test_save_unlabeled_batch(self, tmp_db):
        """Batch without true labels is accepted."""
        init_frontend_db()
        results = [
            {"filename": "a.png", "predicted_label": 1},
            {"filename": "b.png", "predicted_label": 2},
        ]
        save_batch_frontend("batch_f2", results)
        history = get_frontend_history(limit=10)
        assert len(history) == 2


class TestGetFrontendHistory:
    def test_pagination(self, tmp_db):
        """Verify limit and offset work correctly."""
        init_frontend_db()
        for i in range(5):
            save_single_prediction(predicted_label=i, filename=f"img_{i}.png", session_id="s1")

        page1 = get_frontend_history(limit=2, offset=0)
        page2 = get_frontend_history(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # Different pages should have different record sets
        assert page1[0]["id"] != page2[0]["id"]

    def test_empty_history(self, tmp_db):
        """No records → empty list."""
        init_frontend_db()
        history = get_frontend_history()
        assert history == []
