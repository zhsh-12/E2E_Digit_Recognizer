import sqlite3
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "digit_recognizer.db"


DB_VERSION = 1

def _get_user_version(cursor) -> int:
    """Get the database user version"""
    cursor.execute("PRAGMA user_version")
    return cursor.fetchone()[0]


def _set_user_version(cursor, version: int):
    """Set the database user version"""
    cursor.execute(f"PRAGMA user_version = {version}")


def init_db():
    """Initialize the predictions Table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    current_version = _get_user_version(cursor)

    if current_version == 0:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                filename TEXT,
                predicted_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
            )
        ''')
        _set_user_version(cursor, 1)
        current_version = 1
        print("Database initialized (version 1)")

    conn.commit()
    conn.close()


def save_single_prediction(predicted_label: int, confidence: float, filename=None, batch_id=None):
    """Save a single prediction result, return the new record id"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (batch_id, filename, predicted_label, confidence)
        VALUES (?, ?, ?, ?)
    ''', (batch_id, filename, predicted_label, confidence))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def save_batch_predictions(batch_id: str, results: list):
    """Save batch prediction results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for item in results:
        cursor.execute('''
            INSERT INTO predictions (batch_id, filename, predicted_label, confidence)
            VALUES (?, ?, ?, ?)
        ''', (batch_id, item.get('filename'), item['predicted_label'], item['confidence']))
    conn.commit()
    conn.close()


def get_history(limit=50, offset=0):
    """Query prediction history, ordered by time descending"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
