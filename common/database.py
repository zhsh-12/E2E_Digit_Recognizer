import sqlite3
from common.config import db_path


# ============================================================
# CLI database (SQL Table name: predictions)
# ============================================================

# Current database schema version
# Version 1: Initial version, true_label / batch_accuracy have NOT NULL constraints
# Version 2: Removed NOT NULL constraints on true_label / batch_accuracy, supports unlabeled mode
# Version 3: created_at default changed to datetime('now','localtime'), fixed UTC time issue

CLI_DB_VERSION = 3

def _get_user_version(cursor) -> int:
    """Get the database user version"""
    cursor.execute("PRAGMA user_version")
    return cursor.fetchone()[0]


def _set_user_version(cursor, version: int):
    """Set the database user version"""
    cursor.execute(f"PRAGMA user_version = {version}")


def _migrate_cli_v1_to_v2(cursor):
    """Migration: version 1 → version 2, remove NOT NULL constraints"""
    cursor.executescript("""
        CREATE TABLE predictions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            true_label INTEGER,
            batch_accuracy REAL,
            created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
        );
        INSERT INTO predictions_new SELECT * FROM predictions;
        DROP TABLE predictions;
        ALTER TABLE predictions_new RENAME TO predictions;
    """)


def _migrate_cli_v2_to_v3(cursor):
    """Migration: version 2 → version 3, created_at default changed to local time"""
    cursor.executescript("""
        CREATE TABLE predictions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            true_label INTEGER,
            batch_accuracy REAL,
            created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
        );
        INSERT INTO predictions_new SELECT * FROM predictions;
        DROP TABLE predictions;
        ALTER TABLE predictions_new RENAME TO predictions;
    """)


def init_cli_db():
    """Initialize the CLI predictions Table, auto-detect version and run migrations"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    current_version = _get_user_version(cursor)

    if current_version == 0:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                batch_accuracy REAL NOT NULL,
                created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
            )
        ''')
        _set_user_version(cursor, 1)
        current_version = 1
        print("Database initialized (version 1)")

    if current_version == 1:
        _migrate_cli_v1_to_v2(cursor)
        _set_user_version(cursor, 2)
        current_version = 2
        print("Database migrated to version 2 (supports unlabeled mode)")

    if current_version == 2:
        _migrate_cli_v2_to_v3(cursor)
        _set_user_version(cursor, 3)
        current_version = 3
        print("Database migrated to version 3 (created_at uses local time)")

    conn.commit()
    conn.close()


def save_batch_results(batch_id: str, results: list, batch_accuracy: float):
    """
    Save batch prediction results (CLI scenario)
    results: list of (filename, predicted_label, true_label)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    data = [(batch_id, filename, pred, true, batch_accuracy) for filename, pred, true in results]
    cursor.executemany('''
        INSERT INTO predictions (batch_id, filename, predicted_label, true_label, batch_accuracy)
        VALUES (?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()
    print(f"Batch {batch_id}: {len(results)} results saved to database")


# ============================================================
# Frontend database (SQL Table name: frontend_predictions)
# ============================================================

# Version 1: Initial version, frontend_predictions table, created_at uses CURRENT_TIMESTAMP (UTC)
# Version 2: created_at default changed to datetime('now','localtime'), fixed UTC time issue

FRONTEND_DB_VERSION = 2

def _migrate_frontend_v1_to_v2(cursor):
    """Migration: version 1 → version 2, created_at changed to local time"""
    cursor.executescript("""
        CREATE TABLE frontend_predictions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            predicted_label INTEGER NOT NULL,
            true_label INTEGER,
            is_correct BOOLEAN,
            batch_id TEXT,
            batch_accuracy REAL,
            source TEXT DEFAULT 'frontend',
            created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
        );
        INSERT INTO frontend_predictions_new SELECT * FROM frontend_predictions;
        DROP TABLE frontend_predictions;
        ALTER TABLE frontend_predictions_new RENAME TO frontend_predictions;
    """)


def init_frontend_db():
    """Initialize the frontend_predictions Table, auto-detect version and run migrations"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    current_version = _get_user_version(cursor)

    if current_version == 0:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frontend_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filename TEXT,
                predicted_label INTEGER NOT NULL,
                true_label INTEGER,
                is_correct BOOLEAN,
                batch_id TEXT,
                batch_accuracy REAL,
                source TEXT DEFAULT 'frontend',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        _set_user_version(cursor, 1)
        current_version = 1
        print("Database initialized (version 1)")

    if current_version == 1:
        _migrate_frontend_v1_to_v2(cursor)
        _set_user_version(cursor, 2)
        current_version = 2
        print("Database migrated to version 2 (created_at uses local time)")

    conn.commit()
    conn.close()


def save_single_prediction(predicted_label: int, true_label=None, filename=None, session_id=None):
    """Save a single prediction result, return the new record id"""
    is_correct = None
    if true_label is not None:
        is_correct = (predicted_label == true_label)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO frontend_predictions (session_id, filename, predicted_label, true_label, is_correct)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, filename, predicted_label, true_label, is_correct))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def save_batch_frontend(batch_id: str, results: list, batch_accuracy=None, session_id=None):
    """Save frontend batch prediction results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for item in results:
        is_correct = None
        if item.get('true_label') is not None:
            is_correct = (item['predicted_label'] == item['true_label'])
        cursor.execute('''
            INSERT INTO frontend_predictions (session_id, filename, predicted_label, true_label, is_correct, batch_id, batch_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, item.get('filename'), item['predicted_label'], item.get('true_label'), is_correct, batch_id, batch_accuracy))
    conn.commit()
    conn.close()


def get_frontend_history(limit=50, offset=0):
    """Query frontend prediction history, ordered by time descending"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM frontend_predictions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
