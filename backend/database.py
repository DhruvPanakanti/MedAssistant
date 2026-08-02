"""
Lightweight SQLite storage for prediction history across all conditions.
"""
import os
import sqlite3
import json
from datetime import datetime, timezone

# Anchored to this file's own directory, not the process's current working
# directory — otherwise running the app from a different folder (a
# shortcut, an IDE run config, a different terminal location) silently
# creates/reads a *different* predictions.db elsewhere, and history
# appears to vanish even though nothing was actually lost.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            condition_key TEXT NOT NULL,
            input_json TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL,
            risk_label TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(condition_key, input_data, result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, condition_key, input_json, prediction, probability, risk_label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            condition_key,
            json.dumps(input_data),
            result["prediction"],
            result["probability"],
            result["risk_label"],
        )
    )
    conn.commit()
    conn.close()


def get_history(limit=50, condition_key=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if condition_key:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE condition_key = ? ORDER BY id DESC LIMIT ?",
            (condition_key, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
