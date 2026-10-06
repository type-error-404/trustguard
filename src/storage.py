"""
storage.py
----------
Stores TrustGuard scan history.

SQLite is used by default.
PostgreSQL can be connected later without
changing the detection pipeline.
"""

import os
import sqlite3
from datetime import datetime


ROOT = os.path.join(
    os.path.dirname(__file__),
    ".."
)

DB_PATH = os.path.join(
    ROOT,
    "outputs",
    "trustguard.db"
)


def initialize_database():

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            modality TEXT,
            label TEXT,
            confidence REAL,
            risk_score REAL
        )
        """
    )

    conn.commit()

    conn.close()


def save_scan(
    modality,
    label,
    confidence,
    risk_score
):

    initialize_database()

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO scans
        (timestamp, modality, label,
         confidence, risk_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            modality,
            label,
            confidence,
            risk_score
        )
    )

    conn.commit()

    conn.close()