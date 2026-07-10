import sqlite3
import time
from typing import Optional, List, Dict, Any
import numpy as np
from .config import CONFIG
from .models import MemoryRecord, map_row_to_record

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG["system"]["db_path"]

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def initialize_db(self):
        memories_schema = """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            store TEXT,
            created_at REAL,
            last_accessed REAL,
            initial_salience REAL,
            current_strength REAL,
            lambda_eff REAL,
            base_lambda REAL,
            importance REAL,
            locked BOOLEAN,
            reinforcement_count INTEGER,
            consolidated BOOLEAN,
            explained_last TEXT,
            vector BLOB
        );
        """
        events_schema = """
        CREATE TABLE IF NOT EXISTS reinforcement_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT,
            timestamp REAL,
            event_type TEXT,
            strength_at_time REAL,
            FOREIGN KEY(memory_id) REFERENCES memories(id)
        );
        """
        with self._get_connection() as conn:
            conn.execute(memories_schema)
            conn.execute(events_schema)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_mem_id ON reinforcement_events(memory_id);")
            conn.commit()
        print("Database initialized with normalized schema.")

    def add_memory(self, record: MemoryRecord, vector: np.ndarray):
        query = """
        INSERT INTO memories (
            id, content, store, created_at, last_accessed, initial_salience,
            current_strength, lambda_eff, base_lambda, importance, locked,
            reinforcement_count, consolidated, explained_last, vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    record.id, record.content, record.store, record.created_at,
                    record.last_accessed, record.initial_salience, record.current_strength,
                    record.lambda_eff, record.base_lambda, record.importance,
                    record.locked, record.reinforcement_count, record.consolidated,
                    record.explained_last, vector.tobytes(),
                ),
            )
            conn.commit()

    def get_memory_by_id(self, memory_id):
        query = "SELECT * FROM memories WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (memory_id,))
            row = cursor.fetchone()
            return map_row_to_record(row) if row else None

    def get_reinforcement_history(self, memory_id):
        query = "SELECT timestamp, event_type, strength_at_time FROM reinforcement_events WHERE memory_id = ? ORDER BY timestamp ASC"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (memory_id,))
            return [
                {"timestamp": r[0], "event": r[1], "strength": r[2]}
                for r in cursor.fetchall()
            ]

    def add_reinforcement_event(self, memory_id, event_type, strength):
        query = "INSERT INTO reinforcement_events (memory_id, timestamp, event_type, strength_at_time) VALUES (?, ?, ?, ?)"
        with self._get_connection() as conn:
            conn.execute(query, (memory_id, time.time(), event_type, strength))
            conn.commit()

    def update_memory(self, memory_id, updates):
        if not updates:
            return
        keys = list(updates.keys())
        values = list(updates.values())
        set_clause = ", ".join([f"{k} = ?" for k in keys])
        query = f"UPDATE memories SET {set_clause} WHERE id = ?"
        with self._get_connection() as conn:
            conn.execute(query, tuple(values) + (memory_id,))
            conn.commit()

    def get_all_vectors(self):
        query = "SELECT id, vector FROM memories"
        with self._get_connection() as conn:
            cursor = conn.execute(query)
            return [
                (row[0], np.frombuffer(row[1], dtype="float32"))
                for row in cursor.fetchall()
            ]
