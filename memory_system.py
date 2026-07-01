import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration Loader
# ==========================================


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# ==========================================
# Vector Engine (Embedding Logic)
# ==========================================


class VectorEngine:
    """Handles text-to-vector conversion and similarity searches."""

    def __init__(self):
        model_name = CONFIG["model"]["embedding_model_name"]
        self.model = SentenceTransformer(model_name)
        self.dimension = CONFIG["model"]["vector_dimension"]

    def embed(self, text: str) -> np.ndarray:
        """Converts text to a normalized float32 numpy array."""
        embedding = self.model.encode(text, convert_to_numpy=True).astype("float32")
        # Normalize for cosine similarity (dot product of normalized vectors)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two normalized vectors."""
        return float(np.dot(vec1, vec2))


# ==========================================
# Data Model Implementation
# ==========================================


@dataclass
class MemoryRecord:
    id: str
    content: str
    store: str
    created_at: float
    last_accessed: float
    initial_salience: float
    current_strength: float
    lambda_eff: float
    base_lambda: float
    importance: float
    locked: bool
    reinforcement_count: int
    consolidated: bool
    explained_last: Optional[str]
    vector: Optional[np.ndarray] = None

    @staticmethod
    def from_row(row: tuple) -> "MemoryRecord":
        """Converts a database row into a MemoryRecord object."""
        # row[14] is the vector stored as a BLOB
        vector_blob = row[14]
        vector = np.frombuffer(vector_blob, dtype="float32") if vector_blob else None

        return MemoryRecord(
            id=row[0],
            content=row[1],
            store=row[2],
            created_at=row[3],
            last_accessed=row[4],
            initial_salience=row[5],
            current_strength=row[6],
            lambda_eff=row[7],
            base_lambda=row[8],
            importance=row[9],
            locked=bool(row[10]),
            reinforcement_count=row[11],
            consolidated=bool(row[13]),
            explained_last=row[14]
            if not vector_blob
            else row[14],  # wait, row index fix below
            vector=vector,
        )


# Redoing from_row with explicit index mapping to avoid errors
def map_row_to_record(row: tuple) -> MemoryRecord:
    # memories table columns: 0:id, 1:content, 2:store, 3:created_at, 4:last_accessed,
    # 5:initial_salience, 6:current_strength, 7:lambda_eff, 8:base_lambda, 9:importance,
    # 10:locked, 11:reinforcement_count, 12:consolidated, 13:explained_last, 14:vector
    vector_blob = row[14]
    vector = np.frombuffer(vector_blob, dtype="float32") if vector_blob else None
    return MemoryRecord(
        id=row[0],
        content=row[1],
        store=row[2],
        created_at=row[3],
        last_accessed=row[4],
        initial_salience=row[5],
        current_strength=row[6],
        lambda_eff=row[7],
        base_lambda=row[8],
        importance=row[9],
        locked=bool(row[10]),
        reinforcement_count=row[11],
        consolidated=bool(row[12]),
        explained_last=row[13],
        vector=vector,
    )


def create_memory_record(
    content: str, store: str = "episodic", initial_salience: float = None
) -> MemoryRecord:
    """Factory function to initialize a new memory using YAML config defaults."""
    now = time.time()
    if initial_salience is None:
        initial_salience = CONFIG["system"]["default_salience"]

    store_cfg = CONFIG["stores"].get(store, CONFIG["stores"]["episodic"])

    return MemoryRecord(
        id=str(uuid.uuid4()),
        content=content,
        store=store,
        created_at=now,
        last_accessed=now,
        initial_salience=initial_salience,
        current_strength=initial_salience,
        lambda_eff=store_cfg["lambda_eff"],
        base_lambda=store_cfg["base_lambda"],
        importance=initial_salience,
        locked=False,
        reinforcement_count=0,
        consolidated=False,
        explained_last="Initial encoding",
    )


# ==========================================
# Database Manager Implementation
# ==========================================


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or CONFIG["system"]["db_path"]

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def initialize_db(self):
        """Initializes the database with normalized tables."""
        # 1. Main memories table (WITHOUT history string, WITH vector BLOB)
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
        # 2. Normalized reinforcement events table
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_mem_id ON reinforcement_events(memory_id);"
            )
            conn.commit()
        print("Database initialized with normalized schema.")

    def add_memory(self, record: MemoryRecord, vector: np.ndarray):
        """Adds a new memory record and its vector."""
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
                    record.id,
                    record.content,
                    record.store,
                    record.created_at,
                    record.last_accessed,
                    record.initial_salience,
                    record.current_strength,
                    record.lambda_eff,
                    record.base_lambda,
                    record.importance,
                    record.locked,
                    record.reinforcement_count,
                    record.consolidated,
                    record.explained_last,
                    vector.tobytes(),
                ),
            )
            conn.commit()

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        query = "SELECT * FROM memories WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (memory_id,))
            row = cursor.fetchone()
            return map_row_to_record(row) if row else None

    def get_reinforcement_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieves history from the normalized events table."""
        query = "SELECT timestamp, event_type, strength_at_time FROM reinforcement_events WHERE memory_id = ? ORDER BY timestamp ASC"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (memory_id,))
            return [
                {"timestamp": r[0], "event": r[1], "strength": r[2]}
                for r in cursor.fetchall()
            ]

    def add_reinforcement_event(self, memory_id: str, event_type: str, strength: float):
        """Logs a reinforcement event in the normalized table."""
        query = "INSERT INTO reinforcement_events (memory_id, timestamp, event_type, strength_at_time) VALUES (?, ?, ?, ?)"
        with self._get_connection() as conn:
            conn.execute(query, (memory_id, time.time(), event_type, strength))
            conn.commit()

    def update_memory(self, memory_id: str, updates: Dict[str, Any]):
        if not updates:
            return
        keys = list(updates.keys())
        values = list(updates.values())
        set_clause = ", ".join([f"{k} = ?" for k in keys])
        query = f"UPDATE memories SET {set_clause} WHERE id = ?"
        with self._get_connection() as conn:
            conn.execute(query, (*values, memory_id))
            conn.commit()

    def get_all_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """Returns all memory IDs and their vectors for similarity search."""
        query = "SELECT id, vector FROM memories"
        with self._get_connection() as conn:
            cursor = conn.execute(query)
            return [
                (row[0], np.frombuffer(row[1], dtype="float32"))
                for row in cursor.fetchall()
            ]


# ==========================================
# Unified Memory System (The Orchestrator)
# ==========================================


class MemorySystem:
    """The high-level API for the memory system."""

    def __init__(self):
        self.db = DatabaseManager()
        self.vectors = VectorEngine()
        self.db.initialize_db()

    def encode(self, text: str, store: str = "episodic", salience: float = None):
        """Encodes text into a record, generates vector, and stores in DB."""
        record = create_memory_record(text, store=store, initial_salience=salience)
        vector = self.vectors.embed(text)
        self.db.add_memory(record, vector)
        return record.id

    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.db.get_memory_by_id(memory_id)

    def simulate_access(self, memory_id: str):
        """Simulates memory retrieval and reinforces it."""
        record = self.db.get_memory_by_id(memory_id)
        if not record:
            return

        # 1. Log event in normalized table
        self.db.add_reinforcement_event(memory_id, "retrieval", record.current_strength)

        # 2. Update memory metrics
        updates = {
            "reinforcement_count": record.reinforcement_count + 1,
            "last_accessed": time.time(),
        }
        self.db.update_memory(memory_id, updates)
        print(f"Reinforced memory {memory_id}. Count: {updates['reinforcement_count']}")

    def semantic_search(
        self, query_text: str, top_k: int = 3
    ) -> List[Tuple[MemoryRecord, float]]:
        """Finds the most similar memories using vector cosine similarity."""
        query_vec = self.vectors.embed(query_text)
        all_vectors = self.db.get_all_vectors()

        results = []
        for mem_id, mem_vec in all_vectors:
            score = self.vectors.cosine_similarity(query_vec, mem_vec)
            results.append((mem_id, score))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for mem_id, score in results[:top_k]:
            record = self.db.get_memory_by_id(mem_id)
            top_results.append((record, score))

        return top_results


# ==========================================
# Testing Pipeline
# ==========================================

if __name__ == "__main__":
    sys = MemorySystem()

    # 1. Test Encoding
    print("\n--- Encoding Memories ---")
    m1 = sys.encode("The capital of France is Paris", store="semantic")
    m2 = sys.encode("I had a great coffee at the park this morning", store="episodic")
    m3 = sys.encode("The Python language is widely used for AI", store="semantic")
    print(f"Stored memories: {m1}, {m2}, {m3}")

    # 2. Test Vector Semantic Search
    print("\n--- Testing Semantic Search ---")
    query = "Tell me about French cities"
    results = sys.semantic_search(query)
    for rec, score in results:
        print(f"Score: {score:.4f} | Content: {rec.content}")

    # 3. Test Normalized History
    print("\n--- Testing Reinforcement ---")
    sys.simulate_access(m1)
    sys.simulate_access(m1)

    history = sys.db.get_reinforcement_history(m1)
    print(f"Memory {m1} has {len(history)} access events in the normalized table.")

    # 4. Verify Config Defaults
    rec = sys.retrieve_by_id(m1)
    print(f"Semantic Base Lambda: {rec.base_lambda} (from config.yaml)")
