import numpy as np
import time
from typing import List, Tuple, Optional
from .database import DatabaseManager
from .vector_engine import VectorEngine
from .models import create_memory_record, MemoryRecord
from .decay import calculate_static_strength

class MemorySystem:
    """The high-level API for the memory system."""

    def __init__(self):
        self.db = DatabaseManager()
        self.vectors = VectorEngine()
        self.db.initialize_db()

    def encode(self, text, store="episodic", salience=None):
        record = create_memory_record(text, store=store, initial_salience=salience)
        vector = self.vectors.embed(text)
        self.db.add_memory(record, vector)
        return record.id

    def raw_encode(self, text):
        """
        Pure RAG Encoding:
        Stores text and vector without any cognitive metadata.
        """
        record = create_memory_record(text, store="raw")
        vector = self.vectors.embed(text)
        self.db.add_memory(record, vector)
        return record.id

    def simple_retrieve(self, query_text, top_k=3):
        """
        Pure RAG Retrieval:
        Just cosine similarity, no decay or weighting.
        """
        query_vec = self.vectors.embed(query_text)
        all_vectors = self.db.get_all_vectors()

        if not all_vectors:
            return []

        ids = [v[0] for v in all_vectors]
        vecs = np.stack([v[1] for v in all_vectors]).astype("float32")
        scores = np.dot(vecs, query_vec)
        
        top_indices = np.argsort(scores)[::-1][:top_k]

        top_results = []
        for idx in top_indices:
            mem_id = ids[idx]
            score = float(scores[idx])
            record = self.db.get_memory_by_id(mem_id)
            if record:
                top_results.append((record, score))

        return top_results

    def retrieve_by_id(self, memory_id):
        return self.db.get_memory_by_id(memory_id)

    def get_strength(self, memory_id, current_time=None, lambda_fixed=0.05):
        """Returns the current R(t) strength of a memory."""
        if current_time is None:
            current_time = time.time()
        record = self.db.get_memory_by_id(memory_id)
        if not record:
            return 0.0
        from .decay import calculate_static_strength
        return calculate_static_strength(record, current_time, lambda_fixed)

    def simulate_access(self, memory_id):
        record = self.db.get_memory_by_id(memory_id)
        if not record:
            return

        self.db.add_reinforcement_event(memory_id, "retrieval", record.current_strength)
        updates = {
            "reinforcement_count": record.reinforcement_count + 1,
            "last_accessed": time.time(),
        }
        self.db.update_memory(memory_id, updates)
        print(f"Reinforced memory {memory_id}. Count: {updates['reinforcement_count']}")

    def retrieve(self, query_text, top_k=3, current_time=None, lambda_fixed=0.05):
        """
        Strength-Weighted Retrieval:
        Score = CosineSimilarity(query, memory) * R(t)
        """
        if current_time is None:
            current_time = time.time()

        query_vec = self.vectors.embed(query_text)
        all_vectors = self.db.get_all_vectors()

        if not all_vectors:
            return []

        ids = [v[0] for v in all_vectors]
        vecs = np.stack([v[1] for v in all_vectors]).astype("float32")
        similarities = np.dot(vecs, query_vec)

        weighted_scores = []
        for i in range(len(ids)):
            record = self.db.get_memory_by_id(ids[i])
            if record:
                rt = calculate_static_strength(record, current_time, lambda_fixed)
                final_score = similarities[i] * rt
                weighted_scores.append((record, final_score))

        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        return weighted_scores[:top_k]
