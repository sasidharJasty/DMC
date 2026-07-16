import numpy as np
from sentence_transformers import SentenceTransformer
from .config import CONFIG

class VectorEngine:
    """Handles text-to-vector conversion and similarity searches."""

    def __init__(self):
        model_name = CONFIG["model"]["embedding_model_name"]
        self.model = SentenceTransformer(model_name)
        self.dimension = CONFIG["model"]["vector_dimension"]

    def embed(self, text):
        """Converts text to a normalized float32 numpy array."""
        embedding = self.model.encode(text, convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two normalized vectors."""
        return float(np.dot(vec1, vec2))
