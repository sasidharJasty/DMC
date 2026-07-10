import uuid
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .config import CONFIG

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

def map_row_to_record(row):
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

def create_memory_record(content, store="episodic", initial_salience=None):
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
