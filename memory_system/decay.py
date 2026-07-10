import math
import time
from .models import MemoryRecord

def calculate_static_strength(record: MemoryRecord, current_time: float, lambda_fixed: float = 0.05) -> float:
    """
    Calculates the current strength of a memory using the static Ebbinghaus formula:
    R(t) = S0 * exp(-lambda * delta_t)
    
    Args:
        record: The memory record containing initial salience (S0).
        current_time: The current mock or real time.
        lambda_fixed: The fixed decay rate (per hour).
        
    Returns:
        The current strength R(t).
    """
    # Convert delta_t from seconds to hours
    delta_t_seconds = current_time - record.last_accessed
    delta_t_hours = max(0, delta_t_seconds) / 3600.0
        
    # S0 is initial_salience
    strength = record.initial_salience * math.exp(-lambda_fixed * delta_t_hours)
    return strength
