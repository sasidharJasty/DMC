from src.manager import MemorySystem

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
