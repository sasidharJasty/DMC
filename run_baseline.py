import json
import time
import numpy as np
from memory_system.manager import MemorySystem
from memory_system.decay import calculate_static_strength

def run_baseline_simulation(dataset_path="eval_dataset.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    sys = MemorySystem()
    
    # Mock clock: each step is 1 hour (3600 seconds)
    step_duration = 3600 
    current_mock_time = time.time()
    
    metrics = {
        "total_queries": 0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "encodings": 0
    }
    
    encoded_ids = []

    print("Starting Simulation Loop...")
    for item in dataset:
        # Advance mock clock
        current_mock_time += step_duration
        
        if item["type"] == "encode":
            mem_id = sys.encode(item["fact"], store=item.get("store", "episodic"), salience=item.get("salience"))
            encoded_ids.append(mem_id)
            metrics["encodings"] += 1
            print(f"Step {item['step']}: Encoded -> {item['fact']}")
            
        elif item["type"] == "query":
            metrics["total_queries"] += 1
            results = sys.retrieve(item["question"], top_k=3, current_time=current_mock_time)
            
            # Check if target_fact is in results
            target = item["target_fact"]
            found_at = -1
            for i, (rec, score) in enumerate(results):
                if target.lower() in rec.content.lower() or rec.content.lower() in target.lower():
                    found_at = i + 1
                    break
            
            if found_at == 1:
                metrics["recall_at_1"] += 1
            if found_at != -1 and found_at <= 3:
                metrics["recall_at_3"] += 1
            
            print(f"Step {item['step']}: Query '{item['question']}' | Target: {target} | Result: {'Found' if found_at != -1 else 'Not Found'} (at rank {found_at})")

    # Final Metrics Calculation
    recall_1 = metrics["recall_at_1"] / metrics["total_queries"] if metrics["total_queries"] > 0 else 0
    recall_3 = metrics["recall_at_3"] / metrics["total_queries"] if metrics["total_queries"] > 0 else 0
    
    # Survival Rate: R(t) > 0.1
    survivors = 0
    for mid in encoded_ids:
        rec = sys.retrieve_by_id(mid)
        if rec:
            strength = calculate_static_strength(rec, current_mock_time)
            if strength > 0.1:
                survivors += 1
    
    survival_rate = survivors / metrics["encodings"] if metrics["encodings"] > 0 else 0

    results_summary = {
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
        "survival_rate": survival_rate,
        "total_encodings": metrics["encodings"],
        "total_queries": metrics["total_queries"]
    }
    
    with open("baseline_results.json", "w") as f:
        json.dump(results_summary, f, indent=4)
        
    print("\n--- Simulation Results ---")
    print(f"Recall@1: {recall_1:.2%}")
    print(f"Recall@3: {recall_3:.2%}")
    print(f"Survival Rate: {survival_rate:.2%}")
    print("Results saved to baseline_results.json")

if __name__ == "__main__":
    run_baseline_simulation()
