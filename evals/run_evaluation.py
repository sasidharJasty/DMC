import json
import time
import numpy as np
from src.manager import MemorySystem

def mock_token_count(text):
    """Approximate token count (1 token approx 4 chars)."""
    return len(text) // 4 if text else 0

def mock_llm_judge(query, retrieved_memories, target_fact):
    """
    Simulates an LLM-as-a-judge.
    If the target fact is present in the retrieved context, the LLM 
    will likely answer correctly.
    """
    for rec, score in retrieved_memories:
        if target_fact.lower() in rec.content.lower() or rec.content.lower() in target_fact.lower():
            return True
    return False

def run_rigorous_evaluation(dataset_path="data/eval_dataset.json", eviction_threshold=0.1):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    sys = MemorySystem()
    step_duration = 3600 
    current_mock_time = time.time()
    
    metrics = {
        "total_queries": 0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "task_successes": 0,
        "total_tokens": 0,
        "encodings": 0,
        "core_facts": [],
        "distractor_facts": []
    }
    
    print("Running Rigorous Evaluation...")
    for item in dataset:
        current_mock_time += step_duration
        
        if item["type"] == "encode":
            mem_id = sys.encode(item["fact"], store=item.get("store", "episodic"), salience=item.get("salience"))
            metrics["encodings"] += 1
            if item.get("is_core", False):
                metrics["core_facts"].append(mem_id)
            else:
                metrics["distractor_facts"].append(mem_id)
            print(f"Step {item['step']}: Encoding {'Core' if item.get('is_core') else 'Distractor'} fact.")
            
        elif item["type"] == "query":
            metrics["total_queries"] += 1
            # Retrieval
            results = sys.retrieve(item["question"], top_k=3, current_time=current_mock_time)
            
            # 1. Retrieval Accuracy
            target = item["target_fact"]
            found_at = -1
            for i, (rec, score) in enumerate(results):
                if target.lower() in rec.content.lower() or rec.content.lower() in target.lower():
                    found_at = i + 1
                    break
            
            if found_at == 1: metrics["recall_at_1"] += 1
            if found_at != -1 and found_at <= 3: metrics["recall_at_3"] += 1
            
            # 2. Agent Performance (Task Success)
            if mock_llm_judge(item["question"], results, target):
                metrics["task_successes"] += 1
                
            # 3. Token Efficiency
            # Sum tokens of all retrieved memories (simulating the prompt context)
            tokens = sum(mock_token_count(rec.content) for rec, score in results)
            metrics["total_tokens"] += tokens
            
            print(f"Step {item['step']}: Query -> Recall@1: {'Yes' if found_at==1 else 'No'} | Success: {found_at != -1}")

    # Final Calculations
    total_q = metrics["total_queries"]
    
    # Survival Rate
    survivors = 0
    for mid in (metrics["core_facts"] + metrics["distractor_facts"]):
        strength = sys.get_strength(mid, current_mock_time)
        if strength > eviction_threshold:
            survivors += 1
    survival_rate = survivors / metrics["encodings"] if metrics["encodings"] > 0 else 0

    # Cognitive Metrics (Avg Strength)
    def get_avg_strength(ids):
        if not ids: return 0.0
        return sum(sys.get_strength(mid, current_mock_time) for mid in ids) / len(ids)

    avg_core = get_avg_strength(metrics["core_facts"])
    avg_dist = get_avg_strength(metrics["distractor_facts"])

    final_json = {
        "model_config": "Static_Baseline_v1",
        "total_queries_tested": total_q,
        "retrieval_metrics": {
            "recall_at_1": metrics["recall_at_1"] / total_q if total_q > 0 else 0,
            "recall_at_3": metrics["recall_at_3"] / total_q if total_q > 0 else 0,
            "task_success_rate": metrics["task_successes"] / total_q if total_q > 0 else 0
        },
        "efficiency_metrics": {
            "survival_rate": survival_rate,
            "avg_tokens_per_query": metrics["total_tokens"] / total_q if total_q > 0 else 0
        },
        "cognitive_metrics": {
            "avg_core_fact_strength": avg_core,
            "avg_distractor_fact_strength": avg_dist
        }
    }

    with open("results/evaluation_results.json", "w") as f:
        json.dump(final_json, f, indent=2)
        
    print("\n--- Final Rigorous Evaluation Report ---")
    print(json.dumps(final_json, indent=2))

if __name__ == "__main__":
    run_rigorous_evaluation()
