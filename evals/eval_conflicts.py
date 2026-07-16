import json
import numpy as np
import sys
import os
# Ensure root directory is in path for imports
sys.path.append(os.getcwd())

from src.manager import MemorySystem

class ConflictEvaluator:
    """
    Evaluates how the system handles contradictory, overlapping, or 
    evolving information (Freshness vs. Obsolescence).
    """
    def __init__(self, dataset_path="data/conflict_dataset.json"):
        self.dataset_path = dataset_path

    def run(self):
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)

        results = []
        
        for scenario in dataset:
            category = scenario["category"]
            sequence = scenario["sequence"]
            
            # Fresh state for each scenario
            sys = MemorySystem()
            
            # Process the sequence
            for item in sequence:
                if item["type"] == "encode":
                    sys.encode(item["fact"], salience=item.get("salience"))
                elif item["type"] == "query":
                    question = item["question"]
                    target = item["target"]
                    conflict_pair = item.get("conflict_pair", [])
                    
                    # Retrieve top 3
                    retrieved = sys.simple_retrieve(question, top_k=3)
                    if not retrieved:
                        results.append({"category": category, "success": False, "outdated_hit": False})
                        continue
                    
                    top_rec, top_score = retrieved[0]
                    
                    # Check for Freshness/Consistency
                    success = target.lower() in top_rec.content.lower()
                    
                    # Check if we retrieved the 'outdated' part of the conflict pair
                    outdated_hit = False
                    if conflict_pair:
                        outdated_val = conflict_pair[0]
                        if outdated_val.lower() in top_rec.content.lower():
                            outdated_hit = True
                    
                    results.append({
                        "category": category,
                        "success": success,
                        "outdated_hit": outdated_hit,
                        "retrieved": top_rec.content,
                        "target": target
                    })

        # Aggregate Metrics
        total = len(results)
        freshness_rate = np.mean([1 if r["success"] else 0 for r in results]) if total > 0 else 0
        obsolescence_rate = np.mean([1 if r["outdated_hit"] else 0 for r in results]) if total > 0 else 0
        
        final_report = {
            "total_scenarios": total,
            "metrics": {
                "freshness_rate": freshness_rate,
                "obsolescence_rate": obsolescence_rate,
                "conflict_resolution_ratio": freshness_rate / (obsolescence_rate + 1e-6)
            }
        }

        with open("results/conflict_evaluation_results.json", "w") as f:
            json.dump(final_report, f, indent=2)

        print("\n" + "="*40)
        print("CONFLICT & FRESHNESS EVALUATION")
        print("="*40)
        print(f"Freshness Rate: {freshness_rate:.2%}")
        print(f"Obsolescence Rate: {obsolescence_rate:.2%}")
        print(f"Conflict Resolution Ratio: {final_report['metrics']['conflict_resolution_ratio']:.2f}")
        print("="*40)

if __name__ == "__main__":
    evaluator = ConflictEvaluator()
    evaluator.run()
