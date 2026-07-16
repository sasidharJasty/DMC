import json
import numpy as np
import sys
import os
# Ensure root directory is in path for imports
sys.path.append(os.getcwd())

from src.manager import MemorySystem
from src.vector_engine import VectorEngine

class PureRAGEvaluator:
    """
    Rigorous Baseline Evaluation using Pure RAG.
    Compares retrieval against Reference Memories, Anchors, and Answer vectors.
    """
    def __init__(self, mem_path="data/PerLTQA/perltmem_en_v2.json", qa_path="data/PerLTQA/perltqa_en_v2.json"):
        self.mem_path = mem_path
        self.qa_path = qa_path
        self.sys = MemorySystem()
        self.vec_engine = VectorEngine()

    def flatten_persona_mems(self, persona_data):
        """
        Converts complex PerLTMem structure into (content, ref_tag, category) pairs.
        """
        flattened = []

        # 1. Profile (Category: bio)
        profile = persona_data.get("profile", {})
        if isinstance(profile, dict):
            for key, value in profile.items():
                flattened.append((f"{key}: {value}", key, "bio"))

        # 2. Social (Category: social)
        social = persona_data.get("social_relationship", {})
        if isinstance(social, dict):
            for rel_id, data in social.items():
                if isinstance(data, dict):
                    desc = data.get("Description", "")
                    if desc: flattened.append((desc, rel_id, "social"))
                elif isinstance(data, list):
                    for item in data: flattened.append((str(item), rel_id, "social"))

        # 3. Events (Category: episodic)
        events = persona_data.get("events", {})
        if isinstance(events, dict):
            for event_id, data in events.items():
                if isinstance(data, dict):
                    content = data.get("content", "")
                    if content: flattened.append((content, event_id, "episodic"))

        return flattened

    def calculate_jaccard_similarity(self, text1, text2):
        """Measures the overlap of unique words between two strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2: return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def run(self):
        with open(self.mem_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        with open(self.qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        stats = {
            "total_queries": 0,
            "recall_at_1": 0,
            "recall_at_3": 0,
            "recall_at_10": 0,
            "avg_jaccard_sim": 0,
            "category_performance": {
                "bio": {"queries": 0, "hits": 0},
                "social": {"queries": 0, "hits": 0},
                "episodic": {"queries": 0, "hits": 0}
            },
            "anchor_hits": 0
        }

        for persona_dict in qa_data:
            for persona_name, qa_content in persona_dict.items():
                print(f"Evaluating Persona: {persona_name}")
                self.sys = MemorySystem()

                persona_mems = mem_data.get(persona_name, {})
                if not persona_mems: continue

                facts = self.flatten_persona_mems(persona_mems)
                id_to_ref = {}
                for content, ref_tag, cat in facts:
                    mid = self.sys.raw_encode(content)
                    id_to_ref[mid] = (ref_tag, cat)

                all_questions = []
                if "profile" in qa_content:
                    for q in qa_content["profile"]: all_questions.append((q, "bio"))
                if "social_relationship" in qa_content:
                    for rel_dict in qa_content["social_relationship"]:
                        for rel_list in rel_dict.values():
                            for q in rel_list: all_questions.append((q, "social"))
                if "events" in qa_content:
                    for event_dict in qa_content["events"]:
                        for event_list in event_dict.values():
                            for q in event_list: all_questions.append((q, "episodic"))

                for q_tuple, category in all_questions:
                    q_item = q_tuple
                    question = q_item["Question"]
                    answer = q_item["Answer"]
                    ref_mem = q_item.get("Reference Memory")
                    anchors = q_item.get("Memory Anchors", [])

                    stats["total_queries"] += 1
                    stats["category_performance"][category]["queries"] += 1

                    retrieved = self.sys.simple_retrieve(question, top_k=10)
                    if not retrieved: continue

                    top_rec, top_score = retrieved[0]

                    # Jaccard Similarity
                    j_sim = self.calculate_jaccard_similarity(top_rec.content, answer)
                    stats["avg_jaccard_sim"] += j_sim

                    # Recall check
                    found_at = -1
                    for i, (rec, score) in enumerate(retrieved):
                        actual_ref, _ = id_to_ref.get(rec.id, ("Unknown", ""))
                        if isinstance(ref_mem, list):
                            if actual_ref in ref_mem: found_at = i + 1; break
                        elif ref_mem and actual_ref == ref_mem:
                            found_at = i + 1; break

                    if found_at == 1: stats["recall_at_1"] += 1
                    if found_at != -1 and found_at <= 3: stats["recall_at_3"] += 1
                    if found_at != -1 and found_at <= 10: stats["recall_at_10"] += 1

                    if found_at != -1:
                        stats["category_performance"][category]["hits"] += 1

                    # Anchor Match
                    for anchor_dict in anchors:
                        for keyword in anchor_dict.keys():
                            if keyword.lower() in top_rec.content.lower():
                                stats["anchor_hits"] += 1
                                break

        tq = stats["total_queries"]
        final_report = {
            "total_queries": tq,
            "global_metrics": {
                "recall_at_1": stats["recall_at_1"] / tq if tq > 0 else 0,
                "recall_at_3": stats["recall_at_3"] / tq if tq > 0 else 0,
                "recall_at_10": stats["recall_at_10"] / tq if tq > 0 else 0,
                "avg_text_overlap_jaccard": stats["avg_jaccard_sim"] / tq if tq > 0 else 0,
                "anchor_hit_rate": stats["anchor_hits"] / tq if tq > 0 else 0,
            },
            "category_breakdown": {
                cat: (data["hits"] / data["queries"] if data["queries"] > 0 else 0)
                for cat, data in stats["category_performance"].items()
            }
        }

        with open("results/pure_rag_baseline_v2.json", "w") as f:
            json.dump(final_report, f, indent=2)

        print("\n" + "="*40)
        print("ENHANCED PURE RAG BASELINE RESULTS")
        print("="*40)
        print(f"Total Queries: {tq}")
        print(f"Recall@1: {final_report['global_metrics']['recall_at_1']:.2%}")
        print(f"Recall@3: {final_report['global_metrics']['recall_at_3']:.2%}")
        print(f"Recall@10: {final_report['global_metrics']['recall_at_10']:.2%}")
        print(f"Avg Text Overlap (Jaccard): {final_report['global_metrics']['avg_text_overlap_jaccard']:.4f}")
        print(f"Anchor Hit Rate: {final_report['global_metrics']['anchor_hit_rate']:.2%}")
        print("\nCategory Accuracy:")
        for cat, acc in final_report["category_breakdown"].items():
            print(f" - {cat}: {acc:.2%}")
        print("="*40)

if __name__ == "__main__":
    evaluator = PureRAGEvaluator()
    evaluator.run()
