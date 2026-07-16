import json
import numpy as np
from memory_system.manager import MemorySystem

class PerLTQA_FullEvaluator:
    """
    Evaluator for the PerLTQA dataset.
    Integrates perltmem (memory facts) and perltqa (queries).
    """
    def __init__(self, mem_path="perltmem_en_v2.json", qa_path="perltqa_en_v2.json"):
        self.mem_path = mem_path
        self.qa_path = qa_path
        self.sys = MemorySystem()

    def flatten_memories(self, persona_data):
        """
        Converts the complex nested structure of perltmem into a list of strings.
        """
        facts = []

        # 1. Profile Facts
        profile = persona_data.get("profile", {})
        if isinstance(profile, dict):
            for key, value in profile.items():
                facts.append(f"{persona_data.get('Protagonist', 'The user')}'s {key} is {value}.")

        # 2. Social Relationships
        social = persona_data.get("social_relationship", {})
        if isinstance(social, dict):
            for rel_id, data in social.items():
                if isinstance(data, dict):
                    desc = data.get("Description", "")
                    if desc: facts.append(desc)
                elif isinstance(data, list):
                    for item in data:
                        facts.append(str(item))

        # 3. Events
        events = persona_data.get("events", {})
        if isinstance(events, dict):
            for event_id, data in events.items():
                if isinstance(data, dict):
                    content = data.get("content", "")
                    summary = data.get("summary", "")
                    if content: facts.append(content)
                    if summary: facts.append(f"Summary: {summary}")

        return facts

    def run(self):
        with open(self.mem_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        with open(self.qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        total_queries = 0
        correct_at_1 = 0
        correct_at_3 = 0

        for persona_name, qa_content in qa_data.items():
            print(f"\nEvaluating Persona: {persona_name}")
            self.sys = MemorySystem()
            persona_mems = mem_data.get(persona_name, {})
            if not persona_mems:
                continue

            facts = self.flatten_memories(persona_mems)
            for fact in facts:
                self.sys.encode(fact, store="semantic")

            all_questions = []
            if "profile" in qa_content: all_questions.extend(qa_content["profile"])
            if "social_relationship" in qa_content:
                for rel_list in qa_content["social_relationship"].values():
                    all_questions.extend(rel_list)
            if "events" in qa_content:
                for event_list in qa_content["events"].values():
                    all_questions.extend(event_list)

            for q_item in all_questions:
                question = q_item["Question"]
                answer = q_item["Answer"]

                total_queries += 1
                # USE SIMPLE RETRIEVAL (Base RAG)
                results = self.sys.simple_retrieve(question, top_k=3)

                found_at = -1
                for i, (rec, score) in enumerate(results):
                    if any(word.lower() in rec.content.lower() for word in answer.split() if len(word) > 3):
                        found_at = i + 1
                        break

                if found_at == 1: correct_at_1 += 1
                if found_at != -1 and found_at <= 3: correct_at_3 += 1

        accuracy_1 = correct_at_1 / total_queries if total_queries > 0 else 0
        accuracy_3 = correct_at_3 / total_queries if total_queries > 0 else 0

        print("\n" + "="*30)
        print("FINAL PERLTQA BASE RAG RESULTS")
        print("="*30)
        print(f"Total Queries: {total_queries}")
        print(f"Recall@1: {accuracy_1:.2%}")
        print(f"Recall@3: {accuracy_3:.2%}")
        print("="*30)

        return {
            "total_queries": total_queries,
            "recall_at_1": accuracy_1,
            "recall_at_3": accuracy_3
        }


if __name__ == "__main__":
    evaluator = PerLTQA_FullEvaluator(
        mem_path="perltmem_en_v2.json",
        qa_path="perltqa_en_v2.json"
    )
    evaluator.run()
