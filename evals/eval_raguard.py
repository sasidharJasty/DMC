import json
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv
# Ensure root directory is in path for imports
sys.path.append(os.getcwd())

from openai import OpenAI
from src.manager import MemorySystem

# Load environment variables from .env file
load_dotenv()

class RAGuardEvaluator:
    """
    Evaluates the system using the RAGuard dataset.
    Compares claims against a document corpus using a Pure RAG baseline 
    and an LLM-as-a-Judge to verify accuracy and faithfulness.
    """
    def __init__(self, claims_path="data/RAGuard/claims.csv", docs_path="data/RAGuard/documents.csv", api_key=None):
        self.claims_path = claims_path
        self.docs_path = docs_path
        self.sys = MemorySystem()
        
        # Initialize OpenAI Client
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini" # Cheapest/most efficient model

    def ingest_documents(self):
        """Loads documents.csv and encodes them into the vector DB."""
        print("Ingesting RAGuard documents into vector DB...")
        df = pd.read_csv(self.docs_path)
        
        for _, row in df.iterrows():
            content = f"Title: {row['Title']}\nText: {row['Full Text']}"
            # Use raw_encode to keep it a Pure RAG baseline
            self.sys.raw_encode(content)
        print(f"Successfully ingested {len(df)} documents.")

    def judge_claim(self, claim, context):
        """
        LLM-as-a-Judge: Determines if the claim is supported by the context.
        Designed for maximum token efficiency.
        """
        system_prompt = (
            "You are a strict fact-checker. Determine if the claim is SUPPORTED, "
            "CONTRADICTED, or UNKNOWN based ONLY on the provided context. "
            "Respond in JSON: {\"verdict\": \"SUPPORTED\"|\"CONTRADICTED\"|\"UNKNOWN\", \"reason\": \"short explanation\"}"
        )
        user_prompt = f"Context:\n{context}\n\nClaim: {claim}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling LLM judge: {e}")
            return {"verdict": "ERROR", "reason": str(e)}

    def run(self, top_k=5):
        # 1. Setup
        self.ingest_documents()
        claims_df = pd.read_csv(self.claims_path)
        
        results = []
        total_queries = 0
        correct_verdicts = 0
        
        print(f"Evaluating {len(claims_df)} claims...")
        
        for _, row in claims_df.iterrows():
            claim_text = row['Claim']
            ground_truth = str(row['Verdict']).strip().lower() # e.g., "true" or "false"
            ref_doc_ids = str(row['Document IDs']) # This is a string representation of a list
            
            total_queries += 1
            
            # Step A: Retrieve Top-K documents
            retrieved = self.sys.simple_retrieve(claim_text, top_k=top_k)
            
            # Construct context for LLM
            context = "\n---\n".join([rec.content for rec, score in retrieved])
            
            # Step B: LLM Judge
            judgment = self.judge_claim(claim_text, context)
            predicted_verdict = judgment['verdict'].upper()
            
            # Normalize ground truth for comparison
            # mapping "true" -> "SUPPORTED", "false" -> "CONTRADICTED"
            gt_normalized = "SUPPORTED" if ground_truth == "true" else "CONTRADICTED" if ground_truth == "false" else "UNKNOWN"
            
            is_correct = (predicted_verdict == gt_normalized)
            if is_correct:
                correct_verdicts += 1
            
            # Metric: Reference Match (Did we retrieve the specific docs listed in claims.csv?)
            # Note: simple_retrieve doesn't store the original Document ID from CSV 
            # unless we added it to the content. We check if content overlaps.
            ref_match = False
            # Extract IDs from the string "[1, 2, 3]"
            try:
                target_ids = json.loads(ref_doc_ids)
                # We'd need to have stored IDs in the DB to do this accurately. 
                # For now, we track the result based on LLM verdict.
            except:
                pass

            results.append({
                "claim_id": row['Claim ID'],
                "ground_truth": gt_normalized,
                "predicted": predicted_verdict,
                "correct": is_correct,
                "reason": judgment['reason']
            })

        # Final Metrics
        accuracy = correct_verdicts / total_queries if total_queries > 0 else 0
        
        final_report = {
            "total_claims": total_queries,
            "metrics": {
                "verdict_accuracy": accuracy,
                "model_used": self.model,
                "top_k": top_k
            },
            "details": results
        }

        with open("results/raguard_baseline.json", "w") as f:
            json.dump(final_report, f, indent=2)

        print("\n" + "="*40)
        print("RAGUARD EVALUATION RESULTS")
        print("="*40)
        print(f"Total Claims: {total_queries}")
        print(f"Verdict Accuracy: {accuracy:.2%}")
        print(f"Model: {self.model}")
        print("="*40)

if __name__ == "__main__":
    evaluator = RAGuardEvaluator()
    evaluator.run()
