"""
======================================================================
Dataset Generation Pipeline (generate_dataset.py)
Role: Fetches real-world benchmark data from the Hugging Face Hub
(deepset/prompt-injections) to ensure a standardized academic testbed.
======================================================================
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_benchmark_dataset():
    logging.info("Fetching 'deepset/prompt-injections' dataset...")
    
    try:
        # Contacts Hugging Face API and downloads the raw JSON records
        hf_dataset = load_dataset("deepset/prompt-injections", split="train")
    except Exception as e:
        logging.error(f"Failed to fetch dataset: {e}")
        return

    # Slices 100 Malicious inputs (label 1) and 100 Benign inputs (label 0)
    malicious_samples = [row["text"] for row in hf_dataset if row["label"] == 1][:100]
    benign_samples = [row["text"] for row in hf_dataset if row["label"] == 0][:100]

    dataset = []
    
    # Structure Malicious Data
    for idx, text in enumerate(malicious_samples, 1):
        dataset.append({"id": idx, "category": "malicious", "prompt": text, "expected_action": "block"})
        
    # Structure Benign Data
    for idx, text in enumerate(benign_samples, 101):
        dataset.append({"id": idx, "category": "benign", "prompt": text, "expected_action": "allow"})

    # Export to local JSON for benchmark.py to read
    output_path = Path("benchmark_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    logging.info(f"Dataset built at '{output_path.resolve()}' ({len(dataset)} entries).")

if __name__ == "__main__":
    build_benchmark_dataset()