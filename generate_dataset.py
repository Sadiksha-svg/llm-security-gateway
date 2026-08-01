import json
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_benchmark_dataset(samples_per_category=5):
    """
    Downloads and prepares a 10-prompt trial dataset directly from Hugging Face:
    - 5 Malicious Injection vectors (Label = 1)
    - 5 Benign Task vectors (Label = 0)
    """
    logging.info("Connecting to Hugging Face Hub to fetch 'deepset/prompt-injections' dataset...")
    
    try:
        hf_dataset = load_dataset("deepset/prompt-injections", split="train")
        logging.info(f"Successfully loaded raw entries from Hugging Face.")
    except Exception as e:
        logging.error(f"Failed to fetch dataset from Hugging Face: {e}")
        return

    malicious_samples = [row["text"] for row in hf_dataset if row["label"] == 1]
    benign_samples = [row["text"] for row in hf_dataset if row["label"] == 0]

    dataset = []
    prompt_id = 1

    # Sample 5 Malicious Prompts
    for i in range(min(samples_per_category, len(malicious_samples))):
        dataset.append({
            "id": prompt_id,
            "category": "malicious",
            "prompt": malicious_samples[i],
            "expected_action": "block",
            "source": "HuggingFace: deepset/prompt-injections"
        })
        prompt_id += 1

    # Sample 5 Benign Prompts
    for i in range(min(samples_per_category, len(benign_samples))):
        dataset.append({
            "id": prompt_id,
            "category": "benign",
            "prompt": benign_samples[i],
            "expected_action": "allow",
            "source": "HuggingFace: deepset/prompt-injections"
        })
        prompt_id += 1

    output_path = Path("benchmark_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    logging.info(f"Dataset successfully built at '{output_path.resolve()}' with {len(dataset)} total prompts.")

if __name__ == "__main__":
    build_benchmark_dataset(samples_per_category=5)