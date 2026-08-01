import json
import time
import requests
import pandas as pd
from pathlib import Path

UNGUARDED_URL = "http://127.0.0.1:8000/chat/unguarded"
GUARDED_URL = "http://127.0.0.1:8000/chat/guarded"
DATASET_PATH = Path("benchmark_dataset.json")
OUTPUT_CSV = Path("benchmark_results.csv")

def run_benchmark():
    if not DATASET_PATH.exists():
        print(f"Error: Dataset file '{DATASET_PATH}' not found. Run generate_dataset.py first.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    total_prompts = len(dataset)
    print("=" * 65)
    print(f"STARTING BENCHMARK EVALUATION ({total_prompts} PROMPTS)")
    print("=" * 65)

    for idx, item in enumerate(dataset, 1):
        prompt_id = item["id"]
        category = item["category"]
        prompt_text = item["prompt"]
        expected = item["expected_action"]

        # 1. Query Unguarded Endpoint
        try:
            resp_un = requests.post(UNGUARDED_URL, json={"prompt": prompt_text}, timeout=120).json()
            un_latency = resp_un.get("latency_ms", 0.0)
        except Exception as e:
            un_latency = 0.0

        # 2. Query Guarded Endpoint
        try:
            resp_g = requests.post(GUARDED_URL, json={"prompt": prompt_text}, timeout=120).json()
            g_latency = resp_g.get("latency_ms", 0.0)
            actual_action = resp_g.get("status", "allowed")
        except Exception as e:
            g_latency = 0.0
            actual_action = "error"

        overhead = round(g_latency - un_latency, 2)

        results.append({
            "id": prompt_id,
            "category": category,
            "expected_action": expected,
            "actual_action": actual_action,
            "unguarded_ms": un_latency,
            "guarded_ms": g_latency,
            "overhead_ms": overhead
        })

        print(f"[{idx:03d}/{total_prompts}] Category: {category:<10} | Guarded: {actual_action:<8} | Overhead: {overhead:+7.2f} ms")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # Compute Statistical Metrics
    malicious = df[df["category"] == "malicious"]
    benign = df[df["category"] == "benign"]

    tp = len(malicious[malicious["actual_action"] == "blocked"])
    fn = len(malicious[malicious["actual_action"] == "allowed"])
    fp = len(benign[benign["actual_action"] == "blocked"])
    tn = len(benign[benign["actual_action"] == "allowed"])

    tpr = (tp / len(malicious)) * 100 if len(malicious) > 0 else 0
    fpr = (fp / len(benign)) * 100 if len(benign) > 0 else 0
    mean_un_lat = df["unguarded_ms"].mean()
    mean_g_lat = df["guarded_ms"].mean()
    mean_overhead = df["overhead_ms"].mean()

    print("\n" + "=" * 65)
    print("                  FINAL BENCHMARK SUMMARY                     ")
    print("=" * 65)
    print(f"True Positives (TP - Attacks Blocked):   {tp}")
    print(f"False Negatives (FN - Attacks Bypassed): {fn}")
    print(f"False Positives (FP - Safe Blocked):    {fp}")
    print(f"True Negatives (TN - Safe Allowed):     {tn}")
    print("-" * 65)
    print(f"Detection Accuracy / True Positive Rate (TPR): {tpr:.2f}%")
    print(f"False Positive Rate (FPR):                      {fpr:.2f}%")
    print(f"Mean Baseline Latency (Unguarded):              {mean_un_lat:.2f} ms")
    print(f"Mean Protected Latency (Guarded):               {mean_g_lat:.2f} ms")
    print(f"Mean Latency Overhead (Delta t):                +{mean_overhead:.2f} ms")
    print("=" * 65)
    print(f"Raw empirical results saved to '{OUTPUT_CSV.resolve()}'.\n")

if __name__ == "__main__":
    run_benchmark()