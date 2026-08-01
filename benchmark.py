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
        print(f"[!] Error: Dataset file '{DATASET_PATH}' not found. Run generate_dataset.py first.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    total_prompts = len(dataset)
    print("=" * 70)
    print(f"STARTING BENCHMARK EVALUATION ({total_prompts} PROMPTS TRIAL RUN)")
    print("=" * 70)

    # Model Warmup Call (Loads Llama 3 weights into RAM before latency tracking begins)
    print("[*] Warming up local LLM model weights in RAM...")
    try:
        requests.post(UNGUARDED_URL, json={"prompt": "warmup test"}, timeout=300)
        print("[+] Warmup complete. Starting benchmark evaluation loop...\n")
    except Exception as e:
        print(f"[!] Warmup ping finished: {e}\n")

    for idx, item in enumerate(dataset, 1):
        prompt_id = item["id"]
        category = item["category"]
        prompt_text = item["prompt"]
        expected = item["expected_action"]

        # 1. Query Unguarded Endpoint
        try:
            resp_un = requests.post(UNGUARDED_URL, json={"prompt": prompt_text}, timeout=300).json()
            un_latency = resp_un.get("latency_ms", 0.0)
        except Exception as e:
            un_latency = 0.0
            print(f"\n[!] Unguarded Error on prompt {idx}: {e}")

        # Pause 1 second between endpoint requests to let CPU recover
        time.sleep(1.0)

        # 2. Query Guarded Endpoint
        try:
            resp_g = requests.post(GUARDED_URL, json={"prompt": prompt_text}, timeout=300).json()
            g_latency = resp_g.get("latency_ms", 0.0)
            actual_action = resp_g.get("status", "allowed")
        except Exception as e:
            g_latency = 0.0
            actual_action = "error"
            print(f"\n[!] Guarded Error on prompt {idx}: {e}")

        overhead = round(g_latency - un_latency, 2)

        record = {
            "id": prompt_id,
            "category": category,
            "expected_action": expected,
            "actual_action": actual_action,
            "unguarded_ms": un_latency,
            "guarded_ms": g_latency,
            "overhead_ms": overhead
        }
        results.append(record)

        # Print Terminal Progress Stream
        print(f"[{idx:02d}/{total_prompts:02d}] Category: {category:<10} | Guarded: {actual_action:<8} | Overhead: {overhead:+7.2f} ms")

        # Live Save to CSV immediately after every prompt
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

        # Pause 3 seconds between loop iterations to cool down RAM/CPU
        time.sleep(3.0)

    # Matrix Summary
    df = pd.DataFrame(results)

    malicious = df[df["category"] == "malicious"]
    benign = df[df["category"] == "benign"]

    tp = len(malicious[malicious["actual_action"] == "blocked"])
    fn = len(malicious[malicious["actual_action"] == "allowed"])
    fp = len(benign[benign["actual_action"] == "blocked"])
    tn = len(benign[benign["actual_action"] == "allowed"])
    err_count = len(df[df["actual_action"] == "error"])

    tpr = (tp / len(malicious)) * 100 if len(malicious) > 0 else 0.0
    fpr = (fp / len(benign)) * 100 if len(benign) > 0 else 0.0
    mean_un_lat = df["unguarded_ms"].mean()
    mean_g_lat = df["guarded_ms"].mean()
    mean_overhead = df["overhead_ms"].mean()

    print("\n" + "=" * 70)
    print("              10-PROMPT TRIAL BENCHMARK SUMMARY               ")
    print("=" * 70)
    print(f"True Positives (TP - Attacks Blocked):   {tp}")
    print(f"False Negatives (FN - Attacks Bypassed): {fn}")
    print(f"False Positives (FP - Safe Blocked):    {fp}")
    print(f"True Negatives (TN - Safe Allowed):     {tn}")
    print(f"Total Errors / Timeouts:                {err_count}")
    print("-" * 70)
    print(f"Detection Accuracy / True Positive Rate (TPR): {tpr:.2f}%")
    print(f"False Positive Rate (FPR):                      {fpr:.2f}%")
    print(f"Mean Baseline Latency (Unguarded):              {mean_un_lat:.2f} ms")
    print(f"Mean Protected Latency (Guarded):               {mean_g_lat:.2f} ms")
    print(f"Mean Latency Overhead (Delta t):                +{mean_overhead:.2f} ms")
    print("=" * 70)
    print(f"Raw empirical results saved to '{OUTPUT_CSV.resolve()}'.\n")

if __name__ == "__main__":
    run_benchmark()