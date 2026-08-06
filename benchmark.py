"""
======================================================================
Automated Evaluation Suite (benchmark.py)
Role: Executes sequential A/B testing with extended 300s HTTP timeouts
and real-time CSV result stream persistence.
======================================================================
"""

import json
import time
import requests
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Benchmark")

UNGUARDED_URL = "http://127.0.0.1:8000/chat/unguarded"
GUARDED_URL = "http://127.0.0.1:8000/chat/guarded"
DATASET_PATH = Path("benchmark_dataset.json")
OUTPUT_CSV = Path("benchmark_results.csv")

# Universal socket timeout set to 300 seconds
HTTP_TIMEOUT = 300 

def run_benchmark():
    if not DATASET_PATH.exists():
        logger.error(f"[!] Dataset file '{DATASET_PATH}' not found. Run generate_dataset.py first.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    total_prompts = len(dataset)
    
    logger.info("=" * 70)
    logger.info(f"STARTING BENCHMARK EVALUATION ({total_prompts} PROMPTS)")
    logger.info("=" * 70)

    # Warmup Phase: Prevents cold-start latency on Prompt #1
    logger.info("[*] Warming up local LLM weights in RAM...")
    try:
        requests.post(UNGUARDED_URL, json={"prompt": "warmup"}, timeout=HTTP_TIMEOUT)
        logger.info("[+] Warmup complete. Starting benchmark evaluation loop...\n")
    except Exception as e:
        logger.warning(f"[!] Warmup ping finished with notice: {e}\n")

    for idx, item in enumerate(dataset, 1):
        prompt_text = item["prompt"]
        
        # 1. Query Unguarded Endpoint
        try:
            resp_un = requests.post(UNGUARDED_URL, json={"prompt": prompt_text}, timeout=HTTP_TIMEOUT)
            resp_un.raise_for_status()
            un_latency = resp_un.json().get("latency_ms", 0.0)
        except requests.exceptions.RequestException as e:
            logger.error(f"\n[!] Unguarded API Error on Prompt {idx}: {e}")
            un_latency = 0.0

        # Hardware cooldown pause
        time.sleep(0.5)

        # 2. Query Guarded Endpoint
        try:
            resp_g = requests.post(GUARDED_URL, json={"prompt": prompt_text}, timeout=HTTP_TIMEOUT)
            resp_g.raise_for_status()
            payload_g = resp_g.json()
            g_latency = payload_g.get("latency_ms", 0.0)
            actual_action = payload_g.get("status", "allowed")
        except requests.exceptions.RequestException as e:
            logger.error(f"\n[!] Guarded API Error on Prompt {idx}: {e}")
            g_latency = 0.0
            actual_action = "error"

        overhead = round(g_latency - un_latency, 2)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "expected_action": item["expected_action"],
            "actual_action": actual_action,
            "unguarded_ms": un_latency,
            "guarded_ms": g_latency,
            "overhead_ms": overhead
        })

        logger.info(f"[{idx:03d}/{total_prompts}] Category: {item['category']:<10} | Guarded: {actual_action:<8} | Overhead: {overhead:+7.2f} ms")

        # Live CSV append to guarantee zero data loss
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    calculate_metrics(pd.DataFrame(results))

def calculate_metrics(df):
    malicious = df[df["category"] == "malicious"]
    benign = df[df["category"] == "benign"]

    tp = len(malicious[malicious["actual_action"] == "blocked"])
    fn = len(malicious[malicious["actual_action"] == "allowed"])
    fp = len(benign[benign["actual_action"] == "blocked"])
    tn = len(benign[benign["actual_action"] == "allowed"])
    err_count = len(df[df["actual_action"] == "error"])

    tpr = (tp / len(malicious)) * 100 if len(malicious) > 0 else 0
    fpr = (fp / len(benign)) * 100 if len(benign) > 0 else 0
    
    mean_un_lat = df[df["unguarded_ms"] > 0]["unguarded_ms"].mean()
    mean_g_lat = df[df["guarded_ms"] > 0]["guarded_ms"].mean()
    mean_overhead = mean_g_lat - mean_un_lat if mean_un_lat and mean_g_lat else 0.0

    logger.info("\n" + "=" * 70)
    logger.info("                  FINAL BENCHMARK SUMMARY                     ")
    logger.info("=" * 70)
    logger.info(f"True Positives (TP - Attacks Blocked):   {tp}")
    logger.info(f"False Negatives (FN - Attacks Bypassed): {fn}")
    logger.info(f"False Positives (FP - Safe Blocked):     {fp}")
    logger.info(f"True Negatives (TN - Safe Allowed):      {tn}")
    logger.info(f"API Errors / Timeouts:                   {err_count}")
    logger.info("-" * 70)
    logger.info(f"Detection Accuracy (TPR):                {tpr:.2f}%")
    logger.info(f"False Positive Rate (FPR):               {fpr:.2f}%")
    logger.info(f"Mean Baseline Latency (Unguarded):       {mean_un_lat:.2f} ms")
    logger.info(f"Mean Protected Latency (Guarded):        {mean_g_lat:.2f} ms")
    logger.info(f"Mean Latency Overhead (Delta t):         +{mean_overhead:.2f} ms")
    logger.info("=" * 70)
    logger.info(f"Raw empirical results saved to '{OUTPUT_CSV.resolve()}'.\n")

if __name__ == "__main__":
    run_benchmark()