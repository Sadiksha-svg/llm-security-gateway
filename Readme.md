# Localized LLM Security Gateway

This repository contains the software artefact for benchmarking localized Large Language Model (LLM) security. It implements an asynchronous Python reverse-proxy integrated with NVIDIA NeMo Guardrails to intercept, evaluate, and block prompt injection attacks against a locally hosted Meta Llama 3 model.

The primary goal of this project is to measure the trade-off between adversarial detection accuracy (security) and computational latency (performance overhead) in a resource-constrained, offline environment.

## 📁 Repository Structure
* `main.py`: The FastAPI reverse-proxy handling traffic routing and Guardrail execution.
* `benchmark.py`: The automated evaluation suite that calculates True Positive Rate, False Positive Rate, and latency overhead.
* `generate_dataset.py`: Fetches and structures benign and malicious prompt data from Hugging Face (`deepset/prompt-injections`).
* `config/`: Contains the NeMo Guardrails YAML configurations (`config.yml`) and Colang (`rails.co`) security policies.

---

## ⚙️ Prerequisites & System Setup

This project is designed to run locally. We will need Python 3.10+ and [Ollama](https://ollama.com/) installed on machine.

### 1. Configure the LLM Engine (Ollama)

First, download and install Ollama. Because LLM weights are large files, we can configure Ollama to store models on a secondary drive (e.g., D: drive) to save space on the primary OS drive.

Open a terminal as Administrator and run:
```cmd
setx OLLAMA_MODELS "D:\ollama_models"
```

Next, pull the Llama 3 (8B) model to the local machine:
```cmd
ollama pull llama3
```

### 2. Python Environment Setup

Open a terminal in the root directory of this repository and create a virtual environment to isolate the project dependencies:
```cmd
python -m venv venv
```
Activate the virtual environment:
```cmd
PowerShell: .\venv\Scripts\Activate.ps1
Command Prompt: venv\Scripts\activate
```
Upgrade the core packaging tools and install the required project dependencies:
```cmd
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
### 3. Running the Evaluation Suite
**Step 1: Generate the Benchmark Dataset**

Before running the proxy, pull the evaluation dataset (benign enterprise queries and malicious jailbreak payloads):
```cmd
python generate_dataset.py
```
This will create a structured benchmark_dataset.json file in the root directory.

**Step 2: Launch the Security Gateway Proxy**

Start the FastAPI reverse-proxy. This acts as the middleware between the client and the local Llama 3 model:
```cmd
python main.py
```
The proxy will initialize NeMo Guardrails and begin listening on http://127.0.0.1:8000.

**Step 3: Execute the Benchmark**

Open a new, separate terminal window, activate the virtual environment (.\venv\Scripts\Activate.ps1), and run the automated benchmarking suite:
```cmd
python benchmark.py
```
The script will automatically execute a model warmup ping, run 200 prompts sequentially across both unguarded and guarded endpoints, stream live data to benchmark_results.csv, and print the final metrics summary (TPR, FPR, Mean Latency Overhead).

## 🔒 Data Encoding & Formatting Preservation
To ensure empirical validity during benchmarking, the evaluation pipeline strictly preserves the raw text formatting of all input prompts:

**UTF-8 Encoding:** All data ingestion and output files (benchmark_dataset.json, gateway.log, and benchmark_results.csv) enforce explicit utf-8 encoding to prevent character corruption when processing non-English translation queries or telegraphic search terms.

**Payload Sanitization Safety:** Input text is serialized using Pydantic schema validation (PromptPayload) in FastAPI, preserving multi-line structures, spaces, and escape sequences without un-escaping raw string delimiters prior to semantic evaluation.

**CSV Stream Escaping:** benchmark.py uses pandas stream persistence to automatically escape quotes, commas, and newlines in benchmark outputs, ensuring raw prompt integrity across consecutive dataset runs.
