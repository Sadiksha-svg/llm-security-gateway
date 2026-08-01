Download and Install ollama.

To direct all future model downloads to Drive D
# setx OLLAMA_MODELS "D:\ollama_models"

Install llama3 models to drive D.
# ollama pull llama3

Create virtual environment.
# python -m venv venv

Activate the Virtual Environment
# .\venv\Scripts\Activate.ps1
# venv\Scripts\activate

upgrade pip and install all your required project packages
# python -m pip install --upgrade pip setuptools wheel
# pip install nemoguardrails fastapi uvicorn requests pandas

verify everything is ready
# pip list

generate a requirements.txt
# pip freeze > requirements.txt

To set up the project on a new machine or rebuild your venv.
# pip install -r requirements.txt

Generate dataset.
# python generate_dataset

Launch API.
# python main.py

Execute Benchmark Evaluation Suite.
# python benchmark.py


