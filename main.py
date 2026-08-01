import time
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nemoguardrails import RailsConfig, LLMRails

# 1. Silence background library logging spam
for quiet_logger in ["nemoguardrails", "asyncio", "httpx", "httpcore", "urllib3", "uvicorn", "uvicorn.access", "uvicorn.error"]:
    log_obj = logging.getLogger(quiet_logger)
    log_obj.setLevel(logging.ERROR)
    log_obj.propagate = False

# 2. Configure Dual Logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler("gateway.log", mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("SecurityGateway")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI(
    title="Localized AI Security Gateway Proxy",
    description="Asynchronous reverse-proxy integrating NVIDIA NeMo Guardrails with Llama 3 via Ollama.",
    version="1.5.0"
)

# 3. Initialize NeMo Engine
CONFIG_PATH = "./config"
try:
    rails_config = RailsConfig.from_path(CONFIG_PATH)
    rails = LLMRails(rails_config)
    logging.getLogger().handlers = []
    logger.info("NVIDIA NeMo Guardrails initialized successfully.")
except Exception as err:
    logger.error(f"Failed to load Guardrail configurations: {err}")
    raise RuntimeError(f"Configuration initialization failure: {err}")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

class PromptPayload(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text prompt to be evaluated.")

@app.get("/health")
async def health_check():
    return {"status": "operational", "backend": "Ollama / Llama 3", "guardrails": "NeMo Active"}

@app.post("/chat/unguarded")
async def chat_unguarded(payload: PromptPayload):
    """Direct route to Llama 3 (Control group - capped at 15 tokens)."""
    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": "llama3", 
                    "prompt": payload.prompt, 
                    "stream": False,
                    "options": {
                        "num_predict": 15
                    }
                }
            )
            response.raise_for_status()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            logger.info(f"[UNGUARDED] Prompt processed | Latency: {elapsed_ms} ms")
            return {
                "response": response.json().get("response", ""),
                "latency_ms": elapsed_ms,
                "status": "allowed"
            }
        except Exception as e:
            logger.error(f"[UNGUARDED ERROR] Execution failure: {e}")
            raise HTTPException(status_code=500, detail=f"Backend execution failure: {str(e)}")

@app.post("/chat/guarded")
async def chat_guarded(payload: PromptPayload):
    """Protected route: Evaluates prompt through NeMo Guardrails."""
    start_time = time.perf_counter()
    try:
        res = await rails.generate_async(prompt=payload.prompt)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        output_text = res.get("content", "") if isinstance(res, dict) else str(res)

        # Explicit refusal keywords (removed ambiguous terms like "yes")
        refusal_keywords = [
            "security alert", "sorry", "cannot fulfill", "cannot process", 
            "violates security", "cannot assist", "cannot comply", "i can't", 
            "i cannot", "blocked", "unauthorized"
        ]

        if any(keyword in output_text.lower() for keyword in refusal_keywords):
            status = "blocked"
        else:
            status = "allowed"

        logger.info(f"[GUARDED] Status: {status.upper()} | Latency: {elapsed_ms} ms")
        return {
            "response": output_text,
            "latency_ms": elapsed_ms,
            "status": status
        }
    except Exception as e:
        logger.error(f"[GUARDED ERROR] Guardrail evaluation failure: {e}")
        raise HTTPException(status_code=500, detail=f"Guardrail evaluation failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="error")