"""
======================================================================
Localized AI Security Gateway API (main.py)
Role: Asynchronous FastAPI reverse-proxy enforces NeMo Guardrails on 
local Llama 3 execution with extended 300s socket timeouts.
======================================================================
"""

import time
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nemoguardrails import RailsConfig, LLMRails

# Industry-Standard Audit Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gateway.log")]
)
logger = logging.getLogger("SecurityGateway")

# Silence noisy background logging channels
for noisy_logger in ["httpx", "httpcore", "nemoguardrails"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

app = FastAPI(
    title="Localized AI Security Gateway Proxy",
    description="Asynchronous reverse-proxy integrating NVIDIA NeMo Guardrails with Llama 3 via Ollama.",
    version="3.1.0"
)

# Initialize NeMo Engine
CONFIG_PATH = "./config"
try:
    rails_config = RailsConfig.from_path(CONFIG_PATH)
    rails = LLMRails(rails_config)
    logger.info("NVIDIA NeMo Guardrails initialized successfully.")
except Exception as err:
    logger.error(f"Failed to load Guardrail configurations: {err}", exc_info=True)
    raise RuntimeError(f"Configuration initialization failure: {err}")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

class PromptPayload(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text prompt to be evaluated.")

@app.post("/chat/unguarded")
async def chat_unguarded(payload: PromptPayload):
    """Control Route: Direct to Ollama. Extended 300s timeout for local CPU/RAM."""
    start_time = time.perf_counter()
    # Increased timeout to 300.0s to prevent socket drops on heavy CPU runs
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                OLLAMA_ENDPOINT,
                json={"model": "llama3", "prompt": payload.prompt, "stream": False}
            )
            response.raise_for_status()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            logger.info(f"[UNGUARDED] 200 OK | Latency: {elapsed_ms}ms")
            return {
                "response": response.json().get("response", ""),
                "latency_ms": elapsed_ms,
                "status": "allowed"
            }
        except Exception as e:
            logger.error(f"[UNGUARDED] Backend execution failure: {e}")
            raise HTTPException(status_code=502, detail=f"Ollama Backend Error: {str(e)}")

@app.post("/chat/guarded")
async def chat_guarded(payload: PromptPayload):
    """Protected Route: Passes through NeMo Guardrails before execution."""
    start_time = time.perf_counter()
    try:
        res = await rails.generate_async(prompt=payload.prompt)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        output_text = res.get("content", "") if isinstance(res, dict) else str(res)

        # Deterministic check for the exact tag emitted in rails.co
        if "[GATEWAY_SECURITY_BLOCK]" in output_text:
            status = "blocked"
        else:
            status = "allowed"

        logger.info(f"[GUARDED] Status: {status.upper()} | Latency: {elapsed_ms}ms")
        return {
            "response": output_text,
            "latency_ms": elapsed_ms,
            "status": status
        }
    except Exception as e:
        logger.error(f"[GUARDED] Guardrail evaluation failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Guardrail Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
