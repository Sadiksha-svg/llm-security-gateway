import time
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nemoguardrails import RailsConfig, LLMRails

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityGateway")

app = FastAPI(
    title="Localized AI Security Gateway Proxy",
    description="Asynchronous reverse-proxy integrating NVIDIA NeMo Guardrails with Llama 3 via Ollama.",
    version="1.0.0"
)

# Initialize Guardrails Instance
CONFIG_PATH = "./config"
try:
    rails_config = RailsConfig.from_path(CONFIG_PATH)
    rails = LLMRails(rails_config)
    logger.info("NVIDIA NeMo Guardrails initialized successfully.")
except Exception as err:
    logger.error(f"Failed to load Guardrail configurations: {err}")
    raise RuntimeError("Configuration initialization failure.")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

class PromptPayload(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text prompt to be evaluated.")

@app.get("/health")
async def health_check():
    return {"status": "operational", "backend": "Ollama / Llama 3", "guardrails": "NeMo Active"}

@app.post("/chat/unguarded")
async def chat_unguarded(payload: PromptPayload):
    """Direct route to local Llama 3 model (Control group - no safety checking)."""
    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                OLLAMA_ENDPOINT,
                json={"model": "llama3", "prompt": payload.prompt, "stream": False}
            )
            response.raise_for_status()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            return {
                "response": response.json().get("response", ""),
                "latency_ms": elapsed_ms,
                "status": "allowed"
            }
        except Exception as e:
            logger.error(f"Unguarded endpoint error: {e}")
            raise HTTPException(status_code=500, detail=f"Backend execution failure: {str(e)}")

@app.post("/chat/guarded")
async def chat_guarded(payload: PromptPayload):
    """Protected route: Intercepts prompt through NeMo Guardrails prior to execution."""
    start_time = time.perf_counter()
    try:
        # Evaluate semantic policies asynchronously
        res = await rails.generate_async(prompt=payload.prompt)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        output_text = res.get("content", "") if isinstance(res, dict) else str(res)

        # Flag status if refusal trigger activated
        if "SECURITY ALERT" in output_text or "violates security" in output_text.lower():
            status = "blocked"
        else:
            status = "allowed"

        return {
            "response": output_text,
            "latency_ms": elapsed_ms,
            "status": status
        }
    except Exception as e:
        logger.error(f"Guarded endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Guardrail evaluation failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")