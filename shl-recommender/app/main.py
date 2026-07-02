import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.catalog import load_catalog
from app.retrieval import CatalogIndex
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.agent import handle_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

_state = {"index": None, "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        items = load_catalog()
        _state["index"] = CatalogIndex(items)
        _state["ready"] = True
        logger.info("Startup complete: %d catalog items indexed", len(items))
    except Exception as e:
        # Don't crash the process on a bad catalog fetch - /health should
        # still respond so the host doesn't think the service is dead; we
        # just fail /chat calls with a clear error until it recovers.
        logger.exception("Startup catalog load failed: %s", e)
        _state["ready"] = False
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    index = _state.get("index")
    if not _state.get("ready") or index is None:
        raise HTTPException(status_code=503, detail="Catalog not ready yet, please retry shortly.")
    try:
        return handle_chat(request.messages, index)
    except Exception as e:
        logger.exception("Unhandled error in /chat: %s", e)
        raise HTTPException(status_code=500, detail="Internal error handling chat turn.")
