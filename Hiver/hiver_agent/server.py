"""Production HTTP surface for the support agent.

FastAPI is the only web dependency. It gives us JSON validation, an SSE
streaming endpoint for the live console, a health check for orchestrators, and
static hosting of the built frontend -- all from one process.

    uvicorn hiver_agent.server:app --port 8000
    python -m hiver_agent.server            # convenience wrapper
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .rag import RagAgent, embed, ollama, ollama_healthy

STATE: dict = {}

EXAMPLES = [
    {"label": "Late driver", "text": "My driver was 20 minutes late for pickup and I missed my flight.", "kind": "routine"},
    {"label": "Double charge", "text": "Why was I charged twice for the same trip this morning?", "kind": "routine"},
    {"label": "Promo failed", "text": "My promo code FIRST50 did not apply at checkout even though it is still valid.", "kind": "routine"},
    {"label": "Off-app cash", "text": "My driver asked me for an extra 200 rupees in cash at the end of the ride.", "kind": "escalate"},
    {"label": "Safety", "text": "My driver threatened me and I feel unsafe, what do I do?", "kind": "escalate"},
    {"label": "Account locked", "text": "I cannot log in, my account says it has been disabled.", "kind": "escalate"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = Path(settings.index_path)
    if not index.exists():
        raise SystemExit(
            f"RAG index not found: {index}. Run scripts/build_rag_index.py first."
        )
    STATE["agent"] = RagAgent.from_index(str(index))
    STATE["index_size"] = len(STATE["agent"].pairs)
    _warm_up()
    yield
    STATE.clear()


def _warm_up() -> None:
    """Load the embed + generation models into VRAM so the first real request
    is not paying a multi-second cold start."""
    try:
        embed(["warm up"])
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        ollama(
            "/api/generate",
            {"model": settings.gen_model, "prompt": "ok", "stream": False,
             "think": False, "keep_alive": settings.gen_keep_alive,
             "options": {"num_predict": 1}},
            timeout=settings.gen_timeout_s, retries=1,
        )
    except Exception:  # pragma: no cover - best effort
        pass


app = FastAPI(title="Hiver Support Console", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AssessRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def _agent() -> RagAgent:
    agent = STATE.get("agent")
    if agent is None:  # pragma: no cover - only before lifespan
        raise HTTPException(503, "Agent not ready")
    return agent


@app.get("/api/health")
def health() -> dict:
    ok, models = ollama_healthy()
    return {
        "status": "ok" if ok else "degraded",
        "ollama": ok,
        "ollama_models": models,
        "gen_model": settings.gen_model,
        "embed_model": settings.embed_model,
        "index_size": STATE.get("index_size", 0),
        "fastpath": settings.enable_fastpath,
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    return EXAMPLES


@app.post("/api/assess")
def assess(req: AssessRequest) -> dict:
    return _agent().assess(req.message).to_dict()


@app.post("/api/assess/stream")
def assess_stream(req: AssessRequest) -> StreamingResponse:
    agent = _agent()

    def gen():
        for event in agent.stream(req.message):
            payload = dict(event)
            if payload.get("stage") == "done":
                payload = {"stage": "done", "result": payload["result"].to_dict(),
                           "cached": payload.get("cached", False)}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- static frontend (built assets) -------------------------------------------
_static = Path(settings.static_dir)
if _static.is_dir():
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(_static / "index.html")

    @app.get("/{path:path}")
    def _spa(path: str) -> FileResponse:
        candidate = _static / path
        return FileResponse(candidate if candidate.is_file() else _static / "index.html")


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default=settings.host)
    p.add_argument("--port", type=int, default=settings.port)
    p.add_argument("--index", default=settings.index_path)
    p.add_argument("--model", default=settings.gen_model)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    import os

    os.environ["HIVER_INDEX_PATH"] = args.index
    os.environ["HIVER_MODEL"] = args.model
    print(f"Hiver Support Console -> http://{args.host}:{args.port}")
    uvicorn.run("hiver_agent.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
