"""
FastAPI application entrypoint.

Assembles middleware, exception handlers, and routes into a single app instance.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import setup_middleware
from app.api.routes import auth, chats, documents, evaluate, health, ingest, query, usage
from app.core.config import get_settings
from app.core.events import lifespan
from app.core.exceptions import RAGException
from app.core.logging import setup_logging

# ── Logging setup (must happen before anything else) ──────────────────────────
setup_logging()

# ── App creation ──────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade RAG backend with hybrid search, multi-strategy pipelines, and evaluation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# ── Middleware ────────────────────────────────────────────────────────────────
setup_middleware(app)

# ── Exception handlers ───────────────────────────────────────────────────────


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": type(exc).__name__},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    from app.core.logging import get_logger
    _logger = get_logger("unhandled")
    _logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "InternalError"},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
# The frontend vite proxy rewrites /api/* → /* so all routes are at root level.

app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(documents.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(evaluate.router)
app.include_router(health.router)
app.include_router(usage.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "RAG Pipeline API", "version": "1.0.0"}
