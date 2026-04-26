"""
FastAPI middleware — request logging, timing, CORS, and Prometheus metrics.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import get_settings
from app.core.logging import get_logger
from app.monitoring.prometheus import ACTIVE_CONNECTIONS, REQUEST_COUNT, REQUEST_LATENCY

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach request_id, log request/response, record Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Bind request context to structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        method = request.method
        path = request.url.path

        ACTIVE_CONNECTIONS.inc()
        t0 = time.perf_counter()

        try:
            response = await call_next(request)
            elapsed = time.perf_counter() - t0

            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed * 1000:.1f}ms"

            # Prometheus
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)

            # Structured log
            log_level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
            getattr(logger, log_level)(
                "http_request",
                method=method,
                path=path,
                status=status_code,
                duration_ms=round(elapsed * 1000, 1),
            )

            return response
        except Exception:
            elapsed = time.perf_counter() - t0
            REQUEST_COUNT.labels(method=method, endpoint=path, status=500).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
            logger.exception("http_request_error", method=method, path=path)
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
            structlog.contextvars.clear_contextvars()


class StripApiPrefixMiddleware(BaseHTTPMiddleware):
    """Strip the /api path prefix when present.

    Firebase Hosting rewrites /api/** → Cloud Run preserving the full path,
    so Cloud Run receives /api/auth/google instead of /auth/google.
    This middleware normalises the path so all deployment modes work:

      Firebase Hosting → Cloud Run  : /api/auth/google → /auth/google  ✓
      Vite dev proxy (strips /api)  : /auth/google     → /auth/google  ✓
      nginx (strips /api)           : /auth/google     → /auth/google  ✓
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path: str = request.scope["path"]
        if path.startswith("/api"):
            stripped = path[4:] or "/"
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware on the FastAPI app."""
    settings = get_settings()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Strip /api prefix forwarded by Firebase Hosting → Cloud Run rewrite
    app.add_middleware(StripApiPrefixMiddleware)

    # Request logging (must be added after CORS so CORS runs first)
    app.add_middleware(RequestLoggingMiddleware)
