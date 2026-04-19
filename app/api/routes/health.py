"""
Health check route — service connectivity status.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.dependencies import verify_services
from app.monitoring.prometheus import get_metrics
from app.models.responses import (
    HealthResponse,
    PipelineLatencyResponse,
    ServiceHealthResponse,
)

router = APIRouter(tags=["health"])


def _check_service(name: str, url: str, check_fn) -> ServiceHealthResponse:
    """Run a health check for a single service and return its status."""
    t0 = time.perf_counter()
    try:
        check_fn()
        latency = (time.perf_counter() - t0) * 1000
        return ServiceHealthResponse(
            name=name, url=url, status="healthy", latency_ms=round(latency, 1)
        )
    except Exception:
        latency = (time.perf_counter() - t0) * 1000
        return ServiceHealthResponse(
            name=name, url=url, status="unhealthy", latency_ms=round(latency, 1)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of all services concurrently."""
    from app.api.dependencies import (
        get_cloud_storage_client,
        get_firestore_client,
        get_qdrant_client,
    )
    from app.core.config import get_settings

    settings = get_settings()
    loop = asyncio.get_running_loop()

    def _check_qdrant():
        get_qdrant_client().get_collections()

    def _check_firestore():
        get_firestore_client()

    def _check_gcs():
        get_cloud_storage_client()

    def _check_ollama():
        from app.generation.ollama_provider import OllamaProvider
        if not OllamaProvider().health_check():
            raise RuntimeError("Ollama not reachable")

    checks = [
        ("qdrant",        f"http://{settings.qdrant_host}:{settings.qdrant_port}", _check_qdrant),
        ("firestore",     f"firestore/{settings.firestore_database}",              _check_firestore),
        ("cloud_storage", f"gs://{settings.gcs_bucket}",                           _check_gcs),
        ("ollama",        settings.ollama_base_url,                                _check_ollama),
    ]

    services: list[ServiceHealthResponse] = list(await asyncio.gather(*[
        loop.run_in_executor(None, lambda name=name, url=url, fn=fn: _check_service(name, url, fn))
        for name, url, fn in checks
    ]))

    healthy_count = sum(1 for s in services if s.status == "healthy")
    overall = "healthy" if healthy_count == len(services) else "degraded" if healthy_count > 0 else "unhealthy"

    return HealthResponse(
        overall=overall,
        checked_at=datetime.now(timezone.utc).isoformat(),
        services=services,
        pipeline=PipelineLatencyResponse(),
    )


@router.get("/metrics")
async def prometheus_metrics() -> bytes:
    """Expose Prometheus metrics."""
    from fastapi.responses import Response

    return Response(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
