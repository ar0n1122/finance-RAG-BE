# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for native extensions (pdf parsing, crypto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app/ ./app/

# Install deps into a virtual env that we can copy to runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    # Install CPU-only PyTorch first so docling doesn't pull the CUDA variant (~1.7 GB savings)
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Copy application
COPY app/ ./app/

# Drop privileges
USER appuser

# Cloud Run injects PORT (default 8080); the app also reads RAG_PORT.
ENV PORT=8080

# Version label (set via --build-arg APP_VERSION=v1.0.42 during CI build)
ARG APP_VERSION=dev
LABEL org.opencontainers.image.version="${APP_VERSION}"
ENV APP_VERSION="${APP_VERSION}"

EXPOSE ${PORT}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
