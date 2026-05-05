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

# Pre-download Docling ML models so subprocess workers skip the HF Hub hit
# on their first run.  Uses the same DocumentConverter constructor as the worker
# (most reliable API).  Models land in HF_HOME; we chown them to appuser.
# NOTE: HF_HUB_OFFLINE=1 is set at Cloud Run runtime (deploy workflow), NOT here,
# so local docker builds / docker-compose still work without a model cache.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN mkdir -p "$HF_HOME" && \
    ( python -c "from docling.datamodel.base_models import InputFormat; from docling.datamodel.pipeline_options import PdfPipelineOptions; from docling.document_converter import DocumentConverter as _DC, PdfFormatOption; opts=PdfPipelineOptions(); opts.do_ocr=False; opts.generate_page_images=False; opts.generate_picture_images=False; _DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}); print('Docling models cached OK')" \
    && echo "Model pre-download complete" \
    || echo "WARNING: Docling model pre-download failed -- models download at first subprocess run" ) ; \
    chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true

# Drop privileges
USER appuser

# Cloud Run injects PORT (default 8080); the app also reads RAG_PORT.
ENV PORT=8080
    
    EXPOSE ${PORT}
    
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
