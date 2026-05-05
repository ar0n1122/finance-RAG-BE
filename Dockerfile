# syntax=docker/dockerfile:1
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
    # Install CPU-only PyTorch + torchvision from the CPU wheel index.
    # torchvision from PyPI is a CUDA build — its C++ extension (_C) fails to load
    # without CUDA libs, so torchvision::nms is never registered, which breaks
    # docling's import chain (granite_vision → AutoProcessor → torchvision).
    # Installing from the CPU index gives a CUDA-free wheel that loads cleanly.
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# ── Pre-download Docling ML models ────────────────────────────────────────────
# Must happen BEFORE COPY app/ so Docker can cache this expensive layer
# independently of app code changes.
#
# HF_HOME      = huggingface_hub cache (some assets)
# DOCLING_CACHE_DIR = where Docling itself stores downloaded ONNX/Torch models
#                    (default: ~/.cache/docling — which would be /root/.cache/docling
#                     during build, inaccessible to appuser at runtime)
#
# We explicitly point both to appuser's home so models survive the USER switch.
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV DOCLING_CACHE_DIR=/home/appuser/.cache/docling
RUN <<'EOF'
set -e
mkdir -p "$HF_HOME" "$DOCLING_CACHE_DIR"
# Hard failure: if either download fails the image build fails.
# Do NOT add '|| true' — a silently broken image would set HF_HUB_OFFLINE=1
# but have no cached models, causing LocalEntryNotFoundError at runtime.
python -c "
# With CPU torchvision (installed from PyTorch's CPU wheel index) the full
# docling import chain works: granite_vision → AutoProcessor → torchvision.
# Pre-download the layout (Heron) + table (TableFormer) ONNX models so the
# subprocess worker never has to reach HuggingFace at runtime.
import sys
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as _DC, PdfFormatOption
opts = PdfPipelineOptions()
opts.do_ocr = False
opts.do_chart_extraction = False
opts.do_code_enrichment = False
opts.do_formula_enrichment = False
opts.generate_page_images = False
opts.generate_picture_images = False
opts.generate_parsed_pages = False
print('Downloading Docling layout/table ONNX models...', flush=True)
_DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
print('Docling ONNX models OK', flush=True)
# sentence-transformers/all-MiniLM-L6-v2 tokenizer required by HybridChunker
# in the main API process (get_ingestion_pipeline → DoclingHybridChunker.__init__).
print('Downloading sentence-transformers/all-MiniLM-L6-v2 tokenizer...', flush=True)
from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer
get_default_tokenizer()
print('All models cached OK', flush=True)
"
echo 'Pre-download complete'
chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true
EOF

# Lock HuggingFace to offline mode — models are now in the image cache.
# Prevents runtime downloads and eliminates huggingface.co connectivity errors.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application (after model download so the layer above is cache-friendly)
COPY app/ ./app/

# Drop privileges
USER appuser

# Cloud Run injects PORT (default 8080); the app also reads RAG_PORT.
ENV PORT=8080
EXPOSE ${PORT}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
