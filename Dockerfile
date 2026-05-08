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

# Install system libraries required by OpenCV (cv2) at runtime.
# cv2 is imported by docling_ibm_models (TableFormer) and needs libxcb1
# even in headless/non-display mode on python:3.12-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

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
# Increment MODELS_CACHE_BUST to force re-download of ML models on next build.
ARG MODELS_CACHE_BUST=4
RUN <<'EOF'
set -e
mkdir -p "$HF_HOME" "$DOCLING_CACHE_DIR"
# Hard failure: if either download fails the image build fails.
# Do NOT add '|| true' — a silently broken image would set HF_HUB_OFFLINE=1
# but have no cached models, causing LocalEntryNotFoundError at runtime.
python -c "
# With CPU torchvision (installed from PyTorch's CPU wheel index) the full
# docling import chain works: granite_vision → AutoProcessor → torchvision.
# Pre-download layout + table (TableFormer) ONNX models so the subprocess
# worker never has to reach HuggingFace at runtime.
#
# We cache BOTH heron (default) AND egret_medium so the worker succeeds
# regardless of which RAG_DOCLING_LAYOUT_MODEL is set in the Cloud Run env.
import sys, io
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as _DC, PdfFormatOption

def _base_opts():
    o = PdfPipelineOptions()
    o.do_ocr = False
    o.do_chart_extraction = False
    o.do_code_enrichment = False
    o.do_formula_enrichment = False
    o.generate_page_images = False
    o.generate_picture_images = False
    o.generate_parsed_pages = False
    o.do_picture_classification = False
    o.do_picture_description = False
    return o

def _make_test_pdf():
    'Build a minimal valid 1-page PDF entirely in memory (no external tools).'
    objs = {
        1: b'<</Type/Catalog/Pages 2 0 R>>',
        2: b'<</Type/Pages/Kids[3 0 R]/Count 1>>',
        3: b'<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>',
        5: b'<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>',
    }
    stream = b'BT /F1 12 Tf 72 720 Td (DoclingBuildTest) Tj ET'
    objs[4] = b'<</Length ' + str(len(stream)).encode() + b'>>'
    buf = bytearray(b'%PDF-1.4\n')
    offsets = {}
    for i in range(1, 6):
        offsets[i] = len(buf)
        if i == 4:
            buf += (str(i) + ' 0 obj').encode() + objs[i] + b'\nstream\n' + stream + b'\nendstream\nendobj\n'
        else:
            buf += (str(i) + ' 0 obj').encode() + objs[i] + b'\nendobj\n'
    xpos = len(buf)
    buf += b'xref\n0 6\n0000000000 65535 f \n'
    for i in range(1, 6):
        buf += (f'{offsets[i]:010d} 00000 n \n').encode()
    buf += b'trailer<</Size 6/Root 1 0 R>>\nstartxref\n' + str(xpos).encode() + b'\n%%EOF\n'
    return bytes(buf)

def _test_convert(dc, label):
    'Run a minimal conversion to flush ALL lazy-loaded models into the HF cache.'
    from docling.datamodel.document import DocumentStream
    src = DocumentStream(name='test.pdf', stream=io.BytesIO(_make_test_pdf()))
    result = dc.convert(src)
    print(f'{label} test conversion status={result.status}', flush=True)

# 1. Heron (default, 42.9M RT-DETR v2) — used when RAG_DOCLING_LAYOUT_MODEL=heron
print('Downloading Docling heron layout + table ONNX models...', flush=True)
dc_heron = _DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_base_opts())})
print('Heron ONNX models OK — running test convert to warm lazy models...', flush=True)
_test_convert(dc_heron, 'heron')
del dc_heron
print('Heron warm-up done', flush=True)

# 2. Egret-medium (19.5M D-Fine) — used when RAG_DOCLING_LAYOUT_MODEL=egret_medium
# This is the model currently set in Cloud Run (RAG_DOCLING_LAYOUT_MODEL=egret_medium).
print('Downloading Docling egret_medium layout ONNX model...', flush=True)
from docling.datamodel.pipeline_options import LayoutOptions, DOCLING_LAYOUT_EGRET_MEDIUM
opts2 = _base_opts()
opts2.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_MEDIUM)
dc_egret = _DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts2)})
print('Egret-medium ONNX model OK — running test convert to warm lazy models...', flush=True)
_test_convert(dc_egret, 'egret_medium')
del dc_egret
print('Egret-medium warm-up done', flush=True)

# sentence-transformers/all-MiniLM-L6-v2 tokenizer required by HybridChunker
# in the main API process (get_ingestion_pipeline → DoclingHybridChunker.__init__).
print('Downloading sentence-transformers/all-MiniLM-L6-v2 tokenizer...', flush=True)
from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer
get_default_tokenizer()
print('All models cached and warm-up complete', flush=True)
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
