"""
Subprocess worker — Docling PDF conversion + chunking in an isolated process.

This module runs in a **separate process** spawned by the ingestion pipeline.
All Docling models (layout detection, TableFormer) are loaded here, not in
the main FastAPI server process.  When this process exits, *all* model memory
is released by the OS.

Why a subprocess?
  - Docling loads ~1–2 GB of ML models for layout + table detection.
  - Processing large PDFs (financial reports, 100+ pages) can peak at several
    GB of RAM.  Running this **in the server process** causes ``std::bad_alloc``
    which crashes or freezes the entire API — no more health checks, no
    document listings, nothing.
  - A subprocess gives complete memory isolation.  If Docling OOMs, only this
    worker dies.  The server keeps running.  Memory is 100% reclaimed by the OS.

Protocol:
  Input  — CLI args: ``<pdf_path> <output_json_path> <document_id> <filename>``
  Output — JSON file at ``output_json_path`` with chunks + metadata.
  Exit   — 0 on success, 1 on conversion error, 2 on setup error.

Usage::

    python -m app.ingestion.docling_worker \\
        /tmp/rag_uploads/abc_report.pdf \\
        /tmp/rag_uploads/abc_report.json \\
        abc-document-id \\
        report.pdf
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _run(pdf_path: str, output_path: str, document_id: str, filename: str) -> None:
    """Execute Docling conversion + chunking, write JSON result."""
    import time as _time
    _t0 = _time.perf_counter()

    from app.core.config import get_settings
    from app.ingestion.chunking.docling_hybrid import DoclingHybridChunker
    from app.ingestion.document_converter import DocumentConverter

    settings = get_settings()

    _t_imports = _time.perf_counter()
    converter = DocumentConverter()
    _t_model_load = _time.perf_counter()

    chunker = DoclingHybridChunker(
        max_tokens=settings.chunk_size,
        min_chunk_chars=settings.min_chunk_chars,
    )

    # ── Convert ───────────────────────────────────────────────────────────
    _t_conv_start = _time.perf_counter()
    parsed = converter.convert(
        source=Path(pdf_path),
        document_id=document_id,
        filename=filename,
    )
    _t_conv_end = _time.perf_counter()

    # ── Chunk ─────────────────────────────────────────────────────────────
    chunks = chunker.chunk(parsed)
    _t_chunk_end = _time.perf_counter()

    print(
        f"[worker_timing] imports={_t_imports - _t0:.2f}s "
        f"model_load={_t_model_load - _t_imports:.2f}s "
        f"conversion={_t_conv_end - _t_conv_start:.2f}s "
        f"chunking={_t_chunk_end - _t_conv_end:.2f}s "
        f"total={_t_chunk_end - _t0:.2f}s "
        f"pages={parsed.total_pages} chunks={len(chunks)}",
        flush=True,
    )

    # ── Serialize ─────────────────────────────────────────────────────────
    output = {
        "status": "success",
        "total_pages": parsed.total_pages,
        "conversion_status": parsed.metadata.get("conversion_status", "success"),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "enriched_text": c.enriched_text,
                "content_type": c.content_type.value,
                "document_id": c.document_id,
                "page": c.page,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "metadata": c.metadata or {},
            }
            for c in chunks
        ],
    }

    Path(output_path).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 5:
        err = {
            "status": "error",
            "error": "Usage: python -m app.ingestion.docling_worker <pdf_path> <output_json> <document_id> <filename>",
        }
        print(json.dumps(err), file=sys.stderr)
        sys.exit(2)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    document_id = sys.argv[3]
    filename = sys.argv[4]

    try:
        _run(pdf_path, output_path, document_id, filename)
    except Exception as exc:
        # Write error details to the output file so the parent can read them
        error_output = {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        try:
            Path(output_path).write_text(
                json.dumps(error_output, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # best-effort; parent reads stderr
        print(f"docling_worker failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
