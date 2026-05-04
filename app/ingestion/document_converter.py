"""
Document conversion using Docling's ``DocumentConverter``.

Docling converts PDFs into a structured ``DoclingDocument`` AST that faithfully
preserves the document layout — headings, tables, figures, lists, reading
order, and provenance (page numbers).  This AST is the input to
``HybridChunker`` for structure-aware chunking.

**This is the primary parsing strategy for the project.**  The legacy
``PDFParser`` (PyMuPDF + PDFPlumber) is no longer used.

Why Docling over PyMuPDF/PDFPlumber?
  - Single unified pipeline: parse → structure detection → export (no dual-engine quirks)
  - Understands reading order across columns, footnotes, headers/footers
  - Native table detection with robust cell extraction (TableFormer ACCURATE mode)
  - Heading-level hierarchy preserved in the AST → heading-chain breadcrumbs
  - Built-in Markdown export for clean text output
  - Superior accuracy on financial reports (10-K, 10-Q, annual reports)

Pipeline configuration (controlled via ``Settings``):
  - ``PdfPipelineOptions`` with table structure, OCR, layout analysis
  - ``TableStructureOptions(mode=ACCURATE, do_cell_matching=True)`` for financial tables
  - ``AcceleratorOptions`` for GPU/CPU control
  - ``document_timeout`` for production safety (default 120s)
  - ``DocumentStream`` for zero-copy bytes handling (no temp files)
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.models.domain import ParsedDocument

logger = get_logger(__name__)


class DocumentConverter:
    """
    Wraps Docling's ``DocumentConverter`` to produce ``ParsedDocument`` objects.

    The converter handles:
      - PDF parsing with layout analysis (Heron layout model by default)
      - Structure detection (headings, tables, lists, figures)
      - Table structure extraction (TableFormer in ACCURATE mode)
      - Optional OCR for scanned pages
      - Building a ``DoclingDocument`` AST
      - Exporting clean Markdown

    All pipeline knobs are driven by ``Settings`` so they can be tuned via
    environment variables without code changes.

    Image extraction is intentionally de-emphasised: financial reports
    (quarterly/annual) are text-and-table heavy.  Images are preserved
    in the AST but not separately embedded.
    """

    def __init__(self) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.document_converter import (
                DocumentConverter as _DoclingDC,
                PdfFormatOption,
            )

            from app.core.config import get_settings

            settings = get_settings()
            slim = settings.docling_mode == "slim"
            medium = settings.docling_mode == "medium"

            # ── Build PDF pipeline options ────────────────────────────────
            pipeline_options = PdfPipelineOptions()

            # Table structure — critical for financial reports
            pipeline_options.do_table_structure = settings.docling_do_table_structure
            if settings.docling_do_table_structure:
                # slim mode forces FAST; medium/full respect the config setting
                if slim:
                    mode = TableFormerMode.FAST
                else:
                    mode = (
                        TableFormerMode.ACCURATE
                        if settings.docling_table_mode == "accurate"
                        else TableFormerMode.FAST
                    )
                pipeline_options.table_structure_options = TableStructureOptions(
                    mode=mode,
                    do_cell_matching=settings.docling_do_cell_matching,
                )

            # OCR — enable for scanned PDFs
            pipeline_options.do_ocr = settings.docling_do_ocr

            # Timeout — prevents runaway parsing on malformed PDFs
            pipeline_options.document_timeout = settings.docling_document_timeout

            # Hardware acceleration
            try:
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )

                device_map = {
                    "auto": AcceleratorDevice.AUTO,
                    "cpu": AcceleratorDevice.CPU,
                    "cuda": AcceleratorDevice.CUDA,
                    "mps": AcceleratorDevice.MPS,
                }
                device = device_map.get(
                    settings.docling_device, AcceleratorDevice.AUTO
                )
                pipeline_options.accelerator_options = AcceleratorOptions(
                    num_threads=settings.docling_num_threads,
                    device=device,
                )
            except ImportError:
                logger.debug("accelerator_options_not_available")

            # Disable features we don't need for financial docs
            pipeline_options.generate_page_images = False
            pipeline_options.generate_picture_images = False
            pipeline_options.do_picture_classification = False
            pipeline_options.do_picture_description = False
            pipeline_options.do_code_enrichment = False
            pipeline_options.do_formula_enrichment = False
            pipeline_options.do_chart_extraction = False
            pipeline_options.generate_parsed_pages = False

            if slim:
                # ── Slim-mode memory optimizations ────────────────────────
                # Process 1 page at a time (default=4) to cut peak RAM
                pipeline_options.layout_batch_size = 1
                pipeline_options.table_batch_size = 1
                pipeline_options.ocr_batch_size = 1
                # Lower page-render resolution → less RAM per page image
                pipeline_options.images_scale = 0.25
                # Use PDF backend's native text layer directly instead of
                # running the layout model's text extraction on rendered
                # page images.  Safe for programmatic PDFs (10-K, 10-Q,
                # annual reports) that have a reliable text layer.
                pipeline_options.force_backend_text = True
                logger.info(
                    "docling_slim_mode",
                    hint="batch=1, images_scale=0.25, force_backend_text=True, table_mode=fast",
                )
            elif medium:
                # ── Medium-mode: speed + accuracy sweet spot ──────────────
                # batch=2 → 2× faster than slim (270 pages → 135 iterations)
                # force_backend_text keeps RAM safe (~800-900 MB peak)
                # table_mode follows config (ACCURATE by default) — better
                # cell detection than slim's forced FAST, with no extra RAM
                # cost because force_backend_text already saves the headroom.
                pipeline_options.layout_batch_size = 2
                pipeline_options.table_batch_size = 2
                pipeline_options.ocr_batch_size = 2
                pipeline_options.images_scale = 0.25
                pipeline_options.force_backend_text = True
                logger.info(
                    "docling_medium_mode",
                    hint="batch=2, images_scale=0.25, force_backend_text=True, table_mode=config",
                )
            else:
                pipeline_options.images_scale = settings.docling_images_scale

            # ── Create converter with format-specific options ─────────────
            self._converter = _DoclingDC(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                },
            )
            self._available = True
            logger.info(
                "docling_converter_ready",
                mode=settings.docling_mode,
                do_ocr=settings.docling_do_ocr,
                table_mode="fast" if slim else settings.docling_table_mode,
                do_table_structure=settings.docling_do_table_structure,
                timeout=settings.docling_document_timeout,
                device=settings.docling_device,
            )
        except ImportError:
            self._converter = None
            self._available = False
            logger.warning(
                "docling_not_installed",
                hint="Install docling: pip install 'enterprise-rag[docling]'",
            )

    @property
    def is_available(self) -> bool:
        """True when docling is installed and the converter is ready."""
        return self._available

    # ── Public API ────────────────────────────────────────────────────────────

    def convert(
        self,
        source: Path | bytes,
        *,
        document_id: str = "",
        filename: str = "",
    ) -> ParsedDocument:
        """
        Convert a PDF to a ``ParsedDocument``.

        Parameters
        ----------
        source :
            Path to a PDF file **or** raw PDF bytes.
        document_id :
            Unique identifier for the document.
        filename :
            Original upload filename (used for financial metadata extraction).

        Returns
        -------
        ParsedDocument
            Contains ``docling_document`` (the AST), ``markdown``, page count,
            and file-level metadata.
        """
        if not self._available:
            return self._fallback_convert(source, document_id=document_id, filename=filename)

        try:
            # Docling natively supports Path, str, and DocumentStream —
            # no temp files needed for in-memory bytes.
            conv_source = self._build_source(source, filename)
            result = self._converter.convert(conv_source)

            # ── Check conversion status ───────────────────────────────────
            from docling.document_converter import ConversionStatus

            if result.status == ConversionStatus.FAILURE:
                errors = "; ".join(e.error_message for e in result.errors) if result.errors else "unknown"
                raise IngestionError(
                    f"Docling conversion failed for '{filename}': {errors}"
                )

            if result.status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(
                    "docling_partial_success",
                    document_id=document_id,
                    filename=filename,
                    errors=[str(e) for e in (result.errors or [])],
                )

            dl_doc = result.document
            markdown = dl_doc.export_to_markdown()
            total_pages = self._count_pages(dl_doc)

            logger.info(
                "document_converted",
                document_id=document_id,
                filename=filename or self._source_name(source),
                pages=total_pages,
                markdown_chars=len(markdown),
                status=result.status.value,
            )

            return ParsedDocument(
                document_id=document_id,
                filename=filename or self._source_name(source),
                markdown=markdown,
                total_pages=total_pages,
                docling_document=dl_doc,
                metadata={
                    "converter": "docling",
                    "conversion_status": result.status.value,
                    "file_hash": self._hash_bytes(
                        source if isinstance(source, bytes) else source.read_bytes()
                    ),
                },
            )
        except IngestionError:
            raise
        except Exception as exc:
            err_msg = str(exc)
            logger.error(
                "docling_conversion_failed",
                document_id=document_id,
                filename=filename,
                error=err_msg,
            )
            if "bad_alloc" in err_msg or "MemoryError" in type(exc).__name__:
                logger.error(
                    "docling_oom_hint",
                    hint=(
                        "Docling ran out of memory. Try: "
                        "RAG_DOCLING_DO_OCR=false, "
                        "RAG_DOCLING_DO_TABLE_STRUCTURE=false, "
                        "RAG_DOCLING_NUM_THREADS=1, "
                        "RAG_DOCLING_IMAGES_SCALE=0.5"
                    ),
                )
            raise IngestionError(f"Document conversion failed for '{filename}': {exc}") from exc

    # ── Source builders ───────────────────────────────────────────────────────

    @staticmethod
    def _build_source(source: Path | bytes, filename: str) -> Any:
        """
        Return a Docling-compatible source.

        - ``Path`` / ``str``  → passed directly.
        - ``bytes``           → wrapped in ``DocumentStream`` (zero temp files).
        """
        if isinstance(source, (Path, str)):
            p = Path(source)
            if not p.exists():
                raise IngestionError(f"File not found: {p}")
            return p

        # In-memory bytes — use Docling's native DocumentStream
        from docling.datamodel.base_models import DocumentStream

        name = filename or "upload.pdf"
        return DocumentStream(name=name, stream=BytesIO(source))

    @staticmethod
    def _source_name(source: Path | bytes) -> str:
        if isinstance(source, Path):
            return source.name
        return "upload.pdf"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count_pages(dl_doc: Any) -> int:
        """Extract page count from a DoclingDocument."""
        try:
            if hasattr(dl_doc, "pages") and dl_doc.pages is not None:
                return len(dl_doc.pages)
        except Exception:
            pass
        return 0

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ── Fallback (no docling) ─────────────────────────────────────────────────

    def _fallback_convert(
        self,
        source: Path | bytes,
        *,
        document_id: str,
        filename: str,
    ) -> ParsedDocument:
        """
        Minimal fallback when docling is not installed.

        For plain text / markdown files this is sufficient.  For PDFs, a
        warning is logged — install docling for full support.
        """
        logger.warning(
            "docling_fallback",
            filename=filename,
            hint="Install docling for PDF parsing: pip install docling docling-core",
        )

        if isinstance(source, bytes):
            try:
                text = source.decode("utf-8")
            except UnicodeDecodeError:
                raise IngestionError(
                    "Docling is not installed and the file is not valid UTF-8 text.  "
                    "Install docling for PDF support: pip install docling docling-core"
                )
        else:
            text = source.read_text(encoding="utf-8", errors="replace")

        return ParsedDocument(
            document_id=document_id,
            filename=filename or (source.name if isinstance(source, Path) else "unknown"),
            markdown=text,
            total_pages=0,
            docling_document=None,
            metadata={"converter": "fallback_text"},
        )
