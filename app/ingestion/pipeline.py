"""
Ingestion pipeline — orchestrates: convert → chunk → embed → index → store.

Uses Docling ``DocumentConverter`` for PDF→DoclingDocument conversion,
then Docling ``HybridChunker`` (or recursive fallback) for structure-aware
chunking, dense embedding, and Qdrant vector indexing.

PDF conversion + chunking runs in an **isolated subprocess** to prevent
Docling's heavy ML models from crashing the API server with OOM errors.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

_QDRANT_RETRYABLE = (OSError, ConnectionError, ResponseHandlingException)

from app.core.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.chunking.base import ChunkingStrategy
from app.ingestion.document_converter import DocumentConverter
from app.ingestion.embeddings.base import EmbeddingProvider
from app.models.domain import Chunk, ContentType, DocumentStatus, IngestionResult
from app.storage.cloud_storage import CloudStorageClient
from app.storage.firestore import FirestoreClient

logger = get_logger(__name__)

# Maximum chars sent to the text embedding model in one go.
_MAX_EMBED_CHARS = 8_000


class IngestionPipeline:
    """
    End-to-end document ingestion.

    Flow:
      1. Upload raw file to GCS
      2. Convert PDF → ``ParsedDocument`` (Docling ``DocumentConverter``)
      3. Chunk → ``list[Chunk]`` (Docling ``HybridChunker`` or recursive fallback)
      4. Embed text chunks → dense vectors
      5. Upsert to Qdrant (dense + full-text index)
      6. Upload processing manifest to GCS
      7. Update Firestore document metadata
    """

    def __init__(
        self,
        settings: Settings,
        converter: DocumentConverter,
        chunker: ChunkingStrategy,
        text_embedder: EmbeddingProvider,
        qdrant: QdrantClient,
        firestore: FirestoreClient,
        gcs: CloudStorageClient,
    ) -> None:
        self._settings = settings
        self._converter = converter
        self._chunker = chunker
        self._text_embedder = text_embedder
        self._qdrant = qdrant
        self._firestore = firestore
        self._gcs = gcs
        self._collection = settings.qdrant_collection

    # ── Progress helpers ─────────────────────────────────────────────────────

    def _update_progress(self, document_id: str, progress: int, stage: str) -> None:
        """Write processing progress (0-100) and stage label to Firestore."""
        try:
            self._firestore.update_document(document_id, {
                "progress": progress,
                "processing_stage": stage,
            })
        except Exception:
            logger.debug("progress_update_failed", document_id=document_id, progress=progress)

    # ── Public API ────────────────────────────────────────────────────────────

    def upload(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> str:
        """
        Phase 1 — fast: upload raw file to GCS and create Firestore record.

        Returns the ``document_id``.
        """
        document_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(content).hexdigest()

        logger.info(
            "upload_started",
            document_id=document_id,
            filename=filename,
            size_bytes=len(content),
            user_id=user_id,
        )

        # Create Firestore record (status=processing)
        self._firestore.create_document(document_id, {
            "title": filename,
            "filename": filename,
            "source_file": filename,
            "content_type": content_type,
            "status": DocumentStatus.PROCESSING.value,
            "file_size_bytes": len(content),
            "size_bytes": len(content),
            "file_hash_sha256": file_hash,
            "uploaded_by": user_id,
            "user_id": user_id,
        })

        # Upload raw file to GCS
        gcs_raw_path = f"raw/{document_id}/{filename}"
        self._gcs.upload_bytes(content, gcs_raw_path, content_type=content_type)

        # Cache locally so re-processing avoids a GCS re-download for up to 24 h.
        self._write_cache(document_id, filename, content)

        logger.info("upload_complete", document_id=document_id)
        return document_id

    def process(
        self,
        document_id: str,
        filename: str,
        content: bytes | None = None,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> IngestionResult:
        """
        Phase 2 — heavy: convert, chunk, embed, index.

        Expects the Firestore record and GCS file to already exist (from ``upload``).
        If *content* is ``None``, the file is loaded from the local 24-h cache
        (written by ``upload()``) or downloaded from GCS as a fallback.

        PDF conversion + chunking runs in an isolated subprocess so Docling's
        ML models don't consume memory in the API server process.
        """
        start = time.perf_counter()

        if content is None:
            content = self._read_cache(document_id, filename)
            if content is not None:
                logger.debug("process_using_cache", document_id=document_id)
            else:
                _t_dl = time.perf_counter()
                logger.info("process_downloading_from_gcs", document_id=document_id)
                content = self._gcs.download_bytes(f"raw/{document_id}/{filename}")
                self._write_cache(document_id, filename, content)
                logger.info(
                    "gcs_download_complete",
                    document_id=document_id,
                    size_kb=round(len(content) / 1024, 1),
                    elapsed_s=round(time.perf_counter() - _t_dl, 2),
                )

        logger.info("processing_started", document_id=document_id, filename=filename)
        self._update_progress(document_id, 5, "converting")

        try:
            # 1+2. Convert PDF → chunks (subprocess for PDFs, in-process for others)
            _t_conv = time.perf_counter()
            if content_type == "application/pdf":
                chunks, total_pages = self._convert_and_chunk_subprocess(
                    document_id=document_id,
                    filename=filename,
                    content=content,
                )
            else:
                parsed = self._converter.convert(
                    source=content,
                    document_id=document_id,
                    filename=filename,
                )
                self._update_progress(document_id, 25, "chunking")
                chunks = self._chunker.chunk(parsed)
                total_pages = parsed.total_pages

            logger.info(
                "conversion_phase_complete",
                document_id=document_id,
                chunks=len(chunks),
                elapsed_s=round(time.perf_counter() - _t_conv, 2),
            )
            self._update_progress(document_id, 40, "embedding")

            # Cap chunks to prevent runaway embedding costs on table-heavy PDFs
            max_chunks = self._settings.max_chunks_per_document
            if max_chunks and len(chunks) > max_chunks:
                logger.warning(
                    "chunk_count_capped",
                    document_id=document_id,
                    original=len(chunks),
                    capped=max_chunks,
                )
                chunks = chunks[:max_chunks]

            # 3. Ensure Qdrant collection exists
            self._ensure_collection()

            # 4. Embed + index text/table chunks
            _t_emb = time.perf_counter()
            vectors_indexed = self._embed_and_index(chunks, document_id=document_id)
            logger.info(
                "embedding_phase_complete",
                document_id=document_id,
                vectors=vectors_indexed,
                elapsed_s=round(time.perf_counter() - _t_emb, 2),
            )
            self._update_progress(document_id, 90, "finalizing")

            elapsed = time.perf_counter() - start
            tables_found = sum(
                1 for c in chunks if c.content_type == ContentType.TABLE
            )

            # 6. Update Firestore (status=indexed)
            self._firestore.update_document(document_id, {
                "status": DocumentStatus.INDEXED.value,
                "progress": 100,
                "processing_stage": "complete",
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "chunk_count": len(chunks),
                "pages": total_pages,
                "chunks": len(chunks),
                "tables_found": tables_found,
                "processing_time_seconds": round(elapsed, 2),
                "processing_duration_s": round(elapsed, 2),
            })

            result = IngestionResult(
                document_id=document_id,
                status=DocumentStatus.INDEXED,
                total_chunks=len(chunks),
                total_pages=total_pages,
                tables_found=tables_found,
                vectors_indexed=vectors_indexed,
                processing_time_s=round(elapsed, 2),
            )
            logger.info(
                "processing_complete",
                document_id=document_id,
                total_chunks=result.total_chunks,
                total_pages=result.total_pages,
                tables=result.tables_found,
                elapsed_s=result.processing_time_s,
            )
            # Remove the local cache file now that the document is indexed.
            self._delete_cache(document_id, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._firestore.update_document(document_id, {
                "status": DocumentStatus.FAILED.value,
                "error": str(exc),
                "processing_time_seconds": round(elapsed, 2),
            })
            logger.error("processing_failed", document_id=document_id, error=str(exc))
            raise IngestionError(f"Processing failed for '{filename}': {exc}") from exc

    # ── Subprocess-based PDF conversion ───────────────────────────────────────

    def _run_worker(
        self,
        pdf_path: Path,
        output_path: Path,
        document_id: str,
        filename: str,
        timeout: float,
    ) -> dict:
        """
        Spawn one docling_worker subprocess for *pdf_path*.  Returns the
        parsed JSON dict on success; raises ``IngestionError`` on failure.
        """
        backend_root = Path(__file__).resolve().parent.parent.parent
        pdf_size_kb = round(pdf_path.stat().st_size / 1024, 1) if pdf_path.exists() else -1
        logger.info(
            "worker_spawning",
            document_id=document_id,
            filename=filename,
            pdf_size_kb=pdf_size_kb,
            timeout_s=timeout,
        )
        _t_spawn = time.perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                "-m", "app.ingestion.docling_worker",
                str(pdf_path),
                str(output_path),
                document_id,
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=None,   # inherit parent stderr → [worker] lines appear in Cloud Run logs in real-time
            text=True,
            timeout=timeout,
            cwd=str(backend_root),
        )
        logger.info(
            "worker_exited",
            document_id=document_id,
            returncode=result.returncode,
            elapsed_s=round(time.perf_counter() - _t_spawn, 2),
        )

        if not output_path.exists():
            raise IngestionError(
                f"Worker produced no output (exit={result.returncode}). "
                f"See [worker] lines in Cloud Run logs for document_id={document_id}"
            )

        raw = json.loads(output_path.read_text(encoding="utf-8"))

        if raw.get("status") != "success":
            error_msg = raw.get("error", "unknown worker error")
            error_type = raw.get("error_type", "")
            raise IngestionError(
                f"Subprocess conversion failed ({error_type}): {error_msg}"
            )

        # Log worker stdout so [worker_timing] lines appear in Cloud Run logs.
        if result.stdout and result.stdout.strip():
            logger.info(
                "worker_output",
                document_id=document_id,
                output=result.stdout.strip(),
            )

        return raw

    def _convert_and_chunk_subprocess(
        self,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> tuple[list[Chunk], int]:
        """
        Run Docling conversion + chunking in an isolated subprocess per batch.

        When ``docling_pages_per_batch`` > 0 and the PDF has more pages than
        that threshold, the PDF is split into smaller files and one fresh
        subprocess is spawned per batch.  Each subprocess exits after its
        batch — releasing all ONNX C++ memory — so peak RAM is capped at
        one batch worth instead of accumulating across the whole document.

        Returns ``(chunks, total_pages)``.
        """
        # Ensure the PDF is on disk for the subprocess to read
        pdf_path = self._cache_path(document_id, filename)
        if not pdf_path.exists():
            pdf_path.write_bytes(content)

        timeout = self._settings.docling_document_timeout or 1800
        pages_per_batch = self._settings.docling_pages_per_batch

        # ── Count pages without loading Docling ───────────────────────────
        total_pages = 0
        split_paths: list[Path] = []
        split_dir: Path | None = None
        try:
            import pypdfium2 as _pdfium
            src_doc = _pdfium.PdfDocument(str(pdf_path))
            total_pages = len(src_doc)

            if pages_per_batch > 0 and total_pages > pages_per_batch:
                split_dir = Path(tempfile.mkdtemp(prefix="rag_pdf_split_"))
                for idx, start0 in enumerate(range(0, total_pages, pages_per_batch)):
                    end0 = min(start0 + pages_per_batch, total_pages)
                    chunk_doc = _pdfium.PdfDocument.new()
                    chunk_doc.import_pages(src_doc, list(range(start0, end0)))
                    tmp_path = split_dir / f"chunk_{idx:03d}.pdf"
                    chunk_doc.save(str(tmp_path))
                    chunk_doc.close()
                    split_paths.append(tmp_path)
                src_doc.close()
                logger.info(
                    "pdf_split_for_batching",
                    document_id=document_id,
                    total_pages=total_pages,
                    batches=len(split_paths),
                    pages_per_batch=pages_per_batch,
                )
            else:
                src_doc.close()
        except ImportError:
            logger.debug("pypdfium2_not_available_skipping_split")

        use_batching = bool(split_paths)
        batch_files = split_paths if use_batching else [pdf_path]
        n_batches = len(batch_files)

        logger.info(
            "subprocess_convert_start",
            document_id=document_id,
            filename=filename,
            batches=n_batches,
            total_pages=total_pages,
            timeout=timeout,
        )

        # Progress budget: converting = 5% → 38%  (33 points spread across batches)
        # Embedding will use 40% → 88%, finalizing 90%, complete 100%.
        _CONV_START = 5
        _CONV_END   = 38

        all_chunks: list[Chunk] = []
        global_chunk_index = 0

        try:
            for i, batch_pdf in enumerate(batch_files):
                # Update Firestore before spawning so UI shows forward motion
                batch_progress = _CONV_START + round(
                    (_CONV_END - _CONV_START) * i / n_batches
                )
                batch_label = (
                    f"converting pages {i * pages_per_batch + 1}–"
                    f"{min((i + 1) * pages_per_batch, total_pages)} of {total_pages}"
                    if use_batching
                    else "converting"
                )
                self._update_progress(document_id, batch_progress, batch_label)
                logger.info(
                    "batch_start",
                    document_id=document_id,
                    batch=i + 1,
                    of=n_batches,
                    progress=batch_progress,
                    label=batch_label,
                )

                output_path = batch_pdf.with_suffix(f".batch{i}.json")
                _t_batch = time.perf_counter()
                try:
                    raw = self._run_worker(
                        batch_pdf, output_path, document_id, filename, timeout
                    )
                    # total_pages from the first (or only) batch is authoritative
                    if i == 0 and not use_batching:
                        total_pages = raw.get("total_pages", total_pages)
                    elif not use_batching:
                        total_pages = raw.get("total_pages", total_pages)

                    for c in raw["chunks"]:
                        all_chunks.append(Chunk(
                            chunk_id=c["chunk_id"],
                            text=c["text"],
                            enriched_text=c.get("enriched_text"),
                            content_type=ContentType(c["content_type"]),
                            document_id=c["document_id"],
                            page=c["page"],
                            section=c.get("section"),
                            chunk_index=global_chunk_index,
                            token_count=c["token_count"],
                            metadata=c.get("metadata", {}),
                        ))
                        global_chunk_index += 1

                    logger.info(
                        "subprocess_batch_complete",
                        document_id=document_id,
                        batch=i + 1,
                        of=n_batches,
                        batch_chunks=len(raw["chunks"]),
                        total_chunks_so_far=global_chunk_index,
                        elapsed_s=round(time.perf_counter() - _t_batch, 2),
                    )
                except subprocess.TimeoutExpired:
                    logger.error(
                        "subprocess_batch_timeout",
                        document_id=document_id,
                        batch=i + 1,
                        timeout=timeout,
                    )
                    raise IngestionError(
                        f"PDF batch {i + 1}/{len(batch_files)} timed out after {timeout}s"
                    )
                except json.JSONDecodeError as exc:
                    raise IngestionError(
                        f"Worker batch {i + 1} output is not valid JSON: {exc}"
                    ) from exc
                finally:
                    try:
                        output_path.unlink(missing_ok=True)
                    except OSError:
                        pass

        finally:
            # Clean up split temp files
            if split_dir is not None:
                for p in split_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                try:
                    split_dir.rmdir()
                except OSError:
                    pass

        logger.info(
            "subprocess_convert_complete",
            document_id=document_id,
            total_chunks=len(all_chunks),
            total_pages=total_pages,
            batches=len(batch_files),
        )

        return all_chunks, total_pages

    def ingest(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> IngestionResult:
        """
        Run the full ingestion pipeline (upload + process) synchronously.
        Kept for backward compatibility.
        """
        document_id = self.upload(filename, content, content_type, user_id)
        return self.process(document_id, filename, content, content_type, user_id)
    # ── Local upload cache ────────────────────────────────────────────────────

    _CACHE_TTL = 86_400  # 24 hours in seconds

    def _cache_dir(self) -> Path:
        base = Path(self._settings.upload_cache_dir) if self._settings.upload_cache_dir else Path(tempfile.gettempdir())
        d = base / "rag_uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_path(self, document_id: str, filename: str) -> Path:
        # Use document_id prefix so filenames with special chars are safe.
        safe_name = Path(filename).name
        return self._cache_dir() / f"{document_id}_{safe_name}"

    def _write_cache(self, document_id: str, filename: str, content: bytes) -> None:
        try:
            path = self._cache_path(document_id, filename)
            path.write_bytes(content)
            logger.debug("upload_cache_written", document_id=document_id, path=str(path), size=len(content))
        except OSError as exc:
            logger.warning("upload_cache_write_failed", document_id=document_id, error=str(exc))

    def _read_cache(self, document_id: str, filename: str) -> bytes | None:
        try:
            path = self._cache_path(document_id, filename)
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self._CACHE_TTL:
                path.unlink(missing_ok=True)
                logger.debug("upload_cache_expired", document_id=document_id)
                return None
            return path.read_bytes()
        except OSError as exc:
            logger.warning("upload_cache_read_failed", document_id=document_id, error=str(exc))
            return None

    def _delete_cache(self, document_id: str, filename: str) -> None:
        try:
            path = self._cache_path(document_id, filename)
            path.unlink(missing_ok=True)
            logger.debug("upload_cache_deleted", document_id=document_id)
        except OSError:
            pass
    # ── Qdrant collection setup ───────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with text dense vector + sparse + payload indexes."""
        # Retry on transient DNS/network failures (common after long CPU-bound processing).
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                existing = [c.name for c in self._qdrant.get_collections().collections]
                break
            except _QDRANT_RETRYABLE as exc:
                last_exc = exc
                logger.warning("qdrant_connect_retry", attempt=_attempt + 1, error=f"{type(exc).__name__}: {exc}")
                if _attempt < 2:
                    time.sleep(2 ** _attempt)  # 1s, 2s
        else:
            raise last_exc  # type: ignore[misc]
        if self._collection in existing:
            return

        text_dim = self._text_embedder.dimension

        self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config={
                "text": models.VectorParams(
                    size=text_dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
                ),
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(),
            },
        )

        # Keyword payload indexes — mirrors user's proven Qdrant schema
        for field_name, schema in [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("content_type", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("metadata.company", models.PayloadSchemaType.KEYWORD),
            ("metadata.type_of_report", models.PayloadSchemaType.KEYWORD),
            ("metadata.chunk_type", models.PayloadSchemaType.KEYWORD),
            ("metadata.fiscal_year", models.PayloadSchemaType.KEYWORD),
        ]:
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=schema,
            )

        # Integer payload indexes
        for field_name in ["page", "metadata.page_min", "metadata.page_max"]:
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        # Full-text index for keyword search leg
        self._qdrant.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        logger.info(
            "qdrant_collection_created",
            collection=self._collection,
            text_dim=text_dim,
        )

    # ── Embedding + indexing ──────────────────────────────────────────────────

    def _embed_and_index(
        self,
        chunks: list[Chunk],
        batch_size: int | None = None,
        document_id: str = "",
    ) -> int:
        """Embed and upsert text/table chunks to Qdrant.  Returns count."""
        if not chunks:
            return 0

        if batch_size is None:
            batch_size = self._settings.embedding_batch_size

        # Progress budget: embedding = 40% → 88%  (48 points across embed batches)
        _EMB_START = 40
        _EMB_END   = 88
        n_embed_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(
            "embed_start",
            document_id=document_id,
            total_chunks=len(chunks),
            n_batches=n_embed_batches,
            batch_size=batch_size,
        )

        total = 0
        _log_every = max(1, n_embed_batches // 4)   # log ~4 times during embedding
        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size
            if document_id:
                emb_progress = _EMB_START + round(
                    (_EMB_END - _EMB_START) * batch_num / n_embed_batches
                )
                self._update_progress(
                    document_id,
                    emb_progress,
                    f"embedding {min(i + batch_size, len(chunks))}/{len(chunks)} chunks",
                )
                if batch_num % _log_every == 0:
                    logger.info(
                        "embed_batch_progress",
                        document_id=document_id,
                        batch=batch_num + 1,
                        of=n_embed_batches,
                        chunks_done=total,
                        chunks_total=len(chunks),
                    )
            batch = chunks[i : i + batch_size]
            texts: list[str] = []
            for c in batch:
                t = c.enriched_text or c.text
                if len(t) > _MAX_EMBED_CHARS:
                    t = t[:_MAX_EMBED_CHARS]
                texts.append(t)

            embeddings = self._text_embedder.embed_documents(texts)

            points: list[models.PointStruct] = []
            for chunk, vector in zip(batch, embeddings):
                points.append(models.PointStruct(
                    id=chunk.chunk_id,
                    vector={"text": vector},
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content_type": chunk.content_type.value,
                        "text": chunk.text,
                        "page": chunk.page,
                        "section": chunk.section or "",
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "metadata": chunk.metadata,
                    },
                ))

            for _attempt in range(3):
                try:
                    self._qdrant.upsert(collection_name=self._collection, points=points)
                    break
                except _QDRANT_RETRYABLE as exc:
                    if _attempt == 2:
                        raise
                    logger.warning("qdrant_upsert_retry", attempt=_attempt + 1, error=f"{type(exc).__name__}: {exc}")
                    time.sleep(2 ** _attempt)
            total += len(points)

        logger.info("embed_complete", document_id=document_id, vectors_indexed=total)
        return total
