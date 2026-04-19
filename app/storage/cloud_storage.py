"""
GCP Cloud Storage client wrapper.

Manages the ``rag-documents`` bucket with the following structure::

    gs://rag-documents/
    └── raw/{document_id}/{filename}   # original user-uploaded files
"""

from __future__ import annotations

from google.cloud import storage

from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class CloudStorageClient:
    """Typed wrapper around GCS blob operations."""

    def __init__(self, client: storage.Client, bucket_name: str) -> None:
        self._client = client
        self._bucket_name = bucket_name
        self._bucket = client.bucket(bucket_name)

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_bytes(
        self,
        data: bytes,
        destination: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes to GCS.  Returns the ``gs://`` URI."""
        blob = self._bucket.blob(destination)
        blob.upload_from_string(data, content_type=content_type)
        uri = f"gs://{self._bucket_name}/{destination}"
        logger.debug("gcs_bytes_uploaded", destination=destination, size=len(data))
        return uri

    # ── Download ──────────────────────────────────────────────────────────────

    def download_bytes(self, gcs_path: str) -> bytes:
        """Download an object as bytes."""
        blob = self._bucket.blob(gcs_path)
        if not blob.exists():
            raise StorageError(f"Object not found: gs://{self._bucket_name}/{gcs_path}")
        return blob.download_as_bytes()

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_prefix(self, prefix: str) -> int:
        """Delete all blobs under *prefix*.  Returns count of deleted blobs."""
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        count = len(blobs)
        if count > 0:
            self._bucket.delete_blobs(blobs)
        logger.info("gcs_prefix_deleted", prefix=prefix, deleted_count=count)
        return count

    def delete_document_files(self, document_id: str) -> int:
        """Delete all files associated with a document."""
        total = self.delete_prefix(f"raw/{document_id}/")
        logger.info("gcs_document_deleted", document_id=document_id, total_deleted=total)
        return total
