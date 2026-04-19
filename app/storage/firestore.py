"""
GCP Firestore client wrapper.

Provides type-safe CRUD operations for all collections:
  - users
  - documents
  - evaluations
  - queries

The GCP Firestore SDK is synchronous; methods use ``run_in_executor`` for
async compatibility where needed by callers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore

from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class FirestoreClient:
    """Thin, typed wrapper around the Firestore SDK."""

    def __init__(self, client: firestore.Client) -> None:
        self._db = client

    # ── Generic helpers ───────────────────────────────────────────────────────

    def _collection(self, name: str) -> firestore.CollectionReference:
        return self._db.collection(name)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Users ─────────────────────────────────────────────────────────────────

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Return user dict or ``None`` if not found."""
        doc = self._collection("users").document(user_id).get()
        return doc.to_dict() if doc.exists else None

    def upsert_user(self, user_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create or update a user document.  Returns the merged data."""
        ref = self._collection("users").document(user_id)
        existing = ref.get()
        if existing.exists:
            data["last_login"] = self._now_iso()
            ref.update(data)
            logger.info("user_updated", user_id=user_id)
        else:
            data.setdefault("created_at", self._now_iso())
            data["last_login"] = self._now_iso()
            data["role"] = "user"
            data["settings"] = data.get("settings", {
                "default_llm": "ollama",
                "default_top_k": 5,
                "theme": "dark",
            })
            ref.set(data)
            logger.info("user_created", user_id=user_id)
        return {**data, "user_id": user_id}

    # ── Documents ─────────────────────────────────────────────────────────────

    def list_documents(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return all document metadata, most-recent first. Optionally filter by user."""
        col = self._collection("documents")
        if user_id:
            # Query both fields to support docs created by older code that used
            # only 'uploaded_by' before 'user_id' was added.
            seen: set[str] = set()
            result: list[dict[str, Any]] = []
            for q in [
                col.where(filter=firestore.FieldFilter("user_id", "==", user_id)),
                col.where(filter=firestore.FieldFilter("uploaded_by", "==", user_id)),
            ]:
                for d in q.stream():
                    if d.id not in seen:
                        seen.add(d.id)
                        result.append({**d.to_dict(), "id": d.id})
        else:
            docs = list(col.stream())
            result = [{**d.to_dict(), "id": d.id} for d in docs]
        result.sort(key=lambda d: d.get("ingested_at", ""), reverse=True)
        return result

    def get_document(self, document_id: str) -> dict[str, Any]:
        """Return a single document or raise ``NotFoundError``."""
        doc = self._collection("documents").document(document_id).get()
        if not doc.exists:
            raise NotFoundError(f"Document '{document_id}' not found")
        return {**doc.to_dict(), "id": doc.id}

    def create_document(self, document_id: str, data: dict[str, Any]) -> None:
        """Create a new document metadata record."""
        data.setdefault("ingested_at", self._now_iso())
        data.setdefault("status", "queued")
        self._collection("documents").document(document_id).set(data)
        logger.info("document_metadata_created", document_id=document_id)

    def update_document(self, document_id: str, data: dict[str, Any]) -> None:
        """Partial update of document metadata."""
        self._collection("documents").document(document_id).update(data)
        logger.debug("document_metadata_updated", document_id=document_id, fields=list(data.keys()))

    def delete_document(self, document_id: str) -> None:
        """Delete document metadata."""
        self._collection("documents").document(document_id).delete()
        logger.info("document_metadata_deleted", document_id=document_id)

    # ── Evaluations ───────────────────────────────────────────────────────────

    def save_evaluation(self, evaluation_id: str, data: dict[str, Any]) -> None:
        data.setdefault("started_at", self._now_iso())
        self._collection("evaluations").document(evaluation_id).set(data)
        logger.info("evaluation_saved", evaluation_id=evaluation_id)

    def get_latest_evaluation(self) -> dict[str, Any] | None:
        """Return the most recent evaluation report, or ``None``."""
        docs = (
            self._collection("evaluations")
            .order_by("started_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for d in docs:
            return {**d.to_dict(), "evaluation_id": d.id}
        return None

    # ── Queries (history) ─────────────────────────────────────────────────────

    def log_query(self, query_id: str, data: dict[str, Any]) -> None:
        data.setdefault("created_at", self._now_iso())
        self._collection("queries").document(query_id).set(data)

    def get_query_history(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        docs = (
            self._collection("queries")
            .where("user_id", "==", user_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [{**d.to_dict(), "query_id": d.id} for d in docs]

    # ── Chat Sessions ─────────────────────────────────────────────────────────

    MAX_CHAT_SESSIONS = 5

    def list_chat_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Return all chat sessions for a user, newest first."""
        docs = (
            self._collection("chat_sessions")
            .where(filter=firestore.FieldFilter("user_id", "==", user_id))
            .stream()
        )
        results = [{**d.to_dict(), "id": d.id} for d in docs]
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results

    def get_chat_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a single chat session or None."""
        doc = self._collection("chat_sessions").document(session_id).get()
        if not doc.exists:
            return None
        return {**doc.to_dict(), "id": doc.id}

    def create_chat_session(self, session_id: str, user_id: str, title: str) -> dict[str, Any]:
        """Create a new chat session, evicting the oldest if at the cap."""
        existing = self.list_chat_sessions(user_id)
        # Evict oldest sessions beyond the limit (keep MAX - 1 to make room)
        if len(existing) >= self.MAX_CHAT_SESSIONS:
            to_delete = existing[self.MAX_CHAT_SESSIONS - 1:]
            for old in to_delete:
                self.delete_chat_session(old["id"])

        now = self._now_iso()
        data: dict[str, Any] = {
            "user_id": user_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self._collection("chat_sessions").document(session_id).set(data)
        logger.info("chat_session_created", session_id=session_id, user_id=user_id)
        return {**data, "id": session_id}

    def update_chat_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Partial update of a chat session (e.g. messages, title)."""
        data["updated_at"] = self._now_iso()
        self._collection("chat_sessions").document(session_id).update(data)

    def delete_chat_session(self, session_id: str) -> None:
        """Delete a chat session."""
        self._collection("chat_sessions").document(session_id).delete()
        logger.info("chat_session_deleted", session_id=session_id)

    # ── Usage / Cost Records ──────────────────────────────────────────────────

    def save_usage_record(self, record_id: str, data: dict[str, Any]) -> None:
        """Save a per-query usage/cost record."""
        data.setdefault("created_at", self._now_iso())
        self._collection("usage_records").document(record_id).set(data)
        logger.debug("usage_record_saved", record_id=record_id)

    def get_usage_records(
        self,
        user_id: str,
        limit: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return usage records for a user, newest first."""
        query = self._collection("usage_records").where(
            filter=firestore.FieldFilter("user_id", "==", user_id),
        )
        if start_date:
            query = query.where(
                filter=firestore.FieldFilter("created_at", ">=", start_date),
            )
        if end_date:
            query = query.where(
                filter=firestore.FieldFilter("created_at", "<=", end_date),
            )
        query = query.order_by(
            "created_at", direction=firestore.Query.DESCENDING,
        ).limit(limit)
        return [{**d.to_dict(), "id": d.id} for d in query.stream()]

    def get_usage_summary(self, user_id: str) -> dict[str, Any]:
        """Aggregate usage stats for a user across all their records."""
        docs = (
            self._collection("usage_records")
            .where(filter=firestore.FieldFilter("user_id", "==", user_id))
            .stream()
        )
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        total_cost = 0.0
        total_queries = 0
        by_model: dict[str, dict[str, Any]] = {}
        by_operation: dict[str, dict[str, Any]] = {}
        daily_costs: dict[str, float] = {}

        for doc in docs:
            d = doc.to_dict()
            total_prompt += d.get("total_prompt_tokens", 0)
            total_completion += d.get("total_completion_tokens", 0)
            total_tokens += d.get("total_tokens", 0)
            total_cost += d.get("total_cost", 0.0)
            total_queries += 1

            # Aggregate by model
            for model_key, model_data in d.get("breakdown_by_model", {}).items():
                if model_key not in by_model:
                    by_model[model_key] = {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "cost": 0.0, "calls": 0,
                    }
                bm = by_model[model_key]
                bm["prompt_tokens"] += model_data.get("prompt_tokens", 0)
                bm["completion_tokens"] += model_data.get("completion_tokens", 0)
                bm["total_tokens"] += model_data.get("total_tokens", 0)
                bm["cost"] += model_data.get("cost", 0.0)
                bm["calls"] += model_data.get("calls", 0)

            # Aggregate by operation
            for op_key, op_data in d.get("breakdown_by_operation", {}).items():
                if op_key not in by_operation:
                    by_operation[op_key] = {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "cost": 0.0, "calls": 0,
                    }
                bo = by_operation[op_key]
                bo["prompt_tokens"] += op_data.get("prompt_tokens", 0)
                bo["completion_tokens"] += op_data.get("completion_tokens", 0)
                bo["total_tokens"] += op_data.get("total_tokens", 0)
                bo["cost"] += op_data.get("cost", 0.0)
                bo["calls"] += op_data.get("calls", 0)

            # Daily cost aggregation
            day = d.get("created_at", "")[:10]
            if day:
                daily_costs[day] = daily_costs.get(day, 0.0) + d.get("total_cost", 0.0)

        return {
            "user_id": user_id,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_queries": total_queries,
            "avg_cost_per_query": total_cost / total_queries if total_queries else 0.0,
            "avg_tokens_per_query": total_tokens / total_queries if total_queries else 0.0,
            "breakdown_by_model": by_model,
            "breakdown_by_operation": by_operation,
            "daily_costs": dict(sorted(daily_costs.items())),
        }

