"""
Document resolver — uses a lightweight LLM call to determine which of
the user's uploaded documents are relevant to a given query.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.logging import get_logger
from app.storage.firestore import FirestoreClient

logger = get_logger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[.*?]", re.DOTALL)


def _format_document_list(docs: list[dict[str, Any]]) -> str:
    """Build a concise catalogue string for the LLM prompt."""
    lines: list[str] = []
    for doc in docs:
        doc_id = doc["id"]
        title = doc.get("title", "") or doc.get("filename", "")
        status = doc.get("status", "")
        if status != "indexed":
            continue  # skip non-indexed docs
        lines.append(f'- ID: "{doc_id}"  Title: "{title}"')
    return "\n".join(lines) if lines else "(no documents)"


def _parse_llm_ids(text: str, valid_ids: set[str]) -> list[str] | None:
    """Extract document IDs from the LLM response JSON array."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    # "all" sentinel means search everything
    if any(str(v).lower() == "all" for v in parsed):
        return list(valid_ids)
    resolved = [str(v) for v in parsed if str(v) in valid_ids]
    return resolved if resolved else None


def resolve_document_ids(
    firestore: FirestoreClient,
    query: str,
    *,
    user_id: str | None = None,
    explicit_ids: list[str] | None = None,
    gen_pipeline: Any = None,
    user_docs: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> list[str]:
    """
    Determine which document IDs to search.

    Priority order:
    1. If ``explicit_ids`` are provided (frontend selected docs), use them.
    2. Use ``user_docs`` from the frontend (if provided) or fetch from
       Firestore, then ask the LLM which ones the query refers to.
    3. If the LLM call fails or returns nothing, fall back to all of
       the user's documents (user isolation).
    4. If no user, return empty list (search everything — anonymous mode).
    """
    # (1) Explicit override from request body
    if explicit_ids:
        logger.debug("document_resolver", mode="explicit", ids=explicit_ids)
        return explicit_ids

    # (2) Resolve the user's document list
    if not user_id:
        logger.debug("document_resolver", mode="anonymous_all")
        return []

    # Prefer frontend-supplied metadata to avoid an extra Firestore call
    if user_docs is None:
        user_docs = firestore.list_documents(user_id=user_id)
    if not user_docs:
        logger.debug("document_resolver", mode="user_no_docs", user_id=user_id)
        return []

    all_ids = [doc["id"] for doc in user_docs]
    valid_ids = set(all_ids)

    # Fetch recent chat history for context (if session provided)
    chat_history = ""
    if session_id:
        chat_history = _get_chat_context(firestore, session_id)

    # (3) LLM-based resolution
    if gen_pipeline is not None:
        try:
            resolved = _resolve_via_llm(gen_pipeline, query, user_docs, valid_ids, chat_history)
            if resolved is not None:
                return resolved
        except Exception:
            logger.warning("document_resolver_llm_failed", exc_info=True)

    # (4) Fallback — scope to all user's docs
    logger.info("document_resolver", mode="user_all_fallback", count=len(all_ids))
    return all_ids


def _get_chat_context(firestore: FirestoreClient, session_id: str, max_turns: int = 3) -> str:
    """Fetch the last few messages from a chat session as context."""
    try:
        session = firestore.get_chat_session(session_id)
        if not session:
            return ""
        messages = session.get("messages", [])
        if not messages:
            return ""
        # Take the last N messages (user + assistant turns)
        recent = messages[-max_turns:]
        lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate long assistant responses
            if role == "Assistant" and len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    except Exception:
        logger.debug("chat_context_fetch_failed", session_id=session_id)
        return ""


def _resolve_via_llm(
    gen_pipeline: Any,
    query: str,
    user_docs: list[dict[str, Any]],
    valid_ids: set[str],
    chat_history: str = "",
) -> list[str] | None:
    """Ask the LLM to pick the relevant documents for this query."""
    from app.generation.prompts import PromptManager

    prompts: PromptManager = gen_pipeline._prompts
    template = prompts.get("document_resolve_v1")

    doc_list_str = _format_document_list(
        [d for d in user_docs if d.get("status") == "indexed"]
    )
    if doc_list_str == "(no documents)":
        return list(valid_ids)

    # Format chat history block — only include if non-empty
    history_block = ""
    if chat_history:
        history_block = f"Recent conversation:\n{chat_history}\n\n"

    system_prompt, user_prompt = template.format(
        document_list=doc_list_str,
        query=query,
        chat_history=history_block,
    )

    logger.debug("document_resolver_prompt", doc_list_preview=doc_list_str[:300])

    t0 = time.perf_counter()
    response = gen_pipeline.generate_raw(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=200,
        operation="document_resolve",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(
        "document_resolver_llm_response",
        text_repr=repr(response.text[:300]),
        text_len=len(response.text),
        tokens=response.completion_tokens,
    )

    resolved = _parse_llm_ids(response.text, valid_ids)

    if resolved:
        matched_titles = {
            did: next((d.get("title", "") for d in user_docs if d["id"] == did), "")
            for did in resolved
        }
        logger.info(
            "document_resolver",
            mode="llm",
            resolved=matched_titles,
            ms=round(elapsed, 1),
        )
    else:
        logger.info(
            "document_resolver",
            mode="llm_all",
            raw=response.text[:200],
            ms=round(elapsed, 1),
        )
        resolved = list(valid_ids)

    return resolved
