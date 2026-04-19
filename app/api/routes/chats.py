"""
Chat session routes — list, get, create, update, delete chat sessions.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_firestore_client
from app.auth.dependencies import RequiredUser
from app.models.responses import (
    ChatMessageResponse,
    ChatSessionResponse,
    ChatSessionSummary,
)

router = APIRouter(prefix="/chats", tags=["chats"])


def _parse_chat_message(m: dict) -> ChatMessageResponse:
    """Build a ``ChatMessageResponse`` from a raw Firestore dict,
    preserving document_ids, document_metadata, and cost."""
    return ChatMessageResponse(
        id=m.get("id", ""),
        role=m.get("role", "user"),
        content=m.get("content", ""),
        timestamp=m.get("timestamp", ""),
        sources=m.get("sources", []),
        latency=m.get("latency"),
        ragas=m.get("ragas"),
        model=m.get("model"),
        cost=m.get("cost"),
        document_ids=m.get("document_ids"),
        document_metadata=m.get("document_metadata"),
    )


class CreateChatRequest(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class UpdateChatRequest(BaseModel):
    title: str | None = None
    messages: list[ChatMessageResponse] | None = None


@router.get("", response_model=list[ChatSessionSummary])
async def list_chats(user: RequiredUser) -> list[ChatSessionSummary]:
    """List all chat sessions for the current user (max 5, newest first)."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    sessions = await loop.run_in_executor(
        None, lambda: fs.list_chat_sessions(user.user_id)
    )
    return [
        ChatSessionSummary(
            id=s["id"],
            title=s.get("title", "Untitled"),
            message_count=len(s.get("messages", [])),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
        )
        for s in sessions
    ]


@router.post("", response_model=ChatSessionResponse)
async def create_chat(body: CreateChatRequest, user: RequiredUser) -> ChatSessionResponse:
    """Create a new chat session. Evicts the oldest if user already has 5."""
    fs = get_firestore_client()
    session_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(
        None,
        lambda: fs.create_chat_session(session_id, user.user_id, body.title),
    )
    return ChatSessionResponse(
        id=data["id"],
        title=data["title"],
        messages=[],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_chat(session_id: str, user: RequiredUser) -> ChatSessionResponse:
    """Get a chat session with all messages."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: fs.get_chat_session(session_id))
    if not data or data.get("user_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Chat session not found")
    raw_msgs = data.get("messages", [])
    messages = [_parse_chat_message(m) for m in raw_msgs]
    return ChatSessionResponse(
        id=data["id"],
        title=data.get("title", "Untitled"),
        messages=messages,
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


@router.put("/{session_id}", response_model=ChatSessionResponse)
async def update_chat(
    session_id: str, body: UpdateChatRequest, user: RequiredUser
) -> ChatSessionResponse:
    """Update chat session title and/or messages."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    existing = await loop.run_in_executor(None, lambda: fs.get_chat_session(session_id))
    if not existing or existing.get("user_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Chat session not found")

    update: dict = {}
    if body.title is not None:
        update["title"] = body.title
    if body.messages is not None:
        update["messages"] = [m.model_dump() for m in body.messages]

    if update:
        await loop.run_in_executor(
            None, lambda: fs.update_chat_session(session_id, update)
        )

    refreshed = await loop.run_in_executor(None, lambda: fs.get_chat_session(session_id))
    raw_msgs = refreshed.get("messages", []) if refreshed else []
    messages = [_parse_chat_message(m) for m in raw_msgs]
    return ChatSessionResponse(
        id=refreshed["id"] if refreshed else session_id,
        title=refreshed.get("title", "Untitled") if refreshed else body.title or "Untitled",
        messages=messages,
        created_at=refreshed.get("created_at", "") if refreshed else "",
        updated_at=refreshed.get("updated_at", "") if refreshed else "",
    )


@router.delete("/{session_id}")
async def delete_chat(session_id: str, user: RequiredUser) -> dict[str, str]:
    """Delete a chat session entirely."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    existing = await loop.run_in_executor(None, lambda: fs.get_chat_session(session_id))
    if not existing or existing.get("user_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Chat session not found")
    await loop.run_in_executor(None, lambda: fs.delete_chat_session(session_id))
    return {"status": "deleted", "session_id": session_id}
