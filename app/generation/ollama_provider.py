"""
Ollama LLM provider — local inference via Ollama REST API.
"""

from __future__ import annotations

import json
import re

import httpx

from app.core.config import get_settings
from app.core.exceptions import GenerationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.generation.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)

# Patterns for extracting structured answers from thinking content
_JSON_ARRAY_RE = re.compile(r"\[.*?]", re.DOTALL)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_APPROVED_RE = re.compile(r"\b(APPROVED|NEEDS_CORRECTION)\b")


def _extract_from_thinking(thinking: str) -> str:
    """
    Best-effort extraction of the intended answer from the thinking field
    of a thinking model (DeepSeek-R1, QwQ, etc.) when content is empty.

    Only extracts when a known structured pattern is found — JSON arrays,
    yes/no, APPROVED/NEEDS_CORRECTION. Returns empty string otherwise,
    so callers can apply their own empty-response logic (e.g. benefit of
    the doubt).
    """
    if not thinking:
        return ""

    # Check for JSON array (document resolver output)
    json_match = _JSON_ARRAY_RE.search(thinking)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return json_match.group()
        except (json.JSONDecodeError, ValueError):
            pass

    # Check for APPROVED / NEEDS_CORRECTION (reflection)
    approved_match = _APPROVED_RE.search(thinking)
    if approved_match:
        return approved_match.group()

    # Check for yes/no (grading, hallucination)
    # Use the LAST occurrence since it's more likely the conclusion
    yes_no_matches = list(_YES_NO_RE.finditer(thinking))
    if yes_no_matches:
        return yes_no_matches[-1].group().lower()

    # No structured pattern found — return empty so callers use their
    # default handling (benefit of the doubt).
    return ""


class OllamaProvider(LLMProvider):
    """Local Ollama LLM via HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.ollama_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(f"Ollama unreachable at {self._base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(f"Ollama API error: {exc.response.status_code}") from exc

        data = resp.json()
        message = data.get("message", {})
        text = message.get("content", "")

        # Thinking-model fallback: if content is empty but a thinking
        # field exists (e.g. DeepSeek-R1, QwQ), extract the answer from it.
        if not text and message.get("thinking"):
            thinking = message["thinking"]
            text = _extract_from_thinking(thinking)
            if text:
                logger.debug(
                    "ollama_thinking_fallback",
                    thinking_len=len(thinking),
                    extracted_len=len(text),
                )
            else:
                logger.warning(
                    "ollama_empty_content",
                    eval_count=data.get("eval_count"),
                    done_reason=data.get("done_reason"),
                    thinking_preview=thinking[:200],
                )
        elif not text and data.get("eval_count", 0) > 0:
            logger.warning(
                "ollama_empty_content",
                eval_count=data.get("eval_count"),
                message_keys=list(message.keys()),
                done_reason=data.get("done_reason"),
                raw_message=repr(message)[:300],
            )
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

        return LLMResponse(
            text=text.strip(),
            model=self._model,
            provider="ollama",
            usage=usage,
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
