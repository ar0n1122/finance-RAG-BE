"""
OpenRouter LLM provider — OpenAI-compatible API at https://openrouter.ai.

Supports hundreds of models (including free tiers) via a unified API.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.core.exceptions import GenerationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.generation.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider — routes to the best available backend."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.openrouter_model
        self._base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._api_key = api_key or settings.openrouter_api_key

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers["HTTP-Referer"] = "http://localhost:8000"
        headers["X-Title"] = "Enterprise RAG"
        return headers

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(
                f"OpenRouter unreachable at {self._base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"OpenRouter API error: {exc.response.status_code} — "
                f"{exc.response.text[:200]}"
            ) from exc

        data = resp.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage_data = data.get("usage", {})

        return LLMResponse(
            text=text.strip(),
            model=data.get("model", self._model),
            provider="openrouter",
            usage={
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False
