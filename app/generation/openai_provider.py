"""
OpenAI-compatible LLM provider — works with OpenAI API and any compatible server.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.core.exceptions import GenerationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.generation.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider (works with OpenAI, Azure OpenAI, vLLM, etc.)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_fallback_model or "gpt-4o-mini"
        self._base_url = (base_url or settings.llm_fallback_base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or settings.llm_fallback_api_key or ""

    @property
    def provider_name(self) -> str:
        return "openai"

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(f"OpenAI API unreachable at {self._base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(f"OpenAI API error: {exc.response.status_code} — {exc.response.text[:200]}") from exc

        data = resp.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage_data = data.get("usage", {})

        return LLMResponse(
            text=text.strip(),
            model=self._model,
            provider="openai",
            usage={
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(f"{self._base_url}/models", headers=headers)
                return resp.status_code == 200
        except Exception:
            return False
