"""
Fireworks AI LLM provider — OpenAI-compatible endpoint.

API reference: https://docs.fireworks.ai/getting-started/quickstart
Base URL:      https://api.fireworks.ai/inference/v1
Model format:  accounts/fireworks/models/<model-id>
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.core.exceptions import GenerationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.generation.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v3p1"
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class FireworksProvider(LLMProvider):
    """
    Fireworks AI LLM provider.

    Uses the OpenAI-compatible /chat/completions endpoint.
    Set FIREWORKS_API_KEY (or RAG_FIREWORKS_API_KEY) in the environment.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.fireworks_model or _DEFAULT_MODEL
        self._api_key = api_key or settings.fireworks_api_key
        self._base_url = (base_url or settings.fireworks_base_url or _FIREWORKS_BASE_URL).rstrip("/")

    # ── LLMProvider interface ─────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "fireworks"

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
        messages: list[dict] = []
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
                f"Fireworks AI unreachable at {self._base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # Surface auth errors clearly
            if status == 401:
                raise GenerationError(
                    "Fireworks AI: invalid or missing API key (401)"
                ) from exc
            raise GenerationError(
                f"Fireworks AI error: {status} — {exc.response.text[:300]}"
            ) from exc

        data = resp.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage_data = data.get("usage", {})

        logger.debug(
            "fireworks_generation_ok",
            model=self._model,
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
        )

        return LLMResponse(
            text=text.strip(),
            model=data.get("model", self._model),
            provider="fireworks",
            usage={
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        """Check reachability by listing available models."""
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0)) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
