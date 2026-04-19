"""
Prompt management — versioned prompt templates with YAML persistence.

All prompt content lives in YAML files under ``generation/prompts/`` (e.g.
``v1.yaml``).  This module provides the ``PromptTemplate`` data class and the
``PromptManager`` registry that loads and serves those templates.

To iterate on prompts:
  1. Copy ``v1.yaml`` → ``v2.yaml`` and edit.
  2. Keep or change template names — last-loaded file wins (alphabetical).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.core.logging import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptTemplate:
    """A single versioned prompt template with variable slots."""

    def __init__(
        self,
        name: str,
        system: str,
        user: str,
        *,
        version: str = "1",
        description: str = "",
        required_vars: list[str] | None = None,
    ) -> None:
        self.name = name
        self.system = system
        self.user = user
        self.version = version
        self.description = description
        self.required_vars = required_vars or []

    def format(self, **kwargs: Any) -> tuple[str, str]:
        """
        Format *system* and *user* templates with the given variables.

        Returns (system_prompt, user_prompt).
        """
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Prompt '{self.name}' missing vars: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


class PromptManager:
    """
    Registry of prompt templates — loads a specific version file from
    ``prompts/`` dir based on ``settings.prompt_version``.
    """

    def __init__(self, version: str | None = None) -> None:
        from app.core.config import get_settings

        self._version = version or get_settings().prompt_version
        self._templates: dict[str, PromptTemplate] = {}
        self._load_from_yaml()
        if not self._templates:
            logger.warning("no_prompts_loaded", version=self._version, prompts_dir=str(_PROMPTS_DIR))

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(f"Prompt template '{name}' not found. Available: {list(self._templates)}")
        return self._templates[name]

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template
        logger.debug("prompt_registered", name=template.name, version=template.version)

    def list_templates(self) -> list[str]:
        return sorted(self._templates)

    # ── YAML loader ───────────────────────────────────────────────────────────

    def _load_from_yaml(self) -> None:
        """Load prompt templates from the configured version file."""
        if not _PROMPTS_DIR.is_dir():
            return

        # Look for the exact version file first (e.g. v1.yaml)
        target = _PROMPTS_DIR / f"{self._version}.yaml"
        if not target.exists():
            target = _PROMPTS_DIR / f"{self._version}.yml"
        if not target.exists():
            logger.error("prompt_version_not_found", version=self._version, prompts_dir=str(_PROMPTS_DIR))
            return

        try:
            with open(target, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return
            for name, spec in data.items():
                self.register(PromptTemplate(
                    name=name,
                    system=spec.get("system", ""),
                    user=spec.get("user", ""),
                    version=str(spec.get("version", "1")),
                    description=spec.get("description", ""),
                    required_vars=spec.get("required_vars", []),
                ))
            logger.info("prompts_loaded", file=target.name, version=self._version, count=len(data))
        except Exception:
            logger.warning("prompts_yaml_error", file=target.name, exc_info=True)
