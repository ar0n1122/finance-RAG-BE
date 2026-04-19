"""Tests for prompt management."""

from __future__ import annotations

import pytest

from app.generation.prompts import PromptManager, PromptTemplate


class TestPromptManager:
    def setup_method(self):
        self.pm = PromptManager()

    def test_v1_templates_loaded(self):
        templates = self.pm.list_templates()
        assert "qa_v1" in templates
        assert "qa_v2" in templates
        assert "reflection_v1" in templates
        assert "grading_v1" in templates
        assert "query_rewrite_v1" in templates
        assert "hallucination_check_v1" in templates

    def test_qa_v1_format(self):
        t = self.pm.get("qa_v1")
        system, user = t.format(context="Some context", question="What is revenue?")
        assert "context" not in system or "context" in system.lower()
        assert "What is revenue?" in user
        assert "Some context" in user

    def test_missing_template(self):
        with pytest.raises(KeyError):
            self.pm.get("nonexistent_v99")

    def test_missing_vars(self):
        t = self.pm.get("qa_v1")
        with pytest.raises(ValueError):
            t.format(context="ctx")  # Missing 'question'

    def test_register_custom(self):
        custom = PromptTemplate(
            name="custom_v1",
            system="You are a custom bot.",
            user="{query}",
            required_vars=["query"],
        )
        self.pm.register(custom)
        assert "custom_v1" in self.pm.list_templates()
        t = self.pm.get("custom_v1")
        _, user = t.format(query="Hello")
        assert user == "Hello"
