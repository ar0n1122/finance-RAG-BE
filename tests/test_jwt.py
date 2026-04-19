"""Tests for JWT handler."""

from __future__ import annotations

import pytest

from app.auth.jwt_handler import create_token, verify_token
from app.core.exceptions import AuthenticationError


class TestJWT:
    def test_create_and_verify(self):
        token = create_token("user-1", "test@example.com")
        claims = verify_token(token)
        assert claims["sub"] == "user-1"
        assert claims["email"] == "test@example.com"

    def test_invalid_token(self):
        with pytest.raises(AuthenticationError):
            verify_token("invalid.token.here")

    def test_token_is_string(self):
        token = create_token("user-1", "test@example.com")
        assert isinstance(token, str)
        assert len(token) > 20
