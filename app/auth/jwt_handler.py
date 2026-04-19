"""
JWT token creation and validation.

Tokens are signed with HS256 by default; the secret **must** be overridden
in production via the ``RAG_JWT_SECRET_KEY`` environment variable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class JWTHandler:
    """Stateless JWT creation / verification."""

    def __init__(self, settings: Settings) -> None:
        self._secret = settings.jwt_secret_key
        self._algorithm = settings.jwt_algorithm
        self._expiration_hours = settings.jwt_expiration_hours

    def create_token(self, user_id: str, email: str, **extra: Any) -> str:
        """Create a signed JWT containing *user_id* and *email*."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "email": email,
            "iat": now,
            "exp": now + timedelta(hours=self._expiration_hours),
            **extra,
        }
        token: str = jwt.encode(payload, self._secret, algorithm=self._algorithm)
        logger.debug("jwt_created", user_id=user_id)
        return token

    def verify_token(self, token: str) -> dict[str, Any]:
        """
        Decode and verify a JWT.

        Raises ``AuthenticationError`` if the token is expired, malformed, or
        has an invalid signature.
        """
        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                options={"require": ["sub", "email", "exp"]},
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid token: {exc}")
