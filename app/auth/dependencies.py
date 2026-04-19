"""
FastAPI auth dependencies.

Usage in routes::

    @router.get("/protected")
    async def protected(user: User = Depends(require_auth)):
        ...

    @router.get("/optional")
    async def optional(user: User | None = Depends(get_optional_user)):
        ...
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header

from app.auth.jwt_handler import JWTHandler
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.models.domain import User


def _get_jwt_handler(settings: Settings = Depends(get_settings)) -> JWTHandler:
    return JWTHandler(settings)


def _extract_token(authorization: str | None = Header(default=None)) -> str | None:
    """Pull the Bearer token from the ``Authorization`` header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def get_current_user(
    token: str | None = Depends(_extract_token),
    jwt_handler: JWTHandler = Depends(_get_jwt_handler),
) -> User | None:
    """
    Decode JWT and return a ``User`` if valid, otherwise ``None``.

    Does **not** raise — use ``require_auth`` for endpoints that must be
    authenticated.
    """
    if token is None:
        return None
    try:
        claims = jwt_handler.verify_token(token)
        return User(
            user_id=claims["sub"],
            email=claims["email"],
            display_name=claims.get("display_name", ""),
        )
    except AuthenticationError:
        return None


async def require_auth(
    user: User | None = Depends(get_current_user),
) -> User:
    """Raise ``401`` if the request is not authenticated."""
    if user is None:
        raise AuthenticationError("Authentication required")
    return user


async def get_optional_user(
    user: User | None = Depends(get_current_user),
) -> User | None:
    """Return the user or ``None`` — never raises."""
    return user


# Type aliases for route signatures
RequiredUser = Annotated[User, Depends(require_auth)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
