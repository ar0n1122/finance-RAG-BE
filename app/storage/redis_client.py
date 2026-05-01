"""
Async Redis client wrapper — used for rate limiting.

Wraps ``redis.asyncio`` with a minimal interface. All methods are
fail-safe: if Redis is unreachable, callers receive a ``False`` ping
result or a ``RedisUnavailableError`` that they can catch and decide
whether to fail-open or fail-closed.
"""

from __future__ import annotations

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisUnavailableError(Exception):
    """Raised when a Redis operation fails due to connectivity."""


class RedisClient:
    """Thin async wrapper around ``redis.asyncio.Redis``."""

    def __init__(self, url: str) -> None:
        self._url = url
        # Connection pool is created lazily on first command.
        # socket_connect_timeout / socket_timeout keep failures fast.
        self._client: aioredis.Redis = aioredis.Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )

    # ── Atomic counters ───────────────────────────────────────────────────────

    async def incr(self, key: str) -> int:
        """Atomically increment *key* by 1. Creates the key at 0 if absent.

        Returns the new integer value.
        Raises ``RedisUnavailableError`` on connection failure.
        """
        try:
            return int(await self._client.incr(key))
        except RedisError as exc:
            raise RedisUnavailableError(f"Redis incr failed: {exc}") from exc

    async def get_int(self, key: str) -> int:
        """Return the integer value stored at *key*, or ``0`` if the key is absent.

        Raises ``RedisUnavailableError`` on connection failure.
        """
        try:
            val = await self._client.get(key)
            return 0 if val is None else int(val)
        except RedisError as exc:
            raise RedisUnavailableError(f"Redis get failed: {exc}") from exc

    async def delete(self, *keys: str) -> None:
        """Delete one or more keys (no-op if they do not exist).

        Raises ``RedisUnavailableError`` on connection failure.
        """
        try:
            await self._client.delete(*keys)
        except RedisError as exc:
            raise RedisUnavailableError(f"Redis delete failed: {exc}") from exc

    # ── Set operations ────────────────────────────────────────────────────────

    async def sadd(self, key: str, *members: str) -> int:
        """Add one or more *members* to a Redis SET at *key*.

        Returns the number of members actually added (existing members not counted).
        Raises ``RedisUnavailableError`` on connection failure.
        """
        try:
            return int(await self._client.sadd(key, *members))
        except RedisError as exc:
            raise RedisUnavailableError(f"Redis sadd failed: {exc}") from exc

    async def sismember(self, key: str, member: str) -> bool:
        """Return ``True`` if *member* belongs to the SET at *key*.

        Returns ``False`` for a missing key or a non-member.
        Raises ``RedisUnavailableError`` on connection failure.
        """
        try:
            return bool(await self._client.sismember(key, member))
        except RedisError as exc:
            raise RedisUnavailableError(f"Redis sismember failed: {exc}") from exc

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Return ``True`` if Redis responds to PING within the timeout."""
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying connection pool gracefully."""
        try:
            await self._client.aclose()
        except Exception:
            pass
