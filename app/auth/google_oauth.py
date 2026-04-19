"""
Google OAuth 2.0 sign-in service.

Verifies the ID token issued by Google's client-side sign-in flow, creates
or updates the user in Firestore, and returns a platform JWT.
"""

from __future__ import annotations

from typing import Any

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from app.auth.jwt_handler import JWTHandler
from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger
from app.storage.firestore import FirestoreClient

logger = get_logger(__name__)


class GoogleOAuthService:
    """Handles Google sign-in token verification and user provisioning."""

    def __init__(
        self,
        settings: Settings,
        jwt_handler: JWTHandler,
        firestore: FirestoreClient,
    ) -> None:
        self._client_id = settings.google_client_id
        self._jwt = jwt_handler
        self._firestore = firestore

    async def authenticate(self, raw_id_token: str) -> dict[str, Any]:
        """
        Verify a Google ID token and return ``{"access_token": ..., "user": ...}``.

        Steps:
          1. Verify the token with Google's public keys.
          2. Upsert the user in Firestore.
          3. Mint a platform JWT.
        """
        id_info = self._verify_google_token(raw_id_token)

        user_id = id_info["sub"]
        email = id_info.get("email", "")
        display_name = id_info.get("name", "")
        avatar_url = id_info.get("picture", "")

        user_data = self._firestore.upsert_user(user_id, {
            "email": email,
            "display_name": display_name,
            "avatar_url": avatar_url,
        })

        access_token = self._jwt.create_token(
            user_id=user_id,
            email=email,
            display_name=display_name,
        )

        logger.info("google_auth_success", user_id=user_id, email=email)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "user_id": user_id,
                "email": email,
                "display_name": display_name,
                "avatar_url": avatar_url,
                "role": user_data.get("role", "user"),
            },
        }

    def _verify_google_token(self, raw_token: str) -> dict[str, Any]:
        """Verify the Google ID token signature and claims."""
        try:
            id_info: dict[str, Any] = google_id_token.verify_oauth2_token(
                raw_token,
                google_requests.Request(),
                self._client_id,
                clock_skew_in_seconds=10,
            )

            if id_info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
                raise AuthenticationError("Invalid token issuer")

            if not id_info.get("email_verified", False):
                raise AuthenticationError("Email not verified by Google")

            return id_info

        except ValueError as exc:
            logger.warning("google_token_verification_failed", error=str(exc))
            raise AuthenticationError(f"Invalid Google ID token: {exc}")
