"""
 Auth routes — Google OAuth sign-in.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_firestore_client
from app.auth.google_oauth import GoogleOAuthService
from app.auth.jwt_handler import JWTHandler
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.models.requests import GoogleAuthRequest
from app.models.responses import AuthResponse, UserResponse
from app.storage.firestore import FirestoreClient

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=AuthResponse)
async def google_sign_in(
    body: GoogleAuthRequest,
    settings: Settings = Depends(get_settings),
    firestore: FirestoreClient = Depends(get_firestore_client),
) -> AuthResponse:
    """Sign in with Google OAuth ID token."""
    try:
        jwt_handler = JWTHandler(settings)
        service = GoogleOAuthService(settings, jwt_handler, firestore)
        result = await service.authenticate(body.id_token)
        return AuthResponse(
            access_token=result["access_token"],
            token_type=result["token_type"],
            user=UserResponse(
                user_id=result["user"]["user_id"],
                email=result["user"]["email"],
                display_name=result["user"]["display_name"],
                avatar_url=result["user"]["avatar_url"],
                role=result["user"]["role"],
            ),
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
