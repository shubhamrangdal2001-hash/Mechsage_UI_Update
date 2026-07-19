"""
backend/core/security.py
------------------------
Optional API-key middleware.
When settings.require_api_key is True, every request must carry
  X-API-Key: <value matching settings.api_key>
In dev mode (require_api_key=False), all requests pass through.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    x_api_key: str | None = Security(_api_key_header),
) -> None:
    """FastAPI dependency — inject into any router that needs protection."""
    if not settings.require_api_key:
        return  # dev mode: no auth

    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
