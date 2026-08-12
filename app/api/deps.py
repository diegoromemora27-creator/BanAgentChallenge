"""
Dependencias y utilidades comunes para la capa de API HTTP.
"""

from typing import Optional
from fastapi import Header, HTTPException
from app.config import settings

def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> bool:
    """
    Verifica obligatoriamente que la petición incluya una API Key válida.
    Lee el valor desde la variable de entorno API_KEY en Render.com o settings.
    Soporta 'Authorization: Bearer <API_KEY>' o 'X-API-Key: <API_KEY>'.
    """
    expected_key = settings.API_KEY.strip() if settings.API_KEY else "banorte_challenge_api_key_2026"
    provided_key = None
    if authorization:
        provided_key = authorization.replace("Bearer ", "").replace("Basic ", "").strip()
    elif x_api_key:
        provided_key = x_api_key.strip()

    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: API Key requerida e inválida. Envíe 'Authorization: Bearer <API_KEY>' o 'X-API-Key: <API_KEY>'.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True
