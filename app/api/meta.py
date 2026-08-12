"""
Router para metadata del agente, tarjeta A2A, health check y observabilidad.
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Header, Query, HTTPException, Request
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
def root():
    """Endpoint raíz que redirige la bienvenida e indica la documentación."""
    return {
        "message": f"Bienvenido a {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": "/health",
        "agent_card": "/.well-known/agent-card.json"
    }


@router.get("/.well-known/agent-card.json", tags=["Agent Metadata (A2A)"])
def get_agent_card(request: Request):
    """
    Endpoint público que expone el manifiesto de la tarjeta de agente A2A.
    """
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("http://") and "localhost" not in base_url and "127.0.0.1" not in base_url:
        base_url = base_url.replace("http://", "https://", 1)

    responses_endpoint = f"{base_url}/responses"
    v1_responses_endpoint = f"{base_url}/v1/responses"

    prompts_list = [
        "¿Cuál es la trayectoria profesional de Diego Romero Mora?",
        "¿Qué proyectos de Inteligencia Artificial y RAG ha construido?",
        "¿Qué tecnologías y lenguajes de programación domina?",
        "¿Tiene experiencia trabajando con FastAPI, LangGraph y Qdrant Cloud?",
        "¿Cuáles son sus principales logros en sus empleos anteriores?",
        "¿Cómo puedo contactar al candidato?"
    ]

    return {
        "name": "Agente Conversacional CV - Diego Romero Mora",
        "description": "Agente RAG estricto para responder sobre la experiencia laboral, habilidades, proyectos y perfil profesional de Diego Romero Mora.",
        "version": settings.VERSION,
        "url": responses_endpoint,
        "responses_url": responses_endpoint,
        "open_responses_url": responses_endpoint,
        "supportedInterfaces": [
            {
                "name": "open_responses",
                "protocol": "open_responses",
                "url": responses_endpoint
            },
            {
                "name": "Open Responses",
                "protocol": "open_responses",
                "url": v1_responses_endpoint
            },
            {
                "type": "open_responses",
                "url": responses_endpoint
            }
        ],
        "interfaces": [
            {
                "name": "open_responses",
                "protocol": "open_responses",
                "url": responses_endpoint
            }
        ],
        "authentication": {
            "type": "bearer",
            "header": "Authorization",
            "description": "Authorization: Bearer <API_KEY>"
        },
        "capabilities": {
            "open_responses": True,
            "streaming": True,
            "file_input": True,
            "image_input": False
        },
        "starter_prompts": prompts_list,
        "prompts": prompts_list,
        "sample_prompts": prompts_list
    }


@router.get("/health", tags=["Health"])
def health_check():
    """
    Endpoint de verificación de estado con pre-warming liviano de dependencias para mitigar Cold Starts.
    """
    qdrant_status = "connected"
    
    try:
        from app.rag.retriever import qdrant_client
        qdrant_client.get_collections()
    except Exception:
        qdrant_status = "warning"

    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "qdrant_status": qdrant_status,
        "timestamp": int(time.time())
    }


@router.get("/metrics", tags=["Observability"])
def prometheus_metrics(
    authorization: Optional[str] = Header(
        default=None,
        description="Header opcional de autorización para Grafana Cloud."
    ),
    token: Optional[str] = Query(
        default=None,
        description="Token de autorización opcional pasable por la URL."
    )
):
    """
    Endpoint nativo de Prometheus para scraping de métricas en Grafana Cloud / Prometheus Server.
    """
    expected_token = settings.METRICS_TOKEN
    provided_token = None

    if authorization:
        provided_token = authorization.replace("Bearer ", "").replace("Basic ", "").strip()
    elif token:
        provided_token = token.strip()

    if provided_token and provided_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Token de métricas inválido.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return {
            "metrics_status": "enabled",
            "info": "Prometheus scraping endpoint active."
        }
