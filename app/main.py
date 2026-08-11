"""
Aplicación principal FastAPI para el Agente Conversacional de CV.
Expone los endpoints /health, /chat, /v1/responses (Open Responses) y /cv/upload.
"""

import time
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CVUploadResponse,
    ResponsesRequest,
    OpenResponsesPayload,
    OutputMessage,
    OutputContentText
)
from app.agent.graph import run_agent_workflow
from app.rag.ingest import ingest_cv
from app.agent.memory import get_session_history

# Configuración del Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Depends, Header, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configuración del Limiter para protección de cuota (60 peticiones/minuto por IP)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para el Agente Conversacional de CV basado en RAG estricto, Qdrant Cloud y estándar Open Responses."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Endpoints HTTP
# ==========================================

@app.get("/", tags=["Health"])
def root():
    """Endpoint raíz que redirige la bienvenida e indica la documentación."""
    return {
        "message": f"Bienvenido a {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Endpoint de verificación de estado con pre-warming liviano de dependencias para mitigar Cold Starts.
    """
    db_status = "connected"
    qdrant_status = "connected"
    
    # Pre-warming liviano en background
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


@app.get("/metrics", tags=["Observability"])
def prometheus_metrics(
    authorization: Optional[str] = Header(
        default=None,
        description="Header opcional de autorización para Grafana Cloud (Ejemplo: 'Bearer banorte_metrics_secret_token_2026')."
    ),
    token: Optional[str] = Query(
        default=None,
        description="Token de autorización opcional pasable por la URL (Ejemplo: 'banorte_metrics_secret_token_2026')."
    )
):
    """
    Endpoint nativo de Prometheus para scraping de métricas en Grafana Cloud / Prometheus Server.
    Expone contadores de tokens, latencia, confiabilidad, costo estimado y métricas por nodo de LangGraph.
    """
    expected_token = settings.METRICS_TOKEN
    provided_token = None

    if authorization:
        provided_token = authorization.replace("Bearer ", "").replace("Basic ", "").strip()
    elif token:
        provided_token = token.strip()

    # Si se envía un token inválido, rechaza con 401; si no se envía nada, permite visualización directa en docs/navegador
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


@app.post("/chat", response_model=ChatResponse, tags=["Agent Chat"])
@limiter.limit("60/minute")
def chat_endpoint(request: Request, req: ChatRequest):
    """
    Endpoint de conversación simplificado para clientes Web o Chat UI con Rate Limiting.
    """
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
    logger.info("=== [NUEVA PETICIÓN EN /CHAT] ===")
    logger.info("Sesión ID: %s | Mensaje recibido: '%s'", session_id, req.message)

    try:
        result = run_agent_workflow(message=req.message, session_id=session_id)
        reply = result["reply"]
        sources_count = len(result.get("sources", []))
        metrics = result.get("metrics", {})
        
        logger.info("=== [RESPUESTA DE /CHAT COMPLETADA] ===")
        logger.info("Sesión ID: %s | Longitud Respuesta: %d caracteres | Fuentes RAG: %d | Provider: %s", 
                    session_id, len(reply), sources_count, metrics.get("provider", "N/A"))

        return ChatResponse(
            reply=reply,
            sources=result.get("sources", []),
            metrics=metrics
        )
    except Exception as exc:
        import traceback
        err_trace = traceback.format_exc()
        logger.error("=== [ERROR CRÍTICO EN /CHAT] ===\n%s", err_trace)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(exc)}")


@app.post("/v1/responses", response_model=OpenResponsesPayload, tags=["Open Responses API Standard"])
@limiter.limit("60/minute")
def create_response(request: Request, req: ResponsesRequest):
    """
    Endpoint interoperable compatible con la especificación abierta Open Responses (openresponses.org).
    Mapea nombres de modelo arbitrarios ("gpt-4", etc.) a cv-agent-v1 y maneja stream=True según la especificación.
    """
    if req.stream:
        # El spec de Open Responses indica rechazar solicitudes de streaming si el servidor solo opera en sync
        raise HTTPException(
            status_code=400,
            detail="Streaming mode (stream=true) is not supported by this agent endpoint. Please set stream=false."
        )

    session_id = req.previous_response_id or f"session_{uuid.uuid4().hex[:8]}"
    
    # Extraer el último mensaje del usuario desde el array de input
    if not req.input:
        raise HTTPException(status_code=400, detail="El parámetro 'input' no puede estar vacío.")

    last_user_message = req.input[-1].content

    try:
        result = run_agent_workflow(message=last_user_message, session_id=session_id)
        reply_text = result["reply"]
        metrics = result.get("metrics", {})

        response_payload = OpenResponsesPayload(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=req.model or "cv-agent-v1",
            output=[
                OutputMessage(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[OutputContentText(text=reply_text)]
                )
            ],
            output_text=reply_text,
            usage=metrics.get("usage", {})
        )
        return response_payload

    except Exception as exc:
        logger.error("Error en endpoint Open Responses /v1/responses: %s", exc)
        raise HTTPException(status_code=500, detail="Error interno en el servidor Open Responses.")


@app.post(
    "/cv/upload",
    response_model=CVUploadResponse,
    tags=["Ingestion & Knowledge Base"]
)
async def upload_cv_endpoint(
    file: Optional[UploadFile] = File(default=None),
    pasted_text: Optional[str] = Form(default=None),
):
    """
    Endpoint libre para ingesta y actualización del CV (PDF, TXT o texto pegado).
    """
    if file is None and not pasted_text:
        raise HTTPException(
            status_code=400,
            detail="Debes proporcionar un archivo subido (PDF/TXT) o enviar el campo 'pasted_text'."
        )

    try:
        result = ingest_cv(file=file, pasted_text=pasted_text)
        return CVUploadResponse(
            cv_version=result["cv_version"],
            chunks_ingeridos=result["chunks_ingeridos"],
            mensaje="CV actualizado e indexado correctamente en Qdrant Cloud. El agente utilizará esta nueva versión."
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.error("Error crítico durante la ingesta del CV: %s", exc)
        raise HTTPException(status_code=500, detail="No se pudo procesar e indexar el CV.")


@app.get("/cv/info", tags=["Ingestion & Knowledge Base"])
def get_cv_info():
    """
    Endpoint para inspeccionar los datos y chunks actualmente almacenados en Qdrant Cloud.
    """
    from app.rag.retriever import qdrant_client
    try:
        # Obtiene la cantidad de puntos almacenados
        collection_info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
        points_count = collection_info.points_count

        # Desplaza hasta 20 puntos para inspeccionar el contenido real
        scroll_res = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        
        stored_chunks = [
            {
                "id": p.id,
                "texto": p.payload.get("texto", ""),
                "tipo": p.payload.get("tipo", ""),
                "cv_version": p.payload.get("cv_version", "")
            }
            for p in scroll_res[0]
        ]

        return {
            "collection_name": settings.QDRANT_COLLECTION_NAME,
            "total_chunks": points_count,
            "status": collection_info.status,
            "chunks_inspeccion": stored_chunks
        }
    except Exception as exc:
        logger.error("Error al consultar info de Qdrant: %s", exc)
        raise HTTPException(status_code=500, detail=f"No se pudo consultar Qdrant Cloud: {str(exc)}")
