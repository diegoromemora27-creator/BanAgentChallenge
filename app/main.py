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

# Inicialización de la App FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para el Agente Conversacional de CV basado en RAG estricto, Qdrant Cloud y estándar Open Responses."
)

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

@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint de verificación de estado y disponibilidad del servicio."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": int(time.time())
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent Chat"])
def chat_endpoint(req: ChatRequest):
    """
    Endpoint de conversación simplificado para clientes Web o Chat UI.
    """
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
    logger.info("Procesando consulta en /chat para sesión %s", session_id)

    try:
        result = run_agent_workflow(message=req.message, session_id=session_id)
        return ChatResponse(
            reply=result["reply"],
            sources=result.get("sources", [])
        )
    except Exception as exc:
        logger.error("Error procesando solicitud en /chat: %s", exc)
        raise HTTPException(status_code=500, detail="Error interno al procesar la respuesta del agente.")


@app.post("/v1/responses", response_model=OpenResponsesPayload, tags=["Open Responses API Standard"])
def create_response(req: ResponsesRequest):
    """
    Endpoint interoperable compatible con la especificación abierta Open Responses (openresponses.org).
    """
    session_id = req.previous_response_id or f"session_{uuid.uuid4().hex[:8]}"
    
    # Extraer el último mensaje del usuario desde el array de input
    if not req.input:
        raise HTTPException(status_code=400, detail="El parámetro 'input' no puede estar vacío.")

    last_user_message = req.input[-1].content

    try:
        result = run_agent_workflow(message=last_user_message, session_id=session_id)
        reply_text = result["reply"]

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
            output_text=reply_text
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
