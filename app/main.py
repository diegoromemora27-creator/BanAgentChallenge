"""
Aplicación principal FastAPI para el Agente Conversacional de CV.
Expone los endpoints /health, /chat, /v1/responses (Open Responses) y /cv/upload.
"""

import time
import uuid
import json
import logging
from typing import Optional, Any
from fastapi import FastAPI, Depends, Header, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CVUploadResponse,
    ResponsesRequest,
    OpenResponsesPayload,
    OutputMessage,
    OutputContentText,
    AgentCardSchema,
    AgentAuthentication,
    AgentCapabilities,
    AgentInterface
)
from app.agent.graph import run_agent_workflow
from app.rag.ingest import ingest_cv
from app.agent.memory import get_session_history

# Configuración del Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración del Limiter para protección de cuota (60 peticiones/minuto por IP)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para el Agente Conversacional de CV basado en RAG estricto, Qdrant Cloud y estándar Open Responses."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuración de CORS universal para permitir peticiones desde navegadores y la plataforma del Hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://hackathon-2024.com",
        "http://hackathon-2024.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
def auto_ingest_default_cv():
    """Ingesta automáticamente el CV de Diego Romero Mora (data/cv_sample.txt) si Qdrant está vacío."""
    try:
        from app.rag.retriever import qdrant_client, ensure_collection_exists
        ensure_collection_exists()
        info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
        if info.points_count == 0:
            logger.info("Qdrant está vacío. Auto-ingestionando CV de Diego Romero Mora desde data/cv_sample.txt...")
            with open("data/cv_sample.txt", "r", encoding="utf-8") as f:
                pasted_text = f.read()
            ingest_cv(pasted_text=pasted_text)
            logger.info("CV de Diego Romero Mora indexado exitosamente en Qdrant.")
        else:
            logger.info("Qdrant ya contiene %d puntos indexados para el CV.", info.points_count)
    except Exception as exc:
        logger.warning("No se pudo verificar o autoingestar el CV en startup: %s", exc)


def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
):
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


# ==========================================
# Endpoints HTTP
# ==========================================

@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
def root():
    """Endpoint raíz que redirige la bienvenida e indica la documentación."""
    return {
        "message": f"Bienvenido a {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": "/health",
        "agent_card": "/.well-known/agent-card.json"
    }


@app.get("/.well-known/agent-card.json", tags=["Agent Metadata (A2A)"])
def get_agent_card(request: Request):
    """
    Endpoint público que expone el manifiesto de la tarjeta de agente A2A.
    Usado por plataformas como Parley o catálogos A2A para autocompletar formularios y descubrir el agente.
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


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)], tags=["Agent Chat"])
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


def _extract_text_from_content(content: Any) -> str:
    """Extrae texto de manera segura desde content (string, lista de dicts o estructuras multimediales)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ["input_text", "text"] and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") in ["input_file", "file"]:
                    filename = item.get("file_name") or item.get("name") or "adjunto"
                    parts.append(f"[Archivo adjunto recibido: {filename}]")
        return " ".join(parts) if parts else str(content)
    return str(content)


@app.post("/v1/responses", dependencies=[Depends(verify_api_key)], tags=["Open Responses API Standard"])
@app.post("/responses", dependencies=[Depends(verify_api_key)], tags=["Open Responses API Standard"])
@limiter.limit("60/minute")
def create_response(request: Request, req: ResponsesRequest):
    """
    Endpoint interoperable compatible con la especificación abierta Open Responses (openresponses.org).
    Soporta rutas /v1/responses y /responses, streaming SSE cuando stream=true y procesamiento síncrono.
    """
    session_id = (
        req.previous_response_id or 
        req.previousResponseId or 
        req.session_id or 
        req.sessionId or 
        f"session_{uuid.uuid4().hex[:8]}"
    )

    logger.info(
        "=== [PARLEY DIAGNÓSTICO SESIÓN] session_id: %s | previous_response_id: %s | previousResponseId: %s | session_id_req: %s ===",
        session_id, req.previous_response_id, req.previousResponseId, req.session_id
    )

    # Extraer el último mensaje del usuario desde el array de input
    if not req.input:
        raise HTTPException(status_code=400, detail="El parámetro 'input' no puede estar vacío.")

    last_user_message_raw = req.input[-1].content
    last_user_message = _extract_text_from_content(last_user_message_raw)

    if not last_user_message.strip():
        last_user_message = "Hola"

    # Si se solicita streaming (stream=True), responder usando Server-Sent Events (SSE) con el estándar oficial de Open Responses
    if req.stream:
        def sse_event_generator():
            resp_id = f"resp_{uuid.uuid4().hex[:12]}"
            msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            seq_num = 0

            def next_seq():
                nonlocal seq_num
                seq_num += 1
                return seq_num

            # 1. Evento response.created
            created_evt = {
                "type": "response.created",
                "sequence_number": next_seq(),
                "response": {
                    "id": resp_id,
                    "model": req.model or "cv-agent-v1",
                    "status": "in_progress"
                }
            }
            yield f"event: response.created\ndata: {json.dumps(created_evt)}\n\n"

            # 2. Evento response.output_item.added
            msg_added_evt = {
                "type": "response.output_item.added",
                "sequence_number": next_seq(),
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }
            yield f"event: response.output_item.added\ndata: {json.dumps(msg_added_evt)}\n\n"

            # 3. Evento response.content_part.added
            content_part_added_evt = {
                "type": "response.content_part.added",
                "sequence_number": next_seq(),
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "text": ""
                }
            }
            yield f"event: response.content_part.added\ndata: {json.dumps(content_part_added_evt)}\n\n"

            try:
                # 4. EJECUTAR EL GRAFO AGÉNTICO EN SEGUNDO PLANO
                result = run_agent_workflow(
                    message=last_user_message,
                    session_id=session_id,
                    system_instructions=req.instructions
                )
                reply_text = result["reply"]

                # 5. EMITIR FRAMES response.output_text.delta (REPETIDOS N VECES)
                words = reply_text.split(" ")
                chunk_size = 4
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i+chunk_size])
                    if i + chunk_size < len(words):
                        chunk_text += " "
                    
                    delta_evt = {
                        "type": "response.output_text.delta",
                        "sequence_number": next_seq(),
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk_text
                    }
                    yield f"event: response.output_text.delta\ndata: {json.dumps(delta_evt)}\n\n"
                    time.sleep(0.01)

                # 6. Evento response.output_text.done
                text_done_evt = {
                    "type": "response.output_text.done",
                    "sequence_number": next_seq(),
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": reply_text
                }
                yield f"event: response.output_text.done\ndata: {json.dumps(text_done_evt)}\n\n"

                # 7. Evento response.content_part.done
                part_done_evt = {
                    "type": "response.content_part.done",
                    "sequence_number": next_seq(),
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "annotations": [],
                        "text": reply_text
                    }
                }
                yield f"event: response.content_part.done\ndata: {json.dumps(part_done_evt)}\n\n"

                # 8. Evento response.output_item.done
                item_done_evt = {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": reply_text
                            }
                        ]
                    }
                }
                yield f"event: response.output_item.done\ndata: {json.dumps(item_done_evt)}\n\n"

                # 9. Evento response.completed
                completed_evt = {
                    "type": "response.completed",
                    "sequence_number": next_seq(),
                    "response": {
                        "id": resp_id,
                        "model": req.model or "cv-agent-v1",
                        "status": "completed",
                        "output": [
                            {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": reply_text}]
                            }
                        ],
                        "output_text": reply_text
                    }
                }
                yield f"event: response.completed\ndata: {json.dumps(completed_evt)}\n\n"

                # 10. Evento terminal estricto data: [DONE]
                yield "data: [DONE]\n\n"

            except Exception as stream_err:
                logger.error("Error en streaming SSE de Open Responses: %s", stream_err)
                err_evt = {
                    "type": "error",
                    "sequence_number": next_seq(),
                    "error": {"message": str(stream_err)}
                }
                yield f"event: error\ndata: {json.dumps(err_evt)}\n\n"

        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

    try:
        result = run_agent_workflow(
            message=last_user_message,
            session_id=session_id,
            system_instructions=req.instructions
        )
        reply_text = result["reply"]
        metrics = result.get("metrics", {})

        choices_payload = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text
                },
                "finish_reason": "stop"
            }
        ]

        response_payload = OpenResponsesPayload(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=req.model or "cv-agent-v1",
            choices=choices_payload,
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
        logger.error("Error en endpoint Open Responses /responses: %s", exc)
        raise HTTPException(status_code=500, detail="Error interno en el servidor Open Responses.")


@app.post(
    "/cv/upload",
    response_model=CVUploadResponse,
    dependencies=[Depends(verify_api_key)],
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
