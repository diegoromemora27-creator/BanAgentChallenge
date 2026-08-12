"""
Router para la ingesta y consulta de la Base de Conocimiento RAG (/cv/upload y /cv/info).
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from app.api.deps import verify_api_key
from app.config import settings
from app.models.schemas import CVUploadResponse
from app.rag.ingest import ingest_cv

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
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


@router.get("/cv/info", tags=["Ingestion & Knowledge Base"])
def get_cv_info():
    """
    Endpoint para inspeccionar los datos y chunks actualmente almacenados en Qdrant Cloud.
    """
    from app.rag.retriever import qdrant_client
    try:
        collection_info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
        points_count = collection_info.points_count

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
