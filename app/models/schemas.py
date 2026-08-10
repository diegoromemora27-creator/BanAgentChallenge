"""
Definiciones de esquemas de datos Pydantic para el Agente Conversacional de CV.
Incluye modelos para la estructuración del CV, solicitudes/respuestas del chat
y el estándar Open Responses API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==========================================
# 1. Esquemas de Estructuración del CV
# ==========================================

class Experiencia(BaseModel):
    """Representa un bloque de experiencia laboral."""
    id: str = Field(..., description="Identificador único (ej. exp_001)")
    empresa: str = Field(..., description="Nombre de la empresa u organización")
    puesto: str = Field(..., description="Cargo o título ocupado")
    periodo: str = Field(..., description="Rango de fechas o período")
    descripcion: str = Field("", description="Resumen de responsabilidades principales")
    tecnologias: List[str] = Field(default_factory=list, description="Lista de herramientas y tecnologías utilizadas")
    logros: List[str] = Field(default_factory=list, description="Logros relevantes alcanzados")


class Proyecto(BaseModel):
    """Representa un proyecto clave en el CV."""
    id: str = Field(..., description="Identificador único (ej. proj_001)")
    nombre: str = Field(..., description="Nombre del proyecto")
    problema: str = Field("", description="Problema de negocio o técnico a resolver")
    solucion: str = Field("", description="Solución implementada")
    arquitectura: str = Field("", description="Detalles arquitectónicos o stack")
    resultado: str = Field("", description="Impacto o métricas de resultado")


class Perfil(BaseModel):
    """Información general del perfil del candidato."""
    nombre: str = Field(..., description="Nombre completo del profesional")
    resumen: str = Field("", description="Resumen ejecutivo profesional")
    ubicacion: str = Field("", description="Ciudad o país de residencia")


class Skills(BaseModel):
    """Habilidades técnicas y generales."""
    tecnicas: List[str] = Field(default_factory=list, description="Competencias técnicas y lenguajes")
    generales: List[str] = Field(default_factory=list, description="Habilidades blandas o directivas")


class CV(BaseModel):
    """Esquema completo y estructurado del Curriculum Vitae."""
    perfil: Perfil
    experiencia: List[Experiencia] = Field(default_factory=list)
    proyectos: List[Proyecto] = Field(default_factory=list)
    skills: Skills


# ==========================================
# 2. Esquemas para API HTTP (/chat & /cv/upload)
# ==========================================

class ChatRequest(BaseModel):
    """Solicitud enviada al endpoint /chat."""
    message: str = Field(..., description="Mensaje o pregunta del usuario")
    session_id: Optional[str] = Field(None, description="Identificador de sesión conversacional")


class ChatResponse(BaseModel):
    """Respuesta generada por el endpoint /chat."""
    reply: str = Field(..., description="Respuesta del agente conversacional")
    sources: List[str] = Field(default_factory=list, description="Fuentes o trozos de contexto de donde se extrajo la respuesta")


class CVUploadResponse(BaseModel):
    """Respuesta al subir/actualizar el CV en /cv/upload."""
    cv_version: str = Field(..., description="UUID asignado a la versión de CV cargada")
    chunks_ingeridos: int = Field(..., description="Cantidad total de chunks insertados en Qdrant")
    mensaje: str = Field(..., description="Mensaje explicativo del estado de la operación")


# ==========================================
# 3. Esquemas Compatibles con Open Responses API (/v1/responses)
# ==========================================

class ResponseInputItem(BaseModel):
    """Item dentro del arreglo input de Open Responses."""
    role: str = Field("user", description="Rol del emisor: user | assistant | system")
    content: str = Field(..., description="Texto del contenido")


class ResponsesRequest(BaseModel):
    """Solicitud POST enviada a /v1/responses."""
    model: Optional[str] = Field(None, description="Identificador del modelo solicitado")
    input: List[ResponseInputItem] = Field(..., description="Historial de mensajes e instrucciones")
    instructions: Optional[str] = Field(None, description="Instrucciones del sistema opcionales")
    previous_response_id: Optional[str] = Field(None, description="ID de respuesta anterior para encadenar sesión")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Herramientas disponibles (opcional)")
    stream: bool = Field(False, description="Flag para habilitar streaming (por defecto False)")


class OutputContentText(BaseModel):
    """Estructura del texto de salida dentro de OutputItem."""
    type: str = "output_text"
    text: str
    annotations: List[Any] = Field(default_factory=list)


class OutputMessage(BaseModel):
    """Mensaje de respuesta en el estándar Open Responses."""
    type: str = "message"
    id: str
    role: str = "assistant"
    status: str = "completed"
    content: List[OutputContentText]


class OpenResponsesPayload(BaseModel):
    """Payload de respuesta estructurado para /v1/responses."""
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: List[OutputMessage]
    output_text: str
