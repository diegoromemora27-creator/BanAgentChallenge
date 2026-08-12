"""
Cliente primario para interactuar con la API de Groq usando la interfaz compatible con Responses API.
Implementa una cadena de fallback multi-modelo dentro de Groq (Llama-3.3-70B -> Llama-3.1-8B-Instant -> Qwen -> ALLaM).
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Se configura el cliente de Groq con un timeout estricto de 12 segundos para evitar cuelgues
groq_client = OpenAI(
    api_key=settings.GROQ_API_KEY or "dummy_key_if_empty",
    base_url="https://api.groq.com/openai/v1",
    timeout=12.0
)

GROQ_MODELS_FALLBACK_ORDER = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
    "allam-2-7b"
]

def call_groq_llm(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Realiza llamadas a Groq API con fallback automático entre sus modelos disponibles.

    Args:
        system_prompt: Instrucciones del sistema y contexto de grounding.
        input_items: Lista de mensajes precedentes [{"role": "user", "content": "..."}, ...].
        model_name: Modelo preferido (opcional). Si falla o se agota su cuota, se intenta el siguiente modelo de Groq.
        temperature: Parámetro de creatividad.
        max_tokens: Límite máximo de tokens de salida.

    Returns:
        Dict con el texto generado y metadatos del proveedor.
    """
    models_to_try = [model_name] if model_name else GROQ_MODELS_FALLBACK_ORDER
    if model_name and model_name not in GROQ_MODELS_FALLBACK_ORDER:
        models_to_try.extend([m for m in GROQ_MODELS_FALLBACK_ORDER if m != model_name])
    elif not model_name:
        models_to_try = GROQ_MODELS_FALLBACK_ORDER

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(input_items)

    last_exception = None

    for current_model in models_to_try:
        try:
            logger.info("Invocando Groq API (Modelo: %s)...", current_model)
            response = groq_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            output_text = response.choices[0].message.content or ""
            if output_text.strip():
                logger.info("Respuesta generada exitosamente con Groq (Modelo: %s).", current_model)
                return {
                    "text": output_text,
                    "provider": f"Groq ({current_model})",
                    "raw": response
                }
        except Exception as exc:
            logger.warning("Groq API falló con modelo '%s': %s. Intentando modelo de respaldo en Groq...", current_model, exc)
            last_exception = exc

    logger.error("Todos los modelos de Groq en la cadena de fallback fallaron.")
    if last_exception:
        raise last_exception
    raise RuntimeError("Fallaron todos los modelos configurados en Groq API.")
