"""
Cliente primario para interactuar con la API de Groq usando la interfaz compatible con Responses API.
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Se utiliza la base URL de la API OpenAI compatible expuesta por Groq
groq_client = OpenAI(
    api_key=settings.GROQ_API_KEY or "dummy_key_if_empty",
    base_url="https://api.groq.com/openai/v1"
)

def call_groq_llm(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Realiza una llamada a la API de Groq consumiendo su endpoint compatible /v1/responses / completions.

    Args:
        system_prompt: Instrucciones del sistema y contexto de grounding.
        input_items: Lista de mensajes precedentes [{"role": "user", "content": "..."}, ...].
        model_name: Nombre del modelo a utilizar en Groq.
        temperature: Parámetro de creatividad.
        max_tokens: Límite máximo de tokens de salida.

    Returns:
        Dict con el texto generado y metadatos de respuesta.
    """
    logger.info("Invocando LLM principal (Groq: %s)...", model_name)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(input_items)

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_text = response.choices[0].message.content or ""
        return {"text": output_text, "provider": "Groq", "raw": response}
    except Exception as exc:
        logger.error("Error durante la llamada a Groq API: %s", exc)
        raise exc
