"""
Cliente de fallback para interactuar con Hugging Face Inference API / Router.
Incluye un timeout estricto para evitar cuelgues si Hugging Face entra en Cold Start o Rate Limit.
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Se configura el router de Hugging Face mediante SDK compatible de OpenAI con timeout estricto de 8 segundos
hf_client = OpenAI(
    api_key=settings.HF_TOKEN or "dummy_token_if_empty",
    base_url="https://router.huggingface.co/v1",
    timeout=8.0
)

def call_hf_llm_fallback(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Ejecuta llamada de respaldo al proveedor Hugging Face Inference.

    Args:
        system_prompt: Instrucciones del sistema y contexto.
        input_items: Lista de mensajes de conversación.
        model_name: Modelo en Hugging Face Router.
        temperature: Temperatura de muestreo.
        max_tokens: Máximos tokens a generar.

    Returns:
        Dict con el texto generado y metadatos del proveedor.
    """
    logger.info("Invocando LLM Fallback de respaldo (Hugging Face: %s)...", model_name)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(input_items)

    try:
        response = hf_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_text = response.choices[0].message.content or ""
        return {"text": output_text, "provider": "Hugging Face", "raw": response}
    except Exception as exc:
        logger.error("Error o timeout durante la llamada a Hugging Face Fallback API: %s", exc)
        raise exc
