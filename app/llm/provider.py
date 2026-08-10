"""
Orquestador resiliente de proveedores LLM. Maneja la llamada principal a Groq
y el failover automático hacia Hugging Face Inference Providers.
"""

import logging
from typing import List, Dict, Any
from app.llm.groq_client import call_groq_llm
from app.llm.hf_client import call_hf_llm_fallback

logger = logging.getLogger(__name__)

def generate_llm_response(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Intenta generar respuesta usando Hugging Face Inference como proveedor primario (para ahorrar cuota).
    En caso de fallo o límite de cuota, utiliza Groq como respaldo.

    Args:
        system_prompt: Prompt del sistema para grounding.
        input_items: Historial/mensajes de la consulta.
        temperature: Temperatura del modelo.
        max_tokens: Cantidad máxima de tokens.

    Returns:
        Dict con {"text": str, "provider": str, "raw": Any}
    """
    try:
        return call_hf_llm_fallback(
            system_prompt=system_prompt,
            input_items=input_items,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as hf_err:
        logger.warning("Hugging Face API falló (%s). Iniciando fallback hacia Groq API...", hf_err)
        try:
            return call_groq_llm(
                system_prompt=system_prompt,
                input_items=input_items,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as groq_err:
            logger.error("Error crítico: Ambos proveedores LLM (Hugging Face y Groq) fallaron: %s", groq_err)
            return {
                "text": "Actualmente me encuentro experimentando problemas técnicos temporales de conexión con los modelos de lenguaje. Por favor intenta de nuevo en unos momentos.",
                "provider": "None",
                "raw": None
            }
