"""
Orquestador resiliente de proveedores LLM.
Establece Groq API como proveedor PRIMARIO (con fallback multi-modelo interno: Llama-3.3-70B -> Llama-3.1-8B -> Qwen)
y conmuta a Hugging Face únicamente como último recurso si toda la infraestructura de Groq no responde.
"""

import logging
from typing import List, Dict, Any, Optional
from app.llm.groq_client import call_groq_llm
from app.llm.hf_client import call_hf_llm_fallback

logger = logging.getLogger(__name__)

def generate_llm_response(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Intenta generar respuesta usando Groq API como proveedor primario multi-modelo.
    En caso de falla en Groq, conmuta a Hugging Face Router como último respaldo.
    """
    # 1. Intentar proveedor primario: Groq Multi-Modelo
    try:
        return call_groq_llm(
            system_prompt=system_prompt,
            input_items=input_items,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as groq_err:
        logger.warning("Groq API falló en todos sus modelos (%s). Conmutando a Hugging Face de respaldo...", groq_err)
        
        # 2. Respaldo secundario: Hugging Face (con timeout estricto de 8s)
        try:
            return call_hf_llm_fallback(
                system_prompt=system_prompt,
                input_items=input_items,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as hf_err:
            logger.error("Error crítico: Ambos proveedores LLM (Groq y Hugging Face) fallaron o expiraron: %s", hf_err)
            return {
                "text": "Actualmente el sistema está experimentando una alta demanda de procesamiento. Por favor intenta realizar tu pregunta nuevamente.",
                "provider": "None",
                "raw": None
            }
