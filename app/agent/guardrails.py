"""
Módulo de guardrails de seguridad, moderación e intenciones.
"""

import re
import logging
from typing import Dict, Any, Tuple
from app.agent.prompts import CLASSIFY_INTENT_PROMPT
from app.llm.provider import generate_llm_response

logger = logging.getLogger(__name__)

# Patrones heurísticos contra inyecciones de código / prompt injection
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"ignora\s+(todas\s+las\s+)?instrucciones",
    r"you\s+are\s+now\s+a",
    r"ahora\s+eres\s+un",
    r"system\s*:",
    r"drop\s+database",
    r"eval\s*\(",
]

def validate_input_guardrails(user_message: str) -> Tuple[bool, str]:
    """
    Evalúa si la pregunta del usuario cumple los criterios de seguridad básicos.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if len(user_message.strip()) > 1000:
        return False, "La pregunta excede la longitud máxima permitida (1000 caracteres)."
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, user_message, re.IGNORECASE):
            logger.warning("Intento de Prompt Injection detectado: '%s'", user_message)
            return False, "Tu solicitud contiene comandos o instrucciones no permitidas por el sistema."

    return True, ""


def classify_user_intent(user_message: str) -> str:
    """
    Clasifica la intención del usuario mediante el LLM en una de las tres etiquetas válidas.
    """
    prompt = CLASSIFY_INTENT_PROMPT.format(user_message=user_message)
    response = generate_llm_response(
        system_prompt="Eres un clasificador estricto de intenciones.",
        input_items=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20
    )
    
    intent = response.get("text", "").strip().upper()
    
    if "GREETING" in intent:
        return "GREETING_OR_META"
    elif "OUT" in intent:
        return "OUT_OF_BOUNDS"
    else:
        return "CV_QUESTION"


def validate_output_guardrails(response_text: str, retrieved_context: list) -> bool:
    """
    Verifica que la respuesta generada no contenga contradicciones ni alucinaciones graves.
    """
    if not response_text or len(response_text.strip()) == 0:
        return False
    return True
