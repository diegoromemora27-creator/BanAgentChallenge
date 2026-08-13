"""
Cliente de fallback para interactuar con Hugging Face Inference API / Router.
Incluye un timeout estricto para evitar cuelgues si Hugging Face entra en Cold Start o Rate Limit.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Se configura el router de Hugging Face mediante SDK compatible de OpenAI con timeout estricto de 8 segundos
hf_client = OpenAI(
    api_key=settings.HF_TOKEN or "dummy_token_if_empty",
    base_url="https://router.huggingface.co/v1",
    timeout=8.0
)

HF_MODELS_FALLBACK_ORDER = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
]

def call_hf_llm_fallback(
    system_prompt: str,
    input_items: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict[str, Any]:
    """
    Ejecuta llamada de respaldo al proveedor Hugging Face Inference con fallback entre modelos.
    """
    models_to_try = [model_name] if model_name else HF_MODELS_FALLBACK_ORDER
    if model_name and model_name not in HF_MODELS_FALLBACK_ORDER:
        models_to_try.extend([m for m in HF_MODELS_FALLBACK_ORDER if m != model_name])

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(input_items)

    last_exception = None
    for current_model in models_to_try:
        try:
            logger.info("Invocando LLM Fallback (Hugging Face: %s)...", current_model)
            response = hf_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            output_text = response.choices[0].message.content or ""
            if output_text.strip():
                return {"text": output_text, "provider": f"Hugging Face ({current_model})", "raw": response}
        except Exception as exc:
            logger.warning("Hugging Face API falló con modelo '%s': %s. Probando siguiente en HF...", current_model, exc)
            last_exception = exc

    logger.error("Todos los modelos de Hugging Face Fallback fallaron.")
    if last_exception:
        raise last_exception
    raise RuntimeError("Fallaron todos los modelos de Hugging Face.")
