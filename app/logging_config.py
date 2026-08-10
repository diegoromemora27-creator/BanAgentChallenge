"""
Módulo para logging estructurado en formato JSON.
Mide latencias de inferencia, tokens, chunks recuperados y scores.
"""

import json
import logging
import time
from typing import List, Dict, Any

logger = logging.getLogger("cv_agent_structured")

def log_interaction_structured(
    session_id: str,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    response_text: str,
    latency_ms: float,
    provider_used: str = "Hugging Face",
    usage_info: Dict[str, int] = None
):
    """
    Emite un log en formato JSON estructurado a stdout.
    Ideal para observabilidad en Hugging Face Spaces / Render / Langfuse.
    """
    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else None
    
    log_payload = {
        "event": "agent_interaction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "query": query,
        "n_chunks_retrieved": len(retrieved_chunks),
        "top_score": top_score,
        "response_length": len(response_text),
        "latency_ms": round(latency_ms, 2),
        "provider": provider_used,
        "usage": usage_info or {}
    }

    logger.info(json.dumps(log_payload, ensure_ascii=False))
