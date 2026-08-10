"""
Buffer de memoria conversacional simple por sesión.
Mantiene los últimos N turnos de interacción en memoria volatil.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Diccionario global en memoria: session_id -> list of {"role": str, "content": str}
_SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_TURNS_PER_SESSION = 6

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Obtiene el historial reciente de mensajes de una sesión."""
    return _SESSION_MEMORY.get(session_id, [])

def add_message_to_session(session_id: str, role: str, content: str):
    """Agrega un turno de conversación al buffer de sesión."""
    if session_id not in _SESSION_MEMORY:
        _SESSION_MEMORY[session_id] = []
    
    _SESSION_MEMORY[session_id].append({"role": role, "content": content})

    # Mantiene solo los últimos MAX_TURNS_PER_SESSION turnos
    if len(_SESSION_MEMORY[session_id]) > MAX_TURNS_PER_SESSION:
        _SESSION_MEMORY[session_id] = _SESSION_MEMORY[session_id][-MAX_TURNS_PER_SESSION:]

def clear_session(session_id: str):
    """Limpia el buffer de una sesión especificada."""
    if session_id in _SESSION_MEMORY:
        del _SESSION_MEMORY[session_id]
