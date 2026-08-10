"""
Orquestador conversacional basado en LangGraph.
Define el grafo de estados para el flujo RAG estricto con Guardrails.
"""

import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END

from app.agent.prompts import SYSTEM_GROUNDING_PROMPT
from app.agent.guardrails import (
    validate_input_guardrails,
    classify_user_intent,
    validate_output_guardrails
)
from app.agent.memory import get_session_history, add_message_to_session
from app.rag.retriever import retrieve_cv_context
from app.llm.provider import generate_llm_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. Estado del Grafo (AgentState)
# ==========================================

class AgentState(TypedDict):
    user_message: str
    session_id: str
    intent: str
    is_valid_input: bool
    guardrail_error: str
    retrieved_context: List[Dict[str, Any]]
    llm_response: str
    sources: List[str]


# ==========================================
# 2. Nodos del Grafo
# ==========================================

def node_input_guardrails(state: AgentState) -> AgentState:
    """Nodo 1: Validación de Guardrails de entrada."""
    is_valid, err_msg = validate_input_guardrails(state["user_message"])
    state["is_valid_input"] = is_valid
    state["guardrail_error"] = err_msg
    return state


def node_classify_intent(state: AgentState) -> AgentState:
    """Nodo 2: Clasificación de intención."""
    if not state["is_valid_input"]:
        return state

    intent = classify_user_intent(state["user_message"])
    state["intent"] = intent
    return state


def node_retrieve_context(state: AgentState) -> AgentState:
    """Nodo 3: Tool / Recuperación de contexto desde Qdrant Cloud."""
    intent = state.get("intent", "CV_QUESTION")
    
    if intent == "OUT_OF_BOUNDS":
        state["retrieved_context"] = []
        return state

    if intent == "GREETING_OR_META":
        # Para saludos se recupera solo el perfil
        results = retrieve_cv_context(query=state["user_message"], top_k=2, tipo="perfil")
    else:
        results = retrieve_cv_context(query=state["user_message"], top_k=4)

    state["retrieved_context"] = results
    state["sources"] = [r["texto"] for r in results]
    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Nodo 4: Generación con LLM fundamentado en el contexto (Grounding)."""
    # Si hubo error de guardrail de entrada
    if not state["is_valid_input"]:
        state["llm_response"] = state["guardrail_error"]
        return state

    intent = state.get("intent", "CV_QUESTION")

    if intent == "OUT_OF_BOUNDS":
        state["llm_response"] = "Como agente conversacional enfocado en el perfil profesional del candidato, solo estoy capacitado para responder preguntas sobre su experiencia laboral, proyectos, habilidades e historia profesional."
        return state

    # Construcción de la cadena de contexto
    context_chunks = state.get("retrieved_context", [])
    if not context_chunks:
        context_str = "No se encontraron datos ni evidencias en el CV para la consulta realizada."
    else:
        context_str = "\n---\n".join([c["texto"] for c in context_chunks])

    nombre_candidato = "el Candidato"
    system_prompt = SYSTEM_GROUNDING_PROMPT.format(
        nombre_candidato=nombre_candidato,
        context_str=context_str
    )

    # Carga de memoria histórica
    history = get_session_history(state["session_id"])
    input_items = list(history)
    input_items.append({"role": "user", "content": state["user_message"]})

    llm_out = generate_llm_response(
        system_prompt=system_prompt,
        input_items=input_items,
        temperature=0.2,
        max_tokens=600
    )

    reply_text = llm_out.get("text", "")
    
    # Guardrail de salida
    if not validate_output_guardrails(reply_text, context_chunks):
        reply_text = "No dispongo de información suficiente en el CV para responder con exactitud a tu pregunta."

    state["llm_response"] = reply_text

    # Guardar en memoria
    add_message_to_session(state["session_id"], "user", state["user_message"])
    add_message_to_session(state["session_id"], "assistant", reply_text)

    return state


# ==========================================
# 3. Construcción del Grafo LangGraph con Checkpointer (Supabase / Memory)
# ==========================================

from langgraph.checkpoint.memory import MemorySaver
from app.config import settings

checkpointer = None

if settings.DATABASE_URL:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool
        
        logger.info("Conectando a PostgreSQL (Neon) para persistencia de Checkpoints...")
        # autocommit=True es requerido por Neon/PostgreSQL para crear índices concurrentes durante checkpointer.setup()
        connection_pool = ConnectionPool(
            conninfo=settings.DATABASE_URL,
            max_size=10,
            kwargs={"connect_timeout": 5, "autocommit": True}
        )
        checkpointer = PostgresSaver(connection_pool)
        checkpointer.setup()
        logger.info("Checkpointer Postgres (Neon) configurado e inicializado correctamente.")
    except Exception as exc:
        logger.warning("No se pudo conectar a PostgreSQL (%s). Usando MemorySaver por defecto.", exc)
        checkpointer = MemorySaver()
else:
    logger.info("DATABASE_URL no configurada. Usando MemorySaver para estados del agente.")
    checkpointer = MemorySaver()

workflow = StateGraph(AgentState)

# Adición de Nodos
workflow.add_node("guardrails_input", node_input_guardrails)
workflow.add_node("classify_intent", node_classify_intent)
workflow.add_node("retrieve_context", node_retrieve_context)
workflow.add_node("generate_response", node_generate_response)

# Definición de Aristas (Edges)
workflow.set_entry_point("guardrails_input")
workflow.add_edge("guardrails_input", "classify_intent")
workflow.add_edge("classify_intent", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)

# Compilación del Grafo Executable con el Checkpointer configurado
agent_executor = workflow.compile(checkpointer=checkpointer)


def run_agent_workflow(message: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Función de entrada principal para invocar el flujo agéntico completo.
    """
    initial_state: AgentState = {
        "user_message": message,
        "session_id": session_id,
        "intent": "",
        "is_valid_input": True,
        "guardrail_error": "",
        "retrieved_context": [],
        "llm_response": "",
        "sources": []
    }

    # Configuración de sesión / thread en LangGraph
    config = {"configurable": {"thread_id": session_id}}

    final_state = agent_executor.invoke(initial_state, config=config)

    return {
        "reply": final_state["llm_response"],
        "sources": final_state.get("sources", [])
    }
