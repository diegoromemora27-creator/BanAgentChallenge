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

    import time
    from app.logging_config import log_interaction_structured

    start_time = time.time()

    llm_out = generate_llm_response(
        system_prompt=system_prompt,
        input_items=input_items,
        temperature=0.2,
        max_tokens=600
    )

    latency_ms = (time.time() - start_time) * 1000
    reply_text = llm_out.get("text", "")
    provider_used = llm_out.get("provider", "Unknown")
    
    # Extrae métricas de tokens si las devuelve el proveedor
    raw_resp = llm_out.get("raw")
    usage_info = {}
    if raw_resp and hasattr(raw_resp, "usage") and raw_resp.usage:
        usage_info = {
            "prompt_tokens": getattr(raw_resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(raw_resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(raw_resp.usage, "total_tokens", 0)
        }

    # Guardrail de salida seguro
    try:
        if not validate_output_guardrails(reply_text, context_chunks):
            reply_text = "No dispongo de información suficiente en el CV para responder con exactitud a tu pregunta."
    except Exception as g_err:
        logger.warning("Error durante la validación de guardrails de salida: %s", g_err)

    state["llm_response"] = reply_text
    state["latency_ms"] = round(latency_ms, 2)
    state["provider"] = provider_used
    state["usage"] = usage_info

    # Registro estructurado del evento de observabilidad
    try:
        log_interaction_structured(
            session_id=state["session_id"],
            query=state["user_message"],
            retrieved_chunks=context_chunks,
            response_text=reply_text,
            latency_ms=latency_ms,
            provider_used=provider_used,
            usage_info=usage_info
        )
    except Exception:
        pass

    # Guardar en memoria de sesión con fallback seguro
    try:
        add_message_to_session(state["session_id"], "user", state["user_message"])
        add_message_to_session(state["session_id"], "assistant", reply_text)
    except Exception as mem_err:
        logger.warning("No se pudo guardar la sesión en memoria/DB: %s", mem_err)

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

    try:
        final_state = agent_executor.invoke(initial_state, config=config)
    except Exception as exec_err:
        logger.warning("Error ejecutando agente con checkpointer persistente (%s). Reintentando con ejecutor efímero...", exec_err)
        # Fallback a compilador en memoria libre sin PostgreSQL
        fallback_executor = workflow.compile(checkpointer=MemorySaver())
        final_state = fallback_executor.invoke(initial_state, config=config)

    # Cálculo de métricas avanzadas de RAG, confiabilidad y costo estimado
    context_chunks = final_state.get("retrieved_context", [])
    top_score = context_chunks[0]["score"] if context_chunks else 0.0
    n_chunks = len(context_chunks)
    
    # Estimación de Confiabilidad / Grounding Score (0.0 a 1.0)
    # Basado en la relevancia semántica de Qdrant y la presencia de contexto
    reliability_score = round(min(1.0, top_score if top_score > 0 else (0.85 if n_chunks > 0 else 0.40)), 2)

    # Estimación de costo en USD ($0.00015 por 1k tokens en Llama 3.1 8B / 70B en HF & Groq)
    usage = final_state.get("usage", {})
    total_tokens = usage.get("total_tokens", 350)
    estimated_cost_usd = round((total_tokens / 1000) * 0.00015, 6)

    return {
        "reply": final_state.get("llm_response", "No se pudo obtener respuesta del agente."),
        "sources": final_state.get("sources", []),
        "metrics": {
            "latency_ms": final_state.get("latency_ms", 0),
            "provider": final_state.get("provider", "Hugging Face"),
            "rag": {
                "chunks_retrieved": n_chunks,
                "top_similarity_score": round(top_score, 3) if top_score else None,
                "reliability_grounding_score": reliability_score
            },
            "financial": {
                "estimated_cost_usd": estimated_cost_usd,
                "pricing_tier": "Free / Open-Source API"
            },
            "usage": usage
        }
    }
