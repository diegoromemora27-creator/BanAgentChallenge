"""
Orquestador conversacional basado en LangGraph.
Define el grafo de estados para el flujo RAG estricto con Guardrails.
"""

import os
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
    from app.metrics import NODE_EXECUTION_DURATION_SECONDS, NODE_ERRORS_TOTAL
    with NODE_EXECUTION_DURATION_SECONDS.labels(node_name="guardrails_input").time():
        try:
            is_valid, err_msg = validate_input_guardrails(state["user_message"])
            state["is_valid_input"] = is_valid
            state["guardrail_error"] = err_msg
        except Exception:
            NODE_ERRORS_TOTAL.labels(node_name="guardrails_input").inc()
            raise
    return state


def node_classify_intent(state: AgentState) -> AgentState:
    """Nodo 2: Clasificación de intención."""
    from app.metrics import NODE_EXECUTION_DURATION_SECONDS, NODE_ERRORS_TOTAL
    with NODE_EXECUTION_DURATION_SECONDS.labels(node_name="classify_intent").time():
        try:
            if not state["is_valid_input"]:
                return state
            intent = classify_user_intent(state["user_message"])
            state["intent"] = intent
        except Exception:
            NODE_ERRORS_TOTAL.labels(node_name="classify_intent").inc()
            raise
    return state


def node_retrieve_context(state: AgentState) -> AgentState:
    """Nodo 3: Tool / Recuperación de contexto desde Qdrant Cloud."""
    from app.metrics import NODE_EXECUTION_DURATION_SECONDS, RETRIEVAL_LATENCY_SECONDS, TOOL_INVOCATIONS_TOTAL, RAG_RETRIEVED_DOCUMENTS_COUNT, NODE_ERRORS_TOTAL
    with NODE_EXECUTION_DURATION_SECONDS.labels(node_name="retrieve_context").time():
        try:
            intent = state.get("intent", "CV_QUESTION")
            if intent == "OUT_OF_BOUNDS":
                state["retrieved_context"] = []
                RAG_RETRIEVED_DOCUMENTS_COUNT.set(0)
                return state

            TOOL_INVOCATIONS_TOTAL.labels(tool_name="qdrant_vector_search").inc()
            
            with RETRIEVAL_LATENCY_SECONDS.time():
                if intent == "GREETING_OR_META":
                    results = retrieve_cv_context(query=state["user_message"], top_k=2, tipo="perfil")
                else:
                    results = retrieve_cv_context(query=state["user_message"], top_k=4)

            state["retrieved_context"] = results
            state["sources"] = [r["texto"] for r in results]
            RAG_RETRIEVED_DOCUMENTS_COUNT.set(len(results))
        except Exception:
            NODE_ERRORS_TOTAL.labels(node_name="retrieve_context").inc()
            raise
    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Nodo 4: Generación con LLM fundamentado en el contexto (Grounding)."""
    from app.metrics import NODE_EXECUTION_DURATION_SECONDS, LLM_TOKENS_TOTAL, LLM_COST_ESTIMATED_TOTAL, NODE_ERRORS_TOTAL
    with NODE_EXECUTION_DURATION_SECONDS.labels(node_name="generate_response").time():
        try:
            if not state["is_valid_input"]:
                state["llm_response"] = state["guardrail_error"]
                return state

            intent = state.get("intent", "CV_QUESTION")
            if intent == "OUT_OF_BOUNDS":
                state["llm_response"] = "Como agente conversacional enfocado en el perfil profesional del candidato, solo estoy capacitado para responder preguntas sobre su experiencia laboral, proyectos, habilidades e historia profesional."
                return state

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
            
            raw_resp = llm_out.get("raw")
            usage_info = {}
            if raw_resp and hasattr(raw_resp, "usage") and raw_resp.usage:
                p_tok = getattr(raw_resp.usage, "prompt_tokens", 0)
                c_tok = getattr(raw_resp.usage, "completion_tokens", 0)
                t_tok = getattr(raw_resp.usage, "total_tokens", 0)
                usage_info = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": t_tok}
                
                # Actualiza contadores Prometheus de tokens y costos
                LLM_TOKENS_TOTAL.labels(type="prompt", provider=provider_used).inc(p_tok)
                LLM_TOKENS_TOTAL.labels(type="completion", provider=provider_used).inc(c_tok)
                LLM_COST_ESTIMATED_TOTAL.labels(provider=provider_used).inc((t_tok / 1000) * 0.00015)

            try:
                if not validate_output_guardrails(reply_text, context_chunks):
                    reply_text = "No dispongo de información suficiente en el CV para responder con exactitud a tu pregunta."
            except Exception as g_err:
                logger.warning("Error durante la validación de guardrails de salida: %s", g_err)

            state["llm_response"] = reply_text
            state["latency_ms"] = round(latency_ms, 2)
            state["provider"] = provider_used
            state["usage"] = usage_info

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

            try:
                add_message_to_session(state["session_id"], "user", state["user_message"])
                add_message_to_session(state["session_id"], "assistant", reply_text)
            except Exception as mem_err:
                logger.warning("No se pudo guardar la sesión en memoria/DB: %s", mem_err)

        except Exception:
            NODE_ERRORS_TOTAL.labels(node_name="generate_response").inc()
            raise
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

# Inicialización Singleton global de Langfuse CallbackHandler (SDK v3)
_langfuse_handler = None

settings_host = (
    getattr(settings, "LANGFUSE_HOST", "").strip()
    or getattr(settings, "LANGFUSE_BASE_URL", "").strip()
)

if getattr(settings, "LANGFUSE_PUBLIC_KEY", None) and getattr(settings, "LANGFUSE_SECRET_KEY", None):
    try:
        from langfuse.langchain import CallbackHandler
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY.strip()
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY.strip()
        if settings_host:
            os.environ["LANGFUSE_HOST"] = settings_host
            
        _langfuse_handler = CallbackHandler()
        logger.info("Langfuse v3 inicializado usando las credenciales explícitas de 'settings'.")
    except Exception as e:
        logger.error(f"Error al inicializar Langfuse usando el objeto 'settings': {e}")
else:
    try:
        from langfuse.langchain import CallbackHandler
        _langfuse_handler = CallbackHandler()
        logger.info("Variables no encontradas en 'settings'. Inicializado mediante autodetección nativa del entorno de Render.")
    except Exception as e:
        logger.warning(f"Langfuse no se activó automáticamente (Faltan variables en Render): {e}")

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

    # Configuración combinada de LangGraph y metadatos de sesión para Langfuse
    callbacks = [_langfuse_handler] if _langfuse_handler else []
    config = {
        "configurable": {
            "thread_id": session_id
        },
        "callbacks": callbacks,
        "metadata": {
            "langfuse_session_id": session_id,
            "user_id": session_id
        }
    }

    from app.metrics import AGENT_LATENCY_SECONDS, AGENT_REQUESTS_TOTAL, RAG_RELIABILITY_SCORE
    
    with AGENT_LATENCY_SECONDS.time():
        try:
            final_state = agent_executor.invoke(initial_state, config=config)
            AGENT_REQUESTS_TOTAL.labels(status="success").inc()
        except Exception as exec_err:
            logger.warning("Error ejecutando agente con checkpointer persistente (%s). Reintentando con ejecutor efímero...", exec_err)
            fallback_executor = workflow.compile(checkpointer=MemorySaver())
            final_state = fallback_executor.invoke(initial_state, config=config)
            AGENT_REQUESTS_TOTAL.labels(status="fallback").inc()

    # Forzar el envío síncrono de trazas a Langfuse Cloud
    if callbacks:
        try:
            for cb in callbacks:
                if hasattr(cb, "flush"):
                    cb.flush()
        except Exception as fl_err:
            logger.warning("Error al realizar flush en Langfuse: %s", fl_err)

    # Cálculo de métricas avanzadas de RAG, confiabilidad y costo estimado
    context_chunks = final_state.get("retrieved_context", [])
    top_score = context_chunks[0]["score"] if context_chunks else 0.0
    n_chunks = len(context_chunks)
    
    reliability_score = round(min(1.0, top_score if top_score > 0 else (0.85 if n_chunks > 0 else 0.40)), 2)
    RAG_RELIABILITY_SCORE.set(reliability_score)

    usage = final_state.get("usage", {})
    total_tokens = usage.get("total_tokens", 350)
    estimated_cost_usd = round((total_tokens / 1000) * 0.00015, 6)

    # Registrar Traza Enriquecida con Langfuse SDK Nativo
    pub_key = settings.LANGFUSE_PUBLIC_KEY.strip()
    sec_key = settings.LANGFUSE_SECRET_KEY.strip()
    
    logger.info("Langfuse env check -> Public key len: %d, Secret key len: %d", len(pub_key), len(sec_key))

    if pub_key and sec_key:
        try:
            from langfuse import Langfuse
            host_url = settings.LANGFUSE_BASE_URL.strip() or settings.LANGFUSE_HOST.strip() or "https://us.cloud.langfuse.com"
            logger.info("Inicializando cliente Langfuse SDK con host: %s", host_url)
            lf_client = Langfuse(
                public_key=pub_key,
                secret_key=sec_key,
                host=host_url
            )
            
            trace_obj = lf_client.trace(
                name="CV_Agent_Workflow_Execution",
                session_id=session_id,
                input=message,
                output=final_state.get("llm_response", ""),
                metadata={
                    "intent": final_state.get("intent", ""),
                    "reliability_score": reliability_score,
                    "estimated_cost_usd": estimated_cost_usd
                }
            )

            trace_obj.span(
                name="qdrant_vector_retrieval",
                input=message,
                output={"chunks_count": n_chunks, "top_similarity_score": top_score, "sources": final_state.get("sources", [])}
            )

            trace_obj.generation(
                name="llm_grounded_generation",
                model=final_state.get("provider", "Hugging Face Llama-3.1-8B"),
                output=final_state.get("llm_response", ""),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 250),
                    "completion_tokens": usage.get("completion_tokens", 100),
                    "total_tokens": total_tokens
                }
            )
            
            lf_client.flush()
            logger.info("Traza enriquecida de Langfuse SDK enviada con éxito a %s", host_url)
        except Exception as sdk_lf_err:
            logger.warning("No se pudo enviar la traza nativa de Langfuse SDK (%s)", sdk_lf_err)
    else:
        logger.info("Langfuse Tracing omitido: LANGFUSE_PUBLIC_KEY o LANGFUSE_SECRET_KEY no configuradas en el entorno.")

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
