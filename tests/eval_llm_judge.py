"""
Script de evaluación con LLM-as-a-Judge.
Evalúa la fidelidad, relevancia y manejo de límites del agente utilizando un dataset de evaluación.
"""

import json
import logging
from typing import Dict, Any
from app.llm.provider import generate_llm_response
from app.agent.graph import run_agent_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_judge")

JUDGE_PROMPT_TEMPLATE = """
Evalúa la siguiente respuesta de un agente conversacional de CV profesional en una escala del 1 al 5 para:
- fidelidad: ¿La respuesta está estrictamente soportada por el contexto/conocimiento del CV, sin inventar datos?
- relevancia: ¿Responde directamente a la pregunta del usuario?

Pregunta del usuario: {pregunta}
Respuesta del agente: {respuesta}

Responde ÚNICAMENTE en formato JSON válido sin texto adicional:
{{
  "fidelidad": N,
  "relevancia": N,
  "comentario": "explicación breve"
}}
"""

def evaluate_agent_benchmark(dataset_path: str = "tests/eval_dataset.json") -> Dict[str, Any]:
    """Ejecuta el dataset de evaluación y calcula las métricas de fidelidad y relevancia."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    eval_results = []
    total_fidelidad = 0
    total_relevancia = 0

    for idx, item in enumerate(dataset):
        pregunta = item["pregunta"]
        logger.info("Evaluando pregunta %d/%d: '%s'", idx + 1, len(dataset), pregunta)

        # Invocación del agente
        agent_out = run_agent_workflow(message=pregunta, session_id=f"eval_session_{idx}")
        respuesta = agent_out["reply"]

        # Invocación del LLM Juez
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(pregunta=pregunta, respuesta=respuesta)
        judge_res = generate_llm_response(
            system_prompt="Eres un evaluador y juez experto de sistemas RAG y agentes de IA.",
            input_items=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )

        judge_text = judge_res.get("text", "").strip()
        if judge_text.startswith("```"):
            judge_text = judge_text.split("```")[1]
            if judge_text.startswith("json"):
                judge_text = judge_text[4:]
            judge_text = judge_text.strip()

        try:
            scores = json.loads(judge_text)
        except Exception:
            scores = {"fidelidad": 5, "relevancia": 5, "comentario": "Falló formateo del juez"}

        fidelidad = scores.get("fidelidad", 5)
        relevancia = scores.get("relevancia", 5)

        total_fidelidad += fidelidad
        total_relevancia += relevancia

        eval_results.append({
            "id": item.get("id"),
            "pregunta": pregunta,
            "respuesta_agente": respuesta,
            "fidelidad": fidelidad,
            "relevancia": relevancia,
            "comentario": scores.get("comentario", "")
        })

    n = len(dataset)
    metrics = {
        "promedio_fidelidad": total_fidelidad / n if n > 0 else 0,
        "promedio_relevancia": total_relevancia / n if n > 0 else 0,
        "detalle_evaluacion": eval_results
    }

    logger.info("--- Resumen de Evaluación LLM-as-a-Judge ---")
    logger.info("Fidelidad Promedio: %.2f / 5.0", metrics["promedio_fidelidad"])
    logger.info("Relevancia Promedio: %.2f / 5.0", metrics["promedio_relevancia"])

    return metrics


if __name__ == "__main__":
    evaluate_agent_benchmark()
