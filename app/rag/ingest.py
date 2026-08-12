"""
Módulo de ingesta multi-fuente (PDF, TXT, Texto Pegado).
Procesa la ingesta, estructuración mediante LLM, chunking semántico e indexación a Qdrant.
"""

import json
import uuid
import logging
from typing import Optional, Dict, Any, List
import pypdf
from fastapi import UploadFile
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from app.models.schemas import CV
from app.agent.prompts import STRUCTURE_CV_PROMPT
from app.llm.provider import generate_llm_response
from app.rag.retriever import get_embedding, qdrant_client, ensure_collection_exists

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file: UploadFile) -> str:
    """Extrae texto desde un archivo PDF subido."""
    reader = pypdf.PdfReader(file.file)
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    return extracted.strip()


def extract_text_from_txt(file: UploadFile) -> str:
    """Extrae texto desde un archivo de texto plano / Markdown."""
    content = file.file.read().decode("utf-8")
    return content.strip()


def extract_raw_text(file: Optional[UploadFile], pasted_text: Optional[str]) -> str:
    """
    Normaliza la extracción de texto a partir de múltiples fuentes de entrada.
    """
    if pasted_text and pasted_text.strip():
        return pasted_text.strip()
    
    if file is None:
        raise ValueError("Debes proporcionar un archivo (PDF/TXT) o texto pegado.")
    
    filename_lower = file.filename.lower() if file.filename else ""
    
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename_lower.endswith((".txt", ".md")):
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file.filename}")


def structure_with_llm(raw_text: str) -> Dict[str, Any]:
    """
    Utiliza el LLM para estructurar el texto crudo en un diccionario JSON según el esquema de CV.
    """
    prompt = STRUCTURE_CV_PROMPT.format(raw_text=raw_text)
    
    response = generate_llm_response(
        system_prompt="Eres un extractor de datos profesional que responde exclusivamente en JSON válido.",
        input_items=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000
    )
    
    output_text = response.get("text", "").strip()
    
    # Limpieza de bloques de markdown si existieran
    if output_text.startswith("```"):
        output_text = output_text.split("```")[1]
        if output_text.startswith("json"):
            output_text = output_text[4:]
        output_text = output_text.strip()

    try:
        data = json.loads(output_text)
        return data
    except json.JSONDecodeError as exc:
        logger.error("Error decodificando el JSON generado por el LLM durante la estructuración: %s", exc)
        raise ValueError("No se pudo estructurar el CV en un esquema JSON válido.") from exc


def build_chunks(cv_dict: Dict[str, Any], cv_version: str) -> List[Dict[str, Any]]:
    """
    Construye bloques semánticos (chunks) etiquetados con metadatos a partir del CV estructurado.
    """
    chunks = []
    
    # 1. Perfil General
    perfil = cv_dict.get("perfil", {})
    nombre = perfil.get("nombre", "Candidato")
    resumen = perfil.get("resumen", "")
    ubicacion = perfil.get("ubicacion", "")
    contacto = perfil.get("contacto", {}) if isinstance(perfil.get("contacto"), dict) else {}
    email = contacto.get("email", "")
    linkedin = contacto.get("linkedin", "")
    telefono = contacto.get("telefono", "")
    
    texto_perfil = f"Perfil Profesional de {nombre}. Ubicación: {ubicacion}. Resumen: {resumen}."
    chunks.append({
        "texto": texto_perfil,
        "metadata": {"tipo": "perfil", "nombre": nombre, "email": email, "linkedin": linkedin, "telefono": telefono, "cv_version": cv_version}
    })

    # 1b. Chunk explícito de Contacto Directo
    contacto_partes = []
    if email:
        contacto_partes.append(f"correo electrónico {email}")
    if telefono:
        contacto_partes.append(f"teléfono {telefono}")
    if linkedin:
        contacto_partes.append(f"LinkedIn {linkedin}")
    else:
        contacto_partes.append("no tiene LinkedIn registrado en este CV")

    texto_contacto = f"Para contactar a {nombre}: " + ", ".join(contacto_partes) + "."
    chunks.append({
        "texto": texto_contacto,
        "metadata": {"tipo": "contacto", "email": email, "telefono": telefono, "linkedin": linkedin, "cv_version": cv_version}
    })

    # 2. Experiencia Laboral
    exp_list = cv_dict.get("experiencia", [])
    for exp in exp_list:
        tecnologias = ", ".join(exp.get("tecnologias", []))
        logros = "; ".join(exp.get("logros", []))
        texto = f"{exp.get('puesto', '')} en {exp.get('empresa', '')} ({exp.get('periodo', '')}): {exp.get('descripcion', '')}. Tecnologías: {tecnologias}. Logros: {logros}"
        chunks.append({
            "texto": texto,
            "metadata": {"tipo": "experiencia", "cv_version": cv_version, **exp}
        })

    # 2b. Chunk Resumen Jerárquico de Trayectoria Completa
    if exp_list:
        resumen_trayectoria = "; ".join(f"{e.get('puesto', '')} en {e.get('empresa', '')} ({e.get('periodo', '')})" for e in exp_list)
        chunks.append({
            "texto": f"Resumen de trayectoria profesional completa de {nombre}: Ha ocupado {len(exp_list)} puestos clave: {resumen_trayectoria}.",
            "metadata": {"tipo": "experiencia", "cv_version": cv_version, "es_resumen": True}
        })

    # 3. Proyectos
    for proj in cv_dict.get("proyectos", []):
        texto = f"Proyecto {proj.get('nombre', '')}. Problema: {proj.get('problema', '')}. Solución: {proj.get('solucion', '')}. Arquitectura: {proj.get('arquitectura', '')}. Resultado: {proj.get('resultado', '')}"
        chunks.append({
            "texto": texto,
            "metadata": {"tipo": "proyecto", "cv_version": cv_version, **proj}
        })

    # 4. Skills
    skills = cv_dict.get("skills", {})
    tec_str = ", ".join(skills.get("tecnicas", []))
    gen_str = ", ".join(skills.get("generales", []))
    texto_skills = f"Habilidades y Competencias de {nombre}. Técnicas: {tec_str}. Competencias generales: {gen_str}."
    chunks.append({
        "texto": texto_skills,
        "metadata": {"tipo": "skills", "cv_version": cv_version}
    })

    # 5. Educación
    educacion_list = cv_dict.get("educacion", [])
    for edu in educacion_list:
        texto = f"Educación de {nombre}: {edu.get('titulo', '')} en {edu.get('institucion', '')}"
        if edu.get("periodo"):
            texto += f" ({edu['periodo']})"
        texto += "."
        chunks.append({
            "texto": texto,
            "metadata": {"tipo": "educacion", "cv_version": cv_version, **edu}
        })

    if educacion_list:
        resumen_edu = "; ".join(f"{e.get('titulo', '')} en {e.get('institucion', '')}" for e in educacion_list)
        chunks.append({
            "texto": f"Formación académica completa de {nombre}: {resumen_edu}.",
            "metadata": {"tipo": "educacion", "cv_version": cv_version}
        })

    # 6. Certificaciones
    certs = cv_dict.get("certificaciones", [])
    if certs:
        chunks.append({
            "texto": f"Certificaciones de {nombre}: {', '.join(certs)}.",
            "metadata": {"tipo": "certificaciones", "cv_version": cv_version}
        })

    # 7. Cursos Selectos
    cursos = cv_dict.get("cursos_selectos", [])
    if cursos:
        chunks.append({
            "texto": f"Cursos y diplomados adicionales de {nombre}: {', '.join(cursos)}.",
            "metadata": {"tipo": "cursos", "cv_version": cv_version}
        })

    # 8. Colaboración Académica / Docencia
    for colab in cv_dict.get("colaboracion_academica", []):
        texto = f"{nombre} se desempeña como {colab.get('rol', '')} en {colab.get('institucion', '')}"
        if colab.get("periodo"):
            texto += f" ({colab['periodo']})"
        if colab.get("descripcion"):
            texto += f". {colab['descripcion']}"
        chunks.append({
            "texto": texto,
            "metadata": {"tipo": "docencia", "cv_version": cv_version, **colab}
        })

    # 9. Chunk explícito de Límites de Información (Anti-Alucinación)
    chunks.append({
        "texto": f"Este agente contiene únicamente información documentada en el CV de {nombre}. No dispone de datos sobre pretensiones salariales, referencias personales ni información privada no especificada aquí.",
        "metadata": {"tipo": "meta", "cv_version": cv_version}
    })

    return chunks


def replace_cv_version(new_version: str):
    """
    Elimina versiones anteriores de la colección Qdrant para mantener atomicidad de los datos.
    """
    try:
        qdrant_client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must_not=[FieldCondition(key="cv_version", match=MatchValue(value=new_version))]
            )
        )
        logger.info("Versiones antiguas del CV eliminadas de Qdrant. Versión activa: %s", new_version)
    except Exception as exc:
        logger.warning("Error al reemplazar versiones de CV en Qdrant: %s", exc)


def ingest_cv(file: Optional[UploadFile] = None, pasted_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Orquesta la ingesta completa: Extracción -> Estructuración LLM -> Validación Pydantic -> Chunking -> Upsert a Qdrant.
    """
    ensure_collection_exists()

    raw_text = extract_raw_text(file, pasted_text)
    logger.info("Texto extraído exitosamente (longitud: %d caracteres). Estructurando...", len(raw_text))

    cv_dict = structure_with_llm(raw_text)
    
    # Validación estricta con Pydantic
    cv = CV.model_validate(cv_dict)

    cv_version = str(uuid.uuid4())
    chunks = build_chunks(cv.model_dump(), cv_version)

    # Generación de vectores vía la API de Hugging Face (ligero, sin PyTorch ni uso de RAM)
    chunk_texts = [c["texto"] for c in chunks]
    vectors = get_embedding(chunk_texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={"texto": c["texto"], **c["metadata"]}
        )
        for v, c in zip(vectors, chunks)
    ]

    qdrant_client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)
    replace_cv_version(cv_version)

    logger.info("Ingesta completada. %d chunks insertados para la versión %s", len(points), cv_version)

    return {
        "cv_version": cv_version,
        "chunks_ingeridos": len(points)
    }
