# Documento de Arquitectura — Agente Conversacional de CV
### Reto IA Banorte · AI Engineer Generativo

**Autor:** [Tu nombre]
**Versión:** 1.0
**Fecha:** Agosto 2026
**Stack objetivo:** Python · FastAPI · Groq / Hugging Face · Qdrant · Docker · Render / HF Spaces

---

## 0. Cómo usar este documento

Está pensado para seguirse **de arriba hacia abajo, en orden**. Cada sección es una fase del proyecto:

1. Diseño y alcance
2. Preparación de datos (ingesta multi-fuente del CV: PDF, TXT, texto pegado)
3. Capa de embeddings + Qdrant (RAG)
4. Capa de LLM (Groq / Hugging Face)
5. Diseño del agente (herramientas, guardrails, memoria)
6. API (FastAPI)
7. Dockerización
8. Despliegue gratuito (Render, HF Spaces, alternativas)
9. Pruebas y evaluación
10. Observabilidad, seguridad y operación
11. Entregables sugeridos para el reto
12. Roadmap de "si tengo más tiempo"

Cada sección técnica incluye: qué hacer, por qué, y un ejemplo de código o config.

---

## 1. Diseño y alcance

### 1.1 Qué vas a construir

Un **agente conversacional RAG** que responde preguntas sobre tu perfil profesional (experiencia, skills, proyectos), citando información real de tu CV, sin inventar datos, desplegado en un endpoint accesible públicamente.

No es:
- Un chatbot genérico con un prompt gigante pegado al CV en texto plano (sin recuperación, sin control, no escala, alucina).
- Solo una interfaz bonita sin backend real.
- Un sistema multiagente complejo sin justificación (el reto explícitamente lo desincentiva si no aporta valor).

### 1.2 Decisión de arquitectura (resumen ejecutivo)

| Componente | Elección | Por qué |
|---|---|---|
| Vector DB | **Qdrant Cloud** (free tier) | Ya definido por ti. Free tier permanente: 0.5 vCPU, 1GB RAM, 4GB disco, sin tarjeta de crédito. Soporta ~1M vectores de 768 dim. |
| LLM principal | **Groq** (Llama 3.x / GPT-OSS vía Groq API), consumido a través de su endpoint `/v1/responses` | Inferencia extremadamente rápida (LPU), tier gratuito con límites de rate generosos para demo. Su Responses API es **compatible con el estándar abierto Open Responses**, lo que estandariza tu integración. |
| LLM / Embeddings alternos | **Hugging Face Inference Providers** (también con soporte de Open Responses) | Fallback si Groq tiene rate limit, y fuente de modelos de embeddings open-source gratuitos (ej. `sentence-transformers`). |
| Compatibilidad de API | **Open Responses** (openresponses.org) | Tu agente consume modelos y expone su propio endpoint `/v1/responses` con el mismo contrato — ver sección 4. |
| Orquestación | **LangChain o LangGraph** (ligero) | Estandariza RAG + tool calling; LangGraph si quieres un grafo de estados explícito (útil para "manejar preguntas ambiguas" y control de flujo). |
| Backend / API | **FastAPI**, exponiendo `/v1/responses` compatible con **Open Responses** | Async, tipado, fácil de documentar (OpenAPI automático); el endpoint `/v1/responses` hace que tu agente sea interoperable con cualquier cliente que hable el estándar (Agents SDKs, herramientas de evaluación, otros orquestadores). |
| Contenedor | **Docker** | Requisito implícito del reto ("desplegar en un entorno accesible", reproducibilidad). |
| Hosting gratuito | **Hugging Face Spaces (Docker SDK)** como opción primaria, **Render Free Web Service** como alternativa | Ver sección 8 para comparación detallada. |
| Observabilidad | Logs estructurados + límite de trazas gratuito (Langfuse Cloud free tier o logging propio) | Cumple el requisito de "gobernanza y observabilidad" que pide la vacante. |

### 1.3 Arquitectura general (diagrama en texto)

```
                         ┌───────────────────────────┐
                         │        Usuario / Web       │
                         │  (chat UI o cliente HTTP)  │
                         └─────────────┬──────────────┘
                                       │ HTTPS
                                       ▼
                    ┌───────────────────────────────────┐
                    │     FastAPI (contenedor Docker)     │
                    │  /chat  /health  /metrics           │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │   Orquestador del Agente       │  │
                    │  │  (LangGraph / LangChain)       │  │
                    │  │                                 │  │
                    │  │  1. Guardrails de entrada       │  │
                    │  │  2. Router de intención         │  │
                    │  │  3. Tool: retrieve_cv_context   │──┼──► Qdrant Cloud (free)
                    │  │  4. Tool: get_project_detail    │  │    colección "cv_chunks"
                    │  │  5. Llamada al LLM con contexto │──┼──► Groq API (Llama 3.x)
                    │  │  6. Guardrails de salida         │  │    (fallback: HF Inference)
                    │  │  7. Memoria de conversación      │  │
                    │  └───────────────────────────────┘  │
                    │                                     │
                    │  Logging estructurado (JSON)         │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │  Observabilidad / logs     │
                         │  (stdout + Langfuse free)  │
                         └───────────────────────────┘
```

### 1.4 Principio de diseño rector

> El agente **nunca responde desde la memoria del LLM**. Siempre recupera contexto real de Qdrant antes de generar la respuesta (RAG estricto), y si no encuentra evidencia suficiente, responde explícitamente que no tiene esa información. Esto es lo que evalúa el reto cuando pide "evitar inventar empleos, tecnologías, logros".

---

## 2. Preparación de datos (ingesta multi-fuente del CV)

El reto pide que el agente pueda conversar sobre tu CV real. Para que eso sea práctico (y para que el agente pueda **actualizarse cuando cambie tu CV**, no solo cargarse una vez), el pipeline de datos debe aceptar varias fuentes de entrada y convertirlas todas al mismo formato estructurado antes de llegar a Qdrant.

### 2.1 Fuentes soportadas

| Fuente | Cómo se recibe | Extracción de texto |
|---|---|---|
| **PDF** | Archivo subido vía `multipart/form-data` | `pypdf` o `pdfplumber` |
| **TXT / Markdown** | Archivo subido | Lectura directa (`UTF-8`) |
| **Texto pegado** | Campo de texto en el body del request (copiar/pegar en el chat o en un formulario) | Ninguna, ya es texto plano |

El diseño clave: **sin importar la fuente, todas convergen en el mismo texto crudo**, y de ahí en adelante el pipeline es idéntico (extracción de texto → estructuración → chunking → embeddings → upsert a Qdrant). Esto evita tener tres pipelines distintos.

```
PDF ──┐
TXT ──┼──► extract_raw_text() ──► raw_text (str)
Pegado┘
                                        │
                                        ▼
                          structure_with_llm(raw_text)
                                        │
                                        ▼
                              cv.json (estructurado)
                                        │
                                        ▼
                          build_chunks() + embeddings
                                        │
                                        ▼
                          upsert a Qdrant (nueva versión)
```

### 2.2 Extracción de texto por fuente

```python
import pypdf
from fastapi import UploadFile

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = pypdf.PdfReader(file.file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_txt(file: UploadFile) -> str:
    return file.file.read().decode("utf-8")

def extract_raw_text(file: UploadFile | None, pasted_text: str | None) -> str:
    if pasted_text:
        return pasted_text.strip()
    if file is None:
        raise ValueError("Debes proporcionar un archivo o texto pegado.")
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file)
    if file.filename.lower().endswith((".txt", ".md")):
        return extract_text_from_txt(file)
    raise ValueError(f"Formato no soportado: {file.filename}")
```

> Nota sobre PDFs escaneados (imagen, sin texto seleccionable): `pypdf` no extrae texto de imágenes. Si tu CV es un PDF escaneado, sería necesario OCR (ej. `pytesseract`), pero esto normalmente **no aplica a un CV generado digitalmente** (Word/LinkedIn export/Canva), así que queda fuera de alcance salvo que lo necesites — menciónalo como decisión consciente en tu documentación si tu CV es así.

### 2.3 Estructuración: de texto crudo a JSON (vía LLM)

A diferencia de escribir el JSON a mano (enfoque de la v1 de este documento), aquí el texto crudo del PDF/TXT/pegado **no viene estructurado**. La forma robusta de convertirlo es usar el propio LLM (Groq) como paso de extracción, con salida forzada a JSON y un esquema fijo:

```python
STRUCTURE_PROMPT = """
Extrae la información del siguiente CV y regrésala ÚNICAMENTE como JSON válido,
sin texto adicional, siguiendo exactamente este esquema:

{
  "perfil": {"nombre": str, "resumen": str, "ubicacion": str},
  "experiencia": [
    {"id": str, "empresa": str, "puesto": str, "periodo": str,
     "descripcion": str, "tecnologias": [str], "logros": [str]}
  ],
  "proyectos": [
    {"id": str, "nombre": str, "problema": str, "solucion": str,
     "arquitectura": str, "resultado": str}
  ],
  "skills": {"tecnicas": [str], "generales": [str]}
}

Reglas:
- No inventes información que no esté en el texto.
- Si un campo no aparece en el CV, usa "" o [] según corresponda, nunca lo omitas.
- Genera IDs cortos y únicos (ej. "exp_001", "proj_001").
- Responde solo el JSON, sin explicación ni backticks de markdown.

Texto del CV:
---
{raw_text}
---
"""

def structure_with_llm(raw_text: str) -> dict:
    response = groq_client.responses.create(
        model="llama-3.3-70b-versatile",
        input=[{"role": "user", "content": STRUCTURE_PROMPT.format(raw_text=raw_text)}],
        temperature=0,
    )
    return json.loads(response.output_text)
```

**Por qué así y no con regex/parsers de CV tradicionales:** los CVs varían mucho en formato; un LLM con esquema fijo y `temperature=0` generaliza mucho mejor que reglas manuales, y como ya usas un LLM en el resto del sistema, no agregas una dependencia nueva. Como salvaguarda, valida el JSON resultante contra un modelo Pydantic antes de continuar — si falla la validación, rechaza la carga y pide al usuario revisar el archivo, en vez de ingerir datos corruptos a Qdrant.

```python
from pydantic import BaseModel

class Experiencia(BaseModel):
    id: str
    empresa: str
    puesto: str
    periodo: str
    descripcion: str
    tecnologias: list[str]
    logros: list[str]

class Proyecto(BaseModel):
    id: str
    nombre: str
    problema: str
    solucion: str
    arquitectura: str
    resultado: str

class Perfil(BaseModel):
    nombre: str
    resumen: str
    ubicacion: str

class Skills(BaseModel):
    tecnicas: list[str]
    generales: list[str]

class CV(BaseModel):
    perfil: Perfil
    experiencia: list[Experiencia]
    proyectos: list[Proyecto]
    skills: Skills
```

### 2.4 Chunking

- Un chunk por bloque semántico (una experiencia laboral completa, un proyecto completo, no partir logros a la mitad).
- Cada chunk lleva metadata: `{"tipo": "experiencia", "id": "exp_001", "empresa": "...", "cv_version": "..."}`.
- Tamaño recomendado: 150–400 tokens por chunk (los bloques de CV son cortos, no necesitas chunking agresivo tipo "documento largo").

### 2.5 Regla anti-alucinación en el diseño de datos

Incluye explícitamente un chunk de "límites de información":
```json
{"tipo": "meta", "contenido": "Este agente solo tiene información sobre la experiencia profesional documentada en este CV hasta 2026. No tiene información sobre disponibilidad salarial, referencias personales, ni datos no incluidos aquí."}
```
Esto le da al LLM una fuente explícita para responder "no tengo esa información" sin inventar.

### 2.6 Versionado: reemplazar el CV sin mezclar datos viejos

Cuando el agente carga un CV nuevo, **no debes simplemente agregar más puntos a la colección** — eso mezclaría datos del CV viejo y el nuevo, y el agente podría responder con información desactualizada o contradictoria. El patrón correcto:

1. Genera un `cv_version` único (ej. timestamp o UUID) para cada ingesta.
2. Etiqueta todos los chunks de esa ingesta con ese `cv_version` en el payload.
3. Al confirmar que la nueva ingesta fue exitosa, **elimina de Qdrant todos los puntos con un `cv_version` distinto al nuevo** (o mantenlos pero filtra siempre por el `cv_version` activo en las búsquedas, si quieres conservar historial).

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def replace_cv_version(new_version: str):
    # Borra todo lo que no sea la versión recién ingerida
    client.delete(
        collection_name="cv_chunks",
        points_selector=Filter(
            must_not=[FieldCondition(key="cv_version", match=MatchValue(value=new_version))]
        ),
    )
```

Enfoque más simple si no necesitas conservar historial de versiones: en vez de filtrar por `cv_version`, simplemente `recreate_collection()` en cada ingesta nueva (borra y vuelve a crear la colección). Es menos elegante pero mucho más fácil de razonar para el alcance del reto; documenta cuál elegiste y por qué.

### 2.7 Flujo completo de ingesta (función orquestadora)

```python
def ingest_cv(file: UploadFile | None = None, pasted_text: str | None = None) -> dict:
    raw_text = extract_raw_text(file, pasted_text)
    cv_dict = structure_with_llm(raw_text)
    cv = CV.model_validate(cv_dict)  # lanza error si no cumple el esquema

    cv_version = str(uuid.uuid4())
    chunks = build_chunks(cv.model_dump(), cv_version)
    vectors = embedding_model.encode([c["texto"] for c in chunks])

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload={"texto": c["texto"], **c["metadata"]})
        for v, c in zip(vectors, chunks)
    ]
    client.upsert(collection_name="cv_chunks", points=points)
    replace_cv_version(cv_version)

    return {"cv_version": cv_version, "chunks_ingeridos": len(points)}
```

Este es el mismo flujo que corre tanto el script `ingest.py` inicial (sección 3.3) como el endpoint de carga en caliente (`/cv/upload`, sección 6.3) — ambos llaman a `ingest_cv()`, no hay lógica duplicada.

---

## 3. Capa de Embeddings + Qdrant (RAG)

### 3.1 Modelo de embeddings (gratuito)

Usa un modelo open-source vía Hugging Face para no gastar cuota de un proveedor de pago:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, rápido, suficiente para un corpus de CV que es pequeño).
- Alternativa multilingüe (recomendada porque tu CV está en español): `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim).

Puedes correrlo localmente en el contenedor (CPU, es ligero) con `sentence-transformers`, sin necesidad de llamar a una API externa para embeddings. Esto reduce dependencias externas y costo.

### 3.2 Colección en Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.create_collection(
    collection_name="cv_chunks",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

### 3.3 Ingesta (`build_chunks`, usado tanto en el script inicial como en la carga en caliente)

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def build_chunks(cv: dict, cv_version: str) -> list[dict]:
    chunks = []
    for exp in cv["experiencia"]:
        texto = f"{exp['puesto']} en {exp['empresa']} ({exp['periodo']}): {exp['descripcion']}. Tecnologías: {', '.join(exp['tecnologias'])}. Logros: {'; '.join(exp['logros'])}"
        chunks.append({"texto": texto, "metadata": {"tipo": "experiencia", "cv_version": cv_version, **exp}})
    for proj in cv["proyectos"]:
        texto = f"Proyecto {proj['nombre']}. Problema: {proj['problema']}. Solución: {proj['solucion']}. Arquitectura: {proj['arquitectura']}. Resultado: {proj['resultado']}"
        chunks.append({"texto": texto, "metadata": {"tipo": "proyecto", "cv_version": cv_version, **proj}})
    # ... repetir para skills, perfil, meta
    return chunks
```

Para la **carga inicial** (primer CV, antes de tener el endpoint corriendo), corre el flujo completo una sola vez localmente:

```python
# ingest_inicial.py
if __name__ == "__main__":
    with open("data/cv.pdf", "rb") as f:
        upload = UploadFile(filename="cv.pdf", file=f)
        result = ingest_cv(file=upload)
        print(result)  # {"cv_version": "...", "chunks_ingeridos": N}
```

Para **actualizaciones posteriores** (nuevo CV, PDF distinto, texto pegado), usa el endpoint `/cv/upload` de la sección 6.3 — llama exactamente a la misma función `ingest_cv()` (sección 2.7), así que el comportamiento es idéntico sin importar si lo corres localmente o vía API.

### 3.4 Recuperación (tool que usará el agente)

```python
def retrieve_cv_context(query: str, top_k: int = 4, tipo: str | None = None) -> list[dict]:
    query_vector = model.encode(query).tolist()
    query_filter = None
    if tipo:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="tipo", match=MatchValue(value=tipo))])

    results = client.search(
        collection_name="cv_chunks",
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=0.35,  # descarta resultados poco relevantes -> reduce alucinación
    )
    return [{"texto": r.payload["texto"], "score": r.score} for r in results]
```

`score_threshold` es clave: si nada supera el umbral, el agente debe responder "no tengo información suficiente sobre eso" en vez de forzar una respuesta con contexto irrelevante.

---

## 4. Capa de LLM (Groq / Hugging Face) — compatible con Open Responses

### 4.0 Qué es Open Responses y por qué te conviene usarlo aquí

**Open Responses** es una especificación abierta (openresponses.org, impulsada por Hugging Face y la comunidad) que estandariza la forma de llamar modelos, hacer streaming y orquestar tool calling, basada en la Responses API de OpenAI. La idea: describir tu request **una sola vez** y poder correrlo contra OpenAI, Groq, HF Inference Providers, o un modelo local, sin reescribir tu capa de integración.

Para el reto esto es una ventaja doble:
1. Te permite tener **un solo cliente** para Groq y Hugging Face (ambos exponen — o son compatibles con — el endpoint `/v1/responses`), lo que simplifica el patrón de fallback de la sección 4.3.
2. Puedes hacer que **tu propio agente** exponga un endpoint `/v1/responses` compatible (sección 4.5), demostrando que entiendes interoperabilidad de APIs de agentes — algo que la vacante valora explícitamente ("orquestación de aplicaciones basadas en LLMs", "arquitecturas basadas en APIs").

Conceptos clave del spec que usarás:
- **`input`**: lista de *items* (mensajes, resultados de tool calls) que se le pasan al modelo, en vez del array `messages` clásico de Chat Completions.
- **`previous_response_id`**: para continuar una conversación sin reenviar todo el historial — resuelve "memoria de conversación" de forma nativa (ver sección 5.4).
- **Streaming semántico**: eventos como `response.output_text.delta` y `response.completed`, en vez de deltas crudos — útil si luego quieres una UI con streaming real.
- **`tools` / `tool_choice` / `allowed_tools`**: mismo formato de tools que ya definiste para tu RAG tool, portable entre proveedores.

### 4.1 Groq como proveedor principal (vía Responses API)

Groq expone `/v1/responses`, **compatible con el estándar Open Responses / Responses API de OpenAI** (en beta al momento de escribir esto). Puedes usar el SDK oficial de `openai` apuntando al `base_url` de Groq — no necesitas el SDK propietario de Groq para esto.

```python
from openai import OpenAI

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

def call_llm(system_prompt: str, input_items: list[dict]) -> dict:
    response = groq_client.responses.create(
        model="llama-3.3-70b-versatile",   # valida el nombre vigente en console.groq.com
        instructions=system_prompt,
        input=input_items,                  # lista de items tipo {"role": "user", "content": "..."}
        temperature=0.2,
        max_output_tokens=600,
    )
    return {"text": response.output_text, "raw": response}
```

> Nota: la Responses API de Groq aún no soporta conversaciones con estado del lado del servidor (`previous_response_id` persistente); debes seguir enviando el historial reciente en `input` en cada llamada (ver memoria en 5.4). Valida el nombre exacto del modelo vigente en `console.groq.com`, ya que la lista cambia.

### 4.2 Hugging Face como fallback / alterna

Hugging Face tiene soporte temprano de Open Responses vía **Inference Providers**, además de su cliente clásico de chat. Para máxima compatibilidad de formato con el flujo anterior, usa también el SDK `openai` apuntando al router de HF:

```python
from openai import OpenAI

hf_client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
)

def call_llm_fallback(system_prompt: str, input_items: list[dict]) -> dict:
    response = hf_client.responses.create(
        model="meta-llama/Llama-3.1-8B-Instruct",  # o el modelo/proveedor disponible en tu cuenta HF
        instructions=system_prompt,
        input=input_items,
        max_output_tokens=600,
    )
    return {"text": response.output_text, "raw": response}
```

Si el proveedor de HF que tengas configurado aún no soporta `/responses` para el modelo elegido, usa `chat.completions.create(...)` como fallback de ese fallback (Chat Completions sigue siendo el mínimo común denominador universal). Documenta esta decisión en tu README: demuestra que evaluaste el trade-off madurez-vs-estandarización.

### 4.3 Patrón de fallback con manejo de errores

```python
def generate_response(system_prompt, input_items):
    try:
        return call_llm(system_prompt, input_items)
    except Exception as e:
        log.warning(f"Groq falló, usando fallback HF: {e}")
        try:
            return call_llm_fallback(system_prompt, input_items)
        except Exception as e2:
            log.error(f"Fallback también falló: {e2}")
            return {"text": "Estoy teniendo problemas técnicos temporales. Intenta de nuevo en unos segundos.", "raw": None}
```

Como ambos proveedores hablan el mismo formato (`responses.create`, `input`, `output_text`), el código de fallback queda simple y desacoplado del proveedor — ese es justamente el valor de estandarizar sobre Open Responses en vez de mezclar el SDK propietario de Groq con el cliente de HF.

Esto demuestra al evaluador manejo de errores y resiliencia, algo explícitamente pedido en el reto.

### 4.4 System prompt (grounding + anti-alucinación)

```
Eres el agente conversacional del CV de [Tu Nombre]. Tu única fuente de verdad es el
contexto recuperado que se te proporciona en cada turno, extraído del CV real de la persona.

Reglas estrictas:
1. Responde ÚNICAMENTE con base en el contexto proporcionado.
2. Si el contexto no contiene la respuesta, di explícitamente que no tienes esa información
   disponible; no la inventes ni la infieras.
3. No atribuyas empleos, tecnologías ni logros que no aparezcan en el contexto.
4. Si la pregunta es ambigua, pide una aclaración breve o responde a la interpretación más
   probable indicando tu supuesto.
5. Mantén un tono profesional, natural y en primera persona cuando hable de "mi experiencia"
   (representas al candidato, no eres un narrador externo).
6. Si preguntan algo fuera de alcance (no relacionado al CV), indica amablemente que solo
   puedes hablar sobre la trayectoria profesional del candidato.
```

### 4.5 Exponer tu propio agente como servidor Open Responses

Además de *consumir* modelos vía Open Responses, puedes hacer que tu **API del agente** (la que despliegas tú, sección 6) implemente el mismo contrato en `/v1/responses`. Así cualquier cliente que hable el estándar (incluido el propio SDK de `openai` apuntando a tu `base_url`) puede usar tu agente de CV como si fuera "otro proveedor de modelos" — es una forma muy clara de demostrar diseño de APIs interoperable.

Lo mínimo para ser un *implementer* razonable del spec (no necesitas el 100% de la superficie, solo lo relevante a tu caso de uso no-streaming):

- Aceptar `POST /v1/responses` con body JSON: `{"model": "...", "input": [...], "instructions": "...", "tools": [...]}`.
- Tratar `input` como la lista de turnos (mensajes de usuario + resultados de tool calls previos), igual que ya haces internamente.
- Responder con un objeto `Response` con al menos: `id`, `status` (`"completed"` | `"failed"` | `"incomplete"`), y `output` (lista de *items*, típicamente un `message` con `content: [{"type": "output_text", "text": "..."}]`).
- Soportar `previous_response_id`: si lo recibes, cargas el turno anterior desde tu buffer de memoria (sección 5.4) y lo concatenas antes del nuevo `input`, tal como indica el spec.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uuid, time

class ResponsesRequest(BaseModel):
    model: str | None = None
    input: list[dict]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[dict] | None = None
    stream: bool = False

@app.post("/v1/responses")
def create_response(req: ResponsesRequest):
    input_items = resolve_previous_response(req.previous_response_id) + req.input
    result = run_agent_from_items(input_items, req.instructions)

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": req.model or "cv-agent-v1",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": result["reply"], "annotations": []}],
            }
        ],
        "output_text": result["reply"],
    }
```

**No implementes streaming (`stream=True`) si no lo necesitas para el reto** — es la parte más compleja del spec (eventos `response.output_item.added`, `response.output_text.delta`, etc.). Documenta explícitamente que tu implementación soporta el modo no-streaming del contrato y por qué (alcance del reto, tiempo disponible); si quieres ir más allá, la sección "Streaming" del spec de openresponses.org tiene los eventos exactos a emitir sobre `text/event-stream`.

Mantén también tu endpoint `/chat` simplificado de la sección 6 — no lo elimines: `/v1/responses` es la interfaz *estándar* para interoperar con otros sistemas, `/chat` puede seguir siendo tu interfaz simple para un frontend propio o para pruebas manuales rápidas.

---

## 5. Diseño del agente (arquitectura agéntica)

### 5.1 Por qué un grafo de estados simple y no "multiagente"

El reto es explícito: no agregues complejidad (multiagente, N herramientas) sin justificación. Para un agente de CV, la complejidad correcta es:

**Un solo agente, con 2–3 tools, guardrails de entrada/salida, y memoria de conversación corta.** Eso ya demuestra: tool calling, RAG, control de alucinación, manejo de contexto — sin sobre-ingeniería.

### 5.2 Flujo (LangGraph, pseudocódigo del grafo)

```
[entrada usuario]
      │
      ▼
[guardrail_entrada]  ── (rechaza prompt injection, contenido no relacionado)
      │
      ▼
[clasificar_intencion]  ── (¿es sobre CV? ¿es ambigua? ¿fuera de alcance?)
      │
      ├── fuera_de_alcance ──► [respuesta_predefinida]
      │
      ▼
[retrieve_cv_context]  ── tool: busca en Qdrant
      │
      ▼
[generar_respuesta]  ── LLM (Groq) con contexto + historial reciente
      │
      ▼
[guardrail_salida]  ── valida que la respuesta no contenga afirmaciones sin soporte
      │
      ▼
[actualizar_memoria]  ── guarda turno en buffer de conversación
      │
      ▼
[respuesta al usuario]
```

### 5.3 Guardrails prácticos (sin sobre-ingeniería)

- **Entrada:** filtro simple de longitud + regex/heurística contra intentos de "ignora tus instrucciones anteriores" (prompt injection básico) + clasificación con el propio LLM en un paso barato ("¿esta pregunta es sobre el CV, sí/no?").
- **Salida:** verificación ligera de que la respuesta no incluya empresas/tecnologías que no estén en el contexto recuperado (puedes hacer un chequeo simple de substring contra el contexto, o un segundo prompt corto de "juez" solo si quieres ir más allá).

### 5.4 Memoria

Buffer de los últimos 4–6 turnos (no necesitas una base de datos de memoria persistente para este alcance; consistencia entre turnos se resuelve pasando el historial reciente al LLM). Si quieres demostrar algo más sofisticado, puedes resumir la conversación cada N turnos para no exceder contexto.

---

## 6. API (FastAPI)

### 6.1 Estructura del proyecto

```
cv-agent/
├── app/
│   ├── main.py              # FastAPI app, endpoints
│   ├── agent/
│   │   ├── graph.py         # definición del grafo (LangGraph)
│   │   ├── prompts.py
│   │   └── guardrails.py
│   ├── rag/
│   │   ├── ingest.py        # extract_raw_text, structure_with_llm, build_chunks, ingest_cv, replace_cv_version
│   │   └── retriever.py
│   ├── llm/
│   │   ├── groq_client.py
│   │   └── hf_client.py
│   ├── models/
│   │   └── schemas.py       # Pydantic: ChatRequest, ChatResponse, CV, CVUploadResponse
│   └── config.py            # variables de entorno
├── data/
│   └── cv_inicial.pdf       # o .txt — semilla para la primera ingesta
├── tests/
│   ├── test_retrieval.py
│   ├── test_ingest.py       # prueba extracción PDF/TXT y estructuración
│   ├── test_agent_eval.py
│   └── eval_dataset.json
├── Dockerfile
├── docker-compose.yml       # para pruebas locales (opcional, agente + qdrant local)
├── requirements.txt
├── .dockerignore
├── .env.example
└── README.md
```

### 6.2 Endpoints mínimos

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CV Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message, req.session_id)
    return ChatResponse(reply=result["reply"], sources=result["sources"])

# Endpoint compatible con el estándar Open Responses — ver sección 4.5
# @app.post("/v1/responses") -> create_response(...)

# Endpoint de ingesta multi-fuente del CV — ver sección 6.3
# @app.post("/cv/upload") -> upload_cv(...)
```

`/health` es importante porque Render y HF Spaces lo usan (o puedes usarlo tú) para verificar que el contenedor levantó correctamente. `/chat` es tu interfaz simple; `/v1/responses` (sección 4.5) es tu interfaz estándar/interoperable; `/cv/upload` es cómo cargas o actualizas la fuente de verdad del agente — expón las tres.

### 6.3 Endpoint de carga/actualización del CV (PDF, TXT o texto pegado)

Un único endpoint acepta las tres fuentes descritas en la sección 2.1. `file` es opcional (multipart), `pasted_text` es opcional (form field o JSON) — el cliente manda uno u otro.

```python
from fastapi import UploadFile, File, Form, HTTPException

class CVUploadResponse(BaseModel):
    cv_version: str
    chunks_ingeridos: int
    mensaje: str

@app.post("/cv/upload", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile | None = File(default=None),
    pasted_text: str | None = Form(default=None),
):
    if file is None and not pasted_text:
        raise HTTPException(status_code=400, detail="Sube un archivo (PDF/TXT) o envía pasted_text.")
    try:
        result = ingest_cv(file=file, pasted_text=pasted_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Fallo en ingesta de CV: {e}")
        raise HTTPException(status_code=500, detail="No se pudo procesar el CV. Intenta de nuevo.")

    return CVUploadResponse(
        cv_version=result["cv_version"],
        chunks_ingeridos=result["chunks_ingeridos"],
        mensaje="CV actualizado correctamente. El agente ya responde con la nueva información.",
    )
```

Ejemplos de uso desde el cliente:

```bash
# Subir un PDF
curl -X POST https://tu-agente.hf.space/cv/upload \
  -F "file=@mi_cv.pdf"

# Subir un TXT
curl -X POST https://tu-agente.hf.space/cv/upload \
  -F "file=@mi_cv.txt"

# Pegar texto directamente (sin archivo)
curl -X POST https://tu-agente.hf.space/cv/upload \
  -F "pasted_text=Juan Pérez, AI Engineer con 4 años de experiencia..."
```

**Protege este endpoint.** A diferencia de `/chat`, que es de solo lectura, `/cv/upload` **escribe** en tu base de conocimiento — no debe quedar abierto al público sin control. Opciones simples para el alcance del reto:
- Un `API key` propio (header `X-Admin-Key`) validado contra una variable de entorno, distinto de las keys de Groq/HF/Qdrant.
- Si solo tú vas a actualizar el CV, ni siquiera necesitas exponerlo públicamente: puedes dejarlo solo accesible localmente (correr `ingest_cv()` como script) y omitir el endpoint HTTP en producción. Documenta cuál decisión tomaste — ambas son válidas, lo importante es que sea consciente.

```python
from fastapi import Header

def verify_admin_key(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="No autorizado.")

@app.post("/cv/upload", response_model=CVUploadResponse, dependencies=[Depends(verify_admin_key)])
async def upload_cv(...):
    ...
```

### 6.4 Que el propio agente pueda disparar la recarga (opcional, tool adicional)

Si quieres que la conversación misma soporte "aquí tienes mi CV actualizado, cárgalo", puedes exponer `ingest_cv()` como una tool más del agente (ver sección 5), donde el usuario pega el texto directamente en el chat y el agente lo interpreta como una instrucción de actualización en vez de una pregunta:

```python
def tool_actualizar_cv(texto_pegado: str) -> dict:
    """Tool que el agente puede invocar cuando detecta que el usuario pegó
    contenido de CV en vez de hacer una pregunta."""
    return ingest_cv(pasted_text=texto_pegado)
```

Actívala solo si aporta valor real a tu demo — si tu caso de uso es "yo subo mi CV una vez antes de la demo", el endpoint HTTP de la sección 6.3 es suficiente y más simple de operar; agregar esto como tool conversacional es una decisión de alcance, no un requisito.

---

## 7. Dockerización

### 7.1 Qué debe ir en el Dockerfile

- Imagen base ligera (`python:3.11-slim`).
- Instalación de dependencias en capa separada (cache de Docker).
- Usuario no root (buena práctica de seguridad, valorada en el reto).
- Puerto expuesto acorde a la plataforma de destino (**7860 si usas Hugging Face Spaces**, o el que definas si usas Render, que detecta el puerto vía variable `PORT`).
- Healthcheck opcional.

```dockerfile
FROM python:3.11-slim

# Evita prompts interactivos y bytecode innecesario
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas (si sentence-transformers las requiere)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias (se cachea si requirements.txt no cambia)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app
COPY app/ ./app
COPY data/ ./data

# Usuario no root
RUN useradd -m appuser
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 7.2 `.dockerignore`

```
__pycache__/
*.pyc
.env
.venv/
tests/
.git/
*.md
!README.md
```

### 7.3 `docker-compose.yml` (solo para desarrollo local, no para producción)

```yaml
services:
  agent:
    build: .
    ports:
      - "7860:7860"
    env_file: .env
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

Así puedes desarrollar contra un Qdrant local y luego apuntar a Qdrant Cloud solo en producción cambiando `QDRANT_URL`.

### 7.4 Variables de entorno (`.env.example`)

```
GROQ_API_KEY=
HF_TOKEN=
QDRANT_URL=
QDRANT_API_KEY=
LOG_LEVEL=INFO
```

Nunca subas `.env` real al repo; en el hosting configura estos valores como *secrets* (ver sección 8).

---

## 8. Despliegue gratuito

### 8.1 Comparación de opciones (verificado agosto 2026)

| Plataforma | Free tier | Sleep / cold start | Ideal para | Limitación clave |
|---|---|---|---|---|
| **Hugging Face Spaces (Docker SDK)** | CPU Basic 2 vCPU / 16GB RAM, gratis indefinidamente, sin tarjeta | Se duerme tras inactividad si no hay tráfico | Demos de IA, exactamente el caso de uso del reto | Sin garantía de uptime; no pensado como backend "serio" 24/7 |
| **Render (Free Web Service)** | 750 hrs/mes, 512MB RAM, 0.1 CPU | Se duerme tras 15 min de inactividad, cold start 30–60s | Backend tipo API con Docker, deploy por Git push | RAM/CPU limitados para modelos locales pesados (embeddings ligeros sí corren bien) |
| **Railway (Free/Trial)** | Créditos de prueba limitados (no siempre permanente) | Variable | Prototipos cortos | Requiere validar vigencia del free trial al momento de usarlo |
| **Fly.io** | Allowance gratuito reducido según la región | Variable | Apps pequeñas siempre activas | Ha reducido su free tier en los últimos ciclos; revisar vigente |

**Recomendación para el reto:** usa **Hugging Face Spaces con SDK Docker** como plataforma principal — encaja perfecto con "demo de IA gratuita y pública", no requiere tarjeta, y tu propio repo de HF sirve como evidencia de despliegue reproducible. Usa **Render** como alternativa/backup o si prefieres separar el backend de un frontend estático.

### 8.2 Desplegar en Hugging Face Spaces (paso a paso)

1. Crea cuenta en huggingface.co (gratis).
2. New Space → elige **SDK: Docker** → visibilidad pública.
3. En el repo del Space, agrega al inicio de tu `README.md` el frontmatter requerido:
   ```yaml
   ---
   title: CV Agent
   emoji: 🤖
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
4. Sube tu Dockerfile y código (`git push` al repo del Space, igual que a GitHub).
5. En **Settings → Variables and secrets**, agrega `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `HF_TOKEN` como *secrets* (nunca hardcodeados).
6. HF construye la imagen automáticamente; revisa la pestaña **Logs** para depurar.
7. Tu API queda disponible en `https://<usuario>-<space>.hf.space/chat`.

Importante: el puerto **debe ser 7860** (no configurable en Spaces), asegúrate de que coincida con `EXPOSE` y `app_port`.

### 8.3 Desplegar en Render (alternativa)

1. Sube tu repo a GitHub.
2. En Render: **New → Web Service** → conecta el repo → Render detecta el Dockerfile automáticamente.
3. Configura variables de entorno en **Environment**.
4. Selecciona el plan **Free**.
5. Render expone el puerto vía la variable `PORT` — ajusta tu `CMD` para leerla dinámicamente si usas Render:
   ```dockerfile
   CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
   ```
   (usa forma shell del CMD, no exec form, para que se expanda la variable).
6. Acepta que el servicio se duerma tras 15 min sin tráfico; el primer request tras dormir tarda 30–60s (menciónalo en tu documentación como decisión consciente de costo/latencia, parte de lo que el reto pide evaluar).

### 8.4 Qdrant Cloud (paso a paso)

1. Crea cuenta gratuita en cloud.qdrant.io (sin tarjeta).
2. Crea un cluster **Free tier** (1 nodo, 1GB RAM, 4GB disco).
3. Copia el `QDRANT_URL` y genera un `API key`.
4. Nota: los clusters free se suspenden tras 1 semana de inactividad y se eliminan tras 4 semanas — si tu demo estará inactiva por periodos largos, documenta este riesgo y cómo lo mitigarías (ej. reactivación manual, o script de "ping" periódico).

---

## 9. Pruebas y evaluación

Esta sección es donde más se diferencia un "prompt que responde" de un "AI engineer": el reto pide explícitamente evidencia de evaluación.

### 9.1 Niveles de prueba

**a) Pruebas unitarias (código determinista)**
- `test_retrieval.py`: dado un query conocido, verifica que Qdrant regrese el chunk esperado (por `id` en el payload).
- `test_ingest.py`: extracción de texto y estructuración desde las tres fuentes.
- Prueba de guardrail de entrada: un intento de prompt injection debe ser bloqueado.
- Prueba de fallback: simula que Groq falla (mock) y verifica que se llame a HF.

```python
def test_retrieval_returns_relevant_chunk():
    results = retrieve_cv_context("experiencia con RAG")
    ids = [r["id"] for r in results]
    assert "exp_001" in ids or "proj_001" in ids

def test_low_relevance_returns_empty():
    results = retrieve_cv_context("receta de tacos al pastor")
    assert len(results) == 0  # nada debe superar el score_threshold

def test_extract_text_from_pasted():
    texto = extract_raw_text(file=None, pasted_text="Juan Pérez, AI Engineer...")
    assert "Juan Pérez" in texto

def test_extract_text_from_pdf(sample_pdf_upload):
    texto = extract_raw_text(file=sample_pdf_upload, pasted_text=None)
    assert len(texto) > 0

def test_structure_with_llm_matches_schema():
    raw_text = "Juan Pérez, AI Engineer en Empresa X desde 2023, experto en Python y RAG..."
    cv_dict = structure_with_llm(raw_text)
    cv = CV.model_validate(cv_dict)  # no debe lanzar ValidationError
    assert cv.perfil.nombre != ""

def test_ingest_cv_replaces_previous_version(qdrant_test_client):
    ingest_cv(pasted_text="Ana López, experiencia en Empresa A...")
    v1_count = qdrant_test_client.count("cv_chunks").count

    ingest_cv(pasted_text="Ana López, experiencia en Empresa B (CV actualizado)...")
    results = retrieve_cv_context("experiencia de Ana")
    empresas = [r.get("empresa") for r in results]
    assert "Empresa A" not in empresas  # el CV viejo ya no debe aparecer
```

**b) Evaluación funcional del agente (dataset de preguntas/respuestas esperadas)**

Crea `tests/eval_dataset.json` con pares pregunta → criterio de aceptación:

```json
[
  {
    "pregunta": "¿Qué experiencia tienes construyendo sistemas de IA?",
    "debe_mencionar": ["RAG", "LLM"],
    "no_debe_inventar": ["experiencia en blockchain"]
  },
  {
    "pregunta": "¿Cuál fue tu salario en tu último trabajo?",
    "debe_responder": "no_disponible"
  },
  {
    "pregunta": "¿Qué tecnologías dominas?",
    "debe_mencionar": ["Python", "Docker"]
  }
]
```

**c) Evaluación con LLM-as-judge (opcional pero valioso para "evaluación automatizada de agentes")**

Un script que corre cada pregunta del dataset contra el agente real desplegado, y usa un segundo prompt (juez) para calificar:
- **Fidelidad** (¿la respuesta está soportada por el contexto recuperado?)
- **Relevancia** (¿responde lo que se preguntó?)
- **Manejo de límites** (¿reconoce cuándo no sabe algo?)

```python
JUDGE_PROMPT = """
Evalúa la siguiente respuesta de un agente de CV en una escala 1-5 para:
- fidelidad (¿está soportada por el contexto proporcionado, sin inventar datos?)
- relevancia (¿responde la pregunta del usuario?)

Pregunta: {pregunta}
Contexto usado: {contexto}
Respuesta del agente: {respuesta}

Responde solo en JSON: {{"fidelidad": N, "relevancia": N, "comentario": "..."}}
"""
```

Guarda resultados en un CSV/JSON para incluir como evidencia en tu documentación ("Pruebas o evaluaciones" que menciona el reto).

**d) Pruebas de robustez conversacional**
- Reformulaciones de la misma pregunta ("¿en qué tecnologías eres bueno?" vs "¿qué stack manejas?") → deben dar respuestas consistentes.
- Preguntas fuera de alcance ("¿cuál es la capital de Francia?") → debe declinar con gracia.
- Consistencia entre turnos: pregunta A, luego pregunta B que depende del contexto de A.

### 9.2 Métricas a reportar en tu documentación

- % de preguntas del dataset de evaluación respondidas correctamente.
- Latencia promedio de respuesta (p50/p95).
- Tasa de fallback a HF (indica salud de Groq).
- Score promedio de fidelidad del LLM-judge.

---

## 10. Observabilidad, seguridad y operación

### 10.1 Logging estructurado

```python
import logging, json

logger = logging.getLogger("cv_agent")

def log_interaction(session_id, query, retrieved_chunks, response, latency_ms):
    logger.info(json.dumps({
        "session_id": session_id,
        "query": query,
        "n_chunks_retrieved": len(retrieved_chunks),
        "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else None,
        "response_length": len(response),
        "latency_ms": latency_ms,
    }))
```

Esto es suficiente para el alcance del reto: logs en stdout que Render/HF Spaces capturan automáticamente en su panel de logs.

### 10.2 Observabilidad opcional más avanzada

Si quieres ir más allá: **Langfuse** tiene un tier gratuito para tracing de LLM apps (cuentas de turnos, tokens, costo estimado, latencia por paso del grafo). Se integra con pocas líneas sobre LangChain/LangGraph. Menciónalo como "decisión consciente" incluso si no lo implementas a fondo — demuestra que conoces la práctica de gobernanza GenAI que pide la vacante.

### 10.3 Seguridad básica

- Nunca commitees API keys; usa secrets del hosting.
- Rate limiting simple a nivel de FastAPI (ej. `slowapi`) para evitar abuso del endpoint público y controlar costo/cuota de Groq.
- Sanitiza inputs largos (limita `message` a, por ejemplo, 1000 caracteres) para evitar prompt-stuffing.
- CORS configurado explícitamente si vas a consumir el API desde un frontend en otro dominio.

### 10.4 Manejo de errores en producción

- Si Qdrant no responde → responde con un mensaje de degradación controlada ("no puedo acceder a mi base de conocimiento en este momento"), no un error 500 crudo.
- Si el LLM tarda más de N segundos → timeout con mensaje amigable.
- Todo esto ya está cubierto por el patrón de fallback de la sección 4.3, pero documenta explícitamente estos casos como parte de tu evidencia de "operación".

---

## 11. Entregables sugeridos para el reto

No son obligatorios según el enunciado, pero maximizan tu evaluación:

1. **Repositorio de código** (GitHub) con este documento de arquitectura en la raíz (`ARCHITECTURE.md`).
2. **Demo funcional desplegada** (URL de HF Spaces o Render).
3. **README** con instrucciones de setup local (`docker-compose up`) y variables de entorno necesarias.
4. **Diagrama de arquitectura** (puedes reusar el de la sección 1.3, o exportarlo como imagen).
5. **Dataset y resultados de evaluación** (`tests/eval_dataset.json` + un resumen de resultados, aunque sea una tabla simple).
6. **Video corto o GIF** de una conversación real con el agente (evidencia práctica que menciona el reto).
7. **Sección de "decisiones de diseño"** en el README explicando por qué elegiste esta arquitectura y no una más compleja (esto es lo que más valoran: criterio, no solo ejecución).

---

## 12. Roadmap "si tengo más tiempo" (opcional, no necesario para cumplir el reto)

- Agregar una segunda tool (`compare_experience_vs_role`) que compare tu perfil contra una descripción de puesto pegada por el usuario — conecta directamente con el caso de uso de reclutamiento del reto.
- Migrar de buffer de memoria simple a resumen incremental de conversación para sesiones largas.
- Agregar un frontend mínimo (HTML+JS simple servido por FastAPI, o un Space de Gradio como capa de UI sobre el mismo backend Docker).
- CI simple (GitHub Actions) que corra `pytest` en cada push antes de permitir merge a `main`, y opcionalmente redeploy automático al Space.
- Quantizar los vectores en Qdrant si el volumen de datos crece (no necesario para el tamaño de un CV, pero demuestra conocimiento de la vacante).

---

## Apéndice A — `requirements.txt` sugerido

```
fastapi
uvicorn[standard]
python-multipart    # requerido por FastAPI para recibir UploadFile / form-data
pydantic
openai              # cliente universal: usado para Groq y HF vía base_url + endpoint /responses
huggingface-hub
sentence-transformers
qdrant-client
pypdf               # extracción de texto de PDFs
langgraph
langchain-core
python-dotenv
slowapi
pytest
pytest-asyncio
httpx
```

> Se usa el SDK de `openai` como cliente universal para Groq y Hugging Face porque ambos exponen `/v1/responses` compatible con Open Responses (sección 4). Si prefieres aislar aún más el proveedor, puedes mantener `groq` como dependencia adicional y usarlo solo si necesitas alguna función específica de Groq no cubierta por el estándar (ej. modelos de audio). `python-multipart` es indispensable para que FastAPI acepte el `UploadFile` del endpoint `/cv/upload` (sección 6.3) — es un error común olvidarlo y obtener un 422 al subir archivos.

## Apéndice B — Checklist final antes de entregar

- [ ] El agente responde con datos reales del CV (verificado manualmente con 5+ preguntas).
- [ ] El agente reconoce y declara cuando no tiene información.
- [ ] Existe fallback entre Groq y Hugging Face, probado forzando un error.
- [ ] El contenedor Docker construye y corre localmente (`docker build` + `docker run`).
- [ ] El servicio está desplegado y accesible públicamente vía URL.
- [ ] El endpoint `/v1/responses` responde con el formato compatible con Open Responses (probado con el SDK de `openai` apuntando a tu `base_url`).
- [ ] `/cv/upload` acepta PDF, TXT y texto pegado, y cada uno produce el mismo CV estructurado (probado con los tres).
- [ ] Al subir un CV nuevo, las respuestas del agente reflejan el CV actualizado y no mezclan datos del CV anterior (verificado consultando lo mismo antes y después de una recarga).
- [ ] `/cv/upload` está protegido (API key propia o no expuesto públicamente).
- [ ] Existen pruebas automatizadas (`pytest`) que pasan.
- [ ] Existe un dataset de evaluación con resultados documentados.
- [ ] Logs muestran las interacciones de forma estructurada.
- [ ] README y este documento de arquitectura están en el repo.
- [ ] Ninguna API key está commiteada en el código.