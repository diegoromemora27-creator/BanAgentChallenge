# 🤖 Agente Conversacional de CV · Reto IA Banorte (Generative AI Engineer)

Un **Agente Conversacional RAG Estricto** diseñado para responder preguntas sobre la experiencia profesional, habilidades, educación, proyectos y datos de contacto de **Diego Romero Mora** (Senior AI Engineer), garantizando fundamentación (grounding), cero alucinaciones y estricto control de límites de información. 

El agente expone sus capacidades a través de una API modular construida en **FastAPI** (`app/api/`), compatible con la especificación abierta **Open Responses API** (`/v1/responses`), e indexa conocimiento en tiempo real utilizando **Qdrant Cloud** con chunking jerárquico, persistencia de estados con **Supabase PostgreSQL** (`PostgresSaver`), observabilidad y tracing con **Langfuse Cloud (SDK v3)** y monitoreo de métricas en tiempo real con **Prometheus & Grafana Cloud**.

---

## 📐 1. Arquitectura General del Sistema

```
                         ┌───────────────────────────┐
                         │        Usuario / Web       │
                         │  (Chat UI o Cliente HTTP) │
                         └─────────────┬──────────────┘
                                       │ HTTPS (Bearer Auth)
                                       ▼
                    ┌───────────────────────────────────┐
                    │     FastAPI (Contenedor Docker)   │
                    │      Routers Modulares (app/api/) │
                    │ /health  /chat  /v1/responses     │
                    │ /cv/upload  /metrics              │
                    │  ┌─────────────────────────────┐  │
                    │  │   Orquestador LangGraph     │  │
                    │  │                             │  │
                    │  │  1. Guardrails de entrada   │  │
                    │  │  2. Clasificador intenciones│──┼──► Conciencia de Historial
                    │  │  3. Intent-Aware Retrieval  │──┼──► Qdrant Cloud (MatchAny / Score >= 0.0)
                    │  │  4. Grounded LLM Generation │──┼──► Groq LPU (Llama 3.3/3.1) / HF Fallback
                    │  │  5. Guardrails & Fail-Loud  │  │
                    │  │  6. State Checkpointer      │──┼──► Supabase PostgreSQL
                    │  └──────────────┬──────────────┘  │
                    │                 │                 │
                    │  Langfuse CallbackHandler (v3)    │──┼──► Langfuse Cloud (Spans / Chunk IDs)
                    │  Prometheus Metrics Engine        │──┼──► Grafana Cloud (/metrics)
                    └───────────────────────────────────┘
```

### 🧩 Componentes Principales

| Componente | Tecnología Seleccionada | Razón Arquitectónica |
|---|---|---|
| **API Backend Modular** | FastAPI + APIRouters (`app/api/`) | Separación clara de responsabilidades (`chat.py`, `cv.py`, `meta.py`, `deps.py`), OpenAPI auto-generado. |
| **Seguridad & Autenticación** | Header `Authorization` / `X-API-Key` | Control de acceso basado en API Key configurable (`API_KEY`) para proteger endpoints públicos. |
| **Protección & Cuota** | `slowapi` Rate Limiter (60 req/min) | Previene ataques de denegación de servicio y consumo acelerado de cuotas de LLMs. |
| **Estándar Interoperable** | Open Responses (`/v1/responses`) | Compatible con el SDK oficial de OpenAI (`openai` Python/TS) y streaming SSE en tiempo real. |
| **Base de Datos Vectorial** | Qdrant Cloud (`cv_chunks`) | Búsqueda por similitud de cosenos con metadatos filtrados por tipo (`MatchAny`) y fallback a `:memory:`. |
| **Embeddings por API** | Hugging Face Router API (`all-MiniLM-L6-v2`) | Sanitización de consultas (remoción de saludos), reintentos con backoff exponencial y 0 MB de consumo local de RAM. |
| **LLM Principal & Fallback** | Groq LPU (`Llama-3.3-70B` / `Llama-3.1-8B`) + HF Router | Conmutación automática de alta disponibilidad en caso de límites de cuota (latencias < 600 ms). |
| **Orquestación & Grafo** | LangGraph + LangChain Core | Grafo de estados explícito con 4 nodos secuenciales, resolución de anáforas y guardrail *Fail-Loud*. |
| **Tono e Identidad** | Representante Oficial (Tercera Persona) | Responde como el asistente digital oficial de Diego, garantizando máxima transparencia y cero riesgos de suplantación. |
| **Persistencia de Estados** | Supabase PostgreSQL (`PostgresSaver`) | Almacenamiento persistente de conversaciones por `session_id` (o `previous_response_id`) con fallback a `MemorySaver`. |
| **Tracing & Observabilidad** | Langfuse Cloud SDK v3 | Trazabilidad detallada de spans, ejecuciones de grafo y registro explícito de `chunk_ids` y `chunk_types`. |
| **Métricas & Dashboards** | Prometheus Client + Grafana Cloud | Scraping del endpoint `/metrics` protegido por token `METRICS_TOKEN`. |

---

## 🗄️ 2. Arquitectura de Datos, Ingesta y Persistencia

### A. Chunking Jerárquico e Indexación en Qdrant Cloud (`cv_chunks`)
- **Colección:** `cv_chunks`
- **Dimensión:** 384 vectores (`sentence-transformers/all-MiniLM-L6-v2`).
- **Métrica de Distancia:** Coseno (`Distance.COSINE`).
- **Estructuración Semántica Extendida (`app/models/schemas.py`):**
  La ingesta extrae y valida mediante Pydantic todas las secciones del CV sin omitir información:
  1. `perfil` & `contacto` (Email, teléfono `5560438272` y aclaración explícita de LinkedIn).
  2. `experiencia` (Puestos, logros, fechas y tecnologías).
  3. **Chunk Resumen Jerárquico:** Un chunk acumulativo que condensa la trayectoria completa (Teradata, Insulet, Solera, CTIN) para responder preguntas generales de un solo vistazo.
  4. `educacion` (Maestría en Data Science - UTM, Licenciatura - UNAM).
  5. `certificaciones` (AWS Certified, Google AI Leader, Oracle AI Associate).
  6. `cursos_selectos` (Diplomado en Administración de Bases de Datos - UNAM).
  7. `colaboracion_academica` (Docencia en la UNAM en la carrera de Matemáticas Aplicadas y Computación).
  8. `meta` (Chunk explícito anti-alucinaciones).

### B. Retrieval Consciente de Intención (*Intent-Aware Retrieval*)
En lugar de una búsqueda semántica "ciega", el retriever utiliza la clasificación previa de intención para aplicar un filtro `MatchAny` por la clave `tipo` en el payload de Qdrant:
- `CONTACT` $\rightarrow$ `["contacto", "perfil"]`
- `EDUCATION` $\rightarrow$ `["educacion", "cursos", "certificaciones"]`
- `EXPERIENCE` $\rightarrow$ `["experiencia", "docencia"]`
- `SKILLS` $\rightarrow$ `["skills"]`
- `PROJECTS` $\rightarrow$ `["proyectos"]`

Además, se aplica **sanitización de embeddings** (`clean_query_for_embedding`) para remover saludos como `"Hola"`, `"Buenas tardes"` del texto antes de generar el vector, elevando los `top_score` a valores positivos y relevantes.

### C. Supabase PostgreSQL (Checkpointer de LangGraph)
- **Módulo:** `langgraph-checkpoint-postgres` (`PostgresSaver`).
- **Persistencia de Conversación:** Almacena checkpoints mapeando el `previous_response_id` o `session_id` del usuario, manteniendo la memoria de diálogo entre turnos sin duplicar mensajes en el prompt.

---

## 🛡️ 3. Demostración de Grounding Estricto, Tono y Guardrails

El sistema aplica una política de **Grounding Estricto y Fail-Loud** garantizada por un pipeline de cuatro etapas:

1. **Guardrail de Entrada (Input Moderation):** Detecta ataques de Prompt Injection e instrucciones maliciosas antes de tocar el grafo.
2. **Clasificación de Intención Consciente del Contexto (Intent Classification):** Pasa los últimos turnos de la sesión para resolver anáforas y preguntas de seguimiento (*"¿Cuánto tiempo llevó ahí?"* $\rightarrow$ mapea a `EXPERIENCE` sobre Teradata).
3. **Intent-Aware Retrieval:** Consulta Qdrant filtrando por tipo de chunk y sanitizando saludos.
4. **Guardrail Fail-Loud & Grounding Verification:** Si el RAG obtiene `0 chunks` en consultas informativas, el sistema **falla ruidosamente** devolviendo un mensaje seguro de sistema sin permitir que el LLM alucine o invente trayectoria.

### 🧪 Ejemplos de Comportamiento del Agente:

| Caso de Prueba | Entrada del Usuario | Respuesta del Agente (en 3ra Persona) | Explicación del Comportamiento |
|---|---|---|---|
| **Duda en el CV (In-Bounds)** | *¿Qué experiencia tiene Diego en Inteligencia Artificial?* | *"Diego cuenta con experiencia directa como Senior AI Automation Engineer en Teradata, donde diseñó flujos con LangGraph, integró conectores MCP/FastMCP y optimizó APIs de modelos frontera..."* | **Respuesta fundamentada en 3ra persona:** Afirma con seguridad las tecnologías de IA documentadas. |
| **Pregunta de Seguimiento (Anáfora)** | *¿Y cuánto tiempo estuvo ahí?* | *"Diego se desempeñó como Senior AI Automation Engineer en Teradata durante el período de diciembre de 2025 a julio de 2026..."* | **Memoria & Regla Temporal:** El clasificador resuelve que "ahí" es Teradata y aplica la regla de fecha pasada. |
| **Datos de Contacto** | *¿Cómo puedo contactar a Diego?* | *"Puedes contactar a Diego Romero Mora directamente a través de su correo electrónico diegoromemora27@gmail.com o por teléfono al 5560438272..."* | **Retrieval de Contacto:** Recupera el chunk explícito de contacto directo. |
| **Consulta Fuera de Dominio (Out-of-Bounds)** | *¿Cuál es la receta para preparar tacos al pastor?* | *"Como agente conversacional enfocado en el perfil profesional de Diego, solo estoy capacitado para responder preguntas sobre su experiencia laboral, proyectos, habilidades e historia profesional."* | **Filtro de Intención:** Corta la ejecución sin consultar a Qdrant ni gastar tokens inútilmente. |
| **Falla Momentánea de Red en DB** | *¿Qué cursos impartió en la UNAM?* | *"En este momento no pude acceder a los datos de la base de conocimiento del CV de Diego para responder tu consulta. Por favor, intenta realizar tu pregunta nuevamente."* | **Guardrail Fail-Loud (Cero Alucinación):** Si se obtienen 0 chunks por problemas de red, el sistema bloquea cualquier intento de invención. |

---

## 🛰️ 4. Trazabilidad y Observabilidad Avanzada

### A. Langfuse Cloud (SDK v3 Integration)
El sistema utiliza el SDK v3 de Langfuse e inyecta métricas avanzadas de observabilidad en cada turno:
- **`chunk_ids`**: Lista explícita de IDs de vectores leídos en Qdrant.
- **`chunk_types`**: Lista de tipos de contenido recuperados (`['experiencia', 'docencia']`).
- **`top_score`**: Puntaje de similitud Cosine del candidato #1.

### B. Grafana Cloud & Prometheus Metrics
El endpoint `/metrics` expone métricas nativas para scraping en Grafana Cloud:
- **`AGENT_REQUESTS_TOTAL`**: Total de solicitudes procesadas.
- **`NODE_EXECUTION_DURATION_SECONDS`**: Latencia desglosada por cada uno de los 4 nodos de LangGraph.
- **`LLM_TOKENS_TOTAL`**: Total de tokens procesados por proveedor (`Groq`, `Hugging Face`).

---

## 📁 5. Estructura Modular del Proyecto (`app/`)

```
BanAgentChallenge/
├── app/
│   ├── api/                      # Routers modulares HTTP (FastAPI)
│   │   ├── chat.py               # Endpoints /chat, /responses y /v1/responses (SSE Streaming)
│   │   ├── cv.py                 # Endpoints /cv/upload y /cv/info
│   │   ├── meta.py               # Endpoints /, /.well-known/agent-card.json, /health, /metrics
│   │   └── deps.py               # Verificación reutilizable de API Key (Bearer/X-API-Key)
│   ├── agent/                    # Lógica del Agente Conversacional y Grafo
│   │   ├── graph.py              # Definición de AgentState y nodos de LangGraph
│   │   ├── guardrails.py         # Moderación de entrada, salida y clasificación
│   │   ├── memory.py             # Buffer de memoria por session_id
│   │   └── prompts.py            # SYSTEM_GROUNDING_PROMPT, CLASSIFY_INTENT_PROMPT, etc.
│   ├── llm/                      # Proveedores de Inferencia de LLM
│   │   ├── provider.py           # Orquestador resiliente con fallback
│   │   ├── groq_client.py        # Cliente Groq LPU (Llama 3.3 / Llama 3.1)
│   │   └── hf_client.py          # Cliente Hugging Face Router API (Fallback)
│   ├── models/                   # Esquemas Pydantic v2
│   │   └── schemas.py            # Modelos del CV (Perfil, Educacion, ColaboracionAcademica, etc.)
│   ├── rag/                      # Capa de RAG e Indexación Vectorial
│   │   ├── ingest.py             # Extracción LLM, chunking jerárquico y reemplazo en Qdrant
│   │   └── retriever.py          # Búsqueda vectorial, sanitización de query y filtro MatchAny
│   ├── config.py                 # Configuración Pydantic Settings (.env)
│   ├── logging_config.py         # Formateador de logs JSON estructurados
│   ├── metrics.py                # Métricas Prometheus
│   └── main.py                   # Instancia FastAPI principal y registro de routers
├── data/
│   └── cv_sample.txt             # CV crudo oficial de Diego Romero Mora
├── tests/
│   ├── test_unit.py              # Pruebas unitarias de esquemas, intenciones y RAG
│   └── test_agent_card.py        # Validación de tarjeta A2A (.well-known)
├── Dockerfile                    # Configuración de contenedor para producción (Render)
├── docker-compose.yml            # Orquestador local Docker
└── README.md                     # Documentación oficial del proyecto
```

---

## 🛠️ 6. Guía de Instalación y Configuración Local

### Requisitos Previos
- Python 3.10 a 3.14 (Dependencias ancladas).
- Git y Docker.

### Paso 1: Clonar el Repositorio e Instalar Dependencias
```bash
git clone https://github.com/diegoromemora27-creator/BanAgentChallenge.git
cd BanAgentChallenge
python -m venv .venv
# En Windows: .\.venv\Scripts\Activate.ps1 | En Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### Paso 2: Configurar Variables de Entorno (`.env`)
```env
# LLM Providers
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_hf_token_here

# Vector Database (Qdrant Cloud)
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Persistencia de Estado (Supabase / Neon Postgres)
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require

# Observabilidad & Tracing (Langfuse Cloud v3)
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Seguridad
API_KEY=banorte_challenge_api_key_2026
METRICS_TOKEN=your_metrics_token_here
```

### Paso 3: Iniciar Servidor Local
```bash
uvicorn app.main:app --reload --port 7860
```
Accede a Swagger en: `http://localhost:7860/docs`

---

## 🔗 7. Despliegues y Producción

- 🚀 **Servicio API en Producción (Render):** `https://banagentchallenge.onrender.com`
- 📚 **Documentación Swagger / OpenAPI:** `https://banagentchallenge.onrender.com/docs`
- 🩺 **Endpoint de Healthcheck:** `https://banagentchallenge.onrender.com/health`
- 🃏 **Tarjeta de Agente A2A:** `https://banagentchallenge.onrender.com/.well-known/agent-card.json`
- 📊 **Endpoint de Métricas Prometheus:** `https://banagentchallenge.onrender.com/metrics`
- 🔍 **Trazabilidad Langfuse Cloud:** `https://cloud.langfuse.com`
- 🗄️ **Cluster Qdrant Cloud:** `cv_chunks` (384 dimensiones, Cosine)
