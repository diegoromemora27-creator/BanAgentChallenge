# 🤖 Agente Conversacional de CV · Reto IA Banorte (Generative AI Engineer)

Un **Agente Conversacional RAG Estricto** diseñado para responder preguntas sobre la experiencia profesional, habilidades y proyectos de un candidato, garantizando fundamentación (grounding), cero alucinaciones y estricto control de límites de información. 

El agente expone sus capacidades a través de una API construida en **FastAPI**, compatible con la especificación abierta **Open Responses API** (`/v1/responses`), e indexa conocimiento en tiempo real utilizando **Qdrant Cloud**, persistencia de estados con **Neon / Supabase PostgreSQL** y un motor resiliente de LLMs con **Hugging Face Inference Router** (Llama 3.1 8B) y fallback automático a **Groq API** (Llama 3.3 70B).

---

## 📐 1. Arquitectura General del Sistema

```
                         ┌───────────────────────────┐
                         │        Usuario / Web       │
                         │  (Chat UI o Cliente HTTP) │
                         └─────────────┬──────────────┘
                                       │ HTTPS
                                       ▼
                    ┌───────────────────────────────────┐
                    │     FastAPI (Contenedor Docker)   │
                    │ /health  /chat  /v1/responses     │
                    │ /cv/upload  /cv/info              │
                    │  ┌─────────────────────────────┐  │
                    │  │   Orquestador LangGraph     │  │
                    │  │                             │  │
                    │  │  1. Guardrails de entrada   │  │
                    │  │  2. Clasificador intenciones│  │
                    │  │  3. Tool: Vector Retrieval  │──┼──► Qdrant Cloud (cv_chunks)
                    │  │  4. Grounded LLM Generation │──┼──► HF Router / Groq Fallback
                    │  │  5. Guardrails de salida    │  │
                    │  │  6. State Checkpointer      │──┼──► Neon / Supabase Postgres
                    │  └─────────────────────────────┘  │
                    │                                   │
                    │  Logging Estructurado (JSON)       │
                    └───────────────────┬───────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │ Observabilidad / Logs     │
                         │ (Tokens, Latencia, Scores)│
                         └───────────────────────────┘
```

### 🧩 Componentes Principales

| Componente | Tecnología Seleccionada | Razón Arquitectónica |
|---|---|---|
| **API Backend** | FastAPI + Pydantic v2 | Alto rendimiento asíncrono, OpenAPI auto-generado y tipado estricto. |
| **Protección & Cuota** | `slowapi` Rate Limiter (60 req/min) | Previene ataques de denegación de servicio y el consumo acelerado de cuotas de LLM/Embeddings. |
| **Estándar Interoperable** | Open Responses (`/v1/responses`) | Compatible con el SDK oficial de OpenAI y catálogos de agentes; mapea modelos arbitrarios y maneja `stream=false` según la especificación. |
| **Base de Datos Vectorial** | Qdrant Cloud (`cv_chunks`) | Búsqueda por similitud de cosenos, filtros por payload (`tipo`, `cv_version`) e índices de metadatos automáticos. |
| **Embeddings por API** | HF Router API (`bge-small-en-v1.5`) | Generación ligera por API sin descargar PyTorch/Torch localmente (ahorra ~800MB RAM, total app ~120MB). |
| **LLM Principal** | Hugging Face Inference Router (`Llama-3.1-8B-Instruct`) | Inferencia primaria gratuita por API para maximizar ahorro de cuotas. |
| **LLM Fallback** | Groq API (`llama-3.3-70b-versatile`) | Conmutación de respaldo automática de ultra-baja latencia (LPU) en caso de fallos de cuota. |
| **Orquestación & Grafo** | LangGraph + LangChain Core | Grafo de estados explícito para controlar el flujo, guardrails y recuperación. |
| **Persistencia (Checkpointer)** | Neon / Supabase PostgreSQL (`PostgresSaver`) | Almacenamiento persistente de conversaciones por `session_id` con fallback a `MemorySaver`. |
| **Mitigación de Cold Starts**| Pre-warming en `GET /health` | Pings livianos en segundo plano a las bases de datos para evitar timeouts de catálogos externos. |

---

## 🛠️ 2. Guía de Instalación y Ejecución Local Paso a Paso

### Requisitos Previos
- Python 3.10 o superior (Probado en Python 3.11 / 3.14).
- Docker (Opcional, para ejecución contenida).

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/diegoromemora27-creator/BanAgentChallenge.git
cd BanAgentChallenge
```

### Paso 2: Crear y Activar Entorno Virtual
```bash
# En Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# En Linux / macOS / Codespaces:
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias Ligeras
```bash
pip install -r requirements.txt
```

### Paso 4: Configurar Variables de Entorno (`.env`)
Copia la plantilla `.env.example` para crear tu archivo `.env`:

```bash
cp .env.example .env
```

Edita el archivo `.env` configurando tus llaves y credenciales:
```env
GROQ_API_KEY=gsk_tu_groq_key_aqui
HF_TOKEN=hf_tu_huggingface_token_aqui
QDRANT_URL=https://tu-cluster.cloud.qdrant.io
QDRANT_API_KEY=tu_qdrant_api_key_aqui
DATABASE_URL=postgresql://neondb_owner:tu_password@ep-xxx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require
```

> **Nota de Respaldo:** Si dejas `QDRANT_URL` o `DATABASE_URL` vacíos, el sistema iniciará automáticamente en modo de respaldo utilizando **Qdrant en Memoria (`:memory:`)** y **MemorySaver en RAM**, lo cual es ideal para pruebas ultrarrápidas sin dependencias externas.

### Paso 5: Iniciar el Servidor API
```bash
uvicorn app.main:app --reload --port 7860
```

El servidor estará disponible en:
- **Swagger Documentation:** `http://localhost:7860/docs`
- **Healthcheck:** `http://localhost:7860/health`

---

## 🛰️ 3. Endpoints de la API HTTP

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Verificación de estado y salud del servicio. |
| `POST` | `/chat` | Interfaz simplificada de conversación para UIs de chat (`message`, `session_id`). |
| `POST` | `/v1/responses` | Endpoint interoperable compatible con la especificación **Open Responses**. |
| `POST` | `/cv/upload` | Ingesta multi-fuente (PDF, TXT, texto pegado), estructuración LLM e indexación en Qdrant. |
| `GET` | `/cv/info` | Endpoint de inspección para listar los chunks y metadatos actualmente indexados en Qdrant. |

---

## 🧪 4. Suite de Pruebas y Benchmark LLM-as-a-Judge

### Pruebas Unitarias Automatizadas
Verifica la defensa contra Prompt Injection, extracción de texto y validaciones de Pydantic:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Benchmark "LLM-as-a-Judge"
Mide métricas cuantitativas de **Fidelidad** y **Relevancia** (1.0 a 5.0) evaluando el agente contra el conjunto de preguntas de prueba (`tests/eval_dataset.json`):
```bash
python tests/eval_llm_judge.py
```

---

## 📊 5. Observabilidad, Métricas y Muestra de Logs Estructurados

El agente registra automáticamente eventos JSON estructurados en `stdout` con cada interacción para observabilidad en **Hugging Face Spaces**, **Render** o plataformas APM:

```json
{
  "event": "agent_interaction",
  "timestamp": "2026-08-10T22:51:25Z",
  "session_id": "sesion_demo_100",
  "query": "¿Qué experiencia tienes en Python y RAG?",
  "n_chunks_retrieved": 4,
  "top_score": 0.842,
  "response_length": 310,
  "latency_ms": 1350.2,
  "provider": "Hugging Face",
  "usage": {
    "prompt_tokens": 380,
    "completion_tokens": 75,
    "total_tokens": 455
  }
}
```

---

## 🐳 6. Contenerización con Docker y Docker Compose

### Construcción y Ejecución Única con Docker
```bash
# Construir la imagen ligera (~120MB RAM)
docker build -t cv-agent:latest .

# Ejecutar el contenedor
docker run -d -p 7860:7860 --env-file .env --name cv-agent-container cv-agent:latest
```

### Ejecución Completa con Docker Compose (API + Qdrant Local)
```bash
docker-compose up --build
```

---

## 🚀 7. Despliegue en Plataformas Cloud Gratuitas

### A) Despliegue en Hugging Face Spaces (Opción Recomendada)
1. Crea un nuevo Space en [huggingface.co/new-space](https://huggingface.co/new-space).
2. Selecciona **SDK: Docker** (Blank Docker template).
3. Configura el frontmatter al inicio del `README.md` del Space:
   ```yaml
   ---
   title: CV Conversational Agent
   emoji: 🤖
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
4. Agrega los Secretos en **Settings -> Variables and secrets**: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `HF_TOKEN`, `DATABASE_URL`.
5. Haz git push de tu código al repositorio del Space.

### B) Despliegue en Render Free Service
1. Crea un nuevo **Web Service** en Render conectando tu repositorio de GitHub.
2. Render detectará automáticamente el `Dockerfile`.
3. Agrega las Variables de Entorno en el panel de Render.

---

## 🔮 8. Roadmap de Posibles Mejoras Futuras

1. **Re-Ranking de Vectores (Cohere / BGE Re-Ranker):**
   - Incorporar una etapa de re-ordenamiento semántico posterior a la búsqueda inicial de Qdrant para priorizar chunks de alta especificidad.
2. **Evaluación Continua con Langfuse Cloud:**
   - Conectar las trazas de ejecuciones de LangGraph directamente a Langfuse para medir costo de tokens y latencias p95 en tiempo real.
3. **Streaming Semántico Real en `/v1/responses`:**
   - Habilitar soporte para Server-Sent Events (`stream=True`) siguiendo al 100% el estándar Open Responses para respuestas fluídas en UIs web.
4. **Soporte OCR para PDFs Escaneados:**
   - Integrar `pytesseract` o Amazon Textract en `app/rag/ingest.py` para procesar CVs escaneados o en formato imagen.
