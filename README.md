# 🤖 Agente Conversacional de CV · Reto IA Banorte (Generative AI Engineer)

Un **Agente Conversacional RAG Estricto** diseñado para responder preguntas sobre la experiencia profesional, habilidades y proyectos de un candidato, garantizando fundamentación (grounding), cero alucinaciones y estricto control de límites de información. 

El agente expone sus capacidades a través de una API construida en **FastAPI**, compatible con la especificación abierta **Open Responses API** (`/v1/responses`), e indexa conocimiento en tiempo real utilizando **Qdrant Cloud**, persistencia de estados con **Neon / Supabase PostgreSQL** (`PostgresSaver`), observabilidad y tracing con **Langfuse Cloud (SDK v3)** y monitoreo de métricas en tiempo real con **Prometheus & Grafana Cloud**.

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
                    │ /cv/upload  /metrics              │
                    │  ┌─────────────────────────────┐  │
                    │  │   Orquestador LangGraph     │  │
                    │  │                             │  │
                    │  │  1. Guardrails de entrada   │  │
                    │  │  2. Clasificador intenciones│  │
                    │  │  3. Tool: Vector Retrieval  │──┼──► Qdrant Cloud (cv_chunks)
                    │  │  4. Grounded LLM Generation │──┼──► HF Router / Groq Fallback
                    │  │  5. Guardrails de salida    │  │
                    │  │  6. State Checkpointer      │──┼──► Neon / Supabase Postgres
                    │  └──────────────┬──────────────┘  │
                    │                 │                 │
                    │  Langfuse CallbackHandler (v3)    │──┼──► Langfuse Cloud (Trazas)
                    │  Prometheus Metrics Engine        │──┼──► Grafana Cloud (/metrics)
                    └───────────────────────────────────┘
```

### 🧩 Componentes Principales

| Componente | Tecnología Seleccionada | Razón Arquitectónica |
|---|---|---|
| **API Backend** | FastAPI + Pydantic v2 | Alto rendimiento asíncrono, OpenAPI auto-generado y tipado estricto. |
| **Protección & Cuota** | `slowapi` Rate Limiter (60 req/min) | Previene ataques de denegación de servicio y consumo acelerado de cuotas. |
| **Estándar Interoperable** | Open Responses (`/v1/responses`) | Compatible con el SDK oficial de OpenAI y catálogos de agentes. |
| **Base de Datos Vectorial** | Qdrant Cloud (`cv_chunks`) | Búsqueda por similitud de cosenos con metadatos y fallback a `:memory:`. |
| **Embeddings por API** | HF Router API (`bge-small-en-v1.5`) | Generación por API sin carga local en RAM (ahorra ~800MB RAM, app ~120MB). |
| **LLM Principal & Fallback** | HF Inference Router (`Llama-3.1-8B`) + Groq (`Llama-3.3-70B`) | Conmutación automática de alta disponibilidad en caso de límites de cuota. |
| **Orquestación & Grafo** | LangGraph + LangChain Core | Grafo de estados explícito con 4 nodos secuenciales y guardrails estrictos. |
| **Persistencia de Estados** | Neon / Supabase PostgreSQL (`PostgresSaver`) | Almacenamiento persistente de conversaciones por `session_id` con fallback a `MemorySaver`. |
| **Tracing & Observabilidad** | Langfuse Cloud SDK v3 (`from langfuse.langchain import CallbackHandler`) | Trazabilidad detallada de spans, ejecuciones de grafo y latencias. |
| **Métricas & Dashboards** | Prometheus Client + Grafana Cloud | Scraping del endpoint `/metrics` protegido por token. |

---

## 🗄️ 2. Arquitectura de Datos y Persistencia

### A. Qdrant Cloud (Vector DB para RAG)
- **Colección:** `cv_chunks`
- **Dimensión:** 384 vectores (compatible con `bge-small-en-v1.5` / `paraphrase-multilingual-MiniLM-L12-v2`).
- **Métrica de Distancia:** Coseno (`Distance.COSINE`).
- **Indexación y Ingesta:** Endpoint `/cv/upload` permite subir archivos PDF, TXT o texto plano. El sistema segmenta el contenido en chunks de 500 caracteres (solapamiento de 50) y almacena el vector junto con el payload (`texto`, `tipo`, `cv_version`, `timestamp`).
- **Fallback Automático:** Si `QDRANT_URL` o `QDRANT_API_KEY` no están configurados, el sistema inicia automáticamente un servidor **Qdrant en Memoria (`:memory:`)**, permitiendo ejecutar el agente sin dependencias externas.

### B. Neon / Supabase PostgreSQL (Checkpointer de LangGraph)
- **Módulo:** `langgraph-checkpoint-postgres` (`PostgresSaver`).
- **Conexión:** Utiliza `psycopg_pool.ConnectionPool` con opción `autocommit=True` para compatibilidad nativa con índices concurrentes en Postgres de Neon/Supabase.
- **Persistencia de Conversación:** Almacena checkpoints por cada `thread_id` (correspondiente al `session_id` del usuario), manteniendo la memoria de diálogo entre turnos.
- **Fallback Automático:** Si no se especifica `DATABASE_URL`, la aplicación recurre a `MemorySaver` en RAM sin detener la ejecución.

---

## 🛰️ 3. Trazabilidad y Observabilidad Avanzada

### A. Langfuse Cloud (SDK v3 Integration)
El sistema utiliza el SDK v3 de Langfuse a través de la importación oficial moderna:
```python
from langfuse.langchain import CallbackHandler
```

- **Autodetección de Entorno:**
  El inicializador revisa el objeto `settings` de la aplicación y asigna las variables globales `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` y `LANGFUSE_HOST` en `os.environ`.
- **Compatibilidad con Render.com:**
  Si las credenciales no se encuentran en `settings`, invoca el constructor cero-argumentos `CallbackHandler()`, el cual autodetecta de forma nativa las variables de entorno de Render (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- **Asociación de Sesiones:**
  El callback pasa `langfuse_session_id` en el diccionario de metadatos de LangGraph (`config["metadata"]`), agrupando todas las trazas del usuario en el panel de Langfuse.

### B. Grafana Cloud & Prometheus Metrics
La API expone un endpoint `/metrics` en formato Prometheus estándar:
- **Protección de Seguridad:** Soporta autenticación mediante Bearer Token configurado en la variable `METRICS_TOKEN` (`Header Authorization: Bearer <token>` o parámetro query `?token=<token>`).
- **Métricas Exportadas:**
  - `AGENT_REQUESTS_TOTAL`: Total de solicitudes procesadas por el agente.
  - `AGENT_LATENCY_SECONDS`: Histograma de latencia completa del flujo agéntico.
  - `NODE_EXECUTION_DURATION_SECONDS`: Latencia por cada nodo del grafo (`guardrails_input`, `classify_intent`, `retrieve_context`, `generate_response`).
  - `NODE_ERRORS_TOTAL`: Contador de errores por nodo.
  - `RAG_RETRIEVED_DOCUMENTS_COUNT`: Cantidad de chunks recuperados por búsqueda.
  - `RAG_RELIABILITY_SCORE`: Calificación cuantitativa de grounding/relevancia.
  - `LLM_TOKENS_TOTAL`: Tokens consumidos (`prompt` y `completion`) desglosados por proveedor (`Hugging Face`, `Groq`).
  - `LLM_COST_ESTIMATED_TOTAL`: Costo financiero estimado acumulado en USD.
- **Dashboard de Grafana:** Incluido en el repositorio en `grafana_dashboard_banorte.json`, listo para importar en Grafana Cloud.

---

## 🛠️ 4. Guía de Instalación y Configuración Local

### Requisitos Previos
- Python 3.10 o superior (Probado en Python 3.11 / 3.14).
- Git.
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

# En Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Configurar Variables de Entorno (`.env`)
Copia la plantilla `.env.example` para crear tu archivo `.env`:
```bash
cp .env.example .env
```

Edita `.env` agregando tus credenciales:
```env
# LLM Providers
GROQ_API_KEY=gsk_tu_groq_key_aqui
HF_TOKEN=hf_tu_huggingface_token_aqui

# Qdrant Vector Database
QDRANT_URL=https://tu-cluster.cloud.qdrant.io
QDRANT_API_KEY=tu_qdrant_api_key_aqui

# Persistencia Supabase / Neon PostgreSQL
DATABASE_URL=postgresql://neondb_owner:tu_password@ep-xxx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require

# Observabilidad con Langfuse Cloud (v3)
LANGFUSE_PUBLIC_KEY=pk-lf-tu_public_key
LANGFUSE_SECRET_KEY=sk-lf-tu_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Métricas Grafana Cloud
METRICS_TOKEN=banorte_metrics_secret_token_2026
```

### Paso 5: Iniciar el Servidor API
```bash
uvicorn app.main:app --reload --port 7860
```

Accede a la documentación interactiva OpenAPI (Swagger) en:
`http://localhost:7860/docs`

---

## 🛰️ 5. Especificación de Endpoints HTTP

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Verificación de estado del servicio y salud de bases de datos. |
| `POST` | `/chat` | Endpoint de conversación simplificado con Rate Limiting (`message`, `session_id`). |
| `POST` | `/v1/responses` | Endpoint interoperable compatible con el estándar **Open Responses API**. |
| `POST` | `/cv/upload` | Ingesta multi-fuente (PDF, TXT, texto pegado) e indexación en Qdrant. |
| `GET` | `/cv/info` | Inspección de chunks y metadatos actualmente indexados en Qdrant. |
| `GET` | `/metrics` | Endpoint de métricas en formato Prometheus para Grafana Cloud. |

---

## 🧪 6. Suite de Pruebas y Benchmark LLM-as-a-Judge

### Pruebas Unitarias Automatizadas
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Benchmark "LLM-as-a-Judge"
Mide cuantitativamente **Fidelidad** y **Relevancia** (1.0 a 5.0) evaluando el agente contra preguntas de prueba (`tests/eval_dataset.json`):
```bash
python tests/eval_llm_judge.py
```

---

## 🐳 7. Contenerización con Docker y Docker Compose

### ¿Cómo funciona Docker en esta Arquitectura?
Como todas las bases de datos y servicios de observabilidad son administrados en la nube (**Qdrant Cloud**, **Neon PostgreSQL**, **Langfuse Cloud**), la contenerización se centra exclusivamente en empaquetar la aplicación FastAPI de forma portable y segura.

- **`Dockerfile`**: Es el manifiesto principal que utiliza **Render.com** (o cualquier PaaS/Docker Host) para construir e iniciar el servicio en producción (escuchando en `$PORT` o `10000`).
- **`docker-compose.yml`**: Herramienta utilitaria para levantar el contenedor de la aplicación localmente reutilizando las credenciales de tu archivo `.env` sin distorsionar URLs ni levantar contenedores redundantes.

### Construcción y Ejecución con Docker
```bash
# Construir la imagen optimizada
docker build -t cv-agent:latest .

# Ejecutar el contenedor conectándolo a las credenciales del .env
docker run -d -p 10000:10000 --env-file .env --name cv-agent-container cv-agent:latest
```

### Ejecución Local con Docker Compose
```bash
docker-compose up --build
```

---

## 🚀 8. Despliegue en Render.com

1. Conecta este repositorio de GitHub a tu servicio Web en **Render.com**.
2. Render detectará automáticamente el `Dockerfile`.
3. Configura las siguientes variables de entorno en el panel de Render (**Environment**):
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_HOST` (`https://us.cloud.langfuse.com` o la URL de tu instancia)
   - `GROQ_API_KEY`
   - `HF_TOKEN`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `DATABASE_URL`
   - `METRICS_TOKEN`
4. El despliegue levantará automáticamente en HTTPS exponiendo la documentación Swagger en `/docs`.
