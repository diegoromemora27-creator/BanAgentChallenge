# 🤖 Agente Conversacional de CV · Reto IA Banorte (Generative AI Engineer)

Un **Agente Conversacional RAG Estricto** diseñado para responder preguntas sobre la experiencia profesional, habilidades y proyectos de un candidato, garantizando fundamentación (grounding), cero alucinaciones y estricto control de límites de información. 

El agente expone sus capacidades a través de una API construida en **FastAPI**, compatible con la especificación abierta **Open Responses API** (`/v1/responses`), e indexa conocimiento en tiempo real utilizando **Qdrant Cloud**, persistencia de estados con **Supabase PostgreSQL** (`PostgresSaver`), observabilidad y tracing con **Langfuse Cloud (SDK v3)** y monitoreo de métricas en tiempo real con **Prometheus & Grafana Cloud**.

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
                    │  │  6. State Checkpointer      │──┼──► Supabase PostgreSQL
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
| **Seguridad & Autenticación** | Header `Authorization` / `X-API-Key` | Control de acceso basado en API Key configurable (`API_KEY`) para proteger endpoints públicos. |
| **Protección & Cuota** | `slowapi` Rate Limiter (60 req/min) | Previene ataques de denegación de servicio y consumo acelerado de cuotas de LLMs. |
| **Estándar Interoperable** | Open Responses (`/v1/responses`) | Compatible con el SDK oficial de OpenAI (`openai` Python/TS) y catálogos de agentes. |
| **Base de Datos Vectorial** | Qdrant Cloud (`cv_chunks`) | Búsqueda por similitud de cosenos con metadatos y fallback a `:memory:`. |
| **Embeddings por API** | HF Router API (`bge-small-en-v1.5`) | Generación por API sin carga local en RAM (ahorra ~800MB RAM, app ~120MB). |
| **LLM Principal & Fallback** | HF Inference Router (`Llama-3.1-8B`) + Groq (`Llama-3.3-70B`) | Conmutación automática de alta disponibilidad en caso de límites de cuota. |
| **Orquestación & Grafo** | LangGraph + LangChain Core | Grafo de estados explícito con 4 nodos secuenciales y guardrails estrictos. |
| **Persistencia de Estados** | Supabase PostgreSQL (`PostgresSaver`) | Almacenamiento persistente de conversaciones por `session_id` con fallback a `MemorySaver`. |
| **Tracing & Observabilidad** | Langfuse Cloud SDK v3 (`from langfuse.langchain import CallbackHandler`) | Trazabilidad detallada de spans, ejecuciones de grafo y latencias. |
| **Métricas & Dashboards** | Prometheus Client + Grafana Cloud | Scraping del endpoint `/metrics` protegido por token `METRICS_TOKEN`. |

---

## 🗄️ 2. Arquitectura de Datos y Persistencia

### A. Qdrant Cloud (Vector DB para RAG)
- **Colección:** `cv_chunks`
- **Dimensión:** 384 vectores (compatible con `bge-small-en-v1.5` / `paraphrase-multilingual-MiniLM-L12-v2`).
- **Métrica de Distancia:** Coseno (`Distance.COSINE`).
- **Indexación y Ingesta:** Endpoint `/cv/upload` permite subir archivos PDF, TXT o texto plano. El sistema segmenta el contenido en chunks de 500 caracteres (solapamiento de 50) y almacena el vector junto con el payload (`texto`, `tipo`, `cv_version`, `timestamp`).
- **Fallback Automático:** Si `QDRANT_URL` o `QDRANT_API_KEY` no están configurados, el sistema inicia automáticamente un servidor **Qdrant en Memoria (`:memory:`)**, permitiendo ejecutar el agente sin dependencias externas.

### B. Supabase PostgreSQL (Checkpointer de LangGraph)
- **Módulo:** `langgraph-checkpoint-postgres` (`PostgresSaver`).
- **Conexión:** Utiliza `psycopg_pool.ConnectionPool` con opción `autocommit=True` para compatibilidad nativa con índices concurrentes y tablas de checkpoint en Supabase Session Pooler.
- **Persistencia de Conversación:** Almacena checkpoints por cada `thread_id` (correspondiente al `session_id` del usuario), manteniendo la memoria de diálogo entre turnos.
- **Fallback Automático:** Si no se especifica `DATABASE_URL`, la aplicación recurre a `MemorySaver` en RAM sin detener la ejecución.

---

## 🛡️ 3. Demostración de Grounding Estricto y Cero Alucinaciones

El sistema aplica una política de **Grounding Estricto** garantizada por un pipeline de tres etapas:

1. **Guardrail de Entrada (Input Moderation):** Detecta ataques de Prompt Injection e instrucciones maliciosas antes de tocar el grafo.
2. **Clasificación de Intención (Intent Classification):** Etiqueta la consulta como `CV_QUESTION`, `GREETING_OR_META` o `OUT_OF_BOUNDS`. Si la consulta está fuera del dominio (ej. política, recetas de cocina, física cuántica), el nodo se corta inmediatamente.
3. **Guardrail de Salida (Grounding Verification):** Verifica que la respuesta generada por el LLM esté respaldada por los trozos de texto recuperados de Qdrant.

### 🧪 Ejemplos de Comportamiento del Agente:

| Caso de Prueba | Entrada del Usuario | Respuesta del Agente | Explicación del Comportamiento |
|---|---|---|---|
| **Duda en el CV (In-Bounds)** | *¿Qué proyectos ha desarrollado en Inteligencia Artificial?* | *"El candidato lideró el desarrollo del 'Sistema de Ingesta Inteligente de Documentos', un pipeline automatizado con extracción LLM y almacenamiento en Qdrant..."* | **Respuesta fundamentada:** Extrae contexto real indexado en Qdrant y fundamenta la respuesta. |
| **Consulta Fuera de Dominio (Out-of-Bounds)** | *¿Cuál es la receta para preparar tacos al pastor?* | *"Como agente conversacional enfocado en el perfil profesional del candidato, solo estoy capacitado para responder preguntas sobre su experiencia laboral, proyectos, habilidades e historia profesional."* | **Filtro de Intención:** Corta la ejecución sin consultar a Qdrant ni gastar tokens inútilmente. |
| **Duda no evidenciada en el CV** | *¿Cuántos años de experiencia tiene el candidato administrando clústeres de Kubernetes?* | *"No dispongo de información suficiente en el CV para responder con exactitud a tu pregunta."* | **Guardrail de Salida (Cero Alucinación):** Si Qdrant no devuelve evidencia suficiente, el sistema bloquea cualquier intento de invención del LLM. |

---

## 🛰️ 4. Trazabilidad y Observabilidad Avanzada

### A. Langfuse Cloud (SDK v3 Integration)
El sistema utiliza el SDK v3 de Langfuse a través de la importación oficial moderna:
```python
from langfuse.langchain import CallbackHandler
```

#### Paso a Paso para Configurar Langfuse Cloud:
1. **Crear Cuenta y Proyecto:** Registra una cuenta en [Langfuse Cloud](https://cloud.langfuse.com/) y crea un nuevo proyecto.
2. **Generar API Keys:** Ve a **Settings -> API Keys** y genera un nuevo par de llaves (`Public Key` y `Secret Key`).
3. **Configurar Variables en Render.com / `.env`:**
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
   LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
   LANGFUSE_HOST=https://us.cloud.langfuse.com
   ```
4. **Visualización de Trazas:**
   Cada petición enviada al agente agrupa automáticamente sus spans (Guardrails, Intent Classification, Qdrant Retrieval, LLM Generation) bajo el identificador de sesión `langfuse_session_id`, permitiendo analizar latencia p95, costo por llamada y calidad de las respuestas en el dashboard de Langfuse.

---

### B. Grafana Cloud & Prometheus Metrics

La API expone un endpoint `/metrics` en formato Prometheus estándar para scraping continuo.

#### Paso a Paso para Integrar con Grafana Cloud:
1. **Crear Stack en Grafana Cloud:** Crea una cuenta en [Grafana Cloud](https://grafana.com/) y accede a tu instancia de Prometheus.
2. **Configurar el Scraper (Prometheus Scrape Job):**
   Añade la siguiente configuración en tu Prometheus Server o agente de scraping de Grafana Agent / Alloy:
   ```yaml
   scrape_configs:
     - job_name: 'banorte_cv_agent'
       scrape_interval: 15s
       metrics_path: '/metrics'
       params:
         token: ['your_secret_metrics_token_for_grafana_here'] # Configurado en METRICS_TOKEN
       scheme: https
       static_configs:
         - targets: ['tu-app-en-render.onrender.com']
   ```
3. **Importar el Dashboard Pre-construido (`grafana_dashboard_banorte.json`):**
   - En Grafana Cloud, ve al menú **Dashboards -> New -> Import**.
   - Sube o pega el contenido del archivo `grafana_dashboard_banorte.json` ubicado en la raíz del repositorio.
   - Selecciona la fuente de datos Prometheus configurada y haz clic en **Import**.
4. **Métricas Clave Disponibles en el Dashboard:**
   - **`AGENT_REQUESTS_TOTAL`**: Total de solicitudes procesadas agrupadas por status (`success`, `fallback`).
   - **`AGENT_LATENCY_SECONDS`**: Histograma de latencia completa del flujo agéntico.
   - **`NODE_EXECUTION_DURATION_SECONDS`**: Latencia desglosada por cada uno de los 4 nodos de LangGraph (`guardrails_input`, `classify_intent`, `retrieve_context`, `generate_response`).
   - **`NODE_ERRORS_TOTAL`**: Conteo de excepciones por nodo.
   - **`RAG_RETRIEVED_DOCUMENTS_COUNT`**: Cantidad de trozos de texto recuperados de Qdrant por consulta.
   - **`RAG_RELIABILITY_SCORE`**: Puntaje cuantitativo de confiabilidad y grounding.
   - **`LLM_TOKENS_TOTAL`**: Total de tokens (`prompt` y `completion`) por proveedor (`Hugging Face`, `Groq`).
   - **`LLM_COST_ESTIMATED_TOTAL`**: Costo monetario estimado acumulado en USD.

---

## 🛠️ 5. Guía de Instalación y Configuración Local

### Requisitos Previos
- Python 3.10 a 3.14 (Dependencias ancladas y verificadas).
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

### Paso 3: Instalar Dependencias Ancladas
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
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_hf_token_here

# Vector Database (Qdrant Cloud)
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Persistencia de Estado (Neon / Supabase Postgres)
DATABASE_URL=postgresql://neondb_owner:your_password@ep-xxx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require

# Observabilidad & Tracing (Langfuse Cloud v3)
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Seguridad & Control de Acceso
API_KEY=your_optional_api_key_for_endpoints_here
METRICS_TOKEN=your_secret_metrics_token_for_grafana_here
```

### Paso 5: Iniciar el Servidor API
```bash
uvicorn app.main:app --reload --port 7860
```

Accede a la documentación interactiva OpenAPI (Swagger) en:
`http://localhost:7860/docs`

---

## 🛰️ 6. Especificación de Endpoints HTTP e Interoperabilidad (Open Responses)

### 📌 Resumen de Endpoints

| Método | Endpoint | Seguridad / Auth | Descripción |
|---|---|---|---|
| `GET` | `/health` | Pública | Verificación de estado del servicio y salud de bases de datos. |
| `POST` | `/chat` | Bearer Auth (`API_KEY`) | Endpoint de conversación simplificado con Rate Limiting (`message`, `session_id`). |
| `POST` | `/v1/responses` | Bearer Auth (`API_KEY`) | Endpoint interoperable compatible con el estándar **Open Responses API**. |
| `POST` | `/cv/upload` | Bearer Auth (`API_KEY`) | Ingesta multi-fuente (PDF, TXT, texto pegado) e indexación en Qdrant. |
| `GET` | `/cv/info` | Pública | Inspección de chunks y metadatos actualmente indexados en Qdrant. |
| `GET` | `/metrics` | Query Token / Bearer | Endpoint de métricas en formato Prometheus para Grafana Cloud. |

---

### 🌐 A. Estándar Open Responses API (`POST /v1/responses`)

El endpoint `/v1/responses` sigue la especificación abierta **Open Responses API**, permitiendo que cualquier agente externo, cliente HTTP o SDK oficial de OpenAI consuma las respuestas como un modelo LLM estándar.

#### 1. Ejemplo con cURL (Con Autenticación por API Key)
```bash
curl -X POST "https://tu-app-en-render.onrender.com/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu_api_key_aqui" \
  -d '{
    "model": "cv-agent-v1",
    "previous_response_id": "sesion_agente_100",
    "input": [
      {
        "role": "user",
        "content": "¿Qué proyectos relevantes de IA ha liderado el candidato?"
      }
    ]
  }'
```

#### Respuesta JSON del Estándar Open Responses:
```json
{
  "id": "resp_8f91b72a0e414c5c8a91bf5409",
  "object": "response",
  "created_at": 1786423100,
  "status": "completed",
  "model": "cv-agent-v1",
  "output": [
    {
      "type": "message",
      "id": "msg_c9a1b02319df40",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "El candidato lideró el proyecto 'Sistema de Ingesta Inteligente de Documentos', un pipeline automatizado con extracción LLM y Qdrant Cloud que procesó más de 10,000 documentos con 98% de precisión...",
          "annotations": []
        }
      ]
    }
  ],
  "output_text": "El candidato lideró el proyecto...",
  "usage": {
    "prompt_tokens": 615,
    "completion_tokens": 320,
    "total_tokens": 935
  }
}
```

#### 2. Consumo mediante el SDK Oficial de Python (`openai`)
```python
from openai import OpenAI

# Redirige el cliente oficial de OpenAI hacia nuestro agente conversacional
client = OpenAI(
    base_url="https://tu-app-en-render.onrender.com/v1",
    api_key="tu_api_key_aqui" # Pasa la API_KEY configurada en Render.com (ej. banorte_challenge_api_key_2026)
)

response = client.chat.completions.create(
    model="cv-agent-v1",
    messages=[
        {"role": "user", "content": "¿Qué tecnologías domina el candidato en FastAPI y Vector DBs?"}
    ]
)

print("Respuesta del Agente:", response.choices[0].message.content)
```

---

## ⚡ 7. Gestión de Cold Starts y Rendimiento en Render.com (Free Tier)

### ❄️ ¿Por qué ocurre el Cold Start en Render Free Tier?
El plan gratuito de Render desactiva la instancia del contenedor tras 15 minutos de inactividad. Al recibir una nueva petición:
1. Render inicia el contenedor desde cero (~15-30s).
2. FastAPI carga las dependencias e inicializa los pools de conexión a PostgreSQL y Qdrant.

### 🛡️ Estrategias de Mitigación Implementadas:
1. **Pre-Warming en `GET /health`:** El endpoint `/health` ejecuta pings livianos en segundo plano para 'despertar' las conexiones a las bases de datos vectoriales antes de procesar una petición pesada de chat.
2. **Ping Periódico (Keep-Alive Recomendado):** Para pruebas o evaluaciones continuas del jurado, se recomienda utilizar un servicio de Keep-Alive gratuito (ej. [cron-job.org](https://cron-job.org) o [UptimeRobot](https://uptimerobot.com)) haciendo un ping `GET` cada 10 minutos a `https://tu-app-en-render.onrender.com/health`.

---

## 🐳 8. Contenerización con Docker y Docker Compose

### ¿Cómo funciona Docker en esta Arquitectura?
Dado que las bases de datos y herramientas de observabilidad operan en la nube (**Qdrant Cloud**, **Supabase PostgreSQL**, **Langfuse Cloud**), Docker aísla y empaqueta exclusivamente el servicio web FastAPI.

- **`Dockerfile`**: Es el manifiesto que lee **Render.com** (o cualquier servidor Docker) para empaquetar e iniciar la API escuchando en `$PORT` o `10000`.
- **`docker-compose.yml`**: Herramienta de pruebas locales para ejecutar el contenedor de la API cargando las variables de tu archivo `.env`.

### Ejecución Local con Docker Compose
```bash
docker-compose up --build
```

---

## 🧠 9. Justificación de Decisiones Arquitectónicas & Trade-offs (Q&A de la Demo)

Para la evaluación del **Reto IA Banorte**, a continuación se detallan las decisiones clave de arquitectura e ingeniería:

### 1. ¿Por qué LangGraph en lugar de una cadena simple (`LLMChain`)?
- **Control Finito de Estados:** `LLMChain` o pipelines lineales de LangChain son 'cajas negras' difíciles de pausar, bifurcar o auditar.
- **Grafo de Estados Explícito:** LangGraph nos permite implementar un ciclo determinista de 4 nodos secuenciales (`guardrails_input` ➔ `classify_intent` ➔ `retrieve_context` ➔ `generate_response`). Si un guardrail falla, el grafo corta la ejecución inmediatamente sin pasar al LLM ni consumir tokens.

### 2. ¿Por qué Qdrant Cloud y no `pgvector` en PostgreSQL (Supabase)?
- **Aislamiento de Cargas y Escalabilidad:** Aunque ya utilizamos PostgreSQL (Supabase) para el checkpointer de LangGraph, delegar la búsqueda vectorial a Qdrant Cloud evita sobrecargar el pool de conexiones de la base de datos relacional durante consultas masivas de embeddings.
- **Búsqueda Filtrada de Alto Rendimiento:** Qdrant está escrito en Rust y ofrece índices HNSW nativos con filtrado por metadatos (`tipo`, `cv_version`) a nivel de sub-milisegundo, además de fallback transparente a memoria (`:memory:`) para pruebas unitarias sin dependencias externas.

### 3. ¿Qué Trade-offs se aceptaron al elegir Render Free Tier frente a infraestructura dedicada?
- **Trade-off:** La latencia inicial de cold start (15-30s tras inactividad) frente a la ventaja de despliegue en la nube a costo cero y alta seguridad en contenedores Docker HTTPS.
- **Mitigación:** Se implementó pre-warming automático en el endpoint de salud `/health` y fallback de LLM entre Hugging Face Router API y Groq LPU para mantener latencias por debajo de 1.2s una vez caliente el servicio.

---

## 🔗 10. Enlaces y Despliegues del Proyecto

A continuación se presentan los accesos a los servicios y dashboards del proyecto:

- 🚀 **Servicio API en Producción (Render):** `https://tu-app-en-render.onrender.com`
- 📚 **Documentación Swagger / OpenAPI:** `https://tu-app-en-render.onrender.com/docs`
- 🩺 **Endpoint de Healthcheck:** `https://tu-app-en-render.onrender.com/health`
- 📊 **Endpoint de Métricas Prometheus:** `https://tu-app-en-render.onrender.com/metrics?token=your_secret_metrics_token_for_grafana_here`
- 📈 **Dashboard Grafana Cloud:** `https://tu-org.grafana.net/d/tu-dashboard-id`
- 🔍 **Trazabilidad Langfuse Cloud:** `https://cloud.langfuse.com/project/tu-proyecto-id`
- 🗄️ **Cluster Qdrant Cloud:** `https://tu-cluster.cloud.qdrant.io`
- ⚡ **Base de Datos Supabase PostgreSQL:** `https://supabase.com/dashboard/project/wturifeqladmfhdjysse`
