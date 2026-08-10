# 🤖 Agente Conversacional de CV · Reto IA Banorte (Generative AI Engineer)

Un **Agente Conversacional RAG Estricto** diseñado para responder preguntas sobre la experiencia profesional, habilidades y proyectos de un candidato, garantizando fundamentación (grounding), cero alucinaciones y estricto control de límites de información. 

El agente expone sus capacidades a través de una API construida en **FastAPI**, compatible con la especificación abierta **Open Responses API** (`/v1/responses`), e indexa conocimiento en tiempo real utilizando **Qdrant Cloud** y la inferencia ultra-rápida de **Groq** (Llama 3.3 70B) con fallback automático hacia **Hugging Face Inference Providers**.

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
                    │ /cv/upload                        │
                    │  ┌─────────────────────────────┐  │
                    │  │   Orquestador LangGraph     │  │
                    │  │                             │  │
                    │  │  1. Guardrails de entrada   │  │
                    │  │  2. Clasificador intenciones│  │
                    │  │  3. Tool: Vector Retrieval  │──┼──► Qdrant Cloud (cv_chunks)
                    │  │  4. Grounded LLM Generation │──┼──► Groq API (Llama 3.3 70B)
                    │  │  5. Guardrails de salida    │  │    (Fallback: HF Inference)
                    │  │  6. Memoria conversacional  │  │
                    │  └─────────────────────────────┘  │
                    │                                   │
                    │  Logging Estructurado (JSON)       │
                    └───────────────────┬───────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │ Observabilidad / Logs     │
                         │ (stdout / JSON Logs)      │
                         └───────────────────────────┘
```

### 🧩 Componentes Principales
| Componente | Tecnología Seleccionada | Razón Arquitectónica |
|---|---|---|
| **API Backend** | FastAPI + Pydantic v2 | Alto rendimiento asíncrono, OpenAPI auto-generado y tipado estricto. |
| **Estándar Interoperable** | Open Responses (`/v1/responses`) | Permite a cualquier SDK de agentes o cliente OpenAI usar este agente como un proveedor de modelos estandarizado. |
| **Base de Datos Vectorial** | Qdrant Cloud (`cv_chunks`) | Búsqueda por similitud de cosenos, filtros por payload y recreación atómica de versiones. |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo liviano (384 dim) multilenguaje ejecutado localmente en CPU en <30ms. |
| **LLM Principal** | Groq (`llama-3.3-70b-versatile`) | Inferencia de ultra-baja latencia (LPU) con ventana de contexto amplia. |
| **LLM Fallback** | Hugging Face Inference (`meta-llama/Llama-3.1-8B-Instruct`) | Resiliencia garantizada si Groq agota su cuota de rate limit. |
| **Orquestación & Grafo** | LangGraph + LangChain Core | Grafo de estados explícito para controlar el flujo, guardrails y recuperación. |

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

# En Linux/macOS:
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

Edita el archivo `.env` configurando tus llaves y credenciales:
```env
GROQ_API_KEY=gsk_tu_groq_key_aqui
HF_TOKEN=hf_tu_huggingface_token_aqui
QDRANT_URL=https://tu-cluster.cloud.qdrant.io
QDRANT_API_KEY=tu_qdrant_api_key_aqui
```

> **Nota de Respaldo:** Si dejas `QDRANT_URL` vacío, el sistema iniciará en modo **Qdrant en Memoria (`:memory:`)**, lo cual es ideal para pruebas locales rápidas sin conexión a la nube.

### Paso 5: Iniciar el Servidor API
```bash
uvicorn app.main:app --reload --port 7860
```

El servidor estará disponible en:
- **Swagger Documentation:** `http://localhost:7860/docs`
- **Healthcheck:** `http://localhost:7860/health`

---

## 🧪 3. Pruebas y Evaluación LLM-as-a-Judge

### Ejecutar Suite de Pruebas Unitarias
Verifica guardrails de seguridad (Prompt Injection), extracción de texto y validaciones de Pydantic:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Ejecutar Benchmark "LLM-as-a-Judge"
Mide métricas cuantitativas de **Fidelidad** y **Relevancia** (1.0 a 5.0) evaluando el agente contra el conjunto de preguntas de prueba (`tests/eval_dataset.json`):
```bash
python tests/eval_llm_judge.py
```

---

## 🐳 4. Contenerización con Docker y Docker Compose

### Ejecución Única con Docker
```bash
# Construir la imagen
docker build -t cv-agent:latest .

# Ejecutar el contenedor
docker run -d -p 7860:7860 --env-file .env --name cv-agent-container cv-agent:latest
```

### Ejecución Local Completa con Docker Compose (API + Qdrant Local)
```bash
docker-compose up --build
```

---

## 🚀 5. Despliegue en Plataformas Cloud Gratuitas

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
4. Agrega los Secretos en **Settings -> Variables and secrets**: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `HF_TOKEN`.
5. Haz git push de tu código al repositorio del Space.

### B) Despliegue en Render Free Service
1. Crea un nuevo **Web Service** en Render conectando tu repositorio de GitHub.
2. Render detectará automáticamente el `Dockerfile`.
3. Agrega las Variables de Entorno en el panel de Render.
4. Render expondrá la API automáticamente.

---

## 🔮 6. Roadmap de Posibles Mejoras Futuras

Si se dispone de más tiempo o para escalar el proyecto a producción enterprise:

1. **Re-Ranking de Vectores (Cohere / BGE Re-Ranker):**
   - Incorporar una etapa de re-ordenamiento semántico posterior a la búsqueda inicial de Qdrant para priorizar chunks de alta especificidad.
2. **Evaluación Continua con Langfuse Cloud:**
   - Conectar las trazas de ejecuciones de LangGraph directamente a Langfuse para medir costo de tokens y latencias p95 en tiempo real.
3. **Streaming Semántico Real en `/v1/responses`:**
   - Habilitar soporte paraServer-Sent Events (`stream=True`) siguiendo al 100% el estándar Open Responses para respuestas fluídas en UIs web.
4. **Soporte OCR para PDFs Escaneados:**
   - Integrar `pytesseract` o Amazon Textract en `app/rag/ingest.py` para procesar CVs escaneados o en formato imagen.
