# Agente Conversacional de CV — Reto IA Banorte

Un agente conversacional RAG estricto diseñado para responder preguntas sobre la trayectoria profesional de un candidato sin inventar datos ni alucinar, expuesto mediante una API FastAPI interoperable con el estándar abierto **Open Responses** (`/v1/responses`).

---

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **Framework API:** FastAPI / Pydantic v2
- **Vector DB:** Qdrant Cloud (`cv_chunks`)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim)
- **LLM Principal:** Groq API (`llama-3.3-70b-versatile`)
- **LLM Fallback:** Hugging Face Inference Providers (`meta-llama/Llama-3.1-8B-Instruct`)
- **Orquestación:** LangGraph (Grafo de estados con guardrails y memoria)

---

## 🚀 Inicio Rápido Local

### 1. Requisitos Previos
Asegúrate de configurar las variables de entorno copiando el archivo de plantilla:

```bash
cp .env.example .env
```

Edita `.env` agregando tus credenciales de **Groq**, **Hugging Face** y **Qdrant Cloud**:

```env
GROQ_API_KEY=tu_groq_api_key
HF_TOKEN=tu_hf_token
QDRANT_URL=https://tu-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=tu_qdrant_api_key
ADMIN_API_KEY=admin_secret_key_123
```

### 2. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecución del Servidor
```bash
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva disponible en: `http://localhost:8000/docs`

---

## 🛰️ Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Verificación de estado del servidor |
| `POST` | `/chat` | Interfaz simplificada para clientes web/chat UI |
| `POST` | `/v1/responses` | Endpoint compatible con el estándar **Open Responses** |
| `POST` | `/cv/upload` | Ingesta multi-fuente segura para subir o actualizar el CV (PDF/TXT/Pegado) |
