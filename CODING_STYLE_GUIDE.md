# Guía de Estilo de Código y Arquitectura
## Agente Conversacional de CV · Reto IA Banorte

### 1. Principios General de Diseño
1. **Tipado Estricto (Type Annotations)**:
   - Todo código en Python hace uso de type hints explícitos (`str`, `int`, `list[dict]`, `UploadFile | None`).
   - Todos los modelos de datos de entrada/salida y estado usan **Pydantic v2** (`BaseModel`, `Field`).

2. **Manejo de Excepciones y Logging Estructurado**:
   - Cada módulo define su propio `logger = logging.getLogger(__name__)`.
   - Las excepciones se capturan específicamente y se propagan como `HTTPException` con códigos HTTP adecuados (400, 401, 500) en el nivel de FastAPI.

3. **Documentación de Código (Docstrings)**:
   - Cada clase y función pública incluye docstrings formateados estilo Google/NumPy explicando propósito, argumentos y retorno.

4. **Patrón de Fallback y Resiliencia**:
   - Para servicios externos (Groq LLM), se implementa un mecanismo de degradación elegante (Fallback) hacia Hugging Face Inference Providers.

5. **Principios RAG Anti-Alucinación**:
   - Filtrado de vectores por umbral de similitud (`score_threshold = 0.35`).
   - Inyección explícita del contexto recuperado en el System Prompt.
   - Chunk de metadatos de "Límites de información" en la ingesta.

### 2. Estructura de Proyectos y Módulos
```
app/
├── main.py              # Punto de entrada de FastAPI y rutas HTTP
├── config.py            # Gestión centralizada de variables de entorno
├── models/
│   └── schemas.py       # Definición de esquemas Pydantic
├── rag/
│   ├── ingest.py        # Ingesta multi-fuente, extracción e indexación
│   └── retriever.py     # Embeddings y consulta a Qdrant
├── llm/
│   ├── groq_client.py   # Cliente Groq (Open Responses)
│   ├── hf_client.py     # Cliente Fallback Hugging Face
│   └── provider.py      # Gestor de resiliencia LLM
└── agent/
    ├── prompts.py       # Prompts del sistema y de estructuración
    ├── guardrails.py    # Guardrails de seguridad y moderación
    ├── memory.py        # Memoria conversacional de sesión
    └── graph.py         # Grafo de estados con LangGraph
```
