FROM python:3.11-slim

# Evita prompts interactivos y bytecode innecesario
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias (se cachea si requirements.txt no cambia)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación y datos
COPY app/ ./app
COPY data/ ./data

# Usuario no root por mejores prácticas de seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Healthcheck interno del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Comando de inicio compatible con Render (expande variable $PORT) y Hugging Face (puerto 7860)
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"
