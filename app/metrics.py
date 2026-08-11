"""
Módulo de Métricas e Instrumentación nativa para Prometheus / OpenMetrics.
Registra Histogramas, Contadores y Gauges de LangGraph, RAG y Costos Financieros.
"""

try:
    from prometheus_client import Counter, Histogram, Gauge
except ImportError:
    from contextlib import contextmanager
    
    @contextmanager
    def _dummy_cm():
        yield

    class DummyMetric:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def time(self, *args, **kwargs): return _dummy_cm()

    Counter = Histogram = Gauge = DummyMetric

# 1. Métricas de Rendimiento y Latencia (Histograms)
AGENT_LATENCY_SECONDS = Histogram(
    "agent_latency_seconds",
    "Tiempo total de respuesta del agente conversacional",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

NODE_EXECUTION_DURATION_SECONDS = Histogram(
    "node_execution_duration_seconds",
    "Duración de ejecución por nodo individual de LangGraph",
    ["node_name"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Tiempo tomado para buscar vectores en Qdrant Cloud",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

# 2. Métricas de Costo y Uso de Tokens (Counters)
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total acumulado de tokens procesados por el LLM",
    ["type", "provider"]
)

LLM_COST_ESTIMATED_TOTAL = Counter(
    "llm_cost_estimated_total",
    "Costo acumulado estimado en USD basado en el consumo de tokens",
    ["provider"]
)

# 3. Métricas de Fiabilidad y Flujo (Counters / Gauges)
AGENT_REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Número total de solicitudes procesadas por el agente",
    ["status"]
)

NODE_ERRORS_TOTAL = Counter(
    "node_errors_total",
    "Cantidad de fallos o excepciones agrupados por el nombre del nodo",
    ["node_name"]
)

TOOL_INVOCATIONS_TOTAL = Counter(
    "tool_invocations_total",
    "Veces que el agente decide llamar a una herramienta externa",
    ["tool_name"]
)

# 4. Métricas de Calidad RAG (Gauges / Counters)
RAG_RETRIEVED_DOCUMENTS_COUNT = Gauge(
    "rag_retrieved_documents_count",
    "Cantidad de fragmentos de evidencia recuperados de Qdrant por consulta"
)

RAG_RELIABILITY_SCORE = Gauge(
    "rag_reliability_score",
    "Puntaje de confiabilidad y grounding de la respuesta (0.0 a 1.0)"
)
