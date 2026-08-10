"""
Pruebas de la capa de recuperación vectorial y score thresholding.
"""

from app.rag.retriever import retrieve_cv_context

def test_retrieval_returns_list_structure():
    # Consulta general de prueba
    results = retrieve_cv_context("experiencia en Python", top_k=2)
    assert isinstance(results, list)

def test_retrieval_irrelevant_query_threshold():
    # Una pregunta completamente ajena no debe superar el score_threshold = 0.35
    results = retrieve_cv_context("receta de comida para gatos o astronomía galáctica avanzada")
    assert len(results) == 0
