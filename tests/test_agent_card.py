"""
Prueba unitaria para verificar el endpoint /.well-known/agent-card.json y los endpoints de Open Responses.
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app

class TestAgentCardAndOpenResponses(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.api_key = "banorte_challenge_api_key_2026"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def test_get_agent_card_endpoint(self):
        """Verifica que GET /.well-known/agent-card.json devuelva 200 OK y la estructura de la tarjeta A2A."""
        response = self.client.get("/.well-known/agent-card.json")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("name", data)
        self.assertIn("description", data)
        self.assertIn("version", data)
        self.assertIn("supportedInterfaces", data)
        self.assertIn("interfaces", data)
        self.assertIn("starter_prompts", data)
        self.assertIn("authentication", data)
        self.assertIn("capabilities", data)
        
        # Verificar starter prompts
        self.assertGreaterEqual(len(data["starter_prompts"]), 1)
        self.assertLessEqual(len(data["starter_prompts"]), 8)
        
        # Verificar authentication
        self.assertEqual(data["authentication"]["type"], "bearer")

    def test_responses_endpoint_alias(self):
        """Verifica que POST /responses funcione idéntico a /v1/responses."""
        payload = {
            "model": "cv-agent-v1",
            "input": [
                {"role": "user", "content": "¿Cuál es tu experiencia profesional?"}
            ]
        }
        # Petición a /responses sin auth debe dar 401
        unauth_res = self.client.post("/responses", json=payload)
        self.assertEqual(unauth_res.status_code, 401)

        # Petición a /responses con auth válida
        res = self.client.post("/responses", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertIn("output_text", res_data)
        self.assertIn("output", res_data)

    def test_responses_structured_content(self):
        """Verifica que el endpoint /responses acepte arreglos de contenidos (texto + adjuntos)."""
        payload = {
            "model": "gpt-4",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "¿Cuáles son tus habilidades principales?"},
                        {"type": "input_file", "file_name": "resumen.pdf"}
                    ]
                }
            ]
        }
        res = self.client.post("/responses", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertIn("output_text", res_data)

    def test_responses_with_instructions(self):
        """Verifica que el endpoint /responses procese el campo opcional 'instructions'."""
        payload = {
            "model": "cv-agent-v1",
            "instructions": "Responde de forma ejecutiva y muy breve.",
            "input": [
                {"role": "user", "content": "¿Qué proyectos has construido?"}
            ]
        }
        res = self.client.post("/responses", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertIn("output_text", res_data)

if __name__ == "__main__":
    unittest.main()
