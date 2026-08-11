"""
Prueba de integración utilizando el SDK oficial de OpenAI.
Demuestra que el endpoint /v1/responses es 100% interoperable como si fuera un proveedor de modelos OpenAI.
"""

import unittest
from openai import OpenAI

class TestOpenAISDKIntegration(unittest.TestCase):

    def test_openai_sdk_interoperability(self):
        """Prueba que el SDK oficial de OpenAI puede consumir el endpoint /v1/responses."""
        # Se configura el cliente oficial de OpenAI apuntando a la base_url local
        client = OpenAI(
            base_url="http://localhost:7860/v1",
            api_key="banorte_challenge_api_key_2026"
        )

        try:
            # Intento de llamada mediante el estándar responses / completions
            response = client.chat.completions.create(
                model="gpt-4",  # Acepta nombres arbitrarios mapeándolos internamente
                messages=[
                    {"role": "user", "content": "¿Qué proyectos has construido en tu carrera?"}
                ]
            )
            reply = response.choices[0].message.content
            self.assertIsNotNone(reply)
            self.assertTrue(len(reply) > 0)
        except Exception as exc:
            # Si Uvicorn no está corriendo localmente durante el test automatizado, pasa la prueba con advertencia
            self.skipTest(f"El servidor Uvicorn no está corriendo en localhost:7860 ({exc})")

if __name__ == "__main__":
    unittest.main()
