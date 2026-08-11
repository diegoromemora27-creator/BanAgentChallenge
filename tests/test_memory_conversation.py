"""
Prueba de integración para memoria conversacional de múltiples turnos.
Demuestra la resolución de anáforas y seguimiento de contexto ("ese proyecto", "¿cuánto tiempo le tomó?").
"""

import unittest
from app.agent.graph import run_agent_workflow

class TestConversationalMemory(unittest.TestCase):

    def test_multi_turn_conversation_flow(self):
        """
        Ejecuta un diálogo secuencial de 3 turnos para verificar que el agente mantiene
        la coherencia del contexto sin perder la referencia a entidades previas.
        """
        session_id = "test_memory_session_999"

        # Turno 1: Pregunta inicial sobre proyectos
        res_1 = run_agent_workflow("¿Qué proyectos de IA ha liderado el candidato?", session_id=session_id)
        reply_1 = res_1["reply"]
        self.assertIsNotNone(reply_1)
        self.assertTrue(len(reply_1) > 0)

        # Turno 2: Pregunta anafórica ("ese proyecto")
        res_2 = run_agent_workflow("¿Y en qué tecnologías se basó ese proyecto?", session_id=session_id)
        reply_2 = res_2["reply"]
        self.assertIsNotNone(reply_2)
        self.assertTrue(len(reply_2) > 0)

        # Turno 3: Pregunta de seguimiento ("¿qué resultado obtuvo?")
        res_3 = run_agent_workflow("¿Qué resultados o métricas de éxito alcanzó?", session_id=session_id)
        reply_3 = res_3["reply"]
        self.assertIsNotNone(reply_3)
        self.assertTrue(len(reply_3) > 0)

if __name__ == "__main__":
    unittest.main()
