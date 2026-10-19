"""Unit tests for ai_art_critic.py response parsing (no Ollama/OpenAI needed).

Run:  python -m unittest discover -s tests   (needs: pip install -r ai_art_critic_requirements.txt)
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ai_art_critic as critic  # noqa: E402


def fake_ollama(*replies):
    """An Ollama client whose chat() returns the given texts in order."""
    it = iter(replies)
    client = mock.Mock()
    client.chat.side_effect = lambda **kw: SimpleNamespace(message=SimpleNamespace(content=next(it)))
    return client


class CritiqueScoreParsing(unittest.TestCase):
    def post_critique(self, rate_reply):
        client = fake_ollama("A wobbly penguin.", rate_reply)
        with mock.patch.object(critic, "get_ollama_client", return_value=client), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            res = critic.app.test_client().post("/critique", json={"image": "aGk=", "secretWord": "Penguin"})
        self.assertEqual(res.status_code, 200)
        return res.get_json()

    def test_score_out_of_100_is_not_concatenated(self):
        # Used to join every digit: "85/100" -> 851 -> clamped to 100
        self.assertEqual(self.post_critique("SCORE: 85/100 | CRITIQUE: Waddles with intent.")["rating"], 85)

    def test_plain_score_and_critique(self):
        data = self.post_critique("SCORE: 42 | CRITIQUE: Bold. Wrong, but bold.")
        self.assertEqual(data["rating"], 42)
        self.assertEqual(data["critique"], "Bold. Wrong, but bold.")

    def test_missing_score_defaults_to_50(self):
        self.assertEqual(self.post_critique("I cannot rate this masterpiece.")["rating"], 50)


class WinnerParsing(unittest.TestCase):
    SCORES = {"p1": {"name": "Alice", "rating": 40}, "p2": {"name": "Bob", "rating": 90}}

    def test_exact_name(self):
        self.assertEqual(critic.parse_winner("Great show.\nWINNER: Alice", self.SCORES), "p1")

    def test_case_and_brackets(self):
        self.assertEqual(critic.parse_winner("winner: [bob]", self.SCORES), "p2")

    def test_empty_winner_line_matches_nobody(self):
        # `"" in "ALICE"` is True — an empty line used to crown the first player
        self.assertIsNone(critic.parse_winner("Tough call.\nWINNER:", self.SCORES))

    def test_verdict_endpoint_strips_lowercase_winner_line(self):
        client = fake_ollama("What a night of art.\nWinner: Alice")
        with mock.patch.object(critic, "get_ollama_client", return_value=client), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            res = critic.app.test_client().post("/witty-verdict", json={"prompt": "Cat", "scores": self.SCORES})
        data = res.get_json()
        self.assertEqual(data["winnerId"], "p1")
        self.assertEqual(data["wittyVerdict"], "What a night of art.")


if __name__ == "__main__":
    unittest.main()
