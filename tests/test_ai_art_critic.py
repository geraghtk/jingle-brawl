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


EM = "\u2014"


class NoEmDashes(unittest.TestCase):
    """Players never see an em dash: the prompts ask for none, and replies are cleaned anyway."""

    def test_no_dashes_reads_naturally(self):
        self.assertEqual(critic.no_dashes(f"Bold lines {EM} a true penguin."), "Bold lines, a true penguin.")
        self.assertEqual(critic.no_dashes(f"Bold{EM}brave."), "Bold, brave.")
        self.assertEqual(critic.no_dashes(f"A cat {EM}."), "A cat.")
        self.assertEqual(critic.no_dashes(f"{EM} Alice wins."), "Alice wins.")
        self.assertEqual(critic.no_dashes("Bold \u2013 brave, pages 1\u20135."), "Bold, brave, pages 1\u20135.")
        self.assertIsNone(critic.no_dashes(None))

    def test_critique_and_verdict_come_back_without_em_dashes(self):
        client = fake_ollama(f"A penguin {EM} wobbly.", f"SCORE: 80 | CRITIQUE: Charming {EM} truly.")
        with mock.patch.object(critic, "get_ollama_client", return_value=client), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            data = critic.app.test_client().post("/critique", json={"image": "aGk=", "secretWord": "Penguin"}).get_json()
        self.assertEqual(data["critique"], "Charming, truly.")
        self.assertEqual(data["description"], "A penguin, wobbly.")
        for prompt in (c.kwargs["messages"][0]["content"] for c in client.chat.call_args_list):
            self.assertIn("Do not use em dashes", prompt)
            self.assertNotIn(EM, prompt)

        client = fake_ollama(f"What a night {EM} of art.\nWINNER: Alice")
        with mock.patch.object(critic, "get_ollama_client", return_value=client), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            data = critic.app.test_client().post("/witty-verdict", json={"prompt": "Cat", "scores": WinnerParsing.SCORES}).get_json()
        self.assertEqual(data["wittyVerdict"], "What a night, of art.")
        prompt = client.chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Do not use em dashes", prompt)
        self.assertNotIn(EM, prompt)


if __name__ == "__main__":
    unittest.main()
