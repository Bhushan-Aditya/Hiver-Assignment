"""RagAgent behaviour with Ollama stubbed out.

These tests never touch the network. A live end-to-end smoke test that does
call Ollama lives in ``test_live_smoke`` and skips itself when Ollama or the
models are unavailable.
"""
import json
import unittest
import urllib.request

import numpy as np

from hiver_agent import rag
from hiver_agent.core import Pair


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


# Three tight trip_status precedents + one payment precedent.
PAIRS = [
    Pair("1", "my driver was late for pickup", "r1", "Check the driver location in the app; cancel and rebook if needed.", "trip_status"),
    Pair("2", "driver is late and not moving on the map", "r2", "Track the driver in the app; you can cancel and request another ride.", "trip_status"),
    Pair("3", "waiting forever for a late driver", "r3", "Please check the app for the driver ETA and rebook if necessary.", "trip_status"),
    Pair("4", "my card was charged but ride never happened", "r4", "We will review the payment.", "payment"),
]
VECTORS = np.vstack([
    _unit([1, 0, 0, 0]),
    _unit([0.98, 0.05, 0, 0]),
    _unit([0.97, 0.08, 0.02, 0]),
    _unit([0, 0, 1, 0]),
])


class StubMixin:
    def setUp(self):
        self._gen = {"intent": "trip_status", "action": "auto_handle", "confidence": 0.8,
                     "reply": "Track your driver in the app and rebook if needed.", "reason": "Close precedent."}
        self._ollama_calls = []

        def fake_embed(texts, model=None):
            # query "late driver" -> aligns with trip_status cluster
            return VECTORS[:1].copy() if len(texts) == 1 else VECTORS[: len(texts)].copy()

        def fake_ollama(path, payload, timeout=120, retries=3):
            self._ollama_calls.append(path)
            if path == "/api/generate":
                return {"response": json.dumps(self._gen)}
            raise AssertionError(path)

        self._orig_embed, self._orig_ollama = rag.embed, rag.ollama
        self._orig_settings = rag.settings
        rag.embed = fake_embed
        rag.ollama = fake_ollama
        self.agent = rag.RagAgent(list(PAIRS), VECTORS.copy(), model="test")

    def tearDown(self):
        rag.embed, rag.ollama, rag.settings = self._orig_embed, self._orig_ollama, self._orig_settings

    def _no_fastpath_agent(self):
        rag.settings = self._orig_settings.__class__(enable_fastpath=False)
        return rag.RagAgent(list(PAIRS), VECTORS.copy(), model="test")


class FastPathTest(StubMixin, unittest.TestCase):
    def test_confident_routine_uses_fast_path_and_skips_llm(self):
        res = self.agent.assess("driver is very late for pickup")
        self.assertEqual(res.path, "fast")
        self.assertEqual(res.decision.action, "auto_handle")
        self.assertEqual(res.timings_ms["generate"], 0.0)
        self.assertNotIn("/api/generate", self._ollama_calls)

    def test_stream_emits_ordered_stages(self):
        stages = [e["stage"] for e in self.agent.stream("driver is very late for pickup")]
        self.assertEqual(stages[0], "retrieving")
        self.assertIn("retrieved", stages)
        self.assertEqual(stages[-1], "done")

    def test_cache_hit_on_repeat(self):
        self.agent.assess("driver is very late for pickup")
        calls = len(self._ollama_calls)
        events = list(self.agent.stream("  Driver is VERY late for pickup  "))
        self.assertTrue(events[-1].get("cached"))
        self.assertEqual(len(self._ollama_calls), calls)  # no new work


class GuardrailTest(StubMixin, unittest.TestCase):
    def test_off_app_cash_forces_escalate_even_if_model_says_auto(self):
        res = self.agent.assess("my driver asked me for extra 200 rupees in cash")
        self.assertEqual(res.decision.action, "escalate")
        self.assertEqual(res.decision.reply, "")

    def test_risk_word_forces_escalate(self):
        res = self.agent.assess("the driver made a threat against me")
        self.assertEqual(res.decision.action, "escalate")

    def test_paraphrased_safety_signal_forces_escalate(self):
        # No classic risk word, but "following me" must still route to a human.
        for probe in (
            "I left my phone and the driver is following me now",
            "the driver seems drunk and I feel unsafe",
            "he won't let me out of the car",
        ):
            res = self.agent.assess(probe)
            self.assertEqual(res.decision.action, "escalate", probe)
            self.assertEqual(res.decision.reply, "", probe)

    def test_escalate_only_intent_overrides_model(self):
        self._gen = {**self._gen, "intent": "payment"}
        agent = self._no_fastpath_agent()  # force the LLM path
        res = agent.assess("something about my card charge that is ambiguous")
        self.assertEqual(res.decision.action, "escalate")

    def test_model_failure_falls_back_to_escalate(self):
        def boom(path, payload, timeout=120, retries=3):
            raise RuntimeError("model down")

        agent = self._no_fastpath_agent()
        rag.ollama = boom
        res = agent.assess("an unusual message with no clear precedent at all")
        self.assertEqual(res.path, "fallback")
        self.assertEqual(res.decision.action, "escalate")


class LiveSmokeTest(unittest.TestCase):
    """Runs only when a real Ollama with the configured models is reachable."""

    def setUp(self):
        try:
            req = urllib.request.Request(rag.settings.ollama_host + "/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                models = {m["name"] for m in json.load(r).get("models", [])}
        except Exception:
            self.skipTest("Ollama not reachable")
        need = {rag.settings.embed_model, rag.settings.embed_model + ":latest"}
        if not (need & models):
            self.skipTest("embed model not pulled")

    def test_embed_returns_unit_vectors(self):
        v = rag.embed(["hello world", "a second sentence"])
        self.assertEqual(v.shape[0], 2)
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
