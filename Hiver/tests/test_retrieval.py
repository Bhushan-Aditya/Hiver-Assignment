import unittest

import numpy as np

from hiver_agent.core import Pair
from hiver_agent.retrieval import BM25, HybridRetriever, dedupe_pairs, tokenize


def _pair(i, cust, reply, intent=""):
    return Pair(str(i), cust, f"r{i}", reply, intent)


CORPUS = [
    _pair(1, "my driver was late for pickup and never arrived", "Please check the driver location in the app.", "trip_status"),
    _pair(2, "I was charged twice for one trip this morning", "We will review the duplicate charge.", "fare_or_charge"),
    _pair(3, "the promo code FIRST50 did not apply at checkout", "Check the promo expiry in the app.", "promotion"),
    _pair(4, "driver asked me for an extra 200 rupees in cash", "Please contact support about this.", "fare_or_charge"),
    _pair(5, "I cannot log into my account it is disabled", "We can help restore access.", "account_access"),
]


class BM25Test(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_handles(self):
        toks = tokenize("@12345 my driver was VERY late https://t.co/x #uber")
        self.assertIn("driver", toks)
        self.assertIn("late", toks)
        self.assertNotIn("my", toks)
        self.assertNotIn("was", toks)

    def test_bm25_ranks_lexical_match_first(self):
        docs = [tokenize(p.customer_text) for p in CORPUS]
        bm = BM25(docs)
        scores = bm.scores(tokenize("charged twice duplicate trip"))
        self.assertTrue(scores)
        best = max(scores, key=scores.get)
        self.assertEqual(best, 1)  # the "charged twice" doc


class DedupeTest(unittest.TestCase):
    def test_dedupe_collapses_normalized_duplicates(self):
        pairs = [
            _pair(1, "My driver was LATE!!!", "a"),
            _pair(2, "my driver was late", "b"),
            _pair(3, "totally different message", "c"),
        ]
        self.assertEqual(len(dedupe_pairs(pairs)), 2)


class HybridTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(len(CORPUS), 16)).astype(np.float32)
        self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.retr = HybridRetriever(CORPUS, self.vectors)

    def test_search_returns_k_hits_sorted(self):
        q = self.vectors[3]
        hits = self.retr.search("driver asked for extra 200 rupees cash", q, k=3)
        self.assertEqual(len(hits), 3)
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_lexical_signal_surfaces_exact_terms(self):
        # A query with strong lexical overlap but a random query vector should
        # still surface the off-app-cash precedent via BM25 fusion.
        q = self.vectors[0]
        hits = self.retr.search("driver asked me for extra 200 rupees in cash", q, k=5)
        self.assertIn(4, [h.index for h in hits])


if __name__ == "__main__":
    unittest.main()
