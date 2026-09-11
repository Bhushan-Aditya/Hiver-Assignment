import csv
import os
import tempfile
import unittest

from hiver_agent.annotate import next_index
from hiver_agent.core import Pair, SupportAgent, classify
from hiver_agent.evaluation import evaluate
from hiver_agent.io import extract_pairs

PAIRS = [Pair("1", "my driver cancelled the trip", "2", "Please request another ride.", "trip_status")]


class KeywordBaselineTest(unittest.TestCase):
    def test_safety_escalates(self):
        self.assertEqual(SupportAgent(PAIRS).decide("I was assaulted by a driver").action, "escalate")

    def test_precedent_handles_trip(self):
        self.assertEqual(SupportAgent(PAIRS).decide("my driver cancelled").action, "auto_handle")

    def test_unknown_escalates(self):
        self.assertEqual(SupportAgent(PAIRS).decide("what is happening").action, "escalate")

    def test_classification(self):
        self.assertEqual(classify("promo code failed")[0], "promotion")

    def test_late_driver_is_trip_status(self):
        self.assertEqual(classify("my driver was late")[0], "trip_status")

    def test_extra_cash_request_escalates(self):
        d = SupportAgent(PAIRS).decide("My driver asked me for extra 200 rupees")
        self.assertEqual(d.intent, "fare_or_charge")
        self.assertEqual(d.action, "escalate")


class PipelineTest(unittest.TestCase):
    def test_unsafe_auto_rate_conditions_on_auto_handling(self):
        result = evaluate(
            [
                {"customer_text": "my driver cancelled", "intent": "trip_status", "automation_label": "auto_handle"},
                {"customer_text": "I was assaulted", "intent": "safety", "automation_label": "escalate"},
            ],
            PAIRS,
        )
        self.assertEqual(result["unsafe_auto_rate_on_labeled"], 0)

    def test_extract_pairs_follows_parent_link(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False) as f:
            w = csv.DictWriter(
                f,
                fieldnames=["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"],
            )
            w.writeheader()
            w.writerow({"tweet_id": "c1", "author_id": "person", "inbound": "True", "created_at": "", "text": "my ride cancelled", "response_tweet_id": "r1", "in_response_to_tweet_id": ""})
            w.writerow({"tweet_id": "r1", "author_id": "Uber_Support", "inbound": "False", "created_at": "", "text": "request another ride", "response_tweet_id": "", "in_response_to_tweet_id": "c1"})
            path = f.name
        try:
            self.assertEqual(extract_pairs(path, "Uber_Support", 10)[0]["reply_id"], "r1")
        finally:
            os.unlink(path)

    def test_annotation_moves_to_first_incomplete_row(self):
        rows = [{"intent": "trip_status", "automation_label": "auto_handle"}, {"intent": "", "automation_label": ""}]
        self.assertEqual(next_index(rows), 1)


if __name__ == "__main__":
    unittest.main()
