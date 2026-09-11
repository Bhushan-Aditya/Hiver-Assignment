from __future__ import annotations
import csv, json
from pathlib import Path
from .core import Pair, classify

def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f: return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_pairs(path: str) -> list[Pair]:
    return [Pair(**row) for row in read_jsonl(path)]

def extract_pairs(csv_path: str, brand: str, limit: int) -> list[dict]:
    """Extract parent customer messages for one brand in two bounded-memory passes.

    The full Kaggle CSV is ~500 MB; loading every CSV row as a dictionary is both
    unnecessary and unreliable on a laptop. Brand replies point to their parent
    customer message via `in_response_to_tweet_id`, so first collect only those
    response records, then stream once more to find their parents.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        replies_by_parent = {}
        for row in csv.DictReader(f):
            if row.get("author_id") != brand or not row.get("in_response_to_tweet_id"):
                continue
            for parent_id in row["in_response_to_tweet_id"].split(","):
                parent_id = parent_id.strip()
                if parent_id and parent_id not in replies_by_parent:
                    replies_by_parent[parent_id] = row
    output = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for customer in csv.DictReader(f):
            if customer.get("inbound", "").lower() != "true":
                continue
            reply = replies_by_parent.get(customer.get("tweet_id"))
            if not reply:
                continue
            intent, _, _ = classify(customer["text"])
            output.append({"customer_id": customer["tweet_id"], "customer_text": customer["text"], "reply_id": reply["tweet_id"], "reply_text": reply["text"], "intent": intent})
            if len(output) >= limit:
                break
    return output
