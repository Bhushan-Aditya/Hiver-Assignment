"""Persist semantic vectors for fast serving.

    python scripts/build_rag_index.py --pairs dataset/processed/uber_pairs.jsonl \
        --output dataset/processed/uber_rag_index.npz --batch-size 256

Omit --max-pairs to embed the whole corpus. Near-duplicate customer messages
are collapsed before embedding unless --no-dedupe is passed.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiver_agent.io import load_pairs  # noqa: E402
from hiver_agent.rag import build_index  # noqa: E402
from hiver_agent.retrieval import dedupe_pairs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--pairs", required=True)
p.add_argument("--output", required=True)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--max-pairs", type=int)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-dedupe", action="store_true")
args = p.parse_args()

pairs = load_pairs(args.pairs)
if not args.no_dedupe:
    before = len(pairs)
    pairs = dedupe_pairs(pairs)
    print(f"dedupe: {before} -> {len(pairs)} pairs", flush=True)
if args.max_pairs and len(pairs) > args.max_pairs:
    pairs = random.Random(args.seed).sample(pairs, args.max_pairs)

print(f"Embedding {len(pairs)} support pairs (batch={args.batch_size})...", flush=True)
start = time.perf_counter()
count = build_index(pairs, args.output, args.batch_size, dedupe=False)
print(f"Wrote {args.output}  ({count} vectors, {time.perf_counter() - start:.1f}s)", flush=True)
