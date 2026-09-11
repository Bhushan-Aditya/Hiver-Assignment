"""Latency benchmark against a running API (default http://127.0.0.1:8000).

Sends a fixed set of messages twice (cold, then warm/cache) and prints the
per-request and aggregate timings plus the decision-path mix.

    python scripts/benchmark.py --url http://127.0.0.1:8000 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

MESSAGES = [
    "My driver was 20 minutes late for pickup and I missed my flight.",
    "Why was I charged twice for the same trip this morning?",
    "My promo code FIRST50 did not apply at checkout even though it is still valid.",
    "My driver asked me for an extra 200 rupees in cash at the end of the ride.",
    "I feel unsafe, the driver keeps making comments about where I live.",
    "I cannot log in, my account says it has been disabled.",
    "The driver cancelled on me after making me wait 15 minutes.",
    "Driver took a longer route and the fare was much higher than usual.",
]


def call(url: str, message: str) -> tuple[dict, float]:
    req = urllib.request.Request(
        url + "/api/assess",
        data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r), (time.perf_counter() - t) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    all_walls: list[float] = []
    for rnd in range(1, args.rounds + 1):
        walls: list[float] = []
        paths: dict[str, int] = {}
        print(f"\n--- round {rnd} ---")
        for m in MESSAGES:
            d, wall = call(args.url, m)
            walls.append(wall)
            paths[d["path"]] = paths.get(d["path"], 0) + 1
            tm = d["timings_ms"]
            print(f"{wall:8.0f}ms  {d['path']:5s}  {d['action']:11s}  {d['intent']:14s}  "
                  f"gen={tm.get('generate', 0):.0f}ms  cached={d.get('cached', False)}")
        all_walls += walls
        walls.sort()
        print(f"round {rnd}: p50={walls[len(walls)//2]:.0f}ms  "
              f"p90={walls[int(len(walls)*0.9)]:.0f}ms  mean={statistics.mean(walls):.0f}ms  paths={paths}")

    all_walls.sort()
    print(f"\noverall ({len(all_walls)} calls): p50={all_walls[len(all_walls)//2]:.0f}ms  "
          f"p90={all_walls[int(len(all_walls)*0.9)]:.0f}ms  "
          f"min={all_walls[0]:.0f}ms  max={all_walls[-1]:.0f}ms")


if __name__ == "__main__":
    main()
