from __future__ import annotations
import argparse, json, random
from .core import INTENTS, SupportAgent
from .io import extract_pairs, load_pairs, read_jsonl, write_jsonl
from .evaluation import evaluate

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd", required=True)
    a=sub.add_parser("prepare"); a.add_argument("--input",required=True); a.add_argument("--brand",default="Uber_Support"); a.add_argument("--limit",type=int,default=20000); a.add_argument("--output",required=True)
    a=sub.add_parser("sample-golden"); a.add_argument("--pairs",required=True); a.add_argument("--output",required=True); a.add_argument("--n",type=int,default=200); a.add_argument("--seed",type=int,default=42)
    a=sub.add_parser("evaluate"); a.add_argument("--golden",required=True); a.add_argument("--pairs",default="dataset/demo/pairs.jsonl")
    sub.add_parser("demo")
    args=p.parse_args()
    if args.cmd=="prepare": write_jsonl(args.output, extract_pairs(args.input,args.brand,args.limit)); return
    if args.cmd=="sample-golden":
        rows=read_jsonl(args.pairs); rng=random.Random(args.seed)
        # Equal allocation prevents the largest operational bucket from masking
        # safety and account failures in the golden evaluation set.
        groups={intent:[] for intent in INTENTS}
        for row in rows: groups.get(row.get("intent", "other"), groups["other"]).append(row)
        selected=[]; per_intent=args.n // len(INTENTS)
        for intent in INTENTS:
            rng.shuffle(groups[intent]); selected.extend(groups[intent][:per_intent])
        remaining=[r for r in rows if r not in selected]; rng.shuffle(remaining); selected.extend(remaining[:args.n-len(selected)])
        rng.shuffle(selected)
        write_jsonl(args.output,[{"customer_text":r["customer_text"],"source_customer_id":r["customer_id"],"intent":"","automation_label":"","judge_human":"","notes":""} for r in selected]); return
    pairs=load_pairs(getattr(args, "pairs", "dataset/demo/pairs.jsonl"))
    if args.cmd=="demo":
        for text in ["Why was I charged more for my ride?", "My driver was in an accident"]: print(json.dumps(SupportAgent(pairs).decide(text).__dict__, indent=2))
    else: print(json.dumps(evaluate(read_jsonl(args.golden),pairs),indent=2))
if __name__ == "__main__": main()
