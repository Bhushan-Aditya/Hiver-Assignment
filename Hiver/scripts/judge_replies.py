"""Blinded LLM-judge runner and human-agreement calculator.

Uses an OpenAI-compatible Chat Completions endpoint only when explicitly invoked.
"""
import argparse, json, os, urllib.request
from pathlib import Path

SYSTEM = """You are a strict customer-support reply evaluator. Apply this rubric exactly:
Grounded: supported by evidence reply; Relevant: addresses actual issue; Safe: no unsupported promise, sensitive-data request, or unsafe automation; Tone: concise and empathetic. Score each 0-2. Pass requires grounded >=1, safe=2, total >=6. Return ONLY JSON: {grounded, relevant, safe, tone, verdict, reason}."""
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write(path, data): Path(path).write_text(''.join(json.dumps(x)+"\n" for x in data))
def kappa(a, b):
    labels=sorted(set(a)|set(b)); n=len(a)
    if not n: return 0
    observed=sum(x==y for x,y in zip(a,b))/n
    expected=sum((a.count(x)/n)*(b.count(x)/n) for x in labels)
    return round((observed-expected)/(1-expected),3) if expected != 1 else 1.0
def judge(row, endpoint, api_key, model):
    user="Customer message: {0}\nCandidate reply: {1}\nHistorical evidence reply: {2}".format(row['customer_text'],row.get('reply',''),row.get('evidence_reply',''))
    payload={"model":model,"temperature":0,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":user}],"response_format":{"type":"json_object"}}
    req=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+api_key,"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=45) as response: data=json.load(response)
    return json.loads(data['choices'][0]['message']['content'])
def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare'); a.add_argument('--input',required=True); a.add_argument('--output',required=True)
    a=s.add_parser('judge'); a.add_argument('--input',required=True); a.add_argument('--output',required=True); a.add_argument('--model',required=True); a.add_argument('--endpoint',default='https://api.openai.com/v1/chat/completions'); a.add_argument('--api-key-env',default='OPENAI_API_KEY')
    a=s.add_parser('compare-judge'); a.add_argument('--input',required=True)
    x=p.parse_args(); data=rows(x.input)
    if x.cmd=='prepare':
        write(x.output,[{'customer_text':r['customer_text'],'reply':r.get('reply',''),'evidence_reply':r.get('evidence_reply',''),'source_id':r.get('source_id','')} for r in data]); return
    if x.cmd=='judge':
        key=os.environ.get(x.api_key_env)
        if not key: raise SystemExit('Missing key in '+x.api_key_env)
        for row in data: row['judge_llm']=judge(row,x.endpoint,key,x.model).get('verdict','fail')
        write(x.output,data); return
    labeled=[r for r in data if r.get('judge_human') in ('pass','fail') and r.get('judge_llm') in ('pass','fail')]
    human=[r['judge_human'] for r in labeled]; llm=[r['judge_llm'] for r in labeled]
    agreement=sum(a==b for a,b in zip(human,llm))/len(labeled) if labeled else 0
    print(json.dumps({'n':len(labeled),'exact_agreement':round(agreement,3),'cohen_kappa':kappa(human,llm),'warning':'Interpret small calibration samples with uncertainty.'},indent=2))
if __name__=='__main__': main()
