"""Local, reply-blind annotation UI for the golden evaluation set."""
from __future__ import annotations
import argparse, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from .core import INTENTS
from .io import read_jsonl, write_jsonl

def load_state(source: str, output: str) -> list[dict]:
    rows = read_jsonl(output) if Path(output).exists() else read_jsonl(source)
    for row in rows:
        for key in ("intent", "automation_label", "judge_human", "notes"):
            row.setdefault(key, "")
    return rows

def next_index(rows: list[dict], requested: int | None = None) -> int:
    if requested is not None and 0 <= requested < len(rows): return requested
    return next((i for i, row in enumerate(rows) if not row["intent"] or not row["automation_label"]), 0)

def page(rows: list[dict], index: int) -> str:
    row=rows[index]; completed=sum(bool(x['intent'] and x['automation_label']) for x in rows)
    options=lambda values, current: ''.join(f'<option value="{x}" {"selected" if current==x else ""}>{x}</option>' for x in values)
    return f'''<!doctype html><meta charset=utf-8><title>Golden-set labeler</title><style>
body{{font:16px system-ui;max-width:790px;margin:0 auto;padding:40px 20px;background:#f6f8fb;color:#172033}} .card{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 6px #dbe2ee}} select,textarea,button{{font:inherit;padding:9px}} select{{width:100%;margin:5px 0 17px}} textarea{{width:100%;height:70px;box-sizing:border-box}} button{{margin-top:16px;background:#245fd4;color:white;border:0;border-radius:8px;padding:10px 15px;font-weight:650}} .message{{white-space:pre-wrap;line-height:1.6;border-left:3px solid #245fd4;padding-left:13px}} .meta{{color:#607088;font-size:14px}}</style>
<h1>Golden-set labeler</h1><p class=meta>Item {index+1} of {len(rows)} · {completed} complete · Label from the customer message alone.</p><main class=card><p class=message>{html.escape(row['customer_text'])}</p>
<form method=post><input type=hidden name=index value="{index}"><label>Intent</label><select name=intent><option value="">Choose one</option>{options(INTENTS,row['intent'])}</select>
<label>Should the agent auto-handle this?</label><select name=automation_label><option value="">Choose one</option>{options(('auto_handle','escalate'),row['automation_label'])}</select>
<label>Notes (optional)</label><textarea name=notes>{html.escape(row['notes'])}</textarea><br><button>Save and next</button></form></main>'''

def handler(rows: list[dict], output: str):
    class Handler(BaseHTTPRequestHandler):
        def send_page(self, content):
            body=content.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            query=parse_qs(self.path.partition('?')[2]); requested=query.get('item',[None])[0]
            self.send_page(page(rows,next_index(rows,int(requested) if requested and requested.isdigit() else None)))
        def do_POST(self):
            raw=self.rfile.read(int(self.headers.get('Content-Length','0'))).decode(); form=parse_qs(raw)
            index=int(form['index'][0]); row=rows[index]
            intent=form.get('intent',[''])[0]; action=form.get('automation_label',[''])[0]
            if intent in INTENTS: row['intent']=intent
            if action in ('auto_handle','escalate'): row['automation_label']=action
            row['notes']=form.get('notes',[''])[0].strip(); write_jsonl(output,rows)
            self.send_response(303); self.send_header('Location','/?item='+str(index+1 if index+1<len(rows) else 0)); self.end_headers()
        def log_message(self, *_): pass
    return Handler

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',default='dataset/golden/to_label.jsonl'); parser.add_argument('--output',default='dataset/golden/golden_v1.jsonl'); parser.add_argument('--port',type=int,default=8001); args=parser.parse_args()
    rows=load_state(args.input,args.output)
    print(f'Open http://127.0.0.1:{args.port}; labels save to {args.output}')
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(rows,args.output)).serve_forever()
if __name__=='__main__': main()
