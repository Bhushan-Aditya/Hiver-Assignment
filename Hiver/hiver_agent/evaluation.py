from __future__ import annotations
from collections import Counter
from .core import SupportAgent, Pair

def macro_f1(truth: list[str], pred: list[str]) -> float:
    labels = sorted(set(truth) | set(pred)); scores=[]
    for label in labels:
        tp=sum(a==b==label for a,b in zip(truth,pred)); fp=sum(a!=label and b==label for a,b in zip(truth,pred)); fn=sum(a==label and b!=label for a,b in zip(truth,pred))
        scores.append((2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0)
    return round(sum(scores)/len(scores), 3) if scores else 0

def evaluate(golden: list[dict], knowledge: list[Pair]) -> dict:
    agent=SupportAgent(knowledge); truth=[r["intent"] for r in golden]
    majority=Counter(p.intent for p in knowledge).most_common(1)[0][0] if knowledge else "other"
    predicted=[]; auto=[]; automation_truth=[]
    for row in golden:
        d=agent.decide(row["customer_text"]); predicted.append(d.intent); auto.append(d.action == "auto_handle")
        automation_truth.append(row.get("automation_label", ""))
    coverage=sum(auto)/len(auto) if auto else 0
    reviewed_auto=[label for is_auto,label in zip(auto,automation_truth) if is_auto and label]
    unsafe=sum(label != "auto_handle" for label in reviewed_auto)/max(len(reviewed_auto),1)
    return {"n":len(golden), "majority_baseline_accuracy":round(sum(x==majority for x in truth)/len(truth),3), "agent_accuracy":round(sum(a==b for a,b in zip(truth,predicted))/len(truth),3), "agent_macro_f1":macro_f1(truth,predicted), "automation_coverage":round(coverage,3), "unsafe_auto_rate_on_labeled":round(unsafe,3)}
