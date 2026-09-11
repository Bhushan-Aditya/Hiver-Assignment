"""Dependency-free, auditable support-agent core."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import math, re

INTENTS = ("trip_status", "fare_or_charge", "payment", "driver_or_rider", "account_access", "promotion", "safety", "other")
KEYWORDS = {
    "safety": ("accident", "unsafe", "emergency", "assault", "harass", "threat", "police", "hurt"),
    "payment": ("payment", "card", "cash", "wallet", "refund card"),
    "fare_or_charge": ("fare", "charged", "charge", "price", "receipt", "refund", "overcharg"),
    "trip_status": ("trip", "ride", "pickup", "driver", "arrived", "cancel", "waiting", "wait", "late", "eta"),
    "driver_or_rider": ("rider", "passenger", "rating", "rate driver", "rude", "behaviour", "behavior"),
    "account_access": ("account", "login", "sign in", "password", "phone number", "verify"),
    "promotion": ("promo", "coupon", "discount", "code", "offer"),
}
RISK_WORDS = ("__email__", "__phone_number__", "card number", "password", "emergency", "police", "assault", "suicide", "threat")

# Phrase-level safety net. Any match forces escalation with no automated reply,
# on every decision path. Kept broad on purpose -- a false escalation costs a
# human a minute; a false auto-handle on a safety issue is unacceptable.
SAFETY_RE = re.compile(
    r"\b("
    r"unsafe|not safe|feel(ing)? (un)?safe|scared|afraid|terrified|in danger|"
    r"follow(ing|ed)? me|following me|chas(e|ing) me|won'?t (stop|let me (out|go))|"
    r"locked (me )?in|kidnap|abduct|held against|trapped in the car|"
    r"threat(en(ed|ing)?)?|weapon|gun|knife|assault|attack(ed|ing)?|hit me|punch|"
    r"grab(bed)?|touch(ed|ing)? me|grop|harass|stalk|"
    r"drunk driver|driver (is |was )?(drunk|intoxicated|high)|"
    r"accident|crash(ed)?|collision|injur(y|ed)|bleeding|hospital|ambulance|911|emergency"
    r")\b"
)
OFF_APP_PAYMENT_WORDS = ("extra cash", "extra money", "extra rupees", "pay cash", "cash payment", "cash directly", "pay him", "pay her", "asked me for extra", "asked for extra")
FOOD_WORDS = ("ubereats", "uber eats", "food", "restaurant", "order", "delivery", "deliver", "meal")
TOKEN = re.compile(r"[a-z0-9_]+")

def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))

def classify(text: str) -> tuple[str, float, dict[str, int]]:
    lower = text.lower()
    scores = {intent: sum(word in lower for word in words) for intent, words in KEYWORDS.items()}
    # A late/missing driver is a trip-status problem, not a driver-rating issue.
    if "driver" in lower and any(word in lower for word in ("late", "wait", "arriv", "pickup", "cancel", "eta")):
        scores["trip_status"] += 2
    if any(word in lower for word in OFF_APP_PAYMENT_WORDS):
        scores["fare_or_charge"] += 3
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if not best_score:
        return "other", 0.35, scores
    total = sum(scores.values())
    # Margin-based confidence stays deliberately modest for ambiguous messages.
    confidence = min(0.95, 0.55 + 0.15 * best_score + 0.10 * (best_score / max(total, 1)))
    return best, round(confidence, 2), scores

@dataclass
class Pair:
    customer_id: str
    customer_text: str
    reply_id: str
    reply_text: str
    intent: str = ""

def domain(text: str) -> str:
    return "delivery" if any(word in text.lower() for word in FOOD_WORDS) else "ride"

SAFE_DRAFTS = {
    "trip_status": "Sorry your driver is delayed. Please check the driver's location in the app. You can contact the driver, or cancel and request another ride if needed.",
    "fare_or_charge": "Sorry the fare looks unexpected. Open the trip in the app and choose Get trip help so the charge can be reviewed.",
    "promotion": "Sorry the promotion did not apply. Check its expiry and eligibility in the app, then select the trip for help if it still fails.",
    "driver_or_rider": "Sorry you had this experience. Please select the trip in the app and use Get trip help so the relevant details can be reviewed.",
}

@dataclass
class Decision:
    intent: str
    confidence: float
    reply: str
    evidence_reply_id: str | None
    action: str
    reason: str

class SupportAgent:
    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        self.documents = [tokens(p.customer_text) for p in pairs]
        self.idf = self._idf(self.documents)

    @staticmethod
    def _idf(documents: list[set[str]]) -> dict[str, float]:
        count = Counter(t for doc in documents for t in doc)
        n = max(len(documents), 1)
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in count.items()}

    def retrieve(self, text: str, intent: str) -> tuple[Pair | None, float]:
        query = tokens(text)
        query_domain = domain(text)
        best, score = None, 0.0
        for pair, doc in zip(self.pairs, self.documents):
            if pair.intent and pair.intent != intent:
                continue
            if domain(pair.customer_text + " " + pair.reply_text) != query_domain:
                continue
            inter = query & doc
            union = query | doc
            candidate = sum(self.idf.get(t, 1) for t in inter) / max(sum(self.idf.get(t, 1) for t in union), 1)
            if candidate > score:
                best, score = pair, candidate
        return best, round(score, 3)

    def decide(self, text: str) -> Decision:
        intent, confidence, _ = classify(text)
        precedent, similarity = self.retrieve(text, intent)
        risk = [word for word in RISK_WORDS if word in text.lower()]
        if intent == "safety" or SAFETY_RE.search(text.lower()):
            return Decision("safety" if intent == "safety" else intent, confidence, "", None, "escalate", "Safety-sensitive issue requires a human agent.")
        if any(word in text.lower() for word in OFF_APP_PAYMENT_WORDS):
            return Decision(intent, confidence, "", None, "escalate", "Driver-requested off-app payment requires human review.")
        if risk:
            return Decision(intent, confidence, "", None, "escalate", f"Sensitive-content marker: {risk[0]}.")
        if confidence < 0.70:
            return Decision(intent, confidence, "", None, "escalate", "Intent confidence is below the automation threshold.")
        if precedent is None or similarity < 0.08:
            return Decision(intent, confidence, "", None, "escalate", "No sufficiently similar historical resolution was found.")
        if intent in {"payment", "account_access"}:
            return Decision(intent, confidence, "", precedent.reply_id, "escalate", "Financial or account-access issue needs human confirmation.")
        draft = SAFE_DRAFTS.get(intent)
        if not draft:
            return Decision(intent, confidence, "", precedent.reply_id, "escalate", "No safe, approved response template for this intent.")
        return Decision(intent, confidence, draft, precedent.reply_id, "auto_handle", "Low-risk intent with a similar historical resolution and approved reply pattern.")

def decision_dict(decision: Decision) -> dict: return asdict(decision)
