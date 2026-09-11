"""Local semantic RAG: Ollama embeddings + hybrid retrieval + evidence-bounded
generation, with a no-LLM fast path for confident routine cases.

Latency budget on a laptop is dominated by LLM generation. Two things keep the
median response fast without weakening safety:

1. A confident-retrieval fast path returns a templated, pre-approved next step
   when the nearest historical precedent is very close and the intent is
   low-risk -- no generation call at all.
2. Otherwise the model is called once with a trimmed prompt, a small output
   budget, and a warm keep-alive so weights are not reloaded per request.

Deterministic safety guardrails run on every path and can only ever make the
decision *more* conservative.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import (
    ESCALATE_ONLY_INTENTS,
    FASTPATH_BLOCK_WORDS,
    FASTPATH_INTENTS,
    settings,
)
from .core import RISK_WORDS, SAFE_DRAFTS, SAFETY_RE, Decision, Pair
from .retrieval import HybridRetriever, dedupe_pairs

INTENTS = (
    "trip_status", "fare_or_charge", "payment", "driver_or_rider",
    "account_access", "promotion", "safety", "other",
)

_OFF_APP_PAYMENT = re.compile(
    r"(asked.{0,25}extra.{0,25}(cash|money|rupee|rs)|extra\s*\d+\s*(rupee|rs)"
    r"|pay\s+(him|her|driver).{0,25}(cash|direct))"
)


# --------------------------------------------------------------------------- #
# Ollama plumbing
# --------------------------------------------------------------------------- #
def ollama(path: str, payload: dict, timeout: int = 120, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            settings.ollama_host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))  # brief backoff for scheduler/VRAM contention
    raise last  # type: ignore[misc]


def ollama_healthy() -> tuple[bool, list[str]]:
    try:
        req = urllib.request.Request(settings.ollama_host + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            return True, [m["name"] for m in json.load(r).get("models", [])]
    except Exception:  # pragma: no cover - network dependent
        return False, []


def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    data = ollama("/api/embed", {"model": model or settings.embed_model, "input": texts})
    matrix = np.asarray(data["embeddings"], dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# Index build / load
# --------------------------------------------------------------------------- #
def build_index(pairs: list[Pair], output: str, batch_size: int = 64, dedupe: bool = True) -> int:
    """Persist embeddings so serving never re-embeds the corpus. Returns count."""
    if dedupe:
        pairs = dedupe_pairs(pairs)
    docs = [f"Customer: {p.customer_text}\nSupport resolution: {p.reply_text}" for p in pairs]
    vectors = [embed(docs[s:s + batch_size]) for s in range(0, len(docs), batch_size)]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vectors=np.vstack(vectors),
        pairs=np.asarray([json.dumps(asdict(p), ensure_ascii=False) for p in pairs]),
    )
    return len(pairs)


def load_index(path: str) -> tuple[list[Pair], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    pairs = [Pair(**json.loads(row)) for row in data["pairs"]]
    return pairs, data["vectors"]


def parse_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("No JSON object in model response")
    return json.loads(match.group(0))


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    """Decision plus everything the UI and evals need to explain it."""
    decision: Decision
    path: str                       # "fast" | "llm" | "fallback"
    retrieval_score: float
    evidence: list[dict]
    timings_ms: dict[str, float]
    knn_intent: str = "other"
    knn_strength: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self.decision)
        d.update(
            path=self.path,
            retrieval_score=round(self.retrieval_score, 3),
            knn_intent=self.knn_intent,
            knn_strength=round(self.knn_strength, 3),
            evidence=self.evidence,
            timings_ms={k: round(v, 1) for k, v in self.timings_ms.items()},
        )
        return d


class RagAgent:
    def __init__(self, pairs: list[Pair], vectors: np.ndarray, model: str | None = None):
        self.pairs = pairs
        self.vectors = vectors
        self.model = model or settings.gen_model
        self.retriever = HybridRetriever(
            pairs, vectors, dense_weight=settings.dense_weight, rrf_k=settings.rrf_k
        )
        self._cache: OrderedDict[str, AgentResult] = OrderedDict()
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @classmethod
    def from_index(cls, path: str, model: str | None = None) -> "RagAgent":
        pairs, vectors = load_index(path)
        return cls(pairs, vectors, model)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _key(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _embed_query(self, query: str) -> np.ndarray:
        key = self._key(query)
        cached = self._embed_cache.get(key)
        if cached is not None:
            self._embed_cache.move_to_end(key)
            return cached
        vec = embed([query])[0]
        self._embed_cache[key] = vec
        if len(self._embed_cache) > settings.cache_size:
            self._embed_cache.popitem(last=False)
        return vec

    def _cache_put(self, key: str, result: AgentResult) -> None:
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > settings.cache_size:
            self._cache.popitem(last=False)

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[Pair, float]]:
        """Back-compat: (pair, dense_similarity) tuples, best first."""
        hits = self.retriever.search(query, self._embed_query(query), k or settings.evidence_k)
        return [(h.pair, h.dense) for h in hits]

    # -- kNN intent (deterministic, free) ---------------------------------------
    @staticmethod
    def _knn_intent(hits) -> tuple[str, float]:
        """Similarity-weighted vote over retrieved weak labels.

        A cheap kNN classifier over 19k historical pairs. It is well-behaved on
        common intents and is used to (a) rescue the LLM when it returns
        'other', and (b) extend the safety net to paraphrased safety issues.
        Returns (intent, strength) where strength is the winning vote share.
        """
        weights: dict[str, float] = {}
        total = 0.0
        for h in hits:
            if not h.pair.intent:
                continue
            w = max(0.0, h.dense) ** 2  # sharpen toward the closest neighbours
            weights[h.pair.intent] = weights.get(h.pair.intent, 0.0) + w
            total += w
        if not weights or total <= 0:
            return "other", 0.0
        intent, w = max(weights.items(), key=lambda kv: kv[1])
        return intent, round(w / total, 3)

    # -- guardrails ---------------------------------------------------------------
    @staticmethod
    def _guardrails(query: str, intent: str, action: str, reply: str, reason: str,
                    top_sim: float) -> tuple[str, str, str]:
        lower = query.lower()
        if SAFETY_RE.search(lower):
            return "escalate", "", "Message contains a safety signal; routed to a human immediately."
        off_app = bool(_OFF_APP_PAYMENT.search(lower))
        if any(marker in lower for marker in RISK_WORDS) or off_app:
            return "escalate", "", "Sensitive or off-app payment issue requires human review."
        if intent in ESCALATE_ONLY_INTENTS and action == "auto_handle":
            return "escalate", "", f"{intent.replace('_', ' ').title()} issues require human confirmation."
        if action == "auto_handle" and top_sim < settings.auto_min_sim:
            return "escalate", "", "No sufficiently similar semantic precedent was found."
        if action == "escalate":
            return "escalate", "", reason
        return action, reply, reason

    # -- fast path ---------------------------------------------------------------
    def _fast_path(self, query: str, hits) -> Decision | None:
        if not settings.enable_fastpath or not hits:
            return None
        top = hits[0]
        if top.dense < settings.fastpath_sim:
            return None
        # Need a clear majority weak-label among the evidence shown.
        votes = Counter(h.pair.intent for h in hits[: settings.evidence_k] if h.pair.intent)
        if not votes:
            return None
        intent, count = votes.most_common(1)[0]
        if count < settings.evidence_k or intent not in FASTPATH_INTENTS:
            return None
        lower = query.lower()
        if (
            SAFETY_RE.search(lower)
            or any(m in lower for m in RISK_WORDS)
            or _OFF_APP_PAYMENT.search(lower)
            or any(w in lower for w in FASTPATH_BLOCK_WORDS)
        ):
            return None
        draft = SAFE_DRAFTS.get(intent)
        if not draft:
            return None
        confidence = round(min(0.95, 0.60 + 0.35 * (top.dense - settings.fastpath_sim) / max(1e-6, 1 - settings.fastpath_sim)), 2)
        return Decision(
            intent, confidence, draft, top.pair.reply_id, "auto_handle",
            "Very close historical precedent; approved templated next step (no generation needed).",
        )

    # -- llm path ---------------------------------------------------------------
    def _llm_decision(self, query: str, hits) -> Decision:
        citations = "\n\n".join(
            f"[{i + 1}] Customer: {h.pair.customer_text}\nHistorical support reply: {h.pair.reply_text}"
            for i, h in enumerate(hits[: settings.evidence_k])
        )
        prompt = (
            "You are an Uber customer-support triage agent. Decide only from the retrieved historical cases.\n\n"
            f"Customer message: {query}\n\nRetrieved cases:\n{citations}\n\n"
            f"Return JSON only with fields: intent (one of {list(INTENTS)}), action (auto_handle or escalate), "
            "confidence (0 to 1), reply, reason.\n"
            "Intent notes: a late/missing driver is trip_status; driver_or_rider is about conduct/ratings; "
            "fare_or_charge covers fares, duplicate charges, refunds, and extra driver-requested payment.\n"
            "Auto-handle only routine low-risk trip, fare, or promotion questions the evidence supports. "
            "Always escalate safety, threats, account access, payment, off-app cash requests, and unclear cases. "
            "Never ask for credentials, card, phone, or email. Reply <= 2 sentences. Reason <= 18 words. "
            "If escalating, reply must be an empty string."
        )
        raw = ollama(
            "/api/generate",
            {
                "model": self.model, "prompt": prompt, "stream": False, "format": "json",
                "think": False, "keep_alive": settings.gen_keep_alive,
                "options": {
                    "temperature": 0, "num_predict": settings.gen_num_predict,
                    "num_ctx": settings.gen_num_ctx,
                },
            },
            settings.gen_timeout_s,
        )["response"]
        result = parse_json(raw)
        intent = result.get("intent", "other")
        action = result.get("action", "escalate")
        confidence = float(result.get("confidence", 0) or 0)
        reply = str(result.get("reply", "")).strip()
        reason = str(result.get("reason", "Insufficient evidence for automation.")).strip()
        if intent not in INTENTS or action not in ("auto_handle", "escalate"):
            raise ValueError("Invalid model decision")
        return Decision(intent, round(max(0.0, min(confidence, 1.0)), 2),
                        reply, hits[0].pair.reply_id if hits else None, action, reason)

    # -- public ---------------------------------------------------------------
    def assess(self, query: str, drop_index: int | None = None) -> AgentResult:
        for event in self.stream(query, drop_index=drop_index):
            if event["stage"] == "done":
                return event["result"]
        raise RuntimeError("stream produced no result")  # pragma: no cover

    def decide(self, query: str) -> Decision:
        """Back-compat entrypoint used by older callers and tests."""
        return self.assess(query).decision

    def stream(self, query: str, drop_index: int | None = None):
        """Yield progress events, ending with {'stage': 'done', 'result': AgentResult}."""
        query = (query or "").strip()
        key = self._key(query)
        if not query:
            empty = Decision("other", 0.0, "", None, "escalate", "Empty message.")
            yield {"stage": "done", "result": AgentResult(empty, "fallback", 0.0, [], {"total": 0.0})}
            return

        if drop_index is None:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                yield {"stage": "retrieved", "evidence": cached.evidence, "cached": True}
                yield {"stage": "done", "result": cached, "cached": True}
                return

        timings: dict[str, float] = {}
        yield {"stage": "retrieving"}
        t = time.perf_counter()
        qvec = self._embed_query(query)
        timings["embed"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = self.retriever.search(query, qvec, settings.retrieve_k, drop_index=drop_index)
        timings["retrieve"] = (time.perf_counter() - t) * 1000
        evidence = [
            {
                "rank": i + 1,
                "reply_id": h.pair.reply_id,
                "customer_text": h.pair.customer_text,
                "reply_text": h.pair.reply_text,
                "weak_intent": h.pair.intent,
                "similarity": round(h.dense, 3),
                "lexical": round(h.lexical, 2),
                "fused": round(h.score, 3),
            }
            for i, h in enumerate(hits[: settings.evidence_k])
        ]
        knn_intent, knn_strength = self._knn_intent(hits)
        yield {"stage": "retrieved", "evidence": evidence,
               "knn_intent": knn_intent, "knn_strength": knn_strength}

        top_sim = hits[0].dense if hits else 0.0
        fast = self._fast_path(query, hits)
        if fast is not None:
            intent, action, reply, reason = fast.intent, fast.action, fast.reply, fast.reason
            action, reply, reason = self._guardrails(query, intent, action, reply, reason, top_sim)
            decision = Decision(intent, fast.confidence, reply,
                                hits[0].pair.reply_id, action, reason)
            timings["generate"] = 0.0
            timings["total"] = sum(timings.get(k, 0.0) for k in ("embed", "retrieve"))
            result = AgentResult(decision, "fast", float(top_sim), evidence, timings,
                                 knn_intent, knn_strength)
            if drop_index is None:
                self._cache_put(key, result)
            yield {"stage": "done", "result": result}
            return

        yield {"stage": "generating", "model": self.model}
        t = time.perf_counter()
        path = "llm"
        try:
            base = self._llm_decision(query, hits)
        except Exception:
            path = "fallback"
            base = Decision("other", 0.0, "", hits[0].pair.reply_id if hits else None,
                            "escalate", "The local RAG model could not produce a reliable decision.")
        timings["generate"] = (time.perf_counter() - t) * 1000

        # Reconcile the model's intent with the kNN vote over historical labels.
        intent = base.intent
        action, reply, reason = base.action, base.reply, base.reason
        if intent == "other" and knn_strength >= 0.60:
            intent = knn_intent
        if knn_intent == "safety" and knn_strength >= 0.50 and action != "escalate":
            action, reply, reason = "escalate", "", "Nearest historical cases are safety-related; routed to a human."

        action, reply, reason = self._guardrails(query, intent, action, reply, reason, top_sim)
        confidence = base.confidence
        if action == "auto_handle" and confidence < settings.auto_min_conf:
            action, reply, reason = "escalate", "", "Model confidence is below the automation threshold."
        if action == "auto_handle":
            # The historical replies are mostly "DM us" boilerplate, so an
            # auto-handled reply comes from the approved template for the intent,
            # not from the model paraphrasing that boilerplate.
            template = SAFE_DRAFTS.get(intent)
            if template:
                reply = template
            elif not reply:
                action, reason = "escalate", "No approved reply template for this intent."
        decision = Decision(intent, confidence, reply,
                            hits[0].pair.reply_id if hits else None, action, reason)
        timings["total"] = sum(timings.get(k, 0.0) for k in ("embed", "retrieve", "generate"))
        result = AgentResult(decision, path, float(top_sim), evidence, timings,
                             knn_intent, knn_strength)
        if drop_index is None:
            self._cache_put(key, result)
        yield {"stage": "done", "result": result}
