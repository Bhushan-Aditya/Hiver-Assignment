"""Runtime configuration, driven entirely by environment variables.

Every knob has a conservative default so the app runs with zero config on a
laptop, but a deployment can retune retrieval, latency, and safety behaviour
without touching code.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- Ollama / models ---
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    gen_model: str = os.environ.get("HIVER_MODEL", "qwen3:4b")
    embed_model: str = os.environ.get("HIVER_EMBED_MODEL", "nomic-embed-text")
    index_path: str = os.environ.get("HIVER_INDEX_PATH", "dataset/processed/uber_rag_index.npz")

    # --- Retrieval ---
    retrieve_k: int = _i("HIVER_RETRIEVE_K", 8)          # candidates fused/reranked
    evidence_k: int = _i("HIVER_EVIDENCE_K", 3)          # shown to the model + UI
    dense_weight: float = _f("HIVER_DENSE_WEIGHT", 0.65)  # dense vs lexical in fusion
    rrf_k: int = _i("HIVER_RRF_K", 60)                   # reciprocal-rank-fusion constant

    # --- Latency controls ---
    enable_fastpath: bool = _b("HIVER_ENABLE_FASTPATH", True)
    fastpath_sim: float = _f("HIVER_FASTPATH_SIM", 0.74)   # dense sim to skip the LLM
    cache_size: int = _i("HIVER_CACHE_SIZE", 512)
    gen_timeout_s: int = _i("HIVER_GEN_TIMEOUT_S", 90)
    gen_num_predict: int = _i("HIVER_GEN_NUM_PREDICT", 128)
    gen_num_ctx: int = _i("HIVER_GEN_NUM_CTX", 1536)
    gen_keep_alive: str = os.environ.get("HIVER_GEN_KEEP_ALIVE", "30m")

    # --- Safety / automation thresholds ---
    auto_min_sim: float = _f("HIVER_AUTO_MIN_SIM", 0.45)   # floor for any auto_handle
    auto_min_conf: float = _f("HIVER_AUTO_MIN_CONF", 0.55)

    # --- Server ---
    host: str = os.environ.get("HIVER_BIND_HOST", "127.0.0.1")
    port: int = _i("HIVER_PORT", 8000)
    cors_origins: tuple[str, ...] = tuple(
        o for o in os.environ.get("HIVER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o
    )
    static_dir: str = os.environ.get("HIVER_STATIC_DIR", "frontend/dist")


settings = Settings()

# Intents that may ever be auto-handled. Everything else is escalate-only,
# regardless of what the model returns.
AUTO_SAFE_INTENTS = ("trip_status", "fare_or_charge", "promotion", "driver_or_rider")
ESCALATE_ONLY_INTENTS = ("payment", "account_access", "safety")

# The no-LLM fast path is deliberately narrower than AUTO_SAFE_INTENTS: only the
# genuinely low-consequence intents, and only when the message shows no
# financial / account / safety lexical signal at all.
FASTPATH_INTENTS = ("trip_status", "promotion")
FASTPATH_BLOCK_WORDS = (
    "card", "bank", "refund", "charged", "charge", "unauthori", "fraud", "login",
    "log in", "sign in", "password", "account", "hacked", "disabled", "suspend",
    "otp", "verify", "verification", "wallet", "upi", "paypal",
)
