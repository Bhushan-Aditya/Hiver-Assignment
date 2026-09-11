"""Backward-compatible launcher.

The console is now a React client served by :mod:`hiver_agent.server`. This
module is kept so existing docs and muscle memory still work:

    python -m hiver_agent.app --index dataset/processed/uber_rag_index.npz --port 8000
"""
from __future__ import annotations

from .server import main

if __name__ == "__main__":
    main()
