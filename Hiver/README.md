# Hiver Support Console

Hiver is a local, AI-powered customer support triage agent built for the Uber Support dataset. It acts as an intelligent proxy, retrieving semantically similar historical customer conversations and deciding whether a case can be auto-handled or must be escalated to a human agent.

This project features a high-performance **FastAPI** backend and a fully interactive **Three.js + React** frontend with a premium glassmorphic Light Theme.

## Features
- **Privacy-First AI**: 100% local processing. No data is sent to external APIs (OpenAI, Anthropic, etc.).
- **Hybrid Retrieval**: Uses Dense Cosine + BM25 retrieval to find historical precedents against 19,468 unique customer support interactions.
- **Safety Guardrails**: High-consequence intents (threats, accidents, off-app cash) are deterministically hard-routed to a human without waiting for an LLM response.
- **Live 3D Console**: A visually stunning Vite/React frontend featuring a dynamic 3D particle background and real-time Server-Sent Event (SSE) streaming.

---

## Prerequisites

Because Hiver runs entirely locally, you will need the following installed on your machine:
- **Python 3.11+**
- **Node.js 20+**
- **[Ollama](https://ollama.com/)** running locally. Ollama is a local model server that executes the Large Language Models required by this project.

> **Why aren't the LLMs in `requirements.txt`?**
> The `qwen3:4b` and `nomic-embed-text` models are not Python packages. They are large AI weights executed by the standalone Ollama daemon. The instructions below will pull them automatically via the Ollama CLI.

---

## Quick Start

Follow these steps to get the full stack running on your machine.

### 1. Install Dependencies
Initialize the Python virtual environment and install both backend and frontend dependencies:
```bash
make install
```

### 2. Pull the AI Models
Download the embedding and generation models into your local Ollama server:
```bash
make models
```

### 3. Build the Semantic Index
Generate the vector embeddings for the historical dataset (requires Ollama to be running). This step takes ~5 minutes on a modern machine:
```bash
make index
```

### 4. Start the Application
You can run the backend and frontend simultaneously with hot-reloading using the following commands (in separate terminal windows):

**Window 1 (Backend API):**
```bash
make dev-api
```

**Window 2 (Frontend Interface):**
```bash
make dev-web
```

Navigate to [http://localhost:5173](http://localhost:5173) in your browser to experience the Hiver console.

### 5. Reproduce Headline Results
To reproduce the automated evaluation metrics and see the LLM-as-judge scores on the golden dataset (runs in ~2 minutes):
```bash
python scripts/eval_rag.py
```
*Note: For a detailed breakdown of these results, the problem framing, and failure analysis, please see [REPORT.md](./REPORT.md).*

---

## Architecture

```
┌──────────────┐   nomic-embed-text    ┌───────────────────────┐
│  customer    │ ────────────────────▶ │ hybrid retrieval      │
│  message     │                       │ dense cosine + BM25   │
└──────────────┘                       └───────────┬───────────┘
                                                   ▼
               ┌───────────────────────┐   ┌──────────────────────────┐
               │  fast path (no LLM)   │   │  qwen3:4b structured     │
               │  very close precedent │   │  JSON decision           │
               └───────────┬───────────┘   └────────────┬─────────────┘
                           └───────────────┬────────────┘
                                           ▼
                         ┌──────────────────────────────────┐
                         │  deterministic safety guardrails │
                         └───────────────┬──────────────────┘
                         auto-handle ◀───┴───▶ human review
```

## Hardware Requirements
- **VRAM**: The `qwen3:4b` model requires approximately ~2.5 GB of VRAM. A machine with 8GB+ of Unified Memory/RAM is highly recommended. 
- If running on a low-memory system, the first request may take 10-15 seconds to load weights, while subsequent warm requests will resolve in 3-4 seconds. Confident routine cases take the fast path and bypass the LLM entirely (~80ms).
