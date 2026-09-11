.DEFAULT_GOAL := help
PY ?= python
PAIRS ?= dataset/processed/uber_pairs.jsonl
INDEX ?= dataset/processed/uber_rag_index.npz

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python + frontend deps
	$(PY) -m pip install -r requirements.txt
	cd frontend && npm install

models: ## Pull the local Ollama models
	ollama pull nomic-embed-text
	ollama pull qwen3:4b

index: ## Build the full semantic index from all pairs
	$(PY) scripts/build_rag_index.py --pairs $(PAIRS) --output $(INDEX) --batch-size 128

index-demo: ## Build a fast 5k-pair demo index
	$(PY) scripts/build_rag_index.py --pairs $(PAIRS) --output $(INDEX) --max-pairs 5000 --batch-size 128

frontend: ## Build the production frontend bundle
	cd frontend && npm run build

dev-api: ## Run the API with autoreload (proxied by the Vite dev server)
	$(PY) -m uvicorn hiver_agent.server:app --reload --port 8000

dev-web: ## Run the Vite dev server on :5173
	cd frontend && npm run dev

serve: frontend ## Build frontend then serve everything from :8000
	$(PY) -m hiver_agent.server --port 8000

test: ## Run the unit test suite
	$(PY) -m unittest discover -s tests -v

eval: ## Offline RAG eval (weak-label agreement, latency, safety probes)
	$(PY) scripts/eval_rag.py --index $(INDEX) --n 120

bench: ## Latency-only benchmark against a running API on :8000
	$(PY) scripts/benchmark.py

.PHONY: help install models index index-demo frontend dev-api dev-web serve test eval bench
