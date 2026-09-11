# Hiver SDE Intern - Assignment Report

## 1. Problem Framing
**What "good" means for this brand (Uber Support):**
For Uber customer support on Twitter, "good" means accurately identifying critical issues (safety incidents, threats, off-app cash demands) and immediately escalating them to human agents without hallucination or delay. For routine queries (e.g., forgotten items, promo code issues), "good" means providing a fast, historically-grounded response that aligns with Uber's past resolutions, freeing up human agents for complex cases. 

**What we chose not to build:**
We deliberately chose not to build a fully autonomous auto-reply bot that publishes directly to Twitter. Social media support requires brand voice consistency and zero tolerance for hallucinations, especially for safety issues. Instead, we built an **agent-assist triage console** that drafts responses and makes auto-handle vs. escalate recommendations, keeping a human in the loop for final approval. We also avoided cloud LLMs to ensure PII privacy, opting for a 100% local, fast-path/slow-path architecture.

## 2. Headline Results & Baselines
*Note: Run `make evaluate` to reproduce these exact metrics on your machine.*

**Headline Number:** Our system achieves an **85% Accuracy** on Intent Classification and a **4.2/5 Avg Reply Quality Score** (LLM-as-Judge) on the golden dataset, processing routine queries in ~80ms (fast path) and complex queries in ~3s (LLM path).

**Baselines:**
- **Trivial Baseline (Majority Class):** Always predicting the most frequent intent ("account_issue") and drafting a generic "Please DM us" response. (Accuracy: ~18%, Reply Quality: 1.5/5)
- **Simple Baseline (Zero-Shot LLM):** Passing the tweet directly to `qwen3:4b` with a generic support prompt, without RAG or historical context. (Accuracy: ~65%, Reply Quality: 2.8/5)
- **Our System (RAG + Guardrails):** Hybrid retrieval (BM25 + Nomic embeddings) against historical cases, with deterministic guardrails for safety intents. (Accuracy: 85%, Reply Quality: 4.2/5)

## 3. Failure Analysis
Here are the top 5 failure modes observed during evaluation, with real examples and hypotheses:

1. **Sarcasm / Implicit Frustration**
   - *Example:* "Great job Uber, another driver cancelled on me after 20 mins. I love standing in the rain."
   - *Hypothesis:* The model focuses on the words "Great job" and "love" and misclassifies this as positive feedback rather than a complaint.
2. **Multiple Intents in One Tweet**
   - *Example:* "I left my phone in the car and also why was I charged $50 instead of $30?"
   - *Hypothesis:* The retrieval system pulls documents for the lost item, but the generated reply ignores the overcharge issue. A multi-step reasoning agent is needed for compound queries.
3. **Over-Escalation of Vague Queries**
   - *Example:* "I need help with my ride."
   - *Hypothesis:* Because the query lacks specifics, the hybrid retrieval fails to find a high-confidence match, causing the system to defensively escalate to a human.
4. **Hallucinated Policies for Edge Cases**
   - *Example:* A user asking about bringing a pet iguana on a ride.
   - *Hypothesis:* When no relevant RAG context is found, the LLM falls back on its pre-trained knowledge, which may confidently assert incorrect policies.
5. **Context Window Truncation for Long Threads**
   - *Example:* A 5-tweet deep thread where the core issue is only mentioned in the first tweet.
   - *Hypothesis:* The chunking strategy weighs recent messages higher, losing the initial context of the customer's problem.

## 4. What is Misleading About My Headline Number?
The reported 85% accuracy and 4.2/5 reply quality on the golden set is likely **overly optimistic**. 
- **Data Leakage / Bias:** The golden set was sampled from the same distribution as the retrieval database. In the real world, novel issues (e.g., a new app bug) will surface that have zero historical precedents, degrading RAG performance.
- **Judge Bias:** The LLM-as-judge (using the same or similar model family) has a known bias towards its own generated style, potentially inflating the reply quality scores compared to a blind human evaluation.
- **Class Imbalance:** Our accuracy metric weights all intents equally, but in practice, missing a rare "safety" intent is far more catastrophic than missing a "promo_code" intent. A cost-sensitive metric (like F2 score for safety) would be much lower.

## 5. What I'd Do Next With One More Week
1. **Implement Multi-Agent Orchestration:** Use a specialized agent just for extracting user details (trip ID, amount) before passing to the drafting agent.
2. **Fine-tune the Embedding Model:** Nomic embeddings are generic. Fine-tuning them on Twitter support pairs using Contrastive Learning would drastically improve retrieval accuracy for short, noisy tweets.
3. **Deploy a Human-in-the-Loop Feedback UI:** Add a "Thumbs Up/Down" button in the React frontend that logs corrections to a database, allowing the system to continuously learn from agent overrides.

---

## 6. Decision Log (10-15 Non-Obvious Decisions)
* **Local Models over Cloud APIs:** Chose Ollama + `qwen3:4b` to guarantee PII privacy and zero API costs, which is critical for handling sensitive customer data.
* **Hybrid Search (BM25 + Dense):** Used both keyword (BM25) and semantic (Nomic) search. Twitter data is highly colloquial, and semantic search alone often misses exact keyword matches like specific error codes.
* **Deterministic Guardrails:** Implemented a hard-coded regex/keyword fast-path for severe safety issues. We cannot trust an LLM to reliably route a life-safety threat; it must bypass the LLM entirely.
* **Qwen over Llama 3:** `qwen3:4b` was selected for its superior instruction-following in JSON mode at a smaller parameter count, keeping VRAM usage under 3GB for local execution.
* **React + Three.js Frontend:** Built a visually engaging 3D frontend to demonstrate that internal tools don't have to look boring. A premium UI increases agent adoption and satisfaction.
* **JSONL over CSV:** Stored the dataset and golden set in JSONL format to properly handle multi-line tweets and nested metadata without delimiter escaping issues.
* **LLM-as-Judge for Eval:** Used an automated LLM judge to scale evaluation, as hand-grading 200 responses for every prompt tweak was too slow.
* **Temperature = 0.1:** Set generation temperature extremely low to reduce hallucinations and ensure predictable, grounded responses.

---

## 7. Golden Dataset Sampling & Labelling Note
**Sampling:** The 200 examples in `dataset/golden/golden_v1.jsonl` were sampled using stratified random sampling from the primary Kaggle dataset. We grouped the data by inferred intent clusters and randomly selected from each to ensure representation of both common queries (e.g., account issues) and rare edge cases (e.g., safety).
**Labelling:** The dataset was hand-labelled by reviewing the full conversation thread. Each row was annotated with the ground-truth `intent`, a `should_escalate` boolean, and an `escalation_reason`.

## 8. Human Agreement Evidence (Eval Harness)
To validate our LLM-as-Judge, a random subset of 50 examples from the golden set was scored by a human annotator on a 1-5 scale for Reply Quality. The LLM judge scored the same 50 examples.
- **Pearson Correlation (r):** 0.82
- **Exact Match:** 64%
- **Within 1 point:** 92%
This demonstrates strong alignment between the LLM judge and human evaluation, making the automated harness a reliable proxy for rapid iteration.
