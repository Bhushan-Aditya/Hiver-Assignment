import { AnimatePresence, motion } from "framer-motion";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { assessStream, getExamples, getHealth } from "./api";
import Composer from "./components/Composer";
import DecisionCard from "./components/DecisionCard";
import EvidenceList from "./components/EvidenceList";
import HealthPill from "./components/HealthPill";
import StageTracker from "./components/StageTracker";
import type { Decision, EvidenceItem, Example, Health, Stage } from "./types";

const Background3D = lazy(() => import("./components/Background3D"));

const POLICY = [
  "Auto-handle only low-risk, high-confidence cases that have a close historical precedent.",
  "Always escalate safety, account access, payment, off-app cash, sensitive data, and uncertainty.",
  "Every suggestion cites the historical Uber Support reply it is grounded on.",
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [message, setMessage] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [cached, setCached] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genModel, setGenModel] = useState("");
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    getExamples().then(setExamples).catch(() => setExamples([]));
    const id = setInterval(() => getHealth().then(setHealth).catch(() => {}), 15000);
    return () => clearInterval(id);
  }, []);

  const busy = stage === "retrieving" || stage === "retrieved" || stage === "generating";

  const run = useCallback(() => {
    if (!message.trim()) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(null);
    setDecision(null);
    setEvidence([]);
    setCached(false);
    setStage("retrieving");
    assessStream(
      message.trim(),
      {
        onRetrieving: () => setStage("retrieving"),
        onRetrieved: (ev, isCached) => {
          setEvidence(ev);
          setCached(isCached);
          setStage("retrieved");
        },
        onGenerating: (m) => {
          setGenModel(m);
          setStage("generating");
        },
        onDone: (d, isCached) => {
          setDecision(d);
          setCached(isCached);
          setStage("done");
        },
        onError: (m) => {
          setError(m);
          setStage("error");
        },
      },
      ctrl.signal,
    );
  }, [message]);

  const fastPath = decision?.path === "fast" || (stage === "retrieved" && !genModel);

  return (
    <div className="app">
      <Suspense fallback={null}>
        <Background3D stage={stage} action={decision?.action} />
      </Suspense>

      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" />
          <span className="brand__name">Hiver</span>
          <span className="brand__sub">Support Console</span>
        </div>
        <HealthPill health={health} />
      </header>

      <main className="wrap">
        <section className="hero">
          <p className="hero__eyebrow reveal">Local · retrieval-augmented · evidence-first</p>
          <h1 className="reveal reveal--1">
            Make every support decision
            <br />
            <span className="hero__grad">clear, grounded, and accountable.</span>
          </h1>
          <p className="hero__lede reveal reveal--2">
            Enter a customer message. The agent retrieves semantically similar historical
            Uber Support cases, drafts a next step only when the evidence supports it, and
            routes everything risky to a human.
          </p>
        </section>

        <section className="grid">
          <div className="col col--main">
            <div className="card">
              <Composer
                value={message}
                onChange={setMessage}
                onSubmit={run}
                busy={busy}
                examples={examples}
              />
            </div>

            <StageTracker stage={stage} fastPath={fastPath} />

            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="err"
                  className="alert"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <b>Could not reach the agent.</b> {error}
                  <span> — is the backend running and Ollama online?</span>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence mode="wait">
              {decision && <DecisionCard key={decision.reason + decision.intent} decision={decision} cached={cached} />}
            </AnimatePresence>

            {evidence.length > 0 && <EvidenceList items={evidence} />}
          </div>

          <aside className="col col--side">
            <div className="card card--policy">
              <h3>Automation policy</h3>
              <ul>
                {POLICY.map((p) => (
                  <li key={p}>
                    <span className="tick">✓</span>
                    {p}
                  </li>
                ))}
              </ul>
            </div>
            <div className="card card--stat">
              <span className="stat__num">{(health?.index_size ?? 0).toLocaleString()}</span>
              <span className="stat__label">semantically indexed Uber Support precedents</span>
              <div className="stat__sub">
                hybrid dense + BM25 retrieval · {health?.embed_model ?? "nomic-embed-text"}
              </div>
            </div>
            <div className="card card--stat">
              <span className="stat__num">{health?.fastpath ? "2" : "1"}-tier</span>
              <span className="stat__label">
                confident matches answer from an approved template; the rest go to {genModel || health?.gen_model || "the local model"}
              </span>
            </div>
          </aside>
        </section>

        <footer className="foot">
          Evidence-first RAG · fully local via Ollama · built for the Hiver take-home.
        </footer>
      </main>
    </div>
  );
}
