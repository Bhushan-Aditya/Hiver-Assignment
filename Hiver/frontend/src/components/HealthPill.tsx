import type { Health } from "../types";

export default function HealthPill({ health }: { health: Health | null }) {
  if (!health) {
    return (
      <div className="pill pill--muted">
        <span className="pill__dot" /> connecting…
      </div>
    );
  }
  const ok = health.ollama;
  return (
    <div className={`pill ${ok ? "pill--ok" : "pill--warn"}`} title={health.ollama_models.join(", ")}>
      <span className="pill__dot" />
      <span>{ok ? "Ollama online" : "Ollama offline"}</span>
      <span className="pill__sep" />
      <span className="pill__mono">{health.gen_model}</span>
      <span className="pill__sep" />
      <span>{health.index_size.toLocaleString()} precedents</span>
    </div>
  );
}
