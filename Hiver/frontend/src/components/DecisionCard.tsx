import { motion } from "framer-motion";
import type { Decision } from "../types";
import ConfidenceMeter from "./ConfidenceMeter";

const PATH_LABEL: Record<Decision["path"], string> = {
  fast: "Fast path · no generation",
  llm: "Model reasoned",
  fallback: "Model unavailable · safe default",
};

function Timings({ t }: { t: Decision["timings_ms"] }) {
  const rows: [string, number | undefined][] = [
    ["embed", t.embed],
    ["retrieve", t.retrieve],
    ["generate", t.generate],
  ];
  return (
    <div className="timings">
      {rows.map(([k, v]) => (
        <span key={k} className="timings__chip">
          {k} <b>{v === undefined ? "—" : `${Math.round(v)}ms`}</b>
        </span>
      ))}
      <span className="timings__chip timings__chip--total">
        total <b>{Math.round(t.total ?? 0)}ms</b>
      </span>
    </div>
  );
}

export default function DecisionCard({ decision, cached }: { decision: Decision; cached: boolean }) {
  const auto = decision.action === "auto_handle";
  const tone: "auto" | "escalate" = auto ? "auto" : "escalate";
  return (
    <motion.section
      className={`decision decision--${tone}`}
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 130, damping: 18 }}
    >
      <header className="decision__head">
        <div>
          <span className="decision__eyebrow">Recommended action</span>
          <h2>{auto ? "Ready to send" : "Human review required"}</h2>
        </div>
        <span className={`badge badge--${tone}`}>{auto ? "AUTO-HANDLE" : "ESCALATE"}</span>
      </header>

      <div className="decision__facts">
        <span className="chip">intent · {decision.intent.replace(/_/g, " ")}</span>
        <span className="chip">{PATH_LABEL[decision.path]}</span>
        {cached && <span className="chip chip--ghost">cached</span>}
      </div>

      <ConfidenceMeter value={decision.confidence} tone={tone} />

      <div className={`reply reply--${tone}`}>
        <span className="reply__label">{auto ? "Suggested customer reply" : "Routing note"}</span>
        <p>
          {decision.reply ||
            "This message is routed to a trained support agent. No automated reply is sent."}
        </p>
      </div>

      <p className="decision__reason">
        <b>Why</b> {decision.reason}
      </p>

      <div className="decision__evline">
        <span className="decision__diamond">◆</span>
        Grounded on Uber Support reply&nbsp;
        <b>#{decision.evidence_reply_id ?? "not used"}</b>
        <span className="decision__ret">retrieval {Math.round(decision.retrieval_score * 100)}%</span>
      </div>

      <Timings t={decision.timings_ms} />
    </motion.section>
  );
}
