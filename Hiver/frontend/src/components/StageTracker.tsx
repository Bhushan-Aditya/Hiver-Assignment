import { motion } from "framer-motion";
import type { Stage } from "../types";

const STEPS: { key: Stage; label: string }[] = [
  { key: "retrieving", label: "Embed + retrieve" },
  { key: "generating", label: "Reason over evidence" },
  { key: "done", label: "Decision" },
];

const ORDER: Stage[] = ["idle", "retrieving", "retrieved", "generating", "done"];

export default function StageTracker({ stage, fastPath }: { stage: Stage; fastPath: boolean }) {
  if (stage === "idle") return null;
  const pos = ORDER.indexOf(stage);
  return (
    <div className="stages" role="status" aria-live="polite">
      {STEPS.map((s, i) => {
        const sPos = ORDER.indexOf(s.key === "retrieving" ? "retrieving" : s.key);
        const active = pos >= sPos;
        const current = (stage === "retrieving" || stage === "retrieved") && i === 0
          ? true
          : stage === "generating" && i === 1
          ? true
          : stage === "done" && i === 2;
        const skipped = i === 1 && fastPath;
        return (
          <div key={s.key} className={`stages__item${active ? " is-active" : ""}${skipped ? " is-skipped" : ""}`}>
            <span className="stages__dot">
              {current && !skipped && (
                <motion.span
                  className="stages__ping"
                  animate={{ scale: [1, 1.9], opacity: [0.7, 0] }}
                  transition={{ duration: 1.1, repeat: Infinity }}
                />
              )}
            </span>
            <span className="stages__label">
              {s.label}
              {skipped && <em> · skipped (fast path)</em>}
            </span>
          </div>
        );
      })}
    </div>
  );
}
