import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { EvidenceItem } from "../types";

function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="ev__bar">
      <motion.span initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.5 }} />
    </div>
  );
}

export default function EvidenceList({ items }: { items: EvidenceItem[] }) {
  const [open, setOpen] = useState<number | null>(0);
  if (!items.length) return null;
  return (
    <div className="ev">
      <div className="ev__title">Retrieved precedents</div>
      {items.map((it) => {
        const expanded = open === it.rank - 1;
        return (
          <motion.div
            key={it.rank}
            layout
            className={`ev__card${expanded ? " is-open" : ""}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: (it.rank - 1) * 0.06 }}
          >
            <button className="ev__head" onClick={() => setOpen(expanded ? null : it.rank - 1)}>
              <span className="ev__rank">#{it.rank}</span>
              <span className="ev__cust">{it.customer_text}</span>
              <span className="ev__sim">{Math.round(it.similarity * 100)}%</span>
            </button>
            <Bar value={it.similarity} />
            <AnimatePresence initial={false}>
              {expanded && (
                <motion.div
                  className="ev__body"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                >
                  <p className="ev__reply">
                    <span>Historical Uber Support reply</span>
                    {it.reply_text}
                  </p>
                  <div className="ev__meta">
                    <span>reply #{it.reply_id ?? "—"}</span>
                    <span>weak label · {it.weak_intent || "none"}</span>
                    <span>lexical {it.lexical.toFixed(1)}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        );
      })}
    </div>
  );
}
