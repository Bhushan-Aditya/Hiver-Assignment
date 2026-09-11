import { motion } from "framer-motion";

export default function ConfidenceMeter({ value, tone }: { value: number; tone: "auto" | "escalate" }) {
  const pct = Math.round(value * 100);
  return (
    <div className="meter">
      <div className="meter__head">
        <span>Confidence</span>
        <strong>{pct}%</strong>
      </div>
      <div className="meter__track">
        <motion.div
          className={`meter__fill meter__fill--${tone}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 20 }}
        />
      </div>
    </div>
  );
}
