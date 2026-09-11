import { motion } from "framer-motion";
import type { Example } from "../types";

export default function Composer({
  value,
  onChange,
  onSubmit,
  busy,
  examples,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  examples: Example[];
}) {
  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        if (!busy) onSubmit();
      }}
    >
      <label className="composer__label" htmlFor="msg">
        Customer message
      </label>
      <textarea
        id="msg"
        value={value}
        maxLength={2000}
        placeholder="e.g. My driver was 20 minutes late and I missed my flight"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !busy) onSubmit();
        }}
      />
      <div className="composer__row">
        <div className="chips">
          {examples.map((ex) => (
            <button
              type="button"
              key={ex.label}
              className={`chips__btn chips__btn--${ex.kind}`}
              onClick={() => onChange(ex.text)}
            >
              {ex.label}
            </button>
          ))}
        </div>
        <motion.button
          type="submit"
          className="composer__go"
          disabled={busy || !value.trim()}
          whileTap={{ scale: 0.96 }}
        >
          {busy ? "Assessing…" : "Assess message"}
          <span aria-hidden>→</span>
        </motion.button>
      </div>
      <p className="composer__hint">
        Runs fully local: the message is embedded, matched against historical Uber Support
        resolutions, and never leaves this machine. <kbd>⌘</kbd>/<kbd>Ctrl</kbd> + <kbd>Enter</kbd> to send.
      </p>
    </form>
  );
}
