export type Stage = "idle" | "retrieving" | "retrieved" | "generating" | "done" | "error";

export interface EvidenceItem {
  rank: number;
  reply_id: string | null;
  customer_text: string;
  reply_text: string;
  weak_intent: string;
  similarity: number;
  lexical: number;
  fused: number;
}

export interface Decision {
  intent: string;
  confidence: number;
  reply: string;
  evidence_reply_id: string | null;
  action: "auto_handle" | "escalate";
  reason: string;
  path: "fast" | "llm" | "fallback";
  retrieval_score: number;
  evidence: EvidenceItem[];
  timings_ms: { embed?: number; retrieve?: number; generate?: number; total?: number };
}

export interface Health {
  status: string;
  ollama: boolean;
  ollama_models: string[];
  gen_model: string;
  embed_model: string;
  index_size: number;
  fastpath: boolean;
}

export interface Example {
  label: string;
  text: string;
  kind: "routine" | "escalate";
}
