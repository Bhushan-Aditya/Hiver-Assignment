import type { Decision, EvidenceItem, Example, Health } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/api/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getExamples(): Promise<Example[]> {
  const r = await fetch(`${BASE}/api/examples`);
  if (!r.ok) throw new Error(`examples ${r.status}`);
  return r.json();
}

export interface StreamHandlers {
  onRetrieving?: () => void;
  onRetrieved?: (evidence: EvidenceItem[], cached: boolean) => void;
  onGenerating?: (model: string) => void;
  onDone?: (decision: Decision, cached: boolean) => void;
  onError?: (message: string) => void;
}

/** Consume the SSE stream from POST /api/assess/stream. */
export async function assessStream(
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/assess/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    });
  } catch (e) {
    handlers.onError?.((e as Error).message);
    return;
  }
  if (!res.ok || !res.body) {
    handlers.onError?.(`request failed (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt: any;
      try {
        evt = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      switch (evt.stage) {
        case "retrieving":
          handlers.onRetrieving?.();
          break;
        case "retrieved":
          handlers.onRetrieved?.(evt.evidence ?? [], !!evt.cached);
          break;
        case "generating":
          handlers.onGenerating?.(evt.model ?? "");
          break;
        case "done":
          handlers.onDone?.(evt.result as Decision, !!evt.cached);
          break;
      }
    }
  }
}
