/** Explicit QA builds only. Records times/labels and never records user content. */
const enabled =
  (import.meta as ImportMeta & { env?: Record<string, string> }).env
    ?.VITE_OPENECON_QA_METRICS === "1";
type Phase = "shell" | "project" | "typing" | "file" | "run" | "suggestion";
const pending = new Map<Phase, number>();
let token: Promise<string> | undefined;
let typing: number | undefined;
function write(phase: Phase, start: number) {
  if (!enabled) return;
  const now = performance.now();
  const value = {
    phase,
    duration_ms: now - start,
    epoch_ms: performance.timeOrigin + now,
  };
  token ??= fetch("/api/desktop/session")
    .then((r) => r.json())
    .then((r) => r.token);
  void token
    .then((t) =>
      fetch("/api/desktop/qa-performance/event", {
        method: "POST",
        headers: { "X-OpenEcon-Token": t, "Content-Type": "application/json" },
        body: JSON.stringify(value),
      }),
    )
    .catch(() => {});
}
export function qaStart(phase: Phase) {
  if (enabled) pending.set(phase, performance.now());
}
export function qaPaint(phase: Phase) {
  if (!enabled) return;
  const start = pending.get(phase);
  if (start === undefined) return;
  pending.delete(phase);
  requestAnimationFrame(() => requestAnimationFrame(() => write(phase, start)));
}
export function qaShell() {
  if (!enabled) return;
  pending.set("shell", 0);
  qaPaint("shell");
}
export function qaBeforeInput() {
  if (enabled) typing = performance.now();
}
export function qaTyped() {
  if (!enabled || typing === undefined) return;
  const start = typing;
  typing = undefined;
  requestAnimationFrame(() =>
    requestAnimationFrame(() => write("typing", start)),
  );
}
