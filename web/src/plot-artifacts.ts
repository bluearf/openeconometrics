import type { WorkspaceClient } from "./api.ts";

interface NetworkPlot {
  kind: string;
  title: string;
  sample_n: number;
  total_n: number;
  dropped_n: number;
  artifact?: { version: 1; id: string; bytes: number };
}

/** Capture the project client; a late fetch cannot follow a project switch. */
export async function resolvePlot<T extends NetworkPlot>(
  plot: T,
  client: WorkspaceClient | undefined,
  signal: AbortSignal,
): Promise<T> {
  if (!plot.artifact) return plot;
  const ref = plot.artifact;
  if (
    plot.kind !== "network" || ref.version !== 1 ||
    !/^[0-9a-f]{64}$/.test(ref.id) ||
    !Number.isSafeInteger(ref.bytes) || ref.bytes < 1 || ref.bytes > 128 * 1024 * 1024
  ) throw new Error("Could not verify the saved network plot.");
  if (!client) throw new Error("Open this plot in its original local project.");
  const actual = await client.request<T>(`/console/plots/${ref.id}`, { signal });
  if (signal.aborted) throw new DOMException("Request cancelled", "AbortError");
  if (
    actual.kind !== "network" || actual.artifact || actual.title !== plot.title ||
    actual.sample_n !== plot.sample_n || actual.total_n !== plot.total_n ||
    actual.dropped_n !== plot.dropped_n
  ) throw new Error("The saved network plot does not match its history entry.");
  return actual;
}
