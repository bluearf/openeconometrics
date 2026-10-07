/** Discover external local results without repeatedly downloading history. */
import type { ExecutionRecord } from "./types.ts";

/** Follow incoming agent output while preserving an explicit history choice. */
export function incomingAgentSelection(
  previous: readonly Pick<ExecutionRecord, "id" | "source">[],
  next: readonly Pick<ExecutionRecord, "id" | "source">[],
  selectedId: string | null,
): string | null {
  const latest = next.at(-1);
  return latest?.source === "mcp" &&
    !previous.some((record) => record.id === latest.id) &&
    (selectedId === null || selectedId === previous.at(-1)?.id)
    ? latest.id
    : null;
}

export interface ResultRevisionEnvironment {
  isActive: () => boolean;
  onWake: (callback: () => void) => () => void;
  schedule: (callback: () => void, milliseconds: number) => () => void;
}

export interface ResultRevisionOptions {
  readRevision: (signal: AbortSignal) => Promise<{ revision: string }>;
  onChanged: (signal: AbortSignal) => Promise<unknown>;
  onError?: (error: unknown) => void;
}

function browserEnvironment(): ResultRevisionEnvironment {
  return {
    isActive: () =>
      document.visibilityState === "visible" && document.hasFocus(),
    onWake: (callback) => {
      window.addEventListener("focus", callback);
      document.addEventListener("visibilitychange", callback);
      return () => {
        window.removeEventListener("focus", callback);
        document.removeEventListener("visibilitychange", callback);
      };
    },
    schedule: (callback, milliseconds) => {
      const timer = window.setInterval(callback, milliseconds);
      return () => window.clearInterval(timer);
    },
  };
}

export function watchResultRevisions(
  options: ResultRevisionOptions,
  environment: ResultRevisionEnvironment = browserEnvironment(),
): { check: () => Promise<void>; dispose: () => void } {
  let disposed = false;
  let revision: string | undefined;
  let current: AbortController | null = null;

  async function check(): Promise<void> {
    if (disposed || current || !environment.isActive()) return;
    const controller = new AbortController();
    current = controller;
    try {
      const value = await options.readRevision(controller.signal);
      if (disposed || controller.signal.aborted || !environment.isActive())
        return;
      if (
        !value ||
        typeof value.revision !== "string" ||
        !value.revision.length ||
        value.revision.length > 256
      )
        throw new Error("Could not verify the saved results.");
      if (revision !== value.revision) {
        // The first successful check also closes the bootstrap/read race.
        // Failed refreshes retain the old revision so the next check retries.
        await options.onChanged(controller.signal);
        if (!disposed && !controller.signal.aborted) revision = value.revision;
      }
    } catch (error) {
      if (!disposed && !controller.signal.aborted && environment.isActive())
        options.onError?.(error);
    } finally {
      if (current === controller) current = null;
    }
  }

  const wake = () => void check();
  const stopWake = environment.onWake(wake);
  const stopTimer = environment.schedule(wake, 5_000);
  wake();
  return {
    check,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      current?.abort();
      stopWake();
      stopTimer();
    },
  };
}
