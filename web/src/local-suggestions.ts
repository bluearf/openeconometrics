import { nativeInvoke } from "./desktop.ts";
import { qaStart, qaPaint } from "./qa-performance.ts";
import type { InlineSuggestionProvider } from "./editor-inline.ts";

type NativeInvoke = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;

export interface LocalSuggestionProvider extends InlineSuggestionProvider {
  cancelAll(): void;
  dispose(): void;
}

function boundedUtf8(text: string, bytes: number, fromEnd: boolean) {
  const encoder = new TextEncoder();
  if (encoder.encode(text).length <= bytes) return text;
  const characters = Array.from(text);
  let count = 0;
  const kept: string[] = [];
  for (const char of fromEnd ? characters.reverse() : characters) {
    count += encoder.encode(char).length;
    if (count > bytes) break;
    kept.push(char);
  }
  return (fromEnd ? kept.reverse() : kept).join("");
}

/** Only the native local worker receives this bounded editor context. */
export function createLocalSuggestionProvider(
  invoke: NativeInvoke = nativeInvoke,
  identifier: () => string = () => crypto.randomUUID().replaceAll("-", ""),
): LocalSuggestionProvider {
  let disposed = false;
  const pending = new Map<string, () => void>();
  const cancelAll = () => {
    for (const cancel of [...pending.values()]) cancel();
  };
  return {
    cancelAll,
    dispose() {
      disposed = true;
      cancelAll();
    },
    async request({ prefix, suffix, language, signal }) {
      if (disposed || signal.aborted) return null;
      cancelAll();
      const request_id = identifier();
      let cancelled = false;
      const cancel = () => {
        if (cancelled) return;
        cancelled = true;
        pending.delete(request_id);
        try {
          void invoke("suggestions_cancel", { requestId: request_id }).catch(
            () => {},
          );
        } catch {
          // A closing native bridge must not interrupt React/editor cleanup.
        }
      };
      pending.set(request_id, cancel);
      signal.addEventListener("abort", cancel, { once: true });
      try {
        qaStart("suggestion");
        const result = await invoke<{ request_id: string; text: string }>(
          "suggestions_complete",
          {
            request: {
              request_id,
              prefix: boundedUtf8(prefix, 8192, true),
              suffix: boundedUtf8(suffix, 2048, false),
              language,
            },
          },
        );
        if (
          disposed ||
          cancelled ||
          signal.aborted ||
          result?.request_id !== request_id ||
          typeof result.text !== "string"
        )
          return null;
        qaPaint("suggestion");
        return result.text;
      } catch {
        return null;
      } finally {
        pending.delete(request_id);
        signal.removeEventListener("abort", cancel);
      }
    },
  };
}
