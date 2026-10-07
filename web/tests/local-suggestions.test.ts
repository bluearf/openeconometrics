import assert from "node:assert/strict";
import test from "node:test";
import { createLocalSuggestionProvider } from "../src/local-suggestions.ts";

const request = (signal = new AbortController().signal) => ({
  prefix: "values = [1, 2]\nmean = sum(values) / ",
  suffix: "\nprint(mean)",
  language: "python" as const,
  signal,
});

test("local continuation uses only native IPC and bounded source context", async () => {
  const calls: { command: string; args?: Record<string, unknown> }[] = [];
  const provider = createLocalSuggestionProvider(
    async <T>(command: string, args?: Record<string, unknown>) => {
      calls.push({ command, args });
      return { request_id: "request-1", text: "len(values)" } as T;
    },
    () => "request-1",
  );
  assert.equal(await provider.request(request()), "len(values)");
  assert.deepEqual(calls, [
    {
      command: "suggestions_complete",
      args: {
        request: {
          request_id: "request-1",
          prefix: request().prefix,
          suffix: request().suffix,
          language: "python",
        },
      },
    },
  ]);
  provider.dispose();
  assert.equal(calls.length, 1);
});

test("aborting editor context cancels native work and discards its late response", async () => {
  let finish!: (value: unknown) => void;
  const calls: string[] = [];
  const controller = new AbortController();
  const provider = createLocalSuggestionProvider(
    <T>(command: string) => {
      calls.push(command);
      return command === "suggestions_complete"
        ? (new Promise((resolve) => {
            finish = resolve;
          }) as Promise<T>)
        : Promise.resolve(undefined as T);
    },
    () => "request-2",
  );
  const completion = provider.request(request(controller.signal));
  controller.abort();
  finish({ request_id: "request-2", text: "len(values)" });
  assert.equal(await completion, null);
  assert.deepEqual(calls, ["suggestions_complete", "suggestions_cancel"]);
});

test("disposal stops outstanding completion and prevents future native calls", async () => {
  let finish!: (value: unknown) => void;
  const calls: string[] = [];
  const provider = createLocalSuggestionProvider(
    <T>(command: string) => {
      calls.push(command);
      return command === "suggestions_complete"
        ? (new Promise((resolve) => {
            finish = resolve;
          }) as Promise<T>)
        : Promise.resolve(undefined as T);
    },
    () => "request-3",
  );
  const completion = provider.request(request());
  provider.dispose();
  finish({ request_id: "request-3", text: "len(values)" });
  assert.equal(await completion, null);
  assert.equal(await provider.request(request()), null);
  assert.deepEqual(calls, ["suggestions_complete", "suggestions_cancel"]);
});

test("multibyte source stays within native byte budgets without corrupting Unicode", async () => {
  let captured: Record<string, unknown> | undefined;
  const provider = createLocalSuggestionProvider(
    async <T>(_command: string, args?: Record<string, unknown>) => {
      captured = args;
      return { request_id: "unicode", text: "ok" } as T;
    },
    () => "unicode",
  );
  await provider.request({
    ...request(),
    prefix: "😀".repeat(5000) + "son",
    suffix: "ön" + "😀".repeat(3000),
  });
  const sent = captured!.request as { prefix: string; suffix: string };
  assert.ok(new TextEncoder().encode(sent.prefix).length <= 8192);
  assert.ok(new TextEncoder().encode(sent.suffix).length <= 2048);
  assert.ok(sent.prefix.endsWith("son"));
  assert.ok(sent.suffix.startsWith("ön"));
  assert.ok(!sent.prefix.includes("\uFFFD") && !sent.suffix.includes("\uFFFD"));
});

test("wrong request identity and unavailable worker produce no visible suggestion", async () => {
  const mismatch = createLocalSuggestionProvider(
    async <T>() => ({ request_id: "wrong", text: "unsafe" }) as T,
    () => "expected",
  );
  assert.equal(await mismatch.request(request()), null);
  const unavailable = createLocalSuggestionProvider(async () => {
    throw new Error("AI_NOT_READY");
  });
  assert.equal(await unavailable.request(request()), null);
});

test("native bridge disappearing synchronously cannot break editor cleanup", async () => {
  let finish!: (value: unknown) => void;
  const provider = createLocalSuggestionProvider(
    <T>(command: string) => {
      if (command === "suggestions_cancel") throw new Error("Bridge closed");
      return new Promise((resolve) => {
        finish = resolve;
      }) as Promise<T>;
    },
    () => "closing",
  );
  const completion = provider.request(request());
  assert.doesNotThrow(() => provider.dispose());
  finish({ request_id: "closing", text: "len(values)" });
  assert.equal(await completion, null);
});
