import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate } from "node:timers/promises";
import {
  incomingAgentSelection,
  watchResultRevisions,
  type ResultRevisionEnvironment,
} from "../src/result-revision.ts";

test("incoming agent result follows the latest output but preserves an older selection", () => {
  const previous = [{ id: "older" }, { id: "latest" }];
  const incoming = { id: "agent", source: "mcp" as const };
  const next = [...previous, incoming];
  assert.equal(incomingAgentSelection(previous, next, "latest"), "agent");
  assert.equal(incomingAgentSelection(previous, next, null), "agent");
  assert.equal(incomingAgentSelection(previous, next, "older"), null);
  assert.equal(incomingAgentSelection(next, next, "agent"), null);
  assert.equal(incomingAgentSelection([], [incoming], null), "agent");
  assert.equal(
    incomingAgentSelection(previous, [...previous, { id: "python" }], null),
    null,
  );
});

function activity() {
  let active = true;
  let wake: (() => void) | null = null;
  let tick: (() => void) | null = null;
  let stopped = 0;
  const environment: ResultRevisionEnvironment = {
    isActive: () => active,
    onWake: (callback) => {
      wake = callback;
      return () => {
        wake = null;
        stopped += 1;
      };
    },
    schedule: (callback, milliseconds) => {
      assert.equal(milliseconds, 5_000);
      tick = callback;
      return () => {
        tick = null;
        stopped += 1;
      };
    },
  };
  return {
    environment,
    setActive: (value: boolean) => {
      active = value;
    },
    wake: () => wake?.(),
    tick: () => tick?.(),
    stopped: () => stopped,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

test("visible idle workspace loads history only at startup and changed revision", async () => {
  const view = activity();
  let revision = "first";
  let checks = 0;
  let refreshes = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => {
        checks += 1;
        return { revision };
      },
      onChanged: async () => {
        refreshes += 1;
      },
    },
    view.environment,
  );
  await setImmediate();
  assert.equal(refreshes, 1);
  view.tick();
  await setImmediate();
  assert.equal(checks, 2);
  assert.equal(
    refreshes,
    1,
    "unchanged revision must not download full history",
  );
  revision = "new-mcp-result";
  view.tick();
  await setImmediate();
  assert.equal(refreshes, 2);
  watcher.dispose();
});

test("hidden or unfocused workspace stays quiet and wake checks immediately", async () => {
  const view = activity();
  view.setActive(false);
  let checks = 0;
  let refreshes = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => {
        checks += 1;
        return { revision: "ready" };
      },
      onChanged: async () => {
        refreshes += 1;
      },
    },
    view.environment,
  );
  view.tick();
  view.wake();
  await setImmediate();
  assert.equal(checks, 0);
  view.setActive(true);
  view.wake();
  await setImmediate();
  assert.equal(checks, 1);
  assert.equal(refreshes, 1);
  view.setActive(false);
  view.tick();
  await setImmediate();
  assert.equal(checks, 1);
  watcher.dispose();
});

test("overlapping timer and focus events do not duplicate requests or refreshes", async () => {
  const view = activity();
  const pending = deferred<{ revision: string }>();
  const refreshing = deferred<void>();
  let checks = 0;
  let refreshes = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => {
        checks += 1;
        return pending.promise;
      },
      onChanged: async () => {
        refreshes += 1;
        return refreshing.promise;
      },
    },
    view.environment,
  );
  view.tick();
  view.wake();
  assert.equal(checks, 1);
  pending.resolve({ revision: "new" });
  await setImmediate();
  view.tick();
  view.wake();
  assert.equal(checks, 1, "keep one in-flight operation until history is read");
  assert.equal(refreshes, 1);
  refreshing.resolve();
  await setImmediate();
  watcher.dispose();
});

test("project disposal aborts in-flight read and discards its late result", async () => {
  const view = activity();
  const pending = deferred<{ revision: string }>();
  let signal: AbortSignal | undefined;
  let refreshes = 0;
  let errors = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async (value) => {
        signal = value;
        return pending.promise;
      },
      onChanged: async () => {
        refreshes += 1;
      },
      onError: () => {
        errors += 1;
      },
    },
    view.environment,
  );
  watcher.dispose();
  watcher.dispose();
  assert.equal(signal?.aborted, true);
  assert.equal(view.stopped(), 2);
  pending.resolve({ revision: "previous-project" });
  await setImmediate();
  view.tick();
  view.wake();
  await watcher.check();
  assert.equal(refreshes, 0);
  assert.equal(errors, 0);
});

test("disposing during history refresh aborts the same workspace signal", async () => {
  const view = activity();
  const pending = deferred<void>();
  let refreshSignal: AbortSignal | undefined;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => ({ revision: "new" }),
      onChanged: async (signal) => {
        refreshSignal = signal;
        return pending.promise;
      },
    },
    view.environment,
  );
  await setImmediate();
  watcher.dispose();
  assert.equal(refreshSignal?.aborted, true);
  pending.resolve();
  await setImmediate();
});

test("failed history refresh retries unchanged revision without losing a result", async () => {
  const view = activity();
  let refreshes = 0;
  let errors = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => ({ revision: "new" }),
      onChanged: async () => {
        refreshes += 1;
        if (refreshes === 1) throw new TypeError("connection lost");
      },
      onError: () => {
        errors += 1;
      },
    },
    view.environment,
  );
  await setImmediate();
  assert.equal(errors, 1);
  await watcher.check();
  assert.equal(refreshes, 2);
  await watcher.check();
  assert.equal(refreshes, 2);
  watcher.dispose();
});

test("revision read failures or invalid tokens never trigger a history load", async () => {
  const view = activity();
  let attempt = 0;
  let refreshes = 0;
  const failures: unknown[] = [];
  const watcher = watchResultRevisions(
    {
      readRevision: async () => {
        attempt += 1;
        if (attempt === 1) throw new TypeError("offline");
        return { revision: attempt === 2 ? "" : "x".repeat(257) };
      },
      onChanged: async () => {
        refreshes += 1;
      },
      onError: (failure) => {
        failures.push(failure);
      },
    },
    view.environment,
  );
  await setImmediate();
  await watcher.check();
  await watcher.check();
  assert.equal(failures.length, 3);
  assert.equal(refreshes, 0);
  watcher.dispose();
});

test("a response finishing after the window hides is checked again on wake", async () => {
  const view = activity();
  const pending = deferred<{ revision: string }>();
  let reads = 0;
  let refreshes = 0;
  const watcher = watchResultRevisions(
    {
      readRevision: async () => {
        reads += 1;
        return reads === 1 ? pending.promise : { revision: "new" };
      },
      onChanged: async () => {
        refreshes += 1;
      },
    },
    view.environment,
  );
  view.setActive(false);
  pending.resolve({ revision: "new" });
  await setImmediate();
  assert.equal(refreshes, 0);
  view.setActive(true);
  view.wake();
  await setImmediate();
  assert.equal(reads, 2);
  assert.equal(refreshes, 1);
  watcher.dispose();
});
