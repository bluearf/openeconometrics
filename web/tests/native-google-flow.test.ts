import assert from "node:assert/strict";
import test from "node:test";
import {
  waitForDesktopGoogleToken,
  type NativeGoogleDependencies,
} from "../src/native-google-flow.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => {
    resolve = yes;
  });
  return { promise, resolve };
}
function harness(exchange?: Promise<unknown>) {
  const paths: string[] = [];
  const grants: string[] = [];
  let count = 0;
  const deps: NativeGoogleDependencies = {
    request: async <T>(
      path: string,
      _token: string,
      options?: RequestInit,
    ): Promise<T> => {
      paths.push(path);
      assert.ok(options?.signal);
      if (path === "/api/desktop/login") {
        assert.match(
          JSON.parse(String(options?.body)).challenge,
          /^[0-9a-f]{64}$/,
        );
        return { request_id: `grant-${++count}`, code: "ABCD" } as T;
      }
      assert.match(
        JSON.parse(String(options?.body)).verifier,
        /^[0-9a-f]{64}$/,
      );
      return (
        exchange ? await exchange : { custom_token: "synthetic-token" }
      ) as T;
    },
    open: async <T>(_command: string, args?: Record<string, unknown>) => {
      grants.push(String(args?.requestId));
      return undefined as T;
    },
    intervalMs: 1,
  };
  return { deps, paths, grants };
}

test("native abort releases an in-flight exchange immediately and ignores its late credential", async () => {
  const late = deferred<unknown>();
  const h = harness(late.promise);
  const controller = new AbortController();
  const waiting = waitForDesktopGoogleToken(
    controller.signal,
    undefined,
    h.deps,
  );
  const outcome = assert.rejects(waiting, { name: "AbortError" });
  while (h.paths.length < 2)
    await new Promise((resolve) => setTimeout(resolve, 1));
  controller.abort();
  await outcome;
  late.resolve({ custom_token: "old-credential" });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(h.paths.length, 2);
});

test("cancelled grant creation cannot open a browser or start exchange polling", async () => {
  const late = deferred<unknown>();
  let opened = 0;
  let entered = false;
  const controller = new AbortController();
  const waiting = waitForDesktopGoogleToken(controller.signal, undefined, {
    request: async <T>() => {
      entered = true;
      return (await late.promise) as T;
    },
    open: async <T>() => {
      opened++;
      return undefined as T;
    },
    intervalMs: 1,
  });
  const outcome = assert.rejects(waiting, { name: "AbortError" });
  while (!entered) await new Promise((resolve) => setTimeout(resolve, 1));
  controller.abort();
  await outcome;
  late.resolve({ request_id: "old-grant", code: "OLD" });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(opened, 0);
});

test("retry follows only a new verifier and grant, while cancelled poll timers stop", async () => {
  const h = harness();
  h.deps.intervalMs = 1000;
  const first = new AbortController();
  const notices: string[] = [];
  const waiting = waitForDesktopGoogleToken(
    first.signal,
    (text) => notices.push(text),
    h.deps,
  );
  const outcome = assert.rejects(waiting, { name: "AbortError" });
  while (!h.grants.length)
    await new Promise((resolve) => setTimeout(resolve, 1));
  first.abort();
  await outcome;
  h.deps.intervalMs = 1;
  assert.equal(
    await waitForDesktopGoogleToken(
      new AbortController().signal,
      undefined,
      h.deps,
    ),
    "synthetic-token",
  );
  assert.deepEqual(h.grants, ["grant-1", "grant-2"]);
  assert.deepEqual(h.paths, [
    "/api/desktop/login",
    "/api/desktop/login",
    "/api/desktop/login/grant-2/exchange",
  ]);
  assert.match(notices[0], /Waiting for approval/);
});

test("pre-aborted attempts do no work; timeout and network errors are terminal", async () => {
  const h = harness();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    waitForDesktopGoogleToken(controller.signal, undefined, h.deps),
    { name: "AbortError" },
  );
  assert.equal(h.paths.length, 0);
  await assert.rejects(
    waitForDesktopGoogleToken(new AbortController().signal, undefined, {
      ...h.deps,
      timeoutMs: 0,
    }),
    /timed out/,
  );
  const error = new TypeError("network lost");
  await assert.rejects(
    waitForDesktopGoogleToken(new AbortController().signal, undefined, {
      ...h.deps,
      request: async () => {
        throw error;
      },
    }),
    (value) => value === error,
  );
});
