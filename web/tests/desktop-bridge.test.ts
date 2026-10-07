import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiError } from "../src/api.ts";
import { nativeInvoke, isTransientDesktopFailure } from "../src/desktop.ts";

test("native errors preserve access denial separately from transport and local corruption", async () => {
  const previousWindow = globalThis.window;
  let fault = "OPENECON_HTTP:403";
  globalThis.window = {
    __TAURI_INTERNALS__: {}, __TAURI__: { core: { invoke: async () => { throw fault; } } },
  } as unknown as Window & typeof globalThis;
  try {
    await assert.rejects(nativeInvoke("download_project_files"), (error) =>
      error instanceof ApiError && error.status === 403 && !isTransientDesktopFailure(error));
    fault = "OPENECON_NETWORK";
    await assert.rejects(nativeInvoke("download_project_files"), (error) =>
      error instanceof TypeError && isTransientDesktopFailure(error));
    fault = "The cached checksum is invalid.";
    await assert.rejects(nativeInvoke("download_project_files"), (error) =>
      error === fault && !isTransientDesktopFailure(error));
    assert.equal(isTransientDesktopFailure({ code: "auth/user-token-expired" }), false);
    assert.equal(isTransientDesktopFailure({ code: "auth/network-request-failed" }), true);
    assert.equal(isTransientDesktopFailure(new DOMException("cancel", "AbortError")), false);
  } finally {
    if (previousWindow) globalThis.window = previousWindow;
    else delete (globalThis as { window?: unknown }).window;
  }
});
