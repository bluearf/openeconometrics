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


test("renderer native commands are registered, manifested and allowed only by the local workspace capability", async () => {
  const { readFile, readdir } = await import("node:fs/promises");
  const root = new URL("../../", import.meta.url);
  const main = await readFile(new URL("desktop/src-tauri/src/main.rs", root), "utf8");
  const manifest = await readFile(new URL("desktop/src-tauri/build.rs", root), "utf8");
  const handlers = new Set(main.match(/generate_handler!\[([^\]]+)\]/)![1].split(",").map(value => value.trim()));
  const registered = new Set([...manifest.matchAll(/"([a-z_]+)"/g)].map(match => match[1]));
  const allowed = new Set([...main.matchAll(/"allow-([a-z-]+)"/g)].map(match => match[1].replaceAll("-", "_")));
  const invoked = new Set<string>(["save_text_export"]);
  for (const name of await readdir(new URL("web/src/", root))) {
    if (!/\.tsx?$/.test(name)) continue;
    const source = await readFile(new URL("web/src/" + name, root), "utf8");
    for (const match of source.matchAll(/nativeInvoke(?:<[^>\n]+>)?\(\s*"([a-z_]+)"/g)) invoked.add(match[1]);
  }
  assert.equal(invoked.has("save_project_file"), true);
  for (const command of invoked) {
    assert.equal(handlers.has(command), true, command + " must have a native handler");
    assert.equal(registered.has(command), true, command + " must have a command manifest");
    assert.equal(allowed.has(command), true, command + " must have the trusted workspace permission");
  }
  assert.match(main, /"identifier":"local-workspace","windows":\["main"\],"local":false,"remote":\{"urls":\[format!\("\{origin\}\/\*"\)\]\}/);
});
