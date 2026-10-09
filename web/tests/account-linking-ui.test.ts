import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { Auth, User } from "firebase/auth";
import type {
  AccountLinkSession,
  AccountMethod,
} from "../src/account-linking.ts";
const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
for (const name of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
  "HTMLInputElement",
  "Event",
  "MouseEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
let output = ts
  .transpileModule(
    readFileSync(new URL("../src/AccountLinking.tsx", import.meta.url), "utf8"),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    },
  )
  .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "");
output = output.replace(
  /(from\s*)(["'])([^"']+)\2/g,
  (_whole, prefix, _quote, target: string) =>
    prefix +
    JSON.stringify(
      target.startsWith(".")
        ? new URL(`../src/${target.replace(/^\.\//, "")}.ts`, import.meta.url)
            .href
        : pathToFileURL(require.resolve(target)).href,
    ),
);
const { default: AccountLinking } = await import(
  "data:text/javascript;base64," + Buffer.from(output).toString("base64")
);
async function harness(
  method: AccountMethod,
  opts: {
    verifyError?: unknown;
    linkError?: unknown;
    refreshError?: boolean;
    target?: AccountMethod;
    nativeAvailable?: boolean;
  } = {},
) {
  const user = {
    uid: "retained-uid",
    email: "qa@example.test",
    providerData: [{ providerId: method }],
  } as User;
  const auth = { currentUser: user } as Auth;
  const calls: string[] = [];
  let cancelled = 0;
  const session = {
    cancel() {
      cancelled++;
    },
    async reauthenticate(value: string, password: string) {
      calls.push(`verify:${value}`);
      if (value === "password") assert.equal(password, "current-fixture");
      if (opts.verifyError) throw opts.verifyError;
      return { user };
    },
    async link(value: string, password: string) {
      calls.push(`link:${value}`);
      if (value === "password") assert.equal(password, "new-fixture");
      if (opts.linkError) throw opts.linkError;
      user.providerData.push({ providerId: value } as never);
      return { user };
    },
  } as unknown as AccountLinkSession;
  const holder = document.createElement("div");
  document.body.append(holder);
  const root = createRoot(holder);
  await React.act(async () =>
    root.render(
      React.createElement(AccountLinking, {
        auth,
        user,
        target: opts.target,
        nativeAvailable: opts.nativeAvailable,
        session,
        onLinked: async () => {
          calls.push("refresh");
          if (opts.refreshError) throw new Error("transport");
        },
      }),
    ),
  );
  const button = (text: string) => {
    const value = [
      ...holder.querySelectorAll<HTMLButtonElement>("button"),
    ].find((item) => item.textContent === text);
    assert.ok(value, text);
    return value;
  };
  const click = (text: string) => React.act(async () => button(text).click());
  const edit = (index: number, value: string) =>
    React.act(async () => {
      const input = holder.querySelectorAll<HTMLInputElement>("input")[index];
      assert.ok(input);
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  const close = async () => {
    await React.act(async () => root.unmount());
    holder.remove();
  };
  return {
    user,
    holder,
    calls,
    click,
    edit,
    close,
    cancelled: () => cancelled,
  };
}
test("password account UI cannot add Google until separate current-password verification", async () => {
  const h = await harness("password");
  try {
    assert.ok(!h.holder.textContent!.includes("Add Google"));
    await h.edit(0, "current-fixture");
    await h.click("Verify password");
    assert.deepEqual(h.calls, ["verify:password"]);
    assert.equal(h.holder.querySelector("input"), null);
    await h.click("Add Google");
    assert.deepEqual(h.calls, [
      "verify:password",
      "link:google.com",
      "refresh",
    ]);
    assert.equal(h.user.uid, "retained-uid");
    assert.match(h.holder.textContent!, /identity is unchanged/);
  } finally {
    await h.close();
  }
  assert.equal(h.cancelled(), 1);
});
test("Google account UI requires reauthentication, matching new passwords, then one link", async () => {
  const h = await harness("google.com");
  try {
    assert.equal(h.holder.querySelector("input"), null);
    await h.click("Verify with Google");
    await h.edit(0, "new-fixture");
    await h.edit(1, "different-fixture");
    await h.click("Add password");
    assert.match(h.holder.textContent!, /Passwords must match/);
    assert.deepEqual(h.calls, ["verify:google.com"]);
    await h.edit(1, "new-fixture");
    await h.click("Add password");
    assert.deepEqual(h.calls, [
      "verify:google.com",
      "link:password",
      "refresh",
    ]);
    assert.equal(h.holder.querySelector("input"), null);
  } finally {
    await h.close();
  }
});
test("reauth cancellation and provider collision retain the UID and require verification again", async () => {
  const h = await harness("google.com", {
    verifyError: { code: "auth/popup-closed-by-user", message: "private" },
  });
  try {
    await h.click("Verify with Google");
    assert.deepEqual(h.calls, ["verify:google.com"]);
    assert.match(h.holder.textContent!, /cancelled/);
    assert.ok(!h.holder.textContent!.includes("private"));
    assert.equal(h.holder.querySelector("input"), null);
  } finally {
    await h.close();
  }
  const g = await harness("password", {
    linkError: { code: "auth/credential-already-in-use", message: "private" },
  });
  try {
    await g.edit(0, "current-fixture");
    await g.click("Verify password");
    await g.click("Add Google");
    assert.match(g.holder.textContent!, /another account/);
    assert.match(g.holder.textContent!, /Verify your existing/);
    assert.deepEqual(g.calls, ["verify:password", "link:google.com"]);
    assert.equal(g.user.uid, "retained-uid");
    assert.equal(g.holder.querySelector<HTMLInputElement>("input")!.value, "");
  } finally {
    await g.close();
  }
});
test("a committed provider remains successful when project refresh fails", async () => {
  const h = await harness("password", { refreshError: true });
  try {
    await h.edit(0, "current-fixture");
    await h.click("Verify password");
    await h.click("Add Google");
    assert.match(
      h.holder.textContent!,
      /method was added, but projects could not be refreshed/,
    );
    assert.match(h.holder.textContent!, /Sign-in method added/);
    assert.deepEqual(h.calls, [
      "verify:password",
      "link:google.com",
      "refresh",
    ]);
  } finally {
    await h.close();
  }
});
test("an already linked native target still requires reauthentication before acknowledgement", async () => {
  const h = await harness("google.com", { target: "google.com" });
  try {
    assert.ok(!h.holder.textContent!.includes("Confirm linked method"));
    await h.click("Verify with Google");
    await h.click("Confirm linked method");
    assert.deepEqual(h.calls, ["verify:google.com", "refresh"]);
  } finally {
    await h.close();
  }
});

test("older native service offers no credential entry or browser linking action", async () => {
  window.__TAURI_INTERNALS__ = {};
  const h = await harness("password");
  try {
    assert.match(h.holder.textContent!, /unavailable on this server/);
    assert.equal(h.holder.querySelector("input"), null);
    assert.equal(h.holder.querySelector("button"), null);
    assert.deepEqual(h.calls, []);
  } finally {
    await h.close();
    delete window.__TAURI_INTERNALS__;
  }
});
test("native server capability opens only the explicit browser handoff control", async () => {
  window.__TAURI_INTERNALS__ = {};
  const h = await harness("google.com", { nativeAvailable: true });
  try {
    assert.equal(h.holder.querySelector("input"), null);
    assert.ok(
      h.holder
        .querySelector("button")!
        .textContent!.includes("Add password in browser"),
    );
    assert.deepEqual(h.calls, []);
  } finally {
    await h.close();
    delete window.__TAURI_INTERNALS__;
  }
});
