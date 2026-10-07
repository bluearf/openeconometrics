import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
const dom = new JSDOM("<!doctype html><body></body>", {
  url: "http://localhost",
});
for (const name of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
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
const component = readFileSync(
  new URL("../src/TeamApp.tsx", import.meta.url),
  "utf8",
);
const source =
  `import {useState,useRef,useEffect} from ${JSON.stringify(pathToFileURL(require.resolve("react")).href)};
const isDesktop=()=>true;
const Brand=()=> <div>OpenEconometrics</div>;
const Notice=({children})=> <p>{children}</p>;
const message=error=>error.message;
const sendPasswordResetEmail=async()=>{},signInWithEmailAndPassword=async()=>{};
const startGoogleSignIn=(auth,popup,progress,options)=>globalThis.__nativeScreen.start(progress,options);
` +
  component.slice(
    component.indexOf("export function AuthScreen("),
    component.indexOf("function TeamShell("),
  );
const compiled = ts
  .transpileModule(source, {
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext },
  })
  .outputText.replaceAll(
    '"react/jsx-runtime"',
    JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href),
  );
const { AuthScreen } = await import(
  "data:text/javascript;base64," + Buffer.from(compiled).toString("base64")
);
function deferred() {
  let resolve!: () => void, reject!: (error: Error) => void;
  const promise = new Promise<void>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
async function harness() {
  const attempts: Array<
    ReturnType<typeof deferred> & { signal: AbortSignal; commit: () => void }
  > = [];
  (globalThis as any).__nativeScreen = {
    start: (progress: any, options: any) => {
      const item = {
        ...deferred(),
        signal: options.signal,
        commit: options.onCommit,
      };
      attempts.push(item);
      progress("Waiting for approval");
      return item.promise;
    },
  };
  const holder = document.createElement("div");
  document.body.append(holder);
  const root = createRoot(holder);
  await React.act(async () =>
    root.render(
      React.createElement(AuthScreen, {
        auth: {},
        invitation: false,
        onSignup: async () => {},
      }),
    ),
  );
  const button = (text: string) => {
    const item = [...holder.querySelectorAll("button")].find(
      (x) => x.textContent === text,
    );
    assert.ok(item, text);
    return item;
  };
  const click = async (text: string) => {
    await React.act(async () => button(text).click());
  };
  const close = async () => {
    await React.act(async () => root.unmount());
    holder.remove();
  };
  return { attempts, holder, button, click, close };
}
test("cancel releases the form and old completion cannot release a newer attempt lock", async () => {
  const h = await harness();
  try {
    await h.click("Continue with Google");
    assert.equal(h.button("Please wait…").disabled, true);
    await h.click("Cancel Google sign-in");
    assert.equal(h.attempts[0].signal.aborted, true);
    assert.equal(h.button("Sign in").disabled, false);
    await h.click("Continue with Google");
    await React.act(async () => h.attempts[0].resolve());
    assert.equal(h.button("Waiting for Google sign-in…").disabled, true);
    assert.equal(h.attempts[1].signal.aborted, false);
    await h.click("Cancel Google sign-in");
    await React.act(async () => h.attempts[1].resolve());
  } finally {
    await h.close();
  }
});
test("switching to password recovery cancels approval; unmount and page close abort polling", async () => {
  const h = await harness();
  try {
    await h.click("Continue with Google");
    await h.click("Forgot password");
    assert.equal(h.attempts[0].signal.aborted, true);
    assert.match(h.holder.textContent!, /Reset password/);
    await h.click("← Back to sign-in");
    await h.click("Continue with Google");
    await React.act(async () => window.dispatchEvent(new Event("pagehide")));
    assert.equal(h.attempts[1].signal.aborted, true);
    await React.act(async () => h.attempts[1].resolve());
    await h.click("Continue with Google");
  } finally {
    await h.close();
  }
  assert.equal(h.attempts[2].signal.aborted, true);
});
test("a network failure permits retry; cancellation cannot interrupt the identity commit", async () => {
  const h = await harness();
  try {
    await h.click("Continue with Google");
    await React.act(async () =>
      h.attempts[0].reject(new Error("Connection lost")),
    );
    assert.match(h.holder.textContent!, /Connection lost/);
    await h.click("Continue with Google");
    await React.act(async () => h.attempts[1].commit());
    assert.equal(
      [...h.holder.querySelectorAll("button")].some(
        (x) => x.textContent === "Cancel Google sign-in",
      ),
      false,
    );
    assert.equal(h.button("Forgot password").disabled, true);
    await React.act(async () => h.attempts[1].resolve());
    assert.equal(h.button("Sign in").disabled, false);
  } finally {
    await h.close();
  }
});
