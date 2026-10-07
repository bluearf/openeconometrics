import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { LocalSuggestionsStatus } from "../src/InlineSuggestionsSettings.tsx";

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
  "HTMLDialogElement",
  "Event",
  "MouseEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
dom.window.HTMLDialogElement.prototype.showModal = function () {
  this.setAttribute("open", "");
};
dom.window.HTMLDialogElement.prototype.close = function () {
  this.removeAttribute("open");
};
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const urls = new Map<string, string>();
const sourceRoot = new URL("../src/", import.meta.url);
function moduleUrl(name: string): string {
  if (urls.has(name)) return urls.get(name)!;
  const file = new URL(
    `${name}${existsSync(fileURLToPath(new URL(`${name}.tsx`, sourceRoot))) ? ".tsx" : ".ts"}`,
    sourceRoot,
  );
  let output = ts
    .transpileModule(readFileSync(file, "utf8"), {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    })
    .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "");
  output = output.replace(
    /(from\s*|import\s*)(["'])([^"']+)\2/g,
    (_whole, prefix, _quote, target: string) => {
      const url = target.startsWith(".")
        ? moduleUrl(target.replace(/^\.\//, "").replace(/\.tsx?$/, ""))
        : pathToFileURL(require.resolve(target)).href;
      return prefix + JSON.stringify(url);
    },
  );
  const url =
    "data:text/javascript;base64," + Buffer.from(output).toString("base64");
  urls.set(name, url);
  return url;
}
const { default: InlineSuggestionsSettings } = await import(
  moduleUrl("InlineSuggestionsSettings")
);
const TOTAL = 531_068_128;
function status(
  patch: Partial<LocalSuggestionsStatus> = {},
): LocalSuggestionsStatus {
  return {
    enabled: false,
    installed: false,
    state: "not_installed",
    model: "qwen2.5-coder-0.5b-q8_0",
    downloaded_bytes: 0,
    total_bytes: TOTAL,
    ...patch,
  };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
type Command = (
  name: string,
  args?: Record<string, unknown>,
) => Promise<unknown>;
function setup(invoke: Command = async () => status(), desktop = true) {
  document.body.innerHTML = "<div id='app'></div>";
  const calls: { name: string; args?: Record<string, unknown> }[] = [];
  const changes: (LocalSuggestionsStatus | null)[] = [];
  const timers = new Map<number, { delay: number; callback: () => void }>();
  let nextTimer = 0;
  const originalSetTimeout = window.setTimeout;
  const originalClearTimeout = window.clearTimeout;
  window.setTimeout = ((callback: () => void, delay: number) => {
    const id = ++nextTimer;
    timers.set(id, { callback, delay });
    return id;
  }) as typeof window.setTimeout;
  window.clearTimeout = (id) => {
    timers.delete(id);
  };
  window.__TAURI_INTERNALS__ = desktop ? {} : undefined;
  window.__TAURI__ = {
    core: {
      invoke: async <T>(name: string, args?: Record<string, unknown>) => {
        calls.push({ name, args });
        return (await invoke(name, args)) as T;
      },
    },
  };
  const root = createRoot(document.querySelector("#app")!);
  const button = (label: string) =>
    [...document.querySelectorAll("button")].find(
      (item) => item.textContent?.trim() === label,
    )!;
  return {
    calls,
    changes,
    timers,
    async render(strict = false) {
      await React.act(async () => {
        const settings = React.createElement(InlineSuggestionsSettings, {
          onStatusChange: (next: LocalSuggestionsStatus | null) =>
            changes.push(next),
        });
        root.render(
          strict
            ? React.createElement(React.StrictMode, null, settings)
            : settings,
        );
      });
    },
    async click(label: string) {
      await React.act(async () => button(label).click());
    },
    async toggle() {
      await React.act(async () =>
        document
          .querySelector<HTMLInputElement>('input[type="checkbox"]')!
          .click(),
      );
    },
    async tick() {
      const [id, timer] = [...timers.entries()][0];
      assert.ok(timer, "the component scheduled a status refresh");
      timers.delete(id);
      await React.act(async () => timer.callback());
    },
    async cleanup() {
      await React.act(async () => root.unmount());
      window.setTimeout = originalSetTimeout;
      window.clearTimeout = originalClearTimeout;
      delete window.__TAURI__;
      delete window.__TAURI_INTERNALS__;
    },
  };
}

test("desktop settings read real status without downloading or enabling a model", async () => {
  const env = setup();
  try {
    await env.render();
    assert.deepEqual(
      env.calls.map((call) => call.name),
      ["suggestions_status"],
    );
    assert.equal(env.timers.size, 0);
    assert.equal(document.querySelector("dialog"), null);
    await env.click("Suggestions");
    assert.ok(
      document.querySelector("dialog[open][aria-label=\"Inline suggestions\"]"),
    );
    assert.match(document.body.textContent!, /Qwen2\.5-Coder 0\.5B/);
    assert.match(document.body.textContent!, /531 MB/);
    const toggle = document.querySelector<HTMLInputElement>(
      'input[type="checkbox"]',
    )!;
    assert.equal(toggle.checked, false);
    assert.equal(toggle.disabled, true);
    assert.equal(document.querySelector("select"), null);
    assert.equal(document.querySelector('input[type="password"]'), null);
    assert.ok(env.calls.every((call) => call.name === "suggestions_status"));
  } finally {
    await env.cleanup();
  }
});

test("web edition exposes no native suggestion control or request", async () => {
  const env = setup(async () => {
    throw new Error("unexpected native request");
  }, false);
  try {
    await env.render();
    assert.equal(document.body.textContent, "");
    assert.deepEqual(env.calls, []);
    assert.deepEqual(env.changes, []);
  } finally {
    await env.cleanup();
  }
});

test("install requires an explicit click and reports native download progress without enabling", async () => {
  const installed = status({
    installed: true,
    state: "off",
    downloaded_bytes: TOTAL,
  });
  const installation = deferred<LocalSuggestionsStatus>();
  let current = status();
  const env = setup(async (name) => {
    if (name === "suggestions_install") {
      current = status({ state: "installing", downloaded_bytes: 123_000_000 });
      return installation.promise;
    }
    return { ...current };
  });
  try {
    await env.render();
    await env.click("Suggestions");
    assert.equal(
      env.calls.some((call) => call.name === "suggestions_install"),
      false,
    );
    await env.click("Download model");
    assert.equal([...env.timers.values()][0]?.delay, 750);
    await env.tick();
    const progress = document.querySelector<HTMLProgressElement>("progress")!;
    assert.equal(progress.value, 123_000_000);
    assert.equal(progress.max, TOTAL);
    assert.equal(env.changes.at(-1)?.enabled, false);
    current = installed;
    await React.act(async () => installation.resolve(installed));
    assert.equal(document.querySelector("progress"), null);
    assert.equal(
      document.querySelector<HTMLInputElement>('input[type="checkbox"]')!
        .disabled,
      false,
    );
    assert.equal(env.timers.size, 0);
    assert.equal(
      env.calls.some((call) => call.name === "suggestions_configure"),
      false,
    );
    assert.equal(env.changes.at(-1)?.state, "off");
  } finally {
    await env.cleanup();
  }
});

test("stop cancels a pending download and ignores its late result and progress snapshot", async () => {
  const installation = deferred<LocalSuggestionsStatus>();
  const progress = deferred<LocalSuggestionsStatus>();
  let reads = 0;
  const env = setup(async (name) => {
    if (name === "suggestions_install") return installation.promise;
    if (name === "suggestions_configure") return status();
    return ++reads <= 2 ? status() : progress.promise;
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.click("Download model");
    assert.ok(
      [...document.querySelectorAll("button")].find(
        (item) => item.textContent === "Stop",
      ),
    );
    await env.tick();
    await env.click("Stop");
    assert.deepEqual(
      env.calls.find((call) => call.name === "suggestions_configure")?.args,
      { enabled: false },
    );
    assert.equal(env.changes.at(-1)?.state, "not_installed");
    assert.equal(env.timers.size, 0);
    const received = env.changes.length;
    await React.act(async () => {
      progress.resolve(
        status({ state: "installing", downloaded_bytes: 99_000_000 }),
      );
      installation.resolve(
        status({
          enabled: true,
          installed: true,
          state: "ready",
          downloaded_bytes: TOTAL,
        }),
      );
    });
    assert.equal(env.changes.length, received);
    assert.equal(env.changes.at(-1)?.enabled, false);
    assert.equal(document.querySelector("progress"), null);
    assert.equal(document.querySelector('[role="alert"]'), null);
    assert.equal(env.timers.size, 0);
  } finally {
    await env.cleanup();
  }
});

test("stop remains available while starting and cancels before the start promise finishes", async () => {
  const start = deferred<LocalSuggestionsStatus>();
  const stopped = deferred<LocalSuggestionsStatus>();
  let current = status({
    installed: true,
    state: "off",
    downloaded_bytes: TOTAL,
  });
  const env = setup(async (name, args) => {
    if (name === "suggestions_configure") {
      if (args?.enabled) {
        current = { ...current, enabled: true, state: "starting" };
        return start.promise;
      }
      return stopped.promise;
    }
    return { ...current };
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.toggle();
    await env.tick();
    assert.equal(env.changes.at(-1)?.state, "starting");
    await env.click("Stop");
    assert.equal(env.changes.at(-1), null);
    assert.match(
      document.querySelector('[role="status"]')?.textContent ?? "",
      /Stopping/,
    );
    const reads = env.calls.filter(
      (call) => call.name === "suggestions_status",
    ).length;
    await env.tick();
    assert.equal(
      env.calls.filter((call) => call.name === "suggestions_status").length,
      reads,
    );
    current = { ...current, enabled: false, state: "off" };
    await React.act(async () => stopped.resolve(current));
    const received = env.changes.length;
    await React.act(async () => start.reject("AI_CANCELLED"));
    assert.equal(env.changes.length, received);
    assert.equal(env.changes.at(-1)?.state, "off");
    assert.equal(document.querySelector('[role="alert"]'), null);
    assert.equal(env.timers.size, 0);
    assert.equal(
      document.querySelector<HTMLInputElement>('input[type="checkbox"]')!
        .disabled,
      false,
    );
  } finally {
    await env.cleanup();
  }
});

test("on and off send real native configuration, invalidate text on off intent, and stop idle polling", async () => {
  let current = status({
    installed: true,
    state: "off",
    downloaded_bytes: TOTAL,
  });
  const off = deferred<LocalSuggestionsStatus>();
  const env = setup(async (name, args) => {
    if (name === "suggestions_configure") {
      assert.equal(typeof args?.enabled, "boolean");
      if (!args!.enabled) return off.promise;
      current = { ...current, enabled: true, state: "ready" };
    }
    return { ...current };
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.toggle();
    assert.deepEqual(
      env.calls.find((call) => call.name === "suggestions_configure")?.args,
      { enabled: true },
    );
    assert.equal(env.changes.at(-1)?.state, "ready");
    assert.equal([...env.timers.values()][0]?.delay, 5_000);
    await env.toggle();
    assert.equal(env.changes.at(-1), null);
    const stopped = { ...current, enabled: false, state: "off" as const };
    current = stopped;
    await React.act(async () => off.resolve(stopped));
    assert.equal(env.changes.at(-1)?.state, "off");
    assert.equal(env.timers.size, 0);
    assert.deepEqual(
      env.calls
        .filter((call) => call.name === "suggestions_configure")
        .map((call) => call.args),
      [{ enabled: true }, { enabled: false }],
    );
  } finally {
    await env.cleanup();
  }
});

test("an active worker is rechecked when the settings window is closed", async () => {
  let current = status({
    installed: true,
    enabled: true,
    state: "ready",
    downloaded_bytes: TOTAL,
  });
  const env = setup(async () => ({ ...current }));
  try {
    await env.render();
    assert.equal(document.querySelector("dialog"), null);
    assert.equal([...env.timers.values()][0]?.delay, 5_000);
    current = {
      ...current,
      enabled: false,
      state: "error",
      error_code: "AI_ENGINE_UNAVAILABLE",
    };
    await env.tick();
    assert.equal(env.changes.at(-1)?.state, "error");
    assert.equal(env.changes.at(-1)?.enabled, false);
    assert.equal(env.timers.size, 0);
  } finally {
    await env.cleanup();
  }
});

test("loading publishes starting status before native readiness and continues polling", async () => {
  let current = status({
    installed: true,
    state: "off",
    downloaded_bytes: TOTAL,
  });
  const start = deferred<LocalSuggestionsStatus>();
  const env = setup(async (name) => {
    if (name === "suggestions_configure") {
      current = { ...current, enabled: true, state: "starting" };
      return start.promise;
    }
    return { ...current };
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.toggle();
    await env.tick();
    assert.equal(env.changes.at(-1)?.state, "starting");
    assert.match(
      document.querySelector('[role="status"]')?.textContent ?? "",
      /Starting model/,
    );
    assert.equal([...env.timers.values()][0]?.delay, 750);
    current = { ...current, state: "ready" };
    await React.act(async () => start.resolve(current));
    assert.equal(env.changes.at(-1)?.state, "ready");
    assert.equal([...env.timers.values()][0]?.delay, 5_000);
  } finally {
    await env.cleanup();
  }
});

test("a delayed status read cannot overwrite a completed off operation", async () => {
  const ready = status({
    installed: true,
    enabled: true,
    state: "ready",
    downloaded_bytes: TOTAL,
  });
  const pending = deferred<LocalSuggestionsStatus>();
  let reads = 0;
  const env = setup(async (name) => {
    if (name === "suggestions_configure")
      return { ...ready, enabled: false, state: "off" };
    return ++reads <= 2 ? { ...ready } : pending.promise;
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.tick();
    await env.toggle();
    assert.equal(env.changes.at(-1)?.state, "off");
    const received = env.changes.length;
    await React.act(async () => pending.resolve(ready));
    assert.equal(env.changes.length, received);
    assert.equal(env.changes.at(-1)?.enabled, false);
    assert.equal(env.timers.size, 0);
  } finally {
    await env.cleanup();
  }
});

test("native setup errors remain visible after refreshing real status and never expose raw details", async () => {
  const env = setup(async (name) => {
    if (name === "suggestions_install") throw "AI_NETWORK";
    return status();
  });
  try {
    await env.render();
    await env.click("Suggestions");
    await env.click("Download model");
    assert.match(
      document.querySelector('[role="alert"]')?.textContent ?? "",
      /Could not download the model/,
    );
    assert.equal(env.changes.at(-1)?.installed, false);
    assert.equal(env.timers.size, 0);
  } finally {
    await env.cleanup();
  }
});

test("malformed native readiness cannot enable suggestions and can be retried", async () => {
  let broken = true;
  const env = setup(async () =>
    broken ? { ...status(), state: "ready", enabled: true } : status(),
  );
  try {
    await env.render();
    assert.equal(env.changes.at(-1), null);
    await env.click("Suggestions");
    assert.equal(
      document.querySelector<HTMLInputElement>('input[type="checkbox"]')!
        .disabled,
      true,
    );
    assert.match(
      document.querySelector('[role="alert"]')?.textContent ?? "",
      /Could not connect to suggestions/,
    );
    broken = false;
    await env.click("Try again");
    assert.equal(env.changes.at(-1)?.state, "not_installed");
    assert.equal(document.querySelector('[role="alert"]'), null);
  } finally {
    await env.cleanup();
  }
});

test("unmount discards a late native status response and clears scheduled polling", async () => {
  const response = deferred<LocalSuggestionsStatus>();
  const env = setup(async () => response.promise);
  await env.render();
  await env.cleanup();
  await React.act(async () =>
    response.resolve(
      status({
        enabled: true,
        installed: true,
        state: "ready",
        downloaded_bytes: TOTAL,
      }),
    ),
  );
  assert.deepEqual(env.changes, []);
  assert.equal(env.timers.size, 0);
});

test("StrictMode remount refreshes status and discards the abandoned request", async () => {
  const initial = deferred<LocalSuggestionsStatus>();
  const repeated = deferred<LocalSuggestionsStatus>();
  let count = 0;
  const env = setup(async () =>
    ++count === 1 ? initial.promise : repeated.promise,
  );
  try {
    await env.render(true);
    assert.equal(env.calls.length, 2);
    await React.act(async () =>
      initial.resolve(
        status({
          enabled: true,
          installed: true,
          state: "ready",
          downloaded_bytes: TOTAL,
        }),
      ),
    );
    assert.deepEqual(env.changes, []);
    await React.act(async () => repeated.resolve(status()));
    assert.equal(env.changes.at(-1)?.state, "not_installed");
    assert.equal(env.timers.size, 0);
  } finally {
    await env.cleanup();
  }
});
