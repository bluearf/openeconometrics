import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type {
  ScriptFile,
  ScriptConflictSnapshot,
  ScriptSyncNotification,
  WorkspaceClient,
} from "../src/api.ts";
import type { DatasetProfile } from "../src/types.ts";
import type { FileLayout } from "../src/file-layout.ts";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
for (const name of [
  "window",
  "document",
  "navigator",
  "Element",
  "Node",
  "HTMLElement",
  "HTMLInputElement",
  "HTMLTextAreaElement",
  "Event",
  "MouseEvent",
  "KeyboardEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, {
  IS_REACT_ACT_ENVIRONMENT: true,
  requestAnimationFrame: (callback: () => void) => setTimeout(callback, 0),
});
dom.window.matchMedia = () => ({ matches: false }) as MediaQueryList;
dom.window.HTMLAnchorElement.prototype.click = function () {};
dom.window.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
dom.window.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const urls = new Map<string, string>();
const sourceRoot = new URL("../src/", import.meta.url);

// The real App owns switching and persistence. This editor adapter avoids
// CodeMirror's browser layout dependency while retaining its public contract.
const editorSource = `import {forwardRef,useImperativeHandle,useRef} from "react";
export default forwardRef(function Editor(props,ref){
 const input=useRef(null);
 useImperativeHandle(ref,()=>({getCode:()=>input.current.value,
 getSelectionOrLine:()=>input.current.value,focus:()=>input.current.focus(),
 cancelInlineSuggestion:()=>{},
 insert:text=>{if(!props.readOnly)props.onChange(input.current.value+text);}}));
 return <textarea aria-label="Test file editor" data-language={props.language} ref={input} value={props.value}
 readOnly={props.readOnly} onChange={event=>{if(!props.readOnly)props.onChange(event.target.value);}}/>;
});`;
function moduleUrl(name: string): string {
  if (urls.has(name)) return urls.get(name)!;
  const stub =
    name === "CodeEditor"
      ? editorSource
      : name === "WorkspaceTerminal"
        ? 'export default function Stub(props){return <button disabled={props.readOnly} onClick={()=>props.onRun("print(41)")}>Test Python terminal</button>;}'
        : ["PlotView", "LatexView", "PackageManager"].includes(name)
          ? "export default function Stub(){return null;}"
          : null;
  const file = new URL(
    `${name}${existsSync(fileURLToPath(new URL(`${name}.tsx`, sourceRoot))) ? ".tsx" : ".ts"}`,
    sourceRoot,
  );
  let output = ts
    .transpileModule(stub ?? readFileSync(file, "utf8"), {
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
const { default: App } = await import(moduleUrl("App"));
const runtime = await import(moduleUrl("api"));
const first = "a".repeat(32),
  second = "b".repeat(32);
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (failure: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function setup(
  options: {
    readOnly?: boolean;
    preferred?: string;
    onSave?: (path: string, code: string) => Promise<void>;
    onRead?: (id: string, options: RequestInit) => Promise<void>;
    onLayoutSave?: () => Promise<void>;
    onCreate?: () => Promise<void>;
    onLayoutRead?: () => Promise<void>;
    pendingLayout?: boolean;
    onConflictRead?: (id: string) => Promise<ScriptConflictSnapshot>;
    onConflictResolve?: (
      snapshot: ScriptConflictSnapshot,
      code: string,
    ) => Promise<ScriptFile>;
    datasets?: DatasetProfile[];
  } = {},
) {
  document.body.innerHTML = "<div id='app'></div>";
  window.localStorage.clear();
  if (options.preferred)
    window.localStorage.setItem(
      "openecon:active-script:local",
      options.preferred,
    );
  const docs = new Map<string, ScriptFile>([
    [
      "analysis",
      { id: "analysis", name: "analysis.py", code: "main = 1", version: 1 },
    ],
    [first, { id: first, name: "first.py", code: "first = 1", version: 0 }],
    [second, { id: second, name: "untitled.py", code: "", version: 0 }],
  ]);
  const calls: {
    path: string;
    method: string;
    body?: Record<string, unknown>;
  }[] = [];
  const layoutListeners = new Set<(value: FileLayout) => void>();
  const scriptListeners = new Set<(value: ScriptSyncNotification) => void>();
  let controls: { prepareToLeave: () => Promise<void> } | null = null;
  let layout = {
    version: 0,
    entries: [
      ...[...docs.values()].map(({ id, name }) => ({
        kind: "script",
        id,
        name,
        parent: null as string | null,
      })),
      ...(options.datasets ?? []).map(({ id, name }) => ({
        kind: "dataset",
        id,
        name,
        parent: null as string | null,
      })),
    ],
  };
  const client: WorkspaceClient = {
    projectId: null,
    teams: false,
    readScriptConflict: options.onConflictRead,
    resolveScriptConflict: options.onConflictResolve,
    connect: async () => ({
      token: "",
      version: "test",
      environment: "local",
      read_only: options.readOnly,
    }),
    cancelPending() {},
    subscribeScriptSync(listener) {
      scriptListeners.add(listener);
      return () => scriptListeners.delete(listener);
    },
    subscribeFileLayout(listener) {
      layoutListeners.add(listener);
      return () => layoutListeners.delete(listener);
    },
    request: async <T>(path: string, init: RequestInit = {}) => {
      const method = init.method ?? "GET";
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ path, method, body });
      if (path === "/files/layout") {
        if (method === "GET") await options.onLayoutRead?.();
        const known = new Set(layout.entries.map((item) => item.id));
        for (const file of docs.values())
          if (!known.has(file.id))
            layout.entries.push({
              kind: "script",
              id: file.id,
              name: file.name,
              parent: null,
            });
        if (method === "PUT") {
          await options.onLayoutSave?.();
          if (body.version !== layout.version)
            throw new runtime.ApiError(
              "Düzen değişti",
              409,
              "VERSION_CONFLICT",
            );
          layout = {
            version: layout.version + 1,
            entries: structuredClone(body.entries),
          };
        }
        return {
          ...structuredClone(layout),
          pending_sync: options.pendingLayout,
        } as T;
      }
      if (path === "/console")
        return {
          history: [],
          variables: [],
          status: { running: false, session_generation: 0 },
        } as T;
      if (path === "/datasets")
        return { datasets: options.datasets ?? [] } as T;
      if (path === "/console/script") return docs.get("analysis") as T;
      if (path === "/console/execute")
        return {
          id: "run-test",
          code: body.code,
          status: "ok",
          outputs: [],
          stdout: "41",
          variables: [],
        } as T;
      if (path === "/console/scripts") {
        if (method === "POST") {
          await options.onCreate?.();
          const created = {
            id: "c".repeat(32),
            name: String(body.name).replace(/(\.(?:py|md|tex))$/, "_2$1"),
            code: body.code,
            version: 0,
          } as ScriptFile;
          docs.set(created.id, created);
          return created as T;
        }
        return {
          scripts: [...docs.values()].map(({ id, name, version }) => ({
            id,
            name,
            version,
          })),
        } as T;
      }
      if (path.startsWith("/console/scripts/")) {
        const id = path.split("/").at(-1)!;
        if (method === "PUT") {
          await options.onSave?.(path, body.code);
          const old = docs.get(id)!;
          const updated = {
            ...old,
            code: body.code as string,
            version: old.version + 1,
          };
          docs.set(id, updated);
          return updated as T;
        }
        await options.onRead?.(id, init);
        return docs.get(id) as T;
      }
      throw new Error(`Unexpected isolated test request ${method} ${path}`);
    },
  };
  const root = createRoot(document.getElementById("app")!);
  const editor = () => document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Test file editor"]')!;
  const title = () => document.querySelector(".file-tab")?.textContent;
  const button = (label: string) =>
    [...document.querySelectorAll<HTMLButtonElement>("button")].find(
      (b) =>
        b.getAttribute("aria-label") === label ||
        b.textContent?.trim() === label,
    )!;
  const click = async (label: string) =>
    React.act(() => {
      button(label).dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  const contextMenu = async (label: string) =>
    React.act(() => {
      const row = button(label)?.closest(".files-sidebar__row");
      assert.ok(row, `file row ${label} exists`);
      row.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          button: 2,
          clientX: 80,
          clientY: 60,
        }),
      );
    });
  const doubleClick = async (label: string) =>
    React.act(() => {
      const file = button(label);
      assert.ok(file, `file ${label} exists`);
      file.dispatchEvent(
        new MouseEvent("dblclick", {
          bubbles: true,
          cancelable: true,
          button: 0,
          detail: 2,
        }),
      );
    });
  const edit = async (code: string) =>
    React.act(() => {
      Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )!.set!.call(editor(), code);
      editor().dispatchEvent(new Event("input", { bubbles: true }));
    });
  const render = async () =>
    React.act(() =>
      root.render(
        React.createElement(App, {
          client,
          readOnly: options.readOnly,
          onControls: (value: typeof controls) => {
            controls = value;
          },
        }),
      ),
    );
  const cleanup = async () => {
    await React.act(() => root.unmount());
    document.body.innerHTML = "";
  };
  return {
    getLayout: () => structuredClone(layout),
    docs,
    calls,
    root,
    client,
    editor,
    title,
    button,
    click,
    contextMenu,
    doubleClick,
    edit,
    render,
    cleanup,
    controls: () => controls,
    publishLayout: async (value: FileLayout) =>
      React.act(() => {
        for (const listener of layoutListeners) listener(value);
      }),
    layoutSubscriberCount: () => layoutListeners.size,
    publishScript: async (value: ScriptSyncNotification) =>
      React.act(() => {
        for (const listener of scriptListeners) listener(value);
      }),
    scriptSubscriberCount: () => scriptListeners.size,
  };
}

test("App bootstraps the selected empty named script without inserting starter text or writing", async () => {
  const env = setup({ preferred: second });
  try {
    await env.render();
    assert.equal(env.title(), "untitled.py");
    assert.equal(env.editor().value, "");
    assert.equal(
      env.calls.some((c) => c.method !== "GET"),
      false,
    );
  } finally {
    await env.cleanup();
  }
});

test("App waits for outgoing file save, freezes editing, and opens only one selected file", async () => {
  const save = deferred<void>();
  const env = setup({ onSave: async () => save.promise });
  try {
    await env.render();
    await env.edit("important = 123");
    await env.click("first.py");
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.title(), "analysis.py");
    assert.equal(
      env.calls.some((c) => c.path.endsWith(first)),
      false,
    );
    await env.click("untitled.py");
    await React.act(() => save.resolve());
    assert.equal(env.docs.get("analysis")!.code, "important = 123");
    assert.equal(env.title(), "first.py");
    assert.equal(env.editor().value, "first = 1");
    assert.equal(env.calls.filter((c) => c.path.endsWith(second)).length, 0);
  } finally {
    save.resolve();
    await env.cleanup();
  }
});

test("App failed outgoing save keeps the exact local editor and does not read or create another file", async () => {
  const env = setup({
    onSave: async () => {
      throw new Error("owned save failed");
    },
  });
  try {
    await env.render();
    await env.edit("local_unsaved = 7");
    await env.click("first.py");
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().value, "local_unsaved = 7");
    assert.equal(env.editor().readOnly, false);
    assert.equal(
      env.calls.some((c) => c.path.endsWith(first)),
      false,
    );
    assert.ok(
      document
        .querySelector('[role="alert"]')
        ?.textContent?.includes("owned save failed"),
    );
  } finally {
    await env.cleanup();
  }
});

test("App creates a genuinely empty new script after preserving the outgoing document", async () => {
  const env = setup();
  try {
    await env.render();
    await env.edit("old_file = 42");
    await env.click("New file");
    await fillName("new.py");
    await submitName();
    assert.equal(env.docs.get("analysis")!.code, "old_file = 42");
    const creation = env.calls.find((c) => c.method === "POST")!;
    assert.deepEqual(creation.body, { name: "new.py", code: "" });
    assert.equal(env.title(), "new.py");
    assert.equal(env.editor().value, "");
    assert.equal(
      window.localStorage.getItem("openecon:active-script:local"),
      "c".repeat(32),
    );
  } finally {
    await env.cleanup();
  }
});

test("App updates background script acknowledgements without replacing a newer typed draft", async () => {
  const env = setup();
  try {
    await env.render();
    assert.equal(env.scriptSubscriberCount(), 1);
    await env.publishScript({
      file: { ...env.docs.get("analysis")!, pending_sync: true },
    });
    assert.equal(
      document.querySelector(".save-state")?.textContent,
      "Saved locally",
    );
    await env.publishScript({
      file: { ...env.docs.get("analysis")!, pending_sync: false },
    });
    assert.equal(document.querySelector(".save-state")?.textContent, "Saved");
    await env.edit("new_local_draft = 42");
    await env.publishScript({
      file: { ...env.docs.get("analysis")!, pending_sync: false },
    });
    assert.equal(env.editor().value, "new_local_draft = 42");
    assert.equal(document.querySelector(".save-state")?.textContent, "Saving");
  } finally {
    await env.cleanup();
  }
  assert.equal(env.scriptSubscriberCount(), 0);
});

test("a newer teammate source locks the active draft for explicit reload without replacing typed text", async () => {
  const env = setup();
  try {
    await env.render();
    await env.edit("keep_my_typed_work = 42");
    const refreshed = {
      ...env.docs.get("analysis")!,
      code: "teammate_update = 100",
      version: 2,
    };
    env.docs.set("analysis", refreshed);
    await env.publishScript({ file: refreshed, source_changed: true });
    assert.equal(env.editor().value, "keep_my_typed_work = 42");
    assert.ok(document.querySelector(".draft-conflict"));
    assert.equal(
      document.querySelector(".save-state")?.textContent,
      "Version conflict",
    );
    await env.click("first.py");
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.calls.filter((call) => call.method === "PUT").length, 0);
    await env.click("Download local copy and open latest draft");
    assert.equal(env.editor().value, "teammate_update = 100");
    assert.equal(document.querySelector(".draft-conflict"), null);
    await env.publishScript({
      file: { ...refreshed, version: 1 },
      source_changed: true,
    });
    assert.equal(
      document.querySelector(".draft-conflict"),
      null,
      "an older source event cannot relock a freshly reloaded draft",
    );
    assert.equal(env.editor().value, "teammate_update = 100");
  } finally {
    await env.cleanup();
  }
});

test("a background source event arriving during file open guards the stale incoming snapshot", async () => {
  const read = deferred<void>();
  let refresh = false;
  const env = setup({
    onRead: async (id) => {
      if (id === first && refresh) await read.promise;
    },
  });
  try {
    await env.render();
    refresh = true;
    await env.click("first.py");
    assert.equal(env.editor().readOnly, true);
    await env.publishScript({
      file: { ...env.docs.get(first)!, code: "peer_source = 200", version: 1 },
      source_changed: true,
    });
    assert.equal(
      document.querySelector(".draft-conflict"),
      null,
      "an inactive file must not interrupt the outgoing editor",
    );
    await React.act(() => read.resolve());
    assert.equal(env.title(), "first.py");
    assert.equal(env.editor().value, "first = 1");
    assert.ok(
      document.querySelector(".draft-conflict"),
      "a completed old-version open must not silently create a stale writer",
    );
    assert.equal(env.calls.filter((call) => call.method === "PUT").length, 0);
  } finally {
    read.resolve();
    await env.cleanup();
  }
});

test("App presents an active background script conflict and preserves unsaved source", async () => {
  const env = setup();
  try {
    await env.render();
    await env.edit("local_draft = 99");
    await env.publishScript({
      file: {
        ...env.docs.get("analysis")!,
        pending_sync: true,
        sync_conflict: true,
      },
      sync_conflict: true,
      sync_error: {
        message: "cloud version changed",
        status: 409,
        code: "VERSION_CONFLICT",
      },
    });
    assert.ok(document.querySelector(".draft-conflict"));
    assert.equal(
      document.querySelector(".save-state")?.textContent,
      "Version conflict",
    );
    await env.click("first.py");
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().value, "local_draft = 99");
    assert.equal(env.calls.filter((call) => call.method === "PUT").length, 0);
  } finally {
    await env.cleanup();
  }
});

test("opening a locally cached conflicted file restores the conflict flow before further writes", async () => {
  const env = setup();
  env.docs.set(first, {
    ...env.docs.get(first)!,
    sync_conflict: true,
    pending_sync: true,
  });
  try {
    await env.render();
    await env.publishScript({
      file: env.docs.get(first)!,
      sync_conflict: true,
    });
    assert.equal(
      document.querySelector(".draft-conflict"),
      null,
      "an inactive file does not lock the current one",
    );
    await env.click("first.py");
    assert.equal(env.title(), "first.py");
    assert.ok(document.querySelector(".draft-conflict"));
    assert.equal(env.editor().value, "first = 1");
    assert.equal(
      document.querySelector(".save-state")?.textContent,
      "Version conflict",
    );
    await env.click("untitled.py");
    assert.equal(env.title(), "first.py");
    assert.equal(env.calls.filter((call) => call.method === "PUT").length, 0);
  } finally {
    await env.cleanup();
  }
});

test("background script authorization denial locks all editing without losing local text", async () => {
  const env = setup();
  try {
    await env.render();
    await env.edit("keep_this_source = 7");
    await env.publishScript({
      sync_error: {
        message: "project access revoked",
        status: 403,
        code: "ROLE_REQUIRED",
      },
    });
    assert.equal(env.editor().value, "keep_this_source = 7");
    assert.equal(env.editor().readOnly, true);
    assert.equal(
      document.querySelector(".save-state")?.textContent,
      "Read-only",
    );
    assert.ok(
      document
        .querySelector('[role="alert"]')
        ?.textContent?.includes("project access revoked"),
    );
    assert.equal(env.calls.filter((call) => call.method === "PUT").length, 0);
  } finally {
    await env.cleanup();
  }
});

test("App viewer selects named files without save or create writes", async () => {
  const env = setup({ readOnly: true });
  try {
    await env.render();
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.button("New file"), undefined);
    await env.click("untitled.py");
    assert.equal(env.title(), "untitled.py");
    assert.equal(env.editor().value, "");
    assert.equal(
      env.calls.some((c) => c.method !== "GET"),
      false,
    );
  } finally {
    await env.cleanup();
  }
});

test("App conflict reload locks editor and competing selections until its current file returns", async () => {
  const read = deferred<void>();
  let blockRead = false;
  let rejectSave = true;
  const env = setup({
    onSave: async () => {
      if (rejectSave) throw new runtime.ApiError("owned conflict", 409);
    },
    onRead: async (id, options) => {
      if (id === "analysis" && blockRead) {
        assert.equal(
          new Headers(options.headers).get("X-OpenEcon-Resolve-Conflict"),
          "remote",
        );
        await read.promise;
      }
    },
  });
  try {
    await env.render();
    await env.edit("my_local_copy = 99");
    await env.click("first.py");
    assert.ok(document.querySelector(".draft-conflict"));
    await env.click("untitled.py");
    assert.equal(
      env.calls.some((c) => c.path.endsWith(second)),
      false,
    );
    blockRead = true;
    env.docs.set("analysis", {
      id: "analysis",
      name: "analysis.py",
      code: "remote_copy = 100",
      version: 8,
    });
    await env.click("Download local copy and open latest draft");
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.editor().value, "my_local_copy = 99");
    await env.click("first.py");
    assert.equal(
      env.calls.some((c) => c.path.endsWith(first)),
      false,
    );
    await React.act(() => read.resolve());
    assert.equal(env.editor().value, "remote_copy = 100");
    assert.equal(env.editor().readOnly, false);
    assert.equal(document.querySelector(".draft-conflict"), null);
    rejectSave = false;
    await env.edit("after_reload = 101");
    await env.click("first.py");
    assert.deepEqual(
      env.calls.filter((call) => call.method === "PUT").at(-1)?.body,
      { code: "after_reload = 101", version: 8 },
    );
    assert.equal(env.docs.get("analysis")!.code, "after_reload = 101");
  } finally {
    read.resolve();
    await env.cleanup();
  }
});

test.after(() => dom.window.close());

async function fillName(value: string) {
  await React.act(() => {
    const input = document.querySelector<HTMLInputElement>(
      'input[aria-label="New file name"],input[aria-label="Folder name"],input[aria-label="File name"]',
    )!;
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function submitName() {
  await React.act(() => {
    const input = document.querySelector<HTMLInputElement>(
      'input[aria-label="New file name"],input[aria-label="Folder name"],input[aria-label="File name"]',
    )!;
    input.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        bubbles: true,
        cancelable: true,
      }),
    );
  });
}

test("App double-click rename cancels on Escape and persists on Enter without changing source", async () => {
  const env = setup();
  try {
    await env.render();
    assert.equal(env.button("analysis.py options"), undefined);
    await env.click("analysis.py");
    await env.doubleClick("analysis.py");
    await React.act(
      () =>
        new Promise<void>((resolve) => requestAnimationFrame(() => resolve())),
    );
    const renameInput = document.querySelector<HTMLInputElement>(
      'input[aria-label="New file name"]',
    );
    assert.ok(renameInput);
    assert.equal(
      document.activeElement,
      renameInput,
      "the active-file click cannot steal focus from the following rename",
    );
    assert.equal(
      document.querySelector<HTMLInputElement>(
        'input[aria-label="New file name"]',
      )?.value,
      "analysis.py",
    );
    await fillName("cancelled.py");
    await React.act(() => {
      const input = document.querySelector<HTMLInputElement>(
        'input[aria-label="New file name"]',
      );
      assert.ok(input);
      input.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          bubbles: true,
          cancelable: true,
        }),
      );
    });
    assert.equal(
      document.querySelector('input[aria-label="New file name"]'),
      null,
    );
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().value, "main = 1");
    assert.equal(
      env.getLayout().entries.find((entry) => entry.id === "analysis")!.name,
      "analysis.py",
    );
    assert.equal(
      env.calls.filter(
        (call) => call.path === "/files/layout" && call.method === "PUT",
      ).length,
      0,
      "Escape cannot persist the cancelled filename",
    );
    await env.doubleClick("analysis.py");
    await fillName("paper.py");
    await submitName();
    assert.equal(env.title(), "paper.py");
    assert.equal(env.editor().value, "main = 1");
    assert.equal(env.docs.get("analysis")!.name, "analysis.py");
    assert.equal(env.docs.get("analysis")!.version, 1);
    assert.equal(
      env.getLayout().entries.find((entry) => entry.id === "analysis")!.name,
      "paper.py",
    );
    assert.equal(
      env.calls.filter(
        (call) => call.path === "/files/layout" && call.method === "PUT",
      ).length,
      1,
    );
    assert.equal(
      env.calls.some(
        (call) =>
          call.path.startsWith("/console/scripts") && call.method === "PUT",
      ),
      false,
      "renaming only changes the file layout, preserving source and identity",
    );
  } finally {
    await env.cleanup();
  }
});

test("App queues double-click rename during a delayed file open without reopening or changing source", async () => {
  const opening = deferred<void>();
  const env = setup({
    onRead: async (id) => {
      if (id === first) await opening.promise;
    },
  });
  try {
    await env.render();
    const originalLayout = env.getLayout();
    const originalSource = { ...env.docs.get(first)! };
    await env.click("first.py");
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.button("first.py").disabled, false);
    assert.equal(env.button("first.py").getAttribute("aria-disabled"), "true");
    await env.doubleClick("first.py");
    assert.equal(
      document.querySelector('input[aria-label="New file name"]'),
      null,
    );
    assert.equal(
      env.calls.filter(
        (call) =>
          call.path === `/console/scripts/${first}` && call.method === "GET",
      ).length,
      1,
      "the second gesture cannot restart the pending source load",
    );
    assert.equal(
      env.calls.some((call) => call.method !== "GET"),
      false,
    );
    await React.act(() => opening.resolve());
    await React.act(
      () =>
        new Promise<void>((resolve) => requestAnimationFrame(() => resolve())),
    );
    assert.equal(env.title(), "first.py");
    assert.equal(env.editor().value, "first = 1");
    const input = document.querySelector<HTMLInputElement>(
      'input[aria-label="New file name"]',
    );
    assert.ok(
      input,
      "the queued rename opens only after the selected source is ready",
    );
    assert.equal(input.value, "first.py");
    assert.equal(
      document.activeElement,
      input,
      "late editor focus cannot cancel the queued rename",
    );
    await fillName("cancelled.py");
    await React.act(() => {
      input.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          bubbles: true,
          cancelable: true,
        }),
      );
    });
    assert.equal(
      document.querySelector('input[aria-label="New file name"]'),
      null,
    );
    assert.equal(env.title(), "first.py");
    assert.equal(env.editor().value, "first = 1");
    assert.deepEqual(env.docs.get(first), originalSource);
    assert.deepEqual(env.getLayout(), originalLayout);
    assert.equal(
      env.calls.some((call) => call.method !== "GET"),
      false,
    );
    assert.equal(
      env.calls.filter(
        (call) =>
          call.path === `/console/scripts/${first}` && call.method === "GET",
      ).length,
      1,
    );
  } finally {
    opening.resolve();
    await env.cleanup();
  }
});

test("App renames the active file without waiting for an in-flight code save or changing its identity", async () => {
  const save = deferred<void>();
  const env = setup({ onSave: async () => save.promise });
  try {
    await env.render();
    await env.edit("unsaved_important = 42");
    await React.act(
      () => new Promise<void>((resolve) => setTimeout(resolve, 700)),
    );
    assert.ok(
      env.calls.some(
        (c) => c.path === "/console/scripts/analysis" && c.method === "PUT",
      ),
    );
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("paper.py");
    await submitName();
    assert.equal(
      env.calls.some((c) => c.path === "/files/layout" && c.method === "PUT"),
      true,
    );
    assert.equal(env.title(), "paper.py");
    assert.equal(env.editor().readOnly, false);
    assert.equal(env.docs.get("analysis")!.code, "main = 1");
    assert.equal(
      document.querySelector('input[aria-label="New file name"]'),
      null,
    );
    await React.act(() => save.resolve());
    assert.equal(env.docs.get("analysis")!.code, "unsaved_important = 42");
    assert.equal(env.title(), "paper.py");
    assert.equal(env.editor().value, "unsaved_important = 42");
    assert.equal(env.docs.get("analysis")!.name, "analysis.py");
    assert.equal(
      env.getLayout().entries.find((e) => e.id === "analysis")!.name,
      "paper.py",
    );
  } finally {
    save.resolve();
    await env.cleanup();
  }
});

test("App publishes a rename immediately and keeps typing and execution available until local acceptance", async () => {
  const local = deferred<void>();
  const env = setup({ onLayoutSave: () => local.promise });
  try {
    await env.render();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("smooth.py");
    await submitName();
    assert.equal(env.title(), "smooth.py");
    assert.equal(
      document.querySelector('input[aria-label="New file name"]'),
      null,
    );
    assert.equal(env.editor().readOnly, false);
    assert.equal(env.button("Run").disabled, false);
    assert.equal(env.button("Run selection").disabled, false);
    await env.edit("typed_during_rename = 99");
    assert.equal(env.editor().value, "typed_during_rename = 99");
    let left = false;
    const leave = env
      .controls()!
      .prepareToLeave()
      .then(() => {
        left = true;
      });
    await React.act(async () => {
      await Promise.resolve();
    });
    assert.equal(
      left,
      false,
      "Leaving waits only for outstanding durable local acceptance",
    );
    await React.act(() => local.resolve());
    await React.act(() => leave);
    assert.equal(env.docs.get("analysis")!.code, "typed_during_rename = 99");
    assert.equal(env.getLayout().entries[0].name, "smooth.py");
  } finally {
    local.resolve();
    await env.cleanup();
  }
});

test("App preserves a fast background acknowledgement and ignores older layout notifications", async () => {
  const local = deferred<void>();
  const env = setup({ onLayoutSave: () => local.promise, pendingLayout: true });
  try {
    await env.render();
    const stale = env.getLayout();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("acknowledged.py");
    await submitName();
    await env.publishLayout({
      version: 1,
      entries: stale.entries.map((entry) =>
        entry.id === "analysis" ? { ...entry, name: "acknowledged.py" } : entry,
      ),
      pending_sync: false,
    } as FileLayout);
    assert.equal(env.title(), "acknowledged.py");
    await React.act(() => local.resolve());
    assert.equal(
      document
        .querySelector('[role="status"]')
        ?.textContent?.includes("locally"),
      false,
    );
    assert.equal(
      document
        .querySelector(".files-sidebar")
        ?.textContent?.includes("locally"),
      false,
    );
    await env.publishLayout(stale as FileLayout);
    assert.equal(
      env.title(),
      "acknowledged.py",
      "An old cloud task cannot revert the visible name",
    );
    assert.equal(env.editor().value, "main = 1");
    assert.equal(env.layoutSubscriberCount(), 1);
  } finally {
    local.resolve();
    await env.cleanup();
  }
  assert.equal(env.layoutSubscriberCount(), 0);
});

test("App surfaces a background cloud conflict without reverting a durable local rename", async () => {
  const env = setup({ pendingLayout: true });
  try {
    await env.render();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("local-paper.py");
    await submitName();
    assert.equal(env.title(), "local-paper.py");
    await env.publishLayout({
      ...env.getLayout(),
      pending_sync: true,
      sync_conflict: true,
    } as FileLayout);
    assert.ok(
      document.body.textContent?.includes(
        "The file layout changed in another session.",
      ),
    );
    assert.equal(env.title(), "local-paper.py");
    assert.equal(env.editor().readOnly, false);
    assert.equal(env.editor().value, "main = 1");
  } finally {
    await env.cleanup();
  }
});

test("App applies background access denial even when its layout version is older", async () => {
  const env = setup();
  try {
    await env.render();
    const stale = env.getLayout();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("safe.py");
    await submitName();
    await env.publishLayout({
      ...stale,
      sync_error: {
        status: 403,
        code: "ROLE_REQUIRED",
        message: "Üyelik sona erdi.",
      },
    } as FileLayout);
    assert.equal(env.title(), "safe.py");
    assert.equal(env.editor().value, "main = 1");
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.button("Run").disabled, true);
    assert.ok(document.body.textContent?.includes("Üyelik sona erdi."));
  } finally {
    await env.cleanup();
  }
});

test("App creates a folder and new script in the selected folder with durable layout metadata", async () => {
  const env = setup();
  try {
    await env.render();
    await env.click("New folder");
    await fillName("Models");
    await submitName();
    await env.click("Models");
    await env.click("New file");
    await fillName("new.py");
    await submitName();
    const created = env
      .getLayout()
      .entries.find((e) => e.id === "c".repeat(32))!;
    const folder = env.getLayout().entries.find((e) => e.name === "Models")!;
    assert.equal(created.parent, folder.id);
    assert.equal(env.title(), "new.py");
    assert.equal(env.editor().value, "");
    assert.equal(env.docs.get("analysis")!.code, "main = 1");
  } finally {
    await env.cleanup();
  }
});

test("App keeps original code and file header when layout persistence fails", async () => {
  const env = setup({
    onLayoutSave: async () => {
      throw new Error("offline failure");
    },
  });
  try {
    await env.render();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("unwritten.py");
    await submitName();
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().value, "main = 1");
    assert.equal(env.getLayout().entries[0].name, "analysis.py");
    assert.ok(document.body.textContent?.includes("offline failure"));
  } finally {
    await env.cleanup();
  }
});

test("App keeps file and execution controls locked until a new file's layout refresh and folder placement finish", async () => {
  const refresh = deferred<void>();
  let blockRefresh = false;
  const env = setup({
    onLayoutRead: async () => {
      if (blockRefresh) await refresh.promise;
    },
  });
  try {
    await env.render();
    await env.click("New folder");
    await fillName("Models");
    await submitName();
    await env.click("Models");
    blockRefresh = true;
    await env.click("New file");
    await fillName("new.py");
    await submitName();
    assert.equal(env.editor().readOnly, true);
    assert.equal(env.button("New file").disabled, true);
    assert.equal(env.button("analysis.py").disabled, true);
    assert.equal(env.button("Run").disabled, true);
    await env.click("analysis.py");
    await env.click("New file");
    await fillName("new.py");
    await submitName();
    await env.click("Run");
    assert.equal(
      env.calls.filter(
        (c) => c.path === "/console/scripts" && c.method === "POST",
      ).length,
      1,
    );
    assert.equal(
      env.calls.some((c) => c.path === "/console/execute"),
      false,
    );
    await React.act(() => refresh.resolve());
    const saved = env.getLayout();
    assert.equal(
      saved.entries.find((e) => e.id === "c".repeat(32))!.parent,
      saved.entries.find((e) => e.name === "Models")!.id,
    );
    assert.equal(env.title(), "new.py");
    assert.equal(env.editor().readOnly, false);
    assert.equal(env.docs.get("analysis")!.code, "main = 1");
  } finally {
    refresh.resolve();
    await env.cleanup();
  }
});

test("App includes persisted example datasets in organized inventory so renaming a script does not omit them", async () => {
  const dataset: DatasetProfile = {
    id: "d".repeat(32),
    name: "example.csv",
    source: "example",
    row_count: 2,
    column_count: 1,
    columns: [],
    preview: [],
    data_hash: "e".repeat(64),
    created_at: "2026-10-03T00:00:00Z",
    size_bytes: 6,
  };
  const env = setup({ datasets: [dataset] });
  try {
    await env.render();
    assert.ok(env.button("example.csv"));
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("paper.py");
    await submitName();
    assert.equal(env.title(), "paper.py");
    const entries = env.calls.find(
      (c) => c.path === "/files/layout" && c.method === "PUT",
    )!.body!.entries as { kind: string; id: string }[];
    assert.ok(entries.some((e) => e.kind === "dataset" && e.id === dataset.id));
    assert.equal(
      env.getLayout().entries.find((e) => e.id === dataset.id)!.name,
      dataset.name,
    );
  } finally {
    await env.cleanup();
  }
});

test("App refreshes teammate-created files after an inventory conflict and can retry without reopening", async () => {
  let conflict = true;
  const peerId = "f".repeat(32);
  const env = setup({
    onLayoutSave: async () => {
      if (!conflict) return;
      conflict = false;
      env.docs.set(peerId, {
        id: peerId,
        name: "peer.py",
        code: "peer = 1",
        version: 0,
      });
      throw new runtime.ApiError(
        "Dosya listesi değişti",
        409,
        "FILE_INVENTORY_CONFLICT",
      );
    },
  });
  try {
    await env.render();
    await env.contextMenu("analysis.py");
    await env.click("Rename");
    await fillName("paper.py");
    await submitName();
    assert.equal(env.title(), "analysis.py");
    assert.ok(env.button("peer.py"));
    assert.equal(env.editor().value, "main = 1");
    await submitName();
    assert.equal(env.title(), "paper.py");
    assert.equal(
      env.getLayout().entries.find((e) => e.id === peerId)!.name,
      "peer.py",
    );
    assert.equal(env.docs.get(peerId)!.code, "peer = 1");
  } finally {
    await env.cleanup();
  }
});

test("App creates, saves and reopens Markdown and LaTeX documents without executing them", async () => {
  for (const [extension, language, source] of [
    [".md", "markdown", "# Results\nText"],
    [
      ".tex",
      "latex",
      "\\section{Results}\n\\begin{tabular}{lr}X&1\\\\\\end{tabular}",
    ],
  ]) {
    const env = setup();
    try {
      await env.render();
      await env.click("New file");
      await fillName(`notes${extension}`);
      await submitName();
      const creation = env.calls.find(
        (c) => c.path === "/console/scripts" && c.method === "POST",
      )!;
      assert.deepEqual(creation.body, {
        name: `notes${extension}`,
        code: "",
      });
      assert.equal(env.title(), `notes${extension}`);
      assert.equal(env.editor().dataset.language, language);
      assert.equal(env.editor().readOnly, false);
      assert.equal(env.button("Run"), undefined);
      assert.equal(env.button("Run selection"), undefined);
      await env.edit(source);
      await env.click("analysis.py");
      assert.equal(env.docs.get("c".repeat(32))!.code, source);
      await env.click(`notes${extension}`);
      assert.equal(env.editor().value, source);
      await env.click("Test Python terminal");
      const execution = env.calls.find((c) => c.path === "/console/execute")!;
      assert.deepEqual(
        execution.body,
        { code: "print(41)" },
        "terminal is Python without a document identity",
      );
    } finally {
      await env.cleanup();
    }
  }
});

test("App named extension aliases switch editor mode and execution eligibility without rewriting source", async () => {
  const env = setup();
  try {
    await env.render();
    await env.click("first.py");
    const original = env.editor().value;
    for (const [name, language] of [
      ["notes.md", "markdown"],
      ["table.tex", "latex"],
      ["first.py", "python"],
    ]) {
      await env.contextMenu(env.title()!);
      await env.click("Rename");
      await fillName(name);
      await submitName();
      assert.equal(env.title(), name);
      assert.equal(env.editor().dataset.language, language);
      assert.equal(env.editor().value, original);
      assert.equal(Boolean(env.button("Run")), language === "python");
      assert.equal(
        env.docs.get(first)!.name,
        "first.py",
        "effective names are shared layout aliases",
      );
    }
    await env.click("Run");
    assert.deepEqual(
      env.calls.find((c) => c.path === "/console/execute")!.body,
      { code: original, script_id: first },
    );
    assert.equal(
      env.calls.filter(
        (c) => c.path === `/console/scripts/${first}` && c.method === "PUT",
      ).length,
      0,
    );
  } finally {
    await env.cleanup();
  }
});

test("App imports Markdown and LaTeX without appending Python extensions or transforming content", async () => {
  for (const [name, source] of [
    ["README.MD", "# Notes\n**Report**"],
    ["table.tex", "\\begin{tabular}{lr}x&1\\end{tabular}"],
  ]) {
    const env = setup();
    try {
      await env.render();
      const input = document.querySelector<HTMLInputElement>(
        'input[type="file"][accept*=".tex"]',
      )!;
      Object.defineProperty(input, "files", {
        value: [{ name, size: source.length, text: async () => source }],
        configurable: true,
      });
      await React.act(() =>
        input.dispatchEvent(new Event("change", { bubbles: true })),
      );
      assert.deepEqual(
        env.calls.find(
          (c) => c.path === "/console/scripts" && c.method === "POST",
        )!.body,
        { name: name.replace(/\.MD$/, ".md"), code: source },
      );
      assert.equal(env.editor().value, source);
      assert.equal(env.button("Run"), undefined);
      assert.equal(
        env.calls.some((c) => c.path === "/console/execute"),
        false,
      );
    } finally {
      await env.cleanup();
    }
  }
});

test("App names a new document before any creation and preserves its inline draft after a failed request", async () => {
  let reject = true;
  const env = setup({
    onCreate: async () => {
      if (reject) throw new Error("creation unavailable");
    },
  });
  try {
    await env.render();
    await env.click("New file");
    assert.equal(
      env.calls.some((call) => call.method === "POST"),
      false,
    );
    assert.equal(
      document.querySelector('[role="menu"][aria-label="File type"]'),
      null,
    );
    await fillName("Notes.MD");
    await submitName();
    assert.equal(env.title(), "analysis.py");
    assert.equal(
      document.querySelector<HTMLInputElement>('input[aria-label="File name"]')
        ?.value,
      "Notes.MD",
    );
    assert.ok(document.body.textContent?.includes("creation unavailable"));
    assert.equal(env.docs.has("c".repeat(32)), false);
    reject = false;
    await submitName();
    assert.equal(env.title(), "Notes.md");
    assert.equal(env.editor().dataset.language, "markdown");
    assert.equal(document.querySelector('input[aria-label="File name"]'), null);
    assert.equal(env.calls.filter((call) => call.method === "POST").length, 2);
  } finally {
    await env.cleanup();
  }
});

test("App inline creation preserves an outgoing unsaved source when its local save fails", async () => {
  const env = setup({
    onSave: async () => {
      throw new Error("outgoing save unavailable");
    },
  });
  try {
    await env.render();
    await env.edit("valuable_unsaved = 22");
    await env.click("New file");
    await fillName("paper.tex");
    await submitName();
    assert.equal(env.title(), "analysis.py");
    assert.equal(env.editor().value, "valuable_unsaved = 22");
    assert.equal(
      env.calls.some((call) => call.method === "POST"),
      false,
    );
    assert.equal(
      document.querySelector<HTMLInputElement>('input[aria-label="File name"]')
        ?.value,
      "paper.tex",
    );
    assert.ok(document.body.textContent?.includes("outgoing save unavailable"));
  } finally {
    await env.cleanup();
  }
});

test("App does not offer duplicate creation retry after a source is accepted but placement fails", async () => {
  const env = setup({
    onLayoutSave: async () => {
      throw new Error("placement unavailable");
    },
  });
  try {
    await env.render();
    await env.click("New file");
    await fillName("paper.tex");
    await submitName();
    assert.equal(env.calls.filter((call) => call.method === "POST").length, 1);
    assert.equal(env.docs.get("c".repeat(32))!.code, "");
    assert.equal(env.title(), "paper_2.tex");
    assert.equal(env.editor().dataset.language, "latex");
    assert.equal(document.querySelector('input[aria-label="File name"]'), null);
    assert.ok(document.body.textContent?.includes("placement unavailable"));
  } finally {
    await env.cleanup();
  }
});

test("App comparison keeps originals through cancel, another cloud write and CAS retry", async () => {
  let remote = {
    id: "analysis",
    name: "analysis.py",
    code: "cloud = 2",
    version: 2,
  };
  const attempts: { version: number; code: string }[] = [];
  const local = {
    id: "analysis",
    name: "analysis.py",
    code: "main = 1",
    version: 1,
  };
  const env = setup({
    onConflictRead: async (id) => {
      assert.equal(id, "analysis");
      return { base: "base = 0", local: { ...local }, remote: { ...remote } };
    },
    onConflictResolve: async (snapshot, code) => {
      attempts.push({ version: snapshot.remote.version, code });
      if (snapshot.remote.version !== remote.version)
        throw new runtime.ApiError(
          "Another cloud write; read the latest version",
          409,
          "VERSION_CONFLICT",
        );
      remote = { ...remote, code, version: remote.version + 1 };
      env.docs.set("analysis", { ...remote });
      return { ...remote };
    },
  });
  const merged = () =>
    document.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Merged draft"]',
    )!;
  try {
    await env.render();
    await env.edit("local_unsaved = 9");
    await env.publishScript({
      file: { ...local, sync_conflict: true },
      sync_conflict: true,
    });
    await env.click("Compare and merge versions");
    assert.equal(
      document.querySelector('pre[aria-label="Base"]')?.textContent,
      "base = 0",
    );
    assert.equal(
      document.querySelector('pre[aria-label="Your local draft"]')?.textContent,
      "local_unsaved = 9",
    );
    assert.equal(
      document.querySelector('pre[aria-label="Cloud version 2"]')?.textContent,
      "cloud = 2",
    );
    await env.click("Cancel · keep local draft");
    assert.equal(env.editor().value, "local_unsaved = 9");
    assert.equal(env.docs.get("analysis")!.code, "main = 1");
    await env.click("first.py");
    assert.equal(env.title(), "analysis.py");
    await assert.rejects(
      () => env.controls()!.prepareToLeave(),
      /cloud|version|conflict|çakış/i,
    );
    await env.click("Compare and merge versions");
    await React.act(() => {
      Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )!.set!.call(merged(), "merged = 10");
      merged().dispatchEvent(new Event("input", { bubbles: true }));
    });
    remote = { ...remote, code: "cloud = 3", version: 3 };
    await env.click("Save merged draft");
    assert.match(
      document.querySelector('.script-conflict-dialog [role="alert"]')!.textContent!,
      /Another cloud write/,
    );
    assert.equal(merged().value, "merged = 10");
    assert.equal(env.editor().value, "local_unsaved = 9");
    assert.equal(env.docs.get("analysis")!.code, "main = 1");
    await env.click("Read latest cloud version");
    assert.equal(
      document.querySelector('pre[aria-label="Cloud version 3"]')?.textContent,
      "cloud = 3",
    );
    assert.equal(merged().value, "merged = 10");
    assert.ok(
      [...document.querySelectorAll("details pre")].some(
        (node) => node.textContent === "cloud = 2",
      ),
    );
    await env.click("Save merged draft");
    assert.deepEqual(attempts, [
      { version: 2, code: "merged = 10" },
      { version: 3, code: "merged = 10" },
    ]);
    assert.equal(env.editor().value, "merged = 10");
    assert.equal(document.querySelector(".draft-conflict"), null);
    await env.click("first.py");
    assert.equal(env.title(), "first.py");
  } finally {
    await env.cleanup();
  }
});
