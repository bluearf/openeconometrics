import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { WorkspaceClient } from "../src/api.ts";

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
  "HTMLTextAreaElement",
  "Event",
  "MouseEvent",
  "KeyboardEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

class TestPointerEvent extends MouseEvent {
  pointerId: number;
  isPrimary: boolean;
  pointerType: string;
  constructor(type: string, options: PointerEventInit = {}) {
    super(type, options);
    this.pointerId = options.pointerId ?? 1;
    this.isPrimary = options.isPrimary ?? true;
    this.pointerType = options.pointerType ?? "mouse";
  }
}
Object.defineProperty(globalThis, "PointerEvent", { value: TestPointerEvent });
Object.defineProperty(dom.window, "PointerEvent", { value: TestPointerEvent });
const captures = new Map<HTMLElement, number>();
HTMLElement.prototype.setPointerCapture = function (id) {
  captures.set(this, id);
};
HTMLElement.prototype.hasPointerCapture = function (id) {
  return captures.get(this) === id;
};
HTMLElement.prototype.releasePointerCapture = function (id) {
  if (captures.get(this) === id) captures.delete(this);
};

let workspaceWidth = 1240;
let workspaceHeight = 640;
function rectangle(width: number, height: number): DOMRect {
  return {
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: width,
    bottom: height,
    width,
    height,
    toJSON: () => ({}),
  };
}
HTMLElement.prototype.getBoundingClientRect = function () {
  const width = this.classList.contains("workspace-layout")
    ? workspaceWidth
    : this.classList.contains("workbench")
      ? workspaceWidth - 226
      : this.classList.contains("output-pane")
        ? (workspaceWidth - 226) / 2
        : workspaceWidth;
  return rectangle(width, workspaceHeight);
};
for (const [name, axis] of [
  ["clientWidth", "width"],
  ["clientHeight", "height"],
] as const)
  Object.defineProperty(HTMLElement.prototype, name, {
    configurable: true,
    get() {
      return this.getBoundingClientRect()[axis];
    },
  });

const observers = new Set<TestResizeObserver>();
class TestResizeObserver {
  targets = new Set<Element>();
  callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    observers.add(this);
  }
  observe(target: Element) {
    this.targets.add(target);
  }
  unobserve(target: Element) {
    this.targets.delete(target);
  }
  disconnect() {
    observers.delete(this);
    this.targets.clear();
  }
  notify() {
    this.callback(
      [...this.targets].map((target) => ({
        target,
        contentRect: target.getBoundingClientRect(),
      })) as ResizeObserverEntry[],
      this as unknown as ResizeObserver,
    );
  }
}
Object.defineProperty(globalThis, "ResizeObserver", {
  value: TestResizeObserver,
  configurable: true,
});
Object.defineProperty(dom.window, "ResizeObserver", {
  value: TestResizeObserver,
  configurable: true,
});
Object.defineProperty(dom.window, "innerWidth", {
  configurable: true,
  get: () => workspaceWidth,
});
Object.defineProperty(dom.window, "innerHeight", {
  configurable: true,
  get: () => workspaceHeight + 180,
});
dom.window.matchMedia = (query) =>
  ({
    matches: query.includes("720px") && workspaceWidth <= 720,
    media: query,
    addEventListener() {},
    removeEventListener() {},
  }) as MediaQueryList;
Object.assign(globalThis, {
  requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
  cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
});

const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const urls = new Map<string, string>();
const sourceRoot = new URL("../src/", import.meta.url);
// The real App and terminal remain mounted. Only CodeMirror's browser layout
// dependency is adapted; editor identity, buffer and callbacks are observable.
const editorSource = `import {forwardRef,useImperativeHandle,useRef} from "react";
export default forwardRef(function Editor(props,ref){
 const input=useRef(null);
 useImperativeHandle(ref,()=>({getCode:()=>input.current.value,
 getSelectionOrLine:()=>input.current.value,focus:()=>input.current.focus(),
 insert:text=>{if(!props.readOnly)props.onChange(input.current.value+text);}}));
 return <textarea aria-label="Test Python editor" ref={input} value={props.value}
 readOnly={props.readOnly} onChange={event=>{if(!props.readOnly)props.onChange(event.target.value);}}/>;
});`;
function moduleUrl(name: string): string {
  if (urls.has(name)) return urls.get(name)!;
  const stub =
    name === "CodeEditor"
      ? editorSource
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
async function frame() {
  await React.act(
    () =>
      new Promise<void>((resolve) => requestAnimationFrame(() => resolve())),
  );
}

function setup({
  projectId = "pane-test-project",
  clearStorage = true,
  readOnly = false,
  isolatedRuns = false,
  emptyHistory = false,
  sharedHistory = false,
}: {
  projectId?: string;
  clearStorage?: boolean;
  readOnly?: boolean;
  isolatedRuns?: boolean;
  emptyHistory?: boolean;
  sharedHistory?: boolean;
} = {}) {
  document.body.innerHTML = "<div id='app'></div>";
  if (clearStorage) window.localStorage.clear();
  const calls: { path: string; method: string }[] = [];
  const code = "important_result = 123\n";
  const client: WorkspaceClient = {
    projectId,
    teams: sharedHistory,
    connect: async () => ({
      token: "",
      version: "test",
      environment: "local",
    }),
    cancelPending() {},
    request: async <T>(path: string, options: RequestInit = {}) => {
      calls.push({ path, method: options.method ?? "GET" });
      if (path === "/bootstrap") return {
        session: { token: "", version: "test", environment: "team", read_only: true, execution_mode: "isolated" },
        draft: { name: "analysis.py", code, version: 2 }, datasets: [], status: { running: false, session_generation: 0 },
      } as T;
      if (path.startsWith("/console/history?")) return {
        runs: [{ id: "old-archived-run", code_preview: "print('old archive')", status: "ok", state: "finished", created_at: "2026-10-01T00:00:00+00:00", actor_email: "owner@example.com", code_preview_truncated: false }],
        page_cursor: "first", next_cursor: null, scanned: 1, snapshot_at: "2026-10-06T00:00:00+00:00",
      } as T;
      if (path === "/runs/old-archived-run/record") return {
        id: "old-archived-run", code: "print('old archive')", stdout: "saved archive output", outputs: [], variables: [], error: null, status: "ok", duration_ms: 1, session_generation: 0,
      } as T;
      if (path === "/console")
        return {
          history: emptyHistory
            ? []
            : [
                {
                  id: "prior-owned-run",
                  code: "print('already completed')",
                  status: "ok",
                  stdout: "already completed\n",
                  error: null,
                  outputs: [],
                  variables: [],
                  duration_ms: 1,
                  session_generation: 0,
                },
              ],
          variables: [],
          status: { running: false, session_generation: 0 },
        } as T;
      if (path === "/datasets") return { datasets: [] } as T;
      if (path === "/console/script" || path === "/console/scripts/analysis")
        return { id: "analysis", name: "analysis.py", code, version: 2 } as T;
      if (path === "/console/scripts")
        return {
          scripts: [{ id: "analysis", name: "analysis.py", version: 2 }],
        } as T;
      throw new Error(`Unexpected pane-only request ${path}`);
    },
  };
  const root = createRoot(document.getElementById("app")!);
  const render = async () => {
    await React.act(() =>
      root.render(React.createElement(App, { client, readOnly, isolatedRuns })),
    );
    await frame();
  };
  const cleanup = async () => {
    await React.act(() => root.unmount());
    document.body.innerHTML = "";
  };
  const editor = () => document.querySelector<HTMLTextAreaElement>("textarea")!;
  const input = () =>
    document.querySelector<HTMLInputElement>(
      '[aria-label="Python, pip, or uv command"]',
    )!;
  const click = async (label: string) => {
    const button = [
      ...document.querySelectorAll<HTMLButtonElement>("button"),
    ].find(
      (element) =>
        element.getAttribute("aria-label") === label ||
        element.textContent?.trim() === label,
    );
    assert.ok(button, `Expected button ${label}`);
    await React.act(() => {
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  };
  const inputDraft = async (draft: string) =>
    React.act(() => {
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(input(), draft);
      input().dispatchEvent(new Event("input", { bubbles: true }));
    });
  return {
    code,
    calls,
    client,
    root,
    render,
    cleanup,
    editor,
    input,
    click,
    inputDraft,
  };
}

function separator(label: string): HTMLElement {
  const element = document.querySelector<HTMLElement>(
    `[role="separator"][aria-label="${label}"]`,
  );
  assert.ok(element, `Expected separator ${label}`);
  return element;
}
async function keyboard(element: HTMLElement, key: string, shiftKey = false) {
  await React.act(() => {
    element.dispatchEvent(
      new KeyboardEvent("keydown", { key, shiftKey, bubbles: true }),
    );
  });
}
async function pointer(
  element: EventTarget,
  type: string,
  x: number,
  y: number,
  pointerId = 1,
) {
  await React.act(() => {
    element.dispatchEvent(
      new TestPointerEvent(type, {
        clientX: x,
        clientY: y,
        pointerId,
        buttons: type === "pointerup" ? 0 : 1,
        button: 0,
        bubbles: true,
      }),
    );
  });
  await frame();
}
async function resize(width: number, height = workspaceHeight) {
  workspaceWidth = width;
  workspaceHeight = height;
  await React.act(() => {
    window.dispatchEvent(new Event("resize"));
    for (const observer of observers) observer.notify();
  });
  await frame();
}
function assertInert(env: ReturnType<typeof setup>) {
  assert.ok(env.calls.every((call) => call.method === "GET"));
  assert.ok(env.calls.every((call) => !call.path.includes("execute")));
  assert.equal(env.editor().value, env.code);
  assert.equal(document.querySelector(".file-tab")?.textContent, "analysis.py");
}

const labels = {
  sidebar: "Files panel width",
  editor: "Code and results panel width",
  terminal: "Terminal panel height",
};
const preferenceKey = (
  kind: keyof typeof labels,
  projectId = "pane-test-project",
) => `openecon:panes:v1:${encodeURIComponent(projectId)}:${kind}`;
const value = (element: HTMLElement) =>
  Number(element.getAttribute("aria-valuenow"));
function preference(kind: keyof typeof labels, projectId?: string) {
  const saved = window.localStorage.getItem(preferenceKey(kind, projectId));
  assert.notEqual(saved, null, `Expected persisted ${kind} preference`);
  return JSON.parse(saved!);
}

test("App pointer drag and keyboard resize change pane sizes without remounting editor or changing Python state", async () => {
  const env = setup();
  try {
    await env.render();
    const codeEditor = env.editor();
    const commandInput = env.input();
    await env.inputDraft("pip install pandas");
    const side = separator(labels.sidebar);
    const start = value(side);
    await pointer(side, "pointerdown", 220, 200);
    await pointer(document, "pointermove", 300, 200);
    await pointer(document, "pointerup", 300, 200);
    assert.equal(value(separator(labels.sidebar)), start + 80);
    assert.equal(preference("sidebar"), start + 80);
    const codeSplit = separator(labels.editor);
    const ratioBefore = value(codeSplit);
    await keyboard(codeSplit, "ArrowRight");
    assert.ok(value(separator(labels.editor)) > ratioBefore);
    assert.equal(env.editor(), codeEditor);
    assert.equal(env.input(), commandInput);
    assert.equal(env.input().value, "pip install pandas");
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("App remembers all three sizes on reopening the same project and keeps another project's layout independent", async () => {
  const env = setup();
  let reopened: ReturnType<typeof setup> | undefined;
  let other: ReturnType<typeof setup> | undefined;
  try {
    await env.render();
    await keyboard(separator(labels.sidebar), "ArrowRight", true);
    await keyboard(separator(labels.editor), "ArrowRight", true);
    await env.click("Terminal");
    await keyboard(separator(labels.terminal), "ArrowUp", true);
    const saved = {
      sidebar: value(separator(labels.sidebar)),
      editor: value(separator(labels.editor)),
      terminal: value(separator(labels.terminal)),
    };
    for (const kind of Object.keys(labels) as (keyof typeof labels)[])
      assert.equal(typeof preference(kind), "number");
    await env.cleanup();
    reopened = setup({ clearStorage: false });
    await reopened.render();
    await reopened.click("Terminal");
    for (const kind of Object.keys(labels) as (keyof typeof labels)[])
      assert.equal(value(separator(labels[kind])), saved[kind]);
    assertInert(reopened);
    await reopened.cleanup();
    other = setup({ projectId: "another-project", clearStorage: false });
    await other.render();
    assert.notEqual(value(separator(labels.sidebar)), saved.sidebar);
    assert.notEqual(value(separator(labels.editor)), saved.editor);
    assert.equal(
      window.localStorage.getItem(preferenceKey("sidebar", "another-project")),
      null,
    );
    assertInert(other);
  } finally {
    if (other) await other.cleanup();
    else if (reopened) await reopened.cleanup();
    else await env.cleanup();
  }
});

test("App constrains effective sizes when its measured workspace shrinks and restores preferences when space returns", async () => {
  const env = setup();
  try {
    await env.render();
    await keyboard(separator(labels.sidebar), "End");
    await keyboard(separator(labels.editor), "End");
    await env.click("Terminal");
    await keyboard(separator(labels.terminal), "End");
    const wide = {
      sidebar: value(separator(labels.sidebar)),
      editor: value(separator(labels.editor)),
      terminal: value(separator(labels.terminal)),
    };
    const preferences = Object.fromEntries(
      (Object.keys(labels) as (keyof typeof labels)[]).map((kind) => [
        kind,
        preference(kind),
      ]),
    );
    await resize(840, 380);
    assert.ok(value(separator(labels.sidebar)) < wide.sidebar);
    assert.ok(value(separator(labels.terminal)) < wide.terminal);
    for (const kind of Object.keys(labels) as (keyof typeof labels)[]) {
      const handle = separator(labels[kind]);
      assert.ok(value(handle) >= Number(handle.getAttribute("aria-valuemin")));
      assert.ok(value(handle) <= Number(handle.getAttribute("aria-valuemax")));
      assert.equal(preference(kind), preferences[kind]);
    }
    await resize(1240, 640);
    for (const kind of Object.keys(labels) as (keyof typeof labels)[])
      assert.equal(value(separator(labels[kind])), wide[kind]);
    assertInert(env);
  } finally {
    await env.cleanup();
    await resize(1240, 640);
  }
});

test("collapsed sidebar and terminal hide their resize handles while keeping their sizes and terminal draft", async () => {
  const env = setup();
  try {
    await env.render();
    await keyboard(separator(labels.sidebar), "ArrowRight", true);
    const sideWidth = value(separator(labels.sidebar));
    await env.inputDraft("uv pip install pyarrow");
    await env.click("Terminal");
    await keyboard(separator(labels.terminal), "ArrowUp", true);
    const terminalHeight = value(separator(labels.terminal));
    await env.click("Terminal");
    assert.equal(
      document.querySelector(
        `[role="separator"][aria-label="${labels.terminal}"]`,
      ),
      null,
    );
    await env.click("Hide files");
    assert.equal(
      document.querySelector(
        `[role="separator"][aria-label="${labels.sidebar}"]`,
      ),
      null,
    );
    assert.equal(env.input().value, "uv pip install pyarrow");
    await env.click("Show files");
    await env.click("Terminal");
    assert.equal(value(separator(labels.sidebar)), sideWidth);
    assert.equal(value(separator(labels.terminal)), terminalHeight);
    assert.equal(env.input().value, "uv pip install pyarrow");
    assert.ok(
      document
        .querySelector('[role="log"]')
        ?.textContent?.includes("already completed"),
    );
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("pointer cancellation, Escape and window blur restore the drag's starting layout and release capture", async () => {
  const env = setup();
  try {
    await env.render();
    await keyboard(separator(labels.sidebar), "ArrowRight");
    const start = value(separator(labels.sidebar));
    const stored = preference("sidebar");
    for (const reason of ["pointercancel", "Escape", "blur"] as const) {
      const handle = separator(labels.sidebar);
      await pointer(handle, "pointerdown", 230, 200, 7);
      await pointer(document, "pointermove", 320, 200, 7);
      assert.ok(value(separator(labels.sidebar)) > start);
      if (reason === "Escape") await keyboard(handle, "Escape");
      else if (reason === "blur")
        await React.act(() => {
          window.dispatchEvent(new Event("blur"));
        });
      else await pointer(document, reason, 320, 200, 7);
      assert.equal(value(separator(labels.sidebar)), start);
      assert.equal(preference("sidebar"), stored);
      assert.equal(captures.size, 0);
      assert.equal(document.body.style.cursor, "");
      assert.equal(document.body.style.userSelect, "");
    }
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("unmounting during a drag releases pointer and restores pre-existing document interaction styles", async () => {
  const env = setup();
  try {
    document.body.style.cursor = "crosshair";
    document.body.style.userSelect = "text";
    await env.render();
    const handle = separator(labels.sidebar);
    await pointer(handle, "pointerdown", 220, 200, 9);
    await pointer(document, "pointermove", 280, 200, 9);
    assert.ok(captures.size > 0);
    await env.cleanup();
    assert.equal(captures.size, 0);
    assert.equal(document.body.style.cursor, "crosshair");
    assert.equal(document.body.style.userSelect, "text");
    await pointer(document, "pointermove", 410, 200, 9);
    assert.equal(document.body.innerHTML, "");
    assert.ok(env.calls.every((call) => call.method === "GET"));
  } finally {
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    await env.cleanup();
  }
});

test("Home, End and double click expose useful bounds and reset the default layout", async () => {
  const env = setup();
  try {
    await env.render();
    const original = value(separator(labels.editor));
    const handle = separator(labels.editor);
    assert.equal(handle.tabIndex, 0);
    assert.equal(handle.getAttribute("aria-orientation"), "vertical");
    await keyboard(handle, "Home");
    assert.equal(
      value(separator(labels.editor)),
      Number(handle.getAttribute("aria-valuemin")),
    );
    await keyboard(handle, "End");
    assert.equal(
      value(separator(labels.editor)),
      Number(handle.getAttribute("aria-valuemax")),
    );
    await React.act(() => {
      handle.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    assert.equal(value(separator(labels.editor)), original);
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("read-only users can adjust their own layout and isolated execution does not add a terminal handle", async () => {
  const env = setup({ readOnly: true, isolatedRuns: true });
  try {
    await env.render();
    assert.equal(document.querySelector(".workspace-terminal"), null);
    assert.equal(
      document.querySelector(
        `[role="separator"][aria-label="${labels.terminal}"]`,
      ),
      null,
    );
    const handle = separator(labels.editor);
    const before = value(handle);
    await keyboard(handle, "ArrowLeft");
    assert.ok(value(separator(labels.editor)) < before);
    assert.equal(env.editor().readOnly, true);
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("a new project's empty terminal opens a usable resize area without submitting its package draft", async () => {
  const env = setup({ emptyHistory: true });
  try {
    await env.render();
    await env.inputDraft("pip install scipy");
    await env.click("Terminal");
    const log = document.querySelector<HTMLElement>('[role="log"]')!;
    assert.equal(log.hidden, false);
    const handle = separator(labels.terminal);
    const before = value(handle);
    assert.equal(handle.getAttribute("aria-orientation"), "horizontal");
    await pointer(handle, "pointerdown", 700, 500);
    await pointer(document, "pointermove", 700, 440);
    await pointer(document, "pointerup", 700, 440);
    assert.equal(value(separator(labels.terminal)), before + 60);
    assert.equal(env.input().value, "pip install scipy");
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("secondary pointers cannot start or interrupt an owned resize gesture and lost capture restores its original size", async () => {
  const env = setup();
  try {
    await env.render();
    const handle = separator(labels.sidebar);
    const start = value(handle);
    await React.act(() => {
      handle.dispatchEvent(
        new TestPointerEvent("pointerdown", {
          clientX: 220,
          clientY: 200,
          pointerId: 2,
          isPrimary: false,
          pointerType: "touch",
          button: 0,
          bubbles: true,
        }),
      );
    });
    assert.equal(captures.size, 0);
    await React.act(() => {
      handle.dispatchEvent(
        new TestPointerEvent("pointerdown", {
          clientX: 220,
          clientY: 200,
          pointerId: 3,
          button: 2,
          bubbles: true,
        }),
      );
    });
    assert.equal(captures.size, 0);
    await pointer(handle, "pointerdown", 220, 200, 5);
    await pointer(document, "pointermove", 260, 200, 5);
    assert.equal(value(separator(labels.sidebar)), start + 40);
    await pointer(document, "pointermove", 460, 200, 8);
    await pointer(document, "pointerup", 460, 200, 8);
    await pointer(document, "pointercancel", 460, 200, 8);
    assert.equal(value(separator(labels.sidebar)), start + 40);
    assert.equal(captures.get(handle), 5);
    await pointer(handle, "lostpointercapture", 260, 200, 5);
    assert.equal(value(separator(labels.sidebar)), start);
    assert.equal(captures.size, 0);
    assert.equal(window.localStorage.getItem(preferenceKey("sidebar")), null);
    assertInert(env);
  } finally {
    await env.cleanup();
  }
});

test("cancelling a drag in a temporarily narrow window retains the preferred size for a larger window", async () => {
  const env = setup();
  try {
    await env.render();
    await keyboard(separator(labels.sidebar), "End");
    const wide = value(separator(labels.sidebar));
    const saved = preference("sidebar");
    await resize(840, 640);
    const handle = separator(labels.sidebar);
    const constrained = value(handle);
    assert.ok(constrained < wide);
    await pointer(handle, "pointerdown", 352, 200, 7);
    await pointer(document, "pointermove", 292, 200, 7);
    assert.ok(value(separator(labels.sidebar)) < constrained);
    await keyboard(handle, "Escape");
    assert.equal(value(separator(labels.sidebar)), constrained);
    assert.equal(preference("sidebar"), saved);
    await resize(1240, 640);
    assert.equal(value(separator(labels.sidebar)), wide);
    assertInert(env);
  } finally {
    await env.cleanup();
    await resize(1240, 640);
  }
});

test.after(() => dom.window.close());


test("App opens an archived team result without replacing the editor or losing it on a history refresh", async () => {
  Object.defineProperty(dom.window.HTMLDialogElement.prototype, "showModal", { configurable: true, value() { this.setAttribute("open", ""); } });
  Object.defineProperty(dom.window.HTMLDialogElement.prototype, "close", { configurable: true, value() { this.removeAttribute("open"); } });
  const env = setup({ sharedHistory: true, readOnly: true, isolatedRuns: true });
  try {
    await env.render(); await env.click("History");
    const archived = [...document.querySelectorAll<HTMLButtonElement>("button")].find(button => button.textContent?.includes("old-archived-run"))!;
    assert.ok(archived);
    await React.act(() => archived.click());
    assert.equal(env.editor().value, env.code);
    assert.match(document.querySelector(".output-pane")!.textContent!, /saved archive output/);
    await React.act(() => window.dispatchEvent(new Event("focus")));
    await frame();
    assert.match(document.querySelector(".output-pane")!.textContent!, /saved archive output/);
    assert.equal(env.calls.filter(call => call.path === "/runs/old-archived-run/record").length, 1);
    assertInert(env);
  } finally { await env.cleanup(); }
});
