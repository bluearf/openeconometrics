import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
for (const name of [
  "window",
  "document",
  "navigator",
  "Window",
  "HTMLElement",
  "Element",
  "Node",
  "Event",
  "MouseEvent",
  "KeyboardEvent",
  "MutationObserver",
  "DOMRect",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, {
  requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
  cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
  getComputedStyle: dom.window.getComputedStyle.bind(dom.window),
});
dom.window.matchMedia = (media) =>
  ({
    matches: false,
    media,
    addEventListener() {},
    removeEventListener() {},
  }) as MediaQueryList;
const rectangle = () => new dom.window.DOMRect(0, 0, 800, 600);
dom.window.Range.prototype.getBoundingClientRect = rectangle;
dom.window.Range.prototype.getClientRects = function () {
  const rects = [rectangle()] as unknown as DOMRectList;
  rects.item = (index) => rects[index];
  return rects;
};

const { EditorState, Compartment } = await import("@codemirror/state");
const { EditorView, keymap } = await import("@codemirror/view");
const { history, undo, indentWithTab } = await import("@codemirror/commands");
const { autocompletion, startCompletion, closeCompletion, completionStatus } =
  await import("@codemirror/autocomplete");
let source = ts.transpileModule(
  readFileSync(new URL("../src/editor-inline.ts", import.meta.url), "utf8"),
  {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  },
).outputText;
source = source.replace(
  /(\bfrom\s+)(["'])([^"']+)\2/g,
  (_match, prefix, _quote, target) =>
    prefix + JSON.stringify(import.meta.resolve(target)),
);
const {
  inlineSuggestionExtensions,
  normalizeInlineSuggestion,
  inlineSuggestionContext,
  cancelInlineSuggestion,
  INLINE_IDLE_MS,
  INLINE_TIMEOUT_MS,
} = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);

type Request = {
  prefix: string;
  suffix: string;
  language: string;
  signal: AbortSignal;
};
function deferredProvider() {
  const calls: {
    request: Request;
    resolve: (text: string | null) => void;
    reject: (error: Error) => void;
  }[] = [];
  return {
    calls,
    request: (request: Request) =>
      new Promise<string | null>((resolve, reject) =>
        calls.push({ request, resolve, reject }),
      ),
  };
}
const delay = (ms = 30) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));
async function waitFor(predicate: () => boolean, timeout = 1800) {
  const deadline = Date.now() + timeout;
  while (!predicate()) {
    if (Date.now() > deadline)
      assert.fail("Expected editor state did not arrive");
    await delay(15);
  }
}
function fixture(
  code = "value =",
  options: {
    provider?: ReturnType<typeof deferredProvider>;
    contextKey?: string;
  } = {},
) {
  const host = document.createElement("div");
  document.body.append(host);
  const inline = new Compartment(),
    access = new Compartment();
  const view = new EditorView({
    doc: code,
    parent: host,
    extensions: [
      history(),
      keymap.of([indentWithTab]),
      access.of(EditorState.readOnly.of(false)),
      autocompletion({
        activateOnTyping: false,
        override: [
          () => ({ from: 0, filter: false, options: [{ label: "example" }] }),
        ],
      }),
      inline.of(
        inlineSuggestionExtensions({
          provider: options.provider,
          contextKey: options.contextKey ?? "project:file",
        }),
      ),
    ],
  });
  view.focus();
  view.dispatch({ selection: { anchor: code.length } });
  return {
    host,
    view,
    type(text = " ") {
      const pos = view.state.selection.main.head;
      view.dispatch({
        changes: { from: pos, insert: text },
        selection: { anchor: pos + text.length },
        userEvent: "input.type",
      });
    },
    cursor(pos: number) {
      view.dispatch({ selection: { anchor: pos } });
    },
    key(key: string) {
      const event = new KeyboardEvent("keydown", {
        key,
        bubbles: true,
        cancelable: true,
      });
      view.contentDOM.dispatchEvent(event);
      return event;
    },
    ghost() {
      return host.querySelector(".cm-inline-suggestion")?.textContent ?? null;
    },
    readonly(value = true) {
      view.dispatch({
        effects: access.reconfigure(EditorState.readOnly.of(value)),
      });
    },
    configure(
      provider?: ReturnType<typeof deferredProvider>,
      contextKey = "project:file",
    ) {
      view.dispatch({
        effects: inline.reconfigure(
          inlineSuggestionExtensions({ provider, contextKey }),
        ),
      });
    },
    close() {
      view.destroy();
      host.remove();
    },
  };
}

test("normalization preserves Python continuation and strips only redundant typed text", () => {
  assert.equal(
    normalizeInlineSuggestion("oe.ols(data=df)", "model = ", ""),
    "oe.ols(data=df)",
  );
  assert.equal(
    normalizeInlineSuggestion("model = oe.ols(data=df)", "model = ", ""),
    "oe.ols(data=df)",
  );
  assert.equal(
    normalizeInlineSuggestion("```python\nx * x\n```", "return ", ""),
    "x * x",
  );
  assert.equal(
    normalizeInlineSuggestion("\n    return x", "def f(x):", ""),
    "\n    return x",
  );
});

test("actual model suffix repetition and auto-close delimiters are never duplicated", () => {
  assert.equal(
    normalizeInlineSuggestion(
      "x * x\n\nassert square(3) == 9\n",
      "def square(x):\n    return ",
      "\n\nassert square(3) == 9",
    ),
    "x * x",
  );
  assert.equal(normalizeInlineSuggestion("model)", "display(", ")"), "model");
  assert.equal(
    normalizeInlineSuggestion("len(values))\n", "mean(", ")"),
    "len(values)",
  );
  assert.equal(normalizeInlineSuggestion('"wage"]', "df[", "]"), '"wage"');
  assert.equal(normalizeInlineSuggestion(")", "display(", ")"), null);
});

test("empty, explanatory, malformed, excessive and non-string responses are ignored", () => {
  for (const value of [
    null,
    {},
    "  \n",
    "Here is the code:\nx=1",
    "HERE IS THE CODE:\nx=1",
    "İşte kod:\nx=1",
    "```javascript\nx=1\n```",
    "x\0y",
    "a\nb\nc\nd",
    "x".repeat(513),
  ])
    assert.equal(normalizeInlineSuggestion(value, "x=", ""), null);
  assert.equal(
    normalizeInlineSuggestion("<img src=x onerror=alert(1)>", "x=", ""),
    "<img src=x onerror=alert(1)>",
  );
});

test("typing waits for idle and coalesces successive edits into one bounded request", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type(" ");
    await delay(100);
    assert.equal(provider.calls.length, 0);
    item.type("1");
    await delay(100);
    assert.equal(provider.calls.length, 0);
    await waitFor(() => provider.calls.length === 1);
    assert.equal(provider.calls[0].request.prefix, "value = 1");
    assert.equal(provider.calls[0].request.language, "python");
    await delay(INLINE_IDLE_MS + 50);
    assert.equal(provider.calls.length, 1);
  } finally {
    item.close();
  }
});

test("ghost text is only visual until Tab accepts one undoable editor transaction", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("42");
    await waitFor(() => item.ghost() === "42");
    assert.equal(item.view.state.doc.toString(), "value = ");
    assert.equal(item.key("Tab").defaultPrevented, true);
    assert.equal(item.view.state.doc.toString(), "value = 42");
    assert.equal(item.ghost(), null);
    assert.equal(undo(item.view), true);
    assert.equal(item.view.state.doc.toString(), "value = ");
  } finally {
    item.close();
  }
});

test("completion works immediately before an existing auto-close bracket", async () => {
  const provider = deferredProvider(),
    item = fixture("display()", { provider });
  try {
    item.cursor("display(".length);
    item.type(" ");
    await waitFor(() => provider.calls.length === 1);
    assert.equal(provider.calls[0].request.suffix, ")");
    provider.calls[0].resolve("model)");
    await waitFor(() => item.ghost() === "model");
    item.key("Tab");
    assert.equal(item.view.state.doc.toString(), "display( model)");
  } finally {
    item.close();
  }
});

test("Escape dismisses visible and pending suggestions until another source edit", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("42");
    await waitFor(() => item.ghost() !== null);
    assert.equal(item.key("Escape").defaultPrevented, true);
    assert.equal(item.ghost(), null);
    item.cursor(item.view.state.doc.length - 1);
    item.cursor(item.view.state.doc.length);
    await delay(INLINE_IDLE_MS + 50);
    assert.equal(provider.calls.length, 1);
    item.type("4");
    await waitFor(() => provider.calls.length === 2);
    item.key("Escape");
    assert.equal(provider.calls[1].request.signal.aborted, true);
    provider.calls[1].resolve("2");
    await delay();
    assert.equal(item.ghost(), null);
  } finally {
    item.close();
  }
});

test("document changes abort the old request and out-of-order responses cannot overwrite the latest", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    item.type("4");
    assert.equal(provider.calls[0].request.signal.aborted, true);
    await waitFor(() => provider.calls.length === 2);
    provider.calls[1].resolve("2");
    await waitFor(() => item.ghost() === "2");
    provider.calls[0].resolve("wrong");
    await delay();
    assert.equal(item.ghost(), "2");
  } finally {
    item.close();
  }
});

test("cursor movement, read-only, provider changes and file changes invalidate in-flight requests", async () => {
  for (const change of ["cursor", "readonly", "provider", "file"] as const) {
    const provider = deferredProvider(),
      item = fixture("value =", { provider });
    try {
      item.type();
      await waitFor(() => provider.calls.length === 1);
      if (change === "cursor") item.cursor(0);
      if (change === "readonly") item.readonly();
      if (change === "provider") item.configure(deferredProvider());
      if (change === "file")
        item.configure(provider, "other-project:other-file");
      assert.equal(provider.calls[0].request.signal.aborted, true, change);
      provider.calls[0].resolve("stale");
      await delay();
      assert.equal(item.ghost(), null, change);
    } finally {
      item.close();
    }
  }
});

test("focus loss cancels requests and never resurrects a hidden ghost on refocus", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  const button = document.createElement("button");
  document.body.append(button);
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("42");
    await waitFor(() => item.ghost() === "42");
    button.focus();
    await delay();
    assert.equal(item.ghost(), null);
    item.view.focus();
    await delay();
    assert.equal(item.ghost(), null);
    item.key("Tab");
    assert.ok(!item.view.state.doc.toString().includes("42"));
  } finally {
    item.close();
    button.remove();
  }
});

test("active dropdown completion owns the editor and suppresses inline suggestions", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("42");
    await waitFor(() => item.ghost() === "42");
    startCompletion(item.view);
    await waitFor(() => completionStatus(item.view.state) === "active");
    assert.equal(item.ghost(), null);
    const before = item.view.state.doc.toString();
    item.key("Tab");
    assert.ok(!item.view.state.doc.toString().includes("42"));
    assert.notEqual(
      item.view.state.doc.toString(),
      before,
      "Normal indentation still owns Tab",
    );
    closeCompletion(item.view);
  } finally {
    item.close();
  }
});

test("OFF settings, missing file identity, read-only access and middle identifiers make no requests", async () => {
  for (const disabled of ["off", "identity", "readonly", "middle"] as const) {
    const provider = deferredProvider(),
      item = fixture("identifier = value", {
        provider: disabled === "off" ? undefined : provider,
        contextKey: disabled === "identity" ? "" : undefined,
      });
    try {
      if (disabled === "readonly") item.readonly();
      if (disabled === "middle") item.cursor(2);
      item.type();
      await delay(INLINE_IDLE_MS + 70);
      assert.equal(provider.calls.length, 0, disabled);
      assert.equal(item.ghost(), null);
    } finally {
      item.close();
    }
  }
});

test("context extraction bounds large scripts without serializing the entire document", () => {
  const code =
    "# prior note\n".repeat(800) + "value =\n" + "# future note\n".repeat(800);
  const pos = code.indexOf("value =") + "value =".length;
  const state = EditorState.create({ doc: code, selection: { anchor: pos } });
  const context = inlineSuggestionContext(state);
  assert.equal(context.prefix, code.slice(pos - 4096, pos));
  assert.equal(context.suffix, code.slice(pos, pos + 2048));
  assert.equal(context.prefix.length, 4096);
  assert.equal(context.suffix.length, 2048);
});

test("untrusted suggestion output is rendered only as plain widget text", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("<img src=x onerror=alert(1)>");
    await waitFor(() => item.ghost() !== null);
    const ghost = item.host.querySelector(".cm-inline-suggestion")!;
    assert.equal(ghost.querySelector("img") === null, true);
    assert.match(item.ghost()!, /<img/);
  } finally {
    item.close();
  }
});

test("multiline ghost text preserves indentation and inserts only on Tab", async () => {
  const provider = deferredProvider(),
    item = fixture("def f(x):", { provider });
  try {
    item.type("\n    ");
    await waitFor(() => provider.calls.length === 1);
    provider.calls[0].resolve("return x\n\nf(1)");
    await waitFor(() => item.ghost() === "return x\n\nf(1)");
    assert.equal(item.view.state.doc.toString(), "def f(x):\n    ");
    item.key("Tab");
    assert.equal(
      item.view.state.doc.toString(),
      "def f(x):\n    return x\n\nf(1)",
    );
    undo(item.view);
    assert.equal(item.view.state.doc.toString(), "def f(x):\n    ");
  } finally {
    item.close();
  }
});

test("a timed-out local provider is aborted without changing or retrying the script", async () => {
  const provider = deferredProvider(),
    item = fixture("value =", { provider });
  try {
    item.type();
    await waitFor(() => provider.calls.length === 1);
    await waitFor(
      () => provider.calls[0].request.signal.aborted,
      INLINE_TIMEOUT_MS + 500,
    );
    provider.calls[0].resolve("late");
    await delay();
    assert.equal(item.ghost(), null);
    assert.equal(item.view.state.doc.toString(), "value = ");
    assert.equal(provider.calls.length, 1);
  } finally {
    item.close();
  }
});

test("run cancellation and unmount abort pending work without executing its response", async () => {
  for (const action of ["run", "unmount"] as const) {
    const provider = deferredProvider(),
      item = fixture("value =", { provider });
    let closed = false;
    try {
      item.type();
      await waitFor(() => provider.calls.length === 1);
      if (action === "run") cancelInlineSuggestion(item.view);
      else {
        item.close();
        closed = true;
      }
      assert.equal(provider.calls[0].request.signal.aborted, true);
      provider.calls[0].resolve("late");
      await delay();
      assert.equal(item.ghost(), null);
      if (!closed) assert.equal(item.view.state.doc.toString(), "value = ");
    } finally {
      if (!closed) item.close();
    }
  }
});
