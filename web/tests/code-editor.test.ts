import assert from "node:assert/strict";
import test from "node:test";
import type { TestContext } from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { InlineSuggestionProvider } from "../src/editor-inline.ts";

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
  IS_REACT_ACT_ENVIRONMENT: true,
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

const React = await import("react");
const { createRoot } = await import("react-dom/client");
const { EditorView, activateHover } = await import("@codemirror/view");
const { forceParsing } = await import("@codemirror/language");
const { undo } = await import("@codemirror/commands");
const {
  startCompletion,
  currentCompletions,
  acceptCompletion,
  closeCompletion,
  setSelectedCompletion,
} = await import("@codemirror/autocomplete");
const modules = new Map<string, string>();
function moduleUrl(file: URL): string {
  const cached = modules.get(file.href);
  if (cached) return cached;
  let source = file.pathname.endsWith(".json")
    ? `export default ${readFileSync(file, "utf8")};`
    : ts.transpileModule(readFileSync(file, "utf8"), {
        compilerOptions: {
          jsx: ts.JsxEmit.ReactJSX,
          module: ts.ModuleKind.ESNext,
          target: ts.ScriptTarget.ES2022,
        },
      }).outputText;
  source = source.replace(/import\s+["'][^"']+\.css["'];?/g, "");
  source = source.replace(
    /(^import[^\n]*?\bfrom\s+)(["'])([^"'\r\n]+)\2(?:\s+with\s*\{\s*type:\s*["']json["']\s*\})?/gm,
    (_whole, prefix, _quote, target: string) => {
      const dependency = target.startsWith(".")
        ? moduleUrl(
            new URL(
              /\.(tsx?|json)$/.test(target) ? target : `${target}.ts`,
              file,
            ),
          )
        : import.meta.resolve(target);
      return prefix + JSON.stringify(dependency);
    },
  );
  const url =
    "data:text/javascript;base64," + Buffer.from(source).toString("base64");
  modules.set(file.href, url);
  return url;
}
const { default: CodeEditor } = await import(
  moduleUrl(new URL("../src/CodeEditor.tsx", import.meta.url))
);

async function delay(ms = 30) {
  await React.act(
    () => new Promise<void>((resolve) => setTimeout(resolve, ms)),
  );
}

// Advance only the informational-help deadline; parsing, completion and frame
// scheduling continue to use their real clocks in the actual editor.
function helpClock(t: TestContext) {
  const schedule = dom.window.setTimeout.bind(dom.window);
  const cancel = dom.window.clearTimeout.bind(dom.window);
  const timers = new Map<number, { at: number; fire: () => void }>();
  let now = 0;
  let nextId = -1;
  t.mock.method(dom.window, "setTimeout", (callback, ms, ...args) => {
    if (ms !== 8000 || typeof callback !== "function")
      return schedule(callback, ms, ...args);
    const id = nextId--;
    timers.set(id, { at: now + ms, fire: () => callback(...args) });
    return id;
  });
  t.mock.method(dom.window, "clearTimeout", (id) => {
    if (!timers.delete(id)) cancel(id);
  });
  return {
    get pending() {
      return timers.size;
    },
    advance(ms: number) {
      const target = now + ms;
      for (;;) {
        const due = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
        if (!due || due[1].at > target) break;
        const [id, timer] = due;
        timers.delete(id);
        now = timer.at;
        React.act(timer.fire);
      }
      now = target;
    },
  };
}
function setup(
  code = "import openecon as oe\n\noe.ols(",
  inline?: {
    provider: InlineSuggestionProvider;
    contextKey: string;
  },
  documentKey?: string,
) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const changes: string[] = [];
  const runs: string[] = [];
  const positions: [number, number][] = [];
  const ref = React.createRef<{ cancelInlineSuggestion: () => void }>();
  let props = {
    ref,
    documentKey,
    language: "python" as "python" | "markdown" | "latex",
    value: code,
    readOnly: false,
    fullScriptOnly: false,
    inlineSuggestionProvider: inline?.provider,
    inlineSuggestionContextKey: inline?.contextKey,
    onChange: (value: string) => changes.push(value),
    onRun: (value: string) => runs.push(value),
    onCursor: (line: number, column: number) => positions.push([line, column]),
  };
  React.act(() => root.render(React.createElement(CodeEditor, props)));
  const view = EditorView.findFromDOM(host.querySelector(".cm-editor")!)!;
  assert.ok(view, "Actual CodeMirror EditorView mounted");
  React.act(() => view.focus());
  return {
    host,
    view,
    changes,
    runs,
    positions,
    ref,
    update(patch: Partial<typeof props>) {
      props = { ...props, ...patch };
      React.act(() => root.render(React.createElement(CodeEditor, props)));
    },
    key(key: string, modifiers: KeyboardEventInit = {}) {
      const event = new KeyboardEvent("keydown", {
        key,
        bubbles: true,
        cancelable: true,
        ...modifiers,
      });
      React.act(() => view.contentDOM.dispatchEvent(event));
      return event;
    },
    close() {
      React.act(() => root.unmount());
      host.remove();
      assert.equal(document.querySelector(".cm-editor-help"), null);
    },
  };
}

test("editor retains source, cursor callbacks and instance across parent updates", () => {
  const fixture = setup("first = 1\nsecond = 2");
  try {
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: fixture.view.state.doc.length },
      }),
    );
    assert.deepEqual(fixture.positions.at(-1), [2, 11]);
    fixture.update({ value: "third = 3" });
    assert.equal(fixture.view.state.doc.toString(), "third = 3");
    assert.equal(
      EditorView.findFromDOM(fixture.host.querySelector(".cm-editor")!),
      fixture.view,
    );
  } finally {
    fixture.close();
  }
});

test("file switches retain one editor with independent undo and restored selections", () => {
  const fixture = setup("first = 1", undefined, "project:first");
  try {
    React.act(() =>
      fixture.view.dispatch({
        changes: {
          from: fixture.view.state.doc.length,
          insert: "\nfirst_extra = 2",
        },
        selection: { anchor: 5 },
      }),
    );
    const firstSource = fixture.view.state.doc.toString();
    const changes = fixture.changes.length;
    fixture.update({ documentKey: "project:second", value: "second = 3" });
    assert.equal(
      EditorView.findFromDOM(fixture.host.querySelector(".cm-editor")!),
      fixture.view,
    );
    assert.equal(fixture.view.state.selection.main.head, 0);
    assert.equal(
      fixture.changes.length,
      changes,
      "switching is not an edit or autosave",
    );
    assert.equal(
      undo(fixture.view),
      false,
      "a different file cannot undo outgoing text",
    );
    React.act(() =>
      fixture.view.dispatch({
        changes: {
          from: fixture.view.state.doc.length,
          insert: "\nsecond_extra = 4",
        },
        selection: { anchor: 8 },
      }),
    );
    const secondSource = fixture.view.state.doc.toString();
    fixture.update({ documentKey: "project:first", value: firstSource });
    assert.equal(fixture.view.state.selection.main.head, 5);
    assert.deepEqual(fixture.positions.at(-1), [1, 6]);
    React.act(() => undo(fixture.view));
    assert.equal(fixture.view.state.doc.toString(), "first = 1");
    fixture.update({ documentKey: "project:second", value: secondSource });
    assert.equal(fixture.view.state.selection.main.head, 8);
    React.act(() => undo(fixture.view));
    assert.equal(fixture.view.state.doc.toString(), "second = 3");
  } finally {
    fixture.close();
  }
});

test("restored documents use current access and reject outdated source histories", () => {
  const fixture = setup("mine = 1", undefined, "project:main");
  try {
    React.act(() =>
      fixture.view.dispatch({ changes: { from: 8, insert: "0" } }),
    );
    const previous = fixture.view.state.doc.toString();
    fixture.update({ readOnly: true, value: previous });
    fixture.update({ documentKey: "project:other", value: "other = 2" });
    fixture.update({
      documentKey: "project:main",
      value: previous,
      readOnly: false,
    });
    assert.equal(
      fixture.view.state.readOnly,
      false,
      "an outgoing busy state cannot lock a restored file",
    );
    fixture.update({ documentKey: "project:other", value: "other = 2" });
    fixture.update({ documentKey: "project:main", value: "team_update = 3" });
    assert.equal(fixture.view.state.doc.toString(), "team_update = 3");
    assert.equal(
      undo(fixture.view),
      false,
      "changed incoming source invalidates cached undo",
    );
    fixture.update({ documentKey: "another-project:main", value: "mine = 1" });
    assert.equal(
      undo(fixture.view),
      false,
      "project identities cannot share undo histories",
    );
  } finally {
    fixture.close();
  }
});

test("retained document histories are bounded while the EditorView stays mounted", () => {
  const fixture = setup("value = 0", undefined, "project:0");
  try {
    React.act(() =>
      fixture.view.dispatch({ changes: { from: 9, insert: "0" } }),
    );
    const previous = fixture.view.state.doc.toString();
    for (let index = 1; index <= 34; index++)
      fixture.update({
        documentKey: `project:${index}`,
        value: `value = ${index}`,
      });
    fixture.update({ documentKey: "project:0", value: previous });
    assert.equal(
      undo(fixture.view),
      false,
      "the oldest inactive history was released",
    );
    assert.equal(
      EditorView.findFromDOM(fixture.host.querySelector(".cm-editor")!),
      fixture.view,
    );
  } finally {
    fixture.close();
  }
});

test("editor keeps run-selection and run-script shortcuts without executing help", () => {
  const fixture = setup("first = 1\nsecond = 2");
  const modifier = /Mac/.test(navigator.platform)
    ? { metaKey: true }
    : { ctrlKey: true };
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: 10, head: 20 } }),
    );
    fixture.key("Enter", modifier);
    assert.deepEqual(fixture.runs, ["second = 2"]);
    fixture.key("Enter", { ...modifier, shiftKey: true });
    assert.equal(fixture.runs.at(-1), "first = 1\nsecond = 2");
    fixture.update({ fullScriptOnly: true });
    fixture.key("Enter", modifier);
    assert.equal(fixture.runs.at(-1), "first = 1\nsecond = 2");
    fixture.update({ readOnly: true });
    fixture.key("Enter", modifier);
    assert.equal(fixture.runs.length, 3);
  } finally {
    fixture.close();
  }
});

test("OpenEcon member completion inserts one member and remains undoable", async () => {
  const fixture = setup("import openecon as oe\noe.ol");
  try {
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: fixture.view.state.doc.length },
      }),
    );
    React.act(() => {
      startCompletion(fixture.view);
    });
    await delay(180);
    const olsIndex = currentCompletions(fixture.view.state).findIndex(
      (row) => row.label === "ols",
    );
    assert.ok(olsIndex >= 0);
    // The full catalogue also contains ologit/oprobit. Exercise the chosen
    // member's insertion and undo rather than relying on alphabetical rank.
    React.act(() => {
      fixture.view.dispatch({ effects: setSelectedCompletion(olsIndex) });
    });
    React.act(() => {
      acceptCompletion(fixture.view);
    });
    assert.match(fixture.view.state.doc.toString(), /oe\.ols(?:\(|$)/);
    assert.equal(fixture.runs.length, 0);
    React.act(() => {
      undo(fixture.view);
    });
    assert.equal(
      fixture.view.state.doc.toString(),
      "import openecon as oe\noe.ol",
    );
  } finally {
    fixture.close();
  }
});

test("built-in Python keyword completion is still available", async () => {
  const fixture = setup("whi");
  try {
    React.act(() => fixture.view.dispatch({ selection: { anchor: 3 } }));
    React.act(() => {
      startCompletion(fixture.view);
    });
    await delay(90);
    assert.ok(
      currentCompletions(fixture.view.state).some(
        (row) => row.label === "while",
      ),
    );
  } finally {
    fixture.close();
  }
});

test("read-only editor never accepts an offered completion", async () => {
  const fixture = setup("import openecon as oe\noe.ol");
  try {
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: fixture.view.state.doc.length },
      }),
    );
    React.act(() => {
      startCompletion(fixture.view);
    });
    await delay(90);
    fixture.update({ readOnly: true });
    React.act(() => {
      acceptCompletion(fixture.view);
    });
    assert.equal(
      fixture.view.state.doc.toString(),
      "import openecon as oe\noe.ol",
    );
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
    React.act(() => {
      closeCompletion(fixture.view);
    });
  } finally {
    fixture.close();
  }
});

test("F1 opens function documentation and Escape closes it without changing the script", async () => {
  const code = "import openecon as oe\nmodel = oe.ols";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: code.indexOf("oe.ols") + 4 },
      }),
    );
    assert.equal(fixture.key("F1").defaultPrevented, true);
    await delay(50);
    const help = document.querySelector('.cm-editor-help[role="tooltip"]');
    assert.ok(help, "Function documentation is rendered in the real editor");
    assert.equal(
      fixture.host.contains(help),
      false,
      "Help escapes clipping panel ancestors",
    );
    assert.match(help.textContent ?? "", /ols\(/);
    assert.ok(document.querySelector(".cm-help-signature"));
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
    fixture.key("Escape");
    await delay();
    assert.equal(document.querySelector(".cm-editor-help"), null);
  } finally {
    fixture.close();
  }
});

test("read-only access still permits keyboard documentation", async () => {
  const code = "import openecon as research\nresearch.ols";
  const fixture = setup(code);
  try {
    fixture.update({ readOnly: true });
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: code.lastIndexOf("ols") + 1 },
      }),
    );
    fixture.key("F1");
    await delay(50);
    assert.match(
      document.querySelector(".cm-editor-help")?.textContent ?? "",
      /ols\(/,
    );
    assert.equal(fixture.view.state.doc.toString(), code);
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

test("call help highlights a keyword parameter and respects Escape until the cursor changes", async () => {
  const code =
    'import openecon as oe\noe.ols(data=df, x=["education", "experience"], covariance=';
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    await delay(50);
    assert.match(
      document.querySelector(".cm-help-active-parameter")?.textContent ?? "",
      /covariance/,
    );
    fixture.key("Escape");
    await delay();
    assert.equal(document.querySelector(".cm-editor-help"), null);
    await delay();
    assert.equal(document.querySelector(".cm-editor-help"), null);
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length - 1 } }),
    );
    await delay();
    assert.ok(document.querySelector(".cm-editor-help"));
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

test("parameter help refreshes as a long document finishes background parsing", async () => {
  const code =
    "import openecon as oe\n" +
    "# unchanged analysis notes\n".repeat(1800) +
    "oe.ols(covariance=";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    React.act(() => {
      forceParsing(fixture.view, code.length, 1000);
    });
    await delay(60);
    assert.equal(
      document.querySelector(".cm-help-active-parameter")?.textContent,
      "covariance",
    );
    assert.equal(fixture.view.state.doc.toString(), code);
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

test("user function documentation remains plain text and Unicode parameters are highlighted", async () => {
  const code =
    'def düzenle(ücret, *, oran=0.05):\n    """<img src=x onerror=alert(1)> açıklama."""\n    return ücret\ndüzenle(ücret=';
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    await delay(50);
    const help = document.querySelector(".cm-editor-help");
    assert.ok(help);
    assert.equal(
      help.querySelector(".cm-help-active-parameter")?.textContent,
      "ücret",
    );
    assert.match(
      help.textContent ?? "",
      /<img src=x onerror=alert\(1\)> açıklama\./,
    );
    assert.equal(help.querySelector("img"), null);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

test("typing a known call opens required parameter suggestions without running code", async () => {
  const code = "import openecon as oe\noe.ols";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({
        changes: { from: code.length, insert: "(" },
        selection: { anchor: code.length + 1 },
        userEvent: "input.type",
      }),
    );
    await delay(220);
    assert.equal(currentCompletions(fixture.view.state)[0]?.label, "data=");
    fixture.key("Enter");
    assert.equal(fixture.view.state.doc.toString(), `${code}(data=`);
    assert.equal(fixture.runs.length, 0);
    React.act(() => undo(fixture.view));
    assert.equal(fixture.view.state.doc.toString(), `${code}(`);
  } finally {
    fixture.close();
  }
});

test("known data variables rank before generic Python globals in value suggestions", async () => {
  const code = "import openecon as oe\ndf=oe.example()\noe.ols(data=";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    React.act(() => startCompletion(fixture.view));
    await delay(180);
    const options = currentCompletions(fixture.view.state);
    assert.equal(options[0]?.label, "df");
    assert.equal(options[0]?.type, "variable");
    assert.equal(options[0]?.detail, "DataFrame");
    // CodeMirror intentionally ignores acceptance during the first 75 ms
    // after opening. Background parsing can finish near the end of delay(180)
    // on a busy host, so wait from the observed menu rather than from the request.
    await delay(100);
    fixture.key("Enter");
    assert.equal(fixture.view.state.doc.toString(), `${code}df`);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

test("quoted choices accept with Enter, preserve suffixes and stay undoable", async () => {
  for (const marked of [
    'import openecon as oe\noe.ols(covariance="H|C1", data=df)',
    "import openecon as oe\noe.ols(covariance='H|",
  ]) {
    const pos = marked.indexOf("|"),
      code = marked.slice(0, pos) + marked.slice(pos + 1),
      fixture = setup(code);
    try {
      React.act(() => fixture.view.dispatch({ selection: { anchor: pos } }));
      React.act(() => startCompletion(fixture.view));
      await delay(180);
      const index = currentCompletions(fixture.view.state).findIndex(
        (option) => option.label === "HC3",
      );
      assert.ok(index >= 0);
      React.act(() =>
        fixture.view.dispatch({ effects: setSelectedCompletion(index) }),
      );
      fixture.key("Enter");
      assert.equal(
        fixture.view.state.doc.toString(),
        marked.includes('"')
          ? 'import openecon as oe\noe.ols(covariance="HC3", data=df)'
          : "import openecon as oe\noe.ols(covariance='HC3'",
      );
      assert.equal(fixture.runs.length, 0);
      React.act(() => undo(fixture.view));
      assert.equal(fixture.view.state.doc.toString(), code);
    } finally {
      fixture.close();
    }
  }
});

test("the integrated Python sources respect future locals and unsupported receivers", async () => {
  for (const marked of [
    "import openecon as oe\ndef f():\n    oe|\n    oe=object()",
    "def f():\n    len|\n    len=object()",
    "import openecon as oe\ndf=oe.example()\ndf.head().|",
  ]) {
    const pos = marked.indexOf("|"),
      code = marked.slice(0, pos) + marked.slice(pos + 1),
      fixture = setup(code);
    try {
      React.act(() => fixture.view.dispatch({ selection: { anchor: pos } }));
      React.act(() => startCompletion(fixture.view));
      await delay(100);
      const labels = currentCompletions(fixture.view.state).map(
        (option) => option.label,
      );
      if (marked.includes("head().")) assert.equal(labels.length, 0);
      else assert.ok(!labels.includes(marked.includes("len|") ? "len" : "oe"));
      assert.equal(fixture.view.state.doc.toString(), code);
      assert.equal(fixture.runs.length, 0);
    } finally {
      fixture.close();
    }
  }
});

test("expression keywords and constants remain available inside known calls", async () => {
  const code = "def f(value):\n    pass\nf(";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    React.act(() => startCompletion(fixture.view));
    await delay(100);
    const labels = currentCompletions(fixture.view.state).map(
      (option) => option.label,
    );
    for (const label of ["lambda", "not", "None", "True", "False"])
      assert.ok(labels.includes(label), label);
    assert.ok(!labels.includes("def"));
  } finally {
    fixture.close();
  }
});

test("read-only transitions prevent accepting a visible literal suggestion", async () => {
  const code = "import openecon as oe\noe.ols(missing='dr";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    React.act(() => startCompletion(fixture.view));
    await delay(180);
    assert.ok(
      currentCompletions(fixture.view.state).some(
        (option) => option.label === "drop",
      ),
    );
    fixture.update({ readOnly: true });
    fixture.key("Enter");
    assert.equal(fixture.view.state.doc.toString(), code);
    assert.equal(fixture.changes.length, 0);
    assert.equal(fixture.runs.length, 0);
  } finally {
    fixture.close();
  }
});

function inlineProvider() {
  const calls: {
    signal: AbortSignal;
    resolve: (text: string | null) => void;
  }[] = [];
  const provider: InlineSuggestionProvider = {
    request: ({ signal }) =>
      new Promise((resolve) => calls.push({ signal, resolve })),
  };
  return { calls, provider };
}

async function waitInline(predicate: () => boolean) {
  const deadline = Date.now() + 1800;
  while (!predicate()) {
    if (Date.now() > deadline)
      assert.fail("Expected inline editor state did not arrive");
    await delay(25);
  }
}

function typeInline(fixture: ReturnType<typeof setup>, text = " ") {
  React.act(() =>
    fixture.view.dispatch({
      changes: { from: fixture.view.state.doc.length, insert: text },
      selection: { anchor: fixture.view.state.doc.length + text.length },
      userEvent: "input.type",
    }),
  );
}

test("React editor accepts a real ghost suggestion through onChange and undo without running code", async () => {
  const { provider, calls } = inlineProvider();
  const fixture = setup("answer =", { provider, contextKey: "qa:analysis.py" });
  try {
    typeInline(fixture);
    await waitInline(() => calls.length === 1);
    await React.act(async () => {
      calls[0].resolve("42");
    });
    await waitInline(
      () => !!fixture.host.querySelector(".cm-inline-suggestion"),
    );
    assert.equal(
      fixture.host.querySelector(".cm-inline-suggestion")?.textContent,
      "42",
    );
    assert.equal(fixture.view.state.doc.toString(), "answer = ");
    assert.equal(fixture.key("Tab").defaultPrevented, true);
    assert.equal(fixture.changes.at(-1), "answer = 42");
    assert.deepEqual(fixture.runs, []);
    React.act(() => undo(fixture.view));
    assert.equal(fixture.changes.at(-1), "answer = ");
  } finally {
    fixture.close();
  }
});

test("document identity cancels stale ghost requests even when suggestion settings stay unchanged", async () => {
  const { provider, calls } = inlineProvider();
  const fixture = setup(
    "answer =",
    { provider, contextKey: "shared-context" },
    "project:first",
  );
  try {
    typeInline(fixture);
    await waitInline(() => calls.length === 1);
    fixture.update({ documentKey: "project:second", value: "other = 1" });
    assert.equal(calls[0].signal.aborted, true);
    await React.act(async () => calls[0].resolve("stale"));
    await delay();
    assert.equal(fixture.host.querySelector(".cm-inline-suggestion"), null);
    assert.equal(fixture.view.state.doc.toString(), "other = 1");
    assert.equal(
      EditorView.findFromDOM(fixture.host.querySelector(".cm-editor")!),
      fixture.view,
    );
    assert.deepEqual(fixture.runs, []);
  } finally {
    fixture.close();
  }
});

test("React editor Run shortcut cancels an in-flight suggestion before executing existing source", async () => {
  const { provider, calls } = inlineProvider();
  const fixture = setup("answer =", { provider, contextKey: "qa:analysis.py" });
  const modifier = /Mac/.test(navigator.platform)
    ? { metaKey: true }
    : { ctrlKey: true };
  try {
    typeInline(fixture);
    await waitInline(() => calls.length === 1);
    fixture.key("Enter", modifier);
    assert.equal(calls[0].signal.aborted, true);
    assert.deepEqual(fixture.runs, ["answer = "]);
    await React.act(async () => {
      calls[0].resolve("late");
    });
    await delay();
    assert.equal(fixture.host.querySelector(".cm-inline-suggestion"), null);
    assert.equal(fixture.view.state.doc.toString(), "answer = ");
  } finally {
    fixture.close();
  }
});

test("React provider, file and read-only updates preserve the editor while canceling stale work", async () => {
  for (const transition of ["provider", "file", "readonly"] as const) {
    const { provider, calls } = inlineProvider();
    const fixture = setup("answer =", {
      provider,
      contextKey: "qa:analysis.py",
    });
    try {
      typeInline(fixture);
      await waitInline(() => calls.length === 1);
      fixture.update({
        value: fixture.view.state.doc.toString(),
        ...(transition === "provider"
          ? { inlineSuggestionProvider: undefined }
          : {}),
        ...(transition === "file"
          ? { inlineSuggestionContextKey: "qa:other.py" }
          : {}),
        ...(transition === "readonly" ? { readOnly: true } : {}),
      });
      assert.equal(calls[0].signal.aborted, true, transition);
      assert.equal(
        EditorView.findFromDOM(fixture.host.querySelector(".cm-editor")!),
        fixture.view,
      );
      await React.act(async () => {
        calls[0].resolve("stale");
      });
      await delay();
      assert.equal(fixture.host.querySelector(".cm-inline-suggestion"), null);
      assert.equal(fixture.view.state.doc.toString(), "answer = ");
      assert.deepEqual(fixture.runs, []);
    } finally {
      fixture.close();
    }
  }
});

test("React editor unmount cancels a local suggestion and accepts no late DOM updates", async () => {
  const { provider, calls } = inlineProvider();
  const fixture = setup("answer =", { provider, contextKey: "qa:analysis.py" });
  typeInline(fixture);
  await waitInline(() => calls.length === 1);
  fixture.close();
  assert.equal(calls[0].signal.aborted, true);
  await React.act(async () => {
    calls[0].resolve("late");
  });
  await delay();
  assert.equal(document.querySelector(".cm-inline-suggestion"), null);
});

test("visible F1 documentation expires after eight seconds and reopens on explicit F1", async (t) => {
  const clock = helpClock(t);
  const code = "import openecon as oe\nmodel = oe.ols";
  const fixture = setup(code);
  const help = () =>
    document.querySelector(".cm-tooltip-hover .cm-editor-help");
  try {
    React.act(() =>
      fixture.view.dispatch({
        selection: { anchor: code.indexOf("oe.ols") + 4 },
      }),
    );
    fixture.key("F1");
    await delay(50);
    assert.ok(help());
    clock.advance(7999);
    assert.ok(help(), "Visible documentation gets the full reading period");
    clock.advance(1);
    assert.equal(help(), null);
    React.act(() => {
      fixture.view.dispatch({});
      forceParsing(fixture.view, code.length, 1000);
    });
    await delay();
    assert.equal(
      help(),
      null,
      "Background work does not reopen timed-out help",
    );
    fixture.key("F1");
    await delay(50);
    assert.ok(help(), "A new explicit request reopens documentation");
    clock.advance(7999);
    assert.ok(help());
    assert.equal(fixture.view.state.doc.toString(), code);
    assert.deepEqual(fixture.runs, []);
  } finally {
    fixture.close();
  }
  assert.equal(clock.pending, 0, "Unmount cancels outstanding help deadlines");
  clock.advance(8000);
  assert.equal(document.querySelector(".cm-editor-help"), null);
});

test("signature help expires on inactivity, ignores state churn and resumes on actual input", async (t) => {
  const clock = helpClock(t);
  const code = "import openecon as oe\noe.ols(data=df, covariance=";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    await delay(50);
    assert.ok(document.querySelector(".cm-help-active-parameter"));
    clock.advance(7000);
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length - 1 } }),
    );
    await delay();
    clock.advance(7000);
    assert.ok(
      document.querySelector(".cm-editor-help"),
      "Cursor input restarts inactivity time",
    );
    React.act(() => {
      fixture.view.dispatch({ selection: fixture.view.state.selection });
      forceParsing(fixture.view, code.length, 1000);
      fixture.view.dispatch({});
    });
    fixture.update({ value: code });
    await delay();
    clock.advance(999);
    assert.ok(document.querySelector(".cm-editor-help"));
    clock.advance(1);
    assert.equal(
      document.querySelector(".cm-editor-help"),
      null,
      "Identical selections, parsing and React rerenders cannot extend help",
    );
    React.act(() =>
      fixture.view.dispatch({ selection: fixture.view.state.selection }),
    );
    await delay();
    assert.equal(document.querySelector(".cm-editor-help"), null);
    React.act(() =>
      fixture.view.dispatch({
        changes: { from: code.length, insert: "'HC3'" },
        selection: { anchor: code.length + 5 },
        userEvent: "input.type",
      }),
    );
    await delay(50);
    assert.ok(
      document.querySelector(".cm-editor-help"),
      "Typing resumes relevant signature help",
    );
    assert.deepEqual(fixture.runs, []);
  } finally {
    fixture.close();
  }
  assert.equal(clock.pending, 0);
});

test("blur and click-away dismiss informational help and cancel its timers", async (t) => {
  const clock = helpClock(t);
  const code = "import openecon as oe\noe.ols";
  const fixture = setup(code);
  const outside = document.createElement("button");
  document.body.append(outside);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length - 1 } }),
    );
    fixture.key("F1");
    await delay(50);
    assert.ok(document.querySelector(".cm-editor-help"));
    React.act(() => outside.focus());
    await delay();
    assert.equal(document.querySelector(".cm-editor-help"), null);
    assert.equal(clock.pending, 0);
    React.act(() => fixture.view.focus());
    fixture.key("F1");
    await delay(50);
    assert.ok(document.querySelector(".cm-editor-help"));
    React.act(() =>
      outside.dispatchEvent(new MouseEvent("pointerdown", { bubbles: true })),
    );
    assert.equal(document.querySelector(".cm-editor-help"), null);
    assert.equal(clock.pending, 0);
  } finally {
    outside.remove();
    fixture.close();
  }
});

test("Run, toolbar execution and file replacement dismiss help while provider rerenders preserve it", async (t) => {
  const clock = helpClock(t);
  const code = "import openecon as oe\noe.ols";
  const fixture = setup(code);
  const modifier = /Mac/.test(navigator.platform)
    ? { metaKey: true }
    : { ctrlKey: true };
  async function open() {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length - 1 } }),
    );
    fixture.key("F1");
    await delay(50);
    assert.ok(document.querySelector(".cm-editor-help"));
  }
  try {
    await open();
    fixture.key("Enter", modifier);
    assert.equal(document.querySelector(".cm-editor-help"), null);
    assert.deepEqual(fixture.runs, ["oe.ols"]);
    assert.equal(clock.pending, 0);
    await open();
    React.act(() => fixture.ref.current!.cancelInlineSuggestion());
    assert.equal(document.querySelector(".cm-editor-help"), null);
    assert.equal(clock.pending, 0);
    await open();
    fixture.update({ inlineSuggestionProvider: { request: async () => null } });
    assert.ok(
      document.querySelector(".cm-editor-help"),
      "Provider churn preserves the current reading period",
    );
    fixture.update({ inlineSuggestionContextKey: "qa:other.py" });
    assert.equal(document.querySelector(".cm-editor-help"), null);
    assert.equal(clock.pending, 0);
    await open();
    fixture.update({ readOnly: true });
    assert.equal(document.querySelector(".cm-editor-help"), null);
    fixture.update({ readOnly: false });
    await open();
    const replacement = "import openecon as oe\noe.ols(data=";
    fixture.update({ value: replacement });
    await delay(50);
    assert.equal(
      document.querySelector(".cm-editor-help"),
      null,
      "Replacement into a function call cannot resurrect stale help",
    );
    assert.equal(clock.pending, 0);
    assert.equal(fixture.view.state.doc.toString(), replacement);
  } finally {
    fixture.close();
  }
});

test("documentation expiry leaves the completion dropdown available and accepts its selection", async (t) => {
  const clock = helpClock(t);
  const code = "import openecon as oe\noe.ols";
  const fixture = setup(code);
  try {
    React.act(() =>
      fixture.view.dispatch({ selection: { anchor: code.length } }),
    );
    fixture.key("F1");
    await delay(50);
    assert.ok(document.querySelector(".cm-tooltip-hover .cm-editor-help"));
    React.act(() => startCompletion(fixture.view));
    await delay(180);
    assert.ok(document.querySelector(".cm-tooltip-autocomplete"));
    clock.advance(8000);
    assert.equal(
      document.querySelector(".cm-tooltip-hover .cm-editor-help"),
      null,
    );
    assert.ok(
      document.querySelector(".cm-tooltip-autocomplete"),
      "Completion remains interactive after documentation expires",
    );
    assert.ok(
      currentCompletions(fixture.view.state).some((row) => row.label === "ols"),
    );
    React.act(() => acceptCompletion(fixture.view));
    assert.match(fixture.view.state.doc.toString(), /oe\.ols(?:\(|$)/);
    assert.deepEqual(fixture.runs, []);
  } finally {
    fixture.close();
  }
});

test("hover documentation expires without discarding a ready inline suggestion", async (t) => {
  const clock = helpClock(t);
  const { provider, calls } = inlineProvider();
  const code = "import openecon as oe\noe.ols\nanswer =";
  const fixture = setup(code, { provider, contextKey: "qa:analysis.py" });
  try {
    typeInline(fixture);
    await waitInline(() => calls.length === 1);
    await React.act(async () => {
      calls[0].resolve("42");
    });
    await waitInline(
      () => !!fixture.host.querySelector(".cm-inline-suggestion"),
    );
    React.act(() =>
      activateHover(fixture.view, code.indexOf("oe.ols") + 4, -1),
    );
    await delay(50);
    assert.ok(document.querySelector(".cm-tooltip-hover .cm-editor-help"));
    clock.advance(8000);
    assert.equal(
      document.querySelector(".cm-tooltip-hover .cm-editor-help"),
      null,
    );
    assert.equal(
      fixture.host.querySelector(".cm-inline-suggestion")?.textContent,
      "42",
    );
    fixture.key("Tab");
    assert.equal(fixture.view.state.doc.toString(), `${code} 42`);
    assert.deepEqual(fixture.runs, []);
  } finally {
    fixture.close();
  }
});

test("Markdown and LaTeX modes highlight documents and disable Python run, help and local suggestions", async () => {
  const requests: string[] = [];
  const provider: InlineSuggestionProvider = {
    request(request) {
      requests.push(request.prefix);
      return "danger = 1";
    },
  };
  const fixture = setup(
    "# Report\n**Results**",
    { provider, contextKey: "document:notes" },
    "project:notes",
  );
  try {
    const original = fixture.view;
    fixture.update({ language: "markdown" });
    assert.equal(
      fixture.view.contentDOM.getAttribute("aria-label"),
      "File editor",
    );
    assert.equal(
      fixture.view.contentDOM.hasAttribute("aria-description"),
      false,
    );
    forceParsing(fixture.view, fixture.view.state.doc.length);
    await delay(80);
    const { syntaxTree } = await import("@codemirror/language");
    assert.match(syntaxTree(fixture.view.state).toString(), /ATXHeading1/);
    fixture.key("Enter", { ctrlKey: true });
    fixture.key("Enter", { ctrlKey: true, shiftKey: true });
    fixture.key("F1");
    React.act(() =>
      fixture.view.dispatch({
        changes: { from: fixture.view.state.doc.length, insert: "\nmore" },
      }),
    );
    await delay(400);
    assert.deepEqual(fixture.runs, []);
    assert.deepEqual(requests, []);
    assert.equal(document.querySelector(".cm-editor-help"), null);
    fixture.update({
      language: "latex",
      value: "\\section{Results}\n$\\beta = 1$",
    });
    forceParsing(fixture.view, fixture.view.state.doc.length);
    await delay(80);
    assert.match(syntaxTree(fixture.view.state).toString(), /Document/);
    assert.ok(
      fixture.view.contentDOM.querySelector("span"),
      "LaTeX tokens receive syntax styles",
    );
    fixture.key("Enter", { ctrlKey: true });
    assert.deepEqual(fixture.runs, []);
    fixture.update({ language: "python", value: "print(1)" });
    fixture.key("Enter", { ctrlKey: true });
    assert.deepEqual(fixture.runs, ["print(1)"]);
    assert.equal(
      fixture.view,
      original,
      "changing a filename's type reuses the same editor",
    );
  } finally {
    fixture.close();
  }
});
