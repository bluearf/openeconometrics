import assert from "node:assert/strict";
import test from "node:test";
import {
  clampPane,
  editorBounds,
  paneStorageKey,
  PANE_DEFAULTS,
  readPanePreference,
  savePanePreference,
  sidebarBounds,
  terminalBounds,
} from "../src/pane-preferences.ts";

test("pane preferences are isolated by project and by pane", () => {
  assert.notEqual(
    paneStorageKey("one", "sidebar"),
    paneStorageKey("two", "sidebar"),
  );
  assert.notEqual(
    paneStorageKey("one", "sidebar"),
    paneStorageKey("one", "terminal"),
  );
  assert.notEqual(
    paneStorageKey("local", "editor"),
    paneStorageKey("a:b", "editor"),
  );
});

test("malformed and extreme preferences never create an unusable pane", () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  let stored: string | null = null;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { localStorage: { getItem: () => stored } },
  });
  try {
    for (const bad of [
      "{",
      "null",
      '"300"',
      "true",
      "1e999",
      '{"value":300}',
    ]) {
      stored = bad;
      assert.equal(readPanePreference("test", "editor"), PANE_DEFAULTS.editor);
    }
    stored = "10000000";
    assert.equal(readPanePreference("test", "sidebar"), 560);
    assert.equal(readPanePreference("test", "terminal"), 1600);
    assert.equal(readPanePreference("test", "editor"), 0.95);
    stored = "-10000000";
    assert.equal(readPanePreference("test", "sidebar"), 180);
    assert.equal(readPanePreference("test", "editor"), 0.05);
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else Reflect.deleteProperty(globalThis, "window");
  }
});

test("blocked storage does not prevent reading defaults or changing pane sizes", () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    get() {
      throw new Error("Storage is unavailable");
    },
  });
  try {
    assert.equal(readPanePreference("test", "terminal"), 226);
    assert.doesNotThrow(() => savePanePreference("test", 350));
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else Reflect.deleteProperty(globalThis, "window");
  }
});

test("workspace limits reserve usable editor, results, and command input space", () => {
  for (const width of [1, 320, 520, 720, 1440, 10000]) {
    const sidebar = sidebarBounds(width);
    assert.ok(sidebar.maximum >= sidebar.minimum);
    assert.ok(sidebar.maximum <= 560);
    const editor = editorBounds(width);
    assert.ok(editor.minimum <= editor.maximum);
    assert.ok(editor.minimum > 0 && editor.maximum < 1);
    assert.equal(clampPane(-100, editor), editor.minimum);
    assert.equal(clampPane(100, editor), editor.maximum);
    if (width >= 566) {
      assert.ok(editor.minimum * (width - 6) >= 300 - 1e-9);
      assert.ok((1 - editor.maximum) * (width - 6) >= 260 - 1e-9);
    }
  }
  for (const height of [1, 160, 280, 600, 3000]) {
    const terminal = terminalBounds(height);
    assert.ok(terminal.minimum >= 80);
    assert.ok(terminal.minimum <= terminal.maximum);
    assert.ok(terminal.maximum <= 1600);
    if (height >= 260) assert.ok(height - terminal.maximum >= 180);
  }
});
