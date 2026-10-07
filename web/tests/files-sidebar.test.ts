import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { FilesSidebarProps } from "../src/FilesSidebar.tsx";
import type { FileLayoutEntry } from "../src/file-layout.ts";
import type { DatasetProfile } from "../src/types.ts";

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
  "HTMLSelectElement",
  "Element",
  "Node",
  "Event",
  "MouseEvent",
  "KeyboardEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
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
    (_whole, prefix, _quote, target: string) =>
      prefix +
      JSON.stringify(
        target.startsWith(".")
          ? moduleUrl(target.replace(/^\.\//, "").replace(/\.tsx?$/, ""))
          : pathToFileURL(require.resolve(target)).href,
      ),
  );
  const url =
    "data:text/javascript;base64," + Buffer.from(output).toString("base64");
  urls.set(name, url);
  return url;
}
const { default: FilesSidebar } = await import(moduleUrl("FilesSidebar"));
const folder = "a".repeat(32),
  scriptId = "b".repeat(32),
  dataId = "c".repeat(32);
const layout: FileLayoutEntry[] = [
  { kind: "script", id: "analysis", name: "analysis.py", parent: null },
  { kind: "folder", id: folder, name: "Modeller", parent: null },
  { kind: "script", id: scriptId, name: "model.py", parent: folder },
  { kind: "dataset", id: dataId, name: "wage.csv", parent: null },
];
const data = {
  id: dataId,
  name: "wage.csv",
  source: "upload",
} as DatasetProfile;
function setup(
  patch: Partial<FilesSidebarProps> = {},
  persist: (next: FileLayoutEntry[]) => Promise<void> = async () => {},
) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  let closed = false;
  const calls: FileLayoutEntry[][] = [],
    opened: string[] = [],
    created: { parent: string | null; name: string }[] = [];
  let props: FilesSidebarProps = {
    files: [data],
    scripts: [
      { id: "analysis", name: "analysis.py" },
      { id: scriptId, name: "model.py" },
    ],
    collapsed: false,
    onCollapsedChange: () => {},
    onOpen: (file) => opened.push(file.id),
    onOpenScript: (script) => opened.push(script.id),
    onCreateScript: (parent, name) => {
      created.push({ parent, name });
    },
    layout: structuredClone(layout),
    onLayoutChange: async (next) => {
      calls.push(next);
      const previous = props.layout;
      props = { ...props, layout: next };
      if (!closed) root.render(React.createElement(FilesSidebar, props));
      try {
        await persist(next);
      } catch (reason) {
        props = { ...props, layout: previous };
        if (!closed) root.render(React.createElement(FilesSidebar, props));
        throw reason;
      }
    },
    ...patch,
  };
  React.act(() => root.render(React.createElement(FilesSidebar, props)));
  const button = (label: string) => {
    const found = [...host.querySelectorAll("button")].find(
      (item) =>
        item.getAttribute("aria-label") === label ||
        item.textContent?.trim() === label,
    );
    assert.ok(found, `button ${label} exists`);
    return found;
  };
  return {
    host,
    calls,
    opened,
    created,
    button,
    async click(label: string) {
      await React.act(async () => button(label).click());
    },
    async context(label: string, clientX = 40, clientY = 160) {
      let event!: MouseEvent;
      await React.act(async () => {
        event = new dom.window.MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          button: 2,
          clientX,
          clientY,
        });
        button(label).closest(".files-sidebar__row")!.dispatchEvent(event);
      });
      return event;
    },
    async doubleClick(label: string) {
      await React.act(async () => {
        const target = button(label);
        for (const [type, detail] of [
          ["click", 1],
          ["click", 2],
          ["dblclick", 2],
        ] as const)
          target.dispatchEvent(
            new dom.window.MouseEvent(type, {
              bubbles: true,
              cancelable: true,
              detail,
            }),
          );
      });
    },
    async contextKey(label: string, key: string, shiftKey = false) {
      let event!: KeyboardEvent;
      await React.act(async () => {
        button(label).focus();
        event = new dom.window.KeyboardEvent("keydown", {
          key,
          shiftKey,
          bubbles: true,
          cancelable: true,
        });
        button(label).dispatchEvent(event);
      });
      return event;
    },
    async input(value: string) {
      await React.act(async () => {
        const input = host.querySelector("input")!;
        Object.getOwnPropertyDescriptor(
          dom.window.HTMLInputElement.prototype,
          "value",
        )!.set!.call(input, value);
        input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
      });
    },
    async select(value: string) {
      await React.act(async () => {
        const input = host.querySelector("select")!;
        input.value = value;
        input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
      });
    },
    async key(key: string) {
      await React.act(async () =>
        host
          .querySelector("input")!
          .dispatchEvent(
            new dom.window.KeyboardEvent("keydown", { key, bubbles: true }),
          ),
      );
    },
    async drag(
      source: string,
      target: string | null,
      position = 0.5,
      acceptance: "dragover" | "dragenter" = "dragover",
    ) {
      const values = new Map<string, string>();
      const transfer = {
        effectAllowed: "",
        dropEffect: "",
        setData(type: string, value: string) {
          values.set(type, value);
        },
        getData(type: string) {
          return values.get(type) ?? "";
        },
      };
      const dispatch = (node: Element, name: string) => {
        const event = new dom.window.MouseEvent(name, {
          bubbles: true,
          cancelable: true,
          clientY: 100 + position * 40,
        });
        Object.defineProperty(event, "dataTransfer", { value: transfer });
        node.dispatchEvent(event);
        return event;
      };
      let acceptancePrevented = false;
      await React.act(async () => dispatch(button(source), "dragstart"));
      await React.act(async () => {
        const destination = target
          ? button(target)
          : host.querySelector(".files-sidebar__content")!;
        if (target)
          Object.defineProperty(
            destination.closest(".files-sidebar__row")!,
            "getBoundingClientRect",
            {
              configurable: true,
              value: () => ({ top: 100, bottom: 140, height: 40 }),
            },
          );
        const priorCalls = calls.length;
        acceptancePrevented = dispatch(
          destination,
          acceptance,
        ).defaultPrevented;
        assert.equal(
          calls.length,
          priorCalls,
          "accepting a target never persists a move",
        );
        dispatch(destination, "drop");
      });
      await React.act(async () => dispatch(button(source), "dragend"));
      return { ...transfer, acceptancePrevented };
    },
    update(patch: Partial<FilesSidebarProps>) {
      props = { ...props, ...patch };
      React.act(() => root.render(React.createElement(FilesSidebar, props)));
    },
    close() {
      closed = true;
      React.act(() => root.unmount());
      host.remove();
    },
  };
}

test("folder rows expand and collapse without opening a Python or data file", async () => {
  const env = setup();
  try {
    assert.ok(env.button("model.py"));
    await env.click("Modeller");
    assert.equal(env.host.textContent?.includes("model.py"), false);
    await env.click("Modeller");
    assert.ok(env.button("model.py"));
    assert.deepEqual(env.opened, []);
    assert.equal(env.calls.length, 0);
    await env.click("New file");
    await env.input("new_model.py");
    await env.key("Enter");
    assert.deepEqual(env.created, [{ parent: folder, name: "new_model.py" }]);
  } finally {
    env.close();
  }
});

test("the menu renames main script, data and folder using their immutable typed identities", async () => {
  for (const [oldName, name, kind, id] of [
    ["analysis.py", "ücret.py", "script", "analysis"],
    ["wage.csv", "maaş.csv", "dataset", dataId],
    ["Modeller", "Analizler", "folder", folder],
  ]) {
    const env = setup();
    try {
      await env.context(oldName);
      await env.click("Rename");
      await env.input(name);
      await env.key("Enter");
      assert.equal(
        env.calls[0].find((entry) => entry.kind === kind && entry.id === id)
          ?.name,
        name,
      );
      assert.ok(env.button(name));
      assert.deepEqual(env.opened, []);
      assert.equal(env.host.querySelector("input"), null);
    } finally {
      env.close();
    }
  }
});

test("Escape cancels rename, Enter submits once and a failed save retains the draft", async () => {
  const env = setup({}, async () => {
    throw new Error("Başka bir kişi dosya düzenini değiştirdi.");
  });
  try {
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("mine.py");
    await env.key("Escape");
    assert.equal(env.calls.length, 0);
    assert.equal(env.host.querySelector("input"), null);
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("mine.py");
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "mine.py",
    );
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /Başka bir kişi/,
    );
    await env.key("Escape");
    assert.ok(env.button("analysis.py"));
  } finally {
    env.close();
  }
});

test("folder creation appears immediately and closes its form while local acceptance blocks duplicate mutation", async () => {
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  const env = setup({}, async () => pending);
  try {
    await env.click("New folder");
    await env.input("Veriler");
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
    assert.ok(env.button("Veriler"));
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.button("New folder").disabled, true);
    await env.click("New folder");
    await env.context("Veriler");
    assert.equal(env.calls.length, 1);
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector(".files-sidebar__menu"), null);
    await React.act(async () => release());
    assert.ok(env.button("Veriler"));
    assert.equal(env.button("New folder").disabled, false);
    assert.equal(env.calls[0].at(-1)?.kind, "folder");
    assert.match(env.calls[0].at(-1)!.id, /^[a-f0-9]{32}$/);
  } finally {
    env.close();
  }
});

test("folder creation in a collapsed selected folder expands it after a successful save", async () => {
  const env = setup();
  try {
    await env.click("Modeller");
    await env.click("New folder");
    await env.input("İç");
    await env.key("Enter");
    assert.equal(env.calls[0].at(-1)?.parent, folder);
    assert.ok(env.button("İç"));
  } finally {
    env.close();
  }
});

test("rename finishes its visible interaction immediately and repeated Enter accepts only one local write", async () => {
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  const env = setup({}, async () => pending);
  try {
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("sonuç.py");
    const input = env.host.querySelector("input")!;
    await React.act(async () => {
      for (let attempt = 0; attempt < 2; attempt++)
        input.dispatchEvent(
          new dom.window.KeyboardEvent("keydown", {
            key: "Enter",
            bubbles: true,
          }),
        );
    });
    assert.equal(env.calls.length, 1);
    assert.ok(env.button("sonuç.py"));
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector(".files-sidebar__menu"), null);
    assert.equal(env.button("sonuç.py").disabled, true);
    await React.act(async () => release());
    assert.equal(env.button("sonuç.py").disabled, false);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("a failed local folder save rolls back the row and restores the exact draft for a corrected retry", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  let attempts = 0;
  const env = setup({}, async () => {
    if (++attempts === 1) await pending;
  });
  try {
    await env.click("New folder");
    await env.input("Veriler");
    await env.key("Enter");
    assert.ok(env.button("Veriler"));
    assert.equal(env.host.querySelector("input"), null);
    await React.act(async () => reject(new Error("Yerel disk yazılamadı.")));
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Veriler",
    );
    assert.equal(
      [...env.host.querySelectorAll(".files-sidebar__file")].some(
        (button) => button.textContent === "Veriler",
      ),
      false,
    );
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /disk/,
    );
    await env.input("Veriler 2");
    await env.key("Enter");
    assert.equal(env.calls.length, 2);
    assert.ok(env.button("Veriler 2"));
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("moving into a collapsed folder appears immediately and a local failure restores the chosen destination", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  let attempts = 0;
  const env = setup({}, async () => {
    if (++attempts === 1) await pending;
  });
  try {
    await env.click("Modeller");
    await env.context("wage.csv");
    await env.click("Move");
    await env.select(folder);
    await env.click("Move");
    assert.equal(env.host.querySelector("select"), null);
    assert.equal(env.host.querySelector(".files-sidebar__menu"), null);
    assert.equal(env.button("Modeller").getAttribute("aria-expanded"), "true");
    assert.match(
      env
        .button("wage.csv")
        .closest("li")!
        .parentElement!.closest("li")!
        .querySelector(".files-sidebar__file")!.textContent!,
      /Modeller/,
    );
    await React.act(async () =>
      reject(new Error("Yerel taşıma kaydedilemedi.")),
    );
    assert.equal(
      env.host.querySelector<HTMLSelectElement>("select")?.value,
      folder,
    );
    assert.equal(
      env.button("wage.csv").closest("li")!.parentElement!.closest("li"),
      null,
    );
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /taşıma/,
    );
    await env.click("Move");
    assert.equal(env.calls.length, 2);
    assert.equal(
      env.calls[1].find((entry) => entry.id === dataId)?.parent,
      folder,
    );
    assert.equal(env.host.querySelector("select"), null);
  } finally {
    env.close();
  }
});

test("a failed ordering write restores the menu without leaving it open during local acceptance", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  const env = setup({}, async () => pending);
  try {
    await env.context("wage.csv");
    await env.click("Move up");
    assert.equal(env.host.querySelector(".files-sidebar__menu"), null);
    await React.act(async () => reject(new Error("Sıralama kaydedilemedi.")));
    assert.ok(env.button("Move up"));
    assert.equal(
      env.host.querySelector("[role=menu]")?.getAttribute("aria-label"),
      "Actions for wage.csv",
    );
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /Sıralama/,
    );
  } finally {
    env.close();
  }
});

test("data moves into a folder and can move back to the root", async () => {
  const env = setup();
  try {
    await env.context("wage.csv");
    await env.click("Move");
    await env.select(folder);
    await env.click("Move");
    assert.equal(
      env.calls[0].find((entry) => entry.id === dataId)?.parent,
      folder,
    );
    await env.context("wage.csv");
    await env.click("Move");
    await env.select("");
    await env.click("Move");
    assert.equal(
      env.calls[1].find((entry) => entry.id === dataId)?.parent,
      null,
    );
  } finally {
    env.close();
  }
});

test("move destinations exclude self and descendants and ordering keeps nested items in place", async () => {
  const nestedId = "d".repeat(32);
  const env = setup({
    layout: [
      ...layout,
      { kind: "folder", id: nestedId, name: "İç", parent: folder },
    ],
  });
  try {
    await env.context("Modeller");
    await env.click("Move");
    assert.deepEqual(
      [...env.host.querySelectorAll("option")].map((option) => option.value),
      [""],
    );
    await env.click("Cancel");
    await env.context("wage.csv");
    await env.click("Move up");
    assert.deepEqual(
      env.calls[0]
        .filter((entry) => entry.parent === null)
        .map((entry) => entry.name),
      ["analysis.py", "wage.csv", "Modeller"],
    );
    assert.equal(
      env.calls[0].find((entry) => entry.id === scriptId)?.parent,
      folder,
    );
  } finally {
    env.close();
  }
});

test("read-only and busy sidebars prevent mutation while older callers keep the existing file actions", async () => {
  const env = setup({ readOnly: true });
  try {
    assert.equal(
      [...env.host.querySelectorAll("button")].some((button) =>
        button.getAttribute("aria-label")?.includes("options"),
      ),
      false,
    );
    assert.equal(env.host.querySelector("[aria-label=\"New folder\"]"), null);
    await env.click("Modeller");
    assert.equal(env.calls.length, 0);
    env.update({ readOnly: false, disabled: true });
    assert.equal(env.button("New folder").disabled, true);
    assert.equal(env.button("analysis.py").disabled, true);
    env.update({ disabled: false, uploading: true });
    assert.equal(env.button("New folder").disabled, true);
    assert.equal(env.button("analysis.py").disabled, true);
    env.update({
      disabled: false,
      uploading: false,
      layout: undefined,
      onLayoutChange: undefined,
    });
    assert.equal(env.host.querySelector("[aria-label=\"New folder\"]"), null);
    await env.click("analysis.py");
    assert.deepEqual(env.opened, ["analysis"]);
  } finally {
    env.close();
  }
});

test("native drag reorders before files, moves into folders, returns to root and rejects cycles", async () => {
  const env = setup();
  try {
    await env.drag("wage.csv", "analysis.py");
    assert.equal(env.calls[0][0].id, dataId);
    await env.click("Modeller");
    await env.drag("wage.csv", "Modeller");
    assert.equal(
      env.calls[1].find((entry) => entry.id === dataId)?.parent,
      folder,
    );
    assert.ok(env.button("model.py"));
    await env.drag("wage.csv", null);
    assert.equal(
      env.calls[2].find((entry) => entry.id === dataId)?.parent,
      null,
    );
    await env.drag("Modeller", "model.py");
    assert.equal(env.calls.length, 3);
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /into itself/,
    );
  } finally {
    env.close();
  }
});

test("native button drags can place before or after folders without nesting into them", async () => {
  const env = setup();
  try {
    assert.equal(env.button("wage.csv").draggable, true);
    const transfer = await env.drag("wage.csv", "Modeller", 0.1);
    assert.equal(
      transfer.getData("application/x-openecon-file"),
      `dataset:${dataId}`,
    );
    assert.deepEqual(
      env.calls[0]
        .filter((entry) => entry.parent === null)
        .map((entry) => entry.name),
      ["analysis.py", "wage.csv", "Modeller"],
    );
    await env.drag("analysis.py", "Modeller", 0.9);
    assert.deepEqual(
      env.calls[1]
        .filter((entry) => entry.parent === null)
        .map((entry) => entry.name),
      ["wage.csv", "Modeller", "analysis.py"],
    );
    assert.equal(
      env.calls[1].find((entry) => entry.id === scriptId)?.parent,
      folder,
    );
    assert.deepEqual(env.opened, []);
  } finally {
    env.close();
  }
});

test("dropping on a file's lower half inserts after the final sibling", async () => {
  const env = setup();
  try {
    await env.drag("analysis.py", "wage.csv", 0.8);
    assert.deepEqual(
      env.calls[0]
        .filter((entry) => entry.parent === null)
        .map((entry) => entry.name),
      ["Modeller", "wage.csv", "analysis.py"],
    );
    assert.deepEqual(env.opened, []);
  } finally {
    env.close();
  }
});

test("dropping a nested file on itself is a strict no-op and does not bubble to the root", async () => {
  const env = setup();
  try {
    const transfer = await env.drag("model.py", "model.py");
    assert.equal(env.calls.length, 0);
    assert.equal(transfer.dropEffect, "none");
    assert.equal(
      env
        .button("model.py")
        .closest("li")!
        .parentElement!.closest("li")!
        .querySelector(".files-sidebar__file")!.textContent,
      "Modeller",
    );
    assert.deepEqual(env.opened, []);
  } finally {
    env.close();
  }
});

test("a fast dragenter then drop accepts rows and root while self entry remains a no-op", async () => {
  const env = setup();
  try {
    const folderEnter = await env.drag(
      "wage.csv",
      "Modeller",
      0.5,
      "dragenter",
    );
    assert.equal(folderEnter.acceptancePrevented, true);
    assert.equal(
      env.calls[0].find((entry) => entry.id === dataId)?.parent,
      folder,
    );
    const afterEnter = await env.drag("wage.csv", "model.py", 0.9, "dragenter");
    assert.equal(afterEnter.acceptancePrevented, true);
    assert.deepEqual(
      env.calls[1]
        .filter((entry) => entry.parent === folder)
        .map((entry) => entry.name),
      ["model.py", "wage.csv"],
    );
    const selfEnter = await env.drag("wage.csv", "wage.csv", 0.5, "dragenter");
    assert.equal(selfEnter.acceptancePrevented, false);
    assert.equal(selfEnter.dropEffect, "none");
    assert.equal(env.calls.length, 2);
    const rootEnter = await env.drag("wage.csv", null, 0.5, "dragenter");
    assert.equal(rootEnter.acceptancePrevented, true);
    assert.equal(
      env.calls[2].find((entry) => entry.id === dataId)?.parent,
      null,
    );
    assert.deepEqual(env.opened, []);
  } finally {
    env.close();
  }
});

test("external file and text drops cannot move project entries or capture the drop", async () => {
  const env = setup();
  try {
    for (const target of [
      env.button("Modeller"),
      env.host.querySelector(".files-sidebar__content")!,
    ]) {
      for (const type of [
        "application/x-openecon-file",
        "text/plain",
        "Files",
      ]) {
        const transfer = {
          types: [type],
          dropEffect: "",
          getData: () => `script:${scriptId}`,
        };
        for (const name of ["dragenter", "dragover", "drop"]) {
          const event = new dom.window.MouseEvent(name, {
            bubbles: true,
            cancelable: true,
          });
          Object.defineProperty(event, "dataTransfer", { value: transfer });
          await React.act(async () => target.dispatchEvent(event));
          assert.equal(event.defaultPrevented, false);
        }
      }
    }
    assert.equal(env.calls.length, 0);
    assert.equal(env.host.querySelector('[class*="--drop"]'), null);
  } finally {
    env.close();
  }
});

test("sidebar file drops outside the tree cannot insert filenames into editors or inputs", async () => {
  const env = setup();
  const editor = document.createElement("div");
  editor.className = "cm-content";
  editor.contentEditable = "true";
  editor.textContent = "print('unchanged')";
  const input = document.createElement("input");
  input.value = "unchanged.py";
  document.body.append(editor, input);
  const values = new Map<string, string>();
  const transfer = {
    dropEffect: "",
    effectAllowed: "",
    setData: (type: string, value: string) => values.set(type, value),
    getData: (type: string) => values.get(type) ?? "",
  };
  const dispatch = async (target: Element, name: string) => {
    const event = new dom.window.MouseEvent(name, {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(event, "dataTransfer", { value: transfer });
    await React.act(async () => target.dispatchEvent(event));
    return event;
  };
  let editorDrops = 0;
  editor.addEventListener("drop", () => {
    editorDrops++;
    editor.textContent += transfer.getData("text/plain");
  });
  input.addEventListener("drop", () => {
    editorDrops++;
    input.value += transfer.getData("text/plain");
  });
  try {
    for (const target of [editor, input]) {
      await dispatch(env.button("wage.csv"), "dragstart");
      assert.equal(transfer.getData("text/plain"), "wage.csv");
      assert.equal(
        (await dispatch(target, "dragenter")).defaultPrevented,
        true,
      );
      assert.equal((await dispatch(target, "dragover")).defaultPrevented, true);
      assert.equal((await dispatch(target, "drop")).defaultPrevented, true);
      assert.equal(transfer.dropEffect, "none");
      assert.equal(env.calls.length, 0);
      // An outside drop cancels the file move; a later tree drop cannot revive it.
      await dispatch(
        env.host.querySelector(".files-sidebar__content")!,
        "drop",
      );
      assert.equal(env.calls.length, 0);
    }
    assert.equal(editorDrops, 0);
    assert.equal(editor.textContent, "print('unchanged')");
    assert.equal(input.value, "unchanged.py");
    // Ordinary text or external file drops still reach the editor when no sidebar drag is active.
    const external = await dispatch(editor, "drop");
    assert.equal(external.defaultPrevented, false);
    assert.equal(editorDrops, 1);
  } finally {
    env.close();
    editor.remove();
    input.remove();
  }
});

test("a native click immediately after a drag cannot open an editor or toggle the target folder", async () => {
  const env = setup();
  const originalNow = Date.now;
  let now = originalNow();
  Date.now = () => now;
  try {
    await env.drag("analysis.py", "wage.csv", 0.8);
    await env.click("wage.csv");
    await env.click("Modeller");
    assert.deepEqual(env.opened, []);
    assert.equal(env.button("Modeller").getAttribute("aria-expanded"), "true");
    now += 251;
    await env.click("wage.csv");
    assert.deepEqual(env.opened, [dataId]);
    await env.click("Modeller");
    assert.equal(env.button("Modeller").getAttribute("aria-expanded"), "false");
  } finally {
    Date.now = originalNow;
    env.close();
  }
});

test("a parent busy update during a failed rename does not erase the unsaved name", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  const env = setup({}, async () => pending);
  try {
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("important.py");
    await env.key("Enter");
    env.update({ disabled: true });
    assert.ok(env.button("important.py"));
    assert.equal(env.host.querySelector("input"), null);
    await React.act(async () => reject(new Error("Kaydetme başarısız.")));
    env.update({ disabled: false });
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "important.py",
    );
    assert.match(
      env.host.querySelector('[role="alert"]')!.textContent!,
      /başarısız/,
    );
    await env.key("Escape");
    assert.ok(env.button("analysis.py"));
  } finally {
    env.close();
  }
});

test("invalid names never invoke persistence and the user can correct the input", async () => {
  const env = setup();
  try {
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("bad .py");
    await env.key("Enter");
    assert.equal(env.calls.length, 0);
    assert.ok(env.host.querySelector('[role="alert"]'));
    await env.input("good.py");
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
    assert.ok(env.button("good.py"));
  } finally {
    env.close();
  }
});

test("a late failed write cannot revive a dismissed rename after editing permission changes", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  const env = setup({}, async () => pending);
  try {
    await env.context("analysis.py");
    await env.click("Rename");
    await env.input("mine.py");
    await env.key("Enter");
    env.update({ readOnly: true });
    env.update({ readOnly: false });
    await React.act(async () =>
      reject(new Error("Eski yerel kayıt başarısız.")),
    );
    assert.ok(env.button("analysis.py"));
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector("select"), null);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
    await env.click("New folder");
    assert.equal(env.host.querySelector<HTMLInputElement>("input")?.value, "");
  } finally {
    env.close();
  }
});

test("an unmounted sidebar ignores a late local failure", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  const env = setup({}, async () => pending);
  await env.click("New folder");
  await env.input("Geçici");
  await env.key("Enter");
  env.close();
  await React.act(async () => reject(new Error("Geç gelen hata.")));
  assert.equal(env.host.textContent, "");
});

test("offline layout status appears only while cloud synchronization is pending", () => {
  const env = setup();
  try {
    assert.equal(env.host.querySelector(".files-sidebar__pending"), null);
    env.update({ pendingSync: true });
    const badge = env.host.querySelector(".files-sidebar__pending")!;
    assert.equal(badge.textContent, "Local");
    assert.equal(
      badge.getAttribute("title"),
      "The file layout has not been saved to the cloud yet.",
    );
    env.update({ pendingSync: false });
    assert.equal(env.host.querySelector(".files-sidebar__pending"), null);
  } finally {
    env.close();
  }
});

function tooltipClock() {
  const originalSetTimeout = window.setTimeout;
  const originalClearTimeout = window.clearTimeout;
  let now = 0,
    nextId = 0;
  const timers = new Map<number, { deadline: number; callback: () => void }>();
  window.setTimeout = ((callback: () => void, delay = 0) => {
    const id = ++nextId;
    timers.set(id, { deadline: now + delay, callback });
    return id;
  }) as typeof window.setTimeout;
  window.clearTimeout = (id) => {
    timers.delete(id);
  };
  return {
    timers,
    async advance(milliseconds: number) {
      now += milliseconds;
      await React.act(async () => {
        for (const [id, timer] of [...timers]) {
          if (timer.deadline > now) continue;
          timers.delete(id);
          timer.callback();
        }
      });
    },
    restore() {
      window.setTimeout = originalSetTimeout;
      window.clearTimeout = originalClearTimeout;
    },
  };
}
function tooltipFor(button: HTMLButtonElement): HTMLElement {
  return button
    .closest(".files-sidebar__action")!
    .querySelector('[role="tooltip"]')!;
}
async function hoverAction(button: HTMLButtonElement, enter = true) {
  await React.act(async () =>
    button.dispatchEvent(
      new dom.window.MouseEvent(enter ? "mouseover" : "mouseout", {
        bubbles: true,
        relatedTarget: enter ? null : document.body,
      }),
    ),
  );
}

test("sidebar informational tooltip expires after four visible seconds without native-title leftovers", async () => {
  const clock = tooltipClock(),
    env = setup();
  try {
    const button = env.button("New file"),
      tooltip = tooltipFor(button);
    assert.equal(tooltip.hidden, true);
    assert.equal(button.hasAttribute("title"), false);
    assert.equal(button.getAttribute("aria-label"), "New file");
    await hoverAction(button);
    assert.equal(tooltip.hidden, false);
    assert.equal(button.getAttribute("aria-describedby"), tooltip.id);
    assert.equal(clock.timers.size, 1);
    await clock.advance(3999);
    assert.equal(tooltip.hidden, false);
    env.update({ pendingSync: true });
    await clock.advance(1);
    assert.equal(tooltip.hidden, true);
    assert.equal(button.hasAttribute("aria-describedby"), false);
    assert.equal(clock.timers.size, 0);
    await hoverAction(button);
    await React.act(async () => button.focus());
    assert.equal(
      tooltip.hidden,
      true,
      "the same hover/focus session cannot reopen an expired tooltip",
    );
    await hoverAction(button, false);
    await React.act(async () => button.blur());
    await hoverAction(button);
    assert.equal(tooltip.hidden, false);
  } finally {
    env.close();
    clock.restore();
  }
});

test("keyboard tooltip expires and reappears only after leaving and returning focus", async () => {
  const clock = tooltipClock(),
    env = setup();
  try {
    const button = env.button("New folder"),
      tooltip = tooltipFor(button);
    await React.act(async () => button.focus());
    assert.equal(tooltip.hidden, false);
    await clock.advance(4000);
    assert.equal(tooltip.hidden, true);
    await React.act(async () => button.focus());
    assert.equal(tooltip.hidden, true);
    await React.act(async () => button.blur());
    await React.act(async () => button.focus());
    assert.equal(tooltip.hidden, false);
    await React.act(async () =>
      button.dispatchEvent(
        new dom.window.KeyboardEvent("keydown", {
          key: "Escape",
          bubbles: true,
        }),
      ),
    );
    assert.equal(tooltip.hidden, true);
    assert.equal(clock.timers.size, 0);
    await React.act(async () => button.focus());
    assert.equal(tooltip.hidden, true);
  } finally {
    env.close();
    clock.restore();
  }
});

test("click, pointer exit and disabled state immediately dismiss the tooltip without timing out rename forms", async () => {
  const clock = tooltipClock(),
    env = setup();
  try {
    const button = env.button("New folder"),
      tooltip = tooltipFor(button);
    await hoverAction(button);
    await env.click("New folder");
    assert.equal(tooltip.hidden, true);
    assert.equal(clock.timers.size, 0);
    assert.ok(env.host.querySelector("input"));
    await clock.advance(10000);
    assert.ok(
      env.host.querySelector("input"),
      "rename/create form has no tooltip timeout",
    );
    await env.key("Escape");
    await hoverAction(button, false);
    await hoverAction(button);
    assert.equal(tooltip.hidden, false);
    await hoverAction(button, false);
    assert.equal(tooltip.hidden, true);
    assert.equal(clock.timers.size, 0);
    await hoverAction(button);
    env.update({ disabled: true });
    assert.equal(tooltip.hidden, true);
    assert.equal(clock.timers.size, 0);
    env.update({ disabled: false });
    assert.equal(tooltip.hidden, true);
    await hoverAction(button);
    assert.equal(tooltip.hidden, true);
    await hoverAction(button, false);
    await hoverAction(button);
    assert.equal(tooltip.hidden, false);
  } finally {
    env.close();
    clock.restore();
  }
});

test("a pointer exit closes even a focused tooltip and all pending timers clean up on unmount", async () => {
  const clock = tooltipClock(),
    env = setup();
  let closed = false;
  try {
    const button = env.button("New file"),
      tooltip = tooltipFor(button);
    await hoverAction(button);
    await React.act(async () => button.focus());
    assert.equal(tooltip.hidden, false);
    await hoverAction(button, false);
    assert.equal(tooltip.hidden, true);
    assert.equal(clock.timers.size, 0);
    await React.act(async () => button.blur());
    await hoverAction(button);
    assert.equal(clock.timers.size, 1);
    env.close();
    closed = true;
    assert.equal(clock.timers.size, 0);
    await clock.advance(4000);
  } finally {
    if (!closed) env.close();
    clock.restore();
  }
});

test("file action menus remain available after the informational tooltip duration", async () => {
  const clock = tooltipClock(),
    env = setup();
  try {
    await env.context("analysis.py");
    await clock.advance(10000);
    assert.ok(env.button("Rename"));
    assert.ok(env.button("Move"));
    assert.equal(
      env.host.querySelector("[role=menu]")?.getAttribute("aria-label"),
      "Actions for analysis.py",
    );
  } finally {
    env.close();
    clock.restore();
  }
});

test("new file opens one inline explorer draft and only Enter creates a normalized named document", async () => {
  for (const [typed, name] of [
    ["analysis_new", "analysis_new.py"],
    ["README.MD", "README.md"],
    ["table.TEX", "table.tex"],
    ["model.PY", "model.py"],
    ["u\u0308cret.md", "ücret.md"],
  ]) {
    const env = setup();
    try {
      await env.click("New file");
      assert.deepEqual(env.created, []);
      assert.equal(env.host.querySelector('[role="menu"]'), null);
      assert.equal(env.host.querySelectorAll("input").length, 1);
      assert.equal(
        env.host.querySelector("input")?.getAttribute("aria-label"),
        "File name",
      );
      assert.ok(
        env.host.querySelector(
          ".files-sidebar__list .files-sidebar__row input",
        ),
      );
      assert.equal(env.host.querySelector('[aria-label="Save name"]'), null);
      assert.equal(env.host.querySelector("[aria-label=\"Cancel\"]"), null);
      await env.input(typed);
      await env.key("Enter");
      assert.deepEqual(env.created, [{ parent: null, name }]);
      assert.equal(env.host.querySelector("input"), null);
    } finally {
      env.close();
    }
  }
});

test("blank, unsafe, unsupported and sibling duplicate file names never create or discard the draft", async () => {
  const env = setup();
  try {
    await env.click("New file");
    for (const name of [
      "",
      " ",
      " notes.md",
      "notes.md ",
      "../notes.md",
      "notes.txt",
      "analysis.PY",
      "WAGE.csv",
    ]) {
      await env.input(name);
      await env.key("Enter");
      assert.deepEqual(env.created, []);
      assert.equal(
        env.host.querySelector<HTMLInputElement>("input")?.value,
        name,
      );
      assert.ok(env.host.querySelector('[role="alert"]'));
      await React.act(() =>
        env.host.querySelector<HTMLInputElement>("input")!.blur(),
      );
      assert.equal(
        env.host.querySelector<HTMLInputElement>("input")?.value,
        name,
        "invalid draft survives blur",
      );
    }
    await env.input("notes.md");
    await env.key("Enter");
    assert.deepEqual(env.created, [{ parent: null, name: "notes.md" }]);
  } finally {
    env.close();
  }
});

test("new document names cannot collide with case-folded sibling folder aliases", async () => {
  const env = setup({
    layout: layout.map((entry) =>
      entry.kind === "folder" ? { ...entry, name: "NOTES.md" } : entry,
    ),
  });
  try {
    await env.click("New file");
    await env.input("notes.MD");
    await env.key("Enter");
    assert.deepEqual(env.created, []);
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "notes.MD",
    );
    assert.match(
      env.host.querySelector('[role="alert"]')?.textContent ?? "",
      /already contains/,
    );
  } finally {
    env.close();
  }
});

test("Escape and outside blur cancel new-file drafts without creating automatic untitled files", async () => {
  const env = setup();
  try {
    await env.click("New file");
    await env.input("notes.md");
    await env.key("Escape");
    assert.equal(env.host.querySelector("input"), null);
    assert.deepEqual(env.created, []);
    await env.click("New file");
    await env.input("appendix.tex");
    await React.act(() =>
      env.host.querySelector<HTMLInputElement>("input")!.blur(),
    );
    assert.equal(env.host.querySelector("input"), null);
    assert.deepEqual(env.created, []);
  } finally {
    env.close();
  }
});

test("pending file creation submits once and a rejection restores the exact draft for corrected retry", async () => {
  let reject!: (reason: Error) => void;
  const pending = new Promise<void>((_resolve, fail) => {
    reject = fail;
  });
  const calls: { parent: string | null; name: string }[] = [];
  const env = setup({
    onCreateScript: (parent, name) => {
      calls.push({ parent, name });
      return pending;
    },
  });
  try {
    await env.click("New file");
    await env.input("notes.MD");
    await env.key("Enter");
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.disabled,
      true,
    );
    await env.key("Enter");
    await env.key("Escape");
    await React.act(() =>
      env.host.querySelector<HTMLInputElement>("input")!.blur(),
    );
    assert.equal(calls.length, 1);
    await React.act(async () => reject(new Error("Local save failed")));
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "notes.MD",
    );
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.disabled,
      false,
    );
    assert.equal(
      document.activeElement,
      env.host.querySelector("input"),
      "failed creation restores keyboard focus for retry",
    );
    assert.match(
      env.host.querySelector('[role="alert"]')?.textContent ?? "",
      /Local save failed/,
    );
    env.update({
      onCreateScript: (parent, name) => {
        calls.push({ parent, name });
      },
    });
    await env.input("fixed.tex");
    await env.key("Enter");
    assert.deepEqual(calls, [
      { parent: null, name: "notes.md" },
      { parent: null, name: "fixed.tex" },
    ]);
    assert.equal(env.host.querySelector("input"), null);
  } finally {
    env.close();
  }
});

test("new file drafts belong to the selected folder and opening a draft expands a closed parent", async () => {
  const env = setup();
  try {
    await env.click("Modeller");
    assert.equal(env.host.textContent?.includes("model.py"), false);
    await env.click("New file");
    const input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.ok(
      input
        .closest("ul")
        ?.parentElement?.classList.contains("files-sidebar__item"),
      "draft is inside the folder's list",
    );
    assert.ok(env.button("model.py"));
    await env.input("notes.md");
    await env.key("Enter");
    assert.deepEqual(env.created, [{ parent: folder, name: "notes.md" }]);
  } finally {
    env.close();
  }
});

test("F2 opens compact rename, selects only the basename and blur never commits a name", async () => {
  const env = setup();
  try {
    await React.act(() => {
      env.button("model.py").dispatchEvent(
        new dom.window.KeyboardEvent("keydown", {
          key: "F2",
          bubbles: true,
          cancelable: true,
        }),
      );
    });
    let input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(input.value, "model.py");
    assert.equal(input.selectionStart, 0);
    assert.equal(input.selectionEnd, "model".length);
    await env.input("notes.md");
    await React.act(() => input.blur());
    assert.equal(env.calls.length, 0);
    assert.equal(env.host.querySelector("input"), null);
    await env.context("model.py");
    await env.click("Rename");
    await env.input("../invalid.md");
    await env.key("Enter");
    input = env.host.querySelector<HTMLInputElement>("input")!;
    await React.act(() => input.blur());
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "../invalid.md",
    );
    assert.equal(env.calls.length, 0);
    await env.input("notes.md");
    await env.key("Enter");
    assert.equal(
      env.calls[0].find((entry) => entry.id === scriptId)?.name,
      "notes.md",
    );
  } finally {
    env.close();
  }
});

test("double-clicking a filename opens inline rename and the second click never opens it twice", async () => {
  for (const name of ["analysis.py", "model.py", "wage.csv", "Modeller"]) {
    const env = setup({ activeScriptId: scriptId });
    try {
      await env.doubleClick(name);
      const input = env.host.querySelector<HTMLInputElement>("input")!;
      assert.equal(input.value, name);
      assert.equal(document.activeElement, input);
      assert.equal(
        input.selectionEnd,
        name.includes(".") ? name.lastIndexOf(".") : name.length,
      );
      assert.equal(env.opened.length, name === "Modeller" ? 0 : 1);
      assert.equal(env.calls.length, 0);
      await env.key("Escape");
      assert.ok(env.button(name));
      assert.equal(env.calls.length, 0);
    } finally {
      env.close();
    }
  }
});

test("right-click menus replace adjacent row actions without opening files or toggling folders", async () => {
  const downloaded: DatasetProfile[] = [];
  const env = setup({
    onDownload: (file) => {
      downloaded.push(file);
    },
  });
  try {
    assert.equal(env.host.querySelector(".files-sidebar__actions"), null);
    assert.equal(env.host.querySelector(".files-sidebar__download"), null);
    assert.equal(env.host.querySelector('[aria-label$="options"]'), null);
    assert.equal((await env.context("Modeller")).defaultPrevented, true);
    assert.equal(env.button("Modeller").getAttribute("aria-expanded"), "true");
    assert.deepEqual(env.opened, []);
    await env.context("wage.csv");
    const menu = env.host.querySelector('[role="menu"]')!;
    assert.equal(menu.getAttribute("aria-label"), "Actions for wage.csv");
    assert.deepEqual(
      [...menu.querySelectorAll('[role="menuitem"]')].map((item) =>
        item.textContent?.trim(),
      ),
      [
        "Rename",
        "Move",
        "Move up",
        "Move down",
        "Download file",
      ],
    );
    assert.equal(document.activeElement, env.button("Rename"));
    await env.click("Download file");
    assert.equal(downloaded[0].id, dataId);
    assert.deepEqual(env.opened, []);
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    assert.equal(document.activeElement, env.button("wage.csv"));
  } finally {
    env.close();
  }
});

test("keyboard context menus navigate enabled items and Escape restores the originating filename focus", async () => {
  const env = setup();
  try {
    for (const [key, shift] of [
      ["ContextMenu", false],
      ["F10", true],
    ] as const) {
      assert.equal(
        (await env.contextKey("analysis.py", key, shift)).defaultPrevented,
        true,
      );
      assert.deepEqual(env.opened, []);
      assert.equal(document.activeElement, env.button("Rename"));
      const dispatch = async (key: string) => {
        await React.act(async () =>
          document.activeElement!.dispatchEvent(
            new dom.window.KeyboardEvent("keydown", {
              key,
              bubbles: true,
              cancelable: true,
            }),
          ),
        );
      };
      await dispatch("ArrowDown");
      assert.equal(document.activeElement, env.button("Move"));
      await dispatch("ArrowDown");
      assert.equal(
        document.activeElement,
        env.button("Move down"),
        "disabled upward item is skipped",
      );
      await dispatch("Home");
      assert.equal(document.activeElement, env.button("Rename"));
      await dispatch("End");
      assert.equal(document.activeElement, env.button("Move down"));
      await dispatch("Escape");
      assert.equal(env.host.querySelector('[role="menu"]'), null);
      assert.equal(document.activeElement, env.button("analysis.py"));
    }
  } finally {
    env.close();
  }
});

test("outside clicks, scrolling and window resize dismiss the menu without taking focus from the clicked target", async () => {
  const env = setup();
  const outside = document.createElement("button");
  document.body.append(outside);
  try {
    await env.context("analysis.py");
    await React.act(async () => {
      outside.focus();
      outside.dispatchEvent(
        new dom.window.MouseEvent("mousedown", { bubbles: true }),
      );
    });
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    assert.equal(document.activeElement, outside);
    await env.context("analysis.py");
    await React.act(async () =>
      env.host
        .querySelector(".files-sidebar__content")!
        .dispatchEvent(new dom.window.Event("scroll")),
    );
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    await env.context("analysis.py");
    await React.act(async () =>
      window.dispatchEvent(new dom.window.Event("resize")),
    );
    assert.equal(env.host.querySelector('[role="menu"]'), null);
  } finally {
    outside.remove();
    env.close();
  }
});

test("context menu positions remain within the visible sidebar and viewport, including repeated right-clicks", async () => {
  const env = setup();
  const content = env.host.querySelector(".files-sidebar__content")!;
  Object.defineProperty(content, "getBoundingClientRect", {
    configurable: true,
    value: () => ({
      left: 100,
      right: 320,
      width: 220,
      top: 90,
      bottom: 600,
      height: 510,
    }),
  });
  try {
    await env.context("analysis.py", 2000, 2000);
    let menu = env.host.querySelector<HTMLElement>('[role="menu"]')!;
    assert.equal(Number.parseFloat(menu.style.left), 165);
    assert.ok(
      Number.parseFloat(menu.style.top) + 150 <= window.innerHeight - 8,
    );
    await env.context("analysis.py", -200, -200);
    menu = env.host.querySelector<HTMLElement>('[role="menu"]')!;
    assert.equal(Number.parseFloat(menu.style.left), 100);
    assert.equal(Number.parseFloat(menu.style.top), 8);
  } finally {
    env.close();
  }
});

test("readonly datasets retain keyboard download access while rename and busy mutation stay blocked", async () => {
  const downloaded: DatasetProfile[] = [];
  const env = setup({
    readOnly: true,
    onDownload: (file) => {
      downloaded.push(file);
    },
  });
  try {
    await env.doubleClick("analysis.py");
    assert.equal(env.host.querySelector("input"), null);
    await env.context("analysis.py");
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    await env.contextKey("wage.csv", "ContextMenu");
    assert.deepEqual(
      [...env.host.querySelectorAll('[role="menuitem"]')].map((item) =>
        item.textContent?.trim(),
      ),
      ["Download file"],
    );
    await env.click("Download file");
    assert.equal(downloaded.length, 1);
    assert.deepEqual(
      env.opened,
      ["analysis"],
      "readonly dataset was not opened",
    );
    env.update({ readOnly: false, disabled: true });
    await env.doubleClick("analysis.py");
    await env.context("analysis.py");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    assert.equal(env.calls.length, 0);
  } finally {
    env.close();
  }
});

test("filename input text context menus leave invalid drafts and normal copy access intact", async () => {
  const env = setup();
  try {
    await env.doubleClick("analysis.py");
    await env.input("../draft.py");
    await env.key("Enter");
    const input = env.host.querySelector<HTMLInputElement>("input")!;
    let event!: MouseEvent;
    await React.act(async () => {
      event = new dom.window.MouseEvent("contextmenu", {
        bubbles: true,
        cancelable: true,
        button: 2,
      });
      input.dispatchEvent(event);
    });
    assert.equal(event.defaultPrevented, false);
    assert.equal(input.value, "../draft.py");
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    assert.equal(env.calls.length, 0);
  } finally {
    env.close();
  }
});

test("an inactive filename double-click waits for its navigation then renames that typed identity without a second open", async () => {
  let env!: ReturnType<typeof setup>;
  const opened: string[] = [];
  env = setup({
    activeScriptId: "analysis",
    onOpenScript: (script) => {
      opened.push(script.id);
      env.update({ disabled: true, navigationPending: true });
    },
  });
  try {
    const target = env.button("model.py");
    await React.act(async () =>
      target.dispatchEvent(
        new dom.window.MouseEvent("click", { bubbles: true, detail: 1 }),
      ),
    );
    assert.deepEqual(opened, [scriptId]);
    assert.equal(target.disabled, false);
    assert.equal(target.getAttribute("aria-disabled"), "true");
    assert.equal(target.draggable, false);
    assert.equal(env.button("New file").disabled, true);
    await React.act(async () => {
      target.dispatchEvent(
        new dom.window.MouseEvent("click", { bubbles: true, detail: 2 }),
      );
      target.dispatchEvent(
        new dom.window.MouseEvent("dblclick", { bubbles: true, detail: 2 }),
      );
    });
    assert.deepEqual(opened, [scriptId]);
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.calls.length, 0);
    await env.context("analysis.py");
    assert.equal(env.host.querySelector('[role="menu"]'), null);
    env.update({
      disabled: false,
      navigationPending: false,
      activeScriptId: scriptId,
      layout: layout.map((entry) =>
        entry.id === scriptId ? { ...entry, name: "incoming.py" } : entry,
      ),
    });
    const input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(
      input.value,
      "incoming.py",
      "the pending target is looked up by immutable key after loading",
    );
    assert.equal(document.activeElement, input);
    assert.equal(env.calls.length, 0);
    await env.input("renamed.py");
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
    assert.equal(
      env.calls[0].find((entry) => entry.id === scriptId)?.name,
      "renamed.py",
    );
    assert.equal(
      env.calls[0].find((entry) => entry.id === "analysis")?.name,
      "analysis.py",
    );
  } finally {
    env.close();
  }
});

test("Escape, read-only changes and deletion revoke queued filename renames before navigation completes", async () => {
  for (const cancel of ["escape", "readOnly", "delete"]) {
    const env = setup({ disabled: true, navigationPending: true });
    try {
      await env.doubleClick("model.py");
      assert.equal(env.host.querySelector("input"), null);
      if (cancel === "escape")
        await React.act(async () =>
          document.dispatchEvent(
            new dom.window.KeyboardEvent("keydown", {
              key: "Escape",
              bubbles: true,
            }),
          ),
        );
      else if (cancel === "readOnly") env.update({ readOnly: true });
      else
        env.update({
          scripts: [{ id: "analysis", name: "analysis.py" }],
          layout: layout.filter((entry) => entry.id !== scriptId),
        });
      env.update({ disabled: false, navigationPending: false });
      if (cancel === "readOnly") env.update({ readOnly: false });
      assert.equal(
        env.host.querySelector("input"),
        null,
        `${cancel} cancels the queued intent`,
      );
      assert.equal(env.calls.length, 0);
      assert.deepEqual(env.opened, []);
    } finally {
      env.close();
    }
  }
});

test("F2 can wait for navigation but ordinary busy work never queues a later rename", async () => {
  const env = setup({ disabled: true });
  try {
    await env.doubleClick("analysis.py");
    env.update({ disabled: false });
    assert.equal(env.host.querySelector("input"), null);
    env.update({ disabled: true, navigationPending: true });
    await env.contextKey("analysis.py", "F2");
    assert.equal(env.host.querySelector("input"), null);
    env.update({ disabled: true, navigationPending: false });
    env.update({ disabled: false });
    assert.equal(
      env.host.querySelector("input"),
      null,
      "a subsequent non-navigation blocker revokes the queue",
    );
    env.update({ disabled: true, navigationPending: true });
    await env.contextKey("analysis.py", "F2");
    env.update({ disabled: false, navigationPending: false });
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "analysis.py",
    );
    assert.equal(env.calls.length, 0);
  } finally {
    env.close();
  }
});
