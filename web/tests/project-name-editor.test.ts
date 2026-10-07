import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { Project } from "../src/team-api.ts";
import type { ProjectNameEditorProps } from "../src/ProjectNameEditor.tsx";

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
  "Element",
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
const output = ts
  .transpileModule(
    readFileSync(
      new URL("../src/ProjectNameEditor.tsx", import.meta.url),
      "utf8",
    ),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    },
  )
  .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "")
  .replace(
    /(from\s*|import\s*)(["'])([^"']+)\2/g,
    (_whole, prefix, _quote, name: string) =>
      prefix + JSON.stringify(pathToFileURL(require.resolve(name)).href),
  );
const { default: ProjectNameEditor } = await import(
  "data:text/javascript;base64," + Buffer.from(output).toString("base64")
);

const project: Project = {
  id: "a".repeat(32),
  name: "Deneme",
  description: "Açıklama",
  role: "owner",
  created_at: "2026-10-04T00:00:00Z",
};

function deferred() {
  let resolve!: () => void, reject!: (reason: Error) => void;
  const promise = new Promise<void>((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function setup(
  patch: Partial<ProjectNameEditorProps> = {},
  persist: (project: Project, name: string) => Promise<void> = async () => {},
) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  let closed = false;
  let scope = "owner-account:" + project.id;
  let parentClicks = 0,
    parentKeys = 0,
    otherActions = 0;
  const calls: { project: Project; name: string }[] = [];
  let props: ProjectNameEditorProps = {
    project: { ...project },
    onRename: async (submitted, name) => {
      calls.push({ project: submitted, name });
      const previous = props.project;
      const submittedScope = scope;
      props = { ...props, project: { ...props.project, name } };
      render();
      try {
        await persist(submitted, name);
      } catch (reason) {
        if (
          scope === submittedScope &&
          props.project.id === submitted.id &&
          props.project.name === name
        ) {
          props = {
            ...props,
            project: { ...props.project, name: previous.name },
          };
          render();
        }
        throw reason;
      }
    },
    ...patch,
  };
  function render() {
    if (closed) return;
    root.render(
      React.createElement(
        "div",
        {
          onClick: () => parentClicks++,
          onKeyDown: () => parentKeys++,
        },
        !props.showName &&
          React.createElement(
            "strong",
            { "data-project-name": true },
            props.project.name,
          ),
        React.createElement(ProjectNameEditor, { ...props, key: scope }),
        React.createElement(
          "button",
          {
            "aria-label": "Diğer işlem",
            onClick: () => otherActions++,
          },
          "Diğer işlem",
        ),
      ),
    );
  }
  React.act(render);
  const button = (label: string) => {
    const found = host.querySelector<HTMLButtonElement>(
      `button[aria-label="${label}"]`,
    );
    assert.ok(found, `button ${label} exists`);
    return found;
  };
  return {
    host,
    calls,
    button,
    get parentClicks() {
      return parentClicks;
    },
    get parentKeys() {
      return parentKeys;
    },
    get otherActions() {
      return otherActions;
    },
    get displayedName() {
      return host.querySelector("strong")?.textContent;
    },
    async click(label: string) {
      await React.act(async () => button(label).click());
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
    async key(key: string, repeats = 1) {
      const input = host.querySelector("input")!;
      await React.act(async () => {
        for (let index = 0; index < repeats; index++)
          input.dispatchEvent(
            new dom.window.KeyboardEvent("keydown", { key, bubbles: true }),
          );
      });
    },
    update(patch: Partial<ProjectNameEditorProps>) {
      props = { ...props, ...patch };
      React.act(render);
    },
    changeAccount(account: string, next: Project = project) {
      scope = account + ":" + next.id;
      props = { ...props, project: { ...next } };
      React.act(render);
    },
    close() {
      closed = true;
      React.act(() => root.unmount());
      host.remove();
    },
  };
}

test("only owners see the compact edit control and the existing name is not duplicated", () => {
  for (const role of ["owner", "editor", "viewer"] as const) {
    const env = setup({ project: { ...project, role } });
    try {
      assert.equal(env.displayedName, "Deneme");
      assert.equal(env.host.querySelectorAll("[data-project-name]").length, 1);
      const button = env.host.querySelector(
        "button[aria-label=\"Rename project\"]",
      );
      assert.equal(!!button, role === "owner");
      assert.equal(button?.hasAttribute("title") ?? false, false);
      assert.equal(env.host.querySelector('[role="tooltip"]'), null);
      assert.equal(env.host.querySelector("input"), null);
    } finally {
      env.close();
    }
  }
});

test("name mode renders the actual project name once for every role and only owners can click it to edit", async () => {
  for (const role of ["owner", "editor", "viewer"] as const) {
    const env = setup({ project: { ...project, role }, showName: true });
    try {
      assert.equal(env.displayedName, "Deneme");
      assert.equal(env.host.querySelectorAll("strong").length, 1);
      const nameButton = env.host.querySelector<HTMLButtonElement>(
        "button[aria-label=\"Rename project\"]",
      );
      assert.equal(Boolean(nameButton), role === "owner");
      if (nameButton) {
        assert.equal(nameButton.textContent, "Deneme");
        assert.equal(nameButton.querySelector("svg"), null);
        await React.act(async () => nameButton.click());
        assert.equal(
          env.host.querySelector<HTMLInputElement>("input")?.value,
          "Deneme",
        );
        assert.equal(env.parentClicks, 0);
      } else {
        await React.act(async () => env.host.querySelector("strong")!.click());
        assert.equal(env.host.querySelector("input"), null);
      }
      assert.equal(env.calls.length, 0);
    } finally {
      env.close();
    }
  }
});

test("clicking the visible name focuses editing, Escape cancels and repeated Enter saves the name once", async () => {
  const pending = deferred();
  const env = setup({ showName: true }, async () => pending.promise);
  try {
    const name = env.button("Rename project");
    assert.equal(name.textContent, "Deneme");
    await React.act(async () => name.click());
    let input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(document.activeElement, input);
    assert.equal(input.selectionStart, 0);
    assert.equal(input.selectionEnd, "Deneme".length);
    await env.input("Discard this name");
    await env.key("Escape");
    assert.equal(env.calls.length, 0);
    assert.equal(env.displayedName, "Deneme");
    assert.equal(env.host.querySelector("input"), null);
    await React.act(async () => env.button("Rename project").click());
    input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(input.value, "Deneme");
    await env.input("  Research project  ");
    await env.key("Enter", 2);
    assert.equal(env.calls.length, 1);
    assert.equal(env.calls[0].name, "Research project");
    assert.equal(env.displayedName, "Research project");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(
      env.button("Rename project").textContent,
      "Research project",
    );
    assert.equal(env.button("Rename project").disabled, true);
    assert.equal(env.parentClicks, 0);
    assert.equal(env.parentKeys, 0);
    await React.act(async () => pending.resolve());
    assert.equal(env.button("Rename project").disabled, false);
  } finally {
    pending.resolve();
    env.close();
  }
});

test("the name input focuses and selects the current name while Escape and cancel make no writes", async () => {
  const env = setup();
  try {
    await env.click("Rename project");
    let input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(document.activeElement, input);
    assert.equal(input.value, "Deneme");
    assert.equal(input.selectionStart, 0);
    assert.equal(input.selectionEnd, "Deneme".length);
    assert.equal(input.maxLength, 200);
    await env.input("Başka ad");
    await env.key("Escape");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.calls.length, 0);
    await env.click("Rename project");
    input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(input.value, "Deneme");
    await env.input("Kaydetme");
    await env.click("Cancel");
    assert.equal(env.calls.length, 0);
    assert.equal(env.displayedName, "Deneme");
  } finally {
    env.close();
  }
});

test("a delayed rename closes immediately, trims the submitted name and accepts repeated Enter only once", async () => {
  const pending = deferred();
  const env = setup({}, async () => pending.promise);
  try {
    await env.click("Rename project");
    await env.input("  Ücret analizi  ");
    await env.key("Enter", 2);
    assert.equal(env.calls.length, 1);
    assert.equal(env.calls[0].name, "Ücret analizi");
    assert.deepEqual(env.calls[0].project, project);
    assert.equal(env.displayedName, "Ücret analizi");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.button("Rename project").disabled, true);
    assert.equal(env.button("Diğer işlem").disabled, false);
    await env.click("Diğer işlem");
    assert.equal(env.otherActions, 1);
    await env.click("Rename project");
    assert.equal(env.calls.length, 1);
    assert.equal(env.host.querySelector("input"), null);
    await React.act(async () => pending.resolve());
    assert.equal(env.button("Rename project").disabled, false);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("a failed save restores the exact typed draft and a corrected retry clears the local error", async () => {
  const pending = deferred();
  let attempts = 0;
  const env = setup({}, async () => {
    if (++attempts === 1) await pending.promise;
  });
  try {
    await env.click("Rename project");
    await env.input("  Yeni çalışma  ");
    await env.click("Save project name");
    assert.equal(env.host.querySelector("input"), null);
    env.update({ disabled: true });
    await React.act(async () =>
      pending.reject(
        new Error("Proje adı başka bir kişi tarafından değiştirildi."),
      ),
    );
    let input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(input.value, "  Yeni çalışma  ");
    assert.equal(input.disabled, true);
    env.update({ disabled: false });
    input = env.host.querySelector<HTMLInputElement>("input")!;
    assert.equal(input.disabled, false);
    assert.equal(input.getAttribute("aria-invalid"), "true");
    const error = env.host.querySelector('[role="alert"]')!;
    assert.match(error.textContent!, /başka bir kişi/);
    assert.equal(input.getAttribute("aria-describedby"), error.id);
    await env.input("Son çalışma");
    await env.click("Save project name");
    assert.equal(env.calls.length, 2);
    assert.equal(env.displayedName, "Son çalışma");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("invalid empty, overlong and control-character names stay editable without invoking the callback", async () => {
  const env = setup();
  try {
    await env.click("Rename project");
    for (const value of [
      "   ",
      "a".repeat(101),
      "bad\u0001name",
      "bad\uD800name",
      "bad\u2028name",
      "bad\u2029name",
      "\tValid name",
      "Valid name\u2028",
    ]) {
      await env.input(value);
      await env.key("Enter");
      assert.equal(env.calls.length, 0);
      assert.ok(env.host.querySelector('[role="alert"]'));
      assert.equal(
        env.host.querySelector<HTMLInputElement>("input")?.value,
        value,
      );
    }
    await env.input("Doğru ad");
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
    assert.equal(env.displayedName, "Doğru ad");
  } finally {
    env.close();
  }
});

test("Unicode codepoint limits allow 100 astral characters and emoji ZWJ names", async () => {
  const env = setup();
  try {
    await env.click("Rename project");
    await env.input("😀".repeat(100));
    await env.click("Save project name");
    assert.equal(env.calls.length, 1);
    assert.equal(env.calls[0].name.length, 200);
    await env.click("Rename project");
    await env.input("😀".repeat(101));
    await env.click("Save project name");
    assert.equal(env.calls.length, 1);
    await env.input("👩‍💻 Çalışma");
    await env.click("Save project name");
    assert.equal(env.calls.length, 2);
    assert.equal(env.displayedName, "👩‍💻 Çalışma");
  } finally {
    env.close();
  }
});

test("name updates from the parent and blur do not discard an active draft", async () => {
  const env = setup();
  try {
    await env.click("Rename project");
    await env.input("Benim taslağım");
    env.update({ project: { ...project, name: "Takımdan gelen ad" } });
    assert.equal(env.displayedName, "Takımdan gelen ad");
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Benim taslağım",
    );
    await React.act(async () =>
      env.host.querySelector<HTMLInputElement>("input")!.blur(),
    );
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Benim taslağım",
    );
    await env.key("Enter");
    assert.equal(env.calls[0].project.name, "Takımdan gelen ad");
    assert.equal(env.calls[0].name, "Benim taslağım");
  } finally {
    env.close();
  }
});

test("disabled state affects only the rename control and preserves an open draft", async () => {
  const env = setup({ disabled: true });
  try {
    await env.click("Rename project");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.button("Diğer işlem").disabled, false);
    env.update({ disabled: false });
    await env.click("Rename project");
    await env.input("Devam eden taslak");
    env.update({ disabled: true });
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Devam eden taslak",
    );
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.disabled,
      true,
    );
    env.update({ disabled: false });
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Devam eden taslak",
    );
    await env.key("Enter");
    assert.equal(env.calls.length, 1);
  } finally {
    env.close();
  }
});

test("editor click and key events never trigger the surrounding card navigation", async () => {
  const env = setup();
  try {
    await env.click("Rename project");
    await env.input("Yeni ad");
    await env.key("Escape");
    assert.equal(env.parentClicks, 0);
    assert.equal(env.parentKeys, 0);
    await env.click("Rename project");
    await env.click("Save project name");
    assert.equal(env.parentClicks, 0);
    await env.click("Diğer işlem");
    assert.equal(env.parentClicks, 1);
  } finally {
    env.close();
  }
});

test("changing projects clears editing state and an old failure cannot reopen or clear a newer pending rename", async () => {
  const first = deferred(),
    second = deferred();
  const next = { ...project, id: "b".repeat(32), name: "İkinci proje" };
  const env = setup({}, async (submitted) =>
    submitted.id === project.id ? first.promise : second.promise,
  );
  try {
    await env.click("Rename project");
    await env.input("Eski taslak");
    await env.key("Enter");
    env.update({ project: next });
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.button("Rename project").disabled, false);
    await env.click("Rename project");
    await env.input("Yeni proje adı");
    await env.key("Enter");
    await React.act(async () => first.reject(new Error("Eski proje hatası.")));
    assert.equal(env.displayedName, "Yeni proje adı");
    assert.equal(env.host.querySelector("input"), null);
    assert.equal(env.host.querySelector('[role="alert"]'), null);
    assert.equal(env.button("Rename project").disabled, true);
    await React.act(async () => second.resolve());
    assert.equal(env.button("Rename project").disabled, false);
  } finally {
    env.close();
  }
});

test("owner permission changes dismiss drafts and prevent late errors from restoring them", async () => {
  const pending = deferred();
  const env = setup({}, async () => pending.promise);
  try {
    await env.click("Rename project");
    await env.input("Eski sahip taslağı");
    await env.key("Enter");
    env.update({ project: { ...project, role: "viewer" } });
    assert.equal(env.host.querySelector(".project-name-editor"), null);
    env.update({ project: { ...project, role: "owner" } });
    await env.click("Rename project");
    await env.input("Yeni taslak");
    await React.act(async () =>
      pending.reject(new Error("Eski sahip isteği başarısız.")),
    );
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Yeni taslak",
    );
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("an account-scoped remount ignores a late failure even when the project ID is the same", async () => {
  const pending = deferred();
  const env = setup({}, async () => pending.promise);
  try {
    await env.click("Rename project");
    await env.input("Önceki hesap");
    await env.key("Enter");
    env.changeAccount("different-account");
    await env.click("Rename project");
    await env.input("Yeni hesap taslağı");
    await React.act(async () =>
      pending.reject(new Error("Önceki hesap hatası.")),
    );
    assert.equal(
      env.host.querySelector<HTMLInputElement>("input")?.value,
      "Yeni hesap taslağı",
    );
    assert.equal(env.displayedName, "Deneme");
    assert.equal(env.host.querySelector('[role="alert"]'), null);
  } finally {
    env.close();
  }
});

test("an unmounted editor handles a late rejection without updating its old view", async () => {
  const pending = deferred();
  const env = setup({}, async () => pending.promise);
  await env.click("Rename project");
  await env.input("Kaybolan görünüm");
  await env.key("Enter");
  env.close();
  await React.act(async () => pending.reject(new Error("Geç gelen hata.")));
  assert.equal(env.host.textContent, "");
});
