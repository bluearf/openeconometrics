import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { Project, TeamProfile } from "../src/team-api.ts";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
// JSDOM does not implement native dialog visibility; browser rendering is
// checked separately. Keep the real Modal's open/close lifecycle observable.
dom.window.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
dom.window.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
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
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const sourceRoot = new URL("../src/", import.meta.url);
const urls = new Map<string, string>();

// Firebase is only an identity/transport fixture. TeamShell, the name editor,
// optimistic reconciliation and the authenticated API decoder are real modules.
const firebaseApp = `
export const getApps=()=>[{name:"openecon-test"}];
export const initializeApp=(_config,name)=>({name});`;
const firebaseAuth = `
const fixture=()=>globalThis.__projectNameHarness;
export const getAuth=()=>fixture().auth;
export const initializeAuth=()=>fixture().auth;
export const indexedDBLocalPersistence={},browserLocalPersistence={},browserSessionPersistence={};
export const browserPopupRedirectResolver={};
export const getIdToken=async user=>user.uid;
export function onIdTokenChanged(auth,callback){
 fixture().listeners.add(callback); callback(auth.currentUser);
 return ()=>fixture().listeners.delete(callback);
}
export class GoogleAuthProvider {
 addScope(){} setCustomParameters(){}
 static credentialFromResult(){return null;} static credentialFromError(){return null;}
}
export const createUserWithEmailAndPassword=async()=>{},reload=async()=>{},
 sendEmailVerification=async()=>{},sendPasswordResetEmail=async()=>{},
 signInWithEmailAndPassword=async()=>{},updateProfile=async()=>{},
 signInWithPopup=async()=>{},signInWithCustomToken=async()=>{};
export async function signOut(auth){auth.currentUser=null;for(const callback of fixture().listeners)callback(null);}`;

// Keep the workbench's identity, edited source and prepareToLeave contract.
// Rendering the heavy analysis UI is unrelated to project metadata changes.
const workbench = `import {useEffect,useRef,useState} from "react";
export default function Workbench(props){
 const fixture=useRef(globalThis.__projectNameHarness).current;
 const [code,setCode]=useState("keep_me = 1");
 fixture.clients.push(props.client);
 useEffect(()=>{fixture.mounts++;return()=>{fixture.unmounts++;};},[]);
 useEffect(()=>{
  props.onControls({prepareToLeave:async()=>{fixture.flushes++;}});
  return()=>props.onControls(null);
 },[props.onControls]);
 return <section data-testid="workbench" data-readonly={String(props.readOnly)}>
  <span data-testid="workbench-name">{props.projectName}</span>
  <textarea aria-label="Retained Python source" value={code}
   onChange={event=>setCode(event.target.value)}/>
 </section>;
}`;

function moduleUrl(name: string): string {
  const existing = urls.get(name);
  if (existing) return existing;
  let source: string;
  if (name === "firebase/app") source = firebaseApp;
  else if (name === "firebase/auth") source = firebaseAuth;
  else if (name === "App") source = workbench;
  else {
    const extension = existsSync(
      fileURLToPath(new URL(`${name}.tsx`, sourceRoot)),
    )
      ? ".tsx"
      : ".ts";
    source = readFileSync(new URL(`${name}${extension}`, sourceRoot), "utf8");
    if (name === "TeamApp") source += "\nexport { TeamShell };\n";
  }
  const resolve = (target: string) =>
    target.startsWith(".")
      ? moduleUrl(target.replace(/^\.\//, "").replace(/\.tsx?$/, ""))
      : target.startsWith("firebase/")
        ? moduleUrl(target)
        : pathToFileURL(require.resolve(target)).href;
  let output = ts
    .transpileModule(source, {
      fileName:
        name === "App" ||
        existsSync(fileURLToPath(new URL(`${name}.tsx`, sourceRoot)))
          ? `${name}.tsx`
          : `${name}.ts`,
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
      prefix + JSON.stringify(resolve(target)),
  );
  output = output.replace(
    /import\((["'])([^"']+)\1\)/g,
    (_whole, _quote, target: string) =>
      `import(${JSON.stringify(resolve(target))})`,
  );
  const url =
    "data:text/javascript;base64," + Buffer.from(output).toString("base64");
  urls.set(name, url);
  return url;
}
const { TeamShell } = await import(moduleUrl("TeamApp"));
const projectId = "a".repeat(32);
const config = {
  mode: "teams",
  firebase: {
    apiKey: "fixture",
    authDomain: "fixture.invalid",
    projectId: "test",
    appId: "fixture",
  },
};

const createName = () =>
  document.querySelector<HTMLInputElement>(
    'form[aria-label="New project"] input',
  )!;
const createDescription = () =>
  document.querySelector<HTMLTextAreaElement>(
    'form[aria-label="New project"] textarea',
  )!;

test("new-project fields show independent errors before any authenticated POST and keep the draft", async () => {
  const app = setup();
  try {
    await app.render();
    await app.click("New project");
    const name = "a".repeat(101),
      description = "🌍".repeat(501);
    await app.edit(createName(), name);
    await app.edit(createDescription(), description);
    await app.click("Create project");
    assert.equal(createName().getAttribute("aria-invalid"), "true");
    assert.equal(createDescription().getAttribute("aria-invalid"), "true");
    assert.equal(document.querySelectorAll(".team-field-error").length, 2);
    assert.equal(document.activeElement, createName());
    assert.equal(createName().value, name);
    assert.equal(createDescription().value, description);
    assert.equal(app.calls.filter((call) => call.method === "POST").length, 0);
    assert.match(
      document.querySelector('form[aria-label="New project"]')!.textContent!,
      /101 \/ 100 characters/,
    );
    await app.edit(createName(), "Study");
    await app.edit(createDescription(), "Retained\u0000text");
    await app.click("Create project");
    assert.equal(document.activeElement, createDescription());
    assert.match(
      document.querySelector(".team-field-error")!.textContent!,
      /control characters/,
    );
    assert.equal(createDescription().value, "Retained\u0000text");
    assert.equal(app.calls.filter((call) => call.method === "POST").length, 0);
  } finally {
    await app.cleanup();
  }
});

test("new-project astral and trimmed boundaries send normalized fields, not truncated UTF-16 values", async () => {
  const app = setup();
  try {
    await app.render();
    await app.click("New project");
    const name = "🧪".repeat(100),
      description = "🌍".repeat(500);
    await app.edit(createName(), `\u00a0 ${name} \u00a0`);
    await app.edit(createDescription(), `\n\t${description}\r\n`);
    assert.equal(
      createName().maxLength,
      -1,
      "HTML code-unit truncation cannot narrow the Unicode contract",
    );
    assert.equal(document.querySelectorAll(".team-field-error").length, 0);
    await app.click("Create project");
    await app.wait(() => app.header() === name);
    assert.deepEqual(
      app.calls
        .filter((call) => call.method === "POST")
        .map((call) => call.body),
      [{ name, description }],
    );
    const saved = app.profiles
      .get("account-a")!
      .projects.find((project) => project.id === "c".repeat(32))!;
    assert.equal(saved.name, name);
    assert.equal(saved.description, description);
  } finally {
    await app.cleanup();
  }
});

test("new-project API failure preserves both fields for correction and retry", async () => {
  let attempts = 0;
  const app = setup({
    create: async () => {
      if (++attempts === 1)
        return response(
          {
            detail: {
              code: "SERVICE_UNAVAILABLE",
              message: "Try again. Your draft is preserved.",
            },
          },
          503,
        );
    },
  });
  try {
    await app.render();
    await app.click("New project");
    const name = "\ufeffStudy\ufeff",
      description = "Line 1\n\tLine 2";
    await app.edit(createName(), name);
    await app.edit(createDescription(), description);
    await app.click("Create project");
    await app.wait(() =>
      Boolean(
        document.querySelector('form[aria-label="New project"] .team-notice'),
      ),
    );
    assert.equal(createName().value, name);
    assert.equal(createDescription().value, description);
    assert.equal(createName().disabled, false);
    assert.match(
      document.querySelector(".team-notice.is-error")!.textContent!,
      /Your draft is preserved/,
    );
    await app.click("Create project");
    await app.wait(() => app.header() === name);
    assert.equal(attempts, 2);
    assert.deepEqual(
      app.calls
        .filter((call) => call.method === "POST")
        .map((call) => call.body),
      [
        { name, description },
        { name, description },
      ],
    );
  } finally {
    await app.cleanup();
  }
});

test("pending project creation locks fields and blocks duplicate submission", async () => {
  const held = deferred<void>();
  const app = setup({ create: async () => held.promise });
  try {
    await app.render();
    await app.click("New project");
    await app.edit(createName(), "Study");
    await app.click("Create project");
    await app.wait(() => createName().disabled);
    await React.act(() =>
      document
        .querySelector('form[aria-label="New project"]')!
        .dispatchEvent(
          new Event("submit", { bubbles: true, cancelable: true }),
        ),
    );
    assert.equal(app.calls.filter((call) => call.method === "POST").length, 1);
    held.resolve();
    await app.wait(() => app.header() === "Study");
  } finally {
    held.resolve();
    await app.cleanup();
  }
});

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((yes) => {
    resolve = yes;
  });
  return { promise, resolve };
}
function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
type RequestCall = {
  uid: string;
  path: string;
  method: string;
  body: Record<string, unknown> | null;
};
function setup(
  options: {
    open?: boolean;
    role?: Project["role"];
    patch?: (
      call: RequestCall,
      profiles: Map<string, TeamProfile>,
    ) => Promise<Response | void>;
    create?: (call: RequestCall) => Promise<Response | void>;
  } = {},
) {
  window.history.replaceState(
    null,
    "",
    options.open ? `/?project=${projectId}` : "/",
  );
  window.localStorage.clear();
  document.body.innerHTML = "<div id='app'></div>";
  const user = (uid: string) => ({
    uid,
    email: `${uid}@fixture.invalid`,
    emailVerified: true,
    displayName: uid,
  });
  const profile = (uid: string, name: string): TeamProfile => ({
    user: { uid, email: `${uid}@fixture.invalid`, email_verified: true },
    enabled: true,
    projects: [
      {
        id: projectId,
        name,
        role: options.role ?? "owner",
        name_version: 0,
        created_at: "2026-10-04T00:00:00Z",
        description: "Retained description",
      },
    ],
    invitations: [],
  });
  const profiles = new Map([
    ["account-a", profile("account-a", "Original project")],
    ["account-b", profile("account-b", "New account project")],
  ]);
  const fixture = {
    auth: { currentUser: user("account-a") as ReturnType<typeof user> | null },
    listeners: new Set<(value: ReturnType<typeof user> | null) => void>(),
    mounts: 0,
    unmounts: 0,
    flushes: 0,
    clients: [] as unknown[],
  };
  Object.assign(globalThis, { __projectNameHarness: fixture });
  const calls: RequestCall[] = [];
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (input, init = {}) => {
    const path = String(input);
    const uid =
      new Headers(init.headers).get("Authorization")?.replace(/^Bearer /, "") ??
      "";
    const call = {
      uid,
      path,
      method: init.method ?? "GET",
      body: typeof init.body === "string" ? JSON.parse(init.body) : null,
    };
    calls.push(call);
    assert.ok(
      profiles.has(uid),
      "every request must carry the current account token",
    );
    if (path === "/api/me" && call.method === "GET")
      return response(structuredClone(profiles.get(uid)));
    if (path === "/api/projects" && call.method === "POST") {
      const overridden = await options.create?.(call);
      if (overridden) return overridden;
      const project: Project = {
        id: "c".repeat(32),
        name: String(call.body?.name),
        description: String(call.body?.description),
        name_version: 0,
        role: "owner",
        created_at: "2026-10-06T00:00:00Z",
      };
      profiles.get(uid)!.projects.push(project);
      return response(project, 201);
    }
    if (path === `/api/projects/${projectId}` && call.method === "PATCH") {
      const overridden = await options.patch?.(call, profiles);
      if (overridden) return overridden;
      const project = profiles.get(uid)!.projects[0];
      assert.equal(call.body?.name_version, project.name_version);
      project.name = String(call.body?.name);
      project.name_version!++;
      return response({ ...project });
    }
    throw new Error(`Unexpected isolated request ${call.method} ${path}`);
  };
  const root = createRoot(document.getElementById("app")!);
  const button = (label: string) => {
    const found = [
      ...document.querySelectorAll<HTMLButtonElement>("button"),
    ].find(
      (item) =>
        item.getAttribute("aria-label") === label ||
        item.textContent?.trim() === label,
    );
    assert.ok(found, `button ${label} is present`);
    return found;
  };
  const click = async (element: HTMLElement | string) =>
    React.act(() => {
      (typeof element === "string" ? button(element) : element).dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });
  const edit = async (
    element: HTMLInputElement | HTMLTextAreaElement,
    value: string,
  ) =>
    React.act(() => {
      const prototype =
        element instanceof HTMLTextAreaElement
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(
        element,
        value,
      );
      element.dispatchEvent(new Event("input", { bubbles: true }));
    });
  const nameInput = () =>
    document.querySelector<HTMLInputElement>(
      'input[aria-label="Project name"]',
    )!;
  const nameKey = async (key: string) =>
    React.act(() => {
      nameInput().dispatchEvent(
        new KeyboardEvent("keydown", { key, bubbles: true }),
      );
    });
  const sourceInput = () =>
    document.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Retained Python source"]',
    )!;
  const header = () =>
    document.querySelector(".team-topbar-left strong")?.textContent;
  const wait = async (predicate: () => boolean) => {
    for (let attempt = 0; attempt < 30 && !predicate(); attempt++)
      await React.act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
    assert.ok(predicate(), "expected UI state arrived");
  };
  const render = async () => {
    await React.act(() =>
      root.render(React.createElement(TeamShell, { config })),
    );
    await wait(() =>
      options.open
        ? Boolean(sourceInput())
        : Boolean(document.querySelector(".team-project-card")),
    );
  };
  const cleanup = async () => {
    await React.act(() => root.unmount());
    globalThis.fetch = previousFetch;
    document.body.innerHTML = "";
    assert.equal(fixture.listeners.size, 0, "auth subscriptions are disposed");
  };
  const switchAccount = async (uid: string) => {
    await React.act(() => {
      fixture.auth.currentUser = user(uid);
      for (const callback of fixture.listeners)
        callback(fixture.auth.currentUser);
    });
    await wait(() => header() === profiles.get(uid)!.projects[0].name);
  };
  return {
    profiles,
    fixture,
    calls,
    render,
    cleanup,
    button,
    click,
    edit,
    nameInput,
    nameKey,
    sourceInput,
    header,
    wait,
    switchAccount,
  };
}

test("open project rename updates header before cloud acknowledgement without replacing client or edited source", async () => {
  const held = deferred<void>();
  const app = setup({ open: true, patch: async () => held.promise });
  try {
    await app.render();
    await app.edit(app.sourceInput(), "user_edits = 42\nprint(user_edits)");
    const client = app.fixture.clients[0];
    const name = app.button("Rename project");
    assert.equal(name.textContent, "Original project");
    assert.ok(document.querySelector(".team-topbar-left")!.contains(name));
    assert.equal(name.querySelector("svg"), null);
    await app.click(name);
    assert.equal(document.activeElement, app.nameInput());
    await app.edit(app.nameInput(), "Renamed project");
    await app.nameKey("Enter");
    await app.wait(() => app.calls.some((call) => call.method === "PATCH"));
    assert.equal(
      app.header(),
      "Renamed project",
      "optimistic header does not await HTTP",
    );
    assert.equal(
      document.querySelector('[data-testid="workbench-name"]')?.textContent,
      "Renamed project",
    );
    assert.equal(
      app.profiles.get("account-a")!.projects[0].name,
      "Original project",
      "response remains held",
    );
    assert.equal(app.sourceInput().value, "user_edits = 42\nprint(user_edits)");
    assert.equal(app.fixture.mounts, 1);
    assert.equal(app.fixture.unmounts, 0);
    assert.ok(app.fixture.clients.every((value) => value === client));
    assert.equal(
      app.fixture.flushes,
      0,
      "logical rename never invokes prepareToLeave",
    );
    assert.deepEqual(
      app.calls
        .filter((call) => call.method === "PATCH")
        .map((call) => call.body),
      [{ name: "Renamed project", name_version: 0 }],
    );
    await React.act(() => held.resolve());
    await app.wait(() => !app.button("Rename project").disabled);
    assert.equal(app.header(), "Renamed project");
    assert.equal(app.sourceInput().value, "user_edits = 42\nprint(user_edits)");
    assert.equal(app.fixture.mounts, 1);
    assert.ok(app.fixture.clients.every((value) => value === client));
    assert.equal(app.fixture.flushes, 0);
    assert.ok(
      app.calls.every(
        (call) =>
          call.path === "/api/me" || call.path === `/api/projects/${projectId}`,
      ),
      "metadata rename issues no script or runtime requests",
    );
  } finally {
    held.resolve();
    await app.cleanup();
  }
});

test("single-click workspace name opens editing and Escape cancels without changing metadata, client or unsaved source", async () => {
  const app = setup({ open: true });
  try {
    await app.render();
    await app.edit(app.sourceInput(), "unsaved_source = 17");
    const client = app.fixture.clients[0];
    const name = app.button("Rename project");
    assert.equal(name.textContent, "Original project");
    await app.click(name);
    assert.equal(app.nameInput().value, "Original project");
    assert.equal(document.activeElement, app.nameInput());
    await app.edit(app.nameInput(), "Cancelled name");
    await app.nameKey("Escape");
    assert.equal(app.nameInput(), null);
    assert.equal(app.header(), "Original project");
    assert.equal(app.calls.filter((call) => call.method === "PATCH").length, 0);
    assert.equal(app.sourceInput().value, "unsaved_source = 17");
    assert.equal(app.fixture.mounts, 1);
    assert.equal(app.fixture.unmounts, 0);
    assert.equal(app.fixture.flushes, 0);
    assert.ok(app.fixture.clients.every((value) => value === client));
    await app.click(app.button("Rename project"));
    assert.equal(app.nameInput().value, "Original project");
    await app.nameKey("Escape");
  } finally {
    await app.cleanup();
  }
});

test("dashboard project open and rename controls are siblings; renaming never opens the project", async () => {
  const app = setup();
  try {
    await app.render();
    const article = document.querySelector("article.team-project-card")!;
    const open =
      article.querySelector<HTMLButtonElement>(".team-project-open")!;
    const pencil = app.button("Rename project");
    assert.ok(article.contains(pencil));
    assert.ok(
      !open.contains(pencil),
      "rename is outside the project open button",
    );
    assert.equal(
      article.querySelector("button button"),
      null,
      "no nested interactive buttons",
    );
    await app.click(pencil);
    await app.edit(app.nameInput(), "Dashboard rename");
    await app.click("Save project name");
    await app.wait(() => !app.button("Rename project").disabled);
    assert.equal(window.location.search, "");
    assert.equal(
      document.querySelector(".team-project-open h2")?.textContent,
      "Dashboard rename",
    );
    assert.equal(app.fixture.mounts, 0);
    await app.click(open);
    await app.wait(() => Boolean(app.sourceInput()));
    assert.equal(
      new URL(window.location.href).searchParams.get("project"),
      projectId,
    );
    assert.equal(app.header(), "Dashboard rename");
    assert.equal(app.fixture.mounts, 1);
  } finally {
    await app.cleanup();
  }
});

test("CAS conflict refreshes canonical name and version while retaining the user's retry draft", async () => {
  let attempts = 0;
  const app = setup({
    patch: async (_call, profiles) => {
      if (attempts++ === 0) {
        Object.assign(profiles.get("account-a")!.projects[0], {
          name: "Teammate rename",
          name_version: 1,
        });
        return response(
          {
            detail: {
              code: "VERSION_CONFLICT",
              message: "Project changed; retry.",
            },
          },
          409,
        );
      }
    },
  });
  try {
    await app.render();
    await app.click("Rename project");
    await app.edit(app.nameInput(), "  My retry draft  ");
    await app.click("Save project name");
    await app.wait(
      () =>
        Boolean(app.nameInput()) &&
        document.querySelector(".team-project-open h2")?.textContent ===
          "Teammate rename",
    );
    assert.equal(
      app.nameInput().value,
      "  My retry draft  ",
      "typed whitespace and draft survive the refresh",
    );
    assert.match(
      document.querySelector(".project-name-editor__error")?.textContent ?? "",
      /retry/,
    );
    assert.equal(
      app.calls.filter((call) => call.path === "/api/me").length,
      2,
      "conflict refreshes authenticated canonical profile",
    );
    await app.click("Save project name");
    await app.wait(
      () => !app.nameInput() && !app.button("Rename project").disabled,
    );
    assert.deepEqual(
      app.calls
        .filter((call) => call.method === "PATCH")
        .map((call) => call.body),
      [
        { name: "My retry draft", name_version: 0 },
        { name: "My retry draft", name_version: 1 },
      ],
    );
    assert.equal(
      document.querySelector(".team-project-open h2")?.textContent,
      "My retry draft",
    );
    assert.equal(app.profiles.get("account-a")!.projects[0].name_version, 2);
    assert.equal(window.location.search, "");
  } finally {
    await app.cleanup();
  }
});

test("late rename acknowledgement from an old UID cannot overwrite the new account's same project", async () => {
  const held = deferred<void>();
  const app = setup({ open: true, patch: async () => held.promise });
  try {
    await app.render();
    await app.click("Rename project");
    await app.edit(app.nameInput(), "Old account rename");
    await app.click("Save project name");
    await app.wait(() => app.calls.some((call) => call.method === "PATCH"));
    assert.equal(app.header(), "Old account rename");
    await app.switchAccount("account-b");
    await app.edit(app.sourceInput(), "new_account_edits = 7");
    const newClient = app.fixture.clients.at(-1);
    const mounts = app.fixture.mounts;
    await React.act(() => held.resolve());
    assert.equal(app.header(), "New account project");
    assert.equal(
      document.querySelector('[data-testid="workbench-name"]')?.textContent,
      "New account project",
    );
    assert.equal(app.sourceInput().value, "new_account_edits = 7");
    assert.equal(app.fixture.clients.at(-1), newClient);
    assert.equal(app.fixture.mounts, mounts);
    assert.equal(app.fixture.flushes, 0);
    assert.equal(app.profiles.get("account-b")!.projects[0].name_version, 0);
    assert.deepEqual(
      app.calls
        .filter((call) => call.method === "PATCH")
        .map((call) => call.uid),
      ["account-a"],
    );
  } finally {
    held.resolve();
    await app.cleanup();
  }
});

test("viewer has no rename control on the dashboard or in the open workbench", async () => {
  const app = setup({ role: "viewer" });
  try {
    await app.render();
    assert.equal(document.querySelector('[aria-label="Rename project"]'), null);
    await app.click(
      document.querySelector<HTMLButtonElement>(".team-project-open")!,
    );
    await app.wait(() => Boolean(app.sourceInput()));
    assert.equal(app.header(), "Original project");
    assert.equal(document.querySelector('[aria-label="Rename project"]'), null);
    assert.equal(
      document
        .querySelector('[data-testid="workbench"]')
        ?.getAttribute("data-readonly"),
      "true",
    );
    assert.equal(app.calls.filter((call) => call.method === "PATCH").length, 0);
  } finally {
    await app.cleanup();
  }
});
