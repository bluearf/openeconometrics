import assert from "node:assert/strict";
import test from "node:test";
import {
  ApiError,
  VersionedScriptWriter,
  createWorkspaceClient,
  loadWorkspaceBootstrap,
  submitWorkspaceExecution,
  canExecuteLocally,
  workspaceDownloadPath,
} from "../src/api.ts";
import {
  canEditProject,
  createTeamApi,
  teamRoute,
  projectUrl,
  loadAuthConfig,
} from "../src/team-api.ts";
import { LatestValueAutosaver } from "../src/session-state.ts";

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("invitation and project deep links preserve opaque identifiers", () => {
  assert.deepEqual(teamRoute("https://example.com/?invite=abc%2B1"), {
    invitationId: "abc+1",
    projectId: null,
  });
  assert.deepEqual(teamRoute("https://example.com/?project=project-a"), {
    projectId: "project-a",
    invitationId: null,
  });
  assert.equal(projectUrl("a/b"), "/?project=a%2Fb");
  assert.equal(teamRoute("/projects/team-a").projectId, "team-a");
  assert.equal(canEditProject("owner"), true);
  assert.equal(canEditProject("editor"), true);
  assert.equal(canEditProject("viewer"), false);
});

test("project clients keep requests scoped while a different project opens", async (t) => {
  const token = deferred<string>();
  const calls: { path: string; headers: Headers }[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      calls.push({ path, headers: new Headers(options.headers) });
      return json({ ok: true });
    },
  );
  const first = createWorkspaceClient("first", () => token.promise);
  const pending = first.request("/console/script", {
    method: "PUT",
    body: JSON.stringify({ code: "first code", version: 2 }),
  });
  const second = createWorkspaceClient("second", async () => "new-token");
  await second.request("/console");
  token.resolve("original-token");
  await pending;
  assert.deepEqual(
    calls.map((call) => call.path),
    [
      "/api/projects/second/workspace/console",
      "/api/projects/first/workspace/console/script",
    ],
  );
  assert.equal(calls[1].headers.get("Authorization"), "Bearer original-token");
  assert.equal(calls[1].headers.get("Content-Type"), "application/json");
  assert.equal(calls[1].headers.has("X-OpenEcon-Token"), false);
});

test("cancelPending stops a request still awaiting an identity token without poisoning remount", async (t) => {
  const token = deferred<string>();
  let fetches = 0;
  t.mock.method(globalThis, "fetch", async () => {
    fetches++;
    return json({ token: "" });
  });
  const client = createWorkspaceClient("one", () => token.promise);
  const pending = client.request("/console");
  client.cancelPending();
  token.resolve("token");
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(fetches, 0);
  await client.connect();
  assert.equal(fetches, 1);
});

test("team bootstrap opens a saved draft and active-run lock without fetching history", async (t) => {
  const calls: string[] = [];
  const value = {
    session: {
      token: "",
      version: "0.3.0",
      environment: "team",
      read_only: true,
    },
    draft: { code: "saved draft", name: "analysis.py", version: 9 },
    datasets: [{ id: "existing-file" }],
    status: { running: true, session_generation: 7 },
  };
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      calls.push(path);
      assert.equal(
        new Headers(options.headers).get("Authorization"),
        "Bearer token",
      );
      assert.equal(path, "/api/projects/one/workspace/bootstrap");
      return json(value);
    },
  );
  const result = await loadWorkspaceBootstrap(
    createWorkspaceClient("one", async () => "token"),
  );
  assert.deepEqual(result, value);
  assert.equal(calls.length, 1);
  assert.equal(result.initialConsole, undefined);
});

test("team bootstrap fails closed on lost membership without legacy fallback", async (t) => {
  const calls: string[] = [];
  t.mock.method(globalThis, "fetch", async (path: string) => {
    calls.push(path);
    return json(
      { detail: { code: "NOT_FOUND", message: "membership removed" } },
      404,
    );
  });
  await assert.rejects(
    loadWorkspaceBootstrap(createWorkspaceClient("one", async () => "token")),
    { status: 404 },
  );
  assert.deepEqual(calls, ["/api/projects/one/workspace/bootstrap"]);
});

test("a browser team workspace never sends code to the cloud", async (t) => {
  const fetched = t.mock.method(globalThis, "fetch", async () => {
    throw new Error("No request may be sent.");
  });
  const client = createWorkspaceClient("one", async () => "token");
  assert.equal(canExecuteLocally(client), false);
  await assert.rejects(submitWorkspaceExecution(client, "print(1)"), {
    status: 410,
    code: "CLOUD_EXECUTION_RETIRED",
    message: /desktop app/,
  });
  await assert.rejects(submitWorkspaceExecution(client, "print(1)", "analysis"), {
    code: "CLOUD_EXECUTION_RETIRED",
  });
  assert.equal(fetched.mock.callCount(), 0);
});

test("desktop team execution stays on the local runtime with a synchronous record", async () => {
  const record = { id: "local-run", status: "ok", outputs: [] };
  const calls: { path: string; body: unknown }[] = [];
  const client = {
    projectId: "one",
    teams: true,
    desktop: true,
    connect: async () => ({ token: "", version: "test", environment: "local" as const }),
    cancelPending() {},
    request: async <T,>(path: string, options: RequestInit = {}) => {
      calls.push({ path, body: JSON.parse(String(options.body)) });
      return record as T;
    },
  };
  assert.equal(canExecuteLocally(client), true);
  assert.deepEqual(await submitWorkspaceExecution(client, "print(1)", "analysis"), record);
  assert.deepEqual(calls, [
    { path: "/console/execute", body: { code: "print(1)", script_id: "analysis" } },
  ]);
});

test("local execution keeps the synchronous body and real result response", async (t) => {
  const record = {
    id: "actual-run",
    status: "ok",
    outputs: [{ type: "text", data: "1" }],
  };
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      if (path === "/api/session") return json({ token: "local-secret" });
      assert.equal(path, "/api/console/execute");
      assert.equal(
        new Headers(options.headers).get("X-OpenEcon-Token"),
        "local-secret",
      );
      assert.deepEqual(JSON.parse(String(options.body)), { code: "print(1)" });
      return json(record);
    },
  );
  const client = createWorkspaceClient();
  await client.connect();
  assert.equal(canExecuteLocally(client), true);
  const result = await submitWorkspaceExecution(client, "print(1)");
  assert.deepEqual(result, record);
});

test("local bootstrap establishes its token before fetching persistent console state", async (t) => {
  const calls: string[] = [];
  const consoleState = {
    history: [{ id: "existing" }],
    variables: [],
    status: { running: false, session_generation: 4 },
  };
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      calls.push(path);
      if (path === "/api/session")
        return json({
          token: "local-secret",
          environment: "local",
          version: "0.3.0",
        });
      assert.equal(
        new Headers(options.headers).get("X-OpenEcon-Token"),
        "local-secret",
      );
      if (path === "/api/console") return json(consoleState);
      if (path === "/api/console/script")
        return json({ code: "saved", name: "analysis.py" });
      assert.equal(path, "/api/datasets");
      return json({ datasets: [{ id: "input" }] });
    },
  );
  const result = await loadWorkspaceBootstrap(createWorkspaceClient());
  assert.equal(calls[0], "/api/session");
  assert.equal(calls.length, 4);
  assert.deepEqual(result.initialConsole, consoleState);
  assert.deepEqual(result.status, consoleState.status);
  assert.equal(result.draft.code, "saved");
});

test("team identity refreshes exactly once on 401 and never retries forbidden access", async (t) => {
  const refreshes: boolean[] = [];
  const headers: string[] = [];
  let count = 0;
  const api = createTeamApi(async (refresh = false) => {
    refreshes.push(refresh);
    return refresh ? "fresh" : "stale";
  });
  t.mock.method(
    globalThis,
    "fetch",
    async (_path: string, options: RequestInit) => {
      headers.push(new Headers(options.headers).get("Authorization")!);
      return ++count === 1
        ? json({ detail: { code: "TOKEN_EXPIRED", message: "expired" } }, 401)
        : json({ projects: [] });
    },
  );
  await api("/projects");
  assert.deepEqual(refreshes, [false, true]);
  assert.deepEqual(headers, ["Bearer stale", "Bearer fresh"]);
  count = 0;
  refreshes.length = 0;
  t.mock.method(globalThis, "fetch", async () => {
    count++;
    return json(
      { detail: { code: "FORBIDDEN", message: "not a member" } },
      403,
    );
  });
  await assert.rejects(
    api("/projects/private/members"),
    (error) => error instanceof ApiError && error.status === 403,
  );
  assert.equal(count, 1);
  assert.deepEqual(refreshes, [false]);
});

test("optimistic autosave serializes changing text with acknowledged versions", async (t) => {
  const first = deferred<Response>();
  const bodies: { code: string; version: number }[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (_path: string, options: RequestInit) => {
      bodies.push(JSON.parse(String(options.body)));
      return bodies.length === 1
        ? first.promise
        : json({ code: "latest", name: "analysis.py", version: 9 });
    },
  );
  const writer = new VersionedScriptWriter(
    createWorkspaceClient("one", async () => "token"),
    7,
  );
  const saver = new LatestValueAutosaver({
    initialValue: "old",
    save: (value) => writer.save(value),
    delayMs: 60_000,
  });
  saver.update("first");
  const flushing = saver.flush();
  await new Promise((done) => setTimeout(done, 0));
  saver.update("latest");
  first.resolve(json({ code: "first", name: "analysis.py", version: 8 }));
  await flushing;
  assert.deepEqual(
    bodies.map(({ code, version }) => ({ code, version })),
    [
      { code: "first", version: 7 },
      { code: "latest", version: 8 },
    ],
  );
  assert.equal(saver.getSavedValue(), "latest");
  saver.dispose();
});

test("conflict keeps unsaved code and prevents a later overwrite", async (t) => {
  let requests = 0;
  t.mock.method(globalThis, "fetch", async () => {
    requests++;
    return json(
      { detail: { code: "SCRIPT_CONFLICT", message: "newer revision" } },
      409,
    );
  });
  const writer = new VersionedScriptWriter(
    createWorkspaceClient("one", async () => "token"),
    1,
  );
  await assert.rejects(writer.save("local changes"), { status: 409 });
  assert.equal(writer.conflict, true);
  await assert.rejects(writer.save("later local changes"), { status: 409 });
  assert.equal(requests, 1);
});

test("legacy workspace retains its token protocol and excludes version from script writes", async (t) => {
  const requests: { path: string; headers: Headers; body?: unknown }[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      requests.push({
        path,
        headers: new Headers(options.headers),
        body: options.body ? JSON.parse(String(options.body)) : undefined,
      });
      return path === "/api/session"
        ? json({
            token: "local-secret",
            environment: "local",
            version: "0.3.0",
          })
        : json({ code: "x = 1", name: "analysis.py" });
    },
  );
  const client = createWorkspaceClient();
  await client.connect();
  await new VersionedScriptWriter(client).save("x = 1");
  assert.equal(requests[1].headers.get("X-OpenEcon-Token"), "local-secret");
  assert.equal(requests[1].headers.has("Authorization"), false);
  assert.equal("version" in (requests[1].body as object), false);
});

test("downloads require the active project and retain authentication for binary responses", async (t) => {
  assert.equal(
    workspaceDownloadPath(
      "/api/projects/one/workspace/runs/run/files/0",
      "one",
    ),
    "/runs/run/files/0",
  );
  for (const url of [
    "https://evil.test/a",
    "/api/projects/two/workspace/runs/run/files/0",
    "/api/projects/one/workspace/../secret",
  ])
    assert.throws(() => workspaceDownloadPath(url, "one"));
  t.mock.method(
    globalThis,
    "fetch",
    async (_path: string, options: RequestInit) => {
      assert.equal(
        new Headers(options.headers).get("Authorization"),
        "Bearer token",
      );
      return new Response("a,b\n1,2", {
        headers: { "Content-Type": "text/csv" },
      });
    },
  );
  const blob = await createWorkspaceClient(
    "one",
    async () => "token",
  ).request<Blob>("/runs/run/files/0", {}, "blob");
  assert.equal(await blob.text(), "a,b\n1,2");
});

test("auth bootstrap accepts explicit local mode and legacy 404, but fails closed for auth/config errors", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    json({ detail: "missing" }, 404),
  );
  assert.equal(await loadAuthConfig(), "local");
  t.mock.method(globalThis, "fetch", async () => json({ mode: "local" }));
  assert.equal(await loadAuthConfig(), "local");
  t.mock.method(globalThis, "fetch", async () =>
    json({ detail: "forbidden" }, 401),
  );
  await assert.rejects(loadAuthConfig(), { status: 401 });
  t.mock.method(globalThis, "fetch", async () =>
    json({ mode: "teams", firebase: { apiKey: "public-key" } }),
  );
  await assert.rejects(loadAuthConfig(), /configuration is missing/);
  const config = {
    mode: "teams",
    firebase: {
      apiKey: "public-key",
      projectId: "project",
      authDomain: "project.firebaseapp.com",
      appId: "app-id",
    },
  };
  t.mock.method(globalThis, "fetch", async () => json(config));
  assert.deepEqual(await loadAuthConfig(), config);
});
