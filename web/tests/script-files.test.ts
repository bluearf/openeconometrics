import assert from "node:assert/strict";
import test from "node:test";
import {
  ApiError,
  VersionedScriptWriter,
  loadScriptFiles,
  loadScriptFile,
  scriptFilePath,
  submitWorkspaceExecution,
  type WorkspaceClient,
} from "../src/api.ts";
import { LatestValueAutosaver } from "../src/session-state.ts";

const first = "a".repeat(32),
  second = "b".repeat(32);
function client(request: WorkspaceClient["request"]): WorkspaceClient {
  return {
    projectId: null,
    teams: false,
    request,
    connect: async () => ({ token: "", version: "test", environment: "local" }),
    cancelPending() {},
  };
}

test("opening a file verifies its immutable identity and preserves blank content", async () => {
  const read = client(
    async <T>() =>
      ({ id: first, name: "untitled.py", code: "", version: 0 }) as T,
  );
  assert.equal((await loadScriptFile(read, first)).code, "");
  await assert.rejects(loadScriptFile(read, second), /Could not verify/);
  const negative = client(
    async <T>() =>
      ({ id: first, name: "untitled.py", code: "", version: -1 }) as T,
  );
  await assert.rejects(loadScriptFile(negative, first), /Could not verify/);
});

test("a pending save remains bound to its file while another file is saved", async () => {
  const calls: { path: string; body: { code: string; version: number } }[] = [];
  let release!: () => void;
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  const api = client(async <T>(path: string, options?: RequestInit) => {
    const body = JSON.parse(String(options?.body));
    calls.push({ path, body });
    if (path === scriptFilePath(first)) await wait;
    return {
      id: path.split("/").at(-1),
      name: "file.py",
      code: body.code,
      version: body.version + 1,
    } as T;
  });
  const original = new VersionedScriptWriter(api, 4, first);
  const other = new VersionedScriptWriter(api, 9, second);
  const pending = original.save("original = 1");
  await other.save("other = 2");
  release();
  await pending;
  await original.save("original = 3");
  assert.deepEqual(calls, [
    { path: scriptFilePath(first), body: { code: "original = 1", version: 4 } },
    { path: scriptFilePath(second), body: { code: "other = 2", version: 9 } },
    { path: scriptFilePath(first), body: { code: "original = 3", version: 5 } },
  ]);
});

test("empty Python content is saved as an empty file with local concurrency checking", async () => {
  const api = client(async <T>(path: string, options?: RequestInit) => {
    assert.equal(path, "/console/scripts/analysis");
    assert.deepEqual(JSON.parse(String(options?.body)), {
      code: "",
      version: 3,
    });
    return { id: "analysis", name: "analysis.py", code: "", version: 4 } as T;
  });
  assert.equal(
    (await new VersionedScriptWriter(api, 3, "analysis").save("")).code,
    "",
  );
});

test("a failed outgoing autosave blocks leaving and keeps the exact draft available", async () => {
  const draft = "important_result = 123\n";
  const writer = new VersionedScriptWriter(
    client(async () => {
      throw new ApiError("conflict", 409);
    }),
    2,
    first,
  );
  const saver = new LatestValueAutosaver({
    initialValue: "",
    delayMs: 60000,
    save: (code) => writer.save(code),
  });
  saver.update(draft);
  let switched = false;
  await assert.rejects(
    async () => {
      await saver.flush();
      switched = true;
    },
    (error) => error instanceof ApiError && error.status === 409,
  );
  assert.equal(switched, false);
  assert.equal(writer.conflict, true);
  assert.equal(saver.getSavedValue(), undefined);
  saver.dispose();
});

test("one conflicted file cannot send further writes or block another file", async () => {
  const calls: string[] = [];
  const api = client(async <T>(path: string) => {
    calls.push(path);
    if (path === scriptFilePath(first)) throw new ApiError("conflict", 409);
    return { id: second, name: "second.py", code: "ok", version: 1 } as T;
  });
  const blocked = new VersionedScriptWriter(api, 0, first);
  await assert.rejects(blocked.save("mine"));
  await assert.rejects(blocked.save("still mine"));
  await new VersionedScriptWriter(api, 0, second).save("ok");
  assert.deepEqual(calls, [scriptFilePath(first), scriptFilePath(second)]);
});

test("a reply for a different file or an older version is not acknowledged", async () => {
  for (const response of [
    { id: second, version: 8 },
    { id: first, version: 5 },
  ]) {
    const api = client(
      async <T>() => ({ ...response, name: "file.py", code: "mine" }) as T,
    );
    await assert.rejects(
      new VersionedScriptWriter(api, 7, first).save("mine"),
      /save version/,
    );
  }
});

test("unsafe script ids cannot construct an API route", () => {
  for (const id of [
    "../other",
    "analysis.py",
    "",
    first + "/other",
    "%2fsecret",
    "A".repeat(32),
  ])
    assert.throws(() => scriptFilePath(id));
  assert.equal(scriptFilePath("analysis"), "/console/scripts/analysis");
});

test("the script catalog is ordered, includes the legacy main file and keeps pending metadata", async () => {
  const scripts = [
    { id: "analysis", name: "analysis.py", version: 2 },
    { id: first, name: "ücret.py", version: 0, pending_sync: true },
  ];
  const api = client(async <T>(path: string) => {
    assert.equal(path, "/console/scripts");
    return { scripts } as T;
  });
  assert.deepEqual(await loadScriptFiles(api), scripts);
});

test("malformed catalogs fail closed without treating another document as the main file", async () => {
  for (const scripts of [
    [],
    [{ id: first, name: "first.py", version: 0 }],
    [{ id: "analysis", name: "analysis.py", version: -1 }],
    [
      { id: "analysis", name: "analysis.py", version: 0 },
      { id: "analysis", name: "other.py", version: 1 },
    ],
  ]) {
    await assert.rejects(
      loadScriptFiles(client(async <T>() => ({ scripts }) as T)),
    );
  }
});

test("text document responses accept safe Markdown and LaTeX names while reserved main remains Python", async () => {
  for (const name of ["notes.md", "table.tex"]) {
    const reader = client(
      async <T>() => ({ id: first, name, code: "", version: 0 }) as T,
    );
    assert.equal((await loadScriptFile(reader, first)).name, name);
  }
  for (const [id, name] of [
    ["analysis", "notes.md"],
    [first, "../notes.md"],
    [first, "notes.txt"],
    [first, "hidden..tex"],
  ]) {
    const reader = client(
      async <T>() => ({ id, name, code: "", version: 0 }) as T,
    );
    await assert.rejects(loadScriptFile(reader, id));
  }
});

test("file execution submits its immutable identity and terminal execution omits document type", async () => {
  const bodies: unknown[] = [];
  const api = client(async <T>(_path: string, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    return {
      id: "run-1",
      code: "print(1)",
      status: "ok",
      outputs: [],
      stdout: "",
      variables: [],
    } as T;
  });
  await submitWorkspaceExecution(api, "print(1)", first);
  await submitWorkspaceExecution(api, "print(2)");
  assert.deepEqual(bodies, [
    { code: "print(1)", script_id: first },
    { code: "print(2)" },
  ]);
});
