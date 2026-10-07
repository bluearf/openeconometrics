import assert from "node:assert/strict";
import test from "node:test";
import { createLocalProjects } from "../src/local-projects.ts";

test("fresh local projects use only token-protected loopback routes and keep project requests scoped", async () => {
  const original = globalThis.fetch;
  const id = "a".repeat(32);
  const project = { id, name: "Offline research", description: "" };
  const calls: string[] = [];
  globalThis.fetch = async (input, init = {}) => {
    const path = String(input);
    calls.push(path);
    assert.ok(path.startsWith("/api/desktop/"));
    const headers = new Headers(init.headers);
    assert.equal(headers.has("Authorization"), false);
    if (path === "/api/desktop/session") return Response.json({ token: "catalog-token" });
    if (path.startsWith("/api/desktop/local-projects")) {
      assert.equal(headers.get("X-OpenEcon-Token"), "catalog-token");
      return Response.json(init.method === "POST" ? project : { projects: [project] });
    }
    const prefix = `/api/desktop/projects/${id}/workspace`;
    if (path === prefix + "/session") return Response.json({ token: "workspace-token", version: "test", environment: "local" });
    if (path === prefix + "/datasets/upload") {
      assert.ok(init.body instanceof FormData);
      assert.equal(headers.get("X-OpenEcon-Token"), "workspace-token");
      return Response.json({ id: "synthetic-dataset" });
    }
    assert.equal(path, prefix + "/console/scripts/analysis");
    assert.equal(headers.get("X-OpenEcon-Token"), "workspace-token");
    return Response.json({ id: "analysis", name: "analysis.py", code: "local = 1", version: 1 });
  };
  try {
    const catalog = createLocalProjects();
    assert.deepEqual(await catalog.list(), { projects: [project] });
    assert.deepEqual(await catalog.create(project.name, ""), project);
    const opened = await catalog.open(id);
    assert.equal(opened.client.teams, false);
    assert.equal(opened.client.desktop, undefined);
    assert.equal(opened.client.localDesktop, true);
    assert.equal(opened.client.projectId, id);
    await opened.client.connect();
    await opened.client.request("/console/scripts/analysis");
    const file = new FormData();
    file.append("file", new Blob(["x,y\n1,2\n"]), "synthetic.csv");
    await opened.client.request("/datasets/upload", { method: "POST", body: file });
    opened.client.cancelPending();
    assert.equal(calls.filter(path => path === "/api/desktop/session").length, 1);
  } finally { globalThis.fetch = original; }
});

test("invalid or mismatched local identities cannot open a project workspace", async () => {
  const original = globalThis.fetch;
  const calls: string[] = [];
  globalThis.fetch = async input => {
    calls.push(String(input));
    return Response.json(String(input).endsWith("/session") ? { token: "catalog-token" }
      : { id: "b".repeat(32), name: "Wrong project", description: "" });
  };
  try {
    const catalog = createLocalProjects();
    await assert.rejects(catalog.open("../../cloud"), /Invalid local project/);
    assert.equal(calls.length, 0);
    await assert.rejects(catalog.open("a".repeat(32)), /verify the local project/);
    assert.ok(calls.every(path => !path.includes("/workspace")));
  } finally { globalThis.fetch = original; }
});

test("an absent local registration surfaces denial without falling back to cloud access", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async input => Response.json(String(input).endsWith("/session")
    ? { token: "catalog-token" } : { detail: { code: "LOCAL_PROJECT_NOT_FOUND", message: "This is not a local project." } },
    { status: String(input).endsWith("/session") ? 200 : 404 });
  try { await assert.rejects(createLocalProjects().open("a".repeat(32)), /not a local project/); }
  finally { globalThis.fetch = original; }
});
