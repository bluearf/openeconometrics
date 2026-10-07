import assert from "node:assert/strict";
import test from "node:test";
import { ProjectNames, projectName } from "../src/project-name.ts";
import type { Project, TeamProfile } from "../src/team-api.ts";
const first = "a".repeat(32),
  second = "b".repeat(32);
const project = (
  id = first,
  name = "Original",
  name_version = 0,
  role: Project["role"] = "owner",
): Project => ({
  id,
  name,
  name_version,
  role,
  created_at: "2026-01-01",
  updated_at: "2026-01-01",
});
const profile = (projects: Project[], uid = "owner"): TeamProfile => ({
  user: { uid, email: `${uid}@example.com` },
  enabled: true,
  projects,
  invitations: [],
});
function setup() {
  const names = new ProjectNames();
  names.select("owner");
  return names;
}

test("pending rename updates the visible name immediately but never cached canonical metadata", () => {
  const names = setup(),
    original = profile([project()]);
  const change = names.begin(original.projects[0], "  New name  ");
  assert.equal(names.view(original).projects[0].name, "New name");
  assert.equal(names.reconcile(original).projects[0].name, "Original");
  assert.equal(original.projects[0].name, "Original");
  assert.throws(
    () => names.begin(original.projects[0], "Again"),
    /Saving project name/,
  );
  names.accept(change, project(first, "New name", 1));
  assert.equal(names.reconcile(original).projects[0].name, "New name");
  assert.equal(names.busy(first), false);
});

test("failed pending rename reveals the latest shared name rather than a stale rollback", () => {
  const names = setup();
  const change = names.begin(project(), "My proposal");
  const refreshed = profile([project(first, "Other device", 1)]);
  assert.equal(names.view(refreshed).projects[0].name, "My proposal");
  names.cancel(change);
  assert.equal(names.view(refreshed).projects[0].name, "Other device");
});

test("an older profile cannot erase an acknowledgement; a newer shared rename wins", () => {
  const names = setup();
  names.accept(
    names.begin(project(), "Accepted"),
    project(first, "Accepted", 1),
  );
  assert.equal(names.view(profile([project()])).projects[0].name, "Accepted");
  assert.equal(
    names.view(profile([project(first, "Newer", 2)])).projects[0].name,
    "Newer",
  );
  assert.equal(
    names.reconcile(profile([project(first, "Original", 0, "viewer")]))
      .projects[0].role,
    "viewer",
  );
  assert.deepEqual(names.reconcile(profile([])).projects, []);
});

test("old account completions cannot alter another account even with the same project ID", () => {
  const names = setup();
  const old = names.begin(project(), "Old account name");
  names.select("another");
  const next = profile([project(first, "Another account")], "another");
  assert.equal(names.accept(old, project(first, "Old account name", 1)), false);
  names.cancel(old);
  assert.equal(names.view(next).projects[0].name, "Another account");
  assert.equal(names.busy(first), false);
});

test("closing a context suppresses late completion and reopening starts from canonical metadata", () => {
  const names = setup(),
    old = names.begin(project(), "Late");
  names.select(null);
  names.select("owner");
  assert.equal(names.accept(old, project(first, "Late", 1)), false);
  assert.equal(names.view(profile([project()])).projects[0].name, "Original");
});

test("different project operations do not erase each other's pending or acknowledged name", () => {
  const names = setup(),
    original = profile([project(), project(second, "Second")]);
  const a = names.begin(original.projects[0], "A"),
    b = names.begin(original.projects[1], "B");
  names.accept(b, project(second, "B", 1));
  assert.deepEqual(
    names.view(original).projects.map((p) => p.name),
    ["A", "B"],
  );
  assert.deepEqual(
    names.reconcile(original).projects.map((p) => p.name),
    ["Original", "B"],
  );
  names.cancel(a);
  assert.deepEqual(
    names.view(original).projects.map((p) => p.name),
    ["Original", "B"],
  );
});

test("owner and valid version are required before any optimistic name can appear", () => {
  const names = setup();
  for (const role of ["viewer", "editor"] as const)
    assert.throws(
      () => names.begin(project(first, "Original", 0, role), "New"),
      /project owner/,
    );
  for (const bad of [-1, NaN, 1.5, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(
      () => names.begin(project(first, "Original", bad), "New"),
      /version/,
    );
  assert.equal(names.busy(first), false);
});

test("wrong identity, name or version cannot be cached as a successful rename", () => {
  for (const saved of [
    project(second, "New", 1),
    project(first, "Wrong", 1),
    project(first, "New", 0),
    project(first, "New", 2),
    { ...project(first, "New", 1), name_version: undefined },
  ]) {
    const names = setup(),
      change = names.begin(project(), "New");
    assert.throws(() => names.accept(change, saved), /Could not verify/);
    assert.equal(
      names.reconcile(profile([project()])).projects[0].name,
      "Original",
    );
    names.cancel(change);
  }
});

test("legacy projects start at zero and current unchanged names accept an idempotent response", () => {
  const names = setup(),
    legacy = { ...project(), name_version: undefined };
  const change = names.begin(legacy, "Original");
  assert.equal(change.name_version, 0);
  assert.equal(names.accept(change, project()), true);
  assert.equal(names.busy(first), false);
});

test("acknowledged names update invitation labels without replacing role or URL", () => {
  const names = setup();
  names.accept(names.begin(project(), "New"), project(first, "New", 1));
  const original = profile([project()]);
  original.invitations = [
    {
      id: "invitation",
      project_id: first,
      project_name: "Original",
      email: "guest@example.com",
      role: "viewer",
      expires_at: "2027-01-01",
      url: "https://example.com/?invite=invitation",
    },
  ];
  const result = names.reconcile(original);
  assert.deepEqual(result.invitations[0], {
    ...original.invitations[0],
    project_name: "New",
  });
  assert.equal(original.invitations[0].project_name, "Original");
});

test("names trim ordinary whitespace, count Unicode codepoints and reject raw controls", () => {
  assert.equal(projectName("  Ücret araştırması  "), "Ücret araştırması");
  assert.equal(projectName("👩‍🔬"), "👩‍🔬");
  assert.equal(projectName("😀".repeat(100)), "😀".repeat(100));
  for (const value of [
    "",
    "  ",
    "x".repeat(101),
    "\nResearch",
    "Research\t",
    "\u2028Research",
    "Research\u2029",
    "x\u0085y",
    "\ud800",
  ])
    assert.throws(() => projectName(value));
});
