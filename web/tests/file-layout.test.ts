import assert from "node:assert/strict";
import test from "node:test";
import {
  createFileFolder,
  fileEntryKey,
  fileFolderPath,
  mergeFileLayout,
  moveFileEntry,
  placeFileAfter,
  placeFileBefore,
  renameFileEntry,
  reorderFileEntry,
  validFileName,
  validateFileLayout,
  documentExtension,
  documentMimeType,
  localCopyName,
  type FileLayoutEntry,
} from "../src/file-layout.ts";

const a = "a".repeat(32),
  b = "b".repeat(32),
  c = "c".repeat(32);
const original: FileLayoutEntry[] = [
  { kind: "script", id: "analysis", name: "analysis.py", parent: null },
  { kind: "folder", id: a, name: "Modeller", parent: null },
  { kind: "script", id: b, name: "model.py", parent: a },
  { kind: "dataset", id: c, name: "wage.csv", parent: null },
];

test("the main script, data and folders rename without changing identities or paths", () => {
  for (const [key, name] of [
    ["script:analysis", "ücret.py"],
    [`dataset:${c}`, "maaş.csv"],
    [`folder:${a}`, "Analizler"],
  ]) {
    const next = renameFileEntry(original, key, name);
    assert.equal(next.find((entry) => fileEntryKey(entry) === key)?.name, name);
    assert.deepEqual(
      next.map(({ id, kind, parent }) => ({ id, kind, parent })),
      original.map(({ id, kind, parent }) => ({ id, kind, parent })),
    );
  }
  assert.equal(original[0].name, "analysis.py");
  assert.throws(
    () => renameFileEntry(original, `dataset:${c}`, "wage.dta"),
    /extension/,
  );
  assert.throws(
    () => renameFileEntry(original, "script:analysis", "result.txt"),
    /\.py/,
  );
});

test("names normalize NFC and reject unsafe or oversized cross-platform names", () => {
  assert.equal(validFileName("u\u0308cret.py", "script"), "ücret.py");
  for (const name of [
    "",
    ".hidden",
    "..",
    "a/b",
    "a\\b",
    "a:",
    "NUL.csv",
    "con",
    "LPT1.txt",
    "name.",
    "name ",
    " name",
    "x\u0000",
    "x\u200b",
    "x\u2028y",
    "😀".repeat(46),
  ])
    assert.throws(() => validFileName(name, "folder"), /valid name/);
  assert.equal(validFileName("Model.v2", "folder"), "Model.v2");
  assert.throws(() => validFileName("bad .py", "script"), /\.py/);
  assert.throws(() => validFileName("bad..py", "script"), /\.py/);
});

test("sibling collisions include case and common Unicode casefold across kinds", () => {
  const entries = createFileFolder(original, "STRASSE", null, "d".repeat(32));
  assert.throws(
    () => renameFileEntry(entries, `folder:${a}`, "Straße"),
    /already contains/,
  );
  assert.throws(
    () => createFileFolder(original, "ANALYSIS.PY", null, "e".repeat(32)),
    /already contains/,
  );
  assert.doesNotThrow(() =>
    createFileFolder(original, "analysis.py", a, "e".repeat(32)),
  );
});

test("logical moves keep descendants and reject cycles, absent destinations and name collisions", () => {
  const nested = createFileFolder(original, "İç", a, "d".repeat(32));
  assert.throws(
    () => moveFileEntry(nested, `folder:${a}`, "d".repeat(32)),
    /into itself/,
  );
  assert.throws(
    () => moveFileEntry(original, `folder:${a}`, a),
    /into itself/,
  );
  assert.throws(
    () => moveFileEntry(original, "script:analysis", "missing"),
    /not found/,
  );
  const moved = moveFileEntry(original, `dataset:${c}`, a);
  assert.equal(moved.at(-1)?.parent, a);
  assert.equal(moved.find((entry) => entry.id === b)?.parent, a);
  const duplicate = createFileFolder(original, "wage.csv", a, "e".repeat(32));
  assert.throws(() => moveFileEntry(duplicate, `dataset:${c}`, a), /already contains/);
});

test("ordering swaps only siblings and drag placement can move between folders", () => {
  assert.deepEqual(
    reorderFileEntry(original, `dataset:${c}`, -1).map(fileEntryKey),
    ["script:analysis", `dataset:${c}`, `script:${b}`, `folder:${a}`],
  );
  assert.deepEqual(reorderFileEntry(original, `script:${b}`, -1), original);
  const moved = placeFileBefore(original, "script:analysis", `script:${b}`);
  assert.equal(moved.find((entry) => entry.id === "analysis")?.parent, a);
  assert.deepEqual(
    moved.filter((entry) => entry.parent === a).map(fileEntryKey),
    ["script:analysis", `script:${b}`],
  );
});

test("after placement preserves folder descendants and can append the final sibling", () => {
  const afterFolder = placeFileAfter(
    original,
    "script:analysis",
    `folder:${a}`,
  );
  assert.deepEqual(
    afterFolder.filter((entry) => entry.parent === null).map(fileEntryKey),
    [`folder:${a}`, "script:analysis", `dataset:${c}`],
  );
  assert.equal(afterFolder.find((entry) => entry.id === b)?.parent, a);
  const final = placeFileAfter(afterFolder, "script:analysis", `dataset:${c}`);
  assert.deepEqual(
    final.filter((entry) => entry.parent === null).map(fileEntryKey),
    [`folder:${a}`, `dataset:${c}`, "script:analysis"],
  );
  const nested = placeFileAfter(original, "script:analysis", `script:${b}`);
  assert.deepEqual(
    nested.filter((entry) => entry.parent === a).map(fileEntryKey),
    [`script:${b}`, "script:analysis"],
  );
  assert.equal(original[0].parent, null);
});

test("after placement rejects cycles and collisions and self placement leaves order unchanged", () => {
  assert.throws(
    () => placeFileAfter(original, `folder:${a}`, `script:${b}`),
    /into itself/,
  );
  const duplicate = createFileFolder(original, "wage.csv", a, "e".repeat(32));
  assert.throws(
    () => placeFileAfter(duplicate, `dataset:${c}`, `script:${b}`),
    /already contains/,
  );
  assert.throws(
    () => placeFileAfter(original, "script:analysis", "script:missing"),
    /not found/,
  );
  assert.deepEqual(
    placeFileAfter(original, `script:${b}`, `script:${b}`),
    original,
  );
});

test("sixteen folder levels are valid and a seventeenth is rejected", () => {
  const entries: FileLayoutEntry[] = [];
  for (let i = 0; i < 17; i++)
    entries.push({
      kind: "folder",
      id: String(i).padStart(32, "0"),
      name: `folder${i}`,
      parent: i ? String(i - 1).padStart(32, "0") : null,
    });
  assert.doesNotThrow(() => validateFileLayout(entries));
  assert.throws(
    () =>
      createFileFolder(entries, "nested", entries.at(-1)!.id, "f".repeat(32)),
    /16/,
  );
});

test("merging new inventory preserves renamed metadata, folder membership and order", () => {
  const layout = renameFileEntry(original, "script:analysis", "result.py");
  const merged = mergeFileLayout(
    layout,
    [
      { id: "analysis", name: "analysis.py" },
      { id: b, name: "model.py" },
      { id: "d".repeat(32), name: "result.py" },
    ],
    [
      { id: c, name: "wage.csv" },
      { id: "e".repeat(32), name: "wage.csv" },
    ],
  );
  assert.deepEqual(merged.slice(0, 4), layout);
  assert.equal(merged[4].name, "result (2).py");
  assert.equal(merged[5].name, "wage (2).csv");
  assert.equal(merged[4].parent, null);
  assert.equal(
    fileFolderPath(
      createFileFolder(merged, "İç", a, "f".repeat(32)),
      "f".repeat(32),
    ),
    "Modeller / İç",
  );
});

test("defaults reconcile removed files, typed IDs and legacy unsafe names deterministically", () => {
  const sameId = [{ id: "analysis", name: "analysis.py" }];
  const merged = mergeFileLayout(original, sameId, [
    { id: "analysis", name: "ANALYSIS.PY" },
    { id: c, name: "wage.csv" },
  ]);
  assert.equal(
    merged.some((entry) => entry.id === b),
    false,
  );
  assert.equal(merged.at(-1)?.name, "ANALYSIS (2).PY");
  validateFileLayout(merged);
  const long = "ü".repeat(88) + ".csv";
  const defaults = mergeFileLayout(
    undefined,
    [],
    [
      { id: a, name: long },
      { id: b, name: long },
      { id: c, name: "../CON.csv" },
    ],
  );
  for (const entry of defaults)
    assert.ok(new TextEncoder().encode(entry.name).byteLength <= 180);
  validateFileLayout(defaults);
  assert.deepEqual(
    mergeFileLayout(
      undefined,
      [],
      [
        { id: a, name: "wage.csv" },
        { id: b, name: "wage.csv" },
      ],
    ).map((entry) => entry.name),
    ["wage.csv", "wage (2).csv"],
  );
});

test("inventory version and synchronization metadata never leak into layout entries", () => {
  const script = {
    id: "analysis",
    name: "analysis.py",
    version: 5,
    pending_sync: true,
  };
  const entries = mergeFileLayout(undefined, [script], []);
  assert.deepEqual(entries, [
    { kind: "script", id: "analysis", name: "analysis.py", parent: null },
  ]);
});

test("named documents change effective type without changing immutable identity, while main remains Python", () => {
  for (const name of ["paper.md", "paper.tex", "paper.py"]) {
    const renamed = renameFileEntry(original, `script:${b}`, name);
    assert.equal(renamed[2].name, name);
    assert.equal(renamed[2].id, b);
    assert.equal(renamed[2].parent, a);
    validateFileLayout(renamed);
  }
  for (const name of ["main.md", "main.tex"])
    assert.throws(
      () => renameFileEntry(original, "script:analysis", name),
      /main file/,
    );
  for (const name of ["paper .md", "paper..tex", "paper.MD", "paper.txt"])
    assert.throws(() => validFileName(name, "script"));
});

test("document exports preserve the effective extension and content MIME", () => {
  assert.equal(documentExtension("notes.md"), ".md");
  assert.equal(documentExtension("table.tex"), ".tex");
  assert.equal(documentExtension("code.py"), ".py");
  assert.equal(localCopyName("notes.md"), "notes-local-copy.md");
  assert.equal(localCopyName("table.tex"), "table-local-copy.tex");
  assert.equal(documentMimeType("notes.md"), "text/markdown;charset=utf-8");
  assert.equal(
    documentMimeType("table.tex"),
    "application/x-tex;charset=utf-8",
  );
});

test("inline document names normalize case and Unicode and append Python only without a suffix", async () => {
  const { normalizeDocumentName } = await import("../src/file-layout.ts");
  assert.equal(normalizeDocumentName("README"), "README.py");
  assert.equal(normalizeDocumentName("README.MD"), "README.md");
  assert.equal(normalizeDocumentName("paper.TEX"), "paper.tex");
  assert.equal(normalizeDocumentName("u\u0308cret.PY"), "ücret.py");
  for (const value of [
    "",
    " name",
    "name ",
    "name.txt",
    "name.",
    ".hidden",
    "../name.md",
  ])
    assert.throws(() => normalizeDocumentName(value));
});
