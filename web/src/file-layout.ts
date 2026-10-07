export type FileEntryKind = "script" | "dataset" | "folder";

export interface FileLayoutEntry {
  kind: FileEntryKind;
  id: string;
  name: string;
  parent: string | null;
}

export interface FileLayout {
  version: number;
  entries: FileLayoutEntry[];
  pending_sync?: boolean;
  sync_conflict?: boolean;
  sync_error?: { message: string; status: number; code: string };
}

export const fileEntryKey = (entry: Pick<FileLayoutEntry, "kind" | "id">) =>
  `${entry.kind}:${entry.id}`;

export type DocumentExtension = ".py" | ".md" | ".tex";

export function documentExtension(name: string): DocumentExtension | null {
  const match = /\.(py|md|tex)$/.exec(name);
  return match ? (match[0] as DocumentExtension) : null;
}

export function normalizeDocumentName(value: string): string {
  const name = value
    .normalize("NFC")
    .replace(/\.(py|md|tex)$/i, (suffix) => suffix.toLowerCase());
  return validFileName(name.includes(".") ? name : `${name}.py`, "script");
}

export function documentMimeType(name: string): string {
  return documentExtension(name) === ".md"
    ? "text/markdown;charset=utf-8"
    : documentExtension(name) === ".tex"
      ? "application/x-tex;charset=utf-8"
      : "text/x-python;charset=utf-8";
}

export function localCopyName(name: string): string {
  return name.replace(/(\.(?:py|md|tex))$/, "-local-copy$1");
}

export function fileNameFold(name: string): string {
  return name
    .normalize("NFC")
    .toLowerCase()
    .replaceAll("ß", "ss")
    .replaceAll("ς", "σ");
}

function extension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot) : "";
}

export function validFileName(
  value: string,
  kind: FileEntryKind,
  previousName?: string,
): string {
  const name = value.normalize("NFC");
  if (
    !name ||
    name !== name.trim() ||
    name.startsWith(".") ||
    /[. ]$/.test(name) ||
    /[\p{C}\u2028\u2029/\\<>:"|?*]/u.test(name) ||
    new TextEncoder().encode(name).byteLength > 180 ||
    /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name)
  )
    throw new Error("Enter a valid name (up to 180 bytes).");
  if (
    kind === "script" &&
    (!documentExtension(name) ||
      /[. ]$/.test(name.slice(0, -documentExtension(name)!.length)))
  )
    throw new Error("The file extension must be .py, .md or .tex.");
  if (
    kind === "dataset" &&
    previousName !== undefined &&
    fileNameFold(extension(name)) !== fileNameFold(extension(previousName))
  )
    throw new Error("Keep the data file's extension unchanged.");
  return name;
}

export function validateFileLayout(entries: readonly FileLayoutEntry[]): void {
  const keys = new Set<string>();
  const folders = new Map<string, FileLayoutEntry>();
  for (const entry of entries) {
    const key = fileEntryKey(entry);
    if (keys.has(key))
      throw new Error("The same file appears more than once.");
    keys.add(key);
    validFileName(entry.name, entry.kind);
    if (
      entry.kind === "script" &&
      entry.id === "analysis" &&
      documentExtension(entry.name) !== ".py"
    )
      throw new Error("The main file must have a .py extension.");
    if (entry.kind === "folder") folders.set(entry.id, entry);
  }
  const siblings = new Set<string>();
  for (const entry of entries) {
    if (entry.parent !== null && !folders.has(entry.parent))
      throw new Error("Folder not found.");
    const sibling = `${entry.parent ?? ""}\u0000${fileNameFold(entry.name)}`;
    if (siblings.has(sibling))
      throw new Error("This folder already contains an item with that name.");
    siblings.add(sibling);
    const ancestors = new Set<string>();
    if (entry.kind === "folder") ancestors.add(entry.id);
    let parent = entry.parent;
    let depth = 0;
    while (parent !== null) {
      if (ancestors.has(parent))
        throw new Error("You cannot move a folder into itself.");
      ancestors.add(parent);
      if (++depth > 16)
        throw new Error("Folders can be nested up to 16 levels deep.");
      parent = folders.get(parent)!.parent;
    }
  }
}

function trimUtf8(value: string, bytes: number): string {
  let result = "";
  for (const character of value) {
    if (new TextEncoder().encode(result + character).byteLength > bytes) break;
    result += character;
  }
  return result;
}

function defaultName(value: string): string {
  let name = value
    .normalize("NFC")
    .replace(/[\p{C}\u2028\u2029/\\<>:"|?*]/gu, "_")
    .trim()
    .replace(/^\.+/, "")
    .replace(/[. ]+$/, "");
  if (!name) name = "data";
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name))
    name = "_" + name;
  const suffix = extension(name);
  return (
    trimUtf8(
      suffix ? name.slice(0, -suffix.length) : name,
      180 - new TextEncoder().encode(suffix).byteLength,
    ).replace(/[. ]+$/, "") + suffix
  );
}

function uniqueName(
  name: string,
  kind: FileEntryKind,
  parent: string | null,
  entries: readonly FileLayoutEntry[],
): string {
  const taken = new Set(
    entries
      .filter((entry) => entry.parent === parent)
      .map((entry) => fileNameFold(entry.name)),
  );
  if (!taken.has(fileNameFold(name))) return name;
  const suffix = kind === "folder" ? "" : extension(name);
  const stem = suffix ? name.slice(0, -suffix.length) : name;
  for (let number = 2; ; number++) {
    const ending = ` (${number})${suffix}`;
    const candidate =
      trimUtf8(stem, 180 - new TextEncoder().encode(ending).byteLength).replace(
        /[. ]+$/,
        "",
      ) + ending;
    if (!taken.has(fileNameFold(candidate))) return candidate;
  }
}

/** Reconcile additions/deletions while preserving every known entry's order and folder. */
export function mergeFileLayout(
  layout: readonly FileLayoutEntry[] | undefined,
  scripts: readonly { id: string; name: string }[],
  files: readonly { id: string; name: string }[],
): FileLayoutEntry[] {
  const inventory: FileLayoutEntry[] = [
    ...scripts.map((entry) => ({
      id: entry.id,
      name: entry.name,
      kind: "script" as const,
      parent: null,
    })),
    ...files.map((entry) => ({
      id: entry.id,
      name: entry.name,
      kind: "dataset" as const,
      parent: null,
    })),
  ];
  const known = new Set(inventory.map(fileEntryKey));
  const result: FileLayoutEntry[] = [];
  const keys = new Set<string>();
  for (const entry of layout ?? []) {
    const key = fileEntryKey(entry);
    if (keys.has(key) || (entry.kind !== "folder" && !known.has(key))) continue;
    keys.add(key);
    result.push({ ...entry });
  }
  const folders = new Set(
    result.filter((entry) => entry.kind === "folder").map((entry) => entry.id),
  );
  for (const entry of result) {
    if (entry.parent !== null && !folders.has(entry.parent))
      entry.parent = null;
  }
  for (const entry of inventory) {
    if (keys.has(fileEntryKey(entry))) continue;
    result.push({
      ...entry,
      name: uniqueName(defaultName(entry.name), entry.kind, null, result),
    });
    keys.add(fileEntryKey(entry));
  }
  return result;
}

function changedLayout(
  entries: readonly FileLayoutEntry[],
  key: string,
  update: (entry: FileLayoutEntry) => FileLayoutEntry,
): FileLayoutEntry[] {
  let found = false;
  const next = entries.map((entry) => {
    if (fileEntryKey(entry) !== key) return { ...entry };
    found = true;
    return update(entry);
  });
  if (!found) throw new Error("File not found.");
  validateFileLayout(next);
  return next;
}

export function renameFileEntry(
  entries: readonly FileLayoutEntry[],
  key: string,
  name: string,
): FileLayoutEntry[] {
  return changedLayout(entries, key, (entry) => ({
    ...entry,
    name: validFileName(name, entry.kind, entry.name),
  }));
}

export function createFileFolder(
  entries: readonly FileLayoutEntry[],
  name: string,
  parent: string | null = null,
  id = crypto.randomUUID().replaceAll("-", ""),
): FileLayoutEntry[] {
  const next = [
    ...entries.map((entry) => ({ ...entry })),
    {
      kind: "folder" as const,
      id,
      name: validFileName(name, "folder"),
      parent,
    },
  ];
  validateFileLayout(next);
  return next;
}

export function moveFileEntry(
  entries: readonly FileLayoutEntry[],
  key: string,
  parent: string | null,
): FileLayoutEntry[] {
  const entry = entries.find((item) => fileEntryKey(item) === key);
  if (!entry) throw new Error("File not found.");
  const next = entries
    .filter((item) => fileEntryKey(item) !== key)
    .map((item) => ({ ...item }));
  next.push({ ...entry, parent });
  validateFileLayout(next);
  return next;
}

export function reorderFileEntry(
  entries: readonly FileLayoutEntry[],
  key: string,
  direction: -1 | 1,
): FileLayoutEntry[] {
  const index = entries.findIndex((entry) => fileEntryKey(entry) === key);
  if (index < 0) throw new Error("File not found.");
  const siblings = entries
    .map((entry, i) => ({ entry, i }))
    .filter(({ entry }) => entry.parent === entries[index].parent);
  const siblingIndex = siblings.findIndex(({ i }) => i === index);
  const other = siblings[siblingIndex + direction];
  if (!other) return entries.map((entry) => ({ ...entry }));
  const next = entries.map((entry) => ({ ...entry }));
  [next[index], next[other.i]] = [next[other.i], next[index]];
  return next;
}

export function placeFileBefore(
  entries: readonly FileLayoutEntry[],
  key: string,
  targetKey: string,
): FileLayoutEntry[] {
  return placeFileBeside(entries, key, targetKey, "before");
}

export function placeFileAfter(
  entries: readonly FileLayoutEntry[],
  key: string,
  targetKey: string,
): FileLayoutEntry[] {
  return placeFileBeside(entries, key, targetKey, "after");
}

function placeFileBeside(
  entries: readonly FileLayoutEntry[],
  key: string,
  targetKey: string,
  position: "before" | "after",
): FileLayoutEntry[] {
  if (key === targetKey) return entries.map((entry) => ({ ...entry }));
  const source = entries.find((entry) => fileEntryKey(entry) === key);
  const target = entries.find((entry) => fileEntryKey(entry) === targetKey);
  if (!source || !target) throw new Error("File not found.");
  const next = entries
    .filter((entry) => fileEntryKey(entry) !== key)
    .map((entry) => ({ ...entry }));
  next.splice(
    next.findIndex((entry) => fileEntryKey(entry) === targetKey) +
      (position === "after" ? 1 : 0),
    0,
    { ...source, parent: target.parent },
  );
  validateFileLayout(next);
  return next;
}

export function fileFolderPath(
  entries: readonly FileLayoutEntry[],
  id: string,
): string {
  const folders = new Map(
    entries
      .filter((entry) => entry.kind === "folder")
      .map((entry) => [entry.id, entry]),
  );
  const parts: string[] = [];
  const seen = new Set<string>();
  let current: string | null = id;
  while (current !== null && !seen.has(current)) {
    seen.add(current);
    const entry: FileLayoutEntry | undefined = folders.get(current);
    if (!entry) break;
    parts.unshift(entry.name);
    current = entry.parent;
  }
  return parts.join(" / ");
}
