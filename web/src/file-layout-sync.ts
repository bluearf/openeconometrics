/** Logical names and folders synchronize independently from executable files. */
import { ApiError, type WorkspaceClient } from "./api.ts";
import { isTransientDesktopFailure } from "./desktop.ts";
import { type DesktopSyncState } from "./desktop-script-sync.ts";
import {
  validateFileLayout,
  type FileLayout,
  type FileLayoutEntry,
} from "./file-layout.ts";

export interface FileLayoutSyncCache {
  version: number;
  cloud_version: number | null;
  base_entries: FileLayoutEntry[];
  conflict: boolean;
}
interface Dependencies {
  local: WorkspaceClient;
  cloud<T>(path: string, options?: RequestInit): Promise<T>;
  state(): DesktopSyncState | null;
  persist(state: DesktopSyncState): Promise<void>;
}
interface DatasetMapping {
  cloud: Map<string, string>;
  local: Map<string, string>;
  localOnly: Set<string>;
}
const LOCAL = "/desktop/file-layout";
const CACHE = "/desktop/file-layout-sync";
const HEX = /^[0-9a-f]{32}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const json = (body: unknown): RequestInit => ({
  method: "PUT",
  body: JSON.stringify(body),
});
const equal = (
  left: readonly FileLayoutEntry[],
  right: readonly FileLayoutEntry[],
) => JSON.stringify(left) === JSON.stringify(right);
const changed = () =>
  new ApiError(
    "The cloud file layout has changed. Your local layout is preserved.",
    409,
    "VERSION_CONFLICT",
  );
const inventoryChanged = () =>
  new ApiError(
    "The file list has changed. Reopen the project; your local layout is preserved.",
    409,
    "FILE_INVENTORY_CONFLICT",
  );
const denied = () =>
  new ApiError("You do not have permission to edit this project.", 403, "ROLE_REQUIRED");

function entries(value: unknown, cloud = false): FileLayoutEntry[] {
  if (!Array.isArray(value) || value.length > 2000)
    throw new Error("Could not verify the file layout.");
  const result = value.map((item: unknown) => {
    if (!item || typeof item !== "object" || Array.isArray(item))
      throw new Error("Could not verify the file layout.");
    const entry = item as FileLayoutEntry;
    if (
      Object.keys(entry).some(
        (key) => !["kind", "id", "name", "parent"].includes(key),
      ) ||
      !["script", "dataset", "folder"].includes(entry.kind) ||
      typeof entry.id !== "string" ||
      typeof entry.name !== "string" ||
      (entry.parent !== null &&
        (typeof entry.parent !== "string" || !HEX.test(entry.parent))) ||
      (entry.kind === "script" &&
        entry.id !== "analysis" &&
        !HEX.test(entry.id)) ||
      (entry.kind === "folder" && !HEX.test(entry.id)) ||
      (entry.kind === "dataset" &&
        !(HEX.test(entry.id) || (!cloud && UUID.test(entry.id))))
    )
      throw new Error("Could not verify the file layout.");
    return {
      kind: entry.kind,
      id: entry.id,
      name: entry.name,
      parent: entry.parent,
    };
  });
  validateFileLayout(result);
  return result;
}
function layout(value: unknown, cloud = false): FileLayout {
  if (!value || typeof value !== "object")
    throw new Error("Could not verify the file layout.");
  const document = value as FileLayout;
  if (!Number.isSafeInteger(document.version) || document.version < 0)
    throw new Error("Could not verify the file layout version.");
  return {
    version: document.version,
    entries: entries(document.entries, cloud),
  };
}
function cacheDocument(value: unknown): FileLayoutSyncCache {
  if (value === null)
    return {
      version: 0,
      cloud_version: null,
      base_entries: [],
      conflict: false,
    };
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Could not verify the cached file layout.");
  const document = value as FileLayoutSyncCache;
  if (
    Object.keys(document).some(
      (key) =>
        !["version", "cloud_version", "base_entries", "conflict"].includes(key),
    ) ||
    !Number.isSafeInteger(document.version) ||
    document.version < 0 ||
    (document.cloud_version !== null &&
      (!Number.isSafeInteger(document.cloud_version) ||
        document.cloud_version < 0)) ||
    typeof document.conflict !== "boolean"
  )
    throw new Error("Could not verify the cached file layout.");
  return {
    version: document.version,
    cloud_version: document.cloud_version,
    base_entries: entries(document.base_entries, true),
    conflict: document.conflict,
  };
}

/** Cloud operations serialize externally; short local commits have their own queue. */
export function createFileLayoutSync(deps: Dependencies) {
  let active = false;
  let cancelled = false;
  let localSerial: Promise<unknown> = Promise.resolve();
  let unacknowledged: { version: number; entries: FileLayoutEntry[] } | null =
    null;
  function localOperation<T>(operation: () => Promise<T>): Promise<T> {
    const next = localSerial.catch(() => {}).then(operation);
    localSerial = next;
    return next;
  }
  function current(allowDenied = false) {
    if (cancelled) throw new DOMException("Workspace closed", "AbortError");
    const state = deps.state();
    if (!state) throw new Error("Open the project first.");
    if (state.access_denied && !allowDenied) throw denied();
    return state;
  }
  async function cloud<T>(path: string, options?: RequestInit): Promise<T> {
    try {
      return await deps.cloud<T>(path, options);
    } catch (error) {
      if (
        error instanceof ApiError &&
        (error.status === 401 || error.status === 403)
      )
        await deps.persist({ ...current(), access_denied: true });
      throw error;
    }
  }
  async function mapping(): Promise<DatasetMapping> {
    const document = await deps.local.request<{
      files: { cloud_id: string; dataset_id: string }[];
      local_only?: string[];
    }>("/desktop-cached-files");
    if (
      !document ||
      !Array.isArray(document.files) ||
      document.files.length > 2000
    )
      throw new Error("Could not verify the cached data files.");
    const result: DatasetMapping = { cloud: new Map(), local: new Map(), localOnly: new Set() };
    for (const file of document.files) {
      if (
        !file ||
        !HEX.test(file.cloud_id) ||
        !UUID.test(file.dataset_id) ||
        result.cloud.has(file.cloud_id) ||
        result.local.has(file.dataset_id)
      )
        throw new Error("Could not verify the cached data files.");
      result.cloud.set(file.cloud_id, file.dataset_id);
      result.local.set(file.dataset_id, file.cloud_id);
    }
    if (document.local_only !== undefined &&
        (!Array.isArray(document.local_only) || document.local_only.length > 2000))
      throw new Error("Could not verify the local-only data files.");
    for (const id of document.local_only ?? []) {
      if (typeof id !== "string" || !UUID.test(id) || result.local.has(id) || result.localOnly.has(id))
        throw new Error("Could not verify the local-only data files.");
      result.localOnly.add(id);
    }
    return result;
  }
  function cloudEntries(
    local: FileLayout,
    map: DatasetMapping,
  ): FileLayoutEntry[] | null {
    const result: FileLayoutEntry[] = [];
    for (const entry of local.entries) {
      if (entry.kind === "dataset" && map.localOnly.has(entry.id)) continue;
      const id = entry.kind === "dataset" ? map.local.get(entry.id) : entry.id;
      // An unshared local dataset must keep its place, never disappear from a PUT.
      if (!id) return null;
      result.push({ ...entry, id });
    }
    return result;
  }
  function localEntries(
    remote: FileLayout,
    map: DatasetMapping,
    local: FileLayout,
  ): FileLayoutEntry[] {
    const result = remote.entries.map((entry) => {
      const id = entry.kind === "dataset" ? map.cloud.get(entry.id) : entry.id;
      if (!id) throw inventoryChanged();
      return { ...entry, id };
    });
    // Local-only data never belongs to the cloud inventory, but must survive
    // remote renames/reorders and reopening. Anchor it to the next surviving
    // shared entry so consecutive local entries retain their relative order.
    const key = (entry: FileLayoutEntry) => `${entry.kind}:${entry.id}`;
    const privateEntries = local.entries.filter((entry) => entry.kind === "dataset" && map.localOnly.has(entry.id));
    for (const entry of privateEntries) {
      if (entry.parent !== null && !result.some((candidate) => candidate.kind === "folder" && candidate.id === entry.parent))
        throw inventoryChanged(); // A remotely deleted parent cannot erase local data.
      const index = local.entries.findIndex((candidate) => key(candidate) === key(entry));
      const next = local.entries.slice(index + 1).find((candidate) =>
        candidate.parent === entry.parent && result.some((present) => key(present) === key(candidate)));
      const previousPrivate = privateEntries.slice(0, privateEntries.indexOf(entry)).filter((candidate) => candidate.parent === entry.parent).at(-1);
      const minimum = previousPrivate ? result.findIndex((candidate) => key(candidate) === key(previousPrivate)) + 1 : 0;
      const insertion = Math.max(minimum, next ? result.findIndex((candidate) => key(candidate) === key(next)) : result.length);
      result.splice(insertion, 0, { ...entry });
    }
    try { validateFileLayout(result); }
    catch { throw inventoryChanged(); } // Preserve both inventories on name collisions.
    return result;
  }
  function decorated(
    local: FileLayout,
    cache: FileLayoutSyncCache,
    map: DatasetMapping,
  ): FileLayout {
    const desired = cloudEntries(local, map);
    return {
      ...local,
      pending_sync:
        desired === null ||
        (cache.cloud_version === null
          ? local.version > 0
          : !equal(desired, cache.base_entries)),
      sync_conflict: cache.conflict,
    };
  }
  async function snapshot() {
    const [stored, saved, map] = await Promise.all([
      deps.local.request(LOCAL),
      deps.local.request(CACHE),
      mapping(),
    ]);
    return { local: layout(stored), cache: cacheDocument(saved), map };
  }
  async function readLocal(allowDenied = false): Promise<FileLayout> {
    active = true;
    current(allowDenied);
    return localOperation(async () => {
      const { local, cache, map } = await snapshot();
      return decorated(local, cache, map);
    });
  }
  async function saveLocal(body: string): Promise<FileLayout> {
    active = true;
    if (current().role === "viewer") throw denied();
    const payload = JSON.parse(body);
    if (
      !payload ||
      Object.keys(payload).some((key) => !["version", "entries"].includes(key))
    )
      throw new Error("Could not verify the file layout.");
    const requested = layout(payload);
    return localOperation(async () => {
      const [saved, map] = await Promise.all([
        deps.local.request(CACHE),
        mapping(),
      ]);
      const cache = cacheDocument(saved);
      // Known denial/conflict must fail before changing the durable local tree.
      if (current().role === "viewer") throw denied();
      if (cache.conflict) throw changed();
      const local = layout(await deps.local.request(LOCAL, json(requested)));
      return decorated(local, cache, map);
    });
  }
  async function remember(
    previous: FileLayoutSyncCache,
    remote: FileLayout,
    conflict = false,
  ) {
    return cacheDocument(
      await deps.local.request(
        CACHE,
        json({
          version: previous.version,
          cloud_version: remote.version,
          base_entries: remote.entries,
          conflict,
        }),
      ),
    );
  }
  async function markConflict(previous: FileLayoutSyncCache) {
    if (previous.conflict) return previous;
    return cacheDocument(
      await deps.local.request(CACHE, json({ ...previous, conflict: true })),
    );
  }
  async function synchronize(
    local: FileLayout,
    cache: FileLayoutSyncCache,
    map: DatasetMapping,
  ): Promise<FileLayout> {
    if (cache.conflict) throw changed();
    const desired = cloudEntries(local, map);
    if (desired === null || cache.cloud_version === null)
      return decorated(local, cache, map);
    if (equal(desired, cache.base_entries)) return decorated(local, cache, map);
    try {
      unacknowledged = { version: cache.cloud_version, entries: desired };
      const remote = layout(
        await cloud(
          "/files/layout",
          json({ version: cache.cloud_version, entries: desired }),
        ),
        true,
      );
      if (
        !equal(remote.entries, desired) ||
        remote.version < cache.cloud_version
      )
        throw new Error("Could not verify the file layout's cloud record.");
      return await localOperation(async () => {
        current();
        await remember(cache, remote);
        unacknowledged = null;
        const latest = await snapshot();
        return decorated(latest.local, latest.cache, latest.map);
      });
    } catch (error) {
      if (
        error instanceof ApiError &&
        error.status === 409 &&
        error.code !== "FILE_INVENTORY_CONFLICT"
      ) {
        // A successful write whose acknowledgement was lost is safe to settle.
        const remote = layout(await cloud("/files/layout"), true);
        if (equal(remote.entries, desired))
          return localOperation(async () => {
            current();
            await remember(cache, remote);
            unacknowledged = null;
            const latest = await snapshot();
            return decorated(latest.local, latest.cache, latest.map);
          });
        await localOperation(() => markConflict(cache));
        unacknowledged = null;
        throw changed();
      }
      if (!isTransientDesktopFailure(error)) throw error;
      return readLocal();
    }
  }
  async function read(resolveConflict = false): Promise<FileLayout> {
    active = true;
    current();
    let remote: FileLayout;
    try {
      remote = layout(await cloud("/files/layout"), true);
    } catch (error) {
      if (resolveConflict || !isTransientDesktopFailure(error)) throw error;
      return readLocal();
    }
    // Never adopt a snapshot captured before the network await: local renames
    // can complete while that request is still in flight.
    const decision = await localOperation<{
      value: FileLayout;
      upload?: Awaited<ReturnType<typeof snapshot>>;
    }>(async () => {
      current();
      let { local, cache, map } = await snapshot();
      if (
        unacknowledged &&
        cache.cloud_version === unacknowledged.version &&
        remote.version > unacknowledged.version &&
        equal(remote.entries, unacknowledged.entries)
      ) {
        cache = await remember(cache, remote);
        unacknowledged = null;
      }
      const incoming = localEntries(remote, map, local);
      const desired = cloudEntries(local, map);
      const dirty =
        desired === null ||
        (cache.cloud_version === null
          ? local.version > 0
          : !equal(desired, cache.base_entries));
      if (resolveConflict || (!dirty && !cache.conflict)) {
        if (!equal(local.entries, incoming))
          local = layout(
            await deps.local.request(
              LOCAL,
              json({ version: local.version, entries: incoming }),
            ),
          );
        cache = await remember(cache, remote);
        return { value: decorated(local, cache, map) };
      }
      if (desired !== null && equal(desired, remote.entries)) {
        cache = await remember(cache, remote);
        return { value: decorated(local, cache, map) };
      }
      if (
        cache.conflict ||
        (cache.cloud_version === null
          ? remote.version > 0
          : remote.version !== cache.cloud_version)
      ) {
        cache = await markConflict(cache);
        return { value: decorated(local, cache, map) };
      }
      if (cache.cloud_version === null) cache = await remember(cache, remote);
      return {
        value: decorated(local, cache, map),
        ...(current().role === "viewer"
          ? {}
          : { upload: { local, cache, map } }),
      };
    });
    return decision.upload
      ? synchronize(
          decision.upload.local,
          decision.upload.cache,
          decision.upload.map,
        )
      : decision.value;
  }
  async function save(body: string): Promise<FileLayout> {
    await saveLocal(body);
    const result = await read();
    if (result.sync_conflict) throw changed();
    return result;
  }
  async function retry() {
    if (!active) return;
    const state = deps.state();
    if (!state || state.access_denied || state.role === "viewer") return;
    try {
      await read();
    } catch (error) {
      if (!(
        error instanceof ApiError &&
        error.status === 409 &&
        error.code !== "FILE_INVENTORY_CONFLICT"
      ))
        throw error;
    }
  }
  return {
    read,
    readLocal,
    save,
    saveLocal,
    retry,
    cancel: () => {
      cancelled = true;
      active = false;
    },
  };
}
