import type { ConsoleState, DatasetProfile, ExecutionRecord } from "./types.ts";
import type { ResultSharingSnapshot } from "./result-sharing.ts";
import {
  documentExtension,
  validFileName,
  type FileLayout,
} from "./file-layout.ts";

const friendlyErrors: Record<string, string> = {
  CONSOLE_BUSY:
    "Python is already running. Stop the current command or wait for it to finish.",
  INVALID_CODE:
    "Enter Python code to run. Each run supports up to 64,000 characters.",
  missing_values:
    'The model variables contain missing values. Correct the data or explicitly use missing="drop".',
  invalid_binary_outcome:
    "For logit and probit, the outcome variable must contain both 0 and 1.",
  separation_detected:
    "The selected variables perfectly or partly separate the binary outcome. This model cannot produce a finite estimate; review the variables and sample.",
  singular_design:
    "Some variables are perfectly collinear or represent an empty category. Review the variable selection.",
  constant_predictor:
    "A selected predictor is constant throughout the sample. Remove it and try again.",
  nonconvergence:
    "The model did not converge. No valid result was saved; review the model and data.",
  non_finite_values:
    "A selected variable contains infinite or invalid numeric values. Correct the data and upload it again.",
  insufficient_clusters:
    "Clustered standard errors require at least two distinct clusters.",
  DATA_INTEGRITY:
    "The saved data has changed since import. Upload the original file again.",
  TOKEN_REQUIRED: "The session was renewed. Refresh the page to reconnect.",
  OWNER_REQUIRED:
    "Sign in with the owner's Google account to access this workspace.",
};

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(message: string, status: number, code = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}
export type TokenGetter = (forceRefresh?: boolean) => Promise<string>;
export interface WorkspaceSession {
  token: string;
  version: string;
  environment: "local" | "cloud" | "team";
  persistent?: boolean;
  read_only?: boolean;
  execution_mode?: "persistent" | "isolated";
}
export interface ScriptDraft {
  code: string | null;
  name: string;
  version?: number;
  pending_sync?: boolean;
}
export interface ScriptSummary {
  id: string;
  name: string;
  version: number;
  pending_sync?: boolean;
}
export interface ScriptFile extends ScriptSummary {
  code: string | null;
  sync_conflict?: boolean;
}
export interface ScriptSyncNotification {
  file?: ScriptFile;
  sync_error?: { message: string; status: number; code: string };
  sync_conflict?: boolean;
  source_changed?: boolean;
  read_only?: boolean;
}

export interface ScriptConflictSnapshot {
  base: string;
  local: ScriptFile;
  remote: ScriptFile;
}

export function scriptFilePath(id: string): string {
  if (id !== "analysis" && !/^[0-9a-f]{32}$/.test(id))
    throw new Error("Could not verify the file identity.");
  return `/console/scripts/${id}`;
}

export function validateScriptFile(
  value: ScriptFile,
  expectedId?: string,
): ScriptFile {
  if (!value) throw new Error("Could not verify the file.");
  scriptFilePath(value.id);
  if (
    (expectedId !== undefined && value.id !== expectedId) ||
    typeof value.name !== "string" ||
    !documentExtension(value.name) ||
    (value.id === "analysis" && documentExtension(value.name) !== ".py") ||
    !Number.isSafeInteger(value.version) ||
    value.version < 0 ||
    (value.code !== null &&
      (typeof value.code !== "string" ||
        Array.from(value.code).length > 64_000))
  )
    throw new Error("Could not verify the file.");
  validFileName(value.name, "script");
  return value;
}

export async function loadScriptFile(
  client: WorkspaceClient,
  id: string,
  options?: RequestInit,
): Promise<ScriptFile> {
  return validateScriptFile(
    await client.request<ScriptFile>(scriptFilePath(id), options),
    id,
  );
}

export async function loadScriptFiles(
  client: WorkspaceClient,
): Promise<ScriptSummary[]> {
  const value = await client.request<{ scripts: ScriptSummary[] }>(
    "/console/scripts",
  );
  if (!Array.isArray(value.scripts) || value.scripts.length > 200)
    throw new Error("Could not verify the file list.");
  const identifiers = new Set<string>();
  for (const file of value.scripts) {
    scriptFilePath(file.id);
    if (
      identifiers.has(file.id) ||
      typeof file.name !== "string" ||
      !documentExtension(file.name) ||
      (file.id === "analysis" && documentExtension(file.name) !== ".py") ||
      !Number.isSafeInteger(file.version) ||
      file.version < 0
    )
      throw new Error("Could not verify the file list.");
    validFileName(file.name, "script");
    identifiers.add(file.id);
  }
  if (!identifiers.has("analysis"))
    throw new Error("The main Python file could not be found.");
  return value.scripts;
}
export interface WorkspaceClient {
  readonly projectId: string | null;
  readonly teams: boolean;
  readonly desktop?: boolean;
  /** Native local runtime tools; its transport never uses the cloud broker. */
  readonly localDesktop?: boolean;
  readScriptConflict?: (id: string) => Promise<ScriptConflictSnapshot>;
  resolveScriptConflict?: (
    snapshot: ScriptConflictSnapshot,
    code: string,
  ) => Promise<ScriptFile>;
  connect: () => Promise<WorkspaceSession>;
  request: <T>(
    path: string,
    options?: RequestInit,
    plainText?: boolean | "blob",
  ) => Promise<T>;
  cancelPending: () => void;
  subscribeFileLayout?: (listener: (layout: FileLayout) => void) => () => void;
  subscribeScriptSync?: (
    listener: (value: ScriptSyncNotification) => void,
  ) => () => void;
  subscribeResultSharing?: (
    listener: (value: ResultSharingSnapshot) => void,
  ) => () => void;
}

export interface WorkspaceBootstrap {
  session: WorkspaceSession;
  draft: ScriptDraft;
  datasets: DatasetProfile[];
  status: ConsoleState["status"];
  initialConsole?: ConsoleState;
}

export interface ExecutionAccepted {
  accepted: true;
  id: string;
}

export function isExecutionAccepted(
  value: ExecutionRecord | ExecutionAccepted,
): value is ExecutionAccepted {
  return "accepted" in value && value.accepted === true;
}

/** Cloud tasks outlive an HTTP request; local sessions keep synchronous results. */
export async function submitWorkspaceExecution(
  client: WorkspaceClient,
  code: string,
  scriptId?: string,
): Promise<ExecutionRecord | ExecutionAccepted> {
  const value = await client.request<ExecutionRecord | ExecutionAccepted>(
    "/console/execute",
    {
      method: "POST",
      body: JSON.stringify({
        code,
        ...(scriptId === undefined ? {} : { script_id: scriptId }),
        ...(client.teams ? { wait_for_result: false } : {}),
      }),
    },
  );
  if (
    "accepted" in value &&
    (!client.teams ||
      value.accepted !== true ||
      typeof value.id !== "string" ||
      !value.id)
  )
    throw new Error(
      "Could not verify that the run was accepted. Refresh the project status.",
    );
  return value;
}

/** Teams open the draft without waiting for historical result blobs. */
export async function loadWorkspaceBootstrap(
  client: WorkspaceClient,
): Promise<WorkspaceBootstrap> {
  if (client.teams) return client.request<WorkspaceBootstrap>("/bootstrap");
  // Persistent local sessions retain their existing token and console protocol.
  const session = await client.connect();
  const [initialConsole, draft, data] = await Promise.all([
    client.request<ConsoleState>("/console"),
    client.request<ScriptDraft>("/console/script"),
    client.request<{ datasets: DatasetProfile[] }>("/datasets"),
  ]);
  return {
    session,
    draft,
    datasets: data.datasets,
    status: initialConsole.status,
    initialConsole,
  };
}

export async function decodeResponse<T>(
  response: Response,
  plainText: boolean | "blob" = false,
): Promise<T> {
  if (!response.ok) {
    let message = `Could not complete the request (${response.status}).`;
    let code = "";
    try {
      const error = await response.json();
      if (typeof error.detail === "string") message = error.detail;
      else if (Array.isArray(error.detail))
        message = error.detail
          .map(
            (item: { msg: string; loc?: string[] }) =>
              `${item.loc?.slice(1).join(".") || "Data"}: ${item.msg}`,
          )
          .join(" · ");
      else if (error.detail?.message) {
        code = error.detail.code || "";
        message = friendlyErrors[code] || error.detail.message;
      }
    } catch {
      /* Keep the HTTP status when the server returned no JSON body. */
    }
    throw new ApiError(message, response.status, code);
  }
  if (response.status === 204) return undefined as T;
  if (plainText === "blob") return response.blob() as Promise<T>;
  return plainText
    ? (response.text() as Promise<T>)
    : (response.json() as Promise<T>);
}

/** Every request captures this project's prefix; switching projects cannot retarget it. */
export function createWorkspaceClient(
  projectId: string | null = null,
  getToken?: TokenGetter,
  localPrefix?: string,
): WorkspaceClient {
  const prefix =
    localPrefix ??
    (projectId === null
      ? "/api"
      : `/api/projects/${encodeURIComponent(projectId)}/workspace`);
  const pending = new Set<AbortController>();
  let localToken = "";
  async function request<T>(
    path: string,
    options: RequestInit = {},
    plainText: boolean | "blob" = false,
  ): Promise<T> {
    if (!path.startsWith("/") || path.startsWith("//"))
      throw new Error("Invalid API path.");
    const controller = new AbortController();
    pending.add(controller);
    const signal = options.signal
      ? AbortSignal.any([controller.signal, options.signal])
      : controller.signal;
    try {
      const perform = async (refresh = false) => {
        const headers = new Headers(options.headers);
        if (getToken) {
          const token = await getToken(refresh);
          if (!token)
            throw new ApiError("You need to sign in.", 401, "AUTH_REQUIRED");
          headers.set("Authorization", `Bearer ${token}`);
        } else if (path !== "/session")
          headers.set("X-OpenEcon-Token", localToken);
        if (signal.aborted)
          throw new DOMException("Request cancelled", "AbortError");
        if (options.body && !(options.body instanceof FormData))
          headers.set("Content-Type", "application/json");
        return fetch(`${prefix}${path}`, { ...options, signal, headers });
      };
      let response = await perform();
      if (response.status === 401 && getToken && !signal.aborted)
        response = await perform(true);
      return await decodeResponse<T>(response, plainText);
    } finally {
      pending.delete(controller);
    }
  }
  return {
    projectId,
    teams: projectId !== null,
    request,
    connect: async () => {
      const session = await request<WorkspaceSession>("/session");
      localToken = session.token || "";
      return session;
    },
    cancelPending: () => {
      for (const controller of pending) controller.abort();
      pending.clear();
    },
  };
}

/** Serial autosaver's optimistic concurrency adapter; conflicts never overwrite local text. */
export class VersionedScriptWriter {
  private conflictError: ApiError | null = null;
  private readonly client: WorkspaceClient;
  private readonly fileId?: string;
  private version: number;
  constructor(client: WorkspaceClient, version = 0, fileId?: string) {
    this.client = client;
    this.version = version;
    if (fileId !== undefined) scriptFilePath(fileId);
    this.fileId = fileId;
  }
  get conflict(): boolean {
    return this.conflictError !== null;
  }
  async save(code: string): Promise<ScriptDraft> {
    if (this.conflictError) throw this.conflictError;
    try {
      const result = await this.client.request<ScriptDraft>(
        this.fileId ? scriptFilePath(this.fileId) : "/console/script",
        {
          method: "PUT",
          body: JSON.stringify({
            code,
            ...(this.fileId ? {} : { name: "analysis.py" }),
            ...(this.client.teams || this.fileId
              ? { version: this.version }
              : {}),
          }),
        },
      );
      if (this.client.teams || this.fileId) {
        if (
          !Number.isInteger(result.version) ||
          result.version! < this.version ||
          (this.fileId && (result as ScriptFile).id !== this.fileId)
        )
          throw new Error(
            "Could not verify the save version. Reload the current draft.",
          );
        this.version = result.version!;
      }
      return result;
    } catch (error) {
      if (error instanceof ApiError && error.status === 409)
        this.conflictError = error;
      throw error;
    }
  }
}

const localClient = createWorkspaceClient();
export const connect = localClient.connect;
export const api = localClient.request;
export { localClient };

export function download(
  content: string,
  filename: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function downloadBundle(resultId: string) {
  const session = await localClient.connect();
  const response = await fetch(
    `/api/results/${encodeURIComponent(resultId)}/bundle`,
    { headers: { "X-OpenEcon-Token": session.token } },
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      error.detail?.message || "Could not prepare the analysis bundle.",
    );
  }
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `openecon-${resultId}.zip`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function workspaceDownloadPath(url: string, projectId: string): string {
  const expected = `/api/projects/${encodeURIComponent(projectId)}/workspace`;
  if (
    !url.startsWith(`${expected}/`) ||
    url.includes("..") ||
    url.includes("\\")
  )
    throw new Error("Could not verify that the file belongs to this project.");
  return url.slice(expected.length);
}
