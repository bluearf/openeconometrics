import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import type { WorkspaceClient } from "./api";
import { download } from "./api";
import {
  boundedPackageLog,
  normalizePackageName,
  type ProjectPackages,
  type SharedPackages,
  type PortableEnvironment,
  type EnvironmentPreview,
} from "./package-types";
import "./package-styles.css";
import { normalizePackageCommand } from "./package-command";

export interface PackageManagerProps {
  client: WorkspaceClient;
  disabled?: boolean;
  readOnly?: boolean;
  onRunCommand: (source: string) => void;
  onEnvironmentChanged?: () => void | Promise<void>;
}

function failureMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Could not complete the operation.";
}

const jobMessages = {
  running: "Installing…",
  complete: "Complete.",
  error: "Installation failed.",
  cancelled: "Cancelled.",
};

export default function PackageManager({
  client,
  disabled = false,
  readOnly = false,
  onRunCommand,
  onEnvironmentChanged,
}: PackageManagerProps) {
  const [open, setOpen] = useState(false);
  const [environment, setEnvironment] = useState<ProjectPackages | null>(null);
  const [shared, setShared] = useState<SharedPackages | null>(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [command, setCommand] = useState("");
  const [imported, setImported] = useState<{
    document: PortableEnvironment;
    preview: EnvironmentPreview;
  } | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const mounted = useRef(false);
  const currentClient = useRef(client);
  const currentOpen = useRef(open);
  const pendingOperation = useRef(false);
  const pendingCancel = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const terminalJob = useRef<string | null>(null);
  const changedCallback = useRef(onEnvironmentChanged);
  const titleId = useId();
  const commandId = useId();
  currentClient.current = client;
  currentOpen.current = open;
  changedCallback.current = onEnvironmentChanged;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
    };
  }, []);

  const readEnvironment = useCallback(
    async (signal: AbortSignal) => {
      const snapshot = await client.request<ProjectPackages>("/environment", {
        signal,
      });
      if (
        !signal.aborted &&
        mounted.current &&
        currentClient.current === client
      ) {
        setEnvironment(snapshot);
      }
      return snapshot;
    },
    [client],
  );

  const readShared = useCallback(
    async (signal: AbortSignal) => {
      if (!client.teams) return;
      const snapshot = await client.request<SharedPackages>(
        "/environment/shared",
        { signal },
      );
      if (
        !signal.aborted &&
        mounted.current &&
        currentClient.current === client
      ) {
        setShared(snapshot);
      }
    },
    [client],
  );

  const notifyEnvironmentChanged = useCallback(() => {
    try {
      void Promise.resolve(changedCallback.current?.()).catch(
        (failure: unknown) => {
          if (mounted.current && currentOpen.current)
            setError(failureMessage(failure));
        },
      );
    } catch (failure) {
      if (mounted.current && currentOpen.current)
        setError(failureMessage(failure));
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const abort = new AbortController();
    controller.current = abort;
    const element = dialog.current;
    element?.showModal();
    setLoading(true);
    setEnvironment(null);
    setShared(null);
    setError("");
    setNotice("");
    setImported(null);
    void readEnvironment(abort.signal)
      .catch((failure: unknown) => {
        if (!abort.signal.aborted) setError(failureMessage(failure));
      })
      .finally(() => {
        if (!abort.signal.aborted) setLoading(false);
      });
    void readShared(abort.signal).catch((failure: unknown) => {
      if (!abort.signal.aborted) setError(failureMessage(failure));
    });
    return () => {
      abort.abort();
      element?.close();
      if (controller.current === abort) controller.current = null;
    };
  }, [open, readEnvironment, readShared]);

  const running = environment?.job?.state === "running";
  useEffect(() => {
    if (!open || !running) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await readEnvironment(abort.signal);
        if (!abort.signal.aborted && next.job?.state === "running") {
          timer = setTimeout(() => void poll(), 1000);
        }
      } catch (failure) {
        if (!abort.signal.aborted) {
          setError(failureMessage(failure));
          timer = setTimeout(() => void poll(), 2000);
        }
      }
    }
    timer = setTimeout(() => void poll(), 1000);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [open, running, readEnvironment]);

  useEffect(() => {
    const job = environment?.job;
    if (
      !open ||
      !job ||
      job.state === "running" ||
      terminalJob.current === job.id
    )
      return;
    terminalJob.current = job.id;
    notifyEnvironmentChanged();
    const signal = controller.current?.signal;
    if (signal) {
      void readShared(signal).catch((failure: unknown) => {
        if (!signal.aborted) setError(failureMessage(failure));
      });
    }
  }, [
    open,
    environment?.job?.id,
    environment?.job?.state,
    readShared,
    notifyEnvironmentChanged,
  ]);

  const installed = useMemo(
    () =>
      [...(environment?.installed ?? [])].sort((a, b) =>
        a.name.localeCompare(b.name),
      ),
    [environment],
  );
  const explicit = useMemo(
    () =>
      new Set(
        (environment?.requirements ?? []).map((item) =>
          normalizePackageName(item.name),
        ),
      ),
    [environment],
  );
  const locked =
    loading ||
    pending ||
    running ||
    disabled ||
    readOnly ||
    !environment?.available;

  function close() {
    currentOpen.current = false;
    controller.current?.abort();
    setOpen(false);
    setImported(null);
  }

  async function exportEnvironment() {
    const signal = controller.current?.signal;
    if (locked || !signal || pendingOperation.current) return;
    pendingOperation.current = true;
    setPending(true);
    setError("");
    try {
      const document = await client.request<PortableEnvironment>(
        "/environment/export",
        { signal },
      );
      if (
        !signal.aborted &&
        currentClient.current === client &&
        currentOpen.current
      ) {
        download(
          JSON.stringify(document),
          "openeconometrics-environment.json",
        );
        setNotice("Environment exported.");
      }
    } catch (failure) {
      if (!signal.aborted && currentClient.current === client)
        setError(failureMessage(failure));
    } finally {
      pendingOperation.current = false;
      if (mounted.current) setPending(false);
    }
  }

  async function previewEnvironment(file?: File) {
    const signal = controller.current?.signal;
    if (!file || locked || !signal || pendingOperation.current) return;
    pendingOperation.current = true;
    setPending(true);
    setImported(null);
    setError("");
    setNotice("");
    try {
      if (file.size > 64 * 1024)
        throw new Error("The environment document exceeds 64 KiB.");
      let document: PortableEnvironment;
      try {
        document = JSON.parse(await file.text());
      } catch {
        throw new Error("Choose a valid JSON environment document.");
      }
      if (signal.aborted || currentClient.current !== client) return;
      const preview = await client.request<EnvironmentPreview>(
        "/environment/import/preview",
        {
          method: "POST",
          body: JSON.stringify({ document }),
          signal,
        },
      );
      if (
        !signal.aborted &&
        currentClient.current === client &&
        currentOpen.current
      ) {
        setImported({ document, preview });
        setNotice("Environment checked. No packages have been installed.");
      }
    } catch (failure) {
      if (!signal.aborted && currentClient.current === client)
        setError(failureMessage(failure));
    } finally {
      pendingOperation.current = false;
      if (mounted.current) setPending(false);
    }
  }

  async function perform(path: string, body?: object) {
    if (locked || pendingOperation.current) return;
    pendingOperation.current = true;
    setPending(true);
    setError("");
    setNotice("");
    try {
      await client.request(path, {
        method: "POST",
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      if (
        path !== "/environment/share" &&
        mounted.current &&
        currentClient.current === client
      ) {
        notifyEnvironmentChanged();
      }
      const signal = controller.current?.signal;
      if (
        currentClient.current === client &&
        currentOpen.current &&
        signal &&
        !signal.aborted
      ) {
        await readEnvironment(signal);
        if (path === "/environment/share") {
          await readShared(signal);
          setNotice("Environment shared.");
        }
      }
    } catch (failure) {
      if (
        mounted.current &&
        currentOpen.current &&
        currentClient.current === client
      ) {
        setError(failureMessage(failure));
      }
    } finally {
      pendingOperation.current = false;
      if (mounted.current) setPending(false);
    }
  }

  function install(event: FormEvent) {
    event.preventDefault();
    if (locked) return;
    try {
      const source = normalizePackageCommand(command);
      close();
      onRunCommand(source);
    } catch (failure) {
      setError(failureMessage(failure));
    }
  }

  async function cancel() {
    if (!running || pendingCancel.current) return;
    pendingCancel.current = true;
    setCancelling(true);
    setError("");
    try {
      await client.request("/environment/cancel", { method: "POST" });
      const signal = controller.current?.signal;
      if (
        currentOpen.current &&
        currentClient.current === client &&
        signal &&
        !signal.aborted
      ) {
        await readEnvironment(signal);
      }
    } catch (failure) {
      if (mounted.current && currentOpen.current)
        setError(failureMessage(failure));
    } finally {
      pendingCancel.current = false;
      if (mounted.current) setCancelling(false);
    }
  }

  if (!client.desktop && !client.localDesktop) return null;
  const job = environment?.job;
  const log = job ? boundedPackageLog(job.log) : "";
  return (
    <>
      <button type="button" disabled={disabled} onClick={() => setOpen(true)}>
        Packages
      </button>
      {open && (
        <dialog
          ref={dialog}
          className="package-dialog"
          aria-labelledby={titleId}
          onClose={close}
          onCancel={(event) => {
            event.preventDefault();
            close();
          }}
        >
          <header>
            <h2 id={titleId}>Packages</h2>
            <button
              type="button"
              className="package-close"
              aria-label="Close"
              onClick={close}
            >
              ×
            </button>
          </header>
          <div
            className="package-body"
            aria-busy={loading || pending || running}
          >
            {loading && <p role="status">Loading…</p>}
            {error && (
              <p className="package-message" role="alert">
                {error}
              </p>
            )}
            {notice && (
              <p className="package-message" role="status">
                {notice}
              </p>
            )}
            {environment && !environment.available && (
              <p role="status">Package installation is unavailable.</p>
            )}
            {environment?.available && (
              <>
                <form onSubmit={install} className="package-install">
                  <label htmlFor={commandId}>
                    Command
                    <input
                      id={commandId}
                      value={command}
                      onChange={(event) => setCommand(event.target.value)}
                      placeholder="pip install scikit-learn"
                      autoComplete="off"
                      autoCapitalize="none"
                      spellCheck={false}
                      maxLength={64000}
                      required
                      disabled={locked}
                    />
                  </label>
                  <button
                    type="submit"
                    className="package-primary"
                    disabled={locked || !command.trim()}
                  >
                    Run
                  </button>
                </form>
                {job && (
                  <section
                    className="package-job"
                    aria-label="Installation status"
                  >
                    <div className="package-job-status">
                      <p role={job.state === "error" ? "alert" : "status"}>
                        {job.state === "error"
                          ? job.message || jobMessages.error
                          : jobMessages[job.state]}
                      </p>
                      {running && (
                        <button
                          type="button"
                          disabled={cancelling}
                          onClick={() => void cancel()}
                        >
                          {cancelling ? "Cancelling…" : "Cancel"}
                        </button>
                      )}
                    </div>
                    {log && (
                      <details className="package-log">
                        <summary>Log</summary>
                        <pre>{log}</pre>
                      </details>
                    )}
                  </section>
                )}
                {installed.length ? (
                  <ul className="package-list" aria-label="Installed packages">
                    {installed.map((item) => (
                      <li key={normalizePackageName(item.name)}>
                        <span>
                          <strong>{item.name}</strong>
                          <code>{item.version}</code>
                        </span>
                        {explicit.has(normalizePackageName(item.name)) && (
                          <button
                            type="button"
                            disabled={locked}
                            aria-label={`Remove ${item.name}`}
                            onClick={() =>
                              void perform("/environment/remove", {
                                name: item.name,
                              })
                            }
                          >
                            Remove
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="package-empty">No packages added yet.</p>
                )}
                {!!environment.base.length && (
                  <details className="package-core">
                    <summary>Core</summary>
                    <ul className="package-list" aria-label="Core packages">
                      {environment.base.map((item) => (
                        <li key={normalizePackageName(item.name)}>
                          <span>
                            <strong>{item.name}</strong>
                            <code>{item.version}</code>
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
                {client.teams && shared && !shared.offline && (
                  <div className="package-shared">
                    <button
                      type="button"
                      disabled={locked || shared.localMatches}
                      onClick={() => void perform("/environment/share")}
                    >
                      Share with team
                    </button>
                    {shared.manifest && !shared.localMatches && (
                      <button
                        type="button"
                        disabled={locked}
                        onClick={() =>
                          void perform("/environment/restore-shared")
                        }
                      >
                        Sync environment
                      </button>
                    )}
                  </div>
                )}
                <section
                  className="package-portable"
                  aria-label="Portable environment"
                >
                  <button
                    type="button"
                    disabled={locked}
                    onClick={() => void exportEnvironment()}
                  >
                    Export environment
                  </button>
                  <label>
                    Import environment
                    <input
                      type="file"
                      accept=".json,application/json"
                      disabled={locked}
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        event.target.value = "";
                        void previewEnvironment(file);
                      }}
                    />
                  </label>
                  {imported && (
                    <div>
                      <p>
                        {imported.preview.manifest.locked.length} packages ·
                        Python {imported.preview.manifest.python}
                      </p>
                      <ul aria-label="Environment preview">
                        {imported.preview.manifest.locked.map((item) => (
                          <li key={item.name}>
                            {item.name} {item.version}
                          </li>
                        ))}
                      </ul>
                      <p>
                        Restore installs these versions and replaces this
                        project's additional packages. Core packages stay fixed.
                      </p>
                      <button
                        type="button"
                        disabled={locked || imported.preview.matches}
                        onClick={() => {
                          void perform("/environment/import/restore", {
                            document: imported.document,
                          });
                          setImported(null);
                        }}
                      >
                        {imported.preview.matches
                          ? "Already matches"
                          : "Restore environment"}
                      </button>
                      <button
                        type="button"
                        disabled={pending}
                        onClick={() => setImported(null)}
                      >
                        Discard import
                      </button>
                    </div>
                  )}
                </section>
              </>
            )}
          </div>
        </dialog>
      )}
    </>
  );
}
