import { useCallback, useEffect, useRef, useState } from "react";
import { isDesktop, nativeInvoke } from "./desktop";
import "./inline-suggestions.css";

export interface LocalSuggestionsStatus {
  enabled: boolean;
  installed: boolean;
  state:
    "off" | "not_installed" | "installing" | "starting" | "ready" | "error";
  model: "qwen2.5-coder-0.5b-q8_0";
  downloaded_bytes: number;
  total_bytes: number;
  error_code?: string;
}

interface Props {
  onStatusChange?: (status: LocalSuggestionsStatus | null) => void;
}

type Operation = "install" | "start" | "stop";

const MODEL_BYTES = 531_068_128;
const STATES = new Set([
  "off",
  "not_installed",
  "installing",
  "starting",
  "ready",
  "error",
]);

function statusFrom(value: unknown): LocalSuggestionsStatus {
  if (!value || typeof value !== "object") throw new Error("AI_INVALID_STATUS");
  const status = value as LocalSuggestionsStatus;
  if (
    typeof status.enabled !== "boolean" ||
    typeof status.installed !== "boolean" ||
    !STATES.has(status.state) ||
    status.model !== "qwen2.5-coder-0.5b-q8_0" ||
    status.total_bytes !== MODEL_BYTES ||
    !Number.isSafeInteger(status.downloaded_bytes) ||
    status.downloaded_bytes < 0 ||
    status.downloaded_bytes > status.total_bytes ||
    (status.error_code !== undefined &&
      typeof status.error_code !== "string") ||
    (status.state === "ready" && (!status.enabled || !status.installed))
  )
    throw new Error("AI_INVALID_STATUS");
  return status;
}

function errorText(failure: unknown): string {
  const code =
    typeof failure === "string"
      ? failure
      : failure instanceof Error
        ? failure.message
        : "";
  const messages: Record<string, string> = {
    AI_NETWORK:
      "Could not download the model. Check your connection and try again.",
    AI_MODEL_INVALID: "Could not verify the model. Download it again.",
    AI_ENGINE_UNAVAILABLE: "The suggestion engine could not start in this installation.",
    AI_TIMEOUT: "Could not start the model. Try again.",
    AI_NOT_READY: "The model is not ready yet.",
  };
  return messages[code] ?? "Could not connect to suggestions. Try again.";
}

const stateLabels: Record<LocalSuggestionsStatus["state"], string> = {
  off: "Off",
  not_installed: "Model not downloaded",
  installing: "Downloading model…",
  starting: "Starting model…",
  ready: "Ready",
  error: "Could not start model",
};

export default function InlineSuggestionsSettings({ onStatusChange }: Props) {
  const desktop = isDesktop();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<LocalSuggestionsStatus | null>(null);
  const [failure, setFailure] = useState("");
  const [operation, setOperation] = useState<Operation | null>(null);
  const mounted = useRef(false);
  const statusGeneration = useRef(0);
  const statusPending = useRef<number | null>(null);
  const operationGeneration = useRef(0);
  const operationRef = useRef<Operation | null>(null);
  const callback = useRef(onStatusChange);
  callback.current = onStatusChange;
  const dialog = useRef<HTMLDialogElement>(null);

  const publish = useCallback((value: unknown) => {
    const next = statusFrom(value);
    if (!mounted.current) return;
    setStatus(next);
    setFailure(next.error_code ? errorText(next.error_code) : "");
    callback.current?.(next);
  }, []);

  const refresh = useCallback(async () => {
    if (
      !desktop ||
      statusPending.current !== null ||
      !mounted.current ||
      operationRef.current === "stop"
    )
      return;
    const ticket = ++statusGeneration.current;
    statusPending.current = ticket;
    try {
      const next = await nativeInvoke<unknown>("suggestions_status");
      if (mounted.current && ticket === statusGeneration.current) publish(next);
    } catch (error) {
      if (mounted.current && ticket === statusGeneration.current) {
        setStatus(null);
        setFailure(errorText(error));
        callback.current?.(null);
      }
    } finally {
      if (statusPending.current === ticket) statusPending.current = null;
    }
  }, [desktop, publish]);

  useEffect(() => {
    mounted.current = true;
    if (desktop) void refresh();
    return () => {
      mounted.current = false;
      ++statusGeneration.current;
      statusPending.current = null;
      ++operationGeneration.current;
      operationRef.current = null;
    };
  }, [desktop, refresh]);

  useEffect(() => {
    if (!open) return;
    const current = dialog.current;
    current?.showModal();
    void refresh();
    return () => current?.close();
  }, [open, refresh]);

  useEffect(() => {
    const busy =
      Boolean(operation) ||
      status?.state === "installing" ||
      status?.state === "starting";
    // An active worker can exit after an allocation failure. Refresh its real
    // readiness, while an idle, disabled installation needs no polling.
    if (!desktop || (!busy && !(status?.enabled && status.state === "ready")))
      return;
    const timer = window.setTimeout(() => void refresh(), busy ? 750 : 5_000);
    return () => window.clearTimeout(timer);
  }, [desktop, status, operation, refresh]);

  async function change(
    command: "suggestions_install" | "suggestions_configure",
    enabled?: boolean,
  ) {
    const stopping = command === "suggestions_configure" && enabled === false;
    if (operationRef.current === "stop" || (operationRef.current && !stopping))
      return;
    const ticket = ++operationGeneration.current;
    ++statusGeneration.current;
    const nextOperation =
      command === "suggestions_install"
        ? "install"
        : stopping
          ? "stop"
          : "start";
    operationRef.current = nextOperation;
    setOperation(nextOperation);
    setFailure("");
    // Stop showing previously received text as soon as the user turns it off.
    if (enabled === false) callback.current?.(null);
    try {
      const next = await nativeInvoke<unknown>(
        command,
        command === "suggestions_configure" ? { enabled } : undefined,
      );
      if (mounted.current && ticket === operationGeneration.current) {
        ++statusGeneration.current;
        publish(next);
      }
    } catch (error) {
      if (mounted.current && ticket === operationGeneration.current) {
        ++statusGeneration.current;
        setFailure(errorText(error));
        callback.current?.(null);
        operationRef.current = null;
        await refresh();
        if (mounted.current && ticket === operationGeneration.current)
          setFailure(errorText(error));
      }
    } finally {
      if (mounted.current && ticket === operationGeneration.current) {
        operationRef.current = null;
        setOperation(null);
      }
    }
  }

  if (!desktop) return null;
  const busy =
    Boolean(operation) ||
    status?.state === "installing" ||
    status?.state === "starting";
  const caption =
    operation === "stop"
      ? "Stopping…"
      : operation === "install" && status?.state !== "installing"
        ? "Preparing download…"
        : operation === "start" && status?.state !== "starting"
          ? "Updating…"
          : status
            ? stateLabels[status.state]
            : failure
              ? "Unavailable"
              : "Connecting…";

  return (
    <>
      <button
        type="button"
        title="Local code suggestions as you type"
        onClick={() => setOpen(true)}
      >
        Suggestions
      </button>
      {open && (
        <dialog
          ref={dialog}
          className="inline-suggestions-dialog"
          aria-label="Inline suggestions"
          onCancel={() => setOpen(false)}
          onClick={(event) => {
            if (event.target === event.currentTarget) setOpen(false);
          }}
        >
          <div className="dialog-head">
            <h2>Inline suggestions</h2>
            <button
              type="button"
              className="inline-suggestions-close"
              aria-label="Close dialog"
              onClick={() => setOpen(false)}
            >
              ×
            </button>
          </div>
          <div className="inline-suggestions-body">
            <div className="inline-suggestions-model">
              <strong>Qwen2.5-Coder 0.5B</strong>
              <span>Runs on your computer · 531 MB</span>
            </div>
            <label className="inline-suggestions-toggle">
              <span>Suggest as you type</span>
              <input
                type="checkbox"
                checked={Boolean(status?.enabled)}
                disabled={!status?.installed || busy}
                onChange={(event) =>
                  void change("suggestions_configure", event.target.checked)
                }
              />
            </label>
            <p className="inline-suggestions-state" role="status">
              {caption}
            </p>
            {status?.state === "installing" && (
              <div className="inline-suggestions-progress">
                <progress
                  value={status.downloaded_bytes}
                  max={status.total_bytes}
                  aria-label="Model download progress"
                />
                <span>
                  {Math.floor(
                    (100 * status.downloaded_bytes) / status.total_bytes,
                  )}
                  %
                </span>
              </div>
            )}
            {failure && (
              <p className="inline-suggestions-error" role="alert">
                {failure}
              </p>
            )}
            {(operation === "install" ||
              operation === "start" ||
              status?.state === "installing" ||
              status?.state === "starting") && (
              <button
                type="button"
                disabled={operation === "stop"}
                onClick={() => void change("suggestions_configure", false)}
              >
                Stop
              </button>
            )}
            {status && !status.installed && (
              <button
                type="button"
                className="inline-suggestions-install"
                disabled={busy}
                onClick={() => void change("suggestions_install")}
              >
                Download model
              </button>
            )}
            {!status && failure && (
              <button type="button" onClick={() => void refresh()}>
                Try again
              </button>
            )}
            <p className="inline-suggestions-shortcuts">
              Tab: accept · Escape: dismiss
            </p>
          </div>
        </dialog>
      )}
    </>
  );
}
