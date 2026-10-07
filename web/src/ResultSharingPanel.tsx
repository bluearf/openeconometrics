import { useEffect, useRef, useState } from "react";
import { ApiError, type WorkspaceClient } from "./api";
import type { ResultSharingRecord, ResultSharingSnapshot } from "./result-sharing";

export function sharingLabel(state: ResultSharingRecord["state"], partial = false): string {
  if (state === "shared" && partial) return "Summary shared · graph local";
  return {
    pending: "Waiting to share",
    shared: "Shared to team",
    failed: "Sharing failed",
    local_only: "Local only",
  }[state];
}

interface Props {
  client: WorkspaceClient;
  resultId?: string;
  readOnly: boolean;
  ready: boolean;
  partial?: boolean;
  onOpenHistory: () => void;
  onSnapshot: (value: ResultSharingSnapshot | null) => void;
  onAccessDenied: () => void;
}

/** Delivery is independent of the Python run and its selected result. */
export default function ResultSharingPanel(props: Props) {
  const { client, resultId, readOnly, onOpenHistory } = props;
  const callbacks = useRef(props);
  callbacks.current = props;
  const context = useRef<{ active: boolean; client: WorkspaceClient; revision: number } | null>(null);
  const [snapshot, setSnapshot] = useState<ResultSharingSnapshot | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const retrying = useRef(false);
  const enabled = Boolean(props.ready && client.desktop && client.teams && client.subscribeResultSharing);

  useEffect(() => {
    const current = { active: true, client, revision: 0 };
    context.current = current;
    retrying.current = false;
    setSnapshot(null);
    setError("");
    setBusy(false);
    callbacks.current.onSnapshot(null);
    if (!enabled) return () => { current.active = false; };
    let received = false;
    const accept = (value: ResultSharingSnapshot) => {
      if (!current.active || context.current !== current) return;
      received = true;
      current.revision++;
      setSnapshot(value);
      setError("");
      callbacks.current.onSnapshot(value);
      if ([401, 403].includes(value.error?.status ?? 0))
        callbacks.current.onAccessDenied();
    };
    const unsubscribe = client.subscribeResultSharing?.(accept);
    void client.request<ResultSharingSnapshot>("/result-sharing").then((value) => {
      // A newer subscription notification wins over the bootstrap response.
      if (!received) accept(value);
    }).catch((failure) => {
      if (!current.active || context.current !== current || received) return;
      setError(failure instanceof Error ? failure.message : "Could not check sharing.");
      if (failure instanceof ApiError && [401, 403].includes(failure.status))
        callbacks.current.onAccessDenied();
    });
    return () => { current.active = false; unsubscribe?.(); };
  }, [client, enabled]);

  async function retry(id?: string) {
    const current = context.current;
    if (!current?.active || current.client !== client || (id && readOnly) || retrying.current) return;
    retrying.current = true;
    setBusy(true);
    setError("");
    let readRevision: number | null = null;
    try {
      if (id) await client.request(`/result-sharing/${encodeURIComponent(id)}/retry`, { method: "POST" });
      readRevision = current.revision;
      const value = await client.request<ResultSharingSnapshot>("/result-sharing");
      if (!current.active || context.current !== current || current.revision !== readRevision) return;
      current.revision++;
      setSnapshot(value);
      callbacks.current.onSnapshot(value);
      if ([401, 403].includes(value.error?.status ?? 0))
        callbacks.current.onAccessDenied();
    } catch (failure) {
      if (!current.active || context.current !== current) return;
      if (readRevision !== null && current.revision !== readRevision
          && !(failure instanceof ApiError && [401, 403].includes(failure.status))) return;
      setError(failure instanceof Error ? failure.message : "Could not share this result.");
      if (failure instanceof ApiError && [401, 403].includes(failure.status))
        callbacks.current.onAccessDenied();
    } finally {
      if (current.active && context.current === current) {
        retrying.current = false;
        setBusy(false);
      }
    }
  }

  if (!enabled) return null;
  const record = snapshot?.records.find((item) => item.id === resultId);
  const pending = snapshot?.pending_count ?? 0;
  const failed = snapshot?.failed_count ?? 0;
  const message = snapshot?.error?.message || error || record?.error?.message;
  if (!record && !pending && !failed && !message) return null;
  return (
    <div className="result-sharing" aria-label="Result sharing" role="status">
      <div className="result-sharing-line">
        {record && <span className="result-sharing-state">{sharingLabel(record.state, props.partial)}</span>}
        {(pending > 0 || failed > 0) && (
          <button className="subtle-button" onClick={onOpenHistory} aria-label="Open sharing history">
            {[pending ? `${pending} waiting` : "", failed ? `${failed} failed` : ""].filter(Boolean).join(" · ")}
          </button>
        )}
        {record?.retryable && (
          <button className="subtle-button" disabled={readOnly || busy} onClick={() => void retry(record.id)}>
            {busy ? "Sending…" : "Retry sharing"}
          </button>
        )}
        {message && !record && (
          <button className="subtle-button" disabled={busy} onClick={() => void retry()}>
            {busy ? "Checking…" : "Check sharing"}
          </button>
        )}
      </div>
      {message && <p className="result-sharing-error">{message}</p>}
    </div>
  );
}
