import { useEffect, useRef, useState } from "react";
import { ApiError, type WorkspaceClient } from "./api";
import type { ExecutionRecord } from "./types";
import { sharingLabel } from "./ResultSharingPanel";
import type { ResultSharingRecord } from "./result-sharing";

export interface HistoryEntry {
  id: string;
  created_at: string;
  state: string;
  status: ExecutionRecord["status"] | null;
  code_preview: string;
  code_preview_truncated: boolean;
  actor_email: string;
}
export interface HistoryPage {
  runs: HistoryEntry[];
  next_cursor: string | null;
  page_cursor: string;
  snapshot_at: string;
  scanned: number;
}
type Filters = { query: string; since: string; until: string };
const emptyFilters = { query: "", since: "", until: "" };

export default function ExecutionHistory({
  client,
  history,
  localLoading = false,
  localError = "",
  sharingRecords,
  onSelect,
  onOpen,
}: {
  client: WorkspaceClient;
  history: ExecutionRecord[];
  localLoading?: boolean;
  localError?: string;
  sharingRecords?: ReadonlyMap<string, ResultSharingRecord>;
  onSelect: (id: string) => void;
  onOpen: (record: ExecutionRecord) => void;
}) {
  const [shared, setShared] = useState(client.teams && !client.desktop);
  const [draft, setDraft] = useState<Filters>(emptyFilters);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [trail, setTrail] = useState<(string | null)[]>([null]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);

  async function load(nextFilters: Filters, nextTrail: (string | null)[]) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    const params = new URLSearchParams({ ...nextFilters, limit: "20" });
    const cursor = nextTrail.at(-1);
    if (cursor) params.set("cursor", cursor);
    try {
      const value = await client.request<HistoryPage>(
        `/console/history?${params}`,
        { signal: controller.signal },
      );
      if (controller.signal.aborted || request.current !== controller) return;
      if (
        !Array.isArray(value.runs) ||
        value.runs.length > 50 ||
        new Set(value.runs.map((run) => run.id)).size !== value.runs.length ||
        (value.next_cursor !== null && typeof value.next_cursor !== "string") ||
        typeof value.page_cursor !== "string"
      )
        throw Error("Could not verify the history page.");
      setPage(value);
      setFilters(nextFilters);
      setTrail([...nextTrail.slice(0, -1), value.page_cursor]);
    } catch (failure) {
      if (controller.signal.aborted || request.current !== controller) return;
      if (
        failure instanceof ApiError &&
        [401, 403, 404].includes(failure.status)
      )
        setPage(null);
      setError(
        failure instanceof Error ? failure.message : "Could not load history.",
      );
    } finally {
      if (!controller.signal.aborted && request.current === controller)
        setBusy(false);
    }
  }
  useEffect(() => {
    if (shared) void load(emptyFilters, [null]);
    return () => {
      request.current?.abort();
    };
  }, [client, shared]);

  async function open(id: string) {
    if (busy) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try {
      const record = await client.request<ExecutionRecord>(
        `/runs/${encodeURIComponent(id)}/record`,
        { signal: controller.signal },
      );
      if (controller.signal.aborted || request.current !== controller) return;
      if (
        record.id !== id ||
        !Array.isArray(record.outputs) ||
        typeof record.code !== "string"
      )
        throw Error("Could not verify the saved result.");
      onOpen(record);
    } catch (failure) {
      if (!controller.signal.aborted && request.current === controller)
        setError(
          failure instanceof Error
            ? failure.message
            : "Could not open the result.",
        );
    } finally {
      if (!controller.signal.aborted && request.current === controller)
        setBusy(false);
    }
  }

  return (
    <>
      {client.teams && client.desktop && (
        <div className="history-sources" aria-label="History source">
          <button
            className="subtle-button"
            aria-pressed={!shared}
            onClick={() => setShared(false)}
          >
            This device
          </button>
          <button
            className="subtle-button"
            aria-pressed={shared}
            onClick={() => setShared(true)}
          >
            Shared history
          </button>
        </div>
      )}
      {shared ? (
        <>
          <form
            className="history-search"
            onSubmit={(event) => {
              event.preventDefault();
              if (Array.from(draft.query).length > 256) {
                setError("Search with up to 256 characters.");
                return;
              }
              void load(draft, [null]);
            }}
          >
            <label>
              Run ID or code
              <input
                value={draft.query}
                onChange={(e) => setDraft({ ...draft, query: e.target.value })}
              />
            </label>
            <div className="history-dates">
              <label>
                From (UTC)
                <input
                  type="date"
                  value={draft.since}
                  onChange={(e) =>
                    setDraft({ ...draft, since: e.target.value })
                  }
                />
              </label>
              <label>
                Through (UTC)
                <input
                  type="date"
                  value={draft.until}
                  onChange={(e) =>
                    setDraft({ ...draft, until: e.target.value })
                  }
                />
              </label>
            </div>
            <button className="subtle-button" type="submit">
              Search / refresh
            </button>
          </form>
          {error && (
            <p className="error-banner" role="alert">
              {error}
            </p>
          )}
          {busy && <p role="status">Loading history…</p>}
          <div className="history-list" aria-busy={busy}>
            {page?.runs.map((run) => (
              <button
                key={run.id}
                disabled={
                  busy ||
                  !["finished", "failed", "cancelled"].includes(run.state)
                }
                onClick={() => void open(run.id)}
              >
                <span className={`history-dot ${run.status ?? "pending"}`} />
                <span>
                  <strong>
                    {run.code_preview || "Python"}
                    {run.code_preview_truncated ? "…" : ""}
                  </strong>
                  <small>
                    {run.created_at} · {run.status ?? run.state}
                  </small>
                  <small>
                    {run.id} · {run.actor_email}
                  </small>
                </span>
              </button>
            ))}
          </div>
          {page && !page.runs.length && (
            <p>
              {page.next_cursor
                ? "No matching runs in this page. Continue to older runs."
                : "No matching runs."}
            </p>
          )}
          <div className="dialog-actions">
            <button
              className="subtle-button"
              disabled={busy || trail.length < 2}
              onClick={() => void load(filters, trail.slice(0, -1))}
            >
              Previous page
            </button>
            <button
              className="subtle-button"
              disabled={busy || !page?.next_cursor}
              onClick={() => void load(filters, [...trail, page!.next_cursor])}
            >
              Older runs
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="history-list">
            {[...history].reverse().map((run, index) => (
              <button key={run.id} onClick={() => onSelect(run.id)}>
                <span className={`history-dot ${run.status}`} />
                <span>
                  <strong>
                    {run.code
                      .split("\n")
                      .find((line) => line.trim() && !line.startsWith("#")) ||
                      "Python"}
                  </strong>
                  <small>
                    #{history.length - index} ·{" "}
                    {new Intl.NumberFormat("en-US", {
                      maximumFractionDigits: 2,
                    }).format(run.duration_ms / 1000)}{" "}
                    s · {run.status}
                    {sharingRecords?.has(run.id) && ` · ${sharingLabel(sharingRecords.get(run.id)!.state,
                      run.outputs.some((output) => output.type === "plot" && !!output.data &&
                        typeof output.data === "object" && "artifact" in output.data))}`}
                  </small>
                </span>
              </button>
            ))}
          </div>
          {!history.length && (
            <p>
              {localLoading
                ? "Loading history…"
                : localError
                  ? "Could not load history."
                  : "No runs yet."}
            </p>
          )}
        </>
      )}
    </>
  );
}
