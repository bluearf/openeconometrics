import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import {
  ApiError,
  download,
  loadWorkspaceBootstrap,
  isExecutionAccepted,
  submitWorkspaceExecution,
  localClient,
  workspaceDownloadPath,
  VersionedScriptWriter,
  loadScriptFiles,
  loadScriptFile,
  validateScriptFile,
  type ScriptDraft,
  type ScriptFile,
  type ScriptSummary,
  type ScriptSyncNotification,
  type WorkspaceClient,
  type ScriptConflictSnapshot,
} from "./api";
import CodeEditor, { type EditorHandle } from "./CodeEditor";
import { qaStart, qaPaint } from "./qa-performance";
import ScriptConflict from "./ScriptConflict";
import PlotView, { type PlotSpec } from "./PlotView";
import LatexView from "./LatexView";
import PackageManager from "./PackageManager";
import ResultSharingPanel from "./ResultSharingPanel";
import type { ResultSharingSnapshot } from "./result-sharing";
import InlineSuggestionsSettings, {
  type LocalSuggestionsStatus,
} from "./InlineSuggestionsSettings";
import { createLocalSuggestionProvider } from "./local-suggestions";
import FilesSidebar from "./FilesSidebar";
import ProjectTransfers from "./ProjectTransfers";
import ProjectFileDownload from "./ProjectFileDownload";
import ExecutionHistory from "./ExecutionHistory";
import {
  mergeFileLayout,
  moveFileEntry,
  renameFileEntry,
  validateFileLayout,
  validFileName,
  normalizeDocumentName,
  fileNameFold,
  documentExtension,
  documentMimeType,
  localCopyName,
  type FileLayout,
  type FileLayoutEntry,
} from "./file-layout";
import { WorkspaceLayout, WorkbenchPanels } from "./WorkspacePanels";
import WorkspaceTerminal, {
  type WorkspaceTerminalHandle,
} from "./WorkspaceTerminal";
import { normalizePackageCommand } from "./package-command";
import {
  latexDocument,
  outputLabel,
  modelTargetLabel,
  publicationNumber,
  significanceStars,
  uninferredModelRows,
} from "./latex-export";
import { LatestValueAutosaver, ResponseGate } from "./session-state";
import { incomingAgentSelection, watchResultRevisions } from "./result-revision";
import {
  orderedExecutionEvents,
  selectExecutionEvents,
  timelineLatexOutputs,
} from "./output-order";
import type {
  AppConfig,
  ConsoleOutput,
  ConsoleState,
  ConsoleVariable,
  DatasetProfile,
  ExecutionRecord,
  ResultBundle,
} from "./types";

const STARTER = `import openecon as oe

# 01 · Load the data — 480 synthetic observations
df = oe.example()
display(df.head())

# 02 · Wages, education, and experience
model = oe.ols(
    data=df,
    y="wage",
    x=["education", "experience"],
    covariance="HC3",
)
display(model)

# 03 · Create a chart from the same code
oe.plot.scatter(
    data=df, x="education", y="wage",
    title="Education and wages",
)
`;
const fmt = (v: number | null | undefined, digits = 4) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(
        v,
      );
const errorMessage = (e: unknown) =>
  e instanceof Error ? e.message : "The action could not be completed.";
const EMPTY_OUTPUTS: ConsoleOutput[] = [];
function Symbol({
  name,
  size = 17,
}: {
  name:
    | "play"
    | "stop"
    | "file"
    | "data"
    | "download"
    | "plus"
    | "code"
    | "chart"
    | "history"
    | "agent"
    | "close"
    | "reset";
  size?: number;
}) {
  const paths: Record<string, ReactNode> = {
    play: <path d="m8 4 12 8-12 8Z" />,
    stop: <rect x="5" y="5" width="14" height="14" rx="2" />,
    file: (
      <>
        <path d="M6 3h9l4 4v14H6Z" />
        <path d="M14 3v5h5M9 12h7m-7 4h7" />
      </>
    ),
    data: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M3 10h18M9 4v16" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-4-4 4 4 4-4M5 17v4h14v-4" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    code: <path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18" />,
    chart: (
      <>
        <path d="M4 4v16h17M8 14l4-6 4 4 5-8" />
      </>
    ),
    history: (
      <>
        <path d="M3 11a9 9 0 1 1 2 7M3 4v7h7M12 7v5l4 2" />
      </>
    ),
    agent: (
      <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z" />
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    reset: <path d="M4 8a9 9 0 1 1-1 8M4 3v6h6" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
function Dialog({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    d?.showModal();
    return () => d?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      aria-label={title}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog-head">
        <h2>{title}</h2>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <Symbol name="close" />
        </button>
      </div>
      {children}
    </dialog>
  );
}
function ModelTable({
  data,
  publication = false,
}: {
  data: ResultBundle;
  publication?: boolean;
}) {
  if (data.inference?.available === false) {
    return (
      <div className="table-wrap">
        <table className={publication ? "publication-grid" : undefined}>
          <thead>
            <tr>
              <th>Quantity</th>
              <th>Estimate</th>
            </tr>
          </thead>
          <tbody>
            {uninferredModelRows(data).map(([label, value], index) => (
              <tr key={index}>
                <th>{label}</th>
                <td>{publication ? publicationNumber(value) : fmt(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (publication) {
    const grouped = data.coefficients.some(
      (coefficient) => coefficient.equation != null,
    );
    return (
      <>
        <div className="table-wrap">
          <table className="publication-grid">
            <thead>
              <tr>
                <th />
                <th>(1)</th>
              </tr>
              <tr>
                <th />
                <th>{data.spec.outcome}</th>
              </tr>
            </thead>
            <tbody>
              {data.coefficients.map((coefficient, index) => (
                <Fragment key={coefficient.term}>
                  {grouped &&
                    (index === 0 ||
                      coefficient.equation !==
                        data.coefficients[index - 1].equation) && (
                      <tr>
                        <th colSpan={2}>
                          [{coefficient.equation ?? "Other parameters"}]
                        </th>
                      </tr>
                    )}
                  <tr className="publication-estimate">
                    <th>{coefficient.term}</th>
                    <td>
                      {publicationNumber(coefficient.estimate)}
                      <sup>{significanceStars(coefficient.p_value)}</sup>
                    </td>
                  </tr>
                  <tr className="publication-standard-error">
                    <th />
                    <td>({publicationNumber(coefficient.std_error)})</td>
                  </tr>
                </Fragment>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th>Observations</th>
                <td>{data.nobs}</td>
              </tr>
            </tfoot>
          </table>
        </div>
        <p className="publication-notes">
          {String(data.inference?.covariance ?? data.spec.covariance)} standard
          errors in parentheses. *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10.
        </p>
      </>
    );
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Variable</th>
            <th>Coefficient</th>
            <th>Std. error</th>
            <th>p-value</th>
            <th>{fmt(100 * (1 - data.spec.alpha), 0)}% CI</th>
          </tr>
        </thead>
        <tbody>
          {data.coefficients.map((coefficient) => (
            <tr key={coefficient.term}>
              <th>{coefficient.term}</th>
              <td>{fmt(coefficient.estimate)}</td>
              <td>{fmt(coefficient.std_error)}</td>
              <td>
                {coefficient.p_value < 0.001
                  ? "< 0.001"
                  : fmt(coefficient.p_value)}
              </td>
              <td className="interval">
                [{fmt(coefficient.ci_low)}; {fmt(coefficient.ci_high)}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function ModelOutput({
  data,
  math,
  publication,
  notes,
}: {
  data: ResultBundle;
  math?: string | null;
  publication?: boolean;
  notes?: string[];
}) {
  const inferenceUnavailable = data.inference?.available === false;
  const fitStatistic =
    data.metrics.r_squared ??
    data.metrics.rsquared ??
    data.metrics.pseudo_r_squared ??
    data.metrics.pseudo_r2;
  return (
    <section
      className={`model-output${publication ? " publication-output" : ""}`}
    >
      <div className="model-title">
        <span className="model-tag">{data.spec.estimator.toUpperCase()}</span>
        <strong>{data.spec.outcome}</strong>
        <span>{data.nobs} observations</span>
      </div>
      <div className="model-metrics">
        <span>
          {inferenceUnavailable
            ? modelTargetLabel(data)
            : `${String(data.inference?.covariance ?? data.spec.covariance)} standard errors`}
        </span>
        {math && !publication && !inferenceUnavailable && (
          <span>{fmt(100 * (1 - data.spec.alpha), 2)}% CI</span>
        )}
        {fitStatistic != null && (
          <span>
            {(data.metrics.r_squared ?? data.metrics.rsquared) != null
              ? "R²"
              : "Pseudo R²"}{" "}
            ={" "}
            {publication ? publicationNumber(fitStatistic) : fmt(fitStatistic)}
          </span>
        )}
      </div>
      {math ? (
        <LatexView
          source={math}
          fallback={<ModelTable data={data} publication={publication} />}
        />
      ) : (
        <ModelTable data={data} publication={publication} />
      )}
      {notes?.map((note, index) => (
        <p className="publication-notes" key={index}>
          {note}
        </p>
      ))}
      {data.warnings.map((w, i) => (
        <p className="output-warning" key={i}>
          {w}
        </p>
      ))}
      <div className="model-footer">
        <span>
          {data.dropped_rows
            ? `${data.dropped_rows} observations excluded`
            : ""}
        </span>
        <button
          className="subtle-button"
          onClick={() =>
            download(JSON.stringify(data, null, 2), `openecon-${data.id}.json`)
          }
        >
          {data.display_omitted?.length ? "Display JSON ↓" : "JSON ↓"}
        </button>
      </div>
      {!!data.display_omitted?.length && (
        <p className="output-note">
          Display JSON omits {data.display_omitted.join(", ")}. Save the
          complete result with <code>model.model_dump_json()</code>.
        </p>
      )}
    </section>
  );
}
function Output({
  output,
  client,
}: {
  output: ConsoleOutput;
  client?: WorkspaceClient;
}) {
  if (output.type === "model")
    return (
      <ModelOutput
        data={output.data as ResultBundle}
        math={output.latex_math}
        publication={output.latex_style?.startsWith("publication-")}
        notes={output.latex_notes}
      />
    );
  if (output.type === "plot")
    return <PlotView plot={output.data as PlotSpec} client={client} />;
  if (output.type === "table") {
    const t = output.data as {
      columns: string[];
      rows: unknown[][];
      index_names?: (string | null)[];
      index?: unknown[][];
      total_rows: number;
      total_rows_known?: boolean;
      total_columns: number;
    };
    const table = (
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{t.index_names?.filter(Boolean).join(" / ") || "#"}</th>
              {t.columns.map((column, index) => (
                <th key={index}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row, index) => (
              <tr key={index}>
                <td className="row-number">
                  {t.index?.[index]
                    ?.map((value) => (value == null ? "—" : String(value)))
                    .join(" / ") ?? index + 1}
                </td>
                {row.map((value, column) => (
                  <td key={column}>
                    {value == null
                      ? "—"
                      : typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    return (
      <div className="data-output">
        <div className="output-label">
          Table{" "}
          <span>
            {t.total_rows_known === false ? "Preview" : `${t.total_rows} rows`}{" "}
            · {t.total_columns ?? t.columns.length} columns
          </span>
        </div>
        {output.latex_math ? (
          <LatexView source={output.latex_math} fallback={table} />
        ) : (
          table
        )}
        {(t.total_rows_known === false ||
          t.total_rows > t.rows.length ||
          t.total_columns > t.columns.length) && (
          <div className="output-note">
            Showing the first {t.rows.length} rows and {t.columns.length}{" "}
            columns.
          </div>
        )}
      </div>
    );
  }
  if (output.type === "latex") {
    const source =
      output.latex_math ??
      (typeof output.data === "string" ? output.data : (output.latex ?? ""));
    return (
      <>
        <LatexView source={source} />
        {output.latex_notes?.map((note, index) => (
          <p className="publication-notes" key={index}>
            {note}
          </p>
        ))}
      </>
    );
  }
  const text = (
    <pre className="text-output">
      {typeof output.data === "string"
        ? output.data
        : JSON.stringify(output.data, null, 2)}
    </pre>
  );
  return output.latex_math ? (
    <LatexView source={output.latex_math} fallback={text} />
  ) : (
    text
  );
}
export interface WorkbenchControls {
  prepareToLeave: () => Promise<void>;
  downloadDraft: () => void;
  cancelPending: () => void;
}
interface AppProps {
  client?: WorkspaceClient;
  readOnly?: boolean;
  isolatedRuns?: boolean;
  projectName?: string;
  toolbarTarget?: HTMLElement | null;
  onControls?: (controls: WorkbenchControls | null) => void;
}
function App({
  client = localClient,
  readOnly = false,
  isolatedRuns = false,
  projectName,
  toolbarTarget,
  onControls,
}: AppProps = {}) {
  const api = client.request;
  const [serverReadOnly, setServerReadOnly] = useState(false);
  const cannotEdit = readOnly || serverReadOnly;
  const [suggestionsStatus, setSuggestionsStatus] =
    useState<LocalSuggestionsStatus | null>(null);
  const [draftConflict, setDraftConflict] = useState(false);
  const conflictRef = useRef(false);
  const conflictBase = useRef("");
  const [conflictCopies, setConflictCopies] = useState<
    ScriptConflictSnapshot[]
  >([]);
  const [compareConflict, setCompareConflict] = useState(false);
  const [conflictLocal, setConflictLocal] = useState("");
  const [conflictBusy, setConflictBusy] = useState(false);
  const [conflictError, setConflictError] = useState("");
  const mounted = useRef(true);
  const [cloud, setCloud] = useState(false);
  const [code, setCode] = useState(STARTER),
    [savedCode, setSavedCode] = useState(""),
    [saveState, setSaveState] = useState("Connecting");
  const [history, setHistory] = useState<ExecutionRecord[]>([]),
    [variables, setVariables] = useState<ConsoleVariable[]>([]),
    [datasets, setDatasets] = useState<DatasetProfile[]>([]);
  const [sharingSnapshot, setSharingSnapshot] = useState<ResultSharingSnapshot | null>(null);
  const [fileDownload, setFileDownload] = useState<{
    id: string; name: string;
  } | null>(null);
  useEffect(() => {
    setFileDownload(null);
    return client.subscribeFileDownload?.(setFileDownload);
  }, [client]);
  const sharingRecords = useMemo(
    () => new Map(sharingSnapshot?.records.map((record) => [record.id, record]) ?? []),
    [sharingSnapshot],
  );
  const [historyLoading, setHistoryLoading] = useState(true),
    [historyError, setHistoryError] = useState("");
  const [accessLost, setAccessLost] = useState(false);
  const [ready, setReady] = useState(false),
    [running, setRunning] = useState(false),
    [uploading, setUploading] = useState(false),
    [error, setError] = useState("");
  const [generation, setGeneration] = useState(0);
  const [activeCommand, setActiveCommand] = useState("");
  const [filesCollapsed, setFilesCollapsed] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(max-width: 720px)").matches,
  );
  const [scripts, setScripts] = useState<ScriptSummary[]>([]);
  const [activeScript, setActiveScript] = useState<ScriptSummary>({
    id: "analysis",
    name: "analysis.py",
    version: 0,
  });
  const [fileBusy, setFileBusy] = useState(false);
  const [fileNavigating, setFileNavigating] = useState(false);
  const fileBusyRef = useRef(false);
  const [layoutBusy, setLayoutBusy] = useState(false);
  const layoutBusyRef = useRef(false);
  const layoutOperation = useRef<Promise<void> | null>(null);
  const layoutNotification = useRef<FileLayout | null>(null);
  const layoutClient = useRef(client);
  layoutClient.current = client;
  const [fileLayout, setFileLayout] = useState<FileLayout | null>(null);
  const fileLayoutRef = useRef(fileLayout);
  fileLayoutRef.current = fileLayout;
  const importParent = useRef<string | null>(null);
  const uploadParent = useRef<string | null>(null);
  const fileEntries = useMemo(
    () => mergeFileLayout(fileLayout?.entries ?? [], scripts, datasets),
    [fileLayout, scripts, datasets],
  );
  const activeScriptName =
    fileEntries.find(
      (entry) => entry.kind === "script" && entry.id === activeScript.id,
    )?.name ?? activeScript.name;
  const activeExtension = documentExtension(activeScriptName) ?? ".py";
  const pythonFile = activeExtension === ".py";
  const inlineContext = `${client.projectId ?? "local"}:${activeScript.id}`;
  const nativeWorkspace = Boolean(client.desktop || client.localDesktop);
  const inlineAllowed = Boolean(
    nativeWorkspace &&
    suggestionsStatus?.enabled &&
    suggestionsStatus.state === "ready" &&
    ready &&
    !cannotEdit &&
    !fileBusy &&
    !running &&
    !draftConflict &&
    !accessLost &&
    pythonFile,
  );
  const inlineProvider = useMemo(
    () => (inlineAllowed ? createLocalSuggestionProvider() : undefined),
    [inlineAllowed, inlineContext],
  );
  useEffect(() => () => inlineProvider?.dispose(), [inlineProvider]);
  const activeScriptRef = useRef(activeScript);
  activeScriptRef.current = activeScript;
  const activeScriptVersion = useRef(0);
  const changedScriptVersions = useRef(new Map<string, number>());
  const scriptSelectionKey = `openecon:active-script:${client.projectId ?? "local"}`;
  const [outputTab, setOutputTab] = useState<"results" | "plots" | "log">(
    "results",
  );
  const [outputFormat, setOutputFormat] = useState<"result" | "latex">(
    "result",
  );
  const [outputChoice, setOutputChoice] = useState("all");
  const [latexCopied, setLatexCopied] = useState(false);
  const [latexCopyError, setLatexCopyError] = useState(false);
  useEffect(() => {
    const failed = (event: Event) => setError(String((event as CustomEvent).detail));
    window.addEventListener("openecon-export-error", failed);
    return () => window.removeEventListener("openecon-export-error", failed);
  }, []);
  const [mobilePane, setMobilePane] = useState<"editor" | "output">("editor"),
    [cursor, setCursor] = useState([1, 1]),
    [selectedId, setSelectedId] = useState<string | null>(null);
  const [historicalRecord, setHistoricalRecord] = useState<{
    client: WorkspaceClient; record: ExecutionRecord;
  } | null>(null);
  const historyRef = useRef(history);
  historyRef.current = history;
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;
  const [showAgent, setShowAgent] = useState(false),
    [config, setConfig] = useState<AppConfig | null>(null),
    [agentError, setAgentError] = useState(""),
    [copied, setCopied] = useState(""),
    [showReset, setShowReset] = useState(false);
  const [showVariables, setShowVariables] = useState(false),
    [showHistory, setShowHistory] = useState(false);
  const terminal = useRef<WorkspaceTerminalHandle>(null);
  const editor = useRef<EditorHandle>(null),
    fileInput = useRef<HTMLInputElement>(null),
    scriptInput = useRef<HTMLInputElement>(null);
  const busy = useRef(false),
    initialized = useRef(false);
  const serverRunning = useRef(false);
  const acceptedRun = useRef<string | null>(null);
  const stateResponses = useRef(new ResponseGate());
  const agentResponses = useRef(new ResponseGate());
  const agentRequest = useRef<AbortController | null>(null);
  const agentClient = useRef(client);
  agentClient.current = client;
  const draftSaver = useRef<LatestValueAutosaver | null>(null);
  const codeRef = useRef(code);
  codeRef.current = code;
  function acceptLayoutNotification(value: FileLayout) {
    if (!Number.isSafeInteger(value.version) || value.version < 0) return;
    validateFileLayout(value.entries);
    if (value.sync_error) {
      if (!value.sync_conflict) setError(value.sync_error.message);
      if ([401, 403].includes(value.sync_error.status)) {
        setAccessLost(true);
        setServerReadOnly(true);
        draftSaver.current?.dispose();
      }
    }
    if (layoutBusyRef.current) {
      if (
        !layoutNotification.current ||
        value.version >= layoutNotification.current.version
      )
        layoutNotification.current = value;
      return;
    }
    if (fileLayoutRef.current && value.version < fileLayoutRef.current.version)
      return;
    fileLayoutRef.current = value;
    setFileLayout(value);
  }
  useEffect(() => {
    let active = true;
    const unsubscribe = client.subscribeFileLayout?.((value) => {
      if (active && mounted.current) acceptLayoutNotification(value);
    });
    return () => {
      active = false;
      unsubscribe?.();
    };
  }, [client]);
  useEffect(() => {
    let active = true;
    changedScriptVersions.current.clear();
    const unsubscribe = client.subscribeScriptSync?.(
      (value: ScriptSyncNotification) => {
        if (!active || !mounted.current) return;
        if (value.sync_error && !value.sync_conflict)
          setError(value.sync_error.message);
        const accessDenied = Boolean(
          value.read_only ||
          (value.sync_error && [401, 403].includes(value.sync_error.status)),
        );
        if (accessDenied) {
          setServerReadOnly(true);
          setAccessLost(true);
          draftSaver.current?.dispose();
          setSaveState("Read-only");
        }
        if (!value.file) return;
        const file = validateScriptFile(value.file);
        if (value.source_changed)
          changedScriptVersions.current.set(
            file.id,
            Math.max(
              changedScriptVersions.current.get(file.id) ?? -1,
              file.version,
            ),
          );
        setScripts((items) =>
          items.map((item) =>
            item.id === file.id && file.version >= item.version
              ? {
                  ...item,
                  version: file.version,
                  pending_sync: file.pending_sync,
                }
              : item,
          ),
        );
        if (
          activeScriptRef.current.id !== file.id ||
          file.version < activeScriptVersion.current
        )
          return;
        if (
          value.sync_conflict ||
          file.sync_conflict ||
          (value.source_changed && file.version > activeScriptVersion.current)
        ) {
          conflictRef.current = true;
          setDraftConflict(true);
          setSaveState("Version conflict");
          draftSaver.current?.dispose();
          return;
        }
        // A remote acknowledgement may update status, never the editor text
        // or an in-flight draft typed after the acknowledged local snapshot.
        if (
          !conflictRef.current &&
          !accessDenied &&
          file.code === codeRef.current
        ) {
          setSavedCode(file.code ?? "");
          setSaveState(file.pending_sync ? "Saved locally" : "Saved");
        }
      },
    );
    return () => {
      active = false;
      unsubscribe?.();
    };
  }, [client]);
  const initializeDraft = useCallback(
    (
      draft: ScriptDraft & { sync_conflict?: boolean },
      allowSave: boolean,
      file?: ScriptSummary,
    ) => {
      draftSaver.current?.dispose();
      draftSaver.current = null;
      const identity = file ?? {
        id: "analysis",
        name: "analysis.py",
        version: draft.version ?? 0,
      };
      const sourceChanged =
        (changedScriptVersions.current.get(identity.id) ?? -1) >
        (draft.version ?? 0);
      conflictRef.current = Boolean(draft.sync_conflict || sourceChanged);
      setDraftConflict(conflictRef.current);
      activeScriptVersion.current = draft.version ?? 0;
      activeScriptRef.current = identity;
      setActiveScript(identity);
      setCode(draft.code ?? STARTER);
      setSavedCode(draft.code ?? "");
      conflictBase.current = draft.code ?? "";
      setSaveState(
        allowSave
          ? conflictRef.current
            ? "Version conflict"
            : draft.code === null
              ? "Draft"
              : draft.pending_sync
                ? "Saved locally"
                : "Saved"
          : "Read-only",
      );
      if (!allowSave || conflictRef.current) return;
      const writer = new VersionedScriptWriter(
        client,
        draft.version ?? 0,
        file?.id,
      );
      let pendingSync = Boolean(draft.pending_sync);
      const saver = new LatestValueAutosaver({
        initialValue: draft.code ?? "",
        save: async (value) => {
          try {
            const saved = await writer.save(value);
            pendingSync = Boolean(saved.pending_sync);
            if (
              draftSaver.current === saver &&
              activeScriptRef.current.id === identity.id
            )
              activeScriptVersion.current = Math.max(
                activeScriptVersion.current,
                saved.version ?? 0,
              );
            if (mounted.current && file) {
              setScripts((items) =>
                items.map((item) =>
                  item.id === identity.id
                    ? {
                        ...item,
                        version: saved.version ?? item.version,
                        pending_sync: pendingSync,
                      }
                    : item,
                ),
              );
            }
            return saved;
          } catch (failure) {
            if (
              writer.conflict &&
              mounted.current &&
              activeScriptRef.current.id === identity.id
            ) {
              conflictRef.current = true;
              setDraftConflict(true);
              setSaveState("Version conflict");
              saver.dispose();
            }
            throw failure;
          }
        },
        onStatus: (status) => {
          if (
            !mounted.current ||
            conflictRef.current ||
            activeScriptRef.current.id !== identity.id
          )
            return;
          setSaveState(
            status === "saved"
              ? pendingSync
                ? "Saved locally"
                : "Saved"
              : status === "saving"
                ? "Saving"
                : "Could not save",
          );
          if (status === "saved") {
            setSavedCode(saver.getSavedValue() ?? "");
            conflictBase.current = saver.getSavedValue() ?? "";
          }
        },
      });
      draftSaver.current = saver;
    },
    [client],
  );
  const selected = (historicalRecord?.client === client && historicalRecord.record.id === selectedId ? historicalRecord.record : null)
    || history.find((r) => r.id === selectedId) || history.at(-1);
  const currentOutputs = selected?.outputs ?? EMPTY_OUTPUTS;
  const plots = useMemo(
    () => currentOutputs.filter((output) => output.type === "plot"),
    [currentOutputs],
  );
  const availableOutputs = outputTab === "plots" ? plots : currentOutputs;
  const visibleOutputs = useMemo(() => {
    const chosenOutput = Number(outputChoice);
    return outputChoice === "all" || !availableOutputs[chosenOutput]
      ? availableOutputs
      : [availableOutputs[chosenOutput]];
  }, [availableOutputs, outputChoice]);
  const outputTimeline = useMemo(
    () => (selected ? orderedExecutionEvents(selected) : []),
    [selected],
  );
  const visibleTimeline = useMemo(() => {
    const chosenOutput = availableOutputs[Number(outputChoice)];
    return selectExecutionEvents(outputTimeline, {
      plotsOnly: outputTab === "plots",
      outputIndex:
        outputChoice === "all" || !chosenOutput
          ? undefined
          : currentOutputs.indexOf(chosenOutput),
    });
  }, [
    outputTimeline,
    availableOutputs,
    currentOutputs,
    outputChoice,
    outputTab,
  ]);
  const latexSource = useMemo(
    () => latexDocument(timelineLatexOutputs(visibleTimeline)),
    [visibleTimeline],
  );
  useEffect(() => {
    setOutputChoice("all");
    setLatexCopied(false);
    setLatexCopyError(false);
  }, [selected?.id, outputTab]);
  useEffect(() => {
    setLatexCopied(false);
    setLatexCopyError(false);
  }, [latexSource]);
  async function copyLatex() {
    try {
      await navigator.clipboard.writeText(latexSource);
      setLatexCopied(true);
      setLatexCopyError(false);
    } catch {
      setLatexCopyError(true);
    }
  }
  const refreshState = useCallback(async (signal?: AbortSignal) => {
    const ticket = stateResponses.current.issue();
    try {
      const s = await api<ConsoleState>("/console", { signal });
      if (
        signal?.aborted ||
        !mounted.current ||
        !stateResponses.current.accept(ticket)
      )
        return s;
      setHistory(s.history);
      const completed = acceptedRun.current
        ? s.history.find((record) => record.id === acceptedRun.current)
        : null;
      if (completed) {
        acceptedRun.current = null;
        setSelectedId(completed.id);
        setOutputTab(
          completed.status !== "ok" || !completed.outputs.length
            ? "log"
            : "results",
        );
      } else {
        const incoming = incomingAgentSelection(
          historyRef.current,
          s.history,
          selectedIdRef.current,
        );
        if (incoming) {
          setSelectedId(incoming);
          setOutputTab("results");
        }
      }
      historyRef.current = s.history;
      setVariables(isolatedRuns ? [] : s.variables);
      setGeneration(s.status.session_generation);
      serverRunning.current = s.status.running;
      setRunning(busy.current || s.status.running);
      setHistoryLoading(false);
      setHistoryError("");
      return s;
    } catch (failure) {
      if (signal?.aborted) throw failure;
      if (mounted.current && stateResponses.current.accept(ticket)) {
        setHistoryLoading(false);
        setHistoryError(errorMessage(failure));
        if (
          failure instanceof ApiError &&
          [401, 403, 404].includes(failure.status)
        ) {
          // Live auth/membership failures stop editing and scheduled writes.
          // A page reload must re-read the draft and role before resuming.
          setAccessLost(true);
          setServerReadOnly(true);
          setReady(false);
          draftSaver.current?.dispose();
        }
      }
      throw failure;
    }
  }, [api, isolatedRuns]);
  useEffect(() => {
    setConfig(null);
    setShowAgent(false);
    setAgentError("");
    setCopied("");
    return () => {
      agentRequest.current?.abort();
      agentResponses.current.invalidate();
    };
  }, [client]);
  useEffect(() => {
    let active = true;
    mounted.current = true;
    stateResponses.current.invalidate();
    acceptedRun.current = null;
    initialized.current = false;
    layoutBusyRef.current = false;
    layoutOperation.current = null;
    layoutNotification.current = null;
    setLayoutBusy(false);
    setReady(false);
    setHistoryLoading(true);
    setHistoryError("");
    (async () => {
      try {
        const { session, datasets, status, initialConsole } =
          await loadWorkspaceBootstrap(client);
        const files = await loadScriptFiles(client);
        let layout: FileLayout | null = null;
        try {
          layout = await readFileLayout();
        } catch (failure) {
          if (
            !(failure instanceof ApiError && failure.status === 404) &&
            active
          )
            setError(errorMessage(failure));
        }
        let preferredId: string | null = null;
        try {
          preferredId = window.localStorage.getItem(scriptSelectionKey);
        } catch {
          /* Selection is optional. */
        }
        const selectedFile =
          files.find((file) => file.id === preferredId) ??
          files.find((file) => file.id === "analysis")!;
        const selectedDraft = await loadScriptFile(client, selectedFile.id);
        if (!active) return;
        setAccessLost(false);
        setServerReadOnly(Boolean(session.read_only));
        setScripts(files);
        setFileLayout(layout);
        initializeDraft(
          selectedDraft,
          !readOnly && !session.read_only,
          selectedFile,
        );
        setCloud(
          session.environment === "cloud" || session.environment === "team",
        );
        setHistory(initialConsole?.history ?? []);
        setVariables(isolatedRuns ? [] : (initialConsole?.variables ?? []));
        setGeneration(status.session_generation);
        setRunning(status.running);
        serverRunning.current = status.running;
        setDatasets(datasets);
        initialized.current = true;
        setReady(true);
        qaPaint("project");
        if (initialConsole) setHistoryLoading(false);
        else void refreshState().catch(() => {});
      } catch (e) {
        if (active) {
          setHistoryLoading(false);
          setError(errorMessage(e));
        }
      }
    })();
    return () => {
      active = false;
      stateResponses.current.invalidate();
    };
  }, [
    client,
    initializeDraft,
    readOnly,
    isolatedRuns,
    refreshState,
    api,
    scriptSelectionKey,
  ]);
  useEffect(() => {
    if (ready && initialized.current && !cannotEdit)
      draftSaver.current?.update(code);
  }, [code, ready, cannotEdit]);
  useEffect(
    () => () => {
      mounted.current = false;
      draftSaver.current?.dispose();
      stateResponses.current.invalidate();
      client.cancelPending();
    },
    [client],
  );
  useEffect(() => {
    if (!client.teams) return;
    const protectDraft = (event: BeforeUnloadEvent) => {
      if (
        cannotEdit ||
        !initialized.current ||
        (!layoutBusyRef.current &&
          draftSaver.current?.getSavedValue() === codeRef.current)
      )
        return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", protectDraft);
    return () => window.removeEventListener("beforeunload", protectDraft);
  }, [client, cannotEdit]);
  useEffect(() => {
    onControls?.({
      prepareToLeave: async () => {
        await layoutOperation.current;
        if (conflictRef.current)
          throw new Error(
            "Your draft conflicts with another version. Download your local copy before leaving.",
          );
        if (!cannotEdit) {
          draftSaver.current?.update(codeRef.current);
          await draftSaver.current?.flush();
        }
      },
      downloadDraft: () =>
        download(
          codeRef.current,
          localCopyName(activeScriptName),
          documentMimeType(activeScriptName),
        ),
      cancelPending: () => {
        draftSaver.current?.dispose();
        client.cancelPending();
      },
    });
    return () => onControls?.(null);
  }, [onControls, cannotEdit, client, activeScriptName]);
  useEffect(() => {
    if (!ready || !running) return;
    const timer = window.setInterval(() => {
      void refreshState().catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [running, ready, refreshState]);
  useEffect(() => {
    if (
      !ready ||
      accessLost ||
      (client.teams && !client.desktop) ||
      (cloud && !client.desktop)
    )
      return;
    const watcher = watchResultRevisions({
      readRevision: (signal) =>
        api<{ revision: string }>("/results/revision", { signal }),
      onChanged: refreshState,
      onError: (failure) => {
        if (
          failure instanceof ApiError &&
          [401, 403].includes(failure.status)
        ) {
          setAccessLost(true);
          setServerReadOnly(true);
          setReady(false);
          setHistoryError(errorMessage(failure));
          draftSaver.current?.dispose();
        }
      },
    });
    return () => watcher.dispose();
  }, [ready, accessLost, client, cloud, api, refreshState]);
  async function execute(source: string, scriptId?: string) {
    if (
      cannotEdit ||
      !source.trim() ||
      !ready ||
      busy.current ||
      fileBusyRef.current ||
      uploading ||
      running ||
      serverRunning.current ||
      (scriptId !== undefined && !pythonFile)
    )
      return;
    editor.current?.cancelInlineSuggestion();
    inlineProvider?.cancelAll();
    if (isolatedRuns && scriptId !== undefined)
      source = editor.current?.getCode() || code;
    busy.current = true;
    qaStart("run");
    stateResponses.current.invalidate();
    setRunning(true);
    setError("");
    setSelectedId(null);
    setActiveCommand(source);
    try {
      const r = await submitWorkspaceExecution(client, source, scriptId);
      if (!mounted.current) return;
      // Polls or the initial history request may predate this new record.
      stateResponses.current.invalidate();
      if (isExecutionAccepted(r)) {
        acceptedRun.current = r.id;
        serverRunning.current = true;
        setRunning(true);
        setSelectedId(r.id);
        setOutputTab("results");
        setMobilePane("output");
        // No placeholder record: only /console supplies validated output.
        void refreshState().catch(() => {});
        return;
      }
      setHistory((items) => [...items.filter((x) => x.id !== r.id), r]);
      qaPaint("run");
      setSelectedId(r.id);
      setOutputTab(r.status !== "ok" || !r.outputs.length ? "log" : "results");
      setMobilePane("output");
      await refreshState();
    } catch (e) {
      if (mounted.current) {
        setError(errorMessage(e));
        if (client.teams && (!(e instanceof ApiError) || e.status >= 500)) {
          // A lost acceptance response may still have launched a task. Keep
          // polling the durable lock instead of suggesting a second launch.
          stateResponses.current.invalidate();
          serverRunning.current = true;
          void refreshState().catch(() => {});
        }
      }
    } finally {
      busy.current = false;
      if (mounted.current) setRunning(serverRunning.current);
    }
  }
  function executeTerminal(source: string) {
    try {
      const isPackageCommand =
        /^[%!](?:pip|uv)(?:\s|$)|^(?:pip|uv)\s+(?:install|pip|add)(?:\s|$)/.test(
          source.trim(),
        );
      void execute(isPackageCommand ? normalizePackageCommand(source) : source);
    } catch (failure) {
      setError(errorMessage(failure));
    }
  }
  async function interrupt() {
    if (cannotEdit) return;
    stateResponses.current.invalidate();
    try {
      await api("/console/interrupt", { method: "POST" });
      await refreshState();
    } catch (e) {
      setError(errorMessage(e));
    }
  }
  async function reset() {
    if (cannotEdit || isolatedRuns) return;
    setShowReset(false);
    stateResponses.current.invalidate();
    try {
      await api("/console/reset", { method: "POST" });
      await refreshState();
      setError("");
    } catch (e) {
      setError(errorMessage(e));
    }
  }
  async function upload(file?: File, parent: string | null = null) {
    if (
      cannotEdit ||
      !ready ||
      fileBusyRef.current ||
      layoutBusyRef.current ||
      uploading ||
      running
    )
      return;
    setUploading(true);
    setError("");
    try {
      const f = new FormData();
      if (file) f.append("file", file);
      const d = await api<DatasetProfile>("/datasets/upload", {
        method: "POST",
        ...(client.desktop ? {} : { body: f }),
      });
      setDatasets((items) => [d, ...items.filter((x) => x.id !== d.id)]);
      void client.transferActions?.refresh().catch(() => {});
      addSnippet(
        d.python_path
          ? `df = oe.read(${JSON.stringify(d.python_path)})\ndisplay(df.head())`
          : `df = oe.load_dataset(${JSON.stringify(d.id)})\ndisplay(df.head())`,
      );
      fileBusyRef.current = true;
      setFileBusy(true);
      try {
        draftSaver.current?.update(
          editor.current?.getCode() ?? codeRef.current,
        );
        await draftSaver.current?.flush();
        await organizeCreatedFile("dataset", d.id, parent);
      } finally {
        fileBusyRef.current = false;
        if (mounted.current) setFileBusy(false);
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError"))
        setError(errorMessage(e));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }
  async function openAgent() {
    agentRequest.current?.abort();
    const controller = new AbortController();
    agentRequest.current = controller;
    const ticket = agentResponses.current.issue();
    setShowAgent(true);
    setConfig(null);
    setAgentError("");
    setCopied("");
    try {
      const value = await api<AppConfig>("/config", {
        signal: controller.signal,
      });
      if (
        !controller.signal.aborted &&
        mounted.current &&
        agentClient.current === client &&
        agentResponses.current.accept(ticket)
      )
        setConfig(value);
    } catch (e) {
      if (
        !controller.signal.aborted &&
        mounted.current &&
        agentClient.current === client &&
        agentResponses.current.accept(ticket)
      )
        setAgentError(errorMessage(e));
    } finally {
      if (agentRequest.current === controller) agentRequest.current = null;
    }
  }
  function closeAgent() {
    agentRequest.current?.abort();
    agentResponses.current.invalidate();
    setShowAgent(false);
  }
  async function copy(value: string, key: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(key);
    } catch {
      setAgentError("Could not copy. Select the command to copy it manually.");
    }
  }
  async function reloadConflictedDraft() {
    if (!ready || fileBusyRef.current) return;
    fileBusyRef.current = true;
    setFileBusy(true);
    setError("");
    const identity = activeScriptRef.current;
    try {
      const exported = await download(
        editor.current?.getCode() ?? codeRef.current,
        localCopyName(activeScriptName),
        documentMimeType(activeScriptName),
      );
      if (["cancelled", "failed"].includes(exported.status))
        throw new Error("Keep the local draft until its copy has been exported.");
      const draft = await loadScriptFile(client, identity.id, {
        headers: { "X-OpenEcon-Resolve-Conflict": "remote" },
      });
      if (!mounted.current || activeScriptRef.current.id !== identity.id)
        return;
      const file = {
        ...identity,
        name: draft.name,
        version: draft.version,
        pending_sync: draft.pending_sync,
      };
      setScripts((items) =>
        items.map((item) => (item.id === identity.id ? file : item)),
      );
      initializeDraft(draft, !cannotEdit, file);
    } catch (failure) {
      if (mounted.current) setError(errorMessage(failure));
    } finally {
      fileBusyRef.current = false;
      if (mounted.current) setFileBusy(false);
    }
  }
  async function readConflict() {
    if (!ready || fileBusyRef.current || conflictBusy) return;
    if (conflictCopies.length >= 20) {
      setConflictError(
        "Download the comparison copies before opening more versions.",
      );
      return;
    }
    const identity = activeScriptRef.current;
    setConflictBusy(true);
    setConflictError("");
    try {
      const snapshot = client.readScriptConflict
        ? await client.readScriptConflict(identity.id)
        : {
            base: conflictBase.current,
            local: {
              ...identity,
              version: activeScriptVersion.current,
              code: editor.current?.getCode() ?? codeRef.current,
            },
            remote: await loadScriptFile(client, identity.id),
          };
      if (!mounted.current || activeScriptRef.current.id !== identity.id)
        return;
      setConflictCopies((items) => [...items, snapshot]);
      if (!compareConflict)
        setConflictLocal(editor.current?.getCode() ?? codeRef.current);
      setCompareConflict(true);
    } catch (failure) {
      if (mounted.current) setConflictError(errorMessage(failure));
    } finally {
      if (mounted.current) setConflictBusy(false);
    }
  }
  async function saveConflict(code: string) {
    if (conflictBusy || cannotEdit || !compareConflict) return;
    const snapshot = conflictCopies.at(-1);
    if (!snapshot) return;
    setConflictBusy(true);
    setConflictError("");
    fileBusyRef.current = true;
    setFileBusy(true);
    try {
      const saved = client.resolveScriptConflict
        ? await client.resolveScriptConflict(snapshot, code)
        : validateScriptFile(
            await api<ScriptFile>(`/console/scripts/${snapshot.remote.id}`, {
              method: "PUT",
              body: JSON.stringify({ code, version: snapshot.remote.version }),
            }),
            snapshot.remote.id,
          );
      if (!mounted.current || activeScriptRef.current.id !== snapshot.local.id)
        return;
      if (saved.code !== code)
        throw new Error("Could not verify the merged draft.");
      changedScriptVersions.current.delete(saved.id);
      initializeDraft(saved, true, saved);
      setCompareConflict(false);
      setConflictCopies([]);
      setConflictLocal("");
    } catch (failure) {
      if (mounted.current) setConflictError(errorMessage(failure));
    } finally {
      fileBusyRef.current = false;
      if (mounted.current) {
        setFileBusy(false);
        setConflictBusy(false);
      }
    }
  }
  async function downloadArtifact(artifact: { name: string; url: string }) {
    if (!client.projectId) return;
    try {
      const blob = await api<Blob>(
        workspaceDownloadPath(artifact.url, client.projectId),
        {},
        "blob",
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = artifact.name;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (failure) {
      if (mounted.current) setError(errorMessage(failure));
    }
  }
  function addSnippet(source: string) {
    if (cannotEdit || !ready || fileBusyRef.current || !pythonFile) return;
    editor.current?.insert(source);
    setMobilePane("editor");
    requestAnimationFrame(() => editor.current?.focus());
  }
  async function changeScript(
    operation: () => Promise<ScriptFile>,
    afterChange?: (draft: ScriptFile) => Promise<void>,
    inlineFailure = false,
    navigation = false,
  ) {
    if (
      !ready ||
      fileBusyRef.current ||
      layoutBusyRef.current ||
      uploading ||
      running
    ) {
      if (inlineFailure)
        throw new Error("The file cannot be created right now.");
      return false;
    }
    if (conflictRef.current) {
      if (inlineFailure)
        throw new Error("Resolve the open file’s version conflict first.");
      setError(
        "Resolve the version conflict before switching files. Your code is preserved in the editor.",
      );
      return false;
    }
    fileBusyRef.current = true;
    setFileBusy(true);
    setFileNavigating(navigation);
    setError("");
    try {
      if (!cannotEdit) {
        draftSaver.current?.update(
          editor.current?.getCode() ?? codeRef.current,
        );
        await draftSaver.current?.flush();
      }
      const draft = validateScriptFile(await operation());
      if (!mounted.current) return;
      const file = {
        id: draft.id,
        name: draft.name,
        version: draft.version,
        pending_sync: draft.pending_sync,
      };
      setScripts((items) =>
        items.some((item) => item.id === draft.id)
          ? items.map((item) => (item.id === draft.id ? file : item))
          : [...items, file],
      );
      initializeDraft(draft, !cannotEdit, file);
      try {
        window.localStorage.setItem(scriptSelectionKey, draft.id);
      } catch {
        /* Selection is optional. */
      }
      setMobilePane("editor");
      setCursor([1, 1]);
      try {
        await afterChange?.(draft);
      } catch (failure) {
        // Creation is already durable. Report a placement failure without
        // offering to create the same source a second time.
        if (mounted.current) setError(errorMessage(failure));
      }
      requestAnimationFrame(() => {
        if (document.activeElement?.closest(".files-sidebar__edit")) return;
        editor.current?.focus();
      });
      return true;
    } catch (failure) {
      if (inlineFailure) throw failure;
      if (mounted.current) setError(errorMessage(failure));
      return false;
    } finally {
      fileBusyRef.current = false;
      if (mounted.current) {
        setFileBusy(false);
        setFileNavigating(false);
      }
    }
  }
  async function openScript(id: string) {
    if (id === activeScriptRef.current.id) {
      setMobilePane("editor");
      requestAnimationFrame(() => {
        if (document.activeElement?.closest(".files-sidebar__edit")) return;
        editor.current?.focus();
      });
      return;
    }
    qaStart("file");
    await changeScript(
      () => loadScriptFile(client, id),
      undefined,
      false,
      true,
    );
  }
  async function readFileLayout(options?: RequestInit): Promise<FileLayout> {
    const value = await api<FileLayout>("/files/layout", options);
    if (!value || !Number.isSafeInteger(value.version) || value.version < 0)
      throw new Error("The file layout could not be verified.");
    validateFileLayout(value.entries);
    return value;
  }
  async function saveFileLayout(entries: FileLayoutEntry[]) {
    if (
      cannotEdit ||
      !ready ||
      fileBusyRef.current ||
      layoutBusyRef.current ||
      running
    )
      throw new Error("The file layout cannot be changed right now.");
    if (conflictRef.current)
      throw new Error("Resolve your code’s version conflict first.");
    const previous = fileLayoutRef.current;
    if (!previous) throw new Error("Reload the file layout.");
    validateFileLayout(entries);
    const owner = client;
    const optimistic = { ...previous, entries };
    layoutBusyRef.current = true;
    setLayoutBusy(true);
    fileLayoutRef.current = optimistic;
    setFileLayout(optimistic);
    editor.current?.cancelInlineSuggestion();
    const operation = persistFileLayout(previous, entries);
    layoutOperation.current = operation;
    try {
      // Logical names do not alter the executable file or its autosave identity.
      await operation;
    } catch (failure) {
      if (
        mounted.current &&
        layoutClient.current === owner &&
        fileLayoutRef.current === optimistic
      ) {
        fileLayoutRef.current = previous;
        setFileLayout(previous);
      }
      throw failure;
    } finally {
      if (layoutClient.current === owner) {
        layoutBusyRef.current = false;
        if (layoutOperation.current === operation)
          layoutOperation.current = null;
        if (mounted.current) {
          setLayoutBusy(false);
          const queued = layoutNotification.current;
          layoutNotification.current = null;
          if (queued) acceptLayoutNotification(queued);
        }
      }
    }
  }
  async function persistFileLayout(
    previous: FileLayout,
    entries: FileLayoutEntry[],
  ) {
    const owner = client;
    try {
      const saved = await readFileLayout({
        method: "PUT",
        body: JSON.stringify({ version: previous.version, entries }),
      });
      if (mounted.current && layoutClient.current === owner) {
        fileLayoutRef.current = saved;
        setFileLayout(saved);
      }
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 409) {
        try {
          const [currentScripts, currentData] = await Promise.all([
            loadScriptFiles(client),
            api<{ datasets: DatasetProfile[] }>("/datasets"),
          ]);
          const fresh = await readFileLayout();
          if (mounted.current && layoutClient.current === owner) {
            setScripts(currentScripts);
            setDatasets(currentData.datasets);
            fileLayoutRef.current = fresh;
            setFileLayout(fresh);
          }
        } catch {
          /* Keep the current layout if refresh is unavailable. */
        }
      }
      throw failure;
    }
  }
  async function organizeCreatedFile(
    kind: "script" | "dataset",
    id: string,
    parent: string | null,
    name?: string,
  ) {
    if (!fileLayoutRef.current) return;
    const fresh = await readFileLayout();
    if (!mounted.current) return;
    fileLayoutRef.current = fresh;
    setFileLayout(fresh);
    let entries = fresh.entries;
    if (
      name !== undefined &&
      entries.some(
        (entry) =>
          entry.kind === kind && entry.id === id && entry.name !== name,
      )
    )
      entries = renameFileEntry(entries, `${kind}:${id}`, name);
    if (parent !== null)
      entries = moveFileEntry(entries, `${kind}:${id}`, parent);
    if (entries !== fresh.entries) {
      await persistFileLayout(fresh, entries);
    }
  }
  async function resolveFileLayoutConflict() {
    if (!fileLayoutRef.current || fileBusyRef.current) return;
    fileBusyRef.current = true;
    setFileBusy(true);
    try {
      const exported = await download(
        JSON.stringify(fileLayoutRef.current, null, 2),
        "file-layout-local.json",
      );
      if (["cancelled", "failed"].includes(exported.status))
        throw new Error("Keep the local file layout until its copy has been exported.");
      const fresh = await readFileLayout({
        headers: { "X-OpenEcon-Resolve-Conflict": "remote" },
      });
      if (mounted.current) {
        fileLayoutRef.current = fresh;
        setFileLayout(fresh);
      }
    } catch (failure) {
      if (mounted.current) setError(errorMessage(failure));
    } finally {
      fileBusyRef.current = false;
      if (mounted.current) setFileBusy(false);
    }
  }
  async function createScript(
    file?: File,
    parent: string | null = null,
    requestedName?: string,
  ) {
    if (cannotEdit) throw new Error("You cannot create files in this project.");
    const name =
      requestedName === undefined
        ? undefined
        : normalizeDocumentName(requestedName);
    if (name !== undefined) {
      const entries = fileLayoutRef.current?.entries ?? [];
      if (
        parent !== null &&
        !entries.some((entry) => entry.kind === "folder" && entry.id === parent)
      )
        throw new Error("Folder not found.");
      if (
        entries.some(
          (entry) =>
            entry.parent === parent &&
            fileNameFold(entry.name) === fileNameFold(name),
        )
      )
        throw new Error("This name already exists in the folder.");
    }
    await changeScript(
      async () => {
        if (file && file.size > 256 * 1024)
          throw new Error("Files can contain up to 64,000 characters.");
        const text = file ? await file.text() : "";
        const fileName = validFileName(
          file
            ? file.name.replace(/\.(py|md|tex|txt)$/i, (suffix) =>
                suffix.toLowerCase() === ".txt" ? ".py" : suffix.toLowerCase(),
              )
            : (name ?? "untitled.py"),
          "script",
        );
        if (Array.from(text).length > 64_000)
          throw new Error("Files can contain up to 64,000 characters.");
        const created = await api<ScriptFile>("/console/scripts", {
          method: "POST",
          body: JSON.stringify({ name: fileName, code: text }),
        });
        return created;
      },
      (created) => organizeCreatedFile("script", created.id, parent, name),
      name !== undefined,
    );
  }
  const toolbar = (
    <nav className="menu-bar" aria-label="Workspace tools">
      <div className="file-actions">
        <button
          onClick={() =>
            download(code, activeScriptName, documentMimeType(activeScriptName))
          }
        >
          Download file
        </button>
        <button onClick={() => setShowHistory(true)}>History</button>
        {nativeWorkspace && (
          <PackageManager
            client={client}
            disabled={!ready || running || uploading}
            readOnly={cannotEdit}
            onRunCommand={executeTerminal}
            onEnvironmentChanged={async () => {
              await refreshState();
            }}
          />
        )}
        {nativeWorkspace && (
          <InlineSuggestionsSettings onStatusChange={setSuggestionsStatus} />
        )}
        {!isolatedRuns && (
          <button onClick={() => setShowVariables(true)}>Variables</button>
        )}
        {!isolatedRuns && (
          <button
            disabled={!ready || cannotEdit}
            onClick={() => void openAgent()}
          >
            Connect agent
          </button>
        )}
      </div>
    </nav>
  );
  return (
    <div className="studio">
      {!client.projectId && (
        <header className="app-header">
          <span className="brand">OpenEconometrics</span>
          {projectName && <span>{projectName}</span>}
        </header>
      )}
      {toolbarTarget ? createPortal(toolbar, toolbarTarget) : toolbar}
      <input
        ref={fileInput}
        type="file"
        hidden
        accept=".csv,.parquet,.dta,.xlsx"
        aria-label="Data file"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void upload(f, uploadParent.current);
        }}
      />
      <input
        ref={scriptInput}
        type="file"
        hidden
        accept=".py,.md,.tex,.txt"
        aria-label="Import file"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void createScript(f, importParent.current);
          e.target.value = "";
        }}
      />
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button
            className="icon-button"
            aria-label="Dismiss error"
            onClick={() => setError("")}
          >
            <Symbol name="close" />
          </button>
        </div>
      )}
      {draftConflict && (
        <div className="draft-conflict" role="alert">
          <span>
            Someone else updated this file. Your local changes are preserved in
            the editor; autosave is paused.
          </span>
          <button disabled={conflictBusy} onClick={() => void readConflict()}>Compare and merge versions</button>
          <button onClick={() => void reloadConflictedDraft()}>
            Download local copy and open latest draft
          </button>
        </div>
      )}
      {draftConflict && conflictError && !compareConflict && (
        <p role="alert">{conflictError}</p>
      )}
      {compareConflict && conflictCopies.length > 0 && (
        <ScriptConflict
          snapshots={conflictCopies}
          initialCode={conflictLocal}
          busy={conflictBusy}
          error={conflictError}
          onSave={(code) => void saveConflict(code)}
          onRefresh={() => void readConflict()}
          onClose={() => setCompareConflict(false)}
        />
      )}
      {fileLayout?.sync_conflict && (
        <div className="draft-conflict" role="alert">
          <span>The file layout changed in another session.</span>
          <button
            disabled={fileBusy}
            onClick={() => void resolveFileLayoutConflict()}
          >
            Download local layout and open latest layout
          </button>
        </div>
      )}
      {cloud && !isolatedRuns && (
        <div className="cloud-notice" role="note">
          Temporary session. Files are deleted when the server restarts;
          download any outputs you need.
        </div>
      )}
      <div className="mobile-switch">
        <button
          className={mobilePane === "editor" ? "active" : ""}
          onClick={() => setMobilePane("editor")}
        >
          Code editor
        </button>
        <button
          className={mobilePane === "output" ? "active" : ""}
          onClick={() => setMobilePane("output")}
        >
          Results {history.length ? `(${history.length})` : ""}
        </button>
      </div>
      <WorkspaceLayout
        projectId={client.projectId}
        collapsed={filesCollapsed}
        sidebar={
          <FilesSidebar
            files={datasets}
            collapsed={filesCollapsed}
            onCollapsedChange={setFilesCollapsed}
            scripts={scripts}
            layout={fileEntries}
            pendingSync={Boolean(fileLayout?.pending_sync)}
            onLayoutChange={fileLayout ? saveFileLayout : undefined}
            activeScriptId={activeScript.id}
            disabled={!ready || fileBusy || layoutBusy || uploading || running}
            navigationPending={
              fileNavigating && ready && !layoutBusy && !uploading && !running
            }
            readOnly={cannotEdit}
            uploading={uploading}
            footer={
              <>
                {fileDownload && client.cancelFileDownload && (
                  <ProjectFileDownload
                    key={fileDownload.id}
                    file={fileDownload}
                    cancel={client.cancelFileDownload}
                  />
                )}
                {ready && client.transferActions && (
                  <ProjectTransfers
                    actions={client.transferActions}
                    readOnly={cannotEdit}
                    disabled={!ready || fileBusy || layoutBusy || uploading || running}
                    onReady={async (file) => {
                      if (!mounted.current) return;
                      setDatasets((items) => [file, ...items.filter((item) => item.id !== file.id)]);
                      try {
                        await organizeCreatedFile("dataset", file.id, null);
                      } catch (failure) {
                        if (mounted.current) setError(errorMessage(failure));
                      }
                    }}
                  />
                )}
              </>
            }
            onOpenScript={(file) => void openScript(file.id)}
            onCreateScript={(parent, name) =>
              createScript(undefined, parent, name)
            }
            onOpen={(file) =>
              addSnippet(
                file.python_path
                  ? `df = oe.read(${JSON.stringify(file.python_path)})\ndisplay(df.head())`
                  : `df = oe.load_dataset(${JSON.stringify(file.id)})\ndisplay(df.head())`,
              )
            }
            onUpload={(parent) => {
              uploadParent.current = parent ?? null;
              client.desktop
                ? void upload(undefined, parent)
                : fileInput.current?.click();
            }}
            onImportScript={(parent) => {
              importParent.current = parent ?? null;
              scriptInput.current?.click();
            }}
            onDownload={
              client.teams && client.projectId
                ? (file) =>
                    void downloadArtifact({
                      name:
                        fileEntries.find(
                          (entry) =>
                            entry.kind === "dataset" && entry.id === file.id,
                        )?.name ?? file.name,
                      url: `/api/projects/${encodeURIComponent(client.projectId!)}/workspace/files/${encodeURIComponent(file.id)}/download`,
                    })
                : undefined
            }
          />
        }
      >
        <WorkbenchPanels projectId={client.projectId} mobilePane={mobilePane}>
          <section className="editor-pane" aria-label="File workspace">
            <div className="editor-filebar">
              <div className="file-tab" title={activeScriptName}>
                {activeScriptName}
              </div>
              <div className="editor-actions">
                <span className="save-state" role="status">
                  {cannotEdit
                    ? "Read-only"
                    : code !== savedCode && saveState === "Saved"
                      ? "Modified"
                      : saveState}
                </span>
                {pythonFile && !isolatedRuns && (
                  <button
                    className="subtle-button selection-run"
                    disabled={
                      !ready || running || cannotEdit || fileBusy || uploading
                    }
                    onClick={() =>
                      void execute(
                        editor.current?.getSelectionOrLine() || "",
                        activeScript.id,
                      )
                    }
                    title="Run the selection or current line (⌘/Ctrl + Enter)"
                  >
                    Run selection
                  </button>
                )}
                {running ? (
                  <button
                    className="run-button"
                    disabled={cannotEdit}
                    onClick={() => void interrupt()}
                  >
                    <Symbol name="stop" size={13} />
                    Stop
                  </button>
                ) : pythonFile ? (
                  <button
                    className="run-button"
                    disabled={!ready || cannotEdit || fileBusy || uploading}
                    onClick={() =>
                      void execute(
                        editor.current?.getCode() || code,
                        activeScript.id,
                      )
                    }
                    title={
                      isolatedRuns
                        ? "Run the entire file in a fresh Python environment (⌘/Ctrl + Enter)"
                        : "Run the file (⌘/Ctrl + Shift + Enter)"
                    }
                  >
                    <Symbol name="play" size={13} />
                    Run
                  </button>
                ) : null}
              </div>
            </div>
            <CodeEditor
              documentKey={inlineContext}
              language={
                activeExtension === ".md"
                  ? "markdown"
                  : activeExtension === ".tex"
                    ? "latex"
                    : "python"
              }
              ref={editor}
              value={code}
              onChange={(value) => {
                if (!cannotEdit) setCode(value);
              }}
              readOnly={cannotEdit || !ready || fileBusy}
              fullScriptOnly={isolatedRuns}
              onRun={(source) => void execute(source, activeScript.id)}
              onCursor={(line, column) => setCursor([line, column])}
              inlineSuggestionProvider={inlineProvider}
              inlineSuggestionContextKey={
                inlineAllowed ? inlineContext : undefined
              }
            />
          </section>
          <section className="output-pane" aria-label="Python results">
            <div
              className="output-tabs"
              role="tablist"
              aria-label="Output type"
            >
              <button
                role="tab"
                aria-selected={outputTab === "results"}
                className={outputTab === "results" ? "active" : ""}
                onClick={() => setOutputTab("results")}
              >
                Results
              </button>
              <button
                role="tab"
                aria-selected={outputTab === "plots"}
                className={outputTab === "plots" ? "active" : ""}
                onClick={() => setOutputTab("plots")}
              >
                Charts{plots.length > 0 && <span>{plots.length}</span>}
              </button>
              <button
                role="tab"
                aria-selected={outputTab === "log"}
                className={outputTab === "log" ? "active" : ""}
                onClick={() => setOutputTab("log")}
              >
                Log
              </button>
              <div className="output-tools">
                <button
                  className="icon-button"
                  disabled={!history.length}
                  aria-label="Download execution log"
                  title="Download log"
                  onClick={() =>
                    download(
                      history
                        .map(
                          (r) =>
                            `# ${r.id} · ${r.status}\n${r.code}\n${r.stdout}\n${r.error?.traceback || ""}\n`,
                        )
                        .join("\n"),
                      "openecon-session.txt",
                      "text/plain",
                    )
                  }
                >
                  <Symbol name="download" size={15} />
                </button>
              </div>
            </div>
            {selected && outputTab !== "log" && visibleTimeline.length > 0 && (
              <div
                className="result-toolbar"
                aria-label="Results view and export"
              >
                <div className="result-format" role="group" aria-label="View">
                  <button
                    aria-pressed={outputFormat === "result"}
                    className={outputFormat === "result" ? "active" : ""}
                    onClick={() => setOutputFormat("result")}
                  >
                    Results
                  </button>
                  <button
                    aria-pressed={outputFormat === "latex"}
                    className={outputFormat === "latex" ? "active" : ""}
                    onClick={() => setOutputFormat("latex")}
                  >
                    LaTeX
                  </button>
                </div>
                {availableOutputs.length > 1 && (
                  <select
                    className="result-output-select"
                    aria-label="Output to display"
                    value={outputChoice}
                    onChange={(event) => setOutputChoice(event.target.value)}
                  >
                    <option value="all">All outputs</option>
                    {availableOutputs.map((output, index) => (
                      <option key={index} value={index}>
                        {outputLabel(output, index)}
                      </option>
                    ))}
                  </select>
                )}
                <div className="result-export-actions">
                  <button
                    className="subtle-button"
                    onClick={() => void copyLatex()}
                    title="Copy LaTeX source"
                    aria-label={
                      latexCopied ? "LaTeX copied" : "Copy LaTeX source"
                    }
                  >
                    {latexCopied ? "Copied" : "Copy"}
                  </button>
                  <button
                    className="subtle-button"
                    onClick={() =>
                      download(
                        latexSource,
                        "results.tex",
                        "application/x-tex;charset=utf-8",
                      )
                    }
                    title="Download LaTeX file"
                    aria-label="Download LaTeX file"
                  >
                    .tex ↓
                  </button>
                </div>
              </div>
            )}
            <div className="output-scroll" aria-live="polite">
              <ResultSharingPanel
                client={client}
                resultId={selected?.id}
                readOnly={cannotEdit}
                ready={ready}
                partial={selected?.outputs.some((output) => output.type === "plot" &&
                  !!output.data && typeof output.data === "object" && "artifact" in output.data)}
                onSnapshot={setSharingSnapshot}
                onOpenHistory={() => setShowHistory(true)}
                onAccessDenied={() => {
                  setAccessLost(true);
                  setServerReadOnly(true);
                  draftSaver.current?.dispose();
                  setSaveState("Read-only");
                }}
              />
              {latexCopyError && (
                <p className="output-note" role="alert">
                  Could not copy. Select the LaTeX source to copy it manually.
                </p>
              )}
              {historyError && (
                <div className="error-banner" role="alert">
                  <span>Could not load history. {historyError}</span>
                  <button
                    className="subtle-button"
                    disabled={historyLoading}
                    onClick={() => {
                      if (accessLost) window.location.reload();
                      else {
                        setHistoryLoading(true);
                        void refreshState().catch(() => {});
                      }
                    }}
                  >
                    {accessLost ? "Reload page" : "Try again"}
                  </button>
                </div>
              )}
              {running && (
                <div className="execution-status">
                  <span className="spinner" />
                  Running…
                </div>
              )}
              {!selected ? (
                historyError ? null : (
                  <div className="output-empty">
                    <p>
                      {historyLoading ? "Loading history…" : "No outputs yet."}
                    </p>
                  </div>
                )
              ) : (
                <>
                  <div className="execution-heading">
                    <span className={`execution-dot ${selected.status}`} />
                    <strong>
                      {selected.status === "ok"
                        ? "Completed"
                        : selected.status === "timeout"
                          ? "Time limit exceeded"
                          : selected.status === "interrupted"
                            ? "Execution stopped"
                            : "Python error"}
                    </strong>
                    {selected.source === "mcp" && <span>Agent</span>}
                    <span>{fmt(selected.duration_ms / 1000, 2)} s</span>
                    {selected.outputs.some((output) => output.type === "plot" && !!output.data && typeof output.data === "object" && "artifact" in output.data) && (
                      <span title="Large network plots stay in this local project. Export HTML to share the full interactive view.">Local graph</span>
                    )}
                    <button
                      className="subtle-button"
                      onClick={() => addSnippet(selected.code)}
                      title="Add this run’s code to the editor"
                      disabled={cannotEdit || !pythonFile}
                    >
                      Add code
                    </button>
                  </div>
                  {!isolatedRuns &&
                    selected.source !== "mcp" &&
                    selected.session_generation !== generation && (
                      <p className="output-note historical-note">
                        Output from a previous session. Variables may no longer
                        be in memory.
                      </p>
                    )}
                  {!!selected.artifacts?.length && (
                    <div className="execution-artifacts">
                      <span>Generated files</span>
                      {selected.artifacts.map((artifact, index) => (
                        <button
                          key={`${artifact.name}-${index}`}
                          onClick={() => void downloadArtifact(artifact)}
                        >
                          {artifact.name}{" "}
                          <small>
                            {Math.ceil(artifact.size_bytes / 1024)} KB
                          </small>
                          <span>↓</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {selected.state_reset &&
                    (!isolatedRuns || selected.status !== "ok") && (
                      <p className="output-warning">
                        {isolatedRuns
                          ? "This run’s environment has closed. Run the entire script to execute it again."
                          : "The session restarted. Run the code again to recreate its variables."}
                      </p>
                    )}
                  {outputTab === "log" ? (
                    <>
                      <pre className="source-echo">
                        <span>
                          In [
                          {history.findIndex((r) => r.id === selected.id) + 1}]
                        </span>
                        {selected.code}
                      </pre>
                      {selected.stdout && (
                        <pre className="text-output">{selected.stdout}</pre>
                      )}
                      {selected.error && (
                        <pre className="traceback" role="alert">
                          {selected.error.traceback ||
                            `${selected.error.type}: ${selected.error.message}`}
                        </pre>
                      )}
                      {!selected.stdout && !selected.error && (
                        <p className="output-note">
                          Execution succeeded. Tables and charts are available
                          in their tabs.
                        </p>
                      )}
                    </>
                  ) : outputFormat === "latex" && visibleTimeline.length > 0 ? (
                    <>
                      <pre className="latex-source" aria-label="LaTeX source">
                        {latexSource}
                      </pre>
                      {outputTab === "results" && selected.error && (
                        <pre className="traceback" role="alert">
                          {selected.error.traceback || selected.error.message}
                        </pre>
                      )}
                    </>
                  ) : outputTab === "plots" ? (
                    plots.length ? (
                      visibleOutputs.map((o, i) => (
                        <Output key={i} output={o} client={client} />
                      ))
                    ) : (
                      <div className="panel-empty">
                        <p>No charts.</p>
                      </div>
                    )
                  ) : (
                    <>
                      {visibleTimeline.map((event, index) =>
                        event.type === "stdout" ? (
                          <pre key={`stdout-${index}`} className="text-output">
                            {event.text}
                          </pre>
                        ) : (
                          <Output
                            key={`output-${event.index}`}
                            output={event.output}
                            client={client}
                          />
                        ),
                      )}
                      {selected.error && (
                        <pre className="traceback" role="alert">
                          {selected.error.traceback || selected.error.message}
                        </pre>
                      )}
                      {!currentOutputs.length &&
                        !selected.stdout &&
                        !selected.error && (
                          <div className="panel-empty">
                            <p>
                              {isolatedRuns
                                ? "No output. Use display(...) to show a result."
                                : "No output."}
                            </p>
                          </div>
                        )}
                    </>
                  )}
                </>
              )}
            </div>
            {!isolatedRuns && (
              <WorkspaceTerminal
                ref={terminal}
                projectId={client.projectId}
                history={history}
                activeCommand={activeCommand}
                running={running}
                readOnly={!ready || cannotEdit || fileBusy || uploading}
                onRun={executeTerminal}
                onStop={() => void interrupt()}
              />
            )}
          </section>
        </WorkbenchPanels>
      </WorkspaceLayout>
      <footer className="status-bar">
        <span role="status">
          {running ? "Running…" : ready ? "Ready" : "Connecting…"}
        </span>
        <span
          title={
            isolatedRuns
              ? "Each run starts in a fresh Python environment."
              : "Python session"
          }
        >
          Line {cursor[0]}, Column {cursor[1]}
        </span>
      </footer>
      {showReset && (
        <Dialog
          title="Reset Python session"
          onClose={() => setShowReset(false)}
        >
          <p>
            Session variables will be cleared. Your code file and previous
            outputs will be preserved.
          </p>
          <div className="dialog-actions">
            <button
              className="subtle-button"
              onClick={() => setShowReset(false)}
            >
              Cancel
            </button>
            <button className="run-button" onClick={() => void reset()}>
              Reset session
            </button>
          </div>
        </Dialog>
      )}
      {showVariables && (
        <Dialog
          title="Session variables"
          onClose={() => setShowVariables(false)}
        >
          <div className="variable-list">
            {variables.map((v) => (
              <button
                key={v.name}
                onClick={() => {
                  terminal.current?.setCommand(v.name);
                  setShowVariables(false);
                  setMobilePane("output");
                }}
              >
                <span className="variable-icon">x</span>
                <span>
                  <strong>{v.name}</strong>
                  <small>{v.type}</small>
                  <span className="variable-preview">{v.preview}</span>
                </span>
              </button>
            ))}
          </div>
          {!variables.length && <p>No variables yet.</p>}
          <div className="dialog-actions">
            <button
              className="subtle-button"
              disabled={running}
              onClick={() => {
                setShowVariables(false);
                setShowReset(true);
              }}
            >
              Reset session
            </button>
          </div>
        </Dialog>
      )}
      {showHistory && (
        <Dialog title="Execution history" onClose={() => setShowHistory(false)}>
          <ExecutionHistory client={client} history={history} localLoading={historyLoading} localError={historyError}
            sharingRecords={sharingRecords}
            onSelect={(id) => {
              setHistoricalRecord(null); setSelectedId(id); setOutputTab("results"); setMobilePane("output"); setShowHistory(false);
            }}
            onOpen={(record) => {
              setHistoricalRecord({ client, record }); setSelectedId(record.id);
              setOutputTab(record.status === "ok" && record.outputs.length ? "results" : "log");
              setMobilePane("output"); setShowHistory(false);
            }} />
        </Dialog>
      )}
      {showAgent && (
        <Dialog
          title="Connect an agent to the workspace"
          onClose={closeAgent}
        >
          {agentError && (
            <p role="alert" className="error-banner">
              {agentError}
            </p>
          )}
          {config?.mcp_available === false ? (
            <p>{config.notice}</p>
          ) : config ? (
            <>
              {[
                ["Codex", config.codex_command],
                ["Claude Code", config.claude_command],
              ].map(([name, cmd]) => (
                <div className="agent-config" key={name}>
                  <div>
                    <strong>{name}</strong>
                    <button
                      className="subtle-button"
                      onClick={() => void copy(cmd, name)}
                    >
                      {copied === name ? "Copied" : "Copy"}
                    </button>
                  </div>
                  <code>{cmd}</code>
                </div>
              ))}
              <p className="fine-print">
                MCP provides defined analysis tools for saved datasets. The
                editor’s Python session runs separately. Results sent to your
                agent client may be forwarded to your model provider.
              </p>
            </>
          ) : (
            <p>Preparing connection details…</p>
          )}
        </Dialog>
      )}
    </div>
  );
}
export default App;
