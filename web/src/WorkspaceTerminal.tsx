import {
  forwardRef,
  useEffect,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type { ExecutionRecord } from "./types";
import {
  ResizeSeparator,
  usePaneExtent,
  usePanePreference,
} from "./PaneResize";
import { terminalBounds } from "./pane-preferences";
import "./workspace-terminal.css";

export interface WorkspaceTerminalProps {
  projectId?: string | null;
  history: ExecutionRecord[];
  running: boolean;
  activeCommand?: string;
  readOnly?: boolean;
  onRun: (source: string) => void;
  onStop?: () => void;
}

export interface WorkspaceTerminalHandle {
  setCommand: (source: string) => void;
  focus: () => void;
}

const MAX_RUNS = 8;
const MAX_TEXT = 8000;
const MAX_COMMANDS = 50;

function preview(value: string, limit: number, tail = false): string {
  if (limit <= 0) return "";
  if (value.length <= limit) return value;
  if (limit === 1) return "…";
  let text = tail ? value.slice(-(limit - 1)) : value.slice(0, limit - 1);
  if (tail && /[\uDC00-\uDFFF]/.test(text[0])) text = text.slice(1);
  if (!tail && /[\uD800-\uDBFF]/.test(text.at(-1) ?? ""))
    text = text.slice(0, -1);
  return tail ? "…" + text : text + "…";
}

function transcript(history: ExecutionRecord[], activeCommand?: string) {
  const recent: {
    id: string;
    code: string;
    status: ExecutionRecord["status"] | "running";
    stdout: string;
    error: ExecutionRecord["error"];
  }[] = history.slice(-(activeCommand ? MAX_RUNS - 1 : MAX_RUNS));
  if (activeCommand) {
    recent.push({
      id: "pending",
      code: activeCommand,
      status: "running",
      stdout: "",
      error: null,
    });
  }
  const perRun = Math.floor(MAX_TEXT / Math.max(1, recent.length));
  return recent.map((record) => {
    const command = preview(record.code, Math.min(512, Math.floor(perRun / 3)));
    const remaining = perRun - command.length;
    const failure = record.error
      ? record.error.traceback ||
        `${preview(record.error.type, 128)}: ${preview(record.error.message, 4000)}`
      : record.status === "timeout"
        ? "Timed out."
        : record.status === "interrupted"
          ? "Stopped."
          : record.status === "error"
            ? "Could not complete the command."
            : "";
    const error = preview(failure, Math.floor(remaining / 2), true);
    return {
      id: record.id,
      status: record.status,
      command,
      stdout: preview(record.stdout, remaining - error.length),
      error,
    };
  });
}

function commandHistory(history: ExecutionRecord[], local: string[]): string[] {
  const candidates = [
    ...history.slice(-MAX_COMMANDS).map((record) => record.code),
    ...local,
  ];
  const commands = [];
  const seen = new Set<string>();
  let characters = 0;
  for (let index = candidates.length - 1; index >= 0; index--) {
    const code = candidates[index];
    if (
      code.length > MAX_TEXT ||
      code.includes("\n") ||
      !code.trim() ||
      seen.has(code)
    )
      continue;
    if (characters + code.length > MAX_TEXT || commands.length === MAX_COMMANDS)
      break;
    characters += code.length;
    seen.add(code);
    commands.push(code);
  }
  return commands.reverse();
}

const WorkspaceTerminal = forwardRef<
  WorkspaceTerminalHandle,
  WorkspaceTerminalProps
>(function WorkspaceTerminal(
  {
    projectId,
    history,
    running,
    activeCommand,
    readOnly = false,
    onRun,
    onStop,
  },
  ref,
) {
  const [command, setCommand] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [localCommands, setLocalCommands] = useState<string[]>([]);
  const terminalElement = useRef<HTMLElement>(null);
  const extent = usePaneExtent(terminalElement, "height", true);
  const bounds = terminalBounds(extent || 600);
  const size = usePanePreference(projectId, "terminal", bounds);
  const panelId = useId();
  const input = useRef<HTMLInputElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const followOutput = useRef(true);
  const draft = useRef("");
  const navigation = useRef<string[] | null>(null);
  const cursor = useRef(-1);
  const pendingCommand =
    running && activeCommand?.trim() ? activeCommand : undefined;
  const entries = useMemo(
    () => transcript(history, pendingCommand),
    [history, pendingCommand],
  );
  const commands = useMemo(
    () => commandHistory(history, localCommands),
    [history, localCommands],
  );

  useEffect(() => {
    if (expanded && followOutput.current && panel.current) {
      panel.current.scrollTop = panel.current.scrollHeight;
    }
  }, [entries, expanded]);

  useEffect(() => {
    if (pendingCommand) {
      followOutput.current = true;
      setExpanded(true);
    }
  }, [pendingCommand]);

  function resetNavigation() {
    cursor.current = -1;
    navigation.current = null;
  }

  useImperativeHandle(ref, () => ({
    setCommand: (source) => {
      setCommand(source);
      resetNavigation();
      input.current?.focus({ preventScroll: true });
    },
    focus: () => input.current?.focus({ preventScroll: true }),
  }));

  function submit() {
    if (running || readOnly || !command.trim()) return;
    onRun(command);
    setLocalCommands((previous) => commandHistory([], [...previous, command]));
    setCommand("");
    resetNavigation();
    followOutput.current = true;
    setExpanded(true);
    input.current?.focus({ preventScroll: true });
  }

  return (
    <section
      ref={terminalElement}
      className={`workspace-terminal${expanded ? " workspace-terminal--expanded" : ""}`}
      aria-label="Terminal"
      style={
        expanded
          ? ({ "--terminal-pane-height": `${size.value}px` } as CSSProperties)
          : undefined
      }
    >
      {expanded && (
        <ResizeSeparator
          label="Terminal panel height"
          orientation="horizontal"
          {...bounds}
          {...size}
          direction={-1}
          onChange={size.change}
          onCommit={size.commit}
          onCancel={size.cancel}
        />
      )}
      <div className="workspace-terminal-header">
        <div className="workspace-terminal-title">
          <button
            type="button"
            className="workspace-terminal-toggle"
            aria-expanded={expanded}
            aria-controls={panelId}
            onClick={() => {
              followOutput.current = true;
              setExpanded((value) => !value);
            }}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              aria-hidden="true"
            >
              <path
                d={expanded ? "M3 7.5 6 4.5 9 7.5" : "M3 4.5 6 7.5 9 4.5"}
              />
            </svg>
            Terminal
          </button>
          <span className="workspace-terminal-scope">Python · pip / uv</span>
        </div>
        <div className="workspace-terminal-actions">
          <span className="workspace-terminal-status" role="status">
            {running ? "Running…" : readOnly ? "Read only" : "Ready"}
          </span>
          {running && onStop && (
            <button
              type="button"
              className="workspace-terminal-stop"
              disabled={readOnly}
              onClick={onStop}
            >
              Stop
            </button>
          )}
        </div>
      </div>
      <div
        id={panelId}
        ref={panel}
        className="workspace-terminal-transcript"
        hidden={!expanded}
        role="log"
        aria-label="Command history"
        aria-live="off"
        tabIndex={0}
        onScroll={(event) => {
          const element = event.currentTarget;
          followOutput.current =
            element.scrollHeight - element.clientHeight - element.scrollTop <
            24;
        }}
      >
        {entries.map((entry) => (
          <div key={entry.id} className="workspace-terminal-entry">
            <div className="workspace-terminal-command">
              <span aria-hidden="true">›</span>
              <pre aria-label="Command">{entry.command}</pre>
              {entry.status !== "ok" && (
                <span className="workspace-terminal-result-state">
                  {entry.status === "running"
                    ? "Running…"
                    : entry.status === "error"
                      ? "Error"
                      : entry.status === "timeout"
                        ? "Timed out"
                        : "Stopped"}
                </span>
              )}
            </div>
            {entry.stdout && (
              <pre className="workspace-terminal-output">{entry.stdout}</pre>
            )}
            {entry.error && (
              <pre
                className="workspace-terminal-error"
                aria-label="Error output"
              >
                {entry.error}
              </pre>
            )}
          </div>
        ))}
      </div>
      <form
        className="workspace-terminal-input"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <span aria-hidden="true">›</span>
        <input
          ref={input}
          value={command}
          onChange={(event) => {
            setCommand(event.target.value);
            resetNavigation();
          }}
          disabled={running || readOnly}
          maxLength={64000}
          aria-label="Python, pip, or uv command"
          placeholder="pip install scikit-learn"
          spellCheck={false}
          autoComplete="off"
          autoCapitalize="off"
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              event.stopPropagation();
              if (!event.shiftKey && !event.nativeEvent.isComposing) submit();
            } else if (event.key === "ArrowUp") {
              if (!commands.length) return;
              event.preventDefault();
              if (cursor.current === -1) {
                draft.current = command;
                navigation.current = commands;
                cursor.current = commands.length;
              }
              cursor.current = Math.max(0, cursor.current - 1);
              setCommand(navigation.current![cursor.current]);
            } else if (event.key === "ArrowDown" && cursor.current >= 0) {
              event.preventDefault();
              cursor.current++;
              if (cursor.current >= navigation.current!.length) {
                setCommand(draft.current);
                resetNavigation();
              } else {
                setCommand(navigation.current![cursor.current]);
              }
            }
          }}
        />
        <button
          type="submit"
          aria-label="Run command"
          disabled={running || readOnly || !command.trim()}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            aria-hidden="true"
          >
            <path d="m4.5 3.5 5 3.5-5 3.5z" />
          </svg>
        </button>
      </form>
    </section>
  );
});

export default WorkspaceTerminal;
