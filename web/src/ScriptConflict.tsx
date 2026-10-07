import { useEffect, useRef, useState } from "react";
import { download, type ScriptConflictSnapshot } from "./api";

export default function ScriptConflict({
  snapshots,
  initialCode,
  busy,
  error,
  onSave,
  onClose,
  onRefresh,
}: {
  snapshots: ScriptConflictSnapshot[];
  initialCode: string;
  busy: boolean;
  error: string;
  onSave: (code: string) => void;
  onClose: () => void;
  onRefresh: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [code, setCode] = useState(initialCode);
  const snapshot = snapshots.at(-1)!;
  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="script-conflict-dialog"
      aria-label="Compare file versions"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onClose();
      }}
    >
      <h2>Compare file versions</h2>
      <p>
        Autosave is paused. Edit the merged draft, then save against the
        displayed cloud version.
      </p>
      {error && <p role="alert">{error}</p>}
      <div className="script-conflict-sources">
        {(
          [
            ["Base", snapshot.base],
            ["Your local draft", initialCode],
            [
              "Cloud version " + snapshot.remote.version,
              snapshot.remote.code ?? "",
            ],
          ] as const
        ).map(([label, source]) => (
          <section key={label}>
            <h3>{label}</h3>
            <pre tabIndex={0} aria-label={label}>
              {source}
            </pre>
            <button disabled={busy} onClick={() => setCode(source)}>
              Use in merged draft
            </button>
          </section>
        ))}
      </div>
      <label>
        Merged draft
        <textarea
          aria-label="Merged draft"
          value={code}
          disabled={busy}
          onChange={(event) => setCode(event.target.value)}
          spellCheck={false}
        />
      </label>
      <p>{Array.from(code).length.toLocaleString()} / 64,000 characters</p>
      <div className="script-conflict-actions">
        <button
          disabled={busy || Array.from(code).length > 64000}
          onClick={() => onSave(code)}
        >
          Save merged draft
        </button>
        <button disabled={busy} onClick={onRefresh}>
          Read latest cloud version
        </button>
        <button
          onClick={() =>
            download(
              JSON.stringify(
                { snapshots, localCode: initialCode, mergedCode: code },
                null,
                2,
              ),
              "file-conflict-copies.json",
            )
          }
        >
          Download all copies
        </button>
        <button disabled={busy} onClick={onClose}>
          Cancel · keep local draft
        </button>
      </div>
      {snapshots.length > 1 && (
        <details>
          <summary>Earlier cloud copies ({snapshots.length - 1})</summary>
          {snapshots.slice(0, -1).map((item, index) => (
            <pre key={index}>{item.remote.code}</pre>
          ))}
        </details>
      )}
    </dialog>
  );
}
