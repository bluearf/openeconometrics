import { useEffect, useId, useRef, useState } from "react";
import type { Project } from "./team-api";
import "./project-name-editor.css";

export interface ProjectNameEditorProps {
  project: Project;
  onRename: (project: Project, name: string) => Promise<void>;
  disabled?: boolean;
  showName?: boolean;
}

function EditIcon({ kind }: { kind: "edit" | "save" | "cancel" }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {kind === "edit" && (
        <path d="m15 4 5 5M4 20l5-1 11-11a2 2 0 0 0-5-5L4 14z" />
      )}
      {kind === "save" && <path d="m5 12 4 4L19 6" />}
      {kind === "cancel" && <path d="m6 6 12 12M6 18 18 6" />}
    </svg>
  );
}

export default function ProjectNameEditor({
  project,
  onRename,
  disabled = false,
  showName = false,
}: ProjectNameEditorProps) {
  const errorId = useId();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const mounted = useRef(true);
  const request = useRef<symbol | null>(null);
  const editGeneration = useRef(0);
  const owner = project.role === "owner";
  const context = useRef({ id: project.id, owner, generation: 0 });
  if (context.current.id !== project.id || context.current.owner !== owner)
    context.current = {
      id: project.id,
      owner,
      generation: context.current.generation + 1,
    };
  const blocked = disabled || pending;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      editGeneration.current += 1;
    };
  }, []);
  useEffect(() => {
    editGeneration.current += 1;
    request.current = null;
    setPending(false);
    setEditing(false);
    setDraft("");
    setError("");
  }, [project.id, owner]);
  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  function cancel() {
    editGeneration.current += 1;
    setEditing(false);
    setError("");
  }

  async function save() {
    if (disabled || !owner || request.current) return;
    const name = draft.trim();
    const length = Array.from(name).length;
    if (length < 1 || length > 100) {
      setError("The project name must be 1–100 characters.");
      return;
    }
    if (/[\p{Cc}\p{Cs}\u2028\u2029]/u.test(draft)) {
      setError("The project name cannot contain control characters.");
      return;
    }
    setError("");
    const current = Symbol("project-name-save");
    const submitted = {
      draft,
      context: context.current.generation,
      edit: editGeneration.current,
    };
    request.current = current;
    setPending(true);
    try {
      const accepted = onRename(project, name);
      setEditing(false);
      await accepted;
    } catch (reason) {
      if (
        !mounted.current ||
        !context.current.owner ||
        submitted.context !== context.current.generation ||
        submitted.edit !== editGeneration.current ||
        request.current !== current
      )
        return;
      setDraft(submitted.draft);
      setEditing(true);
      setError(
        reason instanceof Error ? reason.message : "Could not save the project name.",
      );
    } finally {
      if (request.current === current) {
        request.current = null;
        if (mounted.current) setPending(false);
      }
    }
  }

  if (!owner) return showName ? <strong>{project.name}</strong> : null;
  return (
    <div
      className={`project-name-editor${showName ? " project-name-editor--name" : ""}`}
      aria-busy={pending || undefined}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      {editing ? (
        <form
          className="project-name-editor__form"
          aria-label="Rename project"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <input
            ref={input}
            autoFocus
            aria-label="Project name"
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? errorId : undefined}
            value={draft}
            maxLength={200}
            disabled={blocked}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !blocked) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
              if (event.key === "Escape" && !pending) {
                event.preventDefault();
                cancel();
              }
            }}
          />
          <button
            type="submit"
            className="project-name-editor__button"
            aria-label="Save project name"
            disabled={blocked}
          >
            <EditIcon kind="save" />
          </button>
          <button
            type="button"
            className="project-name-editor__button"
            aria-label="Cancel"
            disabled={pending}
            onClick={cancel}
          >
            <EditIcon kind="cancel" />
          </button>
        </form>
      ) : (
        <button
          type="button"
          className={
            showName
              ? "project-name-editor__name"
              : "project-name-editor__button"
          }
          aria-label="Rename project"
          disabled={blocked}
          onClick={() => {
            if (request.current) return;
            editGeneration.current += 1;
            setDraft(project.name);
            setError("");
            setEditing(true);
          }}
        >
          {showName ? (
            <strong>{project.name}</strong>
          ) : (
            <EditIcon kind="edit" />
          )}
        </button>
      )}
      {error && (
        <span id={errorId} className="project-name-editor__error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
