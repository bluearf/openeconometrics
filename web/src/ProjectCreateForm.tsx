import { useId, useRef, useState } from "react";
import {
  PROJECT_CREATE_LIMITS,
  validateProjectDraft,
  type ProjectCreateInput,
} from "./project-create";

export default function ProjectCreateForm({
  name,
  description,
  busy,
  error,
  onName,
  onDescription,
  onCreate,
}: {
  name: string;
  description: string;
  busy: boolean;
  error: string;
  onName: (value: string) => void;
  onDescription: (value: string) => void;
  onCreate: (input: ProjectCreateInput) => void;
}) {
  const id = useId();
  const nameInput = useRef<HTMLInputElement>(null);
  const descriptionInput = useRef<HTMLTextAreaElement>(null);
  const [submitted, setSubmitted] = useState(false);
  const draft = validateProjectDraft(name, description);
  const nameError = submitted || name ? draft.errors.name : undefined;
  const descriptionError = draft.errors.description;
  return (
    <form
      className="team-modal-body"
      aria-label="New project"
      noValidate
      aria-busy={busy || undefined}
      onSubmit={(event) => {
        event.preventDefault();
        if (busy) return;
        setSubmitted(true);
        if (draft.errors.name) nameInput.current?.focus();
        else if (draft.errors.description) descriptionInput.current?.focus();
        else onCreate(draft.input);
      }}
    >
      <label>
        Project name
        <input
          ref={nameInput}
          autoFocus
          aria-label="Project name"
          value={name}
          onChange={(event) => onName(event.target.value)}
          disabled={busy}
          aria-required="true"
          aria-invalid={nameError ? true : undefined}
          aria-describedby={`${id}-name-hint${nameError ? ` ${id}-name-error` : ""}`}
        />
        <small id={`${id}-name-hint`}>
          {draft.lengths.name} / {PROJECT_CREATE_LIMITS.name} characters
        </small>
        {nameError && (
          <span
            className="team-field-error"
            id={`${id}-name-error`}
            role="alert"
          >
            {nameError}
          </span>
        )}
      </label>
      <label>
        Description <small>optional</small>
        <textarea
          ref={descriptionInput}
          aria-label="Description"
          value={description}
          rows={3}
          onChange={(event) => onDescription(event.target.value)}
          disabled={busy}
          aria-invalid={descriptionError ? true : undefined}
          aria-describedby={`${id}-description-hint${descriptionError ? ` ${id}-description-error` : ""}`}
        />
        <small id={`${id}-description-hint`}>
          {draft.lengths.description} / {PROJECT_CREATE_LIMITS.description}{" "}
          characters
        </small>
        {descriptionError && (
          <span
            className="team-field-error"
            id={`${id}-description-error`}
            role="alert"
          >
            {descriptionError}
          </span>
        )}
      </label>
      {error && (
        <div className="team-notice is-error" role="alert">
          {error}
        </div>
      )}
      <button className="team-primary" type="submit" disabled={busy}>
        {busy ? "Creating…" : "Create project"}
      </button>
    </form>
  );
}
