import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { WorkbenchControls } from "./App";
import type { WorkspaceClient } from "./api";
import { createLocalProjects, type LocalProject } from "./local-projects";
import { qaShell, qaStart } from "./qa-performance";
const App = lazy(() => import("./App"));

export default function LocalProjects({ onTeams }: { onTeams: () => void }) {
  const catalog = useMemo(createLocalProjects, []);
  const [projects, setProjects] = useState<LocalProject[]>([]);
  const [selected, setSelected] = useState<{
    project: LocalProject;
    client: WorkspaceClient;
  } | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const controls = useRef<WorkbenchControls | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    void catalog
      .list()
      .then((value) => {
        if (alive.current) {
          setProjects(value.projects);
          qaShell();
        }
      })
      .catch((failure) => {
        if (alive.current) setError(String(failure.message ?? failure));
      });
    return () => {
      alive.current = false;
    };
  }, [catalog]);
  useEffect(() => () => selected?.client.cancelPending(), [selected]);
  async function operate(action: () => Promise<void>) {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (failure) {
      if (alive.current)
        setError(
          failure instanceof Error
            ? failure.message
            : "Could not open the project.",
        );
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function open(project: LocalProject) {
    qaStart("project");
    await controls.current?.prepareToLeave();
    const next = await catalog.open(project.id);
    if (alive.current) setSelected(next);
  }
  return (
    <div className={selected ? "team-workspace" : "team-home"}>
      <header className="team-topbar">
        <div className="team-topbar-left">
          <strong className="team-brand">OpenEconometrics</strong>
          <span>{selected?.project.name ?? "Local projects"}</span>
        </div>
        <div className="team-topbar-right">
          {selected && (
            <button
              className="team-secondary"
              disabled={busy}
              onClick={() =>
                void operate(async () => {
                  await controls.current?.prepareToLeave();
                  setSelected(null);
                })
              }
            >
              ← Local projects
            </button>
          )}
          {!selected && (
            <button
              className="team-secondary"
              disabled={busy}
              onClick={onTeams}
            >
              Team projects · sign in
            </button>
          )}
        </div>
      </header>
      {error && (
        <p role="alert" className="error-banner">
          {error}
        </p>
      )}
      {selected ? (
        <Suspense fallback={<p role="status">Opening local workspace…</p>}>
          <App
            key={selected.project.id}
            client={selected.client}
            projectName={selected.project.name}
            onControls={(value) => {
              controls.current = value;
            }}
          />
        </Suspense>
      ) : (
        <main className="team-dashboard">
          <h1>Local projects</h1>
          <p>
            Saved on this computer. No account or internet connection required.
          </p>
          <form
            className="local-project-form"
            onSubmit={(event) => {
              event.preventDefault();
              void operate(async () => {
                const project = await catalog.create(name, "");
                if (!alive.current) return;
                setProjects((items) => [...items, project]);
                setName("");
                await open(project);
              });
            }}
          >
            <label>
              Project name{" "}
              <input
                aria-label="Local project name"
                maxLength={120}
                required
                value={name}
                disabled={busy}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <button className="team-primary" disabled={busy || !name.trim()}>
              Create local project
            </button>
          </form>
          <div className="team-project-grid">
            {projects.map((project) => (
              <article className="team-project-card" key={project.id}>
                <button
                  className="team-project-open"
                  disabled={busy}
                  onClick={() => void operate(() => open(project))}
                >
                  <h2>{project.name}</h2>
                  <span>Local</span>
                </button>
              </article>
            ))}
          </div>
          <p>
            To share work, choose Team projects and explicitly create or open a
            team project, then import your files.
          </p>
        </main>
      )}
    </div>
  );
}
