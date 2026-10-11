import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { getApps, initializeApp } from "firebase/app";
import {
  getAuth,
  initializeAuth,
  indexedDBLocalPersistence,
  browserLocalPersistence,
  browserSessionPersistence,
  getIdToken,
  onIdTokenChanged,
  signOut,
  type Auth,
  type User,
} from "firebase/auth";
import type { WorkbenchControls } from "./App";
import { ApiError, createWorkspaceClient } from "./api";
import {
  canEditProject,
  createTeamApi,
  loadAuthConfig,
  projectUrl,
  teamRoute,
  type Invitation,
  type Member,
  type Project,
  type TeamAuthConfig,
  type TeamProfile,
} from "./team-api";
import { googleAuthError, startGoogleSignIn } from "./google-auth";
import { isDesktop, isTransientDesktopFailure } from "./desktop";
import { createDesktopWorkspaceClient } from "./desktop-sync";
import {
  cacheDesktopProfile,
  cachedDesktopProfile,
  forgetDesktopProfile,
} from "./desktop-cache";
import ProjectNameEditor from "./ProjectNameEditor";
import ProjectCreateForm from "./ProjectCreateForm";
import type { ProjectCreateInput } from "./project-create";
import { ProjectNames } from "./project-name";
import "./team-styles.css";
import LocalProjects from "./LocalProjects";
import { EmailCodeSession } from "./email-code";
import EmailCodeForm from "./EmailCodeForm";

const App = lazy(() => import("./App"));

function WorkbenchLoading() {
  return (
    <div className="team-loading" role="status">
      Opening workspace…
    </div>
  );
}

const roleNames = {
  owner: "Owner",
  editor: "Editor",
  viewer: "Viewer",
};
const authErrors: Record<string, string> = {
  "auth/invalid-email": "Enter a valid email address.",
  "auth/too-many-requests": "Too many attempts. Wait a moment and try again.",
  "auth/network-request-failed":
    "Could not connect. Check your internet connection.",
};
function message(error: unknown) {
  const code =
    error && typeof error === "object" && "code" in error
      ? String(error.code)
      : "";
  return (
    googleAuthError(error) ||
    authErrors[code] ||
    (error instanceof Error
      ? error.message
      : "The action could not be completed.")
  );
}
function Brand() {
  return <span className="team-brand">OpenEconometrics</span>;
}

function Notice({
  children,
  error = false,
}: {
  children: ReactNode;
  error?: boolean;
}) {
  return children ? (
    <div
      className={`team-notice ${error ? "is-error" : ""}`}
      role={error ? "alert" : "status"}
    >
      {children}
    </div>
  ) : null;
}
function Modal({
  title,
  children,
  close,
}: {
  title: string;
  children: ReactNode;
  close: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      className="team-dialog"
      ref={ref}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      aria-label={title}
    >
      <header>
        <h2>{title}</h2>
        <button className="team-icon-button" onClick={close} aria-label="Close">
          ×
        </button>
      </header>
      {children}
    </dialog>
  );
}

export default function ApplicationRoot() {
  const [teams, setTeams] = useState(
    () =>
      !isDesktop() ||
      Boolean(new URLSearchParams(window.location.search).get("project")),
  );
  if (!teams) return <LocalProjects onTeams={() => setTeams(true)} />;
  return (
    <ConnectedApplicationRoot
      onLocal={isDesktop() ? () => setTeams(false) : undefined}
    />
  );
}

function ConnectedApplicationRoot({ onLocal }: { onLocal?: () => void }) {
  const [config, setConfig] = useState<TeamAuthConfig | "local" | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    (async () => {
      try {
        setConfig(await loadAuthConfig(controller.signal));
      } catch (failure) {
        if (!controller.signal.aborted) setError(message(failure));
      }
    })();
    return () => controller.abort();
  }, [attempt]);
  if (config === "local")
    return (
      <Suspense fallback={<WorkbenchLoading />}>
        <App />
      </Suspense>
    );
  if (config) return <TeamShell config={config} onLocal={onLocal} />;
  return (
    <div className="team-loading">
      <Brand />
      <p>{error || "Opening workspace…"}</p>
      {onLocal && (
        <button className="team-secondary" onClick={onLocal}>
          ← Local projects
        </button>
      )}
      {error && (
        <button
          className="team-primary"
          onClick={() => setAttempt((x) => x + 1)}
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function AuthScreen({
  auth,
  invitation,
  emailCodeAvailable = false,
  onLocal,
}: {
  auth: Auth;
  invitation: boolean;
  emailCodeAvailable?: boolean;
  onLocal?: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [googleBusy, setGoogleBusy] = useState(false);
  const [emailBusy, setEmailBusy] = useState(false);
  const pending = useRef(false);
  const googleAttempt = useRef<AbortController | null>(null);
  const googleCommitting = useRef(false);
  const emailSession = useMemo(() => new EmailCodeSession(auth), [auth]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  function cancelGoogleSignIn() {
    const attempt = googleAttempt.current;
    if (!attempt || googleCommitting.current) return;
    googleAttempt.current = null;
    attempt.abort();
    pending.current = false;
    setBusy(false);
    setGoogleBusy(false);
    setNotice(
      "Google sign-in cancelled. Choose a sign-in method when you are ready.",
    );
  }
  useEffect(() => {
    const leaving = () => {
      googleAttempt.current?.abort();
      emailSession.cancel();
    };
    window.addEventListener("pagehide", leaving);
    window.addEventListener("beforeunload", leaving);
    return () => {
      leaving();
      googleAttempt.current = null;
      window.removeEventListener("pagehide", leaving);
      window.removeEventListener("beforeunload", leaving);
    };
  }, [emailSession]);
  async function googleSignIn() {
    if (pending.current) return;
    const attempt = new AbortController();
    googleAttempt.current = attempt;
    googleCommitting.current = false;
    pending.current = true;
    setBusy(true);
    setGoogleBusy(true);
    setError("");
    setNotice("");
    try {
      // Keep the SDK call in this user gesture: do not fetch or await beforehand.
      await startGoogleSignIn(
        auth,
        undefined,
        (text) => {
          if (googleAttempt.current === attempt && !attempt.signal.aborted)
            setNotice(text);
        },
        {
          signal: attempt.signal,
          onCommit: () => {
            if (googleAttempt.current === attempt) {
              googleCommitting.current = true;
              setGoogleBusy(false);
            }
          },
        },
      );
    } catch (failure) {
      if (googleAttempt.current === attempt && !attempt.signal.aborted)
        setError(message(failure));
    } finally {
      if (googleAttempt.current === attempt) {
        googleAttempt.current = null;
        pending.current = false;
        setBusy(false);
        setGoogleBusy(false);
      }
    }
  }
  return (
    <main className="team-auth">
      {onLocal && (
        <button className="team-back" onClick={onLocal}>
          ← Local projects
        </button>
      )}
      <section className="team-auth-panel">
        <div className="team-auth-card">
          <Brand />
          <h1>Sign in</h1>
          {invitation && <p>Sign in with the invited email address.</p>}
          <Notice error>{error}</Notice>
          <Notice>{notice}</Notice>
          <button
            type="button"
            className="team-google-button"
            disabled={busy}
            aria-busy={googleBusy}
            onClick={() => void googleSignIn()}
          >
            <span>
              {googleBusy
                ? "Waiting for Google sign-in…"
                : "Continue with Google"}
            </span>
          </button>
          {googleBusy && isDesktop() && (
            <button type="button" onClick={cancelGoogleSignIn}>
              Cancel Google sign-in
            </button>
          )}
          <div className="team-auth-divider">
            <span>or</span>
          </div>
          <EmailCodeForm
            session={emailSession}
            available={emailCodeAvailable}
            disabled={busy && !emailBusy}
            onBusy={(value) => {
              setEmailBusy(value);
              pending.current = value;
              setBusy(value);
            }}
          />
        </div>
      </section>
    </main>
  );
}

function TeamShell({
  config,
  onLocal,
}: {
  config: TeamAuthConfig;
  onLocal?: () => void;
}) {
  const auth = useMemo(() => {
    const name = `openecon-${config.firebase.projectId}`;
    const existing = getApps().find((item) => item.name === name);
    // Popup support is supplied only when the user chooses Google sign-in.
    const value = existing
      ? getAuth(existing)
      : initializeAuth(initializeApp(config.firebase, name), {
          persistence: [
            indexedDBLocalPersistence,
            browserLocalPersistence,
            browserSessionPersistence,
          ],
        });
    value.languageCode = "tr";
    return value;
  }, [config]);
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [revision, setRevision] = useState(0);
  const [profile, setProfile] = useState<TeamProfile | null>(null);
  const profileSnapshot = useRef(profile);
  profileSnapshot.current = profile;
  const projectNames = useRef(new ProjectNames());
  const [, updateProjectNames] = useState(0);
  const [route, setRoute] = useState(() => teamRoute(window.location.href));
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [membersOpen, setMembersOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [description, setDescription] = useState("");
  const [leaveFailure, setLeaveFailure] = useState<{
    message: string;
    action: () => void | Promise<void>;
  } | null>(null);
  const controls = useRef<WorkbenchControls | null>(null);
  const profileSequence = useRef(0);
  const [toolbarTarget, setToolbarTarget] = useState<HTMLDivElement | null>(
    null,
  );
  const registerControls = useCallback((value: WorkbenchControls | null) => {
    controls.current = value;
  }, []);
  useEffect(
    () =>
      onIdTokenChanged(
        auth,
        (next) => {
          setUser(next);
          setAuthReady(true);
          setRevision((x) => x + 1);
        },
        (failure) => {
          setError(message(failure));
          setAuthReady(true);
        },
      ),
    [auth],
  );
  const uid = user?.uid;
  projectNames.current.select(uid || null);
  useEffect(() => {
    projectNames.current.select(uid || null);
    return () => projectNames.current.select(null);
  }, [uid]);
  const getToken = useCallback(
    async (refresh = false) => {
      const current = auth.currentUser;
      if (!current || current.uid !== uid)
        throw new ApiError(
          "Your session changed. Sign in again.",
          401,
          "AUTH_REQUIRED",
        );
      return getIdToken(current, refresh);
    },
    [auth, uid],
  );
  const api = useMemo(() => createTeamApi(getToken), [getToken]);
  const refreshProfile = useCallback(async () => {
    if (!uid) return;
    const sequence = ++profileSequence.current;
    setLoading(true);
    try {
      const next = await api<TeamProfile>("/me");
      if (
        sequence === profileSequence.current &&
        auth.currentUser?.uid === uid
      ) {
        const canonical = projectNames.current.reconcile(next);
        profileSnapshot.current = canonical;
        setProfile(canonical);
        if (isDesktop()) cacheDesktopProfile(canonical);
        setError("");
      }
    } catch (failure) {
      if (sequence === profileSequence.current) {
        const transient = isTransientDesktopFailure(failure);
        const cached =
          isDesktop() && transient ? cachedDesktopProfile(uid) : null;
        if (cached) {
          const canonical = projectNames.current.reconcile(cached);
          profileSnapshot.current = canonical;
          setProfile(canonical);
        } else if (!transient) {
          profileSnapshot.current = null;
          setProfile(null);
          if (isDesktop()) forgetDesktopProfile(uid);
        }
        setError(cached ? "Offline" : message(failure));
      }
    } finally {
      if (sequence === profileSequence.current) setLoading(false);
    }
  }, [api, auth, uid]);
  useEffect(() => {
    if (!uid) {
      ++profileSequence.current;
      profileSnapshot.current = null;
      setProfile(null);
      setLoading(false);
      return;
    }
    void refreshProfile();
    return () => {
      ++profileSequence.current;
    };
  }, [uid, revision, refreshProfile]);
  const currentProfile =
    profile && profile.user.uid === uid
      ? projectNames.current.view(profile)
      : null;
  const selected = currentProfile?.projects.find(
    (project) => project.id === route.projectId,
  );
  const client = useMemo(
    () =>
      selected && uid
        ? isDesktop()
          ? createDesktopWorkspaceClient(
              selected.id,
              getToken,
              selected.role,
              uid,
            )
          : createWorkspaceClient(selected.id, getToken)
        : null,
    [selected?.id, selected?.role, getToken, uid],
  );
  async function leave(action: () => void | Promise<void>) {
    try {
      await controls.current?.prepareToLeave();
      await action();
    } catch (failure) {
      setLeaveFailure({ message: message(failure), action });
    }
  }
  const leaveRef = useRef(leave);
  leaveRef.current = leave;
  useEffect(() => {
    function pop() {
      const next = teamRoute(window.location.href);
      window.history.replaceState(
        null,
        "",
        route.projectId
          ? projectUrl(route.projectId)
          : route.invitationId
            ? `/?invite=${encodeURIComponent(route.invitationId)}`
            : "/",
      );
      void leaveRef.current(() => {
        window.history.replaceState(
          null,
          "",
          next.projectId
            ? projectUrl(next.projectId)
            : next.invitationId
              ? `/?invite=${encodeURIComponent(next.invitationId)}`
              : "/",
        );
        setRoute(next);
      });
    }
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, [route]);
  function navigate(projectId: string | null) {
    window.history.pushState(null, "", projectUrl(projectId));
    setRoute({ projectId, invitationId: null });
    setMembersOpen(false);
    setError("");
  }
  async function logout() {
    await leave(() => signOut(auth));
  }
  async function accept(invitationId: string) {
    setBusy(true);
    setError("");
    try {
      await api(`/invitations/${encodeURIComponent(invitationId)}/accept`, {
        method: "POST",
      });
      await refreshProfile();
      navigate(null);
      setNotice("Invitation accepted. You can now open the project.");
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }
  async function createProject(input: ProjectCreateInput) {
    setBusy(true);
    setError("");
    try {
      const project = await api<Project>("/projects", {
        method: "POST",
        body: JSON.stringify(input),
      });
      await refreshProfile();
      setCreateOpen(false);
      setProjectName("");
      setDescription("");
      navigate(project.id);
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }
  async function renameProject(project: Project, name: string) {
    const change = projectNames.current.begin(project, name);
    updateProjectNames((value) => value + 1);
    try {
      const saved = await api<Project>(
        `/projects/${encodeURIComponent(project.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            name: change.name,
            name_version: change.name_version,
          }),
        },
      );
      if (
        auth.currentUser?.uid !== change.account ||
        !projectNames.current.accept(change, saved)
      )
        return;
      const current = profileSnapshot.current;
      if (current?.user.uid === change.account) {
        const canonical = projectNames.current.reconcile(current);
        profileSnapshot.current = canonical;
        setProfile(canonical);
        if (isDesktop()) cacheDesktopProfile(canonical);
      }
      updateProjectNames((value) => value + 1);
    } catch (failure) {
      if (
        !projectNames.current.current(change) ||
        auth.currentUser?.uid !== change.account
      )
        return;
      projectNames.current.cancel(change);
      updateProjectNames((value) => value + 1);
      setError(message(failure));
      void refreshProfile();
      throw failure;
    }
  }
  if (!authReady)
    return (
      <div className="team-loading">
        <Brand />
        <p>Checking session…</p>
      </div>
    );
  if (!user || !user.emailVerified)
    return (
      <AuthScreen
        auth={auth}
        invitation={Boolean(route.invitationId)}
        emailCodeAvailable={config.email_code_available === true}
        onLocal={onLocal}
      />
    );
  const verified =
    user.emailVerified && currentProfile?.user.email_verified !== false;
  const invitations = currentProfile?.invitations || [];
  const directInvitation =
    route.invitationId &&
    !invitations.some((item) => item.id === route.invitationId);
  const desktopLogin = new URL(window.location.href).searchParams.get(
    "desktop_login",
  );
  if (desktopLogin && /^[0-9a-f]{32}$/.test(desktopLogin) && verified)
    return (
      <DesktopLoginApproval
        requestId={desktopLogin}
        auth={auth}
        user={user}
        api={api}
      />
    );
  return (
    <div className={selected && verified ? "team-workspace" : "team-home"}>
      <header className="team-topbar">
        <div className="team-topbar-left">
          {selected && verified ? (
            <>
              <button
                className="team-back"
                onClick={() => void leave(() => navigate(null))}
              >
                ← <span>Projects</span>
              </button>
              <span className="team-topbar-divider" />
              <ProjectNameEditor
                key={`${uid}:${selected.id}`}
                project={selected}
                onRename={renameProject}
                disabled={projectNames.current.busy(selected.id)}
                showName
              />
            </>
          ) : (
            <Brand />
          )}
        </div>
        {selected && verified && (
          <div className="team-workspace-toolbar" ref={setToolbarTarget} />
        )}
        <div className="team-topbar-right">
          {!selected && onLocal && (
            <button
              className="team-secondary"
              onClick={() => void leave(onLocal)}
            >
              Local projects
            </button>
          )}
          {selected && verified && (
            <button
              className="team-secondary"
              onClick={() => setMembersOpen(true)}
            >
              Members
            </button>
          )}
          <span className="team-user-email" title={user.email || ""}>
            {user.displayName || user.email}
          </span>
          <button className="team-logout" onClick={() => void logout()}>
            Sign out
          </button>
        </div>
      </header>
      {!verified ? (
        <main className="team-gate">
          <h1>Verify your email</h1>
          <p>Sign in with Google or request an email sign-in code.</p>
          <Notice error>{error}</Notice>
          <button className="team-primary" onClick={() => void logout()}>
            Continue to sign in
          </button>
        </main>
      ) : selected && client ? (
        <Suspense fallback={<WorkbenchLoading />}>
          <App
            key={`${user.uid}:${selected.id}`}
            client={client}
            readOnly={!canEditProject(selected.role)}
            syncOnly={!isDesktop()}
            projectName={selected.name}
            toolbarTarget={toolbarTarget}
            onControls={registerControls}
          />
        </Suspense>
      ) : (
        <main className="team-dashboard">
          <div className="team-dashboard-heading">
            <h1>Projects</h1>
            {currentProfile?.enabled && (
              <button
                className="team-primary"
                onClick={() => setCreateOpen(true)}
              >
                New project
              </button>
            )}
          </div>
          <Notice error>{error}</Notice>
          <Notice>{notice}</Notice>
          {route.projectId && currentProfile && !selected && (
            <Notice error>
              This project was not found, or you do not have access. Ask the
              project owner for an invitation.
            </Notice>
          )}
          {loading && !currentProfile && (
            <p className="team-muted" role="status">
              Loading your projects…
            </p>
          )}
          {!currentProfile && !loading && (
            <button
              className="team-secondary"
              onClick={() => void refreshProfile()}
            >
              Reload
            </button>
          )}
          {(invitations.length > 0 || directInvitation) && (
            <section className="team-invites">
              <h2>Invitations</h2>
              {invitations.map((invite) => (
                <article className="team-invite" key={invite.id}>
                  <div>
                    <strong>{invite.project_name}</strong>
                    <p>
                      Join as {roleNames[invite.role]} · {invite.email}
                    </p>
                  </div>
                  <button
                    className="team-secondary"
                    disabled={busy}
                    onClick={() => void accept(invite.id)}
                  >
                    Accept invitation
                  </button>
                </article>
              ))}
              {directInvitation && (
                <article className="team-invite">
                  <div>
                    <strong>Project invitation</strong>
                    <p>Join with the invited email address.</p>
                  </div>
                  <button
                    className="team-secondary"
                    disabled={busy}
                    onClick={() => void accept(route.invitationId!)}
                  >
                    Accept invitation
                  </button>
                </article>
              )}
            </section>
          )}
          {currentProfile && !currentProfile.enabled && (
            <div className="team-empty">
              <h2>Awaiting an invitation</h2>
              <p>
                Ask the project owner to invite <strong>{user.email}</strong>.
              </p>
              <button
                className="team-secondary"
                disabled={loading}
                onClick={() => void refreshProfile()}
              >
                Check invitations
              </button>
            </div>
          )}
          {currentProfile?.enabled && (
            <>
              <div className="team-project-grid">
                {currentProfile.projects.map((project) => (
                  <article className="team-project-card" key={project.id}>
                    <div className="team-project-card-top">
                      <button
                        className="team-project-open"
                        onClick={() => navigate(project.id)}
                      >
                        <h2>{project.name}</h2>
                        <span className={`team-role ${project.role}`}>
                          {roleNames[project.role]}
                        </span>
                      </button>
                      <ProjectNameEditor
                        key={`${uid}:${project.id}`}
                        project={project}
                        onRename={renameProject}
                        disabled={projectNames.current.busy(project.id)}
                      />
                    </div>
                    {project.description && <p>{project.description}</p>}
                  </article>
                ))}
              </div>
              {!currentProfile.projects.length && (
                <p className="team-empty-note">No projects yet.</p>
              )}
            </>
          )}
          <footer className="team-dashboard-footer">
            <button onClick={() => void refreshProfile()} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh projects"}
            </button>
          </footer>
        </main>
      )}
      {createOpen && (
        <Modal
          title="New project"
          close={() => {
            if (!busy) setCreateOpen(false);
          }}
        >
          <ProjectCreateForm
            name={projectName}
            description={description}
            busy={busy}
            error={error}
            onName={setProjectName}
            onDescription={setDescription}
            onCreate={(input) => void createProject(input)}
          />
        </Modal>
      )}
      {membersOpen && selected && (
        <Members
          project={selected}
          api={api}
          close={() => setMembersOpen(false)}
        />
      )}
      {leaveFailure && (
        <Modal title="Keep your draft" close={() => setLeaveFailure(null)}>
          <div className="team-modal-body">
            <Notice error>{leaveFailure.message}</Notice>
            <p>
              You can download a local copy without changing the draft on the
              server.
            </p>
            <button
              className="team-primary"
              onClick={() => {
                const action = leaveFailure.action;
                controls.current?.downloadDraft();
                controls.current?.cancelPending();
                setLeaveFailure(null);
                void Promise.resolve(action()).catch((failure) =>
                  setError(message(failure)),
                );
              }}
            >
              Download draft and leave
            </button>
            <button
              className="team-secondary"
              onClick={() => setLeaveFailure(null)}
            >
              Keep editing
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function DesktopLoginApproval({
  requestId,
  auth,
  user,
  api,
}: {
  requestId: string;
  auth: Auth;
  user: User;
  api: ReturnType<typeof createTeamApi>;
}) {
  const [busy, setBusy] = useState(false);
  const [approved, setApproved] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    void api<{ kind: string }>(`/desktop/login/${requestId}`)
      .then((value) => {
        if (!current) return;
        if (value.kind !== "login")
          throw new Error(
            "Return to the app and choose Google or an email sign-in code. No password is needed.",
          );
        setReady(true);
      })
      .catch((failure) => {
        if (current) setError(message(failure));
      });
    return () => {
      current = false;
    };
  }, [api, requestId, user.uid]);
  async function approve() {
    setBusy(true);
    setError("");
    try {
      await api(`/desktop/login/${requestId}/authorize`, { method: "POST" });
      setApproved(true);
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="team-auth">
      <section className="team-auth-panel">
        <div className="team-auth-card">
          <Brand />
          <h1>
            {approved
              ? "Sign-in complete"
              : "Sign in to the OpenEconometrics app"}
          </h1>
          {approved ? (
            <p>You can return to the app.</p>
          ) : (
            <>
              <p>{user.email}</p>
              <p>
                This must match the code shown in the app:{" "}
                <strong>{requestId.slice(0, 8).toUpperCase()}</strong>
              </p>
              <button
                className="team-primary"
                disabled={busy || !ready}
                onClick={() => void approve()}
              >
                {busy ? "Signing in…" : "Approve sign-in"}
              </button>
            </>
          )}
          <Notice error>{error}</Notice>
          {!approved && (
            <button
              className="team-text-button"
              disabled={busy}
              onClick={() => void signOut(auth)}
            >
              Use another account
            </button>
          )}
        </div>
      </section>
    </main>
  );
}

function Members({
  project,
  api,
  close,
}: {
  project: Project;
  api: ReturnType<typeof createTeamApi>;
  close: () => void;
}) {
  const [members, setMembers] = useState<Member[]>([]);
  const [invitations, setInvitations] = useState<Invitation[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"editor" | "viewer">("editor");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [inviteUrl, setInviteUrl] = useState("");
  const [remove, setRemove] = useState<Member | null>(null);
  const path = `/projects/${encodeURIComponent(project.id)}`;
  const active = useRef(true);
  const refresh = useCallback(async () => {
    const result = await api<{ members: Member[]; invitations: Invitation[] }>(
      `${path}/members`,
    );
    if (active.current) {
      setMembers(result.members);
      setInvitations(result.invitations);
    }
  }, [api, path]);
  useEffect(() => {
    active.current = true;
    void refresh()
      .catch((failure) => {
        if (active.current) setError(message(failure));
      })
      .finally(() => {
        if (active.current) setLoading(false);
      });
    return () => {
      active.current = false;
    };
  }, [refresh]);
  async function mutate(suffix: string, method: string, body?: object) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api(`${path}${suffix}`, {
        method,
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      await refresh();
      setRemove(null);
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }
  async function invite(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    setInviteUrl("");
    try {
      const result = await api<Invitation>(`${path}/invitations`, {
        method: "POST",
        body: JSON.stringify({ email: email.trim(), role }),
      });
      setInviteUrl(
        result.url ||
          `${window.location.origin}/?invite=${encodeURIComponent(result.id)}`,
      );
      setEmail("");
      await refresh();
      setNotice("Share the invitation link. No email is sent automatically.");
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }
  async function copy(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setNotice("Invitation link copied.");
    } catch {
      setInviteUrl(url);
      setNotice("Select the link in the field below to copy it manually.");
    }
  }
  return (
    <Modal title="Project members" close={close}>
      <div className="team-modal-body">
        <Notice error>{error}</Notice>
        <Notice>{notice}</Notice>
        {loading && <p role="status">Loading members…</p>}
        <ul className="team-member-list">
          {members.map((member) => (
            <li key={member.uid}>
              <div className="team-member-name">
                <strong>{member.name || member.email}</strong>
                {member.name && <span>{member.email}</span>}
              </div>
              {project.role === "owner" && member.role !== "owner" ? (
                <>
                  <select
                    aria-label={`Access role for ${member.email}`}
                    value={member.role}
                    disabled={busy}
                    onChange={(e) =>
                      void mutate(
                        `/members/${encodeURIComponent(member.uid)}`,
                        "PATCH",
                        { role: e.target.value },
                      )
                    }
                  >
                    <option value="editor">Editor</option>
                    <option value="viewer">Viewer</option>
                  </select>
                  <button
                    className="team-remove"
                    onClick={() => setRemove(member)}
                    disabled={busy}
                  >
                    Remove
                  </button>
                </>
              ) : (
                <span className={`team-role ${member.role}`}>
                  {roleNames[member.role]}
                </span>
              )}
            </li>
          ))}
        </ul>
        {remove && (
          <div className="team-remove-confirm">
            <p>
              Remove <strong>{remove.email}</strong> from this project?
            </p>
            <button
              className="team-danger"
              disabled={busy}
              onClick={() =>
                void mutate(
                  `/members/${encodeURIComponent(remove.uid)}`,
                  "DELETE",
                )
              }
            >
              Remove from project
            </button>
            <button
              className="team-text-button"
              onClick={() => setRemove(null)}
            >
              Cancel
            </button>
          </div>
        )}
        {project.role === "owner" && (
          <>
            <form
              className="team-invite-form"
              onSubmit={(event) => void invite(event)}
            >
              <h3>Invite</h3>
              <label>
                Email
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  maxLength={254}
                />
              </label>
              <div className="team-invite-form-row">
                <label>
                  Project role
                  <select
                    value={role}
                    onChange={(e) =>
                      setRole(e.target.value as "editor" | "viewer")
                    }
                  >
                    <option value="editor">Editor</option>
                    <option value="viewer">Viewer</option>
                  </select>
                </label>
                <button
                  className="team-primary"
                  disabled={busy || !email.trim()}
                >
                  Create invitation
                </button>
              </div>
            </form>
            {inviteUrl && (
              <div className="team-copy-link">
                <input
                  aria-label="Invitation link"
                  readOnly
                  value={inviteUrl}
                  onFocus={(e) => e.target.select()}
                />
                <button
                  className="team-secondary"
                  onClick={() => void copy(inviteUrl)}
                >
                  Copy
                </button>
              </div>
            )}
            {invitations.length > 0 && (
              <section className="team-pending-invites">
                <h3>Pending invitations</h3>
                {invitations.map((item) => (
                  <div key={item.id}>
                    <span>
                      <strong>{item.email}</strong>
                      <small>
                        {roleNames[item.role]} · Expires:{" "}
                        {new Date(item.expires_at).toLocaleDateString("en-US")}
                      </small>
                    </span>
                    <button
                      className="team-text-button"
                      onClick={() =>
                        void copy(
                          item.url ||
                            `${window.location.origin}/?invite=${encodeURIComponent(item.id)}`,
                        )
                      }
                    >
                      Link
                    </button>
                    <button
                      className="team-remove"
                      disabled={busy}
                      onClick={() =>
                        void mutate(
                          `/invitations/${encodeURIComponent(item.id)}`,
                          "DELETE",
                        )
                      }
                    >
                      Cancel invitation
                    </button>
                  </div>
                ))}
              </section>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}
