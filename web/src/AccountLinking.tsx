import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import * as firebase from "firebase/auth";
import type { Auth, User } from "firebase/auth";
import {
  AccountLinkSession,
  accountLinkError,
  accountMethods,
  type AccountMethod,
} from "./account-linking";
import { linkDesktopAccount } from "./native-account-link";
import { isDesktop } from "./desktop";
import "./account-linking.css";

export interface AccountLinkingProps {
  auth: Auth;
  user: User;
  target?: AccountMethod;
  nativeAvailable?: boolean;
  onLinked: () => Promise<void>;
  onBusy?: (busy: boolean) => void;
  refreshFailureMessage?: string;
  /** Injection is used by component tests, never by production auth selection. */
  session?: AccountLinkSession;
}
export default function AccountLinking({
  auth,
  user,
  target,
  nativeAvailable = false,
  onLinked,
  onBusy,
  refreshFailureMessage = "The method was added, but projects could not be refreshed. Reload to check your project list.",
  session: injected,
}: AccountLinkingProps) {
  const session = useMemo(
    () => injected || new AccountLinkSession(auth, user),
    [auth, user, injected],
  );
  const [methods, setMethods] = useState(() => accountMethods(user));
  const [checking, setChecking] = useState(!injected);
  const [method, setMethod] = useState<AccountMethod>(
    methods.includes("password") ? "password" : "google.com",
  );
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [verified, setVerified] = useState(false);
  const [busy, setBusy] = useState<"verify" | "link" | "native" | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controller = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      session.cancel();
      controller.current?.abort();
    };
  }, [session]);
  useEffect(() => {
    if (injected) return;
    let current = true;
    setChecking(true);
    void firebase
      .reload(user)
      .then(() => {
        if (!current || auth.currentUser?.uid !== user.uid) return;
        const next = accountMethods(user);
        setMethods(next);
        setMethod((value) =>
          next.includes(value) ? value : next[0] || "google.com",
        );
      })
      .catch(() => {
        if (current)
          setError(
            "Could not refresh linked methods. Check your connection and reopen Account.",
          );
      })
      .finally(() => {
        if (current) setChecking(false);
      });
    return () => {
      current = false;
    };
  }, [auth, user, injected]);
  useEffect(() => {
    onBusy?.(busy === "link");
  }, [busy, onBusy]);
  function resetSecrets() {
    setCurrentPassword("");
    setNewPassword("");
    setConfirmation("");
  }
  async function verify(event?: FormEvent) {
    event?.preventDefault();
    setBusy("verify");
    setError("");
    setNotice("");
    try {
      await session.reauthenticate(method, currentPassword);
      if (mounted.current) setVerified(true);
    } catch (failure) {
      if (mounted.current) {
        setVerified(false);
        setError(accountLinkError(failure));
      }
    } finally {
      if (mounted.current) {
        setCurrentPassword("");
        setBusy(null);
      }
    }
  }
  async function linked() {
    // The link is already committed in Firebase. A profile refresh failure must
    // not be described as a failed link or trigger an automatic unlink/merge.
    setMethods(accountMethods(user));
    setVerified(false);
    resetSecrets();
    setNotice("Sign-in method added. Your account identity is unchanged.");
    try {
      await onLinked();
    } catch {
      if (mounted.current) setError(refreshFailureMessage);
    }
  }
  async function add(next: AccountMethod, event?: FormEvent) {
    event?.preventDefault();
    if (next === "password" && newPassword !== confirmation) {
      setError("Passwords must match.");
      return;
    }
    setBusy("link");
    setError("");
    setNotice("");
    try {
      await session.link(next, newPassword);
      if (mounted.current) await linked();
    } catch (failure) {
      if (mounted.current) {
        setVerified(false);
        setError(accountLinkError(failure));
        resetSecrets();
      }
    } finally {
      if (mounted.current) setBusy(null);
    }
  }
  async function native(next: AccountMethod) {
    if (!nativeAvailable) return;
    controller.current = new AbortController();
    setBusy("native");
    setError("");
    setNotice("");
    try {
      await linkDesktopAccount(
        auth,
        user,
        next,
        controller.current.signal,
        setNotice,
      );
      if (mounted.current) await linked();
    } catch (failure) {
      if (mounted.current) {
        setNotice("");
        setError(
          (failure as Error)?.name === "AbortError"
            ? "Browser handoff cancelled. A method already added in the browser remains linked; reopen Account to check."
            : accountLinkError(failure),
        );
      }
    } finally {
      if (mounted.current) {
        setBusy(null);
      }
      controller.current = null;
    }
  }
  const missing = (["google.com", "password"] as const).filter(
    (value) => !methods.includes(value) && (!target || target === value),
  );
  return (
    <div className="team-modal-body account-linking">
      <p className="account-link-email">{user.email}</p>
      <p>
        Linked methods:{" "}
        {methods
          .map((value) => (value === "google.com" ? "Google" : "Password"))
          .join(" · ") || "None"}
      </p>
      {checking && <p role="status">Checking linked methods…</p>}
      <p>
        Adding a method keeps this account and its projects. A method owned by
        another account cannot be linked.
      </p>
      {notice && (
        <p role="status" className="account-link-notice">
          {notice}
        </p>
      )}
      {error && (
        <p role="alert" className="team-notice team-error">
          {error}
        </p>
      )}
      {!missing.length && !target ? (
        <p>Both available sign-in methods are linked.</p>
      ) : isDesktop() && !nativeAvailable ? (
        <p role="status">
          Account linking is unavailable on this server. Update the service
          before adding a sign-in method in the app.
        </p>
      ) : isDesktop() ? (
        <>
          {missing.map((value) => (
            <button
              key={value}
              className="team-primary"
              disabled={!!busy || checking}
              onClick={() => void native(value)}
            >
              Add {value === "google.com" ? "Google" : "password"} in browser
            </button>
          ))}
          {busy === "native" && (
            <button
              className="team-secondary"
              onClick={() => controller.current?.abort()}
            >
              Cancel browser handoff
            </button>
          )}
        </>
      ) : !verified ? (
        <form onSubmit={(event) => void verify(event)}>
          <h3>1. Verify your existing sign-in</h3>
          {methods.length > 1 && (
            <label>
              Current method
              <select
                aria-label="Current sign-in method"
                value={method}
                disabled={!!busy || checking}
                onChange={(event) => {
                  setMethod(event.target.value as AccountMethod);
                  setCurrentPassword("");
                }}
              >
                {methods.map((value) => (
                  <option key={value} value={value}>
                    {value === "google.com" ? "Google" : "Password"}
                  </option>
                ))}
              </select>
            </label>
          )}
          {method === "password" && (
            <label>
              Current password
              <input
                type="password"
                autoComplete="current-password"
                required
                value={currentPassword}
                disabled={!!busy || checking}
                onChange={(event) => setCurrentPassword(event.target.value)}
              />
            </label>
          )}
          <button
            className="team-primary"
            disabled={!!busy || checking || !methods.length}
          >
            {busy
              ? "Verifying…"
              : method === "password"
                ? "Verify password"
                : "Verify with Google"}
          </button>
        </form>
      ) : (
        <section>
          <h3>2. Add a sign-in method</h3>
          {target && methods.includes(target) && (
            <>
              <p>This method is already linked to this account.</p>
              <button
                className="team-primary"
                disabled={!!busy || checking}
                onClick={() => void linked()}
              >
                Confirm linked method
              </button>
            </>
          )}
          {missing.includes("google.com") && (
            <button
              className="team-primary"
              disabled={!!busy || checking}
              onClick={() => void add("google.com")}
            >
              Add Google
            </button>
          )}
          {missing.includes("password") && (
            <form onSubmit={(event) => void add("password", event)}>
              <label>
                New password
                <input
                  type="password"
                  autoComplete="new-password"
                  minLength={6}
                  required
                  value={newPassword}
                  disabled={!!busy || checking}
                  onChange={(event) => setNewPassword(event.target.value)}
                />
              </label>
              <label>
                Confirm new password
                <input
                  type="password"
                  autoComplete="new-password"
                  minLength={6}
                  required
                  value={confirmation}
                  disabled={!!busy || checking}
                  onChange={(event) => setConfirmation(event.target.value)}
                />
              </label>
              <button className="team-primary" disabled={!!busy || checking}>
                Add password
              </button>
            </form>
          )}
          {busy === "link" && (
            <p role="status">
              Adding method… Wait for confirmation before closing.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

/** Refresh the same user, without signing in under a new token or UID. */
export async function refreshLinkedUser(auth: Auth, user: User) {
  const uid = user.uid;
  if (auth.currentUser?.uid !== uid)
    throw new Error("The signed-in account changed.");
  await firebase.reload(user);
  if (auth.currentUser?.uid !== uid)
    throw new Error("The signed-in account changed.");
  await firebase.getIdToken(user, true);
  if (auth.currentUser?.uid !== uid)
    throw new Error("The signed-in account changed.");
}
