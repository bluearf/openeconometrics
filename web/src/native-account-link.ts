/** UID-bound browser handoff. This bridge never signs in or carries credentials. */
import * as firebase from "firebase/auth";
import type { Auth, User } from "firebase/auth";
import { desktopCloudRequest, nativeInvoke } from "./desktop.ts";
import type { AccountMethod } from "./account-linking.ts";

export interface NativeLinkDependencies {
  request: typeof desktopCloudRequest;
  open: typeof nativeInvoke;
  reload: typeof firebase.reload;
  token: typeof firebase.getIdToken;
  intervalMs?: number;
  timeoutMs?: number;
}
function cancelled() {
  return new DOMException("Account linking cancelled.", "AbortError");
}
function check(signal?: AbortSignal) {
  if (signal?.aborted) throw cancelled();
}
function abortable<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending;
  return new Promise((resolve, reject) => {
    const abort = () => {
      signal.removeEventListener("abort", abort);
      reject(cancelled());
    };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    pending.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        if (signal.aborted) reject(cancelled());
        else resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}
function pause(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    };
    const abort = () => {
      cleanup();
      reject(cancelled());
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}
export async function linkDesktopAccount(
  auth: Auth,
  user: User,
  target: AccountMethod,
  signal?: AbortSignal,
  progress?: (value: string) => void,
  deps: NativeLinkDependencies = {
    request: desktopCloudRequest,
    open: nativeInvoke,
    reload: firebase.reload,
    token: firebase.getIdToken,
  },
): Promise<void> {
  const uid = user.uid;
  const sameUser = () => {
    check(signal);
    if (auth.currentUser?.uid !== uid)
      throw Object.assign(new Error("Account changed."), {
        code: "account/session-changed",
      });
  };
  sameUser();
  const token = await abortable(deps.token(user), signal);
  sameUser();
  const verifier = Array.from(
    crypto.getRandomValues(new Uint8Array(32)),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
  const digest = await abortable(
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    signal,
  );
  const challenge = Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  sameUser();
  const grant = await abortable(
    deps.request<{ request_id: string; code: string }>(
      "/api/desktop/account-link",
      token,
      { method: "POST", body: JSON.stringify({ challenge, target }), signal },
    ),
    signal,
  );
  // Cancellation does not undo a method already linked by the browser. It only
  // revokes this handoff and prevents a late acknowledgement in the app.
  let revocationRequested = false;
  const cancelGrant = () => {
    if (revocationRequested) return;
    revocationRequested = true;
    void deps
      .request(`/api/desktop/account-link/${grant.request_id}`, token, {
        method: "DELETE",
      })
      .catch(() => {});
  };
  signal?.addEventListener("abort", cancelGrant, { once: true });
  try {
    sameUser();
    progress?.(
      `Account code: ${grant.code}. Verify your existing sign-in in the browser, then add the method.`,
    );
    await abortable(
      deps.open("open_desktop_login", { requestId: grant.request_id }),
      signal,
    );
    const deadline = Date.now() + (deps.timeoutMs ?? 5 * 60_000);
    while (Date.now() < deadline) {
      await pause(
        Math.min(deps.intervalMs ?? 1500, Math.max(0, deadline - Date.now())),
        signal,
      );
      sameUser();
      if (Date.now() >= deadline) break;
      const response = await abortable(
        deps.request<{
          pending?: boolean;
          linked?: boolean;
          uid?: string;
          target?: string;
        }>(`/api/desktop/account-link/${grant.request_id}/exchange`, token, {
          method: "POST",
          body: JSON.stringify({ verifier }),
          signal,
        }),
        signal,
      );
      sameUser();
      if (response.linked === true) {
        if (response.uid !== uid || response.target !== target)
          throw Object.assign(new Error("Account changed."), {
            code: "account/session-changed",
          });
        await abortable(deps.reload(user), signal);
        sameUser();
        if (
          !user.providerData.some((provider) => provider.providerId === target)
        )
          throw new Error("The method is not linked to this Firebase user.");
        await abortable(deps.token(user, true), signal);
        sameUser();
        return;
      }
      if (response.pending !== true)
        throw new Error("Could not verify account linking.");
    }
    throw new Error("Account linking timed out. Start again from the app.");
  } finally {
    signal?.removeEventListener("abort", cancelGrant);
    cancelGrant();
  }
}
