/** Cancellable native login exchange. No credential is committed here. */
import { desktopCloudRequest, nativeInvoke } from "./desktop.ts";

export function cancelledSignIn() {
  return new DOMException("Google sign-in cancelled.", "AbortError");
}

export function checkSignInSignal(signal?: AbortSignal) {
  if (signal?.aborted) throw cancelledSignIn();
}

/** The native request may finish later; cancellation releases its caller now. */
function abortable<T>(operation: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return operation;
  return new Promise((resolve, reject) => {
    const abort = () => {
      cleanup();
      reject(cancelledSignIn());
    };
    const cleanup = () => signal.removeEventListener("abort", abort);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    operation.then(
      (value) => {
        cleanup();
        if (signal.aborted) reject(cancelledSignIn());
        else resolve(value);
      },
      (error) => {
        cleanup();
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
      reject(cancelledSignIn());
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}

export interface NativeGoogleDependencies {
  request: typeof desktopCloudRequest;
  open: typeof nativeInvoke;
  intervalMs?: number;
  timeoutMs?: number;
}

export async function waitForDesktopGoogleToken(
  signal?: AbortSignal,
  progress?: (text: string) => void,
  deps: NativeGoogleDependencies = {
    request: desktopCloudRequest,
    open: nativeInvoke,
  },
): Promise<string> {
  checkSignInSignal(signal);
  const verifier = Array.from(
    crypto.getRandomValues(new Uint8Array(32)),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
  const digest = await abortable(
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    signal,
  );
  checkSignInSignal(signal);
  const challenge = Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  const grant = await abortable(
    deps.request<{ request_id: string; code: string }>(
      "/api/desktop/login",
      "",
      {
        method: "POST",
        body: JSON.stringify({ challenge }),
        signal,
      },
    ),
    signal,
  );
  checkSignInSignal(signal);
  progress?.(
    `Sign-in code: ${grant.code}. Waiting for approval in your browser…`,
  );
  checkSignInSignal(signal);
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
    checkSignInSignal(signal);
    if (Date.now() >= deadline) break;
    const response = await abortable(
      deps.request<{ pending?: boolean; custom_token?: string }>(
        `/api/desktop/login/${grant.request_id}/exchange`,
        "",
        {
          method: "POST",
          body: JSON.stringify({ verifier }),
          signal,
        },
      ),
      signal,
    );
    checkSignInSignal(signal);
    if (response.custom_token) return response.custom_token;
    if (response.pending !== true)
      throw new Error("Could not verify Google sign-in.");
  }
  throw new Error("Sign-in timed out. Try again.");
}
