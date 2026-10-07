import {
  GoogleAuthProvider,
  browserPopupRedirectResolver,
  signInWithPopup,
  signInWithCustomToken,
  type Auth,
} from "firebase/auth";
import { isDesktop } from "./desktop.ts";
import {
  checkSignInSignal,
  waitForDesktopGoogleToken,
} from "./native-google-flow.ts";

async function desktopGoogleSignIn(
  auth: Auth,
  progress?: (text: string) => void,
  options?: { signal?: AbortSignal; onCommit?: () => void },
) {
  const token = await waitForDesktopGoogleToken(options?.signal, progress);
  checkSignInSignal(options?.signal);
  // Cancellation ends before the SDK commits a credential. The UI hides its
  // cancel control during this short, non-cancellable identity commit.
  options?.onCommit?.();
  return signInWithCustomToken(auth, token);
}

/** Called directly from the click handler, without awaiting other work first. */
export function startGoogleSignIn(
  auth: Auth,
  openPopup: typeof signInWithPopup = signInWithPopup,
  progress?: (text: string) => void,
  options?: { signal?: AbortSignal; onCommit?: () => void },
) {
  if (isDesktop()) return desktopGoogleSignIn(auth, progress, options);
  const provider = new GoogleAuthProvider();
  provider.addScope("email");
  provider.setCustomParameters({ prompt: "select_account" });
  // Auth normally initializes with persistence only. Enable its popup resolver
  // explicitly for this gesture; successful sign-in uses the existing listener.
  return openPopup(auth, provider, browserPopupRedirectResolver);
}

const googleErrors: Record<string, string> = {
  "auth/popup-blocked":
    "Your browser blocked the sign-in window. Allow pop-ups for this site and try again.",
  "auth/popup-closed-by-user":
    "The window closed before Google sign-in finished. Try again when you are ready.",
  "auth/cancelled-popup-request":
    "Another sign-in window opened. Continue in the latest window or try again.",
  "auth/unauthorized-domain":
    "This site address is not authorized for Google sign-in. Contact the workspace administrator.",
  "auth/account-exists-with-different-credential":
    "This email address is registered with another sign-in method. Sign in with your existing method first; accounts are not merged automatically.",
  "auth/web-storage-unsupported":
    "Your browser does not allow session storage. Check your cookie and site storage settings and try again.",
};

export function googleAuthError(error: unknown): string | undefined {
  const code =
    error && typeof error === "object" && "code" in error
      ? String(error.code)
      : "";
  return googleErrors[code];
}
