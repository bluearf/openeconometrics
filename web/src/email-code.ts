import { signInWithCustomToken, type Auth } from "firebase/auth";
import { decodeResponse } from "./api.ts";
import { desktopCloudFetch, isDesktop } from "./desktop.ts";

type RequestCode = (
  path: string,
  body: Record<string, string>,
  signal: AbortSignal,
) => Promise<unknown>;
async function requestCode(
  path: string,
  body: Record<string, string>,
  signal: AbortSignal,
) {
  const options = { method: "POST", body: JSON.stringify(body), signal };
  return decodeResponse<unknown>(
    isDesktop()
      ? await desktopCloudFetch(path, "", options)
      : await fetch(path, {
          ...options,
          headers: { "Content-Type": "application/json" },
        }),
  );
}

/** The proof stays in this sign-in screen's memory, never URLs or storage. */
export class EmailCodeSession {
  private attempt: {
    controller: AbortController;
    requestId: string;
    verifier: string;
  } | null = null;
  private committing = false;
  private auth: Auth;
  private request: RequestCode;
  private signIn: typeof signInWithCustomToken;
  constructor(
    auth: Auth,
    request: RequestCode = requestCode,
    signIn: typeof signInWithCustomToken = signInWithCustomToken,
  ) {
    this.auth = auth;
    this.request = request;
    this.signIn = signIn;
  }
  cancel() {
    if (this.committing) return false;
    this.attempt?.controller.abort();
    this.attempt = null;
    return true;
  }
  async send(email: string) {
    if (this.committing) throw new Error("Sign-in is completing.");
    this.cancel();
    const verifier = Array.from(
      crypto.getRandomValues(new Uint8Array(32)),
      (n) => n.toString(16).padStart(2, "0"),
    ).join("");
    const attempt = {
      controller: new AbortController(),
      requestId: "",
      verifier,
    };
    this.attempt = attempt;
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(verifier),
    );
    const challenge = Array.from(new Uint8Array(digest), (n) =>
      n.toString(16).padStart(2, "0"),
    ).join("");
    if (this.attempt !== attempt)
      throw new DOMException("Sign-in cancelled", "AbortError");
    const data = (await this.request(
      "/api/auth/email-code",
      { email: email.trim(), challenge },
      attempt.controller.signal,
    )) as Record<string, unknown>;
    if (this.attempt !== attempt)
      throw new DOMException("Sign-in cancelled", "AbortError");
    if (
      !data ||
      typeof data.request_id !== "string" ||
      !/^[0-9a-f]{32}$/.test(data.request_id) ||
      data.retry_after !== 60 ||
      typeof data.expires_at !== "string" ||
      !Number.isFinite(Date.parse(data.expires_at))
    )
      throw new Error("The sign-in response could not be verified.");
    attempt.requestId = data.request_id;
    return data.retry_after;
  }
  async verify(code: string) {
    const attempt = this.attempt;
    if (!attempt?.requestId || !/^[0-9]{6}$/.test(code) || this.committing)
      throw new Error("Enter the six-digit code from your email.");
    const data = (await this.request(
      `/api/auth/email-code/${attempt.requestId}/verify`,
      { code, verifier: attempt.verifier },
      attempt.controller.signal,
    )) as Record<string, unknown>;
    if (this.attempt !== attempt)
      throw new DOMException("Sign-in cancelled", "AbortError");
    if (
      !data ||
      typeof data.custom_token !== "string" ||
      !data.custom_token.length ||
      data.custom_token.length > 16384 ||
      /\s/.test(data.custom_token)
    )
      throw new Error("The sign-in response could not be verified.");
    this.committing = true;
    try {
      return await this.signIn(this.auth, data.custom_token);
    } finally {
      this.committing = false;
      this.attempt = null;
    }
  }
}
