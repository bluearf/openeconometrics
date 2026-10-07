import type { User } from "firebase/auth";

export interface VerificationDeliveryState {
  uid: string | null;
  status: "idle" | "sending" | "sent" | "error";
  message: string;
  retryAt: number;
  sentAt: number | null;
}
const EMPTY: VerificationDeliveryState = Object.freeze({
  uid: null,
  status: "idle",
  message: "",
  retryAt: 0,
  sentAt: null,
});
interface DeliveryOptions {
  now?: () => number;
  cooldownMs?: number;
}

/** Lives above the signed-out screen; auth changes and profile fetches cannot erase a send result. */
export class VerificationDelivery {
  private readonly states = new Map<string, VerificationDeliveryState>();
  private readonly pending = new Map<
    string,
    Promise<VerificationDeliveryState>
  >();
  private readonly listeners = new Set<() => void>();
  private readonly deliver: (user: User) => Promise<void>;
  private readonly now: () => number;
  private readonly cooldownMs: number;

  constructor(
    deliver: (user: User) => Promise<void>,
    options: DeliveryOptions = {},
  ) {
    this.deliver = deliver;
    this.now = options.now ?? Date.now;
    this.cooldownMs = options.cooldownMs ?? 60_000;
  }
  getSnapshot = (uid?: string): VerificationDeliveryState =>
    uid ? (this.states.get(uid) ?? EMPTY) : EMPTY;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  private publish(state: VerificationDeliveryState) {
    this.states.set(state.uid!, state);
    for (const listener of this.listeners) listener();
  }
  send(user: User): Promise<VerificationDeliveryState> {
    const existing = this.pending.get(user.uid);
    if (existing) return existing;
    const before = this.getSnapshot(user.uid);
    if (this.now() < before.retryAt) return Promise.resolve(before);
    const started = this.now();
    const sending: VerificationDeliveryState = {
      uid: user.uid,
      status: "sending",
      message: "Sending verification email…",
      retryAt: started + this.cooldownMs,
      sentAt: before.sentAt,
    };
    // Register the request before notifying React, including synchronous subscribers.
    const operation = Promise.resolve().then(async () => {
      let result: VerificationDeliveryState;
      try {
        await this.deliver(user);
        result = {
          ...sending,
          status: "sent",
          sentAt: this.now(),
          message:
            "Your verification email is queued. Check your inbox and spam folder; it may take a few minutes to arrive.",
        };
      } catch (error) {
        const code =
          error && typeof error === "object" && "code" in error
            ? String(error.code)
            : "";
        result = {
          ...sending,
          status: "error",
          retryAt:
            code === "auth/too-many-requests"
              ? Math.max(sending.retryAt, this.now() + 120_000)
              : sending.retryAt,
          message: `Could not send the verification email. ${verificationErrorMessage(error)}`,
        };
      }
      this.pending.delete(user.uid);
      this.publish(result);
      return result;
    });
    this.pending.set(user.uid, operation);
    this.publish(sending);
    return operation;
  }
}

/** Updating an optional display name must never prevent the verification email. */
export async function finishSignupVerification(
  user: User,
  name: string,
  delivery: VerificationDelivery,
  saveName: (user: User, name: string) => Promise<void>,
): Promise<{ mail: VerificationDeliveryState; profileError: unknown | null }> {
  const mail = delivery.send(user);
  let profileError: unknown | null = null;
  if (name.trim()) {
    try {
      await saveName(user, name.trim());
    } catch (error) {
      profileError = error;
    }
  }
  return { mail: await mail, profileError };
}

export function verificationErrorMessage(error: unknown): string {
  const code =
    error && typeof error === "object" && "code" in error
      ? String(error.code)
      : "";
  const errors: Record<string, string> = {
    "auth/too-many-requests":
      "Too many attempts. Wait a while and try again.",
    "auth/network-request-failed": "Check your connection and try again.",
    "auth/user-token-expired":
      "Your session may have expired. Sign out and sign in again.",
    "auth/requires-recent-login":
      "Sign out, sign in again, and retry.",
    "auth/user-disabled":
      "This account is disabled. Contact the workspace administrator.",
  };
  return (
    errors[code] ||
    "The operation could not be completed right now. Try again later; if it continues, contact the workspace administrator."
  );
}

/** A changed account invalidates checks without leaving a stale busy flag behind. */
export class VerificationCheckGate {
  private uid: string | undefined;
  private sequence = 0;
  private pending = false;
  select(uid?: string) {
    if (uid !== this.uid) {
      this.uid = uid;
      this.sequence++;
      this.pending = false;
    }
  }
  begin(uid: string): number | null {
    this.select(uid);
    if (this.pending) return null;
    this.pending = true;
    return ++this.sequence;
  }
  current(ticket: number): boolean {
    return ticket === this.sequence;
  }
  finish(ticket: number): boolean {
    if (!this.current(ticket)) return false;
    this.pending = false;
    return true;
  }
}
