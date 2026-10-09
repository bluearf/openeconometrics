/** Explicit provider linking; email is never used to merge Firebase users. */
import * as firebase from "firebase/auth";
import type { Auth, User, UserCredential } from "firebase/auth";

export type AccountMethod = "password" | "google.com";
export interface LinkingDependencies {
  reauthenticatePassword: typeof firebase.reauthenticateWithCredential;
  reauthenticateGoogle: typeof firebase.reauthenticateWithPopup;
  linkPassword: typeof firebase.linkWithCredential;
  linkGoogle: typeof firebase.linkWithPopup;
  now: () => number;
}
const dependencies: LinkingDependencies = {
  reauthenticatePassword: (...args) =>
    firebase.reauthenticateWithCredential(...args),
  reauthenticateGoogle: (...args) => firebase.reauthenticateWithPopup(...args),
  linkPassword: (...args) => firebase.linkWithCredential(...args),
  linkGoogle: (...args) => firebase.linkWithPopup(...args),
  now: Date.now,
};

export function accountMethods(
  user: Pick<User, "providerData">,
): AccountMethod[] {
  return (["password", "google.com"] as const).filter((method) =>
    user.providerData.some((provider) => provider.providerId === method),
  );
}

function failure(code: string) {
  return Object.assign(new Error("Account linking could not be completed."), {
    code,
  });
}
function googleProvider() {
  const provider = new firebase.GoogleAuthProvider();
  provider.addScope("email");
  provider.setCustomParameters({ prompt: "select_account" });
  return provider;
}

/** A short-lived reauthentication belongs to one UID and one linking attempt. */
export class AccountLinkSession {
  private readonly auth: Auth;
  private readonly user: User;
  private readonly deps: LinkingDependencies;
  private readonly uid: string;
  private verifiedAt: number | null = null;
  private epoch = 0;
  private busy = false;
  constructor(
    auth: Auth,
    user: User,
    deps: LinkingDependencies = dependencies,
  ) {
    this.auth = auth;
    this.user = user;
    this.deps = deps;
    this.uid = user.uid;
  }
  cancel() {
    this.epoch++;
    this.verifiedAt = null;
  }
  private sameUser(result?: UserCredential) {
    if (
      this.auth.currentUser?.uid !== this.uid ||
      this.user.uid !== this.uid ||
      (result && result.user.uid !== this.uid)
    )
      throw failure("account/session-changed");
  }
  private begin() {
    this.sameUser();
    if (this.busy) throw failure("account/busy");
    this.busy = true;
    return this.epoch;
  }
  private finish(result: UserCredential, epoch: number) {
    this.sameUser(result);
    if (epoch !== this.epoch) throw failure("account/cancelled");
    return result;
  }
  reauthenticate(
    method: AccountMethod,
    password = "",
  ): Promise<UserCredential> {
    const epoch = this.begin();
    this.verifiedAt = null;
    try {
      if (!accountMethods(this.user).includes(method))
        throw failure("account/method-unavailable");
      // Popup invocation occurs in the originating click, before any await.
      const pending =
        method === "google.com"
          ? this.deps.reauthenticateGoogle(
              this.user,
              googleProvider(),
              firebase.browserPopupRedirectResolver,
            )
          : this.deps.reauthenticatePassword(
              this.user,
              firebase.EmailAuthProvider.credential(this.email(), password),
            );
      return pending
        .then((result) => {
          this.finish(result, epoch);
          this.verifiedAt = this.deps.now();
          return result;
        })
        .finally(() => {
          this.busy = false;
        });
    } catch (error) {
      this.busy = false;
      throw error;
    }
  }
  private email() {
    if (!this.user.email) throw failure("account/email-required");
    return this.user.email;
  }
  link(method: AccountMethod, password = ""): Promise<UserCredential> {
    const epoch = this.begin();
    try {
      if (
        this.verifiedAt === null ||
        this.deps.now() - this.verifiedAt > 5 * 60_000
      )
        throw failure("account/reauth-required");
      if (accountMethods(this.user).includes(method))
        throw failure("auth/provider-already-linked");
      this.verifiedAt = null;
      const pending =
        method === "google.com"
          ? this.deps.linkGoogle(
              this.user,
              googleProvider(),
              firebase.browserPopupRedirectResolver,
            )
          : this.deps.linkPassword(
              this.user,
              firebase.EmailAuthProvider.credential(this.email(), password),
            );
      return pending
        .then((result) => {
          this.finish(result, epoch);
          if (!accountMethods(result.user).includes(method))
            throw failure("account/provider-not-linked");
          return result;
        })
        .finally(() => {
          this.busy = false;
          this.verifiedAt = null;
        });
    } catch (error) {
      this.busy = false;
      this.verifiedAt = null;
      throw error;
    }
  }
}

/** Provider errors can include credentials; never render the raw SDK message. */
export function accountLinkError(error: unknown): string {
  const code = (error as { code?: string } | null)?.code;
  const messages: Record<string, string> = {
    "auth/credential-already-in-use":
      "That sign-in method belongs to another account. No accounts or projects were merged.",
    "auth/email-already-in-use":
      "That email belongs to another account. No accounts or projects were merged.",
    "auth/account-exists-with-different-credential":
      "That credential belongs to another account. Sign in to that account separately.",
    "auth/user-mismatch":
      "Verify the account currently shown here. A different account cannot approve this change.",
    "auth/invalid-credential":
      "The current sign-in could not be verified. Check your password and try again.",
    "auth/wrong-password": "The current password is incorrect.",
    "auth/weak-password":
      "Choose a stronger password that meets the sign-in policy.",
    "auth/password-does-not-meet-requirements":
      "Choose a password that meets the sign-in policy.",
    "auth/popup-closed-by-user":
      "Google verification was cancelled. No new method was added.",
    "auth/cancelled-popup-request":
      "Google verification was cancelled. Try again when ready.",
    "auth/popup-blocked": "Allow pop-ups for this site, then try again.",
    "auth/requires-recent-login":
      "Verify your current sign-in again before adding a method.",
    "auth/provider-already-linked": "This sign-in method is already linked.",
    "account/session-changed":
      "Your signed-in account changed. Close this screen and try again.",
    "account/reauth-required":
      "Verify your current sign-in before adding a method.",
    "account/cancelled":
      "Account verification was cancelled. No further action was taken.",
    "account/busy": "Wait for the current verification to finish.",
    "account/method-unavailable":
      "Choose a sign-in method already linked to this account.",
    "account/email-required":
      "This account has no email address for password sign-in.",
    "account/provider-not-linked":
      "The new method could not be confirmed. Reopen Account to check linked methods.",
  };
  return messages[code || ""] || "Could not update sign-in methods. Try again.";
}
