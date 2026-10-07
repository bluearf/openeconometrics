import assert from "node:assert/strict";
import test from "node:test";
import {
  GoogleAuthProvider,
  browserPopupRedirectResolver,
  type Auth,
  type UserCredential,
} from "firebase/auth";
import { googleAuthError, startGoogleSignIn } from "../src/google-auth.ts";

test("Google sign-in invokes the SDK immediately with explicit popup support and identity-only scopes", async () => {
  const auth = { app: { name: "test-auth" } } as Auth;
  const credential = {
    user: { uid: "firebase-user-id" },
    providerId: "google.com",
  } as UserCredential;
  const completion = Promise.resolve(credential);
  let called = false;
  const result = startGoogleSignIn(auth, (actualAuth, provider, resolver) => {
    called = true;
    assert.equal(actualAuth, auth);
    assert.ok(provider instanceof GoogleAuthProvider);
    assert.equal(provider.providerId, "google.com");
    assert.deepEqual(provider.getScopes().sort(), ["email", "profile"]);
    assert.deepEqual(provider.getCustomParameters(), {
      prompt: "select_account",
    });
    // The installed SDK's binding must be supplied, even when initializeAuth
    // intentionally omitted a default popup resolver.
    assert.equal(resolver, browserPopupRedirectResolver);
    return completion;
  });
  assert.equal(
    called,
    true,
    "opening must not wait for a token request or other async work",
  );
  assert.equal(result, completion);
  assert.equal(
    await result,
    credential,
    "Firebase's UID/result passes through unchanged",
  );
});

test("popup cancellation and account collisions propagate without automatic retry or account linking", async () => {
  for (const code of [
    "auth/popup-closed-by-user",
    "auth/account-exists-with-different-credential",
  ]) {
    const failure = Object.assign(new Error("Firebase rejected sign-in"), {
      code,
    });
    let calls = 0;
    const pending = startGoogleSignIn({} as Auth, () => {
      calls++;
      return Promise.reject(failure);
    });
    await assert.rejects(pending, (error) => error === failure);
    assert.equal(calls, 1);
    assert.ok(googleAuthError(failure));
  }
});

test("a retry uses a fresh Google provider without scopes retained from an earlier attempt", async () => {
  let previous: GoogleAuthProvider | undefined;
  const open = (_auth: Auth, provider: unknown) => {
    assert.ok(provider instanceof GoogleAuthProvider);
    assert.notEqual(provider, previous);
    assert.deepEqual(provider.getScopes().sort(), ["email", "profile"]);
    previous = provider;
    provider.addScope("unexpected-scope-from-another-operation");
    return Promise.resolve({} as UserCredential);
  };
  await startGoogleSignIn({} as Auth, open);
  await startGoogleSignIn({} as Auth, open);
});

test("Google auth failures give actionable English explanations without exposing credential details", () => {
  assert.match(
    googleAuthError({ code: "auth/popup-blocked" })!,
    /Allow pop-ups/,
  );
  assert.match(
    googleAuthError({ code: "auth/unauthorized-domain" })!,
    /administrator/,
  );
  const message = googleAuthError({
    code: "auth/account-exists-with-different-credential",
    customData: { secret: "credential-not-for-display" },
  });
  assert.match(message!, /existing method/);
  assert.ok(!message!.includes("credential-not-for-display"));
  assert.equal(googleAuthError({ code: "auth/unrecognized" }), undefined);
  assert.equal(googleAuthError(null), undefined);
});
