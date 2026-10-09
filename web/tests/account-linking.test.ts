import assert from "node:assert/strict";
import test from "node:test";
import {
  GoogleAuthProvider,
  browserPopupRedirectResolver,
  type Auth,
  type User,
  type UserCredential,
} from "firebase/auth";
import {
  AccountLinkSession,
  accountLinkError,
  accountMethods,
  type LinkingDependencies,
} from "../src/account-linking.ts";
function fixture(method = "password") {
  const user = {
    uid: "original-uid",
    email: "current@example.test",
    providerData: [{ providerId: method }],
  } as User;
  const auth = { currentUser: user } as Auth;
  const calls: string[] = [];
  let now = 1000;
  const result = () => ({ user }) as UserCredential;
  const deps: LinkingDependencies = {
    reauthenticatePassword: async (actual, credential) => {
      assert.equal(actual, user);
      assert.equal(credential.providerId, "password");
      calls.push("reauth-password");
      return result();
    },
    reauthenticateGoogle: async (actual, provider, resolver) => {
      assert.equal(actual, user);
      assert.ok(provider instanceof GoogleAuthProvider);
      assert.deepEqual(provider.getScopes().sort(), ["email", "profile"]);
      assert.deepEqual(provider.getCustomParameters(), {
        prompt: "select_account",
      });
      assert.equal(resolver, browserPopupRedirectResolver);
      calls.push("reauth-google");
      return result();
    },
    linkPassword: async (actual, credential) => {
      assert.equal(actual, user);
      assert.equal(credential.providerId, "password");
      calls.push("link-password");
      user.providerData.push({ providerId: "password" } as never);
      return result();
    },
    linkGoogle: async (actual, provider, resolver) => {
      assert.equal(actual, user);
      assert.ok(provider instanceof GoogleAuthProvider);
      assert.equal(resolver, browserPopupRedirectResolver);
      calls.push("link-google");
      user.providerData.push({ providerId: "google.com" } as never);
      return result();
    },
    now: () => now,
  };
  return {
    user,
    auth,
    calls,
    deps,
    advance: () => {
      now += 300001;
    },
    session: () => new AccountLinkSession(auth, user, deps),
  };
}
function code(value: string) {
  return (error: unknown) => (error as { code?: string }).code === value;
}
test("password verification precedes Google linking and keeps the original UID", async () => {
  const f = fixture();
  const session = f.session();
  assert.throws(
    () => session.link("google.com"),
    code("account/reauth-required"),
  );
  const pending = session.reauthenticate("password", "current-secret");
  assert.deepEqual(
    f.calls,
    ["reauth-password"],
    "SDK invocation is in the originating gesture",
  );
  await pending;
  const linked = await session.link("google.com");
  assert.equal(linked.user.uid, "original-uid");
  assert.equal(f.auth.currentUser, f.user);
  assert.deepEqual(accountMethods(f.user), ["password", "google.com"]);
  assert.deepEqual(f.calls, ["reauth-password", "link-google"]);
});
test("Google reauthentication opens immediately before adding password to the same UID", async () => {
  const f = fixture("google.com");
  const session = f.session();
  const pending = session.reauthenticate("google.com");
  assert.deepEqual(f.calls, ["reauth-google"]);
  await pending;
  await session.link("password", "new-secret");
  assert.equal(f.auth.currentUser?.uid, "original-uid");
  assert.deepEqual(f.calls, ["reauth-google", "link-password"]);
});
test("provider collisions reject without retries, sign-in, merge or deletion", async () => {
  for (const collision of [
    "auth/credential-already-in-use",
    "auth/email-already-in-use",
    "auth/account-exists-with-different-credential",
  ]) {
    const f = fixture();
    const error = Object.assign(new Error("secret-provider-details"), {
      code: collision,
    });
    let links = 0;
    f.deps.linkGoogle = async () => {
      links++;
      throw error;
    };
    const session = f.session();
    await session.reauthenticate("password", "current");
    await assert.rejects(
      session.link("google.com"),
      (value) => value === error,
    );
    assert.equal(links, 1);
    assert.equal(f.auth.currentUser, f.user);
    assert.deepEqual(accountMethods(f.user), ["password"]);
    assert.throws(
      () => session.link("google.com"),
      code("account/reauth-required"),
    );
    assert.ok(!accountLinkError(error).includes("secret-provider-details"));
  }
});
test("wrong password and cancelled Google popup never enable linking", async () => {
  for (const [method, errorCode] of [
    ["password", "auth/invalid-credential"],
    ["google.com", "auth/popup-closed-by-user"],
  ] as const) {
    const f = fixture(method);
    const error = Object.assign(new Error("private"), { code: errorCode });
    if (method === "password")
      f.deps.reauthenticatePassword = async () => {
        throw error;
      };
    else
      f.deps.reauthenticateGoogle = async () => {
        throw error;
      };
    const session = f.session();
    await assert.rejects(
      session.reauthenticate(method, "current"),
      (value) => value === error,
    );
    assert.throws(
      () => session.link(method === "password" ? "google.com" : "password"),
      code("account/reauth-required"),
    );
    assert.deepEqual(f.calls, []);
  }
});
test("cancel invalidates completed and late reauthentication tickets", async () => {
  const f = fixture();
  const session = f.session();
  await session.reauthenticate("password", "current");
  session.cancel();
  assert.throws(
    () => session.link("google.com"),
    code("account/reauth-required"),
  );
  let resolve!: (value: UserCredential) => void;
  f.deps.reauthenticatePassword = () =>
    new Promise((done) => {
      resolve = done;
    });
  const pending = session.reauthenticate("password", "current");
  session.cancel();
  resolve({ user: f.user } as UserCredential);
  await assert.rejects(pending, code("account/cancelled"));
  assert.throws(
    () => session.link("google.com"),
    code("account/reauth-required"),
  );
});
test("tickets expire and cannot authorize concurrent operations", async () => {
  const f = fixture();
  const session = f.session();
  const pending = session.reauthenticate("password", "current");
  assert.throws(
    () => session.reauthenticate("password", "current"),
    code("account/busy"),
  );
  await pending;
  f.advance();
  assert.throws(
    () => session.link("google.com"),
    code("account/reauth-required"),
  );
});
test("changed auth session and wrong SDK result UID refuse linking", async () => {
  const f = fixture();
  const session = f.session();
  await session.reauthenticate("password", "current");
  (f.auth as { currentUser: User }).currentUser = {
    ...f.user,
    uid: "other-uid",
  } as User;
  assert.throws(
    () => session.link("google.com"),
    code("account/session-changed"),
  );
  assert.deepEqual(f.calls, ["reauth-password"]);
  const g = fixture();
  g.deps.reauthenticatePassword = async () =>
    ({ user: { ...g.user, uid: "other-uid" } }) as UserCredential;
  const other = g.session();
  await assert.rejects(
    other.reauthenticate("password", "current"),
    code("account/session-changed"),
  );
  assert.throws(
    () => other.link("google.com"),
    code("account/reauth-required"),
  );
});
test("unavailable and already linked methods never reach SDK mutation", async () => {
  const f = fixture();
  const session = f.session();
  assert.throws(
    () => session.reauthenticate("google.com"),
    code("account/method-unavailable"),
  );
  await session.reauthenticate("password", "current");
  assert.throws(
    () => session.link("password", "new"),
    code("auth/provider-already-linked"),
  );
  assert.deepEqual(f.calls, ["reauth-password"]);
});
test("safe errors never render credentials or raw provider details", () => {
  for (const value of [
    new Error("password=secret"),
    { code: "unknown", customData: { credential: "private" } },
    null,
  ])
    assert.equal(
      accountLinkError(value),
      "Could not update sign-in methods. Try again.",
    );
  assert.match(
    accountLinkError({ code: "auth/user-mismatch" }),
    /different account/,
  );
});

test("a resolved SDK promise without the target provider is not reported as success", async () => {
  const f = fixture();
  f.deps.linkGoogle = async () => ({ user: f.user }) as UserCredential;
  const session = f.session();
  await session.reauthenticate("password", "current");
  await assert.rejects(
    session.link("google.com"),
    code("account/provider-not-linked"),
  );
  assert.equal(f.auth.currentUser, f.user);
  assert.deepEqual(accountMethods(f.user), ["password"]);
});
