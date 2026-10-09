import assert from "node:assert/strict";
import test from "node:test";
import type { Auth, User } from "firebase/auth";
import {
  linkDesktopAccount,
  type NativeLinkDependencies,
} from "../src/native-account-link.ts";
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function fixture(response?: Promise<unknown>) {
  const user = {
    uid: "same-uid",
    providerData: [{ providerId: "google.com" }],
  } as User;
  const auth = { currentUser: user } as Auth;
  const paths: string[] = [];
  const tokens: boolean[] = [];
  let opened = 0;
  let reloaded = 0;
  const deps: NativeLinkDependencies = {
    token: async (actual, force) => {
      assert.equal(actual, user);
      tokens.push(!!force);
      return "synthetic-id-token";
    },
    reload: async (actual) => {
      assert.equal(actual, user);
      reloaded++;
      user.providerData.push({ providerId: "password" } as never);
    },
    request: async <T>(
      path: string,
      token: string,
      options?: RequestInit,
    ): Promise<T> => {
      paths.push(path);
      assert.equal(token, "synthetic-id-token");
      if (options?.method === "DELETE") return { cancelled: true } as T;
      const body = JSON.parse(String(options?.body));
      if (path === "/api/desktop/account-link") {
        assert.deepEqual(Object.keys(body).sort(), ["challenge", "target"]);
        assert.equal(body.target, "password");
        assert.match(body.challenge, /^[0-9a-f]{64}$/);
        return { request_id: "grant", code: "ABC" } as T;
      }
      assert.deepEqual(Object.keys(body), ["verifier"]);
      assert.match(body.verifier, /^[0-9a-f]{64}$/);
      return (
        response
          ? await response
          : { linked: true, uid: "same-uid", target: "password" }
      ) as T;
    },
    open: async <T>(command: string, args?: Record<string, unknown>) => {
      assert.equal(command, "open_desktop_login");
      assert.deepEqual(args, { requestId: "grant" });
      opened++;
      return undefined as T;
    },
    intervalMs: 1,
    timeoutMs: 1000,
  };
  return {
    user,
    auth,
    deps,
    paths,
    tokens,
    opened: () => opened,
    reloaded: () => reloaded,
  };
}
test("native linking refreshes its existing user, never signs in or transports credentials", async () => {
  const f = fixture();
  await linkDesktopAccount(
    f.auth,
    f.user,
    "password",
    new AbortController().signal,
    undefined,
    f.deps,
  );
  assert.equal(f.auth.currentUser, f.user);
  assert.equal(f.reloaded(), 1);
  assert.deepEqual(f.tokens, [false, true]);
  assert.equal(f.opened(), 1);
  assert.deepEqual(f.paths, [
    "/api/desktop/account-link",
    "/api/desktop/account-link/grant/exchange",
    "/api/desktop/account-link/grant",
  ]);
});
test("a browser acknowledgement does not count as a real linked provider", async () => {
  const f = fixture();
  f.deps.reload = async () => {};
  await assert.rejects(
    linkDesktopAccount(
      f.auth,
      f.user,
      "password",
      undefined,
      undefined,
      f.deps,
    ),
    /not linked/,
  );
  assert.deepEqual(f.tokens, [false]);
  assert.equal(f.auth.currentUser, f.user);
});
test("another UID or provider response cannot authorize success", async () => {
  for (const response of [
    { linked: true, uid: "different-uid", target: "password" },
    { linked: true, uid: "same-uid", target: "google.com" },
    { custom_token: "must-not-be-used" },
  ]) {
    const f = fixture(Promise.resolve(response));
    await assert.rejects(
      linkDesktopAccount(
        f.auth,
        f.user,
        "password",
        undefined,
        undefined,
        f.deps,
      ),
    );
    assert.equal(f.reloaded(), 0);
    assert.equal(f.auth.currentUser, f.user);
    assert.deepEqual(f.tokens, [false]);
  }
});
test("abort revokes the handoff immediately and ignores late browser completion", async () => {
  const response = deferred<unknown>();
  const f = fixture(response.promise);
  const controller = new AbortController();
  const pending = linkDesktopAccount(
    f.auth,
    f.user,
    "password",
    controller.signal,
    undefined,
    f.deps,
  );
  const outcome = assert.rejects(pending, { name: "AbortError" });
  while (f.paths.length < 2) await new Promise((done) => setTimeout(done, 1));
  controller.abort();
  await outcome;
  response.resolve({ linked: true, uid: "same-uid", target: "password" });
  await new Promise((done) => setTimeout(done, 5));
  assert.ok(f.paths.includes("/api/desktop/account-link/grant"));
  assert.equal(f.reloaded(), 0);
  assert.deepEqual(f.tokens, [false]);
});
test("auth changes during browser approval cannot refresh the new user", async () => {
  const response = deferred<unknown>();
  const f = fixture(response.promise);
  const pending = linkDesktopAccount(
    f.auth,
    f.user,
    "password",
    undefined,
    undefined,
    f.deps,
  );
  const outcome = assert.rejects(
    pending,
    (error: unknown) =>
      (error as { code: string }).code === "account/session-changed",
  );
  while (f.paths.length < 2) await new Promise((done) => setTimeout(done, 1));
  (f.auth as { currentUser: User }).currentUser = {
    uid: "different-uid",
  } as User;
  response.resolve({ linked: true, uid: "same-uid", target: "password" });
  await outcome;
  assert.equal(f.reloaded(), 0);
});
