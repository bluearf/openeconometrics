import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import type {
  Auth,
  UserCredential,
  signInWithCustomToken,
} from "firebase/auth";
import { EmailCodeSession } from "../src/email-code.ts";

const auth = {} as Auth;
const requestId = "a".repeat(32);
const sent = {
  request_id: requestId,
  expires_at: new Date(Date.now() + 600000).toISOString(),
  retry_after: 60,
};

test("email sign-in sends the code and client proof, then exchanges only the verified token with Firebase", async () => {
  const calls: Array<{ path: string; body: Record<string, string> }> = [];
  let tokens: string[] = [];
  const session = new EmailCodeSession(
    auth,
    async (path, body) => {
      calls.push({ path, body });
      return calls.length === 1 ? sent : { custom_token: "synthetic-token" };
    },
    (async (actual, token) => {
      assert.equal(actual, auth);
      tokens.push(token);
      return {} as UserCredential;
    }) as typeof signInWithCustomToken,
  );
  assert.equal(await session.send(" person@example.test "), 60);
  assert.deepEqual(tokens, []);
  assert.deepEqual(Object.keys(calls[0].body).sort(), ["challenge", "email"]);
  assert.equal(calls[0].body.email, "person@example.test");
  await session.verify("012345");
  assert.equal(calls[1].path, `/api/auth/email-code/${requestId}/verify`);
  assert.deepEqual(Object.keys(calls[1].body).sort(), ["code", "verifier"]);
  assert.equal(
    createHash("sha256").update(calls[1].body.verifier).digest("hex"),
    calls[0].body.challenge,
  );
  assert.deepEqual(tokens, ["synthetic-token"]);
  await assert.rejects(session.verify("012345"));
  assert.equal(calls.length, 2);
});

test("cancelled or stale responses cannot sign in, and proofs are discarded when the screen closes", async () => {
  let finish!: (value: unknown) => void;
  let tokenCalls = 0;
  let lastSignal: AbortSignal | undefined;
  const session = new EmailCodeSession(
    auth,
    async (_path, _body, signal) => {
      lastSignal = signal;
      return new Promise((resolve) => {
        finish = resolve;
      });
    },
    (async () => {
      tokenCalls++;
      return {} as UserCredential;
    }) as typeof signInWithCustomToken,
  );
  const pending = session.send("person@example.test");
  // Wait for actual request admission, not a clock delay.
  while (!finish) await new Promise((resolve) => setImmediate(resolve));
  session.cancel();
  assert.equal(lastSignal!.aborted, true);
  finish(sent);
  await assert.rejects(pending, { name: "AbortError" });
  await assert.rejects(session.verify("123456"));
  assert.equal(tokenCalls, 0);
});

test("wrong/expired code remains an error and never calls Firebase; resending uses a new proof", async () => {
  const challenges: string[] = [];
  let tokens = 0;
  const session = new EmailCodeSession(
    auth,
    async (path, body) => {
      if (path.endsWith("/verify"))
        throw new Error("The code is incorrect or has expired.");
      challenges.push(body.challenge);
      return sent;
    },
    (async () => {
      tokens++;
      return {} as UserCredential;
    }) as typeof signInWithCustomToken,
  );
  await session.send("person@example.test");
  await assert.rejects(session.verify("123456"), /incorrect or has expired/);
  await session.send("person@example.test");
  assert.notEqual(challenges[0], challenges[1]);
  assert.equal(tokens, 0);
});

test("invalid server acknowledgements and malformed codes cannot mint a session", async () => {
  for (const response of [
    null,
    {},
    { ...sent, request_id: "../foreign" },
    { ...sent, retry_after: 0 },
    { ...sent, expires_at: "bad" },
  ]) {
    const session = new EmailCodeSession(auth, async () => response);
    await assert.rejects(
      session.send("person@example.test"),
      /could not be verified/,
    );
  }
  const session = new EmailCodeSession(auth, async () => sent);
  await session.send("person@example.test");
  for (const code of ["12345", "1234567", "１２３４５６", "12345\n"])
    await assert.rejects(session.verify(code));
  session.cancel();
});
