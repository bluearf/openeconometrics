import assert from "node:assert/strict";
import test from "node:test";
import type { User } from "firebase/auth";
import {
  VerificationDelivery,
  VerificationCheckGate,
  finishSignupVerification,
  verificationErrorMessage,
} from "../src/email-verification.ts";
const account = (uid: string) => ({ uid, email: `${uid}@example.com` }) as User;
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

test("signup send failure survives the signed-out screen disappearing and later profile refreshes", async () => {
  const request = deferred<void>();
  const delivery = new VerificationDelivery(() => request.promise);
  const unsubscribe = delivery.subscribe(() => {});
  const pending = delivery.send(account("a"));
  unsubscribe(); // AuthScreen has gone; TeamShell retains the delivery controller.
  request.reject(
    Object.assign(new Error("sensitive provider diagnostic"), {
      code: "auth/network-request-failed",
    }),
  );
  await pending;
  let snapshot = delivery.getSnapshot("a");
  assert.equal(snapshot.status, "error");
  assert.match(snapshot.message, /Could not send/);
  assert.ok(!snapshot.message.includes("sensitive provider diagnostic"));
  const stop = delivery.subscribe(() => {
    snapshot = delivery.getSnapshot("a");
  });
  // Reading/refetching a user profile cannot erase this independent state.
  assert.equal(delivery.getSnapshot("a"), snapshot);
  assert.equal(snapshot.status, "error");
  stop();
});

test("display-name failure does not suppress initial signup email", async () => {
  let sent = 0;
  const delivery = new VerificationDelivery(async () => {
    sent++;
  });
  const nameFailure = new Error("name update failed");
  const result = await finishSignupVerification(
    account("a"),
    " Ada ",
    delivery,
    async (_user, name) => {
      assert.equal(name, "Ada");
      throw nameFailure;
    },
  );
  assert.equal(sent, 1);
  assert.equal(result.mail.status, "sent");
  assert.equal(result.profileError, nameFailure);
});

test("same-turn signup send and resend share one SDK operation even if name update is pending", async () => {
  let sent = 0;
  const request = deferred<void>();
  const saveName = deferred<void>();
  const delivery = new VerificationDelivery(() => {
    sent++;
    return request.promise;
  });
  const signup = finishSignupVerification(
    account("a"),
    "Ada",
    delivery,
    () => saveName.promise,
  );
  const resend = delivery.send(account("a"));
  const secondClick = delivery.send(account("a"));
  assert.equal(resend, secondClick);
  await Promise.resolve();
  assert.equal(sent, 1);
  request.resolve();
  saveName.resolve();
  await signup;
  await resend;
  assert.equal(sent, 1);
});

test("a different account can send while the first is pending, and old completion cannot change its state", async () => {
  const requests = { a: deferred<void>(), b: deferred<void>() };
  const delivery = new VerificationDelivery(
    (user) => requests[user.uid as "a" | "b"].promise,
  );
  const first = delivery.send(account("a"));
  const second = delivery.send(account("b"));
  const bSending = delivery.getSnapshot("b");
  requests.a.reject(new Error("old account error"));
  await first;
  assert.equal(delivery.getSnapshot("b"), bSending);
  assert.equal(delivery.getSnapshot("b").status, "sending");
  requests.b.resolve();
  await second;
  assert.equal(delivery.getSnapshot("b").status, "sent");
  assert.equal(delivery.getSnapshot("a").status, "error");
});

test("cooldown starts before sending, blocks retries, and quota rejection extends it", async () => {
  let now = 1_000;
  let calls = 0;
  const delivery = new VerificationDelivery(
    async () => {
      calls++;
      throw { code: "auth/too-many-requests" };
    },
    { now: () => now },
  );
  const pending = delivery.send(account("a"));
  assert.equal(delivery.getSnapshot("a").retryAt, 61_000);
  await pending;
  assert.equal(delivery.getSnapshot("a").retryAt, 121_000);
  await delivery.send(account("a"));
  assert.equal(calls, 1);
  now = 121_000;
  await delivery.send(account("a"));
  assert.equal(calls, 2);
});

test("controller construction, subscriptions and account reads never automatically send mail", () => {
  let sent = 0;
  const delivery = new VerificationDelivery(async () => {
    sent++;
  });
  const stop = delivery.subscribe(() => {});
  assert.equal(delivery.getSnapshot("a").status, "idle");
  assert.equal(delivery.getSnapshot("b").status, "idle");
  stop();
  assert.equal(sent, 0);
});

test("verification checks reject duplicate clicks and an A→B→A switch does not retain A's busy lock", () => {
  const gate = new VerificationCheckGate();
  const firstA = gate.begin("a")!;
  assert.equal(gate.begin("a"), null);
  gate.select("b");
  const firstB = gate.begin("b")!;
  assert.equal(gate.finish(firstA), false);
  assert.equal(
    gate.begin("b"),
    null,
    "old A must not release B's in-flight lock",
  );
  gate.select("a");
  const nextA = gate.begin("a");
  assert.notEqual(nextA, null);
  assert.equal(gate.current(firstA), false);
  assert.equal(gate.current(firstB), false);
  assert.equal(gate.finish(nextA!), true);
  assert.notEqual(gate.begin("a"), null);
});

test("verification error messages never expose raw SDK errors or credential details", () => {
  assert.match(
    verificationErrorMessage({ code: "auth/user-token-expired" }),
    /sign in again/,
  );
  assert.ok(
    !verificationErrorMessage(new Error("private-debug-token")).includes(
      "private-debug-token",
    ),
  );
  assert.ok(
    !verificationErrorMessage({
      code: "unknown",
      customData: { email: "private@example.com" },
    }).includes("private@example.com"),
  );
});
