import assert from "node:assert/strict";
import test from "node:test";

import {
  LatestValueAutosaver,
  ResponseGate,
  type SaveStatus,
} from "../src/session-state.ts";

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

test("undo to previously saved value is persisted after an in-flight edit", async () => {
  const first = deferred();
  const started = deferred();
  const writes: string[] = [];
  const statuses: SaveStatus[] = [];
  let persisted = "B";
  const saver = new LatestValueAutosaver({
    initialValue: "B",
    delayMs: 100_000,
    onStatus: (status) => statuses.push(status),
    save: async (value) => {
      writes.push(value);
      if (value === "A") {
        started.resolve();
        await first.promise;
      }
      persisted = value;
    },
  });
  saver.update("A");
  const finished = saver.flush();
  await started.promise;
  saver.update("B");
  assert.deepEqual(writes, ["A"]);
  assert.notEqual(statuses.at(-1), "saved");
  first.resolve();
  await finished;
  assert.deepEqual(writes, ["A", "B"]);
  assert.equal(persisted, "B");
  assert.equal(saver.getSavedValue(), "B");
  assert.equal(statuses.at(-1), "saved");
  saver.dispose();
});

test("multiple in-flight edits coalesce to the latest and never overlap", async () => {
  const first = deferred();
  const started = deferred();
  const writes: string[] = [];
  let active = 0;
  let maximum = 0;
  const saver = new LatestValueAutosaver({
    initialValue: "initial",
    delayMs: 100_000,
    save: async (value) => {
      active += 1;
      maximum = Math.max(maximum, active);
      writes.push(value);
      if (writes.length === 1) {
        started.resolve();
        await first.promise;
      }
      active -= 1;
    },
  });
  saver.update("A");
  const one = saver.flush();
  await started.promise;
  saver.update("B");
  saver.update("C");
  const two = saver.flush();
  assert.equal(one, two);
  first.resolve();
  await Promise.all([one, two]);
  assert.deepEqual(writes, ["A", "C"]);
  assert.equal(maximum, 1);
  assert.equal(saver.getSavedValue(), "C");
  saver.dispose();
});

test("unsent edits coalesce and a pre-write undo needs no request", async () => {
  const writes: string[] = [];
  const saver = new LatestValueAutosaver({
    initialValue: "original",
    delayMs: 100_000,
    save: async (value) => {
      writes.push(value);
    },
  });
  saver.update("A");
  saver.update("original");
  await saver.flush();
  assert.deepEqual(writes, []);
  saver.update("B");
  saver.update("C");
  await saver.flush();
  assert.deepEqual(writes, ["C"]);
  saver.dispose();
});

test("a failed older request still repairs the latest undo value", async () => {
  const first = deferred();
  const started = deferred();
  const writes: string[] = [];
  let persisted = "B";
  const saver = new LatestValueAutosaver({
    initialValue: "B",
    delayMs: 100_000,
    save: async (value) => {
      writes.push(value);
      persisted = value; // Server writes, but its response can be lost.
      if (writes.length === 1) {
        started.resolve();
        await first.promise;
      }
    },
  });
  saver.update("A");
  const finished = saver.flush();
  await started.promise;
  saver.update("B");
  first.reject(new Error("response lost"));
  await finished;
  assert.deepEqual(writes, ["A", "B"]);
  assert.equal(persisted, "B");
  saver.dispose();
});

test("a failed latest request reports error and an explicit flush retries it", async () => {
  const writes: string[] = [];
  const statuses: SaveStatus[] = [];
  const failure = new Error("offline");
  const saver = new LatestValueAutosaver({
    initialValue: "initial",
    delayMs: 100_000,
    onStatus: (status) => statuses.push(status),
    save: async (value) => {
      writes.push(value);
      if (writes.length === 1) throw failure;
    },
  });
  saver.update("latest");
  await assert.rejects(saver.flush(), failure);
  assert.equal(saver.getSavedValue(), undefined);
  assert.equal(statuses.at(-1), "error");
  await saver.flush();
  assert.deepEqual(writes, ["latest", "latest"]);
  assert.equal(saver.getSavedValue(), "latest");
  assert.equal(statuses.at(-1), "saved");
  saver.dispose();
});

test("dispose cancels pending debounce and suppresses callbacks from sent requests", async () => {
  const sent = deferred();
  const started = deferred();
  const writes: string[] = [];
  const statuses: SaveStatus[] = [];
  const saver = new LatestValueAutosaver({
    initialValue: "original",
    delayMs: 100_000,
    onStatus: (status) => statuses.push(status),
    save: async (value) => {
      writes.push(value);
      started.resolve();
      await sent.promise;
    },
  });
  saver.update("A");
  const done = saver.flush();
  await started.promise;
  saver.dispose();
  const statusCount = statuses.length;
  saver.update("B");
  sent.resolve();
  await done;
  await saver.flush();
  assert.deepEqual(writes, ["A"]);
  assert.equal(statuses.length, statusCount);

  const unsent = new LatestValueAutosaver({
    initialValue: "A",
    delayMs: 100_000,
    save: async () => {
      throw new Error("Disposed debounce must not write");
    },
  });
  unsent.update("B");
  unsent.dispose();
  await unsent.flush();
});

test("saved callbacks can read the acknowledged value", async () => {
  const values: (string | undefined)[] = [];
  const saver = new LatestValueAutosaver({
    initialValue: "A",
    delayMs: 100_000,
    save: async () => {},
    onStatus: (status) => {
      if (status === "saved") values.push(saver.getSavedValue());
    },
  });
  saver.update("B");
  await saver.flush();
  assert.deepEqual(values, ["B"]);
  saver.dispose();
});

test("a newer response supersedes older snapshots within one epoch", () => {
  const gate = new ResponseGate();
  const older = gate.issue();
  const newer = gate.issue();
  assert.equal(gate.accept(newer), true);
  assert.equal(gate.accept(older), false);
  assert.equal(gate.accept(newer), false);
});

test("starting a new execution rejects old polls even with matching sequence", () => {
  const gate = new ResponseGate();
  const oldPoll = gate.issue();
  gate.invalidate();
  const newPoll = gate.issue();
  assert.equal(oldPoll.sequence, newPoll.sequence);
  assert.equal(gate.accept(oldPoll), false);
  assert.equal(gate.accept(newPoll), true);
});

test("slow overlapping polls do not starve valid snapshots", () => {
  const gate = new ResponseGate();
  const first = gate.issue();
  const second = gate.issue();
  assert.equal(gate.accept(first), true);
  const third = gate.issue();
  assert.equal(gate.accept(second), true);
  assert.equal(gate.accept(third), true);
});

test("reset and interrupt each invalidate preceding generations", () => {
  const gate = new ResponseGate();
  const executePoll = gate.issue();
  gate.invalidate(); // reset begins
  const resetPoll = gate.issue();
  gate.invalidate(); // interrupt or a newer action begins
  const current = gate.issue();
  assert.equal(gate.accept(executePoll), false);
  assert.equal(gate.accept(resetPoll), false);
  assert.equal(gate.accept(current), true);
});

test("background opening history and pre-completion polls cannot replace a newly returned run", async () => {
  const gate = new ResponseGate();
  const initialHistory = deferred<string[]>();
  let displayed = ["saved result"];
  const opening = gate.issue();
  const openingDone = initialHistory.promise.then((value) => {
    if (gate.accept(opening)) displayed = value;
  });
  gate.invalidate(); // The editor starts a run while opening history is pending.
  const duringRun = gate.issue();
  gate.invalidate(); // The execution response returns its new record.
  displayed = ["saved result", "new result"];
  initialHistory.resolve(["saved result"]);
  await openingDone;
  assert.equal(gate.accept(duringRun), false);
  assert.deepEqual(displayed, ["saved result", "new result"]);
  const afterRun = gate.issue();
  assert.equal(gate.accept(afterRun), true);
});

test("launch acceptance invalidates an earlier idle poll before the durable running lock is shown", async () => {
  const gate = new ResponseGate();
  let running = true;
  let history = ["existing result"];
  const beforeLaunch = gate.issue();
  const idleResponse = deferred<{ running: boolean; history: string[] }>();
  const pending = idleResponse.promise.then((state) => {
    if (!gate.accept(beforeLaunch)) return;
    running = state.running;
    history = state.history;
  });
  gate.invalidate(); // 202 accepted: polls begun before launch cannot clear the lock.
  running = true;
  idleResponse.resolve({ running: false, history: [] });
  await pending;
  assert.equal(running, true);
  assert.deepEqual(history, ["existing result"]);
  const completed = gate.issue();
  assert.equal(gate.accept(completed), true);
  running = false;
  history = ["existing result", "validated result"];
  assert.equal(running, false);
  assert.deepEqual(history, ["existing result", "validated result"]);
});
