import assert from 'node:assert/strict';
import test from 'node:test';
import { resolvePlot } from '../src/plot-artifacts.ts';

const inline = { kind: 'network', title: 'Study', sample_n: 10000, total_n: 10000, dropped_n: 0 };
const referenced = { ...inline, artifact: { version: 1 as const, id: 'a'.repeat(64), bytes: 300000 } };

test('inline plots avoid IO, lazy graphs capture the project client and abort signal', async () => {
  const controller = new AbortController();
  assert.equal(await resolvePlot(inline, undefined, controller.signal), inline);
  const seen: unknown[] = [];
  const client = { request: async (path: string, options: RequestInit) => { seen.push(path, options.signal); return inline; } } as any;
  assert.equal(await resolvePlot(referenced, client, controller.signal), inline);
  assert.deepEqual(seen, ['/console/plots/' + 'a'.repeat(64), controller.signal]);
});

test('local graph references cannot fetch URLs or become false history metadata', async () => {
  const signal = new AbortController().signal;
  let called = false;
  const client = { request: async () => { called = true; return { ...inline, sample_n: 3 }; } } as any;
  await assert.rejects(resolvePlot({ ...referenced, artifact: { ...referenced.artifact, id: '../secret' } }, client, signal), /verify/);
  assert.equal(called, false);
  await assert.rejects(resolvePlot(referenced, client, signal), /history/);
  await assert.rejects(resolvePlot(referenced, undefined, signal), /original local project/);
});

test('late graph fetch cannot mount after switching project or unmounting', async () => {
  const controller = new AbortController();
  const client = { request: async () => { controller.abort(); return inline; } } as any;
  await assert.rejects(resolvePlot(referenced, client, controller.signal), error => error instanceof DOMException && error.name === 'AbortError');
});
