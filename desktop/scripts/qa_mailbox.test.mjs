import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import test from 'node:test';
import { seal, unseal, validateContext } from './qa_mailbox.mjs';
const keys = generateKeyPairSync('rsa', { modulusLength: 3072 });
const context = { run_id: '38153397180', nonce: 'a'.repeat(64), challenge: 'b'.repeat(32), kind: 'email-code', role: 'owner' };
const now = Date.now();
const payload = { email: 'anilsen+openecon-qa-owner-abcdef012345@openeconometrics.com', code: '314159', expires_at: now + 60000 };
test('encrypted one-run delivery authenticates every binding and keeps its payload out of the packet', () => {
  const packet = seal(keys.publicKey, context, payload);
  assert.deepEqual(unseal(keys.privateKey, context, packet, now), payload);
  for (const secret of [payload.code, payload.email]) assert.equal(JSON.stringify(packet).includes(secret), false);
  for (const changed of [{ ...context, run_id: '38153397181' }, { ...context, nonce: 'c'.repeat(64) },
    { ...context, challenge: 'd'.repeat(32) }, { ...context, role: 'viewer' }])
    assert.throws(() => unseal(keys.privateKey, changed, packet, now), /invalid, expired/);
});
test('tampered, truncated, expired, oversized and malformed deliveries refuse without exposing the code', () => {
  const packet = seal(keys.publicKey, context, payload);
  for (const field of ['encrypted_key', 'ciphertext', 'iv', 'tag']) {
    const bad = structuredClone(packet); bad[field] = Buffer.alloc(9).toString('base64');
    assert.throws(() => unseal(keys.privateKey, context, bad, now), error => !error.message.includes(payload.code));
  }
  assert.throws(() => unseal(keys.privateKey, context, packet, payload.expires_at));
  assert.throws(() => unseal(keys.privateKey, context, seal(keys.publicKey, context, { ...payload, expires_at: now + 1200001 }), now));
  assert.throws(() => unseal(keys.privateKey, context, seal(keys.publicKey, context, { ...payload, code: '12345' }), now));
  assert.throws(() => seal(keys.publicKey, context, { text: 'x'.repeat(32768) }), /bound/);
  for (const bad of [{ ...context, extra: true }, { ...context, nonce: 'bad' }, { ...context, kind: 'bootstrap' },
    { ...context, role: 'fixture' }, { ...context, challenge: 'bootstrap' }]) assert.throws(() => validateContext(bad));
});
test('bootstrap has a distinct authenticated namespace from email challenges', () => {
  const bootstrap = { ...context, challenge: 'bootstrap', kind: 'bootstrap', role: 'fixture' };
  const fixture = { projects: ['e'.repeat(32)], expires_at: now + 60000 };
  const packet = seal(keys.publicKey, bootstrap, fixture);
  assert.deepEqual(unseal(keys.privateKey, bootstrap, packet, now), fixture);
  assert.throws(() => unseal(keys.privateKey, context, packet, now));
});
