import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { link, mkdir, mkdtemp, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import vm from 'node:vm';
import test from 'node:test';
import { EXECUTION_CODE, EXECUTION_COMMAND, EXPECTED_STDOUT, canonicalWorkspaceIdentity, fixturePresentationMatches, fixtureRecordMatches,
  nativeIdentitySummary, observedIdentityReceipt, observerExpression, offlineIdentityReceipt, parseArguments,
  payloadDigestExpression, readAdmittedQaManifest, requireSameWorkspaceIdentity, requireSeparateWorkspaceIdentity, reserveReceiptOutput, selectProfileIdentity, trustedTarget,
  validateBaseline, validateControllerPaths, validateOfflineIdentityReceipt, validateOnlineIdentityReceipt, validatePhaseWorkspaceIdentity, validateQaManifest, workspaceIdentityFromFacts } from './verify_windows_cloud_ui.mjs';

const projectId = '1'.repeat(32);
const owner = { uid: 'owned-qa-owner', email: 'openecon-qa-owner-abcdef012345@example.com', password: 'private-password-sentinel' };
const viewer = { uid: 'owned-qa-viewer', email: 'openecon-qa-viewer-abcdef012345@example.com', password: 'other-private-password-sentinel' };
const options = { 'project-id': projectId };
const hash = value => createHash('sha256').update(value).digest('hex');
const profileDigest = 'd'.repeat(64);
const workspaceIdentity = { schema: 1, verification: 'canonical_existing_directory',
  canonical_path_sha256: 'e'.repeat(64), filesystem_id_sha256: 'f'.repeat(64) };
const online = {
  status: 'passed', phase: 'password-open', project_id: projectId,
  native_identity: { profile_sha256: profileDigest, workspace_identity: workspaceIdentity },
  authenticated_identity: { ...observedIdentityReceipt({ sequence: 2, status: 200,
    identity: { uid: owner.uid, email: owner.email, project_id: projectId, role: 'owner' } }, owner, projectId, 'owner', 1),
    initial_profile_request_sequence: 1 },
  observed_execution_requests: { local_executions: 0, cloud_executions: 0 },
  checks: { native_cloud_reachable: true, fresh_authenticated_uid_and_live_project_role: true,
    canonical_workspace_identity_before_credentials: true, workspace_identity_rechecked_after_phase: true,
    fresh_authenticated_uid_and_live_project_role_after_phase: true },
};
const baseline = {
  status: 'passed', phase: 'offline-run', project_id: projectId,
  native_identity: { profile_sha256: profileDigest, workspace_identity: workspaceIdentity },
  authenticated_identity: offlineIdentityReceipt(online, owner, projectId, workspaceIdentity, 'c'.repeat(64)),
  original: { id: 'local-run-id', payload_sha256: 'a'.repeat(64), raw_payload_sha256: 'b'.repeat(64), history_matches: 1, nobs: 480,
    marker_occurrences: 1, execution_count: 1, status: 'ok', latex_present: true, fixture_protocol_valid: true,
    sharing_state: 'pending', outbox_matches: 1 },
  observed_execution_requests: { local_executions: 1, cloud_executions: 0 },
  checks: { native_cloud_unreachable: true, offline_cached_owner_bound_to_original_online_identity: true,
    canonical_workspace_identity_before_credentials: true, workspace_identity_rechecked_after_phase: true },
};

test('native observer preserves frozen Tauri exports and exact IPC/fetch behavior then restores originals', async () => {
  const calls = [];
  const response = { status: 200, body: { request_id: '3'.repeat(32) } };
  const invoke = async (...args) => { calls.push(args); return response; };
  const originalCore = Object.freeze({ invoke, existingExport: 'preserved' });
  const originalFetch = async (...args) => { calls.push(args); return response; };
  const window = { __TAURI__: { core: originalCore }, fetch: originalFetch };
  const context = vm.createContext({ window, URL, Request, location: { href: 'http://127.0.0.1:49100/' } });
  assert.equal(vm.runInContext(observerExpression(projectId), context), true);
  assert.equal(Object.isFrozen(window.__TAURI__.core), true);
  assert.equal(window.__TAURI__.core.existingExport, 'preserved');
  const args = { method: 'POST', path: '/api/desktop/login' };
  const ipcOptions = { headers: { owned: 'preserved' } };
  assert.equal(await window.__TAURI__.core.invoke('cloud_request', args, ipcOptions), response);
  assert.deepEqual(calls[0], ['cloud_request', args, ipcOptions]);
  assert.equal(window.__market87Observer.grant, response.body.request_id);
  const request = new Request(`http://127.0.0.1:49100/api/desktop/projects/${projectId}/workspace/desktop-console/execute`, { method: 'POST' });
  assert.equal(await window.fetch(request), response);
  assert.equal(calls[1][0], request);
  assert.equal(window.__market87Observer.local_executions, 1);
  window.__market87Observer.restore();
  assert.equal(window.__TAURI__.core, originalCore);
  assert.equal(window.fetch, originalFetch);
  assert.equal(window.__market87Observer, undefined);
  assert.equal(originalCore.invoke, invoke);
});

test('native observer refuses an immutable containing namespace without touching transport', () => {
  const core = Object.freeze({ invoke: async () => {} });
  const fetch = async () => {};
  const window = { __TAURI__: Object.freeze({ core }), fetch };
  assert.throws(() => vm.runInNewContext(observerExpression(projectId), { window }), /not writable/);
  assert.equal(window.__TAURI__.core, core);
  assert.equal(window.fetch, fetch);
  assert.equal(window.__market87Observer, undefined);
});

function observedWindow(invoke) {
  const window = { __TAURI__: { core: Object.freeze({ invoke }) }, fetch: async () => {} };
  vm.runInNewContext(observerExpression(projectId), { window, URL, Request, location: { href: 'http://127.0.0.1:49100/' } });
  return window;
}

function actualProfile(user = owner, role = 'owner') {
  return { user: { uid: user.uid, email: user.email, name: 'private unused name', email_verified: true },
    projects: [{ id: projectId, role, name: 'owned project' }, { id: '2'.repeat(32), role: 'owner', secret: 'unused project sentinel' }],
    invitations: [{ secret: 'unused invitation sentinel' }], token: 'unused response token sentinel' };
}

test('actual successful native profile response retains only UID/email and owned live role without changing IPC', async () => {
  const response = { status: 200, body: actualProfile() };
  const calls = [];
  const window = observedWindow(async (...args) => { calls.push(args); return response; });
  const args = { method: 'GET', path: '/api/me', token: 'private native token sentinel', body: null };
  const ipcOptions = { original: 'preserved' };
  assert.equal(await window.__TAURI__.core.invoke('cloud_request', args, ipcOptions), response);
  assert.deepEqual(calls[0], ['cloud_request', args, ipcOptions]);
  const observation = JSON.parse(JSON.stringify(window.__market87Observer.profile));
  assert.deepEqual(observation, { sequence: 1, status: 200,
    identity: { uid: owner.uid, email: owner.email, project_id: projectId, role: 'owner' } });
  const receipt = observedIdentityReceipt(observation, owner, projectId, 'owner');
  assert.equal(receipt.uid_sha256, hash(owner.uid));
  for (const secret of [args.token, response.body.token, response.body.user.name,
    response.body.projects[1].secret, response.body.invitations[0].secret])
    assert.equal(JSON.stringify(window.__market87Observer).includes(secret), false);
  assert.equal(JSON.stringify(receipt).includes(owner.uid), false);
  assert.equal(JSON.stringify(receipt).includes(owner.email), false);
  for (const wrongRequest of [{ method: 'POST', path: '/api/me' }, { method: 'GET', path: '/api/me?cached=1' },
    { method: 'GET', path: `/api/projects/${projectId}/workspace/session` }])
    await window.__TAURI__.core.invoke('cloud_request', wrongRequest);
  assert.equal(window.__market87Observer.profile_requests, 1);
});

test('visible matching email cannot certify a different UID, missing or duplicated owned project, or wrong live role', () => {
  const valid = selectProfileIdentity(actualProfile(), projectId);
  assert.equal(observedIdentityReceipt({ sequence: 3, status: 200, identity: valid }, owner, projectId, 'owner', 2).live_verification, true);
  for (const changed of [
    { ...valid, uid: viewer.uid }, { ...valid, email: viewer.email }, { ...valid, role: 'viewer' },
    { ...valid, project_id: '2'.repeat(32) }, null,
  ]) assert.throws(() => observedIdentityReceipt({ sequence: 3, status: 200, identity: changed }, owner, projectId, 'owner', 2));
  for (const mutate of [
    value => { value.projects = value.projects.slice(1); },
    value => { value.projects.push({ ...value.projects[0] }); },
    value => { value.projects[0].role = 'admin'; },
    value => { value.user.uid = 'unsafe/uid'; },
    value => { value.user.email += '\n'; },
  ]) {
    const profile = actualProfile(); mutate(profile);
    assert.equal(selectProfileIdentity(profile, projectId), null);
  }
  for (const observation of [
    { sequence: 2, status: 200, identity: valid }, { sequence: 3, status: 401, identity: valid },
    { sequence: 3, status: 'pending', identity: valid }, { sequence: 3, status: 'failed', identity: valid },
  ]) assert.throws(() => observedIdentityReceipt(observation, owner, projectId, 'owner', 2));
});

test('out-of-order profile responses and a latest denial cannot revive an older authenticated identity', async () => {
  const pending = [];
  const window = observedWindow(() => new Promise((resolve, reject) => pending.push({ resolve, reject })));
  const args = { method: 'GET', path: '/api/me', token: 'private unused token' };
  const first = window.__TAURI__.core.invoke('cloud_request', args);
  const second = window.__TAURI__.core.invoke('cloud_request', args);
  pending[1].resolve({ status: 200, body: actualProfile(viewer, 'viewer') });
  await second;
  pending[0].resolve({ status: 200, body: actualProfile() });
  await first;
  assert.equal(window.__market87Observer.profile.sequence, 2);
  assert.equal(window.__market87Observer.profile.identity.uid, viewer.uid);
  assert.throws(() => observedIdentityReceipt(window.__market87Observer.profile, owner, projectId, 'owner'));
  const older = window.__TAURI__.core.invoke('cloud_request', args);
  const latest = window.__TAURI__.core.invoke('cloud_request', args);
  pending[3].resolve({ status: 403, body: { user: actualProfile().user, private: 'denial body sentinel' } });
  await latest;
  pending[2].resolve({ status: 200, body: actualProfile() });
  await older;
  assert.equal(window.__market87Observer.profile.status, 403);
  assert.equal(window.__market87Observer.profile.identity, null);
  assert.throws(() => observedIdentityReceipt(window.__market87Observer.profile, owner, projectId, 'owner'));
  const failed = window.__TAURI__.core.invoke('cloud_request', args);
  const rejection = new Error('private transport error sentinel');
  pending[4].reject(rejection);
  await assert.rejects(failed, error => error === rejection);
  assert.equal(window.__market87Observer.profile.status, 'failed');
  assert.equal(window.__market87Observer.profile.identity, null);
  assert.equal(JSON.stringify(window.__market87Observer).includes(rejection.message), false);
});

test('offline cached binding requires immutable original online actor, exact project and native profile', () => {
  assert.equal(validateOnlineIdentityReceipt(online, projectId, owner, workspaceIdentity), online);
  const identity = offlineIdentityReceipt(online, owner, projectId, workspaceIdentity, 'c'.repeat(64));
  assert.equal(identity.live_verification, false);
  assert.equal(identity.verification, 'offline_cached');
  assert.equal(identity.online_receipt_sha256, 'c'.repeat(64));
  assert.deepEqual(validateOfflineIdentityReceipt(baseline, projectId, owner, workspaceIdentity), identity);
  for (const mutate of [
    value => { value.status = 'error'; }, value => { value.phase = 'offline-run'; },
    value => { value.project_id = '2'.repeat(32); }, value => { value.native_identity.workspace_identity.canonical_path_sha256 = 'a'.repeat(64); },
    value => { value.native_identity.workspace_identity.filesystem_id_sha256 = 'a'.repeat(64); },
    value => { delete value.native_identity.workspace_identity; },
    value => { value.checks.workspace_identity_rechecked_after_phase = false; },
    value => { value.authenticated_identity.uid_sha256 = hash(viewer.uid); },
    value => { value.authenticated_identity.email_sha256 = hash(viewer.email); },
    value => { value.authenticated_identity.role = 'viewer'; },
    value => { value.authenticated_identity.live_verification = false; },
    value => { value.authenticated_identity.source = 'cached profile'; },
    value => { value.authenticated_identity.profile_request_sequence = 0; },
    value => { value.authenticated_identity.initial_profile_request_sequence = 0; },
    value => { value.authenticated_identity.initial_profile_request_sequence = value.authenticated_identity.profile_request_sequence; },
    value => { value.observed_execution_requests.local_executions = 1; },
    value => { value.checks.fresh_authenticated_uid_and_live_project_role_after_phase = false; },
    value => { value.checks.native_cloud_reachable = false; },
  ]) {
    const changed = structuredClone(online); mutate(changed);
    assert.throws(() => offlineIdentityReceipt(changed, owner, projectId, workspaceIdentity, 'c'.repeat(64)));
  }
  for (const mutate of [
    value => { value.authenticated_identity.live_verification = true; },
    value => { value.authenticated_identity.online_receipt_sha256 = ''; },
    value => { value.authenticated_identity.uid_sha256 = hash(viewer.uid); },
    value => { value.authenticated_identity.role = 'viewer'; },
    value => { value.native_identity.workspace_identity.canonical_path_sha256 = 'a'.repeat(64); },
    value => { value.native_identity.workspace_identity.filesystem_id_sha256 = 'a'.repeat(64); },
    value => { delete value.native_identity.workspace_identity; },
    value => { value.checks.canonical_workspace_identity_before_credentials = false; },
  ]) {
    const changed = structuredClone(baseline); mutate(changed);
    assert.throws(() => validateOfflineIdentityReceipt(changed, projectId, owner, workspaceIdentity));
  }
});

test('existing workspace roots resolve parent aliases and require real stable directory identities', async t => {
  const root = await mkdtemp(join(tmpdir(), 'market87-root-identity-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const ownerRoot = join(root, 'owner'), viewerRoot = join(root, 'viewer');
  await mkdir(ownerRoot); await mkdir(viewerRoot);
  const ownerIdentity = await canonicalWorkspaceIdentity(ownerRoot);
  const viewerIdentity = await canonicalWorkspaceIdentity(viewerRoot);
  assert.equal(requireSeparateWorkspaceIdentity(ownerIdentity, viewerIdentity), true);
  assert.equal(requireSameWorkspaceIdentity(ownerIdentity, await canonicalWorkspaceIdentity(join(ownerRoot, '..', 'owner'))), true);
  const alias = join(root, 'parent-alias');
  await symlink(root, alias, 'junction');
  const aliasIdentity = await canonicalWorkspaceIdentity(join(alias, 'owner'));
  assert.equal(requireSameWorkspaceIdentity(ownerIdentity, aliasIdentity), true);
  assert.throws(() => requireSeparateWorkspaceIdentity(ownerIdentity, aliasIdentity), /distinct canonical/);
  assert.equal(JSON.stringify(ownerIdentity).includes(root), false);
  await assert.rejects(canonicalWorkspaceIdentity(join(root, 'absent')));
  const file = join(root, 'regular-file'); await writeFile(file, 'not a directory');
  await assert.rejects(canonicalWorkspaceIdentity(file), /existing workspace directory/);
  for (const facts of [
    { directory: true, dev: 0n, ino: 2n }, { directory: true, dev: 1n, ino: 0n },
    { directory: true, dev: 1, ino: 2 }, { directory: false, dev: 1n, ino: 2n },
  ]) assert.throws(() => workspaceIdentityFromFacts(ownerRoot, facts), /stable nonzero/);
});

test('owner phases preserve both original root digests while viewer refuses either independent alias', () => {
  const originalBytes = JSON.stringify(baseline);
  for (const phase of ['browser-approval', 'offline-run', 'offline-reopen', 'retry-online'])
    assert.equal(validatePhaseWorkspaceIdentity(phase, workspaceIdentity, baseline), true);
  const separate = { ...workspaceIdentity, canonical_path_sha256: '1'.repeat(64), filesystem_id_sha256: '2'.repeat(64) };
  assert.equal(validatePhaseWorkspaceIdentity('viewer-readback', separate, baseline), true);
  for (const current of [
    workspaceIdentity,
    { ...separate, canonical_path_sha256: workspaceIdentity.canonical_path_sha256 },
    { ...separate, filesystem_id_sha256: workspaceIdentity.filesystem_id_sha256 },
  ]) {
    assert.throws(() => validatePhaseWorkspaceIdentity('viewer-readback', current, baseline), /distinct canonical/);
    if (current !== workspaceIdentity)
      assert.throws(() => validatePhaseWorkspaceIdentity('retry-online', current, baseline), /protected original/);
  }
  const upper = workspaceIdentityFromFacts('C:\\QA\\Owner', { directory: true, dev: 1n, ino: 2n }, { caseInsensitive: true });
  const lower = workspaceIdentityFromFacts('c:/qa/owner', { directory: true, dev: 1n, ino: 3n }, { caseInsensitive: true });
  assert.equal(upper.canonical_path_sha256, lower.canonical_path_sha256);
  assert.notEqual(upper.filesystem_id_sha256, lower.filesystem_id_sha256);
  assert.throws(() => requireSeparateWorkspaceIdentity(upper, lower), /distinct canonical/);
  const legacy = { ...baseline, native_identity: { profile_sha256: profileDigest } };
  assert.throws(() => validatePhaseWorkspaceIdentity('viewer-readback', separate, legacy));
  assert.throws(() => validateBaseline(legacy, projectId, owner));
  // A changed raw path hash is diagnostic; the two canonical identities govern binding.
  const aliasReceipt = structuredClone(online); aliasReceipt.native_identity.profile_sha256 = '3'.repeat(64);
  assert.equal(validateOnlineIdentityReceipt(aliasReceipt, projectId, owner, workspaceIdentity), aliasReceipt);
  assert.equal(JSON.stringify(baseline), originalBytes);
});

test('viewer root admission precedes any credential read and preserves the original receipt', async t => {
  const root = await mkdtemp(join(tmpdir(), 'market87-admission-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const identity = await canonicalWorkspaceIdentity(root);
  const original = { ...baseline, native_identity: { ...baseline.native_identity, workspace_identity: identity } };
  const baselinePath = join(root, 'original.json'), absentCredentials = join(root, 'absent-private.json');
  await writeFile(baselinePath, JSON.stringify(original));
  const bytes = await readFile(baselinePath);
  await assert.rejects(readAdmittedQaManifest(absentCredentials, { ...options, phase: 'viewer-readback' }, identity, original), /distinct canonical/);
  await assert.rejects(readAdmittedQaManifest(absentCredentials, { ...options, phase: 'retry-online' },
    { ...identity, filesystem_id_sha256: '4'.repeat(64) }, original), /protected original/);
  const invalidOnline = { ...online, status: 'error', native_identity: { ...online.native_identity, workspace_identity: identity } };
  await assert.rejects(readAdmittedQaManifest(absentCredentials, { ...options, phase: 'offline-run' }, identity, invalidOnline), /original online owner/);
  const viewerRoot = join(root, 'viewer'); await mkdir(viewerRoot);
  const viewerIdentity = await canonicalWorkspaceIdentity(viewerRoot);
  const privateFixture = join(root, 'private-fixture.json');
  const manifest = { projects: [projectId], users: { owner, viewer }, url: 'https://openecon-291739190496.us-central1.run.app' };
  await writeFile(privateFixture, JSON.stringify(manifest));
  assert.deepEqual(await readAdmittedQaManifest(privateFixture, { ...options, phase: 'viewer-readback' }, viewerIdentity, original), manifest);
  assert.deepEqual(await readFile(baselinePath), bytes);
});

test('phase-end identity refuses a recreated root even when the original path text is unchanged', async t => {
  const root = await mkdtemp(join(tmpdir(), 'market87-replaced-root-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const ownerRoot = join(root, 'owner'); await mkdir(ownerRoot);
  const original = await canonicalWorkspaceIdentity(ownerRoot);
  await rename(ownerRoot, join(root, 'retained-original'));
  await mkdir(ownerRoot);
  const replacement = await canonicalWorkspaceIdentity(ownerRoot);
  assert.equal(original.canonical_path_sha256, replacement.canonical_path_sha256);
  assert.notEqual(original.filesystem_id_sha256, replacement.filesystem_id_sha256);
  assert.throws(() => requireSameWorkspaceIdentity(original, replacement), /workspace root changed/);
  assert.throws(() => requireSeparateWorkspaceIdentity(original, replacement), /distinct canonical/);
});

test('only exact manifest-owned project and distinct disposable password actors are accepted', () => {
  const fixture = { projects: [projectId], users: { owner, viewer }, url: 'https://openecon-291739190496.us-central1.run.app' };
  assert.equal(validateQaManifest(fixture, options), fixture);
  for (const rejected of [
    { ...fixture, projects: ['2'.repeat(32)] },
    { ...fixture, users: { owner, viewer: owner } },
    { ...fixture, users: { owner: { ...owner, email: 'human@example.com' }, viewer } },
    { ...fixture, url: 'https://unrelated.example.com' },
    { ...fixture, url: undefined }, { ...fixture, url: '' },
    { ...fixture, url: fixture.url + '/' }, { ...fixture, url: fixture.url + '/?foreign=1' },
  ]) assert.throws(() => validateQaManifest(rejected, options));
});

test('native and actual approval page must use separate trusted loopback endpoints', () => {
  const grant = '3'.repeat(32);
  const endpoint = 'ws://127.0.0.1:9230/devtools/page/owned';
  assert.equal(trustedTarget({ type: 'page', url: 'http://127.0.0.1:49100/', webSocketDebuggerUrl: endpoint }, 9230, 'native'), true);
  assert.equal(trustedTarget({ type: 'page', url: `https://openecon-291739190496.us-central1.run.app/?desktop_login=${grant}`, webSocketDebuggerUrl: endpoint }, 9230, 'browser', grant), true);
  for (const target of [
    { type: 'page', url: 'https://unrelated.example.com/', webSocketDebuggerUrl: endpoint },
    { type: 'page', url: 'http://127.0.0.1/', webSocketDebuggerUrl: 'ws://remote.example.com:9230/devtools/page/x' },
    { type: 'page', url: `https://openecon-291739190496.us-central1.run.app/?desktop_login=${'4'.repeat(32)}`, webSocketDebuggerUrl: endpoint },
  ]) assert.equal(trustedTarget(target, 9230, 'browser', grant), false);
});

test('real native fields bind broker origins while missing source/UID provenance stays explicitly unknown', () => {
  const info = { local_origin: 'http://127.0.0.1:49100', cloud_origin: 'https://openecon-291739190496.us-central1.run.app',
    version: '0.3.44', data_root: 'private-owned-profile-sentinel' };
  const summary = nativeIdentitySummary(info, info.local_origin);
  assert.equal(summary.observed_desktop_version, '0.3.44');
  assert.equal(summary.installed_source_binding, 'unknown');
  assert.equal(summary.deployed_cloud_source_binding, 'unknown');
  assert.equal(summary.webview_storage_binding, 'unknown');
  assert.match(summary.profile_sha256, /^[0-9a-f]{64}$/);
  assert.equal(JSON.stringify(summary).includes(info.data_root), false);
  for (const changed of [
    { ...info, local_origin: 'http://127.0.0.1:49101' },
    { ...info, cloud_origin: 'https://unrelated.example.com' },
    { ...info, version: undefined }, { ...info, data_root: '' },
  ]) assert.throws(() => nativeIdentitySummary(changed, info.local_origin));
});

test('reopen/retry cannot promote a foreign, recomputed, online or incomplete baseline', () => {
  assert.equal(validateBaseline(baseline, projectId, owner), baseline);
  for (const rejected of [
    { ...baseline, project_id: '2'.repeat(32) },
    { ...baseline, original: { ...baseline.original, execution_count: 2 } },
    { ...baseline, original: { ...baseline.original, sharing_state: 'shared' } },
    { ...baseline, original: { ...baseline.original, latex_present: false } },
    { ...baseline, original: { ...baseline.original, fixture_protocol_valid: undefined } },
    { ...baseline, observed_execution_requests: { local_executions: 1, cloud_executions: 1 } },
    { ...baseline, checks: { native_cloud_unreachable: false } },
    { ...baseline, authenticated_identity: { ...baseline.authenticated_identity, uid_sha256: hash(viewer.uid) } },
    { ...baseline, authenticated_identity: { ...baseline.authenticated_identity, email_sha256: hash(viewer.email) } },
  ]) assert.throws(() => validateBaseline(rejected, projectId, owner));
});

test('unsafe CLI ports, reused secret/output paths and unknown options are refused', () => {
  const args = ['--port', '9230', '--phase', 'password-open', '--state', 'private-state.json', '--output', 'public-receipt.json', '--project-id', projectId, '--project-name', 'openecon-qa-native-share'];
  assert.equal(parseArguments(args).port, 9230);
  for (const rejected of [
    [...args, '--port', '9222'],
    args.map(value => value === '9230' ? 'https://remote.example.com' : value),
    args.map(value => value === 'public-receipt.json' ? 'private-state.json' : value),
    [...args, '--token', 'private-token-sentinel'],
  ]) assert.throws(() => parseArguments(rejected));
  const retryArgs = args.map(value => value === 'password-open' ? 'offline-reopen' : value);
  assert.throws(() => parseArguments([...retryArgs, '--baseline', 'public-receipt.json']));
  assert.throws(() => parseArguments([...retryArgs, '--baseline', 'private-state.json']));
  const offlineArgs = args.map(value => value === 'password-open' ? 'offline-run' : value);
  assert.throws(() => parseArguments(offlineArgs));
  assert.equal(parseArguments([...offlineArgs, '--online-identity', 'original-online.json'])['online-identity'], 'original-online.json');
  for (const path of ['private-state.json', 'public-receipt.json'])
    assert.throws(() => parseArguments([...offlineArgs, '--online-identity', path]));
  assert.throws(() => parseArguments([...args, '--online-identity', 'original-online.json']));
  const browserArgs = args.map(value => value === 'password-open' ? 'browser-approval' : value);
  assert.throws(() => parseArguments([...browserArgs, '--browser-port', '9231']));
  assert.equal(parseArguments([...browserArgs, '--browser-port', '9231', '--online-identity', 'original-online.json']).phase, 'browser-approval');
});

test('canonical parent, case and inode aliases preserve private state and original proof bytes', async t => {
  const root = await mkdtemp(join(tmpdir(), 'openecon-owned-cloud-path-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const state = join(root, 'private-state.json'), baselinePath = join(root, 'original-proof.json'), onlinePath = join(root, 'original-online.json');
  const output = join(root, 'new-receipt.json');
  const stateBytes = Buffer.from('private synthetic fixture; preserve these exact bytes\n');
  const proofBytes = Buffer.from('historical proof; preserve these exact bytes\n');
  await writeFile(state, stateBytes); await writeFile(baselinePath, proofBytes); await writeFile(onlinePath, 'original online identity proof\n');
  const inputs = { state, baseline: baselinePath, 'online-identity': onlinePath, output };
  const alias = join(root, 'linked-parent');
  await symlink(root, alias, 'junction');
  const hardlink = join(root, 'same-input-inode.json');
  await link(state, hardlink);
  const onlineHardlink = join(root, 'online-hardlink.json');
  await link(onlinePath, onlineHardlink);
  for (const rejected of [
    { ...inputs, output: baselinePath }, { ...inputs, output: state },
    { ...inputs, output: join(alias, 'original-proof.json') },
    { ...inputs, baseline: hardlink },
    { ...inputs, output: onlinePath }, { ...inputs, output: join(alias, 'original-online.json') },
    { ...inputs, baseline: onlineHardlink }, { ...inputs, 'online-identity': hardlink },
  ]) await assert.rejects(validateControllerPaths(rejected));
  await assert.rejects(validateControllerPaths({ ...inputs, output: join(root, 'PRIVATE-STATE.JSON') }, { caseInsensitive: true }));
  await writeFile(output, 'already saved phase proof\n');
  await assert.rejects(validateControllerPaths(inputs));
  assert.deepEqual(await readFile(state), stateBytes);
  assert.deepEqual(await readFile(baselinePath), proofBytes);
  assert.equal(await readFile(onlinePath, 'utf8'), 'original online identity proof\n');
  assert.equal(await readFile(output, 'utf8'), 'already saved phase proof\n');
});

test('exclusive output reservation refuses races and never truncates an existing receipt', async t => {
  const root = await mkdtemp(join(tmpdir(), 'openecon-owned-cloud-exclusive-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const state = join(root, 'state.json'), baselinePath = join(root, 'baseline.json'), output = join(root, 'phase.json');
  await writeFile(state, 'private sentinel\n'); await writeFile(baselinePath, 'original proof\n');
  const paths = await validateControllerPaths({ state, baseline: baselinePath, output });
  await writeFile(output, 'receipt appeared after preflight\n');
  await assert.rejects(reserveReceiptOutput(paths.output), { code: 'EEXIST' });
  assert.equal(await readFile(output, 'utf8'), 'receipt appeared after preflight\n');
  assert.equal(await readFile(state, 'utf8'), 'private sentinel\n');
  assert.equal(await readFile(baselinePath, 'utf8'), 'original proof\n');
  const fresh = join(root, 'fresh-phase.json'), handle = await reserveReceiptOutput(fresh);
  try { await handle.writeFile('one completed proof\n'); } finally { await handle.close(); }
  await assert.rejects(reserveReceiptOutput(fresh), { code: 'EEXIST' });
  assert.equal(await readFile(fresh, 'utf8'), 'one completed proof\n');
});

function fixtureRecord() {
  const coefficients = ['Intercept', 'education', 'experience'].map(term => ({
    term, estimate: 1.25, std_error: 0.1, statistic: 12.5, p_value: 1e-24, ci_low: 1.05, ci_high: 1.45,
  }));
  return { code: EXECUTION_COMMAND, status: 'ok', stdout: EXPECTED_STDOUT, stderr: '', error: null,
    outputs: [{ type: 'model', data: { nobs: 480, nobs_original: 480, dropped_rows: 0,
      spec: { estimator: 'ols', outcome: 'wage', predictors: ['education', 'experience'], covariance: 'HC3', intercept: true, alpha: 0.05 },
      inference: { covariance: 'HC3' }, coefficients, display_omitted: ['sample_positions', 'covariance_matrix'] },
      latex: 'exact publication TeX', latex_math: 'exact model math' },
    { type: 'latex', data: 'exact publication TeX', latex: 'exact publication TeX', latex_math: 'exact publication math' }],
    events: [{ type: 'output', index: 0 }, { type: 'output', index: 1 }, { type: 'stdout', text: EXPECTED_STDOUT }] };
}

test('strict actual fixture protocol refuses extra markers, wrong model/source/publication and altered events', () => {
  assert.equal(fixtureRecordMatches(fixtureRecord()), true);
  for (const mutate of [
    value => { value.stdout += 'MARKET87_NATIVE_EXECUTIONS=2\n'; },
    value => { value.stdout += EXPECTED_STDOUT; },
    value => { value.code = EXECUTION_CODE; },
    value => { value.outputs.pop(); },
    value => { value.outputs.push(structuredClone(value.outputs[0])); },
    value => { value.outputs[1].data += ' '; },
    value => { value.outputs[0].data.spec.covariance = 'HC1'; },
    value => { value.outputs[0].data.inference.covariance = 'HC1'; },
    value => { value.outputs[0].data.coefficients[1].term = 'wrong-predictor'; },
    value => { value.outputs[0].data.coefficients[1].estimate = NaN; },
    value => { value.outputs[0].data.covariance_matrix = []; },
    value => { value.outputs[0].data = null; },
    value => { value.events.reverse(); },
    value => { value.events[2].text += 'extra stdout'; },
    value => { value.stderr = 'unexpected error'; },
  ]) {
    const changed = fixtureRecord(); mutate(changed);
    assert.equal(fixtureRecordMatches(changed), false);
  }
});

function presentation({ stdout = EXPECTED_STDOUT, modelReady = true, publicationReady = true,
  publicationHidden = false, extraModel = false, wrongTitle = false, modelFallback = false, incompleteFallback = false } = {}) {
  // These nodes test admission logic only; actual native rendering remains a
  // separately required Windows phase, without any simulated browser proof.
  const node = (textContent = '') => ({ textContent, getBoundingClientRect: () => ({ width: 80, height: 20 }),
    closest: () => null, querySelector: () => null, querySelectorAll: () => [], classList: { contains: () => false } });
  const model = node();
  model.querySelector = selector => selector === '.model-title' ? node(wrongTitle ? 'wrong model' : 'OLS wage 480 observations')
    : selector === '.model-metrics' ? node('HC3 standard errors') : null;
  const rows = ['Intercept', 'education', 'experience'].flatMap(term => {
    const estimate = node(), uncertainty = node();
    estimate.classList.contains = value => value === 'publication-estimate';
    uncertainty.classList.contains = value => value === 'publication-standard-error';
    estimate.querySelector = selector => selector === 'th' ? node(term) : node('1.25');
    uncertainty.querySelector = selector => selector === 'th' ? node('') : node('(0.1)');
    return [estimate, uncertainty];
  });
  model.querySelectorAll = selector => selector === 'table tbody tr' && modelFallback
    ? rows.slice(0, incompleteFallback ? 5 : 6) : [];
  const math = node(); math.closest = selector => selector === '.model-output' ? model : null;
  math.querySelector = selector => selector === 'math' ? node() : null;
  const publication = node();
  publication.closest = selector => selector === '[hidden]' && publicationHidden ? node() : null;
  publication.querySelector = selector => selector === 'math' ? node() : null;
  const pane = { querySelectorAll: selector => selector === '.model-output' ? (extraModel ? [model, node()] : [model])
    : selector === '.text-output' ? [node(stdout)]
    : selector === '.latex-preview' ? [...(modelReady ? [math] : []), ...(publicationReady ? [publication] : [])] : [] };
  return { querySelector: selector => selector === '.output-pane .output-scroll' ? pane : null };
}

test('visible original requires exact stdout, one correct model and separate visible publication MathML', () => {
  assert.equal(fixturePresentationMatches(presentation()), true);
  assert.equal(fixturePresentationMatches(presentation({ modelReady: false, modelFallback: true })), true);
  for (const changed of [
    { stdout: 'MARKET87_WINDOWS_NATIVE_SHARE_OK\n' }, { stdout: EXPECTED_STDOUT + 'extra\n' },
    { modelReady: false }, { publicationReady: false }, { publicationHidden: true },
    { extraModel: true }, { wrongTitle: true },
    { modelReady: false, modelFallback: true, incompleteFallback: true },
  ]) assert.equal(fixturePresentationMatches(presentation(changed)), false);
});

test('archive comparison normalizes only omission-label order and protects scientific/publication/event values', async () => {
  const record = { code: 'display(model)', status: 'ok', stdout: 'once\n', outputs: [
    { type: 'model', data: { coefficients: [{ estimate: 1.25 }], display_omitted: ['sample_positions', 'covariance_matrix'] }, latex: 'exact TeX' },
  ], events: [{ type: 'output', index: 0 }, { type: 'stdout', text: 'once\n' }] };
  const archived = structuredClone(record);
  archived.outputs[0].data.display_omitted.reverse();
  const digest = (value, normalized) => eval(payloadDigestExpression(JSON.stringify(value), normalized));
  assert.notEqual(await digest(record, false), await digest(archived, false));
  const expected = await digest(record, true);
  assert.equal(expected, await digest(archived, true));
  for (const mutate of [
    value => { value.outputs[0].data.coefficients[0].estimate += 0.01; },
    value => { value.outputs[0].latex += ' '; },
    value => { value.events.reverse(); },
    value => { value.outputs[0].data.display_omitted.pop(); },
  ]) {
    const changed = structuredClone(archived); mutate(changed);
    assert.notEqual(expected, await digest(changed, true));
  }
});

test('unsupported host or unprovisioned runner exits before private credentials and prints no passed receipt', () => {
  const sentinel = 'private-token-sentinel';
  const result = spawnSync(process.execPath, [fileURLToPath(new URL('./verify_windows_cloud_ui.mjs', import.meta.url)),
    '--port', '9230', '--phase', 'password-open', '--state', sentinel,
    '--output', 'never-created-public-receipt.json', '--project-id', projectId,
    '--project-name', 'openecon-qa-native-share'], { encoding: 'utf8', env: { ...process.env,
      GITHUB_ACTIONS: process.platform === 'win32' ? 'false' : 'true' } });
  assert.equal(result.status, 1);
  assert.equal((result.stdout + result.stderr).includes(sentinel), false);
  assert.equal(result.stdout.includes('passed'), false);
});
