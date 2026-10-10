// Real installed WebView2 input; no mocked transport, cloud compute, or identity creation.
import { createHash } from 'node:crypto';
import { lstat, open, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, resolve, win32 } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const CLOUD_ORIGIN = 'https://openecon-291739190496.us-central1.run.app';
const MARKER = 'MARKET87_WINDOWS_NATIVE_SHARE_OK';
const PHASES = ['password-open', 'browser-approval', 'offline-run', 'offline-reopen', 'retry-online', 'viewer-readback'];
export const EXECUTION_CODE = `from pathlib import Path
import openecon as oe
counter = Path('market87-native-execution-count.txt')
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count), encoding='utf-8')
assert count == 1, 'The acceptance run must never be recomputed'
data = oe.example()
model = oe.ols(data=data, y='wage', x=['education', 'experience'], covariance='HC3')
assert model.nobs == 480 and len(model.sample_positions) == 480
assert len(model.coefficients) == 3 and len(model.covariance_matrix) == 3
assert model.spec.covariance == 'HC3'
display(model)
display(model.to_latex())
print('${MARKER}')
print('MARKET87_NATIVE_EXECUTIONS=1')`;
export const EXECUTION_COMMAND = `exec(${JSON.stringify(EXECUTION_CODE)})`;
export const EXPECTED_STDOUT = `${MARKER}\nMARKET87_NATIVE_EXECUTIONS=1\n`;

export function parseArguments(values) {
  if (values.length % 2) throw new Error('Explicit paired controller arguments are required.');
  const options = {};
  for (let i = 0; i < values.length; i += 2) {
    const key = values[i];
    if (!/^--[a-z-]+$/.test(key) || options[key.slice(2)] !== undefined
        || typeof values[i + 1] !== 'string' || !values[i + 1])
      throw new Error('Invalid controller arguments.');
    options[key.slice(2)] = values[i + 1];
  }
  const allowed = new Set(['port', 'browser-port', 'phase', 'state', 'output', 'project-id', 'project-name', 'baseline', 'online-identity']);
  if (Object.keys(options).some(key => !allowed.has(key)) || !PHASES.includes(options.phase)
      || !options.state || !options.output
      || !/^[0-9a-f]{32}$/.test(options['project-id'] ?? '')
      || !options['project-name']?.startsWith('openecon-qa-')
      || options['project-name'].length > 120)
    throw new Error('A private owned QA manifest and explicit disposable project are required.');
  options.port = ownedPort(options.port);
  if (options.phase === 'browser-approval') options['browser-port'] = ownedPort(options['browser-port']);
  if (options['browser-port'] === options.port) throw new Error('The browser and native ports must differ.');
  if (['offline-reopen', 'retry-online', 'viewer-readback'].includes(options.phase) && !options.baseline)
    throw new Error('The original offline receipt is required.');
  if (['browser-approval', 'offline-run'].includes(options.phase) && !options['online-identity'])
    throw new Error('The original online owner identity receipt is required.');
  if (options['online-identity'] && !['browser-approval', 'offline-run'].includes(options.phase))
    throw new Error('Only browser approval or the first offline phase accepts an online identity receipt.');
  const paths = [options.state, options.output, options.baseline, options['online-identity']].filter(Boolean).map(value => resolve(value));
  if (new Set(paths.map(value => process.platform === 'win32' ? value.toLowerCase() : value)).size !== paths.length)
    throw new Error('Private state, original baseline and new phase receipt paths must differ.');
  return options;
}

export async function validateControllerPaths(options, { caseInsensitive = process.platform === 'win32' } = {}) {
  const identities = {};
  for (const key of ['state', 'baseline', 'online-identity', 'output'].filter(key => options[key])) {
    const absolute = resolve(options[key]);
    // Resolve the existing parent first: Windows junctions, symlinks and case
    // aliases must be compared before credentials are read or any UI is used.
    const parent = await realpath(dirname(absolute));
    if (!(await stat(parent)).isDirectory()) throw new Error('A controller file parent must exist.');
    let entry;
    try { entry = await lstat(absolute); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (key !== 'output' && !entry) throw new Error('A required controller input file is absent.');
    const canonical = entry ? await realpath(absolute) : join(parent, basename(absolute));
    const info = entry ? await stat(absolute, { bigint: true }) : null;
    if (info && !info.isFile()) throw new Error('A controller input must be a regular file.');
    identities[key] = { path: canonical, entry, info };
  }
  const rows = Object.values(identities);
  const normalized = value => caseInsensitive ? value.toLowerCase() : value;
  for (let i = 0; i < rows.length; i++) for (let j = i + 1; j < rows.length; j++) {
    const a = rows[i], b = rows[j];
    const sameInode = a.info && b.info && a.info.ino !== 0n
      && a.info.dev === b.info.dev && a.info.ino === b.info.ino;
    if (normalized(a.path) === normalized(b.path) || sameInode)
      throw new Error('Controller paths alias a protected input or receipt.');
  }
  if (identities.output.entry) throw new Error('An existing phase receipt must never be overwritten.');
  return Object.fromEntries(Object.entries(identities).map(([key, identity]) => [key, identity.path]));
}

export async function reserveReceiptOutput(path) {
  // Keep this exclusive handle for the entire phase. A receipt appearing after
  // preflight is refused, and later path replacement cannot redirect writes.
  return open(path, 'wx', 0o600);
}

function ownedPort(value) {
  if (!/^\d+$/.test(String(value))) throw new Error('An owned loopback CDP port is required.');
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1024 || port > 65535)
    throw new Error('An owned loopback CDP port is required.');
  return port;
}

export function validateQaManifest(state, options) {
  const projectId = options['project-id'];
  if (!Array.isArray(state.projects) || !state.projects.includes(projectId)
      || !state.users || !['owner', 'viewer'].every(role => {
        const user = state.users[role];
        return user && typeof user.uid === 'string' && user.uid.length > 0 && user.uid.length <= 128
          && !/[\x00-\x1f/]/.test(user.uid)
          && new RegExp(`^openecon-qa-${role}-[0-9a-f]{12}@example\\.com$`).test(user.email)
          && typeof user.password === 'string' && user.password.length >= 12;
      }) || state.users.owner.uid === state.users.viewer.uid)
    throw new Error('Only distinct manifest-owned disposable password identities are allowed.');
  if (state.url !== CLOUD_ORIGIN)
    throw new Error('The manifest must target the native broker cloud origin.');
  return state;
}

export function validateBaseline(value, projectId, owner) {
  if (value?.status !== 'passed' || value.phase !== 'offline-run' || value.project_id !== projectId
      || typeof value.original?.id !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(value.original.id)
      || !/^[0-9a-f]{64}$/.test(value.original.payload_sha256 ?? '')
      || !/^[0-9a-f]{64}$/.test(value.original.raw_payload_sha256 ?? '')
      || value.original.history_matches !== 1 || value.original.nobs !== 480
      || value.original.marker_occurrences !== 1 || value.original.execution_count !== 1
      || value.original.status !== 'ok' || value.original.latex_present !== true
      || value.original.fixture_protocol_valid !== true
      || value.original.sharing_state !== 'pending' || value.original.outbox_matches !== 1
      || value.observed_execution_requests?.local_executions !== 1
      || value.observed_execution_requests?.cloud_executions !== 0 || !value.checks?.native_cloud_unreachable
      || value.authenticated_identity?.verification !== 'offline_cached'
      || value.authenticated_identity.live_verification !== false
      || value.authenticated_identity.project_id !== projectId || value.authenticated_identity.role !== 'owner'
      || !/^[0-9a-f]{64}$/.test(value.authenticated_identity.uid_sha256 ?? '')
      || !/^[0-9a-f]{64}$/.test(value.authenticated_identity.email_sha256 ?? '')
      || !/^[0-9a-f]{64}$/.test(value.authenticated_identity.online_receipt_sha256 ?? '')
      || !/^[0-9a-f]{64}$/.test(value.native_identity?.profile_sha256 ?? '')
      || !validWorkspaceIdentity(value.native_identity?.workspace_identity)
      || !value.checks?.offline_cached_owner_bound_to_original_online_identity
      || !value.checks?.canonical_workspace_identity_before_credentials
      || !value.checks?.workspace_identity_rechecked_after_phase
      || (owner && !matchesIdentityBinding(value.authenticated_identity, owner, projectId, 'owner')))
    throw new Error('The baseline must prove the same single offline native computation.');
  return value;
}

export function trustedTarget(target, port, surface, requestId) {
  try {
    const page = new URL(target.url);
    const socket = new URL(target.webSocketDebuggerUrl);
    if (target.type !== 'page' || socket.protocol !== 'ws:' || socket.hostname !== '127.0.0.1'
        || Number(socket.port) !== port) return false;
    if (surface === 'native') return page.protocol === 'http:' && page.hostname === '127.0.0.1';
    return page.origin === CLOUD_ORIGIN && page.searchParams.get('desktop_login') === requestId;
  } catch { return false; }
}

export function nativeIdentitySummary(info, pageOrigin) {
  const origin = new URL(pageOrigin);
  if (origin.protocol !== 'http:' || origin.hostname !== '127.0.0.1'
      || info?.local_origin !== origin.origin || info.cloud_origin !== CLOUD_ORIGIN
      || typeof info.version !== 'string' || !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(info.version)
      || typeof info.data_root !== 'string' || !info.data_root)
    throw new Error('The actual native broker origins or public identity fields are invalid.');
  return { local_origin: info.local_origin, observed_desktop_version: info.version,
    profile_sha256: createHash('sha256').update(info.data_root).digest('hex'),
    cloud_origin_verified: true, installed_source_binding: 'unknown',
    installed_resource_hash_binding: 'unknown', deployed_cloud_source_binding: 'unknown',
    linked_provider_binding: 'unknown', webview_storage_binding: 'unknown',
    current_authenticated_uid_binding: 'not yet verified in this phase' };
}

export function workspaceIdentityFromFacts(canonical, info, { caseInsensitive = process.platform === 'win32' } = {}) {
  if (typeof canonical !== 'string' || (!isAbsolute(canonical) && !(caseInsensitive && win32.isAbsolute(canonical)))
      || info?.directory !== true || typeof info.dev !== 'bigint' || info.dev <= 0n
      || typeof info.ino !== 'bigint' || info.ino <= 0n)
    throw new Error('An existing workspace directory with stable nonzero filesystem identity is required.');
  const path = caseInsensitive ? win32.normalize(canonical).toLowerCase() : canonical;
  return { schema: 1, verification: 'canonical_existing_directory',
    canonical_path_sha256: identityHash(path), filesystem_id_sha256: identityHash(`${info.dev}:${info.ino}`) };
}

function validWorkspaceIdentity(value) {
  return value?.schema === 1 && value.verification === 'canonical_existing_directory'
    && /^[0-9a-f]{64}$/.test(value.canonical_path_sha256 ?? '')
    && /^[0-9a-f]{64}$/.test(value.filesystem_id_sha256 ?? '');
}

export function requireSameWorkspaceIdentity(original, current) {
  if (!validWorkspaceIdentity(original) || !validWorkspaceIdentity(current)
      || original.canonical_path_sha256 !== current.canonical_path_sha256
      || original.filesystem_id_sha256 !== current.filesystem_id_sha256)
    throw new Error('The workspace root changed or differs from the protected original identity.');
  return true;
}

export function requireSeparateWorkspaceIdentity(original, current) {
  if (!validWorkspaceIdentity(original) || !validWorkspaceIdentity(current)
      || original.canonical_path_sha256 === current.canonical_path_sha256
      || original.filesystem_id_sha256 === current.filesystem_id_sha256)
    throw new Error('The viewer requires a distinct canonical workspace root and filesystem identity.');
  return true;
}

export async function canonicalWorkspaceIdentity(dataRoot, options) {
  const canonical = await realpath(dataRoot);
  const info = await stat(canonical, { bigint: true });
  const original = workspaceIdentityFromFacts(canonical,
    { directory: info.isDirectory(), dev: info.dev, ino: info.ino }, options);
  const checkedPath = await realpath(dataRoot);
  const checked = await stat(checkedPath, { bigint: true });
  const current = workspaceIdentityFromFacts(checkedPath,
    { directory: checked.isDirectory(), dev: checked.dev, ino: checked.ino }, options);
  requireSameWorkspaceIdentity(original, current);
  return original;
}

export function validatePhaseWorkspaceIdentity(phase, current, original) {
  if (!PHASES.includes(phase) || !validWorkspaceIdentity(current))
    throw new Error('A supported phase and verified existing workspace identity are required.');
  if (phase === 'password-open') return true;
  const expected = original?.native_identity?.workspace_identity;
  return phase === 'viewer-readback' ? requireSeparateWorkspaceIdentity(expected, current)
    : requireSameWorkspaceIdentity(expected, current);
}

export async function readAdmittedQaManifest(path, options, workspaceIdentity, original) {
  if (['browser-approval', 'offline-run'].includes(options.phase))
    validateOnlineIdentityReceipt(original, options['project-id']);
  else if (options.phase !== 'password-open') validateBaseline(original, options['project-id']);
  validatePhaseWorkspaceIdentity(options.phase, workspaceIdentity, original);
  return validateQaManifest(JSON.parse(await readFile(path, 'utf8')), options);
}

// Select only actual public identity and this project's membership. The original
// broker response, credentials, other projects and invitations are never retained.
export function selectProfileIdentity(body, projectId) {
  const user = body?.user;
  const projects = Array.isArray(body?.projects) ? body.projects.filter(project => project?.id === projectId) : [];
  if (typeof user?.uid !== 'string' || !user.uid || user.uid.length > 128 || /[\x00-\x1f/]/.test(user.uid)
      || typeof user.email !== 'string' || !user.email || user.email.length > 254 || /[\x00-\x1f]/.test(user.email)
      || projects.length !== 1 || !['owner', 'editor', 'viewer'].includes(projects[0].role)) return null;
  return { uid: user.uid, email: user.email, project_id: projectId, role: projects[0].role };
}

const identityHash = value => createHash('sha256').update(value).digest('hex');
function matchesIdentityBinding(value, user, projectId, role) {
  return value?.uid_sha256 === identityHash(user.uid) && value.email_sha256 === identityHash(user.email)
    && value.project_id === projectId && value.role === role;
}

export function observedIdentityReceipt(observation, user, projectId, role, afterSequence = 0) {
  const actual = observation?.identity;
  if (!Number.isSafeInteger(observation?.sequence) || observation.sequence <= afterSequence
      || observation.status !== 200 || actual?.uid !== user.uid || actual.email !== user.email
      || actual.project_id !== projectId || actual.role !== role)
    throw new Error('A fresh authenticated broker identity must match the exact owned UID and project role.');
  return { verification: 'live_authenticated_native_broker', source: 'GET /api/me', live_verification: true,
    uid_sha256: identityHash(actual.uid), email_sha256: identityHash(actual.email), project_id: actual.project_id,
    role: actual.role, profile_request_sequence: observation.sequence };
}

export function validateOnlineIdentityReceipt(value, projectId, user, workspaceIdentity) {
  const identity = value?.authenticated_identity;
  if (value?.status !== 'passed' || !['password-open', 'browser-approval'].includes(value.phase)
      || value.project_id !== projectId || !value.checks?.fresh_authenticated_uid_and_live_project_role
      || !value.checks?.fresh_authenticated_uid_and_live_project_role_after_phase
      || !value.checks?.canonical_workspace_identity_before_credentials
      || !value.checks?.workspace_identity_rechecked_after_phase
      || !value.checks?.native_cloud_reachable || identity?.verification !== 'live_authenticated_native_broker'
      || identity.source !== 'GET /api/me' || identity.live_verification !== true
      || !Number.isSafeInteger(identity.profile_request_sequence) || identity.profile_request_sequence < 1
      || !Number.isSafeInteger(identity.initial_profile_request_sequence) || identity.initial_profile_request_sequence < 1
      || identity.profile_request_sequence <= identity.initial_profile_request_sequence
      || identity.project_id !== projectId || identity.role !== 'owner'
      || !/^[0-9a-f]{64}$/.test(identity.uid_sha256 ?? '') || !/^[0-9a-f]{64}$/.test(identity.email_sha256 ?? '')
      || (user !== undefined && !matchesIdentityBinding(identity, user, projectId, 'owner'))
      || !/^[0-9a-f]{64}$/.test(value.native_identity?.profile_sha256 ?? '')
      || !validWorkspaceIdentity(value.native_identity?.workspace_identity)
      || value.observed_execution_requests?.local_executions !== 0 || value.observed_execution_requests?.cloud_executions !== 0)
    throw new Error('The offline bridge requires the original online owner identity receipt for the same native profile.');
  if (workspaceIdentity !== undefined) requireSameWorkspaceIdentity(value.native_identity.workspace_identity, workspaceIdentity);
  return value;
}

export function offlineIdentityReceipt(online, user, projectId, workspaceIdentity, receiptDigest) {
  validateOnlineIdentityReceipt(online, projectId, user, workspaceIdentity);
  if (!/^[0-9a-f]{64}$/.test(receiptDigest ?? '')) throw new Error('The original online receipt digest is required.');
  const { uid_sha256, email_sha256, role } = online.authenticated_identity;
  return { verification: 'offline_cached', live_verification: false, source: 'native cached profile; original online identity receipt',
    uid_sha256, email_sha256, project_id: projectId, role, online_receipt_sha256: receiptDigest };
}

export function validateOfflineIdentityReceipt(value, projectId, user, workspaceIdentity) {
  validateBaseline(value, projectId);
  if (!matchesIdentityBinding(value.authenticated_identity, user, projectId, 'owner')
      || value.authenticated_identity.source !== 'native cached profile; original online identity receipt')
    throw new Error('The offline cached identity must remain bound to the original owner and native profile.');
  requireSameWorkspaceIdentity(value.native_identity.workspace_identity, workspaceIdentity);
  return { ...value.authenticated_identity };
}

async function until(operation, timeout = 180000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await operation()) return true;
    await delay(250);
  }
  throw new Error('A required installed UI checkpoint timed out.');
}

class Cdp {
  constructor(socket) { this.socket = socket; this.nextId = 0; this.pending = new Map(); }
  static async connect(port, surface, requestId) {
    let target;
    await until(async () => {
      let response;
      try { response = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.timeout(5000) }); }
      catch { return false; }
      if (!response.ok) return false;
      const targets = (await response.json()).filter(row => trustedTarget(row, port, surface, requestId));
      if (targets.length > 1) throw new Error('The owned UI target is ambiguous.');
      target = targets[0];
      return Boolean(target);
    });
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((accept, reject) => {
      socket.addEventListener('open', accept, { once: true });
      socket.addEventListener('error', () => reject(new Error('Owned CDP connection failed.')), { once: true });
    });
    const client = new Cdp(socket);
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      const pending = client.pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer); client.pending.delete(message.id);
      if (message.error) pending.reject(new Error('Owned CDP command failed.'));
      else pending.resolve(message.result);
    });
    await client.request('Runtime.enable');
    return client;
  }
  request(method, params = {}) {
    const id = ++this.nextId;
    return new Promise((accept, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('Owned CDP command timed out.')); }, 300000);
      this.pending.set(id, { resolve: accept, reject, timer });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expression) {
    const response = await this.request('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error('Installed UI inspection failed.');
    return response.result?.value;
  }
  async click(selector) {
    let point;
    await until(async () => {
      point = await this.evaluate(`(() => {
        const node = document.querySelector(${JSON.stringify(selector)});
        if (!node || node.disabled || node.getBoundingClientRect().width < 2) return null;
        node.scrollIntoView({block:'center'});
        const rect = node.getBoundingClientRect();
        if (rect.height < 2) return null;
        return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};
      })()`);
      return Boolean(point);
    });
    await this.request('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
    await this.request('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 });
  }
  async textButton(text) {
    await until(() => this.evaluate(`(() => {
      const buttons = [...document.querySelectorAll('button')].filter(node => node.textContent.trim() === ${JSON.stringify(text)} && !node.disabled && node.getBoundingClientRect().width > 1);
      if (buttons.length !== 1) return false;
      document.querySelector('[data-market87-button]')?.removeAttribute('data-market87-button');
      buttons[0].setAttribute('data-market87-button','true'); return true;
    })()`));
    await this.click('[data-market87-button="true"]');
  }
  async fill(selector, text) {
    await until(() => this.evaluate(`(() => {
      const node=document.querySelector(${JSON.stringify(selector)});
      if(!node || node.disabled || node.getBoundingClientRect().width<2)return false;
      node.focus(); node.select(); return true;
    })()`));
    // Input events let React and Firebase receive the real typed values.
    await this.request('Input.insertText', { text });
  }
  close() {
    this.socket.close();
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(new Error('Owned CDP closed.')); }
    this.pending.clear();
  }
}

async function enterTeams(client) {
  await until(() => client.evaluate(`!!document.querySelector('.team-auth,.team-dashboard,.team-workspace')`));
  if (await client.evaluate(`document.body.innerText.includes('Team projects · sign in')`))
    await client.textButton('Team projects · sign in');
  await until(() => client.evaluate(`!!document.querySelector('.team-auth input[type="email"],.team-user-email')`));
}

async function passwordSignIn(client, user) {
  if (await client.evaluate(`!!document.querySelector('.team-auth input[type="email"]')`)) {
    await client.fill('.team-auth input[type="email"]', user.email);
    await client.fill('.team-auth input[type="password"]', user.password);
    await client.textButton('Sign in');
  }
  await until(() => client.evaluate(`document.querySelector('.team-user-email')?.title === ${JSON.stringify(user.email)}`));
}

async function refreshLiveIdentity(client, options, user, role) {
  if (await client.evaluate(`!!document.querySelector('.team-workspace')`)) await client.click('.team-back');
  await until(() => client.evaluate(`!!document.querySelector('.team-dashboard')`));
  await until(() => client.evaluate(`[...document.querySelectorAll('.team-dashboard-footer button')].filter(node=>node.textContent.trim()==='Refresh projects' && !node.disabled && node.getBoundingClientRect().width>1).length===1`));
  const before = await client.evaluate(`window.__market87Observer.profile_requests`);
  // This normal UI action obtains its own current Firebase token; the controller
  // neither reads that token nor makes an authenticated request itself.
  await client.textButton('Refresh projects');
  let observed;
  await until(async () => {
    observed = await client.evaluate(`window.__market87Observer.profile`);
    return observed?.sequence > before && observed.status !== 'pending';
  });
  const identity = observedIdentityReceipt(observed, user, options['project-id'], role, before);
  await openProject(client, options, user, role);
  return identity;
}

async function openProject(client, options, user, role) {
  if (!await client.evaluate(`!!document.querySelector('.team-workspace')`)) {
    await until(() => client.evaluate(`(() => {
      const buttons=[...document.querySelectorAll('.team-project-open')].filter(node => node.querySelector('h2')?.textContent === ${JSON.stringify(options['project-name'])});
      if(buttons.length!==1)return false;
      buttons[0].setAttribute('data-market87-project','true');return true;
    })()`));
    await client.click('[data-market87-project="true"]');
  }
  await until(() => client.evaluate(`!!document.querySelector('[aria-label="Python, pip, or uv command"]')`));
  const correct = await client.evaluate(`(() => {
    const cached=JSON.parse(localStorage.getItem(${JSON.stringify(`openecon-desktop-profile:${user.uid}`)}) || 'null');
    return new URL(location.href).searchParams.get('project') === ${JSON.stringify(options['project-id'])}
      && document.querySelector('.team-user-email')?.title === ${JSON.stringify(user.email)}
      && cached?.user?.uid === ${JSON.stringify(user.uid)}
      && cached.projects.some(project => project.id === ${JSON.stringify(options['project-id'])} && project.role === ${JSON.stringify(role)})
      && document.body.innerText.includes(${JSON.stringify(options['project-name'])});
  })()`);
  if (!correct) throw new Error('The actual native project or actor role does not match the owned manifest.');
  if (await client.evaluate(`document.querySelector('.workspace-terminal-toggle')?.getAttribute('aria-expanded')==='false'`))
    await client.click('.workspace-terminal-toggle');
}

export function observerExpression(projectId) {
  return `(() => {
    if(window.__market87Observer)throw new Error('Observer already installed');
    const tauri=window.__TAURI__, core=tauri.core, invoke=core.invoke, fetchOriginal=window.fetch;
    const descriptor=Object.getOwnPropertyDescriptor(tauri,'core');
    if(!descriptor?.writable || typeof invoke!=='function')throw new Error('The native observation boundary is not writable');
    const prefix=${JSON.stringify(`/api/projects/${projectId}/workspace`)};
    const localPrefix=${JSON.stringify(`/api/desktop/projects/${projectId}/workspace`)};
    const selectIdentity=${selectProfileIdentity.toString()};
    const projectId=${JSON.stringify(projectId)};
    const state={local_executions:0,cloud_executions:0,grant:null,opened_grant:null,archive:null,archives:[],profile_requests:0,profile_successes:0,profile:null};
    const observedInvoke=async function(command,args){
      if(command==='cloud_request' && args?.method==='POST' && args.path===prefix+'/console/execute')state.cloud_executions++;
      const profileRequest=command==='cloud_request' && args?.method==='GET' && args.path==='/api/me';
      const sequence=profileRequest?++state.profile_requests:0;
      if(profileRequest)state.profile={sequence,status:'pending',identity:null};
      let response;
      try{response=await invoke.apply(core,arguments);}
      catch(error){if(sequence && sequence===state.profile_requests)state.profile={sequence,status:'failed',identity:null};throw error;}
      if(profileRequest){
        if(response?.status===200)state.profile_successes++;
        if(sequence===state.profile_requests)state.profile={sequence,status:response?.status??'failed',identity:response?.status===200?selectIdentity(response.body,projectId):null};
      }
      if(command==='cloud_request' && args?.path==='/api/desktop/login' && args.method==='POST')state.grant=response?.body?.request_id;
      if(command==='open_desktop_login')state.opened_grant=args?.requestId;
      if(command==='cloud_request' && args?.method==='GET' && args.path?.startsWith(prefix+'/runs/') && args.path.endsWith('/record') && response.status===200)state.archive=response.body;
      if(command==='cloud_request' && args?.method==='POST' && args.path===prefix+'/desktop/results' && response.status<300 && response.body?.shared===true)state.archives.push(response.body.id);
      return response;
    };
    const observedFetch=async function(input,init){
      const url=new URL(typeof input==='string'?input:input.url,location.href);
      if((init?.method || (input instanceof Request?input.method:'GET'))==='POST' && url.pathname===localPrefix+'/desktop-console/execute')state.local_executions++;
      return fetchOriginal.apply(window,arguments);
    };
    // The locked Tauri global core namespace is frozen. Replace only its writable
    // containing export with a faithful wrapper; preserve the original namespace
    // and all actual IPC callbacks/results, and restore it after this phase.
    const observedCore=Object.freeze({...core,invoke:observedInvoke});
    tauri.core=observedCore;window.fetch=observedFetch;
    if(tauri.core!==observedCore || window.fetch!==observedFetch)throw new Error('The native observer was not admitted');
    state.restore=()=>{if(tauri.core===observedCore)tauri.core=core;if(window.fetch===observedFetch)window.fetch=fetchOriginal;delete window.__market87Observer;};
    window.__market87Observer=state;return true;
  })()`;
}

async function installObserver(client, projectId) {
  // Observe original transports; never alter an HTTP response, grant, or model.
  await client.evaluate(observerExpression(projectId));
}

async function nativeCloudReachability(client, reachable) {
  const actual = await client.evaluate(`window.__TAURI__.core.invoke('cloud_request',{
    method:'GET',path:'/api/auth/config',token:'',body:null
  }).then(response=>response.status===200, error=>error==='OPENECON_NETWORK'?false:null)`);
  if (actual !== reachable) throw new Error('The native broker connectivity does not match this acceptance phase.');
}

export function payloadDigestExpression(record, archiveComparison = false) {
  return `(async()=>{
    const record=${record};
    function ordered(value){if(Array.isArray(value))return value.map(ordered);if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([key,item])=>[key,ordered(item)]));return value;}
    const body=Object.fromEntries(['code','status','stdout','stderr','outputs','events'].map(key=>[key,record[key]??null]));
    if(${JSON.stringify(archiveComparison)})body.outputs=body.outputs.map(output=>output.type==='model' && Array.isArray(output.data?.display_omitted)
      ? {...output,data:{...output.data,display_omitted:[...output.data.display_omitted].sort()}}:output);
    const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(ordered(body))));
    return [...new Uint8Array(bytes)].map(value=>value.toString(16).padStart(2,'0')).join('');
  })()`;
}

export function fixtureRecordMatches(record, command = EXECUTION_COMMAND, stdout = EXPECTED_STDOUT) {
  const outputs = record?.outputs;
  const model = outputs?.[0], publication = outputs?.[1], data = model?.data;
  const fields = ['estimate', 'std_error', 'statistic', 'p_value', 'ci_low', 'ci_high'];
  const coefficients = data?.coefficients;
  const events = record?.events;
  return Boolean(record?.code === command && record.status === 'ok' && record.stdout === stdout
    && (record.stderr == null || record.stderr === '') && record.error == null
    && Array.isArray(outputs) && outputs.length === 2 && model?.type === 'model' && publication?.type === 'latex'
    && data && typeof data === 'object'
    && data.nobs === 480 && data.nobs_original === 480 && data.dropped_rows === 0
    && data.spec?.estimator === 'ols' && data.spec.outcome === 'wage' && data.spec.covariance === 'HC3'
    && data.spec.intercept === true && data.spec.alpha === 0.05
    && JSON.stringify(data.spec.predictors) === JSON.stringify(['education', 'experience'])
    && data.inference?.covariance === 'HC3'
    && Array.isArray(coefficients) && coefficients.length === 3
    && JSON.stringify(coefficients.map(item => item?.term)) === JSON.stringify(['Intercept', 'education', 'experience'])
    && coefficients.every(item => fields.every(key => typeof item[key] === 'number' && Number.isFinite(item[key]))
      && item.std_error > 0 && item.p_value >= 0 && item.p_value <= 1 && item.ci_low <= item.estimate && item.estimate <= item.ci_high)
    && Array.isArray(data.display_omitted)
    && JSON.stringify([...data.display_omitted].sort()) === JSON.stringify(['covariance_matrix', 'sample_positions'])
    && !('covariance_matrix' in data) && !('sample_positions' in data)
    && typeof model.latex === 'string' && model.latex.length > 0
    && typeof model.latex_math === 'string' && model.latex_math.length > 0
    && publication.data === model.latex && publication.latex === model.latex
    && typeof publication.latex_math === 'string' && publication.latex_math.length > 0
    && Array.isArray(events) && events.length === 3
    && events.slice(0, 2).every((event, index) => event?.type === 'output' && event.index === index
      && Object.keys(event).sort().join(',') === 'index,type')
    && events[2]?.type === 'stdout' && events[2].text === stdout && Object.keys(events[2]).sort().join(',') === 'text,type');
}

export function fixturePresentationMatches(document, stdout = EXPECTED_STDOUT) {
  const pane = document.querySelector('.output-pane .output-scroll');
  if (!pane) return false;
  const visible = node => {
    const style = node && document.defaultView?.getComputedStyle(node);
    return Boolean(node && !node.closest('[hidden]') && style?.visibility !== 'hidden'
      && style?.display !== 'none' && style?.opacity !== '0'
      && node.getBoundingClientRect().width > 1 && node.getBoundingClientRect().height > 1);
  };
  const models = [...pane.querySelectorAll('.model-output')].filter(visible);
  const printed = [...pane.querySelectorAll('.text-output')].filter(visible);
  if (models.length !== 1 || printed.length !== 1 || printed[0].textContent !== stdout) return false;
  const model = models[0];
  if (!['OLS', 'wage', '480 observations'].every(value => model.querySelector('.model-title')?.textContent.includes(value))
      || !model.querySelector('.model-metrics')?.textContent.includes('HC3 standard errors')) return false;
  const previews = [...pane.querySelectorAll('.latex-preview')].filter(node => visible(node) && node.querySelector('math'));
  const modelMath = previews.some(node => node.closest('.model-output') === model);
  const rows = [...model.querySelectorAll('table tbody tr')].filter(visible);
  const diagnosticTable = rows.length === 3
    && rows.every((row, index) => row.querySelector('th')?.textContent === ['Intercept', 'education', 'experience'][index]);
  const publicationTable = rows.length === 6 && rows.every((row, index) => index % 2 === 0
    ? row.classList.contains('publication-estimate')
      && row.querySelector('th')?.textContent === ['Intercept', 'education', 'experience'][index / 2]
    : row.classList.contains('publication-standard-error') && row.querySelector('th')?.textContent === ''
      && /^\([^()]+\)$/.test(row.querySelector('td')?.textContent ?? ''));
  // A source fallback is useful to the user, but it cannot certify this route's
  // claimed rendered publication. Wait for the separate actual MathML output.
  const publicationMath = previews.filter(node => !node.closest('.model-output'));
  return (modelMath || diagnosticTable || publicationTable) && publicationMath.length === 1;
}

async function localSnapshot(client, projectId) {
  const prefix = `/api/desktop/projects/${projectId}/workspace`;
  return client.evaluate(`(async()=>{
    const session=await fetch(${JSON.stringify(prefix + '/session')}).then(response=>response.ok?response.json():null);
    if(!session?.token)throw new Error('Missing local session');
    const read=async path=>{const response=await fetch(${JSON.stringify(prefix)}+path,{headers:{'X-OpenEcon-Token':session.token}});if(!response.ok)throw new Error('Local read failed');return response.json();};
    const [consoleState,sharing,outbox]=await Promise.all([read('/console'),read('/desktop-sharing'),read('/desktop-outbox')]);
    const matches=consoleState.history.filter(record=>record.stdout?.split('\\n').includes(${JSON.stringify(MARKER)}));
    if(matches.length!==1)return {history_matches:matches.length};
    const record=matches[0],model=record.outputs?.find(output=>output.type==='model');
    return {id:record.id,history_matches:matches.length,status:record.status,nobs:model?.data?.nobs,
      marker_occurrences:record.stdout.split('\\n').filter(line=>line===${JSON.stringify(MARKER)}).length,
      execution_count:record.stdout===${JSON.stringify(EXPECTED_STDOUT)}?1:null,
      fixture_protocol_valid:(${fixtureRecordMatches.toString()})(record,${JSON.stringify(EXECUTION_COMMAND)},${JSON.stringify(EXPECTED_STDOUT)}),
      latex_present:Array.isArray(record.outputs)&&record.outputs.length===2&&record.outputs.every(output=>typeof output.latex==='string'&&output.latex.length>0),
      payload_sha256:await ${payloadDigestExpression('record', true)},
      raw_payload_sha256:await ${payloadDigestExpression('record')},
      sharing_state:sharing.records.find(item=>item.id===record.id)?.state,
      retryable:sharing.records.find(item=>item.id===record.id)?.retryable,
      outbox_matches:outbox.items.filter(item=>item.record.id===record.id).length};
  })()`);
}

function assertOriginal(snapshot, baseline) {
  if (snapshot.history_matches !== 1 || snapshot.status !== 'ok' || snapshot.nobs !== 480
      || snapshot.marker_occurrences !== 1 || snapshot.execution_count !== 1 || !snapshot.latex_present || !snapshot.fixture_protocol_valid)
    throw new Error('The actual persisted native model and execution counter are invalid.');
  if (baseline && (snapshot.id !== baseline.original.id || snapshot.payload_sha256 !== baseline.original.payload_sha256
      || snapshot.raw_payload_sha256 !== baseline.original.raw_payload_sha256))
    throw new Error('The original model, publication LaTeX, stdout, or ordered events changed.');
}

async function visibleOriginal(client) {
  await until(() => client.evaluate(`(${fixturePresentationMatches.toString()})(document,${JSON.stringify(EXPECTED_STDOUT)})`));
}

async function run(options) {
  if (process.platform !== 'win32' || process.env.GITHUB_ACTIONS !== 'true')
    throw new Error('Only an explicitly provisioned disposable GitHub Windows runner is supported.');
  const paths = await validateControllerPaths(options);
  const baseline = paths.baseline ? validateBaseline(JSON.parse(await readFile(paths.baseline, 'utf8')), options['project-id']) : null;
  const onlineBytes = paths['online-identity'] ? await readFile(paths['online-identity']) : null;
  const onlineIdentity = onlineBytes ? JSON.parse(onlineBytes.toString('utf8')) : null;
  const receiptOutput = await reserveReceiptOutput(paths.output);
  const receipt = { status: 'running', phase: options.phase, project_id: options['project-id'],
    controller: 'Real installed Windows WebView2 and native broker; loopback CDP input', checks: {} };
  let native, browser;
  try {
    native = await Cdp.connect(options.port, 'native');
    if (!await native.evaluate(`typeof window.__TAURI__?.core?.invoke==='function' && !!window.__TAURI_INTERNALS__`))
      throw new Error('An actual installed native bridge is required.');
    const nativeInfo = await native.evaluate(`window.__TAURI__.core.invoke('desktop_info')`);
    receipt.native_identity = nativeIdentitySummary(nativeInfo, await native.evaluate(`location.origin`));
    const workspaceIdentity = await canonicalWorkspaceIdentity(nativeInfo.data_root);
    // The runner separately restricts the manifest's Windows ACL to its QA user.
    const state = await readAdmittedQaManifest(paths.state, options, workspaceIdentity, baseline ?? onlineIdentity);
    receipt.native_identity.workspace_identity = workspaceIdentity;
    receipt.checks.canonical_workspace_identity_before_credentials = true;
    if (options.phase === 'viewer-readback') receipt.checks.distinct_viewer_workspace_before_credentials = true;
    if (baseline) validateBaseline(baseline, options['project-id'], state.users.owner);
    if (onlineIdentity) validateOnlineIdentityReceipt(onlineIdentity, options['project-id'], state.users.owner, workspaceIdentity);
    const role = options.phase === 'viewer-readback' ? 'viewer' : 'owner';
    const user = state.users[role];
    if (options.phase === 'retry-online')
      validateOfflineIdentityReceipt(baseline, options['project-id'], user, workspaceIdentity);
    receipt.checks.installed_native_window_and_trusted_ipc = true;
    await installObserver(native, options['project-id']);
    await enterTeams(native);
    if (options.phase === 'browser-approval') {
      if (await native.evaluate(`!!document.querySelector('.team-logout')`)) await native.click('.team-logout');
      await native.click('.team-google-button');
      let grant;
      await until(async () => {
        grant = await native.evaluate(`window.__market87Observer.grant`);
        return /^[0-9a-f]{32}$/.test(grant ?? '')
          && await native.evaluate(`window.__market87Observer.opened_grant===${JSON.stringify(grant)}`);
      });
      state.desktop_login_ids = [...new Set([...(state.desktop_login_ids ?? []), grant])];
      // Preserve the private manifest so only this real grant can later be cleaned.
      await writeFile(paths.state, JSON.stringify(state, null, 2) + '\n', { mode: 0o600 });
      browser = await Cdp.connect(options['browser-port'], 'browser', grant);
      if (await browser.evaluate(`!!document.querySelector('.team-auth input[type="email"]')`)) {
        await browser.fill('.team-auth input[type="email"]', user.email);
        await browser.fill('.team-auth input[type="password"]', user.password);
        await browser.textButton('Sign in');
      }
      await until(() => browser.evaluate(`document.querySelector('.team-auth')?.innerText.includes(${JSON.stringify(user.email)}) && document.querySelector('.team-auth strong')?.textContent===${JSON.stringify(grant.slice(0, 8).toUpperCase())}`));
      await browser.textButton('Approve sign-in');
      await until(() => browser.evaluate(`document.querySelector('h1')?.textContent==='Sign-in complete'`));
      await until(() => native.evaluate(`document.querySelector('.team-user-email')?.title===${JSON.stringify(user.email)}`));
      receipt.checks.actual_native_grant_opened_in_owned_system_browser = true;
      receipt.checks.password_identity_approved_real_browser_handoff = true;
      receipt.google_provider_acceptance = 'not_tested';
    } else if (!['offline-run', 'offline-reopen'].includes(options.phase)) {
      await passwordSignIn(native, user);
      receipt.checks.actual_password_sign_in_or_same_verified_persisted_actor = true;
    }
    const offline = ['offline-run', 'offline-reopen'].includes(options.phase);
    if (offline) {
      await openProject(native, options, user, role);
      receipt.authenticated_identity = options.phase === 'offline-run'
        ? offlineIdentityReceipt(onlineIdentity, user, options['project-id'], workspaceIdentity, identityHash(onlineBytes))
        : validateOfflineIdentityReceipt(baseline, options['project-id'], user, workspaceIdentity);
      receipt.native_identity.current_authenticated_uid_binding = 'offline cached; bound to original online identity receipt, no live verification';
      receipt.checks.offline_cached_owner_bound_to_original_online_identity = true;
    } else {
      receipt.authenticated_identity = await refreshLiveIdentity(native, options, user, role);
      receipt.native_identity.current_authenticated_uid_binding = 'verified by fresh authenticated native broker GET /api/me';
      receipt.checks.fresh_authenticated_uid_and_live_project_role = true;
    }
    receipt.checks.actual_owned_team_project_and_role = true;
    if (offline) {
      await nativeCloudReachability(native, false);
      receipt.checks.native_cloud_unreachable = true;
    } else {
      await nativeCloudReachability(native, true);
      receipt.checks.native_cloud_reachable = true;
    }
    if (options.phase === 'offline-run') {
      const before = await native.evaluate(`(async()=>{
        const prefix=${JSON.stringify(`/api/desktop/projects/${options['project-id']}/workspace`)};
        const session=await fetch(prefix+'/session').then(response=>response.json());
        const response=await fetch(prefix+'/console',{headers:{'X-OpenEcon-Token':session.token}});
        if(!response.ok)throw new Error('Local read failed');
        const state=await response.json();return state.history.length;
      })()`);
      if (before !== 0) throw new Error('A fresh disposable project with empty local history is required.');
      await native.fill('[aria-label="Python, pip, or uv command"]', EXECUTION_COMMAND);
      await native.click('[aria-label="Run command"]');
      await visibleOriginal(native);
      let snapshot;
      await until(async () => {
        snapshot = await localSnapshot(native, options['project-id']);
        return snapshot.history_matches === 1 && snapshot.retryable === true
          && snapshot.sharing_state === 'pending' && snapshot.outbox_matches === 1;
      });
      assertOriginal(snapshot);
      if (!await native.evaluate(`document.querySelector('.result-sharing-state')?.textContent==='Waiting to share' && document.querySelector('.result-sharing-error')?.textContent.length>0`))
        throw new Error('The actual sharing status and connectivity error are not visible.');
      receipt.original = snapshot;
      receipt.checks.one_local_hc3_model_with_original_publication_and_events = true;
      receipt.checks.visible_pending_count_and_connectivity_error = true;
    } else if (['offline-reopen', 'retry-online'].includes(options.phase)) {
      await visibleOriginal(native);
      const before = await localSnapshot(native, options['project-id']);
      assertOriginal(before, baseline);
      if (options.phase === 'offline-reopen') {
        if (before.sharing_state !== 'pending' || before.outbox_matches !== 1)
          throw new Error('Cold reopen did not preserve the original pending outbox.');
        await native.textButton('Retry sharing');
        await until(() => native.evaluate(`document.querySelector('.result-sharing-error')?.textContent.length>0`));
        const after = await localSnapshot(native, options['project-id']);
        assertOriginal(after, baseline);
        if (after.sharing_state !== 'pending' || after.outbox_matches !== 1)
          throw new Error('Offline manual retry did not preserve the same pending outbox.');
        receipt.original = after;
        receipt.checks.visible_cold_reopen_and_manual_retry_preserve_original = true;
      } else {
        if (before.sharing_state !== 'shared') await native.textButton('Retry sharing');
        await until(async () => {
          const current = await localSnapshot(native, options['project-id']);
          return current.sharing_state === 'shared' && current.outbox_matches === 0;
        });
        const after = await localSnapshot(native, options['project-id']);
        assertOriginal(after, baseline);
        if (!await native.evaluate(`document.querySelector('.result-sharing-state')?.textContent==='Shared to team'`))
          throw new Error('Successful sharing is not visible in the actual native UI.');
        receipt.original = after;
        receipt.archive_id = createHash('sha256').update(`${state.users.owner.uid}:${after.id}`).digest('hex').slice(0, 32);
        receipt.checks.real_reconnect_delivered_same_record_and_cleared_outbox = true;
      }
    } else if (options.phase === 'viewer-readback') {
      const archiveId = createHash('sha256').update(`${state.users.owner.uid}:${baseline.original.id}`).digest('hex').slice(0, 32);
      if (!await native.evaluate(`document.querySelector('[aria-label="Run command"]')?.disabled===true && document.querySelector('[aria-label="Python, pip, or uv command"]')?.disabled===true && document.querySelector('.workspace-terminal-status')?.textContent==='Read only'`))
        throw new Error('The real second member can execute in a viewer workspace.');
      await native.textButton('History');
      await native.textButton('Shared history');
      await native.fill('.history-search input', archiveId);
      await native.textButton('Search / refresh');
      await until(() => native.evaluate(`(() => {
        const rows=[...document.querySelectorAll('.history-list button')].filter(node=>node.querySelector('small')?.textContent.includes(${JSON.stringify(archiveId)}));
        if(rows.length!==1 || rows[0].disabled)return false;
        rows[0].setAttribute('data-market87-archive','true');return true;
      })()`));
      await native.click('[data-market87-archive="true"]');
      await visibleOriginal(native);
      const archived = await native.evaluate(`(async()=>{
        const record=window.__market87Observer.archive;
        if(!record || record.id!==${JSON.stringify(archiveId)})return null;
        return {id:record.id,payload_sha256:await ${payloadDigestExpression('record', true)},nobs:record.outputs.find(output=>output.type==='model')?.data?.nobs,
          fixture_protocol_valid:(${fixtureRecordMatches.toString()})(record,${JSON.stringify(EXECUTION_COMMAND)},${JSON.stringify(EXPECTED_STDOUT)})};
      })()`);
      if (archived?.payload_sha256 !== baseline.original.payload_sha256 || archived.nobs !== 480 || !archived.fixture_protocol_valid)
        throw new Error('The actual second member archive changed the original model, LaTeX, or ordered events.');
      receipt.archive = archived;
      receipt.checks.distinct_real_native_viewer_read_same_archived_result = true;
      receipt.checks.viewer_run_control_disabled = true;
    }
    if (offline) {
      if (await native.evaluate(`window.__market87Observer.profile_successes`) !== 0)
        throw new Error('An offline identity phase unexpectedly observed a successful live profile request.');
    } else {
      const first = receipt.authenticated_identity;
      const final = await refreshLiveIdentity(native, options, user, role);
      if (final.profile_request_sequence <= first.profile_request_sequence)
        throw new Error('The online phase did not freshly recheck the same authenticated actor.');
      receipt.authenticated_identity = { ...final, initial_profile_request_sequence: first.profile_request_sequence };
      receipt.checks.fresh_authenticated_uid_and_live_project_role_after_phase = true;
    }
    const audit = await native.evaluate(`({local_executions:window.__market87Observer.local_executions,cloud_executions:window.__market87Observer.cloud_executions})`);
    if (audit.cloud_executions !== 0 || audit.local_executions !== (options.phase === 'offline-run' ? 1 : 0))
      throw new Error('Retry, reopen or member readback triggered computation.');
    receipt.observed_execution_requests = audit;
    receipt.checks.no_cloud_compute_and_no_recompute_in_observed_phase = true;
    const finalInfo = await native.evaluate(`window.__TAURI__.core.invoke('desktop_info')`);
    nativeIdentitySummary(finalInfo, await native.evaluate(`location.origin`));
    requireSameWorkspaceIdentity(workspaceIdentity, await canonicalWorkspaceIdentity(finalInfo.data_root));
    receipt.checks.workspace_identity_rechecked_after_phase = true;
    receipt.status = 'passed';
  } catch {
    receipt.status = 'error';
    receipt.error = 'Installed Windows cloud UI phase failed. Inspect the sanitized completed checkpoints.';
    process.exitCode = 1;
  } finally {
    if (native) { try { await native.evaluate(`window.__market87Observer?.restore()`); } catch { /* Owned process may have closed. */ } native.close(); }
    browser?.close();
    try { await receiptOutput.writeFile(JSON.stringify(receipt, null, 2) + '\n'); }
    finally { await receiptOutput.close(); }
    console.log(JSON.stringify({ status: receipt.status, phase: receipt.phase, checks: Object.keys(receipt.checks).length }));
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { await run(parseArguments(process.argv.slice(2))); }
  catch { console.error('Windows cloud UI controller refused the runner, arguments, or private fixture.'); process.exitCode = 1; }
}
