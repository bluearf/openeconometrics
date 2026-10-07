import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { JSDOM } from 'jsdom';

const asset = (name: string) => fileURLToPath(new URL('../../packages/openecon-charts/src/openecon_charts/assets/' + name, import.meta.url));
const rendererCode = readFileSync(asset('network-renderer.js'), 'utf8');
const workerCode = readFileSync(asset('network-worker.js'), 'utf8');
const d3Code = readFileSync(asset('d3.min.js'), 'utf8');

function plot(count = 4, overrides: any = {}) {
  const nodes = Array.from({ length: count }, (_, id) => ({ id, label: 'Node ' + id, degree: id + 1, group: id % 3 }));
  const edges = nodes.slice(1).map(node => ({ source: node.id - 1, target: node.id, weight: node.id % 2 ? -0.5 : 2 }));
  const graph = { nodes, edges, directed: false, node_count: count, edge_count: edges.length,
    shown_node_count: count, shown_edge_count: edges.length, sampled: false, selection: 'All nodes and edges', ...overrides };
  return { kind: 'network', title: 'Ağ örneği', x_label: '', y_label: '', data: [],
    sample_n: graph.shown_node_count, total_n: graph.node_count, dropped_n: 0,
    config: { network: graph, options: {} } };
}

function setup(options: { worker?: boolean; inline?: boolean; scriptOrigin?: string } = {}) {
  const dom = new JSDOM('<!doctype html><html><body><main><div id="chart"></div></main></body></html>', {
    runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost:8765/',
  });
  const window = dom.window as any;
  const frames = new Map<number, () => void>();
  let nextFrame = 0, activeObservers = 0;
  window.requestAnimationFrame = (callback: () => void) => { frames.set(++nextFrame, callback); return nextFrame; };
  window.cancelAnimationFrame = (id: number) => frames.delete(id);
  const observers: any[] = [];
  window.ResizeObserver = class {
    active = false;
    callback: () => void;
    constructor(callback: () => void) { this.callback = callback; observers.push(this); }
    observe() { this.active = true; activeObservers++; }
    disconnect() { if (this.active) activeObservers--; this.active = false; }
  };
  const contexts = new Map<any, any>();
  window.HTMLCanvasElement.prototype.getContext = function () {
    if (!contexts.has(this)) {
      const calls: any[][] = [];
      const context: any = { calls, measureText: (text: string) => ({ width: Array.from(text).length * 7 }) };
      for (const name of ['setTransform', 'clearRect', 'fillRect', 'save', 'restore', 'translate', 'scale', 'beginPath',
        'moveTo', 'lineTo', 'arc', 'closePath', 'stroke', 'fill', 'strokeText', 'fillText', 'drawImage'])
        context[name] = (...args: any[]) => calls.push([name, ...args]);
      contexts.set(this, context);
    }
    return contexts.get(this);
  };
  window.HTMLCanvasElement.prototype.toBlob = function (callback: (blob: any) => void) {
    callback(new window.Blob(['png'], { type: 'image/png' }));
  };
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  window.HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open'); this.dispatchEvent(new window.Event('close'));
  };
  const workers: any[] = [];
  if (options.worker) window.Worker = class {
    url: string;
    messages: any[] = [];
    terminated = false;
    onmessage: any = null;
    onerror: any = null;
    constructor(url: string) { this.url = url; workers.push(this); }
    postMessage(message: any) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    emit(message: any) { this.onmessage?.({ data: message }); }
  };
  else window.Worker = undefined;
  if (!options.inline) Object.defineProperty(window.document, 'currentScript', {
    value: { src: (options.scriptOrigin || 'http://localhost:8765') + '/chart-assets/network-renderer.js' },
    configurable: true,
  });
  window.eval(rendererCode);
  const host = window.document.querySelector('#chart') as any;
  Object.defineProperty(host, 'clientWidth', { value: 640, configurable: true });
  const flush = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach(callback => callback()); };
  const api = window.OpenEconNetworkCharts;
  const close = () => { api.unmount(host); dom.window.close(); };
  return { dom, window, host, api, workers, contexts, frames, observers, flush, close, observed: () => activeObservers };
}

async function blobText(window: any, blob: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new window.FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject; reader.readAsText(blob);
  });
}

function timeline(options: any = {}) {
  const spec = plot();
  spec.config.network.frames = [
    { label: 'First', network: structuredClone(spec.config.network) },
    { label: 'Second', network: plot(5).config.network },
  ];
  spec.config.options = { timeline: true, layout: 'd3-force', ...options };
  if (options.layout === 'fixed') {
    for (const graph of [spec.config.network, ...spec.config.network.frames.map(frame => frame.network)])
      graph.nodes.forEach((node, i) => { node.x = i * 40; node.y = 0; });
  }
  return spec;
}

function watchdogs(window: any) {
  const pending = new Map<number, () => void>();
  const callbacks: (() => void)[] = [];
  let next = 10000;
  const originalTimeout = window.setTimeout.bind(window);
  const originalClear = window.clearTimeout.bind(window);
  window.setTimeout = (callback: () => void, delay: number) => {
    if (delay === 31500) {
      callbacks.push(callback); pending.set(++next, callback); return next;
    }
    return originalTimeout(callback, delay);
  };
  window.clearTimeout = (id: number) => {
    if (!pending.delete(id)) originalClear(id);
  };
  return { pending, callbacks };
}

test('initial timeline frame owns one worker and watchdog, with none after completion', async () => {
  for (const frame_index of [1, 0, undefined]) {
    const env = setup({ worker: true });
    const timers = watchdogs(env.window);
    try {
      const chart = await env.api.mount(env.host, timeline(frame_index === undefined ? {} : { frame_index }));
      assert.equal(env.workers.length, 1, 'Constructor starts only the selected frame layout');
      assert.equal(timers.pending.size, 1);
      assert.equal(chart.frameIndex, frame_index ?? 0);
      const worker = env.workers[0];
      assert.equal(worker.messages[0].nodes.length, frame_index === 1 ? 5 : 4);
      worker.emit({ type: 'done', job: chart.job, positions: Array(chart.positions.length).fill(12) });
      assert.equal(worker.terminated, true);
      assert.equal(timers.pending.size, 0);
      assert.equal(chart.workerTimer, null);
      assert.equal(chart.status.textContent, 'Canvas · Ready');
      timers.callbacks.forEach(callback => callback()); // A queued timer can outlive clearTimeout.
      assert.equal(chart.status.textContent, 'Canvas · Ready');
    } finally { env.close(); }
  }
});

test('rapid frame changes reject retired worker events and watchdogs, including forged current jobs', async () => {
  const env = setup({ worker: true });
  const timers = watchdogs(env.window);
  try {
    const chart = await env.api.mount(env.host, timeline({ frame_index: 1 }));
    const retired: { message: any; error: any; timeout: () => void; job: string }[] = [];
    for (const index of [0, 1, 0, 1]) {
      const old = chart.worker;
      retired.push({ message: old.onmessage, error: old.onerror, timeout: timers.callbacks.at(-1)!, job: chart.job });
      chart.setFrame(index);
      assert.equal(old.terminated, true);
      assert.equal(old.onmessage, null);
      assert.equal(old.onerror, null);
      assert.equal(env.workers.filter(worker => !worker.terminated).length, 1);
      assert.equal(timers.pending.size, 1);
    }
    const active = chart.worker;
    const initial = Array.from(chart.positions);
    for (const old of retired) {
      for (const job of [old.job, chart.job]) {
        old.message({ data: { type: 'progress', job, positions: Array(chart.positions.length).fill(999) } });
        old.message({ data: { type: 'error', job, message: 'retired failure' } });
      }
      old.error(); old.timeout();
    }
    assert.deepEqual(Array.from(chart.positions), initial);
    assert.equal(chart.worker, active);
    assert.equal(active.terminated, false);
    assert.equal(timers.pending.size, 1);
    assert.equal(chart.status.textContent, 'Arranging network…');
    const queuedMessage = active.onmessage, queuedError = active.onerror;
    env.api.unmount(env.host);
    assert.equal(timers.pending.size, 0);
    assert.ok(env.workers.every(worker => worker.terminated));
    queuedMessage({ data: { type: 'done', job: chart.job, positions: Array(chart.positions.length).fill(999) } });
    queuedError(); timers.callbacks.forEach(callback => callback());
    chart.setFrame(0); chart.startLayout();
    assert.equal(env.workers.length, 5, 'Destroyed chart cannot restart a layout');
    assert.deepEqual(Array.from(chart.positions), initial);
    assert.equal(env.frames.size, 0);
  } finally { env.close(); }
});

test('fixed initial timeline creates no worker and explicit layout restarts cancel the current job', async () => {
  const env = setup({ worker: true });
  const timers = watchdogs(env.window);
  try {
    const fixed = await env.api.mount(env.host, timeline({ frame_index: 1, layout: 'fixed' }));
    assert.equal(env.workers.length, 0); assert.equal(timers.pending.size, 0);
    fixed.setFrame(0);
    assert.equal(env.workers.length, 0); assert.equal(timers.pending.size, 0);
    const chart = await env.api.mount(env.host, plot());
    const old = chart.worker, timeout = timers.callbacks.at(-1)!;
    chart.startLayout();
    assert.equal(old.terminated, true);
    assert.equal(env.workers.filter(worker => !worker.terminated).length, 1);
    assert.equal(timers.pending.size, 1);
    timeout();
    assert.equal(chart.worker, env.workers.at(-1));
    timers.callbacks.at(-1)!();
    assert.equal(chart.worker, null); assert.equal(chart.workerTimer, null);
    assert.equal(timers.pending.size, 0);
    assert.equal(chart.status.textContent, 'Layout timeout · Positions retained');
  } finally { env.close(); }
});

test('an initial saved timeline view starts no transient worker and worker errors clear their watchdog', async () => {
  const env = setup({ worker: true });
  const timers = watchdogs(env.window);
  try {
    const spec = timeline({ frame_index: 1 });
    const staticChart = await env.api.mount(env.host, timeline({ frame_index: 1, layout: 'fixed' }));
    spec.config.options.view = JSON.parse(JSON.stringify(staticChart.saveView()));
    const restored = await env.api.mount(env.host, spec);
    assert.equal(env.workers.length, 0);
    assert.equal(timers.pending.size, 0);
    assert.equal(restored.frameIndex, 1);
    for (const error of ['event', 'message']) {
      restored.setFrame(0);
      const worker = restored.worker;
      if (error === 'event') worker.onerror();
      else worker.emit({ type: 'error', job: restored.job, message: 'layout failure' });
      assert.equal(worker.terminated, true);
      assert.equal(restored.worker, null);
      assert.equal(timers.pending.size, 0);
      const status = restored.status.textContent;
      timers.callbacks.forEach(callback => callback());
      assert.equal(restored.status.textContent, status);
      assert.match(status, /^Layout error:/);
    }
  } finally { env.close(); }
});

test('browser normalization enforces the Python network envelope, schema and numeric bounds', () => {
  const env = setup();
  try {
    const base = plot();
    const result = env.api.normalize(base);
    assert.equal(result.edges[0].weight, -0.5);
    result.nodes[0].label = 'changed';
    assert.equal(base.config.network.nodes[0].label, 'Node 0');
    const invalid: ((spec: any) => void)[] = [
      spec => { spec.sample_n = 3; }, spec => { spec.total_n = 100; }, spec => { spec.dropped_n = 1; },
      spec => { spec.data = [{}]; }, spec => { spec.x_label = 'x'; }, spec => { spec.y_label = 'y'; },
      spec => { spec.config.custom = '<script>'; }, spec => { spec.config.network.custom = true; },
      spec => { spec.config.network.nodes[0].custom = true; }, spec => { spec.config.network.edges[0].custom = true; },
      spec => { delete spec.config.network.nodes[0].group; }, spec => { spec.config.network.nodes[1].id = 0; },
      spec => { spec.config.network.nodes[0].id = Number.MAX_SAFE_INTEGER + 1; },
      spec => { spec.config.network.nodes[0].degree = Infinity; }, spec => { spec.config.network.nodes[0].degree = -1; },
      spec => { spec.config.network.nodes[0].group = -1; }, spec => { spec.config.network.edges[0].target = 999; },
      spec => { spec.config.network.edges[0].weight = NaN; }, spec => { spec.config.network.edges[0].weight = true; },
      spec => { spec.config.network.shown_node_count = 3; }, spec => { spec.config.network.node_count = 1; },
      spec => { spec.config.network.sampled = true; }, spec => { spec.config.network.directed = 1; },
      spec => { spec.config.network.selection = ''; }, spec => { spec.config.network.selection = 'ö'.repeat(501); },
      spec => { spec.config.options = null; }, spec => { spec.config.options.width = 320.5; },
      spec => { spec.config.options.height = 240.5; }, spec => { spec.config.options.point_size = 25; },
      spec => { spec.config.options.line_width = 0.2; }, spec => { spec.config.options.opacity = 1.01; },
      spec => { spec.config.options.color = 'url(https://example.test)'; },
      spec => { spec.config.options.palette = []; }, spec => { spec.config.options.palette = Array(65).fill('#fff'); },
      spec => { spec.config.options.annotations = [{ text: 'missing position' }]; }, spec => { spec.config.options.__proto__ = null; spec.config.options = JSON.parse('{"__proto__":2}'); },
    ];
    for (const mutate of invalid) { const spec = plot(); mutate(spec); assert.throws(() => env.api.normalize(spec)); }
    const boundary = plot();
    boundary.config.options = { width: 320, height: 1600, point_size: 24, line_width: 12, opacity: 0,
      color: '#123', palette: Array(12).fill('#abcdef') };
    assert.equal(env.api.normalize(boundary).options.height, 1600);
    const empty = plot(0, { node_count: 0, edge_count: 1, sampled: true });
    assert.throws(() => env.api.normalize(empty), /without nodes/);
  } finally { env.close(); }
});

test('Unicode labels stay exact and bounded in UTF-8, with control and aggregate limits', () => {
  const env = setup();
  try {
    const valid = plot(1);
    valid.config.network.nodes[0].label = '😀'.repeat(1024);
    assert.equal(env.api.normalize(valid).nodes[0].label, valid.config.network.nodes[0].label);
    for (const text of ['😀'.repeat(1025), '\ud800', 'A\u0000B', 'A\u007fB', 'A\u0085B', 'A\u009fB']) {
      const invalid = plot(1); invalid.config.network.nodes[0].label = text;
      assert.throws(() => env.api.normalize(invalid), /label/);
    }
    const aggregate = plot(4097);
    aggregate.config.network.nodes.forEach(node => { node.label = 'a'.repeat(4096); });
    assert.throws(() => env.api.normalize(aggregate), /16 MiB/);
    const maximum = plot(4096);
    maximum.config.network.nodes.forEach(node => { node.label = 'a'.repeat(4096); });
    assert.equal(env.api.normalize(maximum).nodes.length, 4096);
  } finally { env.close(); }
});

test('grouping metadata is optional for old charts, exact for new charts and strictly bounded', () => {
  const env = setup();
  try {
    const old = plot();
    assert.equal(env.api.normalize(old).grouping, 'Weak components');
    assert.equal(Object.hasOwn(old.config.network, 'grouping'), false, 'Normalization does not rewrite persisted old specs');
    for (const method of ['Weak components', 'Leiden communities', 'Louvain communities', 'User groups']) {
      assert.equal(env.api.normalize(plot(4, { grouping: method })).grouping, method);
    }
    for (const grouping of [null, true, 1, '', '\u0085', '\ud800', 'x'.repeat(1001), 'ö'.repeat(501)])
      assert.throws(() => env.api.normalize(plot(4, { grouping })), /Grouping|grouping/);
    assert.equal(env.api.normalize(plot(4, { grouping: 'ö'.repeat(500) })).grouping, 'ö'.repeat(500));
    assert.throws(() => env.api.normalize(plot(4, { grouping: 'User groups', custom: true })), /fields/);
  } finally { env.close(); }
});

test('group methods appear safely in the footer, SVG description, PNG composition and CSV metadata', async () => {
  const env = setup();
  try {
    const method = '<img src=x onerror=alert(1)> User groups';
    const chart = await env.api.mount(env.host, plot(4, { grouping: method }));
    assert.ok(chart.counts.textContent.includes('Groups: ' + method));
    assert.equal(env.host.querySelector('img'), null);
    assert.ok(chart.snapshot().querySelector('desc').textContent.includes('Groups: ' + method));
    assert.ok(chart.exportLayout().footerLines.join('').includes('Groups: ' + method));
    const saved: any[] = [];
    chart.save = (blob: any, extension: string) => saved.push({ blob, extension });
    chart.export('nodes'); chart.export('edges');
    for (const file of saved) assert.ok((await blobText(env.window, file.blob)).includes('Groups: ' + method));
    await env.api.mount(env.host, plot());
    assert.ok(env.api.get(env.host).counts.textContent.includes('Groups: Weak components'));
  } finally { env.close(); }
});

test('display guardrails reject excessive nodes and edges instead of silently truncating', () => {
  const env = setup();
  try {
    assert.equal(env.api.normalize(plot(100000)).nodes.length, 100000);
    assert.throws(() => env.api.normalize(plot(100001)), /render limit/);
    const limit = plot(2);
    limit.config.network.edges = Array.from({ length: 1000000 }, () => ({ source: 0, target: 1, weight: 1 }));
    limit.config.network.edge_count = limit.config.network.shown_edge_count = 1000000;
    assert.equal(env.api.normalize(limit).edges.length, 1000000);
    limit.config.network.edges.push({ source: 0, target: 1, weight: 1 });
    limit.config.network.edge_count = limit.config.network.shown_edge_count = 1000001;
    assert.throws(() => env.api.normalize(limit), /render limit/);
  } finally { env.close(); }
});

test('fallback layout is deterministic and mounts large graphs with constant DOM and bounded labels', async () => {
  const env = setup({ inline: true, worker: true });
  try {
    const spec = plot(2000);
    const normalized = env.api.normalize(spec);
    const first = Array.from(env.api.initialPositions(normalized.nodes));
    assert.deepEqual(first, Array.from(env.api.initialPositions(normalized.nodes)));
    assert.ok(first.every(Number.isFinite));
    const chart = await env.api.mount(env.host, spec);
    assert.equal(env.workers.length, 0, 'An inline document must use offline fallback without creating a worker');
    assert.equal(chart.status.textContent, 'Static layout');
    assert.equal(env.host.querySelectorAll('canvas').length, 2);
    assert.equal(env.host.querySelectorAll('circle').length, 0);
    assert.ok(env.host.querySelectorAll('*').length < 60);
    chart.draw();
    const calls = env.contexts.get(chart.canvas).calls;
    assert.ok(calls.filter(call => call[0] === 'stroke').length <= 8, 'Edges are batched by weight and selection');
    assert.ok(calls.filter(call => call[0] === 'fill').length <= 8, 'Nodes are batched by palette');
    assert.ok(calls.filter(call => call[0] === 'fillText').length <= 80);
    chart.select(0);
    assert.ok(chart.labelIndices().length <= 60);
    assert.ok(chart.related(chart.index.get(1)));
    const seed = env.api.layoutSeed(normalized.nodes, normalized.edges);
    assert.equal(seed, env.api.layoutSeed(normalized.nodes, normalized.edges));
    assert.notEqual(seed, env.api.layoutSeed(normalized.nodes, []));
  } finally { env.close(); }
});

test('search, hover and neighbor selection use safe exact text and preserve full-graph degree', async () => {
  const env = setup();
  try {
    const spec = plot(4, { node_count: 100, edge_count: 200, sampled: true, selection: 'Degree ranking' });
    const label = '<img src=x onerror=alert(1)> supplier';
    spec.config.network.nodes[0].label = label;
    spec.config.network.nodes[0].degree = 98;
    const chart = await env.api.mount(env.host, spec);
    chart.search.value = 'supplier'; chart.search.dispatchEvent(new env.window.Event('input'));
    assert.equal(chart.searchResults.querySelector('button').textContent, label);
    chart.searchResults.querySelector('button').click();
    assert.equal(chart.selected, 0);
    assert.equal(chart.related(1), true); assert.equal(chart.related(3), false);
    assert.match(chart.selectionText.textContent, /Degree: 98 · 1 neighbors shown/);
    assert.equal(env.host.querySelector('img'), null);
    const point = { x: chart.positions[0] * chart.transform.k + chart.transform.x,
      y: chart.positions[1] * chart.transform.k + chart.transform.y };
    assert.equal(chart.hit(point), 0);
    chart.hover(point);
    assert.equal(chart.tooltip.hidden, false);
    assert.ok(chart.tooltip.textContent.startsWith(label + '\nDegree: 98'));
    assert.equal(chart.tooltip.querySelector('img'), null);
    assert.match(chart.counts.textContent, /Showing 4 of 100 nodes · 3 of 200 edges/);
    chart.clear.click(); assert.equal(chart.selected, null);
    assert.equal(chart.clear.hidden, true);
  } finally { env.close(); }
});

test('dense groups reject colliding automatic captions and SVG uses the same visible labels', async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot(36));
    chart.transform = { k: 1, x: 0, y: 0 };
    chart.config.nodes.forEach((node: any, index: number) => {
      chart.positions[index * 2] = 100 + Math.floor(index / 12) * 210 + index % 12;
      chart.positions[index * 2 + 1] = 160 + index % 3;
    });
    const context = env.contexts.get(chart.canvas);
    context.calls.length = 0; chart.draw();
    const captions = context.calls.filter((call: any[]) => call[0] === 'fillText');
    assert.ok(captions.length >= 3 && captions.length < 36, 'Each cluster has readable labels without drawing every overlapping caption');
    const boxes = captions.map((call: any[]) => ({ left: call[2] - 3,
      right: call[2] + context.measureText(call[1]).width + 3, top: call[3] - 9, bottom: call[3] + 9 }));
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      assert.ok(a.left >= b.right || a.right <= b.left || a.top >= b.bottom || a.bottom <= b.top,
        'Automatic captions must not overlap in screen space');
    }
    const exported = [...chart.snapshot().querySelectorAll('svg > g > text')].map((node: any) => node.textContent);
    assert.deepEqual(exported, captions.map((call: any[]) => call[1]));
  } finally { env.close(); }
});

test('caption collisions follow screen-space zoom and keep selected and hovered exact identities', async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot(3));
    chart.positions = new Float32Array([0, 0, 40, 0, 80, 0]);
    chart.transform = { k: 1, x: 100, y: 100 };
    assert.equal(chart.labelIndices().length, 2);
    chart.transform.k = 2;
    assert.equal(chart.labelIndices().length, 3, 'Zoom makes space for the third complete caption');

    const spec = plot(100);
    const exact = 'Supplier 🌍 ' + 'α'.repeat(100);
    spec.config.network.nodes[0].label = exact;
    const large = await env.api.mount(env.host, spec);
    large.positions.fill(0); large.transform = { k: 1, x: 320, y: 160 };
    large.selected = 0; large.hovered = 99;
    const context = env.contexts.get(large.canvas);
    context.calls.length = 0; large.draw();
    const captions = context.calls.filter((call: any[]) => call[0] === 'fillText').map((call: any[]) => call[1]);
    assert.ok(captions.includes(exact), 'A selected long identity is never truncated or removed');
    assert.ok(captions.includes('Node 99'), 'An unrelated hovered node remains visible while another is selected');
    assert.equal(new Set(captions).size, captions.length);
    assert.ok(captions.length <= 60);
    const exported = [...large.snapshot().querySelectorAll('svg > g > text')].map((node: any) => node.textContent);
    assert.deepEqual(exported, captions);
    large.selected = large.hovered = null;
    assert.equal(large.labelIndices().length, 0, 'Large unselected graphs keep the existing low-zoom label bound');
    large.transform.k = 2;
    assert.ok(large.labelCandidates().length <= 80);
  } finally { env.close(); }
});

test('zoom keeps the pointer coordinate anchored, respects limits and fit handles empty graphs', async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot());
    const x = 173, y = 112;
    const before = [(x - chart.transform.x) / chart.transform.k, (y - chart.transform.y) / chart.transform.k];
    chart.zoom(1.3, x, y);
    assert.ok(Math.abs((x - chart.transform.x) / chart.transform.k - before[0]) < 1e-8);
    assert.ok(Math.abs((y - chart.transform.y) / chart.transform.k - before[1]) < 1e-8);
    chart.zoom(1e9, x, y); assert.equal(chart.transform.k, 12);
    chart.zoom(1e-20, x, y); assert.equal(chart.transform.k, 0.02);
    chart.fit(); assert.ok(Object.values(chart.transform).every(Number.isFinite));
    const empty = await env.api.mount(env.host, plot(0));
    assert.ok(Object.values(empty.transform).every(Number.isFinite));
    assert.equal(empty.status.textContent, 'No nodes to display');
  } finally { env.close(); }
});

test('hover tooltip expires after three seconds and owns only one cancellable timer', async () => {
  const env = setup();
  try {
    const pending = new Map<number, () => void>(); let next = 10000;
    const originalTimeout = env.window.setTimeout.bind(env.window), originalClear = env.window.clearTimeout.bind(env.window);
    env.window.setTimeout = (callback: () => void, delay: number) => {
      if (delay === 3000) { pending.set(++next, callback); return next; }
      return originalTimeout(callback, delay);
    };
    env.window.clearTimeout = (id: number) => { if (!pending.delete(id)) originalClear(id); };
    const chart = await env.api.mount(env.host, plot(1));
    const point = { x: chart.positions[0] * chart.transform.k + chart.transform.x,
      y: chart.positions[1] * chart.transform.k + chart.transform.y };
    for (let index = 0; index < 20; index++) chart.hover(point);
    assert.equal(pending.size, 1); assert.equal(chart.tooltip.hidden, false);
    const callback = [...pending.values()][0]; pending.clear(); callback();
    assert.equal(chart.tooltip.hidden, true); assert.equal(chart.tooltipTimer, null);
    chart.hover(point); chart.zoom(1.1, 100, 100);
    assert.equal(pending.size, 0); assert.equal(chart.tooltip.hidden, true);
    chart.hover(point); chart.canvas.dispatchEvent(new env.window.Event('pointerleave'));
    assert.equal(pending.size, 0); assert.equal(chart.tooltip.hidden, true);
    chart.hover({ x: chart.positions[0] * chart.transform.k + chart.transform.x,
      y: chart.positions[1] * chart.transform.k + chart.transform.y });
    assert.equal(pending.size, 1);
    env.api.unmount(env.host); assert.equal(pending.size, 0);
  } finally { env.close(); }
});

test('same-origin worker protocol accepts bounded progress and ignores stale or invalid coordinates', async () => {
  const env = setup({ worker: true });
  try {
    const chart = await env.api.mount(env.host, plot());
    const worker = env.workers[0], request = worker.messages[0];
    assert.equal(worker.url, 'http://localhost:8765/chart-assets/network-worker.js');
    assert.equal(request.type, 'layout'); assert.equal(request.job, chart.job);
    assert.equal(request.nodes[0].label, undefined, 'Labels do not enter the layout worker');
    assert.ok(Number.isSafeInteger(request.seed));
    const initial = Array.from(chart.positions);
    worker.emit({ type: 'progress', job: 'old-job', positions: Array(8).fill(100) });
    worker.emit({ type: 'progress', job: chart.job, positions: [NaN, ...Array(7).fill(1)] });
    worker.emit({ type: 'progress', job: chart.job, positions: Array(8).fill(1e300) });
    worker.emit({ type: 'progress', job: chart.job, positions: Array(6).fill(1) });
    assert.deepEqual(Array.from(chart.positions), initial);
    const coordinates = new Float32Array([-80, 0, -20, 10, 20, -10, 80, 0]);
    worker.emit({ type: 'progress', job: chart.job, positions: coordinates });
    assert.deepEqual(Array.from(chart.positions), Array.from(coordinates));
    chart.zoom(1.3, 100, 100);
    const touched = { ...chart.transform };
    worker.emit({ type: 'done', job: chart.job, positions: coordinates });
    assert.deepEqual({ ...chart.transform }, touched);
    assert.equal(chart.status.textContent, 'Canvas · Ready'); assert.equal(chart.worker, null);
    assert.equal(worker.terminated, true);
    assert.equal(worker.messages.at(-1).type, 'cancel');
    assert.equal(chart.workerTimer, null);
  } finally { env.close(); }
});

test('failed or cross-origin worker availability retains the deterministic fallback', async () => {
  const foreign = setup({ worker: true, scriptOrigin: 'https://example.test' });
  try {
    const chart = await foreign.api.mount(foreign.host, plot());
    assert.equal(foreign.workers.length, 0); assert.equal(chart.status.textContent, 'Static layout');
  } finally { foreign.close(); }
  const env = setup({ worker: true });
  try {
    const chart = await env.api.mount(env.host, plot()), worker = env.workers[0];
    const positions = Array.from(chart.positions);
    worker.emit({ type: 'error', job: chart.job });
    assert.equal(worker.terminated, true); assert.match(chart.status.textContent, /Layout error/);
    assert.deepEqual(Array.from(chart.positions), positions);
  } finally { env.close(); }
});

test('unmount terminates work, removes owned listeners and observers, and preserves unrelated hosts', async () => {
  const env = setup({ worker: true });
  try {
    const unrelated = env.window.document.createElement('div');
    unrelated.className = 'bluearf-chart'; unrelated.dataset.chartType = 'line'; unrelated.textContent = 'Keep this plot';
    env.api.unmount(unrelated);
    assert.equal(unrelated.textContent, 'Keep this plot'); assert.equal(unrelated.className, 'bluearf-chart');
    assert.equal(unrelated.dataset.chartType, 'line');
    const chart = await env.api.mount(env.host, plot()), worker = env.workers[0];
    const input = chart.search, results = chart.searchResults;
    assert.equal(env.observed(), 1); assert.ok(env.frames.size > 0);
    env.api.unmount(env.host);
    assert.equal(worker.terminated, true); assert.equal(worker.messages.at(-1).type, 'cancel');
    assert.equal(env.observed(), 0); assert.equal(env.frames.size, 0); assert.equal(chart.abort.signal.aborted, true);
    assert.equal(env.api.get(env.host), undefined); assert.equal(env.host.children.length, 0);
    input.value = 'Node'; input.dispatchEvent(new env.window.Event('input'));
    assert.equal(results.children.length, 0, 'Detached inputs no longer hold active chart listeners');
    assert.throws(() => env.api.snapshot(env.host), /No network chart/);
    env.api.unmount(env.host);
  } finally { env.close(); }
});

test('remount cancels the previous job and external host removal cleans up the new chart', async () => {
  const env = setup({ worker: true });
  try {
    const previous = await env.api.mount(env.host, plot());
    const current = await env.api.mount(env.host, plot(2));
    assert.equal(previous.destroyed, true); assert.equal(env.workers[0].terminated, true);
    assert.equal(env.observed(), 1); assert.equal(env.api.get(env.host), current);
    env.host.remove();
    await Promise.resolve(); env.flush();
    assert.equal(current.destroyed, true); assert.equal(env.workers[1].terminated, true);
    assert.equal(env.observed(), 0); assert.equal(env.api.get(env.host), undefined);
  } finally { env.close(); }
});

test('expanded view owns a separate worker and closes without leaving a detached chart', async () => {
  const env = setup({ worker: true });
  try {
    const chart = await env.api.mount(env.host, plot());
    chart.expand();
    const dialog = env.window.document.querySelector('dialog');
    const expanded = dialog.querySelector('.oe-network-expanded');
    assert.equal(env.observed(), 2); assert.equal(env.workers.length, 2);
    assert.equal(env.api.get(expanded).toolbar.querySelector('[data-action="expand"]').disabled, true);
    dialog.close();
    assert.equal(env.window.document.querySelector('dialog'), null);
    assert.equal(env.workers[1].terminated, true); assert.equal(env.observed(), 1);
    assert.equal(env.api.get(expanded), undefined); assert.equal(chart.destroyed, false);
    chart.expand();
    env.api.unmount(env.host);
    assert.equal(env.workers[2].terminated, true); assert.equal(env.observed(), 0);
    assert.equal(env.window.document.querySelector('dialog'), null);
  } finally { env.close(); }
});

test('SVG and CSV preserve safe labels, signed weights, explicit subset totals and export direction', async () => {
  const env = setup();
  try {
    const spec = plot(3, { node_count: 9999, edge_count: 12345, sampled: true, directed: true,
      selection: 'Top degree nodes, deterministic tie breaks' });
    spec.config.network.nodes[0].label = '=SUM(A1:A2)';
    spec.config.network.nodes[1].label = '<script>alert("x")</script>';
    spec.config.network.edges.push({ source: 1, target: 1, weight: 0 });
    spec.config.network.shown_edge_count = spec.config.network.edges.length;
    const chart = await env.api.mount(env.host, spec);
    const snapshot = env.api.snapshot(env.host);
    assert.equal(snapshot.nodeName, 'svg'); assert.equal(snapshot.querySelectorAll('circle').length, 3);
    assert.equal(snapshot.querySelector('script'), null);
    assert.equal(snapshot.querySelectorAll('polygon').length, 2);
    assert.match(snapshot.querySelector('desc').textContent, /Showing 3 of 9,999 nodes · 3 of 12,345 edges · Directed/);
    assert.ok([...snapshot.querySelectorAll('circle title')].some(node => node.textContent.includes('<script>alert("x")</script>')));
    const drawing = snapshot.querySelector('svg > g');
    assert.equal(drawing.getAttribute('transform'), 'translate(' + chart.transform.x + ' ' + chart.transform.y + ') scale(' + chart.transform.k + ')');
    const saved: any[] = [];
    chart.save = (blob: any, extension: string) => saved.push({ blob, extension });
    chart.export('nodes'); chart.export('edges'); chart.export('svg'); chart.export('png');
    const nodesCSV = await blobText(env.window, saved[0].blob), edgesCSV = await blobText(env.window, saved[1].blob);
    assert.match(nodesCSV, /Showing 3 of 9,999 nodes/); assert.match(nodesCSV, /Rendered filtered network rows only/);
    assert.match(nodesCSV, /0,integer,0,.*'=SUM\(A1:A2\),1,0/);
    assert.match(edgesCSV, /0,1,integer,0,.*integer,1,.*-0.5/);
    assert.equal(env.api.csvCell(-0.5), '-0.5'); assert.equal(env.api.csvCell('-value'), "'-value");
    assert.equal(saved[2].extension, 'svg'); assert.equal(saved[3].extension, 'png');
    const xml = await blobText(env.window, saved[2].blob);
    assert.match(xml, /&lt;script&gt;/); assert.doesNotMatch(xml, /<script>/);
  } finally { env.close(); }
});

test('long export titles and sampled descriptions wrap within the document and keep exact metadata', async () => {
  const env = setup();
  try {
    const spec = plot(1, { node_count: 100, edge_count: 100, sampled: true, selection: 'S'.repeat(1000) });
    spec.title = 'Very long title '.repeat(30);
    const chart = await env.api.mount(env.host, spec);
    const layout = chart.exportLayout(), svg = chart.snapshot();
    assert.ok(layout.titleLines.length > 1); assert.ok(layout.footerLines.length > 1);
    assert.equal(layout.titleLines.join(''), spec.title);
    assert.equal(layout.footerLines.join(''), env.api.coverage(chart.config));
    assert.equal(Number(svg.getAttribute('height')), layout.height);
    assert.equal(svg.querySelector('title').textContent, spec.title);
    assert.ok(Number(svg.querySelector('svg').getAttribute('y')) >= 46);
  } finally { env.close(); }
});

test('PNG encoding preserves resolution, white composition and text, then keeps its URL alive for the queued download', async () => {
  const env = setup();
  try {
    Object.defineProperty(env.window, 'devicePixelRatio', { value: 2 });
    const encodings: any[] = [], downloads: any[] = [], revoked: string[] = [], delayed: any[] = [];
    env.window.HTMLCanvasElement.prototype.toBlob = function (callback: any, mime: string) {
      encodings.push({ canvas: this, callback, mime, width: this.width, height: this.height });
    };
    env.window.URL.createObjectURL = (blob: any) => {
      assert.equal(blob.type, 'image/png'); assert.ok(blob.size > 0); return 'blob:http://localhost:8765/png-export';
    };
    env.window.URL.revokeObjectURL = (url: string) => revoked.push(url);
    env.window.HTMLAnchorElement.prototype.click = function () {
      downloads.push({ href: this.href, filename: this.download, connected: this.isConnected });
    };
    const originalTimeout = env.window.setTimeout.bind(env.window);
    env.window.setTimeout = (callback: () => void, delay: number) => {
      if (delay === 30000) { delayed.push({ callback, delay }); return 30000; }
      return originalTimeout(callback, delay);
    };
    const spec = plot(4, { node_count: 100, edge_count: 1000, sampled: true, selection: 'Degree ranking' });
    const chart = await env.api.mount(env.host, spec), layout = chart.exportLayout();
    const item = chart.toolbar.querySelector('[data-export="png"]');
    item.click(); item.click();
    assert.equal(encodings.length, 1, 'Repeated clicks cannot encode concurrent PNG bitmaps');
    assert.equal(downloads.length, 0, 'Download starts when asynchronous encoding returns the PNG blob');
    assert.equal(chart.status.textContent, 'Preparing PNG…');
    const encoding = encodings[0], context = env.contexts.get(encoding.canvas);
    assert.equal(encoding.mime, 'image/png');
    assert.equal(encoding.width, 1280); assert.equal(encoding.height, layout.height * 2);
    assert.ok(context.calls.some(call => call[0] === 'scale' && call[1] === 2 && call[2] === 2));
    assert.ok(context.calls.some(call => call[0] === 'fillRect' && call[3] === 640 && call[4] === layout.height));
    assert.ok(context.calls.some(call => call[0] === 'drawImage' && call[1] === chart.canvas && call[3] === layout.top && call[4] === 640));
    assert.ok(context.calls.some(call => call[0] === 'fillText' && String(call[1]).includes('Showing 4 of 100 nodes')));
    encoding.callback(new env.window.Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], { type: 'image/png' }));
    assert.deepEqual(downloads, [{ href: 'blob:http://localhost:8765/png-export', filename: 'network.png', connected: true }]);
    assert.equal(chart.pngExport, null); assert.equal(chart.status.textContent, 'Canvas · Ready');
    assert.equal(encoding.canvas.width, 0); assert.equal(encoding.canvas.height, 0);
    assert.equal(env.window.document.querySelectorAll('a[download]').length, 0);
    assert.equal(revoked.length, 0, 'A queued browser download must retain its blob URL');
    assert.equal(delayed.length, 1); assert.equal(delayed[0].delay, 30000);
    env.api.unmount(env.host);
    assert.equal(revoked.length, 0, 'Chart teardown must not invalidate an already initiated download');
    delayed[0].callback(); assert.deepEqual(revoked, ['blob:http://localhost:8765/png-export']);
  } finally { env.close(); }
});

test('asynchronous PNG failures are visible, retryable and release temporary bitmaps', async () => {
  const env = setup();
  try {
    const encodings: any[] = [];
    env.window.HTMLCanvasElement.prototype.toBlob = function (callback: any) { encodings.push({ canvas: this, callback }); };
    const chart = await env.api.mount(env.host, plot());
    chart.export('png'); encodings[0].callback(null);
    assert.equal(chart.status.textContent, 'Could not export the network.');
    assert.equal(chart.pngExport, null); assert.equal(encodings[0].canvas.width, 0);
    env.window.URL.createObjectURL = () => { throw new Error('Download initialization failed'); };
    chart.export('png');
    assert.doesNotThrow(() => encodings[1].callback(new env.window.Blob(['image'], { type: 'image/png' })));
    assert.equal(chart.status.textContent, 'Could not export the network.');
    assert.equal(chart.pngExport, null); assert.equal(encodings[1].canvas.width, 0);
    env.window.HTMLCanvasElement.prototype.toBlob = function () { throw new Error('Canvas encoding failed'); };
    chart.export('png');
    assert.equal(chart.status.textContent, 'Canvas encoding failed'); assert.equal(chart.pngExport, null);
  } finally { env.close(); }
});

test('a PNG callback after unmount cannot download or mutate the replacement chart', async () => {
  const env = setup();
  try {
    let callback: any, exportCanvas: any, downloads = 0;
    env.window.HTMLCanvasElement.prototype.toBlob = function (finish: any) { callback = finish; exportCanvas = this; };
    env.window.URL.createObjectURL = () => { downloads++; return 'blob:unexpected'; };
    const original = await env.api.mount(env.host, plot());
    original.export('png');
    const replacement = await env.api.mount(env.host, plot(2));
    assert.equal(original.destroyed, true); assert.equal(original.pngExport, null);
    assert.equal(exportCanvas.width, 0);
    callback(new env.window.Blob(['image'], { type: 'image/png' }));
    assert.equal(downloads, 0); assert.equal(env.api.get(env.host), replacement);
    assert.equal(replacement.status.textContent, 'Canvas · Ready');
  } finally { env.close(); }
});

function workerHarness() {
  const timers = new Set<any>(), intervals = new Set<any>(), messages: any[] = [], imports: string[] = [];
  let listener: ((message: any) => void) | null = null;
  const scope: any = {
    URL, Float32Array, Date, Math, performance,
    location: { href: 'http://localhost:8765/chart-assets/network-worker.js' },
    setTimeout: (callback: any, delay: number) => {
      const handle = setTimeout(() => { timers.delete(handle); callback(); }, delay); timers.add(handle); return handle;
    },
    clearTimeout: (handle: any) => { timers.delete(handle); clearTimeout(handle); },
    setInterval: (callback: any, delay: number) => { const handle = setInterval(callback, delay); intervals.add(handle); return handle; },
    clearInterval: (handle: any) => { intervals.delete(handle); clearInterval(handle); },
    postMessage: (message: any) => { messages.push(message); listener?.(message); },
  };
  scope.self = scope;
  const context = vm.createContext(scope);
  scope.importScripts = (url: string) => { imports.push(url); vm.runInContext(d3Code, context); };
  vm.runInContext(workerCode, context);
  const send = (message: any) => scope.onmessage({ data: message });
  const close = () => { send({ type: 'cancel', job: messages.at(-1)?.job }); timers.forEach(clearTimeout); intervals.forEach(clearInterval); };
  const request = (id: string, count = 32) => {
    const spec = plot(count).config.network;
    return { type: 'layout', job: id, nodes: spec.nodes.map(({ id, degree, group }) => ({ id, degree, group })),
      edges: spec.edges, seed: 987, point_size: 4 };
  };
  const run = (message: any) => new Promise<any>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Worker did not finish')), 5000);
    listener = result => {
      if (result.job === message.job && ['done', 'error'].includes(result.type)) {
        clearTimeout(timeout); listener = null; resolve(result);
      }
    };
    send(message);
  });
  const onMessage = (callback: (message: any) => void) => { listener = callback; };
  return { send, close, run, request, messages, imports, onMessage };
}

test('vendored D3 worker has deterministic bounded progressive layouts and imports only the local asset', async () => {
  const worker = workerHarness();
  try {
    const first = await worker.run(worker.request('first'));
    const second = await worker.run(worker.request('second'));
    assert.equal(first.type, 'done'); assert.equal(second.type, 'done');
    assert.deepEqual(Array.from(first.positions), Array.from(second.positions));
    assert.ok(Array.from(first.positions).every(Number.isFinite));
    assert.deepEqual(worker.imports, ['http://localhost:8765/chart-assets/d3.min.js']);
    const progress = worker.messages.filter(message => message.job === 'first');
    assert.ok(progress.length > 1 && progress.length <= 20);
    assert.equal(progress.at(-1).type, 'done');
    assert.ok(progress.every(message => message.positions.length === 64 && message.progress >= 0 && message.progress <= 1));
    assert.ok(progress.slice(1).every((message, index) => message.progress > progress[index].progress));
  } finally { worker.close(); }
});

test('worker cancellation stops after its current bounded batch and invalid graph requests fail clearly', async () => {
  const worker = workerHarness();
  try {
    const progress = new Promise<void>(resolve => worker.onMessage(message => {
      if (message.job === 'cancelled' && message.type === 'progress') {
        worker.send({ type: 'cancel', job: 'cancelled' }); resolve();
      }
    }));
    worker.send(worker.request('cancelled'));
    await progress;
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(worker.messages.filter(message => message.job === 'cancelled').length, 1);
    assert.equal(worker.messages.some(message => message.job === 'cancelled' && message.type === 'done'), false);
    const invalid = worker.request('invalid');
    invalid.edges[0].target = 999;
    assert.equal((await worker.run(invalid)).type, 'error');
    const nullNode = worker.request('null-node'); nullNode.nodes[0] = null;
    assert.equal((await worker.run(nullNode)).type, 'error');
    const excessive = worker.request('oversized', 100001);
    assert.equal((await worker.run(excessive)).type, 'error');
    assert.equal((await worker.run(worker.request('empty', 0))).positions.length, 0);
  } finally { worker.close(); }
});
