import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const asset = (name: string) => new URL('../../packages/openecon-charts/src/openecon_charts/assets/' + name, import.meta.url);
const workerCode = readFileSync(asset('network-worker.js'), 'utf8');
const d3Code = readFileSync(asset('d3.min.js'), 'utf8');
function harness(clock?: () => number) {
  const timers = new Set<any>(), intervals = new Set<any>(), messages: any[] = [], imports: string[] = [];
  let listener: ((message: any) => void) | null = null;
  const scope: any = {
    URL, Float32Array, Date: clock ? { now: clock } : Date, performance,
    location: { href: 'http://localhost:8765/chart-assets/network-worker.js' },
    setTimeout: (callback: any, delay: number) => {
      const handle = setTimeout(() => { timers.delete(handle); callback(); }, delay); timers.add(handle); return handle;
    },
    clearTimeout: (handle: any) => { timers.delete(handle); clearTimeout(handle); },
    setInterval: (callback: any, delay: number) => { const handle = setInterval(callback, delay); intervals.add(handle); return handle; },
    clearInterval: (handle: any) => { intervals.delete(handle); clearInterval(handle); },
    postMessage: (message: any, transferred: any[]) => { assert.equal(transferred?.[0], message.positions?.buffer); messages.push(message); listener?.(message); },
  };
  scope.self = scope;
  const context = vm.createContext(scope);
  scope.importScripts = (url: string) => { imports.push(url); vm.runInContext(d3Code, context); };
  vm.runInContext(workerCode, context);
  const send = (message: any) => scope.onmessage({ data: message });
  const run = (message: any, timeoutMs = 10000) => new Promise<any>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Layout did not finish: ' + message.job)), timeoutMs);
    listener = result => { if (result.job === message.job && ['done', 'error'].includes(result.type)) { clearTimeout(timeout); listener = null; resolve(result); } };
    send(message);
  });
  const close = () => { send({ type: 'cancel', job: messages.at(-1)?.job }); timers.forEach(clearTimeout); intervals.forEach(clearInterval); };
  return { run, send, messages, imports, listen: (callback: (message: any) => void) => { listener = callback; }, close };
}
function request(layout = 'circular', n = 12, extra: any = {}) {
  return { type: 'layout', job: 'job', layout, seed: 31, point_size: 4,
    nodes: Array.from({ length: n }, (_, id) => ({ id, degree: id === 0 ? 4 : 2, group: id % 3 })),
    edges: Array.from({ length: Math.max(0, n - 1) }, (_, source) => ({ source, target: source + 1, weight: 1 })), ...extra };
}
const xy = (result: any, index: number) => [result.positions[2 * index], result.positions[2 * index + 1]];
const near = (actual: number, expected: number, tolerance = 1e-4) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('every native layout is deterministic, finite, sparse, and respects pinned coordinates', async () => {
  const worker = harness();
  try {
    for (const layout of ['d3-force', 'forceatlas2', 'circular', 'grid', 'radial', 'hierarchical', 'community', 'geographic', 'fixed']) {
      const first = request(layout); first.nodes[0] = { ...first.nodes[0], x: 100, y: -200, pinned: true };
      if (layout === 'geographic') first.nodes.forEach((node: any, i: number) => { node.longitude = i * 5; node.latitude = i * 2; });
      if (layout === 'fixed') first.nodes.forEach((node: any, i: number) => { node.x ??= i * 10; node.y ??= -i * 4; });
      if (['d3-force', 'forceatlas2'].includes(layout)) first.layout_options = { iterations: 12 };
      const a = await worker.run(first), b = await worker.run({ ...first, job: 'replay' });
      assert.equal(a.type, 'done', `${layout}: ${a.message}`); assert.equal(b.type, 'done');
      assert.deepEqual([...a.positions], [...b.positions]); assert.ok([...a.positions].every(Number.isFinite));
      assert.deepEqual(xy(a, 0), [100, -200]); assert.equal(a.reason, 'complete');
      assert.ok(a.operations < 100000); assert.equal(new Set(Array.from({ length: 12 }, (_, i) => xy(a, i).join(','))).size, 12);
    }
    assert.deepEqual(worker.imports, ['http://localhost:8765/chart-assets/d3.min.js']);
  } finally { worker.close(); }
});

test('circular and grid geometry reflect code parameters and fixed axes', async () => {
  const worker = harness();
  try {
    const circle = await worker.run(request('circular', 4, { layout_options: { radius: 20, angle: 0 } }));
    for (let i = 0; i < 4; i++) near(Math.hypot(...xy(circle, i)), 20);
    near(circle.positions[0], 20); near(circle.positions[1], 0); near(circle.positions[2], 0); near(circle.positions[3], 20);
    const gridRequest = request('grid', 6, { layout_options: { columns: 3, spacing: 20 } }); gridRequest.nodes[0].fx = 99;
    const grid = await worker.run(gridRequest);
    assert.deepEqual(xy(grid, 0), [99, -10]); assert.deepEqual(xy(grid, 1), [0, -10]); assert.deepEqual(xy(grid, 5), [20, 10]);
  } finally { worker.close(); }
});

test('radial layout uses sparse hop distances and covers disconnected components', async () => {
  const worker = harness();
  try {
    const input = request('radial', 6, { edges: [{ source: 0, target: 1, weight: 1 }, { source: 0, target: 2, weight: 1 }, { source: 1, target: 3, weight: 1 }], layout_options: { root: 0, spacing: 10 } });
    const result = await worker.run(input);
    for (const [id, radius] of [[0, 0], [1, 10], [2, 10], [3, 20], [4, 30], [5, 40]]) near(Math.hypot(...xy(result, id)), radius);
    assert.ok(result.operations < 100);
  } finally { worker.close(); }
});

test('hierarchical ranks condense cycles and honor all four directions', async () => {
  const worker = harness();
  try {
    const edges = [{ source: 0, target: 1, weight: 1 }, { source: 1, target: 0, weight: 1 }, { source: 1, target: 2, weight: 1 }, { source: 2, target: 3, weight: 1 }, { source: 0, target: 3, weight: 1 }];
    for (const direction of ['TB', 'BT', 'LR', 'RL']) {
      const result = await worker.run(request('hierarchical', 5, { edges, layout_options: { direction, spacing: 20 } }));
      assert.equal(result.type, 'done'); const axis = ['LR', 'RL'].includes(direction) ? 0 : 1;
      const sign = ['BT', 'RL'].includes(direction) ? -1 : 1;
      near(xy(result, 0)[axis], 0); near(xy(result, 1)[axis], 0); near(xy(result, 2)[axis], sign * 20); near(xy(result, 3)[axis], sign * 40);
      near(xy(result, 4)[axis], 0); assert.notDeepEqual(xy(result, 0), xy(result, 1));
    }
  } finally { worker.close(); }
});

test('geographic coordinates use declared projections with finite Mercator poles', async () => {
  const worker = harness();
  try {
    const nodes = [{ id: 0, degree: 0, group: 0, longitude: 0, latitude: 0 }, { id: 1, degree: 0, group: 0, longitude: 30, latitude: 45 }, { id: 2, degree: 0, group: 0, longitude: -180, latitude: 90 }];
    const flat = await worker.run(request('geographic', 3, { nodes, edges: [], layout_options: { scale: 2 } }));
    assert.deepEqual(xy(flat, 1), [60, -90]); assert.deepEqual(xy(flat, 2), [-360, -180]);
    const mercator = await worker.run(request('geographic', 3, { nodes, edges: [], layout_options: { projection: 'mercator', scale: 1 } }));
    near(xy(mercator, 0)[1], 0); near(xy(mercator, 1)[1], -Math.log(Math.tan(3 * Math.PI / 8)) * 180 / Math.PI);
    near(xy(mercator, 2)[1], -180); assert.ok([...mercator.positions].every(Number.isFinite));
  } finally { worker.close(); }
});

test('ForceAtlas2 one-step fixture matches degree-weighted forces and adaptive-speed equations', async () => {
  const worker = harness();
  try {
    for (const edges of [[], [{ source: 0, target: 1, weight: 1 }]]) {
      const nodes = [{ id: 0, degree: 999, group: 0, x: -5, y: 0 }, { id: 1, degree: 999, group: 0, x: 5, y: 0 }];
      const mass = edges.length ? 2 : 1, force = -2 * mass * mass / 10 + (edges.length ? 10 : 0);
      const swing = 2 * mass * Math.abs(force), traction = swing / 2, estimate = 0.05 * Math.sqrt(2);
      const tolerance = Math.max(Math.sqrt(estimate), Math.min(10, estimate * traction / 4));
      const speed = tolerance * traction / swing;
      const factor = speed / (1 + Math.sqrt(speed * mass * Math.abs(force)));
      const result = await worker.run(request('forceatlas2', 2, { nodes, edges, layout_options: { iterations: 1, gravity: 0, scaling: 2 } }));
      assert.equal(result.type, 'done'); near(result.positions[0], -5 + force * factor); near(result.positions[2], 5 - force * factor);
      near(result.positions[1], 0); near(result.positions[3], 0);
    }
  } finally { worker.close(); }
});

test('ForceAtlas2 topology, LinLog, weight influence, gravity, and slowdown change positions', async () => {
  const worker = harness();
  try {
    const input = request('forceatlas2', 8, { layout_options: { iterations: 2 } });
    const base = await worker.run(input);
    for (const options of [{ linlog: true }, { gravity: 10, strong_gravity: true }, { slowdown: 10 }, { outbound_attraction_distribution: true }, { scaling: 10 }]) {
      const variant = await worker.run({ ...input, layout_options: { ...input.layout_options, ...options } });
      assert.equal(variant.type, 'done'); assert.notDeepEqual([...variant.positions], [...base.positions]);
    }
    const weighted = { ...input, edges: input.edges.map((edge: any) => ({ ...edge, weight: 3 })) };
    const one = await worker.run(weighted), ignored = await worker.run({ ...weighted, layout_options: { iterations: 2, edge_weight_influence: 0 } });
    assert.notDeepEqual([...one.positions], [...ignored.positions]);
    const coincident = request('forceatlas2', 12, { layout_options: { iterations: 5 } }); coincident.nodes.forEach((node: any) => { node.x = 0; node.y = 0; });
    const result = await worker.run(coincident); assert.equal(result.type, 'done'); assert.ok([...result.positions].every(Number.isFinite));
    assert.equal(new Set(Array.from({ length: 12 }, (_, i) => xy(result, i).join(','))).size, 12);
  } finally { worker.close(); }
});

test('strict layout boundaries reject invalid, unknown, or incompatible code configuration', async () => {
  const worker = harness();
  try {
    const invalid: any[] = [
      request('unknown'), request('circular', 100001), request('circular', 2, { edges: Array(1000001).fill({ source: 0, target: 1, weight: 1 }) }),
      request('grid', 2, { layout_options: { charge: -4 } }), request('circular', 2, { layout_options: null }), request('circular', 2, { layout_options: { radius: true } }),
      request('forceatlas2', 2, { layout_options: { theta: 0 } }), request('d3-force', 2, { layout_options: { collision: 1 } }),
      request('radial', 2, { layout_options: { root: 10 } }), request('hierarchical', 2, { layout_options: { direction: 'down' } }),
      request('geographic'), request('fixed'), request('circular', 2, { layout_options: { iterations: 1.5 } }), request('circular', 2, { layout_options: { work_limit: 999 } }),
      request('circular', 2, { layout_options: { time_limit_ms: 120001 } }), request('circular', 2, { positions: new Float32Array([0, 0]) }),
      request('circular', 2, { seed: -1 }), request('circular', 2, { point_size: 25 }), request('forceatlas2', 2, { edges: [{ source: 0, target: 1, weight: -1 }] }),
    ];
    const pinned = request(); pinned.nodes[0].pinned = true; invalid.push(pinned);
    const nan = request(); nan.nodes[0].x = NaN; invalid.push(nan);
    const partial = request(); partial.nodes[0].x = 10; invalid.push(partial);
    const disagreement = request(); disagreement.nodes[0] = { ...disagreement.nodes[0], x: 0, y: 0, fixed: true, pinned: false }; invalid.push(disagreement);
    for (const input of invalid) { const result = await worker.run(input); assert.equal(result.type, 'error', `${input.layout}: ${result.reason}`); }
    const boundary = await worker.run(request('grid', 2, { layout_options: { spacing: 10000, columns: 100000, iterations: 2000, work_limit: 1000000000, time_limit_ms: 120000 } }));
    assert.equal(boundary.type, 'done');
  } finally { worker.close(); }
});

test('100k-node / 1m-edge envelope executes sparse static layout and reports budget stops', async () => {
  const worker = harness();
  try {
    const input = request('grid', 100000); input.edges = Array.from({ length: 1000000 }, (_, i) => ({ source: i % 100000, target: (i + 1) % 100000, weight: 1 }));
    const result = await worker.run(input, 20000);
    assert.equal(result.type, 'done'); assert.equal(result.positions.length, 200000); assert.equal(result.operations, 100000);
    assert.deepEqual(xy(result, 0), [-6320, -6300]);
    const limited = await worker.run({ ...input, job: 'budget', layout_options: { work_limit: 1000 } }, 20000);
    assert.equal(limited.type, 'done'); assert.equal(limited.reason, 'work_limit'); assert.equal(limited.operations, 1000);
    const forceBudget = await worker.run(request('forceatlas2', 5000, { layout_options: { work_limit: 10000, iterations: 100 } }));
    assert.equal(forceBudget.reason, 'work_limit'); assert.ok(forceBudget.operations <= 10000); assert.ok([...forceBudget.positions].every(Number.isFinite));
  } finally { worker.close(); }
});

test('elapsed-time budgets stop distinctly rather than reporting convergence', async () => {
  let time = 0; const worker = harness(() => time += 10);
  try {
    const result = await worker.run(request('grid', 10000, { layout_options: { time_limit_ms: 100 } }));
    assert.equal(result.type, 'done'); assert.equal(result.reason, 'time_limit'); assert.ok(result.operations < 10000);
  } finally { worker.close(); }
});

test('cancelled or superseded layouts cannot publish later coordinates', async () => {
  const worker = harness();
  try {
    const seen = new Promise<void>(resolve => worker.listen(message => {
      if (message.job === 'cancel' && message.type === 'progress') { worker.send({ type: 'cancel', job: 'cancel' }); resolve(); }
    }));
    worker.send(request('forceatlas2', 200, { job: 'cancel', layout_options: { iterations: 200 } })); await seen;
    await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(worker.messages.filter(message => message.job === 'cancel').length, 1);
    worker.send(request('forceatlas2', 200, { job: 'superseded', layout_options: { iterations: 2000 } }));
    const replacement = await worker.run(request('circular', 4, { job: 'replacement' }));
    assert.equal(replacement.type, 'done'); assert.equal(worker.messages.some(message => message.job === 'superseded'), false);
    const initial = new Float32Array([10, 11, 12, 13]);
    const fixed = await worker.run(request('fixed', 2, { positions: initial })); assert.deepEqual([...fixed.positions], [...initial]);
    const rendererShape = request('grid', 2); rendererShape.nodes = rendererShape.nodes.map((node: any) => ({ ...node, x: undefined, y: undefined, fx: undefined, fy: undefined, fixed: false, longitude: undefined, latitude: undefined }));
    assert.equal((await worker.run(rendererShape)).type, 'done');
  } finally { worker.close(); }
});

test('large sparse hierarchical and Barnes-Hut force jobs avoid quadratic work', async () => {
  const worker = harness();
  try {
    const chain = await worker.run(request('hierarchical', 20000, { layout_options: { spacing: 1 } }), 20000);
    assert.equal(chain.type, 'done'); near(chain.positions[39999], 19999); assert.ok(chain.operations < 400000);
    const force = await worker.run(request('forceatlas2', 4000, { layout_options: { iterations: 1, work_limit: 2000000 } }), 20000);
    assert.equal(force.type, 'done'); assert.equal(force.reason, 'complete'); assert.ok(force.operations < 800000, `${force.operations} is not sparse BH work`);
  } finally { worker.close(); }
});

test('D3 refuses pathological fixed coincidences or collision density before unbounded kernels', async () => {
  const worker = harness();
  try {
    const coincident = request('d3-force', 600, { layout_options: { iterations: 1, collision: false } });
    coincident.nodes.forEach((node: any) => { node.x = 0; node.y = 0; });
    const rejected = await worker.run(coincident); assert.equal(rejected.type, 'error'); assert.match(rejected.message, /coincident-node budget/);
    const clustered = request('d3-force', 600, { layout_options: { iterations: 1 } });
    clustered.nodes.forEach((node: any, i: number) => { node.x = i / 1000; node.y = 0; });
    const dense = await worker.run(clustered); assert.equal(dense.type, 'error'); assert.match(dense.message, /collision-density budget/);
    coincident.nodes.forEach((node: any) => { node.fixed = true; });
    const fixed = await worker.run(coincident); assert.equal(fixed.type, 'done'); assert.ok([...fixed.positions].every(value => value === 0));
    const relaxed = await worker.run({ ...clustered, layout_options: { iterations: 1, collision: false } }); assert.equal(relaxed.type, 'done');
  } finally { worker.close(); }
});

test('many communities pack deterministically without coincident discs or overflowing legal spacing', async () => {
  const worker = harness();
  try {
    const input = request('community', 2000, { layout_options: { spacing: 10000 } });
    input.nodes.forEach((node: any, i: number) => { node.group = i < 1000 ? 0 : i; });
    const result = await worker.run(input); assert.equal(result.type, 'done');
    assert.ok([...result.positions].every(value => Number.isFinite(value) && Math.abs(value) <= 1e9));
    assert.equal(new Set(Array.from({ length: 2000 }, (_, i) => xy(result, i).join(','))).size, 2000);
    assert.ok(result.operations <= 5000);
  } finally { worker.close(); }
});
