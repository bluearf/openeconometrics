/* Reproducible worker-thread layout benchmark; does not claim browser/WebGL performance.
 * Run: node benchmarks/network_layouts_scale.mjs [output-json]
 * Includes graph allocation, clone dispatch, real transferable Float32 responses,
 * fixed work/iteration budgets and sampled whole-process RSS (not a peak guarantee). */
import { Worker } from 'node:worker_threads';
import { writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';

const workerPath = fileURLToPath(new URL('../packages/openecon-charts/src/openecon_charts/assets/network-worker.js', import.meta.url));
const d3Path = fileURLToPath(new URL('../packages/openecon-charts/src/openecon_charts/assets/d3.min.js', import.meta.url));
const boot = `
const { parentPort, workerData } = require('node:worker_threads');
const vm = require('node:vm');
const fs = require('node:fs');
const scope = { URL, Float32Array, Date, Math, performance, setTimeout, clearTimeout, setInterval, clearInterval,
  location: { href: 'http://localhost/chart-assets/network-worker.js' },
  postMessage: (value, transfers) => parentPort.postMessage(value, transfers) };
scope.self = scope;
const context = vm.createContext(scope);
scope.importScripts = url => { if (url !== 'http://localhost/chart-assets/d3.min.js') throw new Error('Unexpected import'); vm.runInContext(fs.readFileSync(workerData.d3Path, 'utf8'), context); };
vm.runInContext(fs.readFileSync(workerData.workerPath, 'utf8'), context);
parentPort.on('message', data => scope.onmessage({ data }));
parentPort.postMessage({ type: 'ready' });
`;
async function measure(layout, nodeCount, edgeCount, layoutOptions) {
  const beforeGraph = performance.now();
  const nodes = Array.from({ length: nodeCount }, (_, id) => ({ id, degree: 20, group: id % 20 }));
  const edges = Array.from({ length: edgeCount }, (_, i) => {
    const source = i % nodeCount, step = Math.floor(i / nodeCount) + 1;
    return { source, target: (source + step * 7919) % nodeCount, weight: 1 + i % 3 };
  });
  const allocationMs = performance.now() - beforeGraph;
  const worker = new Worker(boot, { eval: true, workerData: { workerPath, d3Path } });
  await new Promise((resolve, reject) => { worker.once('error', reject); worker.once('message', message => message.type === 'ready' ? resolve() : reject(new Error('Worker bootstrap failed'))); });
  let sampledRssBytes = process.memoryUsage().rss, progressMessages = 0, firstProgressMs = null, dispatchMs;
  const started = performance.now(), monitor = setInterval(() => { sampledRssBytes = Math.max(sampledRssBytes, process.memoryUsage().rss); }, 10);
  try {
    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Benchmark request timeout')), 125000);
      worker.once('error', error => { clearTimeout(timeout); reject(error); });
      worker.on('message', message => {
        if (message.type === 'progress') { progressMessages++; firstProgressMs ??= performance.now() - started; }
        if (message.type === 'error') { clearTimeout(timeout); reject(new Error(message.message)); }
        if (message.type === 'done') { clearTimeout(timeout); resolve(message); }
      });
      const dispatchStarted = performance.now();
      worker.postMessage({ type: 'layout', job: 'benchmark', layout, layout_options: layoutOptions, nodes, edges, seed: 42, point_size: 4 });
      dispatchMs = performance.now() - dispatchStarted;
    });
    if (!(result.positions instanceof Float32Array) || result.positions.length !== nodeCount * 2 || !result.positions.every(Number.isFinite)) throw new Error('Invalid transferred positions');
    const elapsedMs = performance.now() - started;
    sampledRssBytes = Math.max(sampledRssBytes, process.memoryUsage().rss);
    return { layout, nodes: nodeCount, edges: edgeCount, layout_options: layoutOptions, reason: result.reason,
      iterations: result.iterations, work_units: result.operations, graph_allocation_ms: allocationMs,
      synchronous_dispatch_clone_ms: dispatchMs, request_round_trip_ms: elapsedMs, first_progress_ms: firstProgressMs,
      progress_messages: progressMessages, returned_position_bytes: result.positions.byteLength,
      sampled_process_rss_mib: sampledRssBytes / 1048576,
      first_positions: Array.from(result.positions.slice(0, 10)) };
  } finally { clearInterval(monitor); await worker.terminate(); }
}
const outputPath = process.argv[2] || fileURLToPath(new URL('../docs/evidence/network-layouts-2026-10-06.json', import.meta.url));
const evidence = {
  measured_at_utc: new Date().toISOString(), platform: process.platform, arch: process.arch,
  os: os.release(), node: process.version, cpu: os.cpus()[0]?.model, logical_cpus: os.cpus().length,
  memory_scope: '10ms sampled whole Node process RSS includes main graph objects, worker clone and libraries; not isolated layout memory or a peak guarantee',
  timing_scope: 'Worker request round trip includes structured clone, graph validation, initialization, native layout and transferred result; excludes graph allocation and worker bootstrap reported separately',
  solver: 'Native ForceAtlas2 equations with vendored D3 quadtree; no external graph-layout solver',
  benchmark_source: 'benchmarks/network_layouts_scale.mjs',
  worker_source_bytes: readFileSync(workerPath).byteLength,
  worker_source_sha256: createHash('sha256').update(readFileSync(workerPath)).digest('hex'),
  d3_source_sha256: createHash('sha256').update(readFileSync(d3Path)).digest('hex'),
  benchmark_source_sha256: createHash('sha256').update(readFileSync(fileURLToPath(import.meta.url))).digest('hex'),
  runs: [],
};
for (const [layout, nodes, edges, options] of [
  ['grid', 100000, 1000000, { work_limit: 200000000, time_limit_ms: 120000 }],
  ['forceatlas2', 10000, 100000, { iterations: 10, work_limit: 200000000, time_limit_ms: 120000 }],
  ['forceatlas2', 100000, 1000000, { iterations: 2, work_limit: 200000000, time_limit_ms: 120000 }],
]) {
  const measured = await measure(layout, nodes, edges, options); evidence.runs.push(measured);
  process.stdout.write(`${layout} ${nodes}/${edges}: ${measured.request_round_trip_ms.toFixed(1)}ms, ${measured.reason}, ${measured.work_units} work units\n`);
}
writeFileSync(outputPath, JSON.stringify(evidence, null, 2) + '\n');
process.stdout.write(`Evidence: ${outputPath}\n`);
