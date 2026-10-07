// Playwright CLI run-code expression. Open a generated network HTML first.
// Synthetic fixed coordinates isolate mount/upload and pan/zoom frame cost.
// This is not a force-layout convergence or all-topology performance claim.
async (page) => {
  return await page.evaluate(async () => {
    const host = document.querySelector('main');
    if (!window.OpenEconNetworkCharts?.get(host)) throw new Error('Open a portable network chart first.');
    const start = performance.now(), count = 100000, edgeCount = 1000000;
    const nodes = Array.from({ length: count }, (_, id) => ({ id, label: String(id), degree: 20, group: id % 4,
      identity: { type: 'integer', value: String(id) }, x: id % 1000, y: Math.floor(id / 1000) }));
    const edges = Array.from({ length: edgeCount }, (_, i) => ({ source: i % count, target: (i % count + 1 + Math.floor(i / count)) % count, weight: 1 }));
    const built = performance.now();
    const chart = await OpenEconNetworkCharts.mount(host, { kind: 'network', title: '100k nodes / 1m edges — synthetic display',
      x_label: '', y_label: '', data: [], sample_n: count, total_n: count, dropped_n: 0,
      config: { network: { nodes, edges, node_count: count, edge_count: edgeCount, shown_node_count: count,
        shown_edge_count: edgeCount, sampled: false, selection: 'All nodes and edges', directed: false },
        options: { layout: 'fixed', labels: false, point_size: 1, line_width: 0.5, opacity: 0.3, height: 680 } } });
    const mounted = performance.now();
    await new Promise(resolve => requestAnimationFrame(resolve));
    const frames = []; let last = performance.now();
    for (let i = 0; i < 40; i++) {
      chart.transform.x += i % 2 ? -2 : 2;
      chart.schedule();
      await new Promise(resolve => requestAnimationFrame(resolve));
      const now = performance.now(); if (i >= 5) frames.push(now - last); last = now;
    }
    const gl = chart.gpu?.gl, debug = gl?.getExtension('WEBGL_debug_renderer_info');
    const sorted = [...frames].sort((a, b) => a - b), mean = frames.reduce((a, b) => a + b, 0) / frames.length;
    const result = { schema: 1, user_agent: navigator.userAgent, status: chart.status.textContent,
      shown_nodes: chart.drawNodeIndices.length, shown_edges: chart.drawEdgeIndices.length, sampled: chart.config.sampled,
      renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
      synthetic_construction_ms: built - start, mount_validation_and_upload_ms: mounted - built,
      gpu_buffer_storage_bytes: chart.gpu?.buffers.reduce((total, buffer) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer); return total + gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
      }, 0) ?? null,
      gpu_position_texture_bytes: chart.gpu ? chart.gpu.textureWidth * Math.ceil(count / chart.gpu.textureWidth) * 16 : null,
      measured_pan_frames: frames.length, mean_frame_ms: mean, median_frame_ms: sorted[Math.floor(sorted.length / 2)],
      p95_frame_ms: sorted[Math.floor(sorted.length * 0.95)], measured_fps: 1000 / mean,
      observed_js_heap_bytes: performance.memory?.usedJSHeapSize ?? null,
      scope: 'Actual headed Chrome GPU drawing on this host; fixed synthetic positions, complete graph, warm chart assets. Mount timing includes validation/upload, frame timing uses requestAnimationFrame during pan. No force convergence, cold app boot, other GPUs or total process memory claim.' };
    if (!gl || result.shown_nodes !== count || result.shown_edges !== edgeCount || result.sampled) throw new Error('Full WebGL graph was not rendered: ' + JSON.stringify(result));
    return result;
  });
}
