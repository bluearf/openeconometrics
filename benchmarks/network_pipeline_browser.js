/* Actual production renderer, physical Python artifacts, headed-browser measurements. */
(async () => {
  "use strict";
  const state = document.querySelector("#state"), host = document.querySelector("#chart");
  const reports = [], frame = () => new Promise(resolve => requestAnimationFrame(resolve));
  const timed = async function (phases, name, operation) {
    const start = performance.now(), value = await operation();
    phases[name] = performance.now() - start;
    return value;
  };
  const metadata = () => ({user_agent: navigator.userAgent, device_pixel_ratio: devicePixelRatio,
    document_visibility: document.visibilityState,
    viewport: [innerWidth, innerHeight], hardware_concurrency: navigator.hardwareConcurrency,
    js_heap_used_bytes: performance.memory?.usedJSHeapSize ?? null,
    js_heap_scope: "Chrome performance.memory estimate, shared/approximate; not process RSS or a lifetime peak."});
  try {
    const manifest = await (await fetch("/manifest.json", {cache: "no-store"})).json();
    const repeats = Number(new URL(location.href).searchParams.get("repeats") ?? 3);
    if (!Number.isInteger(repeats) || repeats < 1 || repeats > 20) throw Error("Invalid repeats");
    for (const item of manifest.browser_cases) {
      for (let repetition = 1; repetition <= repeats; repetition++) {
        state.textContent = `${item.id}: ${repetition}/${repeats}`;
        const phases = {}, sampledMemory = await (await fetch("/memory/start", {method: "POST"})).json();
        let chart;
        try {
          const bytes = await timed(phases, "http_fetch_body_ms", async () => {
            const response = await fetch(`/data/${item.path}`, {cache: "no-store"});
            if (!response.ok) throw Error(`HTTP ${response.status}`);
            return response.arrayBuffer();
          });
          const digest = await timed(phases, "body_integrity_sha256_ms", async () =>
            [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
              .map(value => value.toString(16).padStart(2, "0")).join(""));
          if (bytes.byteLength !== item.bytes || digest !== item.sha256) throw Error("Artifact integrity mismatch");
          const text = await timed(phases, "utf8_decode_ms", () => new TextDecoder("utf-8", {fatal: true}).decode(bytes));
          const spec = await timed(phases, "json_parse_ms", () => JSON.parse(text));
          const mountStart = performance.now();
          chart = await timed(phases, "mount_validation_upload_ms", () => OpenEconNetworkCharts.mount(host, spec));
          const originalDraw = chart.draw.bind(chart);
          let resolveDraw, drawStarted, drawFinished;
          const firstDraw = new Promise(resolve => { resolveDraw = resolve; });
          chart.draw = function () {
            if (drawStarted !== undefined) return originalDraw();
            drawStarted = performance.now();
            originalDraw();
            // Explicit synchronization measures submitted GPU work, not compositor pixels.
            chart.gpu?.gl.finish();
            drawFinished = performance.now();
            resolveDraw();
          };
          chart.schedule();
          let watchdog;
          try {
            await Promise.race([firstDraw, new Promise((_, reject) => {
              watchdog = setTimeout(() => reject(Error("First drawing exceeded 30s")), 30000);
            })]);
          } finally { clearTimeout(watchdog); }
          phases.first_draw_submit_and_gpu_finish_ms = drawFinished - drawStarted;
          phases.mount_to_first_draw_finished_ms = drawFinished - mountStart;
          await frame(); await frame();
          phases.mount_to_two_raf_paint_opportunity_ms = performance.now() - mountStart;
          const gl = chart.gpu?.gl, debug = gl?.getExtension("WEBGL_debug_renderer_info");
          if (!gl || chart.gpuLost || chart.gpu.lost || gl.isContextLost() ||
              gl.getError() !== gl.NO_ERROR || gl.drawingBufferWidth < 2 || gl.drawingBufferHeight < 2) {
            throw Error("WebGL context/drawing buffer did not complete without errors");
          }
          const frames = [];
          for (let index = 0; index < 20; index++) {
            const start = performance.now();
            chart.transform.x += index % 2 ? -2 : 2;
            chart.schedule(); await frame();
            if (index >= 5) frames.push(performance.now() - start);
          }
          const gpuBytes = chart.gpu?.buffers.reduce((total, buffer) => {
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            return total + gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
          }, 0) ?? null;
          const report = {schema: 1, case: item.id, repetition, metadata: metadata(),
            analytical_metadata: item.metadata, graphs: item.graphs, payload_counts: item.display,
            body_bytes: bytes.byteLength, body_sha256: digest, phases_ms: phases,
            actual_draw_nodes: chart.drawNodeIndices.length, actual_draw_edges: chart.drawEdgeIndices.length,
            renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
            backend: gl ? "WebGL2" : "Canvas", status: chart.status.textContent,
            gpu_uploaded_nodes: chart.gpu.nodeCount,
            gpu_uploaded_edges: chart.gpu.edgeCount + chart.gpu.loopCount,
            drawing_buffer_pixels: [gl.drawingBufferWidth, gl.drawingBufferHeight],
            webgl_context_alive_and_error_free: true,
            rendering_dtype: gl ? "float32 GPU coordinates" : "JavaScript numbers",
            gpu_buffer_storage_bytes: gpuBytes,
            gpu_position_texture_bytes: chart.gpu ? chart.gpu.textureWidth * Math.max(1,
              Math.ceil(chart.drawNodeIndices.length / chart.gpu.textureWidth)) * 16 : null,
            gpu_memory_scope: "Allocated buffer/position texture storage; excludes driver, framebuffer, program and total GPU memory.",
            pan_raf_frames_ms: frames,
            timing_scope: "Warm assets and OS cache; real loopback HTTP, physical artifact, fixed coordinates. draw includes gl.finish synchronization. Two RAFs indicate a paint opportunity, not compositor pixel confirmation or cold app boot. No force-layout convergence measurement."};
          if (report.actual_draw_nodes !== item.display[0].shown_node_count ||
              report.actual_draw_edges !== item.display[0].shown_edge_count ||
              report.gpu_uploaded_nodes !== item.display[0].shown_node_count ||
              report.gpu_uploaded_edges !== item.display[0].shown_edge_count) {
            throw Error("Requested WebGL display coverage was not reached");
          }
          if (spec.config.network.frames?.length) {
            const start = performance.now();
            chart.setFrame(spec.config.network.frames.length - 1);
            await frame(); chart.gpu?.gl.finish(); await frame();
            report.phases_ms.temporal_last_frame_to_two_raf_ms = performance.now() - start;
            report.final_frame_index = chart.frameIndex;
          }
          report.browser_process_rss = await (await fetch("/memory/stop", {method: "POST"})).json();
          report.browser_process_rss.baseline = sampledMemory;
          reports.push(report);
        } finally {
          if (chart && !(item === manifest.browser_cases.at(-1) && repetition === repeats)) {
            OpenEconNetworkCharts.unmount(host);
          }
        }
      }
    }
    const report = {schema: 1, source_sha256: manifest.source_sha256,
      browser_source_sha256: manifest.browser_source_sha256, reports,
      scope: "Actual headed Chrome on the recorded host. Every full/selected count and byte hash is checked; repeated browser cases use fresh mounts in the same tab, without a forced GC. Python repetitions use fresh processes."};
    const response = await fetch("/results", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(report)});
    if (!response.ok) throw Error(`Result persistence failed: ${response.status}`);
    const receipt = await response.json();
    document.querySelector("#result").value = JSON.stringify({receipt, reports: reports.length});
    state.textContent = `Complete: ${reports.length} measurements saved (${receipt.file})`;
  } catch (error) {
    state.textContent = `Failed: ${error.message}`;
    document.querySelector("#result").value = JSON.stringify({error: String(error), completed: reports.length});
  }
})();
