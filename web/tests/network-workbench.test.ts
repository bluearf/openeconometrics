import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const asset = (name: string) =>
  readFileSync(
    fileURLToPath(
      new URL(
        "../../packages/openecon-charts/src/openecon_charts/assets/" + name,
        import.meta.url,
      ),
    ),
    "utf8",
  );
function plot(count = 4, options: any = {}) {
  const nodes = Array.from({ length: count }, (_, id) => ({
    id,
    label: "Node " + id,
    degree: id + 1,
    group: id % 2,
    identity: {
      type: id % 2 ? "string" : "integer",
      value: id % 2 ? "n" + id : String(id),
    },
    attrs: { score: id, sector: id % 2 ? "B" : "A" },
    x: id * 40,
    y: 0,
  }));
  const edges = nodes.slice(1).map((n) => ({
    source: n.id - 1,
    target: n.id,
    weight: n.id,
    attrs: { year: 2020 + n.id },
  }));
  const network = {
    nodes,
    edges,
    directed: true,
    node_count: count,
    edge_count: edges.length,
    shown_node_count: count,
    shown_edge_count: edges.length,
    sampled: false,
    selection: "All nodes and edges",
  };
  return {
    kind: "network",
    title: "Network",
    x_label: "",
    y_label: "",
    data: [],
    sample_n: count,
    total_n: count,
    dropped_n: 0,
    config: { network, options: { layout: "fixed", ...options } },
  };
}
function setup(gpu = false) {
  const dom = new JSDOM('<!doctype html><body><div id="chart"></div></body>', {
      runScripts: "outside-only",
      pretendToBeVisual: true,
      url: "http://localhost:8765/",
    }),
    w = dom.window as any;
  const frames = new Map<number, () => void>();
  let frameID = 0;
  w.requestAnimationFrame = (fn: () => void) => {
    frames.set(++frameID, fn);
    return frameID;
  };
  w.cancelAnimationFrame = (id: number) => frames.delete(id);
  const gl: any = {
    calls: [],
    buffers: [],
    MAX_TEXTURE_SIZE: 1,
    ARRAY_BUFFER: 2,
    STATIC_DRAW: 3,
    TEXTURE_2D: 4,
    RG32F: 5,
    RG: 6,
    FLOAT: 7,
    TRIANGLES: 8,
    POINTS: 9,
    COMPILE_STATUS: 10,
    LINK_STATUS: 11,
    COLOR_BUFFER_BIT: 12,
  };
  for (const name of [
    "shaderSource",
    "compileShader",
    "deleteShader",
    "attachShader",
    "linkProgram",
    "deleteProgram",
    "enable",
    "blendFunc",
    "clearColor",
    "deleteBuffer",
    "bindBuffer",
    "bindTexture",
    "texParameteri",
    "texImage2D",
    "texSubImage2D",
    "deleteTexture",
    "enableVertexAttribArray",
    "vertexAttribPointer",
    "vertexAttribDivisor",
    "viewport",
    "clear",
    "useProgram",
    "activeTexture",
    "uniform1i",
    "uniform1f",
    "uniform2f",
    "uniform3f",
    "drawArraysInstanced",
    "drawArrays",
  ])
    gl[name] = (...args: any[]) => gl.calls.push([name, ...args]);
  gl.createShader = () => ({});
  gl.createProgram = () => ({});
  gl.createTexture = () => ({});
  gl.createBuffer = () => ({});
  gl.getShaderParameter = gl.getProgramParameter = () => true;
  gl.getAttribLocation = (_: any, name: string) =>
    ["a_endpoints", "a_size", "a_color"].indexOf(name);
  gl.getUniformLocation = (_: any, name: string) => name;
  gl.getParameter = () => 4096;
  gl.bufferData = (_: any, data: any) => {
    assert.ok(ArrayBuffer.isView(data));
    gl.buffers.push(data.byteLength);
  };
  const contexts = new Map<any, any>();
  w.HTMLCanvasElement.prototype.getContext = function (type: string) {
    if (type === "webgl2") return gpu ? gl : null;
    if (type !== "2d") return null;
    if (!contexts.has(this)) {
      const context: any = {
        calls: [],
        measureText: (s: string) => ({ width: s.length * 7 }),
      };
      for (const name of [
        "setTransform",
        "clearRect",
        "fillRect",
        "save",
        "restore",
        "translate",
        "scale",
        "beginPath",
        "moveTo",
        "lineTo",
        "arc",
        "closePath",
        "stroke",
        "fill",
        "strokeText",
        "fillText",
        "drawImage",
      ])
        context[name] = (...args: any[]) => context.calls.push([name, ...args]);
      contexts.set(this, context);
    }
    return contexts.get(this);
  };
  w.HTMLCanvasElement.prototype.toBlob = function (fn: any) {
    fn(new w.Blob(["png"]));
  };
  w.Worker = undefined;
  w.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  w.eval(asset("network-webgl.js"));
  w.eval(asset("network-renderer.js"));
  const host = w.document.getElementById("chart");
  Object.defineProperty(host, "clientWidth", {
    value: 640,
    configurable: true,
  });
  return {
    w,
    dom,
    gl,
    frames,
    host,
    api: w.OpenEconNetworkCharts,
    flush: () => {
      const pending = [...frames.values()];
      frames.clear();
      pending.forEach((fn) => fn());
    },
    close: () => {
      w.OpenEconNetworkCharts.unmount(host);
      w.close();
    },
  };
}

test("initial WebGL presentation retries redraw settled frames and then stop", async () => {
  const env = setup(true);
  try {
    const chart = await env.api.mount(env.host, plot()),
      uploads = chart.gpu.uploadCount,
      buffers = env.gl.buffers.length,
      draws = () =>
        env.gl.calls.filter((call: any[]) => call[0] === "clear").length;
    env.flush();
    assert.equal(draws(), 1);
    env.flush();
    assert.equal(
      draws(),
      2,
      "A settled frame redraws even with unchanged host dimensions",
    );
    env.flush();
    assert.equal(draws(), 3);
    assert.equal(
      env.frames.size,
      0,
      "Two presentation retries must not become a render loop",
    );
    env.flush();
    assert.equal(draws(), 3);
    assert.equal(chart.gpu.uploadCount, uploads);
    assert.equal(env.gl.buffers.length, buffers);
  } finally {
    env.close();
  }
});

test("post-mount presentation corrects the initial fallback width without reloading geometry", async () => {
  const env = setup(true);
  try {
    Object.defineProperty(env.host, "clientWidth", {
      value: 0,
      configurable: true,
    });
    const chart = await env.api.mount(env.host, plot()),
      initialCamera = { ...chart.transform },
      uploads = chart.gpu.uploadCount;
    assert.equal(chart.width, 700);
    env.flush();
    Object.defineProperty(env.host, "clientWidth", {
      value: 450,
      configurable: true,
    });
    env.flush();
    env.flush();
    assert.equal(chart.width, 450);
    assert.equal(chart.gpuCanvas.width, 450);
    assert.notDeepEqual({ ...chart.transform }, initialCamera);
    for (let i = 0; i < chart.config.nodes.length; i++) {
      const x = chart.positions[i * 2] * chart.transform.k + chart.transform.x;
      assert.ok(
        x > 0 && x < 450,
        "Untouched graph is fitted to the settled panel",
      );
    }
    assert.equal(chart.gpu.uploadCount, uploads);
    assert.equal(env.frames.size, 0);
  } finally {
    env.close();
  }
});

test("presentation retries preserve saved views and cameras changed before first paint", async () => {
  for (const saved of [false, true]) {
    const env = setup(true);
    try {
      const spec = plot();
      if (saved)
        spec.config.options.view = {
          version: 1,
          positions: spec.config.network.nodes.map((n: any) => ({
            id: n.id,
            identity: n.identity,
            x: n.x,
            y: n.y,
            pinned: false,
          })),
          transform: { k: 1.7, x: 190, y: 210 },
        };
      const chart = await env.api.mount(env.host, spec);
      if (!saved) chart.zoom(1.6, 310, 230);
      const camera = { ...chart.transform },
        uploads = chart.gpu.uploadCount;
      Object.defineProperty(env.host, "clientWidth", {
        value: 420,
        configurable: true,
      });
      for (let frame = 0; frame < 4; frame++) env.flush();
      assert.equal(chart.width, 420);
      assert.deepEqual({ ...chart.transform }, camera);
      assert.equal(chart.gpu.uploadCount, uploads);
      assert.equal(env.frames.size, 0);
    } finally {
      env.close();
    }
  }
});

test("visible, focus and pageshow resume drawing while preserving camera and layout", async () => {
  const env = setup(true);
  try {
    const chart = await env.api.mount(env.host, plot());
    for (let frame = 0; frame < 4; frame++) env.flush();
    chart.transform = { k: 1.8, x: 130, y: -50 };
    // Even an untouched camera is retained when restoring presentation.
    chart.touched = false;
    const camera = { ...chart.transform },
      uploads = chart.gpu.uploadCount,
      buffers = env.gl.buffers.length,
      positions = Array.from(chart.positions);
    let layouts = 0;
    chart.startLayout = () => {
      layouts++;
    };
    const transitions = [
      [env.w, "focus"],
      [env.w, "pageshow"],
      [env.w.document, "visibilitychange"],
    ];
    for (const [target, type] of transitions) {
      const before = env.gl.calls.filter(
        (call: any[]) => call[0] === "clear",
      ).length;
      target.dispatchEvent(new env.w.Event(type));
      for (let frame = 0; frame < 4; frame++) env.flush();
      const after = env.gl.calls.filter(
        (call: any[]) => call[0] === "clear",
      ).length;
      assert.ok(after > before, type + " must redraw unchanged dimensions");
      assert.deepEqual({ ...chart.transform }, camera);
      assert.equal(env.frames.size, 0);
    }
    Object.defineProperty(env.w.document, "visibilityState", {
      value: "hidden",
      configurable: true,
    });
    env.w.document.dispatchEvent(new env.w.Event("visibilitychange"));
    env.w.dispatchEvent(new env.w.Event("focus"));
    assert.equal(
      env.frames.size,
      0,
      "Hidden pages do not queue presentation retries",
    );
    Object.defineProperty(env.w.document, "visibilityState", {
      value: "visible",
      configurable: true,
    });
    Object.defineProperty(env.host, "clientWidth", {
      value: 400,
      configurable: true,
    });
    Object.defineProperty(env.w, "devicePixelRatio", {
      value: 2,
      configurable: true,
    });
    env.w.document.dispatchEvent(new env.w.Event("visibilitychange"));
    for (let frame = 0; frame < 4; frame++) env.flush();
    assert.equal(chart.width, 400);
    assert.equal(chart.gpuCanvas.width, 800);
    assert.deepEqual({ ...chart.transform }, camera);
    assert.deepEqual(Array.from(chart.positions), positions);
    assert.equal(layouts, 0);
    assert.equal(chart.gpu.uploadCount, uploads);
    assert.equal(env.gl.buffers.length, buffers);
    assert.equal(env.frames.size, 0);
  } finally {
    env.close();
  }
});

test("teardown cancels presentation retries and removes visibility listeners", async () => {
  const env = setup(true);
  try {
    const chart = await env.api.mount(env.host, plot());
    env.flush();
    assert.ok(env.frames.size > 0);
    env.api.unmount(env.host);
    assert.equal(chart.presentationFrame, null);
    assert.equal(chart.frame, null);
    assert.equal(env.frames.size, 0);
    env.w.dispatchEvent(new env.w.Event("focus"));
    env.w.dispatchEvent(new env.w.Event("pageshow"));
    env.w.document.dispatchEvent(new env.w.Event("visibilitychange"));
    env.flush();
    assert.equal(env.frames.size, 0);
    assert.equal(
      env.gl.calls.filter((call: any[]) => call[0] === "clear").length,
      1,
    );
  } finally {
    env.close();
  }
});

test("code mappings, induced filters, labels and annotations use finite scalar attributes", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(
      env.host,
      plot(4, {
        node_size: { field: "score", range: [2, 8] },
        node_color: {
          field: "sector",
          scale: "categorical",
          range: ["#fff", "#163d68"],
        },
        node_label: "sector",
        edge_width: { field: "weight", range: [1, 3] },
        edge_color: "#123456",
        legend: true,
        annotations: [{ text: "<b>literal</b>", node_id: 3 }],
        filters: [{ scope: "nodes", field: "score", op: "gte", value: 2 }],
      }),
    );
    env.flush();
    assert.equal(chart.filteredNodes, 2);
    assert.equal(chart.filteredEdges, 1);
    assert.equal(chart.radius(3), 8);
    assert.equal(chart.nodeColor(3), "#163d68");
    assert.equal(chart.nodeLabels[3], "B");
    assert.equal(env.host.querySelector("b"), null);
    assert.equal(chart.snapshot().querySelectorAll("circle").length, 2);
    assert.ok(chart.snapshot().textContent.includes("<b>literal</b>"));
    assert.ok(chart.counts.textContent.includes("Filtered: 2 nodes / 1 edges"));
    chart.setFilters([]);
    assert.equal(chart.filteredNodes, 4);
    assert.equal(chart.filteredEdges, 3);
  } finally {
    env.close();
  }
});

test("saved views round trip camera, positions, pins, selection and declarative filters", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot());
    chart.positions[2] = 123;
    chart.pinned.add(1);
    chart.select(1);
    chart.transform = { k: 2, x: 10, y: 20 };
    chart.setFilters([{ scope: "edges", field: "weight", op: "gt", value: 1 }]);
    const view = chart.saveView();
    chart.positions[2] = 9;
    chart.transform.k = 1;
    chart.loadView(view);
    assert.equal(chart.positions[2], 123);
    assert.equal(chart.transform.k, 2);
    assert.ok(chart.pinned.has(1));
    assert.equal(chart.selected, 1);
    assert.equal(chart.filteredEdges, 2);
    const spec = chart.toSpec();
    assert.equal(env.api.normalize(spec).options.view.positions[1].x, 123);
    assert.throws(() =>
      chart.loadView({ ...view, transform: { k: Infinity, x: 0, y: 0 } }),
    );
    assert.throws(() =>
      chart.loadView({ ...view, positions: [{ id: 999, x: 0, y: 0 }] }),
    );
    assert.throws(() =>
      chart.loadView({
        ...view,
        positions: [view.positions[0], view.positions[0]],
      }),
    );
  } finally {
    env.close();
  }
});

test("Canvas fallback reports actual drawing budgets without pretending to show the full graph", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot(10001));
    assert.equal(chart.filteredNodes, 10001);
    assert.equal(chart.drawNodeIndices.length, 10000);
    assert.ok(
      chart.counts.textContent.includes(
        "Canvas fallback draws 10,000/10,001 nodes",
      ),
    );
    assert.equal(chart.snapshot().querySelectorAll("circle").length, 10000);
    assert.equal(chart.toSpec().config.network.nodes.length, 10001);
  } finally {
    env.close();
  }
});

test("WebGL camera motion does not rebuild GPU geometry; drag updates one position texel", async () => {
  const env = setup(true);
  try {
    const chart = await env.api.mount(env.host, plot());
    env.flush();
    const uploads = chart.gpu.uploadCount,
      buffers = env.gl.buffers.length;
    chart.zoom(1.3, 200, 200);
    env.flush();
    assert.equal(chart.gpu.uploadCount, uploads);
    assert.equal(env.gl.buffers.length, buffers);
    assert.ok(
      env.gl.calls.some(
        (c: any[]) => c[0] === "drawArraysInstanced" && c[3] === 6,
      ),
    );
    chart.positions[0] = 100;
    chart.gpu.updatePositions(chart.positions, 0);
    const call = env.gl.calls
      .filter((c: any[]) => c[0] === "texSubImage2D")
      .at(-1);
    assert.equal(call[5], 1);
    assert.equal(call[6], 1);
  } finally {
    env.close();
  }
});

test("WebGL context loss switches to declared Canvas budgets and restore reloads buffers", async () => {
  const env = setup(true);
  try {
    const chart = await env.api.mount(env.host, plot());
    const loss = new env.w.Event("webglcontextlost", { cancelable: true });
    chart.gpuCanvas.dispatchEvent(loss);
    assert.ok(loss.defaultPrevented);
    assert.equal(chart.gpuLost, true);
    assert.equal(chart.gpuCanvas.hidden, true);
    chart.gpuCanvas.dispatchEvent(new env.w.Event("webglcontextrestored"));
    assert.equal(chart.gpuLost, false);
    assert.equal(chart.gpuCanvas.hidden, false);
    assert.ok(chart.gpu.uploadCount >= 2);
  } finally {
    env.close();
  }
});

test("spatial picking moves with node positions and does not scan every graph node", () => {
  const env = setup();
  try {
    const positions = new Float32Array(200000),
      indices = Array.from({ length: 100000 }, (_, i) => i);
    for (let i = 0; i < indices.length; i++) {
      positions[i * 2] = i * 50;
      positions[i * 2 + 1] = 0;
    }
    const grid = new env.api.SpatialIndex(positions, indices);
    const candidates = grid.query(100, 0, 10);
    assert.deepEqual(Array.from(candidates), [2]);
    positions[4] = 500;
    grid.move(2, 100, 0);
    assert.equal(grid.query(100, 0, 10).length, 0);
    assert.ok(grid.query(500, 0, 10).includes(2));
  } finally {
    env.close();
  }
});

test("timeline frame switches preserve shared node positions and honest counts", async () => {
  const env = setup();
  try {
    const first = plot(3),
      second = plot(4);
    (first.config.network as any).frames = [
      { label: "2020", network: structuredClone(first.config.network) },
      { label: "2021", network: second.config.network },
    ];
    first.config.options.timeline = true;
    const chart = await env.api.mount(env.host, first);
    chart.positions[0] = 777;
    chart.manualPins.add(0);
    chart.pinned.add(0);
    chart.setFrame(1);
    assert.equal(chart.positions[0], 777);
    assert.equal(chart.config.nodes.length, 4);
    assert.equal(chart.frameLabel.textContent, "2021");
    assert.ok(chart.counts.textContent.includes("4 nodes"));
    chart.setFrame(0);
    assert.equal(chart.positions[0], 777);
    assert.equal(chart.config.nodes.length, 3);
  } finally {
    env.close();
  }
});

test("unsafe styles, attribute objects, filter operators and identity fields fail validation", () => {
  const env = setup();
  try {
    for (const mutate of [
      (s: any) => (s.config.options.node_color = "url(javascript:x)"),
      (s: any) => (s.config.network.nodes[0].attrs = { bad: {} }),
      (s: any) =>
        (s.config.network.nodes[0].identity = {
          type: "integer",
          value: "1.5",
        }),
      (s: any) =>
        (s.config.options.filters = [
          { scope: "nodes", field: "score", op: "eval", value: "x" },
        ]),
      (s: any) =>
        (s.config.options.node_size = { field: "score", range: [NaN, 2] }),
      (s: any) =>
        (s.config.options.annotations = [{ text: "x", node_id: 999 }]),
    ]) {
      const spec = plot();
      mutate(spec);
      assert.throws(() => env.api.normalize(spec));
    }
  } finally {
    env.close();
  }
});

test("graph and CSV exports preserve typed identities, attrs and current filtered coverage", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot());
    const files: any[] = [];
    chart.save = (blob: any, extension: string) =>
      files.push({ blob, extension });
    chart.export("graph");
    chart.export("view");
    chart.export("nodes");
    chart.export("edges");
    assert.deepEqual(
      files.map((f) => f.extension),
      ["graph.json", "view.json", "nodes.csv", "edges.csv"],
    );
    const text = (blob: any) =>
      new Promise<string>((resolve) => {
        const r = new env.w.FileReader();
        r.onload = () => resolve(r.result);
        r.readAsText(blob);
      });
    const graph = JSON.parse(await text(files[0].blob));
    assert.equal(graph.config.network.nodes[1].identity.type, "string");
    assert.equal(graph.config.network.nodes[1].identity.value, "n1");
    assert.equal(graph.config.network.nodes[1].attrs.score, 1);
    const csv = await text(files[2].blob);
    assert.ok(csv.includes("identity_type,identity_value"));
    assert.ok(csv.includes("score,sector"));
    assert.ok(csv.includes("Rendered filtered network rows only"));
  } finally {
    env.close();
  }
});

test("GPU helper uploads exactly 100000 nodes and 1000000 edges as typed buffers", () => {
  const env = setup(true);
  try {
    const canvas = env.w.document.createElement("canvas"),
      renderer = new env.w.OpenEconNetworkWebGL.Renderer(canvas),
      nodes = new Array(100000).fill(null),
      links = Array.from({ length: 1000000 }, (_, i) => ({
        a: i % 100000,
        b: (i + 1) % 100000,
      })),
      positions = new Float32Array(200000);
    renderer.upload({
      nodes,
      links,
      positions,
      radius: () => 2,
      nodeColor: () => "#163d68",
      nodeActive: () => true,
      nodeVisible: () => true,
      edgeVisible: () => true,
      edgeWidth: () => 1,
      edgeColor: () => "#163d68",
      edgeAlpha: () => 0.28,
    });
    assert.equal(renderer.nodeCount, 100000);
    assert.equal(renderer.edgeCount, 1000000);
    assert.deepEqual(env.gl.buffers, [2800000, 28000000, 0]);
    renderer.draw({ k: 1, x: 0, y: 0 }, 640, 420, 1, 0.9, false);
    assert.ok(
      env.gl.calls.some(
        (c: any[]) => c[0] === "drawArraysInstanced" && c[4] === 1000000,
      ),
    );
    renderer.dispose();
  } finally {
    env.close();
  }
});

test("timeline Graph JSON keeps original base counts and restores the selected frame with new nodes", async () => {
  const env = setup();
  try {
    const first = plot(2),
      later = plot(4);
    (first.config.network as any).frames = [
      { label: "first", network: structuredClone(first.config.network) },
      { label: "later", network: later.config.network },
    ];
    first.config.options.timeline = true;
    const chart = await env.api.mount(env.host, first);
    chart.setFrame(1);
    chart.select(3);
    chart.positions[6] = 222;
    const spec = chart.toSpec();
    assert.equal(spec.config.network.node_count, 2);
    assert.equal(spec.config.options.frame_index, 1);
    assert.equal(env.api.normalize(spec).options.view.selected, 3);
    const replacement = await env.api.mount(env.host, spec);
    assert.equal(replacement.frameIndex, 1);
    assert.equal(replacement.positions[6], 222);
    assert.equal(replacement.selected, 3);
  } finally {
    env.close();
  }
});

test("visible neighbor counts respect node and edge filters", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(
      env.host,
      plot(4, {
        filters: [{ scope: "edges", field: "weight", op: "gte", value: 2 }],
      }),
    );
    chart.select(1);
    assert.equal(chart.neighborSet(1).size, 1);
    assert.ok(chart.selectionText.textContent.includes("1 neighbors shown"));
    chart.setFilters([{ scope: "nodes", field: "score", op: "gte", value: 2 }]);
    assert.equal(chart.selected, null);
  } finally {
    env.close();
  }
});

test("invalid layout controls fail before a worker starts and label dictionaries map attributes", async () => {
  const env = setup();
  try {
    for (const options of [
      { layout: "grid", layout_options: { charge: -1 } },
      { layout: "forceatlas2", layout_options: { theta: 0 } },
      { layout: "geographic" },
      { layout: "radial", layout_options: { root: 999 } },
      { layout: "fixed", view: null },
    ])
      assert.throws(() => env.api.normalize(plot(4, options)));
    const chart = await env.api.mount(
      env.host,
      plot(4, {
        node_label: {
          field: "sector",
          scale: "categorical",
          domain: ["A", "B"],
          range: ["Sector A", "Sector B"],
        },
      }),
    );
    assert.equal(chart.nodeLabels[1], "Sector B");
  } finally {
    env.close();
  }
});

test("publication snapshots include mapped arrow colors, legend swatches and exact annotations", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(
      env.host,
      plot(4, {
        edge_color: "#abcdef",
        legend: true,
        node_color: {
          field: "score",
          scale: "linear",
          range: ["#000", "#fff"],
        },
        annotations: [{ text: "注釈", node_id: 3 }],
      }),
    );
    chart.select(3);
    const svg = chart.snapshot();
    assert.equal(svg.querySelector("polygon")?.getAttribute("fill"), "#abcdef");
    assert.ok(svg.textContent.includes("score: 0 (linear)"));
    assert.ok(svg.textContent.includes("score: 3"));
    assert.ok(svg.textContent.includes("注釈"));
    assert.equal(
      chart.exportScene().annotations[0].x,
      chart.positions[6] + chart.radius(3) + 8 / chart.transform.k,
    );
    assert.equal(chart.exportScene().nodes.at(-1).highlighted, true);
    assert.equal(chart.exportScene().legend.length, 2);
    assert.ok(chart.exportLayout().legendLines.length >= 2);
  } finally {
    env.close();
  }
});

test("single-color palettes remain valid for numeric color scales", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(
      env.host,
      plot(4, {
        palette: ["#163d68"],
        node_color: { field: "score", scale: "linear" },
      }),
    );
    assert.equal(chart.nodeColor(0), "#163d68");
    assert.equal(chart.nodeColor(3), "#163d68");
  } finally {
    env.close();
  }
});

test("coincident-node picking has an explicit work budget instead of scanning 100000 nodes", () => {
  const env = setup();
  try {
    const positions = new Float32Array(200000),
      grid = new env.api.SpatialIndex(
        positions,
        Array.from({ length: 100000 }, (_, i) => i),
      );
    assert.equal(grid.query(0, 0, 1).length, 0);
    assert.equal(grid.overflow, true);
  } finally {
    env.close();
  }
});

test("frame coordinates override old automatic positions while intentional manual pins persist", async () => {
  const env = setup();
  try {
    const first = plot(2),
      later = plot(2);
    later.config.network.nodes[0].x = 999;
    later.config.network.nodes[1].x = 888;
    (first.config.network as any).frames = [
      { label: "first", network: structuredClone(first.config.network) },
      { label: "later", network: later.config.network },
    ];
    first.config.options.timeline = true;
    const chart = await env.api.mount(env.host, first);
    chart.positions[2] = 777;
    chart.manualPins.add(1);
    chart.pinned.add(1);
    chart.setFrame(1);
    assert.equal(chart.positions[0], 999);
    assert.equal(chart.positions[2], 777);
    assert.ok(chart.pinned.has(1));
    chart.setFrame(0);
    assert.equal(chart.positions[0], 0);
    assert.equal(chart.positions[2], 777);
  } finally {
    env.close();
  }
});

test("timeline switches restart the selected worker layout with current coordinates and new topology", async () => {
  const env = setup();
  try {
    const jobs: any[] = [];
    env.w.OpenEconNetworkWorkerURL = "blob:network-test";
    env.w.Worker = class {
      onmessage: any;
      onerror: any;
      messages: any[] = [];
      terminated = false;
      constructor() {
        jobs.push(this);
      }
      postMessage(m: any) {
        this.messages.push(m);
      }
      terminate() {
        this.terminated = true;
      }
    };
    const first = plot(2, { layout: "circular" }),
      later = plot(3, { layout: "circular" });
    (first.config.network as any).frames = [
      { label: "first", network: structuredClone(first.config.network) },
      { label: "later", network: later.config.network },
    ];
    first.config.options.timeline = true;
    const chart = await env.api.mount(env.host, first);
    assert.equal(jobs.length, 1);
    chart.setFrame(1);
    assert.equal(jobs.length, 2);
    assert.equal(jobs[0].terminated, true);
    assert.equal(jobs[1].messages[0].layout, "circular");
    assert.equal(jobs[1].messages[0].nodes.length, 3);
    assert.equal(jobs[1].messages[0].nodes[2].x, 80);
    assert.notEqual(jobs[1].messages[0].job, jobs[0].messages[0].job);
  } finally {
    env.close();
  }
});

test("attribute filter aliases, missing-null comparisons and numeric mappings match Python semantics", async () => {
  const env = setup();
  try {
    const spec = plot(4, {
      node_size: { field: "attrs.score", range: [2, 8] },
      filters: [{ scope: "nodes", field: "attr.score", op: "gte", value: 2 }],
    });
    delete (spec.config.network.nodes[3].attrs as any).sector;
    const chart = await env.api.mount(env.host, spec);
    assert.equal(chart.filteredNodes, 2);
    assert.equal(chart.radius(3), 8);
    chart.setFilters([
      { scope: "nodes", field: "attrs.sector", op: "eq", value: null },
    ]);
    assert.equal(chart.filteredNodes, 1);
    assert.equal(chart.config.nodes[chart.drawNodeIndices[0]].id, 3);
    chart.setFilters([
      { scope: "nodes", field: "attr.sector", op: "in", value: [null, "A"] },
    ]);
    assert.equal(chart.filteredNodes, 3);
  } finally {
    env.close();
  }
});

test("timeline identities and directedness remain stable across every frame including the base graph", () => {
  const env = setup();
  try {
    const specification = () => {
      const base = plot(2),
        later = plot(3);
      (base.config.network as any).frames = [
        { label: "later", network: later.config.network },
      ];
      return base;
    };
    assert.equal(env.api.normalize(specification()).frames.length, 1);
    for (const mutate of [
      (spec: any) => {
        spec.config.network.frames[0].network.nodes[0].identity.value = "999";
      },
      (spec: any) => {
        spec.config.network.frames[0].network.nodes[0].identity.type = "string";
      },
      (spec: any) => {
        spec.config.network.frames[0].network.directed = false;
      },
      (spec: any) => {
        spec.config.network.frames[0].network.nodes[2].identity = {
          ...spec.config.network.nodes[0].identity,
        };
      },
    ]) {
      const spec = specification();
      mutate(spec);
      assert.throws(() => env.api.normalize(spec), /preserve|stable/);
    }
    const legacy = specification();
    for (const network of [
      legacy.config.network,
      (legacy.config.network as any).frames[0].network,
    ])
      for (const node of network.nodes) delete (node as any).identity;
    assert.equal(env.api.normalize(legacy).frames.length, 1);
  } finally {
    env.close();
  }
});

test("categorical null domains and ordinary frame zero match Python presentation validation", async () => {
  const env = setup();
  try {
    const spec = plot(4, {
      frame_index: 0,
      node_color: {
        field: "sector",
        scale: "categorical",
        domain: [null, "A"],
        range: ["#111", "#fff"],
      },
      node_label: {
        field: "sector",
        scale: "categorical",
        domain: [null, "A"],
        range: ["Missing sector", "Sector A"],
      },
    });
    delete (spec.config.network.nodes[1].attrs as any).sector;
    const chart = await env.api.mount(env.host, spec);
    assert.equal(chart.nodeColor(1), "#111");
    assert.equal(chart.nodeLabels[1], "Missing sector");
    assert.equal(chart.frameIndex, 0);
    assert.throws(
      () => env.api.normalize(plot(4, { frame_index: 1 })),
      /existing frame/,
    );
  } finally {
    env.close();
  }
});

test("saved position identities reject reordered entities before positions or pins change", async () => {
  const env = setup();
  try {
    const chart = await env.api.mount(env.host, plot(3));
    chart.positions[0] = 777;
    chart.pinned.add(0);
    const view = chart.saveView();
    assert.equal(view.positions[0].identity.type, "integer");
    assert.equal(view.positions[0].identity.value, "0");
    const changed = plot(3),
      zero = { ...changed.config.network.nodes[0].identity };
    changed.config.network.nodes[0].identity = {
      ...changed.config.network.nodes[1].identity,
    };
    changed.config.network.nodes[1].identity = zero;
    const replacement = await env.api.mount(env.host, changed),
      positions = Array.from(replacement.positions),
      pins = Array.from(replacement.pinned);
    assert.throws(() => replacement.loadView(view), /identity does not match/);
    assert.deepEqual(Array.from(replacement.positions), positions);
    assert.deepEqual(Array.from(replacement.pinned), pins);
    const unbound = {
      ...view,
      positions: view.positions.map(
        ({ identity, ...position }: any) => position,
      ),
    };
    assert.throws(() => replacement.loadView(unbound), /identity binding/);
  } finally {
    env.close();
  }
});

test("saved identity bindings preserve huge integer strings and allow explicitly legacy unbound views", async () => {
  const env = setup();
  try {
    const spec = plot(2);
    spec.config.network.nodes[0].identity = {
      type: "integer",
      value: "9007199254740993",
    };
    const chart = await env.api.mount(env.host, spec),
      view = chart.saveView();
    assert.equal(view.positions[0].identity.value, "9007199254740993");
    const invalid = structuredClone(view);
    invalid.positions[0].identity.value = "9007199254740992";
    assert.throws(() => chart.loadView(invalid), /does not match/);
    const legacy = plot(2);
    for (const n of legacy.config.network.nodes) delete (n as any).identity;
    const old = await env.api.mount(env.host, legacy),
      oldView = old.saveView();
    assert.equal(Object.hasOwn(oldView.positions[0], "identity"), false);
    assert.doesNotThrow(() => old.loadView(oldView));
    const bound = structuredClone(oldView);
    bound.positions[0].identity = { type: "integer", value: "0" };
    assert.doesNotThrow(() => old.loadView(bound));
    bound.positions[0].identity = { type: "string", value: "0" };
    assert.throws(() => old.loadView(bound), /does not match/);
  } finally {
    env.close();
  }
});

function pooled(spec: any) {
  const graph = spec.config.network,
    nodes: any[] = [],
    edges: any[] = [];
  const intern = (records: any[], pool: any[]) =>
    records.map((record) => {
      const key = JSON.stringify(record);
      const found = pool.findIndex((other) => JSON.stringify(other) === key);
      if (found >= 0) return found;
      pool.push(record);
      return pool.length - 1;
    });
  const ref = (network: any) => ({
    ...network,
    nodes: intern(network.nodes, nodes),
    edges: intern(network.edges, edges),
  });
  const { frames, ...base } = graph;
  const compact: any = {
    encoding: "timeline-pool-v1",
    base: ref(base),
    frames: frames.map((frame: any) => ({
      label: frame.label,
      network: ref(frame.network),
    })),
    nodes,
    edges,
  };
  return { ...spec, config: { ...spec.config, network: compact } };
}

test("pooled timeline shares immutable records, preserves order and camera, switches without asynchronous stale frames", async () => {
  const env = setup();
  try {
    const spec: any = plot();
    const second: any = plot(5).config.network;
    spec.config.network.frames = [
      { label: "first", network: plot().config.network },
      { label: "second", network: second },
      { label: "repeat", network: second },
    ];
    spec.config.options.timeline = true;
    const chart = await env.api.mount(env.host, pooled(spec));
    assert.equal(
      chart.config.nodes[0],
      chart.config.frames[0].network.nodes[0],
    );
    assert.equal(
      chart.config.frames[1].network.nodes[0],
      chart.config.frames[2].network.nodes[0],
    );
    for (const index of [1, 2, 0, 2, 1]) chart.setFrame(index);
    assert.equal(chart.config.nodes.length, 5);
    assert.equal(chart.frameLabel.textContent, "second");
    const exported = chart.toSpec();
    assert.equal(exported.config.network.frames.length, 3);
    assert.equal(exported.config.options.frame_index, 1);
    assert.deepEqual(
      Array.from(exported.config.network.frames, (f: any) => f.label),
      ["first", "second", "repeat"],
    );
    env.api.unmount(env.host);
    assert.equal(env.api.get(env.host), undefined);
  } finally {
    env.close();
  }
});

test("pooled timeline refuses bad indices, amplification and unused records before mounting a worker", () => {
  const env = setup();
  try {
    const spec: any = plot();
    spec.config.network.frames = [
      { label: "same", network: plot().config.network },
    ];
    for (const invalid of [true, -1, 1000, 1.5]) {
      const wire = pooled(spec);
      wire.config.network.base.nodes[0] = invalid;
      assert.throws(() => env.api.normalize(wire), /reference/);
    }
    const huge = pooled(spec);
    huge.config.network.base.nodes = Array(100001).fill(0);
    assert.throws(() => env.api.normalize(huge), /aggregate/);
    const unused = pooled(spec);
    unused.config.network.nodes.push({ ...unused.config.network.nodes[0] });
    assert.throws(() => env.api.normalize(unused), /unused/);
  } finally {
    env.close();
  }
});
