/* Code-driven sparse layouts. D3 supplies its existing force layout and quadtree only.
 * ForceAtlas2 independently implements the equations of Jacomy et al. (2014),
 * https://doi.org/10.1371/journal.pone.0098679: degree mass, Barnes-Hut repulsion,
 * gravity, linear/LinLog attraction and adaptive global/local speed.
 * Explicit work/iteration/time budgets; no V-by-V arrays. */
(function (scope) {
  "use strict";
  const MAX_NODES = 100000, MAX_EDGES = 1000000, COORDINATE_LIMIT = 1e9;
  const LAYOUTS = new Set(["d3-force", "forceatlas2", "circular", "grid", "radial", "hierarchical", "geographic", "community", "fixed"]);
  const KEYS = {
    "d3-force": ["charge", "link_distance", "theta", "collision"],
    forceatlas2: ["scaling", "gravity", "strong_gravity", "edge_weight_influence", "linlog", "outbound_attraction_distribution", "jitter_tolerance", "slowdown", "theta"],
    circular: ["radius", "angle"], grid: ["spacing", "columns"], radial: ["spacing", "root", "angle"],
    hierarchical: ["spacing", "direction"], geographic: ["projection", "scale"], community: ["spacing", "radius"], fixed: [],
  };
  let active = null;
  const now = () => Date.now();
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const coordinate = value => Number.isFinite(value) && Math.abs(value) <= COORDINATE_LIMIT;
  const fail = message => { throw new Error(message); };
  function numeric(raw, key, fallback, min, max, integer) {
    const value = own(raw, key) ? raw[key] : fallback;
    if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) fail("Invalid layout option: " + key);
    return value;
  }
  function boolean(raw, key, fallback) {
    const value = own(raw, key) ? raw[key] : fallback;
    if (typeof value !== "boolean") fail("Invalid layout option: " + key);
    return value;
  }
  function choice(raw, key, fallback, choices) {
    const value = own(raw, key) ? raw[key] : fallback;
    if (!choices.includes(value)) fail("Invalid layout option: " + key);
    return value;
  }
  function optionsFor(message, layout) {
    const raw = message.layout_options === undefined ? {} : message.layout_options;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail("Invalid layout options");
    const allowed = new Set(["iterations", "work_limit", "time_limit_ms", ...KEYS[layout]]);
    if (Object.keys(raw).some(key => !allowed.has(key))) fail("Unknown or incompatible layout option");
    const n = message.nodes.length, o = {
      iterations: numeric(raw, "iterations", layout === "d3-force" ? 160 : layout === "forceatlas2" ? 200 : 1, 1, 2000, true),
      work_limit: numeric(raw, "work_limit", 200000000, 1000, 1000000000, true),
      time_limit_ms: numeric(raw, "time_limit_ms", 30000, 100, 120000, true),
    };
    if (layout === "d3-force") Object.assign(o, {
      charge: numeric(raw, "charge", -45, -10000, 0), link_distance: numeric(raw, "link_distance", 45, 1, 10000),
      theta: numeric(raw, "theta", 0.9, 0.2, 2), collision: boolean(raw, "collision", true),
    });
    if (layout === "forceatlas2") Object.assign(o, {
      scaling: numeric(raw, "scaling", 2, 0.01, 10000), gravity: numeric(raw, "gravity", 1, 0, 100),
      strong_gravity: boolean(raw, "strong_gravity", false), edge_weight_influence: numeric(raw, "edge_weight_influence", 1, 0, 4),
      linlog: boolean(raw, "linlog", false), outbound_attraction_distribution: boolean(raw, "outbound_attraction_distribution", false),
      jitter_tolerance: numeric(raw, "jitter_tolerance", 1, 0.01, 10), slowdown: numeric(raw, "slowdown", 1, 0.01, 100), theta: numeric(raw, "theta", 0.9, 0.2, 2),
    });
    if (["circular", "community"].includes(layout)) o.radius = numeric(raw, "radius", Math.max(40, Math.sqrt(n) * 12), 1, 100000);
    if (["circular", "radial"].includes(layout)) o.angle = numeric(raw, "angle", -90, -360, 360) * Math.PI / 180;
    if (["grid", "radial", "hierarchical", "community"].includes(layout)) o.spacing = numeric(raw, "spacing", layout === "grid" || layout === "community" ? 40 : 60, 1, 10000);
    if (layout === "grid") o.columns = numeric(raw, "columns", Math.max(1, Math.ceil(Math.sqrt(n))), 1, MAX_NODES, true);
    if (layout === "radial" && own(raw, "root")) o.root = numeric(raw, "root", 0, 0, Number.MAX_SAFE_INTEGER, true);
    if (layout === "hierarchical") o.direction = choice(raw, "direction", "TB", ["TB", "BT", "LR", "RL"]);
    if (layout === "geographic") Object.assign(o, {
      projection: choice(raw, "projection", "equirectangular", ["equirectangular", "mercator"]), scale: numeric(raw, "scale", 2, 0.01, 10000),
    });
    return o;
  }
  function validate(message) {
    if (!message || typeof message.job !== "string" || !message.job.length || message.job.length > 100
        || !Array.isArray(message.nodes) || !Array.isArray(message.edges)
        || message.nodes.length > MAX_NODES || message.edges.length > MAX_EDGES) fail("Invalid network layout request");
    const layout = message.layout === undefined ? "d3-force" : message.layout;
    if (!LAYOUTS.has(layout)) fail("Unknown network layout");
    if (message.seed !== undefined && (!Number.isSafeInteger(message.seed) || message.seed < 0 || message.seed > 4294967295)) fail("Invalid layout seed");
    if (message.point_size !== undefined && (!Number.isFinite(message.point_size) || message.point_size < 1 || message.point_size > 24)) fail("Invalid layout point size");
    const positions = message.positions;
    if (positions !== undefined && (!(positions instanceof Float32Array) || positions.length !== message.nodes.length * 2 || !positions.every(coordinate))) fail("Invalid initial layout positions");
    const ids = new Set();
    for (const node of message.nodes) {
      if (!node || !Number.isSafeInteger(node.id) || node.id < 0 || ids.has(node.id)
          || !Number.isFinite(node.degree) || node.degree < 0 || !Number.isSafeInteger(node.group) || node.group < 0) fail("Invalid layout node");
      ids.add(node.id);
      for (const key of ["x", "y", "fx", "fy"]) if (node[key] !== undefined && !coordinate(node[key])) fail("Invalid node coordinate");
      for (const key of ["fixed", "pinned"]) if (node[key] !== undefined && typeof node[key] !== "boolean") fail("Invalid fixed node flag");
      if ((node.x !== undefined) !== (node.y !== undefined)) fail("Initial node coordinates require both x and y");
      if (node.fixed !== undefined && node.pinned !== undefined && node.fixed !== node.pinned) fail("fixed and pinned cannot disagree");
      if ((node.fixed || node.pinned || layout === "fixed") && ((node.x === undefined && node.fx === undefined && !positions) || (node.y === undefined && node.fy === undefined && !positions))) fail("Fixed nodes require coordinates");
      for (const [key, max] of [["longitude", 180], ["latitude", 90]]) {
        if (node[key] !== undefined && (!Number.isFinite(node[key]) || Math.abs(node[key]) > max)) fail("Invalid geographic coordinate");
        if (layout === "geographic" && node[key] === undefined) fail("Geographic layout requires longitude and latitude");
      }
    }
    for (const edge of message.edges) if (!edge || !ids.has(edge.source) || !ids.has(edge.target) || !Number.isFinite(edge.weight)) fail("Invalid layout edge");
    if (layout === "forceatlas2" && message.edges.some(edge => edge.weight < 0)) fail("ForceAtlas2 requires nonnegative edge weights");
    const options = optionsFor(message, layout);
    if (options.root !== undefined && !ids.has(options.root)) fail("Radial root is not in the displayed graph");
    return { layout, options };
  }
  function cancel() {
    if (!active) return;
    active.cancelled = true; clearTimeout(active.timer);
    if (active.simulation) active.simulation.stop();
    active = null;
  }
  function check(job, work = 0) {
    if (active !== job || job.cancelled) fail("cancelled");
    if (work > job.options.work_limit - job.operations) fail("work_limit");
    job.operations += work;
    if (now() - job.started >= job.options.time_limit_ms) fail("time_limit");
  }
  function setPosition(node, x, y) {
    if (!coordinate(x) || !coordinate(y)) fail("Layout coordinates exceeded safe range");
    node.x = node.fx === undefined ? x : node.fx; node.y = node.fy === undefined ? y : node.fy;
  }
  function emit(job, done, reason) {
    if (active !== job || job.cancelled) return;
    const positions = new Float32Array(job.nodes.length * 2);
    for (let i = 0; i < job.nodes.length; i++) {
      const node = job.nodes[i]; if (!coordinate(node.x) || !coordinate(node.y)) fail("Invalid layout coordinates");
      positions[2 * i] = node.x; positions[2 * i + 1] = node.y;
    }
    job.lastEmission = now();
    scope.postMessage({ type: done ? "done" : "progress", job: job.id, layout: job.layout,
      progress: done ? 1 : Math.min(0.999999, job.ticks / job.options.iterations), positions,
      iterations: job.ticks, operations: job.operations, reason: reason || (done ? "complete" : "running") }, [positions.buffer]);
  }
  function initialize(message, validated) {
    const nodes = [], index = new Map(), n = message.nodes.length, seed = message.seed === undefined ? 42 : message.seed;
    const phase = (seed >>> 0) / 4294967296 * 2 * Math.PI, golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const raw = message.nodes[i], r = 10 * Math.sqrt(i + 0.5), angle = phase + i * golden;
      const x = raw.x === undefined ? message.positions ? message.positions[2 * i] : r * Math.cos(angle) : raw.x;
      const y = raw.y === undefined ? message.positions ? message.positions[2 * i + 1] : r * Math.sin(angle) : raw.y;
      const node = { id: raw.id, degree: raw.degree, group: raw.group, x, y, mass: 1, index: i,
        longitude: raw.longitude, latitude: raw.latitude, fx: raw.fx, fy: raw.fy };
      if (raw.fixed || raw.pinned) { if (node.fx === undefined) node.fx = x; if (node.fy === undefined) node.fy = y; }
      setPosition(node, x, y); nodes.push(node); index.set(raw.id, i);
    }
    const sources = new Int32Array(message.edges.length), targets = new Int32Array(message.edges.length), weights = new Float64Array(message.edges.length);
    for (let i = 0; i < message.edges.length; i++) {
      const edge = message.edges[i], s = index.get(edge.source), t = index.get(edge.target);
      sources[i] = s; targets[i] = t; weights[i] = edge.weight; nodes[s].mass++; nodes[t].mass++;
    }
    return { id: message.job, ...validated, nodes, sources, targets, weights, index, seed,
      pointSize: message.point_size === undefined ? 4 : message.point_size, ticks: 0, operations: 0,
      started: now(), lastEmission: 0, cancelled: false, timer: null, simulation: null, iterator: null };
  }
  function requireD3() { if (!scope.d3) scope.importScripts(new URL("d3.min.js", scope.location.href).href); }
  function d3DensityGuard(job) {
    // D3's closed kernels cannot expose visit counts. Bound pathological coincident
    // repulsion and dense collision neighborhoods before entering a synchronous tick.
    const coincident = new Map(), cells = new Map(), cellSize = 2 * (job.pointSize + 9);
    for (const node of job.nodes) {
      const key = node.x + "," + node.y, count = (coincident.get(key) || 0) + 1;
      if (count > 64) fail("D3 coincident-node budget exceeded; use ForceAtlas2 or distinct initial coordinates");
      coincident.set(key, count);
      if (job.options.collision) {
        const x = Math.floor(node.x / cellSize), y = Math.floor(node.y / cellSize), cell = x + "," + y;
        const value = cells.get(cell); if (value) value.count++; else cells.set(cell, { x, y, count: 1 });
      }
    }
    for (const cell of cells.values()) {
      let count = 0;
      for (let x = cell.x - 1; x <= cell.x + 1; x++) for (let y = cell.y - 1; y <= cell.y + 1; y++) count += cells.get(x + "," + y)?.count || 0;
      if (count > 512) fail("D3 collision-density budget exceeded; use ForceAtlas2 or collision=False");
    }
  }
  function* d3Layout(job) {
    requireD3();
    if (job.nodes.every(node => node.fx !== undefined && node.fy !== undefined)) { job.ticks = 1; return "complete"; }
    d3DensityGuard(job);
    const o = job.options, links = Array.from(job.sources, (s, i) => ({ source: job.nodes[s].id, target: job.nodes[job.targets[i]].id, weight: Math.min(100, Math.abs(job.weights[i])) }));
    const simulation = scope.d3.forceSimulation(job.nodes).stop().randomSource(scope.d3.randomLcg(job.seed))
      .alphaDecay(0.035).force("link", scope.d3.forceLink(links).id(node => node.id)
        .distance(edge => Math.max(1, o.link_distance - 7 + 16 / (1 + Math.sqrt(edge.weight))))
        .strength(edge => 0.12 + 0.15 * Math.sqrt(edge.weight / (1 + edge.weight))))
      .force("charge", scope.d3.forceManyBody().strength(o.charge).distanceMax(700).theta(o.theta)).force("center", scope.d3.forceCenter(0, 0));
    if (o.collision) simulation.force("collide", scope.d3.forceCollide(node => job.pointSize + 2 + Math.min(7, Math.sqrt(node.degree))).iterations(1));
    job.simulation = simulation;
    // D3 has no visitor counter: account conservatively for estimated tree and edge work.
    const perTick = Math.ceil(job.nodes.length * (8 + 8 * Math.log2(job.nodes.length + 1)) + job.sources.length * 4);
    const batch = job.nodes.length <= 2000 && job.sources.length <= 10000 ? 8 : 1;
    while (job.ticks < o.iterations) {
      const count = Math.min(batch, o.iterations - job.ticks);
      for (let i = 0; i < count; i++) { check(job, perTick); d3DensityGuard(job); simulation.tick(); job.ticks++; }
      if (job.ticks < o.iterations && simulation.alpha() >= 0.012) emit(job, false);
      yield; if (simulation.alpha() < 0.012) return "converged";
    }
    return "complete";
  }
  function* forceAtlas2(job) {
    requireD3();
    const n = job.nodes.length, o = job.options, nodes = job.nodes;
    const dx = new Float64Array(n), dy = new Float64Array(n), oldX = new Float64Array(n), oldY = new Float64Array(n);
    let speed = 1, efficiency = 1;
    const compensation = o.outbound_attraction_distribution ? nodes.reduce((sum, node) => sum + node.mass, 0) / Math.max(1, n) : 1;
    for (let iteration = 0; iteration < o.iterations; iteration++) {
      oldX.set(dx); oldY.set(dy); dx.fill(0); dy.fill(0); check(job, n * 2);
      const tree = scope.d3.quadtree(nodes, node => node.x, node => node.y);
      tree.visitAfter(quad => {
        let mass = 0, x = 0, y = 0;
        if (quad.length) {
          for (const child of quad) if (child && child.mass) { mass += child.mass; x += child.mass * child.cx; y += child.mass * child.cy; }
        } else for (let leaf = quad; leaf; leaf = leaf.next) { mass += leaf.data.mass; x += leaf.data.mass * leaf.data.x; y += leaf.data.mass * leaf.data.y; }
        quad.mass = mass; quad.cx = mass ? x / mass : 0; quad.cy = mass ? y / mass : 0;
      });
      yield;
      for (let i = 0; i < n; i++) {
        const node = nodes[i];
        if (node.fx !== undefined && node.fy !== undefined) { if ((i & 511) === 511) yield; continue; }
        tree.visit((quad, x0, y0, x1, y1) => {
          check(job, 1); if (!quad.mass) return true;
          let rx = node.x - quad.cx, ry = node.y - quad.cy, distance2 = rx * rx + ry * ry;
          const contains = node.x >= x0 && node.x < x1 && node.y >= y0 && node.y < y1;
          if (quad.length && !contains && distance2 > 0 && (x1 - x0) ** 2 < o.theta * o.theta * distance2) {
            const f = o.scaling * node.mass * quad.mass / distance2; dx[i] += rx * f; dy[i] += ry * f; return true;
          }
          if (quad.length) return false;
          for (let leaf = quad; leaf; leaf = leaf.next) if (leaf.data.index !== i) {
            check(job, 1); const other = leaf.data;
            rx = node.x - other.x; ry = node.y - other.y; distance2 = rx * rx + ry * ry;
            if (distance2 < 1e-12) {
              const sign = i < other.index ? 1 : -1, a = (Math.min(i, other.index) * 0.754877666 + Math.max(i, other.index) * 0.569840291 + job.seed) % (2 * Math.PI);
              rx = sign * 1e-6 * Math.cos(a); ry = sign * 1e-6 * Math.sin(a); distance2 = 1e-12;
            }
            const f = o.scaling * node.mass * other.mass / distance2; dx[i] += rx * f; dy[i] += ry * f;
          }
          return true;
        });
        const distance = Math.hypot(node.x, node.y);
        if (distance > 0) { const f = node.mass * o.gravity * (o.strong_gravity ? 1 : 1 / distance); dx[i] -= node.x * f; dy[i] -= node.y * f; }
        if ((i & 511) === 511) yield;
      }
      for (let e = 0; e < job.sources.length; e++) {
        check(job, 1); const s = job.sources[e], t = job.targets[e];
        if (s !== t) {
          const rx = nodes[s].x - nodes[t].x, ry = nodes[s].y - nodes[t].y, distance = Math.hypot(rx, ry);
          let coefficient = o.edge_weight_influence === 0 ? 1 : Math.pow(job.weights[e], o.edge_weight_influence);
          if (!Number.isFinite(coefficient)) fail("ForceAtlas2 weighted attraction overflow");
          if (o.linlog) coefficient *= distance > 0 ? Math.log1p(distance) / distance : 0;
          if (o.outbound_attraction_distribution) coefficient *= compensation / nodes[s].mass;
          const fx = -coefficient * rx, fy = -coefficient * ry; dx[s] += fx; dy[s] += fy; dx[t] -= fx; dy[t] -= fy;
        }
        if ((e & 8191) === 8191) yield;
      }
      let swing = 0, traction = 0;
      for (let i = 0; i < n; i++) {
        check(job, 1); const node = nodes[i];
        if (node.fx === undefined || node.fy === undefined) {
          const sx = node.fx === undefined ? oldX[i] - dx[i] : 0, sy = node.fy === undefined ? oldY[i] - dy[i] : 0;
          const tx = node.fx === undefined ? oldX[i] + dx[i] : 0, ty = node.fy === undefined ? oldY[i] + dy[i] : 0;
          swing += node.mass * Math.hypot(sx, sy); traction += node.mass * 0.5 * Math.hypot(tx, ty);
        }
        if ((i & 4095) === 4095) yield;
      }
      if (!Number.isFinite(swing) || !Number.isFinite(traction)) fail("ForceAtlas2 numerical overflow");
      const estimate = 0.05 * Math.sqrt(n);
      let tolerance = o.jitter_tolerance * Math.max(Math.sqrt(estimate), Math.min(10, estimate * traction / Math.max(1, n * n)));
      if (swing > 2 * traction) { if (efficiency > 0.05) efficiency *= 0.5; tolerance = Math.max(tolerance, o.jitter_tolerance); }
      const target = swing > 0 ? tolerance * efficiency * traction / swing : speed;
      if (swing > tolerance * traction) { if (efficiency > 0.05) efficiency *= 0.7; } else if (speed < 1000) efficiency *= 1.3;
      speed += Math.min(target - speed, 0.5 * speed);
      let movement = 0;
      for (let i = 0; i < n; i++) {
        check(job, 1); const node = nodes[i], swinging = node.mass * Math.hypot(oldX[i] - dx[i], oldY[i] - dy[i]);
        const factor = speed / (1 + Math.sqrt(speed * swinging)) / o.slowdown;
        const x = node.fx === undefined ? node.x + dx[i] * factor : node.fx, y = node.fy === undefined ? node.y + dy[i] * factor : node.fy;
        movement += Math.hypot(x - node.x, y - node.y); setPosition(node, x, y); if ((i & 4095) === 4095) yield;
      }
      job.ticks++;
      if (job.ticks < o.iterations && (!job.lastEmission || now() - job.lastEmission >= 100)) emit(job, false);
      yield; if (movement / Math.max(1, n) < 1e-6 && iteration >= 2) return "converged";
    }
    return "complete";
  }
  function* adjacency(job, reverse, undirected) {
    const n = job.nodes.length, counts = new Int32Array(n + 1);
    for (let e = 0; e < job.sources.length; e++) {
      check(job, 1); counts[(reverse ? job.targets[e] : job.sources[e]) + 1]++; if (undirected) counts[job.targets[e] + 1]++;
      if ((e & 8191) === 8191) yield;
    }
    for (let i = 1; i <= n; i++) { check(job, 1); counts[i] += counts[i - 1]; if ((i & 8191) === 8191) yield; }
    const cursor = counts.slice(), neighbors = new Int32Array(counts[n]);
    for (let e = 0; e < job.sources.length; e++) {
      check(job, 1); const s = reverse ? job.targets[e] : job.sources[e], t = reverse ? job.sources[e] : job.targets[e];
      neighbors[cursor[s]++] = t; if (undirected) neighbors[cursor[t]++] = s; if ((e & 8191) === 8191) yield;
    }
    return { offsets: counts, neighbors };
  }
  function* radial(job) {
    const graph = yield* adjacency(job, false, true), n = job.nodes.length, depths = new Int32Array(n).fill(-1), queue = new Int32Array(n);
    let root = job.options.root === undefined ? 0 : job.index.get(job.options.root);
    if (job.options.root === undefined) for (let i = 1; i < n; i++) if (job.nodes[i].degree > job.nodes[root].degree) root = i;
    let head = 0, tail = 0, largestDepth = 0, scan = 0;
    if (n) { queue[tail++] = root; depths[root] = 0; }
    while (head < tail || scan < n) {
      if (head === tail) {
        while (scan < n && depths[scan] >= 0) scan++; if (scan >= n) break;
        depths[scan] = largestDepth + 1; queue[tail++] = scan;
      }
      const u = queue[head++]; check(job, 1); largestDepth = Math.max(largestDepth, depths[u]);
      for (let e = graph.offsets[u]; e < graph.offsets[u + 1]; e++) {
        check(job, 1); const v = graph.neighbors[e]; if (depths[v] < 0) { depths[v] = depths[u] + 1; queue[tail++] = v; }
        if ((e & 8191) === 8191) yield;
      }
      if ((head & 2047) === 2047) yield;
    }
    const counts = new Int32Array(n + 1), used = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) counts[depths[i]]++;
    for (let i = 0; i < n; i++) {
      check(job, 1); const d = depths[i], a = job.options.angle + 2 * Math.PI * used[d]++ / counts[d], r = d * job.options.spacing;
      setPosition(job.nodes[i], r * Math.cos(a), r * Math.sin(a)); if ((i & 4095) === 4095) yield;
    }
  }
  function* hierarchical(job) {
    // Iterative Kosaraju SCC condensation: cycles share a rank; no recursive stack overflow.
    const forward = yield* adjacency(job, false, false), reverse = yield* adjacency(job, true, false), n = job.nodes.length;
    const visited = new Uint8Array(n), order = new Int32Array(n), stack = new Int32Array(n), cursors = new Int32Array(n);
    let ordered = 0;
    for (let start = 0; start < n; start++) {
      if (visited[start]) continue;
      let top = 0; stack[0] = start; cursors[0] = forward.offsets[start]; visited[start] = 1;
      while (top >= 0) {
        check(job, 1); const u = stack[top];
        if (cursors[top] < forward.offsets[u + 1]) {
          const v = forward.neighbors[cursors[top]++]; if (!visited[v]) { visited[v] = 1; stack[++top] = v; cursors[top] = forward.offsets[v]; }
        } else { order[ordered++] = u; top--; }
        if ((job.operations & 4095) === 0) yield;
      }
    }
    const component = new Int32Array(n).fill(-1); let components = 0;
    for (let k = ordered - 1; k >= 0; k--) {
      const start = order[k]; if (component[start] >= 0) continue;
      let top = 0; stack[0] = start; component[start] = components;
      while (top >= 0) {
        check(job, 1); const u = stack[top--];
        for (let e = reverse.offsets[u]; e < reverse.offsets[u + 1]; e++) {
          check(job, 1); const v = reverse.neighbors[e]; if (component[v] < 0) { component[v] = components; stack[++top] = v; }
          if ((e & 8191) === 8191) yield;
        }
        if ((job.operations & 4095) === 0) yield;
      }
      components++;
    }
    const incoming = new Int32Array(components), sizes = new Int32Array(components + 1);
    for (let e = 0; e < job.sources.length; e++) {
      check(job, 1); const s = component[job.sources[e]], t = component[job.targets[e]]; if (s !== t) { incoming[t]++; sizes[s + 1]++; }
      if ((e & 8191) === 8191) yield;
    }
    for (let c = 1; c <= components; c++) sizes[c] += sizes[c - 1];
    const links = new Int32Array(sizes[components]), cursor = sizes.slice();
    for (let e = 0; e < job.sources.length; e++) {
      check(job, 1); const s = component[job.sources[e]], t = component[job.targets[e]]; if (s !== t) links[cursor[s]++] = t;
      if ((e & 8191) === 8191) yield;
    }
    const ranks = new Int32Array(components), queue = new Int32Array(components); let head = 0, tail = 0;
    for (let c = 0; c < components; c++) if (!incoming[c]) queue[tail++] = c;
    while (head < tail) {
      const c = queue[head++]; check(job, 1);
      for (let e = sizes[c]; e < sizes[c + 1]; e++) {
        check(job, 1); const t = links[e]; ranks[t] = Math.max(ranks[t], ranks[c] + 1); if (--incoming[t] === 0) queue[tail++] = t;
        if ((e & 8191) === 8191) yield;
      }
      if ((head & 2047) === 2047) yield;
    }
    const counts = new Int32Array(n + 1), used = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) counts[ranks[component[i]]]++;
    for (let i = 0; i < n; i++) {
      check(job, 1); const rank = ranks[component[i]], x = (used[rank]++ - (counts[rank] - 1) / 2) * job.options.spacing, y = rank * job.options.spacing, d = job.options.direction;
      setPosition(job.nodes[i], d === "LR" ? y : d === "RL" ? -y : x, d === "BT" ? -y : d === "LR" || d === "RL" ? x : y);
      if ((i & 4095) === 4095) yield;
    }
  }
  function* staticLayout(job) {
    const n = job.nodes.length, o = job.options;
    if (job.layout === "radial") yield* radial(job);
    else if (job.layout === "hierarchical") yield* hierarchical(job);
    else if (job.layout === "community") {
      // Deterministic group packing; not advertised as a multilevel force algorithm.
      const groups = new Map();
      for (let i = 0; i < n; i++) { check(job, 1); if (!groups.has(job.nodes[i].group)) groups.set(job.nodes[i].group, []); groups.get(job.nodes[i].group).push(i); if ((i & 4095) === 4095) yield; }
      const ordered = [...groups.keys()].sort((a, b) => a - b); let largest = 0;
      for (const key of ordered) largest = Math.max(largest, groups.get(key).length);
      // Adjacent community discs must not overlap, even when every node is its own group.
      const groupRadius = o.spacing * Math.sqrt(largest + 0.5);
      const radius = Math.max(o.radius, ordered.length <= 1 ? 0 : (groupRadius + o.spacing / 2) / Math.sin(Math.PI / ordered.length));
      const columns = Math.ceil(Math.sqrt(ordered.length)), rows = Math.ceil(ordered.length / columns);
      const cellSize = Math.max(2 * groupRadius + o.spacing, 2 * o.radius / Math.max(1, columns - 1));
      for (let g = 0; g < ordered.length; g++) {
        const members = groups.get(ordered[g]), angle = 2 * Math.PI * g / ordered.length;
        // Many communities use a compact grid of nonoverlapping discs instead of
        // an excessively large ring; both paths remain deterministic and sparse.
        const cx = ordered.length > 64 ? (g % columns - (columns - 1) / 2) * cellSize : ordered.length === 1 ? 0 : radius * Math.cos(angle);
        const cy = ordered.length > 64 ? (Math.floor(g / columns) - (rows - 1) / 2) * cellSize : ordered.length === 1 ? 0 : radius * Math.sin(angle);
        for (let j = 0; j < members.length; j++) {
          check(job, 1); const r = members.length === 1 ? 0 : o.spacing * Math.sqrt(j + 0.5), a = j * Math.PI * (3 - Math.sqrt(5));
          setPosition(job.nodes[members[j]], cx + r * Math.cos(a), cy + r * Math.sin(a)); if ((j & 4095) === 4095) yield;
        }
      }
    } else for (let i = 0; i < n; i++) {
      check(job, 1); const node = job.nodes[i];
      if (job.layout === "circular") { const a = o.angle + 2 * Math.PI * i / Math.max(1, n); setPosition(node, o.radius * Math.cos(a), o.radius * Math.sin(a)); }
      if (job.layout === "grid") { const cols = Math.min(o.columns, Math.max(1, n)), rows = Math.ceil(n / cols); setPosition(node, (i % cols - (cols - 1) / 2) * o.spacing, (Math.floor(i / cols) - (rows - 1) / 2) * o.spacing); }
      if (job.layout === "geographic") {
        const lat = Math.min(85.0511287798066, Math.max(-85.0511287798066, node.latitude));
        const y = o.projection === "mercator" ? -Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) * 180 / Math.PI : -node.latitude;
        setPosition(node, node.longitude * o.scale, y * o.scale);
      }
      if ((i & 4095) === 4095) yield;
    }
    job.ticks = 1; return "complete";
  }
  function drive(job) {
    if (active !== job || job.cancelled) return;
    try {
      const started = now();
      while (active === job && !job.cancelled) {
        check(job); const step = job.iterator.next();
        if (step.done) { emit(job, true, step.value || "complete"); cancel(); return; }
        if (job.layout === "d3-force" || now() - started >= 8) break;
      }
      if (active === job && !job.cancelled) job.timer = setTimeout(() => drive(job), 0);
    } catch (error) {
      if (active !== job || job.cancelled) return;
      if (error.message === "work_limit" || error.message === "time_limit") emit(job, true, error.message);
      else scope.postMessage({ type: "error", job: job.id, message: error.message || "Layout unavailable" });
      cancel();
    }
  }
  scope.onmessage = function (event) {
    const message = event.data;
    if (message && message.type === "cancel") { if (active && active.id === message.job) cancel(); return; }
    if (!message || message.type !== "layout") return;
    cancel();
    try {
      const started = now(), validated = validate(message), job = initialize(message, validated); active = job;
      job.started = started;
      job.iterator = job.layout === "d3-force" ? d3Layout(job) : job.layout === "forceatlas2" ? forceAtlas2(job) : staticLayout(job);
      if (!job.nodes.length) { emit(job, true, "complete"); cancel(); } else job.timer = setTimeout(() => drive(job), 0);
    } catch (error) {
      scope.postMessage({ type: "error", job: typeof message.job === "string" ? message.job : "", message: error.message || "Invalid network layout request" }); cancel();
    }
  };
})(self);
