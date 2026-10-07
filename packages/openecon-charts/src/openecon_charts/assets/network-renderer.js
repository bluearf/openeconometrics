/* Code-configured network charts: WebGL2 camera rendering, bounded Canvas fallback, worker layouts. */
(function (root) {
  "use strict";
  const scriptURL = document.currentScript && document.currentScript.src;
  const workerURL = scriptURL
    ? new URL("network-worker.js", scriptURL).href
    : null;
  const MAX_NODES = 100000,
    MAX_EDGES = 1000000,
    MAX_LABEL_BYTES = 4096,
    MAX_LABEL_TOTAL = 16 * 1024 * 1024;
  const CANVAS_NODES = 10000,
    CANVAS_EDGES = 30000,
    VECTOR_PRIMITIVES = 200000;
  const COLORS = [
    "#163d68",
    "#4c78a8",
    "#739bb4",
    "#234e5a",
    "#79949a",
    "#546883",
    "#a5b6c4",
    "#426f83",
  ];
  const instances = new Map();
  let nextJob = 0,
    removalObserver = null,
    sweepFrame = null;

  function require(value, message) {
    if (!value) throw new TypeError(message);
  }
  function integer(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }
  function keys(value, expected, name, optional) {
    require(value && typeof value === "object" && !Array.isArray(value), name +
      " must be an object.");
    const names = Object.keys(value);
    require(names.every((key) => expected.includes(key)) &&
      expected.every(
        (key) => (optional || []).includes(key) || Object.hasOwn(value, key),
      ), name + " contains missing or unsupported fields.");
  }
  function textBytes(value, limit, name) {
    require(typeof value === "string" && value.length <= limit, name +
      " is too long or is not text.");
    let bytes = 0;
    for (const character of value) {
      const code = character.codePointAt(0);
      require(code > 31 &&
        !(code >= 127 && code <= 159) &&
        !(code >= 0xd800 && code <= 0xdfff), name +
        " contains invalid characters.");
      bytes += code < 128 ? 1 : code < 2048 ? 2 : code < 65536 ? 3 : 4;
      require(bytes <= limit, name + " exceeds its UTF-8 size limit.");
    }
    return bytes;
  }
  function finiteOption(value, low, high, name) {
    require(typeof value === "number" &&
      Number.isFinite(value) &&
      value >= low &&
      value <= high, name + " is outside its supported range.");
    return value;
  }
  function color(value) {
    require(typeof value === "string" &&
      /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(
        value,
      ), "Network colors must be opaque hexadecimal colors.");
    return value;
  }
  const LAYOUTS = [
    "d3-force",
    "force",
    "forceatlas2",
    "circular",
    "grid",
    "radial",
    "hierarchical",
    "geographic",
    "community",
    "fixed",
  ];
  const EMPTY_ATTRIBUTES = Object.freeze(Object.create(null));
  function charge(budget, bytes) {
    budget.bytes += bytes;
    require(budget.bytes <=
      128 *
        1024 *
        1024, "Network payload exceeds the conservative 128 MiB browser budget.");
  }
  function attributes(raw, budget) {
    if (raw === undefined) return EMPTY_ATTRIBUTES;
    require(raw &&
      typeof raw === "object" &&
      !Array.isArray(raw) &&
      Object.keys(raw).length <=
        64, "Attributes require at most 64 scalar fields.");
    const result = Object.create(null);
    for (const [key, value] of Object.entries(raw)) {
      require(!["__proto__", "constructor", "prototype"].includes(
        key,
      ), "Reserved attribute name.");
      const keyBytes = textBytes(key, 256, "Attribute key");
      require(value === null ||
        typeof value === "boolean" ||
        (typeof value === "number" && Number.isFinite(value)) ||
        typeof value === "string", "Attributes must be finite scalars.");
      const valueBytes =
        typeof value === "string"
          ? textBytes(value, 4096, "Attribute value") * 2
          : String(value).length;
      if (budget) charge(budget, keyBytes * 2 + valueBytes + 8);
      result[key] = value;
    }
    return result;
  }
  function filters(raw) {
    if (raw === undefined) return [];
    require(Array.isArray(raw) &&
      raw.length <= 100, "Filters require at most 100 rules.");
    return raw.map((rule) => {
      keys(rule, ["scope", "field", "op", "value"], "Filter");
      require(["nodes", "edges"].includes(rule.scope) &&
        ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"].includes(
          rule.op,
        ), "Unsupported filter operation.");
      textBytes(rule.field, 256, "Filter field");
      const values = ["in", "not_in"].includes(rule.op)
        ? rule.value
        : [rule.value];
      require(Array.isArray(values) &&
        values.length <= 10000, "Filter membership must be a bounded list.");
      values.forEach((value) => {
        require(value === null ||
          typeof value === "boolean" ||
          typeof value === "string" ||
          (typeof value === "number" &&
            Number.isFinite(value)), "Filter values must be finite scalars.");
        if (typeof value === "string") textBytes(value, 4096, "Filter value");
      });
      return {
        ...rule,
        value: Array.isArray(rule.value) ? rule.value.slice() : rule.value,
      };
    });
  }
  function mapping(value, key) {
    if (typeof value !== "object" || value === null) {
      if (key === "node_color" || key === "edge_color") return color(value);
      if (key === "node_label") {
        textBytes(value, 256, "Label field");
        return value;
      }
      return finiteOption(
        value,
        key === "node_size" ? 1 : 0.1,
        key === "node_size" ? 256 : 64,
        key,
      );
    }
    keys(value, ["field", "scale", "range", "domain", "missing"], key, [
      "scale",
      "range",
      "domain",
      "missing",
    ]);
    textBytes(value.field, 256, "Mapping field");
    const scale =
      value.scale || (key === "node_label" ? "categorical" : "linear");
    require(["linear", "sqrt", "log", "categorical"].includes(
      scale,
    ), "Unsupported mapping scale.");
    const result = { ...value, scale };
    if (value.range !== undefined) {
      require(Array.isArray(value.range) &&
        value.range.length >= 2 &&
        value.range.length <= 64, "Mapping range must contain 2–64 values.");
      result.range = value.range.map((v) =>
        key.endsWith("color")
          ? color(v)
          : key === "node_label"
            ? (textBytes(String(v), 4096, "Mapped label"), v)
            : finiteOption(
                v,
                key === "node_size" ? 1 : 0.1,
                key === "node_size" ? 256 : 64,
                "Mapping range",
              ),
      );
      if (scale !== "categorical" && key !== "node_label")
        require(value.range.length ===
          2, "Numeric mapping range requires two bounds.");
    }
    if (value.domain !== undefined) {
      require(Array.isArray(value.domain) &&
        value.domain.length >= 2 &&
        value.domain.length <= 10000, "Mapping domain must be bounded.");
      result.domain = value.domain.slice();
      result.domain.forEach((v) =>
        require(v === null ||
          typeof v === "string" ||
          (typeof v === "number" && Number.isFinite(v)) ||
          typeof v ===
            "boolean", "Mapping domain must contain finite scalars."),
      );
      if (scale !== "categorical")
        require(value.domain.length === 2 &&
          value.domain.every(
            (v) => typeof v === "number" && Number.isFinite(v),
          ) &&
          value.domain[0] <
            value
              .domain[1], "Numeric mapping domain requires two increasing bounds.");
    }
    if (value.missing !== undefined) {
      if (key.endsWith("color")) color(value.missing);
      else if (key === "node_label")
        textBytes(value.missing, 4096, "Missing label");
      else
        finiteOption(
          value.missing,
          key === "node_size" ? 1 : 0.1,
          key === "node_size" ? 256 : 64,
          "Missing mapping",
        );
    }
    return result;
  }
  function decodeTimeline(graph) {
    if (!graph || graph.encoding === undefined) return graph;
    keys(
      graph,
      ["encoding", "base", "frames", "nodes", "edges"],
      "Timeline transport",
    );
    require(graph.encoding ===
      "timeline-pool-v1", "Unsupported timeline encoding.");
    require(Array.isArray(graph.frames) &&
      graph.frames.length >= 1 &&
      graph.frames.length <= 60, "Timeline transport requires 1–60 frames.");
    const networks = [graph.base];
    for (const frame of graph.frames) {
      keys(frame, ["label", "network"], "Timeline transport frame");
      textBytes(frame.label, 4096, "Frame label");
      networks.push(frame.network);
    }
    // Admit aggregate reference counts before creating expanded arrays.
    for (const [kind, cap] of [
      ["nodes", MAX_NODES],
      ["edges", MAX_EDGES],
    ]) {
      const pool = graph[kind],
        used = new Set();
      require(Array.isArray(pool) &&
        pool.length <= cap, "Timeline record pool exceeds the display budget.");
      let total = 0;
      for (const network of networks) {
        require(network &&
          typeof network === "object" &&
          !network.frames &&
          Array.isArray(network[kind]), "Invalid timeline reference network.");
        total += network[kind].length;
        require(total <=
          cap, "Timeline references exceed the aggregate display budget.");
        for (const ref of network[kind]) {
          require(integer(ref) &&
            ref < pool.length, "Invalid timeline record reference.");
          used.add(ref);
        }
      }
      require(used.size ===
        pool.length, "Timeline pool contains unused records.");
    }
    const expand = (network) => ({
      ...network,
      nodes: network.nodes.map((index) => graph.nodes[index]),
      edges: network.edges.map((index) => graph.edges[index]),
    });
    return {
      ...expand(graph.base),
      frames: graph.frames.map((frame) => ({
        label: frame.label,
        network: expand(frame.network),
      })),
    };
  }
  function graphData(graph, frame, budget) {
    budget = budget || { bytes: 0 };
    keys(
      graph,
      [
        "nodes",
        "edges",
        "directed",
        "node_count",
        "edge_count",
        "shown_node_count",
        "shown_edge_count",
        "sampled",
        "selection",
        "grouping",
        "frames",
      ],
      "Network data",
      ["grouping", "frames"],
    );
    require(!frame ||
      !Object.hasOwn(
        graph,
        "frames",
      ), "Recursive network frames are not supported.");
    require(Array.isArray(graph.nodes) &&
      Array.isArray(graph.edges) &&
      graph.nodes.length <= MAX_NODES &&
      graph.edges.length <=
        MAX_EDGES, "Network render limit: 100,000 nodes and 1,000,000 edges.");
    const ids = new Set();
    let labelBytes = 0;
    const nodes = graph.nodes.map((node) => {
      keys(
        node,
        [
          "id",
          "label",
          "degree",
          "group",
          "identity",
          "attrs",
          "x",
          "y",
          "fx",
          "fy",
          "fixed",
          "pinned",
          "longitude",
          "latitude",
        ],
        "Network node",
        [
          "identity",
          "attrs",
          "x",
          "y",
          "fx",
          "fy",
          "fixed",
          "pinned",
          "longitude",
          "latitude",
        ],
      );
      require(integer(node.id) &&
        !ids.has(
          node.id,
        ), "Node IDs must be unique nonnegative safe integers.");
      ids.add(node.id);
      const cached = budget.nodeCache?.get(node);
      if (cached) {
        charge(budget, 8);
        labelBytes += textBytes(node.label, MAX_LABEL_BYTES, "Node label");
        require(labelBytes <= MAX_LABEL_TOTAL, "Network labels exceed 16 MiB.");
        return cached;
      }
      require(integer(node.group) &&
        typeof node.degree === "number" &&
        Number.isFinite(node.degree) &&
        node.degree >=
          0, "Node groups and degrees must be valid nonnegative numbers.");
      const nodeLabelBytes = textBytes(
        node.label,
        MAX_LABEL_BYTES,
        "Node label",
      );
      labelBytes += nodeLabelBytes;
      charge(budget, 128 + nodeLabelBytes * 2);
      require(labelBytes <= MAX_LABEL_TOTAL, "Network labels exceed 16 MiB.");
      const result = { ...node, attrs: attributes(node.attrs, budget) };
      if (node.identity !== undefined) {
        keys(node.identity, ["type", "value"], "Node identity");
        require(["integer", "string"].includes(
          node.identity.type,
        ), "Unsupported identity type.");
        charge(
          budget,
          32 + textBytes(node.identity.value, 4096, "Node identity") * 2,
        );
        if (node.identity.type === "integer")
          require(/^-?(?:0|[1-9][0-9]*)$/.test(
            node.identity.value,
          ), "Integer identity must be decimal.");
        result.identity = { ...node.identity };
      }
      for (const key of ["x", "y", "fx", "fy"])
        if (node[key] !== undefined) finiteOption(node[key], -1e9, 1e9, key);
      for (const [key, bound] of [
        ["longitude", 180],
        ["latitude", 90],
      ])
        if (node[key] !== undefined)
          finiteOption(node[key], -bound, bound, key);
      require((node.x === undefined) ===
        (node.y === undefined), "Node coordinates require x and y.");
      for (const key of ["fixed", "pinned"])
        if (node[key] !== undefined)
          require(typeof node[key] === "boolean", key + " must be boolean.");
      budget.nodeCache?.set(node, result);
      return result;
    });
    const edges = graph.edges.map((edge) => {
      const cached = budget.edgeCache?.get(edge);
      if (cached) {
        require(ids.has(edge.source) &&
          ids.has(
            edge.target,
          ), "Every edge must reference displayed node IDs.");
        charge(budget, 8);
        return cached;
      }
      charge(budget, 80);
      keys(edge, ["source", "target", "weight", "attrs"], "Network edge", [
        "attrs",
      ]);
      require(integer(edge.source) &&
        integer(edge.target) &&
        ids.has(edge.source) &&
        ids.has(edge.target), "Every edge must reference displayed node IDs.");
      require(typeof edge.weight === "number" &&
        Number.isFinite(edge.weight), "Edge weights must be finite.");
      const result = { ...edge, attrs: attributes(edge.attrs, budget) };
      budget.edgeCache?.set(edge, result);
      return result;
    });
    for (const key of [
      "node_count",
      "edge_count",
      "shown_node_count",
      "shown_edge_count",
    ])
      require(integer(
        graph[key],
      ), "Network counts must be nonnegative safe integers.");
    require(graph.shown_node_count === nodes.length &&
      graph.shown_edge_count === edges.length &&
      graph.node_count >= nodes.length &&
      graph.edge_count >=
        edges.length, "Network display counts are inconsistent.");
    require(graph.node_count !== 0 ||
      graph.edge_count === 0, "A network without nodes cannot contain edges.");
    require(typeof graph.directed === "boolean" &&
      typeof graph.sampled === "boolean", "Network flags must be booleans.");
    require(graph.sampled ===
      (graph.node_count > nodes.length ||
        graph.edge_count >
          edges.length), "The sampled flag must reflect the displayed counts.");
    require(textBytes(graph.selection, 1000, "Selection description") >
      0, "A selection description is required.");
    const grouping =
      graph.grouping === undefined ? "Weak components" : graph.grouping;
    require(textBytes(grouping, 1000, "Grouping description") >
      0, "A grouping description is required.");
    return { ...graph, nodes, edges, grouping };
  }
  function normalize(spec) {
    require(spec &&
      typeof spec === "object" &&
      spec.kind === "network", "A network PlotSpec is required.");
    keys(spec.config, ["network", "options"], "Network configuration", [
      "options",
    ]);
    const budget = {
      bytes: 0,
      nodeCache: new WeakMap(),
      edgeCache: new WeakMap(),
    };
    const graph = graphData(decodeTimeline(spec.config.network), false, budget);
    require(Array.isArray(spec.data) &&
      spec.data.length === 0 &&
      spec.x_label === "" &&
      spec.y_label === "" &&
      spec.sample_n === graph.shown_node_count &&
      spec.total_n === graph.node_count &&
      spec.dropped_n ===
        0, "The network PlotSpec envelope does not match its graph metadata.");
    const raw = spec.config.options === undefined ? {} : spec.config.options;
    require(raw &&
      typeof raw === "object" &&
      !Array.isArray(raw), "Network options must be an object.");
    const options = {
      height: 420,
      point_size: 4,
      line_width: 1,
      opacity: 0.9,
      palette: COLORS.slice(),
      layout: "d3-force",
      layout_options: {},
      filters: [],
      labels: { show: true, min_zoom: 0.01, max_count: 80 },
      legend: false,
      annotations: [],
    };
    const bounds = {
      width: [320, 2400],
      height: [240, 1600],
      point_size: [1, 24],
      line_width: [0.25, 12],
      opacity: [0, 1],
    };
    for (const [key, value] of Object.entries(raw)) {
      if (Object.hasOwn(bounds, key)) {
        options[key] = finiteOption(value, ...bounds[key], key);
        if (key === "width" || key === "height")
          require(Number.isInteger(value), key + " must be an integer.");
      } else if (key === "color") options.color = color(value);
      else if (key === "palette") {
        require(Array.isArray(value) &&
          value.length >= 1 &&
          value.length <= 64, "A palette requires 1–64 colors.");
        options.palette = value.map(color);
      } else if (key === "layout") {
        require(LAYOUTS.includes(value), "Unsupported network layout.");
        options.layout = value === "force" ? "d3-force" : value;
      } else if (key === "layout_options") {
        require(value &&
          typeof value === "object" &&
          !Array.isArray(value), "Layout options must be an object.");
        options.layout_options = { ...value };
      } else if (
        [
          "node_size",
          "node_color",
          "node_label",
          "edge_width",
          "edge_color",
        ].includes(key)
      )
        options[key] = mapping(value, key);
      else if (key === "filters") options.filters = filters(value);
      else if (key === "legend") {
        require(typeof value === "boolean", "legend must be boolean.");
        options.legend = value;
      } else if (key === "labels") {
        if (typeof value === "boolean")
          options.labels = { ...options.labels, show: value };
        else {
          keys(value, ["show", "min_zoom", "max_count"], "Labels", [
            "show",
            "min_zoom",
            "max_count",
          ]);
          if (value.show !== undefined)
            require(typeof value.show ===
              "boolean", "Label show must be boolean.");
          if (value.min_zoom !== undefined)
            finiteOption(value.min_zoom, 0, 100, "Label minimum zoom");
          if (value.max_count !== undefined)
            require(integer(value.max_count) &&
              value.max_count <=
                10000, "Label max_count must be at most 10,000.");
          options.labels = { ...options.labels, ...value };
        }
      } else if (key === "annotations") {
        require(Array.isArray(value) &&
          value.length <= 100, "Annotations require at most 100 entries.");
        options.annotations = value.map((note) => {
          keys(note, ["text", "node_id", "x", "y", "color"], "Annotation", [
            "node_id",
            "x",
            "y",
            "color",
          ]);
          textBytes(note.text, 4096, "Annotation text");
          require(note.node_id === undefined
            ? Number.isFinite(note.x) && Number.isFinite(note.y)
            : integer(
                note.node_id,
              ), "Annotation requires a known node or finite x/y.");
          if (note.color !== undefined) color(note.color);
          return { ...note };
        });
      } else if (key === "view") {
        options.view = value;
      } else if (key === "seed") {
        require(integer(value) && value <= 4294967295, "Seed must be uint32.");
        options.seed = value;
      } else if (key === "frame_index") {
        require(integer(value) && value < 60, "Frame index must be 0–59.");
        options.frame_index = value;
      } else if (key === "timeline") {
        require(typeof value === "boolean", "timeline must be boolean.");
        options.timeline = value;
      } else throw new TypeError("Unknown network option: " + key);
    }
    options.layout_options = validateLayout(
      options.layout,
      options.layout_options,
      graph.nodes,
    );
    if (options.layout === "fixed")
      require(graph.nodes.every(
        (n) =>
          (n.x !== undefined || n.fx !== undefined) &&
          (n.y !== undefined || n.fy !== undefined),
      ), "Fixed layout requires all node coordinates.");
    if (options.layout === "geographic")
      require(graph.nodes.every(
        (n) => n.longitude !== undefined && n.latitude !== undefined,
      ), "Geographic layout requires longitude and latitude.");
    if (options.layout === "forceatlas2")
      require(graph.edges.every(
        (e) => e.weight >= 0,
      ), "ForceAtlas2 requires nonnegative edge weights.");
    const title = spec.title === undefined ? "" : spec.title;
    textBytes(title, 4096, "Network title");
    let frames;
    if (graph.frames !== undefined) {
      require(Array.isArray(graph.frames) &&
        graph.frames.length > 0 &&
        graph.frames.length <= 60, "Network frames require 1–60 entries.");
      let nn = graph.nodes.length,
        ne = graph.edges.length;
      frames = graph.frames.map((f) => {
        keys(f, ["label", "network"], "Network frame");
        textBytes(f.label, 4096, "Frame label");
        const network = graphData(f.network, true, budget);
        nn += network.nodes.length;
        ne += network.edges.length;
        require(nn <= MAX_NODES &&
          ne <=
            MAX_EDGES, "Combined network frame budget exceeds 100,000 nodes or 1,000,000 edges.");
        return { label: f.label, network };
      });
    }
    if (frames) {
      const identityByID = new Map(),
        idByIdentity = new Map();
      for (const network of [graph, ...frames.map((frame) => frame.network)]) {
        require(network.directed ===
          graph.directed, "Network frames must preserve directedness.");
        for (const node of network.nodes) {
          const identity = node.identity || {
              type: "integer",
              value: String(node.id),
            },
            key = JSON.stringify([identity.type, identity.value]);
          require(!identityByID.has(node.id) ||
            identityByID.get(node.id) ===
              key, "Network frames must preserve each display ID's exact typed identity.");
          require(!idByIdentity.has(key) ||
            idByIdentity.get(key) ===
              node.id, "Network frames must use one stable display ID per typed identity.");
          identityByID.set(node.id, key);
          idByIdentity.set(key, node.id);
        }
      }
    }
    if (options.frame_index !== undefined)
      require(frames
        ? options.frame_index < frames.length
        : options.frame_index === 0, "Frame index requires an existing frame.");
    if (Object.hasOwn(options, "view"))
      options.view = validateView(
        options.view,
        frames && options.frame_index !== undefined
          ? frames[options.frame_index].network.nodes
          : graph.nodes,
      );
    const annotationIDs = new Set(graph.nodes.map((n) => n.id));
    if (frames)
      for (const frame of frames)
        for (const n of frame.network.nodes) annotationIDs.add(n.id);
    for (const note of options.annotations)
      if (note.node_id !== undefined)
        require(annotationIDs.has(
          note.node_id,
        ), "Annotation requires a known node.");
    return { ...graph, title, options, frames };
  }
  function validateLayout(layout, raw, nodes) {
    const specific = {
      "d3-force": ["charge", "link_distance", "theta", "collision"],
      forceatlas2: [
        "scaling",
        "gravity",
        "strong_gravity",
        "edge_weight_influence",
        "linlog",
        "outbound_attraction_distribution",
        "jitter_tolerance",
        "slowdown",
        "theta",
      ],
      circular: ["radius", "angle"],
      grid: ["spacing", "columns"],
      radial: ["spacing", "root", "angle"],
      hierarchical: ["spacing", "direction"],
      geographic: ["projection", "scale"],
      community: ["spacing", "radius"],
      fixed: [],
    };
    const allowed = [
      "iterations",
      "work_limit",
      "time_limit_ms",
      ...specific[layout],
    ];
    const ranges = {
      iterations: [1, 2000],
      work_limit: [1000, 1e9],
      time_limit_ms: [100, 120000],
      charge: [-10000, 0],
      link_distance: [1, 10000],
      theta: [0.2, 2],
      scaling: [0.01, 10000],
      gravity: [0, 100],
      edge_weight_influence: [0, 4],
      jitter_tolerance: [0.01, 10],
      slowdown: [0.01, 100],
      radius: [1, 100000],
      angle: [-360, 360],
      spacing: [1, 10000],
      columns: [1, 100000],
      root: [0, Number.MAX_SAFE_INTEGER],
      scale: [0.01, 10000],
    };
    for (const [key, value] of Object.entries(raw)) {
      require(allowed.includes(key), "Unknown or incompatible layout option: " +
        key);
      if (ranges[key]) {
        finiteOption(value, ...ranges[key], key);
        if (
          [
            "iterations",
            "work_limit",
            "time_limit_ms",
            "columns",
            "root",
          ].includes(key)
        )
          require(integer(value), "Layout " + key + " must be an integer.");
      } else if (
        [
          "collision",
          "strong_gravity",
          "linlog",
          "outbound_attraction_distribution",
        ].includes(key)
      )
        require(typeof value === "boolean", "Layout " +
          key +
          " must be boolean.");
      else if (key === "direction")
        require(["TB", "BT", "LR", "RL"].includes(
          value,
        ), "Invalid hierarchical direction.");
      else if (key === "projection")
        require(["equirectangular", "mercator"].includes(
          value,
        ), "Invalid geographic projection.");
    }
    if (raw.root !== undefined)
      require(nodes.some(
        (n) => n.id === raw.root,
      ), "Radial root is not in the displayed graph.");
    return { ...raw };
  }
  function validateView(view, nodes) {
    keys(
      view,
      ["version", "positions", "transform", "selected", "filters"],
      "Saved network view",
      ["selected", "filters"],
    );
    require(view.version === 1, "Unsupported saved view version.");
    require(Array.isArray(view.positions) &&
      view.positions.length <= MAX_NODES, "Saved positions must be bounded.");
    const nodesByID = new Map(nodes.map((n) => [n.id, n])),
      ids = new Set(nodesByID.keys()),
      seen = new Set();
    const positions = view.positions.map((p) => {
      keys(p, ["id", "x", "y", "pinned", "identity"], "Saved position", [
        "pinned",
        "identity",
      ]);
      require(integer(p.id) &&
        ids.has(p.id) &&
        !seen.has(p.id), "Saved positions require unique known IDs.");
      seen.add(p.id);
      const node = nodesByID.get(p.id),
        expected = node.identity || { type: "integer", value: String(node.id) };
      require(!node.identity ||
        p.identity !==
          undefined, "Saved positions require exact typed identity binding for this graph.");
      let identity;
      if (p.identity !== undefined) {
        keys(p.identity, ["type", "value"], "Saved position identity");
        require(["integer", "string"].includes(
          p.identity.type,
        ), "Unsupported saved identity type.");
        textBytes(p.identity.value, 4096, "Saved position identity");
        if (p.identity.type === "integer")
          require(/^-?(?:0|[1-9][0-9]*)$/.test(
            p.identity.value,
          ), "Integer saved identity must be decimal.");
        require(p.identity.type === expected.type &&
          p.identity.value ===
            expected.value, "Saved position identity does not match the graph entity.");
        identity = { ...p.identity };
      }
      finiteOption(p.x, -1e9, 1e9, "Saved x");
      finiteOption(p.y, -1e9, 1e9, "Saved y");
      if (p.pinned !== undefined)
        require(typeof p.pinned ===
          "boolean", "Saved pinned flag must be boolean.");
      return { ...p, ...(identity ? { identity } : {}) };
    });
    keys(view.transform, ["k", "x", "y"], "Saved transform");
    finiteOption(view.transform.k, 0.02, 12, "Saved zoom");
    finiteOption(view.transform.x, -1e9, 1e9, "Saved pan x");
    finiteOption(view.transform.y, -1e9, 1e9, "Saved pan y");
    require(view.selected === undefined ||
      view.selected === null ||
      ids.has(view.selected), "Saved selection requires a known ID.");
    return {
      version: 1,
      positions,
      transform: { ...view.transform },
      selected: view.selected === undefined ? null : view.selected,
      filters: filters(view.filters),
    };
  }
  function initialPositions(nodes) {
    const groups = Array.from(new Set(nodes.map((node) => node.group))).sort(
      (a, b) => a - b,
    );
    const ranks = new Map(),
      groupIndex = new Map(groups.map((group, index) => [group, index]));
    const positions = new Float32Array(nodes.length * 2),
      angle = Math.PI * (3 - Math.sqrt(5));
    const groupRadius =
      groups.length > 1 ? 90 + Math.sqrt(nodes.length) * 7 : 0;
    nodes.forEach((node, index) => {
      const rank = ranks.get(node.group) || 0;
      ranks.set(node.group, rank + 1);
      const center =
        (2 * Math.PI * groupIndex.get(node.group)) / Math.max(1, groups.length);
      const radius = 12 * Math.sqrt(rank + 0.5),
        theta = rank * angle;
      positions[index * 2] =
        Math.cos(center) * groupRadius + Math.cos(theta) * radius;
      positions[index * 2 + 1] =
        Math.sin(center) * groupRadius + Math.sin(theta) * radius;
    });
    return positions;
  }
  function layoutSeed(nodes, edges) {
    let hash = 2166136261;
    for (const node of nodes) {
      hash = Math.imul(hash ^ node.id, 16777619);
      hash = Math.imul(hash ^ node.group, 16777619);
    }
    for (const edge of edges) {
      hash = Math.imul(hash ^ edge.source, 16777619);
      hash = Math.imul(hash ^ edge.target, 16777619);
    }
    return hash >>> 0;
  }
  function csvCell(value) {
    let text = String(value);
    if (typeof value === "string" && /^[=+@\-\t\r]/.test(text))
      text = "'" + text;
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  function coverage(config) {
    const number = (value) => value.toLocaleString("en-US");
    const counts = config.sampled
      ? "Showing " +
        number(config.nodes.length) +
        " of " +
        number(config.node_count) +
        " nodes · " +
        number(config.edges.length) +
        " of " +
        number(config.edge_count) +
        " edges"
      : number(config.node_count) +
        " nodes · " +
        number(config.edge_count) +
        " edges";
    return (
      counts +
      (config.directed ? " · Directed" : " · Undirected") +
      " · Groups: " +
      (config.grouping || "Weak components") +
      (config.sampled ? " · Selection: " + config.selection : "")
    );
  }
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const icons = {
    fit: '<path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5M8 8h8v8H8z"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    minus: '<path d="M5 12h14"/>',
    expand:
      '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 8 3 3m13 5 5-5M8 16l-5 5m13-5 5 5"/>',
    download: '<path d="M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
  };
  function button(action, label, iconName) {
    const node = element("button", "oe-network-button");
    node.type = "button";
    node.dataset.action = action;
    node.setAttribute("aria-label", label);
    node.title = label;
    node.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      icons[iconName] +
      "</svg>";
    return node;
  }
  function wrappedText(context, text, width) {
    const lines = [];
    let current = "";
    for (const character of text) {
      if (current && context.measureText(current + character).width > width) {
        lines.push(current);
        current = "";
      }
      current += character;
    }
    lines.push(current);
    return lines;
  }

  function field(item, key) {
    let value;
    if (key.startsWith("attrs.") || key.startsWith("attr."))
      value = item.attrs && item.attrs[key.slice(key.indexOf(".") + 1)];
    else
      value = Object.hasOwn(item, key)
        ? item[key]
        : item.attrs && item.attrs[key];
    return value === undefined ? null : value;
  }
  const membershipCache = new WeakMap();
  function membership(rule, value) {
    let values = membershipCache.get(rule);
    if (!values) {
      values = new Set(rule.value);
      membershipCache.set(rule, values);
    }
    return values.has(value);
  }
  function matches(item, rules) {
    return rules.every((rule) => {
      const value = field(item, rule.field),
        expected = rule.value;
      switch (rule.op) {
        case "eq":
          return value === expected;
        case "ne":
          return value !== expected;
        case "gt":
          return typeof value === "number" && value > expected;
        case "gte":
          return typeof value === "number" && value >= expected;
        case "lt":
          return typeof value === "number" && value < expected;
        case "lte":
          return typeof value === "number" && value <= expected;
        case "in":
          return membership(rule, value);
        case "not_in":
          return !membership(rule, value);
        default:
          return false;
      }
    });
  }
  function makeMapper(items, rule, fallback, palette, label) {
    if (rule === undefined) return fallback;
    if (typeof rule !== "object")
      return label
        ? (item) => {
            const value = field(item, rule);
            return value === undefined || value === null ? "" : String(value);
          }
        : () => rule;
    if (label) {
      const domain =
          rule.domain ||
          Array.from(
            new Set(
              items
                .map((item) => field(item, rule.field))
                .filter((v) => v !== undefined && v !== null),
            ),
          ).sort((a, b) => String(a).localeCompare(String(b))),
        ranks = new Map(domain.map((v, i) => [v, i]));
      return (item) => {
        const value = field(item, rule.field),
          rank = ranks.get(value);
        return rule.range && rank !== undefined
          ? String(rule.range[rank % rule.range.length])
          : value === undefined || value === null
            ? String(rule.missing || "")
            : String(value);
      };
    }
    const isColor = typeof fallback(items[0] || {}, 0) === "string",
      range = rule.range || (isColor ? palette : [1, 12]);
    if (rule.scale === "categorical") {
      const domain =
        rule.domain ||
        Array.from(
          new Set(
            items
              .map((item) => field(item, rule.field))
              .filter((v) => v !== undefined && v !== null),
          ),
        ).sort((a, b) => String(a).localeCompare(String(b)));
      const ranks = new Map(domain.map((v, i) => [v, i]));
      return (item, index) => {
        const i = ranks.get(field(item, rule.field));
        return i === undefined
          ? rule.missing === undefined
            ? fallback(item, index)
            : rule.missing
          : range[i % range.length];
      };
    }
    let low = Infinity,
      high = -Infinity;
    if (rule.domain) {
      low = rule.domain[0];
      high = rule.domain[rule.domain.length - 1];
    } else
      for (const item of items) {
        const value = field(item, rule.field);
        if (typeof value === "number" && Number.isFinite(value)) {
          low = Math.min(low, value);
          high = Math.max(high, value);
        }
      }
    const scale = (value) =>
      rule.scale === "sqrt"
        ? Math.sqrt(Math.max(0, value))
        : rule.scale === "log"
          ? Math.log1p(Math.max(0, value))
          : value;
    const a = scale(low),
      b = scale(high);
    return (item, index) => {
      const value = field(item, rule.field);
      if (typeof value !== "number" || !Number.isFinite(value))
        return rule.missing === undefined
          ? fallback(item, index)
          : rule.missing;
      const t = Math.max(
        0,
        Math.min(1, b === a ? 0.5 : (scale(value) - a) / (b - a)),
      );
      if (!isColor) return range[0] + (range[range.length - 1] - range[0]) * t;
      if (range.length === 1) return range[0];
      const segment = t * (range.length - 1),
        i = Math.min(range.length - 2, Math.floor(segment)),
        f = segment - i;
      const c1 = root.OpenEconNetworkWebGL
          ? root.OpenEconNetworkWebGL.rgba(range[i], 1)
          : hexRGB(range[i]),
        c2 = root.OpenEconNetworkWebGL
          ? root.OpenEconNetworkWebGL.rgba(range[i + 1], 1)
          : hexRGB(range[i + 1]);
      return (
        "#" +
        [0, 1, 2]
          .map((k) =>
            Math.round((c1[k] + (c2[k] - c1[k]) * f) * 255)
              .toString(16)
              .padStart(2, "0"),
          )
          .join("")
      );
    };
  }
  function hexRGB(value) {
    let hex = value.slice(1);
    if (hex.length === 3) hex = hex.replace(/./g, (c) => c + c);
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  }
  class SpatialIndex {
    constructor(positions, indices) {
      this.cells = new Map();
      this.positions = positions;
      this.cell = 32;
      this.indices = indices;
      for (const i of indices) this.insert(i);
    }
    key(x, y) {
      return Math.floor(x / this.cell) + "," + Math.floor(y / this.cell);
    }
    insert(i) {
      const key = this.key(this.positions[i * 2], this.positions[i * 2 + 1]);
      let values = this.cells.get(key);
      if (!values) this.cells.set(key, (values = new Set()));
      values.add(i);
    }
    move(i, oldX, oldY) {
      const key = this.key(oldX, oldY),
        values = this.cells.get(key);
      if (values) {
        values.delete(i);
        if (!values.size) this.cells.delete(key);
      }
      this.insert(i);
    }
    query(x, y, radius) {
      this.overflow = false;
      const values = [],
        minX = Math.floor((x - radius) / this.cell),
        maxX = Math.floor((x + radius) / this.cell),
        minY = Math.floor((y - radius) / this.cell),
        maxY = Math.floor((y + radius) / this.cell);
      for (let a = minX; a <= maxX; a++)
        for (let b = minY; b <= maxY; b++) {
          const bucket = this.cells.get(a + "," + b);
          if (bucket)
            for (const i of bucket) {
              if (values.length === 5000) {
                this.overflow = true;
                return [];
              }
              values.push(i);
            }
        }
      return values;
    }
  }
  class NetworkChart {
    constructor(host, spec, config) {
      this.host = host;
      this.spec = spec;
      this.config = config;
      this.destroyed = false;
      this.abort = new root.AbortController();
      this.everConnected = host.isConnected;
      this.positions = initialPositions(config.nodes);
      this.transform = { k: 1, x: 0, y: 0 };
      this.pinned = new Set(
        config.nodes.filter((n) => n.fixed || n.pinned).map((n) => n.id),
      );
      this.manualPins = new Set();
      this.positionMemory = new Map();
      this.frameIndex = 0;
      this.index = new Map();
      this.links = [];
      this.selectedNeighbors = new Set();
      this.neighbors = { get: (id) => this.neighborSet(id) };
      this.maxWeight = 1;
      this.gpu = null;
      this.gpuLost = false;
      this.timelineTimer = null;
      this.configureGraph(config);
      this.job = "network-" + ++nextJob;
      this.selected = null;
      this.hovered = null;
      this.frame = null;
      this.presentationFrame = null;
      this.worker = null;
      this.workerTimer = null;
      this.tooltipTimer = null;
      this.pngExport = null;
      this.dialog = null;
      this.pointer = null;
      this.touched = false;
      this.width = 1;
      this.height = config.options.height;
      host.classList.add("oe-network");
      host.dataset.chartType = "network";
      this.heading = element(
        "h3",
        "oe-network-heading",
        config.title || "Network",
      );
      this.toolbar = element("div", "oe-network-toolbar");
      this.search = element("input", "oe-network-search");
      this.search.type = "search";
      this.search.placeholder = "Find a node";
      this.search.setAttribute("aria-label", "Find a network node");
      this.searchResults = element("div", "oe-network-search-results");
      this.searchResults.hidden = true;
      const searchBox = element("div", "oe-network-search-box");
      searchBox.append(this.search, this.searchResults);
      const tools = element("div", "oe-network-tools");
      tools.append(
        button("minus", "Zoom out", "minus"),
        button("plus", "Zoom in", "plus"),
        button("fit", "Fit network", "fit"),
        button("expand", "Expand network", "expand"),
      );
      this.download = element("details", "oe-network-download");
      const summary = element("summary");
      summary.setAttribute("aria-label", "Download network");
      summary.title = "Download network";
      summary.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
        icons.download +
        "</svg>";
      const formats = element("div", "oe-network-export-menu");
      for (const [format, label] of [
        ["png", "PNG"],
        ["svg", "SVG"],
        ["pdf", "PDF"],
        ["nodes", "Nodes CSV"],
        ["edges", "Edges CSV"],
        ["graph", "Graph JSON"],
        ["view", "View JSON"],
      ]) {
        const item = element("button", "", label);
        item.type = "button";
        item.dataset.export = format;
        formats.append(item);
      }
      this.download.append(summary, formats);
      tools.append(this.download);
      this.toolbar.append(searchBox, tools);
      this.stage = element("div", "oe-network-stage");
      this.canvas = element("canvas", "oe-network-canvas");
      this.canvas.tabIndex = 0;
      this.canvas.setAttribute("role", "img");
      this.canvas.setAttribute(
        "aria-label",
        (config.title || "Network") +
          ". " +
          coverage(config) +
          ". Use arrow keys to pan, plus and minus to zoom, and zero to fit.",
      );
      this.gpuCanvas = element("canvas", "oe-network-gpu");
      this.gpuCanvas.setAttribute("aria-hidden", "true");
      if (root.OpenEconNetworkWebGL)
        try {
          this.gpu = new root.OpenEconNetworkWebGL.Renderer(
            this.gpuCanvas,
            () => {
              this.gpuLost = true;
              this.gpuCanvas.hidden = true;
              this.refreshVisibility();
              this.status.textContent = "GPU context lost · Canvas fallback";
              this.schedule();
            },
            () => {
              this.gpuLost = false;
              this.gpuCanvas.hidden = false;
              this.refreshVisibility();
              this.schedule();
            },
          );
        } catch (_) {
          this.gpu = null;
        }
      this.context = this.canvas.getContext("2d");
      require(this
        .context, "This browser does not support Canvas network charts.");
      this.hint = element(
        "span",
        "oe-network-hint",
        "Drag to pan · Scroll to zoom",
      );
      this.tooltip = element("div", "oe-network-tooltip");
      this.tooltip.hidden = true;
      this.tooltip.setAttribute("role", "tooltip");
      this.stage.append(this.gpuCanvas, this.canvas, this.hint, this.tooltip);
      this.legend = element("div", "oe-network-legend");
      this.stage.append(this.legend);
      this.selection = element("div", "oe-network-selection");
      this.selectionText = element(
        "span",
        "",
        "Select a node to inspect its connections.",
      );
      this.clear = element("button", "", "Clear selection");
      this.clear.type = "button";
      this.clear.hidden = true;
      this.selection.append(this.selectionText, this.clear);
      this.footer = element("div", "oe-network-footer");
      this.counts = element("span", "oe-network-coverage", coverage(config));
      this.status = element(
        "span",
        "oe-network-status",
        config.nodes.length ? "Static layout" : "No nodes to display",
      );
      this.status.setAttribute("role", "status");
      this.footer.append(this.counts, this.status);
      host.replaceChildren(
        this.heading,
        this.toolbar,
        this.stage,
        this.selection,
        this.footer,
      );
      this.events();
      this.refreshVisibility();
      this.createTimeline();
      this.resize(true);
      if (config.frames && config.options.frame_index !== undefined)
        this.setFrame(config.options.frame_index, false);
      if (config.options.view) this.loadView(config.options.view);
      if (root.ResizeObserver) {
        this.observer = new root.ResizeObserver(() => this.resize(false));
        this.observer.observe(host);
      }
      this.startLayout();
      this.retryPresentation(2, true);
    }
    listen(target, type, listener, options) {
      target.addEventListener(type, listener, {
        ...(options || {}),
        signal: this.abort.signal,
      });
    }
    events() {
      const resumePresentation = () => {
        if (document.visibilityState !== "hidden")
          this.retryPresentation(2, false);
      };
      this.listen(document, "visibilitychange", resumePresentation);
      this.listen(root, "focus", resumePresentation);
      this.listen(root, "pageshow", resumePresentation);
      this.listen(this.toolbar, "click", (event) => {
        const exportButton = event.target.closest("[data-export]");
        if (exportButton) {
          this.download.open = false;
          this.export(exportButton.dataset.export);
          return;
        }
        const action = event.target.closest("[data-action]");
        if (!action) return;
        if (action.dataset.action === "fit") {
          this.touched = false;
          this.fit();
        } else if (action.dataset.action === "expand") this.expand();
        else
          this.zoom(
            action.dataset.action === "plus" ? 1.3 : 1 / 1.3,
            this.width / 2,
            this.height / 2,
          );
      });
      this.listen(this.clear, "click", () => this.select(null));
      this.listen(this.search, "input", () => this.find());
      this.listen(this.searchResults, "click", (event) => {
        const result = event.target.closest("button[data-node-id]");
        if (!result || !this.searchResults.contains(result)) return;
        this.select(Number(result.dataset.nodeId), true);
        this.searchResults.hidden = true;
        this.canvas.focus();
      });
      this.listen(this.search, "keydown", (event) => {
        if (event.key === "Enter") {
          const first = this.searchResults.querySelector("button");
          if (first) first.click();
        }
        if (event.key === "Escape") {
          this.search.value = "";
          this.searchResults.hidden = true;
        }
      });
      this.listen(document, "pointerdown", (event) => {
        if (!this.toolbar.contains(event.target)) {
          this.download.open = false;
          this.searchResults.hidden = true;
        }
      });
      this.listen(
        this.canvas,
        "wheel",
        (event) => {
          event.preventDefault();
          const point = this.point(event);
          this.zoom(
            Math.exp(-Math.max(-120, Math.min(120, event.deltaY)) * 0.0025),
            point.x,
            point.y,
          );
        },
        { passive: false },
      );
      this.listen(this.canvas, "pointerdown", (event) => {
        if (event.button !== 0) return;
        const point = this.point(event);
        const node = this.hit(point);
        this.pointer = {
          id: event.pointerId,
          x: point.x,
          y: point.y,
          moved: false,
          node,
        };
        if (node !== null) {
          this.stopWorker();
          this.select(node);
          this.pinned.add(node);
          this.manualPins.add(node);
        }
        if (this.canvas.setPointerCapture)
          this.canvas.setPointerCapture(event.pointerId);
        this.canvas.classList.add("is-panning");
        this.hideTooltip();
      });
      this.listen(this.canvas, "pointermove", (event) => {
        const point = this.point(event);
        if (this.pointer && this.pointer.id === event.pointerId) {
          const dx = point.x - this.pointer.x,
            dy = point.y - this.pointer.y;
          if (Math.abs(dx) + Math.abs(dy) > 2) this.pointer.moved = true;
          if (this.pointer.node !== null) {
            const i = this.index.get(this.pointer.node),
              ox = this.positions[i * 2],
              oy = this.positions[i * 2 + 1];
            this.positions[i * 2] += dx / this.transform.k;
            this.positions[i * 2 + 1] += dy / this.transform.k;
            this.spatial.move(i, ox, oy);
            if (this.gpu && !this.gpuLost)
              this.gpu.updatePositions(this.positions, i);
          } else {
            this.transform.x += dx;
            this.transform.y += dy;
          }
          this.pointer.x = point.x;
          this.pointer.y = point.y;
          this.touched = true;
          this.schedule();
          return;
        }
        this.hover(point);
      });
      this.listen(this.canvas, "pointerup", (event) => {
        if (!this.pointer || this.pointer.id !== event.pointerId) return;
        if (!this.pointer.moved) this.select(this.hit(this.point(event)));
        if (
          this.canvas.releasePointerCapture &&
          this.canvas.hasPointerCapture &&
          this.canvas.hasPointerCapture(event.pointerId)
        )
          this.canvas.releasePointerCapture(event.pointerId);
        this.pointer = null;
        this.canvas.classList.remove("is-panning");
      });
      this.listen(this.canvas, "pointercancel", () => {
        this.pointer = null;
        this.canvas.classList.remove("is-panning");
      });
      this.listen(this.canvas, "pointerleave", () => {
        this.hideTooltip();
        this.hovered = null;
        this.schedule();
      });
      this.listen(this.canvas, "keydown", (event) => {
        if (
          [
            "+",
            "=",
            "-",
            "0",
            "Escape",
            "ArrowLeft",
            "ArrowRight",
            "ArrowUp",
            "ArrowDown",
          ].includes(event.key)
        )
          event.preventDefault();
        if (event.key === "+" || event.key === "=")
          this.zoom(1.3, this.width / 2, this.height / 2);
        else if (event.key === "-")
          this.zoom(1 / 1.3, this.width / 2, this.height / 2);
        else if (event.key === "0") {
          this.touched = false;
          this.fit();
        } else if (event.key === "Escape") this.select(null);
        else if (event.key.startsWith("Arrow")) {
          this.transform.x +=
            event.key === "ArrowLeft"
              ? 30
              : event.key === "ArrowRight"
                ? -30
                : 0;
          this.transform.y +=
            event.key === "ArrowUp" ? 30 : event.key === "ArrowDown" ? -30 : 0;
          this.touched = true;
          this.schedule();
        }
      });
    }
    point(event) {
      const box = this.canvas.getBoundingClientRect();
      return { x: event.clientX - box.left, y: event.clientY - box.top };
    }
    radius(index) {
      return this.radii[index];
    }
    nodeColor(index) {
      return this.nodeColors[index];
    }
    related(index) {
      return (
        this.selected === null ||
        this.selectedNeighbors.has(this.config.nodes[index].id) ||
        this.config.nodes[index].id === this.selected
      );
    }
    edgeAlpha(index) {
      const edge = this.links[index];
      return this.selected === null
        ? 0.28
        : edge.source === this.selected || edge.target === this.selected
          ? 0.7
          : 0.06;
    }
    configureGraph(graph) {
      this.config = { ...this.config, ...graph, options: this.config.options };
      const nodes = this.config.nodes,
        edges = this.config.edges,
        options = this.config.options;
      this.index = new Map(nodes.map((node, index) => [node.id, index]));
      this.positions = initialPositions(nodes);
      this.pinned = new Set(this.manualPins);
      nodes.forEach((n, i) => {
        const previous = this.positionMemory.get(n.id),
          manual = this.manualPins.has(n.id) && previous;
        if (manual) {
          this.positions[i * 2] = previous.x;
          this.positions[i * 2 + 1] = previous.y;
        } else {
          if (n.x !== undefined) {
            this.positions[i * 2] = n.x;
            this.positions[i * 2 + 1] = n.y;
          } else if (previous) {
            this.positions[i * 2] = previous.x;
            this.positions[i * 2 + 1] = previous.y;
          }
          if (n.fx !== undefined) this.positions[i * 2] = n.fx;
          if (n.fy !== undefined) this.positions[i * 2 + 1] = n.fy;
        }
        if (n.fixed || n.pinned) this.pinned.add(n.id);
      });
      this.links = edges.map((edge) => ({
        ...edge,
        a: this.index.get(edge.source),
        b: this.index.get(edge.target),
      }));
      this.maxWeight = 1;
      for (const e of edges)
        this.maxWeight = Math.max(this.maxWeight, Math.abs(e.weight));
      const degree = new Uint32Array(nodes.length);
      for (const e of this.links) {
        degree[e.a]++;
        if (e.a !== e.b) degree[e.b]++;
      }
      this.offsets = new Uint32Array(nodes.length + 1);
      for (let i = 0; i < nodes.length; i++)
        this.offsets[i + 1] = this.offsets[i] + degree[i];
      this.adjacency = new Uint32Array(this.offsets[nodes.length]);
      this.incidentEdges = new Uint32Array(this.offsets[nodes.length]);
      const cursor = this.offsets.slice();
      this.links.forEach((e, i) => {
        let slot = cursor[e.a]++;
        this.adjacency[slot] = e.b;
        this.incidentEdges[slot] = i;
        if (e.a !== e.b) {
          slot = cursor[e.b]++;
          this.adjacency[slot] = e.a;
          this.incidentEdges[slot] = i;
        }
      });
      const radius = makeMapper(
        nodes,
        options.node_size,
        (n) =>
          options.point_size *
          (0.8 + Math.min(1.8, Math.sqrt(Math.log1p(n.degree)) * 0.4)),
        options.palette,
      );
      const nodeColor = makeMapper(
        nodes,
        options.node_color,
        (n) =>
          options.color || options.palette[n.group % options.palette.length],
        options.palette,
      );
      const label = makeMapper(
        nodes,
        options.node_label,
        (n) => n.label,
        options.palette,
        true,
      );
      const edgeWidth = makeMapper(
        edges,
        options.edge_width,
        (e) =>
          options.line_width *
          (0.65 + 1.2 * Math.sqrt(Math.abs(e.weight) / this.maxWeight)),
        options.palette,
      );
      const edgeColor = makeMapper(
        edges,
        options.edge_color,
        () => options.color || "#163d68",
        options.palette,
      );
      this.radii = new Float32Array(nodes.length);
      this.nodeColors = new Array(nodes.length);
      this.nodeLabels = new Array(nodes.length);
      this.edgeWidths = new Float32Array(edges.length);
      this.edgeColors = new Array(edges.length);
      this.maxRadius = 1;
      nodes.forEach((n, i) => {
        this.radii[i] = radius(n, i);
        this.maxRadius = Math.max(this.maxRadius, this.radii[i]);
        this.nodeColors[i] = nodeColor(n, i);
        this.nodeLabels[i] = label(n, i);
      });
      edges.forEach((e, i) => {
        this.edgeWidths[i] = edgeWidth(e, i);
        this.edgeColors[i] = edgeColor(e, i);
      });
      this.labelOrder = nodes
        .map((_, i) => i)
        .sort((a, b) => nodes[b].degree - nodes[a].degree)
        .slice(0, 1000);
    }
    neighborSet(id) {
      const i = this.index.get(id),
        result = new Set();
      if (i === undefined) return result;
      for (let j = this.offsets[i]; j < this.offsets[i + 1]; j++)
        if (!this.drawnEdgeFlags || this.drawnEdgeFlags[this.incidentEdges[j]])
          result.add(this.config.nodes[this.adjacency[j]].id);
      return result;
    }
    refreshVisibility() {
      const options = this.config.options,
        nrules = options.filters.filter((r) => r.scope === "nodes"),
        erules = options.filters.filter((r) => r.scope === "edges");
      this.visibleNodes = new Uint8Array(this.config.nodes.length);
      this.visibleEdges = new Uint8Array(this.links.length);
      this.drawNodeIndices = [];
      this.drawEdgeIndices = [];
      this.filteredNodes = 0;
      this.filteredEdges = 0;
      this.config.nodes.forEach((n, i) => {
        if (matches(n, nrules)) {
          this.visibleNodes[i] = 1;
          this.filteredNodes++;
          if (
            (this.gpu && !this.gpuLost) ||
            this.drawNodeIndices.length < CANVAS_NODES
          )
            this.drawNodeIndices.push(i);
        }
      });
      if (this.selected !== null && this.index.has(this.selected)) {
        const selectedIndex = this.index.get(this.selected);
        if (
          this.visibleNodes[selectedIndex] &&
          !this.drawNodeIndices.includes(selectedIndex)
        ) {
          if (this.drawNodeIndices.length === CANVAS_NODES)
            this.drawNodeIndices.pop();
          this.drawNodeIndices.push(selectedIndex);
        }
      }
      const drawn = new Uint8Array(this.config.nodes.length);
      this.drawnNodeFlags = drawn;
      for (const i of this.drawNodeIndices) drawn[i] = 1;
      this.links.forEach((e, i) => {
        if (
          this.visibleNodes[e.a] &&
          this.visibleNodes[e.b] &&
          matches(e, erules)
        ) {
          this.visibleEdges[i] = 1;
          this.filteredEdges++;
          if (
            drawn[e.a] &&
            drawn[e.b] &&
            ((this.gpu && !this.gpuLost) ||
              this.drawEdgeIndices.length < CANVAS_EDGES)
          )
            this.drawEdgeIndices.push(i);
        }
      });
      this.drawnEdgeFlags = new Uint8Array(this.links.length);
      for (const i of this.drawEdgeIndices) this.drawnEdgeFlags[i] = 1;
      if (
        this.selected !== null &&
        (!this.index.has(this.selected) ||
          !this.visibleNodes[this.index.get(this.selected)])
      ) {
        this.selected = null;
        if (this.clear) this.clear.hidden = true;
        if (this.selectionText)
          this.selectionText.textContent =
            "Select a node to inspect its connections.";
      }
      this.selectedNeighbors = this.neighborSet(this.selected);
      this.spatial = new SpatialIndex(this.positions, this.drawNodeIndices);
      if (this.gpu && !this.gpuLost) this.uploadGPU();
      if (this.counts) {
        let text = coverage(this.config);
        if (options.filters.length)
          text +=
            " · Filtered: " +
            this.filteredNodes.toLocaleString("en-US") +
            " nodes / " +
            this.filteredEdges.toLocaleString("en-US") +
            " edges";
        if (
          this.drawNodeIndices.length < this.filteredNodes ||
          this.drawEdgeIndices.length < this.filteredEdges
        )
          text +=
            " · Canvas fallback draws " +
            this.drawNodeIndices.length.toLocaleString("en-US") +
            "/" +
            this.filteredNodes.toLocaleString("en-US") +
            " nodes and " +
            this.drawEdgeIndices.length.toLocaleString("en-US") +
            "/" +
            this.filteredEdges.toLocaleString("en-US") +
            " edges";
        this.counts.textContent = text;
      }
      this.renderLegend();
    }
    uploadGPU() {
      try {
        this.gpu.upload({
          nodes: this.config.nodes,
          links: this.links,
          positions: this.positions,
          radius: (i) => this.radius(i),
          nodeColor: (i) => this.nodeColor(i),
          nodeActive: (i) => this.related(i),
          nodeVisible: (i) => this.visibleNodes[i] === 1,
          edgeVisible: (i) => this.visibleEdges[i] === 1,
          edgeWidth: (i) => this.edgeWidths[i],
          edgeColor: (i) => this.edgeColors[i],
          edgeAlpha: (i) => this.edgeAlpha(i),
        });
      } catch (error) {
        this.gpu.dispose();
        this.gpu = null;
        this.gpuLost = false;
        this.gpuCanvas.hidden = true;
        this.refreshVisibility();
        if (this.status)
          this.status.textContent = "GPU allocation failed · Canvas fallback";
      }
    }
    renderLegend() {
      this.legendEntries = [];
      if (!this.legend) return;
      this.legend.replaceChildren();
      this.legend.hidden = !this.config.options.legend;
      if (this.legend.hidden) return;
      const rule = this.config.options.node_color;
      if (rule && typeof rule === "object" && rule.scale !== "categorical") {
        let low = Infinity,
          high = -Infinity,
          li = null,
          hi = null;
        for (const i of this.drawNodeIndices) {
          const value = field(this.config.nodes[i], rule.field);
          if (typeof value !== "number" || !Number.isFinite(value)) continue;
          if (value < low) {
            low = value;
            li = i;
          }
          if (value > high) {
            high = value;
            hi = i;
          }
        }
        if (li !== null)
          this.legendEntries.push({
            label: rule.field + ": " + low + " (" + rule.scale + ")",
            color: this.nodeColor(li),
          });
        if (hi !== null && hi !== li)
          this.legendEntries.push({
            label: rule.field + ": " + high,
            color: this.nodeColor(hi),
          });
      } else {
        const seen = new Set();
        for (const i of this.drawNodeIndices) {
          const node = this.config.nodes[i],
            value =
              rule && typeof rule === "object"
                ? field(node, rule.field)
                : node.group;
          if (seen.has(value)) continue;
          seen.add(value);
          if (this.legendEntries.length < 20)
            this.legendEntries.push({
              label: String(
                value === undefined || value === null ? "Missing" : value,
              ),
              color: this.nodeColor(i),
            });
        }
        if (seen.size > 20)
          this.legendEntries.push({
            label: "+" + (seen.size - 20) + " categories",
            color: "#546883",
          });
      }
      for (const entry of this.legendEntries) {
        const label = element("span", "oe-network-legend-item"),
          swatch = element("i");
        swatch.style.backgroundColor = entry.color;
        label.append(swatch, document.createTextNode(entry.label));
        this.legend.append(label);
      }
    }
    setFilters(rules) {
      this.config.options.filters = filters(rules);
      this.refreshVisibility();
      this.schedule();
      return this;
    }
    saveView() {
      return {
        version: 1,
        positions: this.config.nodes.map((n, i) => ({
          id: n.id,
          x: this.positions[i * 2],
          y: this.positions[i * 2 + 1],
          pinned: this.pinned.has(n.id),
          ...(n.identity ? { identity: { ...n.identity } } : {}),
        })),
        transform: { ...this.transform },
        selected: this.selected,
        filters: this.config.options.filters.map((r) => ({
          ...r,
          value: Array.isArray(r.value) ? r.value.slice() : r.value,
        })),
      };
    }
    loadView(raw) {
      const view = validateView(raw, this.config.nodes);
      this.stopWorker();
      for (const p of view.positions) {
        const i = this.index.get(p.id);
        this.positions[i * 2] = p.x;
        this.positions[i * 2 + 1] = p.y;
        if (p.pinned) {
          this.pinned.add(p.id);
          this.manualPins.add(p.id);
        } else {
          this.pinned.delete(p.id);
          this.manualPins.delete(p.id);
        }
      }
      this.transform = { ...view.transform };
      this.config.options.filters = view.filters;
      this.touched = true;
      this.selected = view.selected;
      this.selectedNeighbors = this.neighborSet(view.selected);
      this.refreshVisibility();
      this.select(view.selected);
      this.schedule();
      return this;
    }
    toSpec() {
      const graph = this.config.frames
        ? { ...this.spec.config.network }
        : { ...this.spec.config.network, ...this.config };
      delete graph.title;
      delete graph.options;
      delete graph.frames;
      if (this.config.frames) graph.frames = this.config.frames;
      return {
        ...this.spec,
        sample_n: graph.nodes.length,
        total_n: graph.node_count,
        config: {
          network: graph,
          options: {
            ...this.config.options,
            ...(this.config.frames ? { frame_index: this.frameIndex } : {}),
            view: this.saveView(),
          },
        },
      };
    }
    createTimeline() {
      if (
        !this.config.options.timeline ||
        !this.config.frames ||
        this.config.frames.length < 2
      )
        return;
      const line = element("div", "oe-network-timeline"),
        play = element("button", "", "Play"),
        slider = element("input");
      play.type = "button";
      slider.type = "range";
      slider.min = "0";
      slider.max = String(this.config.frames.length - 1);
      slider.value = "0";
      slider.setAttribute("aria-label", "Network snapshot");
      this.frameLabel = element("span", "", this.config.frames[0].label);
      line.append(play, slider, this.frameLabel);
      this.footer.before(line);
      this.timelineSlider = slider;
      this.listen(slider, "input", () => this.setFrame(Number(slider.value)));
      this.listen(play, "click", () => {
        if (this.timelineTimer) {
          root.clearInterval(this.timelineTimer);
          this.timelineTimer = null;
          play.textContent = "Play";
        } else {
          play.textContent = "Pause";
          this.timelineTimer = root.setInterval(
            () =>
              this.setFrame((this.frameIndex + 1) % this.config.frames.length),
            1000,
          );
        }
      });
    }
    setFrame(index, startLayout = true) {
      if (this.destroyed) return this;
      require(integer(index) &&
        this.config.frames &&
        index < this.config.frames.length, "Unknown network frame.");
      this.stopWorker();
      this.config.nodes.forEach((n, i) =>
        this.positionMemory.set(n.id, {
          x: this.positions[i * 2],
          y: this.positions[i * 2 + 1],
        }),
      );
      this.frameIndex = index;
      this.configureGraph(this.config.frames[index].network);
      this.selected = null;
      this.hovered = null;
      this.hideTooltip();
      this.clear.hidden = true;
      this.selectionText.textContent =
        "Select a node to inspect its connections.";
      this.selectedNeighbors.clear();
      this.refreshVisibility();
      if (this.timelineSlider) this.timelineSlider.value = String(index);
      if (this.frameLabel)
        this.frameLabel.textContent = this.config.frames[index].label;
      if (!this.touched) this.fit();
      else this.schedule();
      if (startLayout) this.startLayout(true);
      return this;
    }

    resize(initial, preserveCamera = false) {
      if (this.destroyed) return;
      const available =
        this.host.clientWidth || this.config.options.width || 700;
      const width = Math.max(
        1,
        Math.round(Math.min(available, this.config.options.width || available)),
      );
      const height = this.config.options.height,
        ratio = Math.min(2, root.devicePixelRatio || 1);
      if (
        !initial &&
        width === this.width &&
        height === this.height &&
        ratio === this.ratio
      )
        return;
      this.width = width;
      this.height = height;
      this.canvas.width = Math.round(this.width * ratio);
      this.canvas.height = Math.round(this.height * ratio);
      this.gpuCanvas.width = this.canvas.width;
      this.gpuCanvas.height = this.canvas.height;
      this.gpuCanvas.style.width = this.width + "px";
      this.gpuCanvas.style.height = this.height + "px";
      this.canvas.style.width = this.width + "px";
      this.canvas.style.height = this.height + "px";
      this.stage.style.height = this.height + "px";
      this.ratio = ratio;
      if (!preserveCamera && (initial || !this.touched)) this.fit();
      else this.schedule();
    }
    retryPresentation(remaining, fitUntouched) {
      if (this.destroyed || this.presentationFrame !== null) return;
      // A native WebView can present the initial Canvas/WebGL layers before
      // the containing panel settles. Retry twice, without a render loop or
      // rebuilding GPU geometry, after the host has had a frame to settle.
      this.presentationFrame = root.requestAnimationFrame(() => {
        this.presentationFrame = null;
        if (this.destroyed) return;
        this.resize(false, true);
        if (fitUntouched && !this.touched && !this.config.options.view)
          this.fit();
        else this.schedule();
        if (remaining > 1) this.retryPresentation(remaining - 1, fitUntouched);
      });
    }
    fit() {
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (let index = 0; index < this.config.nodes.length; index++) {
        const x = this.positions[index * 2],
          y = this.positions[index * 2 + 1],
          radius = this.radius(index);
        minX = Math.min(minX, x - radius);
        maxX = Math.max(maxX, x + radius);
        minY = Math.min(minY, y - radius);
        maxY = Math.max(maxY, y + radius);
      }
      if (!this.config.nodes.length) {
        this.transform = { k: 1, x: this.width / 2, y: this.height / 2 };
      } else {
        const k = Math.max(
          0.02,
          Math.min(
            2.5,
            (this.width - 60) / Math.max(1, maxX - minX),
            (this.height - 60) / Math.max(1, maxY - minY),
          ),
        );
        this.transform = {
          k,
          x: this.width / 2 - ((minX + maxX) / 2) * k,
          y: this.height / 2 - ((minY + maxY) / 2) * k,
        };
      }
      this.schedule();
    }
    zoom(factor, x, y) {
      const old = this.transform.k,
        next = Math.max(0.02, Math.min(12, old * factor));
      this.transform.x = x - ((x - this.transform.x) * next) / old;
      this.transform.y = y - ((y - this.transform.y) * next) / old;
      this.transform.k = next;
      this.touched = true;
      this.hideTooltip();
      this.schedule();
    }
    hit(point) {
      const x = (point.x - this.transform.x) / this.transform.k,
        y = (point.y - this.transform.y) / this.transform.k;
      let found = null,
        distance = Infinity;
      const candidates = this.spatial.query(
        x,
        y,
        this.maxRadius + 5 / this.transform.k,
      );
      if (this.spatial.overflow) {
        this.status.textContent = "Dense view: zoom in or search for a node.";
        return null;
      }
      for (const index of candidates) {
        const dx = x - this.positions[index * 2],
          dy = y - this.positions[index * 2 + 1],
          squared = dx * dx + dy * dy;
        if (
          squared <= (this.radius(index) + 5 / this.transform.k) ** 2 &&
          squared < distance
        ) {
          distance = squared;
          found = this.config.nodes[index].id;
        }
      }
      return found;
    }
    hideTooltip() {
      if (this.tooltipTimer !== null) {
        root.clearTimeout(this.tooltipTimer);
        this.tooltipTimer = null;
      }
      this.tooltip.hidden = true;
    }
    hover(point) {
      const id = this.hit(point);
      this.hovered = id;
      this.hideTooltip();
      if (id === null) this.tooltip.hidden = true;
      else {
        const node = this.config.nodes[this.index.get(id)];
        this.tooltip.textContent =
          node.label + "\nDegree: " + node.degree + " · ID: " + node.id;
        this.tooltip.style.left =
          Math.max(8, Math.min(this.width - 260, point.x + 12)) + "px";
        this.tooltip.style.top =
          Math.max(8, Math.min(this.height - 70, point.y + 12)) + "px";
        this.tooltip.hidden = false;
        this.tooltipTimer = root.setTimeout(() => {
          this.tooltipTimer = null;
          this.tooltip.hidden = true;
        }, 3000);
      }
      this.schedule();
    }
    select(id, focus) {
      if (id !== null && !this.index.has(id)) return;
      this.selected = id;
      if (
        id !== null &&
        this.visibleNodes[this.index.get(id)] &&
        !this.drawnNodeFlags[this.index.get(id)]
      )
        this.refreshVisibility();
      this.selectedNeighbors = this.neighborSet(id);
      this.clear.hidden = id === null;
      if (this.gpu && !this.gpuLost) this.uploadGPU();
      if (id === null)
        this.selectionText.textContent =
          "Select a node to inspect its connections.";
      else {
        const index = this.index.get(id),
          node = this.config.nodes[index];
        this.selectionText.textContent =
          node.label +
          " · Degree: " +
          node.degree +
          " · " +
          this.neighbors.get(id).size +
          " neighbors shown";
        if (focus) {
          this.transform.x =
            this.width / 2 - this.positions[index * 2] * this.transform.k;
          this.transform.y =
            this.height / 2 - this.positions[index * 2 + 1] * this.transform.k;
          this.touched = true;
        }
      }
      this.schedule();
    }
    find() {
      const term = this.search.value.trim().toLocaleLowerCase();
      this.searchResults.replaceChildren();
      this.searchResults.hidden = !term;
      if (!term) return;
      const matches = [];
      for (let i = 0; i < this.config.nodes.length && matches.length < 6; i++) {
        const node = this.config.nodes[i];
        if (
          this.visibleNodes[i] &&
          (node.label.toLocaleLowerCase().includes(term) ||
            String(node.id) === term ||
            (node.identity && node.identity.value.toLocaleLowerCase() === term))
        )
          matches.push(node);
      }
      if (!matches.length)
        this.searchResults.append(
          element("span", "", "No matching node shown."),
        );
      for (const node of matches) {
        const result = element("button", "", node.label || "Node " + node.id);
        result.type = "button";
        result.title = node.label;
        result.dataset.nodeId = String(node.id);
        this.searchResults.append(result);
      }
    }
    startLayout(ignoreSavedView = false) {
      this.stopWorker();
      this.layoutState = { status: "pending" };
      if (this.destroyed) return;
      if (
        !this.config.nodes.length ||
        (!ignoreSavedView && this.config.options.view) ||
        this.config.options.layout === "fixed" ||
        typeof root.Worker !== "function"
      ) {
        this.layoutState = {
          status:
            typeof root.Worker !== "function" &&
            this.config.options.layout !== "fixed" &&
            !this.config.options.view &&
            this.config.nodes.length
              ? "error"
              : "complete",
          reason: "static",
        };
        if (this.status)
          this.status.textContent = this.config.nodes.length
            ? this.gpu
              ? "WebGL2 · Ready"
              : "Canvas · Ready"
            : "No nodes to display";
        return;
      }
      if (root.OpenEconNetworkWorkerSource && !this.workerBlobURL)
        this.workerBlobURL = root.URL.createObjectURL(
          new root.Blob([root.OpenEconNetworkWorkerSource], {
            type: "text/javascript",
          }),
        );
      const chosenURL =
        this.workerBlobURL || root.OpenEconNetworkWorkerURL || workerURL;
      if (!chosenURL) {
        this.layoutState = {
          status: "error",
          reason: "Layout worker is unavailable",
        };
        return;
      }
      try {
        const job = "network-" + ++nextJob;
        if (
          !chosenURL.startsWith("blob:") &&
          new URL(chosenURL).origin !== root.location.origin
        )
          throw new Error("Layout worker must have the same origin");
        const worker = new root.Worker(chosenURL);
        this.worker = worker;
        this.job = job;
        const isCurrent = () =>
          !this.destroyed && this.worker === worker && this.job === job;
        this.status.textContent = "Arranging network…";
        worker.onmessage = (event) => {
          const message = event.data;
          if (!isCurrent() || !message || message.job !== job) return;
          if (message.type === "error") {
            this.stopWorker();
            this.layoutState = {
              status: "error",
              reason: String(
                message.message || message.error || "worker failed",
              ),
            };
            this.status.textContent =
              "Layout error: " +
              String(message.message || message.error || "worker failed");
            return;
          }
          if (
            !["progress", "done"].includes(message.type) ||
            !message.positions ||
            message.positions.length !== this.config.nodes.length * 2
          )
            return;
          for (const value of message.positions)
            if (!Number.isFinite(value)) return;
          const positions =
            message.positions instanceof Float32Array
              ? message.positions
              : new Float32Array(message.positions);
          for (const value of positions) if (!Number.isFinite(value)) return;
          this.positions = positions;
          this.spatial = new SpatialIndex(this.positions, this.drawNodeIndices);
          if (this.gpu && !this.gpuLost) this.gpu.updatePositions(positions);
          if (!this.touched) this.fit();
          else this.schedule();
          if (message.type === "done") {
            this.stopWorker();
            this.layoutState = {
              status: /time|work_limit|cancel/i.test(message.reason || "")
                ? "error"
                : "complete",
              reason: message.reason || "complete",
            };
            this.status.textContent =
              (this.gpu && !this.gpuLost ? "WebGL2" : "Canvas") +
              " · " +
              (message.reason &&
              message.reason !== "complete" &&
              message.reason !== "converged"
                ? "Layout stopped: " + message.reason
                : "Ready");
          }
        };
        worker.onerror = () => {
          if (isCurrent()) {
            this.stopWorker();
            this.layoutState = {
              status: "error",
              reason: "Layout worker could not run",
            };
            this.status.textContent = "Layout error: worker could not run";
          }
        };
        worker.postMessage({
          type: "layout",
          job,
          layout: this.config.options.layout,
          layout_options: this.config.options.layout_options,
          seed:
            this.config.options.seed === undefined
              ? layoutSeed(this.config.nodes, this.config.edges)
              : this.config.options.seed,
          point_size: this.config.options.point_size,
          positions: this.positions,
          nodes: this.config.nodes.map((n, i) =>
            Object.fromEntries(
              Object.entries({
                id: n.id,
                degree: n.degree,
                group: n.group,
                x: this.positions[i * 2],
                y: this.positions[i * 2 + 1],
                fx: this.pinned.has(n.id) ? this.positions[i * 2] : n.fx,
                fy: this.pinned.has(n.id) ? this.positions[i * 2 + 1] : n.fy,
                fixed: this.pinned.has(n.id),
                longitude: n.longitude,
                latitude: n.latitude,
              }).filter(([, v]) => v !== undefined),
            ),
          ),
          edges: this.config.edges.map((e) => ({
            source: e.source,
            target: e.target,
            weight: e.weight,
          })),
        });
        this.workerTimer = root.setTimeout(
          () => {
            if (!isCurrent()) return;
            this.stopWorker();
            this.layoutState = { status: "error", reason: "Layout timeout" };
            if (!this.destroyed)
              this.status.textContent = "Layout timeout · Positions retained";
          },
          (this.config.options.layout_options.time_limit_ms || 30000) + 1500,
        );
      } catch (error) {
        this.stopWorker();
        this.layoutState = {
          status: "error",
          reason: String(error.message || error),
        };
        this.status.textContent = "Static layout";
      }
    }
    stopWorker() {
      const job = this.job;
      this.job = null;
      if (this.workerTimer !== null) {
        root.clearTimeout(this.workerTimer);
        this.workerTimer = null;
      }
      if (this.worker) {
        const worker = this.worker;
        this.worker = null;
        worker.onmessage = null;
        worker.onerror = null;
        try {
          worker.postMessage({ type: "cancel", job });
        } catch (_) {
          /* A crashed worker can already be closed. */
        }
        worker.terminate();
      }
      if (this.workerBlobURL) {
        root.URL.revokeObjectURL(this.workerBlobURL);
        this.workerBlobURL = null;
      }
    }
    schedule() {
      if (this.destroyed || this.frame !== null) return;
      this.frame = root.requestAnimationFrame(() => {
        this.frame = null;
        if (!this.destroyed) this.draw();
      });
    }
    edgePath(context, edge) {
      const x = this.positions[edge.a * 2],
        y = this.positions[edge.a * 2 + 1];
      const tx = this.positions[edge.b * 2],
        ty = this.positions[edge.b * 2 + 1];
      if (edge.a === edge.b) {
        const radius = this.radius(edge.a);
        context.moveTo(x + radius, y);
        context.arc(x, y - radius * 1.6, radius * 1.6, 0.3, Math.PI * 2 - 0.3);
      } else {
        context.moveTo(x, y);
        context.lineTo(tx, ty);
      }
    }
    arrow(context, edge) {
      if (edge.a === edge.b) return;
      const x = this.positions[edge.a * 2],
        y = this.positions[edge.a * 2 + 1],
        tx = this.positions[edge.b * 2],
        ty = this.positions[edge.b * 2 + 1];
      const dx = tx - x,
        dy = ty - y,
        length = Math.hypot(dx, dy);
      if (length < 0.001) return;
      const ux = dx / length,
        uy = dy / length,
        size = Math.max(3, 4 / this.transform.k);
      const endX = tx - ux * (this.radius(edge.b) + 1),
        endY = ty - uy * (this.radius(edge.b) + 1);
      context.moveTo(endX, endY);
      context.lineTo(
        endX - ux * size - uy * size * 0.5,
        endY - uy * size + ux * size * 0.5,
      );
      context.lineTo(
        endX - ux * size + uy * size * 0.5,
        endY - uy * size - ux * size * 0.5,
      );
      context.closePath();
    }
    labelCandidates() {
      const options = this.config.options.labels;
      if (!options.show || this.transform.k < options.min_zoom) return [];
      const priority = [
        ...new Set(
          [this.selected, this.hovered]
            .filter((id) => id !== null && this.index.has(id))
            .map((id) => this.index.get(id)),
        ),
      ];
      if (
        this.config.nodes.length > 60 &&
        this.selected === null &&
        this.transform.k < 1.8
      )
        return priority;
      const others = this.labelOrder.filter(
        (i) =>
          this.drawnNodeFlags[i] &&
          !priority.includes(i) &&
          (this.selected === null || this.related(i)),
      );
      return [...priority, ...others].slice(0, options.max_count);
    }
    labelLayout() {
      // At most 80 rectangles are tested. Screen-space bounds keep captions
      // readable at every zoom level, without allocating per-node DOM elements.
      const result = [],
        transform = this.transform,
        context = this.context;
      const overlaps = (box) =>
        result.some(
          (label) =>
            box.left < label.box.right &&
            box.right > label.box.left &&
            box.top < label.box.bottom &&
            box.bottom > label.box.top,
        );
      context.save();
      context.font = "12px Barlow, system-ui, sans-serif";
      for (const index of this.labelCandidates()) {
        const node = this.config.nodes[index],
          priority = node.id === this.selected || node.id === this.hovered;
        const text = this.nodeLabels[index],
          width = context.measureText(text).width;
        const cx = this.positions[index * 2] * transform.k + transform.x;
        const cy = this.positions[index * 2 + 1] * transform.k + transform.y;
        const radius = this.radius(index) * transform.k;
        if (
          !priority &&
          (width > this.width - 16 ||
            cx < -radius ||
            cx > this.width + radius ||
            cy < 10 ||
            cy > this.height - 10)
        )
          continue;
        const anchors = [
          { x: cx + radius + 5, y: cy },
          { x: cx - radius - 5 - width, y: cy },
        ];
        if (priority)
          anchors.push(
            { x: cx - width / 2, y: cy - radius - 14 },
            { x: cx - width / 2, y: cy + radius + 14 },
          );
        let fallback = null,
          placement = null;
        for (const anchor of anchors) {
          const x = priority
            ? Math.max(8, Math.min(this.width - width - 8, anchor.x))
            : anchor.x;
          const y = priority
            ? Math.max(12, Math.min(this.height - 12, anchor.y))
            : anchor.y;
          if (!priority && (x < 8 || x + width > this.width - 8)) continue;
          const box = {
            left: x - 3,
            right: x + width + 3,
            top: y - 9,
            bottom: y + 9,
          };
          const label = {
            index,
            text,
            x: (x - transform.x) / transform.k,
            y: (y - transform.y) / transform.k,
            box,
          };
          if (fallback === null) fallback = label;
          if (!overlaps(box)) {
            placement = label;
            break;
          }
        }
        // Selected and hovered identities stay exact even when their labels
        // cannot both fit; the full text also remains in selection and tooltip.
        if (placement || (priority && fallback))
          result.push(placement || fallback);
      }
      context.restore();
      return result;
    }
    labelIndices() {
      return this.labelLayout().map((label) => label.index);
    }
    draw() {
      const context = this.context,
        options = this.config.options,
        transform = this.transform;
      context.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
      context.clearRect(0, 0, this.width, this.height);
      const gpu = this.gpu && !this.gpuLost;
      if (gpu)
        this.gpu.draw(
          transform,
          this.width,
          this.height,
          this.ratio,
          options.opacity,
          this.config.directed,
        );
      else {
        context.fillStyle = "#fff";
        context.fillRect(0, 0, this.width, this.height);
      }
      context.save();
      context.translate(transform.x, transform.y);
      context.scale(transform.k, transform.k);
      if (!gpu) {
        const edgeBuckets = new Map();
        for (const i of this.drawEdgeIndices) {
          const edge = this.links[i],
            key =
              this.edgeColors[i] +
              "|" +
              this.edgeWidths[i].toFixed(1) +
              "|" +
              this.edgeAlpha(i);
          let bucket = edgeBuckets.get(key);
          if (!bucket)
            edgeBuckets.set(
              key,
              (bucket = {
                indices: [],
                color: this.edgeColors[i],
                width: this.edgeWidths[i],
                alpha: this.edgeAlpha(i),
              }),
            );
          bucket.indices.push(i);
        }
        for (const bucket of edgeBuckets.values()) {
          context.globalAlpha = options.opacity * bucket.alpha;
          context.strokeStyle = bucket.color;
          context.lineWidth = bucket.width;
          context.beginPath();
          for (const i of bucket.indices) this.edgePath(context, this.links[i]);
          context.stroke();
          if (this.config.directed) {
            context.fillStyle = bucket.color;
            context.beginPath();
            for (const i of bucket.indices) this.arrow(context, this.links[i]);
            context.fill();
          }
        }
        const groups = new Map();
        for (const i of this.drawNodeIndices) {
          const active = this.related(i),
            fill = this.nodeColor(i),
            key = fill + (active ? "a" : "m");
          let group = groups.get(key);
          if (!group) groups.set(key, (group = { indices: [], active, fill }));
          group.indices.push(i);
        }
        for (const group of groups.values()) {
          context.globalAlpha = options.opacity * (group.active ? 1 : 0.14);
          context.fillStyle = group.fill;
          context.beginPath();
          for (const i of group.indices) {
            const x = this.positions[i * 2],
              y = this.positions[i * 2 + 1],
              radius = this.radius(i);
            context.moveTo(x + radius, y);
            context.arc(x, y, radius, 0, Math.PI * 2);
          }
          context.fill();
        }
      }
      for (const id of [this.selected, this.hovered])
        if (id !== null && this.index.has(id)) {
          const i = this.index.get(id);
          context.globalAlpha = 1;
          context.lineWidth = 2 / transform.k;
          context.strokeStyle = "#163d68";
          context.beginPath();
          context.arc(
            this.positions[i * 2],
            this.positions[i * 2 + 1],
            this.radius(i) + 3 / transform.k,
            0,
            Math.PI * 2,
          );
          context.stroke();
        }
      context.globalAlpha = 1;
      context.font = 12 / transform.k + "px Barlow, system-ui, sans-serif";
      context.textBaseline = "middle";
      context.fillStyle = "#14263d";
      context.lineJoin = "round";
      for (const label of this.labelLayout()) {
        context.strokeStyle = "#fff";
        context.lineWidth = 3 / transform.k;
        context.strokeText(label.text, label.x, label.y);
        context.fillText(label.text, label.x, label.y);
      }
      for (const note of options.annotations) {
        let x = note.x,
          y = note.y;
        if (note.node_id !== undefined) {
          const i = this.index.get(note.node_id);
          if (i === undefined || !this.visibleNodes[i]) continue;
          x = this.positions[i * 2] + this.radius(i) + 8 / transform.k;
          y = this.positions[i * 2 + 1];
        }
        context.fillStyle = note.color || "#14263d";
        context.strokeStyle = "#fff";
        context.lineWidth = 3 / transform.k;
        context.strokeText(note.text, x, y);
        context.fillText(note.text, x, y);
      }
      context.restore();
      context.globalAlpha = 1;
    }
    exportScene() {
      require(this.drawNodeIndices.length + this.drawEdgeIndices.length <=
        VECTOR_PRIMITIVES, "Vector export exceeds 200,000 primitives. Use PNG or filter the graph.");
      const transform = { ...this.transform },
        nodes = this.drawNodeIndices.map((i) => {
          const n = this.config.nodes[i];
          return {
            id: n.id,
            identity: n.identity,
            label: this.nodeLabels[i],
            x: this.positions[i * 2],
            y: this.positions[i * 2 + 1],
            radius: this.radius(i),
            color: this.nodeColor(i),
            highlighted: n.id === this.selected || n.id === this.hovered,
            opacity: this.config.options.opacity * (this.related(i) ? 1 : 0.14),
          };
        }),
        edges = this.drawEdgeIndices.map((i) => {
          const e = this.links[i];
          return {
            source: e.source,
            target: e.target,
            x: this.positions[e.a * 2],
            y: this.positions[e.a * 2 + 1],
            tx: this.positions[e.b * 2],
            ty: this.positions[e.b * 2 + 1],
            width: this.edgeWidths[i],
            color: this.edgeColors[i],
            opacity: this.config.options.opacity * this.edgeAlpha(i),
          };
        });
      return {
        width: this.width,
        height: this.height,
        title: this.config.title,
        coverage: this.counts.textContent,
        transform,
        nodes,
        edges,
        labels: this.labelLayout(),
        annotations: this.config.options.annotations
          .filter(
            (note) =>
              note.node_id === undefined ||
              (this.index.has(note.node_id) &&
                this.visibleNodes[this.index.get(note.node_id)]),
          )
          .map((note) => {
            if (note.node_id === undefined) return { ...note };
            const i = this.index.get(note.node_id);
            return {
              ...note,
              x: this.positions[i * 2] + this.radius(i) + 8 / this.transform.k,
              y: this.positions[i * 2 + 1],
            };
          }),
        legend: this.legendEntries.map((entry) => ({ ...entry })),
        directed: this.config.directed,
      };
    }
    exportLayout() {
      this.context.save();
      this.context.font = "600 18px Barlow, system-ui, sans-serif";
      const titleLines = wrappedText(
        this.context,
        this.config.title || "Network",
        Math.max(1, this.width - 36),
      );
      this.context.font = "11px Barlow, system-ui, sans-serif";
      const footerLines = wrappedText(
        this.context,
        this.counts.textContent,
        Math.max(1, this.width - 36),
      );
      const legendLines = [];
      for (const entry of this.legendEntries)
        wrappedText(
          this.context,
          entry.label,
          Math.max(1, this.width - 50),
        ).forEach((text, i) =>
          legendLines.push({ text, color: entry.color, swatch: i === 0 }),
        );
      this.context.restore();
      const top = 24 + titleLines.length * 22;
      return {
        titleLines,
        footerLines,
        legendLines,
        top,
        height:
          top +
          this.height +
          24 +
          (footerLines.length + legendLines.length) * 15,
      };
    }
    snapshot() {
      require(!this.destroyed, "The network chart is no longer mounted.");
      require(this.drawNodeIndices.length + this.drawEdgeIndices.length <=
        VECTOR_PRIMITIVES, "Vector export exceeds 200,000 primitives. Use PNG or filter the graph.");
      const namespace = "http://www.w3.org/2000/svg";
      const svgNode = (tag, attributes, text) => {
        const node = document.createElementNS(namespace, tag);
        for (const [key, value] of Object.entries(attributes || {}))
          node.setAttribute(key, String(value));
        if (text !== undefined) node.textContent = text;
        return node;
      };
      const layout = this.exportLayout();
      const svg = svgNode("svg", {
        xmlns: namespace,
        width: this.width,
        height: layout.height,
        viewBox: "0 0 " + this.width + " " + layout.height,
        role: "img",
      });
      if (root.OpenEconNetworkFont) {
        const font = svgNode(
          "style",
          {},
          "@font-face{font-family:Barlow;src:url(data:font/ttf;base64," +
            root.OpenEconNetworkFont.base64 +
            ") format('truetype');font-weight:100 900;}",
        );
        svg.append(font);
      }
      svg.append(
        svgNode("title", {}, this.config.title || "Network"),
        svgNode("desc", {}, this.counts.textContent),
        svgNode("rect", { width: "100%", height: "100%", fill: "#fff" }),
      );
      layout.titleLines.forEach((line, index) =>
        svg.append(
          svgNode(
            "text",
            {
              x: 18,
              y: 28 + index * 22,
              fill: "#14263d",
              "font-family": "Barlow,system-ui,sans-serif",
              "font-size": 18,
              "font-weight": 600,
            },
            line,
          ),
        ),
      );
      const viewport = svgNode("svg", {
        x: 0,
        y: layout.top,
        width: this.width,
        height: this.height,
        viewBox: "0 0 " + this.width + " " + this.height,
        overflow: "hidden",
      });
      const drawing = svgNode("g", {
        transform:
          "translate(" +
          this.transform.x +
          " " +
          this.transform.y +
          ") scale(" +
          this.transform.k +
          ")",
      });
      for (const edgeIndex of this.drawEdgeIndices) {
        const edge = this.links[edgeIndex];
        const x = this.positions[edge.a * 2],
          y = this.positions[edge.a * 2 + 1],
          tx = this.positions[edge.b * 2],
          ty = this.positions[edge.b * 2 + 1];
        const radius = this.radius(edge.a) * 1.6;
        const length = Math.hypot(tx - x, ty - y),
          endRadius =
            this.config.directed && length > 0 ? this.radius(edge.b) + 1 : 0;
        const endX = length > 0 ? tx - ((tx - x) / length) * endRadius : tx;
        const endY = length > 0 ? ty - ((ty - y) / length) * endRadius : ty;
        const path =
          edge.a === edge.b
            ? "M" +
              (x + radius) +
              "," +
              (y - radius) +
              " a" +
              radius +
              "," +
              radius +
              " 0 1,0 " +
              -2 * radius +
              ",0 a" +
              radius +
              "," +
              radius +
              " 0 1,0 " +
              2 * radius +
              ",0"
            : "M" + x + "," + y + " L" + endX + "," + endY;
        const related =
          this.selected === null ||
          edge.source === this.selected ||
          edge.target === this.selected;
        const line = svgNode("path", {
          d: path,
          fill: "none",
          stroke: this.edgeColors[edgeIndex],
          "stroke-width": this.edgeWidths[edgeIndex],
          opacity:
            this.config.options.opacity *
            (this.selected === null ? 0.28 : related ? 0.7 : 0.06),
        });

        line.append(
          svgNode(
            "title",
            {},
            edge.source + " → " + edge.target + " · Weight: " + edge.weight,
          ),
        );
        drawing.append(line);
        if (this.config.directed && edge.a !== edge.b && length > 0) {
          const ux = (tx - x) / length,
            uy = (ty - y) / length,
            size = Math.max(3, 4 / this.transform.k);
          drawing.append(
            svgNode("polygon", {
              points: [
                endX + "," + endY,
                endX -
                  ux * size -
                  uy * size * 0.5 +
                  "," +
                  (endY - uy * size + ux * size * 0.5),
                endX -
                  ux * size +
                  uy * size * 0.5 +
                  "," +
                  (endY - uy * size - ux * size * 0.5),
              ].join(" "),
              fill: this.edgeColors[edgeIndex],
              opacity: this.config.options.opacity * this.edgeAlpha(edgeIndex),
            }),
          );
        }
      }
      this.drawNodeIndices.forEach((index) => {
        const node = this.config.nodes[index];
        const circle = svgNode("circle", {
          cx: this.positions[index * 2],
          cy: this.positions[index * 2 + 1],
          r: this.radius(index),
          fill: this.nodeColor(index),
          opacity:
            this.config.options.opacity * (this.related(index) ? 1 : 0.14),
        });
        circle.append(
          svgNode(
            "title",
            {},
            node.label + " · ID: " + node.id + " · Degree: " + node.degree,
          ),
        );
        drawing.append(circle);
      });
      for (const id of [this.selected, this.hovered])
        if (id !== null && this.index.has(id)) {
          const i = this.index.get(id);
          if (this.drawnNodeFlags[i])
            drawing.append(
              svgNode("circle", {
                cx: this.positions[i * 2],
                cy: this.positions[i * 2 + 1],
                r: this.radius(i) + 3 / this.transform.k,
                fill: "none",
                stroke: "#163d68",
                "stroke-width": 2 / this.transform.k,
              }),
            );
        }
      for (const label of this.labelLayout())
        drawing.append(
          svgNode(
            "text",
            {
              x: label.x,
              y: label.y,
              fill: "#14263d",
              "font-family": "Barlow,system-ui,sans-serif",
              "font-size": 12 / this.transform.k,
              "dominant-baseline": "central",
              stroke: "#fff",
              "stroke-width": 3 / this.transform.k,
              "paint-order": "stroke",
            },
            label.text,
          ),
        );
      for (const note of this.config.options.annotations) {
        let x = note.x,
          y = note.y;
        if (note.node_id !== undefined) {
          const i = this.index.get(note.node_id);
          if (i === undefined || !this.visibleNodes[i]) continue;
          x = this.positions[i * 2] + this.radius(i) + 8 / this.transform.k;
          y = this.positions[i * 2 + 1];
        }
        drawing.append(
          svgNode(
            "text",
            {
              x,
              y,
              fill: note.color || "#14263d",
              "font-family": "Barlow,system-ui,sans-serif",
              "font-size": 12 / this.transform.k,
            },
            note.text,
          ),
        );
      }
      viewport.append(drawing);
      svg.append(viewport);
      layout.footerLines.forEach((line, index) =>
        svg.append(
          svgNode(
            "text",
            {
              x: 18,
              y: this.height + layout.top + 19 + index * 15,
              fill: "#546883",
              "font-family": "Barlow,system-ui,sans-serif",
              "font-size": 11,
            },
            line,
          ),
        ),
      );
      layout.legendLines.forEach((entry, index) => {
        const y =
          this.height +
          layout.top +
          19 +
          (layout.footerLines.length + index) * 15;
        if (entry.swatch)
          svg.append(
            svgNode("rect", {
              x: 18,
              y: y - 7,
              width: 7,
              height: 7,
              fill: entry.color,
            }),
          );
        svg.append(
          svgNode(
            "text",
            {
              x: 31,
              y,
              fill: "#14263d",
              "font-family": "Barlow,system-ui,sans-serif",
              "font-size": 11,
            },
            entry.text,
          ),
        );
      });
      return svg;
    }
    save(blob, extension) {
      if (this.destroyed) return false;
      require(blob && blob.size > 0, "The network export is empty.");
      const url = root.URL.createObjectURL(blob),
        anchor = element("a");
      anchor.href = url;
      anchor.download = "network." + extension;
      anchor.hidden = true;
      try {
        document.body.append(anchor);
        anchor.click();
      } catch (error) {
        root.URL.revokeObjectURL(url);
        throw error;
      } finally {
        anchor.remove();
      }
      // WebKit can consume the blob after its download navigation has been queued.
      // Match the shared chart renderer's lifetime; chart teardown must not abort an initiated download.
      root.setTimeout(() => root.URL.revokeObjectURL(url), 30000);
      return true;
    }
    export(format) {
      if (this.destroyed || (format === "png" && this.pngExport)) return;
      try {
        if (format === "pdf") {
          require(root.OpenEconNetworkPDF, "The PDF exporter is unavailable.");
          root.OpenEconNetworkPDF.download(this);
          return;
        }
        if (format === "view" || format === "graph") {
          const data = format === "view" ? this.saveView() : this.toSpec();
          this.save(
            new root.Blob([JSON.stringify(data)], { type: "application/json" }),
            format + ".json",
          );
          return;
        }
        if (format === "svg")
          this.save(
            new root.Blob(
              [new root.XMLSerializer().serializeToString(this.snapshot())],
              { type: "image/svg+xml;charset=utf-8" },
            ),
            "svg",
          );
        else if (format === "png") {
          this.draw();
          const exportCanvas = document.createElement("canvas"),
            scale = this.ratio || 1,
            layout = this.exportLayout();
          exportCanvas.width = this.canvas.width;
          exportCanvas.height = Math.round(layout.height * scale);
          const context = exportCanvas.getContext("2d");
          require(context, "PNG export needs a Canvas drawing context.");
          context.scale(scale, scale);
          context.fillStyle = "#fff";
          context.fillRect(0, 0, this.width, layout.height);
          context.fillStyle = "#14263d";
          context.font = "600 18px Barlow, system-ui, sans-serif";
          layout.titleLines.forEach((line, index) =>
            context.fillText(line, 18, 29 + index * 22),
          );
          if (this.gpu && !this.gpuLost)
            context.drawImage(
              this.gpuCanvas,
              0,
              layout.top,
              this.width,
              this.height,
            );
          context.drawImage(
            this.canvas,
            0,
            layout.top,
            this.width,
            this.height,
          );
          context.fillStyle = "#546883";
          context.font = "11px Barlow, system-ui, sans-serif";
          layout.footerLines.forEach((line, index) =>
            context.fillText(
              line,
              18,
              this.height + layout.top + 19 + index * 15,
            ),
          );
          layout.legendLines.forEach((entry, index) => {
            const y =
              this.height +
              layout.top +
              19 +
              (layout.footerLines.length + index) * 15;
            if (entry.swatch) {
              context.fillStyle = entry.color;
              context.fillRect(18, y - 7, 7, 7);
            }
            context.fillStyle = "#14263d";
            context.fillText(entry.text, 31, y);
          });
          const previousStatus = this.status.textContent;
          this.pngExport = exportCanvas;
          this.status.textContent = "Preparing PNG…";
          exportCanvas.toBlob((blob) => {
            const active = this.pngExport === exportCanvas;
            if (active) this.pngExport = null;
            try {
              if (this.destroyed || !active) return;
              require(blob &&
                blob.size > 0, "The PNG image could not be created.");
              this.save(blob, "png");
              if (this.status.textContent === "Preparing PNG…")
                this.status.textContent = previousStatus;
            } catch (error) {
              if (!this.destroyed)
                this.status.textContent = "Could not export the network.";
            } finally {
              exportCanvas.width = 0;
              exportCanvas.height = 0;
            }
          }, "image/png");
        } else {
          const nodeFormat = format === "nodes",
            items = nodeFormat
              ? this.drawNodeIndices.map((i) => this.config.nodes[i])
              : this.drawEdgeIndices.map((i) => this.config.edges[i]);
          const attrs = Array.from(
            new Set(items.flatMap((item) => Object.keys(item.attrs || {}))),
          ).sort();
          const headers = nodeFormat
            ? [
                "id",
                "identity_type",
                "identity_value",
                "identity_json",
                "label",
                "degree",
                "group",
                "x",
                "y",
                "pinned",
                "attrs_json",
                ...attrs,
              ]
            : [
                "source",
                "target",
                "source_identity_type",
                "source_identity_value",
                "source_identity_json",
                "target_identity_type",
                "target_identity_value",
                "target_identity_json",
                "weight",
                "attrs_json",
                ...attrs,
              ];
          const rows = items.map((item) => {
            if (nodeFormat) {
              const i = this.index.get(item.id);
              return [
                item.id,
                item.identity?.type || "integer",
                item.identity?.value || String(item.id),
                JSON.stringify(
                  item.identity || { type: "integer", value: String(item.id) },
                ),
                item.label,
                item.degree,
                item.group,
                this.positions[i * 2],
                this.positions[i * 2 + 1],
                this.pinned.has(item.id),
                JSON.stringify(item.attrs || {}),
                ...attrs.map((key) => item.attrs[key] ?? ""),
              ];
            }
            const source = this.config.nodes[this.index.get(item.source)],
              target = this.config.nodes[this.index.get(item.target)];
            return [
              item.source,
              item.target,
              source.identity?.type || "integer",
              source.identity?.value || String(source.id),
              JSON.stringify(
                source.identity || {
                  type: "integer",
                  value: String(source.id),
                },
              ),
              target.identity?.type || "integer",
              target.identity?.value || String(target.id),
              JSON.stringify(
                target.identity || {
                  type: "integer",
                  value: String(target.id),
                },
              ),
              item.weight,
              JSON.stringify(item.attrs || {}),
              ...attrs.map((key) => item.attrs[key] ?? ""),
            ];
          });
          const metadata = [
            ["# " + this.counts.textContent],
            ["# Rendered filtered network rows only"],
          ];
          const text = [...metadata, headers, ...rows]
            .map((row) => row.map(csvCell).join(","))
            .join("\r\n");
          this.save(
            new root.Blob([text], { type: "text/csv;charset=utf-8" }),
            format + ".csv",
          );
        }
      } catch (error) {
        if (format === "png" && this.pngExport) {
          this.pngExport.width = 0;
          this.pngExport.height = 0;
          this.pngExport = null;
        }
        this.status.textContent =
          error && error.message
            ? error.message
            : "Could not export the network.";
      }
    }
    expand() {
      if (this.dialog) return;
      const dialog = element("dialog", "oe-network-dialog"),
        heading = element("header");
      const close = button("close", "Close expanded network", "close"),
        expandedHost = element("div", "oe-network-expanded");
      heading.append(
        element("span", "", this.config.title || "Network"),
        close,
      );
      dialog.append(heading, expandedHost);
      document.body.append(dialog);
      this.dialog = dialog;
      const closeDialog = () => {
        if (typeof dialog.close === "function") dialog.close();
        else dialog.dispatchEvent(new root.Event("close"));
      };
      this.listen(close, "click", closeDialog);
      this.listen(dialog, "click", (event) => {
        if (event.target === dialog) closeDialog();
      });
      this.listen(dialog, "close", () => {
        unmount(expandedHost);
        dialog.remove();
        this.dialog = null;
        if (!this.destroyed)
          this.toolbar.querySelector('[data-action="expand"]').focus();
      });
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
      const expandedConfig = {
        ...this.config,
        options: {
          ...this.config.options,
          width: undefined,
          height: Math.max(
            240,
            Math.min(1000, (root.innerHeight || 800) - 230),
          ),
        },
      };
      try {
        const chart = new NetworkChart(expandedHost, this.spec, expandedConfig);
        chart.loadView(this.saveView());
        instances.set(expandedHost, chart);
        chart.toolbar.querySelector('[data-action="expand"]').disabled = true;
      } catch (error) {
        expandedHost.textContent = "The expanded network could not be opened.";
      }
    }
    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      this.stopWorker();
      this.hideTooltip();
      if (this.timelineTimer) root.clearInterval(this.timelineTimer);
      if (this.gpu) this.gpu.dispose();
      if (this.pngExport) {
        this.pngExport.width = 0;
        this.pngExport.height = 0;
        this.pngExport = null;
      }
      if (this.frame !== null) {
        root.cancelAnimationFrame(this.frame);
        this.frame = null;
      }
      if (this.presentationFrame !== null) {
        root.cancelAnimationFrame(this.presentationFrame);
        this.presentationFrame = null;
      }
      if (this.observer) this.observer.disconnect();
      if (this.dialog) {
        const expandedHost = this.dialog.querySelector(".oe-network-expanded");
        if (expandedHost) unmount(expandedHost);
        this.dialog.remove();
        this.dialog = null;
      }
      this.abort.abort();
      this.host.replaceChildren();
      this.host.classList.remove("oe-network");
      if (this.host.dataset.chartType === "network")
        delete this.host.dataset.chartType;
    }
  }
  function sweep() {
    for (const [host, chart] of instances) {
      if (host.isConnected) chart.everConnected = true;
      else if (chart.everConnected) unmount(host);
    }
  }
  function observeRemovals() {
    if (removalObserver || !root.MutationObserver || !document.body) return;
    removalObserver = new root.MutationObserver(() => {
      if (sweepFrame !== null) return;
      sweepFrame = root.requestAnimationFrame(() => {
        sweepFrame = null;
        sweep();
      });
    });
    removalObserver.observe(document.body, { childList: true, subtree: true });
  }
  function unmount(host) {
    const chart = instances.get(host);
    if (!chart) return;
    instances.delete(host);
    chart.destroy();
    if (!instances.size) {
      if (removalObserver) {
        removalObserver.disconnect();
        removalObserver = null;
      }
      if (sweepFrame !== null) {
        root.cancelAnimationFrame(sweepFrame);
        sweepFrame = null;
      }
    }
  }
  async function mount(host, spec) {
    require(host &&
      typeof host.replaceChildren ===
        "function", "A chart host element is required.");
    const config = normalize(spec);
    unmount(host);
    let chart;
    try {
      chart = new NetworkChart(host, spec, config);
      instances.set(host, chart);
      observeRemovals();
      return chart;
    } catch (error) {
      if (chart) chart.destroy();
      host.replaceChildren();
      host.classList.remove("oe-network");
      delete host.dataset.chartType;
      throw error;
    }
  }
  function snapshot(host) {
    const chart = instances.get(host);
    require(chart, "No network chart is mounted in this host.");
    return chart.snapshot();
  }
  root.OpenEconNetworkCharts = {
    mount,
    render: mount,
    unmount,
    destroy: unmount,
    snapshot,
    normalize,
    validate: normalize,
    get: (host) => instances.get(host),
    sweep,
    initialPositions,
    layoutSeed,
    csvCell,
    coverage,
    saveView: (host) => {
      const chart = instances.get(host);
      require(chart, "No chart is mounted.");
      return chart.saveView();
    },
    loadView: (host, view) => {
      const chart = instances.get(host);
      require(chart, "No chart is mounted.");
      return chart.loadView(view);
    },
    toSpec: (host) => {
      const chart = instances.get(host);
      require(chart, "No chart is mounted.");
      return chart.toSpec();
    },
    setFilters: (host, rules) => {
      const chart = instances.get(host);
      require(chart, "No chart is mounted.");
      return chart.setFilters(rules);
    },
    validateView,
    SpatialIndex,
    MAX_NODES,
    MAX_EDGES,
    CANVAS_NODES,
    CANVAS_EDGES,
    VECTOR_PRIMITIVES,
    COLORS,
  };
  if (typeof module !== "undefined" && module.exports)
    module.exports = root.OpenEconNetworkCharts;
})(typeof window !== "undefined" ? window : globalThis);
