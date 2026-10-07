/* Native vector PDF1.7 writer. ISO32000-1:2008, sections8.4 and9.7/9.10.
 * No rasterization, remote service, executable PDF action or runtime dependency.
 * Barlow is embedded under SIL OFL1.1; see BARLOW-OFL.txt.
 */
(function (root) {
  "use strict";
  const MAX_PRIMITIVES = 200000, MAX_BYTES = 128 * 1024 * 1024;
  const encode = text => new TextEncoder().encode(text);
  const num = value => {
    if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > 1e9) throw new TypeError("Invalid PDF coordinate.");
    return Number(value.toFixed(5)).toString();
  };
  function hex(value, width) { return value.toString(16).padStart(width, "0").toUpperCase(); }
  function rgb(value) {
    if (!/^#[\da-f]{3}(?:[\da-f]{3})?$/i.test(value)) throw new TypeError("Invalid PDF color.");
    if (value.length === 4) value = "#" + [...value.slice(1)].map(x => x + x).join("");
    return [1, 3, 5].map(i => num(parseInt(value.slice(i, i + 2), 16) / 255)).join(" ");
  }
  let cachedFont = null;
  function font() {
    if (cachedFont) return cachedFont;
    if (!root.OpenEconNetworkFont?.base64) throw new Error("The bundled PDF font is missing.");
    const raw = root.atob(root.OpenEconNetworkFont.base64), data = Uint8Array.from(raw, c => c.charCodeAt(0));
    const view = new DataView(data.buffer), table = {};
    const u16 = p => view.getUint16(p), i16 = p => view.getInt16(p), u32 = p => view.getUint32(p);
    for (let i = 0; i < u16(4); i++) {
      const p = 12 + 16 * i, name = String.fromCharCode(...data.subarray(p, p + 4));
      table[name] = u32(p + 8);
    }
    if (u32(0) !== 0x10000 || table["CFF "] !== undefined) throw new Error("PDF FontFile2 requires a TrueType outline font.");
    for (const name of ["head", "hhea", "hmtx", "cmap", "maxp", "glyf", "loca"]) if (table[name] === undefined) throw new Error("Invalid PDF font tables.");
    const em = u16(table.head + 18), metrics = u16(table.hhea + 34), glyphCount = u16(table.maxp + 4);
    const cmap = table.cmap; let sub = null, format = null;
    for (let i = 0; i < u16(cmap + 2); i++) {
      const record = cmap + 4 + 8 * i, platform = u16(record), encoding = u16(record + 2), candidate = cmap + u32(record + 4), f = u16(candidate);
      if ((platform === 0 || platform === 3 && [1, 10].includes(encoding)) && [4, 12].includes(f)) {
        if (sub === null || f === 12) { sub = candidate; format = f; }
      }
    }
    if (sub === null || !em || !metrics || !glyphCount) throw new Error("Invalid Unicode PDF font.");
    function glyph(code) {
      if (format === 12) {
        let lo = 0, hi = u32(sub + 12) - 1;
        while (lo <= hi) {
          const mid = (lo + hi) >>> 1, p = sub + 16 + 12 * mid, first = u32(p), last = u32(p + 4);
          if (code < first) hi = mid - 1;
          else if (code > last) lo = mid + 1;
          else return u32(p + 8) + code - first;
        }
        return 0;
      }
      if (code > 65535) return 0;
      const count = u16(sub + 6) / 2, ends = sub + 14, starts = ends + 2 * count + 2, deltas = starts + 2 * count, offsets = deltas + 2 * count;
      let lo = 0, hi = count - 1;
      while (lo < hi) { const mid = (lo + hi) >>> 1; if (u16(ends + 2 * mid) < code) lo = mid + 1; else hi = mid; }
      if (code < u16(starts + 2 * lo) || code > u16(ends + 2 * lo)) return 0;
      const offset = u16(offsets + 2 * lo), delta = i16(deltas + 2 * lo);
      if (!offset) return (code + delta) & 65535;
      const result = u16(offsets + 2 * lo + offset + 2 * (code - u16(starts + 2 * lo)));
      return result ? (result + delta) & 65535 : 0;
    }
    cachedFont = { data, em, glyphCount, glyph,
      width: gid => u16(table.hmtx + 4 * Math.min(gid, metrics - 1)) * 1000 / em,
      bbox: [36, 38, 40, 42].map(offset => i16(table.head + offset) * 1000 / em),
      ascent: i16(table.hhea + 4) * 1000 / em, descent: i16(table.hhea + 6) * 1000 / em };
    return cachedFont;
  }
  function bytes(scene) {
    if (!scene || !Array.isArray(scene.nodes) || !Array.isArray(scene.edges)
        || scene.nodes.length + scene.edges.length > MAX_PRIMITIVES) throw new Error("Vector export supports at most200,000 nodes and edges; use PNG for larger views.");
    const f = font(), width = scene.width, height = scene.height;
    if (!(width >= 1 && width <= 2400 && height >= 1 && height <= 1600)) throw new Error("Invalid PDF dimensions.");
    function wrapped(value, size, available) {
      const lines = []; let line = "", measured = 0;
      for (const character of String(value)) {
        if (character === "\n") { lines.push(line); line = ""; measured = 0; continue; }
        const advance = f.width(f.glyph(character.codePointAt(0))) * size / 1000;
        if (line && measured + advance > available) { lines.push(line); line = ""; measured = 0; }
        line += character; measured += advance;
      }
      lines.push(line); return lines;
    }
    const titleLines = wrapped(scene.title || "Network", 18, Math.max(1, width - 36));
    const footerLines = wrapped(scene.coverage || "", 11, Math.max(1, width - 36));
    const legendLines = (scene.legend || []).flatMap(entry =>
      wrapped(entry.label, 11, Math.max(1, width - 50)).map((label, index) =>
        ({label, color: entry.color, swatch: index === 0})));
    const top = 24 + titleLines.length * 22, pageHeight = height + top + 24 + 15 * (footerLines.length + legendLines.length);
    const transform = scene.transform;
    if (!transform || !(transform.k > 0 && transform.k <= 12)) throw new Error("Invalid PDF viewport.");
    const characters = new Map(), glyphs = [0], widths = [0], commands = [], alphas = new Map();
    function text(value, x, y, size, color, halo = false) {
      if (typeof value !== "string" || value.length > 16384) throw new Error("Invalid PDF text.");
      let codes = "";
      for (const character of value) {
        if (!characters.has(character)) {
          const gid = f.glyph(character.codePointAt(0));
          if (!gid || gid >= f.glyphCount) throw new Error("The embedded PDF font cannot render U+" + hex(character.codePointAt(0), 4) + "; export SVG/PNG for this script.");
          if (characters.size >= 65534) throw new Error("PDF character budget exceeded.");
          const cid = characters.size + 1; characters.set(character, cid); glyphs.push(gid); widths.push(f.width(gid));
        }
        codes += hex(characters.get(character), 4);
      }
      const placement = " /F1 " + num(size) + " Tf 1 0 0 -1 " + num(x) + " " + num(y) + " Tm <" + codes + "> Tj ET Q";
      // Stroke first, then fill: PDF's combined text mode would paint the
      // white outline over the small dark glyphs rather than behind them.
      if (halo) commands.push("q 1 1 1 RG " + num(3 / transform.k) + " w BT 1 Tr" + placement);
      commands.push("q " + rgb(color || "#14263d") + " rg BT 0 Tr" + placement);
    }
    function alpha(value) {
      if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("Invalid PDF opacity.");
      const level = Math.round(value * 100);
      if (!alphas.has(level)) alphas.set(level, "A" + level);
      return "/" + alphas.get(level) + " gs";
    }
    function circle(x, y, radius) {
      const c = radius * 0.55228474983;
      return [num(x + radius), num(y), "m", num(x + radius), num(y + c), num(x + c), num(y + radius), num(x), num(y + radius), "c", num(x - c), num(y + radius), num(x - radius), num(y + c), num(x - radius), num(y), "c", num(x - radius), num(y - c), num(x - c), num(y - radius), num(x), num(y - radius), "c", num(x + c), num(y - radius), num(x + radius), num(y - c), num(x + radius), num(y), "c"].join(" ");
    }
    commands.push("q 1 0 0 -1 0 " + num(pageHeight) + " cm 1 1 1 rg 0 0 " + num(width) + " " + num(pageHeight) + " re f");
    titleLines.forEach((line, i) => text(line, 18, 28 + 22 * i, 18));
    commands.push("q 0 " + num(top) + " " + num(width) + " " + num(height) + " re W n " + [transform.k, 0, 0, transform.k, transform.x, top + transform.y].map(num).join(" ") + " cm");
    const nodes = new Map(scene.nodes.map(node => [node.id, node]));
    for (const edge of scene.edges) {
      commands.push("q " + alpha(edge.opacity ?? 0.3) + " " + rgb(edge.color || "#163d68") + " RG " + num(edge.width || 1) + " w");
      if (edge.source === edge.target) {
        const radius = (nodes.get(edge.source)?.radius || 4) * 1.6;
        commands.push(circle(edge.x, edge.y - radius, radius) + " S");
      } else {
        const dx = edge.tx - edge.x, dy = edge.ty - edge.y, length = Math.hypot(dx, dy), r = scene.directed ? (nodes.get(edge.target)?.radius || 4) + 1 : 0;
        const tx = length > 0 ? edge.tx - dx / length * r : edge.tx, ty = length > 0 ? edge.ty - dy / length * r : edge.ty;
        commands.push([edge.x, edge.y].map(num).join(" ") + " m " + [tx, ty].map(num).join(" ") + " l S");
        if (scene.directed && length > 0.001) {
          const ux = dx / length, uy = dy / length, size = Math.max(3, 4 / transform.k);
          commands.push(rgb(edge.color || "#163d68") + " rg " + [tx, ty].map(num).join(" ") + " m " + [tx - ux * size - uy * size * 0.5, ty - uy * size + ux * size * 0.5].map(num).join(" ") + " l " + [tx - ux * size + uy * size * 0.5, ty - uy * size - ux * size * 0.5].map(num).join(" ") + " l h f");
        }
      }
      commands.push("Q");
    }
    for (const node of scene.nodes) {
      commands.push("q " + alpha(node.opacity ?? 1) + " " + rgb(node.color || "#163d68") + " rg " + circle(node.x, node.y, node.radius) + " f Q");
      if (node.highlighted) commands.push("q " + rgb("#163d68") + " RG " + num(2 / transform.k) + " w " + circle(node.x, node.y, node.radius + 3 / transform.k) + " S Q");
    }
    for (const label of scene.labels || []) text(label.text, label.x, label.y + 4 / transform.k, 12 / transform.k, undefined, true);
    for (const annotation of scene.annotations || []) text(annotation.text, annotation.x, annotation.y, (annotation.font_size || 12) / transform.k, annotation.color);
    commands.push("Q");
    footerLines.forEach((line, i) => text(line, 18, height + top + 19 + 15 * i, 11, "#546883"));
    legendLines.forEach((entry, i) => {
      const y = height + top + 19 + 15 * (footerLines.length + i);
      if (entry.swatch) commands.push(rgb(entry.color) + " rg 18 " + num(y - 8) + " 8 8 re f");
      text(entry.label, 32, y, 11, "#546883");
    });
    commands.push("Q");
    const objects = [null], reserve = () => (objects.push(null), objects.length - 1), add = value => (objects.push(value), objects.length - 1);
    function stream(data, attributes = "") {
      if (typeof data === "string") data = encode(data);
      return [encode("<< /Length " + data.length + " " + attributes + ">>\nstream\n"), data, encode("\nendstream")];
    }
    const catalog = reserve(), pages = reserve(), page = reserve(), typeFont = reserve(), cidFont = reserve(), fontDesc = reserve();
    const fontFile = add(stream(f.data, "/Length1 " + f.data.length + " "));
    const gidMap = new Uint8Array(glyphs.length * 2); glyphs.forEach((gid, i) => { gidMap[i * 2] = gid >>> 8; gidMap[i * 2 + 1] = gid & 255; });
    const cidMap = add(stream(gidMap));
    const mapping = [...characters].map(([character, cid]) => "<" + hex(cid, 4) + "> <" + [...character].map(c => {
      let result = ""; for (let i = 0; i < c.length; i++) result += hex(c.charCodeAt(i), 4); return result;
    }).join("") + ">");
    const cmap = ["/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /OpenEconUnicode def /CMapType 2 def 1 begincodespacerange <0000> <FFFF> endcodespacerange"];
    for (let i = 0; i < mapping.length; i += 100) { const block = mapping.slice(i, i + 100); cmap.push(block.length + " beginbfchar", ...block, "endbfchar"); }
    cmap.push("endcmap CMapName currentdict /CMap defineresource pop end end");
    const unicode = add(stream(cmap.join("\n"))), states = [];
    for (const [level, name] of alphas) states.push("/" + name + " " + add("<< /Type /ExtGState /ca " + num(level / 100) + " /CA " + num(level / 100) + " >>") + " 0 R");
    objects[typeFont] = "<< /Type /Font /Subtype /Type0 /BaseFont /Barlow-Regular /Encoding /Identity-H /DescendantFonts [" + cidFont + " 0 R] /ToUnicode " + unicode + " 0 R >>";
    objects[cidFont] = "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Barlow-Regular /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor " + fontDesc + " 0 R /CIDToGIDMap " + cidMap + " 0 R /DW 1000 /W [1 [" + widths.slice(1).map(num).join(" ") + "]] >>";
    objects[fontDesc] = "<< /Type /FontDescriptor /FontName /Barlow-Regular /Flags 32 /FontBBox [" + f.bbox.map(num).join(" ") + "] /ItalicAngle 0 /Ascent " + num(f.ascent) + " /Descent " + num(f.descent) + " /CapHeight 700 /StemV 80 /FontFile2 " + fontFile + " 0 R >>";
    const content = add(stream(commands.join("\n")));
    objects[page] = "<< /Type /Page /Parent " + pages + " 0 R /MediaBox [0 0 " + num(width) + " " + num(pageHeight) + "] /Resources << /Font << /F1 " + typeFont + " 0 R >> /ExtGState << " + states.join(" ") + " >> >> /Contents " + content + " 0 R >>";
    objects[pages] = "<< /Type /Pages /Kids [" + page + " 0 R] /Count 1 >>"; objects[catalog] = "<< /Type /Catalog /Pages " + pages + " 0 R >>";
    const chunks = [encode("%PDF-1.7\n%OpenEconometrics vector network\n")], offsets = [0]; let length = chunks[0].length;
    function append(data) { if (typeof data === "string") data = encode(data); length += data.length; if (length > MAX_BYTES) throw new Error("PDF export exceeds128MiB."); chunks.push(data); }
    for (let i = 1; i < objects.length; i++) { offsets.push(length); append(i + " 0 obj\n"); for (const part of Array.isArray(objects[i]) ? objects[i] : [objects[i]]) append(part); append("\nendobj\n"); }
    const xref = length; append("xref\n0 " + objects.length + "\n0000000000 65535 f \n"); offsets.slice(1).forEach(offset => append(String(offset).padStart(10, "0") + " 00000 n \n"));
    append("trailer\n<< /Size " + objects.length + " /Root " + catalog + " 0 R >>\nstartxref\n" + xref + "\n%%EOF\n");
    const result = new Uint8Array(length); let position = 0; for (const chunk of chunks) { result.set(chunk, position); position += chunk.length; } return result;
  }
  root.OpenEconNetworkPDF = { bytes, download(chart) { return chart.save(new root.Blob([bytes(chart.exportScene())], { type: "application/pdf" }), "pdf"); }, MAX_PRIMITIVES };
})(window);
