/** Off-tree editor-wire proposal. No browser acceptance has run. */
const SCHEMA = "openecon.editor-api.interned.v1";
const MAX_WIRE_BYTES = 500_000;
const MAX_DECODED_BYTES = 8 * 1024 * 1024;
const MAX_ITEMS = 500_000;
const MAX_DEPTH = 32;
const MAX_STRINGS = 8192;
const MAX_ENTRIES = 10_000;
const MAX_STRING_BYTES = 1024 * 1024;
type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
const utf8 = new TextEncoder();

function stringBytes(value: unknown): number {
  if (typeof value !== "string") throw new Error("Expected a string");
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error("Unpaired Unicode surrogate");
    } else if (code >= 0xdc00 && code <= 0xdfff) throw new Error("Unpaired Unicode surrogate");
  }
  const size = utf8.encode(value).length;
  if (size > MAX_STRING_BYTES) throw new Error("String byte budget exceeded");
  return size;
}

function record(value: Json): value is { [key: string]: Json } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) throw new Error("Non-plain metadata object");
  if (Object.values(Object.getOwnPropertyDescriptors(value)).some(field => !Object.hasOwn(field, "value"))) {
    throw new Error("Metadata accessors are unsupported");
  }
  return true;
}

/** Bound depth and reject duplicate keys before admitting parsed metadata. */
function parseStrict(raw: string): Json {
  if (typeof raw !== "string" || stringBytes(raw) > MAX_WIRE_BYTES) throw new Error("Wire byte budget exceeded");
  let cursor = 0;
  let items = 0;
  function space() { while (/[ \t\r\n]/.test(raw[cursor] ?? "") && cursor < raw.length) cursor++; }
  function string(): string {
    const begin = cursor++;
    let escaped = false;
    while (cursor < raw.length) {
      const char = raw[cursor++];
      if (!escaped && char === '"') {
        const value: unknown = JSON.parse(raw.slice(begin, cursor));
        stringBytes(value);
        return value as string;
      }
      if (!escaped && char === "\\") escaped = true; else escaped = false;
    }
    throw new Error("Incomplete JSON string");
  }
  function value(depth: number): Json {
    space();
    if (depth > MAX_DEPTH || ++items > MAX_ITEMS) throw new Error("Wire geometry exceeded");
    const char = raw[cursor];
    if (depth >= MAX_DEPTH && (char === "{" || char === "[")) throw new Error("Wire depth budget exceeded");
    if (char === '"') return string();
    if (char === "{") {
      cursor++; space();
      const result = Object.create(null) as { [key: string]: Json };
      if (raw[cursor] === "}") { cursor++; return result; }
      while (true) {
        space();
        if (raw[cursor] !== '"') throw new Error("Expected JSON object key");
        const key = string(); space();
        if (Object.hasOwn(result, key)) throw new Error("Duplicate JSON key");
        if (raw[cursor++] !== ":") throw new Error("Expected JSON colon");
        result[key] = value(depth+1); space();
        const end = raw[cursor++];
        if (end === "}") return result;
        if (end !== ",") throw new Error("Expected JSON object separator");
      }
    }
    if (char === "[") {
      cursor++; space();
      const result: Json[] = [];
      if (raw[cursor] === "]") { cursor++; return result; }
      while (true) {
        result.push(value(depth+1)); space();
        const end = raw[cursor++];
        if (end === "]") return result;
        if (end !== ",") throw new Error("Expected JSON array separator");
      }
    }
    for (const [token, literal] of [["true", true], ["false", false], ["null", null]] as const) {
      if (raw.slice(cursor, cursor+token.length) === token) { cursor += token.length; return literal; }
    }
    const match = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(raw.slice(cursor));
    if (!match) throw new Error("Invalid JSON value");
    cursor += match[0].length;
    const number = Number(match[0]);
    if (!Number.isFinite(number) || Math.abs(number) > Number.MAX_SAFE_INTEGER || Object.is(number, -0)) {
      throw new Error("Nonfinite, unsafe or negative-zero JSON number");
    }
    return number;
  }
  const result = value(0); space();
  if (cursor !== raw.length) throw new Error("Trailing JSON input");
  return result;
}

function order(left: string, right: string): number {
  const a = utf8.encode(left), b = utf8.encode(right);
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]-b[i];
  return a.length-b.length;
}

function admit(value: Json, state = [0, 0], depth = 0): number[] {
  if (depth > MAX_DEPTH || ++state[0] > MAX_ITEMS) throw new Error("Decoded geometry exceeded");
  if (typeof value === "string") state[1] += 6 * stringBytes(value) + 2;
  else if (value === null || typeof value === "boolean") state[1] += 5;
  else if (typeof value === "number") {
    if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER || Object.is(value, -0)) throw new Error("Invalid number");
    // Same conservative bound for both integer and fractional literals.
    state[1] += 32;
  } else if (Array.isArray(value)) {
    state[1] += value.length + 2;
    for (const child of value) admit(child, state, depth+1);
  } else if (record(value)) {
    state[1] += 2 * Object.keys(value).length + 2;
    for (const [key, child] of Object.entries(value)) { state[1] += 6 * stringBytes(key) + 2; admit(child, state, depth+1); }
  } else throw new Error("Only bounded JSON metadata is supported");
  if (state[1] > MAX_DECODED_BYTES) throw new Error("Decoded byte reserve exceeded");
  return state;
}

function canonicalEnvelope(entries: Json[]): Json {
  const counts = new Map<string, number>();
  function count(node: Json) {
    if (typeof node === "string") counts.set(node, (counts.get(node) ?? 0) + 1);
    else if (Array.isArray(node)) node.forEach(count);
    else if (record(node)) Object.values(node).forEach(count);
  }
  count(entries);
  const candidates: [number, string][] = [];
  for (const [text, occurrences] of counts) {
    const length = utf8.encode(JSON.stringify(text)).length;
    const saving = occurrences * (length - 12) - length - 1;
    if (occurrences >= 2 && length >= 24 && saving > 0) candidates.push([saving, text]);
  }
  candidates.sort((a, b) => b[0] - a[0] || order(a[1], b[1]));
  const strings = candidates.slice(0, MAX_STRINGS).map(pair => pair[1]).sort(order);
  const tokens = new Map(strings.map((text, token) => [text, token]));
  function pack(node: Json): Json {
    if (typeof node === "string" && tokens.has(node)) return { $s: tokens.get(node)! };
    if (Array.isArray(node)) return node.map(pack);
    if (record(node)) {
      if (Object.keys(node).some(key => key.startsWith("$"))) return { $o: Object.entries(node).sort(([left], [right]) => order(left, right)).map(([key, child]) => [key, pack(child)]) };
      const object = Object.create(null) as { [key: string]: Json };
      for (const [key, child] of Object.entries(node)) object[key] = pack(child);
      return object;
    }
    return node;
  }
  return { schema: SCHEMA, strings, entries: pack(entries) };
}

/** Return unchanged legacy stored entries or fully reconstructed new entries. */
export function decodeEditorWire(raw: string): Json[] {
  const value = parseStrict(raw);
  if (Array.isArray(value)) {
    if (value.length > MAX_ENTRIES) throw new Error("Entry count exceeded");
    admit(value); return value;
  }
  if (!record(value) || Object.keys(value).sort().join(",") !== "entries,schema,strings" || value.schema !== SCHEMA) {
    throw new Error("Unknown or mixed editor wire schema");
  }
  const strings = value.strings, entries = value.entries;
  if (!Array.isArray(strings) || strings.length > MAX_STRINGS || !Array.isArray(entries) || entries.length > MAX_ENTRIES) {
    throw new Error("Envelope geometry exceeded");
  }
  const stringTable: Json[] = strings;
  for (let i = 0; i < strings.length; i++) {
    stringBytes(strings[i]);
    if (i && order(strings[i-1] as string, strings[i] as string) >= 0) throw new Error("Dictionary must be unique and UTF-8 sorted");
  }
  admit(value);
  const uses = strings.map(() => 0);
  const state = [0, 0];
  function unpack(node: Json, depth = 0): Json {
    if (depth > MAX_DEPTH) throw new Error("Decoded depth budget exceeded");
    let result: Json;
    if (record(node) && Object.hasOwn(node, "$s")) {
      const token = node.$s;
      if (Object.keys(node).length !== 1 || typeof token !== "number" || !Number.isInteger(token)
          || Object.is(token, -0) || token < 0 || token >= stringTable.length) throw new Error("Invalid string token");
      uses[token]++; result = stringTable[token];
    } else if (record(node) && Object.hasOwn(node, "$o")) {
      if (Object.keys(node).length !== 1 || !Array.isArray(node.$o)) throw new Error("Invalid escaped object");
      const object = Object.create(null) as { [key: string]: Json };
      for (const pair of node.$o) {
        if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || Object.hasOwn(object, pair[0])) throw new Error("Invalid escaped object pair");
        object[pair[0]] = unpack(pair[1], depth+1);
      }
      if (!Object.keys(object).some(key => key.startsWith("$"))) throw new Error("Unnecessary escaped object");
      result = object;
    } else if (record(node)) {
      if (Object.keys(node).some(key => key.startsWith("$"))) throw new Error("Unknown reserved string-wire key");
      const object = Object.create(null) as { [key: string]: Json };
      for (const [key, child] of Object.entries(node)) object[key] = unpack(child, depth+1);
      result = object;
    } else if (Array.isArray(node)) result = node.map(child => unpack(child, depth+1));
    else result = node;
    if (Array.isArray(result) || record(result)) {
      state[0]++;
      state[1] += 2 + (Array.isArray(result) ? result.length : Object.keys(result).reduce((total, key) => total + 4 + 6*stringBytes(key), 0));
    } else admit(result, state);
    if (state[0] > MAX_ITEMS || state[1] > MAX_DECODED_BYTES) throw new Error("Decoded metadata budget exceeded");
    return result;
  }
  const result = unpack(entries);
  if (uses.some(count => count < 2)) throw new Error("Unused or singleton interned value");
  const canonical = canonicalEnvelope(result as Json[]) as { [key: string]: Json };
  if (JSON.stringify(strings) !== JSON.stringify(canonical.strings)
      || JSON.stringify(entries) !== JSON.stringify(canonical.entries)) throw new Error("Noncanonical string interning");
  return result as Json[];
}

/** Separate complete logical limits, after legacy fixed-token expansion. */
export function admitLogicalCatalog(entries: unknown): void {
  if (!Array.isArray(entries) || entries.length > MAX_ENTRIES) throw new Error("Expected bounded logical catalogue entries");
  admit(entries as Json[]);
}
