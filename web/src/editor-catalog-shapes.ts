/** Prospective complete shape wire; source only, no browser acceptance. */
import { decodeEditorWire as decodePreservedWire, admitLogicalCatalog } from "./editor-catalog-intern.ts";
export { admitLogicalCatalog };
const SCHEMA = "openecon.editor-api.shaped-arrays.v1";
const MAX_WIRE_BYTES = 500_000;
const MAX_DECODED_BYTES = 8 * 1024 * 1024;
const MAX_ITEMS = 500_000;
const MAX_DEPTH = 32;
const MAX_STRINGS = 8192;
const MAX_SHAPES = 8192;
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

function order(left: string, right: string): number {
  const a = utf8.encode(left), b = utf8.encode(right);
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]-b[i];
  return a.length-b.length;
}

function shapeOrder(left: string[], right: string[]): number {
  for (let i = 0; i < Math.min(left.length, right.length); i++) {
    const difference = order(left[i], right[i]);
    if (difference) return difference;
  }
  return left.length-right.length;
}

function admit(value: Json, state = [0, 0], depth = 0): number[] {
  if (depth > MAX_DEPTH || ++state[0] > MAX_ITEMS) throw new Error("Decoded geometry exceeded");
  if (typeof value === "string") state[1] += 6 * stringBytes(value) + 2;
  else if (value === null || typeof value === "boolean") state[1] += 5;
  else if (typeof value === "number") {
    if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER || Object.is(value, -0)) throw new Error("Invalid number");
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

/** Keep raw numeric lexemes at their parent coordinates until token admission.
 * Scalars retain the inherited safe finite-number domain. Only new dictionary
 * indices require integer lexemes, so 0.0 and 0e0 cannot become valid tokens by
 * JavaScript Number conversion. The old decoder remains wholly unchanged.
 */
function parseShapedStrict(raw: string): { value: Json; lexemes: WeakMap<object, Map<string, string>> } {
  if (typeof raw !== "string" || stringBytes(raw) > MAX_WIRE_BYTES) throw new Error("Wire byte budget exceeded");
  const lexemes = new WeakMap<object, Map<string, string>>();
  let cursor = 0, items = 0;
  function space() { while (/[ \t\r\n]/.test(raw[cursor] ?? "") && cursor < raw.length) cursor++; }
  function string(): string {
    const begin = cursor++;
    let escaped = false;
    while (cursor < raw.length) {
      const char = raw[cursor++];
      if (!escaped && char === '"') {
        const result: unknown = JSON.parse(raw.slice(begin, cursor));
        stringBytes(result);
        return result as string;
      }
      if (!escaped && char === "\\") escaped = true; else escaped = false;
    }
    throw new Error("Incomplete JSON string");
  }
  function child(parent: object, field: string, depth: number): Json {
    const begin = cursor;
    const result = value(depth);
    if (typeof result === "number") {
      let fields = lexemes.get(parent);
      if (!fields) { fields = new Map<string, string>(); lexemes.set(parent, fields); }
      fields.set(field, raw.slice(begin, cursor).trim());
    }
    return result;
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
        result[key] = child(result, key, depth+1); space();
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
        result.push(child(result, String(result.length), depth+1)); space();
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
  return { value: result, lexemes };
}

function canonicalEnvelope(entries: Json[]): { [key: string]: Json } {
  if (!Array.isArray(entries) || entries.length > MAX_ENTRIES) throw new Error("Entry count exceeded");
  admit(entries);
  const counts = new Map<string, number>(), shapeMap = new Map<string, string[]>();
  function count(node: Json) {
    if (typeof node === "string") counts.set(node, (counts.get(node) ?? 0) + 1);
    else if (Array.isArray(node)) node.forEach(count);
    else if (record(node)) {
      const fields = Object.keys(node).sort(order);
      shapeMap.set(JSON.stringify(fields), fields);
      if (shapeMap.size > MAX_SHAPES) throw new Error("Object shape count exceeded");
      Object.values(node).forEach(count);
    }
  }
  count(entries);
  const candidates: [number, string][] = [];
  for (const [text, occurrences] of counts) {
    const length = utf8.encode(JSON.stringify(text)).length;
    const saving = occurrences * (length - 12) - length - 1;
    if (occurrences >= 2 && length >= 24 && saving > 0) candidates.push([saving, text]);
  }
  candidates.sort((a, b) => b[0]-a[0] || order(a[1], b[1]));
  const strings = candidates.slice(0, MAX_STRINGS).map(pair => pair[1]).sort(order);
  const stringTokens = new Map(strings.map((text, token) => [text, token]));
  const shapes = [...shapeMap.values()].sort(shapeOrder);
  const shapeTokens = new Map(shapes.map((fields, token) => [JSON.stringify(fields), token]));
  function pack(node: Json): Json {
    if (typeof node === "string" && stringTokens.has(node)) return { $s: stringTokens.get(node)! };
    if (Array.isArray(node)) return { $a: node.map(pack) };
    if (record(node)) {
      const fields = Object.keys(node).sort(order);
      return [shapeTokens.get(JSON.stringify(fields))!, ...fields.map(field => pack(node[field]))];
    }
    return node;
  }
  return { schema: SCHEMA, strings, shapes, entries: entries.map(pack) };
}

export function encodeShapedEditorWire(entries: Json[]): string {
  const raw = JSON.stringify(canonicalEnvelope(entries)) + "\n";
  if (stringBytes(raw) > MAX_WIRE_BYTES) throw new Error("Wire byte budget exceeded");
  return raw;
}

export function decodeEditorWire(raw: string): Json[] {
  const { value, lexemes } = parseShapedStrict(raw);
  if (!record(value) || value.schema !== SCHEMA) return decodePreservedWire(raw);
  if (Object.keys(value).sort().join(",") !== "entries,schema,shapes,strings") throw new Error("Unknown or mixed shape-wire fields");
  const strings = value.strings, shapes = value.shapes, entries = value.entries;
  if (!Array.isArray(strings) || strings.length > MAX_STRINGS || !Array.isArray(shapes)
      || shapes.length > MAX_SHAPES || !Array.isArray(entries) || entries.length > MAX_ENTRIES) throw new Error("Shape-wire geometry exceeded");
  const stringTable: Json[] = strings, shapeTable: Json[] = shapes;
  admit(value);
  for (let i = 0; i < strings.length; i++) {
    stringBytes(strings[i]);
    if (i && order(strings[i-1] as string, strings[i] as string) >= 0) throw new Error("Dictionary must be unique and UTF-8 sorted");
  }
  const fieldsByShape: string[][] = [];
  for (const fields of shapes) {
    if (!Array.isArray(fields)) throw new Error("Object shape must be a complete key array");
    for (let i = 0; i < fields.length; i++) {
      stringBytes(fields[i]);
      if (i && order(fields[i-1] as string, fields[i] as string) >= 0) throw new Error("Shape keys must be unique and UTF-8 sorted");
    }
    if (fieldsByShape.length && shapeOrder(fieldsByShape[fieldsByShape.length-1], fields as string[]) >= 0) throw new Error("Shape dictionary must be unique and UTF-8 sorted");
    fieldsByShape.push(fields as string[]);
  }
  const stringUses = strings.map(() => 0), shapeUses = shapes.map(() => 0), state = [1, 2+entries.length];
  function index(parent: object, field: string, token: Json, limit: number): number {
    const spelling = lexemes.get(parent)?.get(field);
    if (typeof token !== "number" || !Number.isSafeInteger(token) || Object.is(token, -0)
        || token < 0 || token >= limit || spelling === undefined || !/^(0|[1-9][0-9]*)$/.test(spelling)) {
      throw new Error("Invalid canonical shape/string index");
    }
    return token;
  }
  function unpack(node: Json, depth: number): Json {
    if (depth > MAX_DEPTH) throw new Error("Decoded depth budget exceeded");
    let result: Json;
    if (Array.isArray(node)) {
      if (!node.length) throw new Error("Object shape token is missing");
      const token = index(node, "0", node[0], shapeTable.length), fields = fieldsByShape[token];
      if (node.length !== fields.length+1) throw new Error("Complete object shape/value count disagrees");
      shapeUses[token]++;
      const object = Object.create(null) as { [key: string]: Json };
      fields.forEach((field, position) => { object[field] = unpack(node[position+1], depth+1); });
      result = object;
    } else if (record(node)) {
      if (Object.keys(node).length === 1 && Object.hasOwn(node, "$s")) {
        const token = index(node, "$s", node.$s, stringTable.length);
        stringUses[token]++; result = stringTable[token];
      } else if (Object.keys(node).length === 1 && Object.hasOwn(node, "$a") && Array.isArray(node.$a)) {
        result = node.$a.map(child => unpack(child, depth+1));
      } else throw new Error("Unknown, mixed or unescaped shape-wire object");
    } else result = node;
    if (Array.isArray(result) || record(result)) {
      state[0]++;
      state[1] += 2 + (Array.isArray(result) ? result.length : Object.keys(result).reduce((total, field) => total+4+6*stringBytes(field), 0));
    } else admit(result, state);
    if (state[0] > MAX_ITEMS || state[1] > MAX_DECODED_BYTES) throw new Error("Decoded metadata budget exceeded");
    return result;
  }
  const result = entries.map(entry => unpack(entry, 1));
  if (stringUses.some(count => count < 2) || shapeUses.some(count => count < 1)) throw new Error("Unused/singleton string or unused object shape");
  const canonical = canonicalEnvelope(result);
  if (JSON.stringify(strings) !== JSON.stringify(canonical.strings)
      || JSON.stringify(shapes) !== JSON.stringify(canonical.shapes)
      || JSON.stringify(entries) !== JSON.stringify(canonical.entries)) throw new Error("Noncanonical complete shape/string wire");
  return result;
}
