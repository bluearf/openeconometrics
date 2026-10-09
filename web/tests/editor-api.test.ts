import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { decodeApiEntry, type StoredApiEntry } from "../src/editor-api.ts";

const fourStageSignatures = JSON.parse(readFileSync(
  new URL("./fixtures/four-stage-signature-suffixes.json", import.meta.url), "utf8",
)) as { token: number; suffix: string }[];
const fourStageParameters = JSON.parse(readFileSync(
  new URL("./fixtures/four-stage-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
const codecSource = readFileSync(new URL("../src/editor-api.ts", import.meta.url), "utf8");
const parameterTokenUpperBound = JSON.parse(
  codecSource.match(/^const STORED_PARAMETER_LISTS[^=]*= (\[[^\n]+\]);$/m)![1],
).length;
const signatureTokenUpperBound = JSON.parse(
  codecSource.match(/^const STORED_SIGNATURE_SUFFIXES = (\[[\s\S]*?\])(?: as const)?;/m)![1]
    .replace(/,\s*\]$/, "]"),
).length;

test("complete fixed parameter lists preserve ordering, literal defaults and independent objects", () => {
  const first = { n: "~saved.restore", S: "(payload)", d: "Replay", P: 0 };
  const decoded = decodeApiEntry(first);
  assert.deepEqual(decoded.parameters.map(p => p.name), ["json_data", "strict", "extra", "context", "by_alias", "by_name"]);
  assert.equal(decoded.parameters[0].kind, "positional-or-keyword");
  assert.ok(decoded.parameters.slice(1).every(p => p.kind === "keyword-only" && p.default === "None"));
  decoded.parameters[0].description = "Locally changed";
  assert.equal(decodeApiEntry(first).parameters[0].description, undefined);
  assert.equal(Object.hasOwn(decoded, "P"), false);
  const dumped = decodeApiEntry({ ...first, P: 1 });
  assert.equal(dumped.parameters.length, 15);
  assert.equal(dumped.parameters.find(p => p.name === "warnings")?.default, "True");
  assert.equal(dumped.parameters.find(p => p.name === "ensure_ascii")?.default, "False");
  assert.equal(dumped.parameters.at(-1)?.name, "polymorphic_serialization");
  assert.equal(decodeApiEntry({ ...first, P: 2 }).parameters[0].name, "obj");
});

test("unknown parameter-list tokens fail while explicit and legacy parameter lists take precedence", () => {
  const first = { n: "~saved.restore", S: "(payload)", d: "Replay", P: 0 };
  for (const token of [true, false, null, -1, parameterTokenUpperBound, 0.5, -0, NaN, Infinity, "0", [], {}]) {
    assert.throws(() => decodeApiEntry({ ...first, P: token } as unknown as StoredApiEntry), /parameters/);
  }
  const parameters = [{ name: "payload", kind: "positional-only", default: "'literal'" }];
  for (const field of ["parameters", "p"]) {
    assert.deepEqual(decodeApiEntry({ ...first, P: true, [field]: parameters } as unknown as StoredApiEntry).parameters, parameters);
  }
});

test("compact signatures restore canonical leaves and preserve explicit legacy signatures", () => {
  for (const [name, kind] of [["~cflogit", undefined], ["~ResultBundle.summary", "m"], ["pandas.DataFrame", "c"]] as const) {
    const stored: StoredApiEntry = { n: name, s: "(*, data: Any, x=None)", d: "Complete help.", p: [], ...(kind ? { kind } : {}) };
    const leaf = name.slice(name.lastIndexOf(".") + 1).replace(/^~/, "");
    assert.equal(decodeApiEntry(stored).signature, `${leaf}(*, data: Any, x=None)`);
    assert.equal(decodeApiEntry({ ...stored, signature: "explicit_legacy()" }).signature, "explicit_legacy()");
  }
});

test("compact null and boolean defaults decode to exact Python literals", () => {
  const entry: StoredApiEntry = { n: "~example", s: "(a=None,b=True,c=False)", d: "Complete.", p: [
    { n: "a", v: null }, { n: "b", v: true }, { n: "c", v: false },
    { n: "legacy", default: "", v: true },
  ] };
  assert.deepEqual(decodeApiEntry(entry).parameters.map(p => p.default), ["None", "True", "False", ""]);
});

test("safe integers and simple quoted defaults restore complete Python metadata", () => {
  const stored: StoredApiEntry = {
    n: "~Example.method", S: "(*, count=5, label='Türkçe')", kind: "m",
    d: "Complete help. Türkçe.", R: "TableSet", p: [
      { n: "zero", k: "k", v: 0, d: "Exact integer.", c: ["0", "1"] },
      { n: "negative", t: "k", v: -25, a: "int" },
      { n: "upper", v: Number.MAX_SAFE_INTEGER },
      { n: "lower", v: -Number.MAX_SAFE_INTEGER },
      { n: "empty", q: "" },
      { n: "label", q: 'Türkçe "label"', c: ["'Türkçe'"] },
      { n: "none", v: null }, { n: "yes", v: true }, { n: "no", v: false },
    ],
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), {
    name: "openecon.Example.method", signature: "method(*, count=5, label='Türkçe')",
    kind: "method", owner: "openecon.Example", description: "Complete help. Türkçe.",
    returns: "openecon.TableSet", parameters: [
      { name: "zero", kind: "keyword-only", default: "0", description: "Exact integer.", choices: ["0", "1"] },
      { name: "negative", kind: "keyword-only", default: "-25", annotation: "int" },
      { name: "upper", kind: "positional-or-keyword", default: "9007199254740991" },
      { name: "lower", kind: "positional-or-keyword", default: "-9007199254740991" },
      { name: "empty", kind: "positional-or-keyword", default: "''" },
      { name: "label", kind: "positional-or-keyword", default: '\'Türkçe "label"\'', choices: ["'Türkçe'"] },
      { name: "none", kind: "positional-or-keyword", default: "None" },
      { name: "yes", kind: "positional-or-keyword", default: "True" },
      { name: "no", kind: "positional-or-keyword", default: "False" },
    ],
  });
  assert.deepEqual(stored, before);
});

test("default precedence is full field then legacy value then quoted marker", () => {
  const stored: StoredApiEntry = { n: "~example", S: "()", d: "Complete help.", p: [
    { n: "full", default: "'full'", v: "'legacy'", q: "quoted" },
    { n: "blank", default: "", v: Number.NaN, q: "bad'quote" },
    { n: "legacy", v: "", q: "quoted" },
    { n: "zero", v: 0, q: "quoted" },
    { n: "false", v: false, q: "quoted" },
    { n: "none", v: null, q: "bad'quote" },
    { n: "quoted", q: "quoted" },
  ] };
  const before = structuredClone(stored);
  const parameters = decodeApiEntry(stored).parameters;
  assert.deepEqual(parameters.map(p => p.default), ["'full'", "", "", "0", "False", "None", "'quoted'"]);
  assert.ok(parameters.every(p => !("v" in p) && !("q" in p)));
  assert.deepEqual(stored, before);
});

test("unsafe numeric spellings and escaped literals remain exact legacy strings", () => {
  const defaults = ["-0", "00", "001", "+1", "1.0", "1e3", "9007199254740992", "-9007199254740992",
    "'can\\'t'", "'escaped\\n'", "'actual\nnewline'", "'actual\rcarriage'", '"double"'];
  const stored: StoredApiEntry = { n: "~example", S: "()", d: "Complete help.",
    p: defaults.map((v, i) => ({ n: `arg${i}`, v })) };
  assert.deepEqual(decodeApiEntry(stored).parameters.map(p => p.default), defaults);
});

test("active compact defaults reject unsafe numbers and non-simple quote interiors", () => {
  const entry: StoredApiEntry = { n: "~example", S: "()", d: "Complete help.", p: [] };
  for (const v of [1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 2 ** 53, -(2 ** 53), -0]) {
    assert.throws(() => decodeApiEntry({ ...entry, p: [{ n: "value", v }] }), /safe decimal integers/);
  }
  for (const q of ["can't", "back\\slash", "new\nline", "carriage\rreturn"]) {
    assert.throws(() => decodeApiEntry({ ...entry, p: [{ n: "value", q }] }), /exact simple string interior/);
  }
});

test("fixed parameter name tokens preserve all metadata and method owners", () => {
  const names = ["max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
    "intercept", "missing", "instruments", "endogenous", "data",
    "alpha", "weights", "result", "device", "time", "cluster", "categorical", "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb"];
  const stored: StoredApiEntry = {
    n: "~Result.method", S: "(*, max_work=5, endogenous=None)", k: "m",
    d: "Complete Türkçe help.", R: "DataFrame",
    p: names.map((_, n) => ({ n, k: "k", a: "Any", d: `Parameter ${n}. Türkçe.`,
      c: ["None", "'Türkçe'"], ...(n === 0 ? { v: 5 } : n === 8 ? { v: null } : { q: "Türkçe" }) })),
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), {
    name: "openecon.Result.method", signature: "method(*, max_work=5, endogenous=None)",
    kind: "method", owner: "openecon.Result", description: "Complete Türkçe help.",
    returns: "openecon.DataFrame", parameters: names.map((name, n) => ({
      name, kind: "keyword-only", annotation: "Any", description: `Parameter ${n}. Türkçe.`,
      choices: ["None", "'Türkçe'"], default: n === 0 ? "5" : n === 8 ? "None" : "'Türkçe'",
    })),
  });
  assert.deepEqual(stored, before);
});

test("legacy parameter names and explicit names take precedence over unused tokens", () => {
  const stored: StoredApiEntry = { n: "~example", S: "()", d: "Complete help.", p: [
    { n: "max_work" }, { n: "maks_çalışma" }, { n: "0" }, { n: "MAX_WORK" },
    { name: "explicit", n: 8 }, { name: "", n: 9 }, { name: "legacy", n: Number.NaN },
  ] };
  const before = structuredClone(stored);
  const parameters = decodeApiEntry(stored).parameters;
  assert.deepEqual(parameters.map(p => p.name), ["max_work", "maks_çalışma", "0", "MAX_WORK", "explicit", "", "legacy"]);
  assert.ok(parameters.every(p => !("n" in p)));
  assert.deepEqual(stored, before);
});

test("active parameter name tokens refuse invalid indices and nonnumeric aliases", () => {
  const entry: StoredApiEntry = { n: "~example", S: "()", d: "Complete help.", p: [] };
  for (const n of [-1, 31, 0.5, Number.MAX_SAFE_INTEGER, Number.NaN, Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY, -0, true, false, null, {}, [], undefined]) {
    const invalid = { ...entry, p: [{ n }] } as unknown as StoredApiEntry;
    assert.throws(() => decodeApiEntry(invalid), /compact integer token from 0 to 30/);
    const explicit = { ...entry, p: [{ name: "explicit", n }] } as unknown as StoredApiEntry;
    assert.equal(decodeApiEntry(explicit).parameters[0].name, "explicit");
  }
});

test("appended data name token preserves full metadata and legacy name precedence", () => {
  const entry: StoredApiEntry = {
    n: "~call", S: "(data, /)", d: "Complete Türkçe help.", r: "dict[str, Any]",
    p: [{ n: 9, k: "p", a: "Any", q: "None", d: "Complete parameter help.",
      c: ["'None'", "None", "False"] }, { n: "data_alias" }, { name: "Data", n: 9 }],
  };
  const snapshot = structuredClone(entry);
  assert.deepEqual(decodeApiEntry(entry).parameters, [
    { name: "data", kind: "positional-only", annotation: "Any", default: "'None'",
      description: "Complete parameter help.", choices: ["'None'", "None", "False"] },
    { name: "data_alias", kind: "positional-or-keyword" },
    { name: "Data", kind: "positional-or-keyword" },
  ]);
  assert.deepEqual(entry, snapshot);
});

test("compact None defaults preserve required parameters and all literal and legacy defaults", () => {
  const stored: StoredApiEntry = { n: "~example", S: "(required, *, option=None)", d: "Complete Türkçe help.", p: [
    { n: "required" }, { n: "none", V: 0 }, { n: "quoted", v: "'None'" },
    { n: "true", v: "True" }, { n: "false", v: "False" },
    { n: "legacy", v: "", V: 17 }, { n: "explicit", default: "", v: "ignored", V: 17 },
  ] };
  const before = structuredClone(stored);
  const parameters = decodeApiEntry(stored).parameters;
  assert.equal(Object.hasOwn(parameters[0], "default"), false);
  assert.deepEqual(parameters.slice(1).map(p => p.default), ["None", "'None'", "True", "False", "", ""]);
  assert.equal(parameters.some(p => Object.hasOwn(p, "V")), false);
  assert.deepEqual(stored, before);
  for (const tag of [null, true, false, 1, 0.5, -0, "0", [], Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => decodeApiEntry({ ...stored, p: [{ n: "x", V: tag }] } as unknown as StoredApiEntry), /integer zero None tag/);
  }
});

test("Boolean default markers retain exact literals, legacy wire and default precedence", () => {
  const cases: Array<[Record<string, unknown>, string | boolean]> = [
    [{ B: 0 }, "False"], [{ B: 1, N: "ignored" }, "True"],
    [{ V: 0, B: "ignored", N: "ignored" }, "None"],
    [{ v: "", V: "ignored", B: "ignored", N: "ignored" }, ""],
    [{ default: false, v: "True", B: "ignored" }, false],
    [{ v: true, B: "ignored" }, "True"], [{ v: false, B: "ignored" }, "False"],
    [{ v: null, B: "ignored" }, "None"],
  ];
  for (const [fields, expected] of cases) {
    const entry = {n: "~example", S: "(x=False)", R: "TableSet", d: "Full help.",
      p: [{n: "x", ...fields}]} as unknown as StoredApiEntry;
    const before = structuredClone(entry);
    const decoded = decodeApiEntry(entry);
    assert.equal(decoded.signature, "example(x=False)");
    assert.equal(decoded.returns, "openecon.TableSet");
    assert.deepEqual(decoded.parameters, [{name: "x", default: expected, kind: "positional-or-keyword"}]);
    assert.deepEqual(entry, before);
  }
  for (const tag of [null, true, false, -1, 2, 0.5, "0", [], Number.NaN]) {
    assert.throws(() => decodeApiEntry({n: "~example", S: "()", d: "Full help.",
      p: [{n: "x", B: tag, N: 1}]} as unknown as StoredApiEntry), /integer zero or one Boolean tag/);
  }
});

test("fixed default dictionary preserves exact literals and rejects invalid tags", () => {
  const literals = ["'raise'", "0.05", "'cpu'", "100000000", "50000000", "'drop'", "0.95", "0",
    "DEFAULT_BYTES", "DEFAULT_WORK", "'two-sided'", "'nonrobust'", "'constant'", "1000000", "2000000000", "100000"];
  const entry: StoredApiEntry = {n: "~example", S: "(**options)", R: "TableSet", d: "Full help.",
    p: literals.map((_, D) => ({n: `p${D}`, D, k: "k"}))};
  const before = structuredClone(entry);
  assert.deepEqual(decodeApiEntry(entry).parameters.map(p => p.default), literals);
  assert.ok(decodeApiEntry(entry).parameters.every(p => !Object.hasOwn(p, "D")));
  assert.deepEqual(entry, before);
  const cases: Array<[Record<string, unknown>, string | boolean]> = [
    [{D: 15, q: "bad'quote", N: "ignored"}, "100000"],
    [{B: 0, D: "ignored", q: "bad'quote", N: "ignored"}, "False"],
    [{q: "Türkçe", N: "ignored"}, "'Türkçe'"],
    [{V: 0, B: "ignored", D: "ignored", N: "ignored"}, "None"],
    [{v: "", D: "ignored"}, ""], [{default: false, v: "True", D: "ignored"}, false],
    [{v: false, D: -1}, "False"], [{v: null, D: true}, "None"],
  ];
  for (const [fields, expected] of cases) {
    const stored = {...entry, p: [{n: "x", ...fields}]} as unknown as StoredApiEntry;
    const snapshot = structuredClone(stored);
    assert.deepEqual(decodeApiEntry(stored).parameters, [{name: "x", default: expected, kind: "positional-or-keyword"}]);
    assert.deepEqual(stored, snapshot);
  }
  for (const tag of [null, true, false, -1, 16, 0.5, "0", [], Number.NaN]) {
    assert.throws(() => decodeApiEntry({...entry, p: [{n: "x", D: tag, N: 1}]} as unknown as StoredApiEntry), /integer fixed-literal tag/);
  }
});

test("None tags combine with parameter name tokens quoted defaults and legacy precedence", () => {
  const stored: StoredApiEntry = {
    n: "~Example.restore", S: "(*, max_work=None, endogenous='None')", k: "m",
    d: "Complete Türkçe help.", R: "TableSet", p: [
      { n: 0, V: 0, q: "bad'quote", k: "k", a: "int | None", d: "Bounded work.", c: ["None", "5"] },
      { n: 1, v: null, V: 17, q: "bad'quote" },
      { n: 2, v: false, V: 17, q: "ignored" },
      { n: 8, q: "None", t: "k" },
      { name: "literal", n: 99, default: "", V: Number.NaN, q: "bad'quote" },
    ],
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), {
    name: "openecon.Example.restore", signature: "restore(*, max_work=None, endogenous='None')",
    kind: "method", owner: "openecon.Example", description: "Complete Türkçe help.",
    returns: "openecon.TableSet", parameters: [
      { name: "max_work", kind: "keyword-only", default: "None", annotation: "int | None",
        description: "Bounded work.", choices: ["None", "5"] },
      { name: "max_iterations", kind: "positional-or-keyword", default: "None" },
      { name: "tolerance", kind: "positional-or-keyword", default: "False" },
      { name: "endogenous", kind: "keyword-only", default: "'None'" },
      { name: "literal", kind: "positional-or-keyword", default: "" },
    ],
  });
  assert.deepEqual(stored, before);
});

test("compact API entries restore function/class/method identity without losing help", () => {
  const stored: StoredApiEntry[] = [
    {
      n: "openecon.mi_chained",
      s: "mi_chained(data, columns, *, methods, m=5)",
      d: "Native conditional imputation with an explicit method map. Türkçe.",
      p: [
        { n: "data", a: "DataFrame" },
        { n: "columns" },
        { n: "methods", kind: "k", c: ["'normal'", "'pmm'", "'logit'"] },
        { n: "m", kind: "k", v: "5", d: "Imputation count. Türkçe." },
      ],
      r: "openecon.MIResult",
    },
    {
      n: "openecon.MIResult",
      s: "MIResult(**data)",
      d: "Complete validated imputation state.",
      p: [{ name: "data", kind: "**" }],
      kind: "c",
      r: "openecon.MIResult",
    },
    {
      n: "openecon.MIResult.dataset",
      s: "dataset(imputation=1)",
      d: "Return a fresh aligned completed dataset.",
      p: [{ name: "imputation", default: "1" }],
      kind: "m",
      r: "openecon.DataFrame",
    },
  ];
  const before = structuredClone(stored);
  assert.deepEqual(stored.map(decodeApiEntry), [
    {
      name: "openecon.mi_chained",
      signature: "mi_chained(data, columns, *, methods, m=5)",
      description: "Native conditional imputation with an explicit method map. Türkçe.",
      returns: "openecon.MIResult",
      kind: "function",
      parameters: [
        { name: "data", annotation: "DataFrame", kind: "positional-or-keyword" },
        { name: "columns", kind: "positional-or-keyword" },
        { name: "methods", kind: "keyword-only", choices: ["'normal'", "'pmm'", "'logit'"] },
        { name: "m", kind: "keyword-only", default: "5", description: "Imputation count. Türkçe." },
      ],
    },
    {
      name: "openecon.MIResult",
      signature: "MIResult(**data)",
      description: "Complete validated imputation state.",
      returns: "openecon.MIResult",
      kind: "class",
      parameters: [{ name: "data", kind: "var-keyword" }],
    },
    {
      name: "openecon.MIResult.dataset",
      signature: "dataset(imputation=1)",
      description: "Return a fresh aligned completed dataset.",
      returns: "openecon.DataFrame",
      kind: "method",
      owner: "openecon.MIResult",
      parameters: [{ name: "imputation", kind: "positional-or-keyword", default: "1" }],
    },
  ]);
  assert.deepEqual(stored, before);
});

test("legacy full API metadata remains readable and absent help fields are refused", () => {
  const legacy: StoredApiEntry = {
    name: "openecon.DataFrame.head",
    signature: "head(n=5)",
    description: "Return the first rows.",
    parameters: [{ name: "n", default: "5" }],
    kind: "method",
    owner: "openecon.DataFrame",
    returns: "openecon.DataFrame",
  };
  assert.deepEqual(decodeApiEntry(legacy), {
    ...legacy,
    parameters: [{ name: "n", kind: "positional-or-keyword", default: "5" }],
  });
  assert.throws(() => decodeApiEntry({
    name: "openecon.example", s: "example()", parameters: [],
  }), /needs a signature and description/);
});

test("None default marker restores only the literal with full and alias precedence", () => {
  const cases: Array<[Record<string, unknown>, string | undefined]> = [
    [{ N: 1 }, "None"],
    [{ V: 0, N: false }, "None"],
    [{ v: "", V: "ignored", N: "ignored" }, ""],
    [{ default: "False", v: "0", V: "ignored", N: "ignored" }, "False"],
    [{ v: "", N: 1 }, ""],
    [{ v: "0", N: 1 }, "0"],
    [{ default: "False", v: "0", N: 1 }, "False"],
    [{ default: "", v: "None", N: 1 }, ""],
    [{ v: "False" }, "False"],
    [{ v: "0" }, "0"],
    [{ v: "" }, ""],
    [{ v: "'None'" }, "'None'"],
    [{}, undefined],
  ];
  for (const [fields, expected] of cases) {
    const entry: StoredApiEntry = {
      n: "~example", S: "()", R: "TableSet", d: "Complete help.",
      p: [{ n: "value", k: "k", ...fields }],
    };
    const before = structuredClone(entry);
    const decoded = decodeApiEntry(entry);
    assert.equal(decoded.signature, "example()");
    assert.equal(decoded.returns, "openecon.TableSet");
    assert.deepEqual(decoded.parameters, [{
      name: "value", kind: "keyword-only",
      ...(expected !== undefined ? { default: expected } : {}),
    }]);
    assert.deepEqual(entry, before);
  }
  for (const tag of [null, true, false, 0, "1", [], Number.NaN]) {
    assert.throws(() => decodeApiEntry({
      n: "~example", S: "()", d: "Complete help.", p: [{ n: "x", N: tag }],
    } as unknown as StoredApiEntry), /integer one None tag/);
  }
});


test("compact parameter names, annotations, defaults and choices are lossless", () => {
  const entry: StoredApiEntry = {
    n: "openecon.example", s: "example(data, *, option=None)", d: "Complete help.",
    p: [{ n: "data", a: "DataFrame", kind: "p" },
      { n: "option", v: "None", c: ["'x'", "'y'"], d: "Choose explicitly.", kind: "k" }],
  };
  const before = structuredClone(entry);
  assert.deepEqual(decodeApiEntry(entry).parameters, [
    { name: "data", annotation: "DataFrame", kind: "positional-only" },
    { name: "option", default: "None", choices: ["'x'", "'y'"],
      description: "Choose explicitly.", kind: "keyword-only" },
  ]);
  assert.deepEqual(entry, before);
  assert.throws(() => decodeApiEntry({ ...entry, p: [{ v: "None" }] }), /parameter needs a name/);
});

test("compact parameter names defaults and choices restore complete metadata", () => {
  const stored: StoredApiEntry = {
    n: "openecon.example",
    s: "example(data, /, *, method='Türkçe', blank='', **kwargs)",
    d: "All parameter metadata remains available.",
    p: [
      { n: "data", k: "p", annotation: "DataFrame" },
      { n: "method", k: "k", v: "'Türkçe'", c: ["'Türkçe'", "'normal'"], d: "Preserve Unicode and Python literals." },
      { n: "blank", k: "k", v: "", c: [], d: "" },
      { n: "kwargs", k: "**" },
    ],
    r: "openecon.DataFrame",
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), {
    name: "openecon.example",
    signature: "example(data, /, *, method='Türkçe', blank='', **kwargs)",
    description: "All parameter metadata remains available.",
    kind: "function",
    parameters: [
      { name: "data", kind: "positional-only", annotation: "DataFrame" },
      { name: "method", kind: "keyword-only", default: "'Türkçe'", choices: ["'Türkçe'", "'normal'"], description: "Preserve Unicode and Python literals." },
      { name: "blank", kind: "keyword-only", default: "", choices: [], description: "" },
      { name: "kwargs", kind: "var-keyword" },
    ],
    returns: "openecon.DataFrame",
  });
  assert.deepEqual(stored, before);
});

test("mixed old and compact parameter records retain full-field precedence", () => {
  const stored: StoredApiEntry = {
    name: "openecon.example",
    signature: "example(a, b='')",
    description: "Mixed catalog generations remain readable.",
    parameters: [
      { name: "a", n: "ignored", kind: "keyword-only", k: "p", t: "*", default: "", v: "ignored", choices: [], c: ["ignored"], description: "", d: "ignored", annotation: "", a: "ignored" },
      { n: "b", k: "p", t: "k", v: "None" },
    ],
  };
  assert.deepEqual(decodeApiEntry(stored).parameters, [
    { name: "a", default: "", choices: [], description: "", kind: "keyword-only", annotation: "" },
    { name: "b", default: "None", kind: "positional-only" },
  ]);
  assert.throws(() => decodeApiEntry({ ...stored, parameters: [{ kind: "k", v: "1" }] }), /name for every parameter/);
});

test("parameter kind key alias restores all kinds with legacy field precedence", () => {
  const stored: StoredApiEntry = {
    n: "~example", s: "example()", d: "Complete help.",
    p: [
      { n: "positional", t: "p" },
      { n: "keyword", t: "k" },
      { n: "args", t: "*" },
      { n: "kwargs", t: "**" },
      { n: "default" },
      { n: "legacy", kind: "positional-only", t: "k" },
    ],
  };
  const before = structuredClone(stored);
  const parameters = decodeApiEntry(stored).parameters;
  assert.deepEqual(parameters.map(p => p.kind), [
    "positional-only", "keyword-only", "var-positional", "var-keyword",
    "positional-or-keyword", "positional-only",
  ]);
  assert.ok(parameters.every(p => !("t" in p)));
  assert.deepEqual(stored, before);
});

test("shared canonical name prefixes restore full names and method owners", () => {
  for (const legacyName of ["openecon.sspace", "openecon.plot.Chart.render"]) {
    const entry: StoredApiEntry = {
      n: legacyName, s: "example()", d: "Complete help.", p: [],
      ...(legacyName.endsWith("render") ? { kind: "m" } : {}),
    };
    const compact = { ...entry, n: `~${legacyName.slice("openecon.".length)}` };
    const before = structuredClone(compact);
    assert.deepEqual(decodeApiEntry(compact), decodeApiEntry(entry));
    assert.equal(decodeApiEntry(compact).name, legacyName);
    assert.deepEqual(compact, before);
  }
  assert.equal(decodeApiEntry({
    n: "pandas.DataFrame", s: "DataFrame()", d: "Pandas help.", p: [], kind: "c",
  }).name, "pandas.DataFrame");
});

test("short and legacy parameter kind fields restore the same complete values", () => {
  const entry: StoredApiEntry = { n: "openecon.example", s: "example(value)", d: "Complete help.", p: [] };
  for (const [storedKind, expectedKind] of [
    ["k", "keyword-only"], ["p", "positional-only"],
    ["*", "var-positional"], ["**", "var-keyword"],
    ["positional-or-keyword", "positional-or-keyword"], ["extension-kind", "extension-kind"],
  ]) {
    const compact = { ...entry, p: [{ n: "value", k: storedKind, a: "DataFrame" }] };
    const legacy = { ...entry, p: [{ name: "value", kind: storedKind, annotation: "DataFrame" }] };
    const before = structuredClone(compact);
    assert.deepEqual(decodeApiEntry(compact), decodeApiEntry(legacy));
    assert.deepEqual(decodeApiEntry(compact).parameters, [
      { name: "value", kind: expectedKind, annotation: "DataFrame" },
    ]);
    assert.deepEqual(compact, before);
  }
});

test("compact return prefixes retain typed help and explicit field precedence", () => {
  const entry: StoredApiEntry = {
    n: "~mediation_binary", s: "mediation_binary()", d: "Complete help.", p: [], r: "~TableSet",
  };
  const before = structuredClone(entry);
  assert.equal(decodeApiEntry(entry).returns, "openecon.TableSet");
  assert.equal(decodeApiEntry({ ...entry, r: "openecon.MIDeltaResult" }).returns, "openecon.MIDeltaResult");
  assert.equal(decodeApiEntry({ ...entry, returns: "legacy.Result" }).returns, "legacy.Result");
  assert.equal(decodeApiEntry({ ...entry, returns: "" }).returns, "");
  assert.deepEqual(entry, before);
});


test("kind field aliases preserve owners and explicit legacy precedence", () => {
  const compact: StoredApiEntry = {
    n: "~Example.run", s: "run(*, flag=False)", d: "Complete help.", k: "m",
    p: [{ n: "flag", k: "k", v: "False" }], R: "DataFrame",
  };
  const before = structuredClone(compact);
  const decoded = decodeApiEntry(compact);
  assert.equal(decoded.kind, "method");
  assert.equal(decoded.owner, "openecon.Example");
  assert.deepEqual(decoded.parameters, [{ name: "flag", kind: "keyword-only", default: "False" }]);
  assert.ok(!("k" in decoded));
  assert.deepEqual(compact, before);
  const mixed = decodeApiEntry({ ...compact, kind: "class", p: [{ n: "flag", k: "k", kind: "positional-only", v: "" }] });
  assert.equal(mixed.kind, "class");
  assert.ok(!("owner" in mixed));
  assert.deepEqual(mixed.parameters, [{ name: "flag", kind: "positional-only", default: "" }]);
});

test("compact parameter kinds preserve call conventions and explicit legacy precedence", () => {
  const stored: StoredApiEntry = { n: "~example", s: "example()", d: "Complete help.",
    p: [{ n: "value", t: "p" }, { n: "flag", t: "k" }, { n: "args", t: "*" },
        { n: "kwargs", t: "**" }, { n: "legacy", kind: "keyword-only", t: "p" }] };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored).parameters.map(p => p.kind),
    ["positional-only", "keyword-only", "var-positional", "var-keyword", "keyword-only"]);
  assert.deepEqual(stored, before);
  assert.equal(Object.hasOwn(decodeApiEntry(stored).parameters[0], "t"), false);
});


test("signature suffix restores canonical calls while preserving aliases and explicit legacy fields", () => {
  const stored: StoredApiEntry = {n: "~Result.method", S: "(value, /, *, label='Türkçe')", d: "Complete help.", p: [], kind: "m"};
  assert.equal(decodeApiEntry(stored).signature, "method(value, /, *, label='Türkçe')");
  assert.equal(decodeApiEntry(stored).owner, "openecon.Result");
  assert.equal(decodeApiEntry({...stored, signature: ""}).signature, "");
  assert.equal(decodeApiEntry({...stored, s: "alias(value)"}).signature, "alias(value)");
  assert.equal(Object.hasOwn(decodeApiEntry(stored), "S"), false);
  assert.equal(decodeApiEntry({n: "builtins.abs", S: "(x, /)", d: "Full help.", p: []}).signature, "abs(x, /)");
  assert.throws(() => decodeApiEntry({...stored, S: "invalid"}), /signature and description/);
});

test("catalog generations retain CF mediation and TwoStep signatures defaults and typed returns", () => {
  const stored: StoredApiEntry[] = [
    { n: "~cflogit", s: "(*, data, intercept=True)", S: "(ignored)",
      d: "Full generated-control inference.", r: "~ResultBundle", R: "Ignored",
      p: [{ n: "intercept", k: "k", t: "p", v: true }] },
    { n: "~mediation_binary", S: "(*, data, weights=None)",
      d: "Fixed empirical control means. Türkçe.", R: "TableSet",
      p: [{ n: "weights", k: "k", v: null }] },
    { n: "~twostep_quality", S: "(result, *, weights=None)",
      d: "Bounded descriptive quality.", r: "TableSet",
      p: [{ n: "result" }, { n: "weights", t: "k", v: null }] },
  ];
  const before = structuredClone(stored);
  assert.deepEqual(stored.map(decodeApiEntry), [
    { name: "openecon.cflogit", signature: "cflogit(*, data, intercept=True)",
      description: "Full generated-control inference.", returns: "openecon.ResultBundle",
      kind: "function", parameters: [{ name: "intercept", kind: "keyword-only", default: "True" }] },
    { name: "openecon.mediation_binary", signature: "mediation_binary(*, data, weights=None)",
      description: "Fixed empirical control means. Türkçe.", returns: "openecon.TableSet",
      kind: "function", parameters: [{ name: "weights", kind: "keyword-only", default: "None" }] },
    { name: "openecon.twostep_quality", signature: "twostep_quality(result, *, weights=None)",
      description: "Bounded descriptive quality.", returns: "TableSet",
      kind: "function", parameters: [{ name: "result", kind: "positional-or-keyword" },
                                     { name: "weights", kind: "keyword-only", default: "None" }] },
  ]);
  assert.deepEqual(stored, before);
  const explicit = decodeApiEntry({ ...stored[0], signature: "", returns: "", parameters: [
    { name: "flag", default: "", v: false, kind: "positional-only", k: "k", t: "**" },
  ] });
  assert.equal(explicit.signature, "");
  assert.equal(explicit.returns, "");
  assert.deepEqual(explicit.parameters, [{ name: "flag", default: "", kind: "positional-only" }]);
});

test("compact null restores only the literal None default while missing defaults remain absent", () => {
  const stored: StoredApiEntry = {n: "~example", s: "example()", d: "Complete help.", p: [
    {n: "required"}, {n: "optional", v: null}, {n: "old", v: "None"},
    {n: "quoted", v: "'None'"}, {n: "empty", default: "", v: null},
    {n: "explicit", default: "5", v: null}, {n: "unknown", v: "MAX_N"},
    {n: "null_literal", v: "null"},
  ]};
  const before = structuredClone(stored);
  const parameters = decodeApiEntry(stored).parameters;
  assert.deepEqual(parameters.map(p => p.default),
    [undefined, "None", "None", "'None'", "", "5", "MAX_N", "null"]);
  assert.equal(Object.hasOwn(parameters[0], "default"), false);
  assert.ok(parameters.slice(1).every(p => Object.hasOwn(p, "default")));
  assert.ok(parameters.every(p => !Object.hasOwn(p, "v")));
  assert.deepEqual(stored, before);
});

test("compact booleans restore Python literals and preserve legacy and explicit defaults", () => {
  const stored: StoredApiEntry = {n: "~example", s: "example()", d: "Complete help.", k: "m", p: [
    {n: "required", k: "k"}, {n: "yes", v: true}, {n: "no", v: false},
    {n: "old_yes", v: "True"}, {n: "old_no", v: "False"},
    {n: "quoted_yes", v: "'True'"}, {n: "quoted_no", v: "'False'"},
    {n: "empty", default: "", v: true}, {n: "explicit", default: "MAX_N", v: false},
    {n: "lowercase", v: "true"},
  ]};
  const before = structuredClone(stored);
  const decoded = decodeApiEntry(stored);
  assert.equal(decoded.kind, "method");
  assert.equal(decoded.owner, "openecon");
  assert.equal(Object.hasOwn(decoded, "k"), false);
  assert.equal(decoded.parameters[0].kind, "keyword-only");
  assert.deepEqual(decoded.parameters.map(p => p.default),
    [undefined, "True", "False", "True", "False", "'True'", "'False'", "", "MAX_N", "true"]);
  assert.equal(Object.hasOwn(decoded.parameters[0], "default"), false);
  assert.ok(decoded.parameters.slice(1).every(p => Object.hasOwn(p, "default")));
  assert.ok(decoded.parameters.every(p => !Object.hasOwn(p, "v") && !Object.hasOwn(p, "k")));
  assert.deepEqual(stored, before);
  assert.equal(decodeApiEntry({...stored, kind: "class"}).kind, "class");
});

test("exact return tokens preserve function class method alias and complete metadata", () => {
  const types = ["openecon.TableSet", "openecon.ResultBundle", "openecon.DataFrame"];
  for (const fields of [
    {n: "~mi_chained", S: "(frame, /, *, option=None)"},
    {n: "~MIResult", S: "(frame, /, *, option=None)", k: "c" as const},
    {n: "~MIResult.model_validate_json", S: "(frame, /, *, option=None)", k: "m" as const},
    {n: "~mi_chained", s: "different_alias(frame, /, *, option=None)"},
  ]) {
    for (const [T, returns] of types.entries()) {
      const stored: StoredApiEntry = {...fields, T, d: "Complete Türkçe help.", p: [
        {n: "frame", k: "p", a: "DataFrame"},
        {n: "option", k: "k", V: 0, d: "", c: ["'Türkçe'", "'None'"]},
      ]};
      const before = structuredClone(stored);
      const decoded = decodeApiEntry(stored);
      assert.equal(decoded.returns, returns);
      assert.equal(Object.hasOwn(decoded, "T"), false);
      assert.deepEqual(decoded, decodeApiEntry({...stored, returns}));
      assert.deepEqual(decoded, decodeApiEntry({...stored, R: returns.slice("openecon.".length)}));
      assert.deepEqual(decoded, decodeApiEntry({...stored, r: "~" + returns.slice("openecon.".length)}));
      assert.equal(decoded.parameters[0].annotation, "DataFrame");
      assert.equal(Object.hasOwn(decoded.parameters[0], "default"), false);
      assert.equal(decoded.parameters[1].default, "None");
      assert.deepEqual(stored, before);
    }
  }
});

test("return token respects canonical legacy and namespace precedence including empty strings", () => {
  const stored = {n: "~mi_chained", S: "()", d: "Complete help.", p: [], T: "ignored"};
  for (const [fields, expected] of [
    [{returns: "", r: "ignored", R: "Ignored"}, ""],
    [{returns: null, r: "ignored", R: "Ignored"}, null],
    [{r: "", R: "Ignored"}, ""],
    [{r: "~ResultBundle", R: "Ignored"}, "openecon.ResultBundle"],
    [{R: "DataFrame"}, "openecon.DataFrame"],
  ] as const) {
    const input = {...stored, ...fields} as unknown as StoredApiEntry;
    const before = structuredClone(input);
    const decoded = decodeApiEntry(input);
    assert.equal(decoded.returns, expected);
    assert.equal(Object.hasOwn(decoded, "T"), false);
    assert.deepEqual(input, before);
  }
});

test("malformed return tokens are refused without a higher-priority type", () => {
  for (const T of [undefined, null, true, false, -1, 3, 0.5, "0", []]) {
    const stored = {n: "~mi_chained", S: "()", d: "Complete help.", p: [], T} as unknown as StoredApiEntry;
    const before = structuredClone(stored);
    assert.throws(() => decodeApiEntry(stored), /recognized integer token/);
    assert.deepEqual(stored, before);
  }
});

test("opaque full defaults remain exact and suppress all unused compact aliases", () => {
  const defaults = [null, false, 0, { label: "Türkçe", values: [0, false] }, ["unknown", 3]];
  const stored = { n: "~Result.restore", S: "()", d: "Complete help.", k: "m", R: "TableSet",
    p: defaults.map((value, n) => ({ n, default: value, v: Number.NaN, V: 17, q: "bad'quote", k: "k" }))
  } as unknown as StoredApiEntry;
  const before = structuredClone(stored);
  const decoded = decodeApiEntry(stored);
  assert.deepEqual(decoded.parameters.map(p => p.default), defaults);
  assert.deepEqual(decoded.parameters.map(p => p.name), ["max_work", "max_iterations", "tolerance", "covariance", "max_bytes"]);
  assert.equal(decoded.owner, "openecon.Result");
  assert.equal(decoded.returns, "openecon.TableSet");
  assert.ok(decoded.parameters.every(p => !("v" in p) && !("V" in p) && !("q" in p)));
  assert.deepEqual(stored, before);
});

test("shared signature tokens restore complete exact suffixes and saved-state metadata", () => {
  const suffixes = [
    "(json_data: str | bytes | bytearray, *, strict: bool | None=None, extra: ExtraValues | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)",
    "(*, indent: int | None=None, ensure_ascii: bool=False, include: IncEx | None=None, exclude: IncEx | None=None, context: Any | None=None, by_alias: bool | None=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, exclude_computed_fields: bool=False, round_trip: bool=False, warnings: bool | Literal['none', 'warn', 'error']=True, fallback: Callable[[Any], Any] | None=None, serialize_as_any: bool=False, polymorphic_serialization: bool | None=None)",
    "(obj: Any, *, strict: bool | None=None, extra: ExtraValues | None=None, from_attributes: bool | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)",
    "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu')",
    "(data, design, outcome, regressors, *, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09)",
    "(data, design, outcome, regressors, *, method, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09, replicates=None, replicate_weights=None, centering='original', rho=None, justification=None, scale=None, rscales=None, df=None)",
    "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu', batch_rows: int | None=None)",
    "(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None=None, model: str='re', covariance: str | None=None, cluster: str | None=None, intpoints: int | None=None, intmethod: str | None=None, corr: str | None=None, corr_order: int | None=None, force: bool=False, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)",
    "(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str], inflate_link: str='logit', covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, exposure: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)",
    "(low, indicator, *, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None)",
    "(*, data, y, x, categorical=None, weights=None, weight_type='aweight', intercept=True, missing='raise', l1_ratio=0.5, selection='cv', penalty=None, lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729, standardize=True, penalty_factors=None, forced_controls=None, max_iterations=200, tolerance=1e-08, max_work=2000000000, device='cpu')",
    "(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0)",
    "(*, data, items: list[str], points: int=41, max_iter: int=200, max_eval: int=400, tolerance: float=1e-06, level: float=0.95)",
  ];
  for (const fields of [
    {n: "~restore"}, {n: "~Result", k: "c" as const},
    {n: "~Result.restore", k: "m" as const},
  ]) {
    for (const [I, suffix] of suffixes.entries()) {
      const stored: StoredApiEntry = {...fields, I, R: "Result | None",
        d: "Complete Türkçe saved-state help.", p: [
          {n: "payload", k: "p", a: "Any"},
          {n: "option", k: "k", V: 0, d: "", c: ["'Türkçe'", "'None'"]},
        ]};
      const before = structuredClone(stored);
      const decoded = decodeApiEntry(stored);
      assert.deepEqual(decoded, decodeApiEntry({...stored, S: suffix}));
      assert.equal(decoded.signature, fields.n.slice(fields.n.lastIndexOf(".") + 1).replace(/^~/, "") + suffix);
      assert.equal(Object.hasOwn(decoded, "I"), false);
      assert.equal(decoded.returns, "openecon.Result | None");
      assert.equal(decoded.parameters[0].annotation, "Any");
      assert.equal(Object.hasOwn(decoded.parameters[0], "default"), false);
      assert.equal(decoded.parameters[1].default, "None");
      assert.deepEqual(stored, before);
    }
  }
});

test("shared signature tokens retain canonical and legacy precedence", () => {
  const stored = {n: "~Result.restore", k: "m", I: "ignored", d: "Complete help.", p: []};
  for (const [fields, expected] of [
    [{signature: "", s: "ignored", S: "(ignored)"}, ""],
    [{s: "old_alias(value)", S: "(ignored)"}, "old_alias(value)"],
    [{s: "(value)", S: "(ignored)"}, "restore(value)"],
    [{S: "(value)"}, "restore(value)"],
  ] as const) {
    const input = {...stored, ...fields} as unknown as StoredApiEntry;
    const before = structuredClone(input);
    const decoded = decodeApiEntry(input);
    assert.equal(decoded.signature, expected);
    assert.equal(decoded.owner, "openecon.Result");
    assert.equal(Object.hasOwn(decoded, "I"), false);
    assert.deepEqual(input, before);
  }
});

test("malformed shared signature tokens fail without an explicit signature", () => {
  for (const I of [undefined, null, true, false, -1, -0, signatureTokenUpperBound, .5, "0", []]) {
    const stored = {n: "~Result.restore", k: "m", I, d: "Complete help.", p: []} as unknown as StoredApiEntry;
    const before = structuredClone(stored);
    assert.throws(() => decodeApiEntry(stored), /recognized integer token/);
    assert.deepEqual(stored, before);
  }
});

test("four-stage compression restores every appended golden signature and preserves complete objects", () => {
  assert.deepEqual(fourStageSignatures.map(({token}) => token), Array.from({length: 60}, (_, i) => 7 + i));
  for (const {token: I, suffix} of fourStageSignatures) {
    for (const fields of [{n: "~restore"}, {n: "~Result", k: "c" as const}, {n: "~Result.restore", k: "m" as const}]) {
      const stored: StoredApiEntry = {...fields, I, R: "Result | None", d: "Complete Türkçe help.",
        p: [{n: "payload", k: "p", a: "Any", c: ["'Türkçe'"]}]};
      const before = structuredClone(stored);
      assert.deepEqual(decodeApiEntry(stored), decodeApiEntry({...stored, S: suffix}));
      assert.equal(decodeApiEntry(stored).signature,
        fields.n.slice(fields.n.lastIndexOf(".") + 1).replace(/^~/, "") + suffix);
      assert.deepEqual(stored, before);
    }
  }
});

test("four-stage compression restores every appended complete parameter list without shared mutations", () => {
  assert.deepEqual(fourStageParameters.map(({token}) => token), Array.from({length: 30}, (_, i) => 42 + i));
  for (const {token: P, parameters} of fourStageParameters) {
    const stored: StoredApiEntry = {n: "~Fixture.call", S: "(payload)", k: "m", R: "dict[str, Any]",
      d: "Complete Türkçe help.", P};
    const before = structuredClone(stored);
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
    const changed = decodeApiEntry(stored);
    changed.parameters[0].description = "Edited locally.";
    changed.parameters.find(parameter => parameter.choices)?.choices?.push("'local'");
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
    for (const field of ["parameters", "p"]) {
      const explicit = [{name: "literal", kind: "positional-only", default: "'kept'", choices: ["'kept'"]}];
      assert.deepEqual(decodeApiEntry({...stored, P: true, [field]: explicit} as unknown as StoredApiEntry).parameters, explicit);
    }
    assert.deepEqual(stored, before);
  }
});


test("appended complete parameter lists preserve every field and isolate nested choice arrays", () => {
  const expectedTemplates = [[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"'aweight'"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"l1_ratio","default":"0.5","kind":"keyword-only"},{"name":"selection","kind":"keyword-only","default":"'cv'"},{"name":"penalty","kind":"keyword-only","default":"None"},{"name":"lambda_path","kind":"keyword-only","default":"None"},{"name":"n_lambdas","default":"20","kind":"keyword-only"},{"name":"lambda_ratio","default":"0.001","kind":"keyword-only"},{"name":"folds","default":"5","kind":"keyword-only"},{"name":"seed","default":"1729","kind":"keyword-only"},{"name":"standardize","kind":"keyword-only","default":"True"},{"name":"penalty_factors","kind":"keyword-only","default":"None"},{"name":"forced_controls","kind":"keyword-only","default":"None"},{"name":"max_iterations","default":"200","kind":"keyword-only"},{"name":"tolerance","default":"1e-08","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"2000000000"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"panel","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"model","kind":"keyword-only","default":"'re'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"intpoints","kind":"keyword-only","default":"None"},{"name":"intmethod","kind":"keyword-only","default":"None"},{"name":"corr","kind":"keyword-only","default":"None"},{"name":"corr_order","kind":"keyword-only","default":"None"},{"name":"force","kind":"keyword-only","default":"False"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"low","kind":"positional-or-keyword"},{"name":"indicator","kind":"positional-or-keyword"},{"name":"rho","kind":"keyword-only"},{"name":"low_periods","kind":"keyword-only"},{"name":"high_periods","kind":"keyword-only"},{"name":"low_frequency","kind":"keyword-only","default":"'Y'"},{"name":"high_frequency","kind":"keyword-only","default":"'Q'"},{"name":"aggregation","kind":"keyword-only","default":"'sum'"},{"name":"as_of","kind":"keyword-only","default":"None"},{"name":"low_releases","kind":"keyword-only","default":"None"},{"name":"high_releases","kind":"keyword-only","default":"None"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only","default":"None"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"numerators","kind":"positional-or-keyword"},{"name":"denominators","kind":"positional-or-keyword"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"inflate","kind":"keyword-only"},{"name":"inflate_link","kind":"keyword-only","default":"'logit'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"exposure","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"dimensions","kind":"positional-or-keyword"},{"name":"count","kind":"positional-or-keyword"},{"name":"levels","kind":"keyword-only"},{"name":"design","kind":"keyword-only"},{"name":"terms","kind":"keyword-only","default":"None"},{"name":"structural","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"max_iter","default":"200","kind":"keyword-only"},{"name":"tol","default":"1e-09","kind":"keyword-only"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"max_work","default":"300000000","kind":"keyword-only"},{"name":"max_bytes","default":"128 * 1024 ** 2","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only","default":"None"},{"name":"group","kind":"keyword-only"},{"name":"random","kind":"keyword-only","default":"None"},{"name":"intpoints","default":"7","kind":"keyword-only"},{"name":"intmethod","kind":"keyword-only","default":"'mvaghermite'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"ll","kind":"keyword-only","default":"None"},{"name":"ul","kind":"keyword-only","default":"None"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"title","kind":"keyword-only","default":"None"},{"name":"unit","kind":"keyword-only","default":"''"},{"name":"palette","kind":"keyword-only","default":"None"},{"name":"aggregate","kind":"keyword-only","default":"None"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"raters","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only"},{"name":"agreement_weights","kind":"keyword-only","default":"'unweighted'"},{"name":"inference","kind":"keyword-only","default":"'none'"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"max_fits","default":"1024","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"keyword-only"},{"name":"variable","kind":"keyword-only"},{"name":"order","default":"1","kind":"keyword-only"},{"name":"interval","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","kind":"keyword-only","default":"2000000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"trend","kind":"keyword-only","default":"'c'"},{"name":"kernel","kind":"keyword-only","default":"'bartlett'"},{"name":"bandwidth","default":"4","kind":"keyword-only"},{"name":"df_adjust","kind":"keyword-only","default":"False"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","choices":["'cluster'","'nonrobust'"],"kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'raise'","'drop'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"kind","choices":["'linear'","'response'"],"kind":"keyword-only","default":"'response'"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"None"}],[{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_component_nodes","default":"24","kind":"keyword-only"},{"name":"max_states","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"treatment","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"var-keyword"}],[{"name":"partition","default":"None","kind":"positional-or-keyword"},{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_matrix_entries","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}]] as const;
  for (const [offset, template] of expectedTemplates.entries()) {
    const stored: StoredApiEntry = {
      n: "~Fixture.call", S: "(payload)", d: "Complete Türkçe help.",
      k: "m", r: "dict[str, Any]", P: 15 + offset,
    };
    const before = structuredClone(stored);
    const expected = {
      name: "openecon.Fixture.call", signature: "call(payload)", description: "Complete Türkçe help.",
      kind: "method", owner: "openecon.Fixture", returns: "dict[str, Any]",
      parameters: template,
    };
    const decoded = decodeApiEntry(stored);
    assert.deepEqual(decoded, expected);
    assert.deepEqual(stored, before);
    decoded.parameters[0].description = "Edited locally.";
    const parameterWithChoices = decoded.parameters.find(parameter => parameter.choices);
    parameterWithChoices?.choices?.push("'locally-edited-choice'");
    assert.deepEqual(decodeApiEntry(stored), expected);
    for (const field of ["parameters", "p"]) {
      const explicit = [{name: "required", kind: "positional-only", annotation: "Any", choices: [], description: "Exact help."}];
      const mixed = {...stored, P: true, [field]: explicit} as unknown as StoredApiEntry;
      assert.deepEqual(decodeApiEntry(mixed).parameters, explicit);
    }
  }
});


test("appended parameter templates retain every pre-existing logical field", () => {
  const golden = JSON.parse(readFileSync(new URL("./fixtures/causal-assignment-parameter-templates.json", import.meta.url), "utf8"));
  for (const { token, parameters } of golden) {
    const stored = { n: "~example", S: "()", d: "Original complete metadata.", P: token };
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
    const changed = decodeApiEntry(stored);
    changed.parameters[0].description = "Local mutation";
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
  }
});


for (const [batch, tokens] of [[2, [37, 38, 39]], [3, [40]], [4, [41]], [5, [72, 73, 74]]] as const) {
test(`integration ${batch} complete parameter lists preserve every golden field and independent objects`, () => {
  const golden = JSON.parse(readFileSync(new URL(`./fixtures/integration${batch}-parameter-templates.json`, import.meta.url), "utf8"));
  assert.deepEqual(golden.map(({ token }: { token: number }) => token), tokens);
  for (const { token, parameters } of golden) {
    const stored = { n: "~integration.example", S: "()", d: "Complete original fields.", P: token };
    const before = structuredClone(stored);
    const decoded = decodeApiEntry(stored);
    assert.deepEqual(decoded.parameters, parameters);
    assert.deepEqual(stored, before);
    decoded.parameters[0].description = "Changed locally";
    decoded.parameters.find(parameter => parameter.choices)?.choices?.push("'local'");
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
  }
});
}


test("incoming streamed control-function signature retains its append-only token", () => {
  const suffix = "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu', batch_rows: int | None=None)";
  const stored: StoredApiEntry = {n: "~cfregress", I: 6, d: "Complete incoming help.", p: []};
  assert.equal(decodeApiEntry(stored).signature, "cfregress" + suffix);
  assert.deepEqual(decodeApiEntry(stored), decodeApiEntry({...stored, S: suffix}));
});

test("new shared signatures retain every literal suffix and legacy override", () => {
  const suffixes = ["(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None=None, model: str='re', covariance: str | None=None, cluster: str | None=None, intpoints: int | None=None, intmethod: str | None=None, corr: str | None=None, corr_order: int | None=None, force: bool=False, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)","(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str], inflate_link: str='logit', covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, exposure: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)","(low, indicator, *, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None)","(*, data, y, x, categorical=None, weights=None, weight_type='aweight', intercept=True, missing='raise', l1_ratio=0.5, selection='cv', penalty=None, lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729, standardize=True, penalty_factors=None, forced_controls=None, max_iterations=200, tolerance=1e-08, max_work=2000000000, device='cpu')","(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0)","(*, data, items: list[str], points: int=41, max_iter: int=200, max_eval: int=400, tolerance: float=1e-06, level: float=0.95)","(*, data: Any, y: str, x: Sequence[str] | None=None, group: str, random: Sequence[str] | None=None, intpoints: int=7, intmethod: str='mvaghermite', covariance: str | None=None, cluster: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)","(*, data: Any, y: str, x: Sequence[str], covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)","(result: TableSet, data: Any, *, missing: str='raise', max_work: int=DEFAULT_WORK, max_bytes: int=DEFAULT_BYTES, device: str='cpu')","(data: Any, raters: list[str], *, categories: list, agreement_weights: Any='unweighted', inference: str='none', level: float=0.95, missing: str='drop', device: str='cpu', weights: Any=None, max_fits: int=1024, max_work: int=100000000)","(data: pd.DataFrame, variables: list[str], *, knots: dict, components: int=2, n_starts: int=3, seed: int=0, maxiter: int=500, tol: float=1e-08, missing: str='drop', max_work: int=WORK, max_bytes: int=BYTES, device: str='cpu')","(low, indicator, *, rho, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None, intercept=True, alpha=0.05)","(*, data, x: str, y: str | Sequence[str], title: str | None=None, unit: str='', palette=None, aggregate: str | None=None, **options)","(*, data: Any, y: str, x: Sequence[str], covariance: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, cluster: str | None=None, missing: str='raise', alpha: float=0.05)","(data, design, numerators, denominators, *, domain=None, missing='raise', alpha=0.05, null=0.0)","(data, design, outcome, *, categories=None, domain=None, missing='raise', alpha=0.05, null=0.0)","(result: ResultBundle, *, data, variable: str, order=1, interval=False, alpha=0.05, missing='raise', max_work=2000000000)","(result: ResultBundle, *, data, variable: str, order=1, interval=True, alpha=0.05, missing='raise', max_work=2000000000)"];
  for (const [offset, suffix] of suffixes.entries()) {
    const stored: StoredApiEntry = {n: "~Fixture.call", I: [7,8,9,10,11,12,13,14,15,16,67,17,18,19,20,21,22,23][offset], d: "Complete literal help.", p: [], k: "m"};
    const before = structuredClone(stored);
    assert.deepEqual(decodeApiEntry(stored), decodeApiEntry({...stored, S: suffix}));
    assert.equal(decodeApiEntry(stored).signature, "call"+suffix);
    assert.equal(decodeApiEntry({...stored, signature: ""}).signature, "");
    assert.equal(decodeApiEntry({...stored, s: "legacy(alias)"}).signature, "legacy(alias)");
    assert.deepEqual(stored, before);
  }
});


test("new complete parameter lists preserve independent defaults annotations descriptions and choices", () => {
  const lists = [[{"name":"data","kind":"positional-or-keyword"},{"name":"variables","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"components","default":"2","kind":"keyword-only"},{"name":"n_starts","default":"3","kind":"keyword-only"},{"name":"seed","kind":"keyword-only","default":"0"},{"name":"maxiter","default":"500","kind":"keyword-only"},{"name":"tol","default":"1e-08","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_bytes","default":"c.LIMIT_BYTES","kind":"keyword-only"},{"name":"max_work","default":"c.WORK","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"positional-or-keyword"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}]];
  for (const [offset, parameters] of lists.entries()) {
    const stored: StoredApiEntry = {n: "~Fixture.call", S: "()", P: 72+offset, d: "Full help."};
    const before = structuredClone(stored);
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
    const changed = decodeApiEntry(stored);
    changed.parameters[0].description = "Local edit";
    changed.parameters.find(p => p.choices)?.choices?.push("'local'");
    assert.deepEqual(decodeApiEntry(stored).parameters, parameters);
    const explicit = [{name: "required", kind: "keyword-only", choices: [], description: "", default: ""}];
    for (const field of ["parameters", "p"]) {
      assert.deepEqual(decodeApiEntry({...stored, [field]: explicit} as StoredApiEntry).parameters, explicit);
    }
    assert.deepEqual(stored, before);
  }
});


test("categorical branch templates remain complete after the incoming compression layout", () => {
  const suffix = "(data: pd.DataFrame, variables: list[str], *, knots: dict, components: int=2, n_starts: int=3, seed: int=0, maxiter: int=500, tol: float=1e-08, missing: str='drop', max_work: int=WORK, max_bytes: int=BYTES, device: str='cpu')";
  const parameters = [[{"name":"data","kind":"positional-or-keyword"},{"name":"variables","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"components","default":"2","kind":"keyword-only"},{"name":"n_starts","default":"3","kind":"keyword-only"},{"name":"seed","kind":"keyword-only","default":"0"},{"name":"maxiter","default":"500","kind":"keyword-only"},{"name":"tol","default":"1e-08","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_bytes","default":"c.LIMIT_BYTES","kind":"keyword-only"},{"name":"max_work","default":"c.WORK","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"positional-or-keyword"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}]];
  assert.equal(signatureTokenUpperBound, 68);
  assert.equal(parameterTokenUpperBound, 80);
  const signature = { n: "~catpca_spline", I: 67, d: "Bounded spline PCA", p: [] };
  assert.deepEqual(decodeApiEntry(signature), decodeApiEntry({ ...signature, S: suffix }));
  for (const [offset, expected] of parameters.entries()) {
    const stored = { n: "~saved_spline", S: "(value)", d: "Retained help", P: 72 + offset };
    const before = structuredClone(stored);
    const decoded = decodeApiEntry(stored);
    assert.deepEqual(decoded.parameters, expected);
    assert.deepEqual(decoded, decodeApiEntry({ ...stored, parameters: expected }));
    decoded.parameters[0].name = "changed";
    assert.deepEqual(decodeApiEntry(stored).parameters, expected);
    assert.deepEqual(stored, before);
  }
});
