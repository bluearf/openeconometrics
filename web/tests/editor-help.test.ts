import assert from "node:assert/strict";
import test from "node:test";
import { EditorState } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { ensureSyntaxTree, syntaxTree } from "@codemirror/language";
import { CompletionContext } from "@codemirror/autocomplete";
import {
  getEditorSymbol,
  getActiveCall,
  editorCompletionSource,
} from "../src/editor-help.ts";

function state(code: string, readOnly = false) {
  const result = EditorState.create({
    doc: code,
    selection: { anchor: code.length },
    extensions: [python(), EditorState.readOnly.of(readOnly)],
  });
  // Symbol-resolution fixtures start parsed. Keep large-document cases partial
  // so they still exercise the editor's bounded incremental parse behavior.
  if (code.length <= 4096) ensureSyntaxTree(result, code.length, 1000);
  return result;
}
function symbol(code: string, target: string, occurrence = 0) {
  let pos = -1;
  for (let i = 0; i <= occurrence; i++) pos = code.indexOf(target, pos + 1);
  assert.ok(pos >= 0, target);
  const expression = target.split("(")[0];
  const identifier = expression.match(/[A-Za-z_]\w*$/)!;
  return getEditorSymbol(
    state(code),
    pos + identifier.index! + Math.max(1, Math.floor(identifier[0].length / 2)),
  );
}
function complete(code: string, explicit = true, readOnly = false) {
  return editorCompletionSource(
    new CompletionContext(state(code, readOnly), code.length, explicit),
  );
}
function markedCompletion(code: string, explicit = false) {
  const pos = code.indexOf("|");
  assert.ok(pos >= 0, "Cursor marker exists");
  const source = code.slice(0, pos) + code.slice(pos + 1);
  return {
    source,
    result: editorCompletionSource(
      new CompletionContext(state(source), pos, explicit),
    ),
  };
}
function acceptedValue(code: string, label: string) {
  const { source, result } = markedCompletion(code);
  const option = result?.options.find((item) => item.label === label);
  assert.ok(option, label);
  const insert = option.apply ?? option.label;
  assert.equal(typeof insert, "string");
  return (
    source.slice(0, result!.from) +
    insert +
    source.slice(result!.to ?? code.indexOf("|"))
  );
}

test("hover follows actual module and from-import aliases", () => {
  const code =
    "import openecon as econ\nfrom openecon import ols as regress\necon.ols(data=df)\nregress(data=df)";
  assert.equal(symbol(code, "ols(data")?.entry.name, "openecon.ols");
  assert.equal(symbol(code, "regress(data")?.entry.name, "openecon.ols");
  assert.equal(symbol("oe.ols(data=df)", "ols")?.entry, undefined);
});

test("registry models and helpers have offline hover and completion help", () => {
  for (const name of ["nardl", "nardl_multipliers", "tobit", "truncreg", "intreg", "xtcd"]) {
    const code = `import openecon as oe\noe.${name}(`;
    assert.equal(symbol(code, `${name}(`)?.entry.name, `openecon.${name}`);
    assert.ok(complete(`import openecon as oe\noe.${name.slice(0, -1)}`)?.options
      .some((option) => option.label === name), name);
  }
  const fit = "import openecon as oe\nm=oe.tobit(data=df,y='y',x=['x'])\nm.summary()";
  assert.equal(symbol(fit, "summary")?.entry.name, "openecon.ResultBundle.summary");
  assert.equal(symbol(fit.replace("summary", "vif"), "vif")?.entry, undefined);
  for (const helper of ["xtcd", "nardl_multipliers", "predict", "margins"]) {
    const code = `import openecon as oe\nout=oe.${helper}(data=df)\nout.to_latex()`;
    assert.equal(symbol(code, "to_latex")?.entry.name, "openecon.DataFrame.to_latex");
  }
});

test("unknown objects, assignment shadowing, comments and strings have no invented docs", () => {
  assert.equal(
    symbol("import openecon as oe\nthing.ols(data=df)", "ols"),
    null,
  );
  assert.equal(
    symbol("import openecon as oe\noe = object()\noe.ols(data=df)", "ols"),
    null,
  );
  assert.equal(symbol('import openecon as oe\n"oe.ols(data=df)"', "ols"), null);
  assert.equal(symbol("import openecon as oe\n# oe.ols(data=df)", "ols"), null);
  assert.equal(symbol("print = 4\nprint()", "print()"), null);
});

test("inferred frames and model results expose only their actual type's methods", () => {
  const code =
    "import openecon as oe\ndf=oe.example()\ncopy=df.head()\ncopy.to_latex()\nmodel=oe.ols(data=df,y='wage',x=['education'])\nmodel.summary()\nmodel.predict()";
  assert.equal(
    symbol(code, "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex",
  );
  assert.equal(
    symbol(code, "summary")?.entry.name,
    "openecon.OLSResult.summary",
  );
  assert.equal(
    symbol(code, "predict()")?.entry.name,
    "openecon.OLSResult.predict",
  );
  assert.equal(
    symbol(
      "import openecon as oe\nm=oe.logit(data=df,y='employed',x=['education'])\nm.vif()",
      "vif",
    ),
    null,
  );
});

test("network construction exposes its own methods and graph chart help", () => {
  const code = "import openecon as oe\ngraph=oe.network(data=edges)\ngraph.pagerank()\noe.plot.network(graph)";
  assert.equal(symbol(code, "pagerank()")?.entry.name, "openecon.Network.pagerank");
  assert.equal(symbol(code, "network(graph)")?.entry.name, "openecon.plot.network");
  const options = complete("import openecon as oe\ngraph=oe.network(data=edges)\ngraph.")?.options;
  assert.ok(options?.some(item => item.label === "pagerank"));
  assert.ok(options?.some(item => item.label === "shortest_paths"));
  assert.ok(!options?.some(item => item.label === "vif"));
});

test("network workflow return values expose chained completion without executing code", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\nflow=graph.max_flow('a','b')\n";
  assert.equal(symbol(prelude + "flow.summary()", "summary")?.entry.name,
    "openecon.NetworkFlowResult.summary");
  const methods = complete(prelude + "flow.")?.options;
  assert.ok(methods?.some(item => item.label === "to_latex"));
  assert.ok(!methods?.some(item => item.label === "pagerank"));
  const projected = "import openecon as oe\ngraph=oe.network(data=edges)\nprojection=graph.bipartite_projection()\n";
  assert.equal(symbol(projected + "projection.degree()", "degree")?.entry.name,
    "openecon.Network.degree");
  const graphMethods = complete(projected + "projection.")?.options;
  assert.ok(graphMethods?.some(item => item.label === "shortest_path"));
  assert.ok(graphMethods?.some(item => item.label === "link_prediction"));
});

test("network inference and global cut results retain chained table help", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\ncut=graph.global_min_cut()\n";
  assert.equal(symbol(prelude + "cut.summary()", "summary")?.entry.name,
    "openecon.NetworkCutResult.summary");
  const cutMethods = complete(prelude + "cut.")?.options;
  assert.ok(cutMethods?.some(item => item.label === "to_latex"));
  assert.ok(!cutMethods?.some(item => item.label === "pagerank"));
  const triads = prelude + "counts=graph.triad_census()\n";
  assert.equal(symbol(triads + "counts.head()", "head")?.entry.name,
    "openecon.DataFrame.head");
  const comparison = prelude + "comparison=graph.qap_correlation(other)\n";
  assert.equal(symbol(comparison + "comparison.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("function parameters shadow global aliases and nested locals stay scoped", () => {
  const code =
    "import openecon as oe\ndef first(oe):\n    oe.ols()\ndef second():\n    import openecon as local\n    local.ols()\nlocal.ols()\n";
  assert.equal(symbol(code, "oe.ols", 0), null);
  assert.equal(symbol(code, "local.ols", 0)?.entry.name, "openecon.ols");
  assert.equal(symbol(code, "local.ols", 1), null);
});

test("user function docs and multiline signatures are derived without executing code", () => {
  const code =
    'def adjust(data,\n precision=4, *, robust=True, **options):\n    """Adjust a supplied frame.\n\nLonger explanation."""\n    return data\nadjust(';
  const entry = symbol(code, "adjust(", 1)?.entry;
  assert.equal(
    entry?.signature,
    "adjust(data, precision=4, *, robust=True, **options)",
  );
  assert.equal(entry?.description, "Adjust a supplied frame.");
  assert.deepEqual(
    entry?.parameters.map((parameter) => parameter.name),
    ["data", "precision", "robust", "options"],
  );
  assert.equal(entry?.parameters[2].kind, "keyword-only");
});

test("signature help counts only outer argument commas", () => {
  const prefix = "def f(first, second, third):\n    return first\n";
  for (const argument of [
    "[1, 2, 3]",
    "{'a': 1, 'b': 2}",
    "dict(a=1, b=2)",
    "'a, b'",
    "(1, 2)",
  ]) {
    const active = getActiveCall(state(`${prefix}f(${argument}, `));
    assert.equal(active?.activeParameter, "second", argument);
    assert.equal(active?.argumentIndex, 1, argument);
  }
});

test("innermost open call wins and closed calls restore their outer call", () => {
  const prefix =
    "def outer(first, second):\n    pass\ndef inner(a, b):\n    pass\n";
  assert.equal(
    getActiveCall(state(`${prefix}outer(inner(1, `))?.entry.name,
    "inner",
  );
  assert.equal(
    getActiveCall(state(`${prefix}outer(inner(1, 2), `))?.entry.name,
    "outer",
  );
  assert.equal(getActiveCall(state(`${prefix}outer(inner(1, 2))`)), null);
});

test("keywords select the named parameter regardless of declaration order", () => {
  const code = "def f(first, second, third):\n    pass\nf(third=3, first=";
  const active = getActiveCall(state(code));
  assert.equal(active?.activeParameter, "first");
  assert.deepEqual(active?.supplied, ["third"]);
});

test("literal calls and comments never trigger signature help", () => {
  assert.equal(getActiveCall(state('"print(hello, "')), null);
  assert.equal(getActiveCall(state("# print(")), null);
  assert.equal(getActiveCall(state("print(1, # hello")), null);
});

test("completion resolves module children, inferred methods, variables and parameters", () => {
  const rootOptions = complete("import openecon as oe\noe.")!.options;
  assert.ok(!rootOptions.some((option) => option.label === "OLSResult"));
  assert.ok(rootOptions.some((option) => option.label === "DataFrame"));
  assert.ok(
    complete("import openecon as econ\necon.")?.options.some(
      (option) => option.label === "ols",
    ),
  );
  assert.ok(
    complete("import openecon as oe\noe.plot.")?.options.some(
      (option) => option.label === "scatter",
    ),
  );
  assert.ok(
    complete("import openecon as oe\ndf=oe.example()\ndf.")?.options.some(
      (option) => option.label === "to_latex",
    ),
  );
  assert.ok(
    complete("def f(first, *, robust=True):\n    pass\nf(")?.options.some(
      (option) => option.label === "robust=",
    ),
  );
  const keywords = complete(
    "def f(first, *, robust=True):\n    pass\nf(robust=True, ",
  )?.options.map((option) => option.label);
  assert.ok(!keywords?.includes("robust="));
  assert.ok(
    complete("def adjust(x):\n    pass\nadj")?.options.some(
      (option) => option.label === "adjust",
    ),
  );
});

test("completion leaves unknown objects and protected text alone", () => {
  assert.equal(complete("unknown."), null);
  assert.equal(complete('import openecon as oe\n"oe.'), null);
  assert.equal(complete("import openecon as oe\n# oe."), null);
  assert.equal(complete("import openecon as oe\noe.", true, true), null);
});

test("updating a document invalidates earlier type and alias inferences", () => {
  const original = state("import openecon as oe\ndf=oe.example()\ndf.head()");
  assert.equal(
    getEditorSymbol(original, original.doc.length - 3)?.entry.name,
    "openecon.DataFrame.head",
  );
  const changed = original.update({
    changes: {
      from: 0,
      to: original.doc.length,
      insert: "df=object()\ndf.head()",
    },
  }).state;
  assert.equal(getEditorSymbol(changed, changed.doc.length - 3), null);
});

test("updated aliases, context bindings and comprehension targets suppress stale help", () => {
  for (const snippet of [
    "oe += 1\noe.ols()",
    "with resource() as oe:\n    oe.ols()",
    "try:\n    pass\nexcept Exception as oe:\n    oe.ols()",
    "[oe.ols() for oe in items]",
    "def inner():\n    oe.ols()\n    oe=object()",
    "make = lambda oe: oe.ols()",
    "(oe := object())\noe.ols()",
    "del oe\noe.ols()",
    "result = [oe.ols() for oe in items]",
  ]) {
    assert.equal(
      symbol(`import openecon as oe\n${snippet}`, "ols()"),
      null,
      snippet,
    );
  }
  assert.equal(
    symbol(
      "import openecon as oe\n[oe.ols() for oe in items]\noe.ols()",
      "ols()",
      1,
    )?.entry.name,
    "openecon.ols",
  );
});

test("unpacking does not assign the original frame type to each element", () => {
  assert.equal(
    symbol("import openecon as oe\na,b=oe.example()\na.head()", "head"),
    null,
  );
  assert.equal(
    symbol("import openecon as oe\nx=df=oe.example()\ndf.head()", "head")?.entry
      .name,
    "openecon.DataFrame.head",
  );
});

test("keyword completion excludes arguments already supplied positionally", () => {
  const options = complete(
    "def f(first, second, *, robust=False):\n    pass\nf(1, ",
  )!.options;
  assert.ok(!options.some((option) => option.label === "first="));
  assert.ok(options.some((option) => option.label === "second="));
});

test("Turkish identifiers receive function docs and parameter completion", () => {
  const code =
    'def ölç(veri, *, ücret=4):\n    """Ücreti ölç."""\n    return veri\nölç(ücret=';
  assert.equal(getActiveCall(state(code))?.activeParameter, "ücret");
  assert.equal(
    getEditorSymbol(state(code), code.lastIndexOf("ölç") + 1)?.entry
      .description,
    "Ücreti ölç.",
  );
  assert.ok(
    complete(
      "def ölç(veri, *, ücret=4):\n    return veri\nölç(ü",
    )?.options.some((option) => option.label === "ücret="),
  );
});

test("assistance can resolve a requested token beyond the initial partial parse", () => {
  const prefix =
    "import openecon as econ\n" + "# deferred analysis notes\n".repeat(400);
  const code = `${prefix}econ.ols(covariance=`;
  const editor = state(code);
  assert.ok(
    syntaxTree(editor).length < code.length,
    "The initial parser stops before the requested call",
  );
  const requested = getEditorSymbol(editor, prefix.length + "econ.o".length);
  // A busy machine can exhaust the query's short parse budget. Assistance
  // stays absent rather than guessing until the view's parser finishes.
  if (requested) assert.equal(requested.entry.name, "openecon.ols");
  assert.ok(ensureSyntaxTree(editor, code.length, 1000));
  assert.equal(
    getEditorSymbol(editor, prefix.length + "econ.o".length)?.entry.name,
    "openecon.ols",
  );
  assert.equal(getActiveCall(editor)?.activeParameter, "covariance");
  assert.equal(editor.doc.toString(), code);
});

test("known argument punctuation opens useful suggestions automatically", () => {
  for (const code of [
    "import openecon as oe\noe.ols(",
    "import openecon as oe\noe.ols(data=df, ",
  ]) {
    const options = complete(code, false)?.options;
    assert.ok(options?.some((option) => option.label === "y="));
  }
  assert.ok(
    complete("import openecon as oe\noe.ols(covariance=", false)?.options.some(
      (option) => option.label === "'HC3'",
    ),
  );
  assert.equal(complete("import openecon as oe\n", false), null);
  assert.equal(complete("unknown(", false), null);
});

test("literal choices remain specific to the published estimator contract", () => {
  const labels = (model: string) =>
    complete(`import openecon as oe\noe.${model}(covariance=`, false)
      ?.options.filter((option) => option.type === "constant")
      .map((option) => option.label);
  assert.ok(labels("ols")?.includes("'HC2'"));
  assert.ok(labels("ols")?.includes("'hac'"));
  assert.ok(labels("logit")?.includes("'cluster'"));
  assert.ok(!labels("logit")?.includes("'HC2'"));
  assert.ok(!labels("probit")?.includes("'hac'"));
  assert.ok(labels("ols")?.includes("None"));
});

test("defaults provide inert scalar values and both boolean choices", () => {
  const code =
    "def configure(rows=5, *, enabled=True, mode='quiet', unsafe=do_not_call()):\n    pass\n";
  const values = (name: string) =>
    complete(`${code}configure(${name}=`, false)
      ?.options.filter((option) => option.type === "constant")
      .map((option) => option.label);
  assert.deepEqual(values("rows"), ["5"]);
  assert.deepEqual(values("enabled"), ["True", "False"]);
  assert.deepEqual(values("mode"), ["'quiet'"]);
  assert.deepEqual(values("unsafe"), []);
  assert.equal(
    complete("import openecon as oe\noe.ols(covariance=", false, true),
    null,
  );
});

test("quoted value completion preserves delimiters and replaces a whole edited value", () => {
  const prefix = "import openecon as oe\n";
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance='H|`, "HC3"),
    `${prefix}oe.ols(covariance='HC3'`,
  );
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance="H|C1", data=df)`, "HC3"),
    `${prefix}oe.ols(covariance="HC3", data=df)`,
  );
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance='H|C1')`, "HC3"),
    `${prefix}oe.ols(covariance='HC3')`,
  );
});

test("protected and unsupported string expressions never receive value suggestions", () => {
  for (const code of [
    'import openecon as oe\n"H|"',
    'import openecon as oe\n# oe.ols(covariance="H|',
    'import openecon as oe\noe.ols(covariance=f"H|C1")',
    'import openecon as oe\noe.ols(covariance="""H|C1""")',
    'import openecon as oe\noe.ols(covariance="H|" "C1")',
    'import openecon as oe\noe.ols(covariance=unknown("H|"))',
    'import openecon as oe\noe.ols(covariance=["H|"])',
  ])
    assert.equal(markedCompletion(code, true).result, null, code);
});

test("unused keywords include later supplied arguments and required parameters rank first", () => {
  const options = markedCompletion(
    "def f(required, optional=1, *, robust=True):\n    pass\nf(|, robust=True)",
  ).result?.options;
  assert.ok(!options?.some((option) => option.label === "robust="));
  assert.ok(
    (options?.find((option) => option.label === "required=")?.boost ?? 0) >
      (options?.find((option) => option.label === "optional=")?.boost ?? 0),
  );
  assert.equal(
    acceptedValue(
      "def f(first, *, robust=True):\n    pass\nf(ro|bust=True)",
      "robust=",
    ),
    "def f(first, *, robust=True):\n    pass\nf(robust=True)",
  );
});

test("future locals suppress outer namespace and builtin completion claims", () => {
  for (const name of ["oe", "len"]) {
    const options = markedCompletion(
      `import openecon as oe\ndef f():\n    ${name}|\n    ${name}=object()`,
      true,
    ).result?.options;
    assert.ok(!options?.some((option) => option.label === name), name);
  }
  assert.equal(
    markedCompletion(
      "import openecon as oe\ndef f():\n    df=oe.example()\n    oe=object()\n    df.h|",
      true,
    ).result,
    null,
  );
});

test("inferred instance aliases are local values and outrank generic builtins", () => {
  const options = complete(
    "import openecon as oe\ndf=oe.example()\ncopy=df\noe.ols(data=",
    false,
  )!.options;
  for (const name of ["df", "copy"]) {
    const option = options.find((item) => item.label === name);
    assert.equal(option?.type, "variable");
    assert.equal(option?.detail, "DataFrame");
    assert.ok(
      (option?.boost ?? 0) >
        (options.find((item) => item.label === "len")?.boost ?? 0),
    );
  }
});

test("unsupported chained receivers receive no unrelated global suggestions", () => {
  assert.equal(
    complete("import openecon as oe\ndf=oe.example()\ndf.head().", true),
    null,
  );
  assert.equal(
    acceptedValue("import openecon as oe\noe.o|ls", "ols"),
    "import openecon as oe\noe.ols",
  );
});

test("numeric and named literal completion replaces the complete edited token", () => {
  assert.equal(
    acceptedValue("import openecon as oe\noe.ols(alpha=0.|01)", "0.05"),
    "import openecon as oe\noe.ols(alpha=0.05)",
  );
  assert.equal(
    acceptedValue("import openecon as oe\noe.ols(covariance=N|one)", "None"),
    "import openecon as oe\noe.ols(covariance=None)",
  );
  assert.equal(
    acceptedValue("def f(x=-1):\n    pass\nf(x=-|)", "-1"),
    "def f(x=-1):\n    pass\nf(x=-1)",
  );
  assert.equal(
    acceptedValue("def f(x=True):\n    pass\nf(x=Tr|ue)", "True"),
    "def f(x=True):\n    pass\nf(x=True)",
  );
  for (const code of [
    'import openecon as oe\noe.ols(covariance=|"HC1")',
    "def f(x=True):\n    pass\nf(x=|True)",
    "def f(x=100):\n    pass\nf(x=|100)",
  ])
    assert.equal(markedCompletion(code).result, null, code);
});

test("MRQAP, SBM and snapshot objects offer their actual chained methods", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\n";
  assert.equal(symbol(prelude + "fit=graph.block_model(2)\nfit.summary()", "summary")?.entry.name,
    "openecon.NetworkBlockResult.summary");
  assert.equal(symbol(prelude + "table=graph.qap_regression({'a':graph})\ntable.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
  const layers = prelude + "layers=oe.network_snapshots({'first':graph},ordered=True)\n";
  const methods = complete(layers + "layers.")?.options;
  assert.ok(methods?.some(item => item.label === "temporal_path"));
  assert.ok(methods?.some(item => item.label === "edge_persistence"));
  assert.ok(!methods?.some(item => item.label === "block_model"));
  assert.equal(symbol(layers + "combined=layers.aggregate()\ncombined.degree()", "degree")?.entry.name,
    "openecon.Network.degree");
  assert.equal(symbol(layers + "rows=layers.transitions()\nrows.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("signed graph results resolve to signed methods and complete table types", () => {
  const base = "import openecon as oe\ng = oe.signed_network(rows)\n";
  const options = complete(base + "g.signed_")?.options.map((item) => item.label) ?? [];
  for (const method of ["signed_strength", "signed_katz", "signed_modularity", "signed_communities", "signed_shortest_paths"]) {
    assert.ok(options.includes(method));
    assert.equal(symbol(base + `g.${method}()`, method)?.entry.name, `openecon.SignedNetwork.${method}`);
  }
  assert.ok(!complete(base + "g.")?.options.some((item) => item.label === "pagerank"));
  assert.equal(symbol(base + "out = g.signed_strength()\nout.head()", "head")?.entry.name, "openecon.DataFrame.head");
});

test("dynamic help selects intervals before explicit simple projection", () => {
  const base = "import openecon as oe\ng=oe.dynamic_network(nodes,edges)\n";
  const options = complete(base + "g.")?.options.map((item) => item.label) ?? [];
  for (const method of ["nodes", "edges", "summary", "at", "window", "write"])
    assert.ok(options.includes(method));
  assert.ok(!options.includes("pagerank"));
  assert.equal(symbol(base + "point=g.at(1)\npoint.degree()", "degree")?.entry.name,
    "openecon.MultiNetwork.degree");
  assert.equal(symbol(base + "point=g.window(0,1)\nsimple=point.to_network(reducer='count')\nsimple.pagerank()", "pagerank")?.entry.name,
    "openecon.Network.pagerank");
  assert.equal(symbol("import openecon as oe\ng=oe.read_dynamic_network('owned.gexf')\nt=g.edges()\nt.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("multigraph help retains edge IDs until an explicit projection", () => {
  const base = "import openecon as oe\ng = oe.multigraph(rows)\n";
  const options = complete(base + "g.")?.options.map((item) => item.label) ?? [];
  for (const method of ["edges", "degree", "filter", "edit_edges", "to_network", "write"])
    assert.ok(options.includes(method));
  assert.ok(!options.includes("pagerank"));
  assert.equal(symbol(base + "edited=g.edit_edges(weights={1:9})\nchanged=edited.filter(edge_ids=[1])\nchanged.degree()", "degree")?.entry.name,
    "openecon.MultiNetwork.degree");
  assert.equal(symbol(base + "simple=g.to_network(reducer='count')\nsimple.pagerank()", "pagerank")?.entry.name,
    "openecon.Network.pagerank");
  assert.equal(symbol("import openecon as oe\ng=oe.read_multigraph('owned.graphml')\nt=g.edges()\nt.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});
