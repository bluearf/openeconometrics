import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import {
  modelTargetLabel,
  outputLatex,
  uninferredModelRows,
} from "../src/latex-export.ts";
import type { ConsoleOutput, ResultBundle } from "../src/types.ts";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
});
for (const name of ["window", "document", "navigator", "HTMLElement"])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const sourceRoot = new URL("../src/", import.meta.url);
const urls = new Map<string, string>();

// Use the real output components and LaTeX renderer. The rest of the workbench
// does not participate in presenting a saved model.
function moduleUrl(name: string): string {
  const existing = urls.get(name);
  if (existing) return existing;
  const extension =
    name === "latex-render" || name === "latex-math" ? ".ts" : ".tsx";
  let source = readFileSync(new URL(name + extension, sourceRoot), "utf8");
  if (name === "App") {
    const tree = ts.createSourceFile(
      "App.tsx",
      source,
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TSX,
    );
    source = tree.statements
      .filter(
        (node) =>
          (ts.isFunctionDeclaration(node) &&
            ["ModelTable", "ModelOutput", "Output"].includes(
              node.name?.text ?? "",
            )) ||
          (ts.isVariableStatement(node) &&
            node.declarationList.declarations.some(
              (item) => ts.isIdentifier(item.name) && item.name.text === "fmt",
            )),
      )
      .map((node) => node.getText(tree))
      .join("\n");
    source =
      'import {Fragment} from "react";\nimport LatexView from "./LatexView";\n' +
      'import {modelTargetLabel,uninferredModelRows,publicationNumber,significanceStars} from "./latex-export";\n' +
      source +
      "\nexport { Output };\n";
  }
  const resolve = (target: string) =>
    target === "./latex-export"
      ? new URL("../src/latex-export.ts", import.meta.url).href
      : target.startsWith(".")
        ? moduleUrl(target.replace("./", ""))
        : pathToFileURL(require.resolve(target)).href;
  let output = ts
    .transpileModule(source, {
      fileName: name + extension,
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    })
    .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "");
  output = output
    .replace(
      /(from\s*|import\s*)(["'])([^"']+)\2/g,
      (_whole, prefix, _quote, target: string) =>
        prefix + JSON.stringify(resolve(target)),
    )
    .replace(
      /import\((["'])([^"']+)\1\)/g,
      (_whole, _quote, target: string) =>
        `import(${JSON.stringify(resolve(target))})`,
    );
  const url =
    "data:text/javascript;base64," + Buffer.from(output).toString("base64");
  urls.set(name, url);
  return url;
}
const { Output } = await import(moduleUrl("App"));

const predictiveModel: ResultBundle = {
  id: "prediction-fixture",
  created_at: "2026-10-07T00:00:00Z",
  dataset_id: "source-owned-fixture",
  dataset_name: "Prediction fixture",
  spec: {
    estimator: "ridge",
    outcome: "y",
    predictors: ["x_%"],
    categorical: [],
    intercept: true,
    covariance: "nonrobust",
    cluster: null,
    missing: "raise",
    alpha: 0.05,
  },
  nobs: 12,
  nobs_original: 12,
  dropped_rows: 0,
  coefficients: [],
  covariance_matrix: [],
  metrics: { training_mse: 0.125 },
  warnings: [],
  predictions: [],
  sample_positions: [],
  provenance: {},
  inference: {
    available: false,
    target: "prediction",
    covariance: "nonrobust",
  },
  extra: {
    target: "prediction",
    terms: ["x_%"],
    penalized_state: { constant: 2.25, coefficients: [1.375] },
  },
};

async function render(output: ConsoleOutput) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  await React.act(async () =>
    root.render(React.createElement(Output, { output })),
  );
  for (
    let attempt = 0;
    attempt < 20 &&
    output.latex_math &&
    host.querySelector(".latex-preview")?.hasAttribute("hidden");
    attempt++
  )
    await React.act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  return {
    host,
    close: async () => {
      await React.act(async () => root.unmount());
      host.remove();
    },
  };
}

for (const publication of [false, true]) {
  for (const math of [
    undefined,
    String.raw`\begin{array}{lr}\text{Intercept}&2.2500\\\text{x\_\%}&1.3750\end{array}`,
    String.raw`\invalidMathForFallback`,
  ]) {
    test(`prediction React output ${publication ? "publication" : "diagnostic"}, ${math === undefined ? "without math" : math.includes("invalid") ? "invalid math fallback" : "native math"} has actual values without inference fiction`, async () => {
      const output: ConsoleOutput = {
        type: "model",
        data: structuredClone(predictiveModel),
        ...(publication ? { latex_style: "publication-v1" } : {}),
        ...(math ? { latex_math: math } : {}),
      };
      const before = JSON.stringify(output);
      const view = await render(output);
      try {
        assert.match(
          view.host.querySelector(".model-metrics")!.textContent!,
          /Prediction target/,
        );
        const visible =
          view.host.querySelector(".katex-html")?.textContent ??
          view.host.querySelector("table")?.textContent;
        assert.ok(visible?.includes("2.25"));
        assert.ok(visible?.includes("1.375"));
        assert.ok(visible?.includes("x_%"));
        assert.doesNotMatch(
          view.host.textContent ?? "",
          /standard errors|Std\. error|p-value|% CI|\*\*\*/,
        );
        assert.equal(
          view.host.querySelectorAll("sup,.publication-standard-error").length,
          0,
        );
        assert.equal(JSON.stringify(output), before);
      } finally {
        await view.close();
      }
    });
  }
}

test("prediction export fallback preserves saved parameters and safely escaped names", () => {
  const output: ConsoleOutput = {
    type: "model",
    data: structuredClone(predictiveModel),
  };
  const before = JSON.stringify(output);
  const tex = outputLatex(output);
  assert.match(tex, /Prediction target/);
  assert.ok(tex.includes(String.raw`x\_\% & $1.3750$`));
  assert.ok(tex.includes("Intercept & $2.2500$"));
  assert.doesNotMatch(tex, /standard errors|p<|p-value|CI|\^\{|\*\*\*/);
  assert.equal(JSON.stringify(output), before);
});

test("local query and fixed evaluation fallbacks retain their actual saved values", async () => {
  const local = structuredClone(predictiveModel);
  local.spec.estimator = "localreg";
  local.extra = {
    target: "prediction",
    terms: ["x_%"],
    query: [[0.25]],
    query_estimates: [3.125],
  };
  assert.deepEqual(uninferredModelRows(local), [
    ["Query 1 (x_%=0.2500)", 3.125],
  ]);
  assert.ok(outputLatex({ type: "model", data: local }).includes("$3.1250$"));
  const fixed = structuredClone(predictiveModel);
  fixed.spec.estimator = "sspace";
  fixed.extra = { system: { transition: [[0.7]] } };
  fixed.inference = { available: false, covariance: "nonrobust" };
  fixed.predictions = [
    { row: 3, observed: 1.1, fitted: 0.875, residual: 0.225 },
  ];
  assert.equal(modelTargetLabel(fixed), "Fixed parameter evaluation");
  assert.deepEqual(uninferredModelRows(fixed), [["Row 3", 0.875]]);
  const view = await render({ type: "model", data: fixed });
  try {
    assert.match(view.host.textContent ?? "", /Fixed parameter evaluation/);
    assert.match(view.host.textContent ?? "", /0\.875/);
    assert.doesNotMatch(
      view.host.textContent ?? "",
      /standard errors|Std\. error|p-value|% CI|\*\*\*/,
    );
  } finally {
    await view.close();
  }
});

test("available-inference model keeps its recorded covariance and standard-error rows", async () => {
  const model = structuredClone(predictiveModel);
  model.spec.estimator = "ols";
  model.inference = { covariance: "HC3", available: true };
  model.coefficients = [
    {
      term: "x",
      estimate: 1.375,
      std_error: 0.2,
      statistic: 6.875,
      p_value: 0.0001,
      ci_low: 0.983,
      ci_high: 1.767,
    },
  ];
  model.extra = {};
  const view = await render({
    type: "model",
    data: model,
    latex_style: "publication-v1",
  });
  try {
    assert.match(view.host.textContent ?? "", /HC3 standard errors/);
    assert.equal(
      view.host.querySelectorAll(".publication-standard-error").length,
      1,
    );
    assert.match(view.host.querySelector("sup")!.textContent!, /\*\*\*/);
  } finally {
    await view.close();
  }
});
