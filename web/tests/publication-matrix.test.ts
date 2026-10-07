import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import katex from "katex";
import { JSDOM } from "jsdom";
import {
  escapeLatex,
  latexDocument,
  outputLatex,
} from "../src/latex-export.ts";
import { latexMathOptions } from "../src/latex-math.ts";
import type { ConsoleOutput, ResultBundle } from "../src/types.ts";

const fixtures: { case: string; output: ConsoleOutput }[] = JSON.parse(
  readFileSync(
    new URL(
      "../../docs/evidence/market-110-publication/fixtures.json",
      import.meta.url,
    ),
    "utf8",
  ),
);

for (const fixture of fixtures) {
  test(`saved ${fixture.case}: native preview, copied TeX and notes retain the same reporting values`, () => {
    const output = fixture.output;
    const model = output.data as ResultBundle;
    const before = JSON.stringify(output);
    assert.equal(outputLatex(output), output.latex);
    const doc = latexDocument([output]);
    assert.ok(doc.includes(output.latex!));
    for (const note of output.latex_notes!)
      assert.ok(doc.includes(escapeLatex(note)));
    const dom = new JSDOM(
      katex.renderToString(output.latex_math!, latexMathOptions()),
    );
    const visible = dom.window.document
      .querySelector(".katex-html")
      ?.textContent?.replace(/\s+/g, "");
    assert.ok(visible);
    for (const coefficient of model.coefficients) {
      const label =
        coefficient.term === "Intercept" &&
        model.spec.intercept &&
        !model.coefficients.some((c) => c.equation != null)
          ? "Constant"
          : coefficient.term;
      assert.ok(visible!.includes(label.replace(/\s+/g, "")));
      for (const number of [coefficient.estimate, coefficient.std_error]) {
        if (Math.abs(number) >= 0.0001)
          assert.ok(visible!.includes(number.toFixed(4).replace("-", "−")));
      }
      if (coefficient.equation)
        assert.ok(
          visible!.includes(`[${coefficient.equation.replace(/\s+/g, "")}]`),
        );
    }
    assert.equal(
      dom.window.document.querySelectorAll("script,img,a").length,
      0,
    );
    assert.equal(JSON.stringify(output), before);
    dom.window.close();
  });
}

test("explicit full TeX retains rows beyond the bounded model preview", () => {
  const source = readFileSync(
    new URL(
      "../../docs/evidence/market-110-publication/stress.tex",
      import.meta.url,
    ),
    "utf8",
  );
  assert.ok(source.includes(String.raw`term\_089`));
  assert.ok(source.includes(String.raw`89 & row\_\%`));
  assert.ok(source.includes("Türkçe: ı İ ş Ş ğ Ğ ü Ü ö Ö ç Ç"));
  assert.ok(source.includes(String.raw`1.2000 \times 10^{-12}`));
  assert.ok(!source.includes(String.raw`\input{`));
});

test("legacy fallback respects recorded covariance, grouping and method notes", () => {
  const fixture = fixtures.find((item) => item.case === "xtreg_robust")!;
  const output = structuredClone(fixture.output);
  delete output.latex;
  delete output.latex_notes;
  const model = output.data as ResultBundle;
  model.spec.covariance = "nonrobust";
  const source = outputLatex(output);
  assert.ok(source.includes("robust standard errors in parentheses"));
  assert.ok(source.includes("id (36 clusters)"));
  assert.ok(source.includes("df = 35"));
  assert.ok(!source.includes("nonrobust standard errors"));
  const arima = structuredClone(
    fixtures.find((item) => item.case === "arima")!.output,
  );
  delete arima.latex;
  assert.ok(outputLatex(arima).includes("tested one-sided against zero"));
  assert.ok(outputLatex(arima).includes("{[ARMA]}"));
});
