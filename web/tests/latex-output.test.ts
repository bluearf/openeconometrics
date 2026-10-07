import assert from "node:assert/strict";
import test from "node:test";
import katex from "katex";
import { JSDOM } from "jsdom";
import {
  escapeLatex,
  latexDocument,
  outputLatex,
  publicationNumber,
  significanceStars,
} from "../src/latex-export.ts";
import { latexMathOptions, validateMathSource } from "../src/latex-math.ts";

test("standalone results preserve native table, model and PGFPlots fragments", () => {
  const fragments = [
    String.raw`\begin{tabular}{lr}Ücret & 1.234\\\end{tabular}`,
    String.raw`\begin{tikzpicture}\begin{axis}\addplot coordinates {(1,2)};\end{axis}\end{tikzpicture}`,
  ];
  const source = latexDocument(
    fragments.map((latex) => ({ type: "text", data: "ignored", latex })),
  );
  assert.ok(source.startsWith("% !TEX program = xelatex\n"));
  assert.ok(source.includes("\\usepackage{fontspec}"));
  assert.ok(source.includes("graphicx,longtable}"));
  assert.ok(source.includes("\\usepackage{adjustbox}"));
  assert.ok(source.includes("\\usepackage[font=small,labelfont=bf]{caption}"));
  assert.ok(source.includes("\\usepackage{placeins}"));
  assert.equal((source.match(/\\FloatBarrier/g) ?? []).length, 1);
  assert.ok(source.indexOf(fragments[0]) < source.indexOf("\\FloatBarrier"));
  assert.ok(source.indexOf("\\FloatBarrier") < source.indexOf(fragments[1]));
  assert.ok(!source.includes("\\setmainfont"));
  assert.ok(source.includes("\\usepackage{tikz,pgfplots}"));
  assert.ok(fragments.every((fragment) => source.includes(fragment)));
  assert.ok(!source.includes("ignored"));
  assert.ok(source.endsWith("\\end{document}\n"));
});

test("old table records export real indexed values with safe TeX escaping and truncation", () => {
  const source = outputLatex({
    type: "table",
    data: {
      columns: ["wage_%", "<script>"],
      rows: [[1e-12, "A&B"]],
      index: [["k_1"]],
      index_names: ["worker_id"],
      total_rows: 10,
      total_columns: 2,
    },
  });
  assert.ok(source.includes(String.raw`worker\_id & wage\_\% & <script>`));
  assert.ok(source.includes(String.raw`k\_1 & 1e-12 & A\&B`));
  assert.ok(source.includes("First 1 row and 2 columns."));
  assert.ok(
    source.includes(String.raw`\begin{adjustbox}{max width=\linewidth}`),
  );
  assert.ok(!source.includes(String.raw`\resizebox{\linewidth}`));
  assert.equal(
    escapeLatex(String.raw`\input{secret}#$&_%~^`),
    String.raw`\textbackslash{}input\{secret\}\#\$\&\_\%\textasciitilde{}\textasciicircum{}`,
  );
});

test("legacy models export publication rows using actual coefficients, standard errors and p-values", () => {
  const source = outputLatex({
    type: "model",
    data: {
      spec: {
        estimator: "ols",
        outcome: "wage_%",
        covariance: "HC3",
        cluster: null,
      },
      nobs: 480,
      dropped_rows: 2,
      coefficients: [
        {
          term: "education_years",
          estimate: 1.25,
          std_error: 0.125,
          p_value: 0.009,
        },
        { term: "small", estimate: 1.2e-12, std_error: 4.8e-14, p_value: 0.04 },
        { term: "experience", estimate: -0.5, std_error: 0.3, p_value: 0.11 },
      ],
      metrics: {
        r_squared: 0.73,
        adjusted_r_squared: 0.71,
        not_a_fit_statistic: 99,
      },
      warnings: [String.raw`A&B: \input{secret}`],
    },
  });
  assert.ok(source.includes(String.raw`\begin{tabular}{@{}lc@{}}`));
  assert.ok(source.includes(" & (1) \\\\"));
  assert.ok(source.includes(String.raw`wage\_\%`));
  assert.ok(source.includes("education\\_years & ${1.2500}^{***}$"));
  assert.ok(source.includes(String.raw` & ($0.1250$) \\[3pt]`));
  assert.ok(source.includes("small & ${1.2000 \\times 10^{-12}}^{**}$"));
  assert.ok(source.includes(String.raw`Observations & 480 \\`));
  assert.ok(source.includes(String.raw`$R^{2}$ & $0.7300$`));
  assert.ok(source.includes(String.raw`Adjusted $R^{2}$ & $0.7100$`));
  assert.ok(!source.includes("not_a_fit_statistic"));
  assert.ok(source.includes("HC3 standard errors in parentheses."));
  assert.ok(source.includes("2 observations excluded from estimation."));
  assert.ok(source.includes(String.raw`\begin{minipage}{\linewidth}`));
  assert.ok(source.includes(String.raw`\footnotesize\raggedright`));
  assert.ok(source.includes(String.raw`A\&B: \textbackslash{}input\{secret\}`));
  assert.ok(!source.includes(String.raw`\input{secret}`));
});

test("star thresholds are strict and missing p-values do not acquire stars", () => {
  for (const [value, expected] of [
    [0, "***"],
    [0.00999, "***"],
    [0.01, "**"],
    [0.04999, "**"],
    [0.05, "*"],
    [0.09999, "*"],
    [0.1, ""],
    [1, ""],
    [-1, ""],
    [NaN, ""],
    [Infinity, ""],
    [null, ""],
    ["0.001", ""],
  ] as [unknown, string][])
    assert.equal(significanceStars(value), expected);
  assert.equal(publicationNumber(1.2e-12), "1.2000e-12");
  assert.equal(publicationNumber(-1.2e-12), "-1.2000e-12");
  assert.equal(publicationNumber(0), "0.0000");
  assert.equal(publicationNumber(0.00001234), "1.2340e-5");
  assert.equal(publicationNumber(0.12345678), "0.1235");
  assert.equal(publicationNumber(null), "—");
  assert.equal(publicationNumber(undefined), "—");
});

test("legacy text exports preserve lines and do not turn data into TeX commands", () => {
  const source = outputLatex({ type: "text", data: "Ücret\n\\input{secret}" });
  assert.ok(source.includes("\\noindent Ücret\\par\n"));
  assert.ok(source.includes(String.raw`\textbackslash{}input\{secret\}`));
  assert.ok(!source.includes(String.raw`\input{secret}`));
});

test("long legacy tables preserve all rows in a page-breaking longtable with repeated headers", () => {
  const rows = Array.from({ length: 80 }, (_, index) => [
    `person_${index}`,
    index + 1,
  ]);
  const source = outputLatex({
    type: "table",
    data: {
      columns: ["person", "value"],
      rows,
      total_rows: 80,
      total_columns: 2,
    },
  });
  assert.ok(source.includes(String.raw`\begin{longtable}`));
  assert.ok(source.includes(String.raw`\endfirsthead`));
  assert.ok(source.includes(String.raw`\endhead`));
  assert.ok(!source.includes(String.raw`\begin{adjustbox}`));
  assert.ok(rows.every(([name]) => source.includes(escapeLatex(name))));
  assert.ok(source.includes(String.raw`person\_79 & 80 \\`));
});

test("automatic legacy page breaks start after 40 body rows", () => {
  const table = (length: number) =>
    outputLatex({
      type: "table",
      data: {
        columns: ["value"],
        rows: Array.from({ length }, (_, index) => [index]),
        total_rows: length,
        total_columns: 1,
      },
    });
  assert.ok(table(40).includes(String.raw`\begin{adjustbox}`));
  assert.ok(!table(40).includes(String.raw`\begin{longtable}`));
  assert.ok(table(41).includes(String.raw`\begin{longtable}`));
});

test("untrusted math cannot create links, image requests, user HTML or shared macros", () => {
  for (const source of [
    String.raw`\href{javascript:alert(1)}{click}`,
    String.raw`\includegraphics{https://external.invalid/tracker}`,
    String.raw`\text{<script>alert(1)</script>}`,
  ]) {
    const html = katex.renderToString(source, latexMathOptions());
    const dom = new JSDOM(html);
    assert.equal(
      dom.window.document.querySelectorAll("script,img,a,.injected").length,
      0,
    );
    dom.window.close();
  }
  assert.throws(
    () =>
      katex.renderToString(
        String.raw`\htmlClass{injected}{x}`,
        latexMathOptions(),
      ),
    /HTML extension is disabled/,
  );
  katex.renderToString(
    String.raw`\gdef\privateMacro{x}\privateMacro`,
    latexMathOptions(),
  );
  assert.throws(() =>
    katex.renderToString(String.raw`\privateMacro`, latexMathOptions()),
  );
});

test("math has a hard input limit, expansion bound and accessible output", () => {
  assert.throws(() => validateMathSource("x".repeat(65_537)), /limit/);
  assert.throws(
    () =>
      katex.renderToString(
        String.raw`\def\loop{\loop}\loop`,
        latexMathOptions(),
      ),
    /expansions/,
  );
  const dom = new JSDOM(
    katex.renderToString(
      String.raw`\begin{array}{lr}\text{Ücret} & 1.25\\\text{N} & 480\end{array}`,
      latexMathOptions(),
    ),
  );
  assert.equal(
    dom.window.document.querySelectorAll(".katex .katex-mathml math").length,
    1,
  );
  assert.ok(
    dom.window.document
      .querySelector("annotation")
      ?.textContent?.includes("480"),
  );
  dom.window.close();
});

test("native dataframe preview parses multiline Turkish text, safe escapes and tiny scientific values", () => {
  const source = String.raw`\begin{array}{lrl}
\hline
\text{} & \text{Ücret\_\%} & \text{etiket} \\
\hline
\text{0} & 1.2000 \times 10^{-12} & \begin{gathered}\text{çok\_satır} \\ \text{A\&B\textbackslash{}\textasciitilde{}\textasciicircum{}}\end{gathered} \\
\text{1} & \infty & \text{<script>alert(1)</script>} \\
\hline
\end{array}`;
  const dom = new JSDOM(katex.renderToString(source, latexMathOptions()));
  assert.equal(dom.window.document.querySelectorAll("script,img,a").length, 0);
  assert.ok(
    dom.window.document
      .querySelector("annotation")
      ?.textContent?.includes("10^{-12}"),
  );
  assert.ok(
    dom.window.document
      .querySelector(".katex-html")
      ?.textContent?.includes("satır"),
  );
  dom.window.close();
});
