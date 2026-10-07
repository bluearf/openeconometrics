import type { ConsoleOutput, ResultBundle } from "./types.ts";

export function escapeLatex(value: unknown): string {
  const text =
    value == null
      ? "—"
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
  const escapes: Record<string, string> = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    $: "\\$",
    "&": "\\&",
    "#": "\\#",
    "%": "\\%",
    _: "\\_",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
  };
  return text
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, "")
    .replace(/[\\{}$&#%_~^]/g, (character) => escapes[character])
    .replace(/\r?\n/g, " ");
}

function textFragment(value: unknown): string {
  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return (text ?? "")
    .split(/\r?\n/)
    .map((line) =>
      line ? `\\noindent ${escapeLatex(line)}\\par` : "\\medskip",
    )
    .join("\n");
}

export function significanceStars(value: unknown): string {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > 1
  )
    return "";
  return value < 0.01 ? "***" : value < 0.05 ? "**" : value < 0.1 ? "*" : "";
}

export function publicationNumber(value: unknown, precision = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value !== 0 && Math.abs(value) < 10 ** -precision
    ? value.toExponential(precision)
    : value.toFixed(precision);
}

export function modelTargetLabel(model: ResultBundle): string {
  const target = model.inference?.target ?? model.extra?.target;
  if (
    target === "prediction" ||
    model.extra?.penalized_state != null ||
    model.extra?.smoother_state != null
  )
    return "Prediction target";
  if (typeof target === "string" && target.trim()) return target;
  return model.coefficients.length
    ? "Parameter estimates; inference unavailable"
    : "Fixed parameter evaluation";
}

export function uninferredModelRows(model: ResultBundle): [string, number][] {
  const extra = model.extra ?? {};
  const state = extra.penalized_state as
    { constant?: unknown; coefficients?: unknown[] } | undefined;
  const terms = Array.isArray(extra.terms)
    ? extra.terms.filter((term): term is string => typeof term === "string")
    : model.spec.predictors;
  const numeric = (value: unknown): value is number =>
    typeof value === "number" && Number.isFinite(value);
  if (state) {
    const rows: [string, number][] =
      model.spec.intercept && numeric(state.constant)
        ? [["Intercept", state.constant]]
        : [];
    state.coefficients?.forEach((value, index) => {
      if (numeric(value) && terms[index] != null)
        rows.push([terms[index], value]);
    });
    return rows;
  }
  if (Array.isArray(extra.query_estimates)) {
    const queries = Array.isArray(extra.query) ? extra.query : [];
    return extra.query_estimates.flatMap((value, index) => {
      if (!numeric(value)) return [];
      const query = queries[index];
      const coordinates = Array.isArray(query)
        ? terms
            .map(
              (term, column) => `${term}=${publicationNumber(query[column])}`,
            )
            .join(", ")
        : "";
      return [
        [
          `Query ${index + 1}${coordinates ? ` (${coordinates})` : ""}`,
          value,
        ] as [string, number],
      ];
    });
  }
  return model.coefficients.length
    ? model.coefficients.map(({ term, estimate }) => [term, estimate])
    : (model.predictions ?? []).flatMap(({ row, fitted }) =>
        numeric(fitted) ? [[`Row ${row}`, fitted] as [string, number]] : [],
      );
}

function numericCell(value: unknown, stars = ""): string {
  const text = publicationNumber(value);
  if (text === "—") return text;
  const scientific = /^(.+)[eE]([+-]?\d+)$/.exec(text);
  const number = scientific
    ? `${scientific[1]} \\times 10^{${Number(scientific[2])}}`
    : text;
  return `$${stars ? `{${number}}^{${stars}}` : number}$`;
}

function boundedTable(lines: string[]): string[] {
  return [
    "\\begingroup",
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "\\renewcommand{\\arraystretch}{1.12}",
    "\\begin{adjustbox}{max width=\\linewidth}",
    ...lines,
    "\\end{adjustbox}",
    "\\endgroup",
  ];
}

function tableFragment(columns: unknown[], rows: unknown[][]): string {
  const width = columns.length;
  if (!width) return "";
  const header = columns.map(escapeLatex).join(" & ") + " \\\\";
  const body = rows.map(
    (row) =>
      Array.from({ length: width }, (_, index) => escapeLatex(row[index])).join(
        " & ",
      ) + " \\\\",
  );
  if (rows.length > 40) {
    return [
      "\\begingroup",
      "\\small",
      "\\setlength{\\tabcolsep}{3pt}",
      "\\renewcommand{\\arraystretch}{1.12}",
      `\\begin{longtable}{@{}*{${width}}{>{\\raggedright\\arraybackslash}p{\\dimexpr\\linewidth/${width}-6pt\\relax}}@{}}`,
      "\\toprule",
      header,
      "\\midrule",
      "\\endfirsthead",
      "\\toprule",
      header,
      "\\midrule",
      "\\endhead",
      "\\midrule",
      `\\multicolumn{${width}}{r}{\\footnotesize Continued on next page} \\\\`,
      "\\endfoot",
      "\\bottomrule",
      "\\endlastfoot",
      ...body,
      "\\end{longtable}",
      "\\endgroup",
    ].join("\n");
  }
  return [
    "\\begin{center}",
    ...boundedTable([
      `\\begin{tabular}{@{}${"l".repeat(width)}@{}}`,
      "\\toprule",
      header,
      "\\midrule",
      ...body,
      "\\bottomrule",
      "\\end{tabular}",
    ]),
    "\\end{center}",
  ].join("\n");
}

function publicationModel(model: ResultBundle, savedNotes?: string[]): string {
  if (model.inference?.available === false) {
    const notes = [
      `${modelTargetLabel(model)}; coefficient inference unavailable.`,
      ...(savedNotes ?? []),
      ...(model.warnings ?? []),
    ];
    return [
      "\\begin{table}[htbp]",
      "\\centering",
      `\\caption{${escapeLatex(modelTargetLabel(model))}}`,
      ...boundedTable([
        "\\begin{tabular}{@{}lr@{}}",
        "\\toprule",
        "Quantity & Estimate \\\\",
        "\\midrule",
        ...uninferredModelRows(model).map(
          ([label, value]) =>
            `${escapeLatex(label)} & ${numericCell(value)} \\\\`,
        ),
        "\\midrule",
        `Observations & ${escapeLatex(model.nobs)} \\\\`,
        "\\bottomrule",
        "\\end{tabular}",
      ]),
      "\\par\\smallskip",
      `\\textit{Notes:} ${notes.map(escapeLatex).join(" ")}`,
      "\\end{table}",
    ].join("\n");
  }
  const covariance = model.inference?.covariance ?? model.spec.covariance;
  const grouped = model.coefficients.some(
    (coefficient) => coefficient.equation != null,
  );
  const hasRSquared = ["r_squared", "rsquared"].some(
    (name) => typeof model.metrics?.[name] === "number",
  );
  const metricGroups: [string[], string][] =
    model.spec.estimator === "ols" || hasRSquared
      ? [
          [["r_squared", "rsquared"], "$R^{2}$"],
          [
            ["adjusted_r_squared", "adjusted_rsquared", "rsquared_adj"],
            "Adjusted $R^{2}$",
          ],
        ]
      : [
          [["pseudo_r_squared", "pseudo_r2"], "Pseudo $R^{2}$"],
          [["log_likelihood", "llf"], "Log likelihood"],
        ];
  const metricRows = metricGroups.flatMap(([names, label]) => {
    const value = names
      .map((name) => model.metrics?.[name])
      .find(
        (candidate) =>
          typeof candidate === "number" && Number.isFinite(candidate),
      );
    return value === undefined ? [] : [`${label} & ${numericCell(value)} \\\\`];
  });
  const clustering =
    model.inference?.cluster_columns ??
    model.inference?.cluster_column ??
    model.spec.cluster;
  const clusterNames =
    typeof clustering === "string"
      ? [clustering]
      : Array.isArray(clustering)
        ? clustering
        : [];
  const counts = Array.isArray(model.inference?.cluster_counts)
    ? model.inference.cluster_counts
    : clusterNames.length === 1
      ? [model.inference?.cluster_count]
      : [];
  const notes = savedNotes?.map(escapeLatex) ?? [
    `${escapeLatex(model.spec.estimator.toUpperCase())}; ${escapeLatex(covariance)} standard errors in parentheses.`,
    ...(clusterNames.length
      ? [
          `Errors are clustered by ${clusterNames.map((name, index) => `${escapeLatex(name)} (${typeof counts[index] === "number" ? `${counts[index]} clusters` : "cluster count unavailable"})`).join(", ")}.`,
        ]
      : []),
    ...(model.inference?.use_t === true
      ? [
          `Student t inference${typeof model.inference.df_inference === "number" ? ` (df = ${model.inference.df_inference})` : " (degrees of freedom unavailable)"}.`,
        ]
      : model.inference?.use_t === false
        ? ["Normal z inference."]
        : []),
    ...(typeof model.nobs_original === "number"
      ? [
          `Physical estimation sample ${model.nobs_original - model.dropped_rows} of ${model.nobs_original} observations.`,
        ]
      : []),
    ...(model.spec.weight_type
      ? [
          `${escapeLatex(model.spec.weight_type)} weights: ${escapeLatex(model.spec.weights)}; effective observations = ${model.nobs}.`,
        ]
      : []),
    ...["correction", "sigma", "kernel", "lags"].flatMap((key) =>
      model.inference?.[key] != null
        ? [`${escapeLatex(key)}: ${escapeLatex(model.inference[key])}.`]
        : [],
    ),
    ...(model.dropped_rows
      ? [
          `${escapeLatex(model.dropped_rows)} observations excluded from estimation.`,
        ]
      : []),
    "\\(^{***}p<0.01\\), \\(^{**}p<0.05\\), \\(^{*}p<0.10\\).",
    ...(model.warnings ?? []).map(escapeLatex),
  ];
  return [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Regression results}",
    ...boundedTable([
      "\\begin{tabular}{@{}lc@{}}",
      "\\toprule",
      " & (1) \\\\",
      ` & ${escapeLatex(model.spec.outcome)} \\\\`,
      "\\midrule",
      ...model.coefficients.flatMap((coefficient, index) => [
        ...(grouped &&
        (index === 0 ||
          coefficient.equation !== model.coefficients[index - 1].equation)
          ? [
              `{[${escapeLatex(coefficient.equation ?? "Other parameters")}]} & \\\\`,
            ]
          : []),
        `${escapeLatex(coefficient.term)} & ${numericCell(coefficient.estimate, significanceStars(coefficient.p_value))} \\\\`,
        ` & (${numericCell(coefficient.std_error)}) \\\\[3pt]`,
      ]),
      "\\midrule",
      `Observations & ${escapeLatex(model.nobs)} \\\\`,
      ...metricRows,
      "\\bottomrule",
      "\\end{tabular}",
    ]),
    "\\par\\smallskip",
    "\\begin{minipage}{\\linewidth}",
    "\\footnotesize\\raggedright",
    `\\textit{Notes:} ${notes.join(" ")}`,
    "\\end{minipage}",
    "\\end{table}",
  ].join("\n");
}

export function outputLatex(output: ConsoleOutput): string {
  if (typeof output.latex === "string" && output.latex.trim())
    return output.latex;
  if (output.type === "latex" && typeof output.data === "string")
    return output.data;
  if (output.type === "table") {
    const table = output.data as {
      columns?: unknown[];
      rows?: unknown[][];
      index?: unknown[][];
      index_names?: unknown[];
      total_rows?: number;
      total_rows_known?: boolean;
      total_columns?: number;
    };
    if (Array.isArray(table?.columns) && Array.isArray(table.rows)) {
      const columns = [
        table.index_names?.filter(Boolean).join(" / ") || "#",
        ...table.columns,
      ];
      const rows = table.rows.map((row, index) => [
        table.index?.[index]?.join(" / ") ?? index + 1,
        ...(Array.isArray(row) ? row : []),
      ]);
      const truncated =
        table.total_rows_known === false ||
        (table.total_rows ?? rows.length) > rows.length ||
        (table.total_columns ?? table.columns.length) > table.columns.length;
      return [
        tableFragment(columns, rows),
        ...(truncated
          ? [
              `\\par\\smallskip{\\footnotesize\\textit{First ${rows.length} ${rows.length === 1 ? "row" : "rows"} and ${table.columns.length} ${table.columns.length === 1 ? "column" : "columns"}.}\\par}`,
            ]
          : []),
      ].join("\n");
    }
  }
  if (output.type === "model") {
    const model = output.data as ResultBundle;
    if (model?.spec && Array.isArray(model.coefficients)) {
      return publicationModel(model, output.latex_notes);
    }
  }
  if (output.type === "plot")
    return "\\noindent\\textit{No LaTeX source was saved for this chart.}\\par";
  return textFragment(output.data);
}

export function latexDocument(outputs: ConsoleOutput[]): string {
  return [
    "% !TEX program = xelatex",
    "\\documentclass[11pt]{article}",
    "\\usepackage[a4paper,margin=25mm]{geometry}",
    "\\usepackage{fontspec}",
    "\\usepackage{amsmath,amssymb,booktabs,array,graphicx,longtable}",
    "\\usepackage{adjustbox}",
    "\\usepackage[font=small,labelfont=bf]{caption}",
    "\\usepackage{placeins}",
    "\\usepackage{tikz,pgfplots}",
    "\\pgfplotsset{compat=1.18}",
    "\\setlength{\\parindent}{0pt}",
    "\\setlength{\\parskip}{6pt}",
    "\\setlength{\\textfloatsep}{16pt}",
    "\\setlength{\\floatsep}{12pt}",
    "\\setlength{\\intextsep}{12pt}",
    "\\renewcommand{\\arraystretch}{1.12}",
    "\\begin{document}",
    ...outputs
      .map(outputLatex)
      .filter(Boolean)
      .flatMap((source, index) =>
        index ? ["\\FloatBarrier", "\\bigskip", source] : [source],
      ),
    "\\end{document}",
    "",
  ].join("\n");
}

export function outputLabel(output: ConsoleOutput, index: number): string {
  const labels = {
    model: "Model",
    table: "Tablo",
    plot: "Grafik",
    text: "Metin",
    latex: "LaTeX",
  };
  const data = output.data as { title?: string; spec?: { outcome?: string } };
  const title = data?.title ?? data?.spec?.outcome;
  return `${index + 1}. ${labels[output.type]}${title ? ` · ${title}` : ""}`;
}
