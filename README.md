# OpenEconometrics

<!-- BEGIN source-generated capability scope -->
Current source: **157 registered fit names**, **92 Dataset fit routes**, **74 common saved predict/margins adapters**. The [generated method/option inventory](docs/capabilities.md) states conditions, exclusions and devices.

Fit routes, common prediction adapters and family-specific helpers/forecasts have separate contracts. Source implementation does not establish independent scientific validation, installed-package verification or public shipment for a method/option. Those require their own dated, source-pinned evidence; historical measurements retain their original scope.
<!-- END source-generated capability scope -->

An open-source Python statistics library and local research workbench with a code editor, persistent Python session, data inspector, plots and MCP integration. Apache-2.0 licensed.

Python package remains `openecon`. Project website: [openeconometrics.com](https://openeconometrics.com).

Reviewed source snapshots and Mac alpha releases are distributed through
[bluearf/openeconometrics](https://github.com/bluearf/openeconometrics).
Mac packages require Apple Silicon and macOS 15 or later. Read each release's
checksum and signing status before installation. Developer ID signing and
notarization remain pending. The published PyPI alpha packages are
[openecon 0.3.18a4](https://pypi.org/project/openecon/0.3.18a4/) and
[openecon-charts 0.3.0a2](https://pypi.org/project/openecon-charts/0.3.0a2/).
Their scope follows the reviewed public snapshot. See
[PUBLIC-SOURCE.md](https://github.com/bluearf/openeconometrics/blob/main/PUBLIC-SOURCE.md) and `SOURCE-MANIFEST.json` in the
public snapshot for its precise source and external-fixture boundary.

This checkout prepares SDK **0.3.19a1** and charts **0.3.1a1**. Those versions
require their own frozen public-source manifest, wheel/sdist installation checks
and verified PyPI publication; the published pair above retains its earlier scope.

Distribution preserves the [third-party notices](THIRD_PARTY.md). See
[contribution guidelines](CONTRIBUTING.md) and the [private security-reporting
process](SECURITY.md). Source availability and published installer access are
verified separately from the license declaration.

**Status: 0.3 alpha.** OpenEconometrics implements a registered native estimator catalogue on one PyTorch float64 core; see the [source-generated current scope](docs/capabilities.md) for estimator, Dataset fitting and saved prediction coverage. [OLS and weighted least squares](docs/ols.md) include robust/cluster/HAC/resampling covariance and postestimation; dense OLS also supports CUDA. The interface is built around writing Python, inspecting results and continuing the same session. There is no numerical-engine selector. Full Stata coverage and numerical parity with Stata have not been established. The interface, Python API and CLI are in English.

The [desktop edition](docs/desktop.md) packages the existing interface in a
Rust/Tauri window with a bundled Python runtime. Code runs locally; cloud projects
store drafts, data and shared results. See [platform build instructions](desktop/README.md).

## Start the workbench

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node.js 22.12+ (or 24 LTS). Python 3.13 is selected by this checkout; uv can provision it.

```sh
./scripts/start.sh
```

Open **http://127.0.0.1:8765**. The launcher installs locked dependencies, builds the interface and starts the local server. The initial installation needs internet access. Stop the server with Ctrl+C.

The equivalent manual steps are:

```sh
uv sync --frozen --extra app
npm --prefix web ci
npm --prefix web run build
uv run --extra app openecon serve --workspace .openecon
```

The Python console keeps variables between commands. Start with the bundled synthetic dataset:

```python
import openecon as oe

df = oe.example()
model = oe.ols(data=df, y="wage", x=["education", "experience"], covariance="HC3")
print(model.summary())
oe.plot.coefficients(model)
```

Continue with a transformation and another plot:

```python
df["experience_squared"] = df["experience"] ** 2
oe.plot.scatter(data=df, x="education", y="wage")
```

The console displays the last expression automatically. `display(...)` can show additional tables, models and OpenEconometrics plots. Charts use the separate `openecon-charts` Python package and the existing D3 visual family from the Dart-D3/Bluearf projects. Scatter, numeric line, histogram, coefficient intervals, grouped/horizontal bars, multi-series area, stacked bars and donut charts are available. Legends, tooltips, a data table, enlargement and SVG/PNG/JPEG/CSV exports share the same renderer. This is not general Matplotlib figure capture.

The editor autosaves its draft and can download that exact draft as a `.py` file. This is console source, not a self-contained analysis bundle: `display()` and automatic last-expression rendering are console conveniences, and imported dataset IDs belong to the current workspace. For a regular Python script, use `print(model.summary())`, explicit data paths, and `chart.save_html("chart.html")` for an independently viewable chart. “Dosyadan ekle” appends imported source to the current draft; it does not execute it.

The editor shows function signatures and documentation on hover or with **F1**.
Call arguments show parameter help, and **Ctrl+Space** opens completion suggestions.
Help covers the bundled OpenEconometrics API, common Python builtins and functions defined
in the current script. It follows import aliases and conservatively identifies
dataframes and model results. This assistance runs locally in the editor without
executing Python. It is not a language server for every third-party package.
See [editor assistance](docs/editor.md) for supported behavior and shortcuts.

Relative file paths in the console resolve inside the active workspace (`.openecon` by default). Imported datasets are available by ID through `oe.load_dataset(...)`; use an absolute path to read a file elsewhere.

**Console code runs with your local account's permissions.** It can read and write files, access the network and import installed packages. The worker is a separate process, not a security sandbox. Run code you trust. A stop, timeout or session reset terminates the worker and clears its variables. Successful commands and ordinary Python errors leave the session alive; errors do not roll back assignments made before the error. Command history is saved, but in-memory variables do not survive a server restart.

## Use the library

Install the published alpha on Python 3.11–3.14:

```sh
python -m pip install 'openecon==0.3.18a4'
```

This version installs its exact chart dependency, `openecon-charts==0.3.0a2`.
The current source candidate is `openecon==0.3.19a1` with
`openecon-charts==0.3.1a1`; it has not replaced the published pair above.
The source checkout and desktop candidate have separate release scope;
see [distribution versions and extras](docs/distribution.md).

The library accepts pandas DataFrames, dictionaries of columns and lists of records. It is not tied to the included wage dataset.

```python
import openecon as oe

rows = [
    {"output": 11.0, "hours": 2.0},
    {"output": 14.0, "hours": 3.0},
    {"output": 12.0, "hours": 4.0},
    {"output": 19.0, "hours": 5.0},
    {"output": 20.0, "hours": 6.0},
]
result = oe.ols(data=rows, y="output", x=["hours"], covariance="HC3")
print(result.summary())
print(result.model_dump_json(indent=2))
```

File loading and more explicit model specifications are also available:

```python
data = oe.read("survey.csv")
result = oe.fit(
    oe.ModelSpec(
        estimator="ols", outcome="wage",
        predictors=["education", "experience", "region"],
        categorical=["region"], covariance="HC3", missing="raise",
    ),
    data=data,
)
```

`oe.logit(...)` and `oe.probit(...)` use the same `data`, `y` and `x` interface. Ordinary OLS and binary models default to classical covariance; sampling-weighted OLS defaults to HC1. Supplying `cluster="firm_id"` selects cluster covariance when covariance is omitted. `ModelSpec` applies the same defaults. Unsupported combinations fail explicitly. `oe.capabilities()` describes implemented methods. See [the complete OLS API](docs/ols.md) for formulas, weights and postestimation.

Library fits return result objects without automatically persisting them. `oe.load_dataset(dataset_id)` loads an imported dataset in the workbench's active workspace; imported IDs are visible in the dataset inspector. The console session, saved structured analyses and imported snapshots are separate records.

## LaTeX results and exports

The results pane switches directly between **Sonuç** (rendered results) and
**LaTeX** (source). Select all outputs or one table, model or chart, then copy
or download the same complete `.tex` document. Tables and coefficients use a
local, lazily loaded KaTeX preview; charts retain their interactive D3 view and
export as self-contained TikZ/PGFPlots source.

```python
df = oe.example()
df.head().to_latex()                         # Rendered table in the workbench
df.to_latex("data.tex", index=False)          # Export every row to a file
baseline = oe.ols(data=df, y="wage", x=["education"])
model = oe.ols(data=df, y="wage", x=["education", "experience"])
model.to_latex("model.tex")                   # Publication layout by default
oe.regression_table([baseline, model], "models.tex")
chart = oe.plot.coefficients(model)
chart.to_latex("chart.tex", standalone=True)
```

`oe.read`, `oe.example` and `oe.load_dataset` return an `oe.DataFrame`, which
retains pandas operations and provides `to_latex()` and `.latex`. Wrap an existing
pandas table with `oe.DataFrame(table)` or use `oe.to_latex(table)`. Model results
also expose `.latex` and `summary(format="latex")`; ordinary `summary()` remains
text. `oe.latex(value)` converts tables, models and plain text; `oe.Latex(source,
math=...)` displays explicit source with an optional math preview.

Publication defaults use booktabs rules without vertical grid lines. Regression
coefficients sit above parenthesized standard errors; columns are numbered,
constants come last and significance stars use the original p-values. Model
notes report the actual covariance method, clustering, sample and inference.
`oe.regression_table` compares fitted models with missing terms left blank.
Long tables repeat their headers across pages. These are paper-style defaults,
not an official NBER template. See [publication tables](docs/publication-tables.md).

Automatic exports include the displayed preview rows, with truncation recorded;
`df.to_latex(...)` exports the full table. Source text safely escapes TeX special
characters. Downloaded documents use XeLaTeX or LuaLaTeX with `fontspec`, `tikz`
`booktabs`, `adjustbox`, `array`, `longtable`, `placeins` and `pgfplots`. Exports run without pandas' Jinja-based LaTeX renderer or a Python
LaTeX installation. See [LaTeX verification](docs/verification.md).

## Network analysis

Build weighted, directed or undirected networks from edge tables. Large local
CSV/Parquet edge lists are read in batches; graph storage remains sparse and
must fit the explicit memory budget. Degrees, strengths, weak/strong components,
weighted PageRank, shortest paths, Louvain/Leiden communities, modularity,
betweenness/closeness/harmonic/eigenvector centrality, clustering, triangles and
k-core use the full network. Expensive exact methods have explicit work budgets;
sampled betweenness is an opt-in estimator.

```python
graph = oe.network(data=edges, source="from", target="to", weight="weight",
                   directed=True)
display(graph.summary())
display(graph.pagerank())
communities = graph.communities(method="leiden", seed=42)
oe.plot.network(graph, groups=communities, title="Connections")
```

Interactive charts use Canvas and a worker for force layout. Large networks
show a bounded subset with full and displayed counts; this does not change
analytical results. PNG, SVG and CSV exports are available; LaTeX exports a
compact summary table. See [network analysis](docs/network.md) for semantics,
memory limits and examples, including static GraphML/GEXF/Pajek import/export and
scalar attributes. The implementation uses Torch sparse numerical buffers and
bounded Python traversal; it does not depend on SciPy or NetworkX.
CUDA PageRank is optional and has not been verified
on physical CUDA hardware.

## Independent chart package

`oe.plot` re-exports the independently installable package in
[`packages/openecon-charts`](packages/openecon-charts). It has no required Python
dependencies and does not import PyTorch, pandas, NumPy or OpenEconometrics. It accepts
column dictionaries, record lists and pandas-like tables through an adapter.
D3 and Barlow assets are bundled; chart viewing needs no CDN or network request.

```python
import openecon_charts as charts

chart = charts.area(
    data={"year": [2022, 2023, 2024, 2025],
          "A": [12, 18, 15, 26], "B": [8, 11, 16, 21]},
    x="year", y=["A", "B"], title="Illustrative series",
)
chart.save_html("chart.html")
# In the workbench: display(chart)
```

Install the published chart alpha independently with
`python -m pip install 'openecon-charts==0.3.0a2'`. From this checkout,
use `python -m pip install ./packages/openecon-charts` for the current source version.
The workspace launcher installs it alongside OpenEconometrics. For distributable wheels,
build both using `uv build --all-packages` and install both wheels together.

Chart helpers never silently aggregate observations into categories. For example,
compute `df.groupby("region", observed=True, as_index=False)["wage"].mean()` before
passing it to `oe.plot.bar(...)`. Missing values, display sampling and exclusions
are preserved in the plot contract. [Chart documentation](packages/openecon-charts/README.md)
covers supported inputs and limits; [provenance](docs/charts-provenance.md) records
which existing renderers were adapted and the bundled licenses.

## Implemented statistical methods

| Method | Supported options |
| --- | --- |
| OLS/WLS | Formulas, interactions, time operators and four weight types; classical, HC0–HC3, one-to-four-way cluster, cluster HC2/HC3, HAC, bootstrap and jackknife covariance |
| Logit and probit | Binary outcomes; numeric and categorical predictors; nonrobust and one-way cluster covariance; convergence and separation checks |
| Inference | Standard errors, confidence intervals, model and contrast tests, explicit sample counts, degrees of freedom and OLS Bell–McCaffrey/Hansen adjustments |
| OLS postestimation | Full/batched predictions, marginal effects, VIF, Breusch–Pagan/White/RESET/BG, residual and influence diagnostics |
| Sample handling | OLS defaults to complete-case deletion; binary models reject missing inputs by default; retained physical positions are recorded or hashed |
| Provenance | Specification, input/sample hashes, category references, precision, solver diagnostics and package versions |

OLS generally uses Student-t inference; bootstrap uses normal inference. Multiway clusters use intersection-specific CR1 corrections and the minimum marginal cluster degrees of freedom; optional HC2/HC3 corrections provide contrast-specific inference. OLS treatment coding omits the first level with an intercept and keeps all levels without one; collinear terms are omitted deterministically. Binary category/rank rules remain unchanged. Pandas category order is respected; other levels are sorted. See [OLS methods and boundaries](docs/ols.md).

The estimator, covariance, inference, likelihood and separation safeguards belong to OpenEconometrics. PyTorch supplies tensor and linear-algebra primitives; pandas provides tables and file interoperability. SciPy is not a runtime dependency: the separation LP uses a native float64 primal-dual solver with independently checked certificates. NumPy remains a transitive table dependency, with no NumPy estimator backend. Statsmodels and SciPy are development-only numerical references. See [the numerical design](docs/native-engine.md).

## Files, CLI and agents

Import CSV, Parquet, XLSX or Stata DTA files through the workbench or CLI:

```sh
uv run --extra cli openecon inspect survey.csv
uv run --extra cli openecon import survey.csv --workspace .openecon
uv run --extra cli openecon fit survey.csv -y wage -x education -x experience --covariance HC3
uv run --extra cli openecon fit survey.csv -y employed -x education --estimator logit
uv run --extra cli openecon --help
```

The imported original file and canonical Parquet snapshot remain in the local workspace. CLI `fit` returns JSON without adding a saved workspace analysis. Workspace API and MCP analyses persist immutable result records. Generated Python/ZIP exports refer to the recorded dataset and model; **a portable ZIP includes the data**. Recorded hashes bind category semantics and metadata as well as rows. Older results retain their original provenance; the current API does not promise replay compatibility with obsolete model specifications.

MCP exposes structured dataset inspection and model-fitting tools to Codex, Claude or other MCP clients. It does not expose the console's arbitrary Python execution. Raw row previews require an explicit option; summaries and coefficient labels can still contain sensitive information and may be forwarded by a client to its AI provider. No AI account is needed for the workbench itself. See [CLI and agent integration](docs/agent-integration.md).

## Current limits

The [Cloud Run team edition](docs/team-deployment.md) provides Google sign-in and verified email/password accounts, persistent projects, email-bound invitations and owner/editor/viewer roles. Firestore and private Cloud Storage retain drafts, data and results. Production executes each complete Python script in a fresh sandbox through a private Cloud Run compute service; the authenticated API never executes user code. Variables do not persist between runs. The managed runtime proof, all 13 broker integration cases and complete five-user preview and production suites passed. The initial sandbox rollout completed its production OLS, D3, input and durable-output check in 11.9 seconds; the LaTeX update passed native dataframe/model/chart exports and shared CSV/TeX downloads in 12.4 seconds; the publication-table release then passed the complete suite with multi-model exports in 13.0 seconds. [Startup verification](docs/verification.md) records the deployment and distinguishes analysis timings from browser opening and cold starts. Previously accepted Job runs retain status/cancellation support. Concurrent draft writes are version-checked; simultaneous collaborative text editing and remote MCP are not included. The earlier [single-owner IAP edition](docs/cloud-deployment.md) remains a separate legacy deployment mode.

- Large local CSV/Parquet imports automatically use a projected, replayable Dataset. The 32 MiB/100,000-row thresholds choose the bounded reader; they do not reject those formats. Large desktop data stays local. Cloud project uploads retain their 24 MiB limit; eager XLSX/DTA have separate format limits. Native model adapters process complete samples through bounded passes and owned temporary storage, with model-specific option conditions in `oe.capabilities()`. See [large local datasets](docs/streaming.md) for algorithms, resource budgets and measured physical workloads.
- Statistical estimates and inference use native PyTorch float64. Supported factor operations can use CUDA float64 or checked Mac Metal preconditioning with original CPU float64 refinement; likelihood/state recursions and disk operations remain CPU. Actual execution devices and fallbacks are recorded. CUDA hardware throughput has not been validated.
- Local console commands have no automatic execution deadline; explicit deadlines and Stop remain available. Output, table previews and history are bounded. Stop/timeout resets the local session; it has no rollback, hard memory quota or OS security sandbox. Cloud computation retains its previous deadlines and independent sandbox limits.
- Workspace fits and MCP `run_analysis` are synchronous and serialized within each process. MCP also offers bounded background jobs with status, cancellation and durable retry IDs; one background job runs per workspace. See the [job contract](docs/agent-integration.md). These jobs are separate from the console's worker.
- The current catalogue includes weights, absorbed fixed effects, IV/GMM, panel models, time series, survival models and supported postestimation. Complex surveys, imputation, SEM, Stata command execution and do-file translation remain separate work; model coverage does not imply complete Stata parity.
- XLSX reads the first worksheet's stored values. CSV expects UTF-8 and comma separators. DTA labels and tagged missing codes are retained as metadata; this is not a full Stata round trip or DTA export.
- The local HTTP server is for one trusted local user and only accepts loopback binding. Cloud teams use a separate authenticated control API with live membership checks and isolated execution. Do not expose the local console or broaden the legacy IAP mode as a substitute for this team architecture.

## Development and evidence

```sh
uv sync --frozen --extra cloud --extra desktop
uv run --no-sync ruff check src tests scripts packages/openecon-charts
npm --prefix web ci
npm --prefix web test
npm --prefix web run build
uv run --no-sync python -m pytest
uv build --all-packages
```

GitHub verification runs on pull requests, pushes to `main`, merge-queue events
and manual dispatch. Windows packaging uses manual dispatch. The local checks
above remain available; a workflow definition alone does not establish that a
particular source revision passed its required checks.

For interface development, run `uv run --extra app openecon serve` and `npm --prefix web run dev` separately. Vite proxies `/api` and `/chart-assets` to port 8765. Build the interface before packaging; the OpenEconometrics wheel includes generated interface assets and the frozen synthetic fixture. The chart wheel contains the single shared D3 renderer and its local assets.

Tests cover independent statistical references, data integrity and metadata, Python/CLI/MCP behavior, local API access, console persistence/interruption and plot validation. Test results do not establish universal numerical correctness or Stata parity. [Architecture notes](docs/architecture.md) describe the execution boundaries. [Performance records](docs/performance.md) are historical **0.2** measurements of the former dual-engine implementation; they do not measure the current single-core workbench.

## License

OpenEconometrics code is licensed under [Apache-2.0](LICENSE). Dependencies retain their licenses; datasets remain subject to their own permissions and licenses.

Library-only users can install `openecon` without the web/CLI/MCP layers. See [distribution extras and measured dependencies](docs/distribution.md).

[Nonlinear SUR](docs/econometrics/nonlinear-sur.md): bounded shared-parameter Gaussian systems with complete saved-state replay and joint postestimation.

## Teaching courses

Explore [80 English teaching labs](docs/teaching/COURSES.md) across Econometrics, Statistics, Microeconomics and Advanced Econometrics. Each course includes runnable OpenEconometrics code, mathematical explanations, computed results and exercises. Use the supplied named Excel workbooks for identical class results. Student handouts and instructor answer guides are distributed separately in the [teaching release](https://github.com/bluearf/openeconometrics/releases/tag/teaching-v0.1.0).
