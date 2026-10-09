# OpenEconometrics Charts

A standalone Python chart library with no required Python dependencies. Build a chart from column mappings, row records, or a pandas-style table, then display it in a notebook or save an offline HTML file. The same chart specifications and renderer power OpenEconometrics's Python workspace.

```python
import openecon_charts as charts

sales = {
    "quarter": ["Q1", "Q2", "Q3"],
    "online": [12, 18, 15],
    "retail": [8, 11, 13],
}
chart = charts.stacked_bar(
    data=sales, x="quarter", y=["online", "retail"],
    title="Sales by quarter", unit="EUR thousands",
    palette=["#163d68", "#6c8eaf"],
)
chart.save_html("sales.html")
```

Within OpenEconometrics, the same API is available through `oe.plot`. All charts run locally; saved HTML embeds its scripts, styles, and font and requires no network access.

## Charts

| Function | Inputs | Behavior |
|---|---|---|
| `scatter(data=..., x=..., y=...)` | Two numeric columns | At most 2,000 evenly spaced finite observations, with explicit sample counts. |
| `line(data=..., x=..., y=...)` | Two numeric columns | Numeric x spacing is preserved. Missing y creates a gap; missing x separates runs. |
| `hist(data=..., x=..., bins=20)` | Numeric column | All finite observations, equal-width bins, inclusive final edge. |
| `coefficients(model)` | Coefficient-record protocol | Estimate and interval for each term; no OpenEconometrics dependency. |
| `bar(data=..., x=..., y=...)` | Category column and one or more value columns | Grouped vertical bars, including negative values. |
| `barh(data=..., x=..., y=...)` | Same as `bar` | Horizontal bars. |
| `area(data=..., x=..., y=...)` | Category column and one or more value columns | Signed area series with explicit missing-value gaps. |
| `stacked_bar(data=..., x=..., y=..., horizontal=False)` | Categories and component columns | Nonnegative, complete components; optional horizontal layout. |
| `donut(data=..., labels=..., values=...)` | Unique category labels and values | Nonnegative, complete values with a positive total. |

Every helper accepts `title` and the applicable appearance options below. Categorical helpers also accept `unit` and `palette`. A palette is a sequence of opaque `#RGB` or `#RRGGBB` colors; short colors are expanded. Supply at least one color per series, or one per category for a donut.

## Appearance and axes

Configure a chart directly in its Python call or create a separately styled copy
with `with_options()`. Both use the same validated settings:

```python
chart = charts.scatter(
    data={"income": [10, 100, 1000], "spending": [8, 60, 450]},
    x="income", y="spending", title="Income and spending",
    x_label="Income", y_label="Spending",
    x_scale="log", xlim=(5, 2000), ylim=(0, 500),
    color="#163d68", point_size=4, opacity=0.65,
    width=900, height=500, grid=True,
)

paper = chart.with_options(
    title="Figure 1. Income and spending", grid=False, width=600, height=400,
)
paper.to_latex("income.tex", standalone=True)
paper.save_html("income.html")
```

`with_options()` returns an independent specification; it does not change the
original chart, its observations, missing values or sample counts. The settings
are included in `model_dump()` and travel with the chart into the workspace,
offline HTML, notebook display and LaTeX export. Defaults retain the white and
navy appearance.

Explicit dimensions are converted from CSS pixels at 96 pixels per inch for
LaTeX. Choose a width that fits the target manuscript, or leave width unset to
use the default `0.9\linewidth` in LaTeX.

| Option | Meaning |
|---|---|
| `title` | Visible title; also retained in exports. |
| `width`, `height` | Chart dimensions in CSS pixels: 320–2,400 wide and 240–1,600 high. The surrounding workspace remains responsive. |
| `color`, `palette` | Hex colors for marks or series. Donut colors map to categories. |
| `opacity` | Mark opacity from 0 to 1. |
| `point_size` | Marker radius in CSS pixels, from 1 to 24, where markers exist. |
| `line_width` | Stroke width in CSS pixels, from 0.25 to 12, where strokes exist. |
| `grid` | Show or hide the Cartesian grid. |
| `legend`, `legend_position` | Show or hide a categorical legend; position it at `"top"` or `"bottom"`. |
| `x_label`, `y_label` | Labels of the physical horizontal and vertical axes. For `barh`, horizontal is the value axis. |
| `xlim`, `ylim` | Finite, increasing axis limits. They change the visible viewport, not the underlying data or export table. |
| `x_scale`, `y_scale` | `"linear"` on numeric axes; `"log"` is supported on numeric scatter and line axes. |
| `x_format`, `y_format` | Numeric tick labels: `"auto"`, `"number"`, `"integer"`, `"percent"` or `"scientific"`. This formats labels without rounding stored values. |

Options must suit the chart. Category axes cannot receive numeric limits or log
scales; donut charts have no Cartesian axes. Bar, histogram and area value
limits must include zero, preserving the baseline; coefficient limits retain
the zero reference. Log scales require positive
finite plotted coordinates and positive limits; they never drop or clamp
nonpositive coordinates to make the chart render. Invalid names, values and
inapplicable settings raise an error instead of being silently ignored.

For example, a horizontal bar chart uses `xlim` for its numeric range:

```python
charts.barh(
    data={"region": ["North", "South"], "sales": [18, 27]},
    x="region", y="sales", x_label="Sales", y_label="Region",
    xlim=(0, 30), color="#163d68", legend=False, grid=True,
)
```

The workspace only offers chart-type conversions compatible with the supplied
settings. An explicit value range also disables percentage view, which would
otherwise replace those units. The export table retains all stored observations
even when axis limits crop the visible plot.

[The complete example](examples/flexible_charts.py) generates all nine chart
types as offline HTML and standalone LaTeX files.

Repeated categorical labels raise an error. Aggregate explicitly before plotting; no helper chooses a grouping or statistical summary for you. Non-finite numeric values count as missing. Integers outside JavaScript’s exact range (±9,007,199,254,740,991) are rejected instead of silently merging adjacent values; rescale them explicitly or use category labels. Bar and area preserve missing values as `null`; composition charts reject missing or negative components.

Numeric lines retain every row up to 10,000 rows. They sort x within runs separated by missing x, retaining missing y positions; they never bridge a gap through downsampling. Histograms accept 1–200 bins; coefficient charts accept at most 1,000 terms. Categorical charts are limited to 1,000 categories, 24 series, and 10,000 cells. Larger inputs require an explicit filter or aggregation. These limits protect interactive rendering without silently changing a composition.

The plotting package is distinct from the model-fitting engine: a large-data
regression can feed a small coefficient plot without shipping every observation
to the browser. Current charts do not provide faceting, arbitrary layered plots,
zoom/pan, geographic maps or automatic statistical fitting. Numeric scatter and
line helpers currently represent one x/y pair; categorical helpers accept
multiple series.

## Annotations

Add plain text, an arrow pointing to a value, reference lines, or shaded numeric
intervals. Each method returns an independent copy and appends to its existing
annotations; observations, fitted values, bin counts and axis ranges stay intact.

```python
chart = charts.line(
    data={"year": [2018, 2019, 2020, 2021], "value": [4, 5, 3, 7]},
    x="year", y="value", title="Annual outcome",
)
chart = (
    chart.vspan(2019.5, 2020.5, text="Policy period", opacity=0.12)
    .hline(5, text="Target", dash="dashed")
    .annotate("Policy introduced", x=2020, y=3, arrow=True, dx=24, dy=-36)
    .annotate("Source: survey", x=0.02, y=0.95, coords="axes", font_size=10)
)
chart.save_html("annotated.html")
chart.with_options(width=600, height=400).to_latex("annotated.tex", standalone=True)
```

| Method | Position |
|---|---|
| `annotate(text, x=..., y=...)` | Text at a data coordinate; `arrow=True` points from its label to the coordinate. |
| `vline(x, text=None, ...)` | Vertical reference line on a numeric horizontal axis. |
| `hline(y, text=None, ...)` | Horizontal reference line on a numeric vertical axis. |
| `vspan(x0, x1, text=None, ...)` | Shaded interval on a numeric horizontal axis. |
| `hspan(y0, y1, text=None, ...)` | Shaded interval on a numeric vertical axis. |

Data coordinates refer to the physical axes. On a vertical bar chart, `x` is an
exact category label and `y` is a numeric value; on horizontal bars the roles
reverse. A coefficient plot uses its estimate for `x` and exact term label for
`y`. Logarithmic coordinates must be positive. Reference lines and spans apply
only to numeric axes.

With `coords="axes"`, `x` and `y` are fractions from 0 to 1: `(0, 0)` is the
bottom-left and `(1, 1)` the top-right of the plotting area. These relative text
and arrow annotations work on every chart, including donut charts. `dx` and `dy`
offset text in CSS pixels; positive `dx` moves right and positive `dy` moves down.
Text alignment accepts `"left"`, `"center"` and `"right"`.

Annotation styles include hex `color`, `font_size` from 8 to 36, reference/arrow
`line_width` from 0.25 to 12, `dash="solid"`, `"dashed"` or `"dotted"`, and band
`opacity` from 0 to 1. Bands draw behind marks; labels, arrows and reference lines
draw above them. Content outside the plotting area is clipped, and annotations
never expand the data domain. Composition percentage view is unavailable when
data-coordinate annotations would change units; relative annotations remain valid.

For structured configuration, pass `annotations=[...]` directly to a chart
helper or `with_options()`. Each dictionary uses a `type` of `"text"`, `"arrow"`,
`"vline"`, `"hline"`, `"vspan"` or `"hspan"`, with the applicable fields above.
`with_options(annotations=[])` clears annotations. A chart accepts up to 100
annotations with at most 500 characters per note. Unknown fields, invalid
coordinates and unsupported combinations raise an error.

Annotations persist in workspace results, saved specifications, expanded view,
SVG/raster export, offline HTML and LaTeX. Text is plain text, not HTML or TeX
code. [The annotated examples](examples/annotated_charts.py) cover all nine types.

## Data and model adapters

```python
charts.scatter(data={"x": [1, 2, 4], "y": [2, 3, 8]}, x="x", y="y")
charts.hist(data=[{"value": 1}, {"value": 2}, {"value": None}], x="value")

# A pandas DataFrame uses bounded positional blocks; pandas is optional.
# charts.line(data=frame, x="time", y="value")

model = {"coefficients": [
    {"term": "education", "estimate": 1.2, "ci_low": 0.8, "ci_high": 1.6}
]}
charts.coefficients(model)
```

Coefficient records may instead be objects with `term`, `estimate`, `ci_low`, and `ci_high` attributes. Invalid or reversed intervals raise a clear error. Labels and all serialized values are validated; non-finite coefficient values are rejected.

## Export and notebooks

`PlotSpec.model_dump()` returns JSON-safe data. `to_html(height=480)` returns a complete document with a visible title; `height` is the minimum chart workspace height, and the renderer may expand for labels; `save_html(path)` writes it. Notebook HTML uses an opaque-origin iframe with script and download permissions, isolating chart content from the host page. Chart labels are escaped during HTML serialization.

The HTML document blocks external network requests. D3, the renderer, Barlow font, and their notices ship inside the package; asset licensing is documented alongside them. The browser can download SVG, PNG, JPEG and CSV; raster exports use browser canvas, rather than a separate Python rendering engine. Browser export behavior depends on the surrounding notebook application's download policy.

Every supported chart also exports self-contained TikZ/PGFPlots source:

```python
source = chart.to_latex()          # Fragment for an existing document
chart.to_latex("sales.tex", standalone=True)
```

`.latex` returns the same fragment. Optional `caption` and `label` use safe TeX
escaping. Fragments need `tikz`, `pgfplots` and `\pgfplotsset{compat=1.18}`;
standalone documents use XeLaTeX or LuaLaTeX with `fontspec` for Unicode labels.
Coordinates, intervals, missing-data gaps and display sampling match the chart
specification. No Python LaTeX engine or additional package is required.

## Development

From the repository root:

```sh
uv run pytest packages/openecon-charts/tests
uv build --package openecon-charts
```

The Python distribution is Apache-2.0. Vendored assets retain their own licenses.
# Large, replayable tables

All data charts accept a source exposing `columns` and
`iter_batches(columns=..., batch_rows=...)`, including OpenEconometrics `Dataset`:

```python
df = oe.read("large.parquet")  # automatically returns Dataset for large tables
oe.plot.scatter(data=df, x="education", y="wage")
oe.plot.hist(data=df, x="wage", bins=40)
oe.plot.bar(data=df, x="region", y="wage", aggregate="mean")
```

Scatter uses at most 2,000 deterministic, evenly spaced finite-pair ranks and
retains endpoints. Histograms count every finite value in two passes; Torch
uses CPU float64 bucketization when available, with an exact standard-library
fallback. Neither path collects all source values. Every replay verifies all
projected source values, including values omitted from the display.

Line charts retain missing-value gaps and their explicit 10,000-row limit.
Categorical charts retain unique-row semantics by default. Explicit `sum`,
`mean`, or nonmissing `count` aggregation is available for bar, horizontal bar,
area, stacked bar and donut charts. Aggregations retain at most 1,000 categories
and 10,000 series/category cells; arbitrary high cardinality fails with an
actionable filter request. Missing values are ignored within groups and
all-missing groups remain gaps. Coefficient-table sources retain their existing
1,000-term limit. `config.processing` records counts, pass count, sampling or
aggregation and integrity scope. These memory limits do not guarantee elapsed
time for a particular row count.

## Resident numeric tables

`hist`, `scatter` and `line` also accept resident pandas DataFrames, mappings
of sized sliceable columns, and sequences of row dictionaries. They project
only requested columns in blocks of at most 65,536 rows, using the same numeric
reductions as `Dataset`. DataFrame rows are sliced before column selection,
and typed NumPy/pandas columns preserve their dtypes and positional order.
Python lists keep their original scalar values, including precision checks
for large integers. Optional pandas and Torch dependencies remain optional.

Histograms count all finite values. Scatter uses finite-pair ranks, complete
counts and full source extents; log-axis validation includes unsampled values.
The 10,000-row line guard runs before copying a sized resident input. Smaller
lines preserve irregular x distances and missing-value gaps. No implicit line
aggregation is performed.

Resident inputs exceeding 2,000 source rows retain `config.processing` through
output validation and persistence, alongside presentation options. Small
resident inputs retain their previous configuration format; explicit replayable
sources always retain processing metadata. Unsized one-shot histogram/scatter
iterables and custom `to_dict` objects keep the legacy eager conversion path
and are excluded from the bounded resident-input claim.

The repository's [code-panel example](../../docs/examples/resident_charts.py),
[memory probe](../../benchmarks/resident_charts.py) and
verification receipts (internal evidence excluded from this public snapshot)
cover one- and five-million-row inputs, numeric contracts, publication validation,
saved output readback and a freshly frozen runtime. Resident loading and the
input table's memory are separate from additional chart workspace.
