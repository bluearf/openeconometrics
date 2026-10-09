# Publication tables

All native table and model exports use publication formatting by default. The
rendered preview and copied/downloaded source share the same saved numbers.
Formatting does not refit a model or change its covariance, p-values or sample.

## Models and comparisons

The following example runs in the workbench, which provides `display`.

```python
import openecon as oe

df = oe.example()
m1 = oe.ols(data=df, y="wage", x=["education"], covariance="HC3")
m2 = oe.ols(data=df, y="wage", x=["education", "experience"], covariance="HC3")

m2.to_latex()  # Display one publication table
comparison = oe.regression_table(
    {"Baseline": m1, "Experience controls": m2},
    caption="Wage regressions",
    label="tab:wages",
    precision=3,
    term_labels={"education": "Education", "experience": "Experience"},
    notes="Each column reports a separately fitted specification.",
)
display(comparison)
# Write the complete fragment without any preview limit:
oe.regression_table([m1, m2], "models.tex", precision=3)
```

The body puts each coefficient over its parenthesized standard error and leaves
cells blank when a specification does not include a term. Constants come last.
Observations remain integers. OLS tables show available R-squared and adjusted
R-squared; binary models show available pseudo R-squared and log likelihood.
No statistic or fixed-effect row is inferred from a model name.

Stars use the original saved p-values: `* p < 0.10`, `** p < 0.05`,
`*** p < 0.01`. Rounding and the model's confidence level do not change those
thresholds. Use `stars=False` to omit stars and their note. Notes preserve the
actual standard-error method, cluster column/count when available, inference
distribution, confidence level, estimation sample, exclusions and warnings.
Test conventions come from the saved result: for example ARIMA `/sigma` uses
one-sided inference, unlike its slope tests. The formatter does not recompute
either test. Advanced models retain their explicit equation groups and ancillary
parameter names; terms from different equations never share a comparison row.
Parameters without an equation label appear under `Other parameters` when the
result also contains named equations. No group is guessed from a term's spelling.
Available equation-specific R-squared/RMSE and panel R-squared are preserved.
`model.to_latex(style="diagnostic")` retains the detailed t/z, p-value and
confidence-interval layout for diagnostics.

## Dataframes

```python
df.head().to_latex(index=False, caption="Analysis sample", precision=3)
df.describe().to_latex("summary.tex", caption="Summary statistics")
df.to_latex("data.tex", index=False)  # All rows, not only the visible preview
```

Numeric columns align to the right and text columns to the left. Both dataframe
and model defaults use booktabs top/middle/bottom rules without vertical lines.
Short tables shrink only when they exceed the available line width. More than
40 body rows selects `longtable` with repeated headers and page breaks. Explicit
`longtable=True/False`, `font_size` and dataframe `column_format` support a
manuscript's specific layout; longtables are not boxed, so their column widths
and font size must fit the manuscript when explicitly overridden. Wide or
long-text longtables use bounded paragraph columns by default to keep the
content within the available width without preventing page breaks. Small nonzero numbers retain scientific
notation instead of silently rounding to zero. Labels, captions and table
contents escape TeX special characters.

Saved automatic table/model outputs are presented with the current publication
formatter on read, without rewriting the execution or rerunning Python.
Explicit user-written `oe.Latex` remains exactly as written.

## Preview and complete exports

Automatic dataframe displays contain at most 50 rows and 30 data columns. Their
copied/downloaded TeX contains exactly that displayed data, with a truncation
notice. `df.to_latex("data.tex")` exports the complete frame separately.
The math preview's 30-column budget includes index columns; its own notice
identifies any further hidden columns. Copied TeX retains every column in the
display transport even when the math preview cannot show all of them.

Model and regression-table math previews contain at most 50 **body rows** and
30 columns. Equation headings and coefficient/SE pairs count toward the row
budget; a boundary moves back to keep a heading, estimate and SE together. The
preview states its displayed/total row and column counts. Copied/downloaded
model TeX retains every parameter and fit-statistic row, even beyond this preview.

Console model `Display JSON` preserves all coefficients, inference, warnings and
fit metrics, but omits the covariance matrix and sample positions (listed in
`display_omitted`). Save a complete fitted result explicitly:

```python
from pathlib import Path
from openecon.models import ResultBundle
Path("model.json").write_text(model.model_dump_json(), encoding="utf-8")
saved = ResultBundle.model_validate_json(Path("model.json").read_text(encoding="utf-8"))
saved.to_latex("model.tex")  # Formatting the saved result does not fit again.
```

The current `publication-v2` presentation upgrades automatic `publication-v1`
history on read without rewriting the stored execution. Plain `latex_notes`
are the same notes included in TeX; they remain visible beside the math preview.

The 7 October validation matrix (internal evidence excluded from this public snapshot)
covers 22 real fits across all 18 registered result families, plus wide, long,
multi-model and Unicode layout fixtures. This is representative publication
coverage, not estimator parity for every one of the 80 registered specifications.

## Manuscript integration

Python exports a fragment suitable for insertion into a manuscript. Include
`booktabs`, `adjustbox`, `array` and `longtable` in the manuscript's preamble. The UI's
complete document includes these, `placeins` to preserve the selected output
order, `fontspec`, and the TikZ/PGFPlots packages for
charts, uses a portable default serif font and compiles with XeLaTeX or LuaLaTeX.
The browser math preview is a convenient view; the compiled document establishes
page layout and is the version to review before submitting a paper.

The default conventions are illustrated by NBER working-paper tables, including
[Generative AI at Work, Table 2](https://www.nber.org/system/files/working_papers/w31161/w31161.pdf)
and [Job Search, Wages, and Inflation, Table 2](https://www.nber.org/system/files/working_papers/w33042/w33042.pdf).
They are not an official NBER submission template or a guarantee that every
journal's formatting rules are identical.
