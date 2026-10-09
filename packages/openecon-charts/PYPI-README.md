# OpenEconometrics Charts

`openecon-charts==0.3.1a1` is an Apache-2.0 Python chart library with no required
Python dependencies. It accepts column mappings, row records and pandas-style
tables without requiring pandas. Python 3.10 or later is supported.

```sh
python -m pip install 'openecon-charts==0.3.1a1'
python -m pip check
```

```python
import openecon_charts as charts

chart = charts.scatter(
    data={"income": [10, 100, 1000], "spending": [8, 60, 450]},
    x="income", y="spending", title="Income and spending",
)
chart.save_html("income.html")
chart.to_latex("income.tex", standalone=True)
```

Saved HTML embeds its scripts, styles and fonts and requires no network access.
The renderer and Barlow font assets are included with their original notices.
The same specifications power charts in the OpenEconometrics workbench.

The package includes scatter, numeric line, histogram, coefficient intervals,
grouped/horizontal bars, signed area, stacked bars, donut and network charts.
Appearance settings and annotations travel with the saved specification.
Helpers preserve observations, missing values and explicit display limits;
they do not silently group observations or choose a statistical estimator.
The [chart guide](https://github.com/bluearf/openeconometrics/blob/main/packages/openecon-charts/README.md)
records chart-specific inputs, limits and unsupported combinations.

Optional network PDF/SVG/PNG export needs an installed Chrome, Chromium or Edge.
It uses a disposable browser profile, does not download a browser and does not
open an existing user profile. Ordinary HTML and LaTeX exports need no browser.

## Source and scope

Charts `0.3.1a1` is also the exact chart dependency of SDK `0.3.19a1`. Installing
charts alone does not install the SDK or its tensor/dataframe dependencies.
The corresponding public source snapshot records the frozen source commit and
file inventory; release provenance records distribution hashes and installation
check scope. Package publication and native desktop acceptance are separate.

- [Public repository and releases](https://github.com/bluearf/openeconometrics)
- [Public source and fixture boundary](https://github.com/bluearf/openeconometrics/blob/main/PUBLIC-SOURCE.md)
- [Chart license](https://github.com/bluearf/openeconometrics/blob/main/packages/openecon-charts/LICENSE)
- [Chart notices](https://github.com/bluearf/openeconometrics/blob/main/packages/openecon-charts/NOTICE)
- [Security reporting](https://github.com/bluearf/openeconometrics/blob/main/SECURITY.md)
