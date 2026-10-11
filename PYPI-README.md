# OpenEconometrics

`openecon` is an Apache-2.0 Python library for statistical and econometric
analysis. It provides model fitting, persistent JSON results, saved prediction,
LaTeX tables and offline charts. The package is an alpha with bounded method and
option coverage; full Stata or other vendor parity has not been established.

These exact alpha packages are distributed as GitHub release assets. The separate PyPI pair remains SDK `0.3.19a1` / charts `0.3.1a1`; the commands below select the GitHub bytes explicitly.

## Install this package version

```sh
python -m pip install 'https://github.com/bluearf/openeconometrics/releases/download/v0.3.46-alpha.1/openecon_charts-0.3.2a1-py3-none-any.whl' 'https://github.com/bluearf/openeconometrics/releases/download/v0.3.46-alpha.1/openecon-0.3.20a1-py3-none-any.whl'
python -m pip check
```

Python must be at least 3.11 and below 3.15. The SDK pins charts `0.3.2a1`
exactly. Charts also works as a standalone package with no required Python
dependencies:

```sh
python -m pip install 'https://github.com/bluearf/openeconometrics/releases/download/v0.3.46-alpha.1/openecon_charts-0.3.2a1-py3-none-any.whl'
```

The base SDK installation does not require the web server, CLI, MCP service or
additional Excel/Stata file readers. Optional application layers are described
in the [distribution guide](https://github.com/bluearf/openeconometrics/blob/main/docs/distribution.md).
For example, Excel and Stata readers use `python -m pip install 'openecon[files] @ https://github.com/bluearf/openeconometrics/releases/download/v0.3.46-alpha.1/openecon-0.3.20a1-py3-none-any.whl'`.

## Fit and save a result

```python
import openecon as oe

frame = oe.example()
result = oe.ols(
    data=frame,
    y="wage",
    x=["education", "experience"],
    covariance="HC3",
    device="cpu",
)
print(result.summary())

saved = result.model_dump_json(indent=2)
restored = oe.ResultBundle.model_validate_json(saved)
print(restored.to_latex())

chart = oe.plot.scatter(data=frame, x="education", y="wage")
chart.save_html("wage.html")
```

Charts embed their scripts, styles and fonts for offline HTML. Optional network
PDF/SVG/PNG export uses an installed Chrome, Chromium or Edge; it does not download
a browser or open a user's existing profile.

## Scope and provenance

SDK `0.3.20a1` and charts `0.3.2a1` form one source-based package pair. The
corresponding public source snapshot records the frozen source commit and file
inventory in `SOURCE-MANIFEST.json`. Release provenance identifies the exact
wheel/sdist hashes and the scope of their installation checks. Numerical tests,
installed-package checks, public release access and registry installation are
separate evidence layers.

Installing these Python packages does not rebuild a native desktop installer.
Device support follows each method's recorded contract; physical CUDA, installed
desktop acceptance, Developer ID signing and notarization have separate evidence.

- [Public repository and releases](https://github.com/bluearf/openeconometrics)
- [Estimator and resource scope](https://github.com/bluearf/openeconometrics/blob/main/docs/capabilities.md)
- [OLS and inference options](https://github.com/bluearf/openeconometrics/blob/main/docs/ols.md)
- [Public source and fixture boundary](https://github.com/bluearf/openeconometrics/blob/main/PUBLIC-SOURCE.md)
- [Apache-2.0 license](https://github.com/bluearf/openeconometrics/blob/main/LICENSE)
- [Third-party notices](https://github.com/bluearf/openeconometrics/blob/main/THIRD_PARTY.md)
- [Security reporting](https://github.com/bluearf/openeconometrics/blob/main/SECURITY.md)
