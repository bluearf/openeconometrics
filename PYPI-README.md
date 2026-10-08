# OpenEconometrics

`openecon` is an Apache-2.0 Python library for statistical and econometric
analysis. It provides model fitting, persistent JSON results, saved prediction,
LaTeX tables and offline charts. The package is an alpha with bounded method and
option coverage; full Stata or other vendor parity has not been established.

## Install

```sh
python -m pip install 'openecon==0.3.18a4' 'openecon-charts==0.3.0a2'
python -m pip check
```

Python must be at least 3.11 and below 3.15. The reviewed Linux package checks
cover Python 3.11 and 3.13. The SDK pins charts `0.3.0a2` exactly. Charts also
works as a standalone, dependency-free package:

```sh
python -m pip install 'openecon-charts==0.3.0a2'
```

The base SDK installation does not require the web server, CLI, MCP service or
additional Excel/Stata file readers. Optional application layers are described
in the [distribution guide](https://github.com/bluearf/openeconometrics/blob/main/docs/distribution.md).
For example, Excel and Stata readers use `python -m pip install 'openecon[files]==0.3.18a4'`.

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

SDK `0.3.18a4` corrects package documentation and metadata. Its estimation,
parser and chart implementations remain those of the reviewed SDK `0.3.18a3`
and charts `0.3.0a2` source. Charts `0.3.0a2` is reused without rebuilding its
published files. Numerical tests, installed-package checks, public releases and
registry installation are separate evidence layers.

The Mac desktop release `v0.3.43-alpha.1` embeds SDK `0.3.18a1`; installing this
Python package does not rebuild that application. Physical CUDA, Windows,
Developer ID signing and notarization acceptance are separate from these package
checks.

- [Public repository and releases](https://github.com/bluearf/openeconometrics)
- [Estimator and resource scope](https://github.com/bluearf/openeconometrics/blob/main/docs/capabilities.md)
- [OLS and inference options](https://github.com/bluearf/openeconometrics/blob/main/docs/ols.md)
- [Public source and fixture boundary](https://github.com/bluearf/openeconometrics/blob/main/PUBLIC-SOURCE.md)
- [Apache-2.0 license](https://github.com/bluearf/openeconometrics/blob/main/LICENSE)
- [Third-party notices](https://github.com/bluearf/openeconometrics/blob/main/THIRD_PARTY.md)
- [Security reporting](https://github.com/bluearf/openeconometrics/blob/main/SECURITY.md)
