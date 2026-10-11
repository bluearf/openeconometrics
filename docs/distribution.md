# Distribution extras and dependency scope

The library wheel supports fit, result persistence, JSON/LaTeX exports and charts
without installing the web server, CLI, MCP or additional file readers.

## Published PyPI alpha

The published public package pair is
[openecon 0.3.19a1](https://pypi.org/project/openecon/0.3.19a1/) and
[openecon-charts 0.3.1a1](https://pypi.org/project/openecon-charts/0.3.1a1/),
published on 9 October 2026. It follows the
[reviewed Python release](https://github.com/bluearf/openeconometrics/releases/tag/python-v0.3.19a1)
and its `SOURCE-MANIFEST.json`, rather than the newer development checkout.
Use an explicit prerelease version pin on Python 3.11–3.14:

```sh
python -m pip install 'openecon==0.3.19a1'
```

The SDK requires exactly `openecon-charts==0.3.1a1`. For charts alone, run
`python -m pip install 'openecon-charts==0.3.1a1'`; it has no required Python dependencies.

| Installation | Included application layer |
| --- | --- |
| `python -m pip install 'openecon==0.3.19a1'` | Scientific library and openecon-charts |
| `python -m pip install 'openecon[files]==0.3.19a1'` | Excel and Stata readers |
| `python -m pip install 'openecon[cli]==0.3.19a1'` | Command-line interface |
| `python -m pip install 'openecon[server]==0.3.19a1'` | Local HTTP workspace service |
| `python -m pip install 'openecon[agent]==0.3.19a1'` | MCP service |
| `python -m pip install 'openecon[app]==0.3.19a1'` | Files, CLI, server and agent together |
| `python -m pip install 'openecon[desktop]==0.3.19a1'` | Application plus bundled pip/uv tooling |
| `python -m pip install 'openecon[cloud]==0.3.19a1'` | Application plus authentication/storage services |

The table describes optional Python layers for the published version. Native
Mac and Windows installers have separate source pins and acceptance. The public
Mac **0.3.43** bundles SDK **0.3.18a1** / charts **0.3.0a1**. Windows **0.3.45**
bundles SDK **0.3.19a1** / charts **0.3.1a1**, but its newer build source differs
from the PyPI pair despite equal package version strings. Read each
[public release's provenance](https://github.com/bluearf/openeconometrics/releases)
before installation. The [dated distribution ledger](validation/distribution-status-2026-10-11.md)
records source/build/tag pins, artifact hashes, producer runs, acceptance layers
and the anonymous download readback.

## Current source and release acceptance

This source checkout uses SDK **0.3.20a1** and charts **0.3.2a1**. Its generated
capability catalogue describes the checkout, including work beyond the published
PyPI snapshot. These candidate versions require their own public-source, package
installation and PyPI publication acceptance before replacing the published pair.
Use `uv sync --frozen --extra app` for the local application,
`--extra cli` for the CLI, and `--extra desktop` for freezing. The launcher and
locked runtime exporter select their extras explicitly.

Desktop candidate **0.3.46** retains its own installed-app and public-release
acceptance. Mac installers require Apple Silicon and macOS 15 or later;
Developer ID signing and notarization are deferred until Apple Developer Program
enrollment; internal ad-hoc signed Mac alpha acceptance is permitted. Method-specific device
contracts retain their documented scope: supported Mac Metal factors use float32
preconditioning with CPU float64 refinement, MPS float64 inference is unsupported,
and physical NVIDIA/CUDA acceptance is deferred and is not a prerequisite for
this CPU alpha. GitHub-hosted Windows acceptance is sufficient; a physical
Windows PC is not required. Licensed Stata/SPSS/EViews executables are not
required for scientific validation, which retains independent analytical,
development-reference and synthetic-fixture evidence without a vendor-parity claim.

The [11 October ledger](validation/distribution-status-2026-10-11.md) separates
quick checks, the scheduled daily full gate, manual release validation and the
Windows installer workflow by exact tested source. A failed full run remains
failed even when another source's quick check passes. New source methods are not
automatically present in the older published installers or PyPI packages.

Missing extras produce an install hint naming the required extra. CSV/Parquet
readers remain in the scientific layer; Excel/Stata reads require `files`.
Development oracle packages belong to the dev dependency group. The frozen
runtime excludes SciPy, statsmodels and linearmodels.

## Historical library-wheel measurement

These measurements predate the first PyPI publication and retain their original
source, platform and dependency scope.

The measurement record (internal evidence excluded from this public snapshot) comes
from a fresh, isolated macOS ARM64 environment with the locally built wheels
installed into site-packages. It verifies a 480-observation HC3 OLS fit, an exact
JSON result round trip, LaTeX output, standalone chart HTML and actionable errors
for missing CLI/server/agent/desktop/file extras.

The environment contains **25 distributions** totaling
**766,389,624 bytes (730.9 MiB)**, measured by summing installed
wheel RECORD files. This is an installed-file measurement, not compressed wheel
size, process memory or the complete desktop application size.

Three `import openecon` timings were 21.4 ms, 13.6 ms, 13.8 ms.
They measure the lazy library import inside fresh processes with process startup
excluded; they do not measure first PyTorch import or model execution. The full
fit/export verification took 4.13 seconds on this machine.

NumPy and NetworkX are still installed transitively through pandas/PyTorch. The
split removes forced application-layer dependencies; it does not claim removal
of those transitive distributions. Exact installed versions and bytes per
package are in the record. Results are specific to this platform and dependency
resolution.

Reproduce after building wheels and installing them in a fresh environment:

```sh
python scripts/verify_library_distribution.py --python /absolute/venv/bin/python --output library-wheel.json
```
