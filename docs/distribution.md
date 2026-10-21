# Distribution extras and dependency scope

The library wheel supports fit, result persistence, JSON/LaTeX exports and charts
without installing the web server, CLI, MCP or additional file readers.

## Metadata-only PyPI alpha refresh

This checkout prepares [openecon 0.3.20a2](https://pypi.org/project/openecon/0.3.20a2/)
and [openecon-charts 0.3.2a2](https://pypi.org/project/openecon-charts/0.3.2a2/).
The refresh corrects package descriptions and installation commands. Its runtime
source matches the published `0.3.20a1` / `0.3.2a1` pair apart from the two
literal package version constants. Existing release files remain immutable.
Protected publication and exact PyPI hash readback establish registry availability;
preparing this source or passing a package check alone does not publish it.
Use an explicit prerelease version pin on Python 3.11–3.14:

```sh
python -m pip install 'openecon==0.3.20a2'
```

The SDK requires exactly `openecon-charts==0.3.2a2`. For charts alone, run
`python -m pip install 'openecon-charts==0.3.2a2'`; it has no required Python dependencies.

| Installation | Included application layer |
| --- | --- |
| `python -m pip install 'openecon==0.3.20a2'` | Scientific library and openecon-charts |
| `python -m pip install 'openecon[files]==0.3.20a2'` | Excel and Stata readers |
| `python -m pip install 'openecon[cli]==0.3.20a2'` | Command-line interface |
| `python -m pip install 'openecon[server]==0.3.20a2'` | Local HTTP workspace service |
| `python -m pip install 'openecon[agent]==0.3.20a2'` | MCP service |
| `python -m pip install 'openecon[app]==0.3.20a2'` | Files, CLI, server and agent together |
| `python -m pip install 'openecon[desktop]==0.3.20a2'` | Application plus bundled pip/uv tooling |
| `python -m pip install 'openecon[cloud]==0.3.20a2'` | Application plus authentication/storage services |

The table describes optional Python layers for this metadata-refresh version. Native
Mac and Windows installers have separate source pins and acceptance. The public
Mac **0.3.43** bundles SDK **0.3.18a1** / charts **0.3.0a1**. Windows **0.3.45**
bundles SDK **0.3.19a1** / charts **0.3.1a1**, but its newer build source differs
from the PyPI pair despite equal package version strings. Read each
[public release's provenance](https://github.com/bluearf/openeconometrics/releases)
before installation. The [dated distribution ledger](validation/distribution-status-2026-10-11.md)
records source/build/tag pins, artifact hashes, producer runs, acceptance layers
and the anonymous download readback.

## Current source and release acceptance

This source checkout uses SDK **0.3.20a2** and charts **0.3.2a2**. The runtime
identity guard pins public baseline `d11ed130feb4ad8e6e990e61bb0bd8a9101cb981`
and admits only the two literal version replacements. The manifest records the
metadata transforms from that accepted source. Fresh wheel/sdist installation,
README-example and package checks retain their own exact build/provenance records.
This metadata refresh does not claim a new complete scientific validation run.
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
