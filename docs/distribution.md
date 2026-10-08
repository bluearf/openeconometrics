# Distribution extras and dependency scope

The library wheel supports fit, result persistence, JSON/LaTeX exports and charts
without installing the web server, CLI, MCP or additional file readers.

| Installation | Included application layer |
| --- | --- |
| `pip install 'openecon==0.3.18a4'` | Scientific library and openecon-charts |
| `pip install 'openecon[files]==0.3.18a4'` | Excel and Stata readers |
| `pip install 'openecon[cli]==0.3.18a4'` | Command-line interface |
| `pip install 'openecon[server]==0.3.18a4'` | Local HTTP workspace service |
| `pip install 'openecon[agent]==0.3.18a4'` | MCP service |
| `pip install 'openecon[app]==0.3.18a4'` | Files, CLI, server and agent together |
| `pip install 'openecon[desktop]==0.3.18a4'` | Application plus bundled pip/uv tooling |
| `pip install 'openecon[cloud]==0.3.18a4'` | Application plus authentication/storage services |

The SDK pins the already-published `openecon-charts==0.3.0a2` distribution.
SDK `0.3.18a4` corrects package documentation and metadata; it keeps the reviewed
a3 estimation/parser implementation. Public release and PyPI installation need
their own completed readback. In a source checkout use `uv sync --frozen --extra app` for the
local application, `--extra cli` for the CLI, and `--extra desktop` for freezing.
The launcher and locked runtime exporter select their extras explicitly.

Missing extras produce an install hint naming the required extra. CSV/Parquet
readers remain in the scientific layer; Excel/Stata reads require `files`.
Development oracle packages belong to the dev dependency group. The frozen
runtime excludes SciPy, statsmodels and linearmodels.

## Historical local wheel measurement

The historical local wheel measurement came from a fresh, isolated macOS ARM64
environment with locally built wheels installed into site-packages. Its internal
receipt is excluded from this public snapshot; these historical figures do not
measure the new PyPI a4 installation. The [public source boundary](../PUBLIC-SOURCE.md)
describes that evidence separation. It verifies a 480-observation HC3 OLS fit, an exact
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
