# Contributing to OpenEconometrics

Coding agents must also read [AGENTS.md](AGENTS.md), which records the current
test cadence, local resource policy and human-approved acceptance boundaries.

Use Python 3.13 and Node.js 22.12 or later. Install the committed dependency
locks from a clean checkout:

```sh
uv sync --frozen --extra app
npm --prefix web ci
npm --prefix web run build
```

Native estimator code uses PyTorch float64. SciPy, statsmodels and other external
estimators belong in development references, not the shipped statistical
runtime. A model change should verify the complete fitted sample, missing and
weight alignment, full covariance, inference, diagnostics and saved results.
Coefficient recovery alone does not establish correctness. Record the exact
reference convention and supported options; do not claim blanket Stata parity.

Run focused checks for the changed behavior. Repository verification commands
include:

```sh
uv run --no-sync ruff check src tests scripts packages/openecon-charts
uv run --no-sync python scripts/generate_capability_docs.py --check
npm --prefix web test
uv run --no-sync python -m pytest
uv build --all-packages
```

Source tests, frozen execution, installed UI, cloud persistence and public
download access are separate evidence layers. Keep fixes and reports explicit
about which layers were verified. Use synthetic data for new tests. Preserve
user project/data identities and never silently truncate a dataset or replace a
scientifically unsupported operation with an unrelated calculation.

Include upstream license texts and attribution when adding dependencies or
assets. Do not commit credentials, customer data, service-account files,
private build output or unreviewed third-party datasets. Retain all license and
notice files in distributions. Report security concerns through the private
process described in `SECURITY.md`, without placing secrets in public issues.

Contributions are submitted under the repository's Apache-2.0 license. Keep PR
descriptions focused on the concrete behavior, reference boundaries and relevant
validation. Pull requests require quick Python 3.11/3.13 and web checks. The full
SDK and package workflow runs daily at 04:17 Europe/Istanbul and can also be
started manually. Check the full result before a release; a quick green check
does not establish complete numerical regression coverage. See the
[CI cadence and exact scopes](docs/validation/ci-cadence.md).
