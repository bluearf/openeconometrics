# OpenEconometrics agent instructions

These instructions apply throughout this repository. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[CI policy](docs/validation/ci-cadence.md) before choosing validation.
The human's latest instructions take precedence.

## Current testing policy (2026-10-10)

This policy replaces older notes that require the entire eight-shard SDK gate
for every PR or issue batch. Do not restore that requirement from memory or an
old worktree. Fetch current `main` and integrate it into your own branch when
needed; preserve other people's branches, worktrees and uncommitted changes.

- For each change, run focused tests that exercise the affected behavior and
  meaningful failure cases. Reuse an appropriate existing environment with the
  committed dependency locks. Documentation-only edits need link/content checks,
  not new unit tests or a local full numerical run.
- PRs, merge-queue candidates and pushes to `main` run
  `.github/workflows/ci-quick.yml`: core Python checks on 3.11 and 3.13, small
  authentication/process fixtures, lint, generated catalogues, web tests and the
  web build. The required check remains `OpenEconometrics / merge gate`.
  Require a fresh successful check on the current candidate and base; keep the
  strict repository rule. Do not bypass, forge or manually mark a check green.
- `.github/workflows/ci.yml` runs the complete registered SDK/package gate daily
  at **01:17 UTC / 04:17 Europe/Istanbul**, or by manual dispatch. Its aggregate
  is `OpenEconometrics / daily full gate`. Preserve both Python minors, all
  eight shards, the registered selector inventory, artifacts and validation
  limits. Read the current inventory from `scripts/verify_merge_candidate.py`;
  counts in old reports are historical snapshots.
- Do not dispatch the full workflow after every small change, commit or issue
  closure. Use it for release validation, an explicit human request, or when a
  material SDK/validation change needs coverage beyond focused and quick tests.
  Check the full result and its exact tested source before a release. A prior
  daily pass on another commit does not prove a later change.
- Treat failed tests as failures. Diagnose the saved log before retrying; do not
  remove cases, relax limits, accept skipped/missing evidence or replace an
  independent reference just to obtain green CI. A daily job failure and a quick
  PR result are separate facts; report their scopes accurately.

## Local memory and process use

- Run heavy local test batches sequentially. Avoid multiple complete pytest
  runs, full SDK matrices or packaging environments at once on the user's Mac.
  Prefer GitHub runners for the full matrix.
- Cap numerical threads for focused tests, for example:

  ```sh
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    uv run --no-sync python -m pytest tests/test_affected_behavior.py
  ```

  Replace the example selector with actual affected tests. Do not casually run
  bare `pytest`, the entire repository suite or `verify_merge_candidate.py` as a
  routine per-issue local check. The hosted workflows already cap native threads.
- Close test-owned workers, child process groups, temporary servers and browser
  sessions on success, failure and timeout. Use the repository's ownership-aware
  cleanup helpers and `finally` blocks. Never broadly kill Python, Chrome, Codex
  or the user's OpenEconometrics session to free memory. Do not delete saved
  projects/results or change test admission limits as a cleanup measure.

## Human-approved acceptance boundaries

- Use the available Mac CPU runtime; use MPS only for a supported implementation
  path. CUDA hardware and CUDA acceptance are **deferred and not required**.
  Do not block other work on obtaining CUDA hardware.
- GitHub-hosted Windows runners are accepted for Windows testing. A physical
  Microsoft/Windows computer is **not required**. A build alone still does not
  prove installation, startup, UI behavior or saved-state restoration; test the
  relevant layer on the hosted runner when claiming it.
- Licensed Stata/SPSS/EViews executable comparisons are **not required and will
  not be provided**. Do not keep issues open for that requirement. Use independent
  analytical results, supported development references and synthetic fixtures;
  never describe this as vendor parity or invent vendor execution evidence.
- Apple Developer ID signing and notarization (MARKET-79) are **deferred** until
  Apple Developer Program enrollment completes, expected in late October or
  early November 2026. Until then, ad-hoc signed Mac alpha builds are accepted
  for internal testing. Do not block the Mac alpha (MARKET-515) or other work on
  Developer ID. Never describe an ad-hoc build as notarized, Developer ID signed
  or Gatekeeper-approved.

## Evidence and issue closure

For model changes, verify the relevant sample/missing/weight alignment, full
covariance and inference, diagnostics, postestimation and saved-result replay,
not coefficients alone. Keep source tests, wheel/frozen execution, installed UI,
Windows acceptance, persistence and public release evidence distinct. Require
the layers relevant to the actual issue; do not repeat unrelated acceptance.

Close only the implemented, tested scope after integration and tracker readback.
A successful quick gate alone does not establish complete scientific coverage,
installed acceptance or a release. Do not close an unverified parent by
implication when only its children are done. State remaining scope and deferred
hardware honestly, applying the human-approved boundaries above.
