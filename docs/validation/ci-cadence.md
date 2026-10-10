# Continuous and daily verification

Repository-wide instructions for agents are in [AGENTS.md](../../AGENTS.md).
Apply that policy instead of older per-PR full-gate requirements in prior notes.

Since 2026-10-10, pull requests, merge-queue candidates and pushes to `main`
run **Quick OpenEconometrics checks**. The repository's existing required check,
`OpenEconometrics / merge gate`, succeeds only when both Python 3.11/3.13 smoke
jobs and the web job succeed. Failed, cancelled and skipped prerequisites fail
this gate. The strict current-base repository rule remains enabled.

Quick checks cover core estimator/reference results, bounded conjoint covariance/target oracles, data readers, LaTeX/saved
output, package metadata, process cleanup and small full-gate authentication
fixtures (including the daily job name and scheduled-event binding). Python 3.13 also checks Ruff and
the generated capability/editor catalogues. The web job runs the existing web
and Cloud source-guard tests and builds the application. SDK and web test commands
have a three-minute limit; job limits include checkout and installation.
Quick receipts explicitly state that they do not cover the complete SDK suite.

**Verify OpenEconometrics** (`ci.yml`) now runs daily at **01:17 UTC / 04:17
Europe/Istanbul**, using the latest default-branch commit. GitHub may delay
scheduled jobs. It can also be started with **Run workflow** in Actions for a
specific branch. Daily and manual runs have separate concurrency groups and are
not cancelled by new pull-request commits.

The full workflow retains both locked Python environments, all eight SDK shards,
all current registered selectors (160 after the BMA, conjoint, simultaneous score-contrast, Weibull and conjoint bootstrap extensions), package builds, original artifacts and the complete
source/run-bound aggregate validator. Its result is named
`OpenEconometrics / daily full gate`. Failed daily runs remain failed and retain
their evidence. A quick green PR check does not establish full numerical
regression coverage; the daily result must be checked before a release.

The 15m37s failure in main run `38039163245` was an intermittent readiness-file
race, not a 15-minute timeout. The Python 3.13 shard ran 3,337 tests, with one
failure: a process-cleanup fixture read an empty PID file between creation and
write. PID readiness markers are now published by atomic replacement, with a
regression that pauses the write and proves no incomplete public marker exists.
This small fixture fix is also present in the separate Actions performance PR
#221; its other dependency/performance changes remain outside this change.
