# Clean-profile Mac installation, update and rollback

MARKET-82 was exercised on 7 October 2026 with the **installed** 0.3.43 QA app,
macOS 26.6.2, Apple M3 Pro (12 logical CPUs, 18 GiB). The
[machine-readable record](evidence/clean-mac-install-rollback-2026-10-07/summary.json)
contains artifact hashes, source tests, runtime receipts and explicit limits.

Two initially absent application/WebKit profiles were used on an existing Mac.
Caches were not purged. The separately named, ad-hoc signed QA app was copied
from a read-only DMG into a per-user Applications directory. Its identifier was
`org.openecon.qa.install-rollback-20261007`; the human production profile was
never replaced. The shell and web assets enable opt-in QA timing instrumentation.
This is local installation evidence, not a signed/notarized public release.

## Findings and fixes

Native uv 0.9.26 split absolute requirement-file arguments at whitespace.
Every normal macOS `Application Support` path contains a space, so the installed
package command failed before resolving a wheel. The installer now supplies its
fixed generated filenames relative to its staging working directory. Real uv
tests cover spaces and Unicode, and fresh/existing installed QA projects both
completed actual PyPI resolution, hashed wheel installation and import checks.

The Homebrew Python used by the first candidate bundled `libcrypto.3`, `libssl.3`,
`liblzma.5` and `libsqlite3` with minimum macOS 26, despite app metadata declaring
15. The finalizer now inspects thin/universal Mach-O minimum-OS commands and
rejects inconsistent installers. It also requires relative, in-bundle aliases.
Rebuilding with official uv-managed CPython 3.13.5 and the locked environment
produced 152 audited Mach-O files with no violation of the declared macOS 15
minimum. Actual execution was tested on 26.6.2; this is not execution evidence
for a macOS 15 host.

A desktop-extra-only rebuild initially changed the protected dependency set.
Opening an existing package overlay correctly failed closed. The final build
retains all 54 protected versions using the locked all-extras build environment;
the compatibility check remains intact. Setuptools already exposes a vendored
`more-itertools` at the pinned version, so an initial fixture assertion about
its import path was corrected. `boltons` supplies the independent overlay import
proof; the receipt retains this distinction.

## Installed behavior and timing

Native-process timing starts at the OS process creation timestamp and ends at
the catalogue's double-animation-frame QA event. Web timing starts at the web
performance origin and is reported separately. These individual observations
include the existing host's cache and concurrent workload.

| Current 0.3.43 observation | Time |
|---|---:|
| First launch with empty application and WebKit storage: process to catalogue | 2.290 s |
| Same profile reopened: process to catalogue | 2.252 s |
| First catalogue, web origin to paint | 83 ms |
| Reopened catalogue, web origin to paint | 146 ms |
| First actual Torch/480-row HC3 OLS/table/chart execution in empty profile | 1.236 s |
| Fresh-profile uv installation of `humanize==4.14.0` | 1.411 s |
| Authenticated owned-project loading, observed upper bound | 7.686 s |

The authentication bound includes automation delay and is not a service latency
benchmark. Real native password login used an existing synthetic QA owner.
Reopen, update, rollback and restoration showed that owner without reentering
credentials. Opening the owned cloud project loaded its draft and history;
no cloud code was executed. The
[timing record](evidence/clean-mac-install-rollback-2026-10-07/timings.json)
also retains the earlier baseline and update observations. A capture process
stopped at an app-exit race on two earlier launches; their native timings remain
unavailable. The capture now skips incomplete exit samples, and subsequent
first/repeat launches have complete process timestamps.

The installed executable's `--smoke-test` passed from `/tmp` with
`env -i PATH=/usr/bin:/bin`. It started its own bundled server, spawned its own
worker, computed OLS/LaTeX and removed its temporary project. Server readiness
was 0.826 s and 0.520 s on repeat; these values are not GUI readiness.

The saved proof script additionally asserts frozen execution, bundle-relative
Torch/SDK imports and no checkout import paths, and records an absent inherited
PATH. It denies
worker socket connections during import, CSV reading, OLS and complete JSON/
LaTeX readback, with zero attempted connections. The OS itself remains online.
Auth, cloud loading and first PyPI downloads require network access; local saved
computation does not. Only this bounded CPU calculation was executed, not every
registered estimator or CUDA operation.

## Preservation

The update/rollback sequence used a pinned earlier QA runtime, desktop 0.3.42,
then the compatible 0.3.43 app, then 0.3.42, then 0.3.43 again. The baseline's
metadata truthfully declares macOS 26 to match its older Homebrew dependencies;
it does not certify a historical public 0.3.42 release.

Before executing code after each replacement, snapshots matched source/data/
saved-result/history bytes, registered data, package generation and overlay
bytes, declared requirements, catalogue, stable loopback port, pane preferences
and the cached identity marker. Initially this covered 9 workspace files and
83 overlay files. After deliberate additional installs it covered the same
9 workspace files and 138 overlay files. Rollback imported the saved packages
and read the saved 480-row model. Pane values remained 250 px / 52% / 246 px.
The authentication token contents were never read for these snapshots.

The extra empty-profile experiment archived and restored only the owned QA
profile and WebKit store; its final restoration snapshot also matched. All 50
audited pre-existing human files remained unchanged. The
[preservation record](evidence/clean-mac-install-rollback-2026-10-07/preservation.json)
contains matching before/after snapshot and component hashes.

## Reproduction and evidence boundaries

Use `scripts/audit_macos_installer.py APP --output audit.json` for the bundle
contract, `scripts/verify_advanced_bundle.py APP --output parity.json` for source
equality, `scripts/prove_installed_runtime.py` inside an owned synthetic installed
project, and `scripts/snapshot_install_profile.py` for a disposable QA profile.
The latter refuses production identifiers and excludes auth token stores.

The final frozen runtime matches 483 owned modules and 70 UI assets at the
recorded implementation commit. Source validation passed 247 targeted tests
and Ruff; source tests and installed results are separate records. Existing
oracle libraries are absent from the bundle. No CI result is claimed.

The current installed QA app and exactly one previous QA rollback remain.
Three owned failed bundle copies and two failed-candidate DMGs were pruned after
payload audit. Canonical release-cache cleanup refused because its required
current-release verification record was missing/invalid; unrelated release
caches were preserved. Developer ID, notarization, public distribution, Windows,
CUDA and licensed vendor comparisons remain separate acceptance work.
