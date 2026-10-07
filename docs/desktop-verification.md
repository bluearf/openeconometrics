# OpenEcon Desktop 0.3.0 verification

Release evidence recorded on 2026-10-02 for the Apple Silicon macOS build.
This is historical 0.3.0 evidence. See [large local datasets](streaming.md) for
0.3.4 features, benchmarks and installer verification; the current local estimators
support categories and disk-backed cluster inference.
The current delivery scope is Mac only. See the [desktop architecture](desktop.md),
[workbench architecture](architecture.md) and [desktop build instructions](../desktop/README.md).
The current [0.3.33 local installation record](evidence/desktop-0.3.33.json)
documents complete native Dataset option families, exact frozen-source
fingerprints, 30 actual bundled-worker fits and saved-output restart,
installed Mac application and preserved project files. The older artifact
and measurements below describe 0.3.0 only.

## Verified artifact

| Field | Value |
|---|---|
| File | `OpenEcon_0.3.0_aarch64.dmg` |
| Platform | Apple Silicon, macOS 15 or later |
| Size | 269,998,765 bytes |
| SHA-256 | `f0678711f0a14da1d56b59af67a4ba4473bc215204d6f087223666ca2d4516ba` |
| Distribution status | Local verification build; not notarized |

The following timings came from the final mounted DMG on this Mac. Caches were
not purged. The first recorded readiness measurement is **not** an installed
cold-start measurement on a fresh machine.

| Measurement | Observed time |
|---|---:|
| Mounted-DMG readiness, first measured launch | 1.022 s |
| Mounted-DMG readiness, repeat launch | 0.562 s |
| Local CPU OLS and LaTeX check | 1.181 s |

These are individual observations, not cross-machine performance guarantees.
Readiness, authenticated use and local analysis were checked separately.

## Test and native-session evidence

| Check | Result and scope |
|---|---|
| Packaged native tests | 11 passed |
| Frontend tests | 77 passed |
| Full Python suite | 1,469 passed, 3 skipped; before the final cloud change |
| Final cloud change | Subsequent targeted 15-test rerun |
| Native Firebase sign-in | Actual authenticated native-app check passed |
| Session persistence | Reopening the native app retained the authenticated session |

The targeted cloud rerun followed the full Python run; a subsequent full-suite
run is not claimed.

## Cloud deployment and live verification

| Field | Preview evidence |
|---|---|
| Revision | `openecon-teams-preview-00001-vxt` |
| Image version | `v12` |
| Image digest | `sha256:2a551cf6830c41a92ab9c64258fd4afa0db85711345ffe7cad092b544444a95b` |

Preview and production authenticated API QA both passed these checks:

- Real Firebase custom-token sign-in and one-use login proof.
- Version-checked draft saving.
- Idempotent archiving of actual local OLS, D3 and LaTeX output, including a
  duplicate acknowledgement, with zero cloud calculation executions during QA.
- Viewer read access, HTTP 403 for viewer writes and HTTP 404 for an outsider.

Production deployment is confirmed at revision `openecon-00010-46z`, serving
100% of traffic with the reviewed image. Post-deployment authenticated API
verification passed all four checks above.

The private compute service remained at revision `openecon-sandbox-00003-qf7`,
with image digest
`sha256:4304cc205d36dc8a71d575a86c618a4ab86378523d72432ddfbcc57a98a98a70`.

Native-app sign-in and session persistence, packaged local-worker execution,
and authenticated API result archiving were verified as separate checks.
A complete native GUI outbox cycle was not exercised.

## Remaining boundaries

- The Windows CI job never started because of a CI billing block. There is no
  verified Windows build or installed Windows result for this release.
- An initial installed cold launch has not been verified. The DMG measurements
  above retain the existing machine's cache state.
- GPU execution, unlimited data and full Stata parity have not been established.
- Existing loader limits remain 100,000 rows and 100 columns, with 32 MiB local
  files and 24 MiB cloud uploads. Local computation does not remove these limits.
- Apple notarization remains incomplete; macOS 15+ is required by this artifact.

For the calculation and data-size boundaries, see the
[scaling audit](scaling.md) and [workbench architecture](architecture.md).

## Required after each verified installation

After the new native app is installed, reopened and verified, run the cache
cleanup from the canonical repository. First inspect its dry-run output, then
apply the same policy:

```sh
python3 desktop/scripts/cleanup_releases.py
python3 desktop/scripts/cleanup_releases.py --apply
```

This is a mandatory final release step. The default keeps one full release
cache: the version in `desktop/src-tauri/tauri.conf.json`. `--keep 2` also keeps
the preceding marked cache when a release needs a second build fallback. The
script requires matching `source-freeze.json`, `native-package-validation.json`
and successful `packaged-runtime-smoke.json` records under that release's
`desktop/artifacts/release-<version>/build/`. It rechecks the retained native
version, source commit, App payload hashes, strict signature and DMG hash before
deleting anything. Active builds, cached app processes and mounted release
images block cleanup.

After a verified replacement, retain the installed current application and
exactly one previous application rollback. Delete additional older rollback
holders only after an explicit audit. The CLI stays within release caches and
never deletes applications or rollback holders in `/Applications`.

An applied run marks the current cache as generated. Later runs prune only
marked older caches' build targets, bundled runtime copies and dependency/build
outputs; they also remove exact older OpenEcon DMG names from retained caches.
Shared `node_modules` leaf symlinks remain in place. Source text, all small
verification records, unmarked directories, newer versions, human backups and
the installed application remain untouched. Legacy unmarked packages require
the separately audited manual cleanup; they are never automatically adopted.
