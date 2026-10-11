# Windows x64 alpha acceptance

The release acceptance environment is a disposable GitHub-hosted Windows x64
runner. The user selected this environment on 9 October 2026 instead of a
physical Windows PC. Receipts retain the actual operating system, runner image,
source commit, package versions and installer hashes. A Windows Server 2025 run
does not identify itself as a Windows 10 or Windows 11 run.

## Published 0.3.45 acceptance and current 0.3.46 candidate

The published **0.3.45** Windows x64 CPU alpha passed the complete hosted
[producer run 37921893600](https://github.com/bluearf/openecon/actions/runs/37921893600)
on 9 October 2026, attempt 1, at build source
`f1d2eb629db815b716a40e32b07ad55f5d5b5088`. Its installer SHA-256 is
`ccaf15ea9472b89c19f292cff62d94dfa6c92d7f639388d7fc6bca490766515c`
(192,720,594 bytes). The
[public release](https://github.com/bluearf/openeconometrics/releases/tag/v0.3.45-windows-alpha.1)
and attached `windows-release-provenance.json` retain all 18 bounded installed
checks, original/cold native records and saved replay with fitting disabled.
The public tag `1379566c7d09ab29ad49efd14dbfbbabf680e936` is a separate
reviewed source anchor, not the private installer build source. Bundled SDK
0.3.19a1/charts 0.3.1a1 version labels do not establish equality with PyPI bytes.

The current source-versioned desktop candidate is **0.3.46**, with SDK
0.3.20a1/charts 0.3.2a1. Its separate hosted acceptance passed on 11 October:
[run 38150649713](https://github.com/bluearf/openecon/actions/runs/38150649713),
source `6691f50aecadbb1d356af247e5e5c2eff1d1c9ce`, completed 15:47:39 UTC.
The actual installed receipt records 19 completed checks and zero owned
processes after cleanup. This accepted retained installer is not yet a public
release; public Windows remains 0.3.45. See the
[11 October source/run/artifact ledger](validation/distribution-status-2026-10-11.md)
for its exact installer hash, reviewed receipt hashes and retained failures. A successful build or the
older 0.3.45 acceptance cannot certify a newer candidate. New publication is
tracked by [MARKET-515](https://linear.app/bluearf/issue/MARKET-515).

The accepted 0.3.45 workflow used the previous public **0.3.44** installer with SHA-256
`6a6587e4b570f57ae783ce37cfbdc383d6862eadae0b74f648b08cd77a403938`
and build source `8d844ef70a1f6c4ec78d3d3da6790c44591a36e9`.
These previous-version inputs retain their historical scope; current candidate
receipts must record their actual upgrade/rollback versions and hashes.

## Hosted workflow contract

The manual Windows workflow verifies the following layers. The published
0.3.45 receipt proves its completed original cohort; these are required checks,
not a declaration that every newer in-progress run has completed them:

- Exact source blobs, versions and critical installed resources, including the
  locked Tauri NSIS bundle-token transformation.
- A real registered Microsoft WebView2 uninstall, observed dependency absence,
  the installer’s embedded bootstrapper and an actual working native WebView2.
  An unsuccessful uninstall is a failed acceptance, not a simulated absence.
- The normal native application window: create an owned local project, run OLS,
  logit and Poisson through the visible terminal, check independent coefficient
  and information-matrix identities, display all three model tables and save
  their complete JSON state. A full native close/reopen must display the saved
  history and model tables without refitting.
- Enabled Windows firewall profiles with owned application-specific outbound
  Internet block rules while the real local UI and CPU calculation work.
- A request from the installed frozen worker to one pinned public wheel must be
  denied under those rules. Only the two verified rules are then disabled for
  real PyPI installation; the same request must return HTTPS 200. All three
  Windows firewall profiles remain enabled. The rules are re-enabled before
  upgrade, rollback and the native cold reopen.
- Frozen CPU OLS/HC3, JSON/LaTeX/offline charts, real pure/native project packages,
  protected-package rejection and per-project package isolation.
- Actual 0.3.44 installation, a synthetic saved project, replacement by 0.3.45,
  rollback to 0.3.44 and restoration of the candidate. Artifact hashes, editor
  document, original history and environment survive; each installed frozen
  runtime replays the saved result without a new fit.
- Synchronous uninstall, persisted-data preservation and owned process/profile
  cleanup. Firewall rules and profile settings are restored in `finally`.

Only fresh owned synthetic profiles are used. Controllers run outside the
application and do not replace its bundled Python. The WebView2 debug endpoint
exists only on the disposable runner, is attributed to the owned native child
and stays on loopback; it is not enabled in product code.
The fresh cohort also executes weighted categorical Gaussian ridge/lasso/elasticnet
and scalar partial PLS. The installed frozen worker is checked against twelve
compiled source modules, saves ten complete result/data/table artifacts, and
replays after a real server restart with fitting disabled. The normal native UI
renders the four complete tables and reads its one original command after cold
reopening; the installer controller independently compares the actual saved file
hashes. These four-method gates stay disabled in the fixed older-binary diagnostic.
WebView2 150 and later ignore environment overrides in an elevated host.
The runner therefore uses a temporary, exact-executable HKLM debugging policy;
an existing value is never replaced, and each phase removes its owned value.
This follows the [Microsoft WebView2 elevation guidance](https://github.com/MicrosoftEdge/WebView2Feedback/issues/5645#issuecomment-4934355430).

This is an unsigned CPU alpha. Authenticode/SmartScreen and Windows ARM64 remain
separate scopes. CUDA hardware is deferred, and a physical Windows computer is
not required: the hosted installed/native checks are the accepted environment.
Licensed Stata/SPSS/EViews executables are not required; independent scientific
references remain necessary within each method's declared scope.

As of 11 October, [MARKET-91](https://linear.app/bluearf/issue/MARKET-91) is Done:
authentication uses Google or a delivered six-digit code for each new email
session. Historical password setup/link/reset is superseded. This does not
establish installed Windows collaboration acceptance. The broader native
team/project/invitation/transfer and offline reconnect checks remain in
[MARKET-801](https://linear.app/bluearf/issue/MARKET-801)/87/109. The completed
live owned-session cleanup in
[MARKET-802](https://linear.app/bluearf/issue/MARKET-802), including zero-residual
readback after the unchanged 690-second wait, remains accepted separately.

## Historical diagnostic replay

The separately named `OpenEconometrics Windows UI diagnostic replay` workflow
is triggered only by a path-scoped push to `codex/windows-hosted-acceptance` or an
explicit manual dispatch on that exact branch. It can reuse only failed producer run `37868521174`, source
`e281bcf7b98c8b3cbb673a9fd749a054af495703`, and installer SHA-256
`df7b47988f548b018695bc4d05f0819e9be9b734787a058bb4d744c8e93c1a05`.
It verifies immutable artifact archives, original build inputs, native reference
and the original five resource hashes before driving that older installed UI.
The controller commit is recorded separately. Its result is named
`windows-diagnostic-replay.json`, a successful result is `diagnostic_passed`,
and it retains only diagnostic evidence for seven days. It produces no release
installer artifact and cannot certify newer compiled source. A fresh complete
build remains necessary for release acceptance.

Debugger ownership requires an exact registered Microsoft WebView2 executable,
an exact fresh app-local profile parsed from its arguments, the requested port,
and a bounded live ancestry chain to the owned native PID with matching process
creation identity. Windows paths use ordinal case-insensitive comparison.
Microsoft documents the [automatically appended EBWebView subdirectory](https://learn.microsoft.com/en-us/microsoft-edge/web-platform/devtools-mcp-server#step-2-find-the-webview2-user-data-directory); this exact child of each absent app-local root is also attributed.
Failure diagnostics retain only the listener, parsed profile and PID ancestry;
complete process arguments and transient credentials are never retained.
