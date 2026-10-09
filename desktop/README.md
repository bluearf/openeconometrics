# OpenEconometrics desktop

<!-- BEGIN source-generated capability scope -->
Current source: **157 registered fit names**, **92 Dataset fit routes**, **74 common saved predict/margins adapters**. The [generated method/option inventory](../docs/capabilities.md) states conditions, exclusions and devices.

Fit routes, common prediction adapters and family-specific helpers/forecasts have separate contracts. Source implementation does not establish independent scientific validation, installed-package verification or public shipment for a method/option. Those require their own dated, source-pinned evidence; historical measurements retain their original scope.
<!-- END source-generated capability scope -->

Current source scope and versions are recorded in the [generated capability inventory](../docs/capabilities.md). Dated build, client and installation records below retain their original verification scope.

The visible application name is **OpenEconometrics** from desktop 0.3.35.
The Python distribution/import remains `openecon`, the native executable remains
`openecon-desktop`, and the application identifier remains `org.openecon.desktop`.
This preserves the existing data directory, remembered loopback origin and
authentication/storage contracts. Installing the renamed bundle must still be
verified separately with preservation of the existing local project data.

Tauri 2 opens the existing interface from a bundled Python server bound to a
randomly assigned `127.0.0.1` port remembered for this OS account. This stable
origin preserves Firebase sign-in across launches. An exclusive file lock
prevents duplicate runtimes; an occupied saved port fails closed instead of
connecting to another local process. The Python console runs on the user's computer in a
persistent spawned process. It has the user's local permissions. Opening a
project or reading its saved draft never executes code.

The production installer includes a PyInstaller **one-directory** runtime with
Python, the `openecon` package, PyTorch, pandas, Arrow and import/export dependencies.
SciPy is a development oracle dependency and is excluded from the desktop runtime.
It includes native uv and pip for project library installation and needs no
checkout, system Python, separate package manager or repeated extraction on startup. The
webview uses macOS WebKit or Windows WebView2 rather than shipping Chromium.

Runtime builds collect every `openecon.econometrics` submodule explicitly because
the estimator catalogue resolves its entries by lazy import strings. Packaging
fails if collection omits a registered estimator, public helper, forecast or
family manifest. This build contract does not update an existing installed app;
a new frozen runtime and bundled-estimator smoke test are required to ship new
econometrics code.

The 0.3.27 Apple Silicon build was installed and verified locally with a fresh
runtime containing 230 econometrics modules and 80 registered estimators.
The release record (internal evidence excluded from this public snapshot) separates catalogue
resolution, seven actual model fits, installed-app checks, saved-file preservation
and old-cache cleanup.

The 0.3.37 Apple Silicon update adds native sparse Louvain/Leiden communities,
centrality and topology analysis, static GraphML/GEXF/Pajek interchange and
community-grouped network charts. The release record (internal evidence excluded from this public snapshot)
separates source tests, all 373 frozen owned modules, 27 actual packaged network
operations, saved-output reopen, installer/installed-byte verification and
preservation of the existing project. New network algorithms use CPU; graph
storage is resident O(V+E), with explicit memory/work guards and no SciPy or
NetworkX runtime dependency. Physical CUDA was not exercised.

The desktop code toolbar includes project-specific library installation and
versioned team environment sharing. See [project packages](../docs/project-packages.md).

The 0.3.10 interface adds draggable boundaries for the file sidebar, code/results
split and expanded terminal. Sizes are remembered per project on this device
and constrained to the current window. Separators also support arrow keys,
Home/End and double-click to reset. This interface release reuses the verified
0.3.9 scientific runtime; Python package version remains 0.3.8a1.

The 0.3.9 interface has separate sidebar actions for a new blank Python file,
importing a Python file and adding data, with mouse and keyboard tooltips.
Each Python file retains its own content and version. Switching files first
saves the outgoing buffer; a failed save keeps it open. Imported code becomes
a separate file, and duplicate names receive a suffix. Named files persist
locally and synchronize through the existing project membership controls.
This version includes the updated 0.3.8a1 Python runtime for named-file storage.

The 0.3.8 desktop interface adds a collapsible file sidebar and a terminal below
the results. Both the terminal and the library dialog accept bare `pip install`
and `uv pip install` commands through the existing project console. This
interface release reuses the verified 0.3.7 scientific runtime.

The 0.3.7 Apple Silicon installer supports `%uv pip install`, `%uv add`, `%pip
install` and `oe.install(...)`, including version ranges, extras and requirements
files. Actual uv resolution and native-wheel installation were verified inside
the read-only mounted installer without external Python. See
0.3.7 release evidence (internal evidence excluded from this public snapshot).

## Native interface

Use `window.__TAURI__.core.invoke(command, args)` only when
`window.__TAURI_INTERNALS__` is present. Arguments use camelCase:

| Command | Arguments | Result |
|---|---|---|
| `desktop_info` | none | `local_origin`, `data_root`, fixed `cloud_origin`, `version` |
| `cloud_request` | `method`, `path`, optional `token`, optional JSON `body` | `{status, body}`; HTTP errors preserve status and JSON |
| `download_project_file` | `projectId`, `fileId`, `token` | `{cloud_id,name,python_path,data_hash,sha256,size_bytes,reused}` |
| `download_project_files` | `projectId`, `token` | array of cache records; authenticated listing, at most 20 files / 8 GiB shared bytes; legacy inline files retain a separate 64 MiB budget |
| `upload_project_file` | `projectId`, `token` | native chooser; negotiated chunked upload or legacy multipart; picker cancellation returns status 204 |
| `project_transfer_action` | `projectId`, optional `requestId`, `token`, `action` | pending local upload list/status, explicit resume/cancel, or active chunked download cancellation |
| `open_desktop_login` | `requestId` (32 hex characters) | opens fixed cloud login URL in the system browser |

Transport failures reject with `OPENECON_NETWORK`; file-listing and download
HTTP failures reject with `OPENECON_HTTP:<status>`. Local validation, cache and
integrity errors retain ordinary errors, so only connection failures permit
the interface's offline fallback and denied access always fails closed.

Cloud requests always target `https://openecon-291739190496.us-central1.run.app`.
The broker rejects arbitrary URLs, encoded paths, query strings, redirects and
every `/console/execute` route. It accepts a narrow membership, draft and data
API allowlist with a 128 KiB request and 10 MiB response limit. The sole exception
is the explicitly invoked local-result archive route, limited to 3 MiB of JSON.
The source `chunked-v1` protocol permits CSV/Parquet up to 2 GiB per file in
4 MiB verified parts, with explicit resumable upload journals and bounded
download staging. The legacy inline path remains 24 MiB per file / 64 MiB per
project. New protocol implementation is distinct from live deployment and
installed two-member cloud acceptance; see [transfer proof boundaries](../docs/team-data-transfers.md).
Tokens pass directly from Firebase in the webview to Rust;
they are neither sent to Python nor written to its environment or files.

Downloads obtain metadata from the authenticated project listing, verify its
size and SHA-256, and atomically install `data-root/projects/<project>/<name>`.
`python_path` is the original name, so saved `oe.read('wages.csv')` code works.
A cached file is reused only after rereading and hashing its contents. Directory
handles, no-follow opens and portable filename checks reject traversal and
symlinks. The browser has no arbitrary filesystem or shell command permission.
Only the exact running loopback origin receives the application capability;
external navigation and new webview windows are denied.
The bundled navy and white startup screen appears immediately while Python
starts in a background thread and receives no native application permission.

## Build on Apple Silicon

Build prerequisites: Xcode command line tools, Rust and Node.js. Python and
PyInstaller are build dependencies only. Use a compatible standalone Python and
a locked environment (`uv sync --frozen --no-dev --all-packages --all-extras`),
then install `pyinstaller==6.22.3` in that environment. Cloud service modules and
development oracle packages are excluded from the frozen desktop. Then:

```sh
cd desktop
npm ci
python scripts/build_desktop.py --bundles app,dmg
```

For a development checkout, `npm run dev` uses `.venv/bin/python` if no frozen
runtime is present. This fallback exists only in debug builds. Production fails
closed if the bundled runtime is missing. To run a Rust toolchain isolated to
this directory, set `CARGO_HOME=desktop/.toolchains/cargo` and
`RUSTUP_HOME=desktop/.toolchains/rustup` and add its `bin` directory to `PATH`.
The build script automatically uses this isolated toolchain when available.

Artifacts are in `src-tauri/target/release/bundle/macos/` and `bundle/dmg/`.
When an earlier app is running from that folder, use a separate output directory
to preserve its executable and resources:

```sh
python scripts/build_desktop.py --bundles app,dmg --target-dir build/releases/0.3.2/target
```

The app and DMG then appear under that directory's `release/bundle/` as
`macos/OpenEconometrics.app` and `dmg/OpenEconometrics_<version>_aarch64.dmg`.
Disk-image creation stages only that finalized application and the Applications
shortcut; a retained legacy `OpenEcon.app` beside it is neither included nor
deleted. Release-cache cleanup recognizes both exact product names, validates
the retained pair against its approved source configuration, and keeps its
existing source, backup, path, signature and active-build guards.
The configured minimum is macOS 15. The finalizer audits the Mach-O minimum OS
of every bundled native dependency and rejects an installer requiring a newer
system. A Python deployment-target setting alone is insufficient: Homebrew's
Python 3.13.5 on the QA host brought in four dylibs requiring macOS 26. Use a
compatible standalone Python and the locked dependency versions. For an update,
also retain the protected bundled dependency set: removing cloud-related HTTP
dependencies from a build environment can change the frozen core and prevent
existing project package overlays from opening. Do not bypass that check.
The [clean-profile installation record](../docs/clean-mac-installation.md)
documents a compatible managed-Python build and an actual update/rollback.
An unsigned local build is suitable for local verification; signed and
notarized distribution requires the project's Apple Developer identity.

## Build on Windows x64

Run `scripts/build_windows.ps1` on Windows x64 with Visual Studio C++ Build
Tools, Rust, Python 3.13 and Node.js. It exports exact versions from `uv.lock`
without modifying the lock and installs that version of the CPU PyTorch wheel
from the official CPU index. It copies freshly built web assets into the runtime
and produces an NSIS installer under `src-tauri/target/release/bundle/nsis/`.
The manual `.github/workflows/openecon-desktop.yml` workflow uses a standard
Windows 2025 x64 runner; `windows-ci.yml` mirrors that workflow. It records the
exact checked-out commit, component versions, lock/config hashes and runner
image before building. Git preserves LF source bytes and package metadata is
written as UTF-8 independently of the Windows locale. Before tests, it preserves
the raw native producer executable as an acceptance reference. Tauri's locked
NSIS bundler changes its bundle-type marker from `UNK` to `NSS` while packaging
and restores the raw executable afterward; installed identity checks reproduce
only that exact three-byte transformation in memory. Every other native byte
and the frozen resources must match. The raw reference is verification evidence,
not a standalone installer.

The workflow tests native boundaries, distribution metadata and worker startup
with both shell shutdown and input-pipe closure, then installs
the actual NSIS package into a path containing spaces and Turkish characters.
`scripts/verify_windows_installation.ps1` exercises the installed resources and
production executable, real WebView2/native IPC, bundled CPU computations,
saved-result recovery, charts and project package installation. It uses an
owned disposable GitHub runner profile and refuses an existing native data
directory; it is not a general script for a user's existing installation.
After a successful build, all validation phases run independently so
one failing phase does not hide later installation failures. Any failed phase
still fails the job. Installer and source-bound acceptance evidence are
retained as separate GitHub artifacts for 30 days; require the final run to
succeed before distributing the installer. A failed validation may retain a
separately named `Windows-built-unverified` installer for seven days solely for
diagnosis. A workflow definition alone is not a passing build or installation
receipt.

The installer embeds the WebView2 bootstrapper, which needs internet access if
WebView2 is missing. Hosted-runner acceptance uses the runner's provisioned
WebView2. The package is an unsigned x64 CPU alpha; code signing/SmartScreen
reputation, Windows ARM, physical CUDA, missing-WebView2 first installation,
previous-version upgrades and live cloud sign-in/team acceptance are separate
checks. Reinstalling the same NSIS package verifies data preservation for that
operation, not compatibility with an older version. The bundled SDK/charts
versions come from this source commit and need not equal the first pair
published to PyPI.

## Verification

`cargo test --manifest-path src-tauri/Cargo.toml` verifies the broker boundary,
file metadata, cache invalidation and symlink rejection. Run the installed
application executable with `--smoke-test` to start its bundled server, execute
`openecon` OLS in a spawned Python process, report JSON and shut down cleanly.
This smoke test performs no cloud operations and removes its temporary project.

After every verified macOS installation, complete the
[required release-cache cleanup](../docs/desktop-verification.md#required-after-each-verified-installation)
to keep one full build cache and avoid accumulating old installers and runtimes.

The runtime emits one bounded readiness record on stdout:
`{type:'ready',url:'http://127.0.0.1:PORT',port:PORT,token:LOCAL_TOKEN}`.
The shell sends `{type:'shutdown'}` over its private stdin pipe when closing.
On Windows, the runtime keeps that existing stream open but clears its process
default input handle before starting workers. This prevents worker interpreter
initialization from querying the shutdown pipe while its reader is waiting.

Implementation references: [Tauri capabilities](https://v2.tauri.app/security/capabilities/),
[Tauri distribution](https://v2.tauri.app/distribute/),
[cap-fs-ext no-follow directories](https://docs.rs/cap-fs-ext/latest/cap_fs_ext/trait.DirExt.html).
