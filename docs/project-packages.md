# Project Python packages

Install packages directly from the Mac desktop workbench's Python editor or
command field:

```python
import openecon as oe

oe.install("humanize>=4,<5", "polars==1.44.2", installer="uv")
import humanize
import polars as pl
```

Installation finishes before the following line runs. Existing session variables
and imports remain available. Requesting an already installed matching version
does not download or reinstall it; a dependency explicitly requested this way
becomes a pinned project requirement. Once an overlay has imported libraries,
its pinned dependency set cannot change in that session. Restart Python to
change those versions; the running session is not replaced silently.

The workspace accepts these commands in the editor and command field:

```python
%uv pip install "humanize>=4,<5" "tabulate[widechars]"
%uv pip install -r requirements.txt
%uv add "polars>=1.44,<2"
%pip install "humanize>=4,<5"
```

The corresponding `!uv pip install`, `!uv add` and `!pip install` forms are also
supported. They invoke OpenEconometrics's project installer directly, never a shell.
`%uv add` is a workspace convenience for installing libraries; it does not edit
a `pyproject.toml` or implement uv's separate project manager. `--upgrade`/`-U`
requests a new compatible resolution; changing an imported version still
requires an explicit Python restart.

These commands are workspace syntax, not ordinary Python. Use `oe.install(...)` in exported `.py`
scripts, and execute those scripts inside the desktop workbench's active project.
Installation requires that desktop project context; it is unavailable in cloud
execution or an unrelated Python interpreter.

**Kütüphaneler** in the code toolbar remains available to inspect, remove and
manage packages. Its single command field also accepts `pip install`,
`uv pip install` and `uv add`. The terminal below the results accepts those bare
commands alongside Python expressions; `%`/`!` prefixes remain supported.
Commands run through the active project's persistent Python console, with
installation output and errors in the terminal history. Up/Down restores earlier
commands without executing them. The terminal transcript is bounded to eight
runs and 8,000 characters; complete records remain in execution history.
Installed distribution and dependency versions are recorded in
the local project manifest. Installs apply only to the active project.

The pinned native uv executable and pip ship inside OpenEconometrics. Users need neither
system Python nor a separately installed package manager. `installer="uv"` uses
uv's actual resolver and offline installer. Omitting it retains pip compatibility.
The bundled CPU engine and its dependencies remain fixed. Additional packages
live in `projects/<id>/.packages/generation-<id>/site-packages`; the local worker
adds only its own project's verified active generation to its import path.

Names, exact versions, PEP 440 ranges, extras and PEP 508 platform conditions are
accepted. Requirements files are read when the command executes and support
comments and relative `-r` includes. A preceding Python statement can create the
file. Files are bounded to 256 KiB combined, 20 includes and 100 requirements;
cycles are rejected. Repeated declarations of the same package, constraint
files and custom installer flags are outside this grammar.

Each operation resolves against the bundled dependency versions, downloads
compatible wheels from PyPI, verifies SHA-256, and prepares a fresh generation.
Previously pinned project roots retain their versions when another package is
added, including roots originally requested with a range. Original extras and
conditions remain in the project manifest. A bounded 256 MiB local wheel cache
avoids repeated downloads; its SHA-256 and archive checks run again on reuse.
Only a successful installation atomically changes the active pointer. Failure
or cancellation preserves the previous environment. Package removal and shared
environment restore reset the Python session; command installs preserve it. Saved code, data and
execution history remain intact. Other package changes cannot overlap a running
command. Project switching closes that project's installer and worker.

**Takımla paylaş** writes the inert manifest to the project's versioned cloud
workspace document. Owners and editors can write; viewers can read. Optimistic
versions reject concurrent overwrites. **Ortamı eşitle** explicitly restores the
shared versions on the current computer. Opening a project never installs or
executes packages. Local packages continue working offline; publishing or
restoring a shared manifest requires a connection. An incompatible Python minor
version or bundled numerical dependencies are rejected instead of silently
changing the lock. Updating only the OpenEconometrics application or included uv keeps
existing project libraries when their dependency metadata remains compatible.
The cloud validates declarations as inert metadata without applying its own
operating system's conditions. Restoring packages on a different platform still
requires that the saved conditions and wheels support that computer.

The default library installer accepts PyPI wheels. Explicit `source_options`
add custom HTTPS indexes, hashed local/HTTPS wheels, commit-pinned Git builds
and local snapshot builds. See the source contract below. `uv run`, `uv sync`,
Python downloads, operating-system packages, editable installs, executable
`.pth` hooks and command-line wrappers are outside this API. Packages must provide
or explicitly build compatible wheels for the bundled Python and operating system.
The overlay supports
100 distributions, 256 MiB of downloaded wheels and 512 MiB unpacked files;
installation requires 2 GiB of free disk space. Third-party libraries execute
with the same local user permissions as explicitly run Python code.

## Explicit sources

```python
oe.install("my-library==1.2", installer="uv", source_options={
    "index_url": "https://packages.example.org/simple",
    "wheel_hosts": ["downloads.example.org"],
})
oe.install("my-library", source_options={"artifacts": [{
    "name": "my-library", "wheel": "/absolute/path/my_library-1.2-py3-none-any.whl",
    "sha256": "<64 hex characters>",
}]})
oe.install("my-library", source_options={"artifacts": [{
    "name": "my-library", "git": "https://example.org/my-library.git",
    "commit": "<full 40-character commit>", "build": True,
}]})
```

`%pip install --index-url URL ...`, `%uv pip install --index-url URL ...`,
and `uv add` command rewriting also pass the explicit index option. Other
arbitrary installer flags are rejected. Artifact roots are named explicitly;
unstructured direct dependency URLs are still rejected by the requirements
parser. Use the API's hashed artifact declaration instead.

An artifact may alternatively use `directory="/absolute/source"` with
`build=True`. Source directories are copied into an owned bounded snapshot;
Git checks out exactly the supplied commit and records a source digest.
The [pip build-system contract](https://pip.pypa.io/en/stable/reference/build-system/)
is used without automatic build-tool installation or build isolation. Required
backends and versions must already be bundled; missing tools fail explicitly.
The source backend executes with the user's permissions only after this explicit
command. It is not a security sandbox. Build products undergo wheel identity,
Python ABI, SHA-256, dependency/core-constraint and overlay checks before promotion.
CLI wrappers and `.pth` hooks continue to be removed.

Sources accept credential-free HTTPS only: embedded user/password, query tokens
and fragments are rejected before an operation starts. Authenticated artifacts
can be downloaded independently and supplied as hashed local wheels. Custom-index
wheel hosts require an explicit host allowlist; resolver/build logs are suppressed
for these operations, leaving generic phase/error messages. Source paths, commits,
hashes, wheel tags, build tools, Python ABI and resolved dependency hashes stay
in local `source-provenance.json`; portable/cloud manifests remain inert names,
versions and ordinary requirement conditions. Opening or previewing them executes
no source/build/install action. Restoring custom sources on another computer
requires that user to explicitly supply the same sources; a version-only restore
does not implicitly fetch VCS or run builds.

All subprocess scratch is owned by the operation. Failure or cancellation removes
staging and keeps the prior active generation. Git builds need a Git executable;
missing build tools or incompatible wheels can be handled by building a wheel
externally, then installing the pinned artifact.

Install the `desktop` extra in a development/build environment before freezing:

```sh
uv sync --extra desktop
```

`scripts/verify_project_packages.py --runtime <bundled executable> --output
<report.json>` verifies actual pure-Python and compiled package installation,
project isolation, failure rollback, restart persistence, exact manifest restore,
removal, and unchanged native OLS/LaTeX behavior using temporary projects only.

The 0.3.3 Apple Silicon release was checked with real `humanize` 4.16.0 and
compiled `polars` 1.44.2 wheels inside the shipped runtime. Team manifest
persistence, concurrent-write rejection and anonymous-access rejection were
checked against the deployed control service; role permissions passed local
tests. See
[`evidence/desktop-0.3.3.json`](evidence/desktop-0.3.3.json) for artifact identity,
source provenance, verification coverage and the separate cloud revision.

The 0.3.5 Apple Silicon release verifies `oe.install(...)` and `%pip install`
against actual pure-Python and compiled wheels inside the read-only mounted
installer, including preserved variables/imports, no-op installs, failed upgrade
rollback, isolation and restart persistence. Its 2,290 Python tests include
Stop, timeout and worker-crash cancellation. See
[`evidence/desktop-0.3.5.json`](evidence/desktop-0.3.5.json).

The 0.3.7 Apple Silicon release includes uv 0.9.26. Real `humanize`, native
`polars` and `tabulate[widechars]` installations passed inside the mounted DMG,
including requirements files, cached wheels, retained pins, exact uv manifest
restore and pip compatibility. The deployed collaboration service preserves
schema 2 declarations, with authenticated owner/viewer readback, write denial,
optimistic conflict and legacy manifest compatibility. See
[`evidence/desktop-0.3.7.json`](evidence/desktop-0.3.7.json).


## Portable environment documents

In the desktop Packages dialog, **Export environment** downloads an inert JSON
file containing declarations (including extras and markers), exact root and
transitive versions, Python minor version, fixed core versions, platform,
architecture and Python ABI. **Import environment** validates and previews that
file without installing packages or changing the current environment. Only
**Restore environment** starts the existing atomic wheel-only installer and
replaces the project's additional packages. Closing the dialog or discarding
an import leaves its environment unchanged.

The document is bounded to 64 KiB. Its SHA-256 detects accidental edits; it is
an integrity check, not a signature or proof of the sender's identity. Different
platform/ABI, Python or core versions are refused. Validation and failed
installation preserve the previous active generation. Exported documents contain
no executable code, package binaries, credentials or private index settings.
Restoring still needs compatible wheels from the existing cache or package index;
opening an exported file works without initiating a package download.

Local API: `GET /environment/export`, `POST /environment/import/preview` and
`POST /environment/import/restore` (both POSTs accept `{document: ...}`). These
routes retain the project session and console/environment concurrency checks.
