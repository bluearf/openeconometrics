# Local computation and cloud collaboration

OpenEconometrics Desktop uses one Python/PyTorch statistics core on the user's computer.
The existing white/navy Python editor, D3 figures and publication LaTeX interface
run in the operating system's webview. Rust/Tauri manages the application window,
local process lifecycle and bounded cloud transfers. There is no engine selector.

| Component | Responsibility |
|---|---|
| Rust / Tauri 2 | Native window, file chooser, fixed-origin HTTPS, process shutdown |
| TypeScript / React | Editor, projects, team controls, D3 and LaTeX views |
| Bundled Python / PyTorch | Persistent local console and native estimators |
| Existing Cloud Run control service | Authentication, permissions, draft versions, result archives |
| Firestore / Cloud Storage | Team membership, code, source files and shared output |

Tauri uses the system webview; its shell does not ship Chromium. The installer
still contains the scientific Python runtime, including PyTorch, pandas and
Arrow. SciPy is not bundled. Shell size and complete application size are
different measurements.
See [the native architecture](https://v2.tauri.app/concept/architecture/) and
[build instructions](../desktop/README.md).

## Opening and running a project

1. The native window opens a local startup page while the bundled server starts.
   Startup and project inspection do not import PyTorch or execute user code.
2. The desktop opens Local projects without contacting authentication or team services.
   Choosing Team projects restores Firebase from the webview's persistent storage.
   A randomly selected local port is remembered across launches to preserve its
   origin. A native lock prevents a second instance; an occupied port fails closed.
3. The cloud checks membership and supplies the versioned draft and file manifest.
   Rust checks each file's size and SHA-256 before caching it in the local project.
   Previously cached bytes are hashed before reuse. Python retains the original
   filenames, so `oe.read("wages.csv")` works on each team member's computer.
4. Run starts one local Python worker. Variables survive subsequent commands in
   that project until reset, interruption, project switching or application exit.
   Run never leaves this computer: the cloud service has no compute.
5. Code is saved locally before a version-checked cloud write. Structured results
   enter a bounded durable outbox, then the cloud validates and archives them for
   teammates. The server does not rerun or certify client-produced calculations.

The native download batch uses one authenticated metadata listing, a maximum of
20 files and 64 MiB total. Transfer buffers are bounded and files import
sequentially. The application bundles Python once; it does not install packages
or extract a scientific runtime each launch.

## Login and permissions

Desktop Google sign-in opens the system browser. A five-minute request binds a
random verifier to a SHA-256 challenge. The signed-in user compares the displayed
code and explicitly approves the request. A one-use exchange issues a short-lived
[Firebase custom token](https://firebase.google.com/docs/auth/admin/create-custom-tokens)
carrying the desktop claim. Firebase then supplies the normal session. Grants
expire and use the `oe_desktop_logins.expires_at` Firestore TTL policy.

Cloud tokens travel from Firebase to Rust; the local Python API rejects cloud
Authorization headers, and workers receive a reduced environment. Cloud writes
still enforce the current project role. Viewer and removed-member restrictions
are checked by the server, including when archiving results.

The native bridge permits only a fixed OpenEconometrics origin and explicit application
routes. Redirects, arbitrary URLs, cloud execution routes, traversal and symlinks
are rejected. The webview has no general shell or filesystem command permission.

**User Python runs with the local OS account's permissions.** Process separation
and environment filtering are not an OS sandbox: trusted Python code can access
local files, network resources and other user-owned storage. Shared scripts must
be reviewed before running. Tokens are not passed to Python by the application;
this does not claim isolation from malicious code with full OS user privileges.

## Offline behavior and boundaries

Fresh local projects can be created and run without an account or network. Their
code, imported data, package manifest and execution history survive restart. Python
variables in memory are recreated by running code after restart. The local catalog
uses generated project IDs and cannot register cached team projects.

Sharing requires choosing Team projects, creating or opening an authorized team
project and explicitly importing the selected files. Local results are never
automatically enqueued for team publication.

An already opened, cached team project can continue locally if the network fails.
New team projects, membership changes and uncached downloads require a connection.
An explicit cloud 401/403/404 does not become an offline permission fallback.
Downloaded data stays on the user's computer; removing membership cannot erase
copies the user already received.

Offline edits remain on disk. On reconnect, optimistic versions prevent an older
local draft from overwriting a teammate's change. A conflict preserves the local
code. **Compare and merge versions** shows the full base, local and cloud sources
and a separate editable merged draft. Saving uses the displayed cloud version as
a compare-and-swap condition. Another cloud write keeps the originals and merged
draft visible for a fresh comparison. Another local write prevents the cloud
write. Cancelling keeps the editor untouched; unresolved conflicts block file and
project switching. All observed versions can be downloaded together. Users select
the merged code explicitly.

The outbox keeps at most 20 results / 8 MiB, with 3 MiB per result. It retries on
project opening, connectivity restoration or a subsequent Run, without rerunning
Python. The result header and execution history distinguish waiting, failed,
shared and local-only output. **Retry sharing** resends the original result;
it does not submit a Python execution. Queue and permission errors remain visible.

The local runtime saves each run's actor and original input hashes in history
before attempting to enqueue it. Short summaries of local network artifacts are
captured at the same time, so an application update cannot change a pending
publication's body. If the queue is full, that exact record can be
queued later. A different signed-in account cannot republish it under its own
identity. Local-only data stays local. A permanently rejected item does not block
other queued results; an authorization failure stops publication until access is
verified again. Confirmed shared items are terminal, including when the app exits
between persisting that confirmation and removing the queue entry.

Cloud acknowledgements are checked against the actor-bound result identity.
Identical uploads retain their original request body and return the same archive
identity, including when an acknowledgement was lost after input files changed.
Current membership is still required. Concurrent attempts use separate immutable
objects and select one archive transactionally. A transport failure with an
unknown commit outcome can leave an unused object for later lifecycle cleanup;
the client keeps its pending result instead of declaring success.

These are source contracts; tests and deployment/native verification are distinct
checks. Generated arbitrary local files are not uploaded automatically; the
shared archive contains supported table/model/plot/LaTeX output and source hashes.
Large local network artifacts share an explicit summary, with the full graph
remaining on the source computer. The UI labels that scope separately.

Small imports use the eager reader; large CSV/Parquet inputs automatically use a
Dataset. XLSX/DTA retain separate eager limits, and cloud project uploads remain
limited to 24 MiB. Local `oe.scan()` always reads CSV/Parquet in batches without
a total row/file-size ceiling. All 80 registered model names have native Dataset
routes, including the estimation methods and option families listed in
[large local datasets](streaming.md). Groups, parameter matrices and algorithms
with intrinsically quadratic work retain explicit structural budgets. Cluster
scores and global matching state spill to disk. Selected GPU factor operations
have been verified;
whole-model GPU execution and full Stata parity remain separate contracts.

## Platform verification

See the [release verification record](desktop-verification.md) for artifact
identity, measured timings, test scope and outstanding platform checks.

Apple Silicon/macOS 15+ packaging includes a self-contained scientific runtime.
Native tests and actual bundled-worker analysis checks are required before each
release. A local unsigned package is not an Apple-notarized public distribution.
Windows x64 has a CPU-only NSIS build and a bundled-runtime smoke test in the
Windows CI workflow. A workflow definition alone does not establish a successful
Windows build or installed Windows behavior; check its artifact and test result.

Cold installation launch, repeat launch, first worker start and warm estimation
must be reported separately. A quick window or a warm solver benchmark is not
evidence of a quick first analysis on a newly installed machine.


## Cancelling Google approval

While native sign-in waits for browser approval, **Cancel Google sign-in** returns
immediately to the form. **Forgot password**, **Create account** and **Back to
sign-in** also cancel that approval before switching methods. Each retry uses a
fresh verifier and grant; a cancelled attempt's late response cannot commit a
credential or release the newer attempt's busy state. Leaving the page or
unmounting the form aborts polling. An already dispatched native HTTP request
may finish in the background, but its response is ignored. Once Firebase starts
committing an approved credential, the cancel action disappears until that
short identity operation finishes.
