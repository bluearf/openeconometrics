# Installed Windows cloud sharing acceptance (MARKET-87)

This controller drives the normal installed WebView2 window with actual mouse
and keyboard events. Its observer calls the original native broker and `fetch`
unchanged, records execution counts and selects only public grant/result fields
and the actual `/api/me` UID/email/owned-project role. Actual UID/email stay in
the transient observer; receipts retain only their SHA256 digests. Tokens,
passwords, account IDs and email addresses are never written to receipts or
printed. A phase receipt is not complete issue acceptance.

The active Windows installer/workflow/local UI controller is unchanged. The new
controller is prepared for later integration; no hosted cloud phase has yet run.

## Provisioning before the runner

1. Deploy the merged control image through the existing native result-schema
   gate. Read back the running image revision and its `source_commit` from the
   public `/api/auth/config`, together with the account-link and transfer
   feature flags.
2. Use the existing administrative preparation route to create two verified
   disposable password identities, enable the owner, create one empty synthetic
   team project and give the distinct viewer a viewer membership. Do not reuse
   `verify_team_live.py verify` here: it creates its own project and changes
   memberships. It never starts a computation; the cloud has no compute.
3. Transfer only the required QA fixture to the private runner file. It must
   contain `users.owner` and `users.viewer` (`uid`, `email`, `password`), the exact
   project ID in `projects`, and `url` equal to the native production origin.
   Emails follow `openecon-qa-{role}-{12 hex characters}@example.com`. The project
   name starts with `openecon-qa-`. Restrict the fixture's Windows ACL to the QA
   runner user; exclude it from uploaded artifacts and never put it in CLI args.
4. Install the accepted Windows package normally. Start the normal application
   using its isolated QA data/WebView2 profile and a loopback-only CDP port.
   Owner and viewer use distinct isolated data/WebView2 profiles. Node 24 is
   sufficient; no browser automation package or cloud administrator token is
   required on the runner.
   The controller verifies the existing native `data_root` through realpath and
   two filesystem observations. Receipts retain separate SHA256 digests of its
   canonical path and stable nonzero device/inode identity, without raw paths.
   Windows path normalization and case folding apply. The actual `desktop_info`
   response does not expose a WebView2 storage directory: that binding remains
   unknown and requires separate real launcher/process evidence.
5. For browser approval, configure the disposable runner's default browser to
   its isolated CDP-enabled browser profile. The controller only attaches to the
   actual production approval URL opened by `open_desktop_login`; it does not
   manufacture a grant or authorize through an HTTP shortcut. If Windows opens
   a browser outside the owned endpoint, that phase fails until runner browser
   setup is corrected.

## Ordered phases

Every command also supplies `--port <native CDP port> --state <private fixture>
--project-id <manifest project> --project-name <exact QA name> --output <receipt>`.
The private fixture, original offline baseline, original online identity receipt
and each new receipt must use distinct
files. Their parent directories must already exist. The controller refuses
canonical parent/junction/symlink aliases, Windows case aliases, shared file
inodes and every existing output receipt before reading credentials or touching
the UI. It creates the new receipt exclusively and retains that file handle;
repeating a phase requires a fresh receipt path. `GITHUB_ACTIONS=true` and
Windows are required; only loopback CDP endpoints are accepted.

| Phase | External runner action | What the installed UI must prove |
| --- | --- | --- |
| `password-open` | Online owner process | Real password sign-in, fresh actual authenticated `/api/me` UID/email/owner role before and after the phase, actual team project opened and cached. No execution. |
| `browser-approval` | Online owner; system browser already CDP configured; add `--browser-port <separate port>` and `--online-identity <original password-open receipt>` | Same original canonical owner workspace, actual native grant, OS browser open, matching displayed code, real owner password sign-in and real Approve sign-in click. Native exchange returns to the same actor, verified by fresh `/api/me` before and after the phase. This proves browser approval with a password identity; **Google provider acceptance is not tested**. |
| `offline-run` | Keep owner process open; add `--online-identity <passed password-open or browser-approval receipt>`. Apply an actual outbound deny rule to the installed native executable, retaining loopback. Validate the rule independently. | Exact owner/project/native-profile binding bridges to the immutable online receipt's SHA256; identity is explicitly offline cached. Native broker GET fails with the real network sentinel; UI runs one 480-observation HC3 OLS, renders model/LaTeX, persists stdout/events and one pending result. Visible waiting count and connectivity error. The Python counter must equal one. |
| `offline-reopen` | Keep network deny; terminate and cold-start the actual native process using the same owner profile. Add `--baseline <offline-run receipt>`. | Same local run ID and hash, model rendered, pending outbox retained; actual manual Retry sharing click preserves the result. Zero new Python execution requests. |
| `retry-online` | Remove the deny rule and independently read it back; keep/reopen the same owner profile. Add the original baseline. | Fresh actual `/api/me` owner UID/role before and after the phase, real native broker reconnect; actual UI Retry sharing if not already automatically delivered; shared status and empty outbox. Original ID/model/LaTeX/stdout/events hash unchanged. No recompute. |
| `viewer-readback` | Start the distinct viewer native profile online. Add the original baseline. | Fresh actual `/api/me` viewer UID/role before and after the phase, disabled terminal input, actual Shared history search/open of the deterministic archive ID. Viewer renders the model/stdout and archive payload hash equals the original. No local or cloud execution. |

The online identity checkpoint clicks the normal Projects back button when
needed, then Refresh projects. The normal UI obtains its own current Firebase
token and calls the existing native broker GET `/api/me`; the controller never
reads or exports that token. A successful actual response must return the exact
manifest UID and email plus exactly one membership for the owned project with
the expected role. A matching visible email or cached profile does not pass.
The observer preserves the original response and rejection, retains request
order, and refuses a stale response or a latest failed/denied request. The same
normal refresh and exact identity check runs after every online phase, retaining
both ordered request sequence numbers. The refresh button must be enabled before
the observation checkpoint starts, so an earlier initial refresh cannot count. This
endpoint performs its usual server profile metadata refresh as part of normal
UI behaviour; the controller adds no endpoint or profile mutation.

Offline phases never claim a live authenticated UID check. The first offline
receipt records the exact original online receipt's file SHA256 and requires
the same actor, role, project and both canonical workspace identity digests. Cold offline reopen
inherits that binding through the original offline baseline. A successful live
profile response during a blocked phase is refused. Original online/offline
receipt files are protected inputs and are never overwritten. These future
Windows phases must use a fresh owned fixture: existing historical MARKET-87
manifests, identities, queue records and receipts cannot be reclassified or
recreated to substitute for their original acceptance.

Before reading the private fixture or performing UI input, owner browser,
offline, warm/cold reopen and retry phases must match both original canonical
workspace digests in the protected online/offline receipt. Viewer readback must
differ from the original owner on both digests; either a shared canonical path
or a shared filesystem identity refuses admission. Parent junctions, symlinks,
case aliases and unchanged path text after directory replacement cannot prove
separation. The old raw `profile_sha256` is diagnostic only. At phase end the
controller rereads native metadata and the real directory, requiring both root
identities to remain unchanged. Protected receipt files are never rewritten;
older receipts lacking this proof cannot be promoted as current acceptance.

The controller checks native broker reachability, rather than setting only
WebView2's simulated offline flag (which would leave native cloud requests
online). The runner's process/network/start-stop receipts are required in
addition to UI receipts: the controller cannot itself prove a cold process
restart or the firewall rule's system configuration. The phase observer starts
when attached; it does not count requests made before that point. Cloud project
generation/idle/archive-count preflight and final readback must prove no cloud
compute throughout the full route.

The raw local result digest covers exact code, status, stdout, stderr,
model/publication outputs and ordered events; cold reopen/retry require it
unchanged. The archive comparison digest normalizes only the order of the
model's `display_omitted` labels: the local worker and cloud validator currently
list the same omitted fields in opposite order. Scientific values, publication
bytes and event order remain exact. Archive identity/creation time/actor labels
may differ under the documented cloud archive protocol. The second member reads
through the real history UI; it never injects a result or fetches an archive
with an owner token.

Every persisted local and viewer record must also satisfy the exact fixture
protocol: the submitted command, the two complete stdout lines, one three-term
480-observation HC3 OLS and its matching publication LaTeX, and the original
three ordered events. The visible check requires the exact stdout, one correct
model section with rendered math or its coefficient-table fallback, and the
separate visible publication MathML. A publication source fallback alone does
not pass this rendered-publication checkpoint.

## Source and provider binding still required

The controller verifies the real `desktop_info` local origin against the
attached page and its fixed Cloud origin against production; it records the
observed desktop version and only a digest of the private data root. The
response exposes no source commit, executable hash, SDK
version or deployed image revision. The public Cloud auth config exposes
feature flags and Firebase public configuration, while `/healthz` exposes only
status; neither certifies the running source revision. The prepared online
actor gates now require a fresh successful authenticated `/api/me` response,
whose production server verifies the Firebase token and reads membership for
the verified UID. That response exposes no linked-provider inventory. No live
online identity gate has yet been exercised by this preparation.

The normal Windows workflow supplies `build-inputs.json`,
`native-build-reference.json`, `windows-installed-acceptance.json` and
`installed-frozen-runtime.json`. Bind a future Cloud route to these real
receipts and their hashes. Normal installation acceptance subsequently
uninstalls its own application, so its receipt cannot identify a later active
process by itself: retain a separate launch receipt proving that process's
executable/runtime/static hashes match the accepted installer chain, alongside
the actual broker origin/version and isolated profile readback. Cloud source
binding needs separate real deployment/image/revision receipts. Installed and
deployed source binding and linked-provider binding remain explicitly unknown;
phase receipts alone do not certify them. Current authenticated UID binding is
live only in an actually passed online phase, while an offline phase records
only its cached binding to the original online identity receipt. Canonical
workspace proof does not identify WebView2 storage or an installed source hash;
those require independent launcher/process and installer evidence.

## Owned cleanup

Keep each new grant in `desktop_login_ids` in the private manifest; the native
browser phase records the exact real grant before approval. When the later
MARKET-91 real account-link route is run, record each exact quota document key
as `account_link_limit_ids`, e.g. `account-link-{owned UID}-{UTC YYYYMMDD}`. These
are document IDs without `oe_limits/`; no glob, shared daily login quota or
compute quota is permitted.

Automatic `cleanup` is disabled because no enforced admission or quiescence
contract spans Firestore, Cloud Storage and Firebase Auth. It refuses before
credentials or private-manifest access; quiet flags and claimed receipts cannot
enable deletion. `inspect-cleanup` is an explicitly requested read-only
ownership/quota snapshot: it always reports `inspection_only=true`,
`safe_to_delete=false`, `quiescence_enforced=false` and preserves every resource
and the exact private manifest. Its result cannot authorize deletion. Existing
historical QA identities, queues, manifests and receipts remain retained until
a separately implemented, reviewed cleanup contract exists.

## Separate remaining acceptance

- MARKET-87 quota refusal/requeue and authoritative 401/403 behavior retain the
  existing Python/web preservation tests. This controller adds the live installed
  offline/restart/reconnect/second-member route; it does not simulate backend
  refusal codes or claim a native lost-ack transport test.
- MARKET-91 requires real Google popup/link/reauthentication and preservation of
  the Firebase UID in both method directions. Firebase Admin password accounts
  or custom tokens cannot establish that Google provider acceptance.
- MARKET-109 still needs the actual native picker and explicit download approval
  for a real file larger than 24 MiB, cancellation/staging cleanup and verified
  local readback. Package/environment sharing and restore are also separate
  real installed gates. This controller deliberately does not claim them.

Source checks: `node --check desktop/scripts/verify_windows_cloud_ui.mjs`,
`node --test desktop/scripts/verify_windows_cloud_ui.test.mjs`, and the exact
cleanup preservation suite `tests/test_verify_team_cleanup.py`. Their success
does not stand in for running the hosted Windows phases above.
The exact controller Python fixture is additionally exercised through a real
local worker, cold history readback and the actual cloud archive validator in
`tests/test_verify_windows_cloud_ui_fixture.py`.
