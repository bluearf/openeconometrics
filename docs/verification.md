# Alpha verification record

## Advanced publication matrix - 7 October 2026

[MARKET-110 evidence](evidence/market-110-publication/README.md) validates 22
real fits across all 18 registered result families, saved JSON readback,
KaTeX preview/copy/download and 22 compiled/visually reviewed PDF pages.
It records version scope, inference disclosures, numerical preservation,
wide/long grouping, Unicode/special characters and full export/preview limits.
This is publication coverage, without implying all 80 estimators' scientific
parity or installed/frozen/cloud/public release deployment.

## Publication table defaults — 2 October 2026

Models now default to paper-style booktabs tables: numbered specification
columns, coefficients over parenthesized standard errors, constants last,
integer observations and estimator-appropriate fit statistics. The new
`oe.regression_table` compares named or listed fitted models. Original p-values
determine strict 10/5/1 percent stars; real covariance, cluster, inference,
confidence, sample, exclusions and warnings are retained in notes. Diagnostic
coefficient tables remain explicitly available. Existing saved automatic outputs
upgrade their presentation on read without rewriting the stored execution;
explicit user TeX remains unchanged.

The final Python suite passed **1,418 tests, with 3 optional live tests skipped**;
all **64 web tests**, the final TypeScript/Vite build and Ruff passed. The final
wide-table repair separately passed 78 focused formatter/output/startup/live-QA
checks. Native KaTeX parser checks cover tiny scientific coefficients with stars,
Turkish text and escaped cells. One test-library Starlette deprecation warning
remains; it is unrelated to table formatting.

The exact complete-document helper compiled successfully with the native editor.
A six-model table, actual 18/12-cluster models, integer/zero/binary/Unicode data,
12-column short table and 90-row longtable passed PDF rendering/readback. Standard
float barriers preserve selected output order. A 75-row, 8-column table with
long text previously clipped its last columns despite successful compilation;
the repaired paragraph-column defaults export all data over six pages, without
overfull warnings or clipping. The final padding guard was recompiled from the
exact source; page text and bounds remained identical to the visually approved
PDF. Long regression tables keep each estimate with its standard error at page
breaks. Style conventions were compared with two official NBER working-paper
tables, without claiming an official NBER template.

The actual local browser rendered native single/comparison/summary tables,
switched to source, selected outputs and displayed copy feedback. The downloaded
complete document matched visible source exactly and compiled with the native
editor. Safe evidence is `publication-local-qa.json`, `publication-browser.tex`,
`publication-rendered-ui.png`, and `latex-publication/qa.json` plus native PDF/PNG
artifacts under `artifacts/verification`. These are synthetic data only.

Cloud Build **`e184c361-fd63-4137-9a41-82b38b7d06cb`** succeeded at
`2026-10-02T13:28:39.945241Z`. Production **`openecon-00009-28v`** serves
100% of traffic using control digest
`sha256:7fc9e78250dc238712e72920b51be752772ba34cf7a38aa14e7c9ffe9a8a6106`.
Private compute **`openecon-sandbox-00003-qf7`** is pinned to 100% of its traffic
using digest
`sha256:4304cc205d36dc8a71d575a86c618a4ab86378523d72432ddfbcc57a98a98a70`.
The preceding v8 broker remains unchanged for rollback; minimum instances are
zero. These changes do not modify the launcher or broker authentication code.

The complete five-user application suite passed in preview and production,
including the new multi-model TeX artifact, exact estimates/SE/star thresholds,
native dataframe exports and persisted readback by another member. Production
analysis plus artifact downloads took **13.0 seconds**. The fresh preview first
stopped at a platform-generated HTML 401 on one artifact GET; the accepted run
was resumed read-only and all exports and remaining membership checks passed,
without repeating the analysis. The QA helper now retries only this narrow
read-only platform error; application JSON denials and all mutations are never
retried. Its focused 17 tests passed. Preview browser sign-in similarly succeeded
on the explicit second attempt. Real rendered/source views were inspected and
the exact browser-generated saved comparison document compiled successfully.
Safe reports are `publication-team-preview-live.json`,
`publication-team-production-live.json` and `publication-production-deployment.json`.

Independent Cloud Run readback confirmed the v9 image, created/ready revision,
observed generation, 100% traffic and exact private service/image binding. The
actual source verifier and policy checks passed for the new broker and the
unchanged v8 rollback broker, with only the fixed control service account allowed
to invoke private compute. Public HTML, the exact new App bundle and team auth
config returned 200. Runtime endpoint response samples are ordinary HTTP checks,
not demonstrated browser/cold-start measurements.

Cleanup removed both disposable projects, all five identities, four invitations
and their scoped storage prefixes. Absence was read back in Firestore, Firebase
Auth, Cloud Storage and the preview Cloud Run API. The credential state and
local test listener are absent. `publication-cleanup-final.json` records complete
cleanup; production and its preserved v8 rollback broker remain available.

## LaTeX results and dataframe API — 2 October 2026

Native dataframe, model and chart exports now carry LaTeX source alongside the
existing structured console output. The results pane provides **Sonuç / LaTeX**,
selection of all or one output, copying and complete `.tex` downloads. Table and
coefficient math renders with local, lazily loaded KaTeX; D3 charts export as
TikZ/PGFPlots. Older stored results are enriched on read without rewriting them
or loading pandas/Torch in the control process.

The full Python suite passed **1,340 tests, with 3 optional live tests skipped**.
After deployment-helper and live-QA additions, 129 focused tests passed. Final
console edge repairs passed 119 console/server/startup tests: indexed frames
with no columns retain their rows, and an oversized optional math preview cannot
discard valid explicit LaTeX source. All **60 web tests**, TypeScript/Vite build
and repository Ruff checks passed. These changes affect presentation rather
than estimator algorithms.

The actual local browser exercised native `df.head().to_latex()`, rendered model
results, both view modes, selection and copying feedback. A downloaded combined
`.tex` file matched the visible source exactly. The native editor compiler
successfully compiled a 14-chart document and the combined dataframe/model/chart
document after removing a font-name lookup unsupported by that compiler; the
export now uses `fontspec`'s portable default. A browser-generated document with
Turkish labels, escaped `A&B` and a nonzero `1.2e-12` value also compiled
successfully. Browser console error/warning logs were empty. Safe evidence is in
`artifacts/verification/latex-*.tex`, `latex-charts-qa.json`,
`latex-rendered-ui.png` and `latex-source-ui.png`.

Cloud Build **`fb954e13-92af-498b-9964-122edff2b85d`** succeeded at
`2026-10-02T12:16:57Z`, including nonroot/CPU Torch, lazy control startup,
native OLS, dataframe/model/chart LaTeX and clean guest-runtime checks. Production
revision **`openecon-00008-k98`** serves 100% of traffic and has its generation
observed. Its control image is
`sha256:821adad15684c387a1a3a0659a340d0f062b3e66bb7ae2b33c8b8d002200ad6a`.
It uses the separately deployed private service **`openecon-sandbox-latex`**,
pinned to **`openecon-sandbox-latex-00001-t7l`** and
`sha256:fd5163cc7cfe1c1d31a5199b5aa4c77a2d25cec66da4bddb1e0a2937ea9683be`.
Both services retain zero minimum instances; source, readiness, traffic and
image bindings were read back. The previous private broker remains available
for accepted work and explicit rollback, avoiding a digest mismatch during
the control transition.

The complete five-user suite passed in preview and production. In addition to
existing invitation, access, draft, upload and role checks, it verified native
DataFrame compatibility, exact dataframe/model/PGFPlots source values, explicit
LaTeX display records and a directly written `.tex` artifact, with persisted
readback and downloads by another member. The analysis plus durable-file
checks took **27.8 seconds in preview** and **12.4 seconds in production**;
these are complete integration timings, not browser opening or demonstrated
cold/warm container measurements. The unchanged first preview attempt stopped
at a platform-generated HTTP 401 on `/api/me`; the subsequent complete suite
passed without a source change. The preview browser also displayed the saved
table/model results and switched to complete LaTeX source with empty console
error/warning logs. Production HTML and the exact new client asset returned 200.

Safe records are `latex-team-preview-live.json`,
`latex-team-production-live.json`, `latex-production-deployment.json`, and
`latex-preview-rendered.png` / `latex-preview-source.png` under
`artifacts/verification`. These presentation checks do not repeat the initial
rollout's adversarial sandbox suite below; launcher, guest privilege reduction
and broker authentication code were unchanged.

Cleanup removed the two owned disposable projects, five QA identities and
their private credential state. The owned public preview service was deleted
with absence read back. `artifacts/verification/latex-cleanup-final.json`
records complete cleanup; production and the previous rollback broker remain.

## Analysis startup replacement — production sandbox rollout

On 2 October 2026, production switched from per-analysis Jobs to the private
sandbox compute service. Revision **`openecon-00007-p7s`** is Ready, matches the
created revision, has its generation observed and serves 100% of latest traffic.
The fixed-code managed Cloud Run Sandbox runtime proof, **all 13 actual broker
integration cases**, and complete five-user preview and production suites passed.
The core and broker records are
`artifacts/verification/sandbox-runtime-managed-dns-live.json` and
`artifacts/verification/sandbox-broker-final-live.json`, both with no pending
cleanup scopes. Cloud Run Sandboxes remains a Preview feature.

Cloud Build `b757743b-6758-499d-837e-28b983c3a729` succeeded and published the
reviewed trusted supervisor. The subsequent control build
`f3070a0a-87c6-4f05-abb7-bf0718f40261` succeeded at `2026-10-02T00:01:54Z`,
recorded in `artifacts/verification/sandbox-control-final-cloud-build.json`.
Production, the private service and the verified preview use separate pinned images:

- Nonroot web/control: `sha256:56f1cda7f05e8f97fc2d18d1041c4de5728b8fcb5e87daedfe0934e10c076d57`.
- Trusted sandbox supervisor: `sha256:96123803caec5a46cc71f8b39277d5128b39919e614cee5d9ff186056ced73fe`.

Production revision `openecon-00007-p7s` uses the final control digest with
`OPENECON_RUNNER=sandbox`; private revision `openecon-sandbox-00002-7dc` retains
the reviewed supervisor digest. Public preview revision
`openecon-teams-preview-00001-vwv` used the same control and compute pair.
The extra compute image produced by the
final control build was not deployed; the successful runtime and broker proof
remain attached to the previously reviewed supervisor.

The supervisor needs host UID0 to create the managed network namespace. The
guest also has a platform-fixed UID0, without SETUID/SETGID. Before accepting
input, its bootstrap clears effective, permitted, inheritable and ambient
capabilities and sets `NoNewPrivs`. UID10001 is the web/control identity; it is
not the managed guest's identity. Earlier nonroot launcher probes failed with
network-namespace/veth `EPERM`, so a nonroot launcher is not claimed here.

The live proof established clean per-run writable overlays, namespace-local
standard devices, parent file/process/environment denial, metadata denial by
both IP and DNS with GCS egress enabled, unchanged host/lower runtime files,
no files carried between sandboxes, and deletion of hostile descendants. It also
verified a positive signed GCS beacon and, after the managed watchdog killed
the parent, absence of the child beacon while a replacement instance remained
active. Fixed system logs established parent instance termination/replacement.
Disconnect cleanup and the watchdog-after-disconnect path passed.

Three fresh diagnostic sandboxes produced the 480-observation OLS, table,
coefficient plot and scatter plot successfully:

| Diagnostic run | Entire sandbox execution | Console analysis portion |
| --- | ---: | ---: |
| First | 15.75 s | 14.51 s |
| Repeat 1 | 5.92 s | 4.76 s |
| Repeat 2 | 6.13 s | 4.95 s |

The initial diagnostic `/probe` HTTP request took **39.71 seconds**, including
its isolation sandbox and all three analysis sandboxes executed in sequence.
It is not a standalone container startup measurement or one user's analysis
latency. These fixed diagnostic requests bypass the production control/store,
capability manifest, actual broker result validation and team authentication.
They do not establish production performance.

The runtime record deliberately marks complete network isolation and application
authentication as unverified: the guest's gateway can reach the parent listener.
The actual broker therefore verifies the full signed Google ID token itself,
in addition to Cloud Run's IAM edge gate. Authorized empty-body access returned
400, establishing that the fixed control identity reached body validation without
launching a sandbox. Submitted guest attempts with missing, forged, or only
platform headers returned 401. A valid analysis immediately after the gateway
probe passed, establishing the broker's application authentication boundary
separately from the diagnostic record's intentionally unverified field.

The final private broker integration measured **13.42 seconds** for its initial
full analysis request and **12.43 / 11.91 / 12.01 seconds** for repeats. These
measurements include signed GCS input preparation, broker result handling and
cleanup. They are private compute-path measurements, before a public workbench
cutover. The record does not confirm reuse of one warm instance, so these are
repeat requests rather than demonstrated warm-instance timings.

All 13 cases passed: the initial analysis, guest gateway authentication, analysis
after that probe, three repeated analyses, cancellation, Python timeout, bounded
stdout, detached fork, positive child beacon, negative beacon after sandbox
deletion, and a final analysis after the adversarial cases. The detached-fork
case recognized the exact reproducible Python deprecation warning alongside the
expected marker; no arbitrary stderr was ignored. Anonymous private invocation
was denied, the external orphan beacon proof passed, and every staging scope
was removed. These results establish the actual compute transport separately
from the authenticated preview workbench gate below.

The complete five-user team preview suite passed against
`openecon-teams-preview-00001-vwv`, recorded without credentials in
`artifacts/verification/sandbox-team-preview-live.json`. It verified anonymous
and unverified-account denial, invitation gating, one-time email-bound
invitation acceptance, cross-project isolation, persisted shared drafts,
stale-write and viewer-write rejection, upload/download persistence, and viewer
execution denial. Native OLS, D3 plots, uploaded input and durable output passed
in **11.5 seconds**. Another member read the saved execution and chart; role
changes and member removal took effect on the next request.

The first preview attempt had already completed the same analysis in
**12.9 seconds** before a subsequent platform GET returned 401. An unchanged
read-only retry of that same GET returned 200, then the entire five-user suite
was rerun successfully without a source change. The transient platform response
is recorded rather than treated as a complete first-attempt pass. These are
preview measurements, separate from the production evidence below.

The complete five-user suite then passed against the
[production origin](https://openecon-291739190496.us-central1.run.app/), with exit
code 0. Native OLS, D3 plots, uploaded input and durable output passed in
**11.9 seconds**. Viewer readback and immediate role changes/member removal also
passed. Safe records are `artifacts/verification/sandbox-team-production-live.json`
and `artifacts/verification/sandbox-production-deployment.json`; no credentials
are included. The public control and private compute service keep minimum
instances at zero.

Two ordinary anonymous HTML requests returned 200 in **0.252 / 0.159 seconds**.
These are ordinary HTTP samples; no fresh-browser opening or zero-instance cold
opening was measured for this revision. They do not establish a cold-start or
browser-opening improvement. The measured production analysis demonstrates the
new execution path without the previous Job scheduling wait.

The initial local preparation passed 541 focused tests and Ruff across
authentication, projects, output validation, old/new runners, storage, cleanup,
startup, migration routing and execution races. Later focused helper tests cover
exact scoped temporary grants, readback after unknown writes, private state,
role ownership, and direct ID-token issuance without access-token impersonation.
Both temporary QA bindings were revoked with absence and completed cleanup
verified by IAM readback; the signer helper also removed its owned custom role.
Cleanup removed only the three owned disposable projects and five identities;
the private QA credential state file is absent. The owned diagnostic and team
preview services were deleted, with subsequent API readback confirming both
absent. `artifacts/verification/sandbox-cleanup-final.json` records complete
cleanup, including temporary IAM bindings/owned signer role and zero pending
runtime or broker scopes. The production control and private compute services
remain. See
[architecture, deployment contract and rollback](sandbox-compute.md).

## Opening performance and queued analyses

On 1 October 2026, the previous production revision showed a 30.248-second
request for the saved project page. The corresponding Cloud Run instance started
at 13:14:02.518 UTC, Uvicorn listened at 13:14:27.981, and the old five-second HTTP
probe succeeded at 13:14:32.853. These are actual startup logs, distinct from
warm HTTP samples (about 0.23 seconds for the page/auth config) and the existing
signed-in project's warm browser samples (3.304 and 2.414 seconds until the saved
draft appeared in a dedicated measurement tab).

The web entry point now imports the public analysis API lazily and does not load
Torch, pandas, the console or legacy server. Content-hashed assets use gzip and
immutable browser caching. The editor loads only when opening a workbench;
one authenticated project bootstrap returns the draft, role, files and active-run
lock, while result history loads separately. Revocation, verified-email and
live membership checks are retained; no authentication result is cached.

A separate live analysis never reached task startup before the old 300-second
lease cancelled it. The bounded lease now includes a 900-second platform wait
plus the unchanged task budget (Python's requested timeout, at most 120 seconds,
plus 120 seconds overhead). Transfer capabilities last 30 minutes without
expanding their object/generation/size scope. Explicit cancellation remains
separate from total wait expiry. Team clients request a 202 launch acceptance
and poll for validated results instead of keeping an HTTP request open across
the platform wait; older synchronous and local clients retain their protocol.
These changes do not establish faster Cloud Run Job provisioning.

The final TypeScript/Vite build and all **54 frontend tests** passed. Selected
Python startup, cloud, team auth/store/server/storage/runner/job tests and Ruff
passed. Production image checks verify the lightweight control imports before
checking nonroot/CPU Torch/native OLS. No production Python analysis was submitted
for this performance check.

Cloud Build `81fd5624-7234-4504-8914-64bbda2c91ce` succeeded. Revision
`openecon-00006-g7f` serves 100% of traffic; both web and compute use image
`sha256:7de257aa46d84e1c712210688bba6fe268733f9572394cb9c584c2d70abe1708`.
Zero minimum instances, maximum instances, CPU/memory, identities and worker task
settings are unchanged. The HTTP startup probe now checks each second with the
same 120-second startup allowance.

During this deployment, the new instance started at 13:52:05.423 UTC, Uvicorn
listened at 13:52:13.126 and the probe succeeded at 13:52:13.667: **7.703 seconds
to listening / 8.244 seconds to readiness**, compared with the older
25.463 / 30.335-second observed cold instance. Deployment-rollout startup and an
idle-triggered startup are different platform events; these samples do not
promise a fixed future cold-start time.

The first signed-in project opening after deployment took **5.320 seconds**,
including the new bundle and initial server-side identity lookup. Subsequent
reloads of the same saved draft in the same measurement tab took **1.943 and
1.761 seconds**, versus the previous **3.304 and 2.414 seconds**. The draft's
visible contents matched; history appeared afterward, and no browser console
errors were captured. Public page/bundles returned 200 and unauthenticated
project bootstrap returned 401/no-store; authenticated bootstrap requests
returned 200. JavaScript/CSS responses confirmed gzip and immutable caching.
The main/editor assets transfer about 112/156 kB compressed, versus about
391/462 kB uncompressed in the previous release. This is a transfer comparison,
separate from the browser and instance measurements above.

Evidence: `artifacts/verification/opening-before-startup.json`,
`opening-after-logs.json`, `opening-before-browser.json`,
`opening-after-browser.json`, `opening-after-http.json` and
`opening-deployed-service.json`. The public `/healthz` request receives an
upstream 404; readiness here is established by the internal Cloud Run HTTP probe
and successful application/API requests, not that public path.

## White and navy interface

On 1 October 2026, the workbench, authentication, projects and membership screens
were simplified to white surfaces, navy controls and restrained borders. The
team workbench uses its existing project header without a second brand header;
files and code examples now open from one dialog. Repeated runtime explanations,
marketing copy, alpha badges and the persistent inspector were removed. Auth
errors, save/conflict feedback, data-coverage warnings and permission checks
remain. Successful isolated runs no longer show a routine teardown warning.
Local session controls and the warning about ephemeral legacy cloud data remain
available in their appropriate modes. The team edition does not show a connection
button for its unavailable remote agent protocol.

All **46 frontend tests** passed, including the **14 chart renderer tests**;
**33 Python chart tests** and the TypeScript/Vite build passed. The last cosmetic
cleanup was followed by another successful build. No new tests were added for
colors or copy. Local browser checks used a separate disposable workspace:
the starter OLS/table/scatter analysis completed, chart and file dialogs opened,
a snippet was inserted, and the 390-pixel code/result switch worked. The compact
login was rendered separately without signing in; its 346-pixel form fits the
390-pixel viewport. Local previews are distinct from production verification.

Cloud Build `f18e3808-19be-4f23-a086-bbfbe9e1217f` succeeded with its in-image
checks. Revision `openecon-00005-2tx` serves 100% of production traffic; web and
compute use image
`sha256:7120974e2f65b2b28835a84d48596f141fb51026f6f8435e2f67249b734f6da5`.
Live checks returned 200 for the page and new JavaScript bundle, and 401 for
anonymous project access. The authenticated owner's saved project reopened with
its existing draft and results in the simplified interface. No production
analysis was run or user code edited for this visual check. Evidence:
`artifacts/verification/minimal-ui-live-check.json`, `minimal-ui-service.json`,
`minimal-workbench-live.png` and `minimal-projects-live.png`. The live project
list and membership dialog also opened correctly; no invitations or memberships
were changed.


## Verification email feedback

On 1 October 2026, a missing verification email report exposed two client-side
failure paths: registration switched away from the sign-up component before
its mail error could be displayed, and an optional display-name update could
prevent the first mail request. The parent auth shell now owns the initial send;
mail status is scoped by Firebase UID and separate from profile refreshes.
Concurrent sends are deduplicated, retries wait 60 seconds (120 after provider
throttling), and verification checks invalidate stale account-switch results.
The interface reports provider acceptance rather than claiming inbox delivery.

All **46 frontend tests** passed, including eight focused deferred-request,
account-switch, retry and safe-error tests. TypeScript/Vite and an independent
read-only review passed. No verification requirement or project permission was
relaxed. The cooldown is a client UX safeguard, not a server abuse limit.

Read-only production checks found the reported password account enabled and
unverified, default provider email delivery configured, and the expected Firebase
action callback and authorized application domain. No past send result or inbox
delivery could be established: activity logging was not enabled, and the browser
was using a different, verified Google account. No message was resent from that
unrelated session; a sign-in to the reported account was requested for diagnosis.
The feedback fix alone does not establish the cause of the missing message.

Cloud Build `85fa8390-cddb-4a13-a85f-cc256fbf4c63` succeeded, including the
production image's nonroot/CPU Torch/native OLS checks. Revision
`openecon-00004-xvw` is Ready and serves 100% of traffic with image
`sha256:a6368ac4a2e1d389f4297d5b1408947c59616444feac89a70d11c246f5bd7553`.
Live checks returned 200 for the page and its new `index-DpuuXvR7.js` bundle,
confirmed the new verification feedback and retry text, and returned 401 for
anonymous project access. The provider project remains `openecon-workbench`.
Evidence: `artifacts/verification/verification-email-live-check.json` and
`verification-email-service.json`. These checks do not prove inbox delivery.

## Google sign-in release

On 1 October 2026, Firebase's Google provider was enabled in the dedicated
`openecon-workbench` project with its own OAuth client. Password sign-in remains
enabled, anonymous sign-in disabled, and one account per email retained. Only
profile/email sign-in scopes are requested. The client secret stays with the
identity provider and is not embedded in application configuration.

The auth/team HTTP suite passed **127 tests**; the frontend passed **38 tests**,
TypeScript and Vite. Existing cloud identity/server and team store regressions
also passed; Ruff and whitespace checks passed. New tests cover immediate popup
invocation with an explicit resolver, provider scope isolation, cancellation,
credential collisions, verified Google claims, same-UID membership, denial for
the same email under a different UID, revocation and strict auth-domain/CSP
validation. The existing Starlette/httpx deprecation warning remains.

Cloud Build `8d4f3603-f11d-473d-8da4-102f07138f06` succeeded, including its
production image checks. Revision `openecon-00003-2ht` serves 100% of production
traffic; its image and the compute job image are
`sha256:f7b4581819f5421a74ad05ed29b694e90b40b17ff0ae90e7067ead5b3a397b13`.
Live HTTP checks returned 200 for the application and 401 for anonymous project
access. CSP permits only the required Google script host and this project's
Firebase auth iframe; COOP allows the OAuth popup. Frame embedding remains denied.

At the first browser check, an existing unverified password session was active.
At a later check on 1 October, the real owner had completed Google sign-in and
opened their own saved project in the published interface. A read-only Firebase
lookup confirmed the `google.com` provider and a verified email for that account.
The earlier password account remains separate and unverified. Legacy
password-account UID preservation through a live Google link is still unverified.

## Team edition verification

On 1 October 2026, the complete Python suite passed **695 tests**, with **3 CUDA
hardware checks skipped**. The frontend passed **34 tests**, TypeScript and Vite;
Ruff passed. npm audit reported **zero known vulnerabilities** after pinning the
patched gRPC transitive dependency. One pre-existing Starlette/httpx deprecation
warning remains.

New checks cover verified/revoked identity, project isolation, all viewer write
denials, email-bound invitation acceptance/revocation/expiry/replay, protected
ownership, concurrent draft edits, project creation quotas, run leases,
membership removal during publication/readback, streamed request limits,
generation-pinned storage capabilities and malformed/oversized worker output.
The plot boundary has 100 cases including genuine helper roundtrips. These are
local/injected-service checks; live cloud execution and browser checks are
recorded separately after deployment.

The final Cloud Run SDK compatibility fix also passed all **36 runner tests**,
including two new cases with actual protobuf `Any`/`Execution` messages. The
completed-condition field can be `type_` in the SDK's normalized response.

Live preview integration passed with five disposable identities. Anonymous and
unverified access were denied; invitation email binding, single acceptance,
project isolation, shared draft persistence/conflicts, immutable CSV upload and
download, viewer execution denial, role changes and immediate removal were
verified against real Firebase, Firestore and Cloud Storage. A separate Cloud
Run Job fitted the 480-observation OLS model, produced a D3 coefficient chart,
read the uploaded CSV and published a generated CSV that another member read
back. End-to-end elapsed time for that run was **145.5 seconds**, including cold
job startup. This is a measured preview run, not a latency guarantee.

The first preview exposed an IAM scoping issue: job-level permissions allow job
execution but do not cover region-level operation polling. A custom role with
only `run.operations.get` was added to the control identity at project scope;
the integration then passed. The worker remains without granted data roles.
The separate live IAM review has **19 passing checks**; its coverage and limits
are recorded in [team security review](team-security-review.md).

Production revision **`openecon-00002-zq5`** is Ready and serves 100% of traffic at
<https://openecon-291739190496.us-central1.run.app>. Cloud Build
`d1c75a22-ecda-4400-b61c-d9192afa5f78` succeeded, including its in-image nonroot,
CPU Torch and native OLS checks; both the web service and fixed compute job use
`sha256:93d9a578be7528fc0edad9a55237abcb1576a6055c8c7496620b3482922d159b`.
The complete live integration was repeated against the production URL and
passed, including an actual OLS/chart/input/output job in **97.4 seconds**.
The publicly reachable login replaces IAP; anonymous project API calls return
401 and verified project membership is checked separately.

Browser checks created a project, issued an owner invitation, signed in as the
recipient and accepted it. The owner's full starter script fitted the expected
HC3 model and rendered all 480 scatter points. After the deployment changed,
the invited editor signed in on the production origin and opened the same
saved draft and model/chart output. Its membership dialog contained no owner
administration controls. Evidence images are
`artifacts/verification/openecon-teams-login-live.jpg`,
`openecon-teams-shared-live.jpg`, `openecon-teams-projects-preview.jpg` and
`openecon-teams-members-preview.jpg`. These checks use disposable accounts;
the real owner's registration and email verification remain user actions.

Two further live jobs verified the actual compute identity boundary. Using its
metadata identity, the worker received **403** from a Firestore project document,
Firestore collection listing, private bucket listing and Cloud Run job read.
No credential-bearing environment variable names were present. A generated
output from an earlier run was absent from the next run's inputs. A second
independent job had neither the first job's sentinel variable nor its marker
file. The two end-to-end durations were **153.11 and 217.04 seconds**. The report
is `artifacts/verification/team-isolation-live.json`; IAM signing was not directly
probed. These are concrete boundary tests, not a claim of exhaustive penetration
testing or a fixed startup latency.

After verification, all **five disposable Auth identities, four QA projects,
their invitations and active test file objects** were deleted; the ignored local
credential file was removed. The temporary preview service and its authorized
Auth domain were also removed. Existing Cloud Storage soft-delete retention and
provider audit/build/job history still apply. The production browser was signed
out and left on the application login screen. Nonsecret integration assertions
are retained in `artifacts/verification/team-live-qa.json`.
Post-cleanup readback still returned 200 for the login page/config and 401 for
anonymous project access. An external request to `/healthz` returned a 404 HTML
response; it is not used as evidence of public application health. Cloud Run's
internal startup probe and the revision Ready state were checked separately.

## Cloud access and packaging verification

Verified on 1 October 2026. The full local suite passed **379 tests** with **3 hardware-dependent CUDA checks skipped**; the frontend passed **25 tests**, TypeScript/Vite and Ruff. There is one existing third-party Starlette/httpx deprecation warning.

The new authentication cases exercise real P-256 signatures with controlled test keys, wrong owners, issuer/audience mismatches, expired/malformed tokens and bounded key-fetch failures. Server cases verify that session bootstrap and chart assets require verified identity, unsigned email headers cannot grant access, same-origin/CSRF protections remain active, and cloud configuration does not advertise container-local MCP commands.

Cloud Build `c08495e5-9b04-4203-b8fc-9c0bc9d88dcf` successfully built the production Linux image and ran its nonroot/CPU Torch/package checks plus a native 480-observation OLS fit. Image digest: `sha256:f142b3fc57a842f53e5a975177356702894f667884aaf8bd4f3150a6855f64cc`. This initial build ran in `sustainarf-fsk4hp` before the user selected a dedicated OpenEcon project; the verified image can be copied without rebuilding or creating a runtime dependency on that project. See [cloud deployment](cloud-deployment.md) for the selected target and operating limits.

The identical image was copied into `openecon-workbench` and deployed successfully as revision `openecon-00001-5dw`, serving 100% of traffic at <https://openecon-291739190496.us-central1.run.app>. The ready state, IAP-enabled setting, dedicated runtime identity and exact service/IAP policies were read back. Anonymous requests to `/` and `/api/session` both returned HTTP 302 to Google sign-in. Owner access is limited to `anilsen@bluearf.com`. The old task-created OpenEcon service, runtime account and image repository in `sustainarf-fsk4hp` were deleted after migration; historical build records remain there.

Authenticated browser execution on the live deployment passed after the owner completed Google sign-in. The unmodified starter file fitted the 480-observation synthetic wage model with HC3 standard errors (R² 0.8253; intercept 3.0922, education 1.7289, experience 0.4183) and rendered all 480 scatter observations with zero exclusions. A subsequent command imported the independent `openecon_charts` package and rendered the same model's three coefficient estimates with 95% confidence intervals, with zero exclusions. The session retained `df` and `model` between commands. No browser console errors were recorded. The visible cloud notice correctly states that files, drafts and results are temporary; this check does not establish durable persistence or multiple-user isolation. Live screenshots are saved as `artifacts/verification/openecon-cloud-live.jpg` and `artifacts/verification/openecon-cloud-chart.jpg`. Deployment and policy snapshots are kept under the ignored `artifacts/verification/cloud-*.json` files.

## Independent D3 chart package verification

Verified locally on 1 October 2026, macOS arm64 / Python 3.13.5.

The new `openecon-charts` distribution owns chart specifications, table/model
adapters, offline HTML generation, the adapted D3 renderer and its local assets.
`oe.plot` re-exports it; the workbench serves its assets from `/chart-assets`.
The source repositories were kept unchanged.

| Check | Observed result |
| --- | --- |
| Python suite | 293 passed, 3 internal CUDA checks skipped. Includes 33 independent chart-package cases. One third-party Starlette/httpx deprecation warning. |
| Frontend tests | 25 passed: 14 real-D3 DOM/renderer cases and 11 session-state cases. Covers adaptive axis precision, selected-data SVG/CSV, single-series horizontal stacks, failed mount cleanup and dialog teardown. |
| Runtime independence | A fresh environment containing only the chart wheel generated standalone HTML. No OpenEcon, pandas, NumPy or Torch module was installed; chart metadata has no required dependencies. |
| Input semantics | Mapping/records/table adapters; duplicate requested columns rejected; negative/zero/missing values retained; numeric line gaps and irregular spacing preserved; display sampling declared; histogram counts use the reported bin boundaries. |
| Limits and precision | Unsafe integer conversion and overflowing ranges rejected; categorical aggregation remains explicit; compositions reject missing/negative components. |
| Browser integration | Real Python editor execution rendered the 480-observation scatter and OLS confidence intervals. Commands also rendered multi-series area, stacked and signed horizontal bar charts. Legend toggling, data tables, percent mode and enlargement worked. |
| Layout | 390px mobile document width remained 390px with a 345px chart host; no document overflow. Default 837px workbench also checked. |
| Packaging | Both wheels and source archives built. Final chart wheel bytes match all nine Python/asset source files. A production environment installed both wheels and ran a real OLS model plus a separate console worker displaying a chart. |
| Standalone HTML | Embedded D3/CSS/font document rendered outside the OpenEcon server, with external connections disabled by its CSP. |
| Exports | SVG and CSV download paths tested against actual D3 DOM output, including visible-series metadata and embedded font. Browser PNG action produced no observed error; native file-save completion was not independently established. |

The current port supports nine Python helpers; the source chart catalogue is
larger. Browser exports currently expose SVG, PNG, JPEG and CSV; PDF/XLSX are
not bundled. Notebook iframe output is implemented and serialization-tested;
no live Jupyter host was exercised. No new estimation or performance claim is
made by this chart integration. See [chart provenance](charts-provenance.md).

## Python workbench 0.3 verification

Verified locally on 30 September 2026, macOS arm64 / Python 3.13.5.

| Check | Observed result |
| --- | --- |
| Python tests | Full run: 257 passed, 3 internal CUDA checks skipped; subsequently added table-index/large-integer regression: 1 passed. Current total: 258 passing cases and 3 hardware skips. One third-party Starlette/httpx deprecation warning. |
| Numerical core | One mandatory PyTorch CPU float64 estimator. Analytical fixtures, independent statsmodels comparisons, custom normal/Student-t distribution checks, separation, rank, tails, location/scaling and covariance regressions passed. |
| Framework-independent input | DataFrame, mapping and record-list inputs; public API has no engine/device selector; outputs use ordinary Python/JSON values. |
| Console lifecycle | Real persistent child process; ordinary errors preserve prior variables; startup failure, timeout, interrupt, worker exit and subprocess cleanup verified. Saved source never auto-executes. |
| Console tables | DataFrame/Series/MultiIndex labels retained; integers beyond JavaScript's safe range remain exact strings. |
| Frontend state | 11 deterministic Node tests passed for serialized latest-draft writes, undo during a pending save, failed-write recovery and rejection of stale status responses. |
| Browser execution | Editor button and keyboard shortcut ran real Python; subsequent commands reused model/data variables; grouped tables, histogram and scatter plots displayed; a deliberate division error recovered and a sleeping command was interrupted. |
| Browser persistence/layout | Editor draft survived reload; desktop 1440px, normal 837px and narrow 390px layouts checked; narrow layout exposes variables/history and switches between editor/results without document overflow. |
| Clean production wheel | Installed in a fresh environment without statsmodels; required Torch 2.14.1, all three estimators, summary output, packaged 480-row example and plot specifications worked. Main test environment used locked Torch 2.14.0. |
| Packaging | Wheel includes console worker, single tensor engine, frontend and example CSV; no NumPy engine or direct NumPy/statsmodels requirement. Source distribution includes frontend state tests. TypeScript/Vite and Ruff passed. |

The NumPy package is still installed transitively through pandas/SciPy; no separate NumPy estimator exists. SciPy supplies only the separation linear-program safeguard. This record does not establish full Stata parity, GPU acceleration or a new performance advantage. The 0.2 timing results below are historical. Remote CI was configured but not run here.

Editor `.py` downloads are console drafts; they are not standalone bundles. Plot SVGs include displayed/valid sample counts and excluded-count metadata. The browser's native file-save completion was not independently established in this session.

## Historical native engine 0.2 verification

This section records the previous dual-engine release. Backend selectors, optional
Torch installation and the associated benchmark timings do not describe the
current single-core code workbench. Current implementation must be verified with
its current test suite; earlier passes are not carried forward automatically.

Verified locally on 30 September 2026 with Python 3.13.5 / PyTorch 2.14.0 on Apple Silicon.

| Check | Observed result |
| --- | --- |
| Full Python suite | **218 passed, 43 CUDA cases skipped** because no CUDA device was present; one Starlette/httpx deprecation warning |
| Independent estimator comparisons | 80 passed across native NumPy and PyTorch CPU; 40 CUDA cases explicitly skipped |
| Core numerical checks | Analytic covariance, finite-difference likelihood derivatives, extreme tails, scaling/location invariance, rank, convergence, leverage and cluster cases |
| Runtime independence | Subprocess blocked statsmodels and Torch imports while all three NumPy estimators ran |
| Clean production install | Wheel installed in a fresh environment containing neither statsmodels nor Torch; OLS, logit, probit and packaged HTML all worked; optional Torch request failed explicitly |
| Browser | Both NumPy and Torch CPU selected and executed; saved provenance showed the actual backend, CPU and float64; unavailable CUDA showed a Turkish error |
| PyTorch replay | ZIP retrieved through the live HTTP API, extracted elsewhere and executed; actual backend remained PyTorch, sample hash matched, coefficients/inference and covariance agreed at 1e-12 tolerance |
| Distribution and static checks | Python wheel/sdist, TypeScript/Vite production build and Ruff checks passed |
| Performance | Same-input, same-boundary comparisons against the preserved former OpenEcon adapter; complete protocol and measured variability in [performance.md](performance.md) |

CUDA code has not been hardware-validated on this machine. Skipped cases are not passes. The production wheel has no statsmodels dependency; development environments include it only for independent comparisons. The CI workflow now requests the Torch extra, but remote CI has not run in this task.

## Previous 0.1 validation

Verified locally on 30 September 2026, macOS arm64 / Python 3.13.5.

| Check | Observed result |
| --- | --- |
| Python suite | 83 passed; one Starlette/httpx test-client deprecation warning |
| Static checks | `uv run ruff check src tests scripts` passed |
| Frontend | TypeScript checking and production Vite build passed |
| Distribution | Source distribution and wheel built; wheel contains HTML, JS and CSS assets |
| MCP | Actual stdio subprocess initialized, tools discovered, analysis run and persisted result retrieved |
| Replay | ZIP fetched from the running HTTP server, extracted into another directory, script executed; sample hash matched and coefficients agreed at 1e-12 tolerance |
| Browser: data | Synthetic example and a real CSV upload loaded with their profiles |
| Browser: models | OLS/HC3, categorical OLS at 90% confidence, logit and one-way-cluster probit ran successfully |
| Browser: errors | Invalid binary outcome blocked; missing-value policy rejected by default and explicit deletion worked; few-cluster warning displayed |
| Browser: persistence | Saved analysis reopened after page reload; changed model options marked the previous result as stale |
| Browser: presentation | Desktop and 390-pixel layouts visually checked; final console error/warning log empty |
| Agent setup | Correct absolute Codex/Claude command lines shown; user client settings were not modified or exercised |

The UI ZIP button issued a successful HTTP 200 export request. The browser automation's download event timed out, so completion of its native file-save flow was not established. The exported archive itself was independently retrieved through the live HTTP API and replayed as described above.

These checks establish the listed alpha workflows. They do not validate full Stata parity, remote serving, platform portability, peak RAM, or cloud model integration. The GitHub Actions workflow is included but has not run remotely in this task. Performance methodology and measured scope are in [performance.md](performance.md).
