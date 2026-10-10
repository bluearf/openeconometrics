# Team workspace security review

> Cloud execution was retired on 10 October 2026: the cloud service is now a
> team sync layer and every analysis runs in the desktop app. Entries below
> that describe sandbox, Job or browser cloud runs are historical records of
> the removed backends, not current behavior. See
> [team sync backend deployment](sync-backend-deployment.md).

Review date: 1 October 2026. This is an implementation and deployment checklist,
not a penetration-test certificate. Local test results, cloud policy readback
and authenticated browser behavior are separate evidence.

## Required separation

The public control API authenticates users, authorizes project access, stores
metadata and schedules computation. It must not execute submitted Python,
deserialize pickle, restore a user archive into its filesystem, or parse
spreadsheet/dataframe formats. `team_server.py` stores uploads as bytes; all
user code belongs in a separate Cloud Run Job execution.

The existing local `ConsoleSession` is not a sandbox. In particular,
`Connection.recv()` unpickles values from its Python child. Reusing that parent
inside the credentialed control process would allow hostile child code to cross
the boundary. It may be used inside the disposable, roleless compute job where
the entire instance is already untrusted. The job-to-control protocol is bounded
JSON only. See [Python's connection warning](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.connection.Connection.recv).

Each execution uses a fixed image, one task, a bounded timeout, no retries and
a service account with no IAM grants, secrets or shared volumes. Cloud Run still
exposes that account's identity through metadata: having a dedicated account
without privileges is necessary even if user code can reach metadata. The job
must never inherit the API account or URL-signing account. See
[Cloud Run service identity](https://docs.cloud.google.com/run/docs/securing/service-identity).

Each run starts fresh; permanent project files, drafts and results do not imply
permanent Python variables. A future persistent interactive worker needs its own
isolation design. Cloud Run's nested sandbox facility is currently a preview,
not an assumed property of an ordinary Python subprocess.
[Sandbox documentation](https://docs.cloud.google.com/run/docs/configuring/jobs/sandboxes).

## Authentication and authorization

`team_auth.py` verifies ID tokens with Firebase Admin against the explicit
Firebase project and uses `check_revoked=True`. UID is the account key; email
matches an invitation only after verification. The accepted sign-in providers
are email/password and Google (`password`, `google.com`). Other providers and
Firebase tenant tokens are rejected. A raw Google OAuth token is not a Firebase
ID token and cannot authenticate to this API.
Emulator environment settings cannot disable production signature checks.
`/api/me` may inspect an unverified identity; project resources may not.
[Token verification](https://firebase.google.com/docs/auth/admin/verify-id-tokens)
and [revocation behavior](https://firebase.google.com/docs/reference/admin/python/firebase_admin.auth#verify_id_token).

Signup alone does not enable the application. The configured verified owner may
bootstrap; another account must accept an unexpired invitation addressed to its
verified email. Invitations are accepted in the app; creating one does not send
email. Enabled accounts may own at most five projects. Account enablement is
separate from access to any existing project.

Google sign-in does not change this authorization model. `TeamStore` looks up
memberships and profiles using the verified Firebase UID. An identical email
with a different UID does not inherit existing projects; provider profile data
cannot assign a role. The email bootstrap enables an account to create its own
projects, not to take over projects owned by an earlier UID.

Keep Firebase's one-account-per-email policy and let Firebase apply its provider
trust rules. Some collisions require authentication through the existing method
before explicit linking. The current interface offers the existing sign-in
method rather than performing a custom merge. Any future linking feature must
retain the existing Firebase UID and reject credentials already attached to a
different account. Do not infer account ownership from email equality.
[Firebase linking](https://firebase.google.com/docs/auth/web/account-linking) and
[Identity Platform provider trust](https://docs.cloud.google.com/identity-platform/docs/concepts-manage-users).

Keep email enumeration protection enabled. Sign-in-method discovery returns an
empty list under that protection, so it is not a reliable collision-recovery
mechanism. Provider enablement, authenticated Google popup completion, and
same-UID access to a legacy password account's projects require separate live
evidence; mocked token tests cannot establish Firebase's linking behavior.
[Firebase sign-in-method lookup](https://firebase.google.com/docs/reference/js/auth#fetchsigninmethodsforemail).

| Operation | Owner | Editor | Viewer |
| --- | --- | --- | --- |
| Read project files, drafts and results | Yes | Yes | Yes |
| Upload files, edit draft, run/interrupt Python | Yes | Yes | No |
| Invite, revoke invitations, change/remove members | Yes | No | No |
| Remove/demote/transfer the project's fixed owner | No | No | No |

Membership must be checked using the requested project's live record for every
resource, including downloads and polling. Mutations check it inside the same
transaction as the write. User-supplied role, UID, owner and project claims are
not authorization. Removing/demoting an editor prevents publication of that
editor's pending run. Raw store methods such as `run`, `update_run` and
`finish_run` are internal; HTTP handlers must authorize their use.

Direct Firestore client reads/writes are denied. Server libraries bypass those
rules, so the API's checks remain mandatory. The API identity needs only the
reviewed database, job, bucket and signing permissions; Firebase revocation
lookup requires `firebaseauth.users.get`. No account-management write role is
needed for token verification.
[Firestore server authorization](https://firebase.google.com/docs/firestore/security/rules-conditions)
and [Firebase Auth permissions](https://docs.cloud.google.com/iam/docs/roles-permissions/firebaseauth).

## File and execution capabilities

- Generate object keys from server-owned project/run IDs; filenames are labels,
  never arbitrary paths. Inputs are immutable and generation-pinned.
- Keep the bucket private. Use a dedicated signing account with only input-read
  and staging-create permissions. Grant the API `iam.serviceAccounts.signBlob`
  on that account; grant the worker neither signing nor bucket roles.
- Signed URLs are bearer capabilities until expiry. Do not log them or claim
  that membership revocation immediately invalidates an issued URL. The API
  proxies user downloads after membership checks.
- Enforce output size in the signed POST policy, using an exact run-specific
  key and `content-length-range`. Expiration alone does not bound upload size.
- Inspect output size and pin its object generation before download. Validate
  the untrusted JSON and publish under fresh immutable control-owned keys;
  never let worker output replace a project's current metadata or input files.
- Final publication rechecks editor membership and cancellation. Concurrent
  finalizers and retries must be idempotent, with unreachable staged blobs
  cleaned by a lifecycle policy. Never release an ambiguously launched job's
  lock merely because the launch response was lost.
- Bound request bodies while receiving them, including missing/chunked
  `Content-Length`; multipart parsing happens before endpoint file iteration.
  Bound aggregate response/history bytes as well as individual records.
- Treat downloadable HTML/SVG as attachments, not same-origin application code.
  Validate every model/table/plot field used by the UI, including array limits
  and index metadata; valid JSON alone does not make a safe rendering contract.
- If archive import is later added, reject traversal, absolute paths, symlinks,
  hardlinks/devices, duplicate/colliding names and decompression bombs; bound
  compressed bytes, expanded bytes, entries and depth. Current runs use explicit
  flat file manifests instead of archive extraction.

[Signed URLs](https://docs.cloud.google.com/storage/docs/access-control/signed-urls),
[upload policy size conditions](https://docs.cloud.google.com/storage/docs/authentication/signatures)
and [generation preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions)
describe the cloud primitives; the project/run binding is application policy.

## Attack and failure matrix

| Test | Required result |
| --- | --- |
| Foreign-project, revoked, disabled, malformed, unsigned/emulator or wrong-provider token | Reject without exposing token/provider details |
| Unverified account calls `/api/me`, then a project route | Status may be shown; project access denied |
| Exact foreign project/file/run/invitation ID is supplied | No data disclosure or mutation |
| Viewer executes/uploads/edits; editor manages membership | Reject; no job, storage or metadata side effect |
| Wrong-email, expired, revoked, reused or concurrently accepted invitation | At most one authorized acceptance |
| Owner removal/demotion and self-promotion | Reject; fixed ownership retained |
| Concurrent last project slot, draft save or project run | Quota/version/lock admits exactly one valid winner |
| Membership removed during execution or output publication | Pending output is not published to the project |
| Launch timeout, cancellation failure, job crash or API restart | Bounded work; durable status; eventual lock recovery |
| Staging object replaced between size check and download | Read the pinned generation or reject |
| Oversized/chunked upload, enormous result/history or malformed plot/table/model | Bounded resources and controlled error, no persistent UI crash |
| Worker reads metadata or invokes IAM/database/storage/job APIs | No unrelated authority; only its signed run capabilities work |
| Browser injects HTML in code, labels, filenames or results | Text remains text; downloads cannot execute on the app origin |
| Two projects run concurrently | No shared filesystem, variables, capabilities or project results |

Per-project daily quotas do not provide a global spending or concurrency limit.
Bound active jobs and request rates across accounts as the service grows; retain
audit records for role, invitation, upload and run actions without credentials.

## Evidence and remaining deployment checks

- Authentication unit suite: **48 passed**. Uses injected verifiers and a mocked
  SDK boundary; it proves claim policy and revocation-call wiring, not live
  Firebase configuration.
- Store unit suite: **50 passed**. Includes actual threaded races, rollback,
  immediate role changes, revoked publication, quotas and staged Firestore
  read-before-write ordering. It does not substitute for live Firestore tests.
- API, transfer/runner and browser verification must be recorded by the owning
  integration tasks after fixes. Review identified body streaming limits,
  ambiguous launches/cancellation recovery, complete output schema validation,
  control/worker filename consistency and bounded history as integration gates.
  A subsequent source read confirms these changes were implemented; endpoint,
  runner and live-cloud verification remain separate checks.
- Plot boundary suite: **100 passed**. All ten helper variants survive the JSON
  boundary; invalid geometry, oversized arrays, nonfinite/unsafe numbers,
  malformed labels/options and excessive explicit heights are rejected.
- Before replacing the private owner-only service: read back Firebase provider
  configuration, deny-client Firestore rules, bucket privacy, scoped signer and
  control IAM, roleless job identity, fixed image/limits/retries and public API
  authentication. Verify owner/editor/viewer with separate real sessions and
  repeat a cross-project denial and membership-revocation run.

No live security setting was changed by this review.

## Initial preview IAM and storage readback — 1 October 2026

The sanitized local snapshot is `artifacts/verification/team-iam-audit.json`.
It records the policies, exact custom permissions, conditional object prefixes,
rules source/hash and deployed image. All reads succeeded; no cloud settings
were changed by the audit.

| Boundary | Observed live configuration |
| --- | --- |
| Compute identity | `openecon-compute` has no direct project/organization role, no binding on either project bucket, and no grant on any of the six project service accounts permitting it to impersonate another account. |
| Control identity | Project roles are `datastore.user`, custom `openeconAuthReader` containing only `firebaseauth.users.get`, and custom `openeconOperationReader` containing only `run.operations.get`. Job-scoped grants are `run.jobsExecutorWithOverrides` and `run.viewer`. Project-bucket access is `storage.objectAdmin`. |
| Transfer signer | No project/organization role. Bucket grants contain only `storage.objects.get` under `projects/` or `staging/`, and `storage.objects.create` under `staging/`. Control can call only `iam.serviceAccounts.signBlob` on this signer through its custom signing role. |
| Permanent project bucket | Public access prevention is enforced, uniform bucket-level access is enabled, and no public IAM member is present. Object ACLs cannot grant public access under uniform access. Staging objects have a one-day deletion lifecycle; seven-day soft deletion is also configured. |
| Direct Firestore clients | Active `cloud.firestore` release points to ruleset `79475825-c2fc-4772-b861-81502619faea`; its only rule allows all document reads/writes only `if false`. Rules SHA-256: `314dbea92af30a17fde8fe352ef780a015458138dd53286747dd556ebfbf913a`. |
| Compute job | Ready; dedicated compute identity, one task, parallelism one, zero retries, 240-second hard timeout, 2 CPU and 4 GiB. No base environment variables or attached volumes. |
| Team preview | `openecon-teams-preview-00001-rlm` is Ready with 100% traffic and the control identity. Its public Cloud Run invocation grant is intentional; Firebase/application authorization remains the HTTP security boundary. |

The job and preview both use image digest
`sha256:4241ffe7a763f165d89e39e872320c753f5282bbb405b8f4c30c2693448bc7ec`.
These describe the initial preview deployment. The final production readback
below is a later observation; neither readback substitutes for authenticated
team browser tests.

A subsequent targeted IAM readback confirmed the operation-status exception:
Cloud Run's long-running operation belongs to the project's region, so the
job-scoped viewer binding does not authorize the control API to poll it.
`openeconOperationReader` therefore grants `run.operations.get` at project scope.
This allows reading a known operation's status elsewhere in this same project;
it is not restricted to the fixed job. The custom role has no operation-list,
execution, cancellation, deployment or update permission. Job execution and
job/execution viewing retain their resource-scoped bindings. The updated audit
records this additional read permission explicitly; IAM readback alone does not
prove that live status polling succeeded after propagation.

The project and its organization IAM policies were inspected. Organization
group memberships were not expanded, external-project grants were not
enumerated, and no hostile worker execution was performed by this audit.
Legacy/default privileged identities still exist separately in this project;
the dedicated worker does not inherit another service account's roles.

## Final production readback — 1 October 2026

The separate `production_readback` section in the sanitized audit records the
final observations from **30 September 2026, 23:50:34–23:52:46 UTC** (1 October in
Istanbul). The earlier 19 checks remain explicit point-in-time evidence for the
initial configuration and operation-reader update. The final production record
contains a separate set of **19 passing checks**.

- Production `openecon-00002-zq5` is Ready and receives **100%** of service
  traffic at `https://openecon-291739190496.us-central1.run.app`.
- The web service uses `openecon-control`; the compute job uses
  `openecon-compute`. Both use digest
  `sha256:93d9a578be7528fc0edad9a55237abcb1576a6055c8c7496620b3482922d159b`.
  Regional build `d1c75a22-ecda-4400-b61c-d9192afa5f78` is `SUCCESS` and reports
  this digest.
- IAP is not enabled in the Cloud Run configuration. Anonymous `GET /` returns
  **200 HTML**, without an authentication redirect; anonymous `GET /api/me`
  returns **401 JSON**. The public invoker binding is intentional: authenticated
  project access is enforced by the application.
- Service and revision maximum instances are both **2**. No minimum instance
  count is configured, so the Cloud Run default is **0**; the audit distinguishes
  this default from an explicitly returned zero.
- The compute job remains Ready with one task, parallelism one, zero retries
  and a 240-second hard timeout. Fresh project, organization, two-bucket and
  six-service-account policy reads found no direct compute identity grant.
  Control project roles, job roles, transfer prefix grants and service-account
  bindings match the preceding audit.

This production pass performed only configuration/policy reads and anonymous
HTTP checks. It did not execute user Python or perform the separate worker
permission-denial probe. No cloud resource or IAM policy was changed.
