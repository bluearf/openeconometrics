# OpenEconometrics team edition

The team edition adds an application login, persistent projects and project
membership to the existing Python workbench. The local edition remains a local
Python console. Production teams use a fresh sandbox for each script through a
private Cloud Run compute service;
the credentialed web/API process never executes user code or parses uploaded
statistical data formats.

Live URL: <https://openecon-291739190496.us-central1.run.app>. The current
publication-table release is `openecon-00009-28v` in the dedicated
`openecon-workbench` project, region `us-central1`, Ready and serving 100% of
traffic on 2 October 2026. Its nonroot control image is
`sha256:7fc9e78250dc238712e72920b51be752772ba34cf7a38aa14e7c9ffe9a8a6106`,
built and checked by Cloud Build `e184c361-fd63-4137-9a41-82b38b7d06cb`.
The private compute revision is `openecon-sandbox-00003-qf7`, pinned to the
separately reviewed supervisor image
`sha256:4304cc205d36dc8a71d575a86c618a4ab86378523d72432ddfbcc57a98a98a70`.
The preceding `openecon-sandbox-latex` broker is retained for in-flight work
and an explicit v8 rollback. Both active services keep minimum instances at zero.
See [sandbox deployment, control commands and rollback](sandbox-compute.md)
for the current production path. The Job backend below is retained for legacy
execution handles and explicit rollback; it is not the deployed primary path.
The first owner signs in as `anilsen@bluearf.com` through the application login
screen; the previous Google IAP login is not an application account.

## Account and project flow

Users can continue with Google or register with email and password through
Firebase Authentication / Identity Platform. Password registration requires
email verification; Google sign-in uses the verification claim issued by
Firebase. Both methods still require a verified email for project access.
Passwords and Google credentials are handled by the provider. The API accepts
the resulting Firebase ID token, checks the explicit project, and verifies
revocation and disabled-user status. An unverified account can view its
verification state but cannot access project resources.

Registration sends one verification email independently of the optional profile
name update. The verification screen retains the provider's success/error result
through profile refreshes and displays a resend cooldown. A successful request
means the provider accepted the send; it does not confirm inbox delivery.
No email is automatically resent when an existing account opens the application.

Enable the Google provider in Firebase Authentication, retain **one account per
email address**, and authorize the deployed application domain. The browser uses
the project's Firebase auth handler and requests only normal sign-in profile and
email information. OAuth client secrets belong in provider configuration, never
the browser bundle. [Firebase Google sign-in setup](https://firebase.google.com/docs/auth/web/google-signin).

Existing projects belong to the **Firebase UID**, not an email address or a
Google provider-specific identifier. Firebase-linked methods retain that UID.
Automatic same-email linking depends on the provider and email trust rules; it
is not guaranteed for every password account. When Firebase reports a provider
collision, the interface preserves password sign-in and asks the user to use
their existing method. This release does not provide a manual account-linking
screen or merge project data by email. A future linking flow must authenticate
the existing user and use Firebase's linking API.
[Account linking](https://firebase.google.com/docs/auth/web/account-linking) and
[provider trust rules](https://docs.cloud.google.com/identity-platform/docs/concepts-manage-users).

The configured owner's verified email bootstraps access. Other accounts gain
access by accepting a project invitation addressed to their verified email.
Creating an account alone does not grant project access. Accepted users can own
up to five projects. Removing someone from a project removes that project's
access; it does not disable their independent account or other memberships.

Project owners create or revoke invitations, change editor/viewer roles and
remove members. Invitations expire after seven days, are single-use and cannot
be accepted by another email account. The interface produces a shareable link;
it does **not** automatically send invitation emails. Provider verification and
password-reset emails remain available from their corresponding account flows.

| Role | Read and download | Edit/upload/run | Manage membership |
| --- | --- | --- | --- |
| Owner | Yes | Yes | Yes |
| Editor | Yes | Yes | No |
| Viewer | Yes | No | No |

The original project owner cannot be removed or demoted. Ownership transfer,
account administration, MFA enrollment, live simultaneous text editing and a
remote agent protocol are not part of this release. A shared `analysis.py`
draft uses version checks: stale writes fail instead of overwriting another
member's work. The browser preserves the conflicting local draft for download.

## Persistence and execution boundary

Firestore stores accounts, live project memberships, invitations, versioned
drafts, audit events and execution metadata. A private Cloud Storage bucket
stores immutable uploaded files, analysis records and generated artifacts.
Direct client Firestore access is explicitly denied by rules. Project IDs and
roles supplied by a browser are never an authorization source: every request
checks current membership, and writes check it inside their transaction.

Each execution receives only its own code and generation-pinned project input
files. Production explicitly selects `OPENECON_RUNNER=sandbox`. A private broker
verifies the control service account's Google identity and creates a fresh managed
sandbox with a curated rootfs, reduced guest privileges and bounded lifetime.
The worker receives no application tokens, database credentials or secrets;
its compute service account has no granted project/bucket permissions. A separate transfer
identity signs thirty-minute download capabilities and a POST policy constrained
to the execution's staging object and maximum upload size. This covers the bounded
platform startup wait and the task budget; Python still runs for at most 120 seconds.
It can read the
input prefixes and create staging objects, but cannot overwrite project files.

Worker output is untrusted JSON. The control plane verifies its size, result
schema, plot geometry and artifact names before writing immutable result
objects. No pickle or archive from the worker is loaded in the control plane.
Membership/cancellation is checked again before publishing a result and before
returning downloaded data. A run capability already used by a worker cannot
make previously read data unreadable when a member is removed.

Team clients receive a launch acceptance and obtain the validated result through
status polling. The internal broker stream reports progress, while only the
parent-written completion marker after confirmed sandbox deletion releases a run.
The broker's fatal lifetime bound is at most 240 seconds; the retained control
lease also covers older Job operations and unknown launch outcomes. Python's own
time limit, concurrency and daily quotas remain separate. See
[completion and cleanup](sandbox-compute.md#completion-disconnects-and-hard-lifetime-bound).

Every run starts a **fresh Python interpreter**. Run the complete script for a
reproducible analysis; variables are not retained between commands. Original
uploads remain immutable. New regular files created at the workspace top level
are captured as downloadable result artifacts; modified input files, nested
directories, symlinks and special files are not published as new project inputs.
Outbound GCS access is enabled for scoped transfers while metadata access is
blocked; the private broker separately rejects guest attempts to invoke it.
Worker output is not proof of statistical correctness: an editor
can deliberately fabricate data or results within their own project.

### Legacy Job backend and rollback

The retained `OPENECON_RUNNER=jobs` path runs one task in a fixed Cloud Run Job,
with zero retries and a dedicated roleless compute identity. Job provisioning
can take minutes. Its control lease allows 900 seconds for platform waiting plus
the unchanged task budget, at most 1140 seconds; that does not extend Python's
120-second execution limit. Older clients retain synchronous execution by
default. The worker's network is available but its identity has no data/API grants.

Previously accepted Job operations remain pollable and stoppable under the
production migration adapter. For rollback, use the reviewed candidate control
image with `OPENECON_RUNNER=jobs` and preserve all sandbox service/origin/image
settings so accepted sandbox handles can still be read and cancelled. Do not
route back to a pre-sandbox image until every accepted sandbox run has drained
with confirmed cleanup. Use the
[current control helper and rollback commands](sandbox-compute.md#control-deployment-and-rollback-commands).

## Limits and recovery

- Upload: 24 MiB per file; 20 files and 64 MiB of inputs per project.
- Code: 64,000 characters, up to 120 seconds of user execution, plus bounded
  startup/transfer time. Zero minimum instances can add cold-start latency.
- Generated files: 20 files, 16 MiB total. Structured outputs: 20 items, 2 MiB.
- Recent history: up to 50 records within an 8 MiB response budget.
- One active run per project, four active runs across the installation.
- 100 runs per project per UTC day, 200 across the installation per UTC day.
- A launch with an unknown outcome keeps its lease until its hard deadline.
  There is no automatic retry that could silently duplicate user code.
- Staging inputs/outputs have a one-day bucket lifecycle. Project files/results
  have no automatic deletion policy. Existing bucket soft-delete settings also
  apply. Durable storage is distinct from a tested backup/restore plan. The
  [project recovery runbook](team-recovery.md) now defines scoped backup/restore,
  RPO/RTO targets and a synthetic independent-environment drill; live cloud
  recovery and operational scheduling remain separate deployment checks.

## Production wiring

`python -m openecon.cloud` selects the team edition only when
`OPENECON_MODE=teams`; missing team configuration fails closed. Required values:

| Variable | Purpose |
| --- | --- |
| `OPENECON_PROJECT_ID` | Explicit Firebase/Firestore/Cloud Run project |
| `OPENECON_PUBLIC_ORIGIN` | Exact HTTPS origin for the interface |
| `OPENECON_OWNER_EMAIL` | Verified initial account bootstrap |
| `OPENECON_FIREBASE_CONFIG` | Public Firebase web app configuration JSON |
| `OPENECON_BUCKET` | Private project file bucket |
| `OPENECON_SIGNER_EMAIL` | Narrow file-transfer signing identity |
| `OPENECON_RUNNER` | Explicitly `sandbox` in production; `jobs` is the legacy/rollback path |
| `OPENECON_COMPUTE_SERVICE` | Fixed private sandbox service |
| `OPENECON_COMPUTE_ORIGIN` | Exact private service HTTPS origin / ID-token audience |
| `OPENECON_COMPUTE_IMAGE` | Reviewed immutable supervisor image digest |
| `OPENECON_BROKER_CALLER_SUB` | Fixed control service account unique subject |
| `OPENECON_COMPUTE_JOB` | Retained fixed Job for accepted legacy handles and rollback |
| `OPENECON_COMPUTE_EMAIL` | Expected roleless compute identity |
| `OPENECON_REGION` | Region, default `us-central1` |

The source retains `jobs` when `OPENECON_RUNNER` is omitted for older deployments;
current production therefore sets `sandbox` explicitly. Use
[the reviewed sandbox/control deployment helper](sandbox-compute.md#control-deployment-and-rollback-commands)
to preserve the existing private environment and public control configuration.

The web service uses `openecon-control`, with database access, bucket object
management, read-only Auth user lookup, `signBlob` only on `openecon-transfer`,
private sandbox invocation and retained execution/view permissions on the fixed
legacy Job. A narrow project-scoped custom
role, `openeconOperationReader`, adds only `run.operations.get`: long-running
operation status belongs to the project's region and cannot inherit the job's
viewer binding. This permits reading a known operation's status across this
project, without operation listing or execution/update permissions from that
role. The control identity does not have job update permissions or runtime
identity delegation. The worker uses
`openecon-compute`. The transfer identity has only the scoped storage roles.

The login interface is publicly reachable; project APIs require verified
Firebase tokens. This replaces the earlier service-wide owner-only IAP gate.
The old single-owner IAP entry point remains available in the source for that
deployment mode. Do not combine multi-user access with the legacy local-console
runtime. Build and inspect the team image before changing the edge gate.

The live web service runs as `openecon-control` with 2 CPU, 4 GiB, request
concurrency 16, a 360-second request timeout, zero minimum and two maximum
instances. Session affinity is disabled: durable state is external. The private
sandbox service uses its separate supervisor image, 2 CPU and 4 GiB, concurrency
1, zero minimum and at most four maximum instances. The retained legacy Job
uses 2 CPU and 4 GiB, a single task and zero retries. Public invocation permits
reaching the login page; it does
not bypass Firebase or project membership checks on the API.

For verification, use disposable identities with
`scripts/verify_team_live.py prepare`, then `verify --url URL`, followed by
`cleanup`. The script never impersonates the real owner and does not send email.
Its ignored credential file is mode 0600 and removed during cleanup. Browser
verification and the real owner's initial registration are distinct checks.

## References

- [Firebase: verify ID tokens](https://firebase.google.com/docs/auth/admin/verify-id-tokens)
- [Firebase: manage sessions and revocation](https://firebase.google.com/docs/auth/admin/manage-sessions)
- [Identity Platform: password policy](https://docs.cloud.google.com/identity-platform/docs/password-policy)
- [Cloud Run: job execution overrides](https://docs.cloud.google.com/run/docs/execute/jobs)
- [Cloud Run: IAM roles](https://docs.cloud.google.com/run/docs/reference/iam/roles)
- [Cloud Storage: signed policy documents](https://docs.cloud.google.com/storage/docs/authentication/signatures)

Local test evidence and live release evidence are recorded separately in
[verification](verification.md) and [security review](team-security-review.md).
