# OpenEconometrics team edition

The team edition adds an application login, persistent projects and project
membership to the local-first desktop workbench. **Analyses always run on the
member's own computer.** The cloud service is a team sync layer only: accounts,
project membership and roles, shared code and documents, dataset transfer,
account linking and archived results that members share from the desktop app.
It never executes user code, estimators or LaTeX, and it does not parse uploaded
statistical data formats.

The browser team page shows projects, members, drafts, files and shared history.
It has no Run control: opening the project in the desktop app runs the code
locally. Older clients that still call the retired execution routes receive
`CLOUD_EXECUTION_RETIRED`.

The public URL is <https://openecon-291739190496.us-central1.run.app> in the
dedicated `openecon-workbench` project, region `us-central1`. The deployed
revision is not recorded here: read `source_commit` from
`/api/auth/config`. Build, deployment, verification and rollback are in
[team sync backend deployment](sync-backend-deployment.md). The first owner
signs in through the application login screen.

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
their existing method. Project data is never merged by email. Adding a sign-in
method to the same UID uses the authenticated
[account-linking flow](account-linking.md) and Firebase's linking API.
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

| Role | Read and download | Edit, upload and share results | Manage membership |
| --- | --- | --- | --- |
| Owner | Yes | Yes | Yes |
| Editor | Yes | Yes | No |
| Viewer | Yes | No | No |

Every member runs analyses on their own computer in the desktop app; a role
controls what may be synced, not where code runs.

The original project owner cannot be removed or demoted. Ownership transfer,
account administration, MFA enrollment, live simultaneous text editing and a
remote agent protocol are not part of this release. A shared `analysis.py`
draft uses version checks: stale writes fail instead of overwriting another
member's work. The browser preserves the conflicting local draft for download.

## Persistence boundary

Firestore stores accounts, live project memberships, invitations, versioned
drafts, audit events and result metadata. A private Cloud Storage bucket stores
immutable uploaded files and archived result records. Direct client Firestore
access is explicitly denied by rules. Project IDs and roles supplied by a client
are never an authorization source: every request checks current membership, and
writes check it inside their transaction.

A desktop result is untrusted data. The service verifies its size, result
schema, plot geometry and output order before writing an immutable record; it
does not rerun or certify the calculation. Membership is checked again before
publishing a result and before returning downloaded data. An editor can still
deliberately fabricate data or results within their own project.

Results and generated files created by the retired cloud execution backends
remain readable. A historical cloud run that never finished is shown as stopped;
its stored data is not rewritten. See
[saved records](sync-backend-deployment.md#retired-execution-routes-and-saved-records).

## Limits and recovery

- Upload: 24 MiB per direct upload; 20 files and 64 MiB of directly uploaded
  inputs per project. Larger datasets use [chunked transfers](team-data-transfers.md).
- Code and documents: 64,000 characters per draft or named file.
- Shared results: 3 MiB per archived desktop record.
- Recent history: up to 50 records within an 8 MiB response budget; older
  results are available through paged history.
- Project files and results have no automatic deletion policy. Existing bucket
  soft-delete settings also apply. Durable storage is distinct from a tested
  backup/restore plan. The [project recovery runbook](team-recovery.md) defines
  scoped backup/restore, RPO/RTO targets and a synthetic independent-environment
  drill; live cloud recovery and operational scheduling remain separate checks.

## Production wiring

`python -m openecon.cloud` serves the team sync backend only when
`OPENECON_MODE=teams`; any other mode or missing configuration fails closed.
The required variables, service identity and resources are listed in
[production configuration](sync-backend-deployment.md#production-configuration).

The login interface is publicly reachable; project APIs require verified
Firebase tokens. Public invocation permits reaching the login page; it does not
bypass Firebase or project membership checks on the API. Session affinity is
disabled: durable state is external.

For verification, use disposable identities with
`scripts/verify_team_live.py prepare`, then `verify --url URL`. The script never
impersonates the real owner, does not send email and never starts a
computation. Automatic cleanup is refused; `inspect-cleanup` is read-only.
Browser verification and the real owner's initial registration are distinct
checks.

## References

- [Firebase: verify ID tokens](https://firebase.google.com/docs/auth/admin/verify-id-tokens)
- [Firebase: manage sessions and revocation](https://firebase.google.com/docs/auth/admin/manage-sessions)
- [Identity Platform: password policy](https://docs.cloud.google.com/identity-platform/docs/password-policy)
- [Cloud Run: IAM roles](https://docs.cloud.google.com/run/docs/reference/iam/roles)

Local test evidence and historical release evidence are recorded separately in
[verification](verification.md) and [security review](team-security-review.md).
Entries there that describe sandbox or Job execution are historical records of
the retired backends.
