# Team sync backend deployment

OpenEconometrics is a local-first desktop application for macOS and Windows.
Every analysis runs on the user's computer. The cloud service is only a team
sync layer, comparable to a shared project store: it does not execute user
code, estimators or LaTeX.

On 10 October 2026 the cloud execution backends were removed from the source:
the private sandbox broker services, the Cloud Run Jobs worker, the compute
image and the single-owner IAP cloud workbench. The control service described
here is the only cloud component that remains.

## What the service does

| Responsibility | Routes and storage |
| --- | --- |
| Accounts and sign-in | Firebase Authentication / Identity Platform ID tokens; desktop login handoff (`/api/desktop/login`) and account linking (`/api/desktop/account-link`) |
| Projects, membership and roles | `oe_projects`, `oe_users`, `oe_invitations` in Firestore |
| Code and documents | Versioned drafts, named scripts, file layout and package manifests |
| Datasets | Small uploads and chunked transfers (`/workspace/transfers`) in the private project bucket |
| Shared results | Desktop-computed records published to `/workspace/desktop/results`, validated as data and archived without rerunning |
| History | `/workspace/console`, `/workspace/console/history` and `/workspace/runs/{id}/record` read saved results |

The desktop app computes results locally and uploads only the saved record.
The server checks project membership, the record schema, plot geometry, output
order and size limits; it never certifies the calculation.

## Retired execution routes and saved records

`POST /workspace/console/execute`, `/console/interrupt` and `/console/reset`
remain registered only to refuse older clients. After the normal bearer-token
and membership checks they return HTTP 410 with code
`CLOUD_EXECUTION_RETIRED` and a message pointing to the desktop app. They do
not read the request body, create a run, reserve capacity or write data.

Saved data is never deleted or migrated:

- Results produced by the retired cloud backends remain readable from history,
  and their generated files remain downloadable from
  `/workspace/runs/{id}/files/{index}`.
- A cloud run that never reached a terminal state is shown as stopped
  (`interrupted`). Its stored run document and the project's `active_run`
  marker are left unchanged; neither reports a running analysis.
- The project recovery tool still refuses to back up a project whose stored
  `active_run` marker is set. Inspect such projects before a recovery drill; the
  service does not rewrite them.

`/api/auth/config` reports `cloud_execution_available: false`.

## Build

`deploy/cloudbuild-control.yaml` is the only build configuration. It builds the
nonroot control image, then runs an in-image gate before publishing:

- the image is bound to a full source commit ID (`OPENECON_SOURCE_COMMIT`);
- no retired execution module or sandbox root filesystem is present;
- packaged interface and chart assets are present and Torch is CPU-only;
- the desktop result schema gate validates a real native OLS result, its
  publication LaTeX and interleaved output order through `DesktopResult` and
  `validate_worker_result`. This reference model is built inside the gate only;
  the service never runs user code.

Triggered builds take the commit from `COMMIT_SHA`. A manual build from a clean
checkout of the reviewed commit must pass it explicitly; the gate fails without
one:

```sh
gcloud builds submit . --project=openecon-workbench --region=us-central1 \
  --config=deploy/cloudbuild-control.yaml \
  --substitutions=_SOURCE_COMMIT="$(git rev-parse HEAD)"
```

Use the image digest reported by the successful build, never a tag.

## Deploy

`scripts/deploy_team_control.py` is the only deployment helper. It accepts one
immutable image digest and touches only the fixed control services:

```sh
image='us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:<digest>'
.venv/bin/python scripts/deploy_team_control.py --image "$image"            # read-only plan
.venv/bin/python scripts/deploy_team_control.py --image "$image" --apply    # disposable preview
.venv/bin/python scripts/deploy_team_control.py --image "$image" --production --apply
```

Without `--apply` it prints a safe summary for the preview service
`openecon-teams-preview`. `--apply` creates or updates that preview and grants
public invocation so the login page is reachable; Firebase and project
membership still protect every API. `--production --apply` updates the live
service `openecon`.

The helper reads the live service in memory and refuses to continue if its
identity, ingress, scaling, resources, volumes or required sync settings differ
from the reviewed baseline. It preserves every other environment entry,
including secret references, and never exports them. It removes the retired
compute settings (`OPENECON_RUNNER`, `OPENECON_COMPUTE_*`,
`OPENECON_BROKER_CALLER_SUB`) and any service-level `OPENECON_SOURCE_COMMIT`, so
the reported commit always comes from the image. The first applied run for each
target saves a mode-0600 summary under `artifacts/verification/` with the
previous image and ready revision.

To roll back, deploy the previous sync-only control digest with the same
command. An image built before the execution backends were removed expects
compute settings and services that are no longer configured; do not restore it.

## Verify the live revision without gcloud

```sh
curl -s https://openecon-291739190496.us-central1.run.app/api/auth/config
```

The public response contains `source_commit`, `cloud_execution_available`,
`desktop_login_available`, `account_link_available` and
`dataset_transfer_available`. A missing or malformed build commit is reported as
`null`. For disposable end-to-end checks, `scripts/verify_team_live.py verify
--url URL --expect-commit COMMIT` uses prepared QA identities to check sign-in,
roles, drafts, uploads, the execution refusal and membership changes, and
`scripts/verify_desktop_cloud_live.py --url URL` checks desktop login and
result archiving. Neither starts a computation.

## Production configuration

`python -m openecon.cloud` serves the sync backend only when
`OPENECON_MODE=teams`; any other mode, or missing configuration, fails closed.

| Variable | Purpose |
| --- | --- |
| `OPENECON_MODE` | Must be `teams` |
| `OPENECON_PROJECT_ID` | Firebase/Firestore/Cloud Run project |
| `OPENECON_PUBLIC_ORIGIN` | Exact HTTPS origin of the interface |
| `OPENECON_OWNER_EMAIL` | Verified initial account bootstrap |
| `OPENECON_FIREBASE_CONFIG` | Public Firebase web app configuration JSON |
| `OPENECON_BUCKET` | Private project file bucket |
| `OPENECON_SIGNER_EMAIL` | `signBlob` identity for short-lived desktop sign-in tokens |
| `OPENECON_REGION` | Region, default `us-central1` |
| `OPENECON_SOURCE_COMMIT` | Set by the image build; read-only |

The service runs as `openecon-control` with Firestore access, object
management on the project bucket, read-only Auth user lookup and `signBlob` on
`openecon-transfer`. It needs no Cloud Run invoker, Jobs or operation-reader
permission. Resources stay at 2 CPU, 4 GiB, request concurrency 16, a 360-second
request timeout, zero minimum and two maximum instances.

## Retired cloud resources

After the sync-only revision is live and verified, the private compute services
(`openecon-sandbox`, `openecon-sandbox-latex`), the legacy Cloud Run Job
`openecon-compute`, the compute service account, the compute-only IAM bindings
and custom roles, and compute images in Artifact Registry are no longer used.
Remove them only after the owner's explicit approval; the source never deletes
cloud resources or stored user data.
