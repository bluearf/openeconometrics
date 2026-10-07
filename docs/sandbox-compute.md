# Private sandbox compute service

The publication-table update on 2 October 2026 runs production control
revision **`openecon-00009-28v`** against pinned private broker
**`openecon-sandbox-00003-qf7`**. Its full five-user production suite passed
native paper tables, a two-model comparison, source-value checks and shared
CSV/TeX files in 13.0 seconds. Both active services keep minimum instances at
zero. The previous v8 broker `openecon-sandbox-latex-00001-t7l` and image remain
unchanged for accepted work and explicit rollback. [Publication verification](verification.md)
records current immutable images, readiness, traffic and readback. The security
architecture is unchanged; the initial isolation evidence below remains valid.

## Initial sandbox isolation evidence

The initial managed runtime proof passed on 2 October 2026, including actual
watchdog death of the parent and child, and all 13 actual broker integration
cases passed with every staging scope removed. Initial supervisor image
`sha256:96123803caec5a46cc71f8b39277d5128b39919e614cee5d9ff186056ced73fe`
was built by `b757743b-6758-499d-837e-28b983c3a729`; initial nonroot control image
`sha256:56f1cda7f05e8f97fc2d18d1041c4de5728b8fcb5e87daedfe0934e10c076d57`
was built by `f3070a0a-87c6-4f05-abb7-bf0718f40261`. Initial private revision
`openecon-sandbox-00002-7dc` and control `openecon-00007-p7s` passed complete
five-user preview and production suites; production analysis and durable outputs
took 11.9 seconds. The later v8 LaTeX rollout passed these application flows in
12.4 seconds. These are full analysis/file integration timings, not measured
cold/warm browser opening. The operator credentials were renewed during the
initial rollout, not as a table-formatting requirement.

The observed slow opening is computation provisioning: a previous successful
Job waited about 187.5 seconds before its actual task started; active task time
was about 19.4 seconds. The latest completed Job waited 203.925 seconds, then
ran for 18.094 seconds, based on task timestamps. A Job cancelled after about
312 seconds had no task start time. The 480-row example's local fresh-console
median was about 0.8 seconds,
which is local evidence only. Earlier page-startup improvements did not shorten
Job provisioning.

This path replaces Cloud Run Job scheduling with a private Cloud Run service.
Diagnostic runtime, actual broker transport, authenticated preview workbench,
production deployment and production team QA have passed. Cloud Run Sandboxes
is a Preview feature. The control and legacy Job entry points remain
nonroot; the trusted launcher uses its separate reviewed supervisor image.

## Fixed deployment contract

The trusted supervisor runs as host UID0 under Tini because the managed launcher
requires it to establish the network namespace. Its guest has platform-fixed
UID/GID `0:0`, without SETUID/SETGID. Before any input is consumed, the worker
removes all effective/permitted/inheritable/ambient capabilities and permanently
sets `NoNewPrivs`. This guest identity is not the host root identity. The public
control remains UID/GID `10001:10001`. Use a roleless compute service account
with no project, bucket, Firestore, signing, or service-account impersonation grants.
Only `openecon-control@PROJECT.iam.gserviceaccount.com` may invoke this service.
The control plane supplies the same complete Google ID token in both
`X-Serverless-Authorization` and `Authorization`. Cloud Run IAM validates the
first; the broker validates the full signature, issuer, audience, email, unique
subject and expiry from the second using Google's public certificates. This
second gate is mandatory because guest gateway traffic can reach the parent
listener without passing through the Cloud Run IAM edge. Header presence and
namespace networking alone are not authorization.

The service verifier requires a digest-pinned image, one container, generation 2,
concurrency 1, min instances 0, max instances at most 4, 2 CPU, 4 GiB RAM,
360-second request timeout, and the sandbox launcher enabled. CPU is allocated
while requests execute; no detached background analysis may continue afterward.
There are no volume mounts, secrets, VPC attachment, or user-supplied command flags.
The configured command is exactly:

```text
/usr/bin/tini -- /opt/venv/bin/python -m openecon.team_sandbox_broker
```

Only these six deployment environment entries are accepted by the verifier:

```text
OPENECON_MODE=sandbox-broker
OPENECON_SANDBOX_ROOTFS=/opt/openecon-sandbox-rootfs
OPENECON_SANDBOX_ENABLED=1
OPENECON_BROKER_AUDIENCE=https://openecon-sandbox-291739190496.us-central1.run.app
OPENECON_BROKER_CALLER_EMAIL=openecon-control@openecon-workbench.iam.gserviceaccount.com
OPENECON_BROKER_CALLER_SUB=CONTROL_SERVICE_ACCOUNT_UNIQUE_ID
```

Cloud Run's current enabling flag is `--sandbox-launcher` on
`gcloud beta run deploy`; the REST container field is `sandboxLauncher: true`.
Deployment and IAM changes must go through the project's reviewed deploy helper.
See [sandbox service configuration](https://docs.cloud.google.com/run/docs/configuring/services/sandboxes).

## Broker and child boundary

`POST /execute` accepts at most 65,536 bytes of strict JSON, with exactly:

```json
{
  "execution_id": "RUN_HEX32",
  "input_url": "SIGNED_GENERATION_PINNED_GCS_URL",
  "cancel_url": "SIGNED_UNPINNED_GCS_URL",
  "completion_upload": {"url": "GCS_BUCKET_URL", "fields": {}},
  "timeout_seconds": 120
}
```

The example omits the required signed POST fields. Validation binds all three
capabilities to one bucket and `staging/PROJECT_HEX32/RUN_HEX32/`, with fixed
`input.json`, `cancel.json`, and `completion.json` object names. The completion
policy permits exactly 1–4,096 bytes of JSON. The input is generation-pinned;
the cancellation object is deliberately unpinned so it can be created later.
The outer control-store operation name remains `sandbox:PROJECT_HEX32:RUN_HEX32`.

For every request the broker generates a new random name and executes only:

```text
/usr/local/gcp/bin/sandbox run oe-RANDOM_HEX32 \
  --rootfs /opt/openecon-sandbox-rootfs \
  --write --allow-egress --workdir /tmp -- \
  /opt/venv/bin/python -I -m openecon.team_sandbox_worker
```

Only the input URL and a newline travel through stdin. No caller header,
cancellation URL, completion policy, or broker environment reaches the worker.
The parent discards launcher stdout/stderr, never deserializes Python objects,
and never imports or executes submitted code. The worker permanently removes
active capabilities, prevents privilege gains, clears its environment, then uses the existing
`team_job` adapter inside the sandbox. Its internal ConsoleSession pickle channel
is entirely inside this isolated, untrusted boundary.

The Docker build creates a curated root filesystem containing Python, system
runtime libraries, analysis dependencies, and OpenEconometrics. It excludes the parent's
`/tmp`, `/run`, `/home`, `/app`, sandbox launcher, and cloud administration modules.
Runtime files and directories are root-owned/read-only; `/tmp` is initially empty
with mode 1777 so either launcher UID mode can use the private writable overlay.
The broker never writes below this root and validates the empty lower temporary
directories at startup. No bind mounts, inherited private host directory, reused
overlay, import tarball, or sandbox reuse is permitted. Torch is hardlinked into
the root filesystem in the same image layer to avoid storing its payload twice.

The published [CLI reference](https://docs.cloud.google.com/run/docs/reference/sandbox-cli)
defines `run`, `--rootfs`, `--write`, and explicit deletion. Google's
[ADK integration](https://github.com/google/adk-python/blob/main/src/google/adk/integrations/cloud_run/_cloud_run_sandbox_code_executor.py)
demonstrates stdin forwarding for `sandbox do`; the live probe must independently
verify stdin forwarding and lifecycle semantics for the named `run` command used
here. The fixed-code runtime proof established these platform mechanics; the
complete actual broker transport also passed its separate integration suite.
There is no local subprocess
or job fallback within this broker.

## Completion, disconnects, and hard lifetime bound

The HTTP request remains open with bounded newline-delimited JSON heartbeats.
Each frame contains the run identifier and a status; each is less than 4,096 bytes.
Frames are informational. Only the separately signed parent-written completion
object can release the control plane's execution lease:

```json
{"execution_id":"RUN_HEX32","status":"succeeded","cleanup_confirmed":true}
```

The other allowed statuses are `failed` and `cancelled`. A completed worker may
still have produced an analysis error record; that record is validated separately.
The broker polls cancellation every loop (one second between worker wait polls)
and aborts on unavailable or malformed cancellation state. Before any completion
marker is uploaded, it runs:

```text
/usr/local/gcp/bin/sandbox delete oe-RANDOM_HEX32 --force
```

Only a successful delete followed by reaping the launcher confirms cleanup.
Deleting the whole sandbox also removes descendants outside the original process
group. A failed or hanging deletion poisons the host and immediately terminates
the broker. No completion marker is written and no new execution is accepted.
Client disconnects cancel work and enter shielded, bounded deletion before the
ASGI request finishes; they do not leave background analysis running.

A kernel `ITIMER_REAL` alarm with the default fatal `SIGALRM` action is armed at
request acceptance. After parsing, its absolute deadline is acceptance time plus
`ceil(timeout_seconds) + 120`, at most 240 seconds. It is not a Python watchdog
thread or a Python signal handler. The run loop stops 30 seconds earlier, reserving
15 seconds for deletion, 5 for launcher reaping, and 5 for marker upload. The alarm
also bounds body download, subprocess creation, stdin drain, network operations,
and stalled cleanup. It is disarmed only after cleanup or a request that never
attempted a launch. The 360-second Cloud Run request timeout is not relied upon to
kill processes. Unknown completion storage outcomes leave the lease pending.

Python must run beneath Tini rather than as PID 1; Linux PID 1 can ignore
default-fatal signals. The service configuration verifies this entrypoint and
the broker rejects PID 1. The fixed managed alarm proof established parent
instance replacement and absence of a delayed child beacon after teardown, with
a working positive beacon and observation request keeping the replacement active.
The managed runtime proof also verified disconnect cleanup; the actual broker
suite separately passed cancellation and Python timeout cases.

Control wiring uses `OPENECON_RUNNER=sandbox` with explicit private service,
origin and image digest. Its migration adapter sends new work to the sandbox
service while keeping previously accepted Job operations pollable and stoppable.
For rollback, keep all sandbox service/origin/image settings and set only
`OPENECON_RUNNER=jobs` on the candidate image. New work then uses Jobs while
existing sandbox handles retain their own status/cancellation reader. Returning
traffic to the old image is safe only after all accepted sandbox runs have drained
with confirmed cleanup; that old image does not understand sandbox handles.
Project concurrency, daily quotas and the 0.05–120-second Python limit remain.
Minimum instances remain zero; no permanently warm instance cost was enabled.

## Control deployment and rollback commands

`scripts/deploy_sandbox_control.py` uses the explicit OpenEconometrics project and
reviewed immutable images. It reads the current public service in memory,
preserves its existing environment and control identity/resources, and checks
the private service configuration and IAM before a sandbox deployment. It does
not export private configuration. Without `--apply`, it prints only a safe plan
for the fixed disposable preview service. `--production` explicitly selects the
live control service. The reviewed image pair on 2 October 2026 is:

```sh
openecon_control_image='us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:7fc9e78250dc238712e72920b51be752772ba34cf7a38aa14e7c9ffe9a8a6106'
openecon_compute_image='us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:4304cc205d36dc8a71d575a86c618a4ab86378523d72432ddfbcc57a98a98a70'
.venv/bin/python scripts/deploy_sandbox_control.py \
  --image "$openecon_control_image" --compute-image "$openecon_compute_image"
```

After reviewing the plan, append `--apply` to create/update the fixed preview.
For the reviewed production cutover, use the same image pair with
`--production --apply`. A rollback that starts new work on Jobs uses the candidate
control image, preserving accepted sandbox handle support:

```sh
.venv/bin/python scripts/deploy_sandbox_control.py \
  --image "$openecon_control_image" --compute-image "$openecon_compute_image" \
  --production --backend jobs --apply
```

This changes only which backend accepts new work; the migration adapter continues
reading and cancelling previously accepted handles from both backends. The helper
does not require a healthy private service for this Jobs rollback, and never
modifies the private compute service.

For a new compute image, both helpers also accept
`--compute-service openecon-sandbox-latex`. This is the only additional allowed
private broker. Deploying it separately preserves the previous broker and the
public control's exact image verification while the candidate is tested. Pass
the same service selector to the preview and production control deployments;
their origin, token audience and image verifier remain bound to that selected
service. Minimum instances remain zero. Accepted runs retain their scoped
storage read/cancel path; keep the old private service available for in-flight
work and explicit rollback.

## Live evidence required for deployment

The private Cloud Run probe must verify named-run stdin forwarding; CPU Torch and
OpenEconometrics execution under the exact curated rootfs and namespace-local standard
devices such as `/dev/urandom` and `/dev/null`; enforced managed guest capability
reduction and `NoNewPrivs`; no readable
parent environment, process, temporary file or capability; no cross-sandbox files;
metadata denial by both IP and DNS **with outbound GCS access enabled**; forced
deletion of hostile descendants; and cleanup on timeout and client disconnect.
It must also prove fatal watchdog/container exit removes the nested sandbox, even
after the request disconnects and request-based CPU would otherwise become idle.
The worker's bounded read of the non-secret metadata instance-ID endpoint is only
an additional guard, not proof that every metadata path is blocked.

The [Cloud Run code execution guide](https://docs.cloud.google.com/run/docs/code-execution)
describes isolation of environment, filesystem, and metadata. Our use of egress,
a custom rootfs, managed guest privilege reduction, and explicit cleanup must be
proven together in the deployment probe before treating this architecture as
operationally ready.

## Temporary integration QA access and cleanup

Direct private-broker QA uses the exact control service account ID token and
transfer service account signing. The human operator's existing credentials
could not mint either; narrowly scoped temporary grants were explicitly prepared.
The control account grants only OIDC ID-token creation, and the transfer account
uses a custom role containing only `iam.serviceAccounts.signBlob`. Both bindings
have unique conditions expiring within two hours, with ignored owner-only state.
No service account key or broad TokenCreator grant is used. `generateIdToken`
is called directly, since generic CLI impersonation also requires an access-token
permission that is intentionally absent.

After all QA requests and child cleanup have completed, the cleanup commands are:

```sh
.venv/bin/python scripts/sandbox_qa_identity.py --revoke
.venv/bin/python scripts/sandbox_qa_signer.py --revoke
```

These helpers remove only their owned bindings using fresh policy etags and
verified readback. The signer deletes its custom role only if it created that
role and its ownership is unchanged; preexisting roles are kept. Both temporary
QA bindings have been revoked, with owned-binding absence and completed cleanup
verified by readback. The signer helper also removed its owned custom role.
Disposable test-project cleanup must also refuse active or unresolved runs.
A completed cleanup removed the three owned QA projects and five identities and
removed the private QA credential state file.
A final readback confirmed that the owned diagnostic and team preview services
were absent. `artifacts/verification/sandbox-cleanup-final.json` records removal
of both services, absence of private QA credential state and temporary IAM
bindings, removal of the owned signer role, and zero pending runtime/broker
cleanup scopes. The production control and private compute services remain.
A custom role deletion is soft deletion, so complete the signer tests under one
grant before removing that role.

The immediate v8 presentation rollback uses the unchanged alternate broker;
review this plan before appending `--apply`:

```bash
uv run python scripts/deploy_sandbox_control.py --production \
  --compute-service openecon-sandbox-latex \
  --image us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:821adad15684c387a1a3a0659a340d0f062b3e66bb7ae2b33c8b8d002200ad6a \
  --compute-image us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:fd5163cc7cfe1c1d31a5199b5aa4c77a2d25cec66da4bddb1e0a2937ea9683be
```

This preserves sandbox handle reading and cancellation on the presentation
candidate; do not restore a pre-sandbox image while accepted sandbox work remains.
