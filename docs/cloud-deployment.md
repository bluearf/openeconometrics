# Private Cloud Run deployment

This document describes the original single-owner IAP edition. The new account,
project and team architecture is documented in [team deployment](team-deployment.md).
Consult the verification record for the currently deployed edition.

Historical private revision: `openecon-00001-5dw`, verified on 1 October 2026.
The existing URL <https://openecon-291739190496.us-central1.run.app> now serves
the [team edition](team-deployment.md), with its own application login. The IAP
configuration below applies only to the legacy single-owner deployment mode.

The web interface and Python/PyTorch worker run in the same container. Sites' server publishing path requires Cloudflare Workers-compatible output; this CPython/PyTorch application uses Cloud Run instead of a nonfunctional static copy.

## Access and scope

This deployment is for one trusted owner, not a public multi-user Python service.
Google Cloud IAP is enabled on the Cloud Run service. The application independently verifies each IAP assertion's ES256 signature, issuer, service audience, lifetime and exact owner email. Unsigned identity headers do not grant access. Session bootstrap and assets are protected as well as analysis routes. `/healthz` is the only unauthenticated in-container probe, returning only `{"status":"ok"}`; the Cloud Run edge still enforces IAP for external requests.

The local `openecon serve` command remains loopback-only. Cloud serving uses a separate entry point, `python -m openecon.cloud`, and requires:

- `OPENECON_PUBLIC_ORIGIN`: the exact HTTPS origin, without a trailing slash.
- `OPENECON_IAP_AUDIENCE`: `/projects/PROJECT_NUMBER/locations/REGION/services/SERVICE_NAME`.
- `OPENECON_OWNER_EMAIL`: one Google account authorized to operate the workspace.
- `OPENECON_WORKSPACE`: defaults to `/tmp/openecon`.
- `PORT`: supplied by Cloud Run, defaults to 8080.

The console can execute arbitrary owner-supplied Python, access its container filesystem and network, and obtain its runtime service account identity through Google metadata. Its process separation is not a security sandbox. Use a dedicated runtime service account without project roles, secrets, keys, database access or unrelated bucket access. Do not add other viewers or broaden IAP access as a substitute for per-user isolated workers.

## Temporary cloud workspace

Uploaded files, draft code, execution history and results reside in the temporary container filesystem. They disappear when the instance is replaced or terminated. Python variables disappear when the worker resets. The interface explicitly labels this and provides existing draft/result downloads. Browser refresh may preserve the current instance's files; it is not a persistence guarantee.

One Uvicorn worker, a maximum of one Cloud Run instance and session affinity reduce routing splits. Cloud Run can still overlap instances during replacement or deployment: this is not a distributed singleton guarantee. The instance session token fails closed if a request reaches a replacement instance; refresh to reconnect. A durable multi-user edition needs external storage and isolated worker/session routing before access can be expanded.

Cloud uploads are limited to 24 MiB to leave room for multipart overhead within Cloud Run's HTTP/1 request limit. The local edition keeps its existing 32 MiB limit. Cloud stdio MCP commands are deliberately not advertised as remote connections; remote MCP is not implemented.

## Image and deployment

`Dockerfile` builds the existing Vite interface and both Python distributions. Production dependencies are exported from `uv.lock` with hashes; Torch is replaced with the same release's CPU-only wheel from the official PyTorch index. The image has no statsmodels, CUDA or Triton dependency, runs as uid 10001 under `tini`, and contains only application code and the synthetic example dataset. `.gcloudignore` and `.dockerignore` use explicit allowlists: the local workspace, credentials, environment files and verification artifacts are excluded.

`deploy/cloudbuild.yaml` builds the Linux amd64 image, then runs an in-image check for nonroot execution, CPU Torch, packaged interface/chart assets and a real 480-observation OLS fit before publishing the image.

The selected dedicated project is `openecon-workbench`, region `us-central1`, service `openecon`; runtime identity is `openecon-runtime@openecon-workbench.iam.gserviceaccount.com`. Image repository: `us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon`. IAM, IAP access policies and the runtime identity belong to this project, separately from other applications. The existing billing account is reused, with usage attributable to the separate project.

```sh
gcloud builds submit . --project=openecon-workbench --region=us-central1 \
  --config=deploy/cloudbuild.yaml --async
```

Deploy only a successful, verified image with IAP and IAM authentication required. Use 2 CPU, 4 GiB RAM, zero minimum instances, maximum one instance, session affinity, one Uvicorn worker, request concurrency 4, and a 300-second request timeout. Concurrency must exceed one so a running code request does not block its own interrupt request. Zero minimum instances permits scale-to-zero and cold starts; it does not make active execution, builds or artifact storage free.

Grant `roles/run.invoker` on this service only to the Google IAP service agent. Grant `roles/iap.httpsResourceAccessor` on this service to the owner. Never grant `allUsers` or `allAuthenticatedUsers`. Verify the service's actual URL, ready revision, IAP setting, IAM policies and anonymous denial after deployment, then verify owner access and an actual Python analysis separately.

## References

- [Google: configure IAP directly on Cloud Run](https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run)
- [Google: validate signed IAP headers](https://docs.cloud.google.com/iap/docs/signed-headers-howto)
- [Google: Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract)
