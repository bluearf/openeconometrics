# Publishing reviewed packages to PyPI

The manual `Publish reviewed packages to PyPI` workflow publishes the existing
public Python release `python-v0.3.18a4`: `openecon-charts==0.3.0a2` followed by
`openecon==0.3.18a4`. The SDK requires that exact charts version. SDK a4 changes
only its package README, documentation and version metadata from reviewed a3.
Charts a2 is the exact already-published pair originally built from
`ca7854593e2591f56c380dc19f7abeba0b7b0117`; it is downloaded and hash-checked,
not rebuilt. The Mac release `v0.3.43-alpha.1` still embeds SDK `0.3.18a1`; this
Python publication does not rebuild the application or import newer private
implementations. Existing PyPI publishers and the `pypi` environment stay in use.

## Account and publisher setup

Use a verified PyPI account with two-factor authentication. For the first setup,
register the charts pending publisher, publish charts, then register the SDK
pending publisher. PyPI rejects two pending projects with the same publisher
configuration; once charts has its normal publisher, the SDK can use the same
workflow and environment. Use these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `openecon-charts` or `openecon`, respectively |
| GitHub owner | `bluearf` |
| GitHub repository | `openeconometrics` |
| Workflow filename | `publish-pypi.yml` |
| GitHub environment | `pypi` |

Create the `pypi` environment in this repository and restrict deployment to the
protected `main` branch. Choose any required reviewers under the project's
release policy; for this single-maintainer setup, select `anilsen13` and permit
self-review. The workflow itself also refuses publication from other
repositories or branches. Pending publishers become normal publishers after the first upload.
No permanent PyPI password or API token belongs in GitHub secrets.

References: [pending publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Creating a pending publisher alone does not reserve an unused project name.

## Initial publication

1. Merge the reviewed publishing workflow into `main`. Confirm the GitHub
   `pypi` environment and charts pending publisher match the table above.
   Bootstrap charts with `publish=true` and `publish_sdk=false`, approve the
   environment, and verify charts ownership and hashes. Then add the SDK pending
   publisher with the same configuration. This first charts-only run deliberately
   skips SDK publication.
2. The PR runs verification without publishing. A manual run on `main` defaults
   to `publish=false` and also performs verification only. After the account,
   pending publishers, and environment are ready, manually run
   `Publish reviewed packages to PyPI` on `main` with `publish=true` and
   `publish_sdk=true` and approve
   the environment's release review.
3. The verification job downloads the four fixed public distribution files and
   release provenance. It verifies hardcoded SHA256, derivative source/base/patch
   identities, successful Linux/Mac package and sandboxed browser export receipts, package versions,
   SDK dependency, license notices, excluded
   source material, and metadata using Twine 7.0.0. It builds no new packages.
4. Separate jobs publish charts first, then the SDK, through short-lived GitHub
   OIDC identity. Only those jobs have `id-token: write`. Distribution hashes are
   checked again after artifact transfer. PyPI attestations identify the
   publishing workflow; original build provenance remains the release's
   `RELEASE-PROVENANCE.json`.
5. Each job reads back the exact non-yanked filenames and SHA256 values from
   PyPI. Every request includes a distinct run/attempt/phase/poll query and
   no-cache headers, so a previous negative CDN response cannot be reused by a
   later readback. Only an HTTP 404 or an incomplete matching file set is retried,
   for at most 12 polls. SDK publication starts only after charts readback succeeds.
6. Confirm project ownership and publisher settings in PyPI. Install from PyPI
   in a fresh environment and run the fit, JSON persistence, LaTeX, and chart
   checks before recording end-to-end publication as complete.

For example, in a clean Python 3.11–3.14 environment:

```sh
python -m pip install 'openecon==0.3.18a4' 'openecon-charts==0.3.0a2'
python -m pip check
```

The version is an alpha; an explicit pin or `--pre` is needed when selecting
prereleases. CPU installation and testing remain independent of physical CUDA
acceptance. Application extras are described in [distribution.md](distribution.md).

Safe reruns compare any already uploaded file against the exact expected
SHA256 and omit only matching files from the next upload. A different file,
unexpected filename, or yanked existing release stops publication. The action's
`skip-existing` option is disabled, so a race or unreviewed collision fails
instead of being silently accepted.

## Updating to a future release

Each package version is immutable. Never replace the bytes of an already
published version, rebuild private current main using an old version number,
or retarget a fixed release after approval.

1. Finish the scoped release acceptance checks for the new source commit.
   Select new versions only for changed packages. Reuse the exact existing
   distributions for an unchanged charts version and preserve the SDK's pin.
2. Prepare the public snapshot from the reviewed HEAD in an isolated private
   source checkout using `scripts/prepare_public_source.py`. Preserve the
   private development checkout and history. Review the snapshot manifest,
   secret scan, license notices, and exclusion of external research fixture
   bytes and internal evidence. Public snapshot generation excludes `.github`;
   maintain this workflow separately in the public repository.
3. Publish the reviewed snapshot and GitHub release through the normal release
   process. Build and verify distributions from that snapshot, including
   metadata, clean installation, and scoped scientific/package checks. Preserve
   source commit, public snapshot commit, and distribution hashes in release
   provenance. Publishing a package is distinct from desktop signing,
   notarization, vendor parity, and CUDA acceptance.
4. Open a separate PR updating **all** fixed URLs, provenance commit/version
   checks, distribution filenames/hashes, artifact selections, publishing-job
   version constants, environment URLs, and this guide. Pin and review action
   revisions when upgrading them. Do not replace the fixed URL with `latest`,
   mutable input text, or an unreviewed private-repository build.
5. Check workflow syntax with `actionlint`, replay its download/verification
   script against the proposed public release, and run Twine's strict metadata
   check. Review the PR before merging and manually dispatching the new release.

The fixed SDK a4 files were built from public source
`03add0e10867885ed10cf426c6b6dccbd00dedee`, merged as
`090b3656c95c25b2f707516b5868bed529396763`. Its documentation-only source ancestor
is `aad8962af4c5d26bfb8ad0ff2c43f35bc8542eed`. The six previously reviewed scoped
private patches remain inherited; no new implementation is imported. Charts a2
retains its published hashes and original `ca7854593e2591f56c380dc19f7abeba0b7b0117`
build source. New SDK a4 Mac installs and existing charts a2 registry/Chrome
receipts retain their actual versions and origins in release provenance.
Linux Python 3.11/3.13 acceptance runs from the new SDK source with the exact
existing charts pair: 219 parser/OLS cases, 18 sandboxed export cases, zero skips,
and two SDK plus two standalone charts wheel/sdist installations per job.
See the [fixed SDK a4 release and provenance](https://github.com/bluearf/openeconometrics/releases/tag/python-v0.3.18a4)
and the [original charts a2 build](https://github.com/bluearf/openeconometrics/releases/tag/python-v0.3.18a3).

Later publisher-only workflow/documentation commits do not change either fixed
package's bytes or their release provenance. The current repository's
`SOURCE-MANIFEST.json` tracks current listed source/documentation files;
`RELEASE-PROVENANCE.json` separately pins the manifest and source files as they
were at the SDK build commit. A verified GitHub release and merged publishing
workflow alone do not prove successful PyPI upload or a fresh registry installation;
complete steps 5–6 before recording those delivery layers as passed.
