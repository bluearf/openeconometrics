# Distribution status and evidence · 11 October 2026

This is a dated reconciliation for
[MARKET-833](https://linear.app/bluearf/issue/MARKET-833), checked at
**15:44 UTC / 18:44 Europe/Istanbul on 11 October 2026**. The
[machine-readable receipt](distribution-status-2026-10-11.json) records anonymous
HTTP downloads, complete streamed SHA-256/byte checks and authenticated private
workflow metadata separately. It does not run a new installer, publish a release
or certify current scientific coverage. Later delivery must add its own dated
source, run and artifact evidence rather than silently inheriting this snapshot.

## Source, published packages and installed applications

The inspected development source is
`6eb04e2b9824ac230d14a034961d9b91521e98d1`: desktop **0.3.46**, SDK
**0.3.20a1**, charts **0.3.2a1**. These are source candidate versions; none has
replaced the public artifacts below at this readback. The generated
[capability catalogue](../capabilities.md) describes development source, not the
contents of an older binary or registry wheel.

| Public delivery | Exact build source and public anchor | Producer and accepted layer | Publication / anonymous readback |
| --- | --- | --- | --- |
| Mac arm64 0.3.43; bundled SDK 0.3.18a1/charts 0.3.0a1 | Build `6cd9079a330b03a4f1a568e28aca04a23a8d897b`; public source `7cb7517bf8bacf91f92c85f8df9a2904e76e8974` | Local Mac packaging; no hosted producer run is recorded in the public provenance. Historical exact-DMG native OLS/LaTeX smoke, source/bundle identities and bounded saved-result checks. Actual execution macOS 26.6.2; macOS 15 is the audited binary minimum, not a claimed execution environment. | [Release](https://github.com/bluearf/openeconometrics/releases/tag/v0.3.43-alpha.1), published 7 October 05:31:57 UTC; anonymous DMG/provenance bytes rechecked 11 October. |
| Windows x64 CPU 0.3.45; bundled SDK 0.3.19a1/charts 0.3.1a1 | Build `f1d2eb629db815b716a40e32b07ad55f5d5b5088`; public tag `1379566c7d09ab29ad49efd14dbfbbabf680e936` is a separate source anchor | [Producer 37921893600](https://github.com/bluearf/openecon/actions/runs/37921893600), attempt 1, success 9 October 11:42:37 UTC. GitHub-hosted Windows 2025 x64 installed/native/offline/upgrade/rollback/cold-state acceptance: 18 bounded checks. | [Release](https://github.com/bluearf/openeconometrics/releases/tag/v0.3.45-windows-alpha.1), published 9 October 12:06:36 UTC; anonymous installer/provenance bytes rechecked 11 October. |
| PyPI SDK 0.3.19a1 and charts 0.3.1a1 | Package source `a8eb8799ca46e00ebedd362ad6c69461a172a8ca`; public release anchor `2060fb8ef4ea9c6e8a4371cef3de42c63d9f2b31` | [Fixed-pair producer 37894416408](https://github.com/bluearf/openeconometrics/actions/runs/37894416408), both Linux Python 3.11/3.13; separate macOS installed-package receipts. [Protected-main PyPI publication 37907165716](https://github.com/bluearf/openeconometrics/actions/runs/37907165716) and post-publication registry installation receipt. | [Python release](https://github.com/bluearf/openeconometrics/releases/tag/python-v0.3.19a1), 9 October 08:40:24 UTC; charts uploaded 08:51 UTC, SDK 08:54 UTC. All four current PyPI files anonymously downloaded and hashed 11 October. |

The Windows bundled SDK and the PyPI SDK share a version label, but have different
build source pins. The Windows public tag's automatic source archive is not its
private installer build source. The Python pair's accepted source bridge is
documented by its release provenance; it does not cover later source methods.
Mac 0.3.43 release notes saying “Neither package is published to PyPI” describe
its original 0.3.18a1/0.3.0a1 pair and 7 October release. The later independent
Python release above supersedes that historical package-delivery status without
changing the Mac bundle or rewriting its original receipt.

## Exact artifacts

The following files were downloaded without credentials and their **complete
bytes**, rather than metadata alone, matched the advertised SHA-256 and size.
Original URLs and per-file timestamps are in the JSON receipt.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [Mac DMG 0.3.43](https://github.com/bluearf/openeconometrics/releases/download/v0.3.43-alpha.1/OpenEconometrics_0.3.43_aarch64.dmg) | 317,969,037 | `3668701f4e7461faad743ffd20979af91e5fac1a8e8e54f62ad8043d37fde801` |
| [Windows installer 0.3.45](https://github.com/bluearf/openeconometrics/releases/download/v0.3.45-windows-alpha.1/OpenEconometrics_0.3.45_x64-setup.exe) | 192,720,594 | `ccaf15ea9472b89c19f292cff62d94dfa6c92d7f639388d7fc6bca490766515c` |
| [SDK 0.3.19a1 wheel](https://pypi.org/project/openecon/0.3.19a1/#files) | 5,669,381 | `09377fe0220c4c35ff27289236fa29f63666684ba72edb4de82ae89b4cbdc85a` |
| [SDK 0.3.19a1 sdist](https://pypi.org/project/openecon/0.3.19a1/#files) | 11,233,675 | `98bbbea50d7225788dd72413d1ecda85ef025e05c80b41353284f84727ab54c7` |
| [Charts 0.3.1a1 wheel](https://pypi.org/project/openecon-charts/0.3.1a1/#files) | 440,214 | `4badfb26cbb1860dbb266d59cc726ad0057b640b007c9cbfb673bf95108ae4b4` |
| [Charts 0.3.1a1 sdist](https://pypi.org/project/openecon-charts/0.3.1a1/#files) | 475,337 | `9b053443ebca72332b8da7d7d45548654e742d303073e1e5b398399331fa0e3c` |
| [Reviewed Python source archive](https://github.com/bluearf/openeconometrics/releases/download/python-v0.3.19a1/openeconometrics-source-a8eb8799-public.tar.gz) | 11,719,833 | `11e92922b7b398aadc2d52437c7a427dc733b1060fa25cb9b6bd81323fcbc780` |

Provenance was also anonymously downloaded and hashed: [Mac](https://github.com/bluearf/openeconometrics/releases/download/v0.3.43-alpha.1/RELEASE-PROVENANCE.json)
`68640b4faf7ecbdf893b0596bd28ce83002da5c64395abd1f5b24d1c2add1014`,
[Windows](https://github.com/bluearf/openeconometrics/releases/download/v0.3.45-windows-alpha.1/windows-release-provenance.json)
`bdb95aa8375a554c6f023667a9e99264e311f00c116e477ccc5060ef3866e771`,
and [Python pair](https://github.com/bluearf/openeconometrics/releases/download/python-v0.3.19a1/RELEASE-PROVENANCE.json)
`26975f921b1f250620eb14e6d616077f6e45b8ffcbb2817fad2801772fb1f998`.
These are original acceptance receipts, not newly repeated installation tests.

## Download site and release notes

Anonymous HTTP readback of [the website](https://openeconometrics.com), its
`site.js` and the public releases API returned 200 with a standard browser
user-agent. The initial default-urllib request returned 403; that failed access
is retained in the receipt. The website controller's current API selection
resolves the Mac 0.3.43 DMG and Windows 0.3.45 installer listed above. Its signing
notes match those releases. This verifies HTTP/source/controller and public
download bytes; it does not claim a rendered-browser interaction or new native
installation. The site and original release notes have not been republished by
this documentation change.

## Exact-source CI and candidate acceptance

[Current policy](ci-cadence.md) runs a strict quick merge gate for PRs/current
bases and a full daily gate scheduled 01:17 UTC/04:17 Europe/Istanbul, with manual
full runs for release/material coverage. A scheduled run can start late. The
aggregate full result is named `OpenEconometrics / daily full gate` for both
scheduled and manually dispatched full runs; the event and tested source below
distinguish them.

| Workflow / event | Exact tested source | Observed result at 15:44 UTC, 11 October |
| --- | --- | --- |
| [Quick 38150617510](https://github.com/bluearf/openecon/actions/runs/38150617510), push | `6691f50aecadbb1d356af247e5e5c2eff1d1c9ce` | **Success**. Quick scope only. |
| [Quick 38151775204](https://github.com/bluearf/openecon/actions/runs/38151775204), push | `6eb04e2b9824ac230d14a034961d9b91521e98d1` | **Success**. Quick scope only; this document's inspected source. |
| [Daily full 38120796245](https://github.com/bluearf/openecon/actions/runs/38120796245), schedule | `7a195f6231023c025104115b2e3c8ebea411ff51` | **Failure**, completed 07:37:48 UTC. Eight shard jobs passed but the original aggregate cohort exceeded the unchanged 900-second limit; retained [timing analysis](sdk-scheduling-38120796245.json). |
| [Release full 38146156565](https://github.com/bluearf/openecon/actions/runs/38146156565), manual | `fd12cc1310b9e68a6bc0370989ba5adc5091c2e9` | **Failure**, completed 14:38:20 UTC. Historical Weibull catalogue fixture disagreed with the accepted Ng–Perron constant inference description. |
| [Windows 38146158847](https://github.com/bluearf/openecon/actions/runs/38146158847), manual | `fd12cc1310b9e68a6bc0370989ba5adc5091c2e9` | **Failure**, completed 14:42:55 UTC. Upgrade retained an old bundled runtime; expected frozen-core binding failed. |
| [Release full 38150647133](https://github.com/bluearf/openecon/actions/runs/38150647133), manual | `6691f50aecadbb1d356af247e5e5c2eff1d1c9ce` | **Failure**, completed 15:35:38 UTC. Python 3.11/3.13 shard 0 and shard 2 verification failed; the genuine aggregate rejected the incomplete successful union. Saved SDK logs remain diagnostic evidence; no numerical cause is inferred from aggregate status alone. |
| [Windows 38150649713](https://github.com/bluearf/openecon/actions/runs/38150649713), manual | `6691f50aecadbb1d356af247e5e5c2eff1d1c9ce` | **In progress**. A completed build does not establish installation/upgrade/native/cold-state acceptance. |
| [Separate full 38151908157](https://github.com/bluearf/openecon/actions/runs/38151908157), manual | `69946e2a9cc74cb3871eb71fd02d95b7519f797f` | **In progress**, separate candidate. Its result cannot be assigned to 6691f50 or 6eb04e2b. |

[PR 256](https://github.com/bluearf/openecon/pull/256) repaired the historical
catalogue assertion, [PR 257](https://github.com/bluearf/openecon/pull/257) the
stale bundled-runtime upgrade, and
[PR 259](https://github.com/bluearf/openecon/pull/259) native empty/bodyless POST
framing. Their integration and quick checks are source evidence. They do not
turn any failed receipt green or prove a complete new Mac/Windows release.
[MARKET-515](https://linear.app/bluearf/issue/MARKET-515) retains the unified
same-source Mac/package/public-source/installer/site publication scope.

## Authentication, cleanup and agreed boundaries

[MARKET-91](https://linear.app/bluearf/issue/MARKET-91) is **Done**. PR 251 merged
at 7a195f62; production revision `openecon-00029-ztv` accepted Google sign-in and
actual delivered six-digit email codes, preserving Firebase UID and roles.
Each new email session needs a fresh code; Google needs no additional code.
Password setup/link/reset was removed and superseded, and is not an outstanding
acceptance requirement. Production auth and bounded native auth acceptance do
not certify all collaboration or public installer flows.

[MARKET-802](https://linear.app/bluearf/issue/MARKET-802) is **Done**: the existing
owned live-session create/use/cleanup run retained the unchanged 690-second wait
and read back `residual_resources=0`. Other QA/human accounts were preserved.
Historical “deploy missing” or “live cleanup not yet run” paragraphs are dated
receipts superseded by their later accepted results. They are not current
blockers. Full installed Mac/Windows team/project picker, invitation/transfer,
offline queue reconnect without refitting and second-member readback remain in
[MARKET-801](https://linear.app/bluearf/issue/MARKET-801)/87/109; HTTP/API success
does not establish those installed UI and persistence layers.

The human-approved [repository boundaries](../../AGENTS.md) apply:

- Mac CPU is accepted; CUDA hardware and CUDA acceptance are deferred and not
  required. Supported MPS factors retain their own documented precision path.
- GitHub-hosted Windows installation/native acceptance is sufficient; a physical
  Windows PC is not required. The actual Windows 2025 runner is named honestly.
- Developer ID/notarization is deferred until enrollment, expected late October
  or early November 2026. Internal ad-hoc Mac alpha builds are accepted. No
  notarized, Developer ID signed or Gatekeeper-approved claim is made.
- Licensed Stata/SPSS/EViews executables are not required. Independent
  analytical, development-reference and synthetic-fixture validation remains
  method/option bounded and does not establish blanket vendor parity.

This reconciliation leaves method parents, newer platform acceptance,
collaboration and publication issues open within their actual remaining scope.

## Later readback in this reconciliation · 15:50 UTC

The original 15:44 snapshot above is retained. At the final refresh, default
branch source had advanced to
`aa9adfc6ea69e8f3ba97b3f368389dcefa0a020c` (PR 261, committed 15:45:14 UTC),
with the same desktop 0.3.46 / SDK 0.3.20a1 / charts 0.3.2a1 version declarations.
Its [quick run 38152602603](https://github.com/bluearf/openecon/actions/runs/38152602603)
passed at 15:47:54 UTC. Earlier scientific/platform results are not assigned to
this newer source by version equality.

Windows [run 38150649713](https://github.com/bluearf/openecon/actions/runs/38150649713)
completed **successfully at 15:47:39 UTC** on its original exact source
`6691f50aecadbb1d356af247e5e5c2eff1d1c9ce`, attempt 1. The downloaded
[acceptance artifact 11707640642](https://github.com/bluearf/openecon/actions/runs/38150649713/artifacts/11707640642)
contains the matching build/controller source and actual Windows 2025 x64 runner
receipt, 19 true installed checks and zero owned processes at cleanup. This
includes genuine missing-WebView2/bootstrap, offline local computation, native
original/cold history, saved replay, four regularized families, eight saved
binomial systems, two-step/adaptive replay, and actual 0.3.44 upgrade/rollback.
The JSON supplement records the individually reviewed receipt hashes, source
pin, runner image and exact private retained installer identity.

The retained installer was also downloaded through authenticated GitHub access
and its complete 194,960,419 bytes independently hashed to
`db4afee5ddd6a9c7c23edc5e2dd4d40cf0f723725359b17ec673c0eff28c158b`.
It is [artifact 11707775511](https://github.com/bluearf/openecon/actions/runs/38150649713/artifacts/11707775511),
not an anonymous public release asset. The receipt distinguishes that private
readback from the anonymous public installer/package downloads above.

This establishes that bounded **0.3.46 hosted installed acceptance**. It does not
publish a Windows release, prove current Mac or collaboration acceptance, or
make the same-source failed full SDK run 38150647133 successful. The separate
full run 38151908157 remained in progress at this refresh. Public installer and
PyPI versions in the original table were unchanged.
