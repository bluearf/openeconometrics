# Team project backup and recovery

This operator runbook covers a **single project** and a new, isolated recovery
installation. The public team API does not expose backup/restore endpoints.
`python -m openecon.team_recovery` uses the existing Firestore/GCS adapters;
it neither executes saved Python nor installs packages. The accepted drill is
offline and synthetic. Live cloud IAM, provider login, network transfer and a
production-size RTO are separate deployment checks, not claims made by the drill.

## Recovery objectives and ownership

The service operator owns backup creation, off-host storage, receipt retention,
alerts for failed/missed backups, and the recovery decision. The operational
policy is:

| Objective | Target and measurement |
| --- | --- |
| RPO | At most 24 hours, measured from the snapshot in the last complete, verified backup. Run daily and before a deployment/data migration. A missed/failed backup breaks this target. |
| RTO | At most 30 minutes from the recovery decision to verified access to the restored isolated project, including IAM/configuration and provider UID checks. This is an operational target, not a production benchmark. |
| Retention | Keep seven daily and four weekly complete backups, plus the last verified pre-migration backup until rollout is accepted. Store the trusted receipt separately from the bundle. |
| Drill | Quarterly and after a schema/core/storage change. Record backup age, wall-clock recovery time, hashes, roles, links, and actual provider checks. |

No scheduler, cloud retention rule, or production alert is installed by this
change. Enable those operational controls before claiming that the RPO policy
is in force. A partial folder without a valid `manifest.json` is not a backup.

## Covered state

The read-only transaction captures one consistent metadata snapshot:

- Project ID, name/versions, owner, exact owner/editor/viewer membership UIDs,
  file metadata and data versions.
- Main source, all named source documents and their index, explorer folder
  ordering/aliases, the versioned inert Python/core/package environment manifest.
- **All** terminal run metadata within the budget, including older history,
  failed/cancelled runs, desktop archives, and project audit records.
- Project invitations, including accepted/revoked history, and member profiles
  scoped to this project. Other project IDs are removed from those profile
  projections; unrelated profiles/projects are never exported.
- Immutable generations referenced by current inputs, historical run inputs,
  saved result JSON and generated artifacts. Objects are copied by generation
  and checked against saved size/data SHA-256 and source/result/artifact links.

An active project computation or nonterminal run refuses backup. A result whose
artifact metadata is still being published also refuses backup; retry after
publication finishes. Once the transaction snapshot is complete, later source
edits do not change it. Blob generations make the subsequent copy refer to the
same immutable content. Firestore supports
[read-only transactions](https://cloud.google.com/python/docs/reference/firestore/latest/google.cloud.firestore_v1.base_transaction.BaseTransaction)
and [atomic transaction writes](https://firebase.google.com/docs/firestore/manage-data/transactions).

Excluded: Firebase accounts/passwords/tokens/provider grants, service secrets,
IAM, deployments, local desktop projects, installed package files, active
compute quotas/leases, and expiring `staging/` transfers. Terminal run metadata
may retain old staging/operation identifiers as historical data, but recovery
never fetches them or resumes execution. The environment manifest remains inert.

Limits fail explicitly: 400 metadata documents, 4 MiB complete manifest,
24 MiB per object, 512 MiB total referenced object bytes, and 512 KiB for the
object mapping in the recovery receipt. There is no truncated-success result.
The copy holds one bounded object at a time; this is not a total-process RSS
guarantee. A larger installation needs a separately validated bulk recovery
path. Managed Firestore exports must explicitly cover subcollections and a
consistent snapshot; ordinary project collection export alone is insufficient.
See [Firestore export/import](https://firebase.google.com/docs/firestore/manage-data/export-import).

## Create and retain a backup

Use a dedicated operator identity with source document reads and generation
reads on the private source bucket. It needs no compute or `signBlob` right.
Run with the source revision/core version recorded in the receipt/evidence.

```sh
python -m openecon.team_recovery export \
  --gcp-project SOURCE_GCP_PROJECT \
  --bucket SOURCE_PRIVATE_BUCKET \
  --project-id PROJECT_ID \
  --output /secure-backups/project-YYYYMMDD
```

The output directory must not exist; the parent must exist. The tool creates
private directories (0700) and files (0600), copies objects under SHA-256 names,
and writes `manifest.json` last. Failed exports remove only the newly created
directory. Existing directories and source documents/objects are untouched.

Retain the printed `manifest_sha256`, project ID, `snapshot_started_at`, source
revision and byte/document counts in a trusted, separate receipt. Verify the
bundle after transfer to private off-host storage. SHA-256 detects corruption
against that receipt; it is not a signature and cannot authenticate a bundle
when both the bundle and its supposed trusted receipt have been replaced.
Do not commit real backups or member information to the repository.
The recorded time is the start of snapshot acquisition, a conservative RPO
age bound; it is not the later time when copying completes.

## Restore in an independent environment

1. Prepare a **different cloud project**, empty Firestore database and different
   private GCS bucket. Keep the destination unavailable to end users during the
   operation. Existing `oe_projects`, `oe_users`, `oe_invitations`, recovery
   receipts, and orphaned documents at restored paths cause refusal. This tool
   does not merge multiple projects or overwrite a production installation.
2. Restore/configure authentication independently. Preserve the original
   Firebase UIDs, either with the original trusted identity provider or a
   separately reviewed Firebase account migration. Changing the destination
   Firebase project alone does not recreate accounts. Email equivalence does
   not establish UID identity. Keep account enablement/revocation controls intact.
3. Use the compatible recovery tool version and the **trusted** manifest hash.
   All schema, metadata, object size/hash and relationship checks run before
   target writes. The CLI checks the bundle before loading cloud credentials.

```sh
python -m openecon.team_recovery restore \
  --gcp-project RECOVERY_GCP_PROJECT \
  --bucket RECOVERY_PRIVATE_BUCKET \
  --input /secure-backups/project-YYYYMMDD \
  --manifest-sha256 TRUSTED_MANIFEST_SHA256
```

Each object is uploaded to a new immutable
`projects/PROJECT_ID/recovery/RECOVERY_ID/SHA256` key and read back. New GCS
generation references replace old references; project/file/source/run IDs and
the saved result bytes/relative artifact URLs remain the same. Only after every
object passes does one transaction **create** all metadata and an
`oe_recoveries/RECOVERY_ID` receipt. The transaction rechecks destination
emptiness and existing restored paths. It uses creates, never overwrites.
Every document is then read back and compared.

If copying or Stop fails before publication, cleanup attempts to remove all
keys attempted in this owned recovery scope, including an upload whose reply
was lost. A failed cleanup reports the recovery ID. It never removes a source
object or an unrelated destination key. Keep the target isolated while checking
for delayed upload completion or any remaining unreferenced scope.

A timeout/lost reply **after metadata publication begins** may mean an unknown
commit. The tool retains copied objects and reports the recovery ID; it does
not retry the transaction or delete potentially referenced content. Read the
receipt and project metadata to reconcile the outcome. If committed, compare
every document and mapping, then continue verification. If not committed,
confirm no project references that recovery scope before removing its orphan
objects and starting a fresh restore. Never blindly repeat an ambiguous restore.

## Verify before cutover

Compare every scoped metadata document byte-for-byte after reversing the
receipt's object-reference translation. Compare the SHA-256 of every source
and destination object, including original saved result JSON. Read sources,
folder layout, environment and older history through the actual control API.
Open an old result and download both its original input and generated artifact.

Confirm that owner/editor/viewer roles are unchanged, editors can access editor
operations, viewers cannot write, editors cannot manage owner-only invitations,
and outsiders are refused. On a real cloud drill also verify actual Firebase
login/revocation and GCS/Firestore IAM; injected synthetic authentication is
only authorization-contract evidence. Keep the original installation/data until
the recovery environment and measured RPO/RTO are accepted. Endpoint/configuration
cutover and rollback are separate operator actions.

## Reproduce the synthetic drill

```sh
PYTHONPATH=src:packages/openecon-charts/src python scripts/verify_team_recovery.py \
  --report /tmp/openecon-team-recovery.json
```

Four fresh processes seed a synthetic source, export, restore into independent
disk stores, and reopen/verify the destination through the real team HTTP API.
The fixtures contain owner/editor/viewer and outsider identities, two sources,
a folder, a CSV, a package manifest, invitations, 137 successful records including
a desktop archive, two failed/cancelled records, and a generated artifact.
An unrelated synthetic project checks scope isolation. No human project,
credentials, cloud service, package installation or computation is touched.
Owned temporary drill directories are removed after completion.

The persistent report records per-stage time/peak process RSS, all hash checks,
source preservation, history count, role denials and exact proof limits. See
[MARKET-88 evidence](evidence/market-88-team-recovery/README.md). This report
does not establish production-size recovery time or a live Firebase/GCS result.
