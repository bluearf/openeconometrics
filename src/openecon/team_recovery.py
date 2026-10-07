"""Operator-only, inert project backups; never registered on the public API.

Metadata is read at one transaction snapshot. Immutable, generation-pinned
objects can then be copied outside that transaction. Restore publishes metadata
only after every durable object has been checked and copied into a fresh scope.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from uuid import uuid4

from openecon import __version__
from openecon.team_storage import MAX_TRANSFER_BYTES, TeamStorage
from openecon.team_store import (FirestoreDocuments, MemoryDocuments, TeamStore,
                                 ident, now, validate_environment_manifest)

MAX_DOCUMENTS = 400  # Includes scoped profiles/invitations; no silent truncation.
MAX_METADATA_BYTES = 4 * 1024**2
MAX_BACKUP_BYTES = 512 * 1024**2
COLLECTIONS = ('workspace', 'scripts', 'runs', 'audit')
NAMESPACES = ('oe_projects', 'oe_users', 'oe_invitations', 'oe_recoveries')
_HASH = re.compile(r'[0-9a-f]{64}\Z')


class RecoveryError(ValueError):
    """Stable operator error; contains no provider response or credentials."""


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _scoped_profile(profile, uid, pid):
    if not profile or pid not in profile.get('project_ids', []):
        raise RecoveryError('A member profile is missing its project link.')
    # Never export another project's IDs, credentials or provider login grants.
    return {**{key: deepcopy(profile[key]) for key in
               ('uid', 'enabled', 'email', 'name', 'created_at') if key in profile},
            'uid': uid, 'project_ids': [pid]}


def snapshot(documents, project_id):
    """Read the complete supported project schema without source mutations."""
    pid = ident(project_id)
    root = f'oe_projects/{pid}'

    def collect(get, query):
        project = get(root)
        if not project:
            raise RecoveryError('The source project does not exist.')
        values = {root: project}

        def add(path, value):
            values[path] = value
            if len(values) > MAX_DOCUMENTS:
                raise RecoveryError('The project exceeds the 400-document recovery budget.')
            if len(encoded(values)) > MAX_METADATA_BYTES:
                raise RecoveryError('The project exceeds the 4 MiB metadata recovery budget.')

        for collection in COLLECTIONS:
            for path, value in query(f'{root}/{collection}'):
                add(path, value)
        for path, value in query('oe_invitations', pid):
            add(path, value)
        for uid in project['members']:
            uid = ident(uid)
            add(f'oe_users/{uid}', _scoped_profile(get(f'oe_users/{uid}'), uid, pid))
        _validate_documents(values, pid)
        return values

    if isinstance(documents, MemoryDocuments):
        with documents.lock:
            def query(collection, project=None):
                return [(path, deepcopy(value)) for path, value in sorted(documents.data.items())
                        if path.startswith(collection + '/')
                        and path.count('/') == collection.count('/') + 1
                        and (project is None or value.get('project_id') == project)]
            return collect(documents.get, query)
    if not isinstance(documents, FirestoreDocuments):
        raise TypeError('A supported transactional document repository is required.')
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    @firestore.transactional
    def read(transaction):
        def get(path):
            return documents.client.document(path).get(transaction=transaction, timeout=25).to_dict()

        def query(collection, project=None):
            q = documents.client.collection(collection)
            if project is not None:
                q = q.where(filter=FieldFilter('project_id', '==', project))
            # One lookahead makes an over-budget project fail, never truncate.
            return ((item.reference.path, item.to_dict()) for item in
                    q.limit(MAX_DOCUMENTS + 1).stream(transaction=transaction, timeout=25))
        return collect(get, query)

    return read(documents.client.transaction(read_only=True, max_attempts=1))


def _reference(value, pid):
    if (not isinstance(value, dict) or set(value) != {'key', 'generation', 'size'}
            or not isinstance(value['key'], str)
            or not value['key'].startswith(f'projects/{pid}/')
            or len(value['key'].encode('utf-8')) > 1024
            or any(part in {'', '.', '..'} for part in value['key'].split('/'))
            or type(value['generation']) is not int or value['generation'] < 1
            or type(value['size']) is not int or not 0 <= value['size'] <= MAX_TRANSFER_BYTES):
        raise RecoveryError('A durable blob reference is invalid or belongs to another project.')
    return value


def references(values, pid):
    """Durable heads plus inputs captured by old runs; exclude expiring staging."""
    result = {}

    def add(reference):
        reference = _reference(reference, pid)
        key = (reference['key'], reference['generation'])
        if key in result and result[key] != reference:
            raise RecoveryError('A blob generation has inconsistent size metadata.')
        result[key] = reference

    for item in values[f'oe_projects/{pid}']['files']:
        add(item['blob'])
    for path, run in values.items():
        if not path.startswith(f'oe_projects/{pid}/runs/'):
            continue
        for item in [*run.get('files', []), *run.get('artifacts', [])]:
            add(item['blob'])
        if run.get('result') is not None:
            add(run['result'])
    return [deepcopy(result[key]) for key in sorted(result)]


def _validate_documents(values, pid):
    if (not isinstance(values, dict) or not 1 <= len(values) <= MAX_DOCUMENTS
            or len(encoded(values)) > MAX_METADATA_BYTES):
        raise RecoveryError('The metadata recovery budget is exceeded.')
    root = f'oe_projects/{pid}'
    project = values.get(root)
    if not isinstance(project, dict) or project.get('id') != pid:
        raise RecoveryError('The project identity is invalid.')
    if project.get('active_run') is not None:
        raise RecoveryError('Wait for every active computation to finish before taking a backup.')
    members = project.get('members')
    if (not isinstance(members, dict) or not 1 <= len(members) <= 50
            or project.get('owner_uid') not in members
            or sum(member.get('role') == 'owner' for member in members.values()) != 1
            or members[project['owner_uid']].get('role') != 'owner'):
        raise RecoveryError('The project owner/membership graph is invalid.')
    profiles = set()
    invitations = set()
    for path, value in values.items():
        if not isinstance(path, str) or not isinstance(value, dict):
            raise RecoveryError('Document paths and values must be inert JSON metadata.')
        parts = path.split('/')
        for part in parts:
            ident(part)
        if path == root:
            continue
        if len(parts) == 2 and parts[0] == 'oe_users' and parts[1] in members:
            uid = parts[1]
            if (value.get('uid') != uid or value.get('project_ids') != [pid]
                    or set(value) - {'uid', 'enabled', 'email', 'name', 'created_at', 'project_ids'}
                    or type(value.get('enabled')) is not bool):
                raise RecoveryError('Member profiles must be scoped to this project.')
            profiles.add(uid)
        elif len(parts) == 2 and parts[0] == 'oe_invitations' and value.get('project_id') == pid:
            if value.get('id') != parts[1] or value.get('role') not in {'editor', 'viewer'}:
                raise RecoveryError('An invitation link is invalid.')
            invitations.add(parts[1])
        elif len(parts) == 4 and parts[:2] == ['oe_projects', pid] and parts[2] in COLLECTIONS:
            if parts[2] == 'runs':
                if (value.get('id') != parts[3]
                        or (value.get('project_id') != pid
                            and not ('project_id' not in value and value.get('execution_origin') == 'desktop'))):
                    raise RecoveryError('A saved run belongs to another project.')
                if value.get('state') not in {'finished', 'failed', 'cancelled'}:
                    raise RecoveryError('Wait for every saved run to reach a terminal state.')
        else:
            raise RecoveryError('A document lies outside the supported project backup scope.')
    if profiles != set(members) or not set(project.get('invitations', [])) <= invitations:
        raise RecoveryError('The membership/invitation graph is incomplete.')
    for uid, member in members.items():
        if member.get('uid') != uid or member.get('role') not in {'owner', 'editor', 'viewer'}:
            raise RecoveryError('A member identity or role is invalid.')
    db = MemoryDocuments()
    db.data = deepcopy(values)
    store = TeamStore(db, owner_email='', public_origin='https://recovery.invalid')
    # Reuse the real source/inventory validators; no code or packages execute.
    store._main_script(db, pid)
    index = store._script_index(db, pid)
    for row in index['scripts']:
        store._indexed_script(db, pid, row['id'], index)
    layout = values.get(f'{root}/workspace/file-layout')
    if layout is not None:
        from openecon.file_layout import reconcile, stored_document, validate_inventory
        inventory = store._file_inventory(db, pid, project)
        validate_inventory(reconcile(stored_document(layout), inventory), inventory)
    environment = values.get(f'{root}/workspace/environment')
    if environment is not None:
        validate_environment_manifest(environment['manifest'])
    references(values, pid)


def _write(path, payload):
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def export_project(documents, storage, project_id, output, *, source):
    """Create a private directory. manifest.json is its last, complete marker."""
    _location(source)
    snapshot_started_at = now()
    values = snapshot(documents, project_id)
    directory = Path(output)
    directory.mkdir(mode=0o700)  # Never replace an existing directory.
    try:
        (directory / 'blobs').mkdir(mode=0o700)
        rows, total = [], 0
        for reference in references(values, project_id):
            payload = storage.get(reference, maximum=MAX_TRANSFER_BYTES)
            if len(payload) != reference['size']:
                raise RecoveryError('An immutable object does not match its recorded size.')
            total += len(payload)
            if total > MAX_BACKUP_BYTES:
                raise RecoveryError('The project exceeds the 512 MiB blob recovery budget.')
            checksum = digest(payload)
            path = directory / 'blobs' / checksum
            if not path.exists():
                _write(path, payload)
            rows.append({'source': reference, 'sha256': checksum, 'size_bytes': len(payload)})
        manifest = {'schema': 1, 'openecon_version': __version__, 'created_at': snapshot_started_at,
                    'source': source, 'project_id': project_id, 'documents': values, 'blobs': rows}
        payload = encoded(manifest)
        if len(payload) > MAX_METADATA_BYTES:
            raise RecoveryError('The complete manifest exceeds 4 MiB.')
        _mapping_budget(rows, project_id)
        _validate_result_links(manifest, directory)
        _write(directory / 'manifest.json', payload)
        return {'project_id': project_id, 'manifest_sha256': digest(payload),
                'snapshot_started_at': snapshot_started_at,
                'documents': len(values), 'blob_references': len(rows), 'blob_bytes': total}
    except BaseException:
        shutil.rmtree(directory)
        raise


def _read_file(path, maximum):
    # No extraction, URLs, symlinks or arbitrary paths from a manifest.
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    except OSError as exc:
        raise RecoveryError('A backup file is missing or linked.') from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise RecoveryError('A backup file is not regular or exceeds its byte budget.')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            payload = stream.read(maximum + 1)
    finally:
        os.close(fd)
    if len(payload) > maximum:
        raise RecoveryError('A backup file exceeds its byte budget.')
    return payload


def _blob_payload(directory, row):
    payload = _read_file(directory / 'blobs' / row['sha256'], MAX_TRANSFER_BYTES)
    if len(payload) != row['size_bytes'] or digest(payload) != row['sha256']:
        raise RecoveryError('A backup object failed its size/SHA-256 check.')
    return payload


def _validate_result_links(manifest, directory):
    rows = {encoded(row['source']): row for row in manifest['blobs']}
    pid = manifest['project_id']
    inventories = [manifest['documents'][f'oe_projects/{pid}']['files']]
    inventories.extend(run.get('files', []) for path, run in manifest['documents'].items()
                       if path.startswith(f'oe_projects/{pid}/runs/'))
    for inventory in inventories:
        for item in inventory:
            row = rows[encoded(item['blob'])]
            if item.get('data_hash') != row['sha256'] or item.get('size_bytes') != row['size_bytes']:
                raise RecoveryError('A source data file failed its saved data-version/size check.')
    for path, run in manifest['documents'].items():
        if '/runs/' not in path or not run.get('result'):
            continue
        payload = _blob_payload(directory, rows[encoded(run['result'])])
        record = json.loads(payload)
        artifacts = record.get('artifacts', [])
        stored = run.get('artifacts', [])
        if (record.get('id') != run['id'] or record.get('code') != run['code']
                or len(artifacts) != len(stored)
                or any(item.get('name') != saved['name'] or item.get('size_bytes') != saved['size_bytes']
                       or item.get('url') != f'/api/projects/{manifest["project_id"]}/workspace/runs/{run["id"]}/files/{i}'
                       for i, (item, saved) in enumerate(zip(artifacts, stored)))):
            raise RecoveryError('A saved result/source/artifact publication is incomplete or inconsistent.')
        if run.get('execution_origin') == 'desktop':
            files = manifest['documents'][f'oe_projects/{manifest["project_id"]}']['files']
            for supplied in record.get('input_files', []):
                if not any(item['id'] == supplied['id'] and item['data_hash'] == supplied['data_hash']
                           for item in files):
                    raise RecoveryError('An archived desktop result is missing its source data version.')


def validate_bundle(directory, expected_sha256):
    """Validate the entire inert bundle before constructing a target client."""
    directory = Path(directory)
    if (directory.is_symlink() or not directory.is_dir()
            or (directory / 'blobs').is_symlink() or not (directory / 'blobs').is_dir()
            or not isinstance(expected_sha256, str) or not _HASH.fullmatch(expected_sha256)):
        raise RecoveryError('A regular backup directory and trusted manifest SHA-256 are required.')
    payload = _read_file(directory / 'manifest.json', MAX_METADATA_BYTES)
    if digest(payload) != expected_sha256:
        raise RecoveryError('The manifest failed its trusted SHA-256 check.')
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RecoveryError('Duplicate JSON keys are not supported.')
                result[key] = value
            return result
        manifest = json.loads(payload, object_pairs_hook=unique,
                              parse_constant=lambda _: (_ for _ in ()).throw(RecoveryError('Nonfinite JSON.')))
        if (not isinstance(manifest, dict) or set(manifest) != {
                'schema', 'openecon_version', 'created_at', 'source', 'project_id', 'documents', 'blobs'}
                or type(manifest['schema']) is not int or manifest['schema'] != 1
                or manifest['openecon_version'] != __version__
                or set(manifest['source']) != {'gcp_project', 'bucket'}
                or any(not isinstance(v, str) or not v for v in manifest['source'].values())):
            raise RecoveryError('The backup schema/core/source is incompatible with this recovery tool.')
        _location(manifest['source'])
        pid = ident(manifest['project_id'])
        _validate_documents(manifest['documents'], pid)
        expected = {encoded(ref) for ref in references(manifest['documents'], pid)}
        actual, total = set(), 0
        if not isinstance(manifest['blobs'], list) or len(manifest['blobs']) > MAX_DOCUMENTS * 41:
            raise RecoveryError('The blob manifest is invalid.')
        for row in manifest['blobs']:
            if (not isinstance(row, dict) or set(row) != {'source', 'sha256', 'size_bytes'}
                    or not isinstance(row['sha256'], str) or not _HASH.fullmatch(row['sha256'])
                    or type(row['size_bytes']) is not int or row['size_bytes'] != row['source']['size']):
                raise RecoveryError('A backup object entry is invalid.')
            ref = encoded(_reference(row['source'], pid))
            if ref in actual:
                raise RecoveryError('A duplicate blob reference is not supported.')
            actual.add(ref)
            total += row['size_bytes']
            if total > MAX_BACKUP_BYTES:
                raise RecoveryError('The backup exceeds the 512 MiB recovery budget.')
            _blob_payload(directory, row)
        if actual != expected:
            raise RecoveryError('The backup is missing a referenced object or includes foreign objects.')
        _mapping_budget(manifest['blobs'], pid)
        _validate_result_links(manifest, directory)
        return manifest
    except RecoveryError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError, UnicodeError) as exc:
        raise RecoveryError('The project backup metadata or saved result is invalid.') from exc


def _location(location):
    if (not isinstance(location, dict) or set(location) != {'gcp_project', 'bucket'}
            or any(not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9-]{4,62}', value)
                   for value in location.values())):
        raise RecoveryError('Explicit canonical cloud project and bucket names are required.')


def _mapping_budget(rows, pid):
    # Conservative destination generation width; reserve room below Firestore's
    # per-document limit for receipt fields surrounding the mapping.
    estimated = [{'source': row['source'], 'destination': {
        'key': f'projects/{pid}/recovery/{"f" * 32}/{row["sha256"]}',
        'generation': 2**64 - 1, 'size': row['size_bytes']}} for row in rows]
    if len(encoded(estimated)) > 512 * 1024:
        raise RecoveryError('The object mapping exceeds the 512 KiB recovery receipt budget.')


def _ensure_empty(documents, paths=()):
    for namespace in NAMESPACES:
        if documents.scan(namespace, limit=1):
            raise RecoveryError('Recovery requires an empty, independent destination database.')
    if any(documents.get(path) is not None for path in paths):
        raise RecoveryError('The destination contains orphaned project metadata; use an empty database.')


def _publish(documents, values, receipt):
    if isinstance(documents, MemoryDocuments):
        def save(db):
            _ensure_empty(db, values)
            for path, value in values.items():
                db.put(path, value)
            db.put(f'oe_recoveries/{receipt["recovery_id"]}', receipt)
        documents.atomic(save)
        return
    if not isinstance(documents, FirestoreDocuments):
        raise TypeError('A supported transactional document repository is required.')
    from google.cloud import firestore

    @firestore.transactional
    def save(transaction):
        for namespace in NAMESPACES:
            if next(documents.client.collection(namespace).limit(1).stream(
                    transaction=transaction, timeout=25), None) is not None:
                raise RecoveryError('Recovery requires an empty, independent destination database.')
        for path in values:
            if documents.client.document(path).get(transaction=transaction, timeout=25).exists:
                raise RecoveryError('The destination contains orphaned project metadata.')
        for path, value in values.items():
            transaction.create(documents.client.document(path), value)
        transaction.create(documents.client.document(f'oe_recoveries/{receipt["recovery_id"]}'), receipt)
    save(documents.client.transaction(max_attempts=1))


def restore_project(documents, storage, directory, expected_sha256, *, destination):
    manifest = validate_bundle(directory, expected_sha256)
    _location(destination)
    if (destination['gcp_project'] == manifest['source']['gcp_project']
            or destination['bucket'] == manifest['source']['bucket']):
        raise RecoveryError('Use a separate destination cloud project and bucket.')
    _ensure_empty(documents, manifest['documents'])
    pid, recovery_id = manifest['project_id'], uuid4().hex
    replacements, copied, attempted = {}, {}, []
    try:
        for row in manifest['blobs']:
            payload = _blob_payload(Path(directory), row)  # Recheck immediately before upload.
            checksum = row['sha256']
            if checksum not in copied:
                key = f'projects/{pid}/recovery/{recovery_id}/{checksum}'
                # A lost upload reply may still have created this private key.
                # Track it before the request, while no metadata can refer to it.
                attempt = {'key': key}
                attempted.append(attempt)
                reference = storage.put(key, payload)
                attempt.update(reference)
                # Verify the independent store, not only the upload request.
                if digest(storage.get(reference, maximum=MAX_TRANSFER_BYTES)) != checksum:
                    raise RecoveryError('A destination object failed its readback SHA-256 check.')
                copied[checksum] = reference
            replacements[encoded(row['source'])] = copied[checksum]
    except BaseException as exc:
        cleanup_failed = False
        for reference in attempted:
            try:
                storage.delete(reference)
            except Exception:
                cleanup_failed = True
        if cleanup_failed:
            raise RecoveryError(f'Copy cleanup needs reconciliation: recovery {recovery_id}. '
                                'No project metadata was published.') from exc
        if isinstance(exc, RecoveryError):
            raise
        if isinstance(exc, Exception):
            raise RecoveryError(f'Copy stage failed: recovery {recovery_id}. No project metadata was published; '
                                'cleanup was attempted for every owned key. Check for delayed uploads.') from exc
        exc.add_note(f'Interrupted recovery {recovery_id}; no metadata was published. Check for delayed uploads.')
        raise

    def replace(value):
        if isinstance(value, dict):
            replacement = replacements.get(encoded(value))
            if replacement is not None:
                return deepcopy(replacement)
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    values = replace(manifest['documents'])
    receipt = {'schema': 1, 'recovery_id': recovery_id, 'project_id': pid,
               'manifest_sha256': expected_sha256, 'restored_at': now(),
               'source': manifest['source'], 'destination': destination,
               'documents': len(values), 'unique_objects': len(copied),
               'object_map': [{'source': json.loads(key), 'destination': value}
                              for key, value in sorted(replacements.items())]}
    # Once publication starts, a transport error can mean an unknown commit.
    # Never remove objects a committed project may reference. An operator uses
    # this receipt ID to reconcile or remove an unreferenced recovery scope.
    try:
        _publish(documents, values, receipt)
    except Exception as exc:
        raise RecoveryError(f'Metadata publication needs reconciliation: recovery {recovery_id}. '
                            'Copied objects were retained; do not blindly retry.') from exc
    if (documents.get(f'oe_recoveries/{recovery_id}') != receipt
            or any(documents.get(path) != value for path, value in values.items())):
        raise RecoveryError(f'Metadata readback failed: recovery {recovery_id}. Keep the destination isolated.')
    return receipt


def _cloud(gcp_project, bucket):
    """Admin storage needs object operations, never signing or compute rights."""
    from google.cloud import storage
    documents = FirestoreDocuments(gcp_project)
    blobs = object.__new__(TeamStorage)
    blobs.client = storage.Client(project=gcp_project)
    blobs.bucket = blobs.client.bucket(bucket)
    return documents, blobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--gcp-project', required=True)
    export.add_argument('--bucket', required=True)
    export.add_argument('--project-id', required=True)
    export.add_argument('--output', required=True)
    restore = commands.add_parser('restore')
    restore.add_argument('--gcp-project', required=True)
    restore.add_argument('--bucket', required=True)
    restore.add_argument('--input', required=True)
    restore.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(argv)
    location = {'gcp_project': args.gcp_project, 'bucket': args.bucket}
    try:
        _location(location)
        if args.command == 'export':
            db, blobs = _cloud(args.gcp_project, args.bucket)
            result = export_project(db, blobs, args.project_id, args.output, source=location)
        else:
            # Validate offline before reading ADC or contacting a destination.
            manifest = validate_bundle(args.input, args.manifest_sha256)
            if (location['gcp_project'] == manifest['source']['gcp_project']
                    or location['bucket'] == manifest['source']['bucket']):
                raise RecoveryError('Use a separate destination cloud project and bucket.')
            db, blobs = _cloud(args.gcp_project, args.bucket)
            result = restore_project(db, blobs, args.input, args.manifest_sha256, destination=location)
    except RecoveryError as exc:
        parser.exit(2, f'Recovery error: {exc}\n')
    except Exception:
        parser.exit(2, 'Recovery provider operation failed. Keep any destination isolated and reconcile before retrying.\n')
    print(encoded(result).decode())


if __name__ == '__main__':
    main()
