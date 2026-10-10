"""Project authorization and transactional collaboration metadata.

No Python execution, pickle, dataframe parsing or user-controlled filesystem
paths belong in this process. Every mutation reads membership in its transaction.
Run documents are read-only history: desktop-shared results and records from the
retired cloud execution backend. This module never starts or finalizes a run.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
import unicodedata
from threading import RLock
from uuid import uuid4

from openecon import script_contracts as scripts
from openecon import file_layout


MAX_PROJECT_NAME_VERSION = 2**53 - 1
MAX_PROJECT_NAME_LENGTH = 100
MAX_PROJECT_DESCRIPTION_LENGTH = 500


class TeamError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code, self.status_code = code, status_code


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ident(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value):
        raise TeamError("NOT_FOUND", "Record not found.", 404)
    return value


def project_name(value: str) -> str:
    if (not isinstance(value, str)
            or any(unicodedata.category(character) in {'Cc', 'Cs', 'Zl', 'Zp'} for character in value)):
        raise TeamError('INVALID_PROJECT', 'The project name must be text without control characters.', 422)
    name = value.strip()
    if not 1 <= len(name) <= MAX_PROJECT_NAME_LENGTH:
        raise TeamError('INVALID_PROJECT', 'The project name must contain 1–100 characters.', 422)
    return name


def project_description(value: str) -> str:
    if (not isinstance(value, str)
            or any(unicodedata.category(character) in {'Cc', 'Cs'} and character not in '\t\r\n'
                   for character in value)):
        raise TeamError('INVALID_PROJECT', 'The description cannot contain control characters; tabs and line breaks are allowed.', 422)
    description = value.strip()
    if len(description) > MAX_PROJECT_DESCRIPTION_LENGTH:
        raise TeamError('INVALID_PROJECT', 'The description can contain up to 500 characters.', 422)
    return description


def project_name_version(value: int) -> int:
    if type(value) is not int or not 0 <= value <= MAX_PROJECT_NAME_VERSION:
        raise TeamError('INVALID_PROJECT', 'The project name version must be a valid integer.', 422)
    return value


def filename(value: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value.encode('utf-8')) <= 180
            or value.startswith('.') or value != value.strip()
            or any(c in value for c in '/\\:\x00') or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or value in {'analysis.py', 'manifest.json', 'result.json'}):
        raise TeamError("INVALID_FILENAME", "Use a short, safe filename.", 422)
    return value


_ENVIRONMENT_NAME = re.compile(r'[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?\Z')
_ENVIRONMENT_VERSION = re.compile(
    r'(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?'
    r'(?:\.post[0-9]+)?(?:\.dev[0-9]+)?'
    r'(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?\Z', re.IGNORECASE,
)


def validate_environment_manifest(manifest):
    """Validate inert pinned package metadata; never resolve or install it.

    The native environment manager checks Python and protected core versions
    against its bundled runtime before restoring an overlay. This control plane
    only stores strict distribution/version records; URLs, paths and installer
    options have no representation in this schema.
    """
    def invalid():
        raise TeamError('INVALID_ENVIRONMENT', 'The project package list is invalid; use package names with exact versions.', 422)

    def package(name, version):
        if (not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name)
                or not isinstance(version, str) or not 1 <= len(version) <= 100
                or not _ENVIRONMENT_VERSION.fullmatch(version)):
            invalid()
        return re.sub(r'[-_.]+', '-', name).lower(), version

    if isinstance(manifest, dict) and manifest.get('schema') == 2:
        if (type(manifest['schema']) is not int
                or set(manifest) != {'schema', 'python', 'core', 'requirements', 'locked', 'installer', 'specifications'}
                or not isinstance(manifest['installer'], str) or manifest['installer'] not in {'pip', 'uv'}):
            invalid()
        legacy = validate_environment_manifest({key: value for key, value in manifest.items()
                                                if key not in {'installer', 'specifications', 'schema'}} | {'schema': 1})
        try:
            from openecon.package_requirements import validate_specification_pins
            from openecon.project_packages import PackageError
            specs = validate_specification_pins(manifest['specifications'], legacy['requirements'], legacy['core'])
        except (PackageError, ValueError, TypeError, RecursionError):
            invalid()
        return {**legacy, 'schema': 2, 'installer': manifest['installer'], 'specifications': specs}

    if (not isinstance(manifest, dict)
            or set(manifest) != {'schema', 'python', 'core', 'requirements', 'locked'}
            or type(manifest['schema']) is not int or manifest['schema'] != 1
            or not isinstance(manifest['python'], str)
            or not re.fullmatch(r'3\.(?:0|[1-9][0-9]?)', manifest['python'])
            or not isinstance(manifest['core'], dict) or len(manifest['core']) > 100):
        invalid()
    core = {}
    for name, version in manifest['core'].items():
        name, version = package(name, version)
        if name in core:
            invalid()
        core[name] = version
    rows = {}
    for section in ('requirements', 'locked'):
        source = manifest[section]
        if not isinstance(source, list) or len(source) > 100:
            invalid()
        normalized = {}
        for item in source:
            if not isinstance(item, dict) or set(item) != {'name', 'version'}:
                invalid()
            name, version = package(item['name'], item['version'])
            if name in normalized or name in core:
                invalid()
            normalized[name] = version
        rows[section] = normalized
    if any(rows['locked'].get(name) != version
           for name, version in rows['requirements'].items()):
        invalid()
    return {'schema': 1, 'python': manifest['python'], 'core': dict(sorted(core.items())),
            **{section: [{'name': name, 'version': version}
                         for name, version in sorted(rows[section].items())]
               for section in ('requirements', 'locked')}}


class MemoryDocuments:
    """Injectable transactional repository for tests; never selected in cloud."""
    def __init__(self):
        self.data = {}
        self.lock = RLock()

    def get(self, path):
        return deepcopy(self.data.get(path))

    def put(self, path, value):
        self.data[path] = deepcopy(value)

    def delete(self, path):
        self.data.pop(path, None)

    def scan(self, collection, field=None, value=None, limit=100, newest=None):
        with self.lock:
            rows = [deepcopy(v) for k, v in self.data.items()
                    if k.startswith(collection + '/') and k.count('/') == collection.count('/') + 1
                    and (field is None or v.get(field) == value)]
            if newest:
                rows.sort(key=lambda item: item[newest], reverse=True)
            return rows[:limit]

    def atomic(self, fn):
        with self.lock:
            before = deepcopy(self.data)
            try:
                return fn(self)
            except BaseException:
                self.data = before
                raise

    def history_scan(self, collection, *, snapshot, before, limit):
        from openecon.team_history import HISTORY_FIELDS
        with self.lock:
            rows = []
            for path, value in self.data.items():
                if not (path.startswith(collection + '/') and path.count('/') == collection.count('/') + 1):
                    continue
                key = (value['created_at'], path.rsplit('/', 1)[1])
                if key[0] <= snapshot and (before is None or key < before):
                    rows.append({**{field: deepcopy(value[field]) for field in HISTORY_FIELDS if field in value},
                                 '_document_id': key[1]})
            rows.sort(key=lambda row: (row['created_at'], row['_document_id']), reverse=True)
            return rows[:limit]


class FirestoreDocuments:
    def __init__(self, project: str):
        import os
        if os.environ.get('FIRESTORE_EMULATOR_HOST'):
            raise ValueError('Cloud project storage cannot use the Firestore emulator.')
        from google.cloud import firestore
        self.client = firestore.Client(project=project)

    def get(self, path):
        return self.client.document(path).get().to_dict()

    def scan(self, collection, field=None, value=None, limit=100, newest=None):
        from google.cloud.firestore_v1.base_query import FieldFilter
        query = self.client.collection(collection)
        if field is not None:
            query = query.where(filter=FieldFilter(field, '==', value))
        if newest:
            query = query.order_by(newest, direction='DESCENDING')
        return [s.to_dict() for s in query.limit(limit).stream()]

    def history_scan(self, collection, *, snapshot, before, limit):
        from google.cloud.firestore_v1.base_query import FieldFilter
        from google.cloud.firestore_v1.field_path import FieldPath
        from openecon.team_history import HISTORY_FIELDS
        query = (self.client.collection(collection).select(HISTORY_FIELDS)
                 .where(filter=FieldFilter('created_at', '<=', snapshot))
                 .order_by('created_at', direction='DESCENDING')
                 .order_by(FieldPath.document_id(), direction='DESCENDING'))
        if before is not None:
            query = query.start_after({'created_at': before[0],
                                       '__name__': self.client.document(f'{collection}/{before[1]}')})
        return [{**item.to_dict(), '_document_id': item.id} for item in query.limit(limit).stream()]

    def atomic(self, fn):
        from google.cloud import firestore
        client = self.client

        @firestore.transactional
        def apply(transaction):
            class Batch:
                def __init__(self):
                    self.writes = {}
                    self.reads = {}

                def get(self, path):
                    if path in self.writes:
                        return deepcopy(self.writes[path])
                    if path not in self.reads:
                        self.reads[path] = client.document(path).get(transaction=transaction).to_dict()
                    return deepcopy(self.reads[path])

                def put(self, path, value):
                    self.writes[path] = deepcopy(value)

                def delete(self, path):
                    self.writes[path] = None

            batch = Batch()
            result = fn(batch)
            # Firestore requires all reads before any write; stage domain writes.
            for path, value in batch.writes.items():
                reference = client.document(path)
                if value is None:
                    transaction.delete(reference)
                else:
                    transaction.set(reference, value)
            return result

        return apply(self.client.transaction(max_attempts=5))


class TeamStore:
    def __init__(self, documents, *, owner_email: str, public_origin: str):
        self.db, self.owner_email = documents, owner_email.lower()
        self.public_origin = public_origin

    def _project(self, db, project_id, user, minimum='viewer'):
        project = db.get(f'oe_projects/{ident(project_id)}')
        membership = (project or {}).get('members', {}).get(user.uid)
        if not user.email_verified or membership is None:
            raise TeamError('NOT_FOUND', 'The project was not found or your access has been removed.', 404)
        if {'viewer': 0, 'editor': 1, 'owner': 2}.get(membership['role'], -1) < {
            'viewer': 0, 'editor': 1, 'owner': 2
        }[minimum]:
            raise TeamError('ROLE_REQUIRED', 'You do not have sufficient project permissions for this action.', 403)
        return project

    def project(self, project_id, user, minimum='viewer'):
        return self._project(self.db, project_id, user, minimum)

    @staticmethod
    def summary(project, uid):
        return {k: project[k] for k in ('id', 'name', 'description', 'created_at', 'updated_at')} | {
            'role': project['members'][uid]['role'], 'member_count': len(project['members']),
            'name_version': project_name_version(project.get('name_version', 0))}

    def _audit(self, db, project_id, user, action, subject=''):
        event_id = uuid4().hex
        db.put(f'oe_projects/{project_id}/audit/{event_id}', {
            'id': event_id, 'actor_uid': user.uid, 'actor_email': user.email,
            'action': action, 'subject': subject, 'created_at': now(),
        })

    def me(self, user):
        def update(db):
            profile = db.get(f'oe_users/{user.uid}') or {
                'uid': user.uid, 'enabled': False, 'project_ids': [], 'created_at': now(),
            }
            profile.update(email=user.email, name=user.name)
            if user.email_verified and user.email == self.owner_email:
                profile['enabled'] = True
            db.put(f'oe_users/{user.uid}', profile)
            return profile
        profile = self.db.atomic(update)
        invitations = []
        if user.email_verified:
            invitations = [self.invitation_summary(i) for i in
                           self.db.scan('oe_invitations', 'email', user.email)
                           if i['status'] == 'pending' and i['expires_at'] > now()]
        return {'user': {'uid': user.uid, 'email': user.email, 'name': user.name,
                         'email_verified': user.email_verified},
                'enabled': bool(profile['enabled'] and user.email_verified),
                'projects': self.projects(user), 'invitations': invitations}

    def projects(self, user):
        if not user.email_verified:
            return []
        profile = self.db.get(f'oe_users/{user.uid}') or {}
        result = []
        for project_id in profile.get('project_ids', []):
            project = self.db.get(f'oe_projects/{project_id}')
            if project and user.uid in project['members']:
                result.append(self.summary(project, user.uid))
        return sorted(result, key=lambda item: item['updated_at'], reverse=True)

    def create_project(self, user, name, description=''):
        if not user.email_verified:
            raise TeamError('EMAIL_UNVERIFIED', 'Verify your email address first.', 403)
        name, description = project_name(name), project_description(description)
        project_id, timestamp = uuid4().hex, now()
        def create(db):
            profile = db.get(f'oe_users/{user.uid}') or {}
            if not profile.get('enabled'):
                raise TeamError('INVITATION_REQUIRED', 'Accept a team invitation to get started.', 403)
            owned = [db.get(f'oe_projects/{p}') for p in profile.get('project_ids', [])]
            if sum(p is not None and p['owner_uid'] == user.uid for p in owned) >= 5:
                raise TeamError('PROJECT_LIMIT', 'Up to 5 projects can be created per account.', 429)
            project = {'id': project_id, 'name': name, 'name_version': 0, 'description': description,
                       'owner_uid': user.uid, 'created_at': timestamp, 'updated_at': timestamp,
                       'members': {user.uid: {'uid': user.uid, 'email': user.email, 'name': user.name,
                                              'role': 'owner', 'joined_at': timestamp}},
                       'invitations': [], 'files': [], 'active_run': None, 'run_count': 0}
            db.put(f'oe_projects/{project_id}', project)
            profile['project_ids'] = [*profile.get('project_ids', []), project_id]
            db.put(f'oe_users/{user.uid}', profile)
            self._audit(db, project_id, user, 'project.created')
            return self.summary(project, user.uid)
        return self.db.atomic(create)

    def rename_project(self, project_id, user, name, name_version):
        name = project_name(name)
        expected = project_name_version(name_version)
        def rename(db):
            project = self._project(db, project_id, user, 'owner')
            version = project_name_version(project.get('name_version', 0))
            if version != expected:
                raise TeamError('VERSION_CONFLICT', 'The project name was changed elsewhere. Reload the current name.', 409)
            # An unchanged current name is idempotent, but stale versions above
            # never bypass CAS merely because their text matches the new name.
            if project['name'] == name:
                return self.summary(project, user.uid)
            if version >= MAX_PROJECT_NAME_VERSION:
                raise TeamError('VERSION_CONFLICT', 'The project name version limit has been reached.', 409)
            invitations = []
            for invitation_id in project['invitations']:
                invite = db.get(f'oe_invitations/{ident(invitation_id)}')
                if invite and invite.get('project_id') == project['id'] and invite.get('status') == 'pending':
                    invitations.append((invitation_id, invite))
            project.update(name=name, name_version=version + 1, updated_at=now())
            db.put(f'oe_projects/{project_id}', project)
            for invitation_id, invite in invitations:
                db.put(f'oe_invitations/{invitation_id}', {**invite, 'project_name': name})
            self._audit(db, project_id, user, 'project.renamed', name)
            return self.summary(project, user.uid)
        return self.db.atomic(rename)

    def invitation_summary(self, invite):
        return {k: invite[k] for k in ('id', 'project_id', 'project_name', 'email', 'role', 'expires_at')} | {
            'url': f'{self.public_origin}/?invite={invite["id"]}'}

    def members(self, project_id, user):
        project = self.project(project_id, user)
        invites = []
        if project['members'][user.uid]['role'] == 'owner':
            for i in project['invitations']:
                invite = self.db.get(f'oe_invitations/{i}')
                if invite and invite['status'] == 'pending' and invite['expires_at'] > now():
                    invites.append(self.invitation_summary(invite))
        return {'members': list(project['members'].values()), 'invitations': invites}

    def invite(self, project_id, user, email, role):
        email = email.strip().lower()
        if (len(email) > 254 or not re.fullmatch(r'[^\s@/\\]+@[^\s@/\\]+\.[^\s@/\\]+', email)
                or role not in {'editor', 'viewer'}):
            raise TeamError('INVALID_INVITATION', 'Enter a valid email address and select a role.', 422)
        invite_id = uuid4().hex
        def create(db):
            project = self._project(db, project_id, user, 'owner')
            if any(member['email'] == email for member in project['members'].values()):
                raise TeamError('ALREADY_MEMBER', 'This person is already a project member.', 409)
            active = []
            for i in project['invitations']:
                old = db.get(f'oe_invitations/{i}')
                if old and old['status'] == 'pending' and old['expires_at'] > now():
                    if old['email'] == email:
                        raise TeamError('ALREADY_INVITED', 'There is already an active invitation for this address.', 409)
                    active.append(i)
            if len(active) + len(project['members']) >= 50:
                raise TeamError('MEMBER_LIMIT', 'Each project can have up to 50 members and pending invitations combined.', 429)
            invite = {'id': invite_id, 'project_id': project_id, 'project_name': project['name'],
                      'email': email, 'role': role, 'status': 'pending', 'created_at': now(),
                      'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()}
            db.put(f'oe_invitations/{invite_id}', invite)
            project['invitations'] = [*active, invite_id]
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'invitation.created', email)
            return self.invitation_summary(invite)
        return self.db.atomic(create)

    def accept(self, invite_id, user):
        if not user.email_verified:
            raise TeamError('EMAIL_UNVERIFIED', 'Verify the invited email address first.', 403)
        def accept(db):
            invite = db.get(f'oe_invitations/{ident(invite_id)}')
            if (not invite or invite['email'] != user.email or invite['status'] != 'pending'
                    or invite['expires_at'] <= now()):
                raise TeamError('INVITATION_INVALID', 'The invitation is invalid, expired or belongs to another address.', 404)
            project = db.get(f'oe_projects/{invite["project_id"]}')
            if not project or invite_id not in project['invitations']:
                raise TeamError('INVITATION_INVALID', 'This invitation is no longer active.', 404)
            if len(project['members']) >= 50:
                raise TeamError('MEMBER_LIMIT', 'The project member limit has been reached.', 429)
            profile = db.get(f'oe_users/{user.uid}') or {'uid': user.uid, 'created_at': now(), 'project_ids': []}
            if len(profile['project_ids']) >= 100:
                raise TeamError('PROJECT_LIMIT', 'The account project limit has been reached.', 429)
            # Re-acceptance must never demote an existing owner or change roles.
            if user.uid not in project['members']:
                project['members'][user.uid] = {'uid': user.uid, 'email': user.email, 'name': user.name,
                                               'role': invite['role'], 'joined_at': now()}
            project['invitations'].remove(invite_id)
            project['updated_at'] = now()
            profile.update(enabled=True, email=user.email, name=user.name,
                           project_ids=list(dict.fromkeys([*profile['project_ids'], project['id']])))
            invite.update(status='accepted', accepted_by=user.uid, accepted_at=now())
            db.put(f'oe_projects/{project["id"]}', project)
            db.put(f'oe_users/{user.uid}', profile)
            db.put(f'oe_invitations/{invite_id}', invite)
            self._audit(db, project['id'], user, 'invitation.accepted')
            return self.summary(project, user.uid)
        return self.db.atomic(accept)

    def revoke_invite(self, project_id, invite_id, user):
        def revoke(db):
            project = self._project(db, project_id, user, 'owner')
            invite = db.get(f'oe_invitations/{ident(invite_id)}')
            if not invite or invite['project_id'] != project_id:
                raise TeamError('NOT_FOUND', 'Invitation not found.', 404)
            invite['status'] = 'revoked'
            project['invitations'] = [i for i in project['invitations'] if i != invite_id]
            db.put(f'oe_invitations/{invite_id}', invite)
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'invitation.revoked', invite['email'])
        self.db.atomic(revoke)

    def change_member(self, project_id, target_uid, user, role=None):
        def change(db):
            project = self._project(db, project_id, user, 'owner')
            if target_uid == project['owner_uid']:
                raise TeamError('OWNER_PROTECTED', 'The project owner cannot be removed or assigned a different role.', 409)
            if target_uid not in project['members']:
                raise TeamError('NOT_FOUND', 'Member not found.', 404)
            if role is None:
                del project['members'][target_uid]
                profile = db.get(f'oe_users/{ident(target_uid)}')
                if profile:
                    profile['project_ids'] = [p for p in profile['project_ids'] if p != project_id]
                    db.put(f'oe_users/{target_uid}', profile)
            elif role in {'viewer', 'editor'}:
                project['members'][target_uid]['role'] = role
            else:
                raise TeamError('INVALID_ROLE', 'Select the editor or viewer role.', 422)
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'member.removed' if role is None else 'member.role_changed', target_uid)
        self.db.atomic(change)

    def script(self, project_id, user):
        self.project(project_id, user)
        draft = self.db.get(f'oe_projects/{project_id}/workspace/script') or {
            'code': None, 'name': 'analysis.py', 'version': 0}
        self.project(project_id, user)
        return draft

    def save_script(self, project_id, user, code, version):
        code = self._script_value(scripts.script_code, code)
        version = self._script_value(scripts.script_version, version)
        def save(db):
            project = self._project(db, project_id, user, 'editor')
            path = f'oe_projects/{project_id}/workspace/script'
            old = db.get(path) or {'version': 0}
            if old['version'] != version:
                raise TeamError('VERSION_CONFLICT', 'Another team member changed the draft. Download your code and reload the current draft.', 409)
            self._script_value(scripts.script_version, version + 1)
            index = self._script_index(db, project_id)
            self._script_quota(index, code)
            draft = {'code': code, 'name': 'analysis.py', 'version': version + 1,
                     'updated_at': now(), 'updated_by': user.uid}
            project['updated_at'] = draft['updated_at']
            db.put(path, draft)
            db.put(f'oe_projects/{project_id}', project)
            return draft
        return self.db.atomic(save)

    @staticmethod
    def _script_value(validate, *values, **options):
        try:
            return validate(*values, **options)
        except scripts.ScriptValidationError as exc:
            raise TeamError(exc.code, str(exc), 404 if exc.code == 'NOT_FOUND' else
                            429 if exc.code == 'SCRIPT_LIMIT' else 422) from exc

    def _script_index(self, db, project_id):
        index = db.get(f'oe_projects/{project_id}/workspace/scripts')
        if index is None:
            return {'schema': 1, 'scripts': []}
        try:
            if (not isinstance(index, dict) or set(index) != {'schema', 'scripts'}
                    or type(index['schema']) is not int or index['schema'] != 1):
                raise ValueError('Invalid scripts index.')
            return {'schema': 1, 'scripts': scripts.validate_script_rows(index['scripts'])}
        except (ValueError, TypeError, KeyError) as exc:
            raise TeamError('CORRUPT_SCRIPT', 'The file list could not be read.', 422) from exc

    @staticmethod
    def _script_public(document):
        return {key: document[key] for key in ('id', 'name', 'code', 'version')}

    def _main_script(self, db, project_id):
        legacy = db.get(f'oe_projects/{project_id}/workspace/script') or {
            'code': None, 'name': 'analysis.py', 'version': 0}
        try:
            if (legacy['name'] != 'analysis.py'
                    or (legacy['code'] is not None and scripts.script_code(legacy['code']) != legacy['code'])):
                raise ValueError('Invalid main script.')
            scripts.script_version(legacy['version'])
            return {'id': 'analysis', **{key: legacy[key] for key in ('name', 'code', 'version')}}
        except (ValueError, TypeError, KeyError) as exc:
            raise TeamError('CORRUPT_SCRIPT', 'The file could not be read.', 422) from exc

    def _indexed_script(self, db, project_id, identifier, index):
        if identifier == 'analysis':
            return self._main_script(db, project_id)
        row = next((row for row in index['scripts'] if row['id'] == identifier), None)
        if row is None:
            raise TeamError('NOT_FOUND', 'File not found.', 404)
        document = db.get(f'oe_projects/{project_id}/scripts/{identifier}')
        try:
            if (not isinstance(document, dict)
                    or scripts.metadata(document) != scripts.metadata(row)
                    or len(scripts.script_code(document['code']).encode('utf-8')) != row['size']):
                raise ValueError('Invalid script document.')
            return self._script_public(document)
        except (ValueError, TypeError, KeyError) as exc:
            raise TeamError('CORRUPT_SCRIPT', 'The file could not be read.', 422) from exc

    def _script_quota(self, index, main_code):
        if (len(index['scripts']) >= scripts.MAX_SCRIPTS
                or len((main_code or '').encode('utf-8')) + sum(row['size'] for row in index['scripts'])
                > scripts.MAX_SCRIPT_BYTES):
            raise TeamError('SCRIPT_LIMIT', 'The project file limit has been reached.', 429)

    def scripts(self, project_id, user):
        self.project(project_id, user)
        index = self._script_index(self.db, project_id)
        main = self._main_script(self.db, project_id)
        self.project(project_id, user)
        return {'scripts': [scripts.metadata(main), *(scripts.metadata(row) for row in index['scripts'])]}

    def named_script(self, project_id, user, script_id):
        self.project(project_id, user)
        identifier = self._script_value(scripts.script_id, script_id)
        index = self._script_index(self.db, project_id)
        result = self._indexed_script(self.db, project_id, identifier, index)
        self.project(project_id, user)
        return result

    def create_script(self, project_id, user, name='untitled.py', code='', *, script_id=None):
        self.project(project_id, user, 'editor')
        name = self._script_value(scripts.script_name, name)
        code = self._script_value(scripts.script_code, code)
        identifier = (uuid4().hex if script_id is None
                      else self._script_value(scripts.script_id, script_id, creating=True))
        def create(db):
            project = self._project(db, project_id, user, 'editor')
            index = self._script_index(db, project_id)
            document_path = f'oe_projects/{project_id}/scripts/{identifier}'
            if any(row['id'] == identifier for row in index['scripts']):
                existing = self._indexed_script(db, project_id, identifier, index)
                if existing['name'] == name and existing['code'] == code:
                    return existing
                raise TeamError('SCRIPT_ID_CONFLICT', 'The file ID is already in use.', 409)
            if db.get(document_path) is not None:
                raise TeamError('CORRUPT_SCRIPT', 'The file list is inconsistent.', 422)
            chosen_name = self._script_value(scripts.unique_script_name, name,
                                             ['analysis.py', *(row['name'] for row in index['scripts'])])
            document = {'id': identifier, 'name': chosen_name, 'code': code, 'version': 0}
            index['scripts'].append({**scripts.metadata(document), 'size': len(code.encode('utf-8'))})
            self._script_quota(index, self._main_script(db, project_id)['code'])
            timestamp = now()
            db.put(document_path, {**document, 'updated_at': timestamp, 'updated_by': user.uid})
            db.put(f'oe_projects/{project_id}/workspace/scripts', index)
            project['updated_at'] = timestamp
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'script.created', identifier)
            return document
        return self.db.atomic(create)

    def save_named_script(self, project_id, user, script_id, code, version):
        self.project(project_id, user, 'editor')
        identifier = self._script_value(scripts.script_id, script_id)
        code, version = (self._script_value(scripts.script_code, code),
                         self._script_value(scripts.script_version, version))
        if identifier == 'analysis':
            legacy = self.save_script(project_id, user, code, version)
            return {'id': 'analysis', **{key: legacy[key] for key in ('name', 'code', 'version')}}
        def save(db):
            project = self._project(db, project_id, user, 'editor')
            index = self._script_index(db, project_id)
            previous = self._indexed_script(db, project_id, identifier, index)
            if previous['version'] != version:
                raise TeamError('VERSION_CONFLICT', 'The file was changed by someone else.', 409)
            self._script_value(scripts.script_version, version + 1)
            document = {**previous, 'code': code, 'version': version + 1}
            row_index = next(i for i, row in enumerate(index['scripts']) if row['id'] == identifier)
            index['scripts'][row_index] = {**scripts.metadata(document), 'size': len(code.encode('utf-8'))}
            self._script_quota(index, self._main_script(db, project_id)['code'])
            timestamp = now()
            db.put(f'oe_projects/{project_id}/scripts/{identifier}',
                   {**document, 'updated_at': timestamp, 'updated_by': user.uid})
            db.put(f'oe_projects/{project_id}/workspace/scripts', index)
            project['updated_at'] = timestamp
            db.put(f'oe_projects/{project_id}', project)
            return document
        return self.db.atomic(save)

    def environment(self, project_id, user):
        self.project(project_id, user)
        stored = self.db.get(f'oe_projects/{project_id}/workspace/environment')
        # A slow document read must not return project content after removal.
        self.project(project_id, user)
        return ({'manifest': deepcopy(stored['manifest']), 'version': stored['version']}
                if stored else {'manifest': None, 'version': 0})

    @staticmethod
    def _layout_value(operation, *args):
        try:
            return operation(*args)
        except file_layout.FileLayoutError as exc:
            raise TeamError(exc.code, str(exc), 409 if exc.code in {'VERSION_CONFLICT', 'FILE_INVENTORY_CONFLICT'} else 422) from exc

    def _file_inventory(self, db, project_id, project):
        index = self._script_index(db, project_id)
        main = self._main_script(db, project_id)
        return file_layout.inventory([main, *index['scripts']], project['files'])

    def get_file_layout(self, project_id, user):
        def read(db):
            project = self._project(db, project_id, user)
            stored = db.get(f'oe_projects/{project_id}/workspace/file-layout')
            stored = self._layout_value(file_layout.stored_document, stored)
            result = self._layout_value(file_layout.reconcile, stored, self._file_inventory(db, project_id, project))
            self._project(db, project_id, user)
            return result
        return self.db.atomic(read)

    def put_file_layout(self, project_id, user, document):
        validated = self._layout_value(file_layout.validate, document)
        def save(db):
            project = self._project(db, project_id, user, 'editor')
            path = f'oe_projects/{project_id}/workspace/file-layout'
            old = db.get(path)
            previous = self._layout_value(file_layout.stored_document, old) if old is not None else {'version': 0}
            if validated['version'] != previous['version']:
                raise TeamError('VERSION_CONFLICT', 'Another team member changed the file layout. Reload the current list.', 409)
            self._layout_value(file_layout.validate_inventory, validated, self._file_inventory(db, project_id, project))
            saved = {'version': self._layout_value(file_layout.version, previous['version'] + 1),
                     'entries': deepcopy(validated['entries'])}
            timestamp = now()
            db.put(path, {**saved, 'updated_at': timestamp, 'updated_by': user.uid})
            project['updated_at'] = timestamp
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'file-layout.updated')
            return saved
        return self.db.atomic(save)

    def save_environment(self, project_id, user, manifest, version):
        if type(version) is not int or not 0 <= version < 2**63 - 1:
            raise TeamError('INVALID_ENVIRONMENT', 'The package list version is invalid.', 422)
        validated = validate_environment_manifest(manifest)

        def save(db):
            project = self._project(db, project_id, user, 'editor')
            path = f'oe_projects/{project_id}/workspace/environment'
            old = db.get(path) or {'version': 0}
            if old['version'] != version:
                raise TeamError('VERSION_CONFLICT', 'Another team member changed the package list. Reload the current list.', 409)
            record = {'manifest': validated, 'version': version + 1,
                      'updated_at': now(), 'updated_by': user.uid}
            project['updated_at'] = record['updated_at']
            db.put(path, record)
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'environment.updated')
            return {'manifest': deepcopy(validated), 'version': version + 1}
        return self.db.atomic(save)

    def add_file(self, project_id, user, metadata):
        def add(db):
            project = self._project(db, project_id, user, 'editor')
            reservations = list(project.get('transfer_reservations', {}).values())
            if any(f['name'].casefold() == metadata['name'].casefold()
                   for f in [*project['files'], *reservations]):
                raise TeamError('FILE_EXISTS', 'This filename is already in use. Rename the file.', 409)
            if (sum(f['size_bytes'] for f in project['files'])
                    + sum(f['size_bytes'] for f in reservations) + metadata['size_bytes'] > 8 * 1024**3):
                raise TeamError('FILE_LIMIT', 'The project transfer budget is 8 GiB.', 429)
            if len(project['files']) + len(project.get('transfer_reservations', {})) >= 20 or sum(
                    f['size_bytes'] for f in project['files'] if f.get('transfer') != 'chunked-v1') + metadata['size_bytes'] > 64 * 1024**2:
                raise TeamError('FILE_LIMIT', 'Each project can have up to 20 files with a combined size of 64 MiB.', 429)
            project['files'].append(metadata)
            project['updated_at'] = now()
            db.put(f'oe_projects/{project_id}', project)
            self._audit(db, project_id, user, 'file.added', metadata['name'])
        self.db.atomic(add)

    def run(self, project_id, run_id):
        return self.db.get(f'oe_projects/{ident(project_id)}/runs/{ident(run_id)}')

    def runs(self, project_id, user):
        self.project(project_id, user)
        # Single-field ordering uses Firestore's default index.
        runs = self.db.scan(f'oe_projects/{project_id}/runs', limit=50, newest='created_at')
        return sorted(runs, key=lambda r: r['created_at'])[-50:]
