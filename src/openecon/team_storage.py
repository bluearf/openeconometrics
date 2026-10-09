"""Private immutable blobs and narrowly scoped execution capabilities."""
from __future__ import annotations

from datetime import timedelta
import json
import re

from openecon.team_store import TeamError

MAX_TRANSFER_BYTES = 24 * 1024**2
# Covers the bounded queue + task lease (at most 19 minutes) without extending
# the worker's existing 30-minute maximum for signed transfer capabilities.
_TRANSFER_EXPIRATION = timedelta(minutes=30)


def _cancel_key(project_id, run_id):
    if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{32}', value)
           for value in (project_id, run_id)):
        raise ValueError('Canonical project and execution IDs are required.')
    return f'staging/{project_id}/{run_id}/cancel.json'


def _cancel_payload(run_id):
    return json.dumps({'execution_id': run_id, 'cancel_requested': True}).encode()


class TeamStorage:
    def __init__(self, project, bucket, signer_email):
        import google.auth
        from google.auth import iam
        from google.auth.transport.requests import Request
        from google.cloud import storage
        from google.oauth2 import service_account
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        self.client = storage.Client(project=project, credentials=credentials)
        self.bucket = self.client.bucket(bucket)
        self.signer = service_account.Credentials(
            iam.Signer(Request(), credentials, signer_email), signer_email,
            'https://oauth2.googleapis.com/token')

    def put(self, key, payload, content_type='application/octet-stream'):
        if len(payload) > MAX_TRANSFER_BYTES:
            raise TeamError('FILE_LIMIT', 'The file exceeds the size limit.', 413)
        blob = self.bucket.blob(key)
        blob.upload_from_string(payload, content_type=content_type, if_generation_match=0,
                                timeout=45, retry=None)
        return {'key': key, 'generation': int(blob.generation), 'size': len(payload)}

    def get(self, reference, maximum=MAX_TRANSFER_BYTES):
        from google.api_core.exceptions import NotFound
        blob = self.bucket.blob(reference['key'], generation=reference.get('generation'))
        try:
            blob.reload(timeout=15)
            if blob.size is None or blob.size > maximum:
                raise TeamError('RESULT_LIMIT', 'The file exceeds the size limit.', 413)
            # Generation pins the object inspected above, closing size-check races.
            return blob.download_as_bytes(if_generation_match=int(blob.generation), timeout=45)
        except NotFound:
            raise TeamError('NOT_FOUND', 'File not found.', 404) from None

    def find(self, key, maximum=MAX_TRANSFER_BYTES):
        """Inspect one exact immutable key after an ambiguous upload acknowledgement."""
        from google.api_core.exceptions import NotFound
        blob = self.bucket.blob(key)
        try:
            blob.reload(timeout=15)
        except NotFound:
            return None
        if blob.size is None or blob.size > maximum:
            raise TeamError('RESULT_LIMIT', 'The file exceeds the size limit.', 413)
        return {'key': key, 'generation': int(blob.generation), 'size': int(blob.size)}

    def delete(self, reference):
        from google.api_core.exceptions import NotFound
        try:
            self.bucket.blob(reference['key']).delete(
                if_generation_match=reference.get('generation'), timeout=15)
        except NotFound:
            pass

    def download_url(self, reference):
        return self.bucket.blob(reference['key']).generate_signed_url(
            version='v4', expiration=_TRANSFER_EXPIRATION, method='GET',
            credentials=self.signer, query_parameters={'generation': str(reference['generation'])})

    def output_policy(self, key, maximum=MAX_TRANSFER_BYTES):
        if type(maximum) is not int or not 1 <= maximum <= MAX_TRANSFER_BYTES:
            raise ValueError('The upload limit must be an integer within the transfer limit.')
        return self.client.generate_signed_post_policy_v4(
            self.bucket.name, key, expiration=_TRANSFER_EXPIRATION,
            # The SDK adds an exact-match condition for each field. Supplying
            # Content-Type again here creates a duplicate policy condition.
            conditions=[['content-length-range', 1, maximum]],
            fields={'Content-Type': 'application/json'}, credentials=self.signer)

    def signed_cancel_url(self, project_id, run_id):
        # The marker may not exist yet. Pinning a generation would prevent a
        # running sandbox from observing a cancellation created after launch.
        key = _cancel_key(project_id, run_id)
        return self.bucket.blob(key).generate_signed_url(
            version='v4', expiration=_TRANSFER_EXPIRATION, method='GET',
            credentials=self.signer)

    def request_cancel(self, project_id, run_id):
        from google.api_core.exceptions import PreconditionFailed
        key = _cancel_key(project_id, run_id)
        try:
            self.bucket.blob(key).upload_from_string(
                _cancel_payload(run_id), content_type='application/json',
                if_generation_match=0, timeout=45, retry=None)
        except PreconditionFailed:
            # An immutable cancellation marker is already present. Repeated
            # requests must not overwrite it or create a new generation.
            pass
        return {'key': key}

    def manifest(self, project_id, run):
        output_key = f'staging/{project_id}/{run["id"]}/output.json'
        payload = {'execution_id': run['id'], 'code': run['code'],
                   'timeout_seconds': run['timeout_seconds'],
                   'files': [{'name': f['name'], 'download_url': self.download_url(f['blob'])}
                             for f in run['files']],
                   'output_upload': self.output_policy(output_key)}
        reference = self.put(f'staging/{project_id}/{run["id"]}/input.json',
                             json.dumps(payload).encode(), 'application/json')
        return self.download_url(reference), {'key': output_key}, reference


class MemoryStorage:
    """Isolated test double; production always requires Google Cloud Storage."""
    def __init__(self):
        self.objects = {}

    def put(self, key, payload, content_type='application/octet-stream'):
        if key in self.objects:
            raise ValueError('Immutable object already exists')
        self.objects[key] = payload
        return {'key': key, 'generation': 1, 'size': len(payload)}

    def get(self, reference, maximum=MAX_TRANSFER_BYTES):
        value = self.objects.get(reference['key'])
        if value is None:
            raise TeamError('NOT_FOUND', 'File not found.', 404)
        if len(value) > maximum:
            raise TeamError('RESULT_LIMIT', 'The file exceeds the size limit.', 413)
        return value

    def find(self, key, maximum=MAX_TRANSFER_BYTES):
        value = self.objects.get(key)
        if value is None:
            return None
        if len(value) > maximum:
            raise TeamError('RESULT_LIMIT', 'The file exceeds the size limit.', 413)
        return {'key': key, 'generation': 1, 'size': len(value)}

    def delete(self, reference):
        self.objects.pop(reference['key'], None)

    def request_cancel(self, project_id, run_id):
        key = _cancel_key(project_id, run_id)
        self.objects.setdefault(key, _cancel_payload(run_id))
        return {'key': key}

    def manifest(self, project_id, run):
        return 'https://storage.googleapis.com/test/input.json', {'key': f'staging/{run["id"]}'}, {'key': 'input'}
