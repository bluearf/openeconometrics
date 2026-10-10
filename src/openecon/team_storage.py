"""Private immutable project blobs for the team sync backend.

No execution capabilities are issued: analyses run in the desktop app.
"""
from __future__ import annotations

from openecon.team_store import TeamError

MAX_TRANSFER_BYTES = 24 * 1024**2


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
        # signBlob identity for short-lived desktop sign-in custom tokens only.
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
