"""Offline tests of real signed policies and generation-bound blob operations."""
import base64
from datetime import datetime, timezone
import json
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from openecon.team_job import storage_url, validate_manifest
from openecon.team_storage import MAX_TRANSFER_BYTES, MemoryStorage, TeamStorage
from openecon.team_store import TeamError


@pytest.fixture
def storage():
    result = object.__new__(TeamStorage)
    result.client, result.bucket, result.signer = Mock(), Mock(), Mock()
    result.bucket.name = 'openecon-test-runs'
    return result


@pytest.fixture
def signed_storage():
    """Use the Google SDK with a temporary local key; never use ADC/network."""
    sdk = pytest.importorskip('google.cloud.storage')
    from cryptography.hazmat.primitives.asymmetric import rsa
    from google.auth import credentials, crypt
    from google.oauth2 import service_account
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer = crypt.RSASigner(key, 'offline-test-key')
    result = object.__new__(TeamStorage)
    result.signer = service_account.Credentials(signer,
        'openecon-signer@openecon-test.iam.gserviceaccount.com', 'https://oauth2.googleapis.com/token')
    result.client = sdk.Client(project='openecon-test', credentials=credentials.AnonymousCredentials())
    result.bucket = result.client.bucket('openecon-test-runs')
    return result


def test_output_policy_has_fixed_object_and_size_range_and_is_worker_compatible(signed_storage):
    key = 'staging/project/run/output.json'
    result = signed_storage.output_policy(key)
    storage_url(result['url'], signed=False)
    assert result['fields']['key'] == key
    policy = json.loads(base64.b64decode(result['fields']['policy']))
    assert ['content-length-range', 1, MAX_TRANSFER_BYTES] in policy['conditions']
    assert {'key': key} in policy['conditions']
    assert policy['conditions'].count({'Content-Type': 'application/json'}) == 1
    expiry = datetime.fromisoformat(policy['expiration'].replace('Z', '+00:00'))
    assert 1740 < (expiry - datetime.now(timezone.utc)).total_seconds() <= 1800
    # The roleless worker accepts the exact SDK shape, not only a hand-built fixture.
    validate_manifest({'execution_id': 'a' * 32, 'code': '1', 'files': [], 'output_upload': result})


def test_completion_policy_restricts_signed_upload_to_four_kib(signed_storage):
    key = f'staging/{"a" * 32}/{"b" * 32}/completion.json'
    result = signed_storage.output_policy(key, maximum=4096)
    policy = json.loads(base64.b64decode(result['fields']['policy']))
    assert ['content-length-range', 1, 4096] in policy['conditions']
    assert {'key': key} in policy['conditions']
    assert policy['conditions'].count({'Content-Type': 'application/json'}) == 1
    assert result['fields']['key'] == key


def test_real_sdk_capabilities_pass_strict_broker_validation(signed_storage):
    from openecon.team_sandbox_broker import BrokerError, validate_body

    pid, rid = 'a' * 32, 'b' * 32
    prefix = f'staging/{pid}/{rid}/'
    body = {
        'execution_id': rid, 'timeout_seconds': 60,
        'input_url': signed_storage.download_url({'key': prefix + 'input.json', 'generation': 123}),
        'cancel_url': signed_storage.signed_cancel_url(pid, rid),
        'completion_upload': signed_storage.output_policy(prefix + 'completion.json', maximum=4096),
    }
    assert validate_body(json.dumps(body).encode()) == body
    # Reproduce the old SDK integration error: fields already inserts an exact
    # Content-Type condition, so repeating it in conditions fails the contract.
    policy = json.loads(base64.b64decode(body['completion_upload']['fields']['policy']))
    policy['conditions'].append({'Content-Type': 'application/json'})
    body['completion_upload']['fields']['policy'] = base64.b64encode(json.dumps(policy).encode()).decode()
    with pytest.raises(BrokerError):
        validate_body(json.dumps(body).encode())


@pytest.mark.parametrize('maximum', [1, MAX_TRANSFER_BYTES])
def test_output_policy_accepts_inclusive_integer_limits(storage, maximum):
    storage.output_policy('staging/p/r/completion.json', maximum=maximum)
    policy_args = storage.client.generate_signed_post_policy_v4.call_args.kwargs
    assert ['content-length-range', 1, maximum] in policy_args['conditions']


@pytest.mark.parametrize('maximum', [
    True, False, 0, -1, MAX_TRANSFER_BYTES + 1, 4096.0, '4096', None,
    float('inf'), float('nan'),
])
def test_output_policy_rejects_invalid_limits_before_signing(storage, maximum):
    with pytest.raises(ValueError):
        storage.output_policy('staging/p/r/completion.json', maximum=maximum)
    storage.client.generate_signed_post_policy_v4.assert_not_called()


def test_download_url_is_short_lived_and_generation_pinned(signed_storage):
    value = signed_storage.download_url({'key': 'projects/p/files/data.csv', 'generation': 12345})
    storage_url(value, generation=True)
    query = parse_qs(urlsplit(value).query)
    assert query['generation'] == ['12345']
    assert query['X-Goog-Expires'] == ['1800']
    assert urlsplit(value).path == '/openecon-test-runs/projects/p/files/data.csv'


def test_cancel_url_is_unpinned_for_a_future_marker_and_scoped_to_run(signed_storage):
    project_id, run_id = 'a' * 32, 'b' * 32
    value = signed_storage.signed_cancel_url(project_id, run_id)
    storage_url(value)
    query = parse_qs(urlsplit(value).query)
    assert 'generation' not in query
    assert query['X-Goog-Expires'] == ['1800']
    assert urlsplit(value).path == f'/openecon-test-runs/staging/{project_id}/{run_id}/cancel.json'
    other_project = signed_storage.signed_cancel_url('c' * 32, run_id)
    other_run = signed_storage.signed_cancel_url(project_id, 'd' * 32)
    assert len({urlsplit(url).path for url in (value, other_project, other_run)}) == 3


def test_cancel_marker_is_fixed_immutable_and_idempotent(storage):
    exceptions = pytest.importorskip('google.api_core.exceptions')
    project_id, run_id = 'a' * 32, 'b' * 32
    key = f'staging/{project_id}/{run_id}/cancel.json'
    blob = storage.bucket.blob.return_value
    blob.upload_from_string.side_effect = [None, exceptions.PreconditionFailed('already exists')]
    assert storage.request_cancel(project_id, run_id) == {'key': key}
    assert storage.request_cancel(project_id, run_id) == {'key': key}
    assert storage.bucket.blob.call_args.args == (key,)
    assert blob.upload_from_string.call_count == 2
    for call in blob.upload_from_string.call_args_list:
        assert json.loads(call.args[0]) == {'execution_id': run_id, 'cancel_requested': True}
        assert call.kwargs == {'content_type': 'application/json', 'if_generation_match': 0,
                               'timeout': 45, 'retry': None}


def test_cancel_request_propagates_storage_failure(storage):
    exceptions = pytest.importorskip('google.api_core.exceptions')
    storage.bucket.blob.return_value.upload_from_string.side_effect = exceptions.Forbidden('denied')
    with pytest.raises(exceptions.Forbidden):
        storage.request_cancel('a' * 32, 'b' * 32)


@pytest.mark.parametrize('method', ['request_cancel', 'signed_cancel_url'])
@pytest.mark.parametrize('position', [0, 1])
@pytest.mark.parametrize('invalid', [
    '', 'a' * 31, 'a' * 33, 'A' * 32, 'g' * 32, 'a' * 32 + '\n',
    '../' + 'a' * 29, 'a' * 31 + '/', None, True, b'a' * 32,
])
def test_cancel_capabilities_reject_noncanonical_scope_before_storage(storage, method, position, invalid):
    identifiers = ['a' * 32, 'b' * 32]
    identifiers[position] = invalid
    with pytest.raises(ValueError):
        getattr(storage, method)(*identifiers)
    storage.bucket.blob.assert_not_called()


def test_memory_cancel_marker_preserves_first_write_and_isolates_project_and_run():
    storage = MemoryStorage()
    pid, rid = 'a' * 32, 'b' * 32
    first = storage.request_cancel(pid, rid)
    original = storage.get(first)
    assert json.loads(original) == {'execution_id': rid, 'cancel_requested': True}
    # Keeping an existing marker byte-for-byte is the immutable write contract.
    storage.objects[first['key']] = original + b'\n'
    assert storage.request_cancel(pid, rid) == first
    assert storage.get(first) == original + b'\n'
    second = storage.request_cancel('c' * 32, rid)
    third = storage.request_cancel(pid, 'd' * 32)
    assert len({first['key'], second['key'], third['key']}) == 3
    assert len(storage.objects) == 3
    for invalid_ids in [('A' * 32, rid), (pid, '../other')]:
        with pytest.raises(ValueError):
            storage.request_cancel(*invalid_ids)
    assert len(storage.objects) == 3


def test_manifest_uses_pinned_inputs_and_bounded_output_policy(signed_storage):
    captures = {}
    def put(key, payload, content_type):
        captures.update(key=key, payload=payload, content_type=content_type)
        return {'key': key, 'generation': 7, 'size': len(payload)}
    signed_storage.put = put
    run = {'id': 'a' * 32, 'code': '1 + 1', 'timeout_seconds': 45,
           'files': [{'name': 'data.csv', 'blob': {'key': 'projects/p/files/data.csv', 'generation': 456}}]}
    input_url, output, reference = signed_storage.manifest('p', run)
    assert captures['key'] == f'staging/p/{run["id"]}/input.json'
    assert captures['content_type'] == 'application/json'
    assert output == {'key': f'staging/p/{run["id"]}/output.json'}
    assert reference['generation'] == 7
    assert parse_qs(urlsplit(input_url).query)['generation'] == ['7']
    manifest = validate_manifest(json.loads(captures['payload']))
    assert manifest['files'][0]['name'] == 'data.csv'
    assert parse_qs(urlsplit(manifest['files'][0]['download_url']).query)['generation'] == ['456']
    for url in (input_url, manifest['files'][0]['download_url']):
        query = parse_qs(urlsplit(url).query)
        # All inputs remain valid through the longest bounded control lease.
        assert 1140 < int(query['X-Goog-Expires'][0]) <= 1800
        storage_url(url, generation=True)
    policy = json.loads(base64.b64decode(manifest['output_upload']['fields']['policy']))
    expiry = datetime.fromisoformat(policy['expiration'].replace('Z', '+00:00'))
    assert (expiry - datetime.now(timezone.utc)).total_seconds() > 1140


def test_put_requires_new_immutable_object_and_enforces_upload_size(storage):
    blob = storage.bucket.blob.return_value
    blob.generation = '42'
    assert storage.put('projects/p/file', b'abc') == {'key': 'projects/p/file', 'generation': 42, 'size': 3}
    assert blob.upload_from_string.call_args.kwargs['if_generation_match'] == 0
    assert blob.upload_from_string.call_args.kwargs['retry'] is None
    with pytest.raises(TeamError) as error:
        storage.put('projects/p/large', b'x' * (MAX_TRANSFER_BYTES + 1))
    assert error.value.status_code == 413
    assert blob.upload_from_string.call_count == 1


@pytest.mark.parametrize('size', [None, 11])
def test_get_rejects_oversized_or_unknown_blob_before_download(storage, size):
    blob = storage.bucket.blob.return_value
    blob.size, blob.generation = size, '42'
    with pytest.raises(TeamError) as error:
        storage.get({'key': 'staging/p/run/output.json'}, maximum=10)
    assert error.value.status_code == 413
    blob.download_as_bytes.assert_not_called()


def test_get_pins_exact_generation_inspected_to_prevent_replacement_race(storage):
    blob = storage.bucket.blob.return_value
    blob.size, blob.generation = 3, '42'
    blob.download_as_bytes.return_value = b'abc'
    assert storage.get({'key': 'staging/p/run/output.json'}, maximum=10) == b'abc'
    assert blob.download_as_bytes.call_args.kwargs['if_generation_match'] == 42
    assert storage.bucket.blob.call_args.kwargs == {'generation': None}


def test_saved_reference_download_and_cleanup_use_generation_preconditions(storage):
    blob = storage.bucket.blob.return_value
    blob.size, blob.generation = 3, '42'
    reference = {'key': 'projects/p/result', 'generation': 42}
    storage.get(reference)
    assert storage.bucket.blob.call_args.kwargs == {'generation': 42}
    storage.delete(reference)
    assert blob.delete.call_args.kwargs['if_generation_match'] == 42


def test_missing_object_returns_safe_not_found_and_cleanup_is_idempotent(storage):
    exceptions = pytest.importorskip('google.api_core.exceptions')
    blob = storage.bucket.blob.return_value
    blob.reload.side_effect = exceptions.NotFound('private-storage-path')
    with pytest.raises(TeamError) as error:
        storage.get({'key': 'missing'})
    assert error.value.status_code == 404
    assert 'private-storage-path' not in str(error.value)
    blob.delete.side_effect = exceptions.NotFound('gone')
    storage.delete({'key': 'missing', 'generation': 42})
