"""Offline tests of generation-bound blob operations; no execution capabilities remain."""
from unittest.mock import Mock

import pytest

from openecon.team_storage import MAX_TRANSFER_BYTES, MemoryStorage, TeamStorage
from openecon.team_store import TeamError


@pytest.fixture
def storage():
    result = object.__new__(TeamStorage)
    result.client, result.bucket, result.signer = Mock(), Mock(), Mock()
    result.bucket.name = 'openecon-test-projects'
    return result


@pytest.mark.parametrize('cls', [TeamStorage, MemoryStorage])
def test_storage_issues_no_execution_or_staging_capabilities(cls):
    for name in ('manifest', 'output_policy', 'signed_cancel_url', 'request_cancel', 'download_url'):
        assert not hasattr(cls, name)


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
