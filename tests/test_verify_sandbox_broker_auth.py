"""The live proof requests only a fixed control identity token, never broad impersonation."""
import importlib.util
from pathlib import Path
import json
import sys
from unittest.mock import MagicMock, Mock

import google.auth
from google.auth.transport import requests as auth_requests
import pytest

_SPEC = importlib.util.spec_from_file_location(
    'sandbox_broker_live_auth', Path(__file__).parents[1] / 'scripts/verify_sandbox_broker_live.py')
proof = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proof
_SPEC.loader.exec_module(proof)


@pytest.fixture
def issuance(monkeypatch):
    credentials = Mock()
    default = Mock(return_value=(credentials, 'unrelated-default-project'))
    monkeypatch.setattr(google.auth, 'default', default)
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.raw.read.return_value = b'{"token":"header.payload.signature"}'
    session = MagicMock()
    session.__enter__.return_value = session
    session.post.return_value = response
    authorized = Mock(return_value=session)
    monkeypatch.setattr(auth_requests, 'AuthorizedSession', authorized)
    return credentials, default, session, response, authorized


def test_live_proof_requests_only_id_token_for_fixed_control_and_audience(issuance):
    credentials, default, session, response, authorized = issuance
    assert proof.control_identity_token() == 'header.payload.signature'
    default.assert_called_once_with(scopes=['https://www.googleapis.com/auth/cloud-platform'],
                                    quota_project_id=proof.PROJECT)
    assert authorized.call_args.args == (credentials,)
    assert authorized.call_args.kwargs['auth_request'].session.trust_env is False
    assert session.trust_env is False
    session.post.assert_called_once_with(
        'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/'
        + proof.CONTROL + ':generateIdToken',
        json={'audience': proof.ORIGIN, 'includeEmail': True},
        timeout=30, allow_redirects=False, stream=True)
    response.raw.read.assert_called_once_with(32769, decode_content=True)


def test_missing_id_token_permission_never_reads_provider_error_or_falls_back(issuance):
    _, _, session, response, _ = issuance
    response.status_code = 403
    response.raw.read.return_value = b'private-provider-detail'
    with pytest.raises(proof.ProofError, match='fixed control identity') as error:
        proof.control_identity_token()
    assert 'private-provider-detail' not in str(error.value)
    assert session.post.call_count == 1
    response.raw.read.assert_not_called()


def test_adc_or_transport_errors_never_echo_private_diagnostics(issuance):
    _, _, session, _, _ = issuance
    session.post.side_effect = RuntimeError('private-token-sentinel')
    with pytest.raises(proof.ProofError) as error:
        proof.control_identity_token()
    assert 'private-token-sentinel' not in str(error.value)


@pytest.mark.parametrize('token', ['', None, 'header.payload.', 'opaque-token', 'h.p.s\nprivate', 'a' * 8193])
def test_live_proof_requires_full_bounded_jwt_not_signature_stripped_token(issuance, token):
    _, _, _, response, _ = issuance
    response.raw.read.return_value = json.dumps({'token': token}).encode()
    with pytest.raises(proof.ProofError, match='complete signed'):
        proof.control_identity_token()


@pytest.mark.parametrize('raw', [b'x' * 32769, b'{"token":"h.p.s","token":"different.token.signature"}'])
def test_oversized_or_duplicate_issuer_payload_is_rejected(issuance, raw):
    _, _, _, response, _ = issuance
    response.raw.read.return_value = raw
    with pytest.raises(proof.ProofError, match='fixed control identity'):
        proof.control_identity_token()
