"""UID-bound handoff protocol tests; these are not real Firebase/Google proof."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from test_desktop_cloud import CHALLENGE, VERIFIER, api as api, begin, claims, headers


@pytest.mark.parametrize("provider,age,accepted", [
    ("google.com", 0, True), ("google.com", 600, False),
    ("google.com", None, False), ("google.com", -60, False),
    ("password", 0, False), ("custom", 0, False),
])
def test_native_verification_requires_fresh_same_provider_and_one_use_proof(
    api, provider, age, accepted,
):
    before = deepcopy(api.store.db.get(f"oe_projects/{api.pid}"))
    response = api.client.post('/api/desktop/account-link', headers=headers(), json={
        'challenge': CHALLENGE, 'target': 'google.com', 'action': 'verify',
    })
    assert response.status_code == 201
    request_id = response.json()['request_id']
    path = f'/api/desktop/account-link/{request_id}'
    assert api.client.get(f'/api/desktop/login/{request_id}', headers=headers()).json() == {
        'kind': 'account-link', 'uid': 'owner', 'target': 'google.com', 'action': 'verify',
    }
    assert api.client.post(path+'/complete', headers=headers('outsider')).status_code == 403
    timestamp = None if age is None else int(datetime.now(timezone.utc).timestamp()) - age
    api.auth._verifier = lambda uid: {
        **claims(uid, provider, True if provider == 'custom' else None),
        'auth_time': timestamp,
    }
    done = api.client.post(path+'/complete', headers=headers())
    assert done.status_code == (200 if accepted else 403)
    assert api.client.post(path+'/exchange', headers=headers(),
                           json={'verifier': 'b'*64}).status_code == 403
    result = api.client.post(path+'/exchange', headers=headers(), json={'verifier': VERIFIER})
    if accepted:
        assert result.json() == {'verified': True, 'uid': 'owner', 'target': 'google.com'}
        assert api.client.post(path+'/exchange', headers=headers(),
                               json={'verifier': VERIFIER}).status_code == 410
    else:
        assert result.status_code == 202
        assert api.store.db.get(f'oe_desktop_logins/{request_id}')['completed'] is False
    api.issuer.assert_not_called()
    assert api.store.db.get(f'oe_projects/{api.pid}') == before


@pytest.mark.parametrize("value", [True, '123', 123.5, None])
def test_reauthentication_time_is_never_coerced_from_untrusted_claim_types(value):
    identity = __import__('openecon.team_auth', fromlist=['TeamAuth']).TeamAuth(
        'openecon-desktop-test', verifier=lambda token: {**claims('owner'), 'auth_time': value},
    ).verify('signed-token')
    assert identity.auth_time is None and identity.provider == 'google.com'


def begin_link(api, target="password", uid="owner"):
    response = api.client.post(
        "/api/desktop/account-link",
        headers=headers(uid),
        json={"challenge": CHALLENGE, "target": target},
    )
    assert response.status_code == 201
    return response.json()["request_id"]


@pytest.mark.parametrize("target", ["password", "google.com"])
def test_link_is_authenticated_bound_one_use_and_never_issues_token(api, target):
    before = deepcopy(api.store.db.get(f"oe_projects/{api.pid}"))
    request_id = begin_link(api, target, "editor")
    path = f"/api/desktop/account-link/{request_id}"
    assert api.client.get(f"/api/desktop/login/{request_id}", headers=headers("editor")).json() == {
        "kind": "account-link",
        "target": target,
        "uid": "editor",
    }
    assert (
        api.client.post(
            path + "/exchange", headers=headers("editor"), json={"verifier": VERIFIER}
        ).status_code
        == 202
    )
    assert api.client.post(path + "/complete", headers=headers("editor")).json() == {
        "completed": True
    }
    assert api.client.post(path + "/complete", headers=headers("editor")).status_code == 200
    assert (
        api.client.post(
            path + "/exchange", headers=headers("editor"), json={"verifier": "b" * 64}
        ).status_code
        == 403
    )
    result = api.client.post(
        path + "/exchange", headers=headers("editor"), json={"verifier": VERIFIER}
    )
    assert result.json() == {"linked": True, "uid": "editor", "target": target}
    assert (
        api.client.post(
            path + "/exchange", headers=headers("editor"), json={"verifier": VERIFIER}
        ).status_code
        == 410
    )
    api.issuer.assert_not_called()
    assert api.store.db.get(f"oe_projects/{api.pid}") == before


def test_other_uid_cannot_inspect_complete_consume_or_cancel_handoff(api):
    request_id = begin_link(api)
    operations = [
        ("GET", f"/api/desktop/login/{request_id}", None),
        ("POST", f"/api/desktop/account-link/{request_id}/complete", None),
        ("POST", f"/api/desktop/account-link/{request_id}/exchange", {"verifier": VERIFIER}),
        ("DELETE", f"/api/desktop/account-link/{request_id}", None),
    ]
    for method, path, body in operations:
        assert (
            api.client.request(method, path, headers=headers("outsider"), json=body).status_code
            == 403
        )
        assert api.client.request(method, path, json=body).status_code == 401
    assert api.store.db.get(f"oe_desktop_logins/{request_id}")["completed"] is False
    api.issuer.assert_not_called()


def test_link_grants_cannot_be_converted_to_login_or_accept_client_uid(api):
    request_id = begin_link(api)
    assert (
        api.client.post(f"/api/desktop/login/{request_id}/authorize", headers=headers()).status_code
        == 409
    )
    assert (
        api.client.post(
            f"/api/desktop/login/{request_id}/exchange", json={"verifier": VERIFIER}
        ).status_code
        == 409
    )
    for body in [
        {"challenge": CHALLENGE, "target": "password", "uid": "outsider"},
        {"challenge": CHALLENGE, "target": "github.com"},
        {"challenge": "bad", "target": "password"},
    ]:
        assert (
            api.client.post("/api/desktop/account-link", headers=headers(), json=body).status_code
            == 422
        )
    assert (
        api.client.post(
            "/api/desktop/account-link", json={"challenge": CHALLENGE, "target": "password"}
        ).status_code
        == 401
    )
    login_id = begin(api)
    assert api.client.get(f"/api/desktop/login/{login_id}", headers=headers()).json() == {
        "kind": "login"
    }
    assert (
        api.client.post(
            f"/api/desktop/account-link/{login_id}/complete", headers=headers()
        ).status_code
        == 409
    )
    api.issuer.assert_not_called()


def test_link_cancellation_and_expiry_cannot_be_approved_later(api):
    request_id = begin_link(api)
    assert api.client.delete(
        f"/api/desktop/account-link/{request_id}", headers=headers()
    ).json() == {"cancelled": True}
    assert (
        api.client.post(
            f"/api/desktop/account-link/{request_id}/complete", headers=headers()
        ).status_code
        == 410
    )
    request_id = begin_link(api)
    grant = api.store.db.get(f"oe_desktop_logins/{request_id}")
    grant["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    api.store.db.put(f"oe_desktop_logins/{request_id}", grant)
    assert api.client.get(f"/api/desktop/login/{request_id}", headers=headers()).status_code == 410
    assert (
        api.client.post(
            f"/api/desktop/account-link/{request_id}/complete", headers=headers()
        ).status_code
        == 410
    )
    api.issuer.assert_not_called()


def test_link_creation_is_cross_origin_protected_and_has_per_uid_budget(api):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    api.store.db.put(f"oe_limits/account-link-owner-{today}", {"count": 32})
    assert (
        api.client.post(
            "/api/desktop/account-link",
            headers=headers(),
            json={"challenge": CHALLENGE, "target": "password"},
        ).status_code
        == 429
    )
    assert begin_link(api, uid="editor")
    assert (
        api.client.post(
            "/api/desktop/account-link",
            headers={**headers(), "Origin": "https://evil.example"},
            json={"challenge": CHALLENGE, "target": "password"},
        ).status_code
        == 403
    )
