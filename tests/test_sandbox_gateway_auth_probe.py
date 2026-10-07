"""Fixed native guest proof; no real network, credentials or cloud resources."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
import urllib.request

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sandbox_gateway_auth_proof", Path(__file__).parents[1] / "scripts/verify_sandbox_broker_live.py")
proof = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proof
_SPEC.loader.exec_module(proof)

ROUTES = ("Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
          "eth0 00000000 010014AC 0003 0 0 0 00000000 0 0 0\n")


def execute_probe(monkeypatch, capsys, outcome=401, routes=ROUTES):
    requests = []

    def read_route(path):
        assert str(path) == "/proc/net/route"
        return routes

    def open_request(request, timeout):
        requests.append((request, timeout))
        if isinstance(outcome, Exception):
            raise outcome
        if outcome >= 400:
            response = Mock()
            response.read.side_effect = AssertionError("Do not read response bodies")
            raise HTTPError(request.full_url, outcome, "fixed", {}, response)
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.side_effect = AssertionError("Do not read response bodies")
        return response

    opener = Mock()
    opener.open.side_effect = open_request
    monkeypatch.setattr(Path, "read_text", read_route)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)
    exec(compile(proof.GUEST_BROKER_AUTH, "<fixed-guest-auth-proof>", "exec"), {})
    observed = json.loads(capsys.readouterr().out)
    return observed, requests


def test_gateway_probe_sends_only_three_fixed_requests_and_reports_booleans(monkeypatch, capsys):
    observed, requests = execute_probe(monkeypatch, capsys)
    assert len(requests) == 3
    assert all(type(value) is bool and value for value in observed.values())
    for request, timeout in requests:
        assert request.full_url == "http://172.20.0.1:8080/execute"
        assert request.data == b"{}" and request.method == "POST" and timeout == 3
        assert request.get_header("Content-type") == "application/json"
    assert set(dict(requests[0][0].header_items())) == {"Content-type"}
    assert set(dict(requests[1][0].header_items())) == {"Content-type", "Authorization"}
    assert set(dict(requests[2][0].header_items())) == {"Content-type", "X-serverless-authorization"}
    assert requests[1][0].get_header("Authorization") == requests[2][0].get_header("X-serverless-authorization")
    case = next(case for case in proof.CASES if case.name == "guest_gateway_broker_auth")
    checks = proof.fixed_output_checks(case, {"status": "ok", "stdout": json.dumps(observed)}, [])
    assert all(checks.values())
    next_case = proof.CASES[proof.CASES.index(case) + 1]
    assert next_case.name == "analysis_after_guest_auth_probe" and next_case.code == proof.ANALYSIS


@pytest.mark.parametrize("outcome", [200, 403, 503, TimeoutError(), URLError("blocked")])
def test_only_explicit_broker_401_proves_authorization_gate(monkeypatch, capsys, outcome):
    observed, requests = execute_probe(monkeypatch, capsys, outcome)
    assert len(requests) == 3 and observed["gateway_discovered"] and observed["probe_completed"]
    assert not any(observed[name] for name in ("missing_auth_denied", "forged_auth_denied", "serverless_only_denied"))


@pytest.mark.parametrize("routes", ["", ROUTES.replace("0003", "0001"),
                                    ROUTES.replace("010014AC", "0100007F"),
                                    ROUTES + "eth1 00000000 010015AC 0003 0 0 0 00000000 0 0 0\n"])
def test_missing_ambiguous_or_loopback_gateway_does_not_probe_other_addresses(monkeypatch, capsys, routes):
    observed, requests = execute_probe(monkeypatch, capsys, routes=routes)
    assert requests == [] and not any(observed.values())


@pytest.mark.parametrize("mutation", ["missing", "truthy", "extra"])
def test_live_report_requires_complete_exact_boolean_proof(mutation):
    observed = {"gateway_discovered": True, "probe_completed": True, "missing_auth_denied": True,
                "forged_auth_denied": True, "serverless_only_denied": True}
    if mutation == "missing":
        observed.pop("probe_completed")
    elif mutation == "truthy":
        observed["forged_auth_denied"] = "true"
    else:
        observed["extra"] = True
    case = next(case for case in proof.CASES if case.name == "guest_gateway_broker_auth")
    checks = proof.fixed_output_checks(case, {"status": "ok", "stdout": json.dumps(observed)}, [])
    assert not all(checks.values())
