"""Fail-closed bootstrap tests; deployed namespace tests remain mandatory."""
from unittest.mock import Mock

import pytest

from openecon import sandbox_privileges as guest


def test_status_requires_all_capability_sets_and_no_new_privs(monkeypatch):
    monkeypatch.setattr(guest.Path, "read_text", lambda self: "CapEff:\t00000000\n")
    with pytest.raises(RuntimeError, match="state is unavailable"):
        guest.capability_state()


@pytest.mark.parametrize("invalid", ["uid", "euid", "gid", "egid", "groups", "pid", "threads"])
def test_bootstrap_refuses_other_identity_or_existing_threads(monkeypatch, invalid):
    monkeypatch.setattr(guest.sys, "platform", "linux")
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(guest.os, name, lambda: 0)
    monkeypatch.setattr(guest.os, "getgroups", lambda: [])
    monkeypatch.setattr(guest.os, "getpid", lambda: 1)
    monkeypatch.setattr(guest.Path, "iterdir", lambda self: iter([object()]))
    if invalid == "threads":
        monkeypatch.setattr(guest.Path, "iterdir", lambda self: iter([object(), object()]))
    else:
        method = "get" + ("groups" if invalid == "groups" else invalid)
        monkeypatch.setattr(guest.os, method, lambda: [0] if invalid == "groups" else 10001)
    libc = Mock()
    monkeypatch.setattr(guest.ctypes, "CDLL", libc)
    with pytest.raises(RuntimeError, match="bootstrap identity is invalid"):
        guest.reduce_guest_privileges()
    libc.assert_not_called()


def test_unexpected_bounding_capability_refused_before_native_calls(monkeypatch):
    monkeypatch.setattr(guest.sys, "platform", "linux")
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(guest.os, name, lambda: 0)
    monkeypatch.setattr(guest.os, "getgroups", lambda: [])
    monkeypatch.setattr(guest.os, "getpid", lambda: 1)
    monkeypatch.setattr(guest.Path, "iterdir", lambda self: iter([object()]))
    monkeypatch.setattr(guest, "capability_state", lambda: {"CapBnd": 1 << 21})
    libc = Mock()
    monkeypatch.setattr(guest.ctypes, "CDLL", libc)
    with pytest.raises(RuntimeError, match="bounding set is invalid"):
        guest.reduce_guest_privileges()
    libc.assert_not_called()


@pytest.mark.parametrize("failure", ["no_new_privs", "ambient", "capset", "effective", "nnp", "bound"])
def test_reduction_stops_when_syscall_or_postcondition_fails(monkeypatch, failure):
    monkeypatch.setattr(guest.sys, "platform", "linux")
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(guest.os, name, lambda: 0)
    monkeypatch.setattr(guest.os, "getgroups", lambda: [])
    monkeypatch.setattr(guest.os, "getpid", lambda: 1)
    monkeypatch.setattr(guest.Path, "iterdir", lambda self: iter([object()]))
    after = {"CapEff": 0, "CapPrm": 0, "CapInh": 0, "CapAmb": 0,
             "CapBnd": guest.ALLOWED_BOUNDING, "NoNewPrivs": 1}
    if failure == "effective":
        after["CapEff"] = 1 << 5
    elif failure == "nnp":
        after["NoNewPrivs"] = 0
    elif failure == "bound":
        after["CapBnd"] |= 1 << 21
    states = iter([{"CapBnd": guest.ALLOWED_BOUNDING}, after])
    monkeypatch.setattr(guest, "capability_state", lambda: next(states))
    libc = Mock()
    libc.prctl.side_effect = [-1] if failure == "no_new_privs" else [0, -1 if failure == "ambient" else 0]
    libc.capset.return_value = -1 if failure == "capset" else 0
    monkeypatch.setattr(guest.ctypes, "CDLL", lambda *args, **kwargs: libc)
    with pytest.raises(RuntimeError, match="privilege reduction"):
        guest.reduce_guest_privileges()
    if failure in {"no_new_privs", "ambient"}:
        libc.capset.assert_not_called()
