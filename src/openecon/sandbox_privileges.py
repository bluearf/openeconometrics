"""Privilege reduction for the managed guest, before any submitted Python.

This is not a process-isolation substitute. The launcher creates the guest's
filesystem, process and network namespaces. Its fixed UID 0 lacks SETUID/GID;
the deployment must separately prove the namespace and copy-on-write boundary.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

# The observed launcher bounding set: KILL, NET_BIND_SERVICE, AUDIT_WRITE.
# These bits cannot be removed without SETPCAP. None may remain permitted,
# effective, inheritable or ambient; NoNewPrivs prevents reacquisition on exec.
ALLOWED_BOUNDING = (1 << 5) | (1 << 10) | (1 << 29)
_ACTIVE = ("CapEff", "CapPrm", "CapInh", "CapAmb")


def capability_state() -> dict[str, int]:
    raw = Path("/proc/self/status").read_text()
    result = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        if key in (*_ACTIVE, "CapBnd", "NoNewPrivs"):
            result[key] = int(value.strip(), 10 if key == "NoNewPrivs" else 16)
    if set(result) != {*_ACTIVE, "CapBnd", "NoNewPrivs"}:
        raise RuntimeError("Guest privilege state is unavailable.")
    return result


def reduce_guest_privileges() -> None:
    """Irreversibly remove guest privileges in the single-threaded bootstrap."""
    if (sys.platform != "linux" or os.getuid() != 0 or os.geteuid() != 0
            or os.getgid() != 0 or os.getegid() != 0 or os.getgroups()
            or os.getpid() != 1 or len(list(Path("/proc/self/task").iterdir())) != 1):
        raise RuntimeError("The managed guest bootstrap identity is invalid.")
    before = capability_state()
    if before["CapBnd"] & ~ALLOWED_BOUNDING:
        raise RuntimeError("The managed guest bounding set is invalid.")

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    libc = ctypes.CDLL(None, use_errno=True)
    # Once set, NoNewPrivs cannot be unset by the guest, including after exec.
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(47, 4, 0, 0, 0) != 0:
        raise RuntimeError("Guest privilege reduction failed.")
    header, data = Header(0x20080522, 0), (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise RuntimeError("Guest privilege reduction failed.")
    after = capability_state()
    if (any(after[key] for key in _ACTIVE) or after["NoNewPrivs"] != 1
            or after["CapBnd"] & ~ALLOWED_BOUNDING):
        raise RuntimeError("Guest privilege reduction could not be verified.")
