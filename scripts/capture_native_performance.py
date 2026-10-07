"""Capture only one synthetic macOS app's responsible process tree (no UI actions).

Includes WKWebView/GPU XPC processes by macOS responsibility, even when launchd
is their parent. Raw output contains process IDs, roles and resource counters,
never process command lines, code, input content, credentials or other apps.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import platform
import subprocess
import time


class RUsage(C.Structure):
    _fields_ = [("uuid", C.c_uint8 * 16)] + [
        (key, C.c_uint64)
        for key in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


class BSDInfo(C.Structure):
    _fields_ = (
        [
            (key, C.c_uint32)
            for key in (
                "flags",
                "status",
                "xstatus",
                "pid",
                "ppid",
                "uid",
                "gid",
                "ruid",
                "rgid",
                "svuid",
                "svgid",
                "rfu1",
            )
        ]
        + [("comm", C.c_char * 16), ("name", C.c_char * 32)]
        + [(key, C.c_uint32) for key in ("nfiles", "pgid", "pjobc", "e_tdev", "e_tpgid")]
        + [("nice", C.c_int32), ("start_sec", C.c_uint64), ("start_usec", C.c_uint64)]
    )


class Timebase(C.Structure):
    _fields_ = [("numer", C.c_uint32), ("denom", C.c_uint32)]


def capture(binary: Path, output: Path, seconds: float, interval: float):
    if platform.system() != "Darwin":
        raise SystemExit("This capture uses the macOS libproc responsibility contract.")
    lib = C.CDLL("/usr/lib/libproc.dylib")
    system = C.CDLL("/usr/lib/libSystem.B.dylib")
    system.mach_timebase_info.argtypes = [C.POINTER(Timebase)]
    timebase = Timebase()
    if system.mach_timebase_info(C.byref(timebase)) or not timebase.denom:
        raise RuntimeError("Cannot establish the Mach CPU timebase.")
    # libproc CPU counters use Mach ticks, not nanoseconds on Apple Silicon.
    cpu_nanoseconds_per_tick = timebase.numer / timebase.denom
    system.responsibility_get_pid_responsible_for_pid.argtypes = [C.c_int]
    system.responsibility_get_pid_responsible_for_pid.restype = C.c_int
    lib.proc_pidpath.argtypes = [C.c_int, C.c_void_p, C.c_uint32]
    lib.proc_pid_rusage.argtypes = [C.c_int, C.c_int, C.c_void_p]
    lib.proc_pidinfo.argtypes = [C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int]
    binary = binary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    until = time.monotonic() + seconds
    with output.open("a", encoding="utf-8") as stream:
        while time.monotonic() < until:
            tick, epoch = time.monotonic(), time.time()
            pairs = [
                tuple(map(int, row.split()))
                for row in subprocess.check_output(
                    ["ps", "-axo", "pid=,ppid="], text=True
                ).splitlines()
                if row.strip()
            ]
            paths = {}
            for pid, _ in pairs:
                buf = C.create_string_buffer(4096)
                if lib.proc_pidpath(pid, buf, len(buf)) > 0:
                    paths[pid] = buf.value.decode(errors="replace")
            roots = [pid for pid in paths if paths[pid] == str(binary)]
            # A separate identifier/path gives exactly one QA app, never a
            # similarly named human app or another agent's test instance.
            if len(roots) > 1:
                raise RuntimeError("Multiple instances at the exact QA binary path.")
            if roots:
                root = roots[0]
                owned = {root}
                for pid, _ in pairs:
                    if system.responsibility_get_pid_responsible_for_pid(pid) == root:
                        owned.add(pid)
                changed = True
                while changed:
                    changed = False
                    for pid, parent in pairs:
                        if parent in owned and pid not in owned:
                            owned.add(pid)
                            changed = True
                bsd = BSDInfo()
                if lib.proc_pidinfo(root, 3, 0, C.byref(bsd), C.sizeof(bsd)) != C.sizeof(bsd):
                    raise RuntimeError("Cannot establish the native process start time.")
                resources = []
                for pid in sorted(owned):
                    usage = RUsage()
                    if lib.proc_pid_rusage(pid, 2, C.byref(usage)):
                        continue  # exited between enumeration and measurement
                    cpu_ns = (usage.user_time + usage.system_time) * cpu_nanoseconds_per_tick
                    before = previous.get(pid)
                    cpu = (
                        max(0, cpu_ns - before[1]) / ((tick - before[0]) * 1e9) * 100
                        if before
                        else None
                    )
                    previous[pid] = (tick, cpu_ns)
                    name = Path(paths.get(pid, "")).name
                    role = (
                        "native"
                        if pid == root
                        else (
                            "webkit_gpu"
                            if "GPU" in name
                            else "webkit_web"
                            if "WebContent" in name
                            else "webkit_network"
                            if "Networking" in name
                            else "python_runtime_or_worker"
                            if "openecon-runtime" in name or "python" in name.lower()
                            else "owned_helper"
                        )
                    )
                    resources.append(
                        {
                            "pid": pid,
                            "role": role,
                            "rss_bytes": usage.resident_size,
                            "footprint_bytes": usage.phys_footprint,
                            "cpu_percent": cpu,
                        }
                    )
                record = {
                    "epoch_ms": epoch * 1000,
                    "root_pid": root,
                    "native_start_epoch_ms": (bsd.start_sec + bsd.start_usec / 1e6) * 1000,
                    "rss_bytes": sum(r["rss_bytes"] for r in resources),
                    "cpu_percent": sum(r["cpu_percent"] or 0 for r in resources),
                    "cpu_timebase": {"numer": timebase.numer, "denom": timebase.denom},
                    "processes": resources,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            time.sleep(max(0, interval - (time.monotonic() - tick)))
    print(json.dumps({"output": str(output), "capture_finished": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or not 0.1 <= args.interval <= 2:
        parser.error("Use duration (0,3600] and interval [0.1,2].")
    capture(args.binary, args.output, args.seconds, args.interval)


if __name__ == "__main__":
    main()
