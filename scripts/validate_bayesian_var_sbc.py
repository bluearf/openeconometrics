"""Frozen prospective BVAR study driver; registration never fits or samples.

Launch requires an exact reviewed registration SHA. Protocol tests inject only
fake callbacks. Production modules and RNG libraries load only in an authorized
run. Started-but-uncommitted random/fit/draw phases are never repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid

BASE = "b6347f31e2cd38a069ccd3599a5d147dd9266fae"
PIN_SHA = "bffd135550f66e297b65ab070be6171b580377a71061721ded8c346327e455e5"
DRAFT_SHA = "d70bc9b0f7ee507d8e27370c73601c69a81700f6fb6439207cc0b3d001624974"
SCIPY_SHA = "9c4073ff8ed8d46628552431a0928be062d1e1c89203c2d7d0cbb17d5a7c202c"
TOTAL = 1024
REPLICATES = 128
DRAW_COUNT = 128
MIN_FREE = 8 * 1024**3
MAX_DISK = 4 * 1024**3
MAX_FILE = 32 * 1024**2
PHASES = ("random", "inputs", "fit", "draw", "score")
IRREVERSIBLE = frozenset(("random", "fit", "draw"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class ProtocolError(RuntimeError):
    pass


class DiskGuardError(ProtocolError):
    pass


class ComputationError(RuntimeError):
    def __init__(self, message, evidence=None):
        self.evidence = evidence
        super().__init__(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON keys are refused.")
        result[key] = value
    return result


def read_json(path):
    path = pathlib.Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_FILE:
        raise ProtocolError("Symlink/oversized saved protocol state is refused.")
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"Nonfinite JSON {value}")
            ),
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Saved protocol JSON is invalid.") from exc


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Store:
    """Exclusive immutable phase writes; ambiguous commits are read, not repeated."""

    def __init__(self, root, guard, *, link=os.link):
        self.root, self.guard, self.link = pathlib.Path(root).resolve(), guard, link
        self.counter = 0

    def write(self, relative, value, *, replace=False):
        self.guard.check()
        target = self.root / relative
        if target.is_symlink() or self.root not in target.resolve().parents:
            raise ProtocolError("Protocol writes must remain inside their owned directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        temporary = target.parent / f".{target.name}.{os.getpid()}.{self.counter}.partial"
        byte_count, expected = 0, hashlib.sha256()
        try:
            with temporary.open("xb") as handle:
                for block in json.JSONEncoder(
                    sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                ).iterencode(value):
                    encoded = block.encode()
                    byte_count += len(encoded)
                    if byte_count > MAX_FILE:
                        raise ProtocolError("One phase packet exceeds 32 MiB.")
                    expected.update(encoded)
                    handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self.guard.check(additional_bytes=byte_count)
            if replace:
                os.replace(temporary, target)
            else:
                self.link(temporary, target)
            fsync_directory(target.parent)
        except BaseException as exc:
            if hasattr(self.guard, "note_write"):
                self.guard.note_write(temporary)
                self.guard.note_write(target)
            # os.link/fsync may have committed before raising. Never rerun callback.
            if (
                target.exists()
                and file_sha(target) == expected.hexdigest()
                and read_json(target) == value
            ):
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    # Retain the committed packet, but honor deliberate interruption.
                    raise
                return {"sha256": file_sha(target), "ambiguous_commit_readback": True}
            raise
        if temporary.exists():
            temporary.unlink()  # The same inode/value is retained by the final link.
        if hasattr(self.guard, "note_write"):
            self.guard.note_write(temporary)
            self.guard.note_write(target)
        if file_sha(target) != expected.hexdigest() or read_json(target) != value:
            raise ProtocolError("Committed packet failed exact readback.")
        return {"sha256": expected.hexdigest(), "ambiguous_commit_readback": False}


def runtime_pin():
    packages = {}
    for name in ("torch", "pandas", "numpy", "scipy", "pydantic", "pydantic_core"):
        distribution = importlib.metadata.distribution(name)
        records = [
            distribution.locate_file(p)
            for p in (distribution.files or ())
            if str(p).endswith(".dist-info/RECORD")
        ]
        installed = {
            str(pathlib.Path(distribution.locate_file(p)).resolve()): file_sha(
                distribution.locate_file(p)
            )
            for p in (distribution.files or ())
            if pathlib.Path(p).suffix in {".py", ".pyc", ".so", ".dylib", ".pyd"}
            and pathlib.Path(distribution.locate_file(p)).is_file()
        }
        packages[name] = {
            "version": distribution.version,
            "record_files": {str(p): file_sha(p) for p in records},
            "installed_source_and_native_files": installed,
        }
    return {
        "python": platform.python_version(),
        "sys_version": sys.version,
        "executable": sys.executable,
        "executable_sha256": file_sha(pathlib.Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "native_workers_declared": 1,
        "native_threads_declared": 1,
        "execution": "CPU float64; actual thread/dtype/device state recorded at launch",
        "invocation_policy": invocation_policy(),
    }


def invocation_policy():
    return {
        "isolated": bool(sys.flags.isolated),
        "ignore_PYTHON_environment": bool(sys.flags.ignore_environment),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "native_thread_declarations": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "local_source_insertion": str(ROOT / "src"),
        "actual_Torch_default_dtype_device": "deferred first worker receipt before any fit/RNG",
        "Accelerate_actual_thread_count": "unobserved; VECLIB declaration is policy only",
    }


def enforce_invocation_policy(policy):
    if (
        policy["isolated"] is not True
        or policy["ignore_PYTHON_environment"] is not True
        or policy["dont_write_bytecode"] is not True
    ):
        raise ProtocolError(
            "Registration/launch requires the existing interpreter's explicit -I -B invocation."
        )
    if policy["native_thread_declarations"] != {name: "1" for name in THREAD_VARIABLES}:
        raise ProtocolError(
            "Declare OMP/MKL/OPENBLAS/VECLIB=1 before the interpreter/native imports."
        )


def bind_native_state(registration, ledger, state):
    """Validate later native capture; protocol tests supply labeled fake metadata only."""
    if (
        any(
            type(state.get(key)) is not int or state.get(key) != 1
            for key in ("native_threads", "interop_threads")
        )
        or state.get("default_device") != "cpu"
    ):
        raise ProtocolError("Actual authorized Torch native/inter-op/device policy disagrees.")
    if state.get("default_dtype") not in {
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }:
        raise ProtocolError("The first actual Torch floating default must be explicitly captured.")
    if (
        state.get("torch_version") != registration["runtime"]["packages"]["torch"]["version"]
        or state.get("invocation_policy") != registration["runtime"]["invocation_policy"]
    ):
        raise ProtocolError("Actual native version or pre-import invocation policy changed.")
    first = next(
        (e["actual_native_state"] for e in ledger.entries if e["event"] == "worker_started"), None
    )
    if first is not None and first != state:
        raise ProtocolError(
            "Restart native defaults/policy differ from the first bound pre-effect state."
        )
    return digest(state)


class Guard:
    def __init__(self, root, registration, *, free=shutil.disk_usage):
        self.root, self.registration, self.free = pathlib.Path(root).resolve(), registration, free
        self.stamps = {}
        self.sizes = {str(p): p.stat().st_size for p in self.root.rglob("*") if p.is_file()}
        self.check(full=True)

    def note_write(self, path):
        path = pathlib.Path(path)
        if path.exists():
            self.sizes[str(path)] = path.stat().st_size
        else:
            self.sizes.pop(str(path), None)

    def check(self, additional_bytes=0, *, full=False):
        resources = self.registration["resources"]
        if full:
            self.sizes = {str(p): p.stat().st_size for p in self.root.rglob("*") if p.is_file()}
        if self.free(self.root).free < resources["min_free_bytes"]:
            raise DiskGuardError("Actual free disk fell below the registered 8 GiB guard.")
        # Include existing archives/checkpoints, failed temps and ledger copies.
        size = sum(self.sizes.values())
        if size + additional_bytes > resources["max_disk_bytes"]:
            raise DiskGuardError("The complete study exceeds its 4 GiB logical reservation.")
        for path, sha in self.registration["source_files"].items():
            item = pathlib.Path(path)
            stat = item.stat()
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if full or self.stamps.get(path) != stamp:
                if item.is_symlink() or file_sha(item) != sha:
                    raise ProtocolError(f"Registered source drift: {path}")
                self.stamps[path] = stamp
        if full and runtime_pin() != self.registration["runtime"]:
            raise ProtocolError("Registered interpreter/runtime drift.")


class Ledger:
    def __init__(self, store, registration_sha):
        self.store, self.registration_sha = store, registration_sha
        self.entries, self.previous = [], registration_sha
        directory = store.root / "ledger"
        for i, path in enumerate(sorted(directory.glob("*.json"))):
            if path.name != f"{i:08d}.json":
                raise ProtocolError("Ledger sequence has a gap or duplicate.")
            entry = read_json(path)
            payload = {k: v for k, v in entry.items() if k != "entry_sha256"}
            if (
                entry.get("sequence") != i
                or entry.get("previous_sha256") != self.previous
                or entry.get("registration_sha256") != registration_sha
                or digest(payload) != entry.get("entry_sha256")
            ):
                raise ProtocolError("Append-only ledger chain is corrupt.")
            self.entries.append(entry)
            self.previous = entry["entry_sha256"]

    def append(self, event, **data):
        if (
            set(data) & {"sequence", "previous_sha256", "event", "entry_sha256"}
            or data.get("registration_sha256", self.registration_sha) != self.registration_sha
        ):
            raise ProtocolError("Ledger identity fields cannot be overridden.")
        payload = {
            "sequence": len(self.entries),
            "previous_sha256": self.previous,
            "registration_sha256": self.registration_sha,
            "event": event,
            **data,
        }
        entry = payload | {"entry_sha256": digest(payload)}
        self.store.write(f"ledger/{len(self.entries):08d}.json", entry)
        self.entries.append(entry)
        self.previous = entry["entry_sha256"]
        return entry


def coordinates(ordinal):
    if type(ordinal) is not int or not 0 <= ordinal < TOTAL:
        raise ProtocolError("Only the fixed 1,024 registered ordinals are allowed.")
    return ordinal // REPLICATES, ordinal % REPLICATES


def compact_score(score):
    return {key: score[key] for key in ("rank_bins", "coverage", "point_mass_rank_bins")}


def aggregate_row(outcome):
    """Keep full originals on disk; final aggregation needs bounded counters only."""
    row = {key: outcome[key] for key in ("ordinal", "cell", "replicate", "status")}
    row["original_outcome_sha256"] = digest(outcome)
    if outcome["status"] == "success":
        row["score"] = compact_score(outcome["score"])
    return row


def validate_registration(registration):
    draft = registration["draft"]
    if (
        registration.get("total_cases") != TOTAL
        or registration.get("replicates_per_cell") != REPLICATES
        or registration.get("posterior_draws") != DRAW_COUNT
    ):
        raise ProtocolError("The fixed 1,024 denominator and 128×128 design cannot change.")
    if digest(draft) != registration["draft_canonical_sha256"] or len(draft["cells"]) != 8:
        raise ProtocolError("Prospective design digest/cell count disagrees.")
    if (
        draft["family"]["total_gates"] != 2616
        or draft["accepted_rank_bin_counts"] != list(range(2, 36))
        or draft["accepted_exact_coverage_counts"] != list(range(108, 129))
    ):
        raise ProtocolError("Reviewed fixed family/acceptance sets cannot change.")
    for i, cell in enumerate(draft["cells"]):
        if (
            type(cell["cell"]) is not int
            or cell["cell"] != i
            or cell["n_response"] != cell["n_original"] - cell["p"]
        ):
            raise ProtocolError("Prospective cell ordering/dimensions disagree.")
    if (
        registration["pause_after_committed"] != 64
        or type(registration["pause_after_committed"]) is not int
    ):
        raise ProtocolError(
            "The one planned same-PID pause remains fixed after 64 committed ordinals."
        )
    return registration


class Engine:
    def __init__(
        self,
        root,
        registration,
        adapter,
        *,
        guard=None,
        stop=lambda: signal.raise_signal(signal.SIGSTOP),
    ):
        self.root, self.registration, self.adapter = (
            pathlib.Path(root).resolve(),
            validate_registration(registration),
            adapter,
        )
        self.registration_sha = file_sha(self.root / "registration.json")
        self.guard = guard or Guard(root, registration)
        self.store = Store(root, self.guard)
        self.ledger = Ledger(self.store, self.registration_sha)
        completed = [e["ordinal"] for e in self.ledger.entries if e["event"] == "case_committed"]
        if completed != list(range(len(completed))):
            raise ProtocolError("Committed cases must be one fixed contiguous prefix.")
        planned = [e for e in self.ledger.entries if e["event"] == "planned_pause"]
        continued = [e for e in self.ledger.entries if e["event"] == "same_pid_continued"]
        if (
            len(planned) > 1
            or len(continued) > 1
            or (planned and not continued)
            or (continued and not planned)
        ):
            raise ProtocolError(
                "A planned same-PID pause cannot be substituted by another process."
            )
        if continued and (
            planned[0]["pid"] != continued[0]["pid"]
            or planned[0]["entry_sha256"] != continued[0]["pause_entry_sha256"]
            or planned[0]["completed"] != 64
            or continued[0]["completed"] != 64
        ):
            raise ProtocolError(
                "The original one-pause PID/prefix/continuation provenance disagrees."
            )
        progress = self.root / "progress.json"
        if progress.exists():
            value = read_json(progress)
            revisions = [e for e in self.ledger.entries if e["event"] == "progress"]
            if not revisions:
                raise ProtocolError("Atomic progress differs from its append-only revision.")
            latest = {
                k: v
                for k, v in revisions[-1].items()
                if k not in {"sequence", "previous_sha256", "entry_sha256", "event"}
            }
            if value != latest:
                if file_sha(progress) != latest["previous_progress_sha256"]:
                    raise ProtocolError("Atomic progress differs from its append-only revision.")
                # A committed revision may precede an interrupted snapshot replacement.
                self.store.write("progress.json", latest, replace=True)
        elif any(e["event"] == "progress" for e in self.ledger.entries):
            latest = {
                k: v
                for k, v in next(
                    e for e in reversed(self.ledger.entries) if e["event"] == "progress"
                ).items()
                if k not in {"sequence", "previous_sha256", "entry_sha256", "event"}
            }
            if latest["previous_progress_sha256"] is not None:
                raise ProtocolError("An original progress snapshot was removed.")
            self.store.write("progress.json", latest, replace=True)
        self.stop = stop

    def _envelope(self, ordinal, phase, parents, data):
        cell, replicate = coordinates(ordinal)
        return {
            "registration_sha256": self.registration_sha,
            "ordinal": ordinal,
            "cell": cell,
            "replicate": replicate,
            "phase": phase,
            "parents": parents,
            "data": data,
            "data_sha256": digest(data),
        }

    def _marker(self, ordinal, phase, parents):
        relative = f"cases/{ordinal:04d}/{phase}.started.json"
        path = self.root / relative
        value = read_json(path)
        pid = value.get("data", {}).get("pid")
        if (
            type(pid) is not int
            or pid <= 0
            or value != self._envelope(ordinal, phase + "_started", parents, {"pid": pid})
        ):
            raise ProtocolError("Original phase marker identity/parents disagree.")
        starts = [
            e
            for e in self.ledger.entries
            if e["event"] == "phase_started"
            and e.get("ordinal") == ordinal
            and e.get("phase") == phase
        ]
        sha = file_sha(path)
        if starts and (len(starts) != 1 or starts[0]["marker_sha256"] != sha):
            raise ProtocolError("Original phase marker differs from its append-only ledger.")
        if not starts:
            self.ledger.append(
                "phase_started",
                ordinal=ordinal,
                phase=phase,
                marker_sha256=sha,
                recovered_ambiguous_commit=True,
            )
        return sha

    def _saved(self, relative, ordinal, phase, parents, *, ambiguous=True):
        path = self.root / relative
        value = read_json(path)
        if value != self._envelope(ordinal, phase, parents, value.get("data")):
            raise ProtocolError("Committed phase identity, parents or payload digest disagree.")
        self._marker(ordinal, phase, parents)
        sha = file_sha(path)
        commits = [
            e
            for e in self.ledger.entries
            if e["event"] == "phase_committed"
            and e.get("ordinal") == ordinal
            and e.get("phase") == phase
        ]
        if commits and (len(commits) != 1 or commits[0]["sha256"] != sha):
            raise ProtocolError("Original phase differs from its append-only ledger.")
        if not commits:
            self.ledger.append(
                "phase_committed",
                ordinal=ordinal,
                phase=phase,
                sha256=sha,
                ambiguous_commit_readback=ambiguous,
            )
        return value["data"], sha

    def case(self, ordinal):
        cell_id, replicate = coordinates(ordinal)
        completed = [e["ordinal"] for e in self.ledger.entries if e["event"] == "case_committed"]
        if ordinal > len(completed):
            raise ProtocolError("Original ordinals cannot be skipped or selected.")
        prefix = f"cases/{ordinal:04d}"
        final = self.root / prefix / "outcome.json"
        if final.exists():
            outcome = read_json(final)
            if (
                any(type(outcome.get(k)) is not int for k in ("ordinal", "cell", "replicate"))
                or outcome.get("ordinal") != ordinal
                or outcome.get("cell") != cell_id
                or outcome.get("replicate") != replicate
                or outcome.get("registration_sha256") != self.registration_sha
                or outcome.get("status") not in {"success", "computational_failure"}
            ):
                raise ProtocolError("Saved original outcome identity disagrees.")
            if list(outcome.get("phase_files", {})) != sorted(
                outcome.get("phase_files", {})
            ) or set(outcome.get("phase_files", {})) - set(PHASES):
                raise ProtocolError("Original phase inventory is invalid.")
            for phase, sha in outcome.get("phase_files", {}).items():
                if file_sha(self.root / prefix / f"{phase}.json") != sha:
                    raise ProtocolError("Saved original phase digest disagrees.")
            commits = [
                e
                for e in self.ledger.entries
                if e["event"] == "case_committed" and e["ordinal"] == ordinal
            ]
            if commits and (len(commits) != 1 or commits[0]["outcome_sha256"] != file_sha(final)):
                raise ProtocolError("Original outcome differs from its append-only ledger.")
            if outcome["status"] == "success":
                saved, links = {}, {}
                for phase in PHASES:
                    saved[phase], links[phase] = self._saved(
                        f"{prefix}/{phase}.json", ordinal, phase, dict(links)
                    )
                if outcome["phase_files"] != links:
                    raise ProtocolError(
                        "Successful outcome must retain the complete original phase inventory."
                    )
                self.adapter.verify(self.registration["draft"]["cells"][cell_id], replicate, saved)
                if outcome["score"] != compact_score(saved["score"]) or outcome.get(
                    "full_score_sha256"
                ) != digest(saved["score"]):
                    raise ProtocolError("Original score differs from its cached phase.")
            else:
                phase = outcome.get("failure_phase")
                if phase not in {*PHASES, "verify"}:
                    raise ProtocolError("Original failure phase is invalid.")
                count = len(PHASES) if phase == "verify" else PHASES.index(phase)
                # A callback/write failure can occur after the phase committed.
                allowed = [set(PHASES[:count]), set(PHASES[: min(count + 1, len(PHASES))])]
                if set(outcome["phase_files"]) not in allowed:
                    raise ProtocolError(
                        "Original failure must retain a complete committed phase prefix."
                    )
                links = {}
                for name in PHASES:
                    if name in outcome["phase_files"]:
                        _, links[name] = self._saved(
                            f"{prefix}/{name}.json", ordinal, name, dict(links)
                        )
                if outcome["phase_files"] != links:
                    raise ProtocolError("Original failure phase parents disagree.")
            if not commits:
                self.ledger.append(
                    "case_committed",
                    ordinal=ordinal,
                    status=outcome["status"],
                    outcome_sha256=file_sha(final),
                    recovered_ambiguous_commit=True,
                )
            return outcome
        if any(
            e["event"] == "case_committed" and e.get("ordinal") == ordinal
            for e in self.ledger.entries
        ):
            raise ProtocolError(
                "A ledger-committed original outcome is missing; no repeated effect is allowed."
            )
        cell, data, parents = self.registration["draft"]["cells"][cell_id], {}, {}
        for phase in PHASES:
            relative = f"{prefix}/{phase}.json"
            marker = f"{prefix}/{phase}.started.json"
            if (self.root / relative).exists():
                if not (self.root / marker).exists():
                    raise ProtocolError("Completed phase has no original start marker.")
                data[phase], parents[phase] = self._saved(relative, ordinal, phase, dict(parents))
                continue
            self.guard.check()
            if any(
                e["event"] == "phase_committed"
                and e.get("ordinal") == ordinal
                and e.get("phase") == phase
                for e in self.ledger.entries
            ):
                raise ProtocolError(
                    "A ledger-committed phase is missing; no repeated effect is allowed."
                )
            if (self.root / marker).exists() and phase in IRREVERSIBLE:
                self._marker(ordinal, phase, dict(parents))
                return self._failure(
                    ordinal,
                    parents,
                    phase,
                    "interrupted_before_phase_commit",
                    {
                        "marker_sha256": file_sha(self.root / marker),
                        "partials": sorted(p.name for p in (self.root / prefix).glob(".*.partial")),
                    },
                )
            if not (self.root / marker).exists():
                self.store.write(
                    marker,
                    self._envelope(
                        ordinal, phase + "_started", dict(parents), {"pid": os.getpid()}
                    ),
                )
                self.ledger.append(
                    "phase_started",
                    ordinal=ordinal,
                    phase=phase,
                    marker_sha256=file_sha(self.root / marker),
                )
            else:
                self._marker(ordinal, phase, dict(parents))
            try:
                result = self.adapter.phase(phase, cell, replicate, data)
                packet = self._envelope(ordinal, phase, dict(parents), result)
                committed = self.store.write(relative, packet)
                data[phase], parents[phase] = self._saved(
                    relative,
                    ordinal,
                    phase,
                    dict(parents),
                    ambiguous=committed["ambiguous_commit_readback"],
                )
            except ProtocolError:
                raise
            except Exception as exc:
                return self._failure(
                    ordinal,
                    parents,
                    phase,
                    type(exc).__name__,
                    {"message": str(exc), "evidence": getattr(exc, "evidence", None)},
                )
        try:
            self.adapter.verify(cell, replicate, data)
        except ProtocolError:
            raise
        except Exception as exc:
            return self._failure(
                ordinal,
                parents,
                "verify",
                type(exc).__name__,
                {"message": str(exc), "evidence": getattr(exc, "evidence", None)},
            )
        outcome = {
            "ordinal": ordinal,
            "cell": cell_id,
            "replicate": replicate,
            "registration_sha256": self.registration_sha,
            "status": "success",
            "phase_files": parents,
            "score": compact_score(data["score"]),
            "full_score_sha256": digest(data["score"]),
        }
        self.store.write(f"{prefix}/outcome.json", outcome)
        self.ledger.append(
            "case_committed", ordinal=ordinal, status="success", outcome_sha256=file_sha(final)
        )
        return outcome

    def _failure(self, ordinal, parents, phase, code, evidence):
        cell, replicate = coordinates(ordinal)
        result = {
            "ordinal": ordinal,
            "cell": cell,
            "replicate": replicate,
            "registration_sha256": self.registration_sha,
            "status": "computational_failure",
            "phase_files": parents,
            "failure_phase": phase,
            "failure_code": code,
            "failure_evidence": evidence,
        }
        relative = f"cases/{ordinal:04d}/outcome.json"
        self.store.write(relative, result)
        self.ledger.append(
            "case_committed",
            ordinal=ordinal,
            status=result["status"],
            outcome_sha256=file_sha(self.root / relative),
        )
        return result

    def progress(self, outcomes, status="running"):
        payload = {
            "registration_sha256": self.registration_sha,
            "pid": os.getpid(),
            "run_id": self.registration["run_id"],
            "fixed_denominator": TOTAL,
            "committed_ordinals": sorted(outcomes),
            "completed": len(outcomes),
            "computational_failures": sum(v["status"] != "success" for v in outcomes.values()),
            "fit_started": sum(
                e["event"] == "phase_started" and e.get("phase") == "fit"
                for e in self.ledger.entries
            ),
            "draw_started": sum(
                e["event"] == "phase_started" and e.get("phase") == "draw"
                for e in self.ledger.entries
            ),
            "status": status,
            "acceptance": False,
            "ledger_sha256": self.ledger.previous,
            "previous_progress_sha256": file_sha(self.root / "progress.json")
            if (self.root / "progress.json").exists()
            else None,
        }
        self.ledger.append("progress", **payload)
        self.store.write("progress.json", payload, replace=True)

    def run(self, *, limit=TOTAL):
        if type(limit) is not int or not 0 <= limit <= TOTAL:
            raise ProtocolError("Protocol-only limit must lie in the fixed original denominator.")
        outcomes = {}
        for ordinal in range(min(limit, TOTAL)):
            outcomes[ordinal] = aggregate_row(self.case(ordinal))
            self.progress(outcomes)
            pause = self.registration["pause_after_committed"]
            if len(outcomes) == pause and not any(
                e["event"] == "planned_pause" for e in self.ledger.entries
            ):
                inventory = {
                    str(p.relative_to(self.root)): file_sha(p)
                    for p in self.root.rglob("*")
                    if p.is_file() and p.name != "progress.json"
                }
                event = self.ledger.append(
                    "planned_pause",
                    pid=os.getpid(),
                    completed=len(outcomes),
                    inventory=inventory,
                    source_sha256=digest(self.registration["source_files"]),
                )
                self.progress(outcomes, "paused_same_pid")
                paused_progress = file_sha(self.root / "progress.json")
                self.ledger.append(
                    "pause_ready",
                    pid=os.getpid(),
                    progress_sha256=paused_progress,
                    pause_entry_sha256=event["entry_sha256"],
                )
                self.stop()
                self.guard.check(full=True)
                if os.getpid() != event["pid"]:
                    raise ProtocolError("Planned pause resumed in a different process.")
                if file_sha(self.root / "progress.json") != paused_progress:
                    raise ProtocolError("Paused progress was modified externally.")
                for relative, sha in inventory.items():
                    if file_sha(self.root / relative) != sha:
                        raise ProtocolError("Committed inventory changed during pause.")
                self.ledger.append(
                    "same_pid_continued",
                    pid=os.getpid(),
                    completed=len(outcomes),
                    pause_entry_sha256=event["entry_sha256"],
                )
        if limit < TOTAL:
            return {
                "complete": False,
                "acceptance": False,
                "fixed_denominator": TOTAL,
                "completed": len(outcomes),
            }
        result = aggregate(outcomes, self.registration)
        if (self.root / "final.json").exists():
            if read_json(self.root / "final.json") != result:
                raise ProtocolError("Saved final differs from all original outcomes.")
            commits = [e for e in self.ledger.entries if e["event"] == "final_committed"]
            if len(commits) > 1 or (
                commits and commits[0]["sha256"] != file_sha(self.root / "final.json")
            ):
                raise ProtocolError("Saved final differs from its append-only ledger.")
            if not commits:
                self.ledger.append(
                    "final_committed",
                    sha256=file_sha(self.root / "final.json"),
                    acceptance=result["acceptance"],
                    recovered_ambiguous_commit=True,
                )
            self.progress(outcomes, "complete")
            return result
        if any(e["event"] == "final_committed" for e in self.ledger.entries):
            raise ProtocolError("The ledger-committed final was removed.")
        self.store.write("final.json", result)
        self.ledger.append(
            "final_committed",
            sha256=file_sha(self.root / "final.json"),
            acceptance=result["acceptance"],
        )
        self.progress(outcomes, "complete")
        return result


def aggregate(outcomes, registration):
    if any(type(k) is not int for k in outcomes) or set(outcomes) != set(range(TOTAL)):
        raise ProtocolError(
            "Final aggregation requires all 1,024 original ordinals, including failures."
        )
    if any(v.get("status") not in {"success", "computational_failure"} for v in outcomes.values()):
        raise ProtocolError(
            "Every original outcome has an explicit success or computational failure status."
        )
    failures = sum(v.get("status") != "success" for v in outcomes.values())
    cells, gates, controls = [], [], []
    for cell in registration["draft"]["cells"]:
        rows = [outcomes[cell["cell"] * REPLICATES + r] for r in range(REPLICATES)]
        for r, value in enumerate(rows):
            if (
                any(type(value.get(k)) is not int for k in ("ordinal", "cell", "replicate"))
                or value.get("ordinal") != cell["cell"] * REPLICATES + r
                or value.get("cell") != cell["cell"]
                or value.get("replicate") != r
            ):
                raise ProtocolError("Ordinal/cell/replicate identity changed or was selected.")
        valid = [v["score"] for v in rows if v["status"] == "success"]
        for score in valid:
            if len(score["rank_bins"]) != cell["rank_targets"] or any(
                type(v) is not int or not 0 <= v < 8 for v in score["rank_bins"]
            ):
                raise ProtocolError("All fixed rank targets require explicit original bins.")
            if len(score["coverage"]) != cell["exact_coverage_targets"] or any(
                type(v) is not bool for v in score["coverage"]
            ):
                raise ProtocolError("All fixed coverage targets require explicit booleans.")
            if len(score["point_mass_rank_bins"]) != (cell["m"] * cell["p"] + 1) * cell["m"]:
                raise ProtocolError("Point-mass control must retain every coefficient target.")
            if any(type(v) is not int or not 0 <= v < 8 for v in score["point_mass_rank_bins"]):
                raise ProtocolError("Point-mass control bins are invalid.")
        cell_failures = REPLICATES - len(valid)
        for target in range(cell["rank_targets"]):
            for bin_index in range(8):
                count = sum(v["rank_bins"][target] == bin_index for v in valid)
                gates.append(
                    {
                        "cell": cell["cell"],
                        "kind": "rank",
                        "target": target,
                        "bin": bin_index,
                        "count": count,
                        "denominator": REPLICATES,
                        "null_valid": cell_failures == 0,
                        "passed": cell_failures == 0
                        and count in registration["draft"]["accepted_rank_bin_counts"],
                    }
                )
        for target in range(cell["exact_coverage_targets"]):
            count = sum(v["coverage"][target] for v in valid)
            gates.append(
                {
                    "cell": cell["cell"],
                    "kind": "coverage",
                    "target": target,
                    "count": count,
                    "denominator": REPLICATES,
                    "null_valid": cell_failures == 0,
                    "passed": cell_failures == 0
                    and count in registration["draft"]["accepted_exact_coverage_counts"],
                }
            )
        rejected = False
        counts = []
        for target in range((cell["m"] * cell["p"] + 1) * cell["m"]):
            bins = [sum(v["point_mass_rank_bins"][target] == b for v in valid) for b in range(8)]
            counts.append(bins)
            rejected |= cell_failures == 0 and any(
                c not in registration["draft"]["accepted_rank_bin_counts"] for c in bins
            )
        controls.append(
            {
                "cell": cell["cell"],
                "point_mass_rejected": rejected,
                "all_coefficient_bin_counts": counts,
            }
        )
        cells.append(
            {
                "cell": cell["cell"],
                "denominator": REPLICATES,
                "computational_failures": cell_failures,
            }
        )
    if len(gates) != 2616:
        raise ProtocolError("The complete fixed family changed.")
    acceptance = (
        failures == 0
        and all(v["passed"] for v in gates)
        and all(v["point_mass_rejected"] for v in controls)
    )
    return {
        "schema": "openecon.bvar_sbc.final.v1",
        "complete": True,
        "fixed_denominator": TOTAL,
        "computational_failures": failures,
        "acceptance": acceptance,
        "cells": cells,
        "gates": gates,
        "negative_controls": controls,
        "original_outcome_sha256": {
            str(k): v.get("original_outcome_sha256", digest(v)) for k, v in outcomes.items()
        },
        "scope": "prospective prior-predictive calibration only; no vendor or blanket method parity",
        "exit_code": 0 if acceptance else (2 if failures else 1),
    }


def jsonable(value):
    """Failure evidence retains nonfinite values as explicit typed tokens."""
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "nonfinite_float64": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ProtocolError(f"Unsupported raw evidence primitive: {type(value).__name__}")


class PhysicsAdapter:
    """Actual approved science, loaded only after a reviewed launch declaration."""

    def __init__(self, registration):
        policy = invocation_policy()
        enforce_invocation_policy(policy)
        if policy != registration["runtime"]["invocation_policy"]:
            raise ProtocolError("Native imports require the exact registered interpreter policy.")
        import torch

        # Configure Torch immediately, before importing local tensor code.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise ProtocolError("Actual Torch native/inter-op threads must equal one.")
        if str(torch.get_default_device()) != "cpu":
            raise ProtocolError("The frozen native sampler requires an actual CPU default device.")
        defaults_before_import = (str(torch.get_default_dtype()), str(torch.get_default_device()))

        import numpy as np
        import pandas as pd
        import scipy.stats as stats

        # -I supplies isolation; this exact source path is the only local insertion.
        if "openecon" in sys.modules:
            raise ProtocolError(
                "OpenEconometrics must not be imported before the native policy boundary."
            )
        sys.path.insert(0, str(ROOT / "src"))
        import openecon
        from openecon.econometrics.bayesian_var import api, draws
        from openecon.econometrics.bayesian_var.posterior import raw, restore

        if not pathlib.Path(openecon.__file__).resolve().is_relative_to(ROOT / "src"):
            raise ProtocolError("The imported production source is outside the pinned worktree.")
        if defaults_before_import != (
            str(torch.get_default_dtype()),
            str(torch.get_default_device()),
        ):
            raise ProtocolError("Local imports changed the captured Torch dtype/device defaults.")
        self.np, self.pd, self.stats, self.torch = np, pd, stats, torch
        self.api, self.draw_module, self.raw, self.restore = api, draws, raw, restore
        self.packets = {}
        self.launch_state = {
            "torch_version": str(torch.__version__),
            "native_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(torch.get_default_device()),
            "invocation_policy": invocation_policy(),
            "numpy_SciPy_Accelerate": {
                "actual_thread_count": None,
                "observation": "unobserved; installed controller has no Accelerate getter",
                "VECLIB_MAXIMUM_THREADS_declared": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            },
        }

    def phase(self, phase, cell, replicate, saved):
        return getattr(self, phase)(cell, replicate, saved)

    def random(self, cell, replicate, saved):
        np, stats = self.np, self.stats
        trace = []

        class RecordingGenerator(np.random.Generator):
            def normal(self, *args, **kwargs):
                value = super().normal(*args, **kwargs)
                trace.append(
                    {
                        "kind": "normal",
                        "args": jsonable(args),
                        "kwargs": jsonable(kwargs),
                        "values": jsonable(value),
                    }
                )
                return value

            def chisquare(self, *args, **kwargs):
                value = super().chisquare(*args, **kwargs)
                trace.append(
                    {
                        "kind": "chisquare",
                        "args": jsonable(args),
                        "kwargs": jsonable(kwargs),
                        "values": jsonable(value),
                    }
                )
                return value

        seed = 202610090000 + cell["cell"] * 1024 + replicate
        rng = RecordingGenerator(np.random.PCG64(seed))
        state_before = jsonable(rng.bit_generator.state)
        result = {
            "truth_seed": seed,
            "posterior_seed": 202610100000 + cell["cell"] * 1024 + replicate,
            "axen_calls": trace,
            "pcg64_state_before": state_before,
        }
        try:
            prior, m, k = cell["prior"], cell["m"], cell["m"] * cell["p"] + 1
            sigma = stats.invwishart.rvs(
                df=prior["degrees_of_freedom"],
                scale=np.asarray(prior["innovation_scale"]),
                size=1,
                random_state=rng,
            )
            result["truth_sigma"] = sigma.tolist()
            result["coefficient_z"] = rng.standard_normal((k, m)).tolist()
            result["dgp_z"] = rng.standard_normal((cell["n_response"], m)).tolist()
            result["held_out_z"] = rng.standard_normal(m).tolist()
            result["tie_uniforms"] = rng.random(cell["rank_targets"]).tolist()
            result["rank_uniforms"] = rng.random(cell["rank_targets"]).tolist()
            result["pcg64_state_after"] = jsonable(rng.bit_generator.state)
            canonical(result)  # Nonfinite numeric inputs are a failure, never normalized.
            return result
        except Exception as exc:
            result["pcg64_state_after"] = jsonable(rng.bit_generator.state)
            raise ComputationError(
                "Independent prior/DGP primitives failed.", jsonable(result)
            ) from exc

    def inputs(self, cell, replicate, saved):
        np = self.np
        source, prior, m, p = saved["random"], cell["prior"], cell["m"], cell["p"]
        evidence = {"original_random_inputs": source}
        try:
            sigma = np.asarray(source["truth_sigma"], dtype=np.float64)
            b = (
                np.asarray(prior["mean"])
                + np.linalg.cholesky(prior["row_scale"])
                @ np.asarray(source["coefficient_z"])
                @ np.linalg.cholesky(sigma).T
            )
            levels = list(np.asarray(cell["fixed_conditioning_rows"], dtype=np.float64))
            c = np.linalg.cholesky(sigma)
            for z in source["dgp_z"]:
                x = np.concatenate([np.ones(1)] + levels[-p:][::-1])
                y = x @ b + c @ np.asarray(z)
                levels.append(y)
                if not np.isfinite(y).all():
                    raise ComputationError(
                        "Unstable original DGP overflowed; no clipping/resampling.",
                        jsonable({"B": b, "Sigma": sigma, "levels_prefix": levels}),
                    )
            x = np.concatenate([np.ones(1)] + levels[-p:][::-1])
            held = x @ b + c @ np.asarray(source["held_out_z"])
            companion = np.zeros((m * p, m * p))
            companion[:m] = np.concatenate(
                [b[1 + j * m : 1 + (j + 1) * m].T for j in range(p)], axis=1
            )
            companion[m:, :-m] = np.eye(m * (p - 1))
            radius = float(abs(np.linalg.eigvals(companion)).max())
            result = {
                "truth_B": b.tolist(),
                "truth_Sigma": sigma.tolist(),
                "levels": np.asarray(levels).tolist(),
                "periods": list(range(cell["n_original"])),
                "held_out_design": x.tolist(),
                "held_out_y": held.tolist(),
                "truth_spectral_radius": radius,
                "truth_stable": radius < 1,
                "series": [f"y{i}" for i in range(m)],
                "time": "period",
                "period_unit": "unit",
            }
            evidence["constructed_inputs"] = jsonable(result)
            canonical(result)
            return result
        except ComputationError:
            raise
        except Exception as exc:
            raise ComputationError(
                "Original deterministic DGP construction failed.", evidence
            ) from exc

    def fit(self, cell, replicate, saved):
        inputs = saved["inputs"]
        frame = self.pd.DataFrame(inputs["levels"], columns=inputs["series"])
        frame.insert(0, "period", inputs["periods"])
        posterior = self.api.bayes_var(
            frame,
            inputs["series"],
            time="period",
            prior=cell["prior"],
            lags=cell["p"],
            intercept=True,
            alpha=0.05,
            max_work=100000000,
            max_bytes=134217728,
        )
        return self.raw(posterior)

    def draw(self, cell, replicate, saved):
        torch = self.torch
        self.packets.clear()  # At most one ordinal owns a typed draw packet.
        before = (
            torch.get_rng_state().clone(),
            str(torch.get_default_dtype()),
            str(torch.get_default_device()),
        )
        trace = []
        old_gamma, old_normal = torch._standard_gamma, torch.randn

        def gamma(*args, **kwargs):
            result = old_gamma(*args, **kwargs)
            trace.append(
                {
                    "kind": "gamma",
                    "shape_values": args[0].tolist(),
                    "shape": list(result.shape),
                    "values": jsonable(result),
                }
            )
            return result

        def normal(*args, **kwargs):
            result = old_normal(*args, **kwargs)
            trace.append(
                {"kind": "normal", "shape": list(result.shape), "values": jsonable(result)}
            )
            return result

        torch._standard_gamma, torch.randn = gamma, normal
        try:
            packet = self.draw_module.bayes_var_draws(
                saved["fit"], draws=DRAW_COUNT, seed=saved["random"]["posterior_seed"]
            )
            result = {"packet": self.raw(packet), "original_native_rng_calls": trace}
            self.packets[saved["random"]["posterior_seed"]] = (
                digest(result["packet"]),
                packet,
                False,
            )
            if not torch.equal(before[0], torch.get_rng_state()) or before[1:] != (
                str(torch.get_default_dtype()),
                str(torch.get_default_device()),
            ):
                raise ComputationError(
                    "Native local-generator phase changed global RNG/dtype/device.", result
                )
            return result
        except Exception as exc:
            raise ComputationError(
                "Original native draw phase failed; no retry/redraw.",
                {"original_native_rng_calls": trace, "seed": saved["random"]["posterior_seed"]},
            ) from exc
        finally:
            torch._standard_gamma, torch.randn = old_gamma, old_normal

    def _targets(self, b, sigma, prior):
        np = self.np
        m, k = b.shape[1], b.shape[0]
        left, right = np.asarray([(-1.0) ** i / (i + 1) for i in range(k)]), np.arange(1, m + 1) / m
        result = list(b.T.reshape(-1)) + [sigma[i, j] for j in range(m) for i in range(j, m)]
        result += [
            left @ b @ right,
            b[0, 0] + b[1, 1],
            (b[1, 0] - prior["mean"][1][0]) ** 2 / sigma[0, 0],
        ]
        return np.asarray(result)

    @staticmethod
    def rank_bins(truth, samples, ties, uniforms):
        bins, original_ranks = [], []
        for j, target in enumerate(truth):
            lower = sum(float(row[j]) < float(target) for row in samples)
            equal = sum(float(row[j]) == float(target) for row in samples)
            rank = lower + int(ties[j] * (equal + 1))
            value = int(8 * (rank + uniforms[j]) / (DRAW_COUNT + 1))
            if not 0 <= value < 8:
                raise ProtocolError("Randomized 129-rank bin is invalid.")
            bins.append(value)
            original_ranks.append({"below": lower, "equal": equal, "randomized_rank": rank})
        return bins, original_ranks

    def score(self, cell, replicate, saved):
        np, stats = self.np, self.stats
        inputs, random = saved["inputs"], saved["random"]
        seed = saved["random"]["posterior_seed"]
        body = saved["draw"]["packet"]
        cached = self.packets.get(seed)
        if cached and cached[0] == digest(body):
            packet = cached[1]
        else:
            packet = self.draw_module.restore_draws(body)
            self.packets[seed] = (digest(body), packet, True)
        posterior = packet.parent
        if digest(self.raw(posterior)) != digest(saved["fit"]):
            raise ProtocolError("Original fit and draw-parent packets disagree.")
        b, sigma = np.asarray(inputs["truth_B"]), np.asarray(inputs["truth_Sigma"])
        truth = self._targets(b, sigma, cell["prior"])
        samples = np.asarray(
            [
                self._targets(np.asarray(bv), np.asarray(sv), cell["prior"])
                for bv, sv in zip(packet.coefficients, packet.innovations, strict=True)
            ]
        )
        bins, ranks = self.rank_bins(
            truth, samples, random["tie_uniforms"], random["rank_uniforms"]
        )
        a, m = posterior.algebra, cell["m"]
        d = a.degrees_of_freedom - m + 1
        if d != a.scalar_degrees_of_freedom:
            raise ProtocolError("Scalar t df must be nuN-m+1.")
        ci = np.asarray(a.coefficient_intervals)
        coverage = list(
            ((ci[:, :, 0] <= b) & (b <= ci[:, :, 1])).T.reshape(-1).astype(bool).tolist()
        )
        sigma_intervals = []
        for j in range(m):
            interval = stats.invgamma.ppf(
                [0.025, 0.975], a=d / 2, scale=a.innovation_scale[j][j] / 2
            )
            sigma_intervals.append(interval.tolist())
            coverage.append(bool(interval[0] <= sigma[j, j] <= interval[1]))
        k = len(posterior.terms)
        left, right = [(-1.0) ** i / (i + 1) for i in range(k)], [(j + 1) / m for j in range(m)]
        contrast = self.api.bayes_var_contrast(posterior, left, right)
        rank_one = float(np.asarray(left) @ b @ np.asarray(right))
        coverage.append(bool(contrast.interval[0] <= rank_one <= contrast.interval[1]))
        prediction = self.api.bayes_var_predict(
            posterior, [inputs["held_out_design"]], terms=posterior.terms
        )
        delta = np.asarray(inputs["held_out_y"]) - np.asarray(prediction.location[0])
        predictive_scale = np.asarray(prediction.row_predictive_scale[0])
        mahalanobis = float(delta @ np.linalg.solve(predictive_scale, delta))
        critical = float(m * stats.f.isf(0.05, m, d))
        coverage.append(bool(mahalanobis <= critical))
        point = np.asarray(a.location).T.reshape(-1)
        control, control_ranks = self.rank_bins(
            b.T.reshape(-1), [point] * DRAW_COUNT, random["tie_uniforms"], random["rank_uniforms"]
        )
        return {
            "rank_bins": bins,
            "coverage": coverage,
            "point_mass_rank_bins": control,
            "all_truth_rank_targets": truth.tolist(),
            "all_original_draw_rank_targets": samples.tolist(),
            "original_ranks": ranks,
            "diagonal_Sigma_IG": {
                "shape": d / 2,
                "scale": [a.innovation_scale[j][j] / 2 for j in range(m)],
                "intervals": sigma_intervals,
            },
            "rank_one": self.raw(contrast),
            "one_step": self.raw(prediction),
            "one_step_F": {"m": m, "df": d, "mahalanobis": mahalanobis, "critical": critical},
            "coefficient_order": "column-major B; saved terms inside each saved series",
            "Sigma_order": "lower-column-major vech Sigma",
            "point_mass_original_ranks": control_ranks,
            "true_stable": inputs["truth_stable"],
            "posterior_stable": list(packet.stable),
        }

    def verify(self, cell, replicate, saved):
        np = self.np
        original = saved["random"]
        expected_seed = 202610090000 + cell["cell"] * 1024 + replicate
        if (
            original["truth_seed"] != expected_seed
            or original["posterior_seed"] != 202610100000 + cell["cell"] * 1024 + replicate
        ):
            raise ProtocolError("Original fixed seed changed.")
        calls, m = original["axen_calls"], cell["m"]
        if len(calls) != 2 or calls[0]["kind"] != "normal" or calls[1]["kind"] != "chisquare":
            raise ProtocolError("Actual pinned Axen primitive-call order changed.")
        expected_chi = [cell["prior"]["degrees_of_freedom"] - m + 1 + i for i in range(m)]
        if calls[1]["kwargs"]["df"] != expected_chi:
            raise ProtocolError("The independent Axen ascending chi df changed.")
        a = np.zeros((m, m))
        a[np.tril_indices(m, -1)] = np.asarray(calls[0]["values"]).reshape(-1)
        a[np.diag_indices(m)] = np.sqrt(np.asarray(calls[1]["values"]).reshape(-1))
        factor = np.linalg.cholesky(cell["prior"]["innovation_scale"]) @ np.linalg.inv(a)
        np.testing.assert_allclose(
            original["truth_sigma"], factor @ factor.T, rtol=2e-12, atol=2e-12
        )
        replay = self.inputs(cell, replicate, {"random": original})
        if digest(replay) != digest(saved["inputs"]):
            raise ProtocolError("Original full DGP/held-out/stability replay changed.")
        # All endpoint/RNG calls remain forbidden throughout complete cached replay.
        fit, primitive = self.api.bayes_var, self.draw_module.primitive_stream
        gamma, normal = self.torch._standard_gamma, self.torch.randn

        def forbidden(*args, **kwargs):
            raise ProtocolError("Cached ordinal verification attempted refit/RNG.")

        self.api.bayes_var = self.draw_module.primitive_stream = forbidden
        self.torch._standard_gamma = self.torch.randn = forbidden
        try:
            body, seed = saved["draw"]["packet"], saved["random"]["posterior_seed"]
            cached = self.packets.get(seed)
            if cached and cached[0] == digest(body) and cached[2]:
                packet = cached[1]
            else:
                packet = self.draw_module.restore_draws(body)
                self.packets[seed] = (digest(body), packet, True)
            posterior = packet.parent
            if digest(self.raw(posterior)) != digest(saved["fit"]) or packet.draws != DRAW_COUNT:
                raise ProtocolError("Complete original draw/parent state differs.")
            trace, position = saved["draw"]["original_native_rng_calls"], 0
            for index in range(DRAW_COUNT):
                call = trace[position]
                position += 1
                expected = [(posterior.algebra.degrees_of_freedom - i) / 2 for i in range(m)]
                if (
                    call["kind"] != "gamma"
                    or call["shape_values"] != expected
                    or call["shape"] != [m]
                ):
                    raise ProtocolError(
                        "Original native descending Bartlett Gamma contract changed."
                    )
                bartlett = np.zeros((m, m))
                bartlett[np.diag_indices(m)] = np.sqrt(2 * np.asarray(call["values"]))
                for i in range(1, m):
                    call = trace[position]
                    position += 1
                    if call["kind"] != "normal" or call["shape"] != [i]:
                        raise ProtocolError("Native off-diagonal normal contract changed.")
                    bartlett[i, :i] = call["values"]
                call = trace[position]
                position += 1
                if call["kind"] != "normal" or call["shape"] != [len(posterior.terms), m]:
                    raise ProtocolError("Native coefficient matrix-normal primitive shape changed.")
                np.testing.assert_allclose(bartlett, packet.bartlett[index], rtol=2e-14, atol=2e-14)
                np.testing.assert_array_equal(call["values"], packet.standard_normals[index])
            if position != len(trace):
                raise ProtocolError("Additional or discarded native RNG calls were recorded.")
            if digest(self.score(cell, replicate, saved)) != digest(saved["score"]):
                raise ProtocolError("Original complete rank/coverage/held-out score differs.")
        finally:
            self.api.bayes_var, self.draw_module.primitive_stream = fit, primitive
            self.torch._standard_gamma, self.torch.randn = gamma, normal
            self.packets.clear()


def create_registration(directory, pin_path, draft_path):
    enforce_invocation_policy(invocation_policy())
    if file_sha(pin_path) != PIN_SHA or file_sha(draft_path) != DRAFT_SHA:
        raise ProtocolError(
            "Only the reviewed candidate-v3 source and prospective family are accepted."
        )
    pin, draft = read_json(pin_path), read_json(draft_path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != BASE or pin["base"] != BASE:
        raise ProtocolError("Registration requires the exact reviewed b634 base.")
    files = {}
    for relative, sha in pin["files"].items():
        path = ROOT / relative
        if file_sha(path) != sha:
            raise ProtocolError("One of the frozen sixteen candidate-v3 files changed.")
        files[str(path)] = sha
    for relative in (
        "scripts/validate_bayesian_var_sbc.py",
        "tests/test_bayesian_var_sbc_protocol.py",
    ):
        path = ROOT / relative
        files[str(path)] = file_sha(path)
    tracked = subprocess.check_output(
        ["git", "ls-files", "src/openecon"], cwd=ROOT, text=True
    ).splitlines()
    for relative in tracked:
        path = ROOT / relative
        expected = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
        if file_sha(path) != hashlib.sha256(expected).hexdigest():
            raise ProtocolError("A base dependency differs from the exact reviewed Git source.")
        files[str(path)] = file_sha(path)
    scipy_source = pathlib.Path(
        importlib.metadata.distribution("scipy").locate_file("scipy/stats/_multivariate.py")
    )
    if file_sha(scipy_source) != SCIPY_SHA:
        raise ProtocolError("Actual original SciPy Axen/variance source changed.")
    files[str(scipy_source)] = SCIPY_SHA
    runtime = runtime_pin()
    files[str(pathlib.Path(sys.executable).resolve())] = runtime["executable_sha256"]
    for package in runtime["packages"].values():
        files.update(package["record_files"])
        files.update(package["installed_source_and_native_files"])
    resources = {
        "native_workers": 1,
        "native_threads": 1,
        "per_fit_max_work": 100000000,
        "per_fit_max_bytes": 134217728,
        "max_total_work_upper_bound": 102400000000,
        "max_disk_bytes": MAX_DISK,
        "min_free_bytes": MIN_FREE,
        "reservation_kind": "logical root-owned 4 GiB reservation; no physically exclusive disk claim",
        "native_thread_count_status": "declared policy; actual Torch getters checked at authorized launch; Accelerate count unobserved",
    }
    pipeline_work = []
    for cell in draft["cells"]:
        m, p, n = cell["m"], cell["p"], cell["n_original"]
        k = m * p + 1
        algebra = 16 * n * k**2 + 80 * k**3
        draw = algebra + DRAW_COUNT * 32 * (k * m**2 + k**2 * m + (m * p) ** 3)
        estimate = 3 * draw + 16 * algebra + 16 * DRAW_COUNT * cell["rank_targets"]
        if estimate > resources["per_fit_max_work"]:
            raise ProtocolError("Registered full ordinal pipeline exceeds its 100M work guard.")
        pipeline_work.append(estimate)
    resources["pipeline_work_per_cell_upper_estimates"] = pipeline_work
    resources["pipeline_work_total_upper_estimate"] = REPLICATES * sum(pipeline_work)
    directory = pathlib.Path(directory).resolve()
    if directory.exists():
        raise ProtocolError(
            "Register only in a new owned study directory; never overwrite an old study."
        )
    if shutil.disk_usage(directory.parent).free < MIN_FREE:
        raise DiskGuardError("Registration needs at least 8 GiB actual free disk.")
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(directory) + digest(files) + DRAFT_SHA))
    value = {
        "schema": "openecon.bvar_sbc.registration.v1",
        "run_id": run_id,
        "base": BASE,
        "root": str(ROOT),
        "study_directory": str(directory),
        "source_files": files,
        "source_pin_sha256": PIN_SHA,
        "reviewed_draft_sha256": DRAFT_SHA,
        "draft": draft,
        "draft_canonical_sha256": digest(draft),
        "runtime": runtime,
        "resources": resources,
        "total_cases": TOTAL,
        "replicates_per_cell": REPLICATES,
        "posterior_draws": DRAW_COUNT,
        "pause_after_committed": 64,
        "registration_free_disk_bytes": shutil.disk_usage(directory.parent).free,
        "stage_rule": "started-uncommitted random/fit/draw => computational failure; never redraw/refit",
        "launch_authorized": False,
        "author_MATLAB_executed": False,
        "native_state_binding": "first worker_started exact state before any fit/RNG; restart must equal that state",
    }
    validate_registration(value)
    directory.mkdir()

    class RegistrationGuard:
        def check(self, additional_bytes=0):
            if shutil.disk_usage(directory).free < MIN_FREE:
                raise DiskGuardError("Free disk guard failed during registration archival.")
            if (
                sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                + additional_bytes
                > MAX_DISK
            ):
                raise DiskGuardError("Registration evidence exceeds its logical disk reservation.")

    store = Store(directory, RegistrationGuard())
    store.write("registration.json", value)
    evidence = {
        "initial-handoff.md": "/tmp/market-734-initial-implementation-handoff.md",
        "driver-resume-proposal.md": "/tmp/market-734-sbc-driver-and-resume-proposal.md",
        "rank-two-oracle-audit.json": "/tmp/market-734-sigma-rank-two-oracle-audit.json",
        "scipy-original-source.py": str(scipy_source),
        "candidate-v3-source.tar.gz": "/tmp/market-734-initial-implementation-exact-source-v3.tar.gz",
        "candidate-v3-pin.json": str(pin_path),
        "prospective-v3.json": str(draft_path),
        "superseded-adaptive-Holm-v1.json": "/tmp/market-734-prospective-sbc-draft-v1.json",
        "reviewed-prospective-v2.json": "/tmp/market-734-prospective-sbc-draft-v2.json",
        "root-mathematical-screen.json": "/tmp/market734-root-readonly-sbc-mathematical-screen.json",
    }
    for relative in (
        *pin["files"],
        "scripts/validate_bayesian_var_sbc.py",
        "tests/test_bayesian_var_sbc_protocol.py",
        *tracked,
    ):
        evidence[f"source/{relative}"] = str(ROOT / relative)
    (directory / "evidence").mkdir()
    for name, original in evidence.items():
        destination = directory / "evidence" / name
        store.guard.check(additional_bytes=pathlib.Path(original).stat().st_size)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(pathlib.Path(original).read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        if file_sha(destination) != file_sha(original):
            raise ProtocolError("Registration evidence archive failed byte equality.")
    fsync_directory(directory / "evidence")
    store.write(
        "evidence/inventory.json",
        {name: file_sha(directory / "evidence" / name) for name in evidence},
    )
    return value


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    register = sub.add_parser("register")
    register.add_argument("directory", type=pathlib.Path)
    register.add_argument(
        "--source-pin",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/market-734-initial-implementation-final-source-pin-v3.json"),
    )
    register.add_argument(
        "--draft",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/market-734-prospective-sbc-draft-v3.json"),
    )
    run = sub.add_parser("run")
    run.add_argument("directory", type=pathlib.Path)
    run.add_argument("--registration-sha", required=True)
    args = parser.parse_args()
    enforce_invocation_policy(invocation_policy())
    if args.mode == "register":
        create_registration(args.directory, args.source_pin, args.draft)
        print(
            json.dumps(
                {
                    "registration": str(args.directory / "registration.json"),
                    "sha256": file_sha(args.directory / "registration.json"),
                    "study_launched": False,
                }
            )
        )
        return 0
    directory = args.directory.resolve()
    if file_sha(directory / "registration.json") != args.registration_sha:
        raise ProtocolError(
            "Launch/resume requires the exact separately reviewed registration SHA."
        )
    registration = validate_registration(read_json(directory / "registration.json"))
    if registration["study_directory"] != str(directory) or registration["root"] != str(ROOT):
        raise ProtocolError("Registered study/source directory changed.")
    guard = Guard(directory, registration)
    store = Store(directory, guard)
    lock = directory / "run.lock.json"
    if lock.exists():
        old = read_json(lock)
        try:
            os.kill(old["pid"], 0)
        except ProcessLookupError:
            archived = directory / f"lease-{old['pid']}-{time.time_ns()}.json"
            os.rename(lock, archived)
            fsync_directory(directory)
            guard.note_write(lock)
            guard.note_write(archived)
        else:
            raise ProtocolError(
                "The existing driver process is still alive; no duplicate worker is allowed."
            )
    store.write(
        "run.lock.json",
        {
            "pid": os.getpid(),
            "run_id": registration["run_id"],
            "registration_sha256": args.registration_sha,
        },
    )
    try:
        engine = Engine(directory, registration, None, guard=guard)
        adapter = PhysicsAdapter(registration)
        engine.adapter = adapter
        native_sha = bind_native_state(registration, engine.ledger, adapter.launch_state)
        engine.ledger.append(
            "worker_started",
            pid=os.getpid(),
            actual_native_state=adapter.launch_state,
            native_state_sha256=native_sha,
        )
        result = engine.run()
        print(
            json.dumps(
                {
                    "final": str(directory / "final.json"),
                    "acceptance": result["acceptance"],
                    "computational_failures": result["computational_failures"],
                    "fixed_denominator": TOTAL,
                }
            )
        )
        return result["exit_code"]
    except BaseException as exc:
        try:
            store.write(
                f"abort-{os.getpid()}-{time.time_ns()}.json",
                {
                    "pid": os.getpid(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "acceptance": False,
                    "fixed_denominator": TOTAL,
                },
            )
        except BaseException:
            pass  # stderr and every existing original checkpoint remain authoritative.
        raise
    finally:
        if lock.exists():
            archived = directory / f"lease-{os.getpid()}-{time.time_ns()}.json"
            os.rename(lock, archived)
            fsync_directory(directory)
            guard.note_write(lock)
            guard.note_write(archived)


if __name__ == "__main__":
    raise SystemExit(main())
