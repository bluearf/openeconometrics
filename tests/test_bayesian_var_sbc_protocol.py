"""Pure stdlib protocol adversaries: no production imports, fit, or RNG.

Run directly with unittest, without the repository's numerical pytest plugins.
The reviewed fixed design supplies dimensions only; callbacks return fake lists.
"""

from __future__ import annotations

import copy
import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

DRIVER = pathlib.Path(__file__).resolve().parents[1] / "scripts/validate_bayesian_var_sbc.py"
SPEC = importlib.util.spec_from_file_location("bvar_sbc_protocol_driver", DRIVER)
sbc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sbc)
# Only reviewed dimensions/fixed gate sets are needed for the fake callbacks.
# No ephemeral study artifact, prior array, scientific computation, or RNG.
DRAFT = {
    "cells": [
        {
            "cell": i,
            "m": m,
            "p": p,
            "n_original": n,
            "n_response": n - p,
            "rank_targets": (m * p + 1) * m + m * (m + 1) // 2 + 3,
            "exact_coverage_targets": (m * p + 1) * m + m + 2,
        }
        for i, (m, p, n) in enumerate([(2, 1, 40), (3, 2, 80), (4, 4, 160), (2, 4, 20)] * 2)
    ],
    "family": {"total_gates": 2616},
    "accepted_rank_bin_counts": list(range(2, 36)),
    "accepted_exact_coverage_counts": list(range(108, 129)),
}


def registration():
    return {
        "draft": copy.deepcopy(DRAFT),
        "draft_canonical_sha256": sbc.digest(DRAFT),
        "total_cases": 1024,
        "replicates_per_cell": 128,
        "posterior_draws": 128,
        "pause_after_committed": 64,
        "run_id": "fixed-protocol-fixture-no-RNG",
        "source_files": {},
    }


def fake_score(cell, replicate):
    return {
        "rank_bins": [replicate % 8] * cell["rank_targets"],
        "coverage": [True] * cell["exact_coverage_targets"],
        "point_mass_rank_bins": [0 if replicate < 64 else 7]
        * ((cell["m"] * cell["p"] + 1) * cell["m"]),
        "full_original_fake_targets": ["transport-only", cell["cell"], replicate],
    }


def fake_outcomes(reg):
    result = {}
    for ordinal in range(1024):
        cell, replicate = sbc.coordinates(ordinal)
        result[ordinal] = {
            "ordinal": ordinal,
            "cell": cell,
            "replicate": replicate,
            "status": "success",
            "score": fake_score(reg["draft"]["cells"][cell], replicate),
        }
    return result


class FakeGuard:
    def __init__(self):
        self.checks = []
        self.fail = False

    def check(self, additional_bytes=0, *, full=False):
        self.checks.append((additional_bytes, full))
        if self.fail:
            raise sbc.DiskGuardError("Fake disk admission failed before the next effect.")


class FakeAdapter:
    def __init__(self, *, interrupt=None, fail=None):
        self.calls, self.verifications = [], []
        self.interrupt, self.fail = interrupt, fail

    def phase(self, phase, cell, replicate, saved):
        self.calls.append((cell["cell"], replicate, phase))
        if phase == self.interrupt:
            raise KeyboardInterrupt("Injected started-but-uncommitted original effect.")
        if phase == self.fail:
            raise sbc.ComputationError(
                "Injected original failure",
                {"finite_prefix": [1, 2], "nonfinite": {"nonfinite_float64": "+inf"}},
            )
        if phase == "score":
            return fake_score(cell, replicate)
        return {
            "phase": phase,
            "cell": cell["cell"],
            "replicate": replicate,
            "parents": sorted(saved),
        }

    def verify(self, cell, replicate, saved):
        self.verifications.append((cell["cell"], replicate))
        if set(saved) != set(sbc.PHASES) or saved["score"] != fake_score(cell, replicate):
            raise sbc.ProtocolError("Fake complete source packet changed.")


class ProtocolFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="market734-protocol-only-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.reg, self.guard = registration(), FakeGuard()
        sbc.Store(self.root, self.guard).write("registration.json", self.reg)

    def engine(self, adapter=None, *, stop=lambda: None):
        return sbc.Engine(
            self.root, self.reg, adapter or FakeAdapter(), guard=self.guard, stop=stop
        )


class StoreTests(ProtocolFixture):
    def test_exclusive_final_refuses_different_value_and_preserves_partial(self):
        store = sbc.Store(self.root, self.guard)
        store.write("value.json", {"v": 1})
        with self.assertRaises(FileExistsError):
            store.write("value.json", {"v": 2})
        self.assertEqual(sbc.read_json(self.root / "value.json"), {"v": 1})
        self.assertEqual(len(list(self.root.glob(".value.json.*.partial"))), 1)

    def test_link_commit_then_error_is_read_back_without_retry(self):
        calls = []

        def commit_then_raise(source, target):
            calls.append(str(target))
            os.link(source, target)
            raise OSError("Injected ambiguous commit.")

        result = sbc.Store(self.root, self.guard, link=commit_then_raise).write(
            "ambiguous.json", {"full": [1, 2, 3]}
        )
        self.assertTrue(result["ambiguous_commit_readback"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(sbc.read_json(self.root / "ambiguous.json"), {"full": [1, 2, 3]})
        self.assertEqual(len(list(self.root.glob(".ambiguous.json.*.partial"))), 1)

    def test_progress_replace_then_error_is_read_back(self):
        original = os.replace

        def commit_then_raise(source, target):
            original(source, target)
            raise OSError("Injected ambiguous progress replacement.")

        with mock.patch.object(sbc.os, "replace", commit_then_raise):
            result = sbc.Store(self.root, self.guard).write(
                "snapshot.json", {"revision": 1}, replace=True
            )
        self.assertTrue(result["ambiguous_commit_readback"])
        self.assertEqual(sbc.read_json(self.root / "snapshot.json"), {"revision": 1})

    def test_path_escape_and_symlink_refused(self):
        store = sbc.Store(self.root, self.guard)
        with self.assertRaises(sbc.ProtocolError):
            store.write("../outside.json", {})
        (self.root / "alias.json").symlink_to(self.root / "registration.json")
        with self.assertRaises(sbc.ProtocolError):
            store.write("alias.json", {})

    def test_json_duplicate_nonfinite_and_oversize_refused(self):
        path = self.root / "invalid.json"
        for body in ('{"v":1,"v":2}', '{"v":NaN}'):
            path.write_text(body)
            with self.assertRaises(sbc.ProtocolError):
                sbc.read_json(path)
        with mock.patch.object(sbc, "MAX_FILE", 8):
            with self.assertRaises(sbc.ProtocolError):
                sbc.Store(self.root, self.guard).write("too-big.json", {"values": [1, 2, 3]})
        self.assertFalse((self.root / "too-big.json").exists())


class LedgerTests(ProtocolFixture):
    def test_gap_and_payload_tampering_refused(self):
        engine = self.engine()
        engine.ledger.append("fixture", original=True)
        path = self.root / "ledger/00000000.json"
        original = path.read_bytes()
        value = sbc.read_json(path)
        value["original"] = False
        path.write_bytes(sbc.canonical(value))
        with self.assertRaises(sbc.ProtocolError):
            self.engine()
        path.write_bytes(original)
        path.rename(path.with_name("00000001.json"))
        with self.assertRaises(sbc.ProtocolError):
            self.engine()

    def test_identity_override_refused(self):
        engine = self.engine()
        for data in (
            {"sequence": 5},
            {"registration_sha256": "different"},
            {"previous_sha256": "different"},
        ):
            with self.assertRaises(sbc.ProtocolError):
                engine.ledger.append("fixture", **data)

    def test_snapshot_tamper_refused_but_committed_revision_recovered(self):
        engine = self.engine()
        engine.progress({}, "before")
        old = (self.root / "progress.json").read_bytes()
        engine.progress({}, "after")
        expected = (self.root / "progress.json").read_bytes()
        (self.root / "progress.json").write_bytes(old)
        self.engine()
        self.assertEqual((self.root / "progress.json").read_bytes(), expected)
        value = sbc.read_json(self.root / "progress.json")
        value["completed"] = 999
        (self.root / "progress.json").write_bytes(sbc.canonical(value))
        with self.assertRaises(sbc.ProtocolError):
            self.engine()

    def test_missing_original_progress_refused(self):
        engine = self.engine()
        engine.progress({}, "first")
        engine.progress({}, "second")
        (self.root / "progress.json").unlink()
        with self.assertRaises(sbc.ProtocolError):
            self.engine()


class ResumeTests(ProtocolFixture):
    def test_committed_target_keyboard_interrupt_and_system_exit_propagate_without_refit(self):
        for interruption in (KeyboardInterrupt, SystemExit):
            with (
                self.subTest(interruption=interruption.__name__),
                tempfile.TemporaryDirectory(prefix="market734-deliberate-stop-") as temporary,
            ):
                root = pathlib.Path(temporary)
                sbc.Store(root, self.guard).write("registration.json", self.reg)
                original = FakeAdapter()
                engine = sbc.Engine(root, self.reg, original, guard=self.guard)
                link, stopped = engine.store.link, []

                def link_then_stop(source, target):
                    link(source, target)
                    if pathlib.Path(target).name == "fit.json" and not stopped:
                        stopped.append(True)
                        raise interruption("Deliberate stop after original committed fake packet.")

                engine.store.link = link_then_stop
                with self.assertRaises(interruption):
                    engine.case(0)
                self.assertEqual(sum(v[2] == "fit" for v in original.calls), 1)
                self.assertTrue((root / "cases/0000/fit.json").exists())
                resumed = FakeAdapter()
                result = sbc.Engine(root, self.reg, resumed, guard=self.guard).case(0)
                self.assertEqual(result["status"], "success")
                self.assertEqual([v[2] for v in resumed.calls], ["draw", "score"])
                self.assertEqual(sum(v[2] == "fit" for v in original.calls + resumed.calls), 1)

    def test_deliberate_interrupt_after_start_marker_remains_exact_failed_denominator(self):
        original = FakeAdapter()
        engine = self.engine(original)
        link = engine.store.link

        def link_then_stop(source, target):
            link(source, target)
            if pathlib.Path(target).name == "fit.started.json":
                raise KeyboardInterrupt("Deliberate stop before original fit callback.")

        engine.store.link = link_then_stop
        with self.assertRaises(KeyboardInterrupt):
            engine.case(0)
        resumed = FakeAdapter()
        outcome = self.engine(resumed).case(0)
        self.assertEqual(outcome["status"], "computational_failure")
        self.assertEqual(outcome["failure_phase"], "fit")
        self.assertEqual(outcome["failure_code"], "interrupted_before_phase_commit")
        self.assertFalse(any(v[2] == "fit" for v in original.calls + resumed.calls))
        self.assertEqual(resumed.calls, [])
        full = fake_outcomes(self.reg)
        full[0] = outcome
        result = sbc.aggregate(full, self.reg)
        self.assertEqual(result["fixed_denominator"], 1024)
        self.assertEqual(result["exit_code"], 2)
        self.assertFalse(result["acceptance"])

    def test_complete_original_resume_has_zero_phase_callbacks(self):
        original = FakeAdapter()
        first = self.engine(original).case(0)
        resumed = FakeAdapter()
        second = self.engine(resumed).case(0)
        self.assertEqual(first, second)
        self.assertEqual(len(original.calls), 5)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(resumed.verifications, [(0, 0)])
        self.assertEqual(len(second["score"]), 3)
        full = sbc.read_json(self.root / "cases/0000/score.json")["data"]
        self.assertEqual(second["full_score_sha256"], sbc.digest(full))
        self.assertIn("full_original_fake_targets", full)

    def test_every_irreversible_uncommitted_phase_is_terminal_without_redraw(self):
        for phase in sbc.IRREVERSIBLE:
            with (
                self.subTest(phase=phase),
                tempfile.TemporaryDirectory(prefix="market734-interruption-") as temporary,
            ):
                root = pathlib.Path(temporary)
                sbc.Store(root, self.guard).write("registration.json", self.reg)
                first = FakeAdapter(interrupt=phase)
                engine = sbc.Engine(root, self.reg, first, guard=self.guard)
                with self.assertRaises(KeyboardInterrupt):
                    engine.case(0)
                resumed = FakeAdapter()
                result = sbc.Engine(root, self.reg, resumed, guard=self.guard).case(0)
                self.assertEqual(result["status"], "computational_failure")
                self.assertEqual(result["failure_phase"], phase)
                self.assertEqual(result["failure_code"], "interrupted_before_phase_commit")
                self.assertEqual(resumed.calls, [])
                self.assertEqual(
                    sbc.Engine(root, self.reg, resumed, guard=self.guard).case(0), result
                )
                self.assertEqual(resumed.calls, [])

    def test_deterministic_uncommitted_inputs_resume_uses_original_random_only(self):
        engine = self.engine(FakeAdapter(interrupt="inputs"))
        with self.assertRaises(KeyboardInterrupt):
            engine.case(0)
        resumed = FakeAdapter()
        result = self.engine(resumed).case(0)
        self.assertEqual(result["status"], "success")
        self.assertEqual([v[2] for v in resumed.calls], ["inputs", "fit", "draw", "score"])

    def test_committed_packet_before_ledger_is_recovered_without_callback(self):
        original = FakeAdapter()
        engine = self.engine(original)
        append = engine.ledger.append

        def interrupted(event, **data):
            if event == "phase_committed":
                raise KeyboardInterrupt("Packet committed before ledger acknowledgment.")
            return append(event, **data)

        engine.ledger.append = interrupted
        with self.assertRaises(KeyboardInterrupt):
            engine.case(0)
        resumed = FakeAdapter()
        result = self.engine(resumed).case(0)
        self.assertEqual(result["status"], "success")
        self.assertNotIn((0, 0, "random"), resumed.calls)
        entry = next(
            e
            for e in self.engine().ledger.entries
            if e["event"] == "phase_committed" and e["phase"] == "random"
        )
        self.assertTrue(entry["ambiguous_commit_readback"])

    def test_marker_before_ledger_is_terminal_not_a_new_random_call(self):
        engine = self.engine()
        engine.store.write(
            "cases/0000/random.started.json",
            engine._envelope(0, "random_started", {}, {"pid": os.getpid()}),
        )
        adapter = FakeAdapter()
        result = self.engine(adapter).case(0)
        self.assertEqual(result["status"], "computational_failure")
        self.assertEqual(adapter.calls, [])
        self.assertTrue(self.engine().ledger.entries[0]["recovered_ambiguous_commit"])

    def test_started_marker_parent_tampering_refused_without_effect(self):
        engine = self.engine()
        engine.store.write(
            "cases/0000/random.started.json",
            engine._envelope(0, "random_started", {"wrong": "parent"}, {"pid": os.getpid()}),
        )
        adapter = FakeAdapter()
        with self.assertRaises(sbc.ProtocolError):
            self.engine(adapter).case(0)
        self.assertEqual(adapter.calls, [])

    def test_completed_packet_tampering_and_deletion_refused_without_effect(self):
        self.engine().case(0)
        path = self.root / "cases/0000/random.json"
        original = path.read_bytes()
        value = sbc.read_json(path)
        value["data"]["selected"] = True
        value["data_sha256"] = sbc.digest(value["data"])
        path.write_bytes(sbc.canonical(value))
        adapter = FakeAdapter()
        with self.assertRaises(sbc.ProtocolError):
            self.engine(adapter).case(0)
        self.assertEqual(adapter.calls, [])
        path.write_bytes(original)
        (self.root / "cases/0000/outcome.json").unlink()
        with self.assertRaises(sbc.ProtocolError):
            self.engine(adapter).case(0)
        self.assertEqual(adapter.calls, [])

    def test_original_computation_failure_preserves_evidence_and_denominator(self):
        first = FakeAdapter(fail="fit")
        outcome = self.engine(first).case(0)
        self.assertEqual(outcome["failure_evidence"]["evidence"]["finite_prefix"], [1, 2])
        self.assertEqual(outcome["status"], "computational_failure")
        resumed = FakeAdapter()
        self.assertEqual(self.engine(resumed).case(0), outcome)
        self.assertEqual(resumed.calls, [])
        row = sbc.aggregate_row(outcome)
        self.assertNotIn("failure_evidence", row)
        self.assertEqual(
            row["original_outcome_sha256"], sbc.file_sha(self.root / "cases/0000/outcome.json")
        )

    def test_skip_or_selected_ordinal_refused(self):
        adapter = FakeAdapter()
        with self.assertRaises(sbc.ProtocolError):
            self.engine(adapter).case(1)
        self.assertEqual(adapter.calls, [])
        for ordinal in (True, -1, 1024, 1.0):
            with self.assertRaises(sbc.ProtocolError):
                sbc.coordinates(ordinal)

    def test_disk_guard_before_effect_leaves_no_callback(self):
        engine, adapter = self.engine(), FakeAdapter()
        engine.adapter = adapter
        self.guard.fail = True
        with self.assertRaises(sbc.DiskGuardError):
            engine.case(0)
        self.assertEqual(adapter.calls, [])
        self.assertFalse((self.root / "cases/0000/random.started.json").exists())

    def test_one_pause_exact64_same_pid_inventory_and_zero_second_pause(self):
        stops = []
        engine = self.engine(stop=lambda: stops.append(os.getpid()))
        result = engine.run(limit=64)
        self.assertEqual(
            result,
            {"complete": False, "acceptance": False, "fixed_denominator": 1024, "completed": 64},
        )
        self.assertEqual(stops, [os.getpid()])
        planned = next(e for e in engine.ledger.entries if e["event"] == "planned_pause")
        continued = next(e for e in engine.ledger.entries if e["event"] == "same_pid_continued")
        self.assertEqual(planned["completed"], 64)
        self.assertEqual(planned["pid"], continued["pid"])
        self.assertEqual(planned["entry_sha256"], continued["pause_entry_sha256"])
        self.assertEqual(
            len(
                [
                    e
                    for e in engine.ledger.entries
                    if e["event"] == "phase_started" and e["phase"] == "fit"
                ]
            ),
            64,
        )
        adapter = FakeAdapter()
        resumed = self.engine(adapter, stop=lambda: stops.append("second"))
        resumed.run(limit=64)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(stops, [os.getpid()])

    def test_planned_pause_cannot_be_substituted_with_restarted_engine(self):
        engine = self.engine()
        engine.ledger.append("planned_pause", pid=os.getpid(), completed=64, inventory={})
        with self.assertRaises(sbc.ProtocolError):
            self.engine()

    def test_final_committed_exact1024_is_idempotent_without_any_phase_callbacks(self):
        adapter, stops = FakeAdapter(), []
        engine = self.engine(adapter, stop=lambda: stops.append(os.getpid()))
        originals = fake_outcomes(self.reg)
        # Fake original outcomes exercise the real final write/ledger path.
        engine.case = lambda ordinal: originals[ordinal]
        engine.progress = lambda outcomes, status="running": engine.store.write(
            "progress.json",
            {"completed": len(outcomes), "status": status, "acceptance": False},
            replace=True,
        )
        result = engine.run()
        final_sha = sbc.file_sha(self.root / "final.json")
        self.assertTrue(result["acceptance"])
        self.assertEqual(engine.run(), result)
        self.assertEqual(sbc.file_sha(self.root / "final.json"), final_sha)
        self.assertEqual(
            len([e for e in engine.ledger.entries if e["event"] == "final_committed"]), 1
        )
        self.assertEqual(stops, [os.getpid()])
        self.assertEqual(adapter.calls, [])
        self.assertEqual(len(result["original_outcome_sha256"]), 1024)


class GuardTests(ProtocolFixture):
    def test_isolated_bytecode_and_all_preimport_thread_declarations_required(self):
        policy = {
            "isolated": True,
            "ignore_PYTHON_environment": True,
            "dont_write_bytecode": True,
            "native_thread_declarations": {name: "1" for name in sbc.THREAD_VARIABLES},
        }
        sbc.enforce_invocation_policy(policy)
        for flag in ("isolated", "ignore_PYTHON_environment", "dont_write_bytecode"):
            with self.subTest(flag=flag), self.assertRaises(sbc.ProtocolError):
                sbc.enforce_invocation_policy(policy | {flag: False})
        for name in sbc.THREAD_VARIABLES:
            altered = copy.deepcopy(policy)
            altered["native_thread_declarations"][name] = "2"
            with self.subTest(variable=name), self.assertRaises(sbc.ProtocolError):
                sbc.enforce_invocation_policy(altered)

    def test_first_native_state_binds_defaults_before_effect_and_restart_drift_refused(self):
        engine = self.engine()
        policy = {"fixture_only": "no native runtime captured"}
        reg = self.reg | {
            "runtime": {
                "packages": {"torch": {"version": "fake-version"}},
                "invocation_policy": policy,
            }
        }
        state = {
            "native_threads": 1,
            "interop_threads": 1,
            "default_device": "cpu",
            "default_dtype": "torch.float32",
            "torch_version": "fake-version",
            "invocation_policy": policy,
            "fixture_only": "metadata only; no native getters or kernels",
        }
        native_sha = sbc.bind_native_state(reg, engine.ledger, state)
        engine.ledger.append(
            "worker_started", actual_native_state=state, native_state_sha256=native_sha
        )
        self.assertEqual(sbc.bind_native_state(reg, engine.ledger, state), native_sha)
        for field, value in (
            ("default_dtype", "torch.float64"),
            ("default_dtype", None),
            ("native_threads", True),
            ("native_threads", 2),
            ("interop_threads", 2),
            ("default_device", "cuda"),
            ("torch_version", "different"),
        ):
            with self.subTest(field=field), self.assertRaises(sbc.ProtocolError):
                sbc.bind_native_state(reg, engine.ledger, state | {field: value})

    def test_bootstrap_sets_torch_policy_before_local_import_and_exact_source_insertion(self):
        tree = ast.parse(DRIVER.read_text())
        adapter = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PhysicsAdapter"
        )
        constructor = next(
            node
            for node in adapter.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        imports = {
            alias.name: node.lineno
            for node in ast.walk(constructor)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        calls = {
            ast.unparse(node.func): node.lineno
            for node in ast.walk(constructor)
            if isinstance(node, ast.Call)
        }
        self.assertLess(imports["torch"], calls["torch.set_num_threads"])
        self.assertLess(calls["torch.set_num_threads"], calls["torch.set_num_interop_threads"])
        self.assertLess(calls["torch.set_num_interop_threads"], imports["numpy"])
        self.assertLess(calls["torch.set_num_interop_threads"], imports["openecon"])
        self.assertLess(calls["sys.path.insert"], imports["openecon"])
        self.assertIn('str(ROOT / "src")', ast.unparse(constructor).replace("'", '"'))

    def test_low_disk_and_source_bytes_drift_refused(self):
        with mock.patch.object(sbc, "runtime_pin", lambda: {"fixture": "stdlib only"}):
            source = self.root / "source.py"
            source.write_text("frozen=1\n")
            reg = self.reg | {
                "source_files": {str(source): sbc.file_sha(source)},
                "runtime": {"fixture": "stdlib only"},
                "resources": {"min_free_bytes": 50, "max_disk_bytes": 100000},
            }
            guard = sbc.Guard(self.root, reg, free=lambda path: types.SimpleNamespace(free=100))
            source.write_text("frozen=2\n")
            with self.assertRaises(sbc.ProtocolError):
                guard.check()
            source.write_text("frozen=1\n")
            with self.assertRaises(sbc.DiskGuardError):
                sbc.Guard(self.root, reg, free=lambda path: types.SimpleNamespace(free=49))

    def test_new_archive_bytes_and_additional_write_count_against_quota(self):
        runtime = {"fixture": "stdlib only"}
        with mock.patch.object(sbc, "runtime_pin", lambda: runtime):
            original = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            reg = self.reg | {
                "source_files": {},
                "runtime": runtime,
                "resources": {"min_free_bytes": 50, "max_disk_bytes": original + 10},
            }
            guard = sbc.Guard(self.root, reg, free=lambda path: types.SimpleNamespace(free=100))
            with self.assertRaises(sbc.DiskGuardError):
                guard.check(additional_bytes=11)
            (self.root / "external-archive").write_bytes(b"x" * 11)
            with self.assertRaises(sbc.DiskGuardError):
                guard.check(full=True)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.reg = registration()
        self.outcomes = fake_outcomes(self.reg)

    def test_full1024_all2616_fixed_gates_and_controls(self):
        result = sbc.aggregate(self.outcomes, self.reg)
        self.assertTrue(result["acceptance"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["fixed_denominator"], 1024)
        self.assertEqual(len(result["gates"]), 2616)
        self.assertEqual(sum(g["kind"] == "rank" for g in result["gates"]), 2352)
        self.assertEqual(sum(g["kind"] == "coverage" for g in result["gates"]), 264)
        self.assertTrue(all(g["denominator"] == 128 for g in result["gates"]))
        self.assertTrue(all(g["count"] == 16 for g in result["gates"] if g["kind"] == "rank"))
        self.assertTrue(all(v["point_mass_rejected"] for v in result["negative_controls"]))

    def test_one_failure_retains_fixed_denominators_and_invalidates_cell_null(self):
        self.outcomes[0] = {
            "ordinal": 0,
            "cell": 0,
            "replicate": 0,
            "status": "computational_failure",
        }
        result = sbc.aggregate(self.outcomes, self.reg)
        self.assertFalse(result["acceptance"])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["computational_failures"], 1)
        self.assertEqual(result["fixed_denominator"], 1024)
        self.assertEqual(len(result["original_outcome_sha256"]), 1024)
        self.assertTrue(all(g["denominator"] == 128 for g in result["gates"]))
        self.assertTrue(
            all(not g["null_valid"] and not g["passed"] for g in result["gates"] if g["cell"] == 0)
        )
        self.assertFalse(result["negative_controls"][0]["point_mass_rejected"])

    def test_statistical_failure_is_nonzero_without_computational_failure(self):
        for value in self.outcomes.values():
            value["score"]["rank_bins"] = [0] * len(value["score"]["rank_bins"])
        result = sbc.aggregate(self.outcomes, self.reg)
        self.assertFalse(result["acceptance"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["computational_failures"], 0)

    def test_missing_original_or_selected_identity_refused(self):
        absent = self.outcomes.copy()
        absent.pop(1023)
        with self.assertRaises(sbc.ProtocolError):
            sbc.aggregate(absent, self.reg)
        self.outcomes[0]["replicate"] = 1
        with self.assertRaises(sbc.ProtocolError):
            sbc.aggregate(self.outcomes, self.reg)

    def test_bool_bin_numeric_coverage_bad_control_and_unknown_status_refused(self):
        for key, value in (("rank_bins", True), ("coverage", 1), ("point_mass_rank_bins", 8)):
            outcomes = copy.deepcopy(self.outcomes)
            outcomes[0]["score"][key][0] = value
            with self.subTest(key=key), self.assertRaises(sbc.ProtocolError):
                sbc.aggregate(outcomes, self.reg)
        self.outcomes[0]["status"] = "discarded"
        with self.assertRaises(sbc.ProtocolError):
            sbc.aggregate(self.outcomes, self.reg)

    def test_fixed_registration_family_and_pause_cannot_change(self):
        for key, value in (
            ("total_cases", 1023),
            ("posterior_draws", 129),
            ("pause_after_committed", 65),
        ):
            altered = self.reg | {key: value}
            with self.subTest(key=key), self.assertRaises(sbc.ProtocolError):
                sbc.validate_registration(altered)

    def test_compact_aggregate_retains_full_original_hash(self):
        rows = {k: sbc.aggregate_row(v) for k, v in self.outcomes.items()}
        result = sbc.aggregate(rows, self.reg)
        self.assertEqual(result["original_outcome_sha256"]["0"], sbc.digest(self.outcomes[0]))
        self.assertNotIn("full_original_fake_targets", rows[0]["score"])

    def test_driver_import_and_all_fake_callbacks_load_no_numerical_runtime(self):
        code = (
            "import importlib.util,sys; s=importlib.util.spec_from_file_location('pure_protocol',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert {'numpy','pandas','torch','scipy','openecon'}.isdisjoint(sys.modules)"
        )
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(DRIVER)], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
