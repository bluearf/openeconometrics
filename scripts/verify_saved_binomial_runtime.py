"""Exact installed frozen saved-score execution, full state and genuine cold replay.

The separate native controller proves visible Windows tables. Neither proof
establishes generic observation intervals, all score providers or vendor parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import CodeType

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/saved_binomial_suest_acceptance.py"
ORACLE = "examples/saved_binomial_suest.py"
MARKER = "SAVED_BINOMIAL_SUEST_EIGHT_OK "
NAMES = (
    "cloglog_offset", "cloglog_frequency_cluster", "fractional_logit_corners",
    "fractional_probit_corners", "fractional_logit_frequency",
    "fractional_probit_probability_cluster", "fractional_logit_analytic",
    "fractional_probit_analytic_cluster",
)
ARTIFACTS = ("saved-binomial-suest-models.json", "saved-binomial-suest-data.parquet",
             *(f"saved-binomial-suest-{name}.tex" for name in NAMES))
MODULES = (
    "__init__", "models", "analysis", "analysis_contracts", "console_worker",
    "output_latex", "latex", "resources", "econometrics.registry",
    "econometrics.core", "econometrics.resident_cpu", "econometrics.glm.__init__",
    "econometrics.glm.commands", "econometrics.glm.glm", "econometrics.glm.common",
    "econometrics.glm.families", "econometrics.glm.kernels",
    "econometrics.postest.common", "econometrics.postest.scores",
    "econometrics.postest.suest", "econometrics.postest.inference",
    "engines.covariance", "engines.optimize", "engines.inference",
    "engines.distributions", "linear_ols.postestimation",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def normalized(code):
    # Preserve every code property except source-location names from packaging.
    return code.replace(co_filename="<bundled-source>", co_consts=tuple(
        normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts))


def committed_source(source_sha, relative):
    assert re.fullmatch(r"[0-9a-f]{40}", source_sha)
    payload = subprocess.check_output(["git", "show", f"{source_sha}:{relative}"], cwd=ROOT)
    assert 0 < len(payload) <= 2 * 1024 * 1024
    # Exact committed input must also be the actual inspected checkout file.
    assert (ROOT / relative).read_bytes() == payload, relative
    return payload


def load_controller(path):
    spec = importlib.util.spec_from_file_location("_saved_binomial_owned_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_code(example, oracle, source_sha, *, replay=False, require_frozen=True):
    assert re.fullmatch(r"[0-9a-f]{40}", source_sha)
    assert "def saved_binomial_acceptance(" in example and "def expected_covariance(" in oracle
    return example + "\nsaved_binomial_acceptance(" + repr(oracle) + ", " + repr(source_sha) + (
        f", replay={replay!r}, require_frozen={require_frozen!r}, emit=display)\n")


def statistical_state(state):
    return {key: value for key, value in state.items() if key not in {"id", "created_at"}}


def verify_run(run, project_root, source_sha, oracle_sha, *, replay=False, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    lines = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1 and len(run["stdout"].encode()) < 64 * 1024
    marker = json.loads(lines[0])
    assert marker["frozen"] is require_frozen and marker["fit_disabled_replay"] is replay
    assert marker["source_sha"] == source_sha and marker["oracle_sha256"] == oracle_sha
    assert marker["cases"] == list(NAMES)
    assert marker["component_models"] == 16 and marker["joint_systems"] == 8
    assert marker["sample_sizes"] == [218, 240] and marker["union_positions"] == list(range(240))
    assert set(marker["files"]) == set(ARTIFACTS)
    for name, expected in marker["files"].items():
        path = project_root / name
        assert not path.is_symlink() and path.is_file()
        assert type(expected["bytes"]) is int and 0 < path.stat().st_size == expected["bytes"] <= 2 * 1024 * 1024
        assert re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) and digest(path) == expected["sha256"]
    payload = json.loads((project_root / ARTIFACTS[0]).read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "source_sha", "oracle_sha256", "cases"}
    assert payload["schema"] == "saved-binomial-suest-v1"
    assert payload["source_sha"] == source_sha and payload["oracle_sha256"] == oracle_sha
    assert [record["name"] for record in payload["cases"]] == list(NAMES)
    assert len(run["outputs"]) == 8
    for index, (record, output) in enumerate(zip(payload["cases"], run["outputs"], strict=True)):
        assert set(record) == {"name", "models", "joint", "reference_covariance", "cross_model_test", "table_rows"}
        models, joint = record["models"], record["joint"]
        assert len(models) == 2 and [len(model["coefficients"]) for model in models] == [3, 2]
        assert [model["sample_positions"] for model in models] == [[i for i in range(240) if i % 11], list(range(240))]
        expected_link = "cloglog" if index < 2 else "logit" if index in (2, 4, 6) else "probit"
        expected_weight = (None, "fweight", None, None, "fweight", "pweight", "aweight", "aweight")[index]
        assert all(model["extra"]["link"] == expected_link and model["spec"]["weight_type"] == expected_weight for model in models)
        assert joint["sample_positions"] == list(range(240)) and len(joint["coefficients"]) == 5
        assert joint["provenance"]["device"] == "cpu" and joint["provenance"]["stata_parity_validated"] is False
        assert joint["inference"]["cluster_column"] == ("g" if index in (1, 5, 7) else None)
        reference = record["reference_covariance"]
        covariance = joint["covariance_matrix"]
        assert len(covariance) == len(reference) == 5
        for actual_row, expected_row in zip(covariance, reference, strict=True):
            assert len(actual_row) == len(expected_row) == 5
            assert all(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-12)
                       for a, b in zip(actual_row, expected_row, strict=True))
        assert max(abs(reference[i][j]) for i in range(3) for j in (3, 4)) > 1e-5
        test = record["cross_model_test"]
        assert test["distribution"] == "chi2" and test["df"] == 1
        difference = joint["coefficients"][1]["estimate"] - joint["coefficients"][4]["estimate"]
        variance = reference[1][1] + reference[4][4] - reference[1][4] - reference[4][1]
        expected_wald = difference ** 2 / variance
        assert math.isclose(test["statistic"], expected_wald, rel_tol=2e-10, abs_tol=2e-12)
        assert math.isclose(test["p_value"], math.erfc(math.sqrt(expected_wald / 2)), rel_tol=2e-10, abs_tol=2e-12)
        assert len(record["table_rows"]) == 7 and all(len(row) == 7 for row in record["table_rows"])
        assert marker["table_rows"][index] == record["table_rows"]
        assert output["type"] == "model"
        visible = output["data"]
        assert visible["display_omitted"] == ["sample_positions", "covariance_matrix"]
        assert statistical_state({key: value for key, value in visible.items() if key != "display_omitted"}) == statistical_state(
            {key: value for key, value in joint.items() if key not in {"sample_positions", "covariance_matrix"}})
        assert output["latex"] == (project_root / ARTIFACTS[index + 2]).read_text(encoding="utf-8")
        assert "\\begin{tabular}" in output["latex"] and output.get("latex_math")
    return {"payload": payload, "artifact_manifest": marker["files"]}


def verify(runtime, sdk_version, source_sha):
    from PyInstaller.archive.readers import CArchiveReader

    runtime = runtime.resolve(strict=True)
    if os.name == "nt" and (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("Windows acceptance requires disposable GitHub-hosted Windows.")
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in MODULES:
        relative = "src/openecon/" + name.replace(".", "/") + ".py"
        source = committed_source(source_sha, relative)
        module = "openecon" if name == "__init__" else ("openecon." + name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(compile(source, relative, "exec", dont_inherit=True)), module
        modules[module] = hashlib.sha256(source).hexdigest()
    for relative in ("scripts/verify_saved_binomial_runtime.py", "scripts/verify_regularized_all_family_runtime.py",
                     "desktop/scripts/verify_windows_runtime.py", "scripts/verify_bai_perron_runtime.py"):
        committed_source(source_sha, relative)
    example = committed_source(source_sha, EXAMPLE).decode()
    oracle = committed_source(source_sha, ORACLE).decode()
    oracle_sha = hashlib.sha256(oracle.encode()).hexdigest()
    guard = ("import importlib,sys\nfrom pathlib import Path\nassert getattr(sys,'frozen',False)\n"
             + "import openecon as oe\nassert oe.__version__ == " + repr(sdk_version) + "\n"
             + "for name in " + repr(tuple(modules)) + ":\n"
             + "    module=importlib.import_module(name)\n"
             + "    assert Path(module.__file__).is_relative_to(Path(sys._MEIPASS))\n")
    # Execute the fixture in its own namespace so its future import is first.
    def code(replay):
        return guard + "exec(" + repr(fixture_code(example, oracle, source_sha, replay=replay)) + ")\n"
    controller = load_controller(ROOT / "scripts/verify_regularized_all_family_runtime.py")
    with tempfile.TemporaryDirectory(prefix="openecon-saved-binomial-") as temporary:
        root = Path(temporary)
        owned = controller.owned_runtime(runtime, root, "unused", sdk_version)
        try:
            owned.project = owned.request("/api/desktop/local-projects", {"name": "Saved binomial score QA"}, desktop=True)["id"]
            owned.open_project()
            project = owned.project
            project_root = root / "projects" / project
            original_code = code(False)
            owned.call("/console/script", {"code": original_code, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": original_code, "timeout_seconds": 180})
            proof = verify_run(run, project_root, source_sha, oracle_sha)
            script = owned.call("/console/script")
            saved = next(item for item in owned.call("/console")["history"] if item["id"] == run["id"])
        finally:
            owned.close()
        owned = controller.owned_runtime(runtime, root, project, sdk_version)
        try:
            owned.open_project()
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
            reopened = next(item for item in owned.call("/console")["history"] if item["id"] == run["id"])
            assert all(reopened[key] == saved[key] for key in ("code", "stdout", "outputs", "events"))
            assert owned.call("/console/script") == script
            replay_run = owned.call("/console/execute", {"code": code(True), "timeout_seconds": 180})
            assert verify_run(replay_run, project_root, source_sha, oracle_sha, replay=True) == proof
        finally:
            owned.close()
    return {"status": "passed", "source_sha": source_sha, "sdk_version": sdk_version,
            "runtime_sha256": digest(runtime), "verifier_sha256": digest(__file__),
            "example_sha256": hashlib.sha256(example.encode()).hexdigest(), "oracle_sha256": oracle_sha,
            "compiled_modules_equal_source": modules, "component_models": 16, "joint_systems": 8,
            "model_outputs_create_and_replay": 16, "full_cross_covariance_and_independent_wald": True,
            "sixteen_complete_models_and_eight_joint_states_equal_after_restart": True,
            "saved_replay_with_fit_disabled": True, "no_worker_needed_for_history": True,
            "code_stdout_outputs_events_equal_after_restart": True, "editor_script_equal_after_restart": True,
            "owned_runtime_stopped": True, "temporary_profile_removed": not root.exists(),
            "source_path_injected": False, "native_window_verified": False, "public_release_delivered": False,
            "artifact_manifest": proof["artifact_manifest"], "complete_saved_state_sha256": digest_json(proof["payload"]),
            "ordered_outputs_sha256": digest_json(saved["outputs"]),
            "saved_code_stdout_outputs_events_sha256": digest_json({key: saved[key] for key in ("code", "stdout", "outputs", "events")})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-version", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        parser.error("--source-sha must be the exact committed source.")
    try:
        receipt = verify(args.runtime, args.sdk_version, args.source_sha)
    except BaseException as error:
        receipt = {"status": "error", "source_sha": args.source_sha, "sdk_version": args.sdk_version,
                   "error": f"{type(error).__name__}: {error}", "native_window_verified": False,
                   "public_release_delivered": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "component_models": 16, "joint_systems": 8}))
