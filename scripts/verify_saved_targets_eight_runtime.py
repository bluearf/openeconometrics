"""Eight restored model targets: compiled identity, complete state and restart."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import zlib

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
MARKER = "SAVED_TARGETS_EIGHT_OK "
METHODS = {1: ["sreg", "mmreg", "ivcue", "mixedflex"], 2: ["sar", "sem", "sac", "sdm"]}
MODULES = ["econometrics.postest.linear_prediction", "econometrics.postest.prediction",
           "econometrics.postest.streaming_prediction", "econometrics.postest.inference",
           "econometrics.postest.index_codec", "econometrics.spatial.prediction",
           "econometrics.spatial.weights", "econometrics.spatial.kernels",
           "econometrics.spatial.estimators", "econometrics.robust.smm",
           "econometrics.robust.__init__", "econometrics.iv.cue", "econometrics.iv.__init__",
           "econometrics.mixed.flexible_lmm", "econometrics.mixed.__init__",
           "prediction_capabilities", "econometrics.summary_state", "econometrics.resident_cpu",
           "resources", "econometrics.core", "engines.distributions", "analysis", "models",
           "engines.linalg", "engines.optimize", "engines.inference", "engines.contracts",
           "econometrics.mixed.extended_common", "econometrics.glm.families",
           "econometrics.registry", "analysis_contracts", "frame", "dataset", "__init__"]



def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_outputs(run, *, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    matches = [json.loads(zlib.decompress(base64.b64decode(line[len(MARKER):])))
               for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(matches) == 1
    proof = matches[0]
    assert proof["frozen"] is require_frozen and proof["methods"] == METHODS[proof["part"]]
    assert proof["all_complete_state_restorations_equal"] and not proof["third_party_estimation_imports"]
    assert len(run["outputs"]) == len(proof["output_contracts"])
    for name, state in proof["summary_states"].items():
        assert hashlib.sha256(json.dumps(state, allow_nan=False, sort_keys=True).encode()).hexdigest() == proof["summary_hashes"][name]
    for output, contract in zip(run["outputs"], proof["output_contracts"], strict=True):
        assert output["type"] == "table"
        table = output["data"]
        assert len(table["rows"]) == table["total_rows"] == contract["rows"]
        assert table["columns"] == contract["columns"]
        original = proof["summary_states"][contract["method"]]["tables"][contract["key"]]
        assert table["rows"] == original["data"][contract["start"]:contract["start"]+contract["rows"]]
        assert table["index"] == [[label] for label in original["index"]]
        assert table["index_names"] == contract["index_names"]
        assert proof["summary_states"][contract["method"]]["attrs"]["table_index_names"][contract["key"]] == contract["index_names"]
        assert any(text in output.get("latex", "") for text in ("\\begin{tabular}", "\\begin{longtable}"))
    assert proof["restored_models_used_without_refit"]
    assert set(proof["model_states"]) == set(proof["methods"])
    for name, model in proof["model_states"].items():
        assert model["spec"]["estimator"] == name
        p = len(model["coefficients"])
        assert len(model["covariance_matrix"]) == p
        assert all(len(row) == p for row in model["covariance_matrix"])
        assert proof["summary_states"][name]["attrs"]["refitted"] is False
    if proof["part"] == 2:
        for name, state in proof["summary_states"].items():
            n = state["attrs"]["n"]
            assert len(state["tables"]["mean_covariance"]["data"]) == n
            assert all(len(row) == n for row in state["tables"]["mean_covariance"]["data"])
            assert len(state["tables"]["sample"]["data"]) == n

    return proof


def example_code(part):
    return """import sys
from pathlib import Path
import importlib
assert getattr(sys, 'frozen', False)
for module_name in ('postest.linear_prediction', 'spatial.prediction'):
    assert Path(importlib.import_module('openecon.econometrics.' + module_name).__file__).is_relative_to(Path(sys._MEIPASS))
""" + (ROOT/f"docs/examples/saved_targets_eight_{part}.py").read_text()



def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in [*MODULES, "econometrics.spatial.__init__"]:
        path = ROOT/"src/openecon"/(name.replace(".","/")+".py")
        module = ("openecon."+name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(compile(path.read_text(), str(path), "exec", dont_inherit=True)), module
        modules[module] = digest(path)
    runs = []
    with tempfile.TemporaryDirectory(prefix="openecon-saved-targets-eight-proof-") as temporary:
        root = Path(temporary)
        owned = OwnedRuntime(runtime,root,"unused")
        try:
            project = owned.request("/api/desktop/local-projects",{"name":"Saved Targets Eight QA"},desktop=True)["id"]
            owned.project = project
            owned.open_project()
            for part in (1,2):
                code = example_code(part)
                owned.call("/console/script",{"code":code,"name":"analysis.py"},method="PUT")
                run = owned.call("/console/execute",{"code":code,"timeout_seconds":120})
                proof = verify_outputs(run)
                assert proof["part"] == part
                runs.append(dict(run=run,proof=proof))
            document = owned.call("/console/script")
        finally:
            owned.close()
        owned = OwnedRuntime(runtime,root,project)
        try:
            owned.open_project()
            history = owned.call("/console")["history"]
            assert len(history) == 2
            for item in runs:
                after = next(row for row in history if row["id"] == item["run"]["id"])
                assert all(after[key] == item["run"][key] for key in ("code","stdout","outputs","events"))
                verify_outputs(after)
            assert owned.call("/console/script") == document
            assert owned.request("/api/desktop/status",desktop=True)["console"]["pid"] is None
        finally:
            owned.close()
    return dict(status="passed",compiled_modules_equal_source=modules,runtime_sha256=digest(runtime),
        runs=[dict(run_id=item["run"]["id"],duration_ms=item["run"]["duration_ms"],proof=item["proof"],
                   hashes={key:hashed(item["run"][key]) for key in ("code","stdout","outputs","events")}) for item in runs],
        examples_sha256={part:digest(ROOT/f"docs/examples/saved_targets_eight_{part}.py") for part in (1,2)},
        complete_output_tables=sum(len(item["proof"]["output_contracts"]) for item in runs),
        all_eight_complete_summaries_saved_and_restored=True,all_eight_complete_model_states_retained=True,full_code_stdout_outputs_events_equal_after_restart=True,
        active_second_script_equal_after_restart=True,first_script_retained_in_run_history=True,
        no_worker_needed_for_history=True,source_path_injected=False,owned_runtime_stopped=True,
        temporary_data_removed=not root.exists(),human_data_access=False,native_window_verified=False,public_release_delivered=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("status","complete_output_tables","all_eight_complete_summaries_saved_and_restored")}))
