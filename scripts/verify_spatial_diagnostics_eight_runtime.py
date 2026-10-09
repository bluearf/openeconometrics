"""Eight saved OLS spatial diagnostics: compiled identity, full state and restart."""
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
MARKER = "SPATIAL_DIAGNOSTICS_EIGHT_OK "
METHODS = {1: ["moran_normal", "moran_gaussian_mc", "lm_error"],
           2: ["lm_lag", "robust_lm_error", "robust_lm_lag"],
           3: ["lm_joint", "wx_f"]}
MODULES = ["econometrics.spatial.diagnostics", "econometrics.spatial.diagnostic_kernels",
           "econometrics.spatial.weights", "econometrics.spatial.__init__",
           "econometrics.postest.common", "econometrics.postest.inference",
           "econometrics.postest.index_codec",
           "econometrics.core", "econometrics.summary_state", "econometrics.resident_cpu",
           "linear_ols.__init__", "linear_ols.spec", "linear_ols.design", "linear_ols.estimation",
           "linear_ols.publication", "linear_ols.postestimation", "linear_ols.lags",
           "engines.linalg", "engines.covariance", "engines.distributions", "engines.inference",
           "engines.contracts", "analysis", "analysis_contracts", "models", "resources",
           "frame", "dataset", "econometrics.registry", "__init__"]



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
    for name, state in proof["complete_summary_states"].items():
        assert hashlib.sha256(json.dumps(state, allow_nan=False, sort_keys=True).encode()).hexdigest() == proof["summary_hashes"][name]
    for output, contract in zip(run["outputs"], proof["output_contracts"], strict=True):
        assert output["type"] == "table"
        table = output["data"]
        assert len(table["rows"]) == table["total_rows"] == contract["rows"]
        assert table["columns"] == contract["columns"]
        original = proof["complete_summary_states"][contract["method"]]["tables"][contract["key"]]
        assert table["rows"] == original["data"]
        assert table["index"] == [[label] for label in original["index"]]
        assert table["index_names"] == contract["index_names"]
        assert proof["complete_summary_states"][contract["method"]]["attrs"]["table_index_names"][contract["key"]] == contract["index_names"]
        assert any(text in output.get("latex", "") for text in ("\\begin{tabular}", "\\begin{longtable}"))
    assert proof["restored_ols_used_without_original_refit"] and proof["complete_source_model_retained"]
    model = proof["source_model_state"]
    assert model["spec"]["estimator"] == "ols" and model["spec"]["covariance"] == "nonrobust"
    assert len(model["coefficients"]) == len(model["covariance_matrix"])
    assert set(proof["complete_summary_states"]) == set(proof["methods"])
    for method, state in proof["complete_summary_states"].items():
        assert json.loads(state["attrs"]["source_model"]) == model
        assert len(state["tables"]["tests"]["data"]) == 1
        assert len(state["tables"]["sample"]["data"]) == model["nobs"]
        assert len(state["tables"]["design"]["data"]) == model["nobs"]
        assert len(state["tables"]["weights"]["data"]) == model["nobs"]
        assert all(len(row) == model["nobs"] for row in state["tables"]["weights"]["data"])
        assert len(state["tables"]["null_simulation"]["data"]) == (39 if method == "moran_gaussian_mc" else 0)

    return proof


def example_code(part):
    return """import sys
from pathlib import Path
import importlib
assert getattr(sys, 'frozen', False)
for module_name in ('spatial.diagnostics', 'spatial.diagnostic_kernels'):
    assert Path(importlib.import_module('openecon.econometrics.' + module_name).__file__).is_relative_to(Path(sys._MEIPASS))
""" + (ROOT/f"docs/examples/spatial_diagnostics_eight_{part}.py").read_text()



def verify(runtime, *, capture_path=None):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in MODULES:
        path = ROOT/"src/openecon"/(name.replace(".","/")+".py")
        module = ("openecon."+name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(compile(path.read_text(), str(path), "exec", dont_inherit=True)), module
        modules[module] = digest(path)
    runs = []
    with tempfile.TemporaryDirectory(prefix="openecon-spatial-diagnostics-eight-proof-") as temporary:
        root = Path(temporary)
        owned = OwnedRuntime(runtime,root,"unused")
        try:
            project = owned.request("/api/desktop/local-projects",{"name":"Spatial Diagnostics Eight QA"},desktop=True)["id"]
            owned.project = project
            owned.open_project()
            for part in METHODS:
                code = example_code(part)
                owned.call("/console/script",{"code":code,"name":"analysis.py"},method="PUT")
                run = owned.call("/console/execute",{"code":code,"timeout_seconds":120})
                if capture_path is not None:
                    capture_path.write_text(json.dumps([item["run"] for item in runs] + [run], indent=2) + "\n")
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
            assert len(history) == len(METHODS)
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
        examples_sha256={part:digest(ROOT/f"docs/examples/spatial_diagnostics_eight_{part}.py") for part in METHODS},
        complete_output_tables=sum(len(item["proof"]["output_contracts"]) for item in runs),
        all_eight_complete_summaries_saved_and_restored=True,complete_source_model_states_retained=True,full_code_stdout_outputs_events_equal_after_restart=True,
        active_final_script_equal_after_restart=True,all_scripts_retained_in_run_history=True,
        no_worker_needed_for_history=True,source_path_injected=False,owned_runtime_stopped=True,
        temporary_data_removed=not root.exists(),human_data_access=False,native_window_verified=False,public_release_delivered=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt = verify(args.runtime, capture_path=args.output.with_name(args.output.stem + "-runs.json"))
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("status","complete_output_tables","all_eight_complete_summaries_saved_and_restored")}))
