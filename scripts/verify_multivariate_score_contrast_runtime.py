"""Owned frozen execution/restart, complete score reconstruction and host oracles.

Native Run and native application Quit/reopen are separate evidence. This
verifier retains its fresh owned result directory and isolated project profile.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized
from verify_frequency_uncertainty_runtime import MODULES as TRAINING_MODULES

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/multivariate_score_contrasts_eight.py"
MARKER = "MULTIVARIATE_SCORE_CONTRASTS_EIGHT_OK "
CASES = ("pca_covariance", "pca_correlation", "cca_iid", "cca_frequency",
         "pf_iid", "target_iid", "pf_frequency", "target_frequency")
MODULES = (*TRAINING_MODULES, "openecon.econometrics.multivariate.score_uncertainty",
           "openecon.econometrics.multivariate.score_contrasts")


def source_identity(runtime, source_ref):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    identities = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert normalized(archive.extract(name)) == normalized(compile(source, str(path), "exec", dont_inherit=True)), name
        identities[name] = hashlib.sha256(source).hexdigest()
    return identities


def verify_outputs(run):
    assert run["status"] == "ok", run.get("error")
    matches = [json.loads(line[len(MARKER):]) for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(matches) == 1
    proof = matches[0]
    assert proof["cases"] == list(CASES) and proof["frozen"]
    assert proof["bootstrap_replications"] == 199 and proof["fixture_rows"] == 603
    assert proof["query_positions"] == [0, 2] and proof["full_saved_roundtrip"]
    assert len(run["outputs"]) == 8
    assert all(item["type"] == "table" and any(token in item["latex"] for token in
               ("\\begin{tabular}", "\\begin{longtable}")) for item in run["outputs"])
    assert "Display limit reached" not in run["stdout"]
    return proof


def validate_files(directory, proof, run):
    hashes = {case: digest(directory / (case + ".json")) for case in CASES}
    assert hashes == proof["hashes"]
    for name, displayed in zip(CASES, run["outputs"], strict=True):
        payload = json.loads((directory / (name + ".json")).read_text())
        fit, post = [json.loads(payload[key]["summary"]) for key in ("fit", "post")]
        attrs, tables = post["attrs"], post["tables"]
        assert attrs["replications"] == 199 and attrs["covariance_divisor"] == 198
        assert attrs["score_attrs"]["query_positions"] == [0, 2] and attrs["score_attrs"]["physical_query_rows"] == 3
        assert len(tables["replicates"]["data"]) == 199
        assert len(tables["covariance"]["data"]) == attrs["contrast_dimension"]
        for key, value in fit["tables"].items():
            assert tables["score__fit__" + key] == value
        assert displayed["data"]["rows"] == tables["estimates"]["data"]
        assert displayed["data"]["columns"] == tables["estimates"]["columns"]
        assert displayed["data"]["total_rows"] == attrs["contrast_dimension"]
    return hashes


def functions():
    source = EXAMPLE.read_text()
    return "\n".join(ast.get_source_segment(source, node) for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in ("restored_frame", "restore_pack"))


def reconstruction_code(directory):
    return """import json, sys
from pathlib import Path
import pandas as pd
import openecon as oe
from openecon.econometrics.postest.index_codec import decode
from openecon.resources import use_workspace_budget
assert getattr(sys, 'frozen', False)
""" + functions() + f"\nroot = Path({str(directory)!r})\n" + """
fixtures = json.loads((root/'fixtures.json').read_text())
for name in """ + repr(CASES) + """:
    payload = json.loads((root/(name+'.json')).read_text())
    original = restore_pack(payload['post'])
    fit = oe.restore_summary(payload['fit']['summary'])
    query = restored_frame(fixtures['canonical_query' if name.startswith('cca') else 'query'])
    function = oe.pca_fweight_bootstrap_score_contrasts if name.startswith('pca') else oe.canon_bootstrap_score_contrasts if name.startswith('cca') else oe.factor_bootstrap_score_contrasts
    with use_workspace_budget(512):
        actual = function(fit, query, original["contrasts"], confidence=original.attrs["confidence"])
    assert oe.summary_state(actual) == payload['post']['summary'], name
print('MULTIVARIATE_SCORE_CONTRAST_FULL_RESULTS_RECONSTRUCTED')
"""


def host_oracles(directory):
    """Independent full refits of frozen outputs, outside the frozen runtime."""
    sys.path.insert(0, str(ROOT / "tests"))
    import numpy as np
    import pandas as pd
    from openecon.econometrics.postest.index_codec import decode
    from openecon.econometrics.summary_state import restore_summary
    from test_multivariate_score_uncertainty import oracle
    fixtures = json.loads((directory / "fixtures.json").read_text())

    def frame(key):
        saved = fixtures[key]
        output = pd.DataFrame(saved["data"], columns=saved["columns"])
        labels, names = [decode(code) for code in saved["index_codes"]], [decode(code) for code in saved["index_names"]]
        output.index = pd.MultiIndex.from_tuples(labels, names=names) if len(names) > 1 else pd.Index(labels, name=names[0], tupleize_cols=False)
        output.columns.name = decode(saved["column_axis_names"][0])
        return output

    for case in CASES:
        payload = json.loads((directory / (case + ".json")).read_text())
        fit, actual = [restore_summary(payload[key]["summary"]) for key in ("fit", "post")]
        names = fit.attrs["variables"]
        source = frame("canonical" if case.startswith("cca") else "spectral")
        query = frame("canonical_query" if case.startswith("cca") else "query").iloc[[0, 2]]
        point, draws, coefficients, coefficient_draws = oracle(case, source, names, "freq", fit.attrs.get("target"), fit, query)
        contrast = actual["contrasts"].to_numpy()
        axes = 4 if case.startswith("cca") else 2
        reduced = contrast[:, [*range(axes), *range(2*axes, 3*axes)]]
        projected_point, projected_draws = reduced @ point, draws @ reduced.T
        covariance = np.cov(projected_draws, rowvar=False, ddof=1)
        se = np.sqrt(covariance.diagonal())
        deviations = (projected_draws-projected_point)/se
        maxima = np.abs(deviations).max(axis=1)
        critical = np.quantile(maxima, .95)
        for table, expected in (("replicates", projected_draws), ("covariance", covariance),
                                ("standardized_deviations", deviations), ("joint_maxima", maxima[:, None]),
                                ("score__replicates", draws), ("score__point_score_coefficients", coefficients),
                                ("score__replicate_score_coefficients", coefficient_draws)):
            np.testing.assert_allclose(actual[table], expected, atol=4e-11)
        np.testing.assert_allclose(actual["estimates"].estimate, projected_point, atol=4e-11)
        np.testing.assert_allclose(actual["estimates"].std_error, se, atol=4e-11)
        np.testing.assert_allclose(actual["estimates"][["ci_lower", "ci_upper"]],
                                   np.array([projected_point-critical*se, projected_point+critical*se]).T, atol=4e-11)
        np.testing.assert_allclose(actual["estimates"][["marginal_percentile_lower", "marginal_percentile_upper"]],
                                   np.quantile(projected_draws, [.025, .975], axis=0).T, atol=4e-11)
        np.testing.assert_allclose(actual["estimates"].bootstrap_bias, projected_draws.mean(0)-projected_point, atol=4e-11)
        np.testing.assert_allclose(actual.attrs["critical_value"], critical, atol=4e-11)
    return {name: digest(ROOT / "tests" / name) for name in ("test_multivariate_score_contrasts.py", "test_multivariate_score_uncertainty.py",
            "test_pca_frequency_uncertainty.py", "test_canon_frequency_uncertainty.py", "test_factor_frequency_uncertainty.py")}


def verify(runtime, results, source_ref):
    assert not results.exists(), "Use a fresh owned result path"
    identities = source_identity(runtime, source_ref)
    code = ("import sys, importlib.util\nassert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\nassert importlib.util.find_spec('statsmodels') is None\n"
            f"CONTRAST_RESULT_DIRECTORY = {str(results)!r}\n" + EXAMPLE.read_text())
    with nullcontext(results.parent / "profile") as root:
        assert not root.exists(), "Preserve previous QA projects; choose a fresh profile"
        root.mkdir(parents=True)
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            project = owned.request("/api/desktop/local-projects", {"name": "Score contrast frozen QA"}, desktop=True)["id"]
            owned.project = project
            owned.open_project()
            owned.call("/console/script", {"code": code, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
            proof = verify_outputs(run)
            hashes = validate_files(results, proof, run)
            document = owned.call("/console/script")
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            history = owned.call("/console")["history"]
            assert len(history) == 1
            assert all(history[0][key] == run[key] for key in ("id", "code", "stdout", "outputs", "events"))
            assert owned.call("/console/script") == document
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
            reopened = owned.call("/console/execute", {"code": reconstruction_code(results), "timeout_seconds": 120})
            assert reopened["status"] == "ok", reopened.get("error")
            assert "MULTIVARIATE_SCORE_CONTRAST_FULL_RESULTS_RECONSTRUCTED" in reopened["stdout"]
        finally:
            owned.close()
    oracles = host_oracles(results)
    return {"status": "passed", "frozen": True, "source_ref": source_ref, "compiled_modules": identities,
            "runtime_sha256": digest(runtime), "duration_ms": run["duration_ms"], "proof": proof,
            "complete_result_hashes": hashes, "fixture_sha256": digest(results / "fixtures.json"),
            "saved_synthetic_project": project, "saved_profile": str(root),
            "full_results_reconstructed_after_runtime_restart": True, "native_ui_verified": False,
            "host_independent_full_refits": 8 * 199, "oracle_sources": oracles,
            "external_reference_packages_absent_in_frozen_runtime": True, "public_release_delivered": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Choose a fresh receipt path"
    receipt = verify(args.runtime.resolve(strict=True), args.results.resolve(), args.source_ref)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "frozen", "duration_ms", "host_independent_full_refits")}))
