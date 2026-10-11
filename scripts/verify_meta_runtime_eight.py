"""Verify frozen meta results, source identities and an owned backend restart.

Retains the isolated synthetic project and complete results. The temporary
executable/build outputs can be removed after this verifier completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import runpy

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/meta_runtime_eight.py"
FIXTURES = ROOT / "tests/fixtures/meta_extensions"
MODULES = (
    "openecon", "openecon.analysis", "openecon.analysis_contracts", "openecon.resources",
    "openecon.latex", "openecon.output_latex", "openecon.econometrics.core",
    "openecon.econometrics.summary_state", "openecon.econometrics.state_lifecycle",
    "openecon.econometrics.resident_cpu", "openecon.engines.distributions",
    "openecon.engines.optimize", "openecon.engines.contracts",
) + tuple("openecon.econometrics.meta" + suffix for suffix in (
    "", ".bias", ".common", ".dependent", ".dependent_kernels", ".dependent_post",
    ".diagnostics", ".effects", ".kernels", ".models", ".multilevel",
    ".multilevel_kernels", ".selection", ".small_sample"))


def identity(runtime, source_ref):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    output = {}
    for module in MODULES:
        path = ROOT / "src" / Path(*module.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert normalized(archive.extract(module)) == normalized(compile(source, str(path), "exec", dont_inherit=True)), module
        assert path.read_bytes() == source, module
        output[module] = hashlib.sha256(source).hexdigest()
    return output


def code(directory, *, replay=False):
    return ("import sys, importlib.util\nassert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"META_FIXTURE_DIRECTORY = {str(FIXTURES)!r}\n"
            f"META_RESULT_DIRECTORY = {str(directory)!r}\n"
            f"META_REPLAY = {replay!r}\n" + EXAMPLE.read_text())


def execute(owned, source):
    run = owned.call("/console/execute", {"code": source, "timeout_seconds": 120})
    assert run["status"] == "ok", run.get("error")
    assert "Display limit reached" not in run["stdout"]
    return run


def references(directory):
    """Check frozen numbers against committed R receipts and independent t laws.

    Runs on the host only. No reference package is imported into the frozen
    backend and no reference tolerance is relaxed from the existing tests.
    """
    import numpy as np
    from scipy import stats

    unpack = runpy.run_path(str(EXAMPLE))["unpack"]
    results = {name: unpack(payload) for name, payload in
               json.loads((directory / "results.json").read_text()).items()}
    oracle = json.loads((FIXTURES / "oracles.json").read_text())["results"]
    comparisons = 0

    def close(actual, expected, *, rtol=1e-6, atol=1e-10):
        nonlocal comparisons
        np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                                   rtol=rtol, atol=atol)
        comparisons += 1

    for name, key in (("ml", "konst_int_ML_z"), ("reml", "che_REML")):
        result, ref = results[name], oracle[key]
        close(result["coefficients"].estimate, np.atleast_1d(ref["beta"]), atol=1e-9)
        close(result["covariance"], np.atleast_2d(ref["vb"]), rtol=2e-6)
        close(result["variance_components"].sigma2, ref["sigma2"], rtol=2e-6)
        close(result["component_covariance"], ref["vvc"], rtol=2e-5)
        close(result["fit"].iloc[0].loglik, ref["fitstats"]["logLik"], rtol=0, atol=2e-9)
        if ref["profile"] is not None:
            close(result["variance_components"][["profile_ci_low", "profile_ci_high"]],
                  np.atleast_2d(ref["profile"])[:, 1:], rtol=2e-6)
        for row, lrt in zip(result["variance_components"].itertuples(), ref["lrt"], strict=True):
            close(row.lrt_statistic, lrt["LRT"], rtol=2e-6, atol=2e-8)
        coef = result["coefficients"]
        critical = stats.norm.isf(.025) if name == "ml" else stats.t.isf(.025, result.attrs["residual_df"])
        p = 2 * (stats.norm.sf(np.abs(coef.statistic)) if name == "ml" else
                 stats.t.sf(np.abs(coef.statistic), result.attrs["residual_df"]))
        close(coef.p_value, p, rtol=1e-8)
        close(coef.ci_low, coef.estimate - critical * coef.std_error, rtol=1e-8)
        close(coef.ci_high, coef.estimate + critical * coef.std_error, rtol=1e-8)
        prediction = results["prediction_" + name]["prediction"]
        close(prediction.estimate, np.atleast_1d(ref["predict"]["pred"]), rtol=2e-6, atol=1e-9)
        close(prediction.std_error, np.atleast_1d(ref["predict"]["se"]), rtol=2e-6)
        se = np.sqrt(prediction.std_error**2 + sum(result.attrs["sigma2"]))
        for field, expected in (("ci_low", prediction.estimate-critical*prediction.std_error),
                                ("ci_high", prediction.estimate+critical*prediction.std_error),
                                ("pi_low", prediction.estimate-critical*se),
                                ("pi_high", prediction.estimate+critical*se)):
            close(prediction[field], expected, rtol=1e-8)
    for name in ("cr2", "dependent_study_ML", "dependent_study_REML", "dependent_effect_ML",
                 "dependent_effect_REML", "dependent_common"):
        result = results[name] if name == "cr2" else results[name + "_cr2"]
        ref = oracle["konst_cr2" if name == "cr2" else name]
        close(result["covariance"], np.atleast_2d(ref["vcov"]), rtol=2e-5)
        close(result["coefficients"].df, np.atleast_1d(ref["df"]), rtol=2e-5)
        close(result["coefficients"].p_value, np.atleast_1d(ref["p"]), rtol=5e-5)
        tests = result["tests"].set_index("test")
        hypotheses = [("joint moderators", "moderators")]
        if name == "cr2":
            hypotheses.append(("contrasts", "contrasts"))
        for label, key in hypotheses:
            for field, reference in (("F_statistic", "F"), ("df2", "df2"), ("p_value", "p")):
                close(tests.loc[label, field], ref[key][reference], rtol=5e-5)
    for label in ("hack_L0", "hack_R0", "hack_right", "asym_L0", "asym_R0"):
        name, variant = label.split("_")
        result, ref = results["trimfill_" + label], oracle[f"trimfill_{name}_REML_{variant}"]
        row, adjusted = result["summary"].iloc[0], result["estimates"].iloc[1]
        assert row.k0 == ref["k0"] and row.side == ref["side"]
        close([adjusted.estimate, adjusted.std_error, adjusted.tau2, adjusted.ci_low, adjusted.ci_high],
              [np.atleast_1d(ref["beta"])[0], ref["se"], ref["tau2"], ref["ci_lb"], ref["ci_ub"]], rtol=2e-7, atol=1e-12)
        assert len(result["filled"]) == int(ref["k0"])
        if int(ref["k0"]):
            close(result["filled"].yi, np.atleast_1d(ref["yi"])[-int(ref["k0"]):], rtol=2e-7, atol=1e-12)
            close(result["filled"].vi, np.atleast_1d(ref["vi"])[-int(ref["k0"]):], rtol=0, atol=0)
    for name in ("hack", "asym"):
        result = results["pet_" + name]
        for model, key, table in (("PET", "pet", "pet_covariance"), ("PEESE", "peese", "peese_covariance")):
            ref = oracle[key + "_" + name]
            rows = result["coefficients"].query("model == @model")
            close(rows.estimate, ref["beta"], rtol=1e-10, atol=1e-13)
            close(result[table], ref["vcov"], rtol=1e-9)
            close(rows.p_value, ref["p"], rtol=1e-8, atol=1e-15)
    for name, key in (("estimated", "hack_4"), ("fixed", "hack_fixed"), ("zero", "hack_ee"), ("moderator", "moderator")):
        result, ref = results["selection_" + name], oracle["selection_" + key]
        close(result["coefficients"].estimate, np.atleast_1d(ref["beta"]), rtol=2e-5, atol=5e-6)
        close(result["coefficients"].std_error, np.atleast_1d(ref["se"]), rtol=5e-5)
        close(result["heterogeneity"].iloc[0].tau2, ref["tau2"], rtol=5e-5, atol=1e-9)
        close(result["fit"].iloc[0].loglik, ref["ll"], rtol=0, atol=1e-8)
        close(result["selection_weights"].estimate, np.atleast_1d(ref["delta"]), rtol=5e-5)
    return {"array_comparisons": comparisons, "source": "committed R fixtures and host SciPy reference laws",
            "oracle_sha256": digest(FIXTURES / "oracles.json"), "all_passed": True}


def verify(runtime, output, source_ref):
    runtime = runtime.resolve(strict=True)
    output = output.resolve(strict=True)
    source_ref = subprocess.check_output(
        ["git", "rev-parse", source_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    identities = identity(runtime, source_ref)
    profile, results = output / "profile", output / "frozen-results"
    assert not profile.exists() and not results.exists(), "Use a fresh directory; preserve previous saved projects"
    profile.mkdir(parents=True)
    owned = OwnedRuntime(runtime, profile, "unused")
    try:
        project = owned.request("/api/desktop/local-projects", {"name": "Meta eight frozen acceptance"}, desktop=True)["id"]
        owned.project = project
        owned.open_project()
        source = code(results)
        owned.call("/console/script", {"code": source, "name": "analysis.py"}, method="PUT")
        run = execute(owned, source)
        marker = "META_EIGHT_OK "
        proofs = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines() if line.startswith(marker)]
        assert len(proofs) == 1
        proof = proofs[0]
        assert proof["frozen"] and proof["full_roundtrip"] and len(proof["negative_cases"]) == 7
        assert digest(results / "results.json") == proof["sha256"]
        source_results = output / "source-results" / "results.json"
        assert results.joinpath("results.json").read_bytes() == source_results.read_bytes(), "Complete source/frozen payload difference"
        assert results.joinpath("failures.json").read_bytes() == (source_results.parent / "failures.json").read_bytes()
        independent = references(results)
        payload = json.loads(source_results.read_text())
        assert set(payload) == set(proof["cases"])
        assert sum(len(case["table_order"]) for case in payload.values()) == proof["tables"]
        assert len(run["outputs"]) == 1 and run["outputs"][0]["type"] == "table"
        assert run["outputs"][0]["data"]["total_rows"] == len(payload)
        assert len(run["outputs"][0]["data"]["rows"]) == len(payload)
        assert "\\begin{tabular}" in run["outputs"][0]["latex"] or "\\begin{longtable}" in run["outputs"][0]["latex"]
        document = owned.call("/console/script")
        saved = next(item for item in owned.call("/console")["history"] if item["id"] == run["id"])
        original = {key: saved[key] for key in ("code", "stdout", "outputs", "events")}
    finally:
        owned.close()
    owned = OwnedRuntime(runtime, profile, project)
    try:
        owned.open_project()
        saved = next(item for item in owned.call("/console")["history"] if item["id"] == run["id"])
        assert {key: saved[key] for key in original} == original
        assert owned.call("/console/script") == document
        replay = execute(owned, code(results, replay=True))
        assert "META_EIGHT_REPLAY_OK" in replay["stdout"]
        assert digest(results / "results.json") == proof["sha256"]
    finally:
        owned.close()
    return {"status": "passed", "source_ref": source_ref, "runtime_sha256": digest(runtime),
            "compiled_modules": identities, "proof": proof,
            "fixture_hashes": {path.name: digest(path) for path in sorted(FIXTURES.glob("*.csv"))},
            "oracle_sha256": digest(FIXTURES / "oracles.json"), "example_sha256": digest(EXAMPLE),
            "independent_reference_checks": independent,
            "verifier_sha256": digest(__file__), "project_id": project, "profile": str(profile),
            "execution_id": run["id"], "execution_duration_ms": run["duration_ms"],
            "replay_execution_id": replay["id"], "replay_duration_ms": replay["duration_ms"],
            "source_frozen_complete_results_equal": True,
            "full_code_stdout_outputs_events_and_script_equal_after_restart": True,
            "optimizer_free_saved_prediction_and_cr2_replay": True,
            "results_files_unchanged_after_restart": True, "owned_workers_stopped": True,
            "saved_projects_results_preserved": True,
            "source_path_injected": False, "development_estimators_present": False,
            "native_installed_ui_verified": False, "windows_verified": False,
            "coverage_simulations_verified": False, "public_release_delivered": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime, args.output_directory, args.source_ref)
    destination = args.output_directory / "frozen-receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cases": len(receipt["proof"]["cases"]),
                      "tables": receipt["proof"]["tables"], "negative_cases": len(receipt["proof"]["negative_cases"])}))
