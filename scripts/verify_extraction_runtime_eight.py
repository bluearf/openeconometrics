"""Verify complete frozen extraction results and an owned backend restart.

The synthetic saved project/results are retained. Generated runtime/build and
dependency files are separate cleanup targets after this verifier completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/extraction_runtime_eight.py"
MODULES = (
    "openecon", "openecon.analysis", "openecon.analysis_contracts", "openecon.resources",
    "openecon.latex", "openecon.output_latex", "openecon.econometrics.core",
    "openecon.econometrics.summary_state", "openecon.econometrics.state_lifecycle",
    "openecon.econometrics.resident_cpu", "openecon.econometrics.postest.index_codec",
    "openecon.engines.distributions", "openecon.engines.optimize", "openecon.engines.contracts",
) + tuple("openecon.econometrics.multivariate" + suffix for suffix in (
    "", ".common", ".factor", ".pca", ".extraction", ".extraction_additional",
    ".extraction_extensions", ".extraction_uncertainty", ".rotation",
    ".rotation_extensions", ".summary", ".uncertainty", ".factor_uncertainty",
    ".replay", ".weighted"))


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
            f"EXTRACTION_RESULT_DIRECTORY = {str(directory)!r}\n"
            f"EXTRACTION_REPLAY = {replay!r}\n" + EXAMPLE.read_text())


def execute(owned, source):
    run = owned.call("/console/execute", {"code": source, "timeout_seconds": 120})
    assert run["status"] == "ok", run.get("error")
    assert "Display limit reached" not in run["stdout"]
    return run


def references(directory):
    """Independent frozen-output moments, objectives, eigensystems and inference.

    NumPy and SciPy run on the development host. No reference estimator is
    injected into the frozen backend. All B draws enter each uncertainty check.
    Focused tests additionally refit every planned independent method oracle.
    """
    import numpy as np
    import pandas as pd
    from scipy import linalg, optimize

    unpack = runpy.run_path(str(EXAMPLE))["unpack"]
    results = {name: unpack(payload) for name, payload in
               json.loads((directory / "results.json").read_text()).items()}
    stored_inputs = json.loads((directory / "inputs.json").read_text())
    one = pd.DataFrame(**stored_inputs["onefactor"])
    two = pd.DataFrame(**stored_inputs["twofactor"])
    complete = two.dropna().to_numpy(dtype=float)
    r = np.corrcoef(complete, rowvar=False)
    comparisons = 0

    def close(actual, expected, *, rtol=1e-10, atol=1e-12):
        nonlocal comparisons
        np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                                   rtol=rtol, atol=atol)
        comparisons += 1

    def gls_objective(psi):
        roots = np.linalg.eigvalsh(r / np.sqrt(np.outer(psi, psi)))
        return np.sum(np.square(roots[:-2]-1)) / 2

    gls = optimize.minimize(gls_objective, np.full(len(r), .45), method="Nelder-Mead",
                            bounds=[(.005, None)] * len(r),
                            options={"xatol": 2e-11, "fatol": 2e-14, "maxiter": 10000})
    assert gls.success and gls.status == 0, gls.message
    psi = gls.x
    roots, vectors = np.linalg.eigh(r / np.sqrt(np.outer(psi, psi)))
    loading = np.sqrt(psi)[:, None] * vectors[:, -2:] * np.sqrt(roots[-2:]-1)
    for scoring in ("regression", "bartlett"):
        result = results["gls_" + scoring]
        close(result["uniqueness"].uniqueness, psi, rtol=2e-6, atol=2e-7)
        close(result["loadings"].to_numpy() @ result["loadings"].to_numpy().T,
              loading @ loading.T, rtol=2e-6, atol=3e-7)
        close(result["reproduced_correlation"], loading @ loading.T + np.diag(psi), rtol=2e-6, atol=3e-7)
        close(result["optimized_uniqueness"], np.diag(psi), rtol=2e-6, atol=2e-7)
        close(result.attrs["discrepancy"], gls.fun, rtol=0, atol=3e-13)
        assert result.attrs["local_minimum_certified"]
        assert result.attrs["gradient_max"] <= result.attrs["gradient_tolerance"]

    inverse = np.linalg.inv(r)
    d = np.diag(1 / inverse.diagonal())
    roots, vectors = linalg.eigh(r, d)
    roots, vectors = roots[::-1], vectors[:, ::-1]
    loading = d @ vectors[:, :2] @ np.diag((roots[:2]-1)/np.sqrt(roots[:2]))
    for scoring in ("regression", "bartlett"):
        result = results["generalized_image_" + scoring]
        close(result["eigenvalues"].eigenvalue, roots)
        close(result["loadings"].to_numpy() @ result["loadings"].to_numpy().T, loading @ loading.T)
        close(result["uniqueness"].uniqueness, 1-np.square(loading).sum(1))
        close(result["conditional_variance"], d)
        close(result["anti_image_covariance"], d @ inverse @ d)
        close(result["image_covariance"], r - 2*d + d @ inverse @ d)

    target = results["complete_oblique_target"]
    a, pattern = target["loadings"].to_numpy(), target["rotated_loadings"].to_numpy()
    phi = target["factor_correlations"].to_numpy()
    close(pattern, target["rotation_target"], rtol=0, atol=3e-8)
    close(pattern, a @ target["rotation_matrix"], rtol=0, atol=1e-12)
    close(phi.diagonal(), np.ones(2), rtol=0, atol=1e-10)
    close(pattern @ phi @ pattern.T, a @ a.T, rtol=2e-10, atol=2e-10)
    close(target["structure"], pattern @ phi, rtol=2e-10, atol=2e-10)
    assert target.attrs["target_complete"] and target.attrs["target_jacobian_rank"] == 2
    assert target.attrs["rotation_projected_gradient"] <= 1e-5

    query = two.iloc[[7, 2, 7, 19]].copy()
    query.iloc[1, 0] = np.nan
    for name in ("gls_regression", "gls_bartlett", "generalized_image_regression",
                 "generalized_image_bartlett", "complete_oblique_target"):
        result = results[name]
        p = result["rotated_loadings"].to_numpy() if "rotated_loadings" in result else result["loadings"].to_numpy()
        phi = result["factor_correlations"].to_numpy() if "factor_correlations" in result else np.eye(2)
        if result.attrs["scores"] == "regression":
            coefficient = np.linalg.solve(r, p @ phi)
        else:
            weighted = p / result["uniqueness"].to_numpy()
            coefficient = weighted @ np.linalg.inv(p.T @ weighted)
        close(result["score_coefficients"], coefficient)
        close(result["descriptives"]["mean"], complete.mean(0))
        close(result["descriptives"].std_dev, complete.std(0, ddof=1))
        standardized = (query.to_numpy()-complete.mean(0)) / complete.std(0, ddof=1)
        close(results[name + "_scores"]["scores"], standardized @ coefficient)
        assert results[name + "_scores"]["scores"].index.tolist() == query.index.tolist()
        assert results[name + "_scores"]["scores"].iloc[1].isna().all()

    complete_one = one.dropna().to_numpy(dtype=float)
    sample_positions = np.flatnonzero(one.notna().all(axis=1)).tolist()
    for method in ("ml", "minres", "alpha", "image_covariance", "generalized_image"):
        result = results["bootstrap_" + method]
        attrs = result.attrs
        assert attrs["method"] == method and attrs["replications"] == attrs["successful_replications"] == 199
        assert attrs["failed_replications"] == [] and attrs["sample_positions"] == sample_positions
        assert attrs["physical_rows"] == 242 and attrs["n"] == 240 and attrs["n_missing"] == 2
        assert attrs["sign_anchor"] == "x3"
        close(result["sample"], complete_one)
        draws = result["replicates"].to_numpy(dtype=float)
        indices = result["replicate_indices"].to_numpy()
        # Public numerical tables use float64, including the exactly integral
        # zero-based raw-row draws; the full typed pack preserves that dtype.
        assert indices.shape == (199, 240) and np.isfinite(indices).all()
        assert np.array_equal(indices, np.floor(indices))
        assert indices.min() >= 0 and indices.max() < 240
        assert np.all(draws[:, 2] > 0)
        covariance = np.cov(draws, rowvar=False, ddof=1)
        close(result["covariance"], covariance)
        estimate = result["estimates"]
        close(estimate.std_error, np.sqrt(covariance.diagonal()))
        close(estimate.ci_lower, np.quantile(draws, .025, axis=0, method="linear"))
        close(estimate.ci_upper, np.quantile(draws, .975, axis=0, method="linear"))
        close(estimate.bootstrap_bias, draws.mean(0)-estimate.estimate)
        assert estimate.p_value.isna().all() and estimate.df.isna().all()
        assert len(result["replicate_diagnostics"]) == 199
        diagnostics = result["replicate_diagnostics"]
        assert (diagnostics.minimum_uniqueness > 1e-8).all()
        assert (diagnostics.local_chart_margin > 1e-8).all()
        assert attrs["p_values_available"] is False and attrs["inference_df_available"] is False
    return {"array_comparisons": comparisons, "source": "host NumPy moments/covariance/percentiles and SciPy GLS/generalized eigensystem",
            "gls_optimizer_success": bool(gls.success), "gls_optimizer_status": int(gls.status),
            "planned_bootstrap_draws": 995, "all_draws_in_inference": True,
            "all_passed": True}


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
        project = owned.request("/api/desktop/local-projects", {"name": "Extraction eight frozen acceptance"}, desktop=True)["id"]
        owned.project = project
        owned.open_project()
        source = code(results)
        owned.call("/console/script", {"code": source, "name": "analysis.py"}, method="PUT")
        run = execute(owned, source)
        marker = "EXTRACTION_EIGHT_OK "
        proofs = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines() if line.startswith(marker)]
        assert len(proofs) == 1
        proof = proofs[0]
        assert proof["frozen"] and proof["full_roundtrip"] and len(proof["negative_cases"]) == 8
        assert set(proof["bootstrap_replications"].values()) == {199}
        assert digest(results / "results.json") == proof["sha256"]
        source_results = output / "source-results" / "results.json"
        for name in ("results.json", "failures.json", "inputs.json"):
            assert (results / name).read_bytes() == (source_results.parent / name).read_bytes(), "Complete source/frozen payload difference: " + name
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
        assert "EXTRACTION_EIGHT_REPLAY_OK" in replay["stdout"]
        assert digest(results / "results.json") == proof["sha256"]
    finally:
        owned.close()
    return {"status": "passed", "source_ref": source_ref, "runtime_sha256": digest(runtime),
            "compiled_modules": identities, "proof": proof, "example_sha256": digest(EXAMPLE),
            "independent_reference_checks": independent, "verifier_sha256": digest(__file__),
            "project_id": project, "profile": str(profile), "execution_id": run["id"],
            "execution_duration_ms": run["duration_ms"], "replay_execution_id": replay["id"],
            "replay_duration_ms": replay["duration_ms"], "source_frozen_complete_results_equal": True,
            "full_code_stdout_outputs_events_and_script_equal_after_restart": True,
            "optimizer_free_saved_scores_and_full_bootstrap_restore": True,
            "results_files_unchanged_after_restart": True, "owned_workers_stopped": True,
            "saved_projects_results_preserved": True, "source_path_injected": False,
            "development_estimators_present": False, "native_installed_ui_verified": False,
            "windows_verified": False, "coverage_simulations_verified": False,
            "latent_posterior_verified": False, "public_release_delivered": False}


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
