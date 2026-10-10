"""Independent original-unit likelihood, full OIM/transforms and curve covariance."""

import importlib.util
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import weibull_regression as api
from openecon.econometrics.survival_ext import weibull_regression_kernels as kernels

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load("examples/interval_weibull_regression.py", "interval_weibull_example")
oracle = load("scripts/verify_interval_weibull_regression_oracles.py", "interval_weibull_oracle")


@pytest.fixture(scope="module")
def fit():
    return api.stinterval_weibull_regression(
        example.case(), lower="lower", upper="upper", x=["x1", "x2"]
    )


@pytest.mark.parametrize("seed,exact", [(121, 0.2), (311, 1.0), (91, 0.0)])
def test_independent_full_original_oim(seed, exact):
    model = api.stinterval_weibull_regression(
        example.case(seed, exact_fraction=exact), lower="lower", upper="upper", x=["x1", "x2"]
    )
    query = api.interval_weibull_regression_predict(
        model, pd.DataFrame({"x1": [-0.5, 0.75], "x2": [0.25, -0.4]}), times=[0.0, 0.5, 1.5, 3.0]
    )
    receipt = oracle.audit(
        {"models": [model.model_dump()["payload"]], "predictions": [query.model_dump()["payload"]]},
        run_r=False,
    )
    assert receipt["passed"], [row for row in receipt["checks"] if not row["passed"]]
    assert receipt["actual_R_executed"] is False
    assert receipt["references"] == []


@pytest.mark.parametrize("case", range(4))
def test_cached_full_original_author_reference(case):
    provenance = json.loads(
        (ROOT / "tests/fixtures/interval_weibull_regression_author_provenance.json").read_text()
    )
    fixture = provenance["actual_reference_fixtures"][case]
    reference = fixture["reference_data"]
    frame = pd.DataFrame(reference["values"], columns=reference["columns"])
    frame["upper"] = pd.Series([row[1] for row in reference["values"]], dtype=object)
    model = api.stinterval_weibull_regression(frame, lower="lower", upper="upper", x=["x1", "x2"])
    receipt = oracle.cached_author_check(model.model_dump()["payload"], fixture)
    assert receipt["passed"], receipt["errors"]
    assert receipt["new_actual_R_run"] is False
    assert receipt["cached_actual_reference"] is True
    assert receipt["version"].startswith("3.8.6 |")
    altered = model.model_dump()["payload"]
    altered["source"]["values"][0][2] += 0.01
    with pytest.raises(ValueError, match="complete exact numeric"):
        oracle.cached_author_check(altered, fixture)


@pytest.mark.parametrize("seed,exact", [(121, 0.2), (311, 1.0), (91, 0.0)])
def test_fresh_actual_original_r_reference(seed, exact):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Live original R reference unavailable; full cached author gate is separate")
    version = subprocess.run(
        [rscript, "-e", "cat(as.character(packageVersion('survival')))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if version.returncode or version.stdout != "3.8.6":
        pytest.skip("Pinned live survival3.8-6 unavailable; cached original reference is separate")
    model = api.stinterval_weibull_regression(
        example.case(seed, exact_fraction=exact), lower="lower", upper="upper", x=["x1", "x2"]
    )
    receipt = oracle.audit({"models": [model.model_dump()["payload"]]})
    assert receipt["passed"], [row for row in receipt["checks"] if not row["passed"]]
    assert receipt["actual_R_executed"] is True
    assert receipt["references"][0]["version"].startswith("3.8.6 |")


def test_covariate_and_time_units_full_transformation():
    base = api.stinterval_weibull_regression(
        example.case(192), lower="lower", upper="upper", x=["x1", "x2"]
    )
    units = (0.001, 1000.0)
    scaled = api.stinterval_weibull_regression(
        example.case(192, time_units=0.001, x_units=units),
        lower="lower",
        upper="upper",
        x=["x1", "x2"],
    )
    j = np.diag([1.0, 1 / units[0], 1 / units[1], 1.0])
    raw = np.array(base.payload["result"]["aft_parameters"])
    expected = j @ raw
    expected[0] += np.log(0.001)
    np.testing.assert_allclose(
        scaled.payload["result"]["aft_parameters"], expected, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        scaled.payload["result"]["aft_covariance"],
        j @ np.array(base.payload["result"]["aft_covariance"]) @ j,
        rtol=1e-8,
        atol=1e-12,
    )
    receipt = oracle.audit({"models": [scaled.model_dump()["payload"]]}, run_r=False)
    assert receipt["passed"], [row for row in receipt["checks"] if not row["passed"]]


def test_full_shape_coupling_differs_from_diagonal_shortcut(fit):
    covariance = np.array(fit.payload["result"]["aft_covariance"])
    raw = np.array(fit.payload["result"]["aft_parameters"])
    _, full, jac = oracle.ph(raw, covariance)
    wrong = jac @ np.diag(np.diag(covariance)) @ jac.T
    assert np.max(np.abs(full - wrong)) > 1e-4 * np.max(np.diag(full))
    np.testing.assert_allclose(full, fit.payload["result"]["ph_covariance"], rtol=1e-10, atol=1e-12)


def test_narrow_interval_keeps_width_and_actual_density_jacobian():
    spec = api.specification(
        lower="lower",
        upper="upper",
        x=["x1"],
        parameterization="aft",
        level=0.95,
        maxiter=100,
        max_work=500000000,
    )
    frame = pd.DataFrame(
        {
            "lower": [0.4, 1.0, 1.2, 2.0],
            "upper": pd.Series([0.4, 1.0 + 1e-10, 1.6, None], dtype=object),
            "x1": [-0.4, 0.1, 0.5, 1.0],
        }
    )
    source = api.c.source(frame, ["lower", "upper", "x1"], upper="upper")
    native = kernels.prepare(source, spec)
    point = kernels.tensor([0.1, 0.2, -0.3])
    logparts = kernels.contributions(point, native).numpy()
    raw = kernels.original(point, native).numpy()
    sigma = np.exp(raw[-1])
    eta = raw[0] + raw[1] * 0.1
    z = (np.log(1.0) - eta) / sigma
    expected = np.log(1e-10) + (-raw[-1] + z - np.exp(z))
    assert abs(logparts[1] - expected) < 2e-6  # binary endpoint width differs from decimal1e-10


def test_intercept_reduces_existing_weibull_core():
    from openecon.econometrics.survival_ext.interval import stinterval_weibull

    data = example.case(198, n=90)
    new = api.stinterval_weibull_regression(data, lower="lower", upper="upper", x=[])
    old = stinterval_weibull(data.lower.tolist(), data.upper.tolist())
    state = old.attrs["prediction_state"]
    assert abs(new.payload["endpoint"]["log_likelihood"] - old.attrs["log_likelihood"]) < 1e-7
    np.testing.assert_allclose(
        new.payload["result"]["aft_parameters"],
        [state["raw_parameters"][1], -state["raw_parameters"][0]],
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize("kind", ["left", "right", "constant_covariate", "collinear"])
def test_unidentified_domains_refuse(kind):
    frame = example.case(4, n=30)
    if kind == "left":
        frame["lower"] = 0.0
        frame["upper"] = 1.0
    if kind == "right":
        frame["lower"] = 1.0
        frame["upper"] = None
    if kind == "constant_covariate":
        frame["x1"] = 1.0
    if kind == "collinear":
        frame["x2"] = frame.x1 * 2
    with pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(frame, lower="lower", upper="upper", x=["x1", "x2"])


def test_calibration_late_failure_retains_denominator_without_partial_hits(
    fit, monkeypatch, tmp_path
):
    calibration = load(
        "scripts/audit_interval_weibull_regression_calibration.py", "weibull_calibration_failure"
    )
    plan_path = ROOT / "docs/econometrics/interval-weibull-covariate-calibration-plan.json"
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    original_read, original_write = Path.read_text, Path.write_text

    def read(path, *args, **kwargs):
        if str(path) == "/tmp/market-689-preregistration.json":
            return json.dumps({"sha256": plan_hash, "test_injected_failure_audit": True})
        return original_read(path, *args, **kwargs)

    def write(path, data, *args, **kwargs):
        if str(path) == "/tmp/market-689-pre-calibration-source-pin.json":
            path = tmp_path / "injected-failure-pin.json"
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(calibration, "stinterval_weibull_regression", lambda *a, **kw: fit)
    monkeypatch.setattr(
        calibration, "interval_weibull_regression_predict", lambda *a, **kw: SimpleNamespace()
    )

    def fail_after_marginal_hits(*args, **kwargs):
        raise ValueError("Injected late full-covariance audit failure")

    monkeypatch.setattr(calibration.np.linalg, "solve", fail_after_marginal_hits)
    output = tmp_path / "injected-failure-receipt.json"
    assert calibration.main(output) is False
    receipt = json.loads(output.read_text())
    assert receipt["full_failure_denominator"] == receipt["failures"] == 128
    assert all(not row["accepted"] for row in receipt["results"])
    for field in ("aft_coverage", "ph_coverage", "wrong_variance_coverage"):
        assert receipt[field] == [0.0] * 4
    assert receipt["joint_coverage"] == 0.0
