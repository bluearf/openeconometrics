"""Independent reference fixtures and scientific-domain guards for 114/115/118/121."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2
from scipy.stats import f as f_distribution
from scipy.stats._hypotests import _cdf_cvm_inf
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.panic_selection import bridge_probabilities, select_factors
from openecon.models import ResultBundle
from test_econ_panic import panel_data

REFERENCE = json.loads((Path(__file__).parent / "fixtures/inference-2026-10-07.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["po"]["cases"])
def test_phillips_ouliaris_arch_calibration(case):
    trends = {"n": "none", "c": "constant", "ct": "trend", "ctt": "quadratic"}
    actual = oe.phillips_ouliaris(
        pd.DataFrame(REFERENCE["po"]["data"]),
        "y",
        ["x0", "x1"],
        test=case["test"],
        trend=trends[case["trend"]],
        kernel=case["kernel"],
        bandwidth=5,
    )
    assert_allclose(actual.attrs["statistic"], case["statistic"], rtol=2e-11, atol=1e-11)
    assert_allclose(actual.attrs["p_value"], case["p_value"], rtol=1e-10, atol=1e-13)
    assert_allclose(
        [actual.attrs["critical_values"][key] for key in ("10%", "5%", "1%")],
        case["critical"],
        atol=2e-12,
    )
    assert actual.attrs["nobs"] == 300 and actual.attrs["n_series"] == 3
    json.dumps(actual.attrs, allow_nan=False)
    assert "tabular" in actual.to_latex()


def test_po_missing_calendar_rank_and_small_calibration_guards():
    data = pd.DataFrame(REFERENCE["po"]["data"])
    data["time"] = range(300)
    duplicate = data.assign(x1=data.x0)
    with pytest.raises(AnalysisError):
        oe.po_za(duplicate, "y", ["x0", "x1"])
    with pytest.raises(AnalysisError):
        oe.po_zt(data.drop(index=2), "y", ["x0"], time="time")
    missing = data.copy()
    missing.loc[4, "y"] = np.nan
    with pytest.raises(AnalysisError):
        oe.po_zt(missing, "y", ["x0"])
    with pytest.raises(AnalysisError):
        oe.po_zt(data.head(10), "y", ["x0", "x1"], bandwidth=1)
    actual = oe.po_zt(data.head(10), "y", ["x0", "x1"], bandwidth=1, inference="none")
    assert actual.attrs["p_value"] is None and actual.attrs["critical_values"] is None


@pytest.mark.parametrize("cohort", REFERENCE["po"].get("uncoupled", []))
def test_po_null_samples_cover_dimensions_and_probability_surfaces(cohort):
    data = pd.DataFrame(cohort["data"])
    names = [f"x{i}" for i in range(cohort["dimensions"])]
    for case in cohort["cases"]:
        actual = oe.phillips_ouliaris(
            data,
            "y",
            names,
            test=case["test"],
            trend={"n": "none", "c": "constant", "ct": "trend", "ctt": "quadratic"}[case["trend"]],
            kernel=case["kernel"],
            bandwidth=3,
        )
        assert_allclose(
            [actual.attrs["statistic"], actual.attrs["p_value"]],
            [case["statistic"], case["p_value"]],
            rtol=3e-10,
            atol=2e-11,
        )
        assert_allclose(
            [actual.attrs["critical_values"][key] for key in ("10%", "5%", "1%")],
            case["critical"],
            atol=2e-12,
        )


@pytest.mark.parametrize("reference", REFERENCE["weak"])
def test_structural_ar_clr_and_fuller_kclass(reference):
    data = pd.DataFrame(reference["data"])
    options = dict(data=data, y="y", endog=["d"], instruments=["z0", "z1", "z2"])
    for method in ("ar", "clr"):
        actual = oe.iv_weak_test(**options, null=1.4, method=method)
        assert_allclose(
            [actual.attrs["statistic"], actual.attrs["p_value"]],
            reference[method],
            rtol=1e-9,
            atol=5e-11,
        )
        json.dumps(actual.attrs, allow_nan=False)
    for method, oracle in reference["models"].items():
        kwargs = {
            "method": "fuller" if method == "fuller(1)" else "kclass" if method == "0.5" else method
        }
        if method == "0.5":
            kwargs["kappa"] = 0.5
        actual = oe.ivregress(**options, **kwargs)
        assert_allclose(
            [row.estimate for row in actual.coefficients],
            oracle["coefficients"],
            rtol=1e-9,
            atol=1e-10,
        )
        if method not in ("2sls",):
            assert_allclose(actual.metrics["kappa"], oracle["kappa"], atol=2e-12)
        restored = ResultBundle.model_validate_json(actual.model_dump_json())
        assert restored.covariance_matrix == actual.covariance_matrix
    confidence = oe.iv_ar_confidence_set(**{**options, "endog": "d"})
    assert_allclose(confidence.attrs["critical_value"], reference["ar_critical"], rtol=1e-11)
    y = data.y.to_numpy() - data.y.mean()
    d = data.d.to_numpy() - data.d.mean()
    z = data[["z0", "z1", "z2"]].to_numpy()
    z -= z.mean(0)
    q = np.linalg.qr(z)[0]
    intervals = confidence.attrs["intervals"]
    for beta in np.linspace(-30, 30, 601):
        residual = y - beta * d
        projected = q @ (q.T @ residual)
        statistic = (
            296 / 3 * (projected @ projected) / ((residual - projected) @ (residual - projected))
        )
        accepted = any(
            (lo is None or beta >= lo) and (hi is None or beta <= hi) for lo, hi in intervals
        )
        assert accepted == (statistic <= reference["ar_critical"])


@pytest.mark.parametrize("covariance", ["robust", "cluster", "hac"])
def test_structural_ar_full_covariance_oracle(covariance):
    data = pd.DataFrame(REFERENCE["weak"][0]["data"])
    data["cluster"] = np.arange(300) % 30
    data["time"] = np.arange(300)
    options = (
        {"cluster": "cluster"}
        if covariance == "cluster"
        else {"time": "time", "lags": 3}
        if covariance == "hac"
        else {}
    )
    actual = oe.iv_weak_test(
        data=data,
        y="y",
        endog=["d"],
        instruments=["z0", "z1", "z2"],
        null=1.4,
        covariance=covariance,
        **options,
    )
    x = np.column_stack([np.ones(300), data[["z0", "z1", "z2"]]])
    y = data.y.to_numpy() - 1.4 * data.d.to_numpy()
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    score = x * (y - x @ beta)[:, None]
    meat = score.T @ score
    if covariance == "cluster":
        sums = np.stack([score[data.cluster == g].sum(0) for g in range(30)])
        meat = sums.T @ sums * 30 / 29
    if covariance == "hac":
        for lag in range(1, 4):
            cross = score[lag:].T @ score[:-lag]
            meat += (1 - lag / 4) * (cross + cross.T)
    expected = inv @ meat @ inv
    assert_allclose(actual.attrs["null_reduced_form_covariance"], expected, rtol=3e-11, atol=1e-12)
    statistic = beta[1:] @ np.linalg.solve(expected[1:, 1:], beta[1:])
    assert_allclose(
        [actual.attrs["statistic"], actual.attrs["p_value"]],
        [statistic, chi2.sf(statistic, 3)],
        rtol=2e-11,
    )


def test_confidence_topology_and_stock_yogo_domain():
    data = pd.DataFrame(REFERENCE["weak"][0]["data"])
    arguments = dict(data=data, y="y", endog=["d"], instruments=["z0", "z1", "z2"])
    reference = oe.stock_yogo(oe.ivregress(**arguments))
    assert reference.attrs["critical_value"] == 22.3
    with pytest.raises(AnalysisError):
        oe.stock_yogo(oe.ivregress(**arguments, covariance="robust"))
    with pytest.raises(AnalysisError):
        oe.iv_weak_test(**arguments, null=1.4, method="clr", covariance="robust")
    with pytest.raises(AnalysisError):
        oe.iv_weak_test(**arguments, null=[1.4, 2.0])
    with pytest.raises(AnalysisError):
        oe.ivregress(**arguments, method="kclass")


@pytest.mark.parametrize("shape", ["empty", "disjoint"])
def test_actual_ar_empty_and_disjoint_sets_against_independent_f_grid(shape):
    draws = np.random.default_rng(649).normal(size=(80, 4))
    orthogonal = np.linalg.qr(draws - draws.mean(0))[0]
    z1, z2, v, e = orthogonal.T
    d = z1 + 0.1 * v if shape == "empty" else 0.05 * z1 + v
    y = z2 + 0.1 * e
    data = pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})
    result = oe.iv_ar_confidence_set(data=data, y="y", endog="d", instruments=["z1", "z2"])
    intervals = result.attrs["intervals"]
    assert result.attrs[shape]
    json.dumps(result.attrs, allow_nan=False)
    for beta in np.linspace(-20, 20, 801):
        residual = y - beta * d
        fitted = orthogonal[:, :2] @ (orthogonal[:, :2].T @ residual)
        statistic = (fitted @ fitted / 2) / ((residual - fitted) @ (residual - fitted) / 77)
        accepted = any(
            (lo is None or lo <= beta) and (hi is None or beta <= hi) for lo, hi in intervals
        )
        assert accepted == (f_distribution.sf(statistic, 2, 77) >= 0.05)


@pytest.mark.parametrize("method", ["icp1", "icp2", "icp3"])
def test_panic_selection_matches_independent_ic(method):
    data = panel_data(units=20, periods=100, factors=2)
    values = data.pivot(index="t", columns="id", values="y").to_numpy().T
    chosen, meta = select_factors(torch.tensor(values), "trend", method, 5)
    differences = np.diff(values, axis=1).T
    differences /= np.abs(differences).max()
    differences -= differences.mean(0)
    t, n = differences.shape
    spectrum = np.linalg.svd(differences, compute_uv=False)
    penalty = {
        "icp1": (n + t) / (n * t) * np.log(n * t / (n + t)),
        "icp2": (n + t) / (n * t) * np.log(min(n, t)),
        "icp3": np.log(min(n, t)) / min(n, t),
    }[method]
    criteria = [np.log((spectrum[k:] ** 2).sum() / (n * t)) + k * penalty for k in range(6)]
    assert chosen == np.argmin(criteria)
    assert_allclose([row["criterion"] for row in meta["candidates"]], criteria, atol=2e-13)


def test_bridge_limit_matches_analytic_cvm_distribution_and_preserves_rng():
    before = torch.random.get_rng_state().clone()
    statistics = torch.tensor([-3.0, -2.5, -2.0, -1.5, -1.0], dtype=torch.float64)
    probability, _, critical, metadata = bridge_probabilities(statistics)
    expected = _cdf_cvm_inf(0.25 / statistics.numpy() ** 2)
    assert_allclose(probability, expected, atol=0.004)
    for level, value in critical.items():
        assert_allclose(_cdf_cvm_inf(0.25 / value**2), float(level[:-1]) / 100, atol=0.004)
    assert torch.equal(before, torch.random.get_rng_state())
    assert metadata["finite_sample_calibration"] is False


def test_panic_auto_lags_zero_factor_and_bridge_pooling():
    data = panel_data(units=20, periods=100, factors=2)
    actual = oe.xtpanic(
        data,
        "y",
        "id",
        "t",
        factors="icp2",
        max_factors=5,
        lags="bic",
        maxlag=3,
        trend="trend",
        inference="bridge",
        pooling="independent",
        components=True,
    )
    assert actual.attrs["factor_selection"]["chosen"] == actual.attrs["factor_count"]
    assert all(0 <= k <= 3 for k in actual.attrs["lag_selection"]["chosen_idiosyncratic"])
    assert "pooled" in actual and actual["idiosyncratic"]["p_value"].notna().all()
    json.dumps(actual.attrs, allow_nan=False)
    rng = np.random.default_rng(914)
    values = rng.normal(size=(100, 70)).cumsum(1)
    independent = pd.DataFrame(
        {
            "id": np.repeat(np.arange(100), 70),
            "time": np.tile(np.arange(70), 100),
            "y": values.ravel(),
        }
    )
    zero = oe.xtpanic(
        independent, "y", "id", "time", factors="icp2", max_factors=3, inference="none"
    )
    assert zero.attrs["factor_count"] == 0 and len(zero["common"]) == 0
    with pytest.raises(AnalysisError):
        oe.xtpanic(data, "y", "id", "t", trend="trend", inference="bridge", memory_mb=1)


@pytest.mark.parametrize("case", REFERENCE["rd"]["cases"])
def test_rd_adjusted_weighted_kink_matches_official_package(case):
    data = pd.DataFrame(REFERENCE["rd"]["data"])
    options = dict(
        data=data,
        y="y",
        running="x",
        covariates=["z0", "z1"],
        weights="w",
        deriv=case["deriv"],
        p=case["deriv"] + 1,
    )
    if case["fuzzy"]:
        options["fuzzy"] = "t"
    if not case["auto"]:
        options.update(h=[0.4, 0.5], b=[0.6, 0.7])
    actual = oe.rdrobust(**options)
    tolerance = 4e-6 if case["auto"] else 2e-10
    assert_allclose(
        [c.estimate for c in actual.coefficients], case["coef"], rtol=tolerance, atol=tolerance
    )
    assert_allclose(
        [c.std_error for c in actual.coefficients], case["se"], rtol=tolerance, atol=tolerance
    )
    assert_allclose([actual.metrics["h_left"], actual.metrics["h_right"]], case["h"], rtol=1e-6)
    assert_allclose([actual.metrics["b_left"], actual.metrics["b_right"]], case["b"], rtol=1e-6)
    assert [actual.metrics["n_h_left"], actual.metrics["n_h_right"]] == case["N_h"]
    ResultBundle.model_validate_json(actual.model_dump_json())
    assert "tabular" in actual.to_latex()


@pytest.mark.parametrize("case", REFERENCE["rd"]["mass"])
def test_rd_masspoints_official_bandwidth_sample(case):
    data = pd.DataFrame(REFERENCE["rd"]["data"])
    data.x = data.x.round(2)
    actual = oe.rdrobust(data=data, y="y", running="x", weights="w", masspoints=case["masspoints"])
    assert_allclose([c.estimate for c in actual.coefficients], case["coef"], rtol=2e-7)
    assert_allclose([c.std_error for c in actual.coefficients], case["se"], rtol=2e-7)
    assert_allclose(
        [actual.metrics["h_left"], actual.metrics["b_left"]],
        [case["h"][0], case["b"][0]],
        rtol=2e-7,
    )
    assert [actual.metrics["n_h_left"], actual.metrics["n_h_right"]] == case["N_h"]


@pytest.mark.parametrize("masspoints", [False, True])
def test_density_manipulation_official_reference(masspoints):
    data = pd.DataFrame(REFERENCE["rd"]["data"])
    if masspoints:
        data.x = data.x.round(2)
    ref = REFERENCE["rd"]["density_mass" if masspoints else "density"]
    actual = oe.rddensity(data=data, running="x", h=[0.4, 0.5])
    row = actual["density"].iloc[1]
    assert_allclose(
        [row.left, row.right, row.difference],
        [ref["hat"]["left"], ref["hat"]["right"], ref["hat"]["diff"]],
        atol=3e-11,
    )
    assert_allclose(
        [row.std_error, row.statistic, row.p_value],
        [ref["se"]["diff"], ref["stat"]["t_jk"], ref["stat"]["p_jk"]],
        atol=3e-11,
    )
    with pytest.raises(AnalysisError):
        oe.rddensity(data=data, running="x", h=0.005)
    bad = data.assign(z1=data.z0)
    with pytest.raises(AnalysisError):
        oe.rdrobust(data=bad, y="y", running="x", covariates=["z0", "z1"], h=0.4)


@pytest.mark.parametrize("adjustment", ["unadjusted", "fixed", "automatic"])
def test_published_stata_senate_replication(adjustment):
    """Calonico et al. (2017), Stata Journal pp. 392, 395--396, printed precision."""
    import hashlib

    path = Path(__file__).parent / "fixtures/rdrobust-senate.csv"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "def1fc6b8ae5af273f2140aa449c81b3ecfe1d0f325aa80ceccb0d9638a2ad44"
    )
    data = pd.read_csv(path)
    covariates = None if adjustment == "unadjusted" else ["class", "termshouse", "termssenate"]
    options = {"h": 17.708, "b": 27.984} if adjustment == "fixed" else {}
    result = oe.rdrobust(
        data=data,
        y="vote",
        running="margin",
        covariates=covariates,
        missing="drop",
        masspoints="off",
        **options,
    )
    h, b, nleft, nright, estimate, se, z, low, high = {
        "unadjusted": (17.708, 27.984, 359, 322, 7.416, 1.4604, 4.3095, 4.09441, 10.9255),
        "fixed": (17.708, 27.984, 309, 280, 6.8595, 1.4165, 4.1911, 3.75238, 10.345),
        "automatic": (17.987, 28.943, 313, 283, 6.8514, 1.4081, 4.1999, 3.72856, 10.2537),
    }[adjustment]
    assert_allclose(
        [result.metrics["h_left"], result.metrics["b_left"]], [h, b], atol=0.0005, rtol=0
    )
    assert [result.metrics["n_h_left"], result.metrics["n_h_right"]] == [nleft, nright]
    conventional, robust = result.coefficients[0], result.coefficients[2]
    assert_allclose(
        [conventional.estimate, conventional.std_error, robust.statistic],
        [estimate, se, z],
        atol=0.000051,
        rtol=0,
    )
    assert_allclose([robust.ci_low, robust.ci_high], [low, high], atol=0.000051, rtol=0)
    assert round(robust.p_value, 3) == 0
