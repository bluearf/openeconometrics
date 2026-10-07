"""cnsreg against the explicit Lagrangian (constrained least squares) solution."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy.linalg import null_space

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ModelSpec, ResultBundle


@pytest.fixture
def data():
    rng = np.random.default_rng(99)
    n = 200
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n),
        "g": np.repeat(np.arange(20), 10), "w": rng.integers(1, 4, size=n).astype(float),
        "aw": rng.uniform(0.5, 2, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
    })
    frame["y"] = 0.5 + 0.7 * frame.x1 + 0.3 * frame.x2 - frame.x3 + rng.normal(size=n) * (1 + frame.x2**2)
    return frame


def lagrangian(x, y, r_matrix, r_value, weights=None):
    """Constrained WLS: b = b_ols - (X'WX)^-1 R' [R (X'WX)^-1 R']^-1 (R b_ols - r)."""
    w = np.ones(len(y)) if weights is None else weights
    xtx_inv = np.linalg.inv(x.T @ (x * w[:, None]))
    b_ols = xtx_inv @ (x.T @ (w * y))
    gain = xtx_inv @ r_matrix.T @ np.linalg.solve(r_matrix @ xtx_inv @ r_matrix.T, np.eye(len(r_value)))
    b = b_ols - gain @ (r_matrix @ b_ols - r_value)
    return b, y - x @ b, xtx_inv - gain @ r_matrix @ xtx_inv


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def test_cnsreg_matches_the_lagrangian_solution(data):
    constraints = [{"terms": {"x1": 1, "x2": 1}, "value": 1}, {"terms": {"x3": 1, "Intercept": -2}, "value": 0}]
    result = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints)
    x = sm.add_constant(data[["x1", "x2", "x3"]]).to_numpy()
    r_matrix = np.array([[0, 1, 1, 0], [-2, 0, 0, 1.0]])
    r_value = np.array([1.0, 0.0])
    n, k = x.shape
    b, e, shape = lagrangian(x, data.y.to_numpy(), r_matrix, r_value)
    df_resid = n - k + 2
    sigma2 = e @ e / df_resid
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", "x3"]
    assert_allclose(estimates(result), b, rtol=1e-12)
    assert_allclose(r_matrix @ estimates(result), r_value, atol=1e-12)
    assert_allclose(np.asarray(result.covariance_matrix), sigma2 * shape, rtol=1e-10, atol=1e-14)
    assert result.metrics["df_resid"] == df_resid and result.metrics["df_constraints"] == 2
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(sigma2), rel=1e-12)
    assert "r_squared" not in result.metrics and result.inference["df_inference"] == df_resid
    assert result.extra["constrained_terms"] == {} and result.extra["constraint_matrix"]["r"] == [1.0, 0.0]
    # The reported covariance has rank K - q and the model test uses that rank.
    assert np.linalg.matrix_rank(np.asarray(result.covariance_matrix), tol=1e-10) == 2
    test = result.tests["model"]
    assert test["df"] == 2 and test["df2"] == df_resid and test["distribution"] == "F"
    free = sigma2 * shape[1:, 1:]
    values, vectors = np.linalg.eigh(free / np.sqrt(np.outer(np.diag(free), np.diag(free))))
    keep = values > 1e-12 * values[-1]
    rotated = vectors[:, keep].T @ (b[1:] / np.sqrt(np.diag(free)))
    assert test["statistic"] == pytest.approx((rotated**2 / values[keep]).sum() / keep.sum(), rel=1e-9)
    assert_allclose([p["fitted"] for p in result.predictions[:3]], (x @ b)[:3])
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    assert "df_constraints: 2" in result.summary() and "Observations" in str(result.to_latex())


@pytest.mark.parametrize("covariance", ["HC1", "robust", "cluster"])
def test_cnsreg_robust_covariances_use_the_reduced_model(data, covariance):
    constraints = [{"terms": {"x1": 1, "x2": -1}, "value": 0}]
    cluster = "g" if covariance == "cluster" else None
    result = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints,
                       covariance=None if cluster else covariance, cluster=cluster)
    x = sm.add_constant(data[["x1", "x2", "x3"]]).to_numpy()
    basis = null_space(np.array([[0, 1, -1, 0.0]]))             # any orthonormal basis gives the same V
    reduced = x @ basis
    fit = sm.OLS(data.y, reduced).fit(cov_type="HC1" if not cluster else "cluster",
                                      cov_kwds={} if not cluster else {"groups": data.g})
    expected = basis @ fit.cov_params() @ basis.T
    assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-10, atol=1e-14)
    assert_allclose(estimates(result), basis @ fit.params, rtol=1e-12)
    n, k = x.shape
    if cluster:
        assert result.inference["small_sample_correction"] == pytest.approx(20 / 19 * (n - 1) / (n - 3))
        assert result.inference["df_inference"] == 19
    else:
        assert result.inference["small_sample_correction"] == pytest.approx(n / (n - 3))


def test_cnsreg_weights_follow_regress_conventions(data):
    constraints = [{"terms": {"x1": 1, "x2": 1}, "value": 1}]
    x = sm.add_constant(data[["x1", "x2", "x3"]]).to_numpy()
    r_matrix, r_value = np.array([[0, 1, 1, 0.0]]), np.array([1.0])
    analytic = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints, weights="aw",
                         weight_type="aweight")
    w = data.aw.to_numpy() * len(data) / data.aw.sum()
    b, e, shape = lagrangian(x, data.y.to_numpy(), r_matrix, r_value, w)
    assert_allclose(estimates(analytic), b, rtol=1e-12)
    assert_allclose(np.asarray(analytic.covariance_matrix), (w * e @ e / (len(data) - 3)) * shape, rtol=1e-10)
    duplicated = data.loc[data.index.repeat(data.w.astype(int))].reset_index(drop=True)
    weighted = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints, weights="w",
                         weight_type="fweight", covariance="HC1")
    plain = oe.cnsreg(data=duplicated, y="y", x=["x1", "x2", "x3"], constraints=constraints, covariance="HC1")
    assert weighted.nobs == plain.nobs == int(data.w.sum())
    assert_allclose(estimates(weighted), estimates(plain), rtol=1e-12)
    assert_allclose(errors(weighted), errors(plain), rtol=1e-11)
    assert weighted.metrics["df_resid"] == plain.metrics["df_resid"]
    sampling = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints, weights="aw",
                         weight_type="pweight")
    assert sampling.spec.covariance == "HC1"
    with pytest.raises(AnalysisError) as caught:
        oe.cnsreg(data=data, y="y", x=["x1"], constraints=[{"terms": {"x1": 1}, "value": 0}], weights="aw",
                  weight_type="pweight", covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"


def test_cnsreg_reports_determined_terms_separately_and_handles_categoricals(data):
    result = oe.cnsreg(data=data, y="y", x=["x1", "x2", "sector"], categorical=["sector"],
                       constraints=[{"terms": {"x1": 1}, "value": 0.7},
                                    {"terms": {"sector[b]": 1, "sector[c]": -1}, "value": 0}])
    assert [c.term for c in result.coefficients] == ["Intercept", "x2", "sector[b]", "sector[c]"]
    assert result.extra["constrained_terms"] == {"x1": 0.7}
    assert result.coefficients[2].estimate == pytest.approx(result.coefficients[3].estimate)
    dummies = pd.get_dummies(data.sector, drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(data)), data.x1, data.x2, dummies.to_numpy()])
    b, e, shape = lagrangian(x, data.y.to_numpy(), np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, -1.0]]),
                             np.array([0.7, 0.0]))
    assert_allclose(estimates(result), b[[0, 2, 3, 4]], rtol=1e-12)
    sigma2 = e @ e / (len(data) - 5 + 2)
    assert_allclose(errors(result), np.sqrt(sigma2 * np.diag(shape))[[0, 2, 3, 4]], rtol=1e-10)
    assert result.metrics["df_resid"] == len(data) - 3
    # The residual of a determined coefficient is exactly reproduced in the chart sample.
    assert_allclose([p["fitted"] for p in result.predictions[:3]], (x @ b)[:3])


def test_cnsreg_constraint_screening_and_error_codes(data):
    base = {"data": data, "y": "y", "x": ["x1", "x2", "x3"]}
    redundant = oe.cnsreg(**base, constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1},
                                               {"terms": {"x1": 2, "x2": 2}, "value": 2},
                                               {"terms": {"x3": 1}, "value": -1}])
    assert redundant.metrics["df_constraints"] == 2 and redundant.extra["constrained_terms"] == {"x3": -1.0}
    assert any("Dropped redundant constraint(s) 2" in warning for warning in redundant.warnings)
    for constraints, code in [
        ([{"terms": {"x1": 1, "x2": 1}, "value": 1}, {"terms": {"x1": 2, "x2": 2}, "value": 3}], "inconsistent_constraints"),
        ([{"terms": {"nope": 1}, "value": 0}], "invalid_constraint"),
        ([{"terms": {"x1": "one"}, "value": 0}], "invalid_constraint"),
        ([{"terms": {"x1": 1}}], "invalid_constraint"),
        ([], "invalid_constraint"),
        ("x1 = 1", "invalid_constraint"),
        ([{"terms": {}, "value": 1}], "inconsistent_constraints"),
        ([{"terms": {}, "value": 0}], "invalid_constraint"),
        ([{"terms": {"Intercept": 1}, "value": 0}, {"terms": {"x1": 1}, "value": 0},
          {"terms": {"x2": 1}, "value": 0}, {"terms": {"x3": 1}, "value": 0}], "invalid_constraint"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.cnsreg(**base, constraints=constraints)
        assert caught.value.code == code, constraints
    collinear = data.assign(twice=2 * data.x1)
    with pytest.raises(AnalysisError) as caught:
        oe.cnsreg(data=collinear, y="y", x=["x1", "twice"], constraints=[{"terms": {"twice": 1}, "value": 0}])
    assert caught.value.code == "invalid_constraint" and "collinearity" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.cnsreg(**{**base, "data": data.assign(x1=np.where(np.arange(200) == 0, np.nan, data.x1))},
                  constraints=[{"terms": {"x1": 1}, "value": 0}])
    assert caught.value.code == "missing_values"
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="cnsreg", outcome="y", predictors=["x1"])
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="cnsreg", outcome="y", predictors=["x1"], options={"constraints": []}, covariance="HC3")
    with pytest.raises((ValidationError, AnalysisError)):        # a cluster column needs the cluster covariance
        oe.cnsreg(**base, constraints=[{"terms": {"x1": 1}, "value": 0}], cluster="g", covariance="HC1")
    with pytest.raises(AnalysisError) as caught:
        oe.cnsreg(data=data, y="y", x="x1", constraints=[{"terms": {"x1": 1}, "value": 0}])
    assert caught.value.code == "invalid_spec"


def test_cnsreg_without_intercept_and_round_trip(data):
    result = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], intercept=False,
                       constraints=[{"terms": {"x1": 1, "x2": 1, "x3": 1}, "value": 0}])
    x = data[["x1", "x2", "x3"]].to_numpy()
    b, e, shape = lagrangian(x, data.y.to_numpy(), np.array([[1, 1, 1.0]]), np.array([0.0]))
    assert_allclose(estimates(result), b, rtol=1e-12)
    assert_allclose(errors(result), np.sqrt(e @ e / (200 - 2) * np.diag(shape)), rtol=1e-10)
    assert result.tests["model"]["df"] == 2
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------- regression tests (repairs)


def test_cnsreg_fixed_terms_are_identified_structurally_not_by_scale(data):
    """Regression: a regressor scaled by 1e8 (tiny coefficient variance) was moved out of the
    coefficient table as if the constraints fixed it. The criterion is now the row space of R."""
    constraints = [{"terms": {"x2": 1}, "value": 0.5}]
    base = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], constraints=constraints)
    for scale in (1e8, 1e-8):
        frame = data.assign(big=scale * data.x1)
        result = oe.cnsreg(data=frame, y="y", x=["big", "x2", "x3"], constraints=constraints)
        assert [c.term for c in result.coefficients] == ["Intercept", "big", "x3"]
        assert result.extra["constrained_terms"] == {"x2": 0.5}
        reference = sm.OLS(frame.y - 0.5 * frame.x2, sm.add_constant(frame[["big", "x3"]])).fit()
        assert_allclose(estimates(result), reference.params.values, rtol=1e-7)
        assert_allclose(errors(result), reference.bse.values, rtol=1e-7)
        # exact rescaling: the coefficient and its standard error scale by 1/scale
        assert_allclose(estimates(result)[1] * scale, estimates(base)[1], rtol=1e-9)
        assert_allclose(errors(result)[1] * scale, errors(base)[1], rtol=1e-9)
        assert result.metrics == pytest.approx(base.metrics, rel=1e-9)
        assert any(warning.startswith("Fixed by the constraints and not estimated: x2 = 0.5")
                   for warning in result.warnings)
    # Fixed through a combination: x1 + x2 = 1 and x1 - x2 = 0 pin both at 0.5 although no
    # single constraint names one coefficient alone.
    both = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"],
                     constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1},
                                  {"terms": {"x1": 1, "x2": -1}, "value": 0}])
    assert [c.term for c in both.coefficients] == ["Intercept", "x3"]
    assert both.extra["constrained_terms"] == pytest.approx({"x1": 0.5, "x2": 0.5}, abs=1e-14)
    reference = sm.OLS(data.y - 0.5 * data.x1 - 0.5 * data.x2, sm.add_constant(data[["x3"]])).fit()
    assert_allclose(estimates(both), reference.params.values, rtol=1e-11)
    assert_allclose(errors(both), reference.bse.values, rtol=1e-11)
    # A constraint that only ties coefficients together fixes none of them: no warning.
    tied = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"],
                     constraints=[{"terms": {"x1": 1, "x2": 1e-9}, "value": 0}])
    assert tied.extra["constrained_terms"] == {} and len(tied.coefficients) == 4
    assert not any("Fixed by the constraints" in warning for warning in tied.warnings)


def test_cnsreg_rejects_non_finite_or_overflowing_constraint_numbers(data):
    """Regression: inf/NaN/huge constraint numbers surfaced as unrelated kernel errors."""
    base = {"data": data, "y": "y", "x": ["x1", "x2", "x3"]}
    for constraints in [
        [{"terms": {"x1": 1}, "value": float("inf")}],
        [{"terms": {"x1": 1}, "value": float("-inf")}],
        [{"terms": {"x1": 1}, "value": float("nan")}],
        [{"terms": {"x1": float("inf")}, "value": 0}],
        [{"terms": {"x1": float("nan")}, "value": 0}],
        [{"terms": {"x1": 1}, "value": True}],
        [{"terms": {"x1": 1}, "value": 1e300}],
        [{"terms": {"x1": 1e-300}, "value": 1e200}],
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.cnsreg(**base, constraints=constraints)
        assert caught.value.code == "invalid_constraint", constraints
        assert "finite" in str(caught.value) or "too large" in str(caught.value)


def test_cnsreg_needs_more_observations_than_design_columns(data):
    """Regression: n <= K was reported as a collinearity omission with a 'valid' fit."""
    constraints = [{"terms": {"x1": 1}, "value": 2}]
    for rows in (2, 3):
        with pytest.raises(AnalysisError) as caught:
            oe.cnsreg(data=data.head(rows), y="y", x=["x1", "x2"], constraints=constraints)
        assert caught.value.code == "insufficient_observations"
        assert "3 design columns" in str(caught.value)
    fitted = oe.cnsreg(data=data.head(5), y="y", x=["x1", "x2"], constraints=constraints)
    assert fitted.nobs == 5 and fitted.metrics["df_resid"] == 3
    assert fitted.provenance["omitted_terms"] == [] and [c.term for c in fitted.coefficients] == ["Intercept", "x2"]


def test_cnsreg_model_test_rank_and_the_spec_level_robust_alias(data):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(60, 10)), columns=[f"z{i}" for i in range(10)])
    frame["g"] = np.repeat(np.arange(5), 12)
    frame["y"] = frame.z0 + rng.normal(size=60)
    columns = [f"z{i}" for i in range(10)]
    constraints = [{"terms": {"z8": 1, "z9": -1}, "value": 0}]
    # structural rank of the nine free slope directions: 10 slopes minus one constraint
    plain = oe.cnsreg(data=frame, y="y", x=columns, constraints=constraints)
    assert plain.tests["model"]["df"] == 9 == plain.metrics["df_model"]
    assert "rank_deficient" not in plain.tests["model"]
    # five clusters give a covariance of rank four: flagged, not silently reduced
    clustered = oe.cnsreg(data=frame, y="y", x=columns, constraints=constraints, cluster="g")
    test = clustered.tests["model"]
    assert test["rank_deficient"] is True and test["df"] == 4 and test["restrictions"] == 9
    assert any("Stata would report a missing F" in warning for warning in clustered.warnings)
    # 'robust' in a directly built spec is HC1
    spec = ModelSpec(estimator="cnsreg", outcome="y", predictors=["x1", "x2", "x3"], covariance="robust",
                     options={"constraints": [{"terms": {"x1": 1, "x2": 1}, "value": 1}]})
    direct = oe.fit(spec, data=data)
    named = oe.cnsreg(data=data, y="y", x=["x1", "x2", "x3"], covariance="HC1",
                      constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1}])
    assert direct.spec.covariance == "robust" and direct.inference["covariance"] == "HC1"
    assert_allclose(np.asarray(direct.covariance_matrix), np.asarray(named.covariance_matrix), rtol=1e-13)
