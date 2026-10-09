"""Independent observed-likelihood optimizer and analytical pattern acceptance."""

import json
import math

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2, multivariate_normal
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.mi import diagnostics as d
from openecon.resources import use_workspace_budget


def _random_missing():
    rng = np.random.default_rng(7163)
    y = rng.multivariate_normal(
        [1.2, -0.3, 2.4], [[1.7, 0.5, 0.2], [0.5, 1.1, -0.1], [0.2, -0.1, 0.8]], size=240
    )
    y[10:45, 0] = np.nan
    y[60:99, 1] = np.nan
    y[110:144, [0, 2]] = np.nan
    y[150:181, 2] = np.nan
    y[200:215, :2] = np.nan
    y[-2:, :] = np.nan
    return pd.DataFrame(
        y,
        columns=["a", "b", "c"],
        index=pd.Index(["duplicate" if i % 2 else i for i in range(len(y))]),
    )


def _observed_likelihood_oracle(frame):
    """Direct optimised observed marginal normal densities; no EM recurrence."""
    y = frame.to_numpy()
    p = y.shape[1]
    mask = ~np.isnan(y)
    groups = []
    for observed in np.unique(mask, axis=0):
        if observed.any():
            rows = np.all(mask == observed, axis=1)
            groups.append((np.flatnonzero(observed), y[rows][:, observed]))
    lower = np.tril_indices(p)

    def unpack(z):
        factor = np.zeros((p, p))
        factor[lower] = z[p:]
        factor[np.diag_indices(p)] = np.exp(factor[np.diag_indices(p)])
        return z[:p], factor @ factor.T

    def objective(z):
        mean, covariance = unpack(z)
        return -sum(
            float(
                multivariate_normal.logpdf(
                    values, mean=mean[ix], cov=covariance[np.ix_(ix, ix)]
                ).sum()
            )
            for ix, values in groups
        )

    start_mean = np.nanmean(y, axis=0)
    complete = y[mask.all(axis=1)]
    start_factor = np.linalg.cholesky(np.cov(complete.T, bias=True))
    start_factor[np.diag_indices(p)] = np.log(start_factor.diagonal())
    initial = np.r_[start_mean, start_factor[lower]]
    solution = minimize(objective, initial, method="BFGS", options={"gtol": 1e-6, "maxiter": 1500})
    assert np.linalg.norm(solution.jac, ord=np.inf) < 2e-4
    mean, covariance = unpack(solution.x)
    return mean, covariance, -solution.fun


def _hand_patterns(all_missing=0):
    # 8 complete balanced independent +/-1 profiles and 3 x-only [0,1,2].
    # Exact fitted means=(3/11,0), ML covariance=diag(134/121,1).
    rows = [[x, y] for _ in range(2) for x in [-1.0, 1.0] for y in [-1.0, 1.0]]
    rows += [[0.0, math.nan], [1.0, math.nan], [2.0, math.nan]]
    rows += [[math.nan, math.nan]] * all_missing
    return pd.DataFrame(rows, columns=["x", "y"], index=["same"] * len(rows))


def test_observed_em_matches_independent_scipy_likelihood_optimizer():
    frame = _random_missing()
    mean, covariance, loglikelihood = _observed_likelihood_oracle(frame)
    result = d.mvnorm_em(frame, ["a", "b", "c"], tolerance=1e-10)
    np.testing.assert_allclose(result.estimates, mean, rtol=0, atol=2e-5)
    np.testing.assert_allclose(result.covariance, covariance, rtol=0, atol=2e-5)
    assert result.observed_loglikelihood == pytest.approx(loglikelihood, abs=2e-8)
    assert min(np.diff(result.loglikelihood_history)) >= -1e-8
    assert result.final_parameter_change <= 1e-10
    assert result.sample.n_total == 240
    assert result.sample.n_informative == 238
    assert result.sample.all_missing_positions == (238, 239)
    assert len(result.sample.row_labels) == 240
    assert result.sample.row_labels[1] == "builtins.str:'duplicate'"
    assert result.covariance_kind == "Gaussian population covariance; maximum likelihood divisor"


def test_little_statistic_matches_closed_pattern_oracle_and_finite_correction():
    frame = _hand_patterns(all_missing=2)
    result = d.little_mcar(frame, ["x", "y"])
    np.testing.assert_allclose(result.estimates, [3 / 11, 0], atol=1e-12)
    np.testing.assert_allclose(result.covariance, [[134 / 121, 0], [0, 1]], atol=1e-12)
    assert result.statistic_covariance_scale == 11 / 10
    assert result.statistic == pytest.approx(1320 / 737, abs=1e-12)
    assert result.df == 1
    assert result.p_value == pytest.approx(chi2.sf(1320 / 737, 1), abs=1e-13)
    empty = next(p for p in result.patterns if not p.observed_columns)
    assert empty.positions == (11, 12)
    assert empty.statistic_contribution == 0.0
    baseline = d.little_mcar(_hand_patterns(), ["x", "y"])
    assert baseline.estimates == result.estimates
    assert baseline.covariance == result.covariance
    assert baseline.statistic == result.statistic
    assert baseline.observed_loglikelihood == result.observed_loglikelihood


def test_complete_em_matches_direct_ml_moments_and_univariate_missing_fit():
    y = np.array([[1, 2], [3, 2], [4, 7], [-2, 0], [6, -1]], dtype=float)
    result = d.mvnorm_em(pd.DataFrame(y, columns=["x", "y"]), ["x", "y"])
    np.testing.assert_allclose(result.estimates, y.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(result.covariance, np.cov(y.T, bias=True), atol=1e-12)
    one = d.mvnorm_em({"x": [1.0, 2.0, 4.0, math.nan]}, ["x"])
    assert one.estimates == pytest.approx((7 / 3,))
    assert one.covariance[0][0] == pytest.approx(14 / 9)


def test_affine_units_and_column_order_restore_full_covariance():
    frame = _random_missing()
    baseline = d.little_mcar(frame, ["a", "b", "c"])
    shifted = frame.copy()
    shifted["a"] = shifted["a"] * 1e12 + 1e10
    shifted["b"] = shifted["b"] * 1e-12 - 1e-13
    changed = d.little_mcar(shifted, ["c", "b", "a"])
    assert changed.statistic == pytest.approx(baseline.statistic, abs=1e-8)
    assert changed.p_value == pytest.approx(baseline.p_value, abs=1e-10)
    transform = np.array([1, 1e-12, 1e12])
    expected = (
        np.array(baseline.covariance)[np.ix_([2, 1, 0], [2, 1, 0])]
        * transform[:, None]
        * transform[None, :]
    )
    np.testing.assert_allclose(changed.covariance, expected, rtol=1e-9, atol=0)


def test_roundtrip_frozen_nested_state_tables_and_latex(tmp_path):
    result = d.little_mcar(_hand_patterns(2), ["x", "y"])
    saved = tmp_path / "diagnostic.json"
    result.to_json(saved)
    assert result == d.MIDiagnosticResult.from_json(saved)
    assert result == d.MIDiagnosticResult.model_validate_json(result.to_json())
    with pytest.raises(ValidationError):
        result.statistic = 0
    with pytest.raises(TypeError):
        result.covariance[0][0] = 0
    with pytest.raises(ValidationError):
        result.sample.n_total = 2
    frames = result.to_frame()
    assert frames.attrs["nobs"] == 11
    assert frames["Little MCAR mean homogeneity"].iloc[0]["statistic"] == result.statistic
    assert "Gaussian ML population covariance" in result.to_latex()
    assert "chi-square" in frames.attrs["inference"]
    assert "does not establish MCAR" in " ".join(result.notes)


@pytest.mark.parametrize(
    "field", ["estimates", "covariance", "patterns", "observed_loglikelihood", "sample"]
)
def test_saved_state_tamper_fails_integrity_or_semantics(field):
    result = d.little_mcar(_hand_patterns(), ["x", "y"])
    value = result.model_dump(mode="json")
    if field == "estimates":
        value[field][0] += 1
    elif field == "covariance":
        value[field][0][0] *= 2
    elif field == "patterns":
        value[field][0]["positions"][0] = 999
    elif field == "observed_loglikelihood":
        value[field] += 1
    else:
        value[field]["row_labels"][0] = "changed"
    with pytest.raises(ValidationError):
        d.MIDiagnosticResult.model_validate_json(json.dumps(value))


def test_semantic_saved_state_validation_survives_recomputed_digest():
    result = d.little_mcar(_hand_patterns(), ["x", "y"])
    value = result.model_dump(mode="json")
    value["statistic"] += 0.5
    value["integrity_sha256"] = d._digest(
        {k: v for k, v in value.items() if k != "integrity_sha256"}
    )
    with pytest.raises(ValidationError, match="statistic/p-value"):
        d.MIDiagnosticResult.model_validate(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iterations", True),
        ("max_iterations", 0),
        ("max_iterations", 10001),
        ("tolerance", True),
        ("tolerance", math.nan),
        ("tolerance", 0),
        ("tolerance", 0.1),
    ],
)
def test_declared_options_fail(field, value):
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(_hand_patterns(), ["x", "y"], **{field: value})
    assert caught.value.code == "invalid_option"


@pytest.mark.parametrize("kwargs", [{"weights": "w"}, {"device": "cuda"}, {"missing": "drop"}])
def test_unsupported_options_fail(kwargs):
    with pytest.raises(TypeError):
        d.mvnorm_em(_hand_patterns(), ["x", "y"], **kwargs)


def test_dataset_rejected_without_materialization():
    source = Dataset.from_frame(_hand_patterns())
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(source, ["x", "y"])
    assert caught.value.code == "unsupported_dataset"


@pytest.mark.parametrize(
    "frame,columns,code",
    [
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [math.nan] * 3}),
            ["x", "y"],
            "mi_nonidentification",
        ),
        (
            pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0, math.nan, math.nan, math.nan],
                    "y": [math.nan] * 3 + [1.0, 2.0, 3.0],
                }
            ),
            ["x", "y"],
            "mi_nonidentification",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}),
            ["x", "y"],
            "mi_singular_pair_coverage",
        ),
        (pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]}), ["x"], "mi_singular_covariance"),
        (pd.DataFrame({"x": [math.nan] * 4}), ["x"], "mi_nonidentification"),
        (pd.DataFrame({"x": [1.0, 2.0, math.inf, 4.0]}), ["x"], "nonfinite_data"),
        (pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["absent"], "missing_columns"),
        (pd.DataFrame({"x": ["1", "2", "3"]}), ["x"], "non_numeric_column"),
        (pd.DataFrame({"x": [True, False, True]}), ["x"], "non_numeric_column"),
        (pd.DataFrame({"x": [1j, 2j, 3j]}), ["x"], "non_numeric_column"),
    ],
)
def test_nonidentification_and_invalid_input_fail_loud(frame, columns, code):
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(frame, columns)
    assert caught.value.code == code


def test_complete_patterns_have_explicit_degenerate_mcar_error():
    frame = _hand_patterns().iloc[:8]
    with pytest.raises(AnalysisError) as caught:
        d.little_mcar(frame, ["x", "y"])
    assert caught.value.code == "mi_degenerate_mcar"


def test_em_nonconvergence_is_not_a_successful_result():
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(_random_missing(), ["a", "b", "c"], max_iterations=1, tolerance=1e-12)
    assert caught.value.code == "mi_nonconvergence"
    assert caught.value.iterations == 1
    assert len(caught.value.loglikelihood_history) == 2
    assert caught.value.final_parameter_change > 1e-12


def test_workspace_rejection_happens_before_tensor_allocation(monkeypatch):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 5000})

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor allocation occurred before workspace rejection")

    monkeypatch.setattr(torch, "tensor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(frame, ["x"])
    assert caught.value.code == "workspace_limit"
    assert caught.value.resource_plan["estimated_workspace_bytes"] > 1024**2


def test_dimension_and_iteration_work_guards_before_tensor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor allocation occurred before admission rejection")

    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(
            pd.DataFrame({f"x{i}": [1.0, 2.0, 3.0] for i in range(17)}),
            [f"x{i}" for i in range(17)],
        )
    assert caught.value.code == "mi_dimension_limit"
    with pytest.raises(AnalysisError) as caught:
        d.mvnorm_em(
            pd.DataFrame({f"x{i}": [1.0, 2.0, 3.0, 4.0] * 1000 for i in range(16)}),
            [f"x{i}" for i in range(16)],
            max_iterations=10000,
        )
    assert caught.value.code == "mi_work_limit"


def test_input_unchanged_and_nullable_float_missingness():
    frame = _hand_patterns(1).astype("Float64")
    before = frame.copy(deep=True)
    result = d.mvnorm_em(frame, ["x", "y"])
    pd.testing.assert_frame_equal(frame, before)
    assert result.sample.all_missing_positions == (11,)


def test_runtime_module_uses_only_torch_numerical_kernels():
    import inspect

    text = inspect.getsource(d)
    assert "import scipy" not in text
    assert "import numpy" not in text
    assert "statsmodels" not in text
    assert "torch.cholesky_solve" in text


def test_mapping_workspace_admission_precedes_dataframe_materialization(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("A refused workspace must not construct a DataFrame")

    monkeypatch.setattr(d, "source", refused)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        d.mvnorm_em({"x": list(range(10000)), "y": list(range(10000))}, ["x", "y"])


def test_diagnostic_copy_revalidation_and_ambient_cpu_float64():
    data = {"x": [1.0, 2.0, 4.0, 5.0, 8.0], "y": [2.0, 4.0, 3.0, float("nan"), 7.0]}
    old_dtype = torch.get_default_dtype()
    torch.set_default_device("meta")
    torch.set_default_dtype(torch.float32)
    try:
        fitted = d.mvnorm_em(data, ["x", "y"])
        result = d.little_mcar(data, ["x", "y"])
        assert result.p_value >= 0
        assert fitted.model_copy(deep=True) == fitted
        with pytest.raises(ValueError):
            fitted.model_copy(update={"estimates": (123.0, 456.0)})
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(old_dtype)


def test_saved_diagnostics_restore_admits_rows_patterns_and_json_before_coercion():
    # One continuous column has an exact complete-data limit; no dense n-square work.
    result = d.mvnorm_em({"x": [float(i % 17) for i in range(2000)]}, ["x"])
    saved = result.model_dump_json()
    with use_workspace_budget(1), pytest.raises(ValueError, match="restoration"):
        d.MIDiagnosticResult.model_validate_json(saved)
