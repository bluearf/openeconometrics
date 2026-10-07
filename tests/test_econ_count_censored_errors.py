"""Observed count events, identification and finite result contracts."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

import openecon as oe
from test_econ_count_censored import coefficients, estimate, make_data


@pytest.mark.parametrize("value", [-1, 1.2, True, np.inf, 2**53 + 1])
@pytest.mark.parametrize("name", ["ll", "ul"])
def test_fixed_count_limits_must_be_exact_nonnegative_integers(name, value):
    with pytest.raises(oe.AnalysisError) as error:
        oe.cpoisson(data=make_data(), y="y", x=["x"], **{name: value})
    assert error.value.code == "invalid_censoring_limit"


@pytest.mark.parametrize("value", [-1., 1.5, np.inf, True, float(2**53 * 2)])
def test_per_row_invalid_limits_are_not_silently_rounded_or_ignored(value):
    data = make_data()
    data["ll"] = value
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code == "invalid_censoring_limit"


def test_unknown_status_and_missing_required_event_limit_are_clear_errors():
    data = make_data()
    data.loc[3, "status"] = "unexpected"
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code == "invalid_censoring_status"
    data = make_data()
    data.loc[3, "ll"] = np.nan
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code == "missing_censoring_limit"


def test_limits_cannot_cross_and_count_cannot_contradict_the_declared_interval():
    data = make_data()
    data.loc[3, "ll"] = data.loc[3, "ul"] + 1
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code == "invalid_censoring_limits"
    data = make_data()
    data.loc[3, "y"] = data.loc[3, "ul"] + 1
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code == "inconsistent_censoring"


@pytest.mark.parametrize("value", [-1, .2, np.inf, 2**53 + 1])
def test_invalid_observed_counts_are_rejected(value):
    data = make_data()
    data["y"] = value
    with pytest.raises(oe.AnalysisError) as error:
        estimate(data, "poisson")
    assert error.value.code in {"invalid_count_outcome", "non_finite_values"}


@pytest.mark.parametrize("kind", ["left", "right", "whole"])
def test_no_information_or_monotone_constant_shift_is_reported(kind):
    data = pd.DataFrame({"x": np.tile([-1., 1.], 25), "y": 1})
    options = {"ll": 1} if kind == "left" else {"ul": 1 if kind == "right" else 0}
    with pytest.raises(oe.AnalysisError) as error:
        oe.cpoisson(data=data, y="y", x=["x"], **options)
    assert error.value.code == ("unidentified_censoring" if kind == "whole" else "separation_detected")


@pytest.mark.parametrize("right", [False, True])
def test_no_intercept_balanced_signs_have_a_finite_maximum_despite_all_zero_or_all_right_events(right):
    x = np.tile([-1., 1.], 30)
    data = pd.DataFrame({"x": x, "y": int(right)})
    result = oe.cpoisson(data=data, y="y", x=["x"], intercept=False, **({"ul": 1} if right else {}))
    def function(beta):
        return np.log(-np.expm1(-np.exp(beta * x))).sum() if right else -np.exp(beta * x).sum()
    oracle = minimize_scalar(lambda beta: -function(beta), bracket=(-1, 0, 1))
    assert coefficients(result)[0] == pytest.approx(oracle.x, abs=1e-8)
    assert result.metrics["log_likelihood"] == pytest.approx(-oracle.fun, rel=1e-12)
    step = 1e-4
    hessian = (function(step) - 2 * function(0) + function(-step)) / step**2
    assert result.covariance_matrix[0][0] == pytest.approx(-1 / hessian, rel=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_predictor_isolating_only_censored_events_has_no_finite_mle(side):
    rng = np.random.default_rng(14)
    n = 200
    indicator = np.repeat([0., 1.], n // 2)
    y = rng.poisson(2., n)
    y[indicator == 1] = 2 if side == "left" else 4
    data = pd.DataFrame({"x": indicator, "y": y,
                         "bound": np.where(indicator == 1, 2 if side == "left" else 4, np.nan)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cpoisson(data=data, y="y", x=["x"], **({"ll": "bound"} if side == "left" else {"ul": "bound"}))
    assert error.value.code == "separation_detected"


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_non_overdispersed_counts_report_the_poisson_dispersion_boundary(form):
    data = pd.DataFrame({"x": np.tile([-1., 0., 1.], 60), "y": np.tile([1, 2, 3], 60)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cnbreg(data=data, y="y", x=["x"], dispersion=form)
    assert error.value.code == "boundary_solution"


def test_zero_weight_events_do_not_change_identification_or_estimates():
    data = make_data("poisson", "auto", n=300)
    data["weights"] = 1.
    reference = oe.cpoisson(data=data, y="y", x=["x"], ll=1, ul=5, exposure="exposure")
    ignored = data.iloc[[0]].copy()
    ignored["y"], ignored["weights"] = 999, 0
    expanded = pd.concat([data, ignored], ignore_index=True)
    actual = oe.cpoisson(data=expanded, y="y", x=["x"], ll=1, ul=5, exposure="exposure",
                         weights="weights", weight_type="iweight")
    assert actual.nobs == reference.nobs
    assert_allclose(coefficients(actual), coefficients(reference), rtol=1e-10)


def test_actual_stopped_newton_iteration_is_not_returned_as_a_valid_model(monkeypatch):
    from openecon.econometrics.count import common
    original = common.maximize
    def stop_early(*args, **kwargs):
        return original(*args, **kwargs, max_iter=1)
    monkeypatch.setattr(common, "maximize", stop_early)
    with pytest.raises(oe.AnalysisError) as error:
        estimate(make_data(), "poisson")
    assert error.value.code == "nonconvergence"


def test_equal_interval_endpoints_reduce_to_a_point_mass_and_collinearity_is_reported():
    data = make_data("poisson", "none", n=300)
    data["status"], data["ll"], data["ul"] = "interval", data.y, data.y
    data["duplicate"] = data.x * 2
    actual = oe.cpoisson(data=data, y="y", x=["x", "duplicate"], ll="ll", ul="ul",
                         censoring="status", exposure="exposure")
    reference = oe.poisson(data=data, y="y", x=["x"], exposure="exposure")
    assert_allclose(coefficients(actual), coefficients(reference), rtol=1e-9)
    assert any("duplicate" in note for note in actual.warnings)


def test_offset_exposure_and_weight_covariance_conflicts_are_rejected():
    data = make_data()
    for options in ({"offset": "offset", "exposure": "exposure"},
                    {"weights": "w", "weight_type": "pweight", "covariance": "nonrobust"},
                    {"dispersion": "unsupported"}):
        function = oe.cnbreg if "dispersion" in options else oe.cpoisson
        with pytest.raises(oe.AnalysisError):
            function(data=data, y="y", x=["x"], **options)


@pytest.mark.parametrize("form", ["mean", "constant"])
@pytest.mark.parametrize("count", [10**9, 10**12, 2**53])
def test_large_nb_count_endpoints_have_a_specific_precision_error_before_starting_fit(form, count):
    data = pd.DataFrame({"x": np.tile([-1., 1.], 25), "y": np.tile([count - 1, count], 25)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cnbreg(data=data, y="y", x=["x"], dispersion=form)
    assert error.value.code == "precision_unsupported"


def test_poisson_unsupported_cumulative_endpoint_is_distinct_from_exact_count_precision():
    data = pd.DataFrame({"x": np.tile([-1., 1.], 25), "y": np.tile([2, 10**12 + 2], 25)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cpoisson(data=data, y="y", x=["x"], ul=10**12 + 2)
    assert error.value.code == "precision_unsupported"


def test_nb_all_start_shapes_outside_domain_keep_precision_error_not_invalid_start():
    data = pd.DataFrame({"x": np.tile([-1., 1.], 25), "y": 5, "offset": np.log(1e8)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cnbreg(data=data, y="y", x=["x"], dispersion="constant", intercept=False, offset="offset")
    assert error.value.code == "precision_unsupported"


def test_uncertified_nb_precision_trial_failure_is_not_reported_as_dispersion_boundary(monkeypatch):
    import openecon.econometrics.count.censored_kernels as kernels
    # An intentionally narrowed numerical region is reached by actual Newton
    # trials before the model's independent zero-dispersion certificate.
    monkeypatch.setattr(kernels, "_NB_MAX_GAMMA_ARGUMENT", 100.)
    data = pd.DataFrame({"x": np.tile([-1., 0., 1.], 60), "y": np.tile([1, 2, 3], 60)})
    with pytest.raises(oe.AnalysisError) as error:
        oe.cnbreg(data=data, y="y", x=["x"])
    assert error.value.code == "precision_unsupported"


def test_unsupported_optional_null_comparison_preserves_certified_fit_and_uses_wald(monkeypatch):
    from openecon.econometrics.count import common
    data = make_data("mean", "auto", n=400)
    reference = oe.cnbreg(data=data, y="y", x=["x"], ll=1, ul=5, exposure="exposure")
    original = common.run
    def unsupported_null(*args, **kwargs):
        if kwargs["what"].endswith(" null model"):
            raise oe.AnalysisError("precision_unsupported", "Injected optional comparison scale failure.")
        return original(*args, **kwargs)
    monkeypatch.setattr(common, "run", unsupported_null)
    actual = oe.cnbreg(data=data, y="y", x=["x"], ll=1, ul=5, exposure="exposure")
    assert_allclose(coefficients(actual), coefficients(reference), rtol=1e-12)
    assert_allclose(actual.covariance_matrix, reference.covariance_matrix, rtol=1e-12)
    assert actual.extra["null_log_likelihood"] is None
    assert any("precision domain" in note and "Wald" in note for note in actual.warnings)


def test_unverified_final_nb_parameters_are_never_returned_as_a_converged_model(monkeypatch):
    from openecon.econometrics.count import common
    original = common.run
    def corrupted_final(*args, **kwargs):
        result, stop = original(*args, **kwargs)
        if kwargs["what"] == "cnbreg" and result is not None and result.converged:
            result.theta[-1] = -25.
        return result, stop
    monkeypatch.setattr(common, "run", corrupted_final)
    with pytest.raises(oe.AnalysisError) as error:
        estimate(make_data("mean", "auto", n=300), "mean")
    assert error.value.code == "precision_unsupported"
