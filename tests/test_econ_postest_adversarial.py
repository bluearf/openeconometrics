"""Adversarial inputs and regression tests of the post-estimation family.

Every invalid input must either work correctly or raise ``AnalysisError`` with
a helpful code; each defect fixed in the verify-and-repair pass has a
regression test here (or in ``test_econ_postest_oracle.py``).
"""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.common import refit_parameters, refit_spec


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(99)
    n = 120
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "g": rng.choice(list("abcdefghij"), n), "one": "same"})
    frame["y"] = 1 + frame.x1 + rng.normal(size=n)
    frame["b"] = (frame.x1 + rng.logistic(size=n) > 0).astype(int)
    frame["cnt"] = rng.poisson(np.exp(0.2 + 0.3 * frame.x1))
    return frame


def code_of(call):
    with pytest.raises(AnalysisError) as error:
        call()
    return error.value.code


# ---- regression tests of fixed defects ------------------------------------------------


def test_seed_outside_the_generator_range_is_refused(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1"])
    assert code_of(lambda: oe.bootstrap(fit, data, reps=3, seed=2**64)) == "invalid_spec"
    assert code_of(lambda: oe.bootstrap(fit, data, reps=3, seed=-1)) == "invalid_spec"
    boot = oe.bootstrap(fit, data, reps=3, seed=2**64 - 1)
    assert boot.inference["seed"] == 2**64 - 1


def test_a_single_cluster_is_refused_with_its_own_code(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1"])
    assert code_of(lambda: oe.bootstrap(fit, data, reps=3, seed=1, cluster="one")) \
        == "insufficient_clusters"
    assert code_of(lambda: oe.jackknife(fit, data, cluster="one")) == "insufficient_clusters"


def test_exact_fits_have_no_resampling_covariance(data):
    exact = data.assign(yexact=1 + 2 * data.x1)
    fit = oe.ols(data=exact, y="yexact", x=["x1"])
    for call in (lambda: oe.bootstrap(fit, exact, reps=10, seed=1),
                 lambda: oe.jackknife(fit, exact)):
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "invalid_covariance" and "exactly" in str(error.value)


def test_multidimensional_tensors_are_refused():
    assert code_of(lambda: oe.fcast_eval(torch.ones(5, 2), torch.ones(5, 2))) == "invalid_data"
    table = oe.fcast_eval(torch.arange(1.0, 6.0).reshape(5, 1), torch.arange(2.0, 7.0))
    assert table.loc["n", "value"] == 5


def test_names_may_be_any_iterable(data):
    m1 = oe.logit(data=data, y="b", x=["x1"])
    m2 = oe.poisson(data=data, y="cnt", x=["x1"])
    table = oe.estat_ic(m1, m2, names=(name for name in ["a", "b"]))
    assert list(table.model) == ["a", "b"]
    joint = oe.suest(m1, m2, data=data, names=iter(["L", "P"]))
    assert joint.coefficients[0].term == "L:Intercept"
    assert code_of(lambda: oe.estat_ic(m1, names="ab")) == "invalid_spec"


def test_refit_exceptions_count_as_failed_replicates():
    def broken(frame):
        raise RuntimeError("linalg.solve: the solver failed")

    values, reason = refit_parameters(broken, pd.DataFrame({"x": [1.0]}), ["x"])
    assert values is None and reason == "numerical_failure (RuntimeError)"


def test_clustered_fit_resampled_by_observation_says_so(data):
    fit = oe.ols(data=data, y="y", x=["x1"], cluster="g")
    boot = oe.bootstrap(fit, data, reps=5, seed=1)
    assert any("pass cluster='g'" in note for note in boot.warnings)
    jk = oe.jackknife(fit, data, cluster="g")
    assert not any("pass cluster=" in note for note in jk.warnings)


def test_mlogit_refits_pin_the_base_category():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x": rng.normal(size=60), "y": rng.integers(0, 3, 60)})
    fit = oe.mlogit(data=frame, y="y", x=["x"])
    assert refit_spec(fit).options["base"] == fit.extra["base"]
    explicit = oe.mlogit(data=frame, y="y", x=["x"], base=2)
    assert refit_spec(explicit) is explicit.spec


# ---- adversarial inputs -----------------------------------------------------------------


def test_tiny_and_degenerate_samples(data):
    tiny = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.5], "y": [1.0, 2.5, 2.9, 4.4]})
    fit = oe.ols(data=tiny, y="y", x=["x"])
    assert code_of(lambda: oe.bootstrap(fit, tiny, reps=20, seed=1)) == "bootstrap_failed"
    jk = oe.jackknife(fit, tiny)
    assert jk.inference["df_inference"] == 3
    poisson = oe.poisson(data=data, y="cnt", x=["x1"])
    assert code_of(lambda: oe.bootstrap(poisson, data, reps=5, seed=1, size=1)) \
        == "bootstrap_failed"
    assert code_of(lambda: oe.fcast_eval([np.nan, 1.0], [1.0, np.nan], missing="drop")) \
        == "insufficient_observations"


def test_option_bounds(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1"])
    assert oe.bootstrap(fit, data, reps=2, seed=1).inference["reps_completed"] == 2
    for bad in ({"reps": 1}, {"reps": 2.0}, {"reps": True}, {"ci": "unknown"}, {"size": 0},
                {"size": 10_000}, {"alpha": 0.0}, {"alpha": 1}):
        options = {"reps": 3, "seed": 1, **bad}
        assert code_of(lambda options=options: oe.bootstrap(fit, data, **options)) \
            == "invalid_spec"
    assert code_of(lambda: oe.jackknife(fit, data, max_refits=1)) == "invalid_spec"
    assert code_of(lambda: oe.jackknife(fit, data, max_refits=50)) == "jackknife_too_large"
    rng = np.random.default_rng(0)
    y, f1, f2 = rng.normal(size=(3, 10))
    assert oe.dm_test(y, f1, f2, horizon=9).iloc[0].df == 9
    for bad in ({"horizon": 10}, {"horizon": 0}, {"horizon": "2"}, {"loss": "pinball"},
                {"kernel": "parzen"}, {"harvey": 1}, {"missing": "keep"}):
        code = code_of(lambda bad=bad: oe.dm_test(y, f1, f2, **bad))
        assert code in {"invalid_spec", "insufficient_observations"}
    assert code_of(lambda: oe.fcast_eval(y, f1, seasonal_period=0)) == "invalid_spec"


def test_wrong_types_and_columns(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1"])
    assert code_of(lambda: oe.bootstrap("fit", data, reps=3)) == "invalid_result"
    assert code_of(lambda: oe.bootstrap(fit, None, reps=3)) == "invalid_data"
    assert code_of(lambda: oe.bootstrap(fit, data, reps=3, cluster="missing")) \
        == "missing_columns"
    assert code_of(lambda: oe.bootstrap(fit, data, reps=3, strata=3)) == "missing_columns"
    assert code_of(lambda: oe.bootstrap(fit, data.iloc[::-1], reps=3)) == "data_mismatch"
    assert code_of(lambda: oe.bootstrap(fit, data.drop(columns="x1"), reps=3)) \
        == "data_mismatch"
    edited = data.copy()
    edited.loc[0, "x1"] += 1e-9
    assert code_of(lambda: oe.jackknife(fit, edited)) == "data_mismatch"
    assert code_of(lambda: oe.fcast_eval(["a", "b"], [1, 2])) == "invalid_data"
    assert code_of(lambda: oe.fcast_eval("y", "x1")) == "invalid_data"
    assert code_of(lambda: oe.fcast_eval("y", "nope", data=data)) == "missing_columns"
    assert code_of(lambda: oe.fcast_eval([1.0, 2.0, 3.0], [1.0, 2.0])) == "length_mismatch"
    assert code_of(lambda: oe.fcast_eval([1.0, np.inf], [1.0, 2.0])) == "missing_values"
    assert code_of(lambda: oe.lrtest(fit, "restricted")) == "invalid_result"
    assert code_of(lambda: oe.lrtest(fit, fit, force="yes")) == "invalid_spec"
    assert code_of(lambda: oe.lrtest(fit, fit)) == "incompatible_models"


def test_missing_and_constant_series_in_forecast_evaluation():
    table = oe.fcast_eval([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 0.0])
    assert pd.isna(table.loc["theil_u2", "value"]) and pd.isna(table.loc["mase", "value"])
    assert any("does not change" in note for note in table.attrs["notes"])
    table = oe.fcast_eval([0.0, 1.0, 2.0], [0.5, 1.0, 2.5])
    assert pd.isna(table.loc["mape", "value"])
    gappy = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0, 5.0, 3.0],
                          "f1": [1.1, 2.2, 3.0, np.nan, 4.5, 3.1],
                          "f2": [0.8, 2.5, 2.0, 3.0, 5.5, 2.0]})
    assert code_of(lambda: oe.dm_test("y", "f1", "f2", data=gappy)) == "missing_values"
    result = oe.dm_test("y", "f1", "f2", data=gappy, missing="drop")
    assert result.iloc[0].n == 4 and any("Excluded 2" in n for n in result.attrs["notes"])
    assert code_of(lambda: oe.dm_test([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 5], [1, 2, 3, 5])) \
        == "constant_loss_differential"


def test_extreme_magnitudes_are_scale_equivariant(data):
    base = oe.suest(oe.logit(data=data, y="b", x=["x1"]),
                    oe.poisson(data=data, y="cnt", x=["x1"]), data=data)
    for scale in (1e8, 1e-8):
        scaled = data.assign(x1=data.x1 * scale)
        joint = oe.suest(oe.logit(data=scaled, y="b", x=["x1"]),
                         oe.poisson(data=scaled, y="cnt", x=["x1"]), data=scaled)
        transform = np.diag([1, 1 / scale, 1, 1 / scale])
        expected = transform @ np.array(base.covariance_matrix) @ transform
        assert_allclose(np.array(joint.covariance_matrix), expected, rtol=1e-8,
                        atol=1e-10 * np.abs(expected).max())
        rng = np.random.default_rng(1)
        y = rng.normal(size=60) * scale
        f1, f2 = y + rng.normal(size=60) * scale, y + 1.2 * rng.normal(size=60) * scale
        unit = oe.dm_test(y / scale, f1 / scale, f2 / scale).iloc[0].statistic
        assert_allclose(oe.dm_test(y, f1, f2).iloc[0].statistic, unit, rtol=1e-9)


def test_row_permutations_do_not_change_the_jackknife_or_suest(data):
    shuffled = data.sample(frac=1, random_state=4).reset_index(drop=True)
    a = oe.jackknife(oe.poisson(data=data, y="cnt", x=["x1"]), data)
    b = oe.jackknife(oe.poisson(data=shuffled, y="cnt", x=["x1"]), shuffled)
    assert_allclose(np.array(a.covariance_matrix), np.array(b.covariance_matrix), rtol=1e-9)
    s1 = oe.suest(oe.logit(data=data, y="b", x=["x1"]), oe.ols(data=data, y="y", x=["x1"]),
                  data=data, cluster="g")
    s2 = oe.suest(oe.logit(data=shuffled, y="b", x=["x1"]),
                  oe.ols(data=shuffled, y="y", x=["x1"]), data=shuffled, cluster="g")
    assert_allclose(np.array(s1.covariance_matrix), np.array(s2.covariance_matrix), rtol=1e-8)


def test_collinear_terms_stay_omitted_in_every_replicate(data):
    frame = data.assign(x3=2 * data.x1)
    fit = oe.ols(data=frame, y="y", x=["x1", "x3"])
    assert [c.term for c in fit.coefficients] == ["Intercept", "x1"]
    boot = oe.bootstrap(fit, frame, reps=10, seed=3)
    assert boot.inference["failed_replicates"] == 0
    assert [c.term for c in boot.coefficients] == ["Intercept", "x1"]


def test_refusals_of_unsupported_combinations(data):
    weighted = oe.poisson(data=data.assign(w=1.0 + (data.x2 > 0)), y="cnt", x=["x1"],
                          weights="w", weight_type="pweight")
    logit = oe.logit(data=data, y="b", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.suest(logit, weighted, data=data.assign(w=1.0 + (data.x2 > 0)))
    assert error.value.code == "suest_unsupported" and "same weight" in str(error.value)
    nbreg = oe.nbreg(data=data, y="cnt", x=["x1"])
    assert oe.suest(logit, nbreg, data=data).nobs == len(data)
    quantile = oe.qreg(data=data, y="y", x=["x1"])
    assert code_of(lambda: oe.suest(logit, quantile, data=data)) == "suest_unsupported"
    boot = oe.bootstrap(logit, data, reps=5, seed=1)
    assert code_of(lambda: oe.bootstrap(boot, data, reps=5)) == "bootstrap_unsupported"
    # suest rebuilds scores from the specification, so the input covariance is irrelevant.
    joint = oe.suest(boot, oe.probit(data=data, y="b", x=["x1"]), data=data)
    plain = oe.suest(logit, oe.probit(data=data, y="b", x=["x1"]), data=data)
    assert_allclose(np.array(joint.covariance_matrix), np.array(plain.covariance_matrix))
    series = pd.DataFrame({"t": np.arange(60), "y": np.random.default_rng(2).normal(size=60)})
    arima = oe.arima(data=series, y="y", time="t", order=(1, 0, 0))
    assert code_of(lambda: oe.bootstrap(arima, series, reps=5)) == "bootstrap_unsupported"
    assert code_of(lambda: oe.jackknife(arima, series)) == "jackknife_unsupported"
    robust = oe.poisson(data=data, y="cnt", x=["x1"], covariance="robust")
    small = oe.poisson(data=data, y="cnt", x=["x1", "x2"], covariance="robust")
    assert code_of(lambda: oe.lrtest(small, robust)) == "incompatible_models"
    assert oe.lrtest(small, robust, force=True).attrs["forced"] is True
    ols = oe.ols(data=data, y="y", x=["x1"])
    assert code_of(lambda: oe.estat_ic(ols, boot.model_copy(update={"metrics": {}}))) \
        == "missing_log_likelihood"
