"""Independent oracles for the bootstrap quantile commands bsqreg, sqreg and iqreg.

The bootstrap samples are redrawn in the test from the recorded seed, every
replicate is expanded to its duplicated rows and solved as an explicit linear
program with SciPy's HiGHS, and the covariance is formed with NumPy.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import linprog

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ModelSpec

X_COLS = ["x1", "x2"]


def lp_quantile(x, y, tau):
    n, k = x.shape
    cost = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1 - tau)])
    a_eq = np.hstack([x, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    solution = linprog(cost, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    assert solution.status == 0
    return solution.x[:k]


def make_data(seed=21, n=110):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.exponential(size=n)})
    noise = rng.standard_t(4, size=n) * (1 + 0.4 * frame.x2)
    frame["y"] = 1 + 0.5 * frame.x1 - 0.3 * frame.x2 + noise
    frame["g"] = rng.integers(0, 18, size=n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def design(frame):
    return np.column_stack([np.ones(len(frame)), frame[X_COLS].to_numpy()])


def params(result):
    return np.array([c.estimate for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def replicate_draws(frame, taus, reps, seed, cluster=None):
    """Bootstrap coefficients [reps, len(taus), k] from explicitly resampled rows."""
    x, y = design(frame), frame.y.to_numpy()
    n = len(y)
    generator = torch.Generator().manual_seed(seed)
    draws = np.empty((reps, len(taus), x.shape[1]))
    if cluster is not None:
        codes, labels = pd.factorize(frame[cluster], sort=False)
        members = [np.flatnonzero(codes == g) for g in range(len(labels))]
    for rep in range(reps):
        if cluster is None:
            rows = torch.randint(n, (n,), generator=generator).numpy()
        else:
            chosen = torch.randint(len(members), (len(members),), generator=generator).numpy()
            rows = np.concatenate([members[g] for g in chosen])
        for j, tau in enumerate(taus):
            draws[rep, j] = lp_quantile(x[rows], y[rows], tau)
    return draws


def test_bsqreg_matches_explicit_bootstrap(data):
    result = oe.bsqreg(data=data, y="y", x=X_COLS, quantile=0.3, reps=15, seed=123)
    point = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.3)
    assert_allclose(params(result), params(point), rtol=0, atol=0)
    assert_allclose(params(result), lp_quantile(design(data), data.y.to_numpy(), 0.3), rtol=1e-8)
    draws = replicate_draws(data, [0.3], 15, 123)[:, 0]
    assert_allclose(covariance(result), np.cov(draws.T, ddof=1), rtol=1e-6, atol=1e-12)
    assert result.spec.covariance == "bootstrap"
    record = result.extra["bootstrap"]
    assert record == result.provenance["bootstrap"]
    assert record["reps"] == record["reps_used"] == 15 and record["seed"] == 123
    assert record["resampling"] == "observations"
    assert result.metrics["reps"] == 15 and result.metrics["quantile"] == 0.3
    for name in ("pseudo_r_squared", "sum_adev", "sum_rdev", "raw_quantile"):
        assert_allclose(result.metrics[name], point.metrics[name], rtol=1e-12)
    # Student t with N - K degrees of freedom.
    df = len(data) - 3
    assert result.inference["df_inference"] == df and result.inference["use_t"]
    coefficient = result.coefficients[2]
    assert_allclose(coefficient.p_value, 2 * stats.t.sf(abs(coefficient.statistic), df), rtol=1e-8)
    assert "bootstrap" in result.inference["correction"]
    assert "bootstrap standard errors" in result.title


def test_bootstrap_is_deterministic_and_seed_is_recorded(data):
    first = oe.bsqreg(data=data, y="y", x=X_COLS, reps=10, seed=5)
    second = oe.bsqreg(data=data, y="y", x=X_COLS, reps=10, seed=5)
    other = oe.bsqreg(data=data, y="y", x=X_COLS, reps=10, seed=6)
    assert first.covariance_matrix == second.covariance_matrix
    assert first.covariance_matrix != other.covariance_matrix
    unseeded = oe.bsqreg(data=data, y="y", x=X_COLS, reps=10)
    seed = unseeded.extra["bootstrap"]["seed"]
    assert isinstance(seed, int) and "seed" not in unseeded.spec.options
    replay = oe.bsqreg(data=data, y="y", x=X_COLS, reps=10, seed=seed)
    assert replay.covariance_matrix == unseeded.covariance_matrix


def test_sqreg_joint_covariance(data):
    taus = [0.2, 0.5, 0.8]
    result = oe.sqreg(data=data, y="y", x=X_COLS, quantiles=taus, reps=12, seed=77)
    labels = ["q20", "q50", "q80"]
    names = ["Intercept", *X_COLS]
    assert [c.term for c in result.coefficients] == [f"{q}:{t}" for q in labels for t in names]
    assert [c.equation for c in result.coefficients] == [q for q in labels for _ in names]
    x, y = design(data), data.y.to_numpy()
    expected = np.concatenate([lp_quantile(x, y, tau) for tau in taus])
    assert_allclose(params(result), expected, rtol=1e-8, atol=1e-9)
    draws = replicate_draws(data, taus, 12, 77).reshape(12, -1)
    assert covariance(result).shape == (9, 9)
    assert_allclose(covariance(result), np.cov(draws.T, ddof=1), rtol=1e-6, atol=1e-12)
    for label, tau in zip(labels, taus, strict=True):
        point = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau)
        assert_allclose(result.metrics[f"pseudo_r_squared_{label}"],
                        point.metrics["pseudo_r_squared"], rtol=1e-12)
        assert_allclose(result.extra["sum_adev"][label], point.metrics["sum_adev"], rtol=1e-12)
        assert_allclose(result.extra["sum_rdev"][label], point.metrics["sum_rdev"], rtol=1e-12)
    assert result.extra["quantiles"] == taus and result.extra["equations"] == labels
    assert result.inference["df_inference"] == len(data) - 3
    # The between-quantile blocks allow cross-quantile tests: Var(b80 - b20) for x2.
    v = covariance(result)
    difference = v[8, 8] + v[2, 2] - 2 * v[8, 2]
    assert_allclose(difference, np.var(draws[:, 8] - draws[:, 2], ddof=1), rtol=1e-6)
    summary = result.summary()
    assert "[q20]" in summary and "q80:x2" in summary and "Simultaneous" in summary
    assert "q50:x1" in result.to_latex()
    assert result.predictions == []


def test_sqreg_default_quantiles_and_labels(data):
    result = oe.sqreg(data=data, y="y", x=X_COLS, reps=4, seed=1)
    assert result.extra["equations"] == ["q25", "q50", "q75"]
    odd = oe.sqreg(data=data, y="y", x=X_COLS, quantiles=[0.025, 0.333], reps=4, seed=1)
    assert odd.extra["equations"] == ["q2_5", "q33_3"]
    single = oe.sqreg(data=data, y="y", x=X_COLS, quantiles=[0.5], reps=6, seed=9)
    same = oe.bsqreg(data=data, y="y", x=X_COLS, quantile=0.5, reps=6, seed=9)
    assert_allclose(covariance(single), covariance(same), rtol=0, atol=0)


def test_iqreg_is_the_difference_of_two_quantile_regressions(data):
    result = oe.iqreg(data=data, y="y", x=X_COLS, quantiles=[0.25, 0.75], reps=14, seed=31)
    x, y = design(data), data.y.to_numpy()
    low, high = lp_quantile(x, y, 0.25), lp_quantile(x, y, 0.75)
    assert [c.term for c in result.coefficients] == ["Intercept", *X_COLS]
    assert_allclose(params(result), high - low, rtol=1e-8, atol=1e-9)
    draws = replicate_draws(data, [0.25, 0.75], 14, 31)
    assert_allclose(covariance(result), np.cov((draws[:, 1] - draws[:, 0]).T, ddof=1),
                    rtol=1e-6, atol=1e-12)
    assert result.metrics["quantile_low"] == 0.25 and result.metrics["quantile_high"] == 0.75
    assert_allclose(list(result.extra["coefficients_low"].values()), low, rtol=1e-8, atol=1e-9)
    assert_allclose(list(result.extra["coefficients_high"].values()), high, rtol=1e-8, atol=1e-9)
    for side, tau in (("low", 0.25), ("high", 0.75)):
        point = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau)
        assert_allclose(result.metrics[f"pseudo_r_squared_{side}"],
                        point.metrics["pseudo_r_squared"], rtol=1e-12)
    default = oe.iqreg(data=data, y="y", x=X_COLS, reps=14, seed=31)
    assert_allclose(covariance(default), covariance(result), rtol=0, atol=0)
    assert "Interquantile" in result.summary()
    assert result.inference["df_inference"] == len(data) - 3


def test_cluster_bootstrap_resamples_whole_clusters(data):
    result = oe.bsqreg(data=data, y="y", x=X_COLS, quantile=0.5, reps=12, seed=8, cluster="g")
    draws = replicate_draws(data, [0.5], 12, 8, cluster="g")[:, 0]
    assert_allclose(covariance(result), np.cov(draws.T, ddof=1), rtol=1e-6, atol=1e-12)
    record = result.extra["bootstrap"]
    assert record["resampling"] == "clusters" and record["cluster_count"] == data.g.nunique()
    assert result.inference["cluster_count"] == data.g.nunique()
    joint = oe.sqreg(data=data, y="y", x=X_COLS, quantiles=[0.3, 0.7], reps=6, seed=8, cluster="g")
    draws = replicate_draws(data, [0.3, 0.7], 6, 8, cluster="g").reshape(6, -1)
    assert_allclose(covariance(joint), np.cov(draws.T, ddof=1), rtol=1e-6, atol=1e-12)
    spread = oe.iqreg(data=data, y="y", x=X_COLS, quantiles=[0.3, 0.7], reps=6, seed=8, cluster="g")
    draws = draws.reshape(6, 2, 3)
    assert_allclose(covariance(spread), np.cov((draws[:, 1] - draws[:, 0]).T, ddof=1),
                    rtol=1e-6, atol=1e-12)


def test_failed_resamples_are_skipped_and_reported():
    rng = np.random.default_rng(4)
    n = 40
    frame = pd.DataFrame({"x1": rng.normal(size=n), "rare": np.r_[1.0, np.zeros(n - 1)]})
    frame["y"] = frame.x1 + rng.normal(size=n)
    result = oe.bsqreg(data=frame, y="y", x=["x1", "rare"], reps=40, seed=2)
    record = result.extra["bootstrap"]
    # The single rare == 1 row is absent from about 1/e of the resamples.
    assert 2 <= record["reps_used"] < 40 and result.metrics["reps"] == record["reps_used"]
    assert any("could not be estimated" in warning for warning in result.warnings)
    # Explicit oracle: skip the same resamples.
    x = np.column_stack([np.ones(n), frame.x1, frame.rare])
    generator = torch.Generator().manual_seed(2)
    kept = []
    for _ in range(40):
        rows = torch.randint(n, (n,), generator=generator).numpy()
        if 0 in rows:
            kept.append(lp_quantile(x[rows], frame.y.to_numpy()[rows], 0.5))
    assert len(kept) == record["reps_used"]
    assert_allclose(covariance(result), np.cov(np.array(kept).T, ddof=1), rtol=1e-6, atol=1e-12)


def test_bootstrap_design_handling(data):
    frame = data.assign(twice=2 * data.x1)
    frame.loc[[2, 9], "x2"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.sqreg(data=frame, y="y", x=["x1", "twice", "x2"], reps=4, seed=1)
    assert caught.value.code == "missing_values"
    result = oe.sqreg(data=frame, y="y", x=["x1", "twice", "x2"], quantiles=[0.4, 0.6], reps=5,
                      seed=1, missing="drop")
    assert result.nobs == len(data) - 2 and result.dropped_rows == 2
    assert result.provenance["omitted_terms"] == ["twice"]
    complete = data.drop(index=[2, 9]).reset_index(drop=True)
    reference = oe.sqreg(data=complete, y="y", x=X_COLS, quantiles=[0.4, 0.6], reps=5, seed=1)
    assert_allclose(params(result), params(reference), rtol=1e-10)
    assert_allclose(covariance(result), covariance(reference), rtol=1e-8)
    no_constant = oe.iqreg(data=data, y="y", x=X_COLS, intercept=False, reps=4, seed=1)
    assert [c.term for c in no_constant.coefficients] == X_COLS
    categorical = oe.bsqreg(data=data.assign(k=np.where(data.x1 > 0, "hi", "lo")), y="y",
                            x=["x2", "k"], categorical=["k"], reps=4, seed=1)
    assert [c.term for c in categorical.coefficients] == ["Intercept", "x2", "k[lo]"]


def test_bootstrap_error_codes(data):
    cases = [
        (oe.bsqreg, {"reps": 1}, "invalid_spec"),
        (oe.bsqreg, {"reps": 2.5}, "invalid_spec"),
        (oe.bsqreg, {"seed": -1}, "invalid_spec"),
        (oe.bsqreg, {"quantile": 1.0}, "invalid_quantile"),
        (oe.sqreg, {"quantiles": [0.25, 1.2]}, "invalid_quantile"),
        (oe.sqreg, {"quantiles": [0.25, 0.25]}, "invalid_quantile"),
        (oe.sqreg, {"quantiles": []}, "invalid_quantile"),
        (oe.sqreg, {"quantiles": 0.5}, "invalid_quantile"),
        (oe.sqreg, {"quantiles": ["a"]}, "invalid_quantile"),
        (oe.iqreg, {"quantiles": [0.75, 0.25]}, "invalid_quantile"),
        (oe.iqreg, {"quantiles": [0.25, 0.5, 0.75]}, "invalid_quantile"),
        (oe.iqreg, {"quantiles": "0.25 0.75"}, "invalid_quantile"),
        (oe.sqreg, {"x": ["x1", "absent"]}, "missing_columns"),
    ]
    for function, options, code in cases:
        arguments = {"data": data, "y": "y", "x": X_COLS, **options}
        with pytest.raises(AnalysisError) as caught:
            function(**arguments)
        assert caught.value.code == code, options
    for name in ("bsqreg", "sqreg", "iqreg"):
        with pytest.raises(Exception) as caught:
            ModelSpec(estimator=name, outcome="y", predictors=X_COLS, weights="g",
                      weight_type="fweight")
        assert "does not support fweights" in str(caught.value)
        with pytest.raises(Exception) as caught:
            ModelSpec(estimator=name, outcome="y", predictors=X_COLS, covariance="nonrobust")
        assert "bootstrap" in str(caught.value)
    exact = data.assign(y=2 - data.x1)
    with pytest.raises(AnalysisError) as caught:
        oe.bsqreg(data=exact, y="y", x=X_COLS, reps=4)
    assert caught.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as caught:
        oe.sqreg(data=data.iloc[:3], y="y", x=X_COLS, reps=4)
    assert caught.value.code == "insufficient_observations"


@pytest.mark.parametrize("function, options", [
    (oe.bsqreg, {"quantile": 0.4}), (oe.sqreg, {"quantiles": [0.3, 0.6]}), (oe.iqreg, {}),
])
def test_bootstrap_round_trip_and_public_api(data, function, options):
    result = function(data=data, y="y", x=X_COLS, reps=6, seed=3, **options)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    json.dumps(result.model_dump())
    assert result.summary() and "tabular" in result.to_latex()
    name = function.__name__
    again = oe.fit(ModelSpec(estimator=name, outcome="y", predictors=X_COLS,
                             options=result.spec.options), data=data)
    assert again.covariance_matrix == result.covariance_matrix
    listing = oe.capabilities()["estimators"][name]
    assert listing["family"] == "quantile" and listing["covariances"] == ["bootstrap"]
    assert listing["weights"] == [] and listing["stata"] == [name]
    assert result.provenance["stata_parity_validated"] is False
