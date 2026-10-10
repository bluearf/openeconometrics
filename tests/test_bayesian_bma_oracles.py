"""Independent exact mixture geometry, credible atoms and joint predictive laws."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian import (
    bma as b,
    bma_kernels as k,
    bma_query as q,
    bma_draws as d,
)
from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior as Prior


def inputs(n=12):
    data = pd.DataFrame({"x1": np.linspace(-1, 1, n), "x2": np.cos(np.arange(n) * 0.6)})
    data["y"] = 0.3 + 0.7 * data.x1 - 0.4 * data.x2 + np.sin(np.arange(n) * 1.7) * 0.3
    priors = []
    for mask in range(4):
        ix = [0, *[j + 1 for j in range(2) if mask & (1 << j)]]
        mean = np.array([0.2, 0.1, -0.2])[ix]
        V = np.diag([1.0, 2.0, 0.8])[np.ix_(ix, ix)]
        priors.append(
            Prior(
                mean=mean.tolist(),
                scale_matrix=V.tolist(),
                shape=2.3 + 0.2 * mask,
                scale=1.7 + 0.1 * mask,
            )
        )
    return data, priors, [2.0, 1.0, 3.0, 4.0]


def fit(n=12):
    data, prior, odds = inputs(n)
    return b.bayes_bma(data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds)


def independent(data, priors, odds):
    full = np.c_[np.ones(len(data)), data[["x1", "x2"]].to_numpy()]
    y = data.y.to_numpy()
    out = []
    logs = []
    for mask, p in enumerate(priors):
        ix = [0, *[j + 1 for j in range(2) if mask & (1 << j)]]
        X = full[:, ix]
        m = np.array(p.mean)
        V = np.array(p.scale_matrix)
        # Observation-space covariance/evidence is independent from parameter precision code.
        A = np.eye(len(y)) + X @ V @ X.T
        res = y - X @ m
        a = p.shape + len(y) / 2
        bb = p.scale + 0.5 * res @ np.linalg.solve(A, res)
        vn = V - V @ X.T @ np.linalg.solve(A, X @ V)
        mn = m + V @ X.T @ np.linalg.solve(A, res)
        location = np.zeros(3)
        location[ix] = mn
        C = np.zeros((3, 3))
        C[np.ix_(ix, ix)] = vn * bb / (a - 1)
        log = (
            -len(y) / 2 * np.log(2 * np.pi)
            - np.linalg.slogdet(A)[1] / 2
            + special.gammaln(a)
            - special.gammaln(p.shape)
            + p.shape * np.log(p.scale)
            - a * np.log(bb)
        )
        out.append(
            dict(
                ix=ix,
                location=location,
                C=C,
                a=a,
                b=bb,
                scale=np.sqrt(np.diag(np.pad(np.zeros((0, 0)), ((0, 0), (0, 0)))))
                if False
                else None,
                vn=vn,
            )
        )
        logs.append(log)
    logpost = np.array(logs) + np.log(odds)
    w = np.exp(logpost - special.logsumexp(logpost))
    return out, w, logs


def test_observation_space_full_weights_inclusion_beta_sigma_and_quantiles():
    data, prior, odds = inputs()
    result = fit()
    r = result.payload["results"]
    ref, w, logs = independent(data, prior, odds)
    np.testing.assert_allclose(r["log_model_evidence"], logs, rtol=0, atol=2e-12)
    np.testing.assert_allclose(r["model_weights"], w, rtol=2e-12, atol=0)
    means = np.array([v["location"] for v in ref])
    mean = w @ means
    within = sum(ww * v["C"] for ww, v in zip(w, ref))
    between = sum(
        ww * np.outer(v["location"] - mean, v["location"] - mean) for ww, v in zip(w, ref)
    )
    np.testing.assert_allclose(r["coefficient_mean"], mean, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(r["within_coefficient_covariance"], within, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(r["between_coefficient_covariance"], between, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        r["coefficient_covariance"], within + between, rtol=2e-12, atol=2e-13
    )
    sigma = np.array([v["b"] / (v["a"] - 1) for v in ref])
    fullmeans = np.c_[means, sigma]
    meanfull = w @ fullmeans
    Cfull = sum(ww * np.outer(mm - meanfull, mm - meanfull) for ww, mm in zip(w, fullmeans))
    Cfull[:3, :3] += within
    Cfull[-1, -1] += sum(
        ww * v["b"] ** 2 / ((v["a"] - 1) ** 2 * (v["a"] - 2)) for ww, v in zip(w, ref)
    )
    np.testing.assert_allclose(r["joint_covariance"], Cfull, rtol=2e-12, atol=2e-13)
    assert abs(Cfull[0, -1]) > 1e-6
    for j in range(3):

        def cf(x, left=False):
            return sum(
                ww
                * (
                    float(x > 0 if left else x >= 0)
                    if j not in v["ix"]
                    else stats.t.cdf(
                        (x - v["location"][j])
                        / math.sqrt(v["vn"][v["ix"].index(j), v["ix"].index(j)] * v["b"] / v["a"]),
                        2 * v["a"],
                    )
                )
                for ww, v in zip(w, ref)
            )

        for pp, value in zip([0.025, 0.975], r["coefficient_quantiles"][j]):
            if cf(0, True) <= pp <= cf(0):
                expected = 0.0
            else:
                expected = optimize.brentq(lambda x: cf(x) - pp, -100, 100, xtol=1e-13)
            assert value == pytest.approx(expected, rel=3e-11, abs=2e-12)
        assert r["coefficient_atom_zero"][j] == pytest.approx(
            sum(ww for ww, v in zip(w, ref) if j not in v["ix"]), abs=2e-15
        )


def test_full_cross_query_C_parameter_query_C_and_mixture_endpoints():
    data, prior, odds = inputs()
    result = fit()
    query = pd.DataFrame(
        {"x1": [-0.8, 0.3, 1.1], "x2": [0.4, -0.2, 0.7]},
        index=pd.Index(["dup", "dup", "last"], name="patient"),
    )
    prediction = q.bayes_bma_predict(result, data=query)
    r = prediction.payload["results"]
    ref, w, _ = independent(data, prior, odds)
    Z = np.c_[np.ones(3), query.to_numpy()]
    means = np.array([Z @ v["location"] for v in ref])
    mu = w @ means
    within = sum(ww * Z @ v["C"] @ Z.T for ww, v in zip(w, ref))
    between = sum(ww * np.outer(m - mu, m - mu) for ww, m in zip(w, means))
    cm = within + between
    noise = sum(ww * v["b"] / (v["a"] - 1) for ww, v in zip(w, ref))
    np.testing.assert_allclose(r["mean_covariance"], cm, rtol=3e-12, atol=2e-13)
    np.testing.assert_allclose(
        r["outcome_covariance"], cm + noise * np.eye(3), rtol=3e-12, atol=2e-13
    )
    J = np.array(result.payload["results"]["joint_covariance"])
    np.testing.assert_allclose(
        np.array(r["parameter_query_covariance"])[:3], J[:3, :3] @ Z.T, rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        np.array(r["parameter_query_covariance"])[-1], J[-1, :3] @ Z.T, rtol=3e-12, atol=2e-13
    )
    for j in range(3):
        for target in ("mean", "outcome"):

            def cf(x):
                return sum(
                    ww
                    * stats.t.cdf(
                        (x - mm[j])
                        / math.sqrt(
                            v["b"]
                            / v["a"]
                            * (Z[j, v["ix"]] @ v["vn"] @ Z[j, v["ix"]] + (target == "outcome"))
                        ),
                        2 * v["a"],
                    )
                    for ww, v, mm in zip(w, ref, means)
                )

            for pp, value in zip([0.025, 0.975], r[target + "_quantiles"][j]):
                assert cf(value) == pytest.approx(pp, abs=2e-12)
    pd.testing.assert_index_equal(prediction.summary().index, query.index)


def test_exact_spike_quantile_and_genuine_between_model_uncertainty():
    laws = [(0.0, 0.0, 6.0), (3.0, 1.0, 6.0)]
    weights = [0.96, 0.04]
    assert k.quantile(0.025, laws, weights) == 0.0
    assert k.quantile(0.5, laws, weights) == 0.0
    left = k.cdf(0, laws, weights, left=True)
    right = k.cdf(0, laws, weights)
    assert right - left == pytest.approx(0.96, abs=2e-15)
    assert k.quantile(left, laws, weights) == 0.0
    assert k.quantile(0.99, laws, weights) > 0


@pytest.mark.parametrize(
    "shape,n,mean_exists,var_exists,sigmavar",
    [
        (0.01, 1, True, False, False),
        (0.01, 2, True, True, False),
        (0.01, 3, True, True, False),
        (0.01, 4, True, True, True),
        (1e-300, 1, True, False, False),
        (1e-300, 2, True, True, False),
    ],
)
def test_increment_moments_truthful(shape, n, mean_exists, var_exists, sigmavar):
    # beta first moments exist for every n>=1 and proper a0>0, even if storeddf rounds1.
    prior = Prior(mean=[0.0], scale_matrix=[[1e-20]], shape=shape, scale=1e-20)
    result = b.bayes_bma(
        data=pd.DataFrame({"y": np.zeros(n)}),
        y="y",
        optional=[],
        priors=[prior],
        model_prior_odds=[1.0],
    )
    rr = result.payload["results"]
    assert rr["mean_exists"] == (mean_exists,)
    assert rr["variance_exists"] == (var_exists,)
    assert (rr["joint_covariance"] is not None) == sigmavar
    assert (rr["coefficient_covariance"] is not None) == var_exists


def test_tiny_positive_supported_weight_cannot_erase_infinite_covariance():
    data = pd.DataFrame({"y": [0.0], "x": [0.0]})
    priors = [
        Prior(mean=[0.0], scale_matrix=[[1.0]], shape=0.01, scale=1.0),
        Prior(mean=[0.0, 0.0], scale_matrix=[[1.0, 0.0], [0.0, 1.0]], shape=3.0, scale=1.0),
    ]
    result = b.bayes_bma(
        data=data, y="y", optional=["x"], priors=priors, model_prior_odds=[1e-250, 1.0]
    )
    r = result.payload["results"]
    assert 0 < r["model_weights"][0] < 1e-200
    assert r["variance_exists"] == (False, True)
    assert r["coefficient_covariance"] is None
    assert r["coefficient_variance"][1] is not None
    assert r["variance_mean"] is None


def test_singleton_reduces_to_exact_core_and_collinear_proper_prior_remains_valid():
    data, prior, _ = inputs()
    p = prior[-1]
    result = b.bayes_bma(
        data=data, y="y", forced=["x1", "x2"], optional=[], priors=[p], model_prior_odds=[19.0]
    )
    np.testing.assert_allclose(
        result.payload["results"]["coefficient_covariance"],
        result.payload["components"][0]["covariance"],
        rtol=0,
        atol=0,
    )
    assert result.payload["results"]["model_weights"] == (1.0,)
    data["x2"] = data.x1
    assert fit().payload["results"]["model_weights"]
    assert b.bayes_bma(
        data=data, y="y", optional=["x1", "x2"], priors=inputs()[1], model_prior_odds=[1.0] * 4
    ).payload["results"]["model_weights"]


def test_units_permutation_and_forced_block_basis_preserve_posterior():
    data, p, odds = inputs()
    base = fit()
    scaled = data.copy()
    scaled.x1 *= 1e-8
    scaled.x2 *= -1e4
    new = []
    for mask, old in enumerate(p):
        ix = [0, *[j + 1 for j in range(2) if mask & (1 << j)]]
        D = np.diag(np.array([1.0, 1e8, -1e-4])[ix])
        new.append(
            Prior(
                mean=(D @ np.array(old.mean)).tolist(),
                scale_matrix=(D @ np.array(old.scale_matrix) @ D.T).tolist(),
                shape=old.shape,
                scale=old.scale,
            )
        )
    r = b.bayes_bma(data=scaled, y="y", optional=["x1", "x2"], priors=new, model_prior_odds=odds)
    np.testing.assert_allclose(
        r.payload["results"]["model_weights"],
        base.payload["results"]["model_weights"],
        rtol=2e-11,
        atol=0,
    )
    D = np.diag([1.0, 1e8, -1e-4])
    np.testing.assert_allclose(
        r.payload["results"]["coefficient_covariance"],
        D @ np.array(base.payload["results"]["coefficient_covariance"]) @ D.T,
        rtol=3e-11,
        atol=0,
    )
    # A nonorthogonal forced-block basis keeps the same space; arbitrary optional mixing does not.
    X = data[["x1", "x2"]].to_numpy()
    A = np.array([[2.0, 0.3], [-0.4, 1.1]])
    inv = np.linalg.inv(A)
    transformed = data.copy()
    transformed[["x1", "x2"]] = X @ A
    P = np.eye(3)
    P[1:, 1:] = inv
    old = p[-1]
    mapped_scale = P @ np.array(old.scale_matrix) @ P.T
    mapped_scale = (mapped_scale + mapped_scale.T) / 2
    prior = Prior(
        mean=(P @ np.array(old.mean)).tolist(),
        scale_matrix=mapped_scale.tolist(),
        shape=old.shape,
        scale=old.scale,
    )
    original = b.bayes_bma(
        data=data, y="y", forced=["x1", "x2"], optional=[], priors=[old], model_prior_odds=[1.0]
    )
    mapped = b.bayes_bma(
        data=transformed,
        y="y",
        forced=["x1", "x2"],
        optional=[],
        priors=[prior],
        model_prior_odds=[1.0],
    )
    np.testing.assert_allclose(
        mapped.payload["results"]["coefficient_mean"],
        P @ np.array(original.payload["results"]["coefficient_mean"]),
        rtol=3e-11,
        atol=2e-13,
    )


def test_draw_law_model_frequency_and_full_joint_query_covariance():
    result = fit()
    query = pd.DataFrame({"x1": [-0.5, 0.7], "x2": [0.2, -0.6]})
    prediction = q.bayes_bma_predict(result, data=query)
    draws = d.bayes_bma_draws(result, draws=10000, seed=190690, data=query)
    r = draws.payload["results"]
    counts = np.bincount(r["model_mask"], minlength=4) / 10000
    np.testing.assert_allclose(
        counts, result.payload["results"]["model_weights"], rtol=0, atol=0.016
    )
    beta = np.array(r["beta"])
    mask = np.array(r["model_mask"])
    assert np.all(beta[(mask & 1) == 0, 1] == 0)
    assert np.all(beta[(mask & 2) == 0, 2] == 0)
    np.testing.assert_allclose(
        np.cov(np.array(r["mean_draws"]), rowvar=False, ddof=0),
        prediction.payload["results"]["mean_covariance"],
        rtol=0.10,
        atol=0.001,
    )
    np.testing.assert_allclose(
        np.cov(np.array(r["outcome_draws"]), rowvar=False, ddof=0),
        prediction.payload["results"]["outcome_covariance"],
        rtol=0.10,
        atol=0.01,
    )


def test_unrepresentable_positive_model_mass_refuses_without_truncation():
    with pytest.raises(AnalysisError, match="underflow"):
        k.normalize([0.0, -1000.0])


def test_inverse_gamma_tiny_increment_full_joint_variance_against_300_digit_law():
    import mpmath as mp

    prior = Prior(mean=[0.0], scale_matrix=[[1.0]], shape=1e-300, scale=1e-200)
    data = pd.DataFrame({"y": [0.0] * 4})
    result = b.bayes_bma(data=data, y="y", optional=[], priors=[prior], model_prior_odds=[1.0])
    with mp.workdps(350):
        a = mp.mpf(prior.shape) + 2
        bn = mp.mpf(prior.scale)
        expected = float(bn**2 / ((a - 1) ** 2 * (a - 2)))
    actual = result.payload["components"][0]["sigma_variance"]
    assert actual == pytest.approx(expected, rel=4e-15, abs=0)
    assert result.payload["results"]["joint_covariance"][-1][-1] == actual
    assert actual > 0


def test_nonrepresentable_positive_inverse_gamma_variance_typed_refuses():
    prior = Prior(mean=[0.0], scale_matrix=[[1.0]], shape=3.0, scale=1e-200)
    with pytest.raises(AnalysisError, match="not float64 representable"):
        b.bayes_bma(
            data=pd.DataFrame({"y": [0.0, 0.0]}),
            y="y",
            optional=[],
            priors=[prior],
            model_prior_odds=[1.0],
        )


def test_positive_low_weight_coefficient_variance_underflow_refuses():
    priors = [
        Prior(mean=[0.0], scale_matrix=[[1.0]], shape=3.0, scale=1e-160),
        Prior(mean=[0.0, 0.0], scale_matrix=[[1.0, 0.0], [0.0, 1e-160]], shape=3.0, scale=1e-160),
    ]
    with pytest.raises(AnalysisError, match="positive mixture coefficient variance"):
        b.bayes_bma(
            data=pd.DataFrame({"y": [0.0, 0.0], "x": [0.0, 0.0]}),
            y="y",
            optional=["x"],
            priors=priors,
            model_prior_odds=[1.0, 1e-200],
        )


def test_unresolved_continuous_quantile_never_spoofs_a_point_interval():
    with pytest.raises(AnalysisError, match="resolve its CDF target"):
        k.quantile(0.025, [(1e20, 1.0, 8.0)], [1.0])
    # Genuine omitted-coordinate atoms retain generalized-inverse limits.
    assert k.quantile(0.025, [(0.0, 0.0, 8.0)], [1.0]) == 0.0


@pytest.mark.parametrize("shape,n,scale", [(1e12, 1, 1e12), (1e-300, 2, 1.0), (1e-300, 4, 1e-200)])
def test_inverse_gamma_mixture_endpoints_independent_at_extreme_shape(shape, n, scale):
    prior = Prior(mean=[0.0], scale_matrix=[[1.0]], shape=shape, scale=scale)
    result = b.bayes_bma(
        data=pd.DataFrame({"y": [0.0] * n}),
        y="y",
        optional=[],
        priors=[prior],
        model_prior_odds=[1.0],
    )
    posterior = result.payload["components"][0]["posterior"]
    for pp, value in zip([0.025, 0.975], result.payload["results"]["sigma2_quantiles"]):
        assert stats.invgamma.cdf(
            value, posterior["shape"], scale=posterior["scale"]
        ) == pytest.approx(pp, abs=2e-10)
