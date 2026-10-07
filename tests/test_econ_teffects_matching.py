"""Brute-force oracles for teffects nnmatch and psmatch.

Matched sets are found by explicit O(n^2) NumPy distance matrices (ties: every
unit at the M-th distance), potential outcomes are imputed and the
Abadie-Imbens variances (2006; 2016 propensity-score correction) are coded
directly from their formulas; the treatment model of psmatch comes from
statsmodels. A small simulation checks that the standard errors track the
sampling spread of the estimators.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.teffects import neighbors
from openecon.models import ResultBundle


def make_data(seed=5, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.4 * x1 - 0.6 * x2)))
    d = (rng.uniform(size=n) < p).astype(int)
    y = 1 + 2 * d + x1 + 0.5 * x2 + 0.8 * d * x1 + rng.normal(size=n) * (1 + 0.3 * np.abs(x1))
    frame = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    frame["k1"] = rng.integers(0, 4, size=n).astype(float)      # discrete: many ties
    frame["k2"] = rng.integers(0, 3, size=n).astype(float)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def sets(query, pool, m, own=False):
    dist = ((query[:, None, :] - pool[None, :, :]) ** 2).sum(-1)
    if own:
        np.fill_diagonal(dist, np.inf)
    kth = np.sort(dist, axis=1)[:, m - 1]
    return dist <= kth[:, None], np.sqrt(kth)


def wls(x, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]


def oracle(frame, coords, estimand, m=1, nn=2, biasadj=None, both=False):
    y, d = frame.y.to_numpy(), frame.d.to_numpy()
    n = len(y)
    groups = [np.flatnonzero(d == 0), np.flatnonzero(d == 1)]
    imputed = np.column_stack([y, y]).astype(float)
    usage, usage2 = np.zeros(n), np.zeros(n)
    xb = None if biasadj is None else sm.add_constant(frame[biasadj].to_numpy())
    directions = [1] if estimand == "atet" and not both else [1, 0]
    masks = {}
    for level in directions:
        q, p = groups[level], groups[1 - level]
        mask, _ = sets(coords[q], coords[p], m)
        masks[level] = mask
        size = mask.sum(1)
        usage[p] += (mask / size[:, None]).sum(0)
        usage2[p] += (mask / size[:, None] ** 2).sum(0)
    for level in directions:
        q, p = groups[level], groups[1 - level]
        mask = masks[level]
        size = mask.sum(1)
        values = (mask * y[p]).sum(1) / size
        if xb is not None:
            beta = wls(xb[p], y[p], usage[p])
            mean_x = (mask @ xb[p]) / size[:, None]
            values = values + (xb[q] - mean_x) @ beta
        imputed[q, 1 - level] = values
    sigma2 = np.zeros(n)
    for level in (0, 1):
        g = groups[level]
        mask, _ = sets(coords[g], coords[g], nn, own=True)
        size = mask.sum(1)
        sigma2[g] = size / (size + 1) * (y[g] - (mask * y[g]).sum(1) / size) ** 2
    effect = imputed[:, 1] - imputed[:, 0]
    if estimand == "ate":
        tau = effect.mean()
        var = (np.sum((effect - tau) ** 2)
               + np.sum((usage ** 2 + 2 * usage - usage2) * sigma2)) / n ** 2
    else:
        t, c = groups[1], groups[0]
        tau = effect[t].mean()
        var = (np.sum((effect[t] - tau) ** 2)
               + np.sum((usage[c] ** 2 - usage2[c]) * sigma2[c])) / len(t) ** 2
    return tau, var, effect


def test_nnmatch_matches_brute_force_with_mahalanobis_metric(data):
    x = data[["x1", "x2"]].to_numpy()
    chol = np.linalg.cholesky(np.cov(x, rowvar=False))
    coords = np.linalg.solve(chol, x.T).T
    for estimand in ("ate", "atet"):
        for m in (1, 3):
            result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"],
                                 method="nnmatch", estimand=estimand, neighbors=m)
            tau, var, _ = oracle(data, coords, estimand, m=m)
            assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
            assert_allclose(result.coefficients[0].std_error, np.sqrt(var), rtol=1e-8)
            assert result.coefficients[0].term == f"{estimand.upper()}:r1vs0.d"


def test_metrics_bias_adjustment_and_ties(data):
    x = data[["x1", "x2"]].to_numpy()
    for metric, coords in (("ivariance", x / x.std(0, ddof=1)), ("euclidean", x)):
        result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="nnmatch",
                             metric=metric, neighbors=2, biasadj=["x1", "x2"], vce_neighbors=3)
        tau, var, _ = oracle(data, coords, "ate", m=2, nn=3, biasadj=["x1", "x2"])
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
        assert_allclose(result.coefficients[0].std_error, np.sqrt(var), rtol=1e-8)
    # integer covariates: matched sets with ties hold more than `neighbors` units
    coords = data[["k1", "k2"]].to_numpy()
    for estimand in ("ate", "atet"):
        result = oe.teffects(data=data, y="y", treatment="d", x=["k1", "k2"], method="nnmatch",
                             metric="euclidean", estimand=estimand, biasadj=["x1"])
        tau, var, _ = oracle(data, coords, estimand, biasadj=["x1"])
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
        assert_allclose(result.coefficients[0].std_error, np.sqrt(var), rtol=1e-8)
        assert result.extra["matches"]["max"] > 10
        assert result.extra["matches"]["units_with_ties"] > 0


def psmatch_oracle(frame, estimand, m=1, nn=2, biasadj=None, probit=False):
    X = sm.add_constant(frame[["x1", "x2"]].to_numpy())
    d = frame.d.to_numpy()
    model = (sm.Probit if probit else sm.Logit)(d, X).fit(disp=0, tol=1e-12, method="newton")
    p = model.predict(X)
    eta = X @ model.params
    f = (stats.norm.pdf(eta) if probit else p * (1 - p))
    v_gamma = model.cov_params()
    coords = p[:, None]
    # psmatch imputes both potential outcomes of every unit (needed by the ATET correction)
    tau, var, effect = oracle(frame, coords, estimand, m=m, nn=nn, biasadj=biasadj, both=True)
    y = frame.y.to_numpy()
    n = len(y)
    groups = [np.flatnonzero(d == 0), np.flatnonzero(d == 1)]
    cov = np.zeros((2, n, X.shape[1]))
    for w in (0, 1):
        g = groups[w]
        for level in (0, 1):
            q = groups[level]
            mask, _ = sets(coords[q], coords[g], nn, own=level == w)
            for row, i in enumerate(q):
                members = g[mask[row]]
                xm, ym = X[members], y[members]
                centred = (xm - xm.mean(0)) * (ym - ym.mean())[:, None]
                cov[w, i] = centred.sum(0) / (len(members) - 1)
    if estimand == "ate":
        c = (f[:, None] * (cov[1] / p[:, None] + cov[0] / (1 - p)[:, None])).sum(0) / n
        adj = c @ v_gamma @ c
    else:
        n1 = d.sum()
        c = (f[:, None] * (cov[1] + (p / (1 - p))[:, None] * cov[0])).sum(0) / n1
        dd = (f * (effect - tau)) @ X / n1
        adj = c @ v_gamma @ c - dd @ v_gamma @ dd
    return tau, var, adj


@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_psmatch_matches_abadie_imbens_2016(data, estimand):
    for tmodel, probit in (("logit", False), ("probit", True)):
        result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="psmatch",
                             estimand=estimand, tmodel=tmodel, neighbors=2)
        tau, var, adj = psmatch_oracle(data, estimand, m=2, probit=probit)
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-8)
        assert_allclose(result.extra["variance"]["abadie_imbens"], var, rtol=1e-6)
        assert_allclose(result.extra["variance"]["propensity_adjustment"], adj, rtol=1e-5)
        assert_allclose(result.coefficients[0].std_error, np.sqrt(var - adj), rtol=1e-6)
    adjusted = oe.teffects(data=data, y="y", treatment="d", tx=["x1", "x2"], method="psmatch",
                           estimand=estimand, biasadj=["x1"])
    tau, var, adj = psmatch_oracle(data, estimand, biasadj=["x1"])
    assert_allclose(adjusted.coefficients[0].estimate, tau, rtol=1e-8)
    assert_allclose(adjusted.coefficients[0].std_error, np.sqrt(var - adj), rtol=1e-6)


def test_sorted_search_equals_brute_force_with_ties():
    rng = np.random.default_rng(0)
    pool = np.sort(np.round(rng.uniform(size=300), 2))          # many exact ties
    query = np.round(rng.uniform(-0.1, 1.1, size=500), 2)
    for m in (1, 2, 5):
        found = neighbors.match_sorted(torch.tensor(query), torch.tensor(pool),
                                       torch.tensor(pool)[:, None], m)
        dist = np.abs(query[:, None] - pool[None, :])
        kth = np.sort(dist, axis=1)[:, m - 1]
        mask = dist <= kth[:, None]
        assert_allclose(found.count.numpy(), mask.sum(1))
        assert_allclose(found.sums[:, 0].numpy(), (mask * pool).sum(1), rtol=1e-12)
        assert_allclose(found.usage.numpy(), (mask / mask.sum(1)[:, None]).sum(0), rtol=1e-12)
        own = neighbors.match_sorted(torch.tensor(pool), torch.tensor(pool),
                                     torch.tensor(pool)[:, None], m,
                                     own=torch.arange(len(pool)))
        dist = np.abs(pool[:, None] - pool[None, :])
        np.fill_diagonal(dist, np.inf)
        kth = np.sort(dist, axis=1)[:, m - 1]
        mask = dist <= kth[:, None]
        assert_allclose(own.count.numpy(), mask.sum(1))
        assert_allclose(own.sums[:, 0].numpy(), (mask * pool).sum(1), rtol=1e-12, atol=1e-12)
        blocks = neighbors.match_blocks(torch.tensor(pool)[:, None], torch.tensor(pool)[:, None],
                                        torch.tensor(pool)[:, None], m,
                                        own=torch.arange(len(pool)))
        assert_allclose(blocks.count.numpy(), mask.sum(1))


def test_caliper_and_failure_contract(data):
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="nnmatch",
                    caliper=1e-4)
    assert caught.value.code == "caliper_violation"
    wide = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="nnmatch",
                       caliper=50.0)
    plain = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="nnmatch")
    assert wide.coefficients[0].estimate == plain.coefficients[0].estimate
    cases = [
        ({"method": "nnmatch", "estimand": "pomeans"}, "unsupported_estimand"),
        ({"method": "nnmatch", "covariance": "cluster", "cluster": "k1"},
         "unsupported_covariance"),
        ({"method": "nnmatch", "weights": "k1", "weight_type": "pweight"}, "unsupported_weights"),
        ({"method": "nnmatch", "tmodel": "probit"}, "invalid_spec"),
        ({"method": "nnmatch", "omodel": "logit"}, "invalid_spec"),
        ({"method": "psmatch", "metric": "euclidean"}, "invalid_spec"),
        ({"method": "psmatch", "vce_neighbors": 1}, "invalid_spec"),
        ({"method": "ra", "biasadj": ["x1"]}, "invalid_spec"),
        ({"method": "nnmatch", "x": []}, "invalid_spec"),
        ({"method": "psmatch", "x": ["one"]}, "invalid_spec"),
    ]
    for update, code in cases:
        arguments = {"data": data.assign(one=1.0), "y": "y", "treatment": "d",
                     "x": ["x1", "x2"], **update}
        with pytest.raises(AnalysisError) as caught:
            oe.teffects(**arguments)
        assert caught.value.code == code, update
    three = data.assign(d=data.d + (data.x1 > 1.5))
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=three, y="y", treatment="d", x=["x1", "x2"], method="nnmatch")
    assert caught.value.code == "invalid_treatment"
    tiny = data.iloc[:6].assign(d=[0, 1, 1, 1, 1, 1])
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=tiny, y="y", treatment="d", x=["x1"], method="nnmatch", neighbors=2)
    assert caught.value.code == "insufficient_observations"


def test_matching_standard_errors_track_sampling_spread():
    rng = np.random.default_rng(11)
    estimates, errors = [], []
    for _ in range(120):
        n = 800
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.7 * x2)))).astype(int)
        y = 1 + 2 * d + x1 + 0.5 * x2 + 0.8 * d * (x1 + 1) + rng.normal(size=n)
        frame = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        row = []
        for method in ("nnmatch", "psmatch"):
            fit = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2"], method=method,
                              biasadj=["x1", "x2"])
            row.append((fit.coefficients[0].estimate, fit.coefficients[0].std_error))
        estimates.append([r[0] for r in row])
        errors.append([r[1] for r in row])
    estimates, errors = np.array(estimates), np.array(errors)
    assert_allclose(estimates.mean(0), 2.8, atol=0.05)
    ratio = errors.mean(0) / estimates.std(0)
    assert np.all(np.abs(ratio - 1) < 0.2), ratio


def test_large_sample_resident_propensity_matching_is_fast():
    import time

    rng = np.random.default_rng(2)
    n = 200_000
    x = rng.normal(size=(n, 4))
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-x[:, 0] + 0.5 * x[:, 1]))).astype(int)
    y = x.sum(1) + d + rng.normal(size=n)
    frame = pd.DataFrame(x, columns=["a", "b", "c", "e"]).assign(y=y, d=d)
    start = time.perf_counter()
    # The public API now automatically chooses disk replay above 100k rows.
    # This historical 20-second guard measures the resident sorted-search kernel;
    # replay has separate full-source correctness and workspace-budget tests.
    from openecon.econometrics.teffects.estimators import fit_teffects
    from openecon.models import ModelSpec
    spec = ModelSpec(estimator="teffects", outcome="y", predictors=["a", "b", "c", "e"],
                     covariance="robust", columns={"treatment": "d"},
                     options={"method": "psmatch", "neighbors": 2})
    fit = fit_teffects(spec, frame)
    assert time.perf_counter() - start < 20
    assert abs(fit.coefficients[0].estimate - 1) < 0.1


def test_serialization_and_rendering(data):
    result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="psmatch")
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.loads(result.model_dump_json())
    assert "propensity-score matching" in result.summary()
    assert "begin{tabular}" in str(result.to_latex())
    assert result.extra["auxiliary_equations"]["TME1"][0]["term"] == "Intercept"
    assert result.provenance["stata_parity_validated"] is False
