"""Independent finite-population and likelihood oracles for two SRSWOR stages.

The reference calculations use raw NumPy rows and SciPy likelihoods. They do
not call a production grouping, linearization, covariance, or fitting helper.
"""

from itertools import combinations, product
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError


DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
FAMILIES = ("regress", "logit", "probit", "poisson")
GATES = DESCRIPTIVE + FAMILIES


def declare(frame):
    return oe.survey_two_stage_design(
        frame, psu="p", ssu="s", strata="h",
        population_psu="N", population_ssu="M",
    )


def fixture():
    """Noncontiguous nested rows, unequal stage weights, and duplicate indices."""
    rng = np.random.default_rng(20261008)
    rows = []
    for h, population in zip(("north", "south", "east"), (5, 8, 7)):
        for p, count in enumerate((3, 4, 5)):
            for s in range(count):
                x, z, noise = rng.normal(size=3)
                rows.append({
                    "h": h, "p": p, "s": s, "N": population,
                    "M": count + 2 + p, "x": x, "z": z,
                    "y": 1.4 + 0.65*x - 0.4*z + noise,
                    "a": 2.0 + 0.3*x + noise/4,
                    "den": 3.5 + abs(z) + s/3,
                    "den2": 2.5 + abs(x) + s/4,
                    "binary": (0, 1, 0, 1, 1)[s],
                    "count": (0, 2, 1, 4, 3)[s] + (p == 2),
                    "category": ("a", "b", "a", "b", "b")[s],
                    "domain": int(not (h == "east" and p == 2)),
                })
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.index = np.resize([37, -4, 37, 2], len(frame))
    return frame, declare(frame)


def geometry(frame):
    """Derive stage probabilities from complete physical identities, not design."""
    weights = np.empty(len(frame), dtype=float)
    strata = []
    for h in pd.unique(frame.h):
        positions = np.flatnonzero(frame.h.to_numpy() == h)
        psus = []
        population = float(frame.N.iloc[positions[0]])
        for p in pd.unique(frame.p.iloc[positions]):
            rows = positions[frame.p.iloc[positions].to_numpy() == p]
            M = float(frame.M.iloc[rows[0]])
            psus.append((rows, len(rows)/M))
        f1 = len(psus)/population
        for rows, f2 in psus:
            weights[rows] = 1/(f1*f2)
        strata.append((psus, f1))
    return weights, strata


def stage_covariance(frame, weighted_rows):
    """Direct multivariate SRSWOR recursion, including the residual f1 factor."""
    weighted_rows = np.asarray(weighted_rows, dtype=float)
    _, strata = geometry(frame)
    k = weighted_rows.shape[1]
    between, within = np.zeros((k, k)), np.zeros((k, k))
    for psus, f1 in strata:
        totals = np.array([weighted_rows[rows].sum(axis=0) for rows, _ in psus])
        if f1 < 1:
            centered = totals - totals.mean(axis=0)
            between += (1-f1)*len(psus)/(len(psus)-1)*(centered.T @ centered)
        for rows, f2 in psus:
            if f2 < 1:
                centered = weighted_rows[rows] - weighted_rows[rows].mean(axis=0)
                within += f1*(1-f2)*len(rows)/(len(rows)-1)*(centered.T @ centered)
    return between, within


def selected_rows(frame, names, *, domain=False):
    selected = frame[list(names)].notna().all(axis=1).to_numpy()
    if domain:
        selected &= frame.domain.to_numpy(dtype=bool)
    return selected


def descriptive_oracle(frame, kind, *, domain=False):
    names = ("category",) if kind == "proportion" else ("y", "a")
    required = names + (("den", "den2") if kind == "ratio" else ())
    selected = selected_rows(frame, required, domain=domain)
    weights, _ = geometry(frame)
    if kind == "proportion":
        values = np.column_stack([
            frame.category.to_numpy() == category for category in ("a", "b", "absent")
        ]).astype(float)
    else:
        values = frame[list(names)].fillna(0).to_numpy(dtype=float)
    values[~selected] = 0
    effective = weights*selected
    numerator = effective @ values
    if kind == "total":
        estimate, influence = numerator, values
    else:
        denominator = (
            frame[["den", "den2"]].fillna(0).to_numpy(dtype=float)
            if kind == "ratio" else np.ones_like(values)
        )
        normalizer = effective @ denominator
        estimate = numerator/normalizer
        influence = selected[:, None]*(values-denominator*estimate)/normalizer
    between, within = stage_covariance(frame, weights[:, None]*influence)
    return estimate, between+within, selected, between, within


def likelihood(family, beta, x, y, weights):
    """Observed score and bread for actual inverse inclusion weights."""
    index = x @ beta
    if family == "regress":
        factor, curvature = y-index, np.ones(len(y))
        objective = weights @ (factor**2/2)
    elif family == "logit":
        probability = special.expit(index)
        factor, curvature = y-probability, probability*(1-probability)
        objective = weights @ (np.logaddexp(0, index)-y*index)
    elif family == "probit":
        sign = 2*y-1
        log_probability = special.log_ndtr(sign*index)
        mills = np.exp(stats.norm.logpdf(index)-log_probability)
        factor, curvature = sign*mills, mills*(mills+sign*index)
        objective = -weights @ log_probability
    else:
        mean = np.exp(index)
        factor, curvature = y-mean, mean
        objective = weights @ (mean-y*index)
    score = (weights*factor)[:, None]*x
    bread = x.T @ ((weights*curvature)[:, None]*x)
    return objective, score, bread


def regression_oracle(frame, family, *, intercept=True, domain=False):
    name = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[family]
    selected = selected_rows(frame, (name, "x", "z"), domain=domain)
    weights, _ = geometry(frame)
    x = frame[["x", "z"]].fillna(0).to_numpy(dtype=float)
    if intercept:
        x = np.column_stack((np.ones(len(frame)), x))
    xs, ys, ws = x[selected], frame[name].to_numpy(dtype=float)[selected], weights[selected]
    if family == "regress":
        beta = np.linalg.solve(xs.T @ (ws[:, None]*xs), xs.T @ (ws*ys))
    else:
        def objective(beta):
            value, score, _ = likelihood(family, beta, xs, ys, ws)
            return value/ws.sum(), -score.sum(axis=0)/ws.sum()

        fit = optimize.minimize(
            objective, np.zeros(xs.shape[1]), jac=True, method="BFGS",
            options={"gtol": 1e-11, "maxiter": 500},
        )
        polished = optimize.root(
            lambda beta: likelihood(family, beta, xs, ys, ws)[1].sum(axis=0),
            fit.x, jac=lambda beta: -likelihood(family, beta, xs, ys, ws)[2],
            tol=1e-11,
        )
        beta = polished.x
        assert np.max(np.abs(objective(beta)[1])) < 1e-9
    _, score, bread = likelihood(family, beta, xs, ys, ws)
    row_scores = np.zeros((len(frame), len(beta)))
    row_scores[selected] = score
    # Solve the full sandwich before centering; stage covariance commutes with
    # this fixed linear transformation and avoids a numerical matrix inverse.
    weighted_influence = np.linalg.solve(bread, row_scores.T).T
    between, within = stage_covariance(frame, weighted_influence)
    return beta, between+within, selected, between, within


def call(frame, design, kind, *, domain=False, missing="raise", intercept=True,
         alpha=0.1, null=0, two_stage=True):
    options = {"domain": "domain" if domain else None, "missing": missing,
               "alpha": alpha, "null": null}
    fn = getattr(oe, ("survey_two_stage_" if two_stage else "survey_")+kind)
    if kind == "ratio":
        return fn(frame, design, ["y", "a"], ["den", "den2"], **options)
    if kind == "proportion":
        return fn(frame, design, "category", categories=["a", "b", "absent"], **options)
    if kind in DESCRIPTIVE:
        return fn(frame, design, ["y", "a"], **options)
    name = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[kind]
    return fn(frame, design, name, ["x", "z"], intercept=intercept,
              max_iter=200, tolerance=1e-11, **options)


def point_estimates(result):
    return np.asarray(result.estimates if hasattr(result, "estimates") else result.coefficients)


def check_inference(result, expected, covariance, *, alpha=0.1, null=0):
    table = result.to_frame()
    np.testing.assert_allclose(table.estimate, expected, rtol=2e-8, atol=2e-10)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    np.testing.assert_allclose(table.std_error, se, rtol=2e-8, atol=2e-10)
    if result.df:
        quantile = stats.t.ppf(1-alpha/2, result.df)
        np.testing.assert_allclose(table.ci_low, expected-quantile*se, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(table.ci_high, expected+quantile*se, rtol=2e-8, atol=2e-9)
        positive = se > 0
        statistics = (expected-np.broadcast_to(null, len(expected)))[positive]/se[positive]
        np.testing.assert_allclose(table.loc[positive, "statistic"].to_numpy(dtype=float), statistics,
                                   rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(table.loc[positive, "p_value"].to_numpy(dtype=float),
                                   2*stats.t.sf(abs(statistics), result.df),
                                   rtol=2e-8, atol=2e-10)
        assert table.loc[~positive, ["statistic", "p_value"]].isna().all().all()
    else:
        assert table.statistic.isna().all() and table.p_value.isna().all()
        positive = se > 0
        assert table.loc[positive, ["ci_low", "ci_high"]].isna().all().all()
        np.testing.assert_array_equal(table.loc[~positive, "ci_low"], expected[~positive])
        np.testing.assert_array_equal(table.loc[~positive, "ci_high"], expected[~positive])
    assert table.attrs["df"] == result.df
    np.testing.assert_allclose(table.attrs["covariance_matrix"], covariance, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("populations", [(3, 3, 3), (3, 4, 3)])
@pytest.mark.parametrize("first_census", [False, True])
def test_exhaustive_ht_total_has_exact_unbiased_design_variance(populations, first_census):
    """Enumerate every sample with its actual conditional sampling probability."""
    N, n, m = len(populations), (len(populations) if first_census else 2), 2
    population = [
        np.array([[1+3*i+j*j, 2-i+(j+1)*(-1)**j] for j in range(M)], dtype=float)
        for i, M in enumerate(populations)
    ]
    truth = np.vstack(population).sum(axis=0)
    estimates, variances, probabilities = [], [], []
    for psus in combinations(range(N), n):
        choices = [tuple(combinations(range(populations[i]), m)) for i in psus]
        probability = 1/math.comb(N, n)/math.prod(len(v) for v in choices)
        for selected in product(*choices):
            records = [
                {"h": "a", "p": i, "s": j, "N": N, "M": populations[i],
                 "y": population[i][j, 0], "a": population[i][j, 1]}
                for i, js in zip(psus, selected) for j in js
            ]
            frame = pd.DataFrame(records)
            actual = oe.survey_two_stage_total(frame, declare(frame), ["y", "a"])
            estimates.append(actual.estimates)
            variances.append(actual.covariance)
            probabilities.append(probability)
    probabilities = np.array(probabilities)
    estimates, variances = np.array(estimates), np.array(variances)
    assert probabilities.sum() == pytest.approx(1, abs=1e-15)
    expected_mean = probabilities @ estimates
    centered = estimates-truth
    exact_covariance = np.einsum("n,ni,nj->ij", probabilities, centered, centered)
    average_estimator = np.einsum("n,nij->ij", probabilities, variances)
    np.testing.assert_allclose(expected_mean, truth, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(average_estimator, exact_covariance, rtol=2e-14, atol=2e-12)
    assert exact_covariance[0, 0] > 0 and abs(exact_covariance[0, 1]) > 1e-8


@pytest.mark.parametrize("kind", DESCRIPTIVE)
@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_joint_descriptive_estimates_full_covariance_and_reference_inference(kind, domain, missing):
    frame, design = fixture()
    if missing:
        # Outcome mutations preserve the original sampling design and counts.
        member = np.flatnonzero(frame.domain.to_numpy(dtype=bool))[0]
        frame.iloc[member, frame.columns.get_loc("category" if kind == "proportion" else "y")] = np.nan
    expected, covariance, selected, between, within = descriptive_oracle(frame, kind, domain=domain)
    null = [0.1]*len(expected)
    actual = call(frame, design, kind, domain=domain, missing="drop" if missing else "raise", null=null)
    np.testing.assert_allclose(actual.estimates, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=2e-12, atol=2e-12)
    assert between[0, 0] > 0 and within[0, 0] > 0
    assert actual.df == 6
    assert actual.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    assert actual.metadata["n_design"] == len(frame)
    assert actual.design.validation.n_psu == 9
    check_inference(actual, expected, covariance, null=null)
    if kind == "proportion":
        assert actual.estimates[-1] == 0
        np.testing.assert_allclose(np.sum(covariance, axis=0), 0, atol=1e-15)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("domain_missing", [False, True])
def test_weighted_likelihood_observed_bread_and_two_stage_sandwich(family, intercept, domain_missing):
    frame, design = fixture()
    if domain_missing:
        member = np.flatnonzero(frame.domain.to_numpy(dtype=bool))[0]
        frame.iloc[member, frame.columns.get_loc("z")] = np.nan
    expected, covariance, selected, between, within = regression_oracle(
        frame, family, intercept=intercept, domain=domain_missing,
    )
    null = [0.2]*len(expected)
    actual = call(frame, design, family, domain=domain_missing,
                  missing="drop" if domain_missing else "raise", intercept=intercept, null=null)
    np.testing.assert_allclose(actual.coefficients, expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=3e-8, atol=2e-10)
    assert between[0, 0] > 0 and within[0, 0] > 0
    assert abs(covariance[0, 1]) > 1e-8
    assert actual.df == 6
    assert actual.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    check_inference(actual, expected, covariance, null=null)


@pytest.mark.parametrize("kind", GATES)
@pytest.mark.parametrize("census", ["first", "second", "both"])
def test_stage_census_removes_only_its_own_covariance_component(kind, census):
    frame, _ = fixture()
    if census in ("first", "both"):
        frame["N"] = 3
    if census in ("second", "both"):
        frame["M"] = frame.groupby(["h", "p"]).s.transform("size")
    design = declare(frame)
    reference = descriptive_oracle(frame, kind) if kind in DESCRIPTIVE else regression_oracle(frame, kind)
    expected, covariance, _, between, within = reference
    actual = call(frame, design, kind)
    np.testing.assert_allclose(point_estimates(actual), expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=3e-8, atol=2e-10)
    if census in ("first", "both"):
        np.testing.assert_array_equal(between, np.zeros_like(between))
    if census in ("second", "both"):
        np.testing.assert_array_equal(within, np.zeros_like(within))
    if census == "second":
        # With every selected PSU fully enumerated, the new recursive design
        # must reduce to the established single-stage FPC sampling contract.
        legacy_frame = frame.assign(_w=geometry(frame)[0])
        legacy_design = oe.survey_design(legacy_frame, weights="_w", psu="p", strata="h", fpc="N")
        legacy = call(legacy_frame, legacy_design, kind, two_stage=False)
        np.testing.assert_allclose(point_estimates(actual), point_estimates(legacy), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(actual.covariance, legacy.covariance, rtol=2e-11, atol=1e-12)
    if census != "both":
        assert covariance[0, 0] > 0
    else:
        np.testing.assert_array_equal(actual.covariance, np.zeros_like(covariance))
    check_inference(actual, expected, covariance)


@pytest.mark.parametrize("kind", GATES)
def test_first_stage_df_zero_retains_positive_within_variance_without_ci(kind):
    frame = pd.DataFrame({
        "h": ["certainty"]*8, "p": [1]*8, "s": range(8), "N": [1]*8, "M": [12]*8,
        "x": np.tile([-1., 0., 1., 2.], 2), "z": np.repeat([-1., 1.], 4),
        "y": [1., 4., 2., 5., 4., 1., 6., 3.], "a": [3., 1., 4., 2., 5., 2., 1., 4.],
        "den": [2., 3., 4., 5., 3., 4., 5., 6.], "den2": [4., 2., 5., 3., 6., 4., 3., 5.],
        "binary": [0, 1, 1, 0, 0, 1, 1, 0], "count": [0, 2, 1, 4, 2, 1, 3, 0],
        "category": ["a", "b"]*4, "domain": [1]*8,
    })
    reference = descriptive_oracle(frame, kind) if kind in DESCRIPTIVE else regression_oracle(frame, kind)
    expected, covariance, _, between, within = reference
    actual = call(frame, declare(frame), kind)
    assert actual.df == 0 and covariance[0, 0] > 0 and within[0, 0] > 0
    np.testing.assert_array_equal(between, np.zeros_like(between))
    np.testing.assert_allclose(point_estimates(actual), expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=3e-8, atol=2e-10)
    check_inference(actual, expected, covariance)


@pytest.mark.parametrize("kind", GATES)
def test_reordered_physical_rows_preserve_estimates_and_full_covariance(kind):
    frame, design = fixture()
    baseline = call(frame, design, kind, domain=True)
    shuffled = frame.iloc[np.random.default_rng(19).permutation(len(frame))].copy()
    result = call(shuffled, declare(shuffled), kind, domain=True)
    np.testing.assert_allclose(point_estimates(result), point_estimates(baseline), rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(result.covariance, baseline.covariance, rtol=3e-8, atol=2e-10)
    assert result.metadata["sample_positions"] == np.flatnonzero(shuffled.domain.to_numpy()).tolist()
    assert result.design.validation.design_input_sha256 != design.validation.design_input_sha256


@pytest.mark.parametrize("kind", GATES)
def test_sorted_json_saved_result_replays_joint_numerics_and_display(kind):
    frame, design = fixture()
    actual = call(frame, design, kind, domain=True)
    state = json.loads(json.dumps(actual.model_dump(mode="json"), sort_keys=True, allow_nan=False))
    restored = type(actual).model_validate_json(json.dumps(state, allow_nan=False))
    assert restored.model_dump(mode="json") == actual.model_dump(mode="json")
    np.testing.assert_array_equal(point_estimates(restored), point_estimates(actual))
    np.testing.assert_array_equal(restored.covariance, actual.covariance)
    pd.testing.assert_frame_equal(restored.to_frame(), actual.to_frame())


@pytest.mark.parametrize("kind", DESCRIPTIVE)
def test_missing_outside_domain_is_ignored_but_in_domain_requires_explicit_drop(kind):
    frame, design = fixture()
    name = "category" if kind == "proportion" else "y"
    frame.loc[frame.domain == 0, name] = np.nan
    expected, covariance, *_ = descriptive_oracle(frame, kind, domain=True)
    actual = call(frame, design, kind, domain=True)
    np.testing.assert_allclose(actual.estimates, expected)
    np.testing.assert_allclose(actual.covariance, covariance)
    position = np.flatnonzero(frame.domain.to_numpy())[0]
    frame.iloc[position, frame.columns.get_loc(name)] = np.nan
    with pytest.raises(AnalysisError, match="Missing in-domain"):
        call(frame, design, kind, domain=True)
    dropped = call(frame, design, kind, domain=True, missing="drop")
    assert dropped.df == actual.df
    assert dropped.design == actual.design
    assert dropped.metadata["outcome_exclusions"] == [int(position)]
