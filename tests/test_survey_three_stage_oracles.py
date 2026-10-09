"""Independent finite-population and likelihood oracles for three SRSWOR stages.

The reference calculations use raw NumPy rows and SciPy likelihoods. They do
not call a production grouping, linearization, covariance, or fitting helper.
"""

from itertools import combinations, product
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import importlib.util
from pathlib import Path

import openecon as oe
from openecon.analysis_contracts import AnalysisError


DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
FAMILIES = ("regress", "logit", "probit", "poisson")
GATES = DESCRIPTIVE + FAMILIES


ROLES = {"psu": "p", "ssu": "s", "tsu": "j", "strata": "h",
         "population_psu": "N", "population_ssu": "M", "population_tsu": "L"}
_spec = importlib.util.spec_from_file_location(
    "three_stage_independent_oracle",
    Path(__file__).resolve().parents[1]/"scripts"/"verify_survey_three_stage_oracles.py",
)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def declare(frame):
    return oe.survey_three_stage_design(frame, **ROLES)


def fixture():
    """Unequal counts at every depth; nested reused IDs and noncontiguous rows."""
    rng = np.random.default_rng(20261008)
    rows = []
    for h, population in zip(("north", "south", "east"), (5, 8, 7)):
        for p, m in enumerate((2, 3, 4)):
            for s in range(m):
                leaf_count = 2+s % 3
                for j in range(leaf_count):
                    x, z, noise = rng.normal(size=3)
                    rows.append({
                        "h": h, "p": p, "s": s, "j": j, "N": population,
                        "M": m+2+p, "L": leaf_count+2+(p+s) % 2,
                        "x": x, "z": z, "y": 1.4+.65*x-.4*z+noise,
                        "a": 2.+.3*x+noise/4, "den": 3.5+abs(z)+s/3,
                        "den2": 2.5+abs(x)+s/4,
                        "binary": (j+p+s) % 2, "count": (0, 2, 1, 4)[j]+(p == 2),
                        "category": "a" if j % 2 else "b",
                        "domain": int(not (h == "east" and p == 2)),
                    })
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.index = np.resize([37, -4, 37, 2], len(frame))
    return frame, declare(frame)


def geometry(frame):
    return oracle.geometry(frame, ROLES)[:2]


def stage_covariance(frame, rows):
    return oracle.covariance_components(np.asarray(rows), geometry(frame)[1])


def specification(kind, intercept=True):
    if kind == "ratio":
        return {"args": [["y", "a"], ["den", "den2"]]}
    if kind == "proportion":
        return {"args": ["category"], "categories": ["a", "b", "absent"]}
    if kind in DESCRIPTIVE:
        return {"args": [["y", "a"]]}
    name = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[kind]
    return {"args": [name, ["x", "z"]], "intercept": intercept}


def reference_case(frame, kind, *, domain=False, intercept=True):
    return oracle.expected_case(frame, ROLES, kind, specification(kind, intercept),
                                {"domain": "domain" if domain else None})


def reference_tuple(frame, kind, *, domain=False, intercept=True):
    state = reference_case(frame, kind, domain=domain, intercept=intercept)
    return tuple(state[key] for key in (
        "estimates", "covariance", "selected", "stage1_covariance", "stage2_covariance", "stage3_covariance",
    ))


def descriptive_oracle(frame, kind, *, domain=False):
    return reference_tuple(frame, kind, domain=domain)


def regression_oracle(frame, family, *, domain=False, intercept=True):
    return reference_tuple(frame, family, domain=domain, intercept=intercept)


def call(frame, design, kind, *, domain=False, missing="raise", intercept=True,
         alpha=.1, null=0, three_stage=True):
    options = {"domain": "domain" if domain else None, "missing": missing,
               "alpha": alpha, "null": null}
    fn = getattr(oe, ("survey_three_stage_" if three_stage else "survey_two_stage_")+kind)
    spec = specification(kind, intercept)
    options.update({key: value for key, value in spec.items() if key != "args"})
    if kind in FAMILIES:
        options.update(max_iter=200, tolerance=1e-11)
    return fn(frame, design, *spec["args"], **options)


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


@pytest.mark.parametrize("census_stages", [
    (), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3),
])
def test_exhaustive_ht_total_has_exact_unbiased_design_variance(census_stages):
    """Every three-level sample, weighted by its exact conditional probability.

    The exceptional three-unit leaf creates unequal final-stage fractions. The
    complete enumeration tests the residual f1*f2 third-stage prefix directly.
    """
    N, M = 3, 3
    n, m = (N if 1 in census_stages else 2), (M if 2 in census_stages else 2)
    population = {
        (p, s): np.array([[1+3*p+s*s+j*j, 2-p+2*s+(j+1)*(-1)**j]
                          for j in range(3 if (p, s) == (0, 0) else 2)], dtype=float)
        for p in range(N) for s in range(M)
    }
    truth = np.vstack(list(population.values())).sum(axis=0)
    estimates, variances, probabilities = [], [], []
    for psus in combinations(range(N), n):
        s_choices = [tuple(combinations(range(M), m)) for _ in psus]
        for selected_ssus in product(*s_choices):
            selected_groups = [(p, s) for p, ssus in zip(psus, selected_ssus) for s in ssus]
            t_choices = [tuple(combinations(range(len(population[group])),
                                           len(population[group]) if 3 in census_stages else 2))
                         for group in selected_groups]
            probability = 1/math.comb(N, n)/math.comb(M, m)**n/math.prod(len(v) for v in t_choices)
            for selected_tsus in product(*t_choices):
                frame = pd.DataFrame([
                    {"h": "a", "p": p, "s": s, "j": j, "N": N, "M": M,
                     "L": len(population[p, s]), "y": population[p, s][j, 0],
                     "a": population[p, s][j, 1]}
                    for (p, s), tsus in zip(selected_groups, selected_tsus) for j in tsus
                ])
                actual = oe.survey_three_stage_total(frame, declare(frame), ["y", "a"])
                estimates.append(actual.estimates)
                variances.append(actual.covariance)
                probabilities.append(probability)
    probabilities = np.array(probabilities)
    estimates, variances = np.array(estimates), np.array(variances)
    assert probabilities.sum() == pytest.approx(1, abs=2e-15)
    exact_covariance = np.einsum("n,ni,nj->ij", probabilities, estimates-truth, estimates-truth)
    average_estimator = np.einsum("n,nij->ij", probabilities, variances)
    np.testing.assert_allclose(probabilities @ estimates, truth, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(average_estimator, exact_covariance, rtol=3e-14, atol=3e-12)
    if set(census_stages) == {1, 2, 3}:
        np.testing.assert_array_equal(exact_covariance, np.zeros((2, 2)))
    else:
        assert exact_covariance[0, 0] > 0


@pytest.mark.parametrize("kind", DESCRIPTIVE)
@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_joint_descriptive_estimates_full_covariance_and_reference_inference(kind, domain, missing):
    frame, design = fixture()
    if missing:
        # Outcome mutations preserve the original sampling design and counts.
        member = np.flatnonzero(frame.domain.to_numpy(dtype=bool))[0]
        frame.iloc[member, frame.columns.get_loc("category" if kind == "proportion" else "y")] = np.nan
    expected, covariance, selected, between, within, leaf = descriptive_oracle(frame, kind, domain=domain)
    null = [0.1]*len(expected)
    actual = call(frame, design, kind, domain=domain, missing="drop" if missing else "raise", null=null)
    np.testing.assert_allclose(actual.estimates, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=2e-12, atol=2e-12)
    assert between[0, 0] > 0 and within[0, 0] > 0 and leaf[0, 0] > 0
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
def test_weighted_likelihood_observed_bread_and_three_stage_sandwich(family, intercept, domain_missing):
    frame, design = fixture()
    if domain_missing:
        member = np.flatnonzero(frame.domain.to_numpy(dtype=bool))[0]
        frame.iloc[member, frame.columns.get_loc("z")] = np.nan
    expected, covariance, selected, between, within, leaf = regression_oracle(
        frame, family, intercept=intercept, domain=domain_missing,
    )
    null = [0.2]*len(expected)
    actual = call(frame, design, family, domain=domain_missing,
                  missing="drop" if domain_missing else "raise", intercept=intercept, null=null)
    np.testing.assert_allclose(actual.coefficients, expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=3e-8, atol=2e-10)
    assert between[0, 0] > 0 and within[0, 0] > 0 and leaf[0, 0] > 0
    assert abs(covariance[0, 1]) > 1e-8
    assert actual.df == 6
    assert actual.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    check_inference(actual, expected, covariance, null=null)


@pytest.mark.parametrize("kind", GATES)
@pytest.mark.parametrize("census", [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
def test_stage_census_removes_only_its_own_covariance_component(kind, census):
    frame, _ = fixture()
    if 1 in census:
        frame["N"] = 3
    if 2 in census:
        frame["M"] = frame.groupby(["h", "p"]).s.transform("nunique")
    if 3 in census:
        frame["L"] = frame.groupby(["h", "p", "s"]).j.transform("size")
    expected, covariance, _, *stages = reference_tuple(frame, kind)
    actual = call(frame, declare(frame), kind)
    np.testing.assert_allclose(point_estimates(actual), expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=3e-8, atol=2e-10)
    for depth, component in enumerate(stages, start=1):
        np.testing.assert_allclose(actual.metadata[f"stage{depth}_covariance"], component,
                                   rtol=3e-8, atol=2e-10)
        if depth in census:
            np.testing.assert_array_equal(component, np.zeros_like(component))
        else:
            assert component[0, 0] > 0
    check_inference(actual, expected, covariance)


@pytest.mark.parametrize("kind", GATES)
def test_first_stage_df_zero_retains_positive_within_variance_without_ci(kind):
    frame = pd.DataFrame({
        "h": ["certainty"]*8, "p": [1]*8, "s": [1]*8, "j": range(8), "N": [1]*8, "M": [1]*8, "L": [12]*8,
        "x": np.tile([-1., 0., 1., 2.], 2), "z": np.repeat([-1., 1.], 4),
        "y": [1., 4., 2., 5., 4., 1., 6., 3.], "a": [3., 1., 4., 2., 5., 2., 1., 4.],
        "den": [2., 3., 4., 5., 3., 4., 5., 6.], "den2": [4., 2., 5., 3., 6., 4., 3., 5.],
        "binary": [0, 1, 1, 0, 0, 1, 1, 0], "count": [0, 2, 1, 4, 2, 1, 3, 0],
        "category": ["a", "b"]*4, "domain": [1]*8,
    })
    reference = descriptive_oracle(frame, kind) if kind in DESCRIPTIVE else regression_oracle(frame, kind)
    expected, covariance, _, between, within, leaf = reference
    actual = call(frame, declare(frame), kind)
    assert actual.df == 0 and covariance[0, 0] > 0 and leaf[0, 0] > 0
    np.testing.assert_array_equal(between, np.zeros_like(between))
    np.testing.assert_array_equal(within, np.zeros_like(within))
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


@pytest.mark.parametrize("kind", GATES)
def test_singleton_census_third_stage_reduces_to_existing_two_stage_contract(kind):
    original, _ = fixture()
    frame = original.groupby(["h", "p", "s"], sort=False).head(1).copy()
    frame["L"] = 1
    result = call(frame, declare(frame), kind)
    design2 = oe.survey_two_stage_design(frame, psu="p", ssu="s", strata="h",
                                        population_psu="N", population_ssu="M")
    old = call(frame, design2, kind, three_stage=False)
    np.testing.assert_allclose(point_estimates(result), point_estimates(old), rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(result.covariance, old.covariance, rtol=2e-11, atol=2e-12)
    np.testing.assert_array_equal(result.metadata["stage3_covariance"], np.zeros_like(result.covariance))


@pytest.mark.parametrize("kind", GATES)
def test_retained_primitive_scores_and_all_covariance_components_match_original_rows(kind):
    frame, design = fixture()
    member = np.flatnonzero(frame.domain.to_numpy(dtype=bool))[0]
    name = "category" if kind == "proportion" else "y" if kind in DESCRIPTIVE else "z"
    frame.iloc[member, frame.columns.get_loc(name)] = np.nan
    expected = reference_case(frame, kind, domain=True)
    actual = call(frame, design, kind, domain=True, missing="drop")
    for key in ("stage1_covariance", "stage2_covariance", "stage3_covariance"):
        np.testing.assert_allclose(actual.metadata[key], expected[key], rtol=3e-8, atol=2e-10)
    for key, value in expected["primitives"].items():
        saved_key = "row_scores" if key == "normalized_row_scores" else key
        np.testing.assert_allclose(actual.metadata[saved_key], value, rtol=3e-8, atol=2e-10)
    weights, _ = geometry(frame)
    assert actual.design.validation.sum_weights == pytest.approx(weights.sum())
    assert actual.design.validation.n_ssu == 27
    assert actual.metadata["out_of_domain_positions"] == np.flatnonzero(~expected["members"]).tolist()
    assert actual.metadata["outcome_exclusions"] == [int(member)]


@pytest.mark.parametrize("kind", DESCRIPTIVE)
def test_saved_descriptive_contrast_uses_full_joint_covariance(kind):
    frame, design = fixture()
    source = call(frame, design, kind, domain=True)
    restored = type(source).model_validate_json(source.model_dump_json())
    reference = reference_case(frame, kind, domain=True)
    vector = np.array([1., -2., .5] if kind == "proportion" else [1., -2.])
    estimate = vector @ reference["estimates"]
    covariance = np.array([[vector @ reference["covariance"] @ vector]])
    table = restored.contrast(vector.tolist(), null=1.)
    expected = {"estimates": np.array([estimate]), "covariance": covariance,
                "df": 6, "alpha": .1, "null": np.array([1.])}
    np.testing.assert_allclose(table.to_numpy(dtype=float), oracle.expected_table(expected),
                               rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(table.attrs["covariance_matrix"], covariance, rtol=3e-8, atol=2e-10)


@pytest.mark.parametrize("family", FAMILIES)
def test_saved_model_prediction_lincom_and_adjusted_f_have_independent_full_covariance(family):
    frame, design = fixture()
    source = call(frame, design, family, domain=True)
    restored = type(source).model_validate_json(source.model_dump_json())
    reference = reference_case(frame, family, domain=True)
    beta, covariance = reference["estimates"], reference["covariance"]
    vector = np.array([1., -2., .5])
    estimate, variance = vector @ beta, vector @ covariance @ vector
    expected = {"estimates": np.array([estimate]), "covariance": np.array([[variance]]),
                "df": 6, "alpha": .1, "null": np.array([1.])}
    contrast = restored.lincom(vector.tolist(), null=1.)
    np.testing.assert_allclose(contrast.to_numpy(dtype=float), oracle.expected_table(expected),
                               rtol=3e-8, atol=2e-10)
    restrictions = np.array([[0., 1., 0.], [0., 0., 1.]])
    residual = restrictions @ beta-np.array([.1, .2])
    joint_covariance = restrictions @ covariance @ restrictions.T
    wald = residual @ np.linalg.solve(joint_covariance, residual)
    statistic = wald*(6-2+1)/(6*2)
    joint = restored.test(restrictions.tolist(), null=[.1, .2])
    np.testing.assert_allclose(joint.iloc[0].to_numpy(dtype=float),
                               [wald, statistic, 2, 5, stats.f.sf(statistic, 2, 5)],
                               rtol=3e-8, atol=2e-10)
    new_frame = pd.DataFrame({"x": [-.7, .2], "z": [.4, -.8]}, index=["low", "high"])
    X = np.column_stack((np.ones(2), new_frame.to_numpy()))
    eta = X @ beta
    if family == "regress":
        prediction, derivative = eta, np.ones(2)
    elif family == "logit":
        prediction = 1/(1+np.exp(-eta))
        derivative = prediction*(1-prediction)
    elif family == "probit":
        prediction, derivative = stats.norm.cdf(eta), stats.norm.pdf(eta)
    else:
        prediction = derivative = np.exp(eta)
    jacobian = derivative[:, None]*X
    predicted_covariance = jacobian @ covariance @ jacobian.T
    expected = {"estimates": prediction, "covariance": predicted_covariance,
                "df": 6, "alpha": .1, "null": np.zeros(2)}
    predicted = restored.predict(new_frame)
    assert predicted.index.tolist() == ["low", "high"]
    assert predicted.attrs["physical_positions"] == [0, 1]
    assert predicted.attrs["conditional_fixed_covariates"] is True
    assert predicted.attrs["population_distribution_uncertainty"] is False
    np.testing.assert_allclose(predicted.to_numpy(dtype=float), oracle.expected_table(expected),
                               rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(predicted.attrs["covariance_matrix"], predicted_covariance,
                               rtol=3e-8, atol=2e-10)
