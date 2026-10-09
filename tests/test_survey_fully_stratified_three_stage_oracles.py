"""Independent full-covariance oracles for stratified SSUs and terminal TSUs.

The reference uses original rows and both supplied positive-cell frames, with
NumPy/SciPy only; no production fitting, grouping or covariance kernels.
"""

from itertools import combinations, product
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

_spec = importlib.util.spec_from_file_location(
    "independent_fully_stratified_math",
    Path(__file__).resolve().parents[1]
    / "scripts/verify_survey_fully_stratified_three_stage_oracles.py",
)
reference_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference_math)
ROLES = reference_math.ROLES
DESCRIPTIVE = reference_math.DESCRIPTIVE
FAMILIES = reference_math.FAMILIES
GATES = DESCRIPTIVE + FAMILIES
geometry = reference_math.geometry
expected_case = reference_math.expected_case
covariance_components = reference_math.covariance_components
specification = reference_math.specification


def declare(frame, **options):
    import openecon as oe

    return oe.survey_fully_stratified_three_stage_design(
        frame,
        **ROLES,
        ssu_frame=reference_math.ssu_frame(frame),
        tsu_frame=reference_math.tsu_frame(frame),
        **options,
    )


def fixture():
    rng = np.random.default_rng(20261009)
    rows = []
    for h, N in zip(("north", "south", "east"), (5, 8, 7)):
        for p in range(3):
            for g in range(2):
                m = 2 + int(p == 1 and g == 0)
                for s in range(m):
                    for q in range(2):
                        leaf_count = 2 + (p + s + q) % 2
                        for j in range(leaf_count):
                            x, z, noise = rng.normal(size=3)
                            rows.append(
                                dict(
                                    h=h,
                                    p=p,
                                    g=g,
                                    s=s,
                                    q=q,
                                    j=j,
                                    N=N,
                                    M=m + 2 + p + g,
                                    L=leaf_count + 2 + g + q,
                                    x=x,
                                    z=z,
                                    y=1.4 + 0.65 * x - 0.4 * z + noise,
                                    a=2.0 + 0.3 * x + noise / 4,
                                    den=3.5 + abs(z) + s / 3,
                                    den2=2.5 + abs(x) + s / 4,
                                    binary=(j + p + s + g + q) % 2,
                                    count=(0, 2, 1)[j] + int(p == 2),
                                    category="a" if (j + p + s + q) % 3 else "b",
                                    domain=int(
                                        not (h == "east" and p == 2)
                                        and not (
                                            h == "south" and p == 0 and g == 1 and s == 1 and q == 1
                                        )
                                    ),
                                )
                            )
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.index = np.resize([37, -4, 37, 2], len(frame))
    return frame, declare(frame)


def call(
    frame,
    design,
    kind,
    *,
    domain=False,
    missing="raise",
    intercept=True,
    alpha=0.1,
    null=0.0,
    old=False,
):
    import openecon as oe

    spec = specification(kind, intercept)
    options = {
        "domain": "domain" if domain else None,
        "missing": missing,
        "alpha": alpha,
        "null": null,
        **{key: value for key, value in spec.items() if key != "args"},
    }
    if kind in FAMILIES:
        options.update(max_iter=200, tolerance=1e-11)
    fn = getattr(
        oe,
        ("survey_stratified_three_stage_" if old else "survey_fully_stratified_three_stage_")
        + kind,
    )
    return fn(frame, design, *spec["args"], **options)


def assert_expected(state, expected):
    numeric = reference_math.numerical
    numeric(state.estimates, expected["estimates"], "ordered point estimates")
    numeric(state.covariance, expected["covariance"], "full covariance")
    assert list(state.labels) == expected["labels"] and state.df == expected["df"]
    assert state.metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
    assert (
        state.metadata["out_of_domain_positions"] == np.flatnonzero(~expected["members"]).tolist()
    )
    assert (
        state.metadata["outcome_exclusions"]
        == np.flatnonzero(expected["members"] & ~expected["selected"]).tolist()
    )
    for key in ("stage1_covariance", "stage2_covariance", "stage3_covariance"):
        numeric(state.metadata[key], expected[key], key)
    for key, values in expected["primitives"].items():
        numeric(state.metadata[key], values, key)
    numeric(state.design.validation.weights, expected["weights"], "cell-specific weights")
    table = state.to_frame()
    assert list(table.index) == expected["labels"]
    reference = reference_math.expected_table(expected)
    for row, wanted in zip(
        table.astype(object).where(pd.notna(table), None).values.tolist(), reference
    ):
        for actual, desired in zip(row, wanted):
            if desired is None:
                assert actual is None
            else:
                numeric(actual, desired, "inference")
    numeric(table.attrs["covariance_matrix"], expected["covariance"], "displayed full covariance")


@pytest.mark.parametrize("census_stages", [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
def test_exhaustive_three_real_stages_and_both_lower_strata_unbiased_ht_full_variance(
    census_stages,
):
    N = 3
    n = N if 1 in census_stages else 2
    populations = {
        (p, g, s, q): np.array(
            [
                [
                    2 + 3 * p + 7 * g + s * s + 11 * q + j * j,
                    3 - p + 5 * g + 2 * s - 9 * q + (-1) ** j,
                ]
                for j in range(
                    3 if (p, g, s, q) == (0, 0, 0, 0) else 4 if (p, g, s, q) == (0, 1, 0, 1) else 2
                )
            ],
            float,
        )
        for p in range(N)
        for g, M in enumerate((2, 3))
        for s in range(M)
        for q in range(2)
    }
    truth = np.vstack(list(populations.values())).sum(0)
    estimates = []
    variances = []
    probabilities = []
    for psus in combinations(range(N), n):
        cells = [(p, g) for p in psus for g in range(2)]
        choices = [
            list(combinations(range((2, 3)[g]), (2, 3)[g] if 2 in census_stages else 2))
            for _, g in cells
        ]
        for drawn_ssus in product(*choices):
            leaves = [
                (p, g, s, q)
                for (p, g), selected in zip(cells, drawn_ssus)
                for s in selected
                for q in range(2)
            ]
            choices3 = [
                list(
                    combinations(
                        range(len(populations[key])),
                        len(populations[key]) if 3 in census_stages else 2,
                    )
                )
                for key in leaves
            ]
            probability = (
                1 / math.comb(N, n) / math.prod(map(len, choices)) / math.prod(map(len, choices3))
            )
            for drawn_tsus in product(*choices3):
                frame = pd.DataFrame(
                    [
                        dict(
                            h="a",
                            p=p,
                            g=g,
                            s=s,
                            q=q,
                            j=j,
                            N=N,
                            M=(2, 3)[g],
                            L=len(populations[p, g, s, q]),
                            y=populations[p, g, s, q][j, 0],
                            a=populations[p, g, s, q][j, 1],
                        )
                        for (p, g, s, q), chosen in zip(leaves, drawn_tsus)
                        for j in chosen
                    ]
                )
                state = call(frame, declare(frame), "total")
                oracle = expected_case(frame, "total")
                np.testing.assert_allclose(
                    state.covariance, oracle["covariance"], rtol=5e-14, atol=5e-11
                )
                estimates.append(state.estimates)
                variances.append(state.covariance)
                probabilities.append(probability)
    probabilities = np.asarray(probabilities)
    estimates = np.asarray(estimates)
    variances = np.asarray(variances)
    assert probabilities.sum() == pytest.approx(1.0, abs=3e-15)
    exact = np.einsum("n,ni,nj->ij", probabilities, estimates - truth, estimates - truth)
    np.testing.assert_allclose(probabilities @ estimates, truth, rtol=2e-14, atol=5e-12)
    np.testing.assert_allclose(
        np.einsum("n,nij->ij", probabilities, variances), exact, rtol=8e-14, atol=6e-11
    )
    if not census_stages:
        assert len(estimates) == 243 and exact[0, 0] > 0 and abs(exact[0, 1]) > 0
    if len(census_stages) == 3:
        np.testing.assert_array_equal(exact, np.zeros((2, 2)))


@pytest.mark.parametrize("kind", DESCRIPTIVE)
@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_joint_descriptives_cell_weights_full_covariance_and_reference_inference(
    kind, domain, missing
):
    frame, design = fixture()
    if missing:
        member = np.flatnonzero(frame.domain.to_numpy(bool))[0]
        name = "category" if kind == "proportion" else "y"
        frame.iat[member, frame.columns.get_loc(name)] = np.nan
    options = {
        "domain": "domain" if domain else None,
        "alpha": 0.1,
        "missing": "drop" if missing else "raise",
    }
    expected = expected_case(frame, kind, options=options)
    state = call(frame, design, kind, domain=domain, missing=options["missing"])
    assert_expected(state, expected)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("domain_missing", [False, True])
def test_observed_bread_weighted_likelihood_and_complete_cell_score_covariance(
    family, intercept, domain_missing
):
    frame, design = fixture()
    if domain_missing:
        member = np.flatnonzero(frame.domain.to_numpy(bool))[0]
        frame.iat[member, frame.columns.get_loc("x")] = np.nan
    options = {"domain": "domain" if domain_missing else None, "alpha": 0.1}
    expected = expected_case(frame, family, spec=specification(family, intercept), options=options)
    state = call(
        frame,
        design,
        family,
        intercept=intercept,
        domain=domain_missing,
        missing="drop" if domain_missing else "raise",
    )
    assert_expected(state, expected)


def test_terminal_cell_centering_does_not_invent_sampling_variance_from_different_cell_means():
    frame = pd.DataFrame(
        [
            dict(h="a", p=p, g=g, s=s, q=q, j=j, N=2, M=2, L=4, y=10.0 * q, a=-5.0 * q)
            for p in range(2)
            for g in range(2)
            for s in range(2)
            for q in range(2)
            for j in range(2)
        ]
    )
    state = call(frame, declare(frame), "total")
    np.testing.assert_array_equal(state.covariance, np.zeros((2, 2)))
    weights, strata = geometry(frame)
    weighted = weights[:, None] * frame[["y", "a"]].to_numpy()
    wrong = np.zeros((2, 2))
    for psus, f1 in strata:
        for cells in psus:
            for ssus, f2 in cells:
                for tcells in ssus:
                    positions = np.concatenate([pos for pos, _ in tcells])
                    f3 = 0.5
                    centered = weighted[positions] - weighted[positions].mean(0)
                    wrong += (
                        f1
                        * f2
                        * (1 - f3)
                        * len(positions)
                        / (len(positions) - 1)
                        * (centered.T @ centered)
                    )
    assert wrong[0, 0] > 0 and abs(wrong[0, 1]) > 0


def test_third_component_requires_cell_f3_and_its_parent_stage2_f2():
    frame, _ = fixture()
    expected = expected_case(frame, "total")
    weighted = expected["primitives"]["row_influences"]
    _, strata = geometry(frame)
    wrong_prefix = np.zeros((2, 2))
    wrong_f3 = np.zeros((2, 2))
    for psus, f1 in strata:
        for cells in psus:
            pooled2 = sum(len(ssus) for ssus, _ in cells) / sum(
                len(ssus) / f2 for ssus, f2 in cells
            )
            for ssus, f2 in cells:
                for tcells in ssus:
                    pooled3 = sum(len(pos) for pos, _ in tcells) / sum(
                        len(pos) / f3 for pos, f3 in tcells
                    )
                    for pos, f3 in tcells:
                        centered = weighted[pos] - weighted[pos].mean(0)
                        block = len(pos) / (len(pos) - 1) * (centered.T @ centered)
                        wrong_prefix += f1 * pooled2 * (1 - f3) * block
                        wrong_f3 += f1 * f2 * (1 - pooled3) * block
    assert not np.allclose(wrong_prefix, expected["stage3_covariance"], rtol=1e-5, atol=1e-8)
    assert not np.allclose(wrong_f3, expected["stage3_covariance"], rtol=1e-5, atol=1e-8)
    assert_expected(call(frame, declare(frame), "total"), expected)


@pytest.mark.parametrize("kind", GATES)
def test_one_terminal_cell_per_ssu_reduces_exactly_to_previous_stratified_family(kind):
    import openecon as oe

    frame, _ = fixture()
    frame["j"] = 10 * frame.q + frame.j
    frame["q"] = "only"
    frame["L"] = frame.groupby(["h", "p", "g", "s"], sort=False).j.transform("nunique") + 3
    new = call(frame, declare(frame), kind)
    old = call(
        frame,
        oe.survey_stratified_three_stage_design(
            frame,
            **{k: v for k, v in ROLES.items() if k != "tsu_strata"},
            ssu_frame=reference_math.ssu_frame(frame),
        ),
        kind,
        old=True,
    )
    np.testing.assert_array_equal(new.estimates, old.estimates)
    np.testing.assert_array_equal(new.covariance, old.covariance)
    assert new.df == old.df
    for part in ("stage1_covariance", "stage2_covariance", "stage3_covariance"):
        np.testing.assert_array_equal(new.metadata[part], old.metadata[part])


@pytest.mark.parametrize("kind", GATES)
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_each_stage_census_removes_only_its_own_component(kind, stage):
    frame, _ = fixture()
    names = {1: ["h"], 2: ["h", "p", "g"], 3: ["h", "p", "g", "s", "q"]}
    counted = {1: "p", 2: "s", 3: "j"}
    population = {1: "N", 2: "M", 3: "L"}
    frame[population[stage]] = frame.groupby(names[stage], sort=False)[counted[stage]].transform(
        "nunique"
    )
    state = call(frame, declare(frame), kind)
    parts = [np.array(state.metadata[f"stage{i}_covariance"]) for i in (1, 2, 3)]
    np.testing.assert_array_equal(parts[stage - 1], np.zeros_like(parts[stage - 1]))
    reference = expected_case(frame, kind)
    assert_expected(state, reference)


@pytest.mark.parametrize("kind", GATES)
def test_first_stage_df_zero_keeps_lower_sampling_uncertainty(kind):
    frame, _ = fixture()
    frame = frame.loc[frame.p.eq(0)].copy()
    frame["N"] = 1
    state = call(frame, declare(frame), kind)
    expected = expected_case(frame, kind)
    assert state.df == 0 and np.trace(np.array(state.covariance)) > 0
    assert_expected(state, expected)


@pytest.mark.parametrize("kind", GATES)
def test_physical_reordering_and_json_restore_preserve_full_joint_numerics(kind):
    frame, design = fixture()
    state = call(frame, design, kind, domain=True)
    moved = frame.iloc[::-1].copy()
    other = call(moved, declare(moved), kind, domain=True)
    np.testing.assert_allclose(other.estimates, state.estimates, rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(other.covariance, state.covariance, rtol=3e-8, atol=2e-10)
    restored = type(state).model_validate_json(
        json.dumps(state.model_dump(mode="json"), sort_keys=True, allow_nan=False)
    )
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())


@pytest.mark.parametrize("kind", DESCRIPTIVE)
def test_saved_contrast_uses_all_joint_covariance_entries(kind):
    frame, design = fixture()
    state = call(frame, design, kind, domain=True)
    expected = expected_case(frame, kind, options={"domain": "domain", "alpha": 0.1})
    a = np.array([1.0, -0.5] + ([0.2] if kind == "proportion" else []))
    result = state.contrast(a.tolist(), null=0.25)
    mean = float(a @ expected["estimates"])
    variance = float(a @ expected["covariance"] @ a)
    np.testing.assert_allclose(result.estimate, [mean], rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(result.std_error, [math.sqrt(variance)], rtol=3e-8, atol=2e-10)
    assert result.attrs["df"] == state.df


@pytest.mark.parametrize("family", FAMILIES)
def test_saved_lincom_joint_adjusted_f_and_prediction_match_independent_covariance(family):
    frame, design = fixture()
    state = call(frame, design, family, domain=True)
    expected = expected_case(frame, family, options={"domain": "domain", "alpha": 0.1})
    beta, cov = expected["estimates"], expected["covariance"]
    a = np.array([0.0, 1.0, -0.4])
    result = state.lincom(a.tolist(), null=0.1)
    estimate = float(a @ beta)
    se = math.sqrt(float(a @ cov @ a))
    np.testing.assert_allclose(result.estimate, [estimate], rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(result.std_error, [se], rtol=3e-8, atol=2e-10)
    restrictions = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    delta = restrictions @ beta
    rv = restrictions @ cov @ restrictions.T
    wald = float(delta @ np.linalg.solve(rv, delta))
    q = 2
    df = state.df
    adjusted = (df - q + 1) / (df * q) * wald
    test = state.test(restrictions.tolist())
    np.testing.assert_allclose(test.statistic, [adjusted], rtol=4e-8, atol=2e-10)
    np.testing.assert_allclose(
        test.p_value, [stats.f.sf(adjusted, q, df - q + 1)], rtol=4e-8, atol=2e-10
    )
    evaluation = pd.DataFrame({"x": [-0.4, 0.7], "z": [0.2, -0.3]}, index=[17, 17])
    predicted = state.predict(evaluation)
    x = np.column_stack((np.ones(2), evaluation.to_numpy()))
    eta = x @ beta
    if family == "regress":
        mean, derivative = eta, np.ones(2)
    elif family == "logit":
        mean = 1 / (1 + np.exp(-eta))
        derivative = mean * (1 - mean)
    elif family == "probit":
        mean = stats.norm.cdf(eta)
        derivative = stats.norm.pdf(eta)
    else:
        mean = np.exp(eta)
        derivative = mean
    np.testing.assert_allclose(predicted.estimate, mean, rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(
        predicted.std_error,
        np.sqrt(np.einsum("ni,ij,nj->n", x, cov, x)) * derivative,
        rtol=3e-8,
        atol=2e-10,
    )
