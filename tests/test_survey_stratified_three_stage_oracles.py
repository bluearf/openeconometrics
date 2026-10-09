"""Independent sampling/likelihood oracles for stratified SSUs within each PSU.

Only NumPy/SciPy implement the reference. Production grouping, covariance,
linearization and fitting helpers are never used to calculate expected results.
"""

from itertools import combinations, product
import importlib.util
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats


DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
FAMILIES = ("regress", "logit", "probit", "poisson")
GATES = DESCRIPTIVE + FAMILIES
ROLES = {
    "psu": "p",
    "ssu_strata": "g",
    "ssu": "s",
    "tsu": "j",
    "strata": "h",
    "population_psu": "N",
    "population_ssu": "M",
    "population_tsu": "L",
}
_spec = importlib.util.spec_from_file_location(
    "independent_unstratified_three_stage_math",
    Path(__file__).resolve().parents[1] / "scripts/verify_survey_three_stage_oracles.py",
)
reference_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference_math)


def identity(value):
    """Independent typed key, with bool distinguished before integral equality."""
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, Integral):
        return "int", int(value)
    if isinstance(value, Real):
        assert math.isfinite(value)
        return "float", float(value).hex()
    assert isinstance(value, str) and value
    return "str", value


def ordered_groups(values):
    groups = {}
    for position, value in enumerate(values):
        groups.setdefault(identity(value), []).append(position)
    return [np.array(positions, dtype=int) for positions in groups.values()]


def ssu_frame(frame, roles=ROLES):
    """Explicit positive-cell declaration, independently assembled from this fixture."""
    names = ([roles["strata"]] if roles.get("strata") else []) + [roles["psu"], roles["ssu_strata"]]
    cells = {}
    for row in frame.to_dict("records"):
        key = tuple(identity(row[name]) for name in names)
        record = {name: row[name] for name in names}
        record[roles["population_ssu"]] = row[roles["population_ssu"]]
        assert key not in cells or cells[key] == record
        cells.setdefault(key, record)
    return list(cells.values())


def declare(frame, declared_frame=None, **options):
    import openecon as oe

    return oe.survey_stratified_three_stage_design(
        frame,
        **ROLES,
        ssu_frame=ssu_frame(frame) if declared_frame is None else declared_frame,
        **options,
    )


def fixture():
    """Multiple cells, unequal f2/f3, repeated labels and noncontiguous physical rows."""
    rng = np.random.default_rng(609616)
    rows = []
    for h, population in zip(("north", "south", "east"), (5, 8, 7)):
        for p in range(3):
            for g in range(2):
                m = 2 + int(p == 1 and g == 0)
                for s in range(m):
                    leaf_count = 2 + (p + s + g) % 3
                    for j in range(leaf_count):
                        x, z, noise = rng.normal(size=3)
                        rows.append(
                            {
                                "h": h,
                                "p": p,
                                "g": ("urban", "rural")[g],
                                "s": 10 * g + s,
                                "j": j,
                                "N": population,
                                "M": m + 2 + p + g,
                                "L": leaf_count + 2 + (p + s + g) % 2,
                                "x": x,
                                "z": z,
                                "y": 1.4 + 0.65 * x - 0.4 * z + noise,
                                "a": 2.0 + 0.3 * x + noise / 4,
                                "den": 3.5 + abs(z) + s / 3,
                                "den2": 2.5 + abs(x) + s / 4,
                                "binary": (j + p + s + g) % 2,
                                "count": (0, 2, 1, 4)[j] + int(p == 2),
                                "category": "a" if j % 2 else "b",
                                "domain": int(not (h == "east" and p == 2)),
                            }
                        )
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.index = np.resize([37, -4, 37, 2], len(frame))
    return frame, declare(frame)


def geometry(frame, roles=ROLES, declared_frame=None):
    """Original-row grouping and each declared cell's independent sampling fraction."""
    declared_frame = ssu_frame(frame, roles) if declared_frame is None else declared_frame
    parent_names = ([roles["strata"]] if roles.get("strata") else []) + [
        roles["psu"],
        roles["ssu_strata"],
    ]
    declared = {
        tuple(identity(row[name]) for name in parent_names): int(row[roles["population_ssu"]])
        for row in declared_frame
    }
    h_values = (
        frame[roles["strata"]].tolist() if roles.get("strata") else ["unstratified"] * len(frame)
    )
    weights = np.zeros(len(frame), dtype=float)
    strata = []
    for hrows in ordered_groups(h_values):
        p_groups = ordered_groups(frame[roles["psu"]].iloc[hrows].tolist())
        f1 = len(p_groups) / int(frame[roles["population_psu"]].iloc[hrows[0]])
        psus = []
        for relative_p in p_groups:
            prows = hrows[relative_p]
            cells = []
            for relative_g in ordered_groups(frame[roles["ssu_strata"]].iloc[prows].tolist()):
                crows = prows[relative_g]
                raw = frame.iloc[crows[0]]
                key = tuple(identity(raw[name]) for name in parent_names)
                s_groups = ordered_groups(frame[roles["ssu"]].iloc[crows].tolist())
                f2 = len(s_groups) / declared[key]
                ssus = []
                for relative_s in s_groups:
                    rows = crows[relative_s]
                    f3 = len(rows) / int(frame[roles["population_tsu"]].iloc[rows[0]])
                    weights[rows] = 1 / (f1 * f2 * f3)
                    ssus.append((rows, f3))
                cells.append((ssus, f2))
            psus.append(cells)
        strata.append((psus, f1))
    return weights, strata


def covariance_components(weighted_rows, strata):
    """V2 centers within each cell; V3 carries its own f1*f2_cell prefix."""
    weighted_rows = np.asarray(weighted_rows, dtype=float)
    parts = [np.zeros((weighted_rows.shape[1],) * 2) for _ in range(3)]
    for psus, f1 in strata:
        psu_totals = []
        for cells in psus:
            total = np.zeros(weighted_rows.shape[1])
            for ssus, f2 in cells:
                ssu_totals = np.array([weighted_rows[positions].sum(0) for positions, _ in ssus])
                total += ssu_totals.sum(0)
                if f2 < 1:
                    centered = ssu_totals - ssu_totals.mean(0)
                    parts[1] += (
                        f1 * (1 - f2) * len(ssus) / (len(ssus) - 1) * (centered.T @ centered)
                    )
                for positions, f3 in ssus:
                    if f3 < 1:
                        centered = weighted_rows[positions] - weighted_rows[positions].mean(0)
                        parts[2] += (
                            f1
                            * f2
                            * (1 - f3)
                            * len(positions)
                            / (len(positions) - 1)
                            * (centered.T @ centered)
                        )
            psu_totals.append(total)
        if f1 < 1:
            psu_totals = np.array(psu_totals)
            centered = psu_totals - psu_totals.mean(0)
            parts[0] += (1 - f1) * len(psus) / (len(psus) - 1) * (centered.T @ centered)
    return tuple(parts)


def specification(kind, intercept=True):
    if kind == "ratio":
        return {"args": [["y", "a"], ["den", "den2"]]}
    if kind == "proportion":
        return {"args": ["category"], "categories": ["a", "b", "absent"]}
    if kind in DESCRIPTIVE:
        return {"args": [["y", "a"]]}
    outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[kind]
    return {"args": [outcome, ["x", "z"]], "intercept": intercept}


def expected_case(frame, kind, *, roles=ROLES, declared_frame=None, spec=None, options=None):
    spec = specification(kind) if spec is None else spec
    options = {**(options or {}), **{key: value for key, value in spec.items() if key != "args"}}
    args = spec["args"]
    members = (
        frame[options["domain"]].to_numpy(dtype=bool)
        if options.get("domain")
        else np.ones(len(frame), bool)
    )
    weights, strata = geometry(frame, roles, declared_frame)
    if kind in DESCRIPTIVE:
        names = [args[0]] if isinstance(args[0], str) else list(args[0])
        denominators = list(args[1]) if kind == "ratio" else []
        selected = members & frame[names + denominators].notna().all(1).to_numpy()
        if kind == "proportion":
            categories = options["categories"]
            values = np.array(
                [
                    [identity(value) == identity(category) for category in categories]
                    if selected[position]
                    else [False] * len(categories)
                    for position, value in enumerate(frame[names[0]])
                ],
                dtype=float,
            )
            labels = [
                f"{names[0]}[{i}]={identity(c)[0]}:{identity(c)[1]}"
                for i, c in enumerate(categories)
            ]
        else:
            values = frame[names].fillna(0).to_numpy(dtype=float)
            labels = [f"{y}/{x}" for y, x in zip(names, denominators)] if kind == "ratio" else names
        values[~selected] = 0
        effective = weights * selected
        denominator = (
            frame[denominators].fillna(0).to_numpy(dtype=float)
            if denominators
            else np.ones_like(values)
        )
        if kind == "ratio":
            denominator[~selected] = 0
        numerator = effective @ values
        if kind == "total":
            estimate, influence = numerator, values
        else:
            normalizer = effective @ denominator
            estimate = numerator / normalizer
            influence = selected[:, None] * (values - denominator * estimate) / normalizer
        weighted_rows = weights[:, None] * influence
        primitive = {
            "primitive_values": values,
            "primitive_denominators": denominator,
            "row_influences": weighted_rows,
        }
    else:
        outcome, regressors = args
        selected = members & frame[[outcome, *regressors]].notna().all(1).to_numpy()
        X = frame[regressors].fillna(0).to_numpy(dtype=float)
        intercept = options.get("intercept", True)
        labels = (["_cons"] if intercept else []) + list(regressors)
        if intercept:
            X = np.column_stack((np.ones(len(frame)), X))
        x, y, w = X[selected], frame[outcome].to_numpy(dtype=float)[selected], weights[selected]
        if kind == "regress":
            estimate = np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y))
        else:

            def objective(beta):
                value, scores, _ = reference_math.likelihood(kind, beta, x, y, w)
                return value / w.sum(), -scores.sum(0) / w.sum()

            fit = optimize.minimize(
                objective,
                np.zeros(x.shape[1]),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-12, "maxiter": 1000},
            )
            root = optimize.root(
                lambda beta: reference_math.likelihood(kind, beta, x, y, w)[1].sum(0),
                fit.x,
                jac=lambda beta: -reference_math.likelihood(kind, beta, x, y, w)[2],
                tol=1e-12,
            )
            estimate = root.x
            assert np.max(np.abs(objective(estimate)[1])) < 1e-10
        _, scores, bread = reference_math.likelihood(kind, estimate, x, y, w)
        full_scores = np.zeros((len(frame), len(estimate)))
        full_scores[selected] = scores
        weighted_rows = np.linalg.solve(bread, full_scores.T).T
        normalization = selected.sum() / w.sum()
        primitive = {
            "primitive_X": x,
            "primitive_y": y,
            "bread": bread * normalization,
            "row_scores": full_scores * normalization,
        }
    parts = covariance_components(weighted_rows, strata)
    return {
        "estimates": estimate,
        "covariance": sum(parts),
        "stage1_covariance": parts[0],
        "stage2_covariance": parts[1],
        "stage3_covariance": parts[2],
        "selected": selected,
        "members": members,
        "weights": weights,
        "labels": labels,
        "primitives": primitive,
        "df": sum(len(psus) for psus, _ in strata) - len(strata),
        "alpha": options.get("alpha", 0.1),
        "null": np.broadcast_to(options.get("null", 0.0), len(estimate)),
    }


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
    fn = getattr(oe, ("survey_three_stage_" if old else "survey_stratified_three_stage_") + kind)
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
def test_exhaustive_three_stage_cell_stratified_total_has_unbiased_full_variance(census_stages):
    N = 3
    n = N if 1 in census_stages else 2
    populations = {
        (p, g, s): np.array(
            [
                [2 + 3 * p + 7 * g + s * s + j * j, 3 - p + 5 * g + 2 * s + (-1) ** j]
                for j in range(3 if (p, g, s) == (0, 1, 0) else 2)
            ],
            float,
        )
        for p in range(N)
        for g, M in enumerate((2, 3))
        for s in range(M)
    }
    truth = np.vstack(list(populations.values())).sum(0)
    estimates, variances, probabilities = [], [], []
    for psus in combinations(range(N), n):
        cells = [(p, g) for p in psus for g in range(2)]
        choices = [
            list(combinations(range((2, 3)[g]), (2, 3)[g] if 2 in census_stages else 2))
            for _, g in cells
        ]
        for drawn_ssus in product(*choices):
            groups = [(p, g, s) for (p, g), selected in zip(cells, drawn_ssus) for s in selected]
            leaves = [
                list(
                    combinations(
                        range(len(populations[group])),
                        len(populations[group]) if 3 in census_stages else 2,
                    )
                )
                for group in groups
            ]
            probability = (
                1
                / math.comb(N, n)
                / math.prod(len(x) for x in choices)
                / math.prod(len(x) for x in leaves)
            )
            for drawn_tsus in product(*leaves):
                frame = pd.DataFrame(
                    [
                        {
                            "h": "a",
                            "p": p,
                            "g": g,
                            "s": 10 * g + s,
                            "j": j,
                            "N": N,
                            "M": (2, 3)[g],
                            "L": len(populations[p, g, s]),
                            "y": populations[p, g, s][j, 0],
                            "a": populations[p, g, s][j, 1],
                        }
                        for (p, g, s), chosen in zip(groups, drawn_tsus)
                        for j in chosen
                    ]
                )
                state = call(frame, declare(frame), "total")
                estimates.append(state.estimates)
                variances.append(state.covariance)
                probabilities.append(probability)
    probabilities = np.array(probabilities)
    estimates = np.array(estimates)
    variances = np.array(variances)
    assert probabilities.sum() == pytest.approx(1.0, abs=2e-15)
    exact = np.einsum("n,ni,nj->ij", probabilities, estimates - truth, estimates - truth)
    np.testing.assert_allclose(probabilities @ estimates, truth, rtol=1e-14, atol=2e-12)
    np.testing.assert_allclose(
        np.einsum("n,nij->ij", probabilities, variances), exact, rtol=4e-14, atol=4e-12
    )
    if not census_stages:
        assert len(estimates) == 51 and exact[0, 0] > 0 and abs(exact[0, 1]) > 0
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


def test_cell_centering_does_not_invent_variance_from_different_cell_means():
    rows = [
        dict(h="a", p=p, g=g, s=10 * g + s, j=j, N=3, M=(2, 3)[g], L=2, y=(4, 40)[g], a=(2, -8)[g])
        for p in range(2)
        for g in range(2)
        for s in range(2)
        for j in range(2)
    ]
    frame = pd.DataFrame(rows)
    state = call(frame, declare(frame), "total")
    np.testing.assert_array_equal(state.covariance, np.zeros((2, 2)))
    weights, strata = geometry(frame)
    weighted = weights[:, None] * frame[["y", "a"]].to_numpy()
    for psus, _ in strata:
        for cells in psus:
            pooled = np.array([weighted[pos].sum(0) for ssus, _ in cells for pos, _ in ssus])
            assert np.linalg.norm(pooled - pooled.mean(0)) > 0


def test_third_prefix_is_each_cells_fraction_not_a_pooled_psu_fraction():
    frame, _ = fixture()
    expected = expected_case(frame, "total")
    weighted = expected["primitives"]["row_influences"]
    _, strata = geometry(frame)
    wrong = np.zeros((2, 2))
    for psus, f1 in strata:
        for cells in psus:
            pooled = sum(len(ssus) for ssus, _ in cells) / sum(len(ssus) / f2 for ssus, f2 in cells)
            for ssus, _ in cells:
                for rows, f3 in ssus:
                    centered = weighted[rows] - weighted[rows].mean(0)
                    wrong += (
                        f1
                        * pooled
                        * (1 - f3)
                        * len(rows)
                        / (len(rows) - 1)
                        * (centered.T @ centered)
                    )
    assert not np.allclose(wrong, expected["stage3_covariance"], rtol=1e-5, atol=1e-8)
    state = call(frame, declare(frame), "total")
    reference_math.numerical(
        state.metadata["stage3_covariance"], expected["stage3_covariance"], "cell third-prefix"
    )


@pytest.mark.parametrize("kind", GATES)
def test_one_cell_per_psu_reduces_exactly_to_existing_three_stage(kind):
    import openecon as oe

    frame, _ = fixture()
    frame["g"] = "only"
    frame["M"] = frame.groupby(["h", "p"], sort=False).s.transform("nunique") + 3
    new = call(frame, declare(frame), kind)
    old = call(
        frame,
        oe.survey_three_stage_design(
            frame, **{k: v for k, v in ROLES.items() if k != "ssu_strata"}
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
    names = {1: ["h"], 2: ["h", "p", "g"], 3: ["h", "p", "g", "s"]}
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
