"""Independent conditional-stratum math, exact randomization and old-API reductions."""

from itertools import combinations, product
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load("fully_four_example", "docs/examples/survey_fully_stratified_four_stage_eight.py")
reference = load(
    "fully_four_reference", "scripts/verify_survey_fully_stratified_four_stage_oracles.py"
)
KINDS = reference.CASES
CASES = {
    "mean": {"args": [["y", "x"]]},
    "total": {"args": [["y", "x"]]},
    "ratio": {"args": [["y", "x"], ["d", "d2"]]},
    "proportion": {"args": ["cgroup"], "categories": ["a", "b", "absent"]},
    "regress": {"args": ["y", ["x", "z"]]},
    "logit": {"args": ["b", ["x", "z"]]},
    "probit": {"args": ["b", ["x", "z"]]},
    "poisson": {"args": ["c", ["x", "z"]]},
}


def call(frame, roles, kind, **options):
    import openecon as oe

    design = oe.survey_fully_stratified_four_stage_design(frame, **roles)
    spec = CASES[kind]
    options = {
        "missing": "drop",
        "alpha": 0.05,
        **options,
        **{k: v for k, v in spec.items() if k != "args"},
    }
    if kind in reference.FAMILIES:
        options.update(tolerance=1e-11, max_iter=100)
    return getattr(oe, "survey_fully_stratified_four_stage_" + kind)(
        frame, design, *spec["args"], **options
    )


def assert_expected(state, expected):
    estimate = state.estimates if hasattr(state, "estimates") else state.coefficients
    reference.numerical(estimate, expected["estimates"], "estimate")
    reference.numerical(state.covariance, expected["covariance"], "full covariance")
    for key in ("stage1_covariance", "stage2_covariance", "stage3_covariance", "stage4_covariance"):
        reference.numerical(state.metadata[key], expected[key], key)
    for key, value in expected["primitives"].items():
        reference.numerical(state.metadata[key], value, key)
    reference.numerical(
        state.design.validation.weights, expected["weights"], "cell-specific weights"
    )
    assert state.metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
    assert (
        state.metadata["out_of_domain_positions"] == np.flatnonzero(~expected["members"]).tolist()
    )
    assert (
        state.metadata["outcome_exclusions"]
        == np.flatnonzero(expected["members"] & ~expected["selected"]).tolist()
    )
    table = state.to_frame()
    assert list(table.index) == expected["labels"] and state.df == expected["df"]
    for row, wanted in zip(
        table.astype(object).where(pd.notna(table), None).values.tolist(),
        reference.expected_table(expected),
        strict=True,
    ):
        for actual, value in zip(row, wanted, strict=True):
            if value is None:
                assert actual is None
            else:
                reference.numerical(actual, value, "reference t inference")
    assert type(state).model_validate_json(state.model_dump_json()).model_dump(
        mode="json"
    ) == state.model_dump(mode="json")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["full", "domain", "all-census", "df-zero"])
def test_all_eight_conditional_strata_full_covariance_and_inference(kind, mode):
    frame, roles = example.fixture()
    if mode == "df-zero":
        frame = frame[frame.p == 0].copy()
        frame["N"] = 1
        roles = example.roles(frame)
    if mode == "all-census":
        frame["N"] = 2
        roles = example.roles(frame, census=(2, 3, 4))
    options = {"missing": "drop", "alpha": 0.05, "domain": "domain" if mode == "domain" else None}
    state = call(frame, roles, kind, **options)
    expected = reference.expected_case(frame, kind, roles=roles, spec=CASES[kind], options=options)
    assert_expected(state, expected)
    if mode == "all-census":
        assert np.count_nonzero(state.covariance) == 0
    if mode == "df-zero":
        assert state.df == 0
        assert state.to_frame().p_value.isna().all()
    if mode == "domain":
        assert state.metadata["n_design"] == 256
        assert state.design.validation.n_psu == 4
        assert all(
            row == [0.0] * len(state.labels)
            for i, row in enumerate(
                state.metadata["row_scores" if kind in reference.FAMILIES else "row_influences"]
            )
            if frame.iloc[i].domain == 0
        )


@pytest.mark.parametrize("kind", KINDS)
def test_exact_existing_four_stage_reduction_with_one_lower_stratum_per_parent(kind):
    import openecon as oe

    frame, _ = example.fixture()
    frame = frame[(frame.g == 0) & (frame.q == 0) & (frame.r == 0)].copy()
    # Each terminal cell now has two physical rows, with one stratum in each parent.
    roles = example.roles(frame)
    old_roles = {
        k: v
        for k, v in roles.items()
        if k
        not in ("ssu_strata", "tsu_strata", "fsu_strata", "ssu_frame", "tsu_frame", "fsu_frame")
    }
    old_roles.update(population_ssu="M", population_tsu="L", population_fsu="K")
    old_design = oe.survey_four_stage_design(frame, **old_roles)
    new = call(frame, roles, kind)
    spec = CASES[kind]
    options = {"missing": "drop", "alpha": 0.05, **{k: v for k, v in spec.items() if k != "args"}}
    if kind in reference.FAMILIES:
        options.update(tolerance=1e-11, max_iter=100)
    old = getattr(oe, "survey_four_stage_" + kind)(frame, old_design, *spec["args"], **options)
    reference.numerical(new.covariance, old.covariance, "old full covariance reduction")
    np.testing.assert_allclose(
        new.to_frame().to_numpy(), old.to_frame().to_numpy(), rtol=3e-8, atol=2e-10, equal_nan=True
    )


def population(stage=0, path=(), active=True):
    if stage == 4:
        nums = [u for u in path[1::2]]
        return np.array(
            [
                2 + sum((i + 2) * u * u for i, u in enumerate(nums)),
                3 + nums[0] - 2 * nums[1] + 5 * nums[2] - nums[3],
            ],
            float,
        )
    strata = (
        [(0, 3)] if stage == 0 else [("active", 3), ("census", 1)] if active else [("constant", 1)]
    )
    return [
        (
            g,
            [
                (u, population(stage + 1, path + (g, u), active and g in (0, "active") and u == 0))
                for u in range(N)
            ],
        )
        for g, N in strata
    ]


def leaves(node):
    if isinstance(node, np.ndarray):
        return [node]
    return [v for _, units in node for _, child in units for v in leaves(child)]


def worlds(node, census, stage=0, path=(), declarations=(), first_N=None):
    if stage == 4:
        h, p, g, s, q, t, r, f = path
        row = dict(h=h, p=p, g=g, s=s, q=q, t=t, r=r, f=f, N=first_N, y=node[0], x=node[1])
        return [([row], list(declarations), 1.0)]
    strata_worlds = []
    for g, units in node:
        N = len(units)
        n = N if stage + 1 in census else min(2, N)
        choices = list(combinations(range(N), n))
        out = []
        for chosen in choices:
            declaration = declarations
            if stage:
                fields = (
                    "stratum",
                    "psu",
                    "ssu_stratum",
                    "ssu",
                    "tsu_stratum",
                    "tsu",
                    "fsu_stratum",
                )[: 2 * stage + 1]
                declaration = declarations + (
                    (
                        stage,
                        dict(zip(fields, path + (g,)))
                        | {("population_ssu", "population_tsu", "population_fsu")[stage - 1]: N},
                    ),
                )
            options = [
                worlds(
                    units[i][1],
                    census,
                    stage + 1,
                    path + (g, units[i][0]),
                    declaration,
                    N if stage == 0 else first_N,
                )
                for i in chosen
            ]
            for selected in product(*options):
                out.append(
                    (
                        [r for rows, _, _ in selected for r in rows],
                        [d for _, ds, _ in selected for d in ds],
                        np.prod([p for _, _, p in selected]) / len(choices),
                    )
                )
        strata_worlds.append(out)
    return [
        (
            [r for rows, _, _ in selected for r in rows],
            [d for _, ds, _ in selected for d in ds],
            np.prod([p for _, _, p in selected]),
        )
        for selected in product(*strata_worlds)
    ]


@pytest.mark.parametrize("mask", range(16))
def test_exact_four_selection_randomization_with_separate_lower_strata(mask):
    import openecon as oe

    census = {i + 1 for i in range(4) if mask & (1 << i)}
    tree = population()
    total = sum(leaves(tree), np.zeros(2))
    samples = worlds(tree, census)
    average = np.zeros(2)
    reported = np.zeros((2, 2))
    actual = np.zeros((2, 2))
    for rows, declarations, probability in samples:
        frame = pd.DataFrame(rows)
        roles = dict(
            psu="p",
            ssu="s",
            tsu="t",
            fsu="f",
            strata="h",
            population_psu="N",
            ssu_strata="g",
            tsu_strata="q",
            fsu_strata="r",
        )
        for stage, key in enumerate(("ssu_frame", "tsu_frame", "fsu_frame"), 1):
            unique = {
                json.dumps(record, sort_keys=True): record
                for level, record in declarations
                if level == stage
            }
            roles[key] = list(unique.values())
        state = oe.survey_fully_stratified_four_stage_total(
            frame, oe.survey_fully_stratified_four_stage_design(frame, **roles), ["y", "x"]
        )
        estimate = np.array(state.estimates)
        average += probability * estimate
        reported += probability * np.array(state.covariance)
        actual += probability * np.outer(estimate - total, estimate - total)
        for stage in census:
            assert np.count_nonzero(state.metadata[f"stage{stage}_covariance"]) == 0
    np.testing.assert_allclose(sum(p for _, _, p in samples), 1.0, atol=1e-14)
    np.testing.assert_allclose(average, total, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(reported, actual, rtol=1e-12, atol=1e-10)


def test_complete_example_and_independent_native_payload_oracle():
    example.display = lambda _: None
    namespace = example.main()
    payload = {k: namespace[k] for k in ("states", "poststates", "oracle_inputs")}
    assert reference.verify_payload(json.loads(json.dumps(payload, allow_nan=False)))["cases"] == 8
