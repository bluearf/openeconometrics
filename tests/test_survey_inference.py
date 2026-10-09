"""Independent design/replicate oracles, inference and saved-state failures."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def fixture(*, fpc=False, equal=False):
    frame = pd.DataFrame(
        {
            "w": [2.0] * 8 if equal else [2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0],
            "p": [1, 1, 2, 2, 1, 1, 2, 2],
            "h": ["a"] * 4 + ["b"] * 4,
            "y": [1.0, 3.0, 2.0, 5.0, 6.0, 8.0, 7.0, 11.0],
            "x": [2.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 6.0],
            "domain": [1, 1, 1, 1, 0, 0, 0, 0],
            "category": ["a", "a", "a", "b", "b", "a", "b", "b"],
        }
    )
    if fpc:
        frame["N"] = [4] * 4 + [8] * 4
    return frame, oe.survey_design(
        frame, weights="w", psu="p", strata="h", fpc="N" if fpc else None
    )


def reference(
    frame,
    target,
    *,
    columns=("y", "x"),
    denominator=("x", "y"),
    domain=False,
    weights=None,
    categories=None,
):
    """NumPy score and full design covariance; production internals are not used."""
    w = frame.w.to_numpy() if weights is None else np.asarray(weights)
    membership = frame.domain.to_numpy().astype(bool) if domain else np.ones(len(frame), dtype=bool)
    valid = (
        frame[list(columns) + list(denominator) if target == "ratio" else list(columns)]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    members = membership & valid
    if target == "proportion":
        y = np.column_stack([frame.category.to_numpy() == v for v in categories]).astype(float)
    else:
        y = frame[list(columns)].fillna(0).to_numpy()
    effective = w * members
    numerator = effective @ y
    if target == "total":
        estimate, score = numerator, members[:, None] * y
    else:
        x = frame[list(denominator)].fillna(0).to_numpy() if target == "ratio" else np.ones_like(y)
        den = effective @ x
        estimate = numerator / den
        score = members[:, None] * (y - x * estimate) / den
    influence = w[:, None] * score
    covariance = np.zeros((y.shape[1], y.shape[1]))
    for h in pd.unique(frame.h):
        mask = frame.h.to_numpy() == h
        psus = pd.unique(frame.loc[mask, "p"])
        totals = np.array([influence[mask & (frame.p.to_numpy() == p)].sum(0) for p in psus])
        correction = 1.0 - len(psus) / float(frame.loc[mask, "N"].iloc[0]) if "N" in frame else 1.0
        if correction:
            centred = totals - totals.mean(0)
            covariance += correction * len(psus) / (len(psus) - 1) * centred.T @ centred
    return estimate, covariance


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("fpc", [False, True])
def test_joint_taylor_oracle_and_inference(kind, domain, fpc):
    frame, design = fixture(fpc=fpc)
    options = {"domain": "domain" if domain else None, "alpha": 0.1}
    if kind == "ratio":
        actual = oe.survey_ratio(frame, design, ["y", "x"], ["x", "y"], **options)
    elif kind == "proportion":
        actual = oe.survey_proportion(
            frame, design, "category", categories=["a", "b", "absent"], **options
        )
    else:
        actual = getattr(oe, "survey_" + kind)(frame, design, ["y", "x"], **options)
    expected, cov = reference(
        frame,
        kind,
        domain=domain,
        columns=("category",) if kind == "proportion" else ("y", "x"),
        categories=["a", "b", "absent"],
    )
    np.testing.assert_allclose(actual.estimates, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual.covariance, cov, rtol=1e-12, atol=1e-13)
    table = actual.to_frame()
    assert actual.df == 2
    se = np.sqrt(cov.diagonal())
    np.testing.assert_allclose(table.std_error, se, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(table.ci_low, expected - stats.t.ppf(0.95, 2) * se)
    np.testing.assert_allclose(table.ci_high, expected + stats.t.ppf(0.95, 2) * se)
    for i, value in enumerate(expected):
        if se[i] > 1e-14:
            assert table.iloc[i].p_value == pytest.approx(
                2 * stats.t.sf(abs(value / se[i]), 2), rel=1e-11
            )
    assert actual.metadata["n_design"] == 8
    assert len(actual.metadata["psu_influence_sums"]) == 4


def test_hand_calculated_full_covariance_and_weight_scale():
    frame, design = fixture(equal=True)
    result = oe.survey_total(frame, design, ["y", "x"])
    np.testing.assert_allclose(result.estimates, [86.0, 54.0])
    np.testing.assert_allclose(result.covariance, [[100.0, 80.0], [80.0, 100.0]])
    np.testing.assert_allclose(
        oe.survey_mean(frame, design, ["y", "x"]).covariance,
        np.array([[100.0, 80.0], [80.0, 100.0]]) / 256.0,
    )
    scaled = frame.assign(w=frame.w * 7)
    d2 = oe.survey_design(scaled, weights="w", psu="p", strata="h")
    total = oe.survey_total(scaled, d2, ["y", "x"])
    np.testing.assert_allclose(total.estimates, np.array(result.estimates) * 7)
    np.testing.assert_allclose(total.covariance, np.array(result.covariance) * 49)
    for fn in [oe.survey_mean, oe.survey_ratio]:
        args = (["y"], ["x"]) if fn == oe.survey_ratio else (["y"],)
        a, b = fn(frame, design, *args), fn(scaled, d2, *args)
        np.testing.assert_allclose(a.estimates, b.estimates)
        np.testing.assert_allclose(a.covariance, b.covariance)


def test_domain_psu_universe_and_missing_policy():
    frame, design = fixture()
    frame.loc[4:, "y"] = np.nan
    actual = oe.survey_mean(frame, design, "y", domain="domain")
    assert actual.metadata["sample_positions"] == [0, 1, 2, 3]
    assert actual.df == 2
    assert actual.metadata["psu_influence_sums"][2:] == [[0.0], [0.0]]
    frame.loc[0, "y"] = np.nan
    with pytest.raises(AnalysisError, match="Missing in-domain"):
        oe.survey_mean(frame, design, "y", domain="domain")
    dropped = oe.survey_mean(frame, design, "y", domain="domain", missing="drop")
    assert dropped.metadata["sample_positions"] == [1, 2, 3]
    assert dropped.metadata["outcome_exclusions"] == [0]
    expected, covariance = reference(frame, "mean", columns=("y",), domain=True)
    np.testing.assert_allclose(dropped.estimates, expected)
    np.testing.assert_allclose(dropped.covariance, covariance)
    assert dropped.df == 2


def test_srs_census_and_singleton_certainty():
    frame = pd.DataFrame({"w": [3.0] * 5, "p": list(range(5)), "y": [1.0, 2.0, 4.0, 6.0, 11.0]})
    design = oe.survey_design(frame, weights="w", psu="p")
    actual = oe.survey_mean(frame, design, "y", deff=True)
    assert actual.covariance[0][0] == pytest.approx(np.var(frame.y, ddof=1) / 5)
    assert actual.metadata["design_effect"] == pytest.approx([1.0])
    frame["N"] = 5
    census = oe.survey_mean(frame, oe.survey_design(frame, weights="w", psu="p", fpc="N"), "y")
    assert census.covariance == ((0.0,),)
    assert pd.isna(census.to_frame().iloc[0].p_value)
    assert census.to_frame().iloc[0].ci_low == census.estimates[0]
    frame = pd.DataFrame({"w": [1.0, 1.0], "p": [1, 1], "N": [1, 1], "y": [2.0, 4.0]})
    d = oe.survey_design(frame, weights="w", psu="p", fpc="N", singleton="certainty")
    r = oe.survey_mean(frame, d, "y")
    assert r.df == 0 and r.covariance == ((0.0,),) and r.estimates == (3.0,)


def replicated(frame, rho=0.0):
    rows = []
    for choices in product([-1.0, 1.0], repeat=len(pd.unique(frame.h))):
        weights = frame.w.to_numpy().copy()
        for h, sign in zip(pd.unique(frame.h), choices):
            mask = frame.h.to_numpy() == h
            pair = pd.unique(frame.loc[mask, "p"])
            weights[mask & (frame.p.to_numpy() == pair[0])] *= 1.0 + (1.0 - rho) * sign
            weights[mask & (frame.p.to_numpy() == pair[1])] *= 1.0 - (1.0 - rho) * sign
        rows.append(weights)
    return pd.DataFrame(np.array(rows).T, columns=[f"rep-{i}" for i in range(len(rows))])


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.8])
@pytest.mark.parametrize("center", ["original", "replicate_mean"])
def test_brr_fay_independent_enumeration(kind, rho, center):
    frame, design = fixture()
    supplied = replicated(frame, rho)
    options = {
        "target": kind,
        "centering": center,
        "categories": ["a", "b"] if kind == "proportion" else None,
        "denominators": ["x"] if kind == "ratio" else None,
    }
    outcomes = ["category"] if kind == "proportion" else ["y"]
    fn = oe.survey_fay if rho else oe.survey_brr
    if rho:
        options["rho"] = rho
    actual = fn(frame, design, outcomes, **options)
    provided = fn(frame, design, outcomes, replicate_weights=supplied, **options)
    estimates = np.array(
        [
            reference(
                frame,
                kind,
                columns=tuple(outcomes),
                denominator=("x",),
                categories=["a", "b"],
                weights=supplied[c],
            )[0]
            for c in supplied
        ]
    )
    theta = reference(
        frame, kind, columns=tuple(outcomes), denominator=("x",), categories=["a", "b"]
    )[0]
    centred = estimates - (theta if center == "original" else estimates.mean(0))
    covariance = centred.T @ centred / (len(estimates) * (1.0 - rho) ** 2)
    np.testing.assert_allclose(actual.covariance, covariance, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(provided.covariance, covariance, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(actual.estimates, theta)
    assert actual.metadata["replicate_count"] == 4
    assert actual.metadata["failed_replicates"] == []


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("center", ["original", "stratum_mean"])
@pytest.mark.parametrize("fpc", [False, True])
def test_jackknife_enumeration(kind, center, fpc):
    frame, design = fixture(fpc=fpc)
    outcomes = ["category"] if kind == "proportion" else ["y", "x"]
    den = ["x", "y"] if kind == "ratio" else None
    actual = oe.survey_jackknife(
        frame,
        design,
        outcomes,
        target=kind,
        denominators=den,
        categories=["a", "b"] if kind == "proportion" else None,
        centering=center,
    )
    cov = np.zeros((len(actual.labels), len(actual.labels)))
    theta = np.array(actual.estimates)
    for h in pd.unique(frame.h):
        mask = frame.h.to_numpy() == h
        psus = pd.unique(frame.loc[mask, "p"])
        estimates = []
        for p in psus:
            weights = frame.w.to_numpy().copy()
            weights[mask] *= len(psus) / (len(psus) - 1)
            weights[mask & (frame.p.to_numpy() == p)] = 0.0
            estimates.append(
                reference(
                    frame,
                    kind,
                    columns=tuple(outcomes),
                    denominator=tuple(den or ["x"]),
                    categories=["a", "b"],
                    weights=weights,
                )[0]
            )
        estimates = np.array(estimates)
        delta = estimates - (theta if center == "original" else estimates.mean(0))
        correction = 1.0 - len(psus) / float(frame.loc[mask, "N"].iloc[0]) if fpc else 1.0
        cov += correction * (len(psus) - 1) / len(psus) * delta.T @ delta
    np.testing.assert_allclose(actual.covariance, cov, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("center", ["original", "replicate_mean"])
def test_bootstrap_supplied_scale_full_covariance_and_df(center):
    frame, design = fixture()
    weights = replicated(frame)
    rscales = [1.0, 2.0, 3.0, 4.0]
    actual = oe.survey_bootstrap(
        frame,
        design,
        ["y", "x"],
        weights,
        justification="Enumerated Rao-Wu n_h-1 PSU draws; no row resampling.",
        scale=0.1,
        rscales=rscales,
        df=2,
        centering=center,
    )
    estimates = np.array([reference(frame, "mean", weights=weights[c])[0] for c in weights])
    delta = estimates - (np.array(actual.estimates) if center == "original" else estimates.mean(0))
    covariance = delta.T @ np.diag(np.array(rscales) * 0.1) @ delta
    np.testing.assert_allclose(actual.covariance, covariance, rtol=1e-12)
    assert actual.df == 2 and actual.metadata["rscales"] == rscales
    assert actual.metadata["replicate_ids"] == list(weights.columns)


def all_results():
    frame, design = fixture()
    return [
        oe.survey_mean(frame, design, ["y", "x"]),
        oe.survey_total(frame, design, ["y", "x"]),
        oe.survey_ratio(frame, design, ["y", "x"], ["x", "y"]),
        oe.survey_proportion(frame, design, "category", categories=["a", "b", "absent"]),
        oe.survey_brr(frame, design, ["y", "x"]),
        oe.survey_fay(frame, design, ["y", "x"]),
        oe.survey_jackknife(frame, design, ["y", "x"]),
        oe.survey_bootstrap(
            frame,
            design,
            ["y", "x"],
            replicated(frame),
            justification="Enumerated stratified two-PSU Rao-Wu weights",
            scale=0.25,
        ),
    ]


@pytest.mark.parametrize("i", range(8))
def test_json_restore_full_inference_latex_and_contrast(tmp_path, i):
    original = all_results()[i]
    path = tmp_path / "state.json"
    path.write_text(original.model_dump_json())
    restored = oe.SurveyResult.model_validate_json(path.read_text())
    assert restored == original
    pd.testing.assert_frame_equal(restored.to_frame(), original.to_frame())
    assert "estimate" in oe.to_latex(restored.to_frame())
    a = np.arange(1, len(original.labels) + 1, dtype=float)
    contrast = restored.contrast(a.tolist(), null=1.0)
    assert contrast.iloc[0].estimate == pytest.approx(a @ np.array(original.estimates))
    assert contrast.iloc[0].std_error ** 2 == pytest.approx(a @ np.array(original.covariance) @ a)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "missing",
        "boolean",
        "complex",
        "string",
        "duplicate",
        "bad_domain",
        "changed_design",
        "unknown_category",
        "denominator",
        "unsupported_deff",
        "nonscalar",
    ],
)
def test_explicit_target_failures(case):
    frame, design = fixture()

    def call():
        return oe.survey_mean(frame, design, "y")

    if case == "empty":
        frame["domain"] = 0

        def call():
            return oe.survey_mean(frame, design, "y", domain="domain")
    elif case == "missing":
        frame.loc[0, "y"] = np.nan
    elif case == "boolean":
        frame["y"] = frame.y.astype(bool)
    elif case == "complex":
        frame["y"] = frame.y.astype(complex)
    elif case == "string":
        frame["y"] = frame.y.astype(str)
    elif case == "duplicate":
        frame = pd.concat([frame, frame[["y"]]], axis=1)
    elif case == "bad_domain":
        frame.loc[0, "domain"] = 2

        def call():
            return oe.survey_mean(frame, design, "y", domain="domain")
    elif case == "changed_design":
        frame.loc[0, "w"] *= 2
    elif case == "unknown_category":

        def call():
            return oe.survey_proportion(frame, design, "category", categories=["a"])
    elif case == "denominator":
        frame["x"] = 0.0

        def call():
            return oe.survey_ratio(frame, design, "y", "x")
    elif case == "unsupported_deff":

        def call():
            return oe.survey_mean(frame, design, "y", deff=True)
    elif case == "nonscalar":
        frame["y"] = frame.y.astype(object)
        frame.at[0, "y"] = [1.0, 2.0]
    with pytest.raises(AnalysisError):
        call()


@pytest.mark.parametrize(
    "case",
    [
        "unbalanced",
        "row_resample",
        "negative",
        "nan",
        "zero_den",
        "bad_ids",
        "rho",
        "fpc",
        "count",
        "rscale",
        "df",
        "justification",
    ],
)
def test_replication_negative_guards(case):
    frame, design = fixture()
    weights = replicated(frame)

    def call():
        return oe.survey_brr(frame, design, "y", replicate_weights=weights)

    if case == "unbalanced":
        weights = weights.iloc[:, :3]
    elif case == "row_resample":
        weights.iloc[0, 0] += 1.0
    elif case == "negative":
        weights.iloc[0, 0] = -1.0
    elif case == "nan":
        weights.iloc[0, 0] = np.nan
    elif case == "zero_den":
        weights["empty"] = 0.0

        def call():
            return oe.survey_bootstrap(
                frame, design, "y", weights, justification="Declared PSU bootstrap", scale=0.2
            )
    elif case == "bad_ids":
        weights.columns = ["same"] * 4
    elif case == "rho":

        def call():
            return oe.survey_fay(frame, design, "y", rho=1.0)
    elif case == "fpc":
        frame, design = fixture(fpc=True)

        def call():
            return oe.survey_brr(frame, design, "y")
    elif case == "count":

        def call():
            return oe.survey_brr(frame, design, "y", replicates=3)
    elif case == "rscale":

        def call():
            return oe.survey_bootstrap(
                frame,
                design,
                "y",
                weights,
                justification="PSU bootstrap",
                scale=0.25,
                rscales=[1.0, -1.0, 1.0, 1.0],
            )
    elif case == "df":

        def call():
            return oe.survey_bootstrap(
                frame, design, "y", weights, justification="PSU bootstrap", scale=0.25, df=4
            )
    elif case == "justification":

        def call():
            return oe.survey_bootstrap(frame, design, "y", weights, justification="", scale=0.25)

    with pytest.raises(AnalysisError):
        call()


def test_domain_failed_replicates_exhaustive():
    frame, design = fixture()
    frame["domain"] = [1, 1, 0, 0, 0, 0, 0, 0]
    with pytest.raises(AnalysisError) as exc:
        oe.survey_brr(frame, design, "y", domain="domain")
    assert exc.value.code == "survey_replicate_failure"
    assert "sylvester-1" in str(exc.value) and "sylvester-3" in str(exc.value)
    assert (
        oe.survey_fay(frame, design, "y", domain="domain", rho=0.5).metadata["failed_replicates"]
        == []
    )


def test_state_tampering_and_category_sum():
    result = all_results()[0]
    raw = result.model_dump(mode="json")
    raw["covariance"][0][0] *= 2
    with pytest.raises(ValueError, match="integrity"):
        oe.SurveyResult.model_validate(raw)
    raw = result.model_dump(mode="json")
    raw["covariance"][0][0] = -1.0
    from openecon.econometrics.survey.common import digest

    raw["integrity_sha256"] = digest({k: v for k, v in raw.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="positive semidefinite"):
        oe.SurveyResult.model_validate(raw)
    result.metadata["missing"] = "changed"
    with pytest.raises(ValueError):
        result.to_frame()
    proportion = all_results()[3]
    np.testing.assert_allclose(np.array(proportion.covariance).sum(axis=0), 0.0, atol=1e-15)
    assert sum(proportion.estimates) == pytest.approx(1.0)
    assert proportion.contrast([1.0] * 3).iloc[0].std_error < 1e-15


def test_long_resident_linear_geometry_and_budget(monkeypatch):
    n = 40_001
    values = np.sin(np.arange(n) / 101.0) + 3.0
    frame = pd.DataFrame({"w": np.ones(n), "p": np.arange(n) % 17, "y": values})
    design = oe.survey_design(frame, weights="w", psu="p", max_memory_mb=128)
    actual = oe.survey_mean(frame, design, "y")
    assert len(actual.metadata["sample_positions"]) == n
    assert len(actual.metadata["psu_influence_sums"]) == 17
    assert actual.estimates[0] == pytest.approx(np.mean(values), rel=1e-13)
    totals = np.bincount(frame.p, weights=(values - values.mean()) / n)
    expected = 17 / 16 * np.sum((totals - totals.mean()) ** 2)
    assert actual.covariance[0][0] == pytest.approx(expected, rel=1e-11)
    from openecon.econometrics.survey import replication

    monkeypatch.setattr(replication, "MAX_CELLS", 3)
    small, d = fixture()
    with pytest.raises(AnalysisError, match="cells/work"):
        oe.survey_brr(small, d, "y")


def test_many_psu_supplied_validation_and_non_square_brr():
    n = 10003
    frame = pd.DataFrame({"w": np.ones(n), "p": np.arange(n), "y": np.arange(n, dtype=float)})
    design = oe.survey_design(frame, weights="w", psu="p")
    weights = pd.DataFrame({"first": np.ones(n), "second": np.full(n, 2.0)})
    result = oe.survey_bootstrap(
        frame,
        design,
        "y",
        weights,
        justification="Declared first-stage PSU multiplicities; admission exercise",
        scale=0.5,
    )
    assert result.estimates[0] == pytest.approx(frame.y.mean())
    assert result.covariance == ((0.0,),)
    from openecon.econometrics.survey.replication import hadamard_signs

    signs = hadamard_signs(4096, 2).numpy()
    assert signs.shape == (4096, 2)
    np.testing.assert_array_equal(signs.T @ signs, 4096 * np.eye(2))


def test_rehashed_replica_covariance_and_invalid_centering_are_rejected():
    from openecon.econometrics.survey.common import digest

    result = all_results()[4]
    raw = result.model_dump(mode="json")
    raw["covariance"][0][0] *= 2.0
    raw["integrity_sha256"] = digest({k: v for k, v in raw.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="differs from complete replicate"):
        oe.SurveyResult.model_validate(raw)
    frame, design = fixture()
    with pytest.raises(AnalysisError):
        oe.survey_brr(frame, design, "y", centering=[])
    assert oe.survey_fay(frame, design, "y", rho=0.0).method == "fay"


def test_public_help_capability_and_lazy_import_contract():
    import json
    import runpy
    from pathlib import Path

    root = Path(__file__).parents[1]
    decode = runpy.run_path(str(root / "scripts/generate_editor_api.py"))["decode_catalog"]
    catalog = {v["name"]: v for v in decode(json.loads((root / "web/src/editor-api.json").read_text()))}
    for name in (
        "survey_mean",
        "survey_total",
        "survey_ratio",
        "survey_proportion",
        "survey_brr",
        "survey_fay",
        "survey_jackknife",
        "survey_bootstrap",
    ):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.SurveyResult"
    assert catalog["openecon.SurveyResult.contrast"]["returns"] == "openecon.DataFrame"
    assert catalog["openecon.SurveyResult.to_frame"]["returns"] == "openecon.DataFrame"
    cap = oe.capabilities()["survey_inference"]
    assert cap["dataset_support"] is False
    assert cap["stata_parity_validated"] is False
    assert "replica" in cap["persistence"]


def test_supplied_brr_large_strata_balance_admission():
    n = 2002
    frame = pd.DataFrame(
        {
            "w": np.ones(n),
            "p": [1, 2] * (n // 2),
            "h": np.repeat(np.arange(n // 2), 2),
            "y": np.ones(n),
        }
    )
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    weights = pd.DataFrame({"first": [2.0, 0.0] * (n // 2), "second": [0.0, 2.0] * (n // 2)})
    with pytest.raises(AnalysisError, match="R > stochastic strata"):
        oe.survey_brr(frame, design, "y", replicate_weights=weights)


def test_supplied_brr_balance_work_is_budgeted(monkeypatch):
    import openecon.econometrics.survey.replication as replication

    frame, design = fixture()
    monkeypatch.setattr(replication, "MAX_BALANCE_WORK", 15)
    with pytest.raises(AnalysisError, match="validation work"):
        oe.survey_brr(frame, design, "y", replicate_weights=replicated(frame))
