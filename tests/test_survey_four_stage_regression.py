"""Independent four-level score covariance and bounded native model admission."""

import numpy as np
import pytest
from scipy import special, stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey import four_stage_regression as fitting

from test_survey_four_stage_regression_state import FAMILIES, declaration, fixture


def independent_parts(frame, influences):
    """Center every actual sampling cell from raw role columns, without saved geometry."""
    width = influences.shape[1]
    parts = [np.zeros((width, width)) for _ in range(4)]

    def add(stage, block, fraction, prefix):
        if fraction < 1:
            centered = block - block.mean(axis=0)
            parts[stage] += (
                prefix * (1 - fraction) * len(block) / (len(block) - 1) * (centered.T @ centered)
            )

    for _, stratum in frame.groupby("h", sort=False):
        psus = list(stratum.groupby("p", sort=False))
        f1 = len(psus) / stratum.N.iloc[0]
        add(0, np.array([influences[p.index].sum(0) for _, p in psus]), f1, 1.0)
        for _, psu in psus:
            ssus = list(psu.groupby("s", sort=False))
            f2 = len(ssus) / psu.M.iloc[0]
            add(1, np.array([influences[s.index].sum(0) for _, s in ssus]), f2, f1)
            for _, ssu in ssus:
                tsus = list(ssu.groupby("t", sort=False))
                f3 = len(tsus) / ssu.L.iloc[0]
                add(2, np.array([influences[t.index].sum(0) for _, t in tsus]), f3, f1 * f2)
                for _, tsu in tsus:
                    f4 = len(tsu) / tsu.K.iloc[0]
                    add(3, influences[tsu.index], f4, f1 * f2 * f3)
    return parts


def response_score(family, y, eta):
    if family == "regress":
        return y - eta, np.ones(len(y))
    if family == "logit":
        mean = special.expit(eta)
        return y - mean, mean * (1 - mean)
    if family == "probit":
        signed = (2 * y - 1) * eta
        mills = np.exp(stats.norm.logpdf(signed) - special.log_ndtr(signed))
        return (2 * y - 1) * mills, mills * (signed + mills)
    mean = np.exp(eta)
    return y - mean, mean


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("census", [None, "first", "one_psu", "one_ssu", "one_tsu", "fourth"])
def test_all_four_score_components_match_independent_raw_nested_geometry(family, census):
    frame, _ = fixture()
    if census == "first":
        frame["N"] = 2
    elif census == "one_psu":
        frame.loc[(frame.h == 0) & (frame.p == 0), "M"] = 2
    elif census == "one_ssu":
        frame.loc[(frame.h == 0) & (frame.p == 0) & (frame.s == 0), "L"] = 2
    elif census == "one_tsu":
        frame.loc[(frame.h == 0) & (frame.p == 0) & (frame.s == 0) & (frame.t == 0), "K"] = 4
    elif census == "fourth":
        frame["K"] = 4
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
    state = getattr(oe, "survey_four_stage_" + family)(
        frame, declaration(frame), outcome, ["x", "z"], tolerance=1e-11
    )
    X = np.column_stack((np.ones(len(frame)), frame.x, frame.z))
    weights = np.asarray(frame.N / 2 * frame.M / 2 * frame.L / 2 * frame.K / 4)
    normalized = weights / weights.mean()
    beta = np.asarray(state.coefficients)
    factor, curvature = response_score(family, frame[outcome].to_numpy(), X @ beta)
    bread = (X.T * (normalized * curvature)) @ X
    scores = X * (normalized * factor)[:, None]
    influences = np.linalg.solve(bread, scores.T).T
    parts = independent_parts(frame, influences)
    np.testing.assert_allclose(state.metadata["bread"], bread, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(state.metadata["row_scores"], scores, rtol=1e-8, atol=1e-13)
    for i, part in enumerate(parts, 1):
        np.testing.assert_allclose(
            state.metadata[f"stage{i}_covariance"], part, rtol=1e-9, atol=1e-15
        )
    np.testing.assert_allclose(state.covariance, sum(parts), rtol=1e-9, atol=1e-15)
    if family == "regress":
        expected = np.linalg.lstsq(
            X * np.sqrt(weights)[:, None], frame.y * np.sqrt(weights), rcond=None
        )[0]
        np.testing.assert_allclose(beta, expected, rtol=1e-11, atol=1e-13)
    if census == "first":
        assert not np.any(parts[0])
        assert all(np.trace(part) > 0 for part in parts[1:])
    if census == "fourth":
        assert not np.any(parts[3])
        assert all(np.trace(part) > 0 for part in parts[:3])


@pytest.mark.parametrize("family", FAMILIES)
def test_domain_missing_and_empty_tsu_preserve_complete_four_stage_scores(family):
    frame, design = fixture()
    frame["take"] = ~((frame.h == 0) & (frame.p == 0) & (frame.s == 0) & (frame.t == 0))
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
    frame[outcome] = frame[outcome].astype(float)
    excluded = int(np.flatnonzero(frame["take"])[0])
    frame.loc[excluded, outcome] = np.nan
    state = getattr(oe, "survey_four_stage_" + family)(
        frame, design, outcome, ["x", "z"], domain="take", missing="drop", tolerance=1e-11
    )
    assert state.design.validation.n_tsu == design.validation.n_tsu == 24
    assert state.metadata["outcome_exclusions"] == [excluded]
    positions = np.asarray(state.metadata["sample_positions"])
    X = np.column_stack((np.ones(len(frame)), frame.x, frame.z))[positions]
    y = frame[outcome].to_numpy()[positions]
    weights = np.asarray(frame.N / 2 * frame.M / 2 * frame.L / 2 * frame.K / 4)[positions]
    normalized = weights / weights.mean()
    factor, curvature = response_score(family, y, X @ np.asarray(state.coefficients))
    bread = (X.T * (normalized * curvature)) @ X
    scores = np.zeros((len(frame), 3))
    scores[positions] = X * (normalized * factor)[:, None]
    influences = np.linalg.solve(bread, scores.T).T
    parts = independent_parts(frame, influences)
    assert np.asarray(state.metadata["row_scores"])[~frame["take"]].tolist() == [[0.0] * 3] * 4
    for i, part in enumerate(parts, 1):
        np.testing.assert_allclose(
            state.metadata[f"stage{i}_covariance"], part, rtol=1e-9, atol=1e-15
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_full_four_stage_census_has_point_intervals(family):
    frame, _ = fixture()
    frame[["N", "M", "L"]] = 2
    frame["K"] = 4
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
    state = getattr(oe, "survey_four_stage_" + family)(
        frame, declaration(frame), outcome, ["x", "z"]
    )
    assert state.covariance == ((0.0,) * 3,) * 3
    result = state.to_frame()
    assert result.ci_low.tolist() == result.estimate.tolist() == result.ci_high.tolist()


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
def test_nonlinear_width_eight_refused_before_native_fit(family, monkeypatch):
    frame, design = fixture()
    names = ["x", "z"] + [f"a{i}" for i in range(5)]
    for i, name in enumerate(names[2:]):
        frame[name] = np.sin((np.arange(len(frame)) + 0.3) * (i + 1.7))

    def forbidden(*args, **kwargs):
        raise AssertionError("native solver preceded complete four-stage work admission")

    monkeypatch.setattr(fitting, "fit_sample", forbidden)
    outcome = "c" if family == "poisson" else "b"
    with pytest.raises(AnalysisError) as error:
        getattr(oe, "survey_four_stage_" + family)(frame, design, outcome, names)
    assert error.value.code == "survey_regression_budget"


@pytest.mark.parametrize("family", FAMILIES)
def test_no_intercept_and_intercept_only_models_have_correct_parameter_order(family):
    frame, design = fixture()
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
    function = getattr(oe, "survey_four_stage_" + family)
    no_intercept = function(frame, design, outcome, ["x", "z"], intercept=False)
    assert no_intercept.labels == ("x", "z")
    assert len(no_intercept.metadata["stage4_covariance"]) == 2
    intercept_only = function(frame, design, outcome, [])
    assert intercept_only.labels == ("_cons",)
    weights = np.asarray(design.validation.weights)
    mean = np.average(frame[outcome], weights=weights)
    expected = {
        "regress": lambda p: p,
        "logit": special.logit,
        "probit": special.ndtri,
        "poisson": np.log,
    }[family](mean)
    np.testing.assert_allclose(intercept_only.coefficients, [expected], rtol=1e-10, atol=1e-12)
