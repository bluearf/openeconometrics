"""Replay retained native artifacts with current source, without any estimator.

These are compatibility checks on complete scientific states. They do not
relabel the historical QA applications as freshly built desktop releases.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, logsumexp
from statsmodels.stats.contingency_tables import StratifiedTable

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"
SURVEY = ("mean", "total", "ratio", "proportion", "brr", "fay", "jackknife", "bootstrap")
IRT = ("rasch", "2pl", "3pl", "grm", "pcm", "rsm")
RELIABILITY = (
    "polychoric", "polyserial", "omega_total", "icc", "cohen_kappa",
    "fleiss_kappa", "krippendorff_alpha", "gwet_ac",
)


def forbidden(*args, **kwargs):
    raise AssertionError("A saved-state replay must not fit an estimator")


@pytest.mark.parametrize("method", SURVEY)
def test_retained_native_survey_joint_contrast_without_data_or_estimation(method, monkeypatch):
    states = json.loads((EVIDENCE / "survey-inference-2026-10-07" / "persisted-states.json").read_text())
    state = states[method]
    from openecon.econometrics.survey import targets, replication

    for module in (targets, replication):
        for name in dir(module):
            if name.startswith("survey_") and callable(getattr(module, name)):
                monkeypatch.setattr(module, name, forbidden)
    restored = oe.SurveyResult.model_validate_json(json.dumps(state))
    # A nontrivial joint contrast exercises covariance off-diagonals, including
    # the singular fixed-category proportion covariance.
    coefficients = np.arange(1, len(state["labels"]) + 1, dtype=float)
    coefficients[1::2] *= -1
    expected = coefficients @ np.asarray(state["estimates"])
    variance = coefficients @ np.asarray(state["covariance"]) @ coefficients
    actual = restored.contrast(coefficients.tolist(), null=0.25).iloc[0]
    assert actual.estimate == pytest.approx(expected, abs=1e-12)
    assert actual.std_error == pytest.approx(np.sqrt(max(variance, 0)), abs=1e-12)
    if variance > 1e-25 and restored.df:
        statistic = (expected - 0.25) / np.sqrt(variance)
        assert actual.p_value == pytest.approx(2 * stats.t.sf(abs(statistic), restored.df), abs=1e-12)
    assert restored.metadata["sample_positions"] == state["metadata"]["sample_positions"]
    assert restored.design.stages == 1


@pytest.mark.parametrize("family", IRT)
def test_retained_native_irt_complete_tables_and_scores_without_refitting(family, monkeypatch):
    artifacts = json.loads(gzip.decompress(
        (EVIDENCE / "irt-eight-2026-10-07" / "complete-results.json.gz").read_bytes()
    ))
    from openecon.econometrics.irt import kernels, models

    monkeypatch.setattr(kernels, "fit", forbidden)
    monkeypatch.setattr(models, "_fit", forbidden)
    restored = oe.irt_restore(artifacts[family + ".json"])
    assert restored.to_json() == artifacts[family + ".json"]
    assert restored.to_latex() == artifacts[family + ".tex"]
    expected = json.loads(artifacts[family + "-tables.json"])
    assert set(restored) == set(expected)
    for name, frame in restored.items():
        assert list(frame.columns) == expected[name]["columns"]
        assert list(frame.index) == expected[name]["index"]
        assert frame.astype(object).where(frame.notna(), None).values.tolist() == expected[name]["data"]
    scores = oe.irt_score(restored)["scores"]
    np.testing.assert_array_equal(scores.eap, restored["people"].eap)
    np.testing.assert_array_equal(scores.posterior_sd, restored["people"].posterior_sd)
    assert list(scores.position) == restored.attrs["state"]["sample_positions"]


@pytest.mark.parametrize("method", RELIABILITY)
def test_retained_native_reliability_all_subject_draws_and_tables_without_refitting(method, monkeypatch):
    with tarfile.open(EVIDENCE / "reliability-eight-2026-10-07" / "complete-results.tar.gz") as archive:
        artifact = json.load(archive.extractfile(method + ".json"))
    from openecon.econometrics.measurement import common

    monkeypatch.setattr(common, "result", forbidden)
    restored = oe.reliability_load(artifact)
    assert oe.reliability_save(restored) == artifact
    payload = artifact["payload"]
    assert set(restored) == set(payload["tables"])
    for name, saved in payload["tables"].items():
        expected = pd.DataFrame(saved["data"], columns=saved["columns"], index=saved["index"])
        pd.testing.assert_frame_equal(restored[name], expected, check_dtype=False, check_exact=True)
    state = restored.attrs["state"]
    draws = np.asarray(state["replicates"])
    assert len(draws) == state["sample"]["n"]
    variance = (len(draws) - 1) / len(draws) * np.square(draws - draws.mean()).sum()
    assert state["covariance"] == pytest.approx(variance, abs=1e-12)
    row = restored["estimate"].iloc[0]
    assert row.std_error == pytest.approx(np.sqrt(variance), abs=1e-12)
    assert row.p_value == pytest.approx(2 * stats.t.sf(abs(row.statistic), len(draws) - 1), abs=1e-12)


def mh_fixture():
    # Explicit finite 2x2 strata; third stratum has no focal people, fourth
    # no correct responses. Both remain visible but have zero information.
    counts = np.array([[18, 9, 11, 14], [7, 12, 5, 16], [3, 2, 0, 0], [0, 4, 0, 3]])
    records = []
    for score, cells in enumerate(counts):
        for (group, y), n in zip([("R", 1), ("R", 0), ("F", 1), ("F", 0)], cells):
            records.extend([dict(group=group, item=y, score=score)] * int(n))
    return pd.DataFrame(records, index=["repeated-index"] * len(records)), counts


@pytest.mark.parametrize("continuity", [False, True])
def test_binary_dif_against_fixed_margin_hypergeometric_and_rbg_oracle(continuity):
    data, counts = mh_fixture()
    result = oe.irt_mh_dif(data, items=["item"], group="group", reference="R", focal="F", match="score",
                           continuity=continuity, alpha=0.1)
    row = result["dif"].iloc[0]
    admitted = counts[:2]
    expected, variance = [], []
    for a, b, c, d in admitted:
        mean, var = stats.hypergeom.stats(a+b+c+d, a+c, a+b, moments="mv")
        expected.append(mean)
        variance.append(var)
    delta = abs(admitted[:, 0].sum() - sum(expected))
    expected_chi2 = max(0, delta - (0.5 if continuity else 0)) ** 2 / sum(variance)
    assert row.chi2 == pytest.approx(expected_chi2, abs=1e-12)
    assert row.p_value == pytest.approx(stats.chi2.sf(expected_chi2, 1), abs=1e-12)
    reference = StratifiedTable(admitted.reshape(-1, 2, 2).transpose(1, 2, 0))
    assert row.odds_ratio_reference_over_focal == pytest.approx(reference.oddsratio_pooled, abs=1e-12)
    assert row.std_error_log == pytest.approx(reference.logodds_pooled_se, abs=1e-12)
    np.testing.assert_allclose([row.ci_low_odds_ratio, row.ci_high_odds_ratio], reference.oddsratio_pooled_confint(alpha=.1), atol=1e-12)
    assert row.delta_mh == pytest.approx(-2.35 * np.log(reference.oddsratio_pooled), abs=1e-12)
    np.testing.assert_allclose(result["strata"].conditional_variance, variance + [0, 0], atol=1e-12)
    assert result["strata"].informative.tolist() == [True, True, False, False]
    assert row.informative_strata == 2 and row.excluded_strata == 2
    assert list(result["sample"].position) == list(range(len(data)))


def test_dif_group_and_item_complement_symmetry_missing_and_saved_replay(monkeypatch):
    data, _ = mh_fixture()
    options = dict(items=["item"], group="group", reference="R", focal="F", match="score")
    original = oe.irt_mh_dif(data, **options)
    reverse = oe.irt_mh_dif(data, **dict(options, reference="F", focal="R"))
    complement = oe.irt_mh_dif(data.assign(item=1-data.item), **options)
    for result in (reverse, complement):
        assert result["dif"].log_odds_ratio.iloc[0] == pytest.approx(-original["dif"].log_odds_ratio.iloc[0])
        assert result["dif"].p_value.iloc[0] == pytest.approx(original["dif"].p_value.iloc[0])
        assert result["dif"].std_error_log.iloc[0] == pytest.approx(original["dif"].std_error_log.iloc[0])
    incomplete = data.astype({"item":float})
    incomplete.iloc[4, incomplete.columns.get_loc("item")] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        oe.irt_mh_dif(incomplete, **options)
    dropped = oe.irt_mh_dif(incomplete, **options, missing="drop")
    assert dropped.attrs["diagnostic_state"]["excluded_positions"] == [4]
    assert list(dropped["sample"].source_index) == ["repeated-index"] * (len(data)-1)
    encoded = dropped.to_json()
    monkeypatch.setattr(oe, "irt_mh_dif", forbidden)
    restored = oe.irt_diagnostics_restore(encoded)
    assert restored.to_json() == encoded and restored.to_latex() == dropped.to_latex()
    for name in dropped:
        pd.testing.assert_frame_equal(restored[name], dropped[name], check_exact=True)


def independent_probabilities(state, theta):
    """Natural item units; independent NumPy category geometry, no kernel call."""
    parameters = np.asarray(state["parameters"])
    answer, cursor = [], 0
    for j, k in enumerate(state["categories"]):
        if state["family"] == "rsm":
            slope = 1
            steps = parameters[j] + parameters[len(state["items"]):]
        else:
            slope = parameters[cursor] if state["family"] in ("2pl", "3pl", "grm") else 1
            cursor += int(state["family"] in ("2pl", "3pl", "grm"))
            steps = parameters[cursor:cursor+k-1]
            cursor += k-1
        if state["family"] in ("rasch", "2pl", "3pl"):
            correct = state["guessing"][j] + (1-state["guessing"][j]) * expit(slope*(theta-steps[0]))
            answer.append(np.column_stack([1-correct, correct]))
        elif state["family"] == "grm":
            cumulative = np.column_stack([np.ones(len(theta)), expit(slope*(theta[:,None]-steps)), np.zeros(len(theta))])
            answer.append(-np.diff(cumulative, axis=1))
        else:
            logits = np.column_stack([np.zeros(len(theta)), np.cumsum(theta[:,None]-steps, axis=1)])
            answer.append(np.exp(logits-logsumexp(logits,axis=1)[:,None]))
    return answer


@pytest.mark.parametrize("family", IRT)
def test_descriptive_fit_all_six_families_independent_q3_mean_squares_and_saved_alignment(family, monkeypatch):
    artifacts = json.loads(gzip.decompress((EVIDENCE / "irt-eight-2026-10-07" / "complete-results.json.gz").read_bytes()))
    model = oe.irt_restore(artifacts[family+".json"])
    state = model.attrs["state"]
    eap = np.asarray(model["people"].eap)
    probabilities = independent_probabilities(state, eap)
    expected = np.column_stack([p @ np.arange(p.shape[1]) for p in probabilities])
    variance = np.column_stack([p @ np.arange(p.shape[1])**2 for p in probabilities]) - expected**2
    residual = np.asarray(state["responses"]) - expected
    correlation = np.corrcoef(residual, rowvar=False)
    values = correlation[np.triu_indices(len(state["items"]),1)]
    from openecon.econometrics.irt import kernels, models
    monkeypatch.setattr(kernels, "fit", forbidden)
    monkeypatch.setattr(models, "_fit", forbidden)
    actual = oe.irt_fit_diagnostics(model)
    np.testing.assert_allclose(actual["pairs"].q3, values, atol=2e-12)
    np.testing.assert_allclose(actual["pairs"].adjusted_q3, values-values.mean(), atol=2e-12)
    np.testing.assert_allclose(actual["items"].infit_mean_square, (residual**2).sum(0)/variance.sum(0), atol=2e-12)
    np.testing.assert_allclose(actual["items"].outfit_mean_square, (residual**2/variance).mean(0), atol=2e-12)
    np.testing.assert_allclose(actual["residuals"].expected.to_numpy().reshape(expected.shape), expected, atol=2e-12)
    assert not any("p_value" in frame.columns for frame in actual.values())
    assert actual["sample"].included.sum() == state["nobs"]
    assert actual["sample"].position.tolist() == list(range(state["nobs_original"]))
    encoded = actual.to_json()
    restored = oe.irt_diagnostics_restore(encoded)
    assert restored.to_json() == encoded
    for name in actual:
        pd.testing.assert_frame_equal(actual[name], restored[name], check_exact=True)


def test_descriptive_fit_exact_original_source_and_missing_sample_drift():
    artifacts = json.loads(gzip.decompress((EVIDENCE / "irt-eight-2026-10-07" / "complete-results.json.gz").read_bytes()))
    model = oe.irt_restore(artifacts["2pl.json"])
    frame = pd.read_csv(Path(__file__).parent/"fixtures"/"irt"/"binary.csv",index_col="person")
    actual = oe.irt_fit_diagnostics(model, data=frame)
    assert actual.to_json() == oe.irt_fit_diagnostics(model).to_json()
    changed = frame.copy()
    changed.iloc[8,0] = 1-changed.iloc[8,0]
    with pytest.raises(AnalysisError, match="exact original"):
        oe.irt_fit_diagnostics(model,data=changed)
    with pytest.raises(AnalysisError, match="exact original"):
        oe.irt_fit_diagnostics(model,data=frame.iloc[::-1])


@pytest.mark.parametrize("change", ["outside_group","boolean_item","fractional_score","no_overlap","infinite_item"])
def test_dif_invalid_admission_domains(change):
    frame,_ = mh_fixture()
    if change == "outside_group":
        frame.iloc[0,frame.columns.get_loc("group")] = "third"
    if change == "boolean_item":
        frame["item"] = frame.item.astype(bool)
    if change == "fractional_score":
        frame["score"] = frame.score.astype(float)+.5
    if change == "no_overlap":
        frame["score"] = np.where(frame.group=="R",0,1)
    if change == "infinite_item":
        frame["item"] = frame.item.astype(float)
        frame.iloc[0,frame.columns.get_loc("item")] = np.inf
    with pytest.raises(AnalysisError):
        oe.irt_mh_dif(frame,items=["item"],group="group",reference="R",focal="F",match="score")


def test_diagnostic_portable_tamper_revalidated_geometry_and_cpu_workspace():
    from openecon.econometrics.irt.diagnostics import _digest
    data,_ = mh_fixture()
    options = dict(items=["item"],group="group",reference="R",focal="F",match="score")
    output = oe.irt_mh_dif(data,**options)
    envelope = json.loads(output.to_json())
    envelope["state"]["counts"][0][0][0] += 1
    with pytest.raises(AnalysisError,match="checksum"):
        oe.irt_diagnostics_restore(envelope)
    envelope["checksum"] = _digest(envelope["state"])
    with pytest.raises(AnalysisError,match="geometry"):
        oe.irt_diagnostics_restore(envelope)
    envelope = json.loads(output.to_json())
    envelope["state"]["excluded_positions"] = [0]
    envelope["checksum"] = _digest(envelope["state"])
    with pytest.raises(AnalysisError,match="geometry"):
        oe.irt_diagnostics_restore(envelope)
    output["dif"].iloc[0,output["dif"].columns.get_loc("p_value")] = .5
    with pytest.raises(AnalysisError,match="tables changed"):
        output.to_json()
    with use_workspace_budget(1),pytest.raises(AnalysisError,match="workspace"):
        oe.irt_mh_dif(pd.concat([data]*50),**options)
    import torch
    with torch.device("meta"):
        resident = oe.irt_mh_dif(data,**options)
    assert resident.attrs["device"] == "cpu" and resident.attrs["dtype"] == "float64"
    artifacts = json.loads(gzip.decompress((EVIDENCE / "irt-eight-2026-10-07" / "complete-results.json.gz").read_bytes()))
    with torch.device("meta"):
        fit = oe.irt_fit_diagnostics(artifacts["2pl.json"])
    assert fit.attrs["device"] == "cpu" and fit.attrs["dtype"] == "float64"
    # Existing IRT state validation wraps its workspace rejection as an
    # invalid saved-state error; the request is still rejected before buffers.
    with use_workspace_budget(1),pytest.raises(AnalysisError):
        oe.irt_fit_diagnostics(artifacts["2pl.json"])
