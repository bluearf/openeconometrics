"""Independent continuous fixed-bank score/curvature and admission oracles."""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import norm

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.irt import calibrated as banks
from openecon.econometrics.irt import calibrated_scores as scores
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def binary(a=1.0, b=0.0, c=0.0, u=1.0):
    return {"family": "binary", "K": 2, "discrimination": a, "difficulty": b,
            "guessing": c, "upper": u, "scores": [0, 1]}


def grm(a=1.2, thresholds=(-0.8, 0.6)):
    return {"family": "grm", "K": len(thresholds)+1, "discrimination": a,
            "thresholds": list(thresholds), "scores": list(range(len(thresholds)+1))}


def gpcm(a=0.9, steps=(0.4, -0.7, 0.9)):
    return {"family": "gpcm", "K": len(steps)+1, "discrimination": a,
            "thresholds": list(steps), "scores": list(range(len(steps)+1))}


def nrm(slopes=(0.0, -0.6, 1.3), intercepts=(0.0, 0.3, -0.2), scoring=(2, 0, 1)):
    return {"family": "nrm", "K": len(slopes), "slopes": list(slopes),
            "intercepts": list(intercepts), "scores": list(scoring)}


def make_bank(definitions, names=None):
    names = names or [f"i{j}" for j in range(len(definitions))]
    if all(definition["family"] == "binary" for definition in definitions):
        return banks.irt_bank_binary(names,
            [definition["discrimination"] for definition in definitions],
            [definition["difficulty"] for definition in definitions],
            guessing=[definition["guessing"] for definition in definitions],
            upper=[definition["upper"] for definition in definitions])
    return banks.irt_bank_polytomous(names, family=[definition["family"] for definition in definitions],
        thresholds=[definition.get("thresholds") for definition in definitions],
        discrimination=[definition.get("discrimination") for definition in definitions],
        slopes=[definition.get("slopes") for definition in definitions],
        intercepts=[definition.get("intercepts") for definition in definitions],
        scores=[definition["scores"] for definition in definitions])


def item_log_score_curvature(definition, theta, code):
    """NumPy formulas independent of Torch kernels and autodifferentiation."""
    family = definition["family"]
    if family == "binary":
        a, b = definition["discrimination"], definition["difficulty"]
        z = a*(theta-b)
        p = expit(z)
        logp = -np.logaddexp(0, -z) if code else -np.logaddexp(0, z)
        return logp, a*(code-p), a*a*p*(1-p)
    if family == "grm":
        a = definition["discrimination"]
        b = np.array(definition["thresholds"])
        z, k = a*(theta-b), definition["K"]
        p = expit(z)
        if code == 0:
            return -np.logaddexp(0, z[0]), -a*p[0], a*a*p[0]*(1-p[0])
        if code == k-1:
            return -np.logaddexp(0, -z[-1]), a*(1-p[-1]), a*a*p[-1]*(1-p[-1])
        logp = (-np.logaddexp(0, -z[code-1])-np.logaddexp(0, z[code])
                +np.log(-np.expm1(z[code]-z[code-1])))
        return logp, a*(1-p[code-1]-p[code]), a*a*(p[code-1]*(1-p[code-1])+p[code]*(1-p[code]))
    if family == "gpcm":
        a = definition["discrimination"]
        slopes = a*np.arange(definition["K"])
        intercepts = np.r_[0.0, -a*np.cumsum(definition["thresholds"])]
    else:
        slopes = np.array(definition["slopes"])
        intercepts = np.array(definition["intercepts"])
    logits = theta*slopes+intercepts
    logp = logits-logsumexp(logits)
    p = np.exp(logp)
    mean = p@slopes
    information = p@((slopes-mean)**2)
    return logp[code], slopes[code]-mean, information


def oracle(definitions, response, theta, prior=None):
    values = [item_log_score_curvature(d, theta, int(y))
              for d, y in zip(definitions, response) if not pd.isna(y)]
    ll = sum(value[0] for value in values)
    gradient = sum(value[1] for value in values)
    information = sum(value[2] for value in values)
    target = ll
    if prior is not None:
        mean, sd = prior
        target -= 0.5*((theta-mean)/sd)**2+math.log(sd)+0.5*math.log(2*math.pi)
        gradient -= (theta-mean)/sd**2
        information += 1/sd**2
    return target, ll, gradient, information


@pytest.mark.parametrize("correct", [1, 2, 3, 4, 5])
def test_identical_rasch_closed_form_mle_and_full_normal_wald(correct):
    count = 6
    definitions = [binary() for _ in range(count)]
    response = [1]*correct+[0]*(count-correct)
    bank = make_bank(definitions)
    result = scores.irt_score_mle(bank, data=[dict(zip([f"i{i}" for i in range(count)], response))], level=0.9)
    row = result["people"].iloc[0]
    expected = math.log(correct/(count-correct))
    information = count*(correct/count)*(1-correct/count)
    assert row.theta == pytest.approx(expected, abs=2e-9)
    assert row.observed_information == pytest.approx(information, rel=1e-9)
    assert row.standard_error == pytest.approx(1/math.sqrt(information), rel=1e-9)
    assert row.ci_lower == pytest.approx(expected-norm.ppf(.95)/math.sqrt(information), abs=3e-9)
    assert row.ci_upper == pytest.approx(expected+norm.ppf(.95)/math.sqrt(information), abs=3e-9)
    assert abs(row.gradient) <= 1e-9
    assert result["certificates"].iloc[0].interior


@pytest.mark.parametrize("definitions,response", [
    ([binary(0.7, -0.9), binary(1.2, 0.2), binary(1.6, 0.8)], [1, 0, 1]),
    ([grm(0.8), grm(1.5, (-1.2, 0.3, 1.1))], [1, 2]),
    ([gpcm(0.7), gpcm(1.4, (0.6, -0.4))], [1, 1]),
    ([nrm(), nrm((0.0, 1.6, -0.5), (0.0, -0.4, 0.6))], [0, 0]),
    ([grm(), gpcm(), nrm()], [1, 2, 0]),
])
@pytest.mark.parametrize("method", ["mle", "map"])
def test_family_and_mixed_modes_match_independent_brent_optimizer_and_curvature(definitions, response, method):
    bank = make_bank(definitions)
    prior = None if method == "mle" else (0.45, 1.3)
    root = brentq(lambda theta: oracle(definitions, response, theta, prior)[2], -12, 12, xtol=1e-13)
    optimum = minimize_scalar(lambda theta: -oracle(definitions, response, theta, prior)[0],
                              bounds=(-12, 12), method="bounded", options={"xatol": 1e-10})
    data = [dict(zip([f"i{i}" for i in range(len(definitions))], response))]
    result = (scores.irt_score_mle(bank, data=data) if method == "mle"
              else scores.irt_score_map(bank, data=data, prior_mean=prior[0], prior_sd=prior[1]))
    row = result["people"].iloc[0]
    expected = oracle(definitions, response, root, prior)
    at_reported = oracle(definitions, response, row.theta, prior)
    assert row.theta == pytest.approx(root, abs=3e-9)
    assert row.theta == pytest.approx(optimum.x, abs=3e-7)
    assert row.objective == pytest.approx(at_reported[0], abs=3e-12)
    assert row.log_likelihood == pytest.approx(at_reported[1], abs=3e-12)
    assert row.gradient == pytest.approx(at_reported[2], abs=3e-12)
    assert row.mode_information == pytest.approx(expected[3], rel=2e-9)
    h = 2e-4
    finite_information = -(oracle(definitions, response, root+h, prior)[0]
                           -2*expected[0]+oracle(definitions, response, root-h, prior)[0])/h**2
    assert row.mode_information == pytest.approx(finite_information, rel=2e-6, abs=2e-7)
    assert row["standard_error" if method == "mle" else "laplace_sd"] == pytest.approx(1/math.sqrt(expected[3]), rel=2e-9)
    if method == "map":
        assert not any("ci_" in name or "pvalue" in name for name in result["people"])
        assert "not the exact posterior SD" in result.attrs["uncertainty"]


def test_nrm_declared_score_order_does_not_drive_mode_or_curvature():
    definitions = [nrm(), nrm((0.0, 1.6, -0.5), (0.0, -0.4, 0.6))]
    data = {"i0": [0], "i1": [0]}
    first = scores.irt_score_mle(make_bank(definitions), data=data)
    alternate = copy.deepcopy(definitions)
    alternate[0]["scores"] = [0, 9, 0]
    second = scores.irt_score_mle(make_bank(alternate), data=data)
    pd.testing.assert_frame_equal(first["people"], second["people"])


def test_missing_partial_positions_duplicate_indices_and_exact_empty_prior():
    definitions = [grm(), gpcm(), nrm()]
    frame = pd.DataFrame({"i0": [1.0, np.nan, 0.0], "i1": [np.nan, np.nan, 1.0],
                          "i2": [1.0, np.nan, np.nan]}, index=["duplicate", "duplicate", "tail"])
    result = scores.irt_score_map(make_bank(definitions), data=frame, prior_mean=0.6, prior_sd=1.2)
    assert list(result["people"].position) == [0, 1, 2]
    assert list(result["people"]["index"]) == ["duplicate", "duplicate", "tail"]
    assert list(result["people"].observed_items) == [2, 0, 2]
    empty = result["people"].iloc[1]
    assert empty.theta == 0.6
    assert empty.laplace_sd == 1.2
    assert empty.log_likelihood == 0
    assert empty.observed_information == 0
    assert empty.mode_information == pytest.approx(1/1.2**2)
    assert empty.iterations == 0
    for position in (0, 2):
        response = frame.iloc[position].tolist()
        root = brentq(lambda theta: oracle(definitions, response, theta, (0.6, 1.2))[2], -12, 12)
        assert result["people"].iloc[position].theta == pytest.approx(root, abs=3e-9)


@pytest.mark.parametrize("bad", [[0, 0, 0], [1, 1, 1], [np.nan, np.nan, np.nan]])
def test_mle_refuses_extreme_or_uninformative_rows_without_partial_results(bad):
    bank = make_bank([binary()]*3)
    frame = pd.DataFrame([[0, 1, 1], bad], columns=["i0", "i1", "i2"])
    with pytest.raises(AnalysisError, match="source position 1"):
        scores.irt_score_mle(bank, data=frame)


@pytest.mark.parametrize("method", ["mle", "map"])
def test_nonconcave_binary_is_refused_before_response_selection(method, monkeypatch):
    bank = make_bank([binary(c=.15), binary()])
    monkeypatch.setattr(banks, "_response_size", lambda *a, **k: pytest.fail("response preflight reached"))
    with pytest.raises(AnalysisError, match="3PL/4PL"):
        getattr(scores, f"irt_score_{method}")(bank, data=object())


def test_boundary_and_iteration_refusals_are_explicit():
    bank = make_bank([binary()]*4)
    data = {"i0": [1], "i1": [1], "i2": [1], "i3": [0]}
    with pytest.raises(AnalysisError, match="interior score root"):
        scores.irt_score_mle(bank, data=data, theta_limit=.5)
    with pytest.raises(AnalysisError, match="max_iter"):
        scores.irt_score_mle(bank, data=data, max_iter=1)
    with pytest.raises(AnalysisError, match="interior score root"):
        scores.irt_score_map(bank, data=data, prior_mean=4, theta_limit=.5)


def test_stable_middle_grm_score_with_subnormal_threshold_gap():
    gap = float(np.nextafter(0.0, 1.0))
    bank = make_bank([grm(1.0, (0.0, gap))])
    result = scores.irt_score_mle(bank, data={"i0": [1]})
    row = result["people"].iloc[0]
    assert row.theta == 0
    assert row.observed_information == 0.5
    assert row.log_likelihood == pytest.approx(math.log(gap)-math.log(4), abs=1e-12)


@pytest.mark.parametrize("bad", [True, "1", 0.5, 2, -1, np.inf])
def test_unknown_or_illegal_responses_are_not_recoded(bad):
    bank = make_bank([binary()]*3)
    with pytest.raises(AnalysisError):
        scores.irt_score_map(bank, data={"i0": [bad], "i1": [1], "i2": [0]})


@pytest.mark.parametrize("options", [
    {"device": "cuda"}, {"theta_limit": True}, {"theta_limit": 13}, {"tolerance": 0},
    {"tolerance": float("nan")}, {"max_iter": True}, {"max_iter": 201},
    {"max_work": True}, {"max_bytes": 0}, {"prior_mean": 5}, {"prior_sd": 0},
    {"prior_sd": "1"},
])
def test_controls_validate_before_bank_or_input_materialization(options, monkeypatch):
    monkeypatch.setattr(banks, "_bank", lambda *a, **k: pytest.fail("bank materialization reached"))
    with pytest.raises(AnalysisError):
        scores.irt_score_map(object(), data=object(), **options)


@pytest.mark.parametrize("level", [0, 1, True, "0.95", np.inf, np.nan, 1e-300,
                                  float(np.nextafter(0.0, 1.0)), float(np.nextafter(1.0, 0.0))])
def test_mle_level_refuses_nonrepresentable_or_invalid_confidence_before_bank(level, monkeypatch):
    monkeypatch.setattr(banks, "_bank", lambda *a, **k: pytest.fail("bank materialization reached"))
    with pytest.raises(AnalysisError):
        scores.irt_score_mle(object(), data=object(), level=level)


def test_mle_refuses_positive_critical_when_interval_endpoints_round_to_the_mode():
    names = [f"i{j}" for j in range(16)]
    bank = make_bank([binary(5.0, 4.0)]*16, names=names)
    data = {name: [int(j < 15)] for j, name in enumerate(names)}
    # At this known-bank MLE the mathematical interval is nonzero, but its
    # width is below the float64 spacing around theta = 4 + log(15)/5.
    assert norm.ppf(1-(1-1e-15)/2) > 0
    with pytest.raises(AnalysisError, match="Wald interval endpoints"):
        scores.irt_score_mle(bank, data=data, level=1e-15)


def test_max_work_and_named_workspace_guard_before_response_copy(monkeypatch):
    bank = make_bank([binary()]*3)
    frame = pd.DataFrame([[0, 1, 1]]*100, columns=["i0", "i1", "i2"])
    monkeypatch.setattr(banks, "_responses", lambda *a, **k: pytest.fail("response copy reached"))
    with pytest.raises(AnalysisError, match="before response selection"):
        scores.irt_score_mle(bank, data=frame, max_work=1)
    with pytest.raises(AnalysisError):
        scores.irt_score_map(bank, data=frame, max_bytes=1)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace"):
            scores.irt_score_map(bank, data=frame)


def test_work_plan_counts_every_trace_evaluation_and_rejects_rows_over_limit():
    bank = make_bank([binary()]*3)
    result = scores.irt_score_mle(bank, data={"i0": [0, 1], "i1": [1, 0], "i2": [1, 0]})
    assert result.attrs["objective_evaluations"] == len(result["trace"])
    assert result.attrs["charged_work"] == len(result["trace"])*64*6
    assert result.attrs["charged_work"] <= result.attrs["planned_work"] <= result.attrs["controls"]["max_work"]
    with pytest.raises(AnalysisError):
        scores.irt_score_map(bank, data={"i0": [0]*1001, "i1": [1]*1001, "i2": [1]*1001})


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("method", ["mle", "map"])
def test_caller_gradient_disable_is_scoped_and_does_not_break_scores(context, method):
    bank = make_bank([binary()]*3)
    with context():
        result = getattr(scores, f"irt_score_{method}")(bank, data={"i0": [0], "i1": [1], "i2": [1]})
        assert not torch.is_grad_enabled()
        if context is torch.inference_mode:
            assert torch.is_inference_mode_enabled()
    target = (math.log(2) if method == "mle" else
              brentq(lambda x: oracle([binary()]*3, [0, 1, 1], x, (0, 1))[2], -12, 12))
    assert result["people"].theta.iloc[0] == pytest.approx(target, abs=2e-9)


def test_cpu_float64_under_meta_default_device_and_unchanged_input():
    bank = make_bank([binary()]*3)
    frame = pd.DataFrame({"i0": [0.0], "i1": [1.0], "i2": [1.0]}, index=[9])
    original = frame.copy(deep=True)
    device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = scores.irt_score_mle(bank, data=frame)
    finally:
        torch.set_default_device(device)
    assert result["people"].theta.iloc[0] == pytest.approx(math.log(2), abs=2e-9)
    pd.testing.assert_frame_equal(frame, original)


def test_complete_restored_bank_reproduces_scoring_and_tamper_refuses_before_selection(monkeypatch):
    bank = make_bank([binary()]*3)
    data = {"i0": [0], "i1": [1], "i2": [1]}
    source = scores.irt_score_mle(bank, data=data)
    restored_bank = banks.irt_bank_restore(summary_state(bank))
    restored_score = scores.irt_score_mle(restored_bank, data=data)
    assert summary_state(source) == summary_state(restored_score)
    assert source.to_latex() == restored_score.to_latex()
    forged = copy.deepcopy(bank)
    forged.attrs["bank_state"]["definitions"][0]["discrimination"] = 1.1
    monkeypatch.setattr(banks, "_response_size", lambda *a, **k: pytest.fail("response selection reached"))
    with pytest.raises(AnalysisError):
        scores.irt_score_mle(forged, data=object())


@pytest.mark.parametrize("method", ["mle", "map"])
def test_complete_portable_tables_attrs_and_latex(method):
    bank = make_bank([grm(), gpcm(), nrm()])
    data = {"i0": [1, 0], "i1": [2, 1], "i2": [0, 0]}
    result = getattr(scores, f"irt_score_{method}")(bank, data=data)
    encoded = summary_state(result)
    restored = restore_summary(encoded)
    assert encoded == summary_state(restored)
    assert result.to_latex() == restored.to_latex()
    assert result.attrs == restored.attrs
    for name in result:
        pd.testing.assert_frame_equal(result[name], restored[name], check_dtype=False)
    payload = json.loads(encoded)
    assert len(payload["tables"]["trace"]["data"]) == result.attrs["objective_evaluations"]
    assert payload["attrs"]["bank_state"]["definitions"][2]["scores"] == [2, 0, 1]
