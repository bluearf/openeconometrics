"""Recurrent-event Cox models (``oe.stcox_recurrent``) against R survival.

References (``scripts/generate_survival_frailty_oracles.R``): R ``coxph`` with
``Surv(start, stop, event)``, ``cluster(id)`` and ``strata()`` on data whose event
numbers, gap times and Wei-Lin-Weissfeld expansion are built independently in R.
R's robust variance applies no G/(G-1); ours does (Stata's cluster convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival.recurrent import event_history, snap_times
from openecon.models import ResultBundle

FIXTURES = Path(__file__).parent / "fixtures" / "survival_frailty"
DATA = pd.DataFrame(json.loads((FIXTURES / "data_recurrent.json").read_text())
                    ["datasets"]["recurrent"])
R = json.loads((FIXTURES / "r_reference.json").read_text())["recurrent"]
TERMS = R["terms"]
G = R["n_subjects"]
VARIANTS = {"ag": ("ag", None), "pwp_total": ("pwp_total", None), "pwp_gap": ("pwp_gap", None),
            "wlw": ("wlw", None), "pwp_total_cap3": ("pwp_total", 3), "wlw_cap3": ("wlw", 3)}


def fit(model, cap=None, ties="breslow", data=DATA, **options):
    return oe.stcox_recurrent(data=data, time="stop", entry="start", id="id", failure="event",
                              x=TERMS, model=model, max_events=cap, ties=ties, **options)


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_matches_r_coxph(variant, ties):
    ref = R[ties][variant]
    model, cap = VARIANTS[variant]
    robust = fit(model, cap, ties)
    assert_allclose(estimates(robust), ref["coef"], rtol=0, atol=1e-10)
    assert_allclose(covariance(robust), np.array(ref["robust"]) * G / (G - 1), rtol=1e-9,
                    atol=1e-14)
    assert robust.metrics["log_likelihood"] == pytest.approx(ref["loglik"], abs=1e-9)
    assert robust.metrics["n_rows"] == ref["n"]
    assert robust.metrics["n_subjects"] == G
    assert robust.inference["cluster_count"] == G
    assert robust.tests["model"]["distribution"] == "chi2"
    naive = fit(model, cap, ties, covariance="nonrobust")
    assert_allclose(covariance(naive), ref["naive"], rtol=1e-9, atol=1e-14)
    stats = robust.extra["strata"]
    if model == "ag":
        assert stats == []
    else:
        assert sum(row["rows"] for row in stats) == ref["n"]
        assert sum(row["events"] for row in stats) == ref["nevent"]


def test_event_history_against_explicit_loop():
    t = torch.tensor(DATA.stop.to_numpy(), dtype=torch.float64)
    t0 = torch.tensor(DATA.start.to_numpy(), dtype=torch.float64)
    d = torch.tensor(DATA.event.to_numpy(), dtype=torch.float64)
    ids = torch.tensor(pd.factorize(DATA.id)[0], dtype=torch.int64)
    shuffled = torch.randperm(len(t), generator=torch.Generator().manual_seed(5))
    prior, last = event_history(t[shuffled], t0[shuffled], d[shuffled], ids[shuffled])
    expected_prior, expected_last = [], []
    for row in shuffled.tolist():
        same = DATA[(DATA.id == DATA.id[row]) & (DATA.start < DATA.start[row])]
        expected_prior.append(int(same.event.sum()))
        events = same.stop[same.event == 1]
        expected_last.append(float(events.max()) if len(events) else 0.0)
    assert prior.tolist() == expected_prior
    assert_allclose(last.numpy(), expected_last, rtol=0, atol=0)


def test_gap_time_ties_follow_the_r_timefix_rule():
    assert (0.3 - 0.1) != 0.2                                 # decimal gaps are not exact
    entry = torch.tensor([0.0, 0.1, 0.0, 0.0], dtype=torch.float64)
    exit_ = torch.tensor([0.1, 0.3, 0.2, 0.5], dtype=torch.float64)
    last = torch.tensor([0.0, 0.1, 0.0, 0.0], dtype=torch.float64)
    start, stop = snap_times(entry - last, exit_ - last)
    assert stop.tolist() == [0.1, 0.3 - 0.1, 0.3 - 0.1, 0.5]  # one run, its smallest value
    assert start.tolist() == [0.0, 0.0, 0.0, 0.0]
    untouched = snap_times(torch.tensor([0.0, 0.0], dtype=torch.float64),
                           torch.tensor([0.2, 0.2 + 1e-6], dtype=torch.float64))
    assert untouched[1].tolist() == [0.2, 0.2 + 1e-6]


def test_ag_equals_stcox_with_id_and_robust():
    ag = fit("ag")
    stcox = oe.stcox(data=DATA, time="stop", entry="start", id="id", failure="event", x=TERMS,
                     covariance="robust", concordance=False)
    assert_allclose(estimates(ag), estimates(stcox), atol=1e-12)
    assert_allclose(covariance(ag), covariance(stcox), rtol=1e-10, atol=1e-15)


def test_ag_with_shared_gamma_frailty_matches_r():
    ref = R["ag_gamma_frailty"]
    result = oe.stcox_frailty(data=DATA, time="stop", entry="start", failure="event", x=TERMS,
                              shared="id", theta=0.5)
    assert_allclose(estimates(result), ref["coef"], atol=1e-8)
    assert_allclose(covariance(result), ref["var"], rtol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(ref["ilik"], abs=1e-8)


def test_row_order_and_missing_policy():
    base = fit("pwp_gap", ties="efron")
    shuffled = DATA.sample(frac=1.0, random_state=11).reset_index(drop=True)
    again = fit("pwp_gap", ties="efron", data=shuffled)
    assert_allclose(estimates(again), estimates(base), atol=1e-12)
    assert_allclose(covariance(again), covariance(base), rtol=1e-10)
    holed = pd.concat([DATA, DATA.iloc[:1].assign(age=np.nan, start=50.0, stop=51.0)],
                      ignore_index=True)
    with pytest.raises(AnalysisError) as error:
        fit("ag", data=holed)
    assert error.value.code == "missing_values"
    dropped = fit("pwp_gap", ties="efron", data=holed, missing="drop")
    assert dropped.dropped_rows == 1
    assert_allclose(estimates(dropped), estimates(base), atol=1e-12)


def test_failure_contract():
    def code(**options):
        arguments = dict(data=DATA, time="stop", entry="start", id="id", failure="event",
                         x=TERMS)
        arguments.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.stcox_recurrent(**arguments)
        return error.value.code

    overlap = DATA.copy()
    second = int(np.flatnonzero(overlap.id.duplicated())[0])
    overlap.loc[second, "start"] = 0.0
    assert code(data=overlap) == "overlapping_records"
    assert code(data=DATA.assign(event=0)) == "no_failures"
    assert code(model="frailty") == "invalid_spec"
    assert code(max_events=0) == "invalid_spec"
    assert code(covariance="cluster") == "invalid_spec"
    single = DATA[DATA.id == DATA.id[second]]
    assert code(data=single) == "insufficient_clusters"
    assert code(data=DATA.assign(event=2)) == "invalid_failure_indicator"
    assert code(x=["c"], data=DATA.assign(c=1.0)) == "no_covariates"
    collinear = oe.stcox_recurrent(data=DATA.assign(t2=DATA.treat), time="stop", entry="start",
                                   id="id", failure="event", x=["treat", "t2", "age"])
    assert [c.term for c in collinear.coefficients] == ["treat", "age"]
    assert any("t2" in warning for warning in collinear.warnings)
    with pytest.raises(TypeError):
        oe.stcox_recurrent(data=DATA, time="stop", id="id", failure="event", x=TERMS,
                           weights="age")


def test_round_trip_and_rendering():
    result = fit("wlw", 3, "efron")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.coefficients == result.coefficients
    assert restored.covariance_matrix == result.covariance_matrix
    assert restored.extra == result.extra and restored.metrics == result.metrics
    text = result.summary()
    assert "Wei-Lin-Weissfeld" in text and "Wald chi2" in text
    assert "\\begin{tabular" in result.summary("latex")
    assert result.provenance["stata_parity_validated"] is False
