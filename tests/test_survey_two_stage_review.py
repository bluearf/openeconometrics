"""Counterproofs for independently discovered two-stage admission defects."""

import numpy as np
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey.two_stage_regression_state import replay_fit
from openecon.survey_two_stage import SurveyTwoStageDesign, TwoStageValidation

from test_survey_two_stage_oracles import fixture, geometry, stage_covariance


@pytest.mark.parametrize("difference", [None, 1e-3, 1e-5])
def test_valid_correlated_linear_regressors_replay_against_independent_weighted_qr(difference):
    frame, _ = fixture()
    if difference is not None:
        frame["z"] = frame.x + difference*np.random.default_rng(8).normal(size=len(frame))
    design = oe.survey_two_stage_design(frame, psu="p", ssu="s", strata="h",
                                       population_psu="N", population_ssu="M")
    X = np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy()))
    y, weights = frame.y.to_numpy(), geometry(frame)[0]
    # QR of the weighted original design is independent of the native scaled
    # normal-equation algorithm and avoids squaring the condition number here.
    weighted = np.sqrt(weights)[:, None]*X
    Q, R = np.linalg.qr(weighted, mode="reduced")
    expected = np.linalg.solve(R, Q.T @ (np.sqrt(weights)*y))
    row_scores = (weights*(y-X @ expected))[:, None]*X
    influences = np.linalg.solve(R, np.linalg.solve(R.T, row_scores.T)).T
    first, second = stage_covariance(frame, influences)
    actual = oe.survey_two_stage_regress(frame, design, "y", ["x", "z"])
    # The implementation intentionally uses float64 normal equations. Its
    # forward error is conditioned by cond(X)^2, rather than a fixed arbitrary
    # tolerance on an inverse-amplified residual correction.
    relative_bound = max(1e-10, 20*np.finfo(float).eps*np.linalg.cond(weighted)**2)
    assert relative_bound < 1e-3
    np.testing.assert_allclose(actual.coefficients, expected, rtol=relative_bound, atol=1e-10)
    np.testing.assert_allclose(actual.covariance, first+second,
                               rtol=4*relative_bound, atol=1e-10)
    restored = oe.SurveyTwoStageRegressionResult.model_validate_json(actual.model_dump_json())
    assert restored.model_dump(mode="json") == actual.model_dump(mode="json")


def separated_record(family):
    """Manufacture a fully coherent but scientifically inadmissible saved fit.

    The production replay helper is used solely to make an adversarial record
    internally coherent. It is never a numerical reference for an assertion.
    Its private certify=False switch cannot bypass public state admission.
    """
    frame, design = fixture()
    name = "binary" if family in ("logit", "probit") else "count"
    original = getattr(oe, "survey_two_stage_"+family)(frame, design, name, ["x", "z"])
    raw = original.model_dump(mode="json")
    if family in ("logit", "probit"):
        y = frame.binary.to_numpy(dtype=float)
        frame["x"] = 2*y-1
        # Probit index8 still leaves representable observed tail curvature.
        beta = [0., 35. if family == "logit" else 8., 0.]
    else:
        y = 2*frame.binary.to_numpy(dtype=float)
        frame["count"] = y
        frame["x"] = np.where(y == 0, -1., 0.)
        beta = [float(np.log(2)), 35., 0.]
    X = torch.tensor(np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy())),
                     dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    positions = raw["metadata"]["sample_positions"]
    replay = replay_fit(design, X, y, positions, torch.tensor(beta, dtype=torch.float64),
                        family, certify=False)
    raw["coefficients"] = beta
    raw["covariance"] = replay["covariance"].tolist()
    metadata = raw["metadata"]
    metadata["primitive_X"], metadata["primitive_y"] = X.tolist(), y.tolist()
    for key in ("bread", "row_scores", "stage1_covariance", "stage2_covariance"):
        metadata[key] = replay[key].tolist()
    values = torch.column_stack((y, X[:, 1:]))
    metadata["sample_input_sha256"] = digest([
        design.validation.design_input_sha256, [True]*len(frame), values.tolist(),
        torch.ones_like(values).tolist(),
    ])
    metadata["convergence"].update(
        tolerance=.001, score_max_abs=float(replay["score"].abs().max())/len(y),
        objective=replay["objective"],
    )
    raw["integrity_sha256"] = digest({key: value for key, value in raw.items()
                                      if key != "integrity_sha256"})
    return frame, design, name, raw


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
def test_rehashed_separating_primitive_fit_cannot_be_restored(family):
    frame, design, name, raw = separated_record(family)
    with pytest.raises(AnalysisError, match="separating|separation|cone"):
        getattr(oe, "survey_two_stage_"+family)(frame, design, name, ["x", "z"], tolerance=.001)
    with pytest.raises((ValueError, AnalysisError), match="separating|separation|cone"):
        oe.SurveyTwoStageRegressionResult.model_validate(raw)


def nested_record(memory):
    return {
        "psu": "p", "ssu": "s", "strata": "h", "population_psu": "N",
        "population_ssu": "M", "max_memory_mb": memory,
        "validation": {
            "nobs": 1000, "n_strata": 1, "n_psu": 2, "design_df": 1,
            "sum_weights": 4000., "design_input_sha256": "0"*64,
            "strata": [{"n_psu": 2, "population_psu": 4, "psu_indices": [0, 1]}],
            "psus": [
                {"stratum_index": 0, "n_ssu": 500, "population_ssu": 1000,
                 "row_positions": list(range(500))},
                {"stratum_index": 0, "n_ssu": 500, "population_ssu": 1000,
                 "row_positions": list(range(500, 1000))},
            ],
        },
    }


@pytest.mark.parametrize("memory", [1, "1", True, None])
def test_saved_design_rejects_budget_before_expanding_any_derived_weights(memory, monkeypatch):
    def forbidden(self):
        pytest.fail("Full derived weight expansion occurred before saved-design budget admission.")

    monkeypatch.setattr(TwoStageValidation, "weights", property(forbidden))
    with pytest.raises((ValueError, AnalysisError)):
        SurveyTwoStageDesign.model_validate(nested_record(memory))


def test_saved_design_admits_sufficient_budget_before_expanding_weights(monkeypatch):
    called = []
    original = TwoStageValidation.weights.fget

    def observed(self):
        called.append(self.nobs)
        return original(self)

    monkeypatch.setattr(TwoStageValidation, "weights", property(observed))
    state = SurveyTwoStageDesign.model_validate(nested_record(2))
    assert called == [1000]
    assert state.validation.nobs == 1000
    assert state.validation.sum_weights == 4000.
