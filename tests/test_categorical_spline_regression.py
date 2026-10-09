"""Independent full-design and convex-cone oracles for degree-one CATREG."""
import copy
import itertools

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical import spline_core as core
from openecon.econometrics.categorical.spline_regression import (
    catreg_mspline, catreg_spline, catreg_spline_predict,
)
from openecon.econometrics.categorical.optimal import _seal
from openecon.econometrics.summary_state import restore_summary, summary_state


def fixture():
    generator = np.random.default_rng(829)
    x = generator.uniform(0, 3, 96)
    z = generator.uniform(-1, 2, 96)
    y = 2+1.1*x+0.7*np.maximum(x-1, 0)-0.9*z-0.4*np.maximum(z, 0)+generator.normal(0, .1, 96)
    return pd.DataFrame({"x": x, "z": z, "y": y}), {"x": [0, 1, 2, 3], "z": [-1, 0, 1, 2]}


def design(frame, knots):
    # Independently assemble continuous hat interpolation at the knot values;
    # remove the first constant direction. This spans the ramp-based design.
    blocks = []
    for name, values in knots.items():
        x = frame[name].to_numpy()
        hats = np.column_stack([np.interp(x, values, np.eye(len(values))[j]) for j in range(len(values))])
        blocks.append(hats[:, 1:])
    return np.column_stack([np.ones(len(frame))]+blocks)


def ramps(frame, knots):
    return np.column_stack([np.clip((frame[name].to_numpy()-a)/(b-a), 0, 1)
                            for name, values in knots.items() for a, b in zip(values, values[1:])])


def test_nonmonotone_joint_ols_and_saved_prediction():
    frame, knots = fixture()
    frame["y"] += -3*np.maximum(frame.x-2, 0)
    result = catreg_spline(frame, "y", ["x", "z"], knots=knots)
    matrix = design(frame, knots)
    coefficient = np.linalg.lstsq(matrix, frame.y, rcond=None)[0]
    np.testing.assert_allclose(result["fitted"].fitted, matrix@coefficient, atol=1e-10)
    query = pd.DataFrame({"x": [0., .5, 1., 2.5, 3.], "z": [-1., -.2, 0., 1.5, 2.]})
    predicted = catreg_spline_predict(restore_summary(summary_state(result)), query)
    np.testing.assert_allclose(predicted["predictions"].fitted, design(query, knots)@coefficient, atol=1e-10)
    transformed = result["transformed"][["x", "z"]].to_numpy()
    np.testing.assert_allclose(transformed.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose((transformed**2).mean(0), 1, atol=1e-12)
    assert "se" not in result["effects"].columns


@pytest.mark.parametrize("knots", [
    {"x": [0, 3], "z": [-1, 2]},
    {"x": [0, .7, 1.8, 3], "z": [-1, .4, 2]},
    {"x": [0, .4, 1, 1.9, 3], "z": [-1, -.1, .5, 1.4, 2]},
])
def test_monotone_all_signed_cones_against_scipy(knots):
    frame, _ = fixture()
    result = catreg_mspline(frame, "y", ["x", "z"], knots=knots)
    matrix = ramps(frame, knots)
    matrix -= matrix.mean(0)
    target = frame.y.to_numpy()-frame.y.mean()
    fits = []
    for signs in itertools.product([-1, 1], repeat=2):
        direction = np.repeat(signs, [len(v)-1 for v in knots.values()])
        coef, _ = scipy.optimize.nnls(matrix*direction, target)
        fits.append((np.sum((target-matrix@(coef*direction))**2), coef*direction))
    best = min(fits, key=lambda v: v[0])
    np.testing.assert_allclose(result["fit"].sse.iloc[0], best[0], atol=1e-10)
    np.testing.assert_allclose(result["fitted"].fitted, frame.y.mean()+matrix@best[1], atol=1e-10)
    for name in knots:
        values = result["spline_knots"].query("variable == @name").quantification.to_numpy()
        assert np.all(np.diff(values) >= -1e-12)
    assert result["effects"].coefficient.iloc[0] > 0
    assert result["effects"].coefficient.iloc[1] < 0


@pytest.mark.parametrize("seed", range(6))
def test_native_nnls_independent_constrained_oracle(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(30, 7))
    target = rng.normal(size=30)
    expected, _ = scipy.optimize.nnls(matrix, target)
    actual = core._nnls(torch.tensor(matrix), torch.tensor(target)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-11)


@pytest.mark.parametrize("column_scale", [1e-10, 1., 1e10])
@pytest.mark.parametrize("response_scale", [1e-10, 1., 1e10])
def test_nnls_acceptance_is_invariant_to_column_and_response_units(column_scale, response_scale):
    rng = np.random.default_rng(491)
    matrix = rng.normal(size=(40, 5))
    target = rng.normal(size=40)
    expected, _ = scipy.optimize.nnls(matrix, target)
    actual = core._nnls(torch.tensor(matrix*column_scale), torch.tensor(target*response_scale)).numpy()
    np.testing.assert_allclose(actual*column_scale/response_scale, expected, atol=1e-11)


def test_active_set_solves_use_compact_design_within_admitted_loop_work(monkeypatch):
    rng = np.random.default_rng(119)
    matrix, target = rng.normal(size=(200, 6)), rng.normal(size=200)
    expected, _ = scipy.optimize.nnls(matrix, target)
    calls = []
    original = torch.linalg.lstsq

    def compact(design, response, **kwargs):
        calls.append(design.shape)
        assert design.shape[0] == 6 and design.shape[1] <= 6
        return original(design, response, **kwargs)

    monkeypatch.setattr(torch.linalg, "lstsq", compact)
    actual = core._nnls(torch.tensor(matrix), torch.tensor(target)).numpy()
    assert calls and len(calls) <= 40*6
    np.testing.assert_allclose(actual, expected, atol=1e-11)


@pytest.mark.parametrize("fit", [catreg_spline, catreg_mspline])
def test_small_observed_ramp_range_resealed_nonoptimal_fit_is_refused(fit):
    from openecon.econometrics.categorical import spline_regression as module

    frame = pd.DataFrame({"x": np.linspace(0, 1e-10, 50)})
    frame["y"] = 3+1e10*frame.x
    result = fit(frame, "y", ["x"], knots={"x": [0, 1]})
    np.testing.assert_allclose(result["fitted"].fitted, frame.y, atol=1e-12)
    catreg_spline_predict(result, frame.iloc[:3])
    state = copy.deepcopy(result.attrs["spline_state"])
    state["coefficient"][0] *= .5
    state["candidate_coefficients"][state["chosen"]][0] *= .5
    x = frame.x.to_numpy()-frame.x.mean()
    target = frame.y.to_numpy()-frame.y.mean()
    state["patterns"][state["chosen"]][2] = float(np.sum((target-x*state["coefficient"][0])**2))
    forged = module._output(state)
    assert forged["fit"].sse.iloc[0] > 1
    with pytest.raises(AnalysisError, match="Saved spline"):
        catreg_spline_predict(forged, frame.iloc[:3])


@pytest.mark.parametrize("field", ["version", "chosen"])
def test_saved_integer_schema_fields_refuse_boolean_aliases(field):
    from openecon.econometrics.categorical import spline_regression as module

    frame, knots = fixture()
    state = copy.deepcopy(catreg_spline(frame, "y", ["x", "z"], knots=knots).attrs["spline_state"])
    state[field] = bool(state[field])
    with pytest.raises(AnalysisError, match="Saved spline"):
        catreg_spline_predict(module._output(state), frame.iloc[:3])


@pytest.mark.parametrize("fit", [catreg_spline, catreg_mspline])
def test_underflowed_ramp_norm_refuses_with_domain_error(fit):
    frame = pd.DataFrame({"x": np.linspace(0, 1e-170, 50)})
    frame["y"] = 3+frame.x/1e-170
    with pytest.raises(AnalysisError, match="rank deficient|ill conditioned"):
        fit(frame, "y", ["x"], knots={"x": [0, 1]})


def test_near_collinear_resealed_nonoptimal_fit_is_refused_in_qr_coordinates():
    from openecon.econometrics.categorical import spline_regression as module

    rng = np.random.default_rng(222)
    x = rng.uniform(.1, .9, 80)
    direction = rng.normal(size=80)
    direction -= direction.mean()
    centered = x-x.mean()
    direction -= np.dot(direction, centered)/np.dot(centered, centered)*centered
    frame = pd.DataFrame({"x": x, "z": x+8e-10*direction, "y": 3+direction})
    knots = {"x": [0, 1], "z": [0, 1]}
    result = catreg_spline(frame, "y", ["x", "z"], knots=knots)
    catreg_spline_predict(result, frame.iloc[:3])
    state = copy.deepcopy(result.attrs["spline_state"])
    state["coefficient"] = [v*.5 for v in state["coefficient"]]
    state["candidate_coefficients"][0] = state["coefficient"].copy()
    prepared = module._data(frame, "y", ["x", "z"], knots, "raise", core.LIMIT_BYTES, core.WORK)
    matrix = torch.cat(prepared["bases"], dim=1)
    target = prepared["y"]-prepared["y"].mean()
    state["patterns"][0][2] = float((target-matrix@torch.tensor(state["coefficient"], dtype=core.DT)).square().sum())
    forged = module._output(state)
    assert forged["fit"].sse.iloc[0] > 10
    with pytest.raises(AnalysisError, match="Saved spline"):
        catreg_spline_predict(forged, frame.iloc[:3])


def test_nearly_cancelling_inactive_cone_directions_cannot_hide_large_improvement():
    rng = np.random.default_rng(330)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(60, 2)))
    matrix = np.column_stack([-orthogonal[:, 0], orthogonal[:, 0]+1e-9*orthogonal[:, 1]])
    # Each raw derivative is tiny, but two positive coefficients can fit the
    # unit target exactly. Zero is therefore not an accepted cone optimum.
    assert not core._stationary(torch.tensor(matrix), torch.tensor(orthogonal[:, 1]),
                                torch.zeros(2, dtype=core.DT), nonnegative=True)


def test_active_residual_tolerance_cannot_mask_cancelling_inactive_improvement():
    rng = np.random.default_rng(330)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(60, 2)))
    matrix = np.column_stack([orthogonal[:, 0], -orthogonal[:, 0]+1e-9*orthogonal[:, 1]])
    target = (1+1e-7)*orthogonal[:, 0]+orthogonal[:, 1]
    assert not core._stationary(torch.tensor(matrix), torch.tensor(target),
                                torch.tensor([1., 0.], dtype=core.DT), nonnegative=True)
    # A nearly singular cone whose float64 solution cannot certify its full
    # KKT error fails explicitly, even if its raw gradient appears small.
    with pytest.raises(AnalysisError) as caught:
        core._nnls(torch.tensor(matrix), torch.tensor(target))
    assert caught.value.code == "nonconvergence"


@pytest.mark.parametrize("fit", [catreg_spline, catreg_mspline])
def test_missing_positions_persistence_and_no_refit(fit, monkeypatch):
    frame, knots = fixture()
    frame.loc[3, "y"] = np.nan
    frame.loc[8, "z"] = np.nan
    frame.index = [7]*len(frame)
    result = fit(frame, "y", ["x", "z"], knots=knots)
    assert result.attrs["sample_positions"] == [i for i in range(len(frame)) if i not in [3, 8]]
    text = summary_state(result)
    restored = restore_summary(text)
    assert summary_state(restored) == text
    assert restored.to_latex() == result.to_latex()
    import openecon.econometrics.categorical.spline_regression as module
    monkeypatch.setattr(module, "_fit_coefficients", lambda *args: pytest.fail("Restoration must not refit"))
    query = frame.iloc[:12].copy()
    prediction = catreg_spline_predict(restored, query, missing="drop")
    assert prediction.attrs["sample_positions"] == [i for i in range(12) if i != 8]


@pytest.mark.parametrize("mutation", ["coefficient", "cone", "raw", "knots", "position", "sse", "settings", "extra"])
def test_resealed_numerical_table_and_metadata_tampering_refused(mutation):
    frame, knots = fixture()
    result = copy.deepcopy(catreg_mspline(frame, "y", ["x", "z"], knots=knots))
    state = result.attrs["spline_state"]
    if mutation == "coefficient":
        state["coefficient"][0] += .1
    elif mutation == "cone":
        state["candidate_coefficients"][0][0] += .5
    elif mutation == "raw":
        state["raw"][0][0] += .2
    elif mutation == "knots":
        state["knots"]["x"][1] += .2
    elif mutation == "position":
        state["positions"][0] = -1
    elif mutation == "sse":
        result["fit"].iloc[0, 1] += .1
    elif mutation == "settings":
        result.attrs["weight_type"] = "survey"
    else:
        result.attrs["forged"] = True
    result.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError, match="Saved spline"):
        catreg_spline_predict(result, frame.iloc[:2])


@pytest.mark.parametrize("mutation", ["outside", "knots", "duplicates", "nonnumeric", "rank", "missing", "budget", "work", "device"])
def test_admission_and_resource_fail_closed(mutation):
    frame, knots = fixture()
    options = {}
    if mutation == "outside":
        frame.loc[0, "x"] = -1
    elif mutation == "knots":
        knots["x"] = [0, 1, 1, 3]
    elif mutation == "duplicates":
        frame = pd.concat([frame, frame[["x"]]], axis=1)
    elif mutation == "nonnumeric":
        frame["x"] = frame.x.astype(str)
    elif mutation == "rank":
        frame["z"] = frame.x-1
    elif mutation == "missing":
        frame.loc[0, "y"] = np.nan
        options["missing"] = "raise"
    elif mutation == "budget":
        options["max_bytes"] = 1
    elif mutation == "work":
        options["max_work"] = 1
    else:
        options["device"] = "cuda"
    with pytest.raises(AnalysisError):
        catreg_spline(frame, "y", ["x", "z"], knots=knots, **options)


def test_query_outside_support_and_empty_prediction_fail():
    frame, knots = fixture()
    result = catreg_spline(frame, "y", ["x", "z"], knots=knots)
    for query in [pd.DataFrame({"x": [4.], "z": [0.]}), pd.DataFrame({"x": [np.nan], "z": [0.]})]:
        with pytest.raises(AnalysisError):
            catreg_spline_predict(result, query, missing="drop")


def test_saved_validation_and_query_share_declared_work_before_tensors(monkeypatch):
    frame, knots = fixture()
    training = pd.concat([frame]*30, ignore_index=True)
    result = catreg_spline(training, "y", ["x", "z"], knots=knots)
    query = pd.concat([frame]*21, ignore_index=True)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("Combined work must refuse before tensor copies"))
    with pytest.raises(AnalysisError) as caught:
        catreg_spline_predict(result, query)
    assert caught.value.code == "resource_limit"
