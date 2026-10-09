"""Independent spline subspace/cone oracles and portable local ALS state."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.spline_pca import (
    _block, _checked, _geometry, _seal, catpca_mspline, catpca_spline,
    catpca_spline_predict,
)
from openecon.econometrics.summary_state import restore_summary, summary_state


def spline_sample():
    rng = np.random.default_rng(2027)
    x = rng.uniform(-2, 2, 100)
    frame = pd.DataFrame({"x": x, "y": x*x+.04*rng.normal(size=100)})
    return frame, {"x": [-2., 0., 2.], "y": [-.2, 1., 4.5]}


@pytest.mark.parametrize("monotone", [False, True])
def test_rank_one_block_against_independent_eigenvalue_or_global_angle(monotone):
    rng = np.random.default_rng(118)
    x = np.sort(rng.uniform(-2, 2, 70))
    raw = np.column_stack([np.clip((x+2)/2, 0, 1), np.clip(x/2, 0, 1)])
    basis = raw-raw.mean(0)
    scores = rng.normal(size=(70, 2))
    scores -= scores.mean(0)
    scores = np.linalg.qr(scores)[0]*np.sqrt(70)
    b, s = torch.tensor(basis), torch.tensor(scores)
    coefficient, quantified, objective = _block(b, s, _geometry([b], monotone)[0], monotone)
    gram = basis.T@basis/70
    cross = basis.T@scores/70
    operator = cross@cross.T
    if monotone:
        def value(angle):
            c = np.array([np.cos(angle), np.sin(angle)])
            return -c@operator@c/(c@gram@c)
        grid = np.linspace(0, np.pi/2, 1001)
        best = grid[np.argmin([value(t) for t in grid])]
        result = minimize_scalar(value, bounds=(max(0, best-.003), min(np.pi/2, best+.003)), method="bounded", options={"xatol": 1e-14})
        expected = -min(value(0), value(np.pi/2), result.fun)
        assert np.min(coefficient.numpy()) >= 0
        assert np.min(np.diff(quantified.numpy())) >= -1e-13
    else:
        expected = eigh(operator, gram)[0][-1]
    assert objective == pytest.approx(expected, abs=2e-12)
    assert float(quantified.square().mean()) == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("method", [catpca_spline, catpca_mspline])
def test_no_interior_knots_reduces_to_independent_correlation_pca(method):
    rng = np.random.default_rng(44)
    frame = pd.DataFrame(rng.uniform(-1, 1, (80, 4)), columns=list("abcd"))
    frame["d"] = .7*frame.a+.3*frame.b
    fit = method(frame, list(frame), knots={name: [-1., 1.] for name in frame}, components=2, n_starts=1)
    z = (frame.to_numpy()-frame.to_numpy().mean(0))/frame.to_numpy().std(0)
    values, axes = np.linalg.eigh(z.T@z/len(z))
    values, axes = values[::-1], axes[:, ::-1][:, :2]
    actual = fit["eigenvectors"].to_numpy(float)
    np.testing.assert_allclose(fit["eigenvalues"].eigenvalue, values, atol=2e-12)
    np.testing.assert_allclose(actual@actual.T, axes@axes.T, atol=2e-12)
    scores = fit["scores"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(scores.T@scores/len(z), np.eye(2), atol=2e-12)


def test_freely_signed_and_monotone_fits_are_distinct_objectives():
    frame, knots = spline_sample()
    free = catpca_spline(frame, list(frame), knots=knots, components=1, n_starts=2, maxiter=150)
    monotone = catpca_mspline(frame, list(frame), knots=knots, components=1, n_starts=2, maxiter=150)
    assert free["fit"].reconstruction_loss.iloc[0] < monotone["fit"].reconstruction_loss.iloc[0]-.15
    for name in frame:
        mapping = monotone["splines"].query("variable == @name")
        assert mapping.coefficient.min() >= 0
    assert free["splines"].coefficient.min() < 0


@pytest.mark.parametrize("method", [catpca_spline, catpca_mspline])
def test_complete_json_latex_restoration_and_continuous_new_rows(method, monkeypatch):
    frame, knots = spline_sample()
    fit = method(frame, list(frame), knots=knots, components=1, n_starts=1, maxiter=150)
    restored = restore_summary(summary_state(fit))
    assert summary_state(restored) == summary_state(fit)
    assert restored.to_latex() == fit.to_latex()
    monkeypatch.setattr("openecon.econometrics.categorical.spline_pca._fit", lambda *a, **kw: pytest.fail("Refit"))
    query = pd.DataFrame({"x": [-.25, .125, 0., 0.], "y": [.4, .75, np.nan, .7]})
    prediction = catpca_spline_predict(restored, query, missing="drop")
    assert prediction.attrs["sample_positions"] == [0, 1, 3]
    state = fit.attrs["spline_state"]
    columns = []
    for j, name in enumerate(frame):
        k = np.array(knots[name])
        raw = query.loc[[0, 1, 3], name].to_numpy()
        ramps = np.clip((raw[:, None]-k[:-1])/(k[1:]-k[:-1]), 0, 1)
        columns.append((ramps-np.array(state["basis_means"][j]))@np.array(state["coefficients"][j]))
    expected = np.column_stack(columns)
    np.testing.assert_allclose(prediction["transformed"].iloc[:, 1:], expected, atol=1e-13)
    projected = expected@np.array(state["axes"])/np.sqrt(state["eigenvalues"][:1])
    np.testing.assert_allclose(prediction["scores"].iloc[:, 1:], projected, atol=1e-13)
    with pytest.raises(AnalysisError, match="span"):
        catpca_spline_predict(restored, pd.DataFrame({"x": [2.01], "y": [.7]}))


@pytest.mark.parametrize("mutation", ["coefficient", "raw", "knots", "trace", "version", "table", "extra"])
def test_saved_calibration_and_trace_tampering_is_refused(mutation):
    frame, knots = spline_sample()
    result = copy.deepcopy(catpca_spline(frame, list(frame), knots=knots, components=1, n_starts=1, maxiter=150))
    state = result.attrs["spline_state"]
    if mutation == "coefficient":
        state["coefficients"][0][0] += .1
    elif mutation == "raw":
        state["raw"][0][0] += .1
    elif mutation == "knots":
        state["knots"]["x"][1] += .1
    elif mutation == "trace":
        state["histories"][-1][3] += .1
        result["iterations"].iloc[-1, 3] += .1
    elif mutation == "version":
        state["version"] = True
    elif mutation == "table":
        result["scores"].iloc[0, 1] += .1
    else:
        state["extra"] = 1
    result.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError):
        _checked(result, 128*1024**2, 300_000_000, "cpu")


def test_early_work_admission_before_tensor_and_bad_support(monkeypatch):
    frame, knots = spline_sample()
    with pytest.raises(AnalysisError):
        catpca_mspline(frame, list(frame), knots={"x": [-2, 0, 0, 2], "y": knots["y"]}, components=1)
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Allocated before work admission"))
    with pytest.raises(AnalysisError, match="work"):
        catpca_spline(frame, list(frame), knots=knots, components=1, max_work=1)


@pytest.mark.parametrize("bad", [["x", "x"], [["x"], "y"], [1, "y"], ["x"*129, "y"]])
def test_invalid_variable_names_are_analysis_errors(bad):
    frame, knots = spline_sample()
    with pytest.raises(AnalysisError):
        catpca_spline(frame, bad, knots=knots, components=1)


def test_missing_physical_alignment_empty_query_and_monotone_discrete_values():
    frame, knots = spline_sample()
    frame.index = [7]*len(frame)
    frame.iloc[3, 0] = np.nan
    fit = catpca_mspline(frame, list(frame), knots=knots, components=1, n_starts=1, maxiter=150)
    assert 3 not in fit.attrs["sample_positions"]
    result = catpca_spline_predict(fit, frame.iloc[:0])
    assert result["scores"].shape == (0, 2)
    with pytest.raises(AnalysisError, match="missing"):
        catpca_mspline(frame, list(frame), knots=knots, components=1, missing="raise")
    # Observed numeric category positions can be discrete; their distances and
    # caller-declared knot positions remain explicit, never inferred labels.
    discrete = pd.DataFrame({"a": [0., 1., 2., 3.]*8, "b": [0., 1., 3., 4.]*8})
    result = catpca_mspline(discrete, ["a", "b"], knots={"a": [0, 1, 3], "b": [0, 2, 4]}, components=1, n_starts=1)
    assert result["splines"].coefficient.min() >= 0


def test_saved_state_admission_precedes_numerical_copy(monkeypatch):
    frame, knots = spline_sample()
    fit = catpca_spline(frame, list(frame), knots=knots, components=1, n_starts=1, maxiter=150)
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Tensor before admission"))
    with pytest.raises(AnalysisError, match="work"):
        catpca_spline_predict(fit, frame.iloc[:1], max_work=1)


def test_saved_validation_plus_query_is_one_work_budget(monkeypatch):
    frame, knots = spline_sample()
    fit = catpca_spline(frame, list(frame), knots=knots, components=1, n_starts=1, maxiter=150)
    _, baseline = _checked(fit, 128*1024**2, 300_000_000, "cpu")
    # Four ramp columns plus two scalar-variable projection products.
    allowance = baseline["planned_work"]+64*len(frame)*(4+2)//2
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Tensor before aggregate budget"))
    with pytest.raises(AnalysisError, match="work"):
        catpca_spline_predict(fit, frame, max_work=allowance)


@pytest.mark.parametrize("method", [catpca_spline, catpca_mspline])
def test_zero_loading_and_tied_roots_do_not_destroy_valid_component_space(method):
    orthogonal = pd.DataFrame({"a": [-1., -1., 1., 1.]*4, "b": [-1., 1., -1., 1.]*4})
    result = method(orthogonal, ["a", "b"], knots={"a": [-1, 1], "b": [-1, 1]}, components=1, n_starts=1)
    assert result["fit"].reconstruction_loss.iloc[0] == pytest.approx(1.)
    # The first all-linear start has rank one, but other starts span two ramp
    # directions. All feasible directions are tied once that full span is fit.
    x = np.linspace(-1, 1, 50)
    repeated = pd.DataFrame(dict(a=x, b=x, c=x))
    result = method(repeated, list(repeated), knots={name: [-1, 0, 1] for name in repeated},
                    components=2, n_starts=3, maxiter=30)
    assert result["starts"].iloc[0].status == "rank_deficient"
    assert result["fit"].reconstruction_loss.iloc[0] < 1e-20
    restored = restore_summary(summary_state(result))
    assert summary_state(restored) == summary_state(result)
    _checked(restored, 128*1024**2, 300_000_000, "cpu")


@pytest.mark.parametrize("method", [catpca_spline, catpca_mspline])
def test_saved_tied_component_basis_is_preserved_without_new_svd(method, monkeypatch):
    raw = np.array([[(-1.)**((i >> j) & 1) for j in range(4)] for i in range(16)])
    frame = pd.DataFrame(raw, columns=list("abcd"))
    fit = method(frame, list(frame), knots={name: [-1, 1] for name in frame}, components=2, n_starts=1)
    np.testing.assert_allclose(fit["eigenvalues"].eigenvalue, np.ones(4), atol=1e-13)
    restored = restore_summary(summary_state(fit))
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **kw: pytest.fail("Replaced saved tied basis"))
    projected = catpca_spline_predict(restored, frame)
    np.testing.assert_allclose(projected["scores"], fit["scores"], atol=1e-13)
