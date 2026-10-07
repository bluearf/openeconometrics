"""Independent nonlinear/robust full-source oracles and replay contracts."""
import hashlib
import json

import numpy as np
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.quantile.nl import fit_nl
from openecon.econometrics.quantile.rreg import fit_rreg
from openecon.econometrics.streaming_nonlinear import fit_streaming_nonlinear, _median, _scratch
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget
from test_econ_quantile_nl import DECAY, DECAY_START, make_data as nonlinear_data, decay_jacobian, scipy_fit
from test_econ_quantile_rreg import make_data as robust_data, rreg_oracle


def params(result):
    return np.array([row.estimate for row in result.coefficients])


def nl_spec(covariance="nonrobust", weight_type=None):
    return ModelSpec(estimator="nl", outcome="y", predictors=["x", "z"], intercept=False,
                     covariance=covariance, cluster="g" if covariance == "cluster" else None,
                     weights="f" if weight_type == "fweight" else "w" if weight_type else None,
                     weight_type=weight_type, options={"formula":DECAY, "start":DECAY_START})


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "HC2", "HC3", "cluster"])
@pytest.mark.parametrize("weight_type", [None, "fweight", "aweight", "pweight"])
def test_nonlinear_global_fit_and_all_covariances_match_independent_oracle(kind, weight_type):
    if weight_type == "pweight" and kind == "nonrobust":
        with pytest.raises(AnalysisError) as error:
            fit_streaming_nonlinear(nl_spec(kind, weight_type), Dataset.from_frame(nonlinear_data()), batch_rows=17)
        assert error.value.code == "unsupported_covariance"
        return
    frame = nonlinear_data()
    spec = nl_spec(kind, weight_type)
    result = fit_streaming_nonlinear(spec, Dataset.from_frame(frame), batch_rows=17)
    dense = fit_nl(spec, frame)
    raw = None if weight_type is None else frame[spec.weights].to_numpy()
    weights = np.ones(len(frame)) if raw is None else raw.copy()
    if weight_type in {"aweight", "pweight"}:
        weights *= len(frame)/weights.sum()
    oracle = scipy_fit(frame, weights=weights)
    np.testing.assert_allclose(params(result), oracle, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(params(result), params(dense), rtol=3e-8, atol=3e-9)
    beta = params(result)
    j = decay_jacobian(frame, beta)
    residual = frame.y.to_numpy()-(beta[0]+beta[1]*np.exp(-beta[2]*frame.x.to_numpy())+beta[3]*frame.z.to_numpy())
    bread = np.linalg.inv(j.T@(weights[:, None]*j))
    n = int(weights.sum()) if weight_type == "fweight" else len(frame)
    if kind == "nonrobust":
        expected = bread*(weights@residual**2/(n-len(beta)))
    else:
        if kind == "cluster":
            scores = j*(residual*weights)[:, None]
            grouped = np.array([scores[frame.g.to_numpy() == group].sum(0) for group in np.unique(frame.g)])
            meat = grouped.T@grouped
            factor = len(grouped)/(len(grouped)-1)*(n-1)/(n-len(beta))
        else:
            adjusted = residual.copy()
            if kind in {"HC2", "HC3"}:
                h = np.einsum("ij,jk,ik->i", j, bread, j)
                if weight_type != "fweight":
                    h *= weights
                adjusted /= np.sqrt(1-h) if kind == "HC2" else 1-h
            scores = j*(adjusted*(np.sqrt(weights) if weight_type == "fweight" else weights))[:, None]
            meat = scores.T@scores
            factor = n/(n-len(beta)) if kind == "robust" else 1.
        expected = bread@(meat*factor)@bread
    np.testing.assert_allclose(result.covariance_matrix, expected, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(result.covariance_matrix, dense.covariance_matrix, rtol=2e-7, atol=2e-9)
    centered = frame.y.to_numpy()-np.average(frame.y, weights=weights)
    assert result.metrics["tss"] == pytest.approx(weights@centered**2, rel=2e-12)
    assert result.metrics["tss"] == pytest.approx(dense.metrics["tss"], rel=2e-12)
    assert result.extra["constant_term"] == "b0"
    assert result.provenance["streaming"]["dense_observation_matrix"] is False
    assert result.provenance["streaming"]["maximum_batch_rows"] <= 17
    assert result.sample_positions == []
    assert ResultBundle.model_validate_json(result.model_dump_json()).nobs == n


@pytest.mark.parametrize("batch_rows", [1, 17, 231])
@pytest.mark.parametrize("categorical", [False, True])
def test_rreg_exact_global_screen_mad_weights_and_pseudovalues(batch_rows, categorical, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = robust_data(n=160)
    predictors = ["x1", "x2", *(["sector"] if categorical else [])]
    spec = ModelSpec(estimator="rreg", outcome="y", predictors=predictors, categorical=["sector"] if categorical else [])
    result = fit_streaming_nonlinear(spec, Dataset.from_frame(frame), batch_rows=batch_rows)
    dense = fit_rreg(spec, frame)
    x = np.column_stack([np.ones(len(frame)), frame[["x1", "x2"]],
                         *([(frame.sector.to_numpy() == level).astype(float) for level in frame.sector.cat.categories[1:]] if categorical else [])])
    oracle = rreg_oracle(x, frame.y.to_numpy())
    np.testing.assert_allclose(params(result), oracle["beta"], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(result.covariance_matrix, oracle["cov"], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(params(result), params(dense), rtol=2e-10, atol=2e-11)
    assert result.metrics["scale"] == pytest.approx(oracle["scale"], rel=2e-11)
    assert result.nobs == int(oracle["keep"].sum())
    assert result.metrics["n_dropped_cooks"] == int((~oracle["keep"]).sum())
    assert result.metrics["huber_iterations"] == sum(stage == "huber" for stage, _ in oracle["log"])
    assert result.metrics["biweight_iterations"] == sum(stage == "biweight" for stage, _ in oracle["log"])
    for key, expected in [("min", oracle["weights"].min()), ("max", oracle["weights"].max()), ("mean", oracle["weights"].mean())]:
        assert result.extra["weights"][key] == pytest.approx(expected, abs=2e-12)
    retained = np.flatnonzero(oracle["keep"]).astype("<i8")
    assert result.provenance["sample_positions_hash"] == hashlib.sha256(retained.tobytes()).hexdigest()
    assert [row["row"] for row in result.predictions] == retained.tolist()[:400]
    assert result.provenance["solver_diagnostics"]["scratch_bytes"] > 0
    assert not list(tmp_path.iterdir())


def test_sqlite_median_is_global_even_ties_and_not_a_median_of_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with _scratch() as (db, _):
        values = [0., 0., 1., 2., 100., 200.]
        db.executemany("INSERT INTO residuals VALUES (?)", ((value,) for value in values))
        center = _median(db)
        assert center == np.median(values)
        assert _median(db, "abs(value-?)", (center,)) == np.median(np.abs(np.array(values)-center))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["nl", "rreg"])
def test_missing_file_backed_meta_scope_reporting_and_scratch_cleanup(estimator, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = nonlinear_data(n=703) if estimator == "nl" else robust_data(n=703)
    column = "z" if estimator == "nl" else "x2"
    frame.loc[[3, 19], column] = np.nan
    spec = nl_spec() if estimator == "nl" else ModelSpec(estimator="rreg", outcome="y", predictors=["x1", "x2"])
    spec = spec.model_copy(update={"missing":"drop"})
    path = tmp_path/"data.parquet"
    frame.to_parquet(path, index=False)
    default = torch.get_default_dtype()
    with torch.device("meta"):
        result = fit_streaming_nonlinear(spec, scan(path), batch_rows=31)
        assert torch.empty(0).device.type == "meta"
    assert torch.get_default_dtype() == default
    assert len(result.predictions) == 400
    assert result.nobs_original == 703
    assert result.dropped_rows >= 2
    assert all(row["row"] not in {3,19} for row in result.predictions)
    assert not list(tmp_path.glob("*.sqlite*"))
    json.loads(result.model_dump_json())


def test_nl_start_rank_and_resource_guards_are_explicit(monkeypatch):
    frame = nonlinear_data()
    bad = nl_spec().model_copy(update={"predictors":["x"], "options":{"formula":"ln({a=-1})+x"}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming_nonlinear(bad, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == "invalid_start"
    redundant = nl_spec().model_copy(update={"predictors":["x"], "options":{"formula":"{a=1}+{b=1}+{c=1}*x"}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming_nonlinear(redundant, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == "not_identified"
    import openecon.econometrics.streaming_nonlinear as implementation
    wide = nl_spec().model_copy(update={"predictors":["x"], "options":{
        "formula":"+".join(f"{{p{index}=1}}*x" for index in range(64))}})
    monkeypatch.setattr(implementation, "_formula_values", lambda *args: pytest.fail("budget guard must precede native formula allocation"))
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit_streaming_nonlinear(wide, Dataset.from_frame(frame))
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("estimator", ["nl", "rreg"])
def test_changed_projected_source_rejects_fit_and_removes_owned_scratch(estimator, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = nonlinear_data() if estimator == "nl" else robust_data()
    column = "x" if estimator == "nl" else "x1"
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        data = frame.copy()
        if calls >= 6:
            data.loc[5, column] += .01
        yield data

    spec = nl_spec() if estimator == "nl" else ModelSpec(estimator="rreg", outcome="y", predictors=["x1", "x2"])
    source = Dataset.from_batches(factory, columns=list(frame.columns))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_nonlinear(spec, source, batch_rows=31)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["nonrobust", "HC3"])
def test_direct_nl_formula_infers_columns_and_uncentered_metrics_without_additive_constant(kind):
    frame = nonlinear_data()
    spec = ModelSpec(estimator="nl", outcome="y", predictors=[], covariance=kind, intercept=False,
                     options={"formula":"{slope=1}*x"})
    result = fit_streaming_nonlinear(spec, Dataset.from_frame(frame), batch_rows=17)
    x, y = frame.x.to_numpy(), frame.y.to_numpy()
    beta = x@y/(x@x)
    residual = y-x*beta
    rss = residual@residual
    expected = rss/(len(frame)-1)/(x@x) if kind == "nonrobust" else (
        np.sum((x*residual/(1-x**2/(x@x)))**2)/(x@x)**2)
    assert params(result)[0] == pytest.approx(beta, rel=2e-12)
    assert result.covariance_matrix[0][0] == pytest.approx(expected, rel=2e-12)
    assert result.metrics["tss"] == pytest.approx(y@y, rel=2e-12)
    assert result.extra["constant_term"] is None
    assert result.spec.predictors == []
    assert result.provenance["streaming"]["source"]["kind"] == "frame"


def test_rreg_without_intercept_matches_global_independent_oracle():
    frame = robust_data()
    spec = ModelSpec(estimator="rreg", outcome="y", predictors=["x1", "x2"], intercept=False)
    result = fit_streaming_nonlinear(spec, Dataset.from_frame(frame), batch_rows=17)
    expected = rreg_oracle(frame[["x1", "x2"]].to_numpy(), frame.y.to_numpy())
    np.testing.assert_allclose(params(result), expected["beta"], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(result.covariance_matrix, expected["cov"], rtol=2e-10, atol=2e-11)
    assert result.metrics["df_model"] == 2
