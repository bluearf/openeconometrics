"""Typed source alignment, semantic replay, local RNG and pre-copy admission."""
import copy
import json

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian import commands
from openecon.econometrics.bayesian.commands import bayes_contrast, bayes_draws, bayes_linear, bayes_predict
from openecon.econometrics.bayesian.core import digest
from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior, PosteriorBundle, PosteriorContrast, PosteriorDraws
from openecon.resources import use_workspace_budget

from pathlib import Path
from copy import deepcopy
import hashlib
import types
from typing import get_args, get_origin, Union
from pydantic import BaseModel
from openecon.econometrics.mi import common as _index_common
from openecon.econometrics.bayesian import core as _index_core, posterior as _index_posterior


def fixture(index=None):
    data = pd.DataFrame(dict(y=[1., 3., 2., 5., 4.], x=[0., 1., 2., 3., 4.]), index=index)
    prior = dict(mean=[.2, -.1], scale_matrix=[[1.2, .1], [.1, .8]], shape=1.2, scale=.8)
    return data, prior


def fit(index=None, **kwargs):
    data, prior = fixture(index)
    return bayes_linear(data=data, y="y", x=["x"], prior=prior, **kwargs)


def rehash(mapping):
    mapping["integrity_sha256"] = digest({k: v for k, v in mapping.items() if k != "integrity_sha256"})
    return mapping


@pytest.mark.parametrize("index", [
    pd.RangeIndex(20, 30, 2, name="rows"),
    pd.Index(["a", "a", "b", "c", "d"], name="row"),
    pd.Index([1, "b", None, 4., ("e", 2)], dtype=object, name=("label", 3)),
    pd.date_range("2026-01-01", periods=5, tz="Europe/Istanbul", name="date"),
    pd.TimedeltaIndex([pd.Timedelta(days=i) for i in range(5)], name="time"),
    pd.CategoricalIndex(["b", "a", "a", "c", "b"], categories=["a", "b", "c", "unused"], ordered=True, name="kind"),
    pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2), ("b", 2)], names=["group", "row"]),
])
def test_exact_typed_index_json_and_saved_predictions(index):
    fitted = fit(index)
    saved = fitted.model_dump_json()
    restored = PosteriorBundle.model_validate_json(saved)
    assert restored.model_dump_json() == saved
    data, _ = fixture(index)
    original = bayes_predict(result=fitted, data=data)
    replay = bayes_predict(result=saved, data=data)
    pd.testing.assert_frame_equal(original, replay)
    pd.testing.assert_index_equal(replay.index, index)


def test_nullable_typed_source_and_original_missing_sample_positions():
    data = pd.DataFrame({"y": pd.Series([1, 2, None, 4, 5], dtype="Int64"),
                         "x": pd.Series([0, None, 2, 3, 4], dtype="Float64")}, index=pd.RangeIndex(5, name="source"))
    _, prior = fixture()
    fitted = bayes_linear(data=data, y="y", x=["x"], prior=prior, missing="drop")
    assert fitted.state.source_dtypes == ("Int64", "Float64")
    assert fitted.state.sample_positions == (0, 3, 4)
    assert fitted.nobs_original == 5 and fitted.nobs == 3
    assert fitted.state.source_values == ((1, 2, None, 4, 5), (0., None, 2., 3., 4.))
    restored = PosteriorBundle.model_validate_json(fitted.model_dump_json())
    predictions = bayes_predict(result=restored, data=data, missing="drop")
    assert predictions.attrs["sample_positions"] == [0, 2, 3, 4]
    pd.testing.assert_index_equal(predictions.index, data.index[[0, 2, 3, 4]])


@pytest.mark.parametrize("field", ["mean", "conditional_scale_matrix", "coefficient_scale_matrix",
                                    "coefficient_covariance", "credible_intervals"])
def test_rehashed_full_numeric_state_tampering_refuses(field):
    body = fit().model_dump(mode="json")
    if field == "mean":
        body[field][0] += .02
    else:
        body[field][0][1] += .02
    with pytest.raises((ValidationError, AnalysisError), match="disagrees|positive|symmetric"):
        PosteriorBundle.model_validate(rehash(body))


@pytest.mark.parametrize("field", ["shape", "scale", "degrees_of_freedom", "variance_mean", "log_marginal_likelihood"])
def test_rehashed_scalar_state_tampering_refuses(field):
    body = fit().model_dump(mode="json")
    body[field] *= 1.03
    with pytest.raises(ValidationError, match="disagrees"):
        PosteriorBundle.model_validate(rehash(body))


def test_rehashed_source_sample_order_design_and_rank_tampering_refuse():
    original = fit().model_dump(mode="json")
    variants = []
    changed = copy.deepcopy(original)
    changed["state"]["sample_positions"] = [1, 0, 2, 3, 4]
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["terms"] = ["x", "Intercept"]
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["source_rank"] = 1
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["state"]["source_values"][0][0] += 1
    source = changed["state"]
    source["source_sha256"] = digest({"columns": source["source_columns"], "dtypes": source["source_dtypes"],
                                     "values": source["source_values"], "index": fit().state.source_index.descriptor()})
    variants.append(changed)
    for variant in variants:
        with pytest.raises((ValidationError, AnalysisError), match="disagrees"):
            PosteriorBundle.model_validate(rehash(variant))


@pytest.mark.parametrize("outcome,predictor", [
    ("y", "Intercept"),  # Otherwise a mapping contrast addresses two coefficients.
    ("y", "y"),
    (" ", "x"),
    ("y", " "),
    ("y" * 1001, "x"),
    ("y", "x" * 1001),
])
def test_rehashed_invalid_source_names_refuse_like_original_fit(outcome, predictor):
    fitted = fit()
    body = fitted.model_dump(mode="json")
    body["terms"] = ["Intercept", predictor]
    state = body["state"]
    state["outcome"], state["predictors"] = outcome, [predictor]
    state["source_columns"] = [outcome, predictor]
    # Renaming does not change the numeric algebra. Refresh both digests so that
    # refusal proves the declared input domain rather than checksum corruption.
    state["source_sha256"] = digest({
        "columns": state["source_columns"], "dtypes": state["source_dtypes"],
        "values": state["source_values"], "index": fitted.state.source_index.descriptor(),
    })
    with pytest.raises(ValidationError, match="Source names|coefficient order"):
        PosteriorBundle.model_validate(rehash(body))
    with pytest.raises((AnalysisError, ValidationError)):
        bayes_predict(result=rehash(body), data=pd.DataFrame({predictor: [1.]}))
    data, prior = fixture()
    data.columns = [outcome, predictor]
    with pytest.raises((AnalysisError, ValidationError)):
        bayes_linear(data=data, y=outcome, x=[predictor], prior=prior)


def test_tiny_covariance_refuses_relative_tampering_without_absolute_tolerance_floor():
    data, prior = fixture()
    data.y *= 1e-15
    prior["mean"] = [v*1e-15 for v in prior["mean"]]
    prior["scale"] *= 1e-30
    fitted = bayes_linear(data=data, y="y", x=["x"], prior=prior)
    body = fitted.model_dump(mode="json")
    body["coefficient_covariance"][0][1] *= 1.01
    with pytest.raises(ValidationError, match="disagrees"):
        PosteriorBundle.model_validate(rehash(body))


def test_subnormal_covariance_and_structural_zero_refuse_rehashed_forgeries():
    fitted = bayes_linear(data=pd.DataFrame({'y': [0., 0.]}), y='y',
                         prior=dict(mean=[0.], scale_matrix=[[1.]], shape=2., scale=1e-320))
    assert 0 < fitted.coefficient_covariance[0][0] < torch.finfo(torch.float64).tiny
    assert PosteriorBundle.model_validate_json(fitted.model_dump_json()) == fitted
    body = fitted.model_dump(mode='json')
    body['coefficient_covariance'][0][0] *= 4
    with pytest.raises(ValidationError, match='disagrees'):
        PosteriorBundle.model_validate(rehash(body))
    zero = bayes_contrast(result=fitted, weights=[0.])
    assert zero.posterior_variance == 0
    assert PosteriorContrast.model_validate_json(zero.model_dump_json()) == zero
    body = zero.model_dump(mode='json')
    body['posterior_variance'] = 1e-320
    with pytest.raises(ValidationError, match='disagrees'):
        PosteriorContrast.model_validate(rehash(body))


def test_saved_draws_target_seed_and_prediction_arrays_replay_exactly():
    fitted = fit()
    query = pd.DataFrame(dict(x=[.2, 1.2, -1.]), index=pd.Index(["same", "same", "other"], name="query"))
    draws = bayes_draws(result=fitted, data=query, draws=23, seed=811)
    restored = PosteriorDraws.model_validate_json(draws.model_dump_json())
    assert restored.model_dump_json() == draws.model_dump_json()
    assert len(restored.beta) == 23 and len(restored.outcome_draws[0]) == 3
    assert bayes_draws(result=fitted.model_dump_json(), data=query, draws=23, seed=811).model_dump_json() == draws.model_dump_json()
    for field in ("beta", "sigma_squared", "mean_draws", "outcome_draws", "seed"):
        body = draws.model_dump(mode="json")
        if field == "seed":
            body[field] += 1
        elif field == "sigma_squared":
            body[field][0] *= 1.1
        else:
            body[field][0][0] += .1
        with pytest.raises(ValidationError, match="disagrees"):
            PosteriorDraws.model_validate(rehash(body))


def test_saved_named_contrast_restores_complete_target_and_inference():
    fitted = fit()
    contrast = bayes_contrast(result=fitted, weights={"x": 1., "Intercept": -.25},
                             threshold=.3, alpha=.1)
    assert contrast.weights == (-.25, 1.) and contrast.terms == fitted.terms
    restored = PosteriorContrast.model_validate_json(contrast.model_dump_json())
    assert restored.model_dump_json() == contrast.model_dump_json()
    assert restored.posterior.model_dump_json() == fitted.model_dump_json()
    data, _ = fixture()
    pd.testing.assert_frame_equal(bayes_predict(result=restored.posterior, data=data),
                                  bayes_predict(result=fitted, data=data))
    for threshold in (-1., 0., 1.):
        zero = bayes_contrast(result=fitted, weights=[0., 0.], threshold=threshold)
        saved_zero = PosteriorContrast.model_validate_json(zero.model_dump_json())
        assert saved_zero.student_t_scale == 0 and saved_zero.posterior_variance == 0
        assert saved_zero.credible_lower == saved_zero.credible_upper == 0
        assert saved_zero.probability_above_threshold == float(0 > threshold)


@pytest.mark.parametrize("field", ["weights", "terms", "mean", "student_t_scale",
                                    "posterior_variance", "degrees_of_freedom", "alpha",
                                    "credible_lower", "credible_upper", "threshold",
                                    "probability_above_threshold", "posterior_sha256"])
def test_fully_rehashed_saved_contrast_inference_tampering_refuses(field):
    contrast = bayes_contrast(result=fit(), weights=[-.25, 1.], threshold=.3, alpha=.1)
    body = contrast.model_dump(mode="json")
    if field == "weights":
        body[field][0] += .05
    elif field == "terms":
        body[field].reverse()
    elif field == "posterior_sha256":
        body[field] = "0" * 64
    elif field in {"alpha", "probability_above_threshold"}:
        body[field] *= .9
    else:
        body[field] += .05
    with pytest.raises(ValidationError, match="disagrees|order"):
        PosteriorContrast.model_validate_json(json.dumps(rehash(body)))


def test_rehashed_contrast_complete_target_and_live_model_copy_refuse():
    fitted = fit()
    contrast = bayes_contrast(result=fitted, weights=[0., 1.])
    forged = fitted.model_copy(update={"mean": (fitted.mean[0] + .1, fitted.mean[1])})
    body = contrast.model_dump(mode="json")
    body["posterior"] = rehash(forged.model_dump(mode="json"))
    body["posterior_sha256"] = body["posterior"]["integrity_sha256"]
    with pytest.raises(ValidationError, match="Saved mean disagrees"):
        PosteriorContrast.model_validate(rehash(body))
    body["posterior"] = forged
    with pytest.raises(ValidationError, match="Saved mean disagrees"):
        PosteriorContrast.model_validate(body)
    copied = contrast.model_copy(update={"posterior": forged})
    with pytest.raises(ValidationError, match="Saved mean disagrees"):
        PosteriorContrast.model_validate(copied)
    copied = contrast.model_copy(update={"mean": contrast.mean + .1})
    with pytest.raises(ValidationError, match="contrast.mean disagrees"):
        PosteriorContrast.model_validate(copied)
    mutable_target = fitted.model_copy(update={"mean": list(fitted.mean)})
    copied = contrast.model_copy(update={"posterior": mutable_target, "weights": list(contrast.weights)})
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        restored = PosteriorContrast.model_validate(copied)
    assert isinstance(restored.posterior.mean, tuple) and isinstance(restored.weights, tuple)
    mutable_target.mean[0] += 1
    assert restored.posterior.mean == fitted.mean


def test_contrast_target_and_array_admission_precedes_copy_or_algebra(monkeypatch):
    fitted = fit()
    contrast = bayes_contrast(result=fitted, weights=[0., 1.])
    body = contrast.model_dump(mode="json")
    data, prior = fixture()
    body["posterior"] = bayes_linear(data=pd.concat([data] * 400, ignore_index=True),
                                     y="y", x=["x"], prior=prior)
    def forbidden(*args, **kwargs):
        raise AssertionError("contrast copy or algebra before admission")
    monkeypatch.setattr(PosteriorBundle, "model_dump", forbidden)
    with use_workspace_budget(1), pytest.raises((AnalysisError, ValidationError), match="budget"):
        PosteriorContrast.model_validate(body)
    body["posterior"] = fitted
    body["weights"] = [1.] * 100_000
    with pytest.raises(ValidationError, match="weights and terms"):
        PosteriorContrast.model_validate(body)


@pytest.mark.parametrize("field", ["weights", "threshold", "alpha", "mean"])
def test_saved_and_live_copied_contrast_refuse_numeric_boolean_coercion(field):
    contrast = bayes_contrast(result=fit(), weights=[0., 0.], threshold=0.)
    body = contrast.model_dump(mode="json")
    body[field] = [False, 0.] if field == "weights" else False
    with pytest.raises((ValidationError, AnalysisError)):
        PosteriorContrast.model_validate(rehash(body))
    copied = contrast.model_copy(update={field: (False, 0.) if field == "weights" else False})
    with pytest.raises((ValidationError, AnalysisError)):
        PosteriorContrast.model_validate(copied)


def test_ambient_torch_defaults_global_rng_and_input_are_preserved():
    data, prior = fixture()
    original = data.copy(deep=True)
    rng, dtype = torch.get_rng_state().clone(), torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            fitted = bayes_linear(data=data, y="y", x=["x"], prior=prior)
            bayes_predict(result=fitted, data=data)
            bayes_contrast(result=fitted, weights=[0., 1.])
            bayes_draws(result=fitted, data=data, draws=7, seed=19)
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(rng, torch.get_rng_state())
        pd.testing.assert_frame_equal(data, original)
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("options", [{"max_work": 1}, {"max_bytes": 1}, {"max_work": True}, {"device": "cuda"}])
def test_source_budget_refuses_before_numeric_or_index_copy(monkeypatch, options):
    data, prior = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("source conversion before admission")
    monkeypatch.setattr(commands, "source_values", forbidden)
    monkeypatch.setattr(commands, "_encode_index", forbidden)
    with pytest.raises(AnalysisError):
        bayes_linear(data=data, y="y", x=["x"], prior=prior, **options)


def test_global_workspace_draw_and_saved_state_admission(monkeypatch):
    fitted = fit()
    def forbidden(*args, **kwargs):
        raise AssertionError("draw allocation before admission")
    monkeypatch.setattr(commands, "_draw_arrays", forbidden)
    with pytest.raises(AnalysisError, match="2,000,000"):
        bayes_draws(result=fitted, draws=10_000, data=pd.DataFrame(dict(x=[1.]*101)))
    body = fitted.model_dump(mode="json")
    body["max_bytes"] = 1
    with pytest.raises((AnalysisError, ValidationError), match="budget"):
        PosteriorBundle.model_validate(rehash(body))
    data, prior = fixture()
    large = pd.concat([data]*400, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        bayes_linear(data=large, y="y", x=["x"], prior=prior)


@pytest.mark.parametrize("change", [
    {"shape": 0}, {"scale": 0}, {"shape": True}, {"shape": float("inf")},
    {"scale_matrix": [[1., .2], [.1, 1.]]}, {"scale_matrix": [[1., 2.], [2., 1.]]},
    {"mean": [0.]}, {"mean": [False, 0.]},
])
def test_improper_nonfinite_indefinite_or_mismatched_priors_refuse(change):
    data, prior = fixture()
    prior.update(change)
    with pytest.raises((AnalysisError, ValidationError)):
        bayes_linear(data=data, y="y", x=["x"], prior=prior)


@pytest.mark.parametrize("column", ["boolean", "complex", "string", "infinite", "unsafe_integer"])
def test_invalid_numeric_source_refuses(column):
    data, prior = fixture()
    replacements = {"boolean": [True]*5, "complex": [1+2j]*5,
                    "string": ["1"]*5, "infinite": [float("inf")]*5,
                    "unsafe_integer": [2**53+1]*5}
    data.x = replacements[column]
    with pytest.raises(AnalysisError):
        bayes_linear(data=data, y="y", x=["x"], prior=prior)


def test_unsupported_options_no_frequentist_inference_and_immutable_results():
    data, prior = fixture()
    for kwargs in ({"weights": "x"}, {"covariance": "robust"}, {"cluster": "x"}):
        with pytest.raises(TypeError):
            bayes_linear(data=data, y="y", x=["x"], prior=prior, **kwargs)
    fitted = fit()
    assert not {"p_value", "std_error", "confidence_interval", "r_hat", "ess"} & type(fitted).model_fields.keys()
    assert isinstance(fitted.mean, tuple) and isinstance(fitted.coefficient_covariance[0], tuple)
    with pytest.raises(ValidationError):
        fitted.alpha = .2
    with pytest.raises(ValidationError):
        fitted.prior.shape = 3
    assert "P>" not in fitted.to_latex()
    assert NormalInverseGammaPrior.model_validate(prior).shape == prior["shape"]


def test_corrupt_extra_unknown_json_state_and_missing_refusal():
    fitted = fit()
    body = json.loads(fitted.model_dump_json())
    body["p_value"] = .01
    with pytest.raises(ValidationError, match="Extra inputs"):
        PosteriorBundle.model_validate(rehash(body))
    data, prior = fixture()
    data.loc[1, "x"] = None
    with pytest.raises(AnalysisError, match="missing"):
        bayes_linear(data=data, y="y", x=["x"], prior=prior)
    with pytest.raises(AnalysisError, match="missing"):
        bayes_predict(result=fitted, data=data)
    for weights in ([1.], {"absent": 1}, [True, 1], [float("nan"), 1]):
        with pytest.raises(AnalysisError):
            bayes_contrast(result=fitted, weights=weights)


@pytest.mark.parametrize('target', ['posterior', 'contrast', 'draws'])
def test_direct_summary_refuses_rehashed_typed_model_copy(target):
    posterior = fit()
    if target == 'posterior':
        original = posterior
        value = tuple(tuple(4*v for v in row) for row in original.coefficient_covariance)
        changes = {'coefficient_covariance': value}
    elif target == 'contrast':
        original = bayes_contrast(result=posterior, weights={'x': 1.})
        changes = {'student_t_scale': 2*original.student_t_scale}
    else:
        original = bayes_draws(result=posterior, draws=5, seed=625)
        value = list(original.beta)
        value[0] = tuple(v+1 for v in value[0])
        changes = {'beta': tuple(value)}
    forged = original.model_copy(update=changes)
    resigned = rehash(forged.model_dump(mode='json'))
    forged = forged.model_copy(update={'integrity_sha256': resigned['integrity_sha256']})
    with pytest.raises((ValueError, ValidationError)):
        forged.summary()


# The following cases validate transport only; they never execute a model fit.
_LEGACY_INDEX_TRANSPORT_PACKET = Path(__file__).parent / "fixtures/index_sortorder_legacy_native_bayes.json"

@pytest.fixture
def forbid_index_transport_science(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('This check must not fit, evaluate _index_posterior algebra, use RNG, sort or reindex.')
    for target, names in [
        (_index_core, ('posterior_algebra', 'matrices')),
        (_index_posterior, ('posterior_algebra', 'matrices')),
        (_index_common.torch, ('rand', 'randn', 'randperm', 'manual_seed', 'multinomial', 'normal', 'bernoulli', 'poisson')),
        (pd.DataFrame, ('sort_index', 'sort_values', 'reindex')),
        (pd.Series, ('sort_index', 'sort_values', 'reindex')),
        (pd.MultiIndex, ('sort_values', 'sortlevel', 'reindex')),
    ]:
        for name in names:
            if hasattr(target, name):
                monkeypatch.setattr(target, name, forbidden)


def _transport_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _transport_indices():
    return [
        pd.RangeIndex(2, 12, 2, name=('range', -0.0)),
        pd.Index(['duplicate', 'duplicate', 'other'], name=None),
        pd.CategoricalIndex(['b', 'a', None, 'a'], categories=['b', 'a', 'unused'], ordered=True, name='category'),
        pd.date_range('2023-01-01', periods=3, tz='Europe/Istanbul', name='date'),
        pd.timedelta_range('1 day', periods=3, name=('time', 2)),
        pd.MultiIndex(levels=[pd.Index(['a', 'b', 'unused']), pd.Index([10, 20, 30])],
                      codes=[[1, 0, 1, -1], [0, 1, 0, 2]], names=[('name', 2), None]),
    ]


@pytest.mark.parametrize('index', _transport_indices(), ids=['range', 'index', 'categorical', 'datetime', 'timedelta', 'multi'])
@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_no_declaration_preserves_all_historical_descriptor_and_none_fields(index):
    encoded = _index_common._encode_index(index)
    assert 'sortorder' not in encoded
    typed = _index_posterior.IndexState.model_validate(encoded)
    raw = typed.model_dump(mode='json')
    assert 'sortorder' not in raw
    # Preserve the entire historical field domain, including all prior None fields.
    historical = {'kind', 'name', 'names', 'levels', 'codes', 'start', 'stop', 'step',
                  'categories', 'ordered', 'dtype', 'values', 'freq'}
    assert set(raw) == historical
    for key in historical - set(encoded):
        assert raw[key] is None
    restored = _index_common._decode_index(typed.descriptor())
    pd.testing.assert_index_equal(restored, index, exact=True)
    assert getattr(restored, 'sortorder', None) == getattr(index, 'sortorder', None)
    assert typed.descriptor() == encoded
    assert _index_posterior.IndexState.model_validate_json(typed.model_dump_json()).model_dump(mode='json') == raw


@pytest.mark.parametrize('order', [0, 1, 2])
@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_declared_order_keeps_exact_unused_levels_duplicate_rows_and_typed_names(order):
    index = pd.MultiIndex(levels=[['a', 'b', 'unused'], pd.CategoricalIndex([10, 20, 30], ordered=True)],
                         codes=[[0, 0, 1, 1], [0, 0, 1, 2]], names=[('axis', 3), None], sortorder=order)
    encoded = _index_common._encode_index(index)
    assert encoded['sortorder'] == order
    _index_common._index_envelope(encoded, len(index))
    typed = _index_posterior.IndexState.model_validate(encoded)
    assert typed.sortorder == order
    portable = json.loads(typed.model_dump_json())
    assert portable['sortorder'] == order
    decoded = _index_common._decode_index(typed.descriptor())
    assert decoded.sortorder == order
    assert [code.tolist() for code in decoded.codes] == [code.tolist() for code in index.codes]
    pd.testing.assert_index_equal(decoded, index, exact=True)
    assert typed.descriptor() == encoded
    assert _index_posterior.IndexState.model_validate_json(json.dumps(portable)).descriptor() == encoded


@pytest.mark.parametrize('bad', [True, False, -1, 3, 1.0, '1', [1], {}], ids=['true', 'false', 'negative', 'too-large', 'float', 'string', 'list', 'dict'])
@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_bad_declared_order_is_refused_by_envelope_and_decoder_before_constructor(bad, monkeypatch):
    encoded = _index_common._encode_index(pd.MultiIndex.from_product([['a', 'b'], [0, 1]]))
    encoded['sortorder'] = bad
    with pytest.raises(ValueError, match='sortorder'):
        _index_common._index_envelope(encoded, 4)
    def forbidden(*args, **kwargs):
        raise AssertionError('Malformed declaration reached the pandas constructor.')
    monkeypatch.setattr(_index_common.pd, 'MultiIndex', forbidden)
    with pytest.raises(ValueError, match='sortorder'):
        _index_common._decode_index(encoded)


@pytest.mark.parametrize('bad', [True, False, -1, 3, 1.0, '1'], ids=['true', 'false', 'negative', 'too-large', 'float', 'string'])
@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_bad_declared_order_is_refused_by_typed_index_state(bad):
    encoded = _index_common._encode_index(pd.MultiIndex.from_product([['a', 'b'], [0, 1]]))
    encoded['sortorder'] = bad
    with pytest.raises(ValidationError):
        _index_posterior.IndexState.model_validate(encoded)


@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_order_inconsistent_with_physical_codes_is_refused_without_sorting():
    encoded = _index_common._encode_index(pd.MultiIndex.from_tuples([('b', 0), ('a', 1), ('b', 0)]))
    encoded['sortorder'] = 2
    before = deepcopy(encoded)
    with pytest.raises(ValueError, match='sortorder'):
        _index_common._decode_index(encoded)
    assert encoded == before


@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_other_index_kind_cannot_carry_nonempty_sortorder():
    encoded = _index_common._encode_index(pd.RangeIndex(3)) | {'sortorder': 0}
    with pytest.raises(ValidationError, match='sortorder'):
        _index_posterior.IndexState.model_validate(encoded)


@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_explicit_none_is_normalized_only_for_the_new_field():
    encoded = _index_common._encode_index(pd.MultiIndex.from_product([['a'], [0, 1]]))
    old = _index_posterior.IndexState.model_validate(encoded).model_dump(mode='json')
    explicit = _index_posterior.IndexState.model_validate(encoded | {'sortorder': None})
    assert explicit.model_dump(mode='json') == old
    assert _transport_digest(explicit.model_dump(mode='json')) == _transport_digest(old)
    assert all(key in old for key in ('name', 'categories', 'ordered', 'start', 'stop', 'step', 'freq', 'values'))
    assert explicit.model_dump(mode='json', include={'kind', 'sortorder', 'name'}) == {'kind': 'multi', 'name': None}
    assert explicit.model_dump(mode='json', exclude={'sortorder'}) == old


def _transport_construct_wire(annotation, value):
    """Serialize existing complete cache values; deliberately do not run statistical validators."""
    origin, args = get_origin(annotation), get_args(annotation)
    if value is None:
        return None
    if origin in (Union, types.UnionType):
        for candidate in args:
            if candidate is type(None):
                continue
            if isinstance(candidate, type) and issubclass(candidate, BaseModel) and isinstance(value, dict):
                return _transport_construct_wire(candidate, value)
            if get_origin(candidate) is tuple and isinstance(value, (list, tuple)):
                return _transport_construct_wire(candidate, value)
        return value
    if origin is tuple:
        return tuple(_transport_construct_wire(args[0] if len(args) == 2 and args[1] is Ellipsis else args[i], item)
                     for i, item in enumerate(value))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation is _index_posterior.IndexState:
            return annotation.model_validate(value)
        values = {key: _transport_construct_wire(annotation.model_fields[key].annotation, item) for key, item in value.items()}
        return annotation.model_construct(**values)
    return value


@pytest.mark.parametrize('name,model', [
    ('posterior', _index_posterior.PosteriorBundle), ('draws', _index_posterior.PosteriorDraws), ('contrast', _index_posterior.PosteriorContrast)],
    ids=['complete-native-posterior', 'complete-native-predictive-draws', 'complete-native-contrast'])
@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_actual_complete_old_native_packet_serialization_keeps_integrity_digest(name, model):
    packet = json.loads(_LEGACY_INDEX_TRANSPORT_PACKET.read_text())[name]
    original = deepcopy(packet)
    wire = _transport_construct_wire(model, packet)
    portable = wire.model_dump(mode='json')
    assert portable == original
    assert json.loads(wire.model_dump_json()) == original
    checksum = portable.pop('integrity_sha256')
    assert _transport_digest(portable) == checksum == original['integrity_sha256']
    assert packet == original


@pytest.mark.usefixtures("forbid_index_transport_science")
def test_index_transport_nested_index_state_serializer_preserves_legacy_fields_and_new_nonempty_order():
    class Carrier(BaseModel):
        source_index: _index_posterior.IndexState
        query_index: _index_posterior.IndexState
        legacy_none: str | None = None
    plain = _index_posterior.IndexState.model_validate(_index_common._encode_index(pd.RangeIndex(3)))
    sorted_index = _index_posterior.IndexState.model_validate(_index_common._encode_index(
        pd.MultiIndex.from_product([['a'], [0, 1, 2]], sortorder=2)))
    carrier = Carrier(source_index=plain, query_index=sorted_index)
    raw = carrier.model_dump(mode='json')
    assert raw['legacy_none'] is None
    assert 'sortorder' not in raw['source_index']
    assert raw['query_index']['sortorder'] == 2
    assert raw['source_index']['freq'] is None and raw['query_index']['freq'] is None
    assert Carrier.model_validate_json(carrier.model_dump_json()).model_dump(mode='json') == raw


@pytest.mark.parametrize('name,model', [
    ('posterior', PosteriorBundle), ('contrast', PosteriorContrast)],
    ids=['posterior', 'contrast'])
def test_index_transport_actual_cached_semantic_restore_without_fit_or_rng(name, model, monkeypatch):
    """Replay two existing small complete states; predictive draw replay requires RNG and stays separate."""
    def forbidden(*args, **kwargs):
        raise AssertionError('Cached semantic restoration cannot fit or draw replacement samples.')
    for target, names in [
        (commands, ('bayes_linear', 'bayes_draws')),
        (_index_posterior, ('_draw_arrays',)),
        (torch, ('rand', 'randn', 'randperm', 'manual_seed', 'multinomial', 'normal', 'bernoulli', 'poisson')),
        (pd.DataFrame, ('sort_index', 'sort_values', 'reindex')),
        (pd.Series, ('sort_index', 'sort_values', 'reindex')),
        (pd.MultiIndex, ('sort_values', 'sortlevel', 'reindex')),
    ]:
        for method in names:
            if hasattr(target, method):
                monkeypatch.setattr(target, method, forbidden)
    packet = json.loads(_LEGACY_INDEX_TRANSPORT_PACKET.read_text())[name]
    original = deepcopy(packet)
    restored = model.model_validate_json(json.dumps(packet))
    assert restored.model_dump(mode='json') == original
    body = restored.model_dump(mode='json', exclude={'integrity_sha256'})
    assert _transport_digest(body) == original['integrity_sha256']
    assert packet == original
