"""Full semantic and before-allocation/source/replay lifecycle boundaries."""

import copy
import json
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian import (
    bma as b,
    bma_common as c,
    bma_kernels as k,
    bma_query as q,
    bma_draws as d,
)
from openecon.econometrics.mi.common import _thaw
from openecon.resources import use_workspace_budget
from test_bayesian_bma_oracles import fit, inputs


@pytest.fixture(scope="module")
def model():
    return fit()


@pytest.fixture(scope="module")
def query(model):
    return q.bayes_bma_predict(
        model,
        data=pd.DataFrame(
            {"x1": [-0.5, 0.8], "x2": [0.2, -0.3]}, index=pd.Index(["a", "b"], name="query")
        ),
    )


@pytest.fixture(scope="module")
def draws(model):
    return d.bayes_bma_draws(
        model, draws=16, seed=190690, data=pd.DataFrame({"x1": [-0.5, 0.8], "x2": [0.2, -0.3]})
    )


def rehash(raw):
    raw["digest"] = c.digest({key: value for key, value in raw.items() if key != "digest"})
    return raw


def forbidden(*a, **kw):
    raise AssertionError("expensive parent/core reached before refusal")


@pytest.mark.parametrize("target", ["model", "query", "draws"])
def test_full_roundtrip_copy_and_excluded_valid_exports(request, target):
    state = request.getfixturevalue(target)
    restore = {
        "model": b.bayes_bma_restore,
        "query": q.bayes_bma_prediction_restore,
        "draws": d.bayes_bma_draws_restore,
    }[target]
    assert restore(state.model_dump_json()).payload == state.payload
    assert type(state).model_validate_json(state.model_dump_json()).payload == state.payload
    assert copy.deepcopy(state).payload == state.payload
    assert state.model_copy(deep=True).payload == state.payload
    assert state.model_dump(exclude={"payload"}) == {"schema_version": state.schema_version}
    assert json.loads(state.model_dump_json(exclude={"payload"})) == {
        "schema_version": state.schema_version
    }


@pytest.mark.parametrize(
    "field",
    [
        "model_weights",
        "coefficient_covariance",
        "joint_covariance",
        "coefficient_quantiles",
        "inclusion_covariance",
    ],
)
def test_cache_shape_and_boolean_primitives_before_factorization(model, monkeypatch, field):
    raw = _thaw(model.payload)
    raw["results"][field] = [True]
    rehash(raw)
    monkeypatch.setattr(k, "posterior_algebra", forbidden)
    with pytest.raises(AnalysisError):
        b.bayes_bma_restore(raw)


@pytest.mark.parametrize("field", ["location", "scale_matrix", "covariance", "indices"])
def test_component_shape_before_factorization(model, monkeypatch, field):
    raw = _thaw(model.payload)
    raw["components"][0][field] = [None]
    rehash(raw)
    monkeypatch.setattr(k, "posterior_algebra", forbidden)
    with pytest.raises(AnalysisError):
        b.bayes_bma_restore(raw)


@pytest.mark.parametrize(
    "field",
    [
        "posterior_inclusion",
        "model_weights",
        "coefficient_quantiles",
        "coefficient_covariance",
        "joint_covariance",
    ],
)
def test_rehashed_numeric_forgery_and_every_typed_export_refuse(model, field):
    raw = _thaw(model.payload)
    value = raw["results"][field]
    if isinstance(value[0], list):
        value[0][0] *= 4
    else:
        value[0] += 0.05
    rehash(raw)
    forged = model.model_copy(update={"payload": raw})
    for read in [
        lambda: b.bayes_bma_restore(forged),
        forged.summary,
        forged.model_summary,
        forged.model_dump,
        lambda: forged.model_dump(exclude={"payload"}),
        forged.model_dump_json,
        lambda: forged.model_dump_json(include={"schema_version"}),
    ]:
        with pytest.raises((AnalysisError, ValidationError)):
            read()


def test_structural_zero_subnormal_rehash_refuses(model):
    raw = _thaw(model.payload)
    raw["components"][0]["scale_matrix"][1][1] = 1e-320
    rehash(raw)
    with pytest.raises(AnalysisError):
        b.bayes_bma_restore(raw)


@pytest.mark.parametrize("target", ["model", "query", "draws"])
def test_dense_parser_copy_and_indent_refuse_before_allocation(request, target, monkeypatch):
    state = request.getfixturevalue(target)
    cls = type(state)
    dense = [{} for _ in range(50000)]
    text = json.dumps({"schema_version": state.schema_version, "payload": dense})
    with use_workspace_budget(1):
        tracemalloc.start()
        try:
            with pytest.raises(AnalysisError):
                cls.model_validate_json(text)
            assert tracemalloc.get_traced_memory()[1] < 32768
        finally:
            tracemalloc.stop()
        forged = cls.model_construct(schema_version=state.schema_version, payload=dense)
        tracemalloc.start()
        try:
            with pytest.raises(AnalysisError):
                forged.model_copy(deep=True)
            assert tracemalloc.get_traced_memory()[1] < 32768
        finally:
            tracemalloc.stop()
    monkeypatch.setattr(cls, "_replay", classmethod(forbidden))
    with pytest.raises(AnalysisError):
        state.model_dump_json(indent=10**9)
    with pytest.raises(AnalysisError):
        state.model_dump_json(indent=True)


@pytest.mark.parametrize("target", ["model", "query", "draws"])
def test_bad_outer_literal_all_public_readers_refuse(request, target):
    state = request.getfixturevalue(target)
    forged = state.model_copy(update={"schema_version": "wrong"})
    for call in [
        forged.summary,
        forged.model_dump,
        forged.model_dump_json,
        lambda: type(state).model_validate({"schema_version": "wrong", "payload": state.payload}),
    ]:
        with pytest.raises((AnalysisError, ValidationError)):
            call()


@pytest.mark.parametrize("bad", [True, 1 + 2j, 2**53 + 1, float("inf")])
def test_public_query_cells_refuse_before_parent(model, monkeypatch, bad):
    monkeypatch.setattr(b, "replay", forbidden)
    with pytest.raises(AnalysisError):
        q.bayes_bma_predict(model, data=pd.DataFrame({"x1": [bad], "x2": [0.2]}))


@pytest.mark.parametrize("bad", [True, -1, 1.5])
def test_draw_seed_before_parent(model, monkeypatch, bad):
    monkeypatch.setattr(b, "replay", forbidden)
    with pytest.raises(AnalysisError):
        d.bayes_bma_draws(model, seed=bad)


@pytest.mark.parametrize("change", ["float32", "missing", "positions", "index", "cache"])
def test_saved_query_complete_admission_before_parent(query, monkeypatch, change):
    raw = _thaw(query.payload)
    if change == "float32":
        raw["source"]["dtypes"][0] = "float32"
        raw["source"]["values"][0][0] = 1 / 3
    elif change == "missing":
        raw["source"]["missing"] = "silentlydrop"
    elif change == "positions":
        raw["source"]["positions"] = [True, 1]
    elif change == "index":
        raw["source"]["index"]["dtype"] = "invalid"
    else:
        raw["results"]["mean_covariance"] = [[]]
    raw["source"]["digest"] = c.digest(
        {key: value for key, value in raw["source"].items() if key != "digest"}
    )
    rehash(raw)
    monkeypatch.setattr(b, "replay", forbidden)
    with pytest.raises(AnalysisError):
        q.bayes_bma_prediction_restore(raw)


def test_common_union_missing_sample_and_original_query_alignment():
    data, prior, odds = inputs()
    data.index = pd.Index(["a", "a", *map(str, range(10))], name="id")
    data.loc[data.index == "1", "x2"] = np.nan
    result = b.bayes_bma(
        data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds, missing="drop"
    )
    positions = result.payload["source"]["positions"]
    assert len(positions) == 11
    for component in result.payload["components"]:
        assert component["posterior"]["shape"] == pytest.approx(
            prior[result.payload["components"].index(component)].shape + 11 / 2
        )
    query = pd.DataFrame(
        {"x1": [0.1, np.nan, 0.2], "x2": [0.2, 0.1, 0.3]},
        index=pd.Index(["dup", "drop", "dup"], name="id"),
    )
    predicted = q.bayes_bma_predict(result, data=query, missing="drop")
    frame = predicted.summary()
    pd.testing.assert_index_equal(frame.index, query.index)
    assert pd.isna(frame.iloc[1]["posterior_mean"])
    assert predicted.payload["results"]["positions"] == (0, 2)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(3, 27, 2, name="row"),
        pd.date_range("2020-01-01", periods=12, tz="Europe/Istanbul", name="when"),
        pd.MultiIndex.from_product([["a", "b"], range(6)], names=["group", "row"], sortorder=2),
        pd.Index(
            pd.array(["a", None, *map(str, range(10))], dtype=pd.StringDtype(storage="python")),
            name="str",
        ),
    ],
)
def test_lossless_original_index_roundtrip(index):
    data, prior, odds = inputs()
    data.index = index
    r = b.bayes_bma(data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds)
    restored = b.bayes_bma_restore(r.model_dump_json())
    actual = c.decode_index(restored.payload["source"]["index"], 12)
    pd.testing.assert_index_equal(actual, index, exact=True)
    assert getattr(actual, "sortorder", None) == getattr(index, "sortorder", None)


def test_before_source_scan_and_metadata_work_budget(monkeypatch):
    data, prior, odds = inputs()
    monkeypatch.setattr(c, "source_values", forbidden)
    with pytest.raises(AnalysisError):
        b.bayes_bma(
            data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds, max_work=1
        )
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            b.bayes_bma(
                data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=[{}] * 50000
            )


def test_ambient_device_dtype_rng_and_no_source_estimator_restore(model, query, draws, monkeypatch):
    monkeypatch.setattr(b, "bayes_bma", forbidden)
    original = torch.get_default_dtype()
    random = torch.random.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            b.bayes_bma_restore(model.model_dump_json())
            q.bayes_bma_prediction_restore(query.model_dump_json())
            d.bayes_bma_draws_restore(draws.model_dump_json())
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(random, torch.random.get_rng_state())
    finally:
        torch.set_default_dtype(original)


def test_unused_large_index_levels_admitted_before_descriptor_copy(monkeypatch):
    data, prior, odds = inputs()
    data.index = pd.MultiIndex(
        levels=[pd.Index(np.arange(60000)), pd.Index(["a", "b"])],
        codes=[list(range(len(data))), [0] * len(data)],
        names=["unused_large_level", "group"],
    )
    monkeypatch.setattr(c, "encode_index", forbidden)
    with use_workspace_budget(16), pytest.raises(AnalysisError):
        b.bayes_bma(data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds)


def _expanded_name(depth):
    name = "name"
    for _ in range(depth):
        name = (name, name)
    return name


def _raw_expanded_index(kind, n):
    if kind == "name":
        return pd.RangeIndex(n, name=_expanded_name(14))
    labels = pd.Index([(j, j + 1, j + 2, j + 3) for j in range(4096)], tupleize_cols=False)
    if kind == "categories":
        return pd.CategoricalIndex(pd.Categorical.from_codes(range(n), categories=labels))
    return pd.MultiIndex(levels=[labels, pd.Index(["a"])], codes=[list(range(n)), [0] * n])


@pytest.mark.parametrize("kind", ["name", "categories", "levels"])
@pytest.mark.parametrize("reader", ["fit", "prediction", "draws"])
def test_raw_expanded_index_admission_before_source_or_parent(model, monkeypatch, kind, reader):
    data, prior, odds = inputs()
    data.index = _raw_expanded_index(kind, len(data))
    monkeypatch.setattr(c, "encode_index", forbidden)
    monkeypatch.setattr(c, "source_values", forbidden)
    monkeypatch.setattr(b, "replay", forbidden)
    with use_workspace_budget(1):
        tracemalloc.start()
        try:
            with pytest.raises(AnalysisError) as error:
                if reader == "fit":
                    b.bayes_bma(
                        data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds
                    )
                elif reader == "prediction":
                    q.bayes_bma_predict(model, data=data)
                else:
                    d.bayes_bma_draws(model, data=data, draws=4, seed=690)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert error.value.code == "workspace_limit"
    assert peak < 65536


def test_raw_index_respects_declared_budget_before_descriptor_or_source(monkeypatch):
    data, prior, odds = inputs()
    data.index = pd.RangeIndex(len(data), name=_expanded_name(8))
    monkeypatch.setattr(c, "encode_index", forbidden)
    monkeypatch.setattr(c, "source_values", forbidden)
    with pytest.raises(AnalysisError, match="raw row-index"):
        b.bayes_bma(
            data=data,
            y="y",
            optional=["x1", "x2"],
            priors=prior,
            model_prior_odds=odds,
            max_bytes=65536,
        )


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(10000, name=((True, 2**128), ("😀", None))),
        pd.Index([("a", (1, 2)), ("b", (3, 4))], tupleize_cols=False, name=("id", "😀")),
        pd.date_range("2020-01-01", periods=3, tz="Europe/Istanbul", name="when"),
        pd.timedelta_range("1 day", periods=3, name="duration"),
        pd.MultiIndex(
            levels=[["a", "unused"], [1, 2]], codes=[[0, 0], [0, 1]], names=[("g", 1), "n"]
        ),
        pd.CategoricalIndex(
            pd.Categorical.from_codes([0, -1], categories=["x", "unused"], ordered=True)
        ),
        pd.Index(pd.array(["a", None], dtype=pd.StringDtype(storage="python")), name="str"),
    ],
)
def test_virtual_index_geometry_matches_complete_lossless_codec(index):
    from openecon.econometrics.state_lifecycle import _metadata_geometry

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    lazy = c._lazy_index(index)
    eager = c.encode_index(index)
    assert expand(lazy) == eager
    assert _metadata_geometry(lazy) == _metadata_geometry(eager)
    assert c.index_buffer_bytes(index) == _metadata_geometry(eager)[0] + 8 * int(
        index.memory_usage(deep=True)
    )


def test_raw_index_depth_refuses_before_recursive_codec(monkeypatch):
    data, prior, odds = inputs()
    name = "n"
    for _ in range(40):
        name = (name,)
    data.index = pd.RangeIndex(len(data), name=name)
    monkeypatch.setattr(c, "encode_index", forbidden)
    with pytest.raises(AnalysisError, match="nesting"):
        b.bayes_bma(data=data, y="y", optional=["x1", "x2"], priors=prior, model_prior_odds=odds)


def test_high_shape_odd_even_evidence_against_100_digit_observation_law():
    import mpmath as mp
    from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior

    for n in (1, 2, 5, 6):
        data = pd.DataFrame({"y": np.linspace(0.2, 0.8, n), "x": np.linspace(-0.5, 0.5, n)})
        priors = [
            NormalInverseGammaPrior(mean=[0.1], scale_matrix=[[0.7]], shape=1e12, scale=1e12),
            NormalInverseGammaPrior(
                mean=[0.1, 0.2],
                scale_matrix=[[0.7, 0.1], [0.1, 0.8]],
                shape=1e12 - 0.25,
                scale=1e12,
            ),
        ]
        result = b.bayes_bma(
            data=data, y="y", optional=["x"], priors=priors, model_prior_odds=[0.3, 0.7]
        )
        expected = []
        with mp.workdps(100):
            for p, X in zip(priors, [np.ones((n, 1)), np.c_[np.ones(n), data.x]]):
                XX = mp.matrix([[mp.mpf(float(v)) for v in row] for row in X])
                V = mp.matrix(p.scale_matrix)
                mean = mp.matrix(p.mean)
                yy = mp.matrix(data.y.tolist())
                A = mp.eye(n) + XX * V * XX.T
                rr = yy - XX * mean
                update = (rr.T * A**-1 * rr)[0] / 2
                aa = mp.mpf(p.shape)
                bb = mp.mpf(p.scale)
                an = aa + mp.mpf(n) / 2
                expected.append(
                    float(
                        -mp.mpf(n) / 2 * mp.log(2 * mp.pi)
                        - mp.log(mp.det(A)) / 2
                        + mp.loggamma(an)
                        - mp.loggamma(aa)
                        - mp.mpf(n) / 2 * mp.log(bb)
                        - an * mp.log1p(update / bb)
                    )
                )
        np.testing.assert_allclose(
            result.payload["results"]["log_model_evidence"], expected, rtol=0, atol=3e-12
        )


def test_complete_256_model_capacity_and_prospective_refusals(monkeypatch):
    from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior

    data = pd.DataFrame({f"x{j}": np.sin(np.arange(12) * (0.2 + 0.1 * j)) for j in range(8)})
    data["y"] = 0.2 + 0.3 * data.x0 - 0.2 * data.x3
    priors = []
    for mask in range(256):
        dim = 1 + mask.bit_count()
        priors.append(
            NormalInverseGammaPrior(
                mean=[0.0] * dim, scale_matrix=np.eye(dim).tolist(), shape=3.0, scale=1.0
            )
        )
    model = b.bayes_bma(
        data=data,
        y="y",
        optional=[f"x{j}" for j in range(8)],
        priors=priors,
        model_prior_odds=[1.0] * 256,
    )
    assert len(model.payload["components"]) == 256
    assert all(value > 0 for value in model.payload["results"]["model_weights"])
    assert abs(sum(model.payload["results"]["model_weights"]) - 1) < 5e-15
    assert len(model.payload["results"]["joint_covariance"]) == 10
    assert b.bayes_bma_restore(model.model_dump_json()).payload == model.payload
    monkeypatch.setattr(k, "posterior_algebra", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        b.bayes_bma(
            data=data,
            y="y",
            optional=[f"x{j}" for j in range(8)],
            priors=priors,
            model_prior_odds=[1.0] * 256,
        )
    with pytest.raises(AnalysisError):
        b.bayes_bma(
            data=data, y="y", optional=[f"x{j}" for j in range(9)], priors=[], model_prior_odds=[]
        )
    with pytest.raises(AnalysisError):
        q.bayes_bma_predict(model, data=pd.DataFrame({f"x{j}": [0.0] * 1200 for j in range(8)}))


def test_full_unavailable_beta_sigma_crossmoment_contract():
    from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior

    data = pd.DataFrame({"y": [0.0, 0.0], "x": [-0.5, 0.5]})
    prior = [
        NormalInverseGammaPrior(
            mean=[0.0] * d, scale_matrix=np.eye(d).tolist(), shape=0.25, scale=1.0
        )
        for d in (1, 2)
    ]
    model = b.bayes_bma(data=data, y="y", optional=["x"], priors=prior, model_prior_odds=[1.0, 1.0])
    result = q.bayes_bma_predict(model, data=pd.DataFrame({"x": [0.3]})).payload["results"]
    assert model.payload["results"]["coefficient_covariance"] is not None
    assert model.payload["results"]["joint_covariance"] is None
    assert result["mean_covariance"] is not None
    assert result["outcome_covariance"] is not None
    assert result["parameter_query_covariance"] is None


@pytest.mark.parametrize(
    "dtype,cell",
    [
        ("int64", 2**53 + 1),
        ("int8", 256),
        ("uint8", -1),
        ("float32", 0.123456789),
        ("float16", 0.123456789),
        ("int64", 1.5),
        ("int64", None),
    ],
)
def test_saved_dtype_precision_range_before_any_series(model, monkeypatch, dtype, cell):
    raw = _thaw(model.payload)
    raw["source"]["dtypes"][0] = dtype
    raw["source"]["values"][0][0] = cell
    raw["source"]["digest"] = c.digest(
        {key: value for key, value in raw["source"].items() if key != "digest"}
    )
    rehash(raw)
    monkeypatch.setattr(c.pd, "Series", forbidden)
    with pytest.raises(AnalysisError):
        b.bayes_bma_restore(raw)


@pytest.mark.parametrize(
    "target,field", [("query", "parameter_query_covariance"), ("draws", "beta")]
)
def test_rehashed_query_and_seeded_draw_uncertainty_all_exports_refuse(request, target, field):
    state = request.getfixturevalue(target)
    raw = _thaw(state.payload)
    raw["results"][field][0][0] += 0.02
    rehash(raw)
    forged = state.model_copy(update={"payload": raw})
    restore = q.bayes_bma_prediction_restore if target == "query" else d.bayes_bma_draws_restore
    for read in [
        lambda: restore(forged),
        forged.summary,
        forged.model_dump,
        lambda: forged.model_dump(exclude={"payload"}),
        forged.model_dump_json,
        lambda: forged.model_dump_json(include={"schema_version"}),
    ]:
        with pytest.raises((AnalysisError, ValidationError)):
            read()


@pytest.mark.parametrize(
    "index",
    [
        pytest.param({"kind": "unsupported"}, id="unsupported-kind"),
        pytest.param(
            {"kind": "range", "start": 0, "stop": 2, "step": 0}, id="zero-range-step"
        ),
        pytest.param(
            {"kind": "range", "start": 0, "stop": 10**100, "step": 1},
            id="overflowing-range-geometry",
        ),
        pytest.param(
            {"kind": "multi", "levels": None, "codes": [], "names": []},
            id="nonarray-multi-levels",
        ),
        pytest.param(
            {
                "kind": "multi",
                "levels": [{"kind": "range", "start": 0, "stop": 2, "step": 1}],
                "codes": [[0]],
                "names": [{"type": "none"}],
                "sortorder": None,
            },
            id="multi-code-row-count",
        ),
        pytest.param(
            {"kind": "categorical", "codes": None, "categories": None},
            id="nonarray-categorical-codes",
        ),
        pytest.param(None, id="nonmapping-index"),
    ],
)
def test_rehashed_saved_source_index_geometry_refuses_before_series(index, monkeypatch):
    source = {
        "n": 2,
        "columns": ["x"],
        "dtypes": ["float64"],
        "values": [[0.25, -0.5]],
        "index": index,
        "positions": [0, 1],
        "missing": "raise",
    }
    source["digest"] = c.digest(source)
    with monkeypatch.context() as isolated:
        isolated.setattr(c.pd, "Series", forbidden)
        isolated.setattr(c, "decode_index", forbidden)
        with pytest.raises(AnalysisError) as error:
            c.source_admission(source, ["x"], "raise")
        assert error.value.code == "invalid_bma_state"


@pytest.mark.parametrize(
    "optional_count,rows,max_bytes,error_code",
    [
        (0, 750, 512 * 1024**2, "dimension_limit"),
        (8, 100, 512 * 1024**2, "dimension_limit"),
        (0, 300, 64 * 1024**2, "workspace_limit"),
    ],
)
def test_complete_query_geometry_refuses_before_source_or_parent(
    monkeypatch, optional_count, rows, max_bytes, error_code
):
    optional = [f"x{i}" for i in range(optional_count)]
    spec = c.specification(
        y="y", forced=[], optional=optional, missing="raise", alpha=0.05,
        max_work=10**12, max_bytes=max_bytes,
    )
    # This complete lightweight header is enough to admit the public query's
    # dimensions. The incomplete parent body must never reach semantic replay.
    parent = {"schema": b.SCHEMA, "spec": spec, "source": {"n": 12}}
    data = pd.DataFrame({name: [0.0] * rows for name in optional}, index=pd.RangeIndex(rows))
    with monkeypatch.context() as patch:
        patch.setattr(c, "capture", forbidden)
        patch.setattr(b, "replay", forbidden)
        patch.setattr(q, "derive", forbidden)
        with use_workspace_budget(512), pytest.raises(AnalysisError) as exc:
            q.bayes_bma_predict(parent, data=data)
    assert exc.value.code == error_code
