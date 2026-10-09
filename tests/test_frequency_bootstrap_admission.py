"""Independent count-law and adversarial admission audit for frequency bootstrap.

These checks compare compressed counts with literal IID copy ranks and expanded
sample moments. They do not duplicate the spectral adapter implementations.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import frequency_bootstrap as fb
from openecon.econometrics.multivariate.pca_frequency_uncertainty import (
    pca_subspace_fweight_bootstrap,
)
from openecon.econometrics.postest.index_codec import encode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.frame import DataFrame


def _prepare(data, names=None, **overrides):
    settings = dict(weights="w", weight_type="fweight", replications=39,
                    confidence=.95, seed=317, missing="drop", parameters=6,
                    fit_work=1, minimum_units=2, label="independent frequency audit")
    settings.update(overrides)
    return fb.prepare(data, ["x", "y"] if names is None else names, **settings)


def _domain(seed, process):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(47, 2))
    if process == "asymmetric":
        raw = np.column_stack((np.exp(.4*raw[:, 0]),
                               .7*raw[:, 0] + rng.standard_t(8, size=len(raw))))
    else:
        raw = raw @ np.array([[2.0, .3], [.2, .6]])
    counts = np.asarray([0 if i % 11 == 0 else i % 4 + 1 for i in range(len(raw))])
    frame = pd.DataFrame(raw, columns=["x", "y"])
    frame["w"] = counts
    return frame, raw, counts


def _expanded_moments(raw, counts):
    expanded = np.repeat(raw, counts, axis=0)
    origin = expanded[0].copy()
    anchored = expanded-origin
    offset = anchored.mean(axis=0)
    offset += (anchored-offset).mean(axis=0)
    deviation = anchored-offset
    covariance = deviation.T @ deviation/(len(expanded)-1)
    return origin+offset, covariance


def test_count_law_exact_exhaustive_copy_rank_distribution(monkeypatch):
    # Three literal observations compressed to two support points; the third
    # physical row is an explicit structural zero. Enumerate all 3**3 IID draws.
    sample = _prepare(pd.DataFrame({"x": [0., 1., 1e308],
                                   "y": [1., 0., -1e308], "w": [1, 2, 0]}))
    literal_copy_category = np.repeat(np.arange(3), [1, 2, 0])
    distribution = Counter()
    for ranks in itertools.product(range(3), repeat=3):
        def supplied_ranks(*args, **kwargs):
            return torch.tensor(ranks, dtype=torch.int64)

        monkeypatch.setattr(fb.torch, "randint", supplied_ranks)
        actual = fb.draw_counts(sample, torch.Generator().manual_seed(317))
        expected = np.bincount(literal_copy_category[list(ranks)], minlength=3)
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert actual.dtype == torch.int64 and int(actual.sum()) == 3
        distribution[tuple(actual.tolist())] += 1
    # Independent binomial count probabilities after aggregation of identical
    # empirical atoms; this catches zero-margin and cumulative-boundary errors.
    assert distribution == Counter({(k, 3-k, 0): math.comb(3, k)*2**(3-k)
                                    for k in range(4)})


@pytest.mark.parametrize("seed,process", [(1307, "elliptical"), (1703, "asymmetric")])
def test_compressed_point_and_seeded_draw_moments_equal_literal_expansion(seed, process):
    frame, raw, counts = _domain(seed, process)
    before = frame.copy(deep=True)
    sample = _prepare(frame)
    generator = torch.Generator().manual_seed(317)
    for frequency in [sample.counts, *(fb.draw_counts(sample, generator) for _ in range(19))]:
        fitted = fb.moments(sample.values, frequency)
        expected_mean, expected_covariance = _expanded_moments(raw, frequency.numpy())
        np.testing.assert_allclose(fitted.mean.numpy(), expected_mean, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(fitted.covariance.numpy(), expected_covariance,
                                   rtol=3e-14, atol=3e-14)
        assert int(frequency.sum()) == int(counts.sum())
        assert not bool((frequency[counts == 0] != 0).any())
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("bad", [True, np.bool_(True), -.5, -1, .5, np.inf,
                                2+0j, np.complex128(2), "2", Decimal("2.5"),
                                2**53+1, Decimal("1E+100000")])
def test_invalid_frequency_cannot_hide_on_a_measurement_missing_row(bad):
    frame, _, _ = _domain(1307, "elliptical")
    frame["w"] = frame["w"].astype(object)
    frame.loc[3, "x"] = np.nan
    frame.loc[3, "w"] = bad
    with pytest.raises(AnalysisError) as caught:
        _prepare(frame, missing="drop")
    assert caught.value.code == "invalid_weights"


def test_zero_count_huge_values_do_not_enter_anchor_or_arithmetic():
    frame, raw, counts = _domain(1307, "elliptical")
    raw[counts == 0] = np.column_stack((np.full(sum(counts == 0), 1e308),
                                      np.full(sum(counts == 0), -1e308)))
    frame.loc[:, ["x", "y"]] = raw
    sample = _prepare(frame)
    fitted = fb.moments(sample.values, sample.counts)
    mean, covariance = _expanded_moments(raw, counts)
    assert bool(torch.isfinite(fitted.mean).all())
    assert bool(torch.isfinite(fitted.covariance).all())
    np.testing.assert_allclose(fitted.mean.numpy(), mean, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(fitted.covariance.numpy(), covariance, rtol=3e-14, atol=3e-14)


def test_missing_weight_and_measurement_use_one_listwise_sample():
    frame, raw, counts = _domain(1703, "asymmetric")
    frame["w"] = frame["w"].astype(float)
    frame.loc[2, "w"] = np.nan
    frame.loc[6, "y"] = np.nan
    sample = _prepare(frame)
    keep = np.array([i for i in range(len(raw)) if i not in [2, 6]])
    np.testing.assert_array_equal(sample.counts.numpy(), counts[keep])
    fitted = fb.moments(sample.values, sample.counts)
    mean, covariance = _expanded_moments(raw[keep], counts[keep])
    np.testing.assert_allclose(fitted.mean.numpy(), mean, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(fitted.covariance.numpy(), covariance, rtol=3e-14, atol=3e-14)
    with pytest.raises(AnalysisError):
        _prepare(frame, missing="raise")


@pytest.mark.parametrize("scale", [1e-158, 1e-160])
def test_correlation_refuses_subnormal_raw_sample_variance(scale):
    frame, _, _ = _domain(1703, "asymmetric")
    frame.loc[:, ["x", "y"]] *= scale
    sample = _prepare(frame)
    with pytest.raises(AnalysisError):
        fb.correlation(fb.moments(sample.values, sample.counts))


def test_normal_tiny_unit_correlation_is_scale_equivariant():
    frame, _, _ = _domain(1703, "asymmetric")
    baseline = _prepare(frame)
    small = frame.copy(deep=True)
    small.loc[:, ["x", "y"]] *= 1e-100
    scaled = _prepare(small)
    np.testing.assert_allclose(
        fb.correlation(fb.moments(baseline.values, baseline.counts)).numpy(),
        fb.correlation(fb.moments(scaled.values, scaled.counts)).numpy(),
        rtol=4e-14, atol=4e-14)


@pytest.mark.parametrize("domain", ["dtype", "mixed"])
def test_boolean_measurements_are_not_silently_numeric(domain):
    frame, _, _ = _domain(1307, "elliptical")
    if domain == "dtype":
        frame["x"] = frame["x"] > 0
    else:
        frame["x"] = frame["x"].astype(object)
        frame.loc[3, "x"] = True
    with pytest.raises(AnalysisError):
        _prepare(frame)


def test_portable_admission_precedes_selected_source_copy(monkeypatch):
    frame, _, _ = _domain(1307, "elliptical")

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("Portable refusal must precede source coercion.")

    monkeypatch.setattr(fb.c, "source", forbidden_copy)
    with pytest.raises(AnalysisError) as caught:
        _prepare(frame, parameters=fb.MAX_PARAMETERS, replications=fb.MAX_REPLICATIONS)
    assert caught.value.code == "resource_limit"


def test_row_admission_precedes_resident_column_iteration(monkeypatch):
    class SizedOnly:
        def __len__(self):
            return fb.MAX_ROWS+1

        def __iter__(self):
            raise AssertionError("Oversized source must be refused before iteration.")

    data = {name: SizedOnly() for name in ["x", "y", "w"]}
    with pytest.raises(AnalysisError) as caught:
        _prepare(data)
    assert caught.value.code == "workspace_limit"


def test_work_admission_precedes_numeric_tensor_copy(monkeypatch):
    frame, _, _ = _domain(1307, "elliptical")

    def forbidden_tensor(*args, **kwargs):
        raise AssertionError("Work refusal must precede numerical tensor materialization.")

    monkeypatch.setattr(fb.torch, "tensor", forbidden_tensor)
    with pytest.raises(AnalysisError) as caught:
        _prepare(frame, fit_work=fb.MAX_WORK)
    assert caught.value.code == "work_limit"


def test_literal_unit_cap_precedes_numeric_tensor_copy(monkeypatch):
    frame, _, _ = _domain(1307, "elliptical")
    frame.loc[3, "w"] = fb.MAX_UNITS+1

    def forbidden_tensor(*args, **kwargs):
        raise AssertionError("Unit refusal must precede numerical tensor materialization.")

    monkeypatch.setattr(fb.torch, "tensor", forbidden_tensor)
    with pytest.raises(AnalysisError) as caught:
        _prepare(frame)
    assert caught.value.code == "resource_limit"


def test_source_ledger_retains_original_typed_ids_including_missing_and_zero():
    frame, _, counts = _domain(1307, "elliptical")
    labels = [(i % 3, str(i)) for i in range(len(frame))]
    frame.index = pd.MultiIndex.from_tuples(labels, names=["wave", Decimal("2.25")])
    frame.columns.name = ("instrument", 7)
    frame.loc[labels[2], "y"] = np.nan
    sample = _prepare(frame)
    assert sample.accounting["index_code"].tolist() == [encode(x) for x in labels]
    assert sample.attrs["index_names_json"] == [encode(x) for x in frame.index.names]
    assert sample.attrs["column_axis_names_json"] == [encode(frame.columns.name)]
    assert sample.positions == [i for i in range(len(frame)) if i != 2]
    assert sample.accounting["complete"].tolist() == [i != 2 for i in range(len(frame))]
    assert sample.accounting["zero_complete_frequency"].tolist() == [i != 2 and w == 0 for i, w in enumerate(counts)]
    assert sample.accounting["frequency_json"].tolist() == [str(w) for w in counts]


def test_oversized_decimal_source_id_is_refused_before_codec(monkeypatch):
    frame, _, _ = _domain(1307, "elliptical")
    frame.index = pd.Index([Decimal("1"*100_000), *range(1, len(frame))], dtype=object)

    def forbidden_encoding(*args, **kwargs):
        raise AssertionError("Oversized Decimal must be bounded before conversion/encoding.")

    monkeypatch.setattr(fb.cu, "encode", forbidden_encoding)
    with pytest.raises(AnalysisError) as caught:
        _prepare(frame)
    assert caught.value.code == "resource_limit"


def _count_state():
    frame = pd.DataFrame({"x": [-1., 0., 1., 3.], "y": [2., -1., 2., 0.],
                          "w": [7, 8, 0, 15]})
    sample = _prepare(frame)
    tables = fb.source_tables(sample)
    tables["replicate_counts"] = pd.DataFrame(np.tile(sample.counts.numpy(), (39, 1)))
    result = TableSet(tables, title="Independent count-state audit", **sample.attrs)
    return fb.seal(result)


def test_complete_count_state_json_roundtrip_preserves_integer_semantics():
    result = _count_state()
    restored = restore_summary(summary_state(result))
    fb.validate_saved(restored)
    assert summary_state(restored) == summary_state(result)


def test_public_api_roundtrip_preserves_all_latex_tables_and_source_types():
    frame, _, _ = _domain(1307, "elliptical")
    frame.index.name = "observation"
    frame.columns.name = ("measurement", 7)
    frame.loc[2, "y"] = np.nan
    result = pca_subspace_fweight_bootstrap(
        frame, ["x", "y"], weights="w", components=1,
        matrix="covariance", replications=39, confidence=.95, seed=317,
    )
    payload = json.loads(json.dumps({
        "summary": summary_state(result),
        "table_order": list(result),
        "table_dtypes": {name: [str(dtype) for dtype in table.dtypes]
                         for name, table in result.items()},
    }, allow_nan=False))
    restored = restore_summary(payload["summary"])
    # The native/full-output pack separately retains display order and dtypes:
    # finite summary JSON sorts keys and represents unavailable float cells as
    # null. Restoring this complete pack must also preserve the exact renderer.
    for name, dtypes in payload["table_dtypes"].items():
        restored[name] = restored[name].astype(dict(zip(restored[name].columns, dtypes)))
    restored = TableSet({name: restored[name] for name in payload["table_order"]},
                        title=restored.title, **restored.attrs)
    fb.validate_saved(restored)
    assert summary_state(restored) == payload["summary"]
    assert list(restored) == list(result)
    assert restored.to_latex() == result.to_latex()
    for name, original in result.items():
        pd.testing.assert_frame_equal(restored[name], original, check_exact=True)
        pd.testing.assert_series_equal(restored[name].dtypes, original.dtypes)
    # Raw pandas frames use a different LaTeX renderer. Keep the source tables
    # in the public frame class before saving, including integer/Boolean ledger
    # fields that must not be promoted by the result-table constructor.
    for name in ("sample", "source_frequencies", "source_index", "source_accounting"):
        assert isinstance(result[name], DataFrame)
        assert isinstance(restored[name], DataFrame)
    assert pd.api.types.is_integer_dtype(result["source_frequencies"]["frequency"])
    for name in ("complete", "zero_complete_frequency"):
        assert pd.api.types.is_bool_dtype(result["source_accounting"][name])


@pytest.mark.parametrize("field,value", [("n_units", 30.), ("n_units", True),
                                         ("replications", 39.), ("replications", True)])
def test_rehashed_nonintegral_or_boolean_saved_dimensions_are_refused(field, value):
    result = _count_state()
    result.attrs[field] = value
    fb.seal(result)
    with pytest.raises(AnalysisError) as caught:
        fb.validate_saved(result)
    assert caught.value.code == "invalid_state"


def test_large_location_weighted_covariance_keeps_small_centered_signal():
    frame, raw, counts = _domain(1703, "asymmetric")
    raw = raw + np.array([1e12, -1e12])
    frame.loc[:, ["x", "y"]] = raw
    sample = _prepare(frame)
    mean, covariance = _expanded_moments(raw, counts)
    fitted = fb.moments(sample.values, sample.counts)
    np.testing.assert_array_equal(fitted.mean.numpy(), mean)
    np.testing.assert_allclose(fitted.covariance.numpy(), covariance, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("change", ["boolean", "negative", "wrong_total", "zero_support", "overflow"])
def test_rehashed_impossible_count_state_is_refused(change):
    result = _count_state()
    if change == "boolean":
        result["replicate_counts"] = result["replicate_counts"].astype(bool)
    elif change == "negative":
        result["replicate_counts"].iloc[0, 0] = -1
        result["replicate_counts"].iloc[0, 1] += 8
    elif change == "wrong_total":
        result["replicate_counts"].iloc[0, 0] += 1
    elif change == "zero_support":
        result["replicate_counts"].iloc[0, 0] -= 1
        result["replicate_counts"].iloc[0, 2] += 1
    else:
        forged = np.asarray([2**62, 2**62, 2**62, 2**62+30], dtype=np.int64)
        result["source_frequencies"] = pd.DataFrame({"frequency": forged})
        result["replicate_counts"] = pd.DataFrame(np.tile(forged, (39, 1)))
    fb.seal(result)  # Updated hashes must not hide impossible integer counts.
    with pytest.raises(AnalysisError) as caught:
        fb.validate_saved(result)
    assert caught.value.code == "invalid_state"
