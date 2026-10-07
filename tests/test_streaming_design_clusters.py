"""Independent pandas-factorize expectations for supported cluster identities."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon import streaming_design


def assert_factorize_parity(series: pd.Series):
    expected = pd.factorize(series, sort=False)[0].tolist()
    encoded = streaming_design.encode_cluster_labels(series)
    assert len(encoded) == len(series) and all(isinstance(key, bytes) for key in encoded)
    actual = pd.factorize(pd.Series(encoded, dtype=object), sort=False)[0].tolist()
    assert actual == expected
    for batch_rows in (1, 2, 7):
        replayed = []
        for start in range(0, len(series), batch_rows):
            replayed.extend(streaming_design.encode_cluster_labels(series.iloc[start:start + batch_rows]))
        assert replayed == encoded


@pytest.mark.parametrize("values", [
    [True, 1, 1., np.int64(1), np.uint64(1), np.float64(1.), np.bool_(True), False, 0],
    [-0., +0., 0, False, np.float32(-0.), np.int32(0)],
    [2**63 + 1, float(2**63), np.uint64(2**63 + 1), 2**63, -(2**63 + 1), float(-(2**63))],
    [2**200 + 1, 2**200, float(2**200), 2**200 + 1],
    ["1", 1, b"1", True, "é", "é", b"\xc3\xa9", "", b""],
    [(1,), (1.,), (True,), ("1",), (1, 2), (1., 2.), (12,), ()],
    [frozenset([1, 2]), frozenset([2., True]), frozenset([1, 3]), frozenset(), ()],
    [(frozenset([1, 2]), ("a", -0.)), (frozenset([2., True]), ("a", 0)),
     (frozenset([1, 3]), ("a", 0)), ((1, 2), frozenset(["a"]))],
    [UUID("12345678-1234-5678-1234-567812345678"), UUID("12345678-1234-5678-1234-567812345678"),
     "12345678-1234-5678-1234-567812345678", bytes.fromhex("12345678123456781234567812345678")],
    [datetime(2026, 1, 1), pd.Timestamp("2026-01-01"), datetime(2026, 1, 2), date(2026, 1, 1)],
    [date(2026, 1, 1), datetime(2026, 1, 1), pd.Timestamp("2026-01-01"), date(2026, 1, 1)],
    [datetime(1500, 1, 1), pd.Timestamp("1500-01-01"), datetime(2026, 1, 1)],
    [datetime(2026, 1, 1, tzinfo=timezone.utc), pd.Timestamp("2026-01-01T03:00:00+03:00"),
     datetime(2025, 12, 31, 19, tzinfo=timezone(timedelta(hours=-5))), pd.Timestamp("2026-01-01"),
     datetime(2026, 1, 1)],
    [pd.Timestamp("2026-01-01T00:00:00.000000001"), pd.Timestamp("2026-01-01T00:00:00.000000001"),
     datetime(2026, 1, 1), pd.Timestamp("2026-01-01")],
    [timedelta(days=1), pd.Timedelta(days=1), timedelta(days=-1, seconds=1),
     pd.Timedelta(days=-1, seconds=1), pd.Timedelta(1, unit="ns"), pd.Timedelta(2, unit="ns")],
    [(datetime(2026, 1, 1), timedelta(days=1)), (pd.Timestamp("2026-01-01"), pd.Timedelta(days=1))],
])
def test_supported_object_labels_preserve_actual_pandas_factorize(values):
    assert_factorize_parity(pd.Series(values, dtype=object))


@pytest.mark.parametrize("series", [
    pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"])),
    pd.Series(pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"])),
    pd.Series(pd.to_timedelta(["-1ns", "2 days", "-1ns", "5us"])),
    pd.Series(["a", "b", "a"], dtype="category"),
    pd.Series([1, 2, 1], dtype="Int64"),
    pd.Series([1., 2., 1.], dtype="Float64"),
])
def test_normal_typed_series_have_stable_cluster_identity(series):
    assert_factorize_parity(series)


def test_typed_pandas_dst_instants_stay_distinct():
    # Typed temporal factorization compares actual instants consistently. Object
    # temporal labels can have incompatible equality/hash rules at a DST fold.
    first = pd.Timestamp(datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0))
    second = pd.Timestamp(datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1))
    series = pd.Series([first, second, first, second])
    assert_factorize_parity(series)
    assert pd.factorize(series)[0].tolist() == [0, 1, 0, 1]


def test_ambiguous_object_timestamps_reject_inconsistent_hash_equality():
    first = pd.Timestamp(datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0))
    second = pd.Timestamp(datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1))
    series = pd.Series([first, second, first.tz_convert("UTC"), second.tz_convert("UTC")], dtype=object)
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(series)
    assert caught.value.code == "invalid_clusters"


@pytest.mark.parametrize("fold", [0, 1])
def test_ambiguous_plain_datetimes_reject_nontransitive_equality(fold):
    value = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=fold)
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([value], dtype=object))
    assert caught.value.code == "invalid_clusters"


@pytest.mark.parametrize("value", [np.datetime64("2026-01-01", "D"), np.datetime64("2026-01-01", "ns"),
                                  np.timedelta64(1, "D"), np.timedelta64(1, "ns")])
def test_object_numpy_temporals_are_explicitly_rejected(value):
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([value], dtype=object))
    assert caught.value.code == "invalid_clusters"


@pytest.mark.parametrize("value,code", [
    ([1], "invalid_clusters"), ({"a": 1}, "invalid_clusters"), ({1, 2}, "invalid_clusters"),
    (None, "invalid_clusters"), (pd.NA, "invalid_clusters"), (pd.NaT, "invalid_clusters"),
    (math.nan, "non_finite_values"), (math.inf, "non_finite_values"), (-math.inf, "non_finite_values"),
    (np.float64(math.nan), "non_finite_values"), (("a", math.inf), "non_finite_values"),
    (("a", None), "invalid_clusters"), ("\ud800", "invalid_clusters"),
])
def test_invalid_nonfinite_and_recursive_missing_labels_fail(value, code):
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([value], dtype=object))
    assert caught.value.code == code


def test_nested_identity_has_a_fixed_depth_limit():
    accepted = 1
    for _ in range(streaming_design.MAX_KEY_DEPTH):
        accepted = (accepted,)
    assert_factorize_parity(pd.Series([accepted, accepted], dtype=object))
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([(accepted,)], dtype=object))
    assert caught.value.code == "cluster_key_budget"


@pytest.mark.parametrize("value", ["x" * (6 * 1024 * 1024), b"x" * (6 * 1024 * 1024)])
def test_individual_label_respects_actual_six_mebibyte_budget(value):
    assert streaming_design.MAX_CLUSTER_BYTES == 6 * 1024 * 1024
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([value], dtype=object))
    assert caught.value.code == "cluster_key_budget"


def test_combined_label_budget_includes_repeated_payloads_and_list_overhead(monkeypatch):
    monkeypatch.setattr(streaming_design, "MAX_CLUSTER_BYTES", 256)
    label = "x" * 80
    assert streaming_design.encode_cluster_labels(pd.Series([label], dtype=object))
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([label, label], dtype=object))
    assert caught.value.code == "cluster_key_budget"


def test_recursive_members_share_the_same_byte_budget(monkeypatch):
    monkeypatch.setattr(streaming_design, "MAX_CLUSTER_BYTES", 256)
    with pytest.raises(AnalysisError) as caught:
        streaming_design.encode_cluster_labels(pd.Series([("x" * 100, "y" * 100)], dtype=object))
    assert caught.value.code == "cluster_key_budget"
