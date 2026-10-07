"""Independent pandas/scalar preparation oracles, replay failures and resource admission."""

from copy import deepcopy
import os

import pandas as pd
import pytest

import openecon as oe
from openecon.data import DataError
from openecon.dataset import Dataset


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(root))
    yield root
    assert not list(root.iterdir())


def collect(source, batch_rows=7):
    return pd.concat(list(source.iter_batches(batch_rows=batch_rows)))


def base():
    frame = pd.DataFrame(
        {
            "key": pd.array([1, 2, 2, None], dtype="Int64"),
            "x": pd.array([2, None, 4, 5], dtype="Int64"),
            "kind": pd.Categorical(
                ["A", "B", "A", "B"], categories=["A", "B", "unused"], ordered=True
            ),
        },
        index=pd.Index([10, 20, 30, 40], name="original"),
    )
    frame.attrs["metadata"] = {
        "column_labels": {"x": "Original value"},
        "value_labels": {"x": {"2": "Two"}},
    }
    return frame


def test_project_filter_map_preserve_original_index_nullable_categories_metadata_and_replay():
    frame = base()
    source = Dataset.from_frame(frame)
    original = deepcopy(frame)
    planned = source.project(["x", "kind"]).filter(
        lambda block: (block["x"] >= 3).astype("boolean"), missing="keep"
    )
    actual = collect(planned)
    expected = frame.loc[frame["x"].ge(3).fillna(True), ["x", "kind"]]
    pd.testing.assert_frame_equal(actual, expected, check_flags=False)
    assert planned.metadata["column_labels"] == {"x": "Original value"}
    assert actual["kind"].dtype == frame["kind"].dtype
    digest = planned.preparation_receipt["output_digest"]
    pd.testing.assert_frame_equal(collect(planned, 1), expected, check_flags=False)
    assert planned.preparation_receipt["output_digest"] == digest and planned.row_count == 3
    mapped = source.map(
        lambda b: b.assign(z=(b["x"] * 2).astype("Int64")),
        schema={**dict(frame.dtypes), "z": "Int64"},
    )
    pd.testing.assert_frame_equal(
        collect(mapped), frame.assign(z=(frame["x"] * 2).astype("Int64")), check_flags=False
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("wrong", ["schema", "index", "mask", "missing"])
def test_invalid_callbacks_refused(wrong):
    source = Dataset.from_frame(base())
    if wrong == "schema":
        planned = source.map(
            lambda b: b, schema={"key": "float64", "x": "Int64", "kind": base()["kind"].dtype}
        )
    elif wrong == "index":
        planned = source.map(lambda b: b.reset_index(drop=True), schema=dict(base().dtypes))
    elif wrong == "mask":
        planned = source.filter(lambda b: pd.Series([True] * len(b)))
    else:
        planned = source.filter(lambda b: b["x"] > 3, missing="error")
    with pytest.raises(DataError):
        collect(planned)


def test_nondeterministic_source_function_metadata_and_one_shot_refused():
    source = Dataset.from_frame(base())
    state = {"n": 0}

    def change(block):
        state["n"] += 1
        return block.assign(z=pd.Series(state["n"], index=block.index, dtype="Int64"))

    planned = source.map(change, schema={**dict(base().dtypes), "z": "Int64"})
    collect(planned)
    with pytest.raises(DataError, match="different output"):
        collect(planned)
    source = Dataset.from_frame(base())
    planned = source.project(["x"])
    collect(planned)
    source._frame.loc[10, "x"] = 99
    with pytest.raises(DataError, match="source changed"):
        collect(planned)
    once = iter([base()])
    planned = Dataset.from_batches(lambda: once, columns=list(base().columns)).project(["x"])
    collect(planned)
    with pytest.raises(DataError, match="one-shot"):
        collect(planned)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_disk_join_matches_pandas_values_and_retains_both_source_indices(how):
    left = base().drop(columns="kind")
    left["key"] = left["key"].fillna(9)
    right = pd.DataFrame(
        {"key": pd.array([1, 2, 3], dtype="Int64"), "r": pd.array([7, 8, 9], dtype="Int64")},
        index=pd.Index(["a", "b", "c"], name="right_original"),
    )
    joined = Dataset.from_frame(left).join(Dataset.from_frame(right), on="key", how=how)
    actual = collect(joined)
    expected = pd.merge(left.reset_index(), right.reset_index(), on="key", how=how, sort=False)
    actual_values = (
        actual.reset_index(drop=True).sort_values(["key", "x", "r"]).reset_index(drop=True)
    )
    expected_values = (
        expected[["key", "x", "r"]].sort_values(["key", "x", "r"]).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual_values, expected_values, check_dtype=False)
    pairs = set(actual.index)
    assert (10, "a") in pairs and (20, "b") in pairs and (30, "b") in pairs
    if how in {"left", "outer"}:
        assert (40, None) in pairs
    if how in {"right", "outer"}:
        assert (None, "c") in pairs
    assert joined.preparation_receipt["scratch_removed"]
    digest = joined.preparation_receipt["output_digest"]
    collect(joined, 2)
    assert digest == joined.preparation_receipt["output_digest"]


def test_join_duplicates_missing_keys_typed_identity_and_expansion_admission():
    frame = pd.DataFrame(
        {"key": pd.array([1, 1, None], dtype="Int64"), "x": pd.array([4, 5, 6], dtype="Int64")}
    )
    left = Dataset.from_frame(frame)
    right = Dataset.from_frame(frame.rename(columns={"x": "y"}))
    with pytest.raises(DataError, match="duplicate keys"):
        collect(left.join(right, on="key"))
    with pytest.raises(DataError, match="before output allocation"):
        next(left.join(right, on="key", validate="m:m", max_rows=3).iter_batches())
    assert len(collect(left.join(right, on="key", validate="m:m"))) == 4
    assert len(collect(left.join(right, on="key", validate="m:m", nulls="equal"))) == 5
    identities = pd.DataFrame({"key": pd.Series([1, "1", True], dtype=object), "v": [10, 20, 30]})
    right = pd.DataFrame({"key": pd.Series([1, "1", True], dtype=object), "r": [100, 200, 300]})
    actual = collect(Dataset.from_frame(identities).join(Dataset.from_frame(right), on="key"))
    assert actual["r"].tolist() == [100, 200, 300] and len(actual) == 3


def test_long_wide_values_levels_duplicate_cells_unknown_and_original_indices():
    source = pd.DataFrame(
        {
            "id": pd.array([1, 2], dtype="Int64"),
            "a": pd.array([2, None], dtype="Int64"),
            "b": pd.array([3, 5], dtype="Int64"),
        },
        index=pd.Index(["first", "second"], name="record"),
    )
    long = Dataset.from_frame(source).reshape_long(id_vars=["id"], value_vars=["a", "b"])
    actual = collect(long)
    assert list(actual.index) == [("first", "a"), ("first", "b"), ("second", "a"), ("second", "b")]
    assert actual["variable"].tolist() == ["a", "b", "a", "b"]
    wide = long.reshape_wide(keys="id", variable="variable", value="value", levels=["a", "b"])
    actual = collect(wide)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), source.reset_index(drop=True))
    assert actual.index[0] == ((1,), (("a", ("first", "a")), ("b", ("first", "b"))))
    assert wide.preparation_receipt["scratch_removed"]
    extra = Dataset.from_frame(
        pd.DataFrame({"id": [1, 1, 1], "variable": ["a", "a", "other"], "value": [2, 3, 4]})
    )
    with pytest.raises(DataError, match="duplicate"):
        collect(extra.reshape_wide(keys="id", variable="variable", value="value", levels=["a"]))
    with pytest.raises(DataError, match="undeclared"):
        collect(
            extra.reshape_wide(
                keys="id", variable="variable", value="value", levels=["a"], duplicates="first"
            )
        )
    assert collect(
        extra.reshape_wide(
            keys="id",
            variable="variable",
            value="value",
            levels=["a"],
            duplicates="last",
            unknown="drop",
        )
    )["a"].tolist() == [3]
    with pytest.raises(DataError, match="before reading"):
        Dataset.from_frame(source).reshape_long(id_vars=["id"], value_vars=["a", "b"], max_rows=3)


def test_memory_disk_work_error_and_early_close_remove_owned_state():
    source = Dataset.from_frame(base())
    for limits in ({"memory_bytes": 1}, {"disk_bytes": 1}, {"max_work": 1}):
        with pytest.raises(DataError):
            collect(source.join(source, on="key", validate="m:m", **limits))
    iterator = source.join(source, on="key", validate="m:m").iter_batches(batch_rows=1)
    next(iterator)
    iterator.close()
    assert not list(os.scandir(os.environ["OPENECON_SCRATCH_DIRECTORY"]))


def test_native_fit_prepared_input_matches_dense_with_missing_rows_and_original_indices():
    frame = pd.DataFrame(
        {
            "x": pd.array(range(101), dtype="Float64"),
            "y": pd.array([2 + 0.7 * i + (i % 7) / 10 for i in range(101)], dtype="Float64"),
        },
        index=pd.Index(range(200, 301), name="sample"),
    )
    frame.loc[225, "y"] = pd.NA
    source = (
        Dataset.from_frame(frame)
        .filter(lambda b: b["x"] > 10)
        .map(
            lambda b: b.assign(z=(b["x"] ** 2).astype("Float64")),
            schema={"x": "Float64", "y": "Float64", "z": "Float64"},
        )
    )
    dense = frame.loc[frame["x"] > 10].assign(z=lambda b: (b["x"] ** 2).astype("Float64"))
    one = oe.ols(data=source, y="y", x=["x", "z"], covariance="HC3", device="cpu")
    two = oe.ols(data=dense, y="y", x=["x", "z"], covariance="HC3", device="cpu")
    assert one.nobs == two.nobs == 89
    for actual, expected in zip(one.coefficients, two.coefficients, strict=True):
        assert actual.estimate == pytest.approx(expected.estimate, rel=1e-8, abs=1e-10)
        assert actual.std_error == pytest.approx(expected.std_error, rel=1e-8, abs=1e-10)
    assert source.row_count == 90


def test_missing_code_semantics_survive_slicing_filter_map_long_wide_and_join():
    frame = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype="Int64"),
            "a": pd.array([None, 2, None], dtype="Float64"),
            "b": pd.array([1, None, 3], dtype="Float64"),
        }
    )
    frame.attrs["metadata"] = {
        "extended_missing_codes": {"a": {"a": [0], "z": [2]}, "b": {"b": [1]}},
        "column_labels": {"a": "A", "b": "B"},
    }
    source = Dataset.from_frame(frame)
    filtered = source.filter(lambda b: b["id"] < 3).project(["id", "a", "b"])
    assert list(filtered.iter_missing_codes("a", batch_rows=1)) == [
        {"a": [0]},
        {"a": []},
    ]
    mapped = filtered.map(lambda b: b.assign(a=b["a"].fillna(0)), schema=dict(frame.dtypes))
    assert all(
        not any(positions for positions in block.values())
        for block in mapped.iter_missing_codes("a", batch_rows=1)
    )
    long = source.reshape_long(id_vars=["id"], value_vars=["a", "b"])
    codes = [
        x
        for block in long.iter_missing_codes("value", batch_rows=1)
        for x, positions in block.items()
        if positions
    ]
    assert codes == ["a", "b", "z"]
    wide = long.reshape_wide(keys="id", variable="variable", value="value", levels=["a", "b"])
    assert any(block.get("a") for block in wide.iter_missing_codes("a"))
    joined = source.join(
        Dataset.from_frame(
            pd.DataFrame({"id": pd.array([1, 2, 3], dtype="Int64"), "x": [1, 2, 3]})
        ),
        on="id",
    )
    assert [
        len(indices)
        for block in joined.iter_missing_codes("a", batch_rows=1)
        for indices in block.values()
        if indices
    ] == [1, 1]


def test_code_identity_and_concurrent_pass_refused_before_source_consumption():
    def fn(block):
        return block.copy()

    source = Dataset.from_frame(base())
    mapped = source.map(fn, schema=dict(base().dtypes))

    def changed(block):
        return block.copy()

    fn.__code__ = changed.__code__
    with pytest.raises(DataError, match="identity"):
        collect(mapped)
    projected = source.project(["x"])
    pending = projected.iter_batches(batch_rows=1)
    next(pending)
    with pytest.raises(DataError, match="pending"):
        collect(projected)
    pending.close()
    assert len(collect(projected)) == 4


def test_physical_multiblock_mutation_empty_schema_and_hard_disk_admission(tmp_path):
    frame = pd.DataFrame(
        {"id": pd.array(range(17001), dtype="Int64"), "x": pd.array(range(17001), dtype="Float64")},
        index=pd.Index(range(30000, 47001), name="record"),
    )
    path = tmp_path / "physical.parquet"
    frame.to_parquet(path, row_group_size=2048)
    source = oe.scan(path).map(lambda b: b.astype(dict(frame.dtypes)), schema=dict(frame.dtypes))
    lookup = Dataset.from_frame(frame.iloc[::1000].copy())
    joined = source.join(lookup, on="id", how="left")
    actual = collect(joined, 103)
    assert len(actual) == 17001 and actual.index[0] == (30000, 30000)
    assert joined.preparation_receipt["scratch_removed"]
    with pytest.raises(DataError):
        collect(source.join(lookup, on="id", how="left", disk_bytes=4096))
    empty = source.filter(lambda b: b["id"] < 0)
    assert list(empty.join(lookup, on="id").iter_batches()) == []
    assert empty.row_count == 0
    path.write_bytes(b"changed")
    with pytest.raises(DataError, match="changed"):
        collect(joined)


def test_changed_values_category_schema_and_ambiguous_tag_indices_are_refused():
    frame = base()
    source = Dataset.from_frame(frame).project(["x"])
    collect(source)
    frame.loc[10, "x"] = 7
    with pytest.raises(DataError, match="changed|different"):
        collect(source)
    tagged = pd.DataFrame({"x": pd.array([None, None], dtype="Float64")}, index=[1, 1])
    tagged.attrs["metadata"] = {"extended_missing_codes": {"x": {"a": [1]}}}
    with pytest.raises(DataError, match="unique original"):
        collect(Dataset.from_frame(tagged).project(["x"]))

    def changing_categories():
        yield pd.DataFrame({"c": pd.Categorical(["a"], categories=["a", "b"])})
        yield pd.DataFrame({"c": pd.Categorical(["a"], categories=["a", "c"])})

    with pytest.raises(DataError, match="schema"):
        collect(Dataset.from_batches(changing_categories, columns=["c"]).project(["c"]))
