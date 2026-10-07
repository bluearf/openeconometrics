"""Projected, replayable file sources and bounded reader failure contracts."""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openecon.data import DataError
from openecon.dataset import Dataset, scan
from openecon.frame import DataFrame


def _parquet(path, values=None):
    frame = pd.DataFrame(values or {"y": [1.0, 2.0, None, 4.0], "x": [10, 20, 30, 40],
                                   "unused": ["a", "b", "c", "d"]})
    frame.to_parquet(path, index=False, row_group_size=2)
    return frame


def test_parquet_projection_nullable_types_and_replay(tmp_path):
    file = tmp_path / "data.parquet"
    _parquet(file)
    source = scan(file)
    assert source.columns == ["y", "x", "unused"]
    assert source.row_count == source.nrows == 4
    passes = [list(source.iter_batches(["x", "y"], batch_rows=2)) for _ in range(2)]
    for batches in passes:
        assert sum(len(batch) for batch in batches) == 4
        assert all(list(batch.columns) == ["x", "y"] and len(batch) <= 2 for batch in batches)
        assert batches[0].x.dtype == pd.ArrowDtype(pa.int64())
        assert pd.concat(batches).y.isna().tolist() == [False, False, True, False]
    pd.testing.assert_frame_equal(pd.concat(passes[0]), pd.concat(passes[1]))


def test_parquet_scanner_is_projected_without_read_ahead_growth(tmp_path):
    file = tmp_path / "data.parquet"
    _parquet(file)
    source = scan(file)
    original = source._arrow_dataset
    calls = []

    class Proxy:
        def scanner(self, **kwargs):
            calls.append(kwargs)
            return original.scanner(**kwargs)

    source._arrow_dataset = Proxy()
    # File readers now execute this local scanner inside the supervised parser.
    # Inspect the scanner kernel here; reader receipts and transfer tests verify
    # isolation and projected data in the public iter_batches path separately.
    list(source._local_batches(("x",), batch_rows=1))
    assert calls == [{"columns": ["x"], "batch_size": 1, "use_threads": False,
                      "batch_readahead": 1, "fragment_readahead": 1}]


def test_parquet_directory_hive_partition_and_deterministic_order(tmp_path):
    for region, rows in [("west", [3, 4]), ("east", [1, 2])]:
        folder = tmp_path / f"region={region}"
        folder.mkdir()
        _parquet(folder / "part.parquet", {"x": rows, "y": [float(row) for row in rows]})
    source = scan(tmp_path)
    assert source.columns == ["x", "y", "region"]
    assert source.row_count == 4
    batch = pd.concat(source.iter_batches(["region", "x"], batch_rows=1), ignore_index=True)
    assert batch.x.tolist() == [1, 2, 3, 4]
    assert batch.region.tolist() == ["east", "east", "west", "west"]


def test_parquet_directory_rejects_heterogeneous_schema(tmp_path):
    _parquet(tmp_path / "a.parquet", {"x": [1, 2]})
    _parquet(tmp_path / "b.parquet", {"x": [1.0, 2.0]})
    with pytest.raises(DataError) as exc:
        scan(tmp_path)
    assert exc.value.code == "SCHEMA_MISMATCH"


def test_parquet_duplicate_columns_rejected(tmp_path):
    file = tmp_path / "duplicate.parquet"
    pq.write_table(pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["x", "x"]), file)
    with pytest.raises(DataError):
        scan(file)


def test_csv_projection_unknown_count_and_replay(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("x,y,unused\n1,2,a\n3,,b\n5,6,c\n", encoding="utf-8")
    source = scan(file)
    assert source.row_count is None
    batches = list(source.iter_batches(["y", "x"], batch_rows=2))
    assert [len(batch) for batch in batches] == [2, 1]
    assert all(list(batch.columns) == ["y", "x"] for batch in batches)
    assert pd.concat(batches).x.tolist() == [1, 3, 5]
    assert pd.concat(batches).y.isna().tolist() == [False, True, False]
    pd.testing.assert_frame_equal(pd.concat(batches),
                                  pd.concat(source.iter_batches(["y", "x"], batch_rows=2)))


@pytest.mark.parametrize("header", ["x,x", "x,", ",y", "x,   ", ""])
def test_csv_raw_bad_headers_are_not_silently_mangled(tmp_path, header):
    file = tmp_path / "data.csv"
    file.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(DataError):
        scan(file)


def test_csv_bom_quoted_header_and_integer_precision(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text('\ufeff"large integer","comma,name"\n9007199254740993,7\n', encoding="utf-8")
    source = scan(file)
    preview = source.head()
    assert preview.columns.tolist() == ["large integer", "comma,name"]
    assert int(preview.iloc[0, 0]) == 9007199254740993
    assert isinstance(preview, DataFrame)
    assert "tabular" in preview.to_latex()


def test_no_total_row_or_byte_ceiling_is_applied_to_sources(tmp_path, monkeypatch):
    from openecon import data
    monkeypatch.setattr(data, "MAX_ROWS", 1)
    monkeypatch.setattr(data, "MAX_FILE_BYTES", 1)
    file = tmp_path / "data.parquet"
    _parquet(file)
    source = scan(file)
    assert sum(len(batch) for batch in source.iter_batches(["x"], batch_rows=1)) == 4


def test_scan_reads_only_metadata_and_head_remains_bounded(tmp_path, monkeypatch):
    file = tmp_path / "data.parquet"
    _parquet(file)

    def forbid(*args, **kwargs):
        raise AssertionError("A streaming source must not use eager read_parquet.")

    monkeypatch.setattr(pd, "read_parquet", forbid)
    source = scan(file)
    assert source.head(1).x.tolist() == [10]
    assert source.head(0).empty
    assert source.head(10).x.tolist() == [10, 20, 30, 40]


def test_file_replacement_between_passes_is_rejected(tmp_path):
    file = tmp_path / "data.parquet"
    _parquet(file)
    source = scan(file)
    list(source.iter_batches(["x"], batch_rows=2))
    replacement = tmp_path / "replacement.parquet"
    _parquet(replacement)
    replacement.replace(file)
    with pytest.raises(DataError) as exc:
        list(source.iter_batches(["x"], batch_rows=2))
    assert exc.value.code == "SOURCE_CHANGED"


def test_file_mutation_during_pass_is_rejected_on_close(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("x\n1\n2\n3\n", encoding="utf-8")
    source = scan(file)
    iterator = source.iter_batches(["x"], batch_rows=1)
    assert next(iterator).x.tolist() == [1]
    file.write_text("x\n4\n5\n6\n", encoding="utf-8")
    with pytest.raises(DataError) as exc:
        iterator.close()
    assert exc.value.code == "SOURCE_CHANGED"


def test_parquet_manifest_added_or_removed_file_is_rejected(tmp_path):
    _parquet(tmp_path / "a.parquet")
    source = scan(tmp_path)
    _parquet(tmp_path / "b.parquet")
    with pytest.raises(DataError) as exc:
        source.assert_unchanged()
    assert exc.value.code == "SOURCE_CHANGED"


def test_deleted_source_is_rejected(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("x\n1\n", encoding="utf-8")
    source = scan(file)
    file.unlink()
    with pytest.raises(DataError) as exc:
        source.head()
    assert exc.value.code == "SOURCE_CHANGED"


def test_frame_wrap_avoids_global_copy_and_copies_only_projected_batches():
    frame = pd.DataFrame({"x": range(9), "y": range(9), "unused": ["s"] * 9})
    frame.attrs["metadata"] = {"label": "source"}
    source = Dataset.from_frame(frame)
    assert source._frame is frame
    assert source.row_count == 9
    batches = list(source.iter_batches(["y", "x"], batch_rows=4))
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert all(batch.columns.tolist() == ["y", "x"] for batch in batches)
    assert source.head(2).attrs["metadata"] == {"label": "source"}
    preview = source.head(2)
    preview.attrs["metadata"]["label"] = "mutated copy"
    assert source.metadata["label"] == "source"


@pytest.mark.parametrize("change", ["count", "dtype", "metadata", "columns", "categories"])
def test_frame_schema_count_and_metadata_changes_rejected(change):
    frame = pd.DataFrame({"x": [1, 2], "category": pd.Categorical(["a", "b"])})
    frame.attrs["metadata"] = {"label": "source"}
    source = Dataset.from_frame(frame)
    if change == "count":
        frame.loc[2] = [3, "a"]
    elif change == "dtype":
        frame["x"] = frame.x.astype("float64")
    elif change == "metadata":
        frame.attrs["metadata"]["label"] = "changed"
    elif change == "columns":
        frame.rename(columns={"x": "z"}, inplace=True)
    else:
        frame["category"] = frame.category.cat.reorder_categories(["b", "a"])
    with pytest.raises(DataError) as exc:
        source.assert_unchanged()
    assert exc.value.code == "SOURCE_CHANGED"


def test_large_unused_category_schema_hashing_is_bounded(monkeypatch):
    import openecon.dataset as module
    frame = pd.DataFrame({"x": range(11), "unused": pd.Categorical(range(11))})
    monkeypatch.setattr(module, "DEFAULT_BATCH_ROWS", 3)
    original = pd.util.hash_pandas_object
    sizes = []

    def capture(value, **kwargs):
        sizes.append(len(value))
        return original(value, **kwargs)

    monkeypatch.setattr(pd.util, "hash_pandas_object", capture)
    source = Dataset.from_frame(frame)
    source.assert_unchanged()
    assert sizes and max(sizes) <= 3
    assert sum(len(batch) for batch in source.iter_batches(["x"], batch_rows=3)) == 11


def test_factory_replays_and_splits_supplied_chunks():
    calls = []

    def factory():
        calls.append("pass")
        yield pd.DataFrame({"x": range(7), "y": range(7)})

    source = Dataset.from_batches(factory, ["x", "y"], row_count=7)
    for _ in range(2):
        batches = list(source.iter_batches(["y"], batch_rows=3))
        assert [len(batch) for batch in batches] == [3, 3, 1]
        assert all(batch.columns.tolist() == ["y"] for batch in batches)
    assert len(calls) == 2


def test_declared_100_billion_count_is_exact_python_integer_without_materializing():
    calls = []

    def factory():
        calls.append("started")
        yield pd.DataFrame({"x": [1, 2]})
        raise AssertionError("Preview must not consume the complete source.")

    source = Dataset.from_batches(factory, ["x"], row_count=100_000_000_000)
    assert source.row_count == source.nrows == 100_000_000_000
    assert isinstance(source.provenance["row_count"], int)
    assert calls == []
    assert source.head(1).x.tolist() == [1]
    assert calls == ["started"]


def test_factory_declared_count_mismatch_is_rejected():
    source = Dataset.from_batches(lambda: iter([pd.DataFrame({"x": [1]})]), ["x"], row_count=2)
    with pytest.raises(DataError) as exc:
        list(source.iter_batches())
    assert exc.value.code == "SOURCE_CHANGED"


def test_factory_one_shot_iterator_is_rejected():
    iterator = iter([pd.DataFrame({"x": [1]})])
    with pytest.raises(DataError) as exc:
        Dataset.from_batches(iterator, ["x"])
    assert exc.value.code == "NON_REPLAYABLE_SOURCE"
    source = Dataset.from_batches(lambda: iterator, ["x"])
    list(source.iter_batches())
    with pytest.raises(DataError) as exc:
        list(source.iter_batches())
    assert exc.value.code == "NON_REPLAYABLE_SOURCE"


def test_factory_batch_schema_errors_are_clear():
    source = Dataset.from_batches(lambda: iter([pd.DataFrame({"z": [1]})]), ["x"])
    with pytest.raises(DataError) as exc:
        list(source.iter_batches())
    assert exc.value.code == "INVALID_BATCH"


@pytest.mark.parametrize("bad", ["https://example.com/data.csv", "s3://bucket/data.parquet",
                                  "file:///tmp/data.csv", 12])
def test_network_or_nonpath_sources_rejected(bad):
    with pytest.raises(DataError) as exc:
        scan(bad)
    assert exc.value.code == "INVALID_SOURCE"


@pytest.mark.parametrize("batch_rows", [0, -1, True, 1.5, 1_000_001])
def test_invalid_batch_sizes_rejected(batch_rows):
    source = Dataset.from_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(DataError):
        list(source.iter_batches(batch_rows=batch_rows))


@pytest.mark.parametrize("columns", [[], ["x", "x"], ["missing"], "x", [2]])
def test_invalid_projection_rejected(columns):
    source = Dataset.from_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(DataError):
        list(source.iter_batches(columns))


def test_batch_byte_limit_applies_only_to_materialized_projection(monkeypatch):
    import openecon.dataset as module
    frame = pd.DataFrame({"x": [1, 2], "huge": ["a" * 10_000, "b" * 10_000]})
    source = Dataset.from_frame(frame)
    monkeypatch.setattr(module, "MAX_BATCH_BYTES", 1000)
    assert sum(len(batch) for batch in source.iter_batches(["x"], batch_rows=1)) == 2
    with pytest.raises(DataError) as exc:
        list(source.iter_batches(batch_rows=1))
    assert exc.value.code == "BATCH_LIMIT"


def test_manifest_and_column_limits_are_bounded(tmp_path, monkeypatch):
    import openecon.dataset as module
    _parquet(tmp_path / "a.parquet")
    _parquet(tmp_path / "b.parquet")
    monkeypatch.setattr(module, "MAX_SOURCE_FILES", 1)
    with pytest.raises(DataError) as exc:
        scan(tmp_path)
    assert exc.value.code == "MANIFEST_LIMIT"
    monkeypatch.setattr(module, "MAX_SOURCE_COLUMNS", 1)
    with pytest.raises(DataError):
        Dataset.from_frame(pd.DataFrame({"x": [1], "y": [2]}))


@pytest.mark.parametrize("index", [pd.Index(["a", "a", "c", "b"], name="label"),
                                  pd.RangeIndex(10, 18, 2, name="position"),
                                  pd.MultiIndex.from_tuples([(1, "a"), (1, "a"), (2, "b"), (3, "c")], names=["id", "label"])])
def test_parquet_projection_keeps_stored_index_without_exposing_storage_columns(tmp_path, index):
    frame = pd.DataFrame({"x": [1., 2., 3., 4.], "unused": [5, 6, 7, 8]}, index=index)
    path = tmp_path / "indexed.parquet"
    frame.to_parquet(path)
    source = scan(path)
    assert source.columns == frame.columns.tolist()
    result = pd.concat(source.iter_batches(["x"], batch_rows=1))
    assert result.index.tolist() == frame.index.tolist()
    assert result.index.names == frame.index.names
    assert result.columns.tolist() == ["x"]
    assert source.content_hash(["x"], batch_rows=1) == source.content_hash(["x"], batch_rows=3)


def test_unindexed_parquet_projection_assigns_global_physical_row_indexes(tmp_path):
    path = tmp_path / "plain.parquet"
    frame = pd.DataFrame({"x": range(7)})
    frame.to_parquet(path, index=False, row_group_size=2)
    result = pd.concat(scan(path).iter_batches(batch_rows=2))
    assert result.index.tolist() == list(range(7))


def test_parquet_projection_preserves_declared_category_order_and_unused_levels(tmp_path):
    frame = pd.DataFrame({"g": pd.Categorical(["B", "A", "B"], categories=["B", "A", "C"], ordered=True),
                          "precise": pd.array([2**63 - 1, None, 3], dtype="Int64")})
    path = tmp_path / "categories.parquet"
    frame.to_parquet(path, index=False, row_group_size=1)
    for block in scan(path).iter_batches(batch_rows=1):
        assert isinstance(block.g.dtype, pd.CategoricalDtype)
        assert block.g.cat.categories.tolist() == ["B", "A", "C"]
        assert block.g.cat.ordered
    assert next(scan(path).iter_batches(["precise"], batch_rows=1)).precise.iloc[0] == 2**63 - 1
