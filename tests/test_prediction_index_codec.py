"""Owned prediction indexes retain identities and specialized dtype semantics."""
from decimal import Decimal
from uuid import UUID

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openecon.econometrics.postest.index_codec import IndexStorage


@pytest.mark.parametrize("index", [
    pd.CategoricalIndex(["A", "B", "A", None, "B"], categories=["B", "A", "unused"], ordered=True, name="id"),
    pd.period_range("2020-01", periods=5, freq="M", name="id"),
    pd.IntervalIndex.from_breaks([0, 1, 2, 3, 4, 5], closed="both", name="id"),
    pd.MultiIndex.from_arrays([
        pd.Categorical(["A", "B", "A", None, "B"], categories=["B", "A", "unused"], ordered=True),
        [1, 2, 1, 3, 2]], names=["category", "id"]),
    pd.Index([None, None, "a", 2, "2"], dtype=object, name="id"),
    pd.Index([Decimal("1.50"), UUID(int=4), b"a", ("a", 4), pd.Timestamp("2024-01-01")], dtype=object, name="id"),
    pd.Index([12, 15, 18, 21, 24], name="id"),
    pd.date_range("2020-01-01", periods=5, tz="Europe/Istanbul", name="id"),
    pd.TimedeltaIndex(pd.to_timedelta([1, 2, 3, 4, 5], unit="D"), name="id"),
    pd.Index(["a", None, "a", "b", "c"], dtype="string", name="id"),
])
def test_index_storage_round_trip_across_real_parquet_blocks(index, tmp_path):
    columns = ["response"]
    storage = IndexStorage(index[:2], columns)
    target = tmp_path / "owned.parquet"
    writer = None
    try:
        for start in range(0, len(index), 2):
            labels = index[start:start + 2]
            frame = pd.DataFrame({"response": range(start, start + len(labels))})
            record = storage.append(pa.Table.from_pandas(frame, preserve_index=False), labels)
            if writer is None:
                writer = pq.ParquetWriter(target, record.schema)
            writer.write_table(record)
    finally:
        if writer is not None:
            writer.close()
    with pq.ParquetFile(target) as reader:
        restored = [storage.restore(block.to_pandas()) for block in reader.iter_batches(batch_size=2)]
    actual = pd.concat(restored)
    pd.testing.assert_index_equal(actual.index, index, exact=True)
    assert actual.response.tolist() == list(range(len(index)))
    if isinstance(index, pd.CategoricalIndex):
        assert actual.index.categories.tolist() == ["B", "A", "unused"]
        assert actual.index.ordered
