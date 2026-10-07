"""Disk sorting preserves exact periods, physical positions and source guards."""
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.core import make_spec
from openecon.econometrics.ordered_replay import OrderedReplay
from openecon.analysis_contracts import AnalysisError


def spec(**kwargs):
    return make_spec("var", outcome="y", columns={"system": ["y", "z"]}, missing="drop", **kwargs)


@pytest.mark.parametrize("kind", ["integer", "datetime", "timezone"])
def test_sorted_projected_snapshot_and_cleanup(kind):
    frame = pd.DataFrame({"y": [2., 1., 3.], "z": [5., 7., 8.], "t": [2**54+1, 2**54, 2**54+2], "unused": [object(), object(), object()]})
    if kind != "integer":
        frame["t"] = pd.to_datetime(["2020-02-01", "2020-01-01", "2020-03-01"], utc=kind == "timezone")
    with OrderedReplay(spec(time="t"), Dataset.from_frame(frame)) as ordered:
        path = ordered.path.parent
        actual = pd.concat(list(ordered.source.iter_batches(batch_rows=2)), ignore_index=True)
        assert actual.y.tolist() == [1., 2., 3.]
        assert actual[ordered.positions_column].tolist() == [1, 0, 2]
        assert "unused" not in actual.columns
        if kind == "integer":
            assert actual.t.tolist() == [2**54, 2**54+1, 2**54+2]
            assert ordered.last_period == 2**54+2
        ordered.verify_original()
    assert not path.exists()


@pytest.mark.parametrize("time", [None, "t"])
def test_missing_only_at_ends(time):
    frame = pd.DataFrame({"y": [None, 1., 2., None], "z": [3., 2., 1., 0.], "t": range(4)})
    with OrderedReplay(spec(time=time), Dataset.from_frame(frame)) as ordered:
        assert ordered.count == 2 and ordered.original_count == 4
    frame.loc[2, "y"] = None
    frame.loc[3, "y"] = 4
    with pytest.raises(AnalysisError) as caught:
        with OrderedReplay(spec(time=time), Dataset.from_frame(frame)):
            pass
    assert caught.value.code == "time_gaps"


def test_duplicate_gaps_and_live_source_mutation():
    frame = pd.DataFrame({"y": [1., 2., 3.], "z": [3., 2., 1.], "t": [0, 1, 1]})
    with pytest.raises(AnalysisError) as caught:
        with OrderedReplay(spec(time="t"), Dataset.from_frame(frame)):
            pass
    assert caught.value.code == "duplicate_time"
    frame.t = [0, 1, 3]
    with pytest.raises(AnalysisError) as caught:
        with OrderedReplay(spec(time="t"), Dataset.from_frame(frame)):
            pass
    assert caught.value.code == "time_gaps"
    frame.t = [0, 1, 2]
    with OrderedReplay(spec(time="t"), Dataset.from_frame(frame)) as ordered:
        frame.loc[0, "y"] = 99.
        with pytest.raises(AnalysisError) as caught:
            ordered.verify_original()
        assert caught.value.code == "source_changed"
