"""Streaming errors preserve the public specification's validation semantics."""
import pandas as pd
import pytest

import openecon as oe


@pytest.mark.parametrize("automatic", [False, True])
def test_generated_intercept_term_collision_is_rejected_before_scan(monkeypatch, automatic):
    import openecon.analysis as analysis
    frame = pd.DataFrame({"y": [1., 2., 4., 3., 5.], "Intercept": [2., 1., 3., 4., 5.]})
    if automatic:
        monkeypatch.setattr(analysis, "_MAX_DESIGN_BYTES", 1)
    source = frame if automatic else oe.Dataset.from_frame(frame)
    with pytest.raises(oe.AnalysisError) as error:
        oe.ols(data=source, y="y", x=["Intercept"])
    assert error.value.code == "duplicate_terms"


def test_nonscalar_predictor_has_structured_error():
    source = oe.Dataset.from_frame(pd.DataFrame({"y": [1., 2., 3., 4., 5.],
                                               "x": [[1], [2], [3], [4], [5]]}))
    with pytest.raises(oe.AnalysisError) as error:
        oe.ols(data=source, y="y", x=["x"])
    assert error.value.code == "unsupported_values"


@pytest.mark.parametrize("rows,code", [(0, "empty_data"), (5, "empty_sample")])
def test_empty_stream_and_empty_retained_sample_are_distinct(rows, code):
    source = oe.Dataset.from_frame(pd.DataFrame({"y": [float("nan")] * rows,
                                               "x": [float("nan")] * rows}))
    with pytest.raises(oe.AnalysisError) as error:
        oe.ols(data=source, y="y", x=["x"], missing="drop")
    assert error.value.code == code
