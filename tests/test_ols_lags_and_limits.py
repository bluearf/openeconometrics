"""Raw time histories and bounded formula expansion regression checks."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.linear_ols.design import OLSDesign


def series():
    generator = torch.Generator().manual_seed(87)
    x = torch.randn(80, dtype=torch.float64, generator=generator)
    lag = torch.cat([torch.tensor([float("nan")]), x[:-1]])
    y = 2 + .8 * lag + .1 * torch.randn(80, dtype=torch.float64, generator=generator)
    return pd.DataFrame({"t": range(1, 81), "x": x.numpy(), "y": y.numpy(), "w": 1.})


@pytest.mark.parametrize("term", ["L(x)", "D(x)", "L(x, 3)", "I(L(x) * x)"])
@pytest.mark.parametrize("batch", [2, 13])
def test_raw_lag_history_survives_missing_outcome_and_zero_weights(term, batch):
    frame = series()
    frame.loc[17, "y"] = float("nan")
    frame.loc[25, "w"] = 0.
    frame = frame.drop(index=[8, 41]).reset_index(drop=True)
    design = OLSDesign([term], [], intercept=True, time="t").prepare(frame)
    x = design.encode(frame, allow_missing=True)[:, 1]
    valid = torch.isfinite(x) & torch.tensor(frame.y.notna().tolist()) & torch.tensor((frame.w > 0).tolist())
    reference = frame.loc[valid.numpy()].copy()
    reference["z"] = x[valid].numpy()
    expected = oe.ols(data=reference, y="y", x=["z"], weights="w", weight_type="aweight", covariance="HC3")
    dense = oe.ols(data=frame, formula=f"y ~ {term}", time="t", weights="w", weight_type="aweight", covariance="HC3")
    source = oe.Dataset.from_batches(lambda: (frame.iloc[start:start+batch] for start in range(0, len(frame), batch)), frame.columns)
    streamed = oe.ols(data=source, formula=f"y ~ {term}", time="t", weights="w", weight_type="aweight", covariance="HC3")
    for result in (dense, streamed):
        assert result.nobs == expected.nobs
        assert [c.estimate for c in result.coefficients] == pytest.approx([c.estimate for c in expected.coefficients], abs=1e-11)
        assert [c.std_error for c in result.coefficients] == pytest.approx([c.std_error for c in expected.coefficients], abs=1e-11)


def test_lag_stream_requires_order_and_exact_integer_time():
    frame = series().dropna()
    for bad in (frame.iloc[::-1], frame.assign(t=frame.t + 2**53)):
        with pytest.raises(oe.AnalysisError) as error:
            oe.ols(data=oe.Dataset.from_frame(bad), formula="y ~ L(x)", time="t")
        assert error.value.code == "invalid_time"


def test_dense_lags_support_unsorted_input_by_time_values():
    frame = series()
    forward = oe.ols(data=frame, formula="y ~ L(x)", time="t")
    reverse = oe.ols(data=frame.iloc[::-1], formula="y ~ L(x)", time="t")
    assert reverse.nobs == forward.nobs
    assert [c.estimate for c in reverse.coefficients] == pytest.approx([c.estimate for c in forward.coefficients], abs=1e-12)


@pytest.mark.parametrize("term", ["L(x)", "L(x, 3)", "D(x)"])
def test_current_missing_predictor_does_not_remove_valid_history_only_rows(term):
    frame = series()
    frame.loc[17, "x"] = float("nan")
    design = OLSDesign([term], [], intercept=True, time="t").prepare(frame)
    values = design.encode(frame, allow_missing=True)[:, 1]
    keep = torch.isfinite(values) & torch.tensor(frame.y.notna().tolist())
    reference = frame.loc[keep.numpy()].copy()
    reference["z"] = values[keep].numpy()
    expected = oe.ols(data=reference, y="y", x=["z"])
    for data in (frame, oe.Dataset.from_batches(lambda: (frame.iloc[i:i+7] for i in range(0, len(frame), 7)), frame.columns)):
        result = oe.ols(data=data, formula=f"y ~ {term}", time="t")
        assert result.nobs == expected.nobs
        assert [coef.estimate for coef in result.coefficients] == pytest.approx([coef.estimate for coef in expected.coefficients], abs=1e-11)


def test_factorial_expansion_rejects_before_combinatorial_allocation():
    with pytest.raises(oe.AnalysisError) as error:
        OLSDesign(["*".join(f"x{i}" for i in range(25))], [], intercept=True)
    assert error.value.code == "model_too_wide"


def test_category_interaction_width_is_checked_before_products():
    frame = pd.DataFrame({name: range(80) for name in ("a", "b", "c")})
    design = OLSDesign(["C(a):C(b):C(c)"], [], intercept=True)
    with pytest.raises(oe.AnalysisError) as error:
        design.prepare(frame)
    assert error.value.code == "model_too_wide"
    assert len(design.terms) == 1


def test_expression_complexity_budget_is_explicit():
    with pytest.raises(oe.AnalysisError) as error:
        OLSDesign(["I(" + "+".join(["x"] * 100) + ")"], [], intercept=True)
    assert error.value.code == "invalid_formula"
