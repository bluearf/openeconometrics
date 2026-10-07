"""Joint global factors preserve SUR, multivariate and IV system estimands."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset


@pytest.fixture
def data():
    generator = torch.Generator().manual_seed(937)
    x = torch.randn(811, 4, dtype=torch.float64, generator=generator)
    frame = pd.DataFrame(x.numpy(), columns=["x", "z", "v", "noise"])
    frame["y1"] = 1 + .6*frame.x + .3*frame.v + frame.noise
    frame["y2"] = 2 - .2*frame.z + .5*frame.x + .3*frame.noise + torch.randn(811, dtype=torch.float64, generator=generator).numpy()
    frame["cat"] = ["a", "b", "c"] * 270 + ["a"]
    frame["fw"] = [1, 2, 3] * 270 + [1]
    frame["aw"] = .2 + frame.x.abs()
    return frame


def source(frame, size):
    def factory():
        for start in range(0, len(frame), size):
            yield frame.iloc[start:start+size]
    return Dataset.from_batches(factory, list(frame.columns), row_count=len(frame))


def compare(dense, streamed):
    assert streamed.nobs == dense.nobs
    assert streamed.nobs_original == dense.nobs_original
    assert streamed.dropped_rows == dense.dropped_rows
    assert streamed.sample_positions == []
    assert len(streamed.predictions) <= 400
    assert streamed.provenance["streaming"]["dense_observation_matrix"] is False
    assert [c.term for c in streamed.coefficients] == [c.term for c in dense.coefficients]
    torch.testing.assert_close(torch.tensor([c.estimate for c in streamed.coefficients], dtype=torch.float64),
                               torch.tensor([c.estimate for c in dense.coefficients], dtype=torch.float64), rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(torch.tensor(streamed.covariance_matrix, dtype=torch.float64),
                               torch.tensor(dense.covariance_matrix, dtype=torch.float64), rtol=2e-10, atol=2e-12)
    assert streamed.metrics["df_resid"] == dense.metrics["df_resid"]
    for name, value in dense.metrics.items():
        if isinstance(value, (int, float)):
            assert streamed.metrics[name] == pytest.approx(value, rel=2e-10, abs=1e-10)
    reference = {row["row"]: row for row in dense.predictions}
    matched = 0
    for b in streamed.predictions:
        if b["row"] in reference:
            matched += 1
            assert reference[b["row"]]["fitted"] == pytest.approx(b["fitted"], rel=2e-10, abs=2e-11)
    assert matched > 100


@pytest.mark.parametrize("weights,weight_type", [(None, None), ("fw", "fweight"), ("aw", "aweight")])
@pytest.mark.parametrize("size", [37, 499])
def test_mvreg_all_rows_weights_categories_and_missing(data, weights, weight_type, size):
    data.loc[[2, 405], "x"] = float("nan")
    kwargs = dict(y=["y1", "y2"], x=["x", "z", "cat"], categorical=["cat"],
                  weights=weights, weight_type=weight_type, missing="drop", corr=True)
    compare(oe.mvreg(data=data, **kwargs), oe.mvreg(data=source(data, size), **kwargs))


@pytest.mark.parametrize("iterate", [False, True])
@pytest.mark.parametrize("small,dfk", [(False, False), (True, True)])
@pytest.mark.parametrize("size", [29, 507])
def test_sureg_global_covariance_and_constraints(data, iterate, small, dfk, size):
    kwargs = dict(equations=[dict(y="y1", x=["x", "v"]), dict(y="y2", x=["z", "x"])],
                  iterate=iterate, small=small, dfk=dfk,
                  constraints=[dict(terms={"y1:x": 1., "y2:x": -1.}, value=0.)])
    compare(oe.sureg(data=data, **kwargs), oe.sureg(data=source(data, size), **kwargs))


@pytest.mark.parametrize("method", ["3sls", "2sls", "sure", "ols"])
@pytest.mark.parametrize("size", [31, 500])
def test_reg3_structural_iv_projection(data, method, size):
    kwargs = dict(equations=[dict(y="y1", x=["x", "v"]), dict(y="y2", x=["z", "x"])], method=method)
    compare(oe.reg3(data=data, **kwargs), oe.reg3(data=source(data, size), **kwargs))


@pytest.mark.parametrize("ireg3", [False, True])
def test_reg3_actual_endogenous_outcome_and_excluded_instruments(data, ireg3):
    first = .6*data.x+.4*data.v+data.noise
    second = -.4*data.z+.3*data.noise+data.y2-2+.2*data.z-.5*data.x-.3*data.noise
    data = data.assign(y1=(first+.2*second)/.94)
    data["y2"] = second+.3*data.y1
    kwargs = dict(equations=[dict(y="y1", x=["y2", "x"]), dict(y="y2", x=["y1", "z"])],
                  instruments=["x", "z", "v"], ireg3=ireg3)
    compare(oe.reg3(data=data, **kwargs), oe.reg3(data=source(data, 47), **kwargs))


@pytest.mark.parametrize("constant", [False, True])
def test_system_factor_preserves_large_levels_and_origin(data, constant):
    data = data.assign(x=data.x*1e-3+10**5, z=data.z*1e3, y1=data.y1+10**7, y2=data.y2+10**6)
    kwargs = dict(y=["y1", "y2"], x=["x", "z"], intercept=constant)
    a, b = oe.mvreg(data=data, **kwargs), oe.mvreg(data=source(data, 101), **kwargs)
    torch.testing.assert_close(torch.tensor([c.estimate for c in a.coefficients], dtype=torch.float64),
                               torch.tensor([c.estimate for c in b.coefficients], dtype=torch.float64), rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(torch.tensor(a.covariance_matrix, dtype=torch.float64),
                               torch.tensor(b.covariance_matrix, dtype=torch.float64), rtol=2e-6, atol=2e-8)


def test_replay_changes_are_refused_and_unsupported_variant_is_not_read(data):
    calls = 0
    def changing():
        nonlocal calls
        calls += 1
        yield data.assign(x=data.x + calls)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.mvreg(data=Dataset.from_batches(changing, list(data.columns)), y=["y1", "y2"], x=["x"])
    assert caught.value.code == "source_changed"
    def unread():
        raise AssertionError("An unsupported Dataset variant must fail before reading rows.")
        yield data
    unsupported = Dataset.from_batches(unread, list(data.columns), row_count=len(data))
    with pytest.raises(oe.AnalysisError) as caught:
        oe.teffects(data=unsupported, y="y1", treatment="z", x=["x"], method="nnmatch", estimand="pomeans")
    assert caught.value.code == "unsupported_estimand"
