"""Global replay GMM parity across partitions, weights and nonlinear moments."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset


@pytest.fixture(scope="module")
def data():
    gen = torch.Generator().manual_seed(621)
    x, z, q, u, v = torch.randn((5, 673), generator=gen, dtype=torch.float64)
    u *= 1+.4*z.abs()
    endog = .6*z+.3*q+.4*u+v
    f = pd.DataFrame({"y": (1+.7*x-.9*endog+u).numpy(), "x": x.numpy(), "endog": endog.numpy(),
                      "z": z.numpy(), "q": q.numpy(), "c": torch.arange(673).remainder(23).numpy(),
                      "w": torch.arange(673).remainder(3).add(1).numpy(), "time": torch.arange(673).numpy()})
    f["count"] = torch.poisson(torch.exp(.2+.3*x+.2*z), generator=gen).numpy()
    return f


def dataset(frame, rows):
    return Dataset.from_batches(lambda: (frame.iloc[i:i+rows] for i in range(0, len(frame), rows)),
                                list(frame.columns), row_count=len(frame))


def compare(a, b):
    assert a.nobs == b.nobs and a.dropped_rows == b.dropped_rows
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    torch.testing.assert_close(torch.tensor([c.estimate for c in a.coefficients], dtype=torch.float64),
                               torch.tensor([c.estimate for c in b.coefficients], dtype=torch.float64), rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(torch.tensor(a.covariance_matrix, dtype=torch.float64),
                               torch.tensor(b.covariance_matrix, dtype=torch.float64), rtol=2e-8, atol=2e-11)
    if a.metrics.get("j") is not None:
        assert a.metrics["j"] == pytest.approx(b.metrics["j"], rel=2e-7, abs=2e-8)
    assert b.provenance["streaming"]["dense_observation_matrix"] is False
    assert b.sample_positions == []


@pytest.mark.parametrize("rows", [31, 401])
@pytest.mark.parametrize("options", [
    {}, {"twostep": False}, {"igmm": True}, {"winitial": "unadjusted"},
    {"wmatrix": "unadjusted"}, {"covariance": "nonrobust"}, {"center": True},
    {"cluster": "c"}, {"cluster": "c", "twostep": False, "center": True},
    {"covariance": "hac", "lags": 3, "time": "time"},
    {"covariance": "hac", "lags": 2, "kernel": "parzen"},
    {"covariance": "hac", "lags": 2, "kernel": "quadratic_spectral", "center": True},
    {"weights": "w", "weight_type": "aweight"},
    {"weights": "w", "weight_type": "fweight"},
    {"weights": "w", "weight_type": "pweight"},
])
def test_global_linear_moments(data, rows, options):
    arguments = dict(moments=["y - {b0} - {b1}*x - {b2}*endog"], instruments=["x", "z", "q"], **options)
    compare(oe.gmm(data=data, **arguments), oe.gmm(data=dataset(data, rows), **arguments))


@pytest.mark.parametrize("shared", [False, True])
def test_nonlinear_and_shared_parameters(data, shared):
    arguments = dict(moments=["count - exp({a} + {b}*x + {c}*z)"], instruments=["x", "z", "q"])
    if shared:
        arguments["moments"].append("y - {d} - {b}*x - {e}*endog")
        arguments["instruments"] = [["x", "z", "q"], ["x", "z", "q"]]
    compare(oe.gmm(data=data, **arguments), oe.gmm(data=dataset(data, 97), **arguments))


def test_missing_rank_and_source_integrity(data):
    frame = data.assign(alias=data.z)
    frame.loc[[2, 503], "x"] = float("nan")
    arguments = dict(moments=["y - {a} - {b}*x - {d}*endog"], instruments=["x", "z", "q", "alias"], missing="drop")
    compare(oe.gmm(data=frame, **arguments), oe.gmm(data=dataset(frame, 121), **arguments))
    calls = 0
    def changed():
        nonlocal calls
        calls += 1
        yield frame.assign(z=frame.z+calls)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.gmm(data=Dataset.from_batches(changed, list(frame.columns)), **arguments)
    assert caught.value.code == "source_changed"


def test_perfect_fit_and_all_lag_hac(data):
    arguments = dict(moments=["y - {a} - {b}*x - {d}*endog"], instruments=["x", "z", "q"])
    exact = data.assign(y=1+2*data.x-3*data.endog)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.gmm(data=dataset(exact, 111), **arguments)
    assert caught.value.code == "perfect_fit"
    options = dict(covariance="hac", lags=2, kernel="quadratic_spectral", **arguments)
    compare(oe.gmm(data=data, **options), oe.gmm(data=dataset(data, 111), **options))
