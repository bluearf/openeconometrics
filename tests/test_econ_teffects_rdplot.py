"""NumPy oracle for oe.rdplot: bin counts, binned means and the global polynomial."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    n = 1500
    x = rng.uniform(-1, 2, n)
    y = 0.5 + x - 0.4 * x ** 2 + 0.8 * (x >= 0.2) + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"x": x, "y": y})


def oracle_bins(dx, y, selector):
    n = len(dx)
    order = np.argsort(dx, kind="stable")
    sx, sy = dx[order], y[order]
    gaps, jumps = np.diff(sx), np.diff(sy)
    beta = np.polynomial.polynomial.polyfit(dx, y, 4)
    deriv = np.polynomial.polynomial.polyder(beta)
    length = sx.max() if dx.min() >= 0 else -sx.min()
    if selector.startswith("es"):
        variance = np.sum(gaps * jumps ** 2) / (2 * length)
        bias = length ** 2 / (12 * n) * np.sum(np.polynomial.polynomial.polyval(dx, deriv) ** 2)
    else:
        variance = np.sum(jumps ** 2) / (2 * n)
        middle = (sx[1:] + sx[:-1]) / 2
        bias = n / 12 * np.sum(gaps ** 2 * np.polynomial.polynomial.polyval(middle, deriv) ** 2)
    if selector.endswith("mv"):
        return int(np.ceil(np.var(y, ddof=1) / variance * n / np.log(n) ** 2))
    return int(np.ceil((2 * bias * n / variance) ** (1 / 3)))


@pytest.mark.parametrize("selector", ["es", "esmv", "qs", "qsmv"])
def test_bin_selection_and_binned_means_match_oracle(data, selector):
    c = 0.2
    result = oe.rdplot(data=data, y="y", running="x", cutoff=c, binselect=selector)
    bins = result["bins"]
    for side, mask, key in (("left", data.x < c, "bins_left"), ("right", data.x >= c,
                                                                "bins_right")):
        dx, y = data.x[mask].to_numpy() - c, data.y[mask].to_numpy()
        assert result.attrs[key] == oracle_bins(dx, y, selector)
        rows = bins[bins.side == side]
        assert rows.n.sum() == mask.sum()
        for _, row in rows.iterrows():
            members = (data.x[mask] >= row.x_low) & (data.x[mask] < row.x_high)
            if row.x_high == rows.x_high.max():
                members |= data.x[mask] == row.x_high
            if side == "left" and row.x_low == rows.x_low.min():
                members |= data.x[mask] == row.x_low
            assert members.sum() == row.n
            assert_allclose(row.y_mean, data.y[mask][members].mean(), rtol=1e-10)
            assert_allclose(row.y_se, data.y[mask][members].std(ddof=1) / np.sqrt(row.n),
                            rtol=1e-8)
    if selector.startswith("qs"):
        counts = bins[bins.side == "right"].n
        assert counts.max() - counts.min() <= 2


def test_polynomial_fit_manual_bins_and_errors(data):
    result = oe.rdplot(data=data, y="y", running="x", cutoff=0.2, nbins=[10, 12], p=3, grid=50)
    assert result.attrs["bins_left"] == 10 and result.attrs["bins_right"] == 12
    assert result.attrs["binselect"] == "manual"
    poly = result["poly"]
    right = data[data.x >= 0.2]
    beta = np.polynomial.polynomial.polyfit(right.x - 0.2, right.y, 3)
    curve = poly[poly.side == "right"]
    assert len(curve) == 50
    assert_allclose(curve.fit, np.polynomial.polynomial.polyval(curve.x - 0.2, beta), rtol=1e-8)
    widths = result["bins"].query("side == 'left'").eval("x_high - x_low")
    assert_allclose(widths, widths.iloc[0], rtol=1e-9)
    assert "bins" in str(result) and "tabular" in result.to_latex()
    for kwargs, code in (({"binselect": "imse"}, "invalid_spec"), ({"nbins": 0}, "invalid_spec"),
                         ({"cutoff": 10.0}, "insufficient_observations"),
                         ({"p": 12}, "invalid_spec")):
        with pytest.raises(AnalysisError) as caught:
            oe.rdplot(data=data, y="y", running="x", **kwargs)
        assert caught.value.code == code, kwargs
    assert callable(oe.rdplot) and "rdplot" in dir(oe)
