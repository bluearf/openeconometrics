"""Independent scalar Gaussian ARCH recurrences and full parameter derivatives."""
import math

import numpy as np
import pytest
import torch

from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.streaming_arch_kernel import StreamingArchLikelihood


def oracle(theta, x, y, z, layout):
    ix, n = layout.index, len(y)
    # Conditional ARMA innovations for the common presample variance, without
    # the ARCH-in-mean term; every presample disturbance/innovation is zero.
    u = y-x@theta[ix.x]
    innovations = np.zeros(n)
    for t in range(n):
        e = u[t]
        for j, lag in zip(ix.ar, layout.ar_lags, strict=True):
            e -= theta[j]*(u[t-lag] if t>=lag else 0.)
        for j, lag in zip(ix.ma, layout.ma_lags, strict=True):
            e -= theta[j]*(innovations[t-lag] if t>=lag else 0.)
        innovations[t] = e
    h0 = np.mean(innovations**2)
    power = theta[ix.p[0]] if ix.p else 2.
    if layout.kind=="egarch":
        pre_news, pre_news2, pre_v = 0., 0., math.log(h0)
    elif layout.kind=="parch":
        pre_news, pre_news2, pre_v = h0**(.5*power), 0., h0**(.5*power)
    else:
        pre_news, pre_news2, pre_v = h0, .5*h0, h0
    news, news2, states, residuals, disturbances, variance = [], [], [], [], [], []
    def lagged(values, lag, fill, t):
        return values[t-lag] if t>=lag else fill
    for t in range(n):
        v = theta[ix.c]+z[t]@theta[ix.z]
        if ix.z and layout.kind!="egarch":
            v = np.exp(v)
        for a, lag in zip(ix.a, layout.arch_lags, strict=True):
            v += theta[a]*lagged(news, lag, pre_news, t)
        for a, lag in zip(ix.g, layout.arch_lags if ix.g else [], strict=True):
            v += theta[a]*lagged(news2, lag, pre_news2, t)
        for b, lag in zip(ix.b, layout.garch_lags, strict=True):
            v += theta[b]*lagged(states, lag, pre_v, t)
        h = np.exp(v) if layout.kind=="egarch" else v**(2/power) if layout.kind=="parch" else v
        disturbance = u[t]
        if ix.m:
            feature = h if layout.archm=="variance" else np.sqrt(h) if layout.archm=="sd" else np.log(h)
            disturbance -= theta[ix.m[0]]*feature
        e = disturbance
        for a, lag in zip(ix.ar, layout.ar_lags, strict=True):
            e -= theta[a]*lagged(disturbances, lag, 0., t)
        for a, lag in zip(ix.ma, layout.ma_lags, strict=True):
            e -= theta[a]*lagged(residuals, lag, 0., t)
        if layout.kind=="egarch":
            news.append(e/np.sqrt(h))
            news2.append(abs(e/np.sqrt(h))-np.sqrt(2/np.pi))
        elif layout.kind=="parch":
            news.append(abs(e)**power)
            news2.append(0.)
        else:
            news.append(e*e)
            news2.append(e*e if e<0 else 0.)
        states.append(v)
        residuals.append(e)
        disturbances.append(disturbance)
        variance.append(h)
    e, h = np.array(residuals), np.array(variance)
    value = -.5*np.sum(np.log(2*np.pi)+np.log(h)+e**2/h)
    return value, e, h, h0


@pytest.mark.parametrize("kind,archm", [("garch", "log"), ("gjr", "variance"), ("egarch", "sd"), ("parch", "sd")])
@pytest.mark.parametrize("block", [7, 51])
def test_independent_scalar_recurrence_and_numeric_tangent_oracle(kind, archm, block):
    rng = np.random.default_rng(73221)
    n = 137
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    y, z = rng.normal(size=n), rng.normal(size=(n, 1))*.2
    layout = Layout(kind, "normal", (1, 3), (1, 2), (1, 3), (2,), archm, 2, 1)
    ix = layout.index
    theta = np.zeros(ix.k)
    theta[ix.x], theta[ix.z], theta[ix.ar], theta[ix.ma] = [.1, -.12], [.13], [.14, -.05], [.1]
    theta[ix.a], theta[ix.b], theta[ix.c] = [.04, .025], [.3, .2], math.log(.4)
    if kind=="egarch":
        theta[ix.a], theta[ix.g], theta[ix.c] = [.02, -.01], [.05, .03], -.1
    elif ix.g:
        theta[ix.g] = [.03, .02]
    if ix.m:
        theta[ix.m] = .07
    if ix.p:
        theta[ix.p] = 1.7
    tx, ty, tz = (torch.tensor(v, dtype=torch.float64) for v in [x, y, z])
    like = StreamingArchLikelihood(lambda: ((tx[j:j+block], ty[j:j+block], tz[j:j+block]) for j in range(0, n, block)), n, layout, budget_bytes=128*1024**2)
    try:
        actual = like.evaluate(torch.tensor(theta), scores=True)
        expected, e, h, h0 = oracle(theta, x, y, z, layout)
        assert actual is not None
        assert actual.value == pytest.approx(expected, abs=1e-10)
        assert actual.presample_variance == pytest.approx(h0, abs=1e-13)
        np.testing.assert_allclose(actual.residual, e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.variance, h, rtol=1e-12, atol=1e-12)
        step = 1e-6
        gradient = np.array([(oracle(theta+step*np.eye(ix.k)[j], x, y, z, layout)[0]-oracle(theta-step*np.eye(ix.k)[j], x, y, z, layout)[0])/(2*step) for j in range(ix.k)])
        np.testing.assert_allclose(actual.gradient, gradient, rtol=3e-6, atol=3e-7)
    finally:
        like.close()
