"""QA-only independent NumPy/SciPy replication of the two original papers.

Not imported by the runtime. SVD least squares and separately refitted MAIC
candidates deliberately differ from the shipped Torch nested-QR algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq


def regression(x, target):
    beta = lstsq(x, target, lapack_driver="gelsd")[0]
    residual = target - x.dot(beta)
    ssr = residual.dot(residual)
    covariance = (ssr / (len(target) - x.shape[1])) * np.linalg.pinv(x).dot(np.linalg.pinv(x).T)
    return beta, ssr, np.sqrt(np.diag(covariance))


def augmented(levels, order, start=None, cubic=False):
    start = order if start is None else start
    differences = np.diff(levels)
    rows = []
    for t in range(start, len(differences)):
        rows.append(
            [levels[t] ** (3 if cubic else 1)] + [differences[t - j] for j in range(1, order + 1)]
        )
    return np.asarray(rows), differences[start:]


def ng_reference(levels, trend, lags=None, maxlag=8):
    y = np.asarray(levels, dtype=float)
    horizon = len(y) - 1
    c = -7 if trend == "constant" else -13.5
    z = np.ones((len(y), 1))
    if trend == "trend":
        z = np.column_stack((z, np.arange(len(y))))
    alpha = 1 + c / horizon
    # Original observations and unscaled integer trend; no origin or unit shift.
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]
    zd = np.vstack((z[0], z[1:] - alpha * z[:-1]))
    theta = lstsq(zd, yd, lapack_driver="gelsd")[0]
    u = y - z.dot(theta)
    selection = []
    if lags is None:
        for k in range(maxlag + 1):
            x, target = augmented(u, k, maxlag)
            beta, ssr, _ = regression(x, target)
            variance = ssr / len(target)
            tau = beta[0] ** 2 * x[:, 0].dot(x[:, 0]) / variance
            selection.append(np.log(variance) + 2 * (tau + k) / len(target))
        lags = int(np.argmin(selection))
    x, target = augmented(u, lags)
    beta, ssr, _ = regression(x, target)
    sar = (ssr / len(target)) / (1 - sum(beta[1:])) ** 2
    integral = sum(value**2 for value in u[:-1]) / horizon**2
    end = u[-1] ** 2 / horizon
    mza = (end - sar) / (2 * integral)
    msb = np.sqrt(integral / sar)
    mpt = (c * c * integral + (-c if trend == "constant" else 1 - c) * end) / sar
    return np.array([mza, mza * msb, msb, mpt]), lags, selection


def kss_reference(levels, trend, lags):
    y = np.asarray(levels, dtype=float)
    if trend == "constant":
        y = y - y.mean()
    elif trend == "trend":
        z = np.column_stack((np.ones(len(y)), np.arange(len(y))))
        y = y - z.dot(lstsq(z, y, lapack_driver="gelsd")[0])
    x, target = augmented(y, lags, cubic=True)
    beta, _, se = regression(x, target)
    return beta[0] / se[0]
