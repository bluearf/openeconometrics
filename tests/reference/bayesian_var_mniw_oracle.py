"""Development-only NumPy/SciPy oracle, using augmented QR rather than native Gram factors."""

import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart


def posterior(levels, p, intercept, mean, row_scale, scale, nu):
    levels = np.asarray(levels, dtype=np.float64)
    n, m = levels.shape
    y = levels[p:]
    x = np.column_stack(
        ([np.ones(n - p)] if intercept else [])
        + [levels[p - lag : n - lag] for lag in range(1, p + 1)]
    )
    whiten = np.linalg.inv(np.linalg.cholesky(row_scale))
    ax = np.vstack((x, whiten))
    ay = np.vstack((y, whiten @ mean))
    q, r = np.linalg.qr(ax, mode="reduced")
    location = np.linalg.solve(r, q.T @ ay)
    ri = np.linalg.inv(r)
    vn = ri @ ri.T
    residual = ay - ax @ location
    sn = scale + residual.T @ residual
    nun = nu + len(y)
    return x, y, location, vn, sn, nun


def matrix_t_logpdf(value, location, row_scale, scale, nu):
    value = np.asarray(value)
    q, m = value.shape
    delta = value - location
    updated = scale + delta.T @ np.linalg.solve(row_scale, delta)
    return (
        multigammaln((nu + q) / 2, m)
        - multigammaln(nu / 2, m)
        - q * m / 2 * np.log(np.pi)
        - m / 2 * np.linalg.slogdet(row_scale)[1]
        + nu / 2 * np.linalg.slogdet(scale)[1]
        - (nu + q) / 2 * np.linalg.slogdet(updated)[1]
    )


def joint_transform(location, vn, sn, bartlett, normal):
    # Direct dense inversion deliberately differs from triangular production solves.
    factor = np.linalg.cholesky(sn) @ np.linalg.inv(bartlett).T
    sigma = factor @ factor.T
    b = location + np.linalg.cholesky(vn) @ normal @ factor.T
    return b, sigma


def sigma_covariance_polarized(scale, nu):
    """Full vech covariance from projected SciPy marginal variances and polarization.

    This does not repeat the production four-index covariance formula.
    A rank-r projection of IW_m has df nu-m+r and the projected scale.
    """
    m = len(scale)

    def variance(a):
        return float(invwishart.var(nu - m + 1, [[a @ scale @ a]]))

    def covariance(a, b):
        if np.linalg.matrix_rank(np.stack((a, b))) == 1:
            multiplier = (a @ b) / (a @ a)
            return multiplier**2 * variance(a)
        projection = np.stack((a, b))
        off_diagonal_variance = invwishart.var(nu - m + 2, projection @ scale @ projection.T)[0, 1]
        return (
            variance(a + b)
            + variance(a - b)
            - 2 * variance(a)
            - 2 * variance(b)
            - 8 * off_diagonal_variance
        ) / 4

    eye = np.eye(m)
    components = []
    for j in range(m):
        for i in range(j, m):
            components.append(
                [(1.0, eye[i])]
                if i == j
                else [(0.5, eye[i] + eye[j]), (-0.5, eye[i]), (-0.5, eye[j])]
            )
    return np.array(
        [
            [
                sum(w * z * covariance(a, b) for w, a in first for z, b in second)
                for second in components
            ]
            for first in components
        ]
    )


def recursive_paths(levels, coefficient, sigma, normals, p, intercept):
    # Separate NumPy row recursion; one coefficient/Sigma draw across the whole horizon.
    histories = [list(np.asarray(levels)[-p:]), list(np.asarray(levels)[-p:])]
    out = [[], []]
    for z in normals:
        for index, history in enumerate(histories):
            x = np.concatenate(([np.ones(1)] if intercept else []) + history[-p:][::-1])
            y = x.dot(coefficient)
            if index:
                y = y + np.linalg.cholesky(sigma).dot(z)
            history.append(y)
            out[index].append(y)
    return tuple(np.asarray(values) for values in out)


def impulse(coefficient, sigma, m, p, intercept, h, orthogonalized):
    # Companion powers provide an independent check of the production lag recursion.
    lag = coefficient[int(intercept) :]
    companion = np.zeros((m * p, m * p))
    companion[:m] = np.concatenate([lag[j * m : (j + 1) * m].T for j in range(p)], axis=1)
    companion[m:, :-m] = np.eye(m * (p - 1))
    impact = np.linalg.cholesky(sigma) if orthogonalized else np.eye(m)
    return np.asarray([np.linalg.matrix_power(companion, j)[:m, :m] @ impact for j in range(h + 1)])
