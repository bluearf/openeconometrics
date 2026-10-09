"""Independent NumPy/SciPy reference for Gaussian nonlinear SUR.

This development-only oracle hand-codes its fixture means. It does not import
the production formula parser, optimizer, derivatives or postestimation helpers.
Physical coordinates are beta followed by row-wise lower-triangle Sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, root


def lower_pairs(equations):
    return [(i, j) for i in range(equations) for j in range(i + 1)]


def jacobian(function, value, step=1e-5):
    """Five-point physical-coordinate derivative, independent of autograd."""
    value = np.asarray(value, dtype=float)
    result = []
    for j in range(len(value)):
        delta = np.zeros_like(value)
        delta[j] = step * max(1., abs(value[j]))
        result.append((-np.asarray(function(value + 2*delta))
                       + 8*np.asarray(function(value + delta))
                       - 8*np.asarray(function(value - delta))
                       + np.asarray(function(value - 2*delta))) / (12*delta[j]))
    return np.stack(result, axis=-1)


def complex_jacobian(function, value):
    value = np.asarray(value, dtype=float)
    columns = []
    for j in range(len(value)):
        shifted = value.astype(complex)
        shifted[j] += 1e-25j
        columns.append(np.asarray(function(shifted)).imag / 1e-25)
    return np.stack(columns, axis=-1)


def fixture(seed=672_679, rows=320, *, mode="quadratic", equations=2,
            dependence=False, heteroskedastic=False):
    """Gaussian-error systems with explicit random design and uncensored draws.

    In the heteroskedastic iid mode, errors are conditionally Gaussian and
    E(scale)=1 under the uniform design distribution. Realized scales are not
    normalized: the full joint HC0 score variance includes random-design noise.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    low, high = (-1.5, 1.5) if mode == "quadratic" else (.1, 2.)
    frame = pd.DataFrame({"x": rng.uniform(low, high, rows),
                          "z": rng.uniform(low, high, rows),
                          "w": rng.uniform(low, high, rows),
                          "cluster": np.arange(rows)//4})
    beta = np.array([2., 1.1, 1., 2.5] + ([1.7] if equations == 3 else []))
    if mode == "quadratic":
        beta = np.array([2., .3, 1.2, .3])
    elif mode == "linear_shared":
        beta = np.array([2., 1.1, 2.5, .4])
    elif mode == "linear_separate":
        beta = np.array([2., 1.1, 2.5, .4])
    sigma = (np.array([[.49, .2, .08], [.2, .81, -.06], [.08, -.06, .6]]) if mode == "quadratic"
             else np.array([[.09, .035, .015], [.035, .12, -.01], [.015, -.01, .075]]))[:equations, :equations]
    reference = Reference(frame, mode=mode, equations=equations)
    innovations = rng.normal(size=(rows, equations))
    if dependence:
        groups = (rows + 3)//4
        innovations = np.sqrt(.4)*rng.normal(size=(groups, equations))[np.arange(rows)//4] + np.sqrt(.6)*innovations
    scale = np.ones(rows)
    if heteroskedastic:
        scale = .4 + 1.2*(frame.x.to_numpy()-low)/(high-low)
    errors = (innovations @ np.linalg.cholesky(sigma).T)*np.sqrt(scale[:, None])
    values = reference.means(beta) + errors
    for j in range(equations):
        frame[f"y{j+1}"] = values[:, j]
    return frame, np.r_[beta, [sigma[i, j] for i, j in lower_pairs(equations)]]


@dataclass
class Reference:
    data: object
    mode: str = "quadratic"
    equations: int = 2

    def __post_init__(self):
        self.frame = self.data.reset_index(drop=True).copy()
        self.pairs = lower_pairs(self.equations)
        self.q = 4 + (self.equations == 3) if self.mode == "nonlinear" else 4
        self.x = self.frame[["x", "z", "w"]].to_numpy(dtype=float)
        self.y = (self.frame[[f"y{i+1}" for i in range(self.equations)]].to_numpy(dtype=float)
                  if "y1" in self.frame else None)
        self.labels = list(dict.fromkeys(self.frame.cluster.tolist())) if "cluster" in self.frame else []
        self.groups = [np.flatnonzero(self.frame.cluster.to_numpy() == label) for label in self.labels]
        self.basis = []
        for i, j in self.pairs:
            matrix = np.zeros((self.equations, self.equations))
            matrix[i, j] = matrix[j, i] = 1.
            self.basis.append(matrix)

    @property
    def beta_names(self):
        if self.mode == "quadratic":
            return ["b0", "b1", "amp", "rate"]
        if self.mode == "nonlinear":
            return ["a1", "b", "d", "a2"] + (["a3"] if self.equations == 3 else [])
        return ["a1", "b", "a2", "c"] if self.mode == "linear_shared" else ["a1", "b1", "a2", "b2"]

    @property
    def definitions(self):
        if self.mode == "quadratic":
            return [{"y": f"y{i+1}", "name": f"eq{i+1}",
                     "formula": "{b0}+{b1}*" + ("x", "z", "w")[i] + "+{amp}*exp(" +
                                ("{rate}", "-{rate}", "2*{rate}")[i] + ")*" + ("x", "z", "w")[i] + "**2"}
                    for i in range(self.equations)]
        if self.mode == "nonlinear":
            return [{"y": f"y{i+1}", "name": f"eq{i+1}",
                     "formula": "{" + f"a{i+1}" + "}+{b}*exp({d}*" + ("x", "z", "w")[i] + ")"}
                    for i in range(self.equations)]
        if self.mode == "linear_shared":
            formulas = ["{a1}+{b}*x", "{a2}+{b}*z+{c}*w"]
        else:
            formulas = ["{a1}+{b1}*x", "{a2}+{b2}*x"]
        return [{"y": f"y{i+1}", "name": f"eq{i+1}", "formula": formula}
                for i, formula in enumerate(formulas)]

    @property
    def start(self):
        if self.mode == "quadratic":
            return dict(zip(self.beta_names, [1.8, .2, 1., .2]))
        values = [1.8, 1., .8, 2.3] + ([1.5] if self.equations == 3 else []) if self.mode == "nonlinear" else [1.8, 1., 2.3, .3]
        return dict(zip(self.beta_names, values))

    def means(self, beta):
        beta = np.asarray(beta)
        if self.mode == "quadratic":
            return np.column_stack([beta[0]+beta[1]*self.x[:, j]+beta[2]*np.exp(sign*beta[3])*self.x[:, j]**2
                                    for j, sign in enumerate((1., -1., 2.)[:self.equations])])
        if self.mode == "nonlinear":
            columns = [beta[0] + beta[1]*np.exp(beta[2]*self.x[:, 0]),
                       beta[3] + beta[1]*np.exp(beta[2]*self.x[:, 1])]
            if self.equations == 3:
                columns.append(beta[4] + beta[1]*np.exp(beta[2]*self.x[:, 2]))
            return np.column_stack(columns)
        if self.mode == "linear_shared":
            return np.column_stack([beta[0]+beta[1]*self.x[:, 0],
                                    beta[2]+beta[1]*self.x[:, 1]+beta[3]*self.x[:, 2]])
        return np.column_stack([beta[0]+beta[1]*self.x[:, 0], beta[2]+beta[3]*self.x[:, 0]])

    def derivatives(self, beta):
        beta = np.asarray(beta)
        n, m, p = len(self.frame), self.equations, self.q
        first = np.zeros((n, m, p))
        second = np.zeros((n, m, p, p))
        if self.mode == "quadratic":
            for j, sign in enumerate((1., -1., 2.)[:m]):
                column = self.x[:, j]
                curvature = np.exp(sign*beta[3])*column**2
                first[:, j, 0], first[:, j, 1] = 1., column
                first[:, j, 2], first[:, j, 3] = curvature, sign*beta[2]*curvature
                second[:, j, 2, 3] = second[:, j, 3, 2] = sign*curvature
                second[:, j, 3, 3] = sign**2*beta[2]*curvature
        elif self.mode == "nonlinear":
            for j in range(m):
                column = self.x[:, j]
                exponential = np.exp(beta[2]*column)
                first[:, j, 0 if j == 0 else 3 if j == 1 else 4] = 1.
                first[:, j, 1] = exponential
                first[:, j, 2] = beta[1]*column*exponential
                second[:, j, 1, 2] = second[:, j, 2, 1] = column*exponential
                second[:, j, 2, 2] = beta[1]*column**2*exponential
        elif self.mode == "linear_shared":
            first[:, 0, 0], first[:, 0, 1] = 1., self.x[:, 0]
            first[:, 1, 2], first[:, 1, 1], first[:, 1, 3] = 1., self.x[:, 1], self.x[:, 2]
        else:
            first[:, 0, 0], first[:, 0, 1] = 1., self.x[:, 0]
            first[:, 1, 2], first[:, 1, 3] = 1., self.x[:, 0]
        return first, second

    def sigma(self, theta):
        theta = np.asarray(theta)
        result = np.zeros((self.equations, self.equations), dtype=theta.dtype)
        for value, (i, j) in zip(theta[self.q:], self.pairs):
            result[i, j] = result[j, i] = value
        return result

    def profile(self, beta):
        residual = self.y - self.means(beta)
        sigma = residual.T @ residual / len(residual)
        sign, determinant = np.linalg.slogdet(sigma)
        if sign <= 0 or not np.isfinite(determinant):
            return np.inf, np.full(self.q, np.nan)
        inverse = np.linalg.inv(sigma)
        first, _ = self.derivatives(beta)
        score = np.einsum("nmp,nm->p", first, residual @ inverse)
        loss = .5*len(residual)*(self.equations*(1.+np.log(2*np.pi)) + determinant)
        return loss, -score

    def fit(self, starts=None):
        if starts is None:
            base = np.array(list(self.start.values()))
            starts = [base]
            if self.mode == "quadratic":
                starts.extend([base*np.array([1., 1., .8, .5]), base*np.array([1., 1., 1.2, 1.5])])
            elif self.mode == "nonlinear":
                starts.extend([base*np.array([1., .8, .6, 1.] + ([1.] if self.equations == 3 else [])),
                               base*np.array([1., 1.2, 1.6, 1.] + ([1.] if self.equations == 3 else []))])
        results = [minimize(self.profile, start, method="BFGS", jac=True,
                            options={"gtol": 1e-9, "maxiter": 1000}) for start in starts]
        best = min(results, key=lambda result: result.fun)
        refined = root(lambda beta: self.profile(beta)[1], best.x, method="hybr", options={"xtol": 1e-10})
        if np.isfinite(refined.x).all() and np.linalg.norm(self.profile(refined.x)[1], ord=np.inf) < 1e-6:
            best.x, best.fun = refined.x, self.profile(refined.x)[0]
        residual = self.y - self.means(best.x)
        sigma = residual.T @ residual / len(residual)
        return np.r_[best.x, [sigma[i, j] for i, j in self.pairs]], best, results

    def moments(self, theta):
        theta = np.asarray(theta, dtype=float)
        sigma = self.sigma(theta)
        inverse = np.linalg.inv(sigma)
        residual = self.y - self.means(theta[:self.q])
        u = residual @ inverse
        first, second = self.derivatives(theta[:self.q])
        beta_scores = np.einsum("nmp,nm->np", first, u)
        sigma_scores = np.column_stack([.5*(np.einsum("ni,ij,nj->n", u, basis, u)-np.trace(inverse @ basis))
                                        for basis in self.basis])
        scores = np.column_stack([beta_scores, sigma_scores])
        information = np.zeros((len(theta), len(theta)))
        information[:self.q, :self.q] = (np.einsum("nmp,mk,nkq->pq", first, inverse, first)
                                       - np.einsum("nm,nmpq->pq", u, second))
        for a, basis in enumerate(self.basis, self.q):
            cross = np.einsum("nmp,mk,kl,nl->p", first, inverse, basis, u)
            information[:self.q, a] = information[a, :self.q] = cross
        for a, left in enumerate(self.basis, self.q):
            for b, right in enumerate(self.basis, self.q):
                information[a, b] = .5*(np.einsum("ni,ij,nj->", u, right @ inverse @ left + left @ inverse @ right, u)
                                        - len(residual)*np.trace(inverse @ right @ inverse @ left))
        loglikelihood = -.5*(self.equations*np.log(2*np.pi) + np.linalg.slogdet(sigma)[1]
                             + np.einsum("ni,ij,nj->n", residual, inverse, residual))
        cluster_scores = np.array([scores[group].sum(axis=0) for group in self.groups])
        return dict(information=information, row_scores=scores, cluster_scores=cluster_scores,
                    row_loglikelihood=loglikelihood, residuals=residual, sigma=sigma)

    def covariance(self, theta, vce="oim"):
        result = self.moments(theta)
        bread = np.linalg.inv(result["information"])
        scores = result["cluster_scores"] if vce == "cr0" else result["row_scores"]
        meat = scores.T @ scores
        result.update(bread=bread, meat=meat, covariance=bread if vce == "oim" else bread @ meat @ bread)
        return result

    def predictions(self, theta, given=()):
        """Mean and lower-triangle conditional residual covariance, row major."""
        theta = np.asarray(theta)
        sigma = self.sigma(theta)
        given = [int(name.removeprefix("eq"))-1 if isinstance(name, str) else int(name) for name in given]
        remaining = [j for j in range(self.equations) if j not in given]
        mean = self.means(theta[:self.q])
        conditional_sigma = sigma[np.ix_(remaining, remaining)]
        predictions = mean[:, remaining]
        if given:
            gain = sigma[np.ix_(remaining, given)] @ np.linalg.inv(sigma[np.ix_(given, given)])
            predictions = predictions + (self.y[:, given]-mean[:, given]) @ gain.T
            conditional_sigma = conditional_sigma - gain @ sigma[np.ix_(given, remaining)]
        values = np.r_[predictions.ravel(), [conditional_sigma[i, j] for i, j in lower_pairs(len(remaining))]]
        return values, predictions, conditional_sigma

    def effects(self, theta, attributes, *, given=(), scale="effect", weights=None):
        theta = np.asarray(theta)
        sigma = self.sigma(theta)
        given = [int(name.removeprefix("eq"))-1 if isinstance(name, str) else int(name) for name in given]
        remaining = [j for j in range(self.equations) if j not in given]
        gradient = np.zeros((len(self.frame), self.equations, len(attributes)), dtype=theta.dtype)
        for ai, attribute in enumerate(attributes):
            column = ("x", "z", "w").index(attribute)
            if self.mode == "quadratic" and column < self.equations:
                sign = (1., -1., 2.)[column]
                gradient[:, column, ai] = theta[1]+2*theta[2]*np.exp(sign*theta[3])*self.x[:, column]
            elif self.mode == "nonlinear" and column < self.equations:
                gradient[:, column, ai] = theta[1]*theta[2]*np.exp(theta[2]*self.x[:, column])
            elif self.mode == "linear_shared":
                equation = 0 if column == 0 else 1
                gradient[:, equation, ai] = theta[1] if column < 2 else theta[3]
            elif self.mode == "linear_separate" and column == 0:
                gradient[:, 0, ai], gradient[:, 1, ai] = theta[1], theta[3]
        effects = gradient[:, remaining].copy()
        if given:
            gain = sigma[np.ix_(remaining, given)] @ np.linalg.inv(sigma[np.ix_(given, given)])
            effects -= np.einsum("mg,nga->nma", gain, gradient[:, given])
        if scale == "elasticity":
            mean = self.predictions(theta, given)[1]
            columns = self.frame[list(attributes)].to_numpy(dtype=float)
            effects *= columns[:, None, :] / mean[:, :, None]
        weights = np.ones(len(self.frame)) if weights is None else np.asarray(weights, dtype=float)
        averages = np.einsum("n,nma->ma", weights/weights.sum(), effects)
        return np.r_[effects.ravel(), averages.ravel()], effects, averages
