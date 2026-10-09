"""Independent positive-integral NumPy/SciPy multinomial-probit oracle.

Development only. No production parser, probability quadrature, optimizer or
autograd is imported. Three alternatives use globally normalized error
differences (epsilon_B-epsilon_A, epsilon_C-epsilon_A), with covariance
[[1, sd*rho], [sd*rho, sd**2]]. A two-coordinate second-order algebra carries
the comparison transformation into physical beta/sd/rho derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr

LOG2PI = math.log(2 * math.pi)


def jacobian(function, value, step=2e-5):
    """Independent five-point derivative in untransformed coordinates."""
    value = np.asarray(value, dtype=float)
    columns = []
    for j in range(len(value)):
        delta = np.zeros_like(value)
        delta[j] = step * max(1., abs(value[j]))
        columns.append((-np.asarray(function(value + 2*delta))
                        + 8*np.asarray(function(value + delta))
                        - 8*np.asarray(function(value - delta))
                        + np.asarray(function(value - 2*delta))) / (12*delta[j]))
    return np.stack(columns, axis=-1)


@dataclass
class Jet:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray

    @classmethod
    def constant(cls, value, k):
        return cls(float(value), np.zeros(k), np.zeros((k, k)))

    @classmethod
    def coordinate(cls, value, k, j):
        result = cls.constant(value, k)
        result.gradient[j] = 1.
        return result

    def coerce(self, other):
        return other if isinstance(other, Jet) else self.constant(other, len(self.gradient))

    def __add__(self, other):
        other = self.coerce(other)
        return Jet(self.value + other.value, self.gradient + other.gradient,
                   self.hessian + other.hessian)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        return Jet(self.value*other.value,
                   self.gradient*other.value + other.gradient*self.value,
                   self.hessian*other.value + other.hessian*self.value
                   + np.outer(self.gradient, other.gradient)
                   + np.outer(other.gradient, self.gradient))

    __rmul__ = __mul__

    def power(self, exponent):
        value = self.value**exponent
        first = exponent*self.value**(exponent-1)
        second = exponent*(exponent-1)*self.value**(exponent-2)
        return Jet(value, first*self.gradient,
                   first*self.hessian + second*np.outer(self.gradient, self.gradient))

    def __truediv__(self, other):
        return self*self.coerce(other).power(-1)

    def __rtruediv__(self, other):
        return self.coerce(other)*self.power(-1)


def bivariate_logcdf(a, b, rho):
    """Positive conditional Gaussian integral, adaptively scaled at its mode.

    This integrates a density times a conditional CDF; it never integrates
    Plackett's derivative over correlation or subtracts signed corrections.
    The symmetric orientation with the smaller upper bound avoids gratuitous
    integration through a nearly complete marginal probability.
    """
    if abs(rho) >= 1:
        raise ValueError("A nonsingular bivariate correlation is required.")
    if b < a:
        a, b = b, a
    if rho == 0:
        return float(log_ndtr(a) + log_ndtr(b))
    scale = math.sqrt((1-rho)*(1+rho))

    def logdensity(t):
        return -.5*t*t-.5*LOG2PI + float(log_ndtr((b-rho*t)/scale))

    lower = -max(30., abs(a)+abs(b)+12.)
    optimum = minimize_scalar(lambda t: -logdensity(t), bounds=(lower, a),
                              method="bounded", options={"xatol": 1e-12})
    mode = min(a, float(optimum.x))
    if logdensity(a) >= logdensity(mode):
        mode = a
    maximum = logdensity(mode)

    def integrand(t):
        return math.exp(logdensity(t)-maximum)

    # The split explicitly exposes a narrow endpoint peak to adaptive quad.
    width = max(1./max(1., abs(mode)), .01)
    split = min(a, mode+max(1., 10*width))
    first = quad(integrand, -np.inf, mode, epsabs=2e-13, epsrel=2e-13,
                 limit=250)[0]
    second = (quad(integrand, mode, split, epsabs=2e-13, epsrel=2e-13,
                   limit=250)[0] if split > mode else 0.)
    third = (quad(integrand, split, a, epsabs=2e-13, epsrel=2e-13,
                  limit=250)[0] if a > split else 0.)
    integral = first + second + third
    if not integral > 0:
        raise ArithmeticError("Independent positive Gaussian integration failed.")
    return maximum+math.log(integral)


def bivariate_log_derivatives(a, b, rho):
    """CDF log value, gradient and Hessian from analytic boundary densities."""
    logp = bivariate_logcdf(a, b, rho)
    s2 = (1-rho)*(1+rho)
    s = math.sqrt(s2)
    u, v = (b-rho*a)/s, (a-rho*b)/s
    logpa = -.5*a*a-.5*LOG2PI+float(log_ndtr(u))
    logpb = -.5*b*b-.5*LOG2PI+float(log_ndtr(v))
    logpr = -.5*a*a-.5*u*u-LOG2PI-math.log(s)
    da, db, dr = (math.exp(value-logp) for value in (logpa, logpb, logpr))
    gradient = np.array([da, db, dr])
    hessian_p_over_p = np.array([
        [-a*da-rho*dr, dr, dr*(rho*b-a)/s2],
        [dr, -b*db-rho*dr, dr*(rho*a-b)/s2],
        [dr*(rho*b-a)/s2, dr*(rho*a-b)/s2,
         dr*(rho/s2+(a*b*(1+rho*rho)-rho*(a*a+b*b))/s2**2)]])
    return logp, gradient, hessian_p_over_p-np.outer(gradient, gradient)


def difference_covariance(theta, q, alternatives=3, fixed=None):
    if alternatives == 2:
        return np.ones((1, 1))
    if fixed is not None:
        return np.asarray(fixed, dtype=float)
    sd, rho = np.asarray(theta)[q:q+2]
    return np.array([[1., sd*rho], [sd*rho, sd*sd]])


def choice_log_derivatives(x, theta, chosen, *, alternatives=3, fixed=None):
    """One available risk set: log probability and full physical derivatives.

    x contains available rows with global integer alternative indices in its
    first column and numeric utility attributes in all following columns.
    The chosen argument is the global integer alternative identity.
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    q, k = x.shape[1]-1, len(theta)
    variables = [Jet.coordinate(value, k, j) for j, value in enumerate(theta)]
    if alternatives == 2:
        sigma = [[Jet.constant(1., k)]]
        embedding = np.array([[0.], [1.]])
    else:
        embedding = np.array([[0., 0.], [1., 0.], [0., 1.]])
        if fixed is None:
            sd, rho = variables[q:q+2]
            sigma = [[Jet.constant(1., k), sd*rho], [sd*rho, sd*sd]]
        else:
            sigma = [[Jet.constant(value, k) for value in row] for row in fixed]
    global_ids = x[:, 0].astype(int)
    selected = int(np.flatnonzero(global_ids == chosen)[0])
    other = [j for j in range(len(x)) if j != selected]
    comparison = embedding[global_ids[other]]-embedding[chosen]
    covariance = []
    for left in comparison:
        covariance.append([sum((left[i]*sigma[i][j]*right[j]
                                for i in range(len(left)) for j in range(len(right))),
                               Jet.constant(0., k)) for right in comparison])
    differences = [sum(((x[selected, j+1]-x[rival, j+1])*variables[j]
                        for j in range(q)), Jet.constant(0., k)) for rival in other]
    cutoffs = [difference/covariance[j][j].power(.5)
               for j, difference in enumerate(differences)]
    if len(other) == 1:
        a = cutoffs[0]
        logp = float(log_ndtr(a.value))
        derivative = math.exp(-.5*a.value*a.value-.5*LOG2PI-logp)
        second = -a.value*derivative-derivative*derivative
        return (logp, derivative*a.gradient,
                derivative*a.hessian + second*np.outer(a.gradient, a.gradient))
    correlation = covariance[0][1]/(covariance[0][0]*covariance[1][1]).power(.5)
    transformed = [*cutoffs, correlation]
    logp, gradient, hessian = bivariate_log_derivatives(
        *[item.value for item in transformed])
    first = np.stack([item.gradient for item in transformed])
    second = np.stack([item.hessian for item in transformed])
    return logp, gradient@first, first.T@hessian@first + np.einsum("a,aij->ij", gradient, second)


def choice_effect(x, theta, outcome, changed, attribute, *, alternatives=3,
                  fixed=None, elasticity=False):
    """Analytic raw-cell derivative through independent Gaussian boundaries."""
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    q = x.shape[1]-1
    ids = x[:, 0].astype(int)
    selected = int(np.flatnonzero(ids == outcome)[0])
    changed_row = int(np.flatnonzero(ids == changed)[0])
    other = [j for j in range(len(x)) if j != selected]
    embedding = np.array([[0.], [1.]]) if alternatives == 2 else np.array([[0., 0.], [1., 0.], [0., 1.]])
    comparison = embedding[ids[other]]-embedding[outcome]
    covariance = comparison@difference_covariance(theta, q, alternatives, fixed)@comparison.T
    deviations = np.sqrt(np.diag(covariance))
    cutoffs = (x[selected, 1:]-x[other, 1:])@theta[:q]/deviations
    if len(other) == 1:
        logp = float(log_ndtr(cutoffs[0]))
        boundary = np.array([math.exp(-.5*cutoffs[0]**2-.5*LOG2PI-logp)])
    else:
        rho = covariance[0, 1]/np.prod(deviations)
        logp, derivatives, _ = bivariate_log_derivatives(*cutoffs, rho)
        boundary = derivatives[:2]
    direction = np.ones(len(other)) if selected == changed_row else -np.array([int(j == changed_row) for j in other])
    log_effect = float(theta[attribute]*np.sum(boundary*direction/deviations))
    return log_effect*x[changed_row, attribute+1] if elasticity else math.exp(logp)*log_effect


def fixture(seed=699_706, cases=256, *, alternatives=3, dependence=False,
            unbalanced=True, positive=False):
    """Uncensored Gaussian random-utility choices with repeated respondents."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    truth = np.array([.55, -.4] + ([1.25, .2] if alternatives == 3 else []))
    labels = ["A", "B", "C"][:alternatives]
    attributes = rng.uniform(.2, 2., (cases, alternatives, 2)) if positive else rng.normal(size=(cases, alternatives, 2))
    sigma = difference_covariance(truth, 2, alternatives)
    innovations = rng.normal(size=(cases, alternatives-1))
    if dependence:
        groups = (cases+3)//4
        innovations = (math.sqrt(.35)*rng.normal(size=(groups, alternatives-1))[np.arange(cases)//4]
                       + math.sqrt(.65)*innovations)
    errors = np.column_stack([np.zeros(cases), innovations@np.linalg.cholesky(sigma).T])
    utilities = attributes@truth[:2]+errors
    rows = []
    patterns = [[0, 1, 2], [0, 1], [1, 2], [0, 2]]
    for case in range(cases):
        available = (patterns[case % 10-6] if alternatives == 3 and unbalanced and case % 10 >= 6
                     else list(range(alternatives)))
        chosen = available[int(np.argmax(utilities[case, available]))]
        for alternative in range(alternatives):
            rows.append(dict(case=case, alternative=labels[alternative],
                             chosen=int(alternative == chosen), available=int(alternative in available),
                             x=float(attributes[case, alternative, 0]),
                             z=float(attributes[case, alternative, 1]), cluster=case//4))
    return pd.DataFrame(rows), truth


@dataclass
class Reference:
    data: object
    attributes: tuple = ("x", "z")
    catalogue: tuple = ("A", "B", "C")
    fixed: object = None

    def __post_init__(self):
        self.frame = self.data.copy()
        self.q = len(self.attributes)
        self.k = self.q + (2 if len(self.catalogue) == 3 and self.fixed is None else 0)
        self.case_ids = list(dict.fromkeys(self.frame.case.tolist()))
        self.cases, self.selected, self.cluster_labels = [], [], []
        for identity in self.case_ids:
            rows = self.frame.loc[(self.frame.case == identity) & self.frame.available.astype(bool)]
            self.cases.append(np.column_stack([
                [self.catalogue.index(value) for value in rows.alternative],
                rows[list(self.attributes)].to_numpy(dtype=float)]))
            choices = rows.loc[rows.chosen.astype(bool), "alternative"].tolist() if "chosen" in rows else []
            self.selected.append(self.catalogue.index(choices[0]) if choices else None)
            self.cluster_labels.append(rows.cluster.iloc[0] if "cluster" in rows else identity)
        labels = list(dict.fromkeys(self.cluster_labels))
        self.groups = [np.flatnonzero(np.asarray(self.cluster_labels) == label) for label in labels]

    def choice(self, theta, case, alternative):
        return choice_log_derivatives(self.cases[case], theta, alternative,
                                      alternatives=len(self.catalogue), fixed=self.fixed)

    def probabilities(self, theta):
        return np.array([math.exp(self.choice(theta, case, int(row[0]))[0])
                         for case, riskset in enumerate(self.cases) for row in riskset])

    def moments(self, theta):
        values = [self.choice(theta, case, selected)
                  for case, selected in enumerate(self.selected)]
        scores = np.stack([value[1] for value in values])
        return dict(case_loglikelihood=np.array([value[0] for value in values]),
                    case_scores=scores, information=-np.sum([value[2] for value in values], axis=0),
                    cluster_scores=np.stack([scores[group].sum(0) for group in self.groups]))

    def covariance(self, theta, vce="oim"):
        result = self.moments(theta)
        result["bread"] = np.linalg.inv(result["information"])
        scores = result["cluster_scores"] if vce == "cr0" else result["case_scores"]
        # Native saved 'meat' always audits score outer products. OIM uses
        # inverse information directly; it does not substitute sample meat.
        result["meat"] = scores.T@scores
        result["covariance"] = (result["bread"] if vce == "oim" else
                                result["bread"]@result["meat"]@result["bread"])
        return result

    def fit(self, starts=None):
        starts = starts or ([np.r_[np.zeros(self.q), 1., 0.], np.r_[np.full(self.q, .15), 1.3, .25],
                             np.r_[np.full(self.q, -.15), .8, -.25]] if self.k > self.q
                            else [np.zeros(self.q), np.full(self.q, .2), np.full(self.q, -.2)])

        def objective(theta):
            values = [self.choice(theta, case, selected)
                      for case, selected in enumerate(self.selected)]
            return -sum(value[0] for value in values), -np.sum([value[1] for value in values], axis=0)

        bounds = [(None, None)]*self.q + ([(.050001, 19.999), (-.979999, .979999)] if self.k > self.q else [])
        results = [minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-8}) for start in starts]
        best = min(results, key=lambda result: result.fun)
        return best.x, best, results

    def effect(self, theta, case, outcome, changed, attribute, *, elasticity=False):
        return choice_effect(self.cases[case], theta, outcome, changed,
                             self.attributes.index(attribute), alternatives=len(self.catalogue),
                             fixed=self.fixed, elasticity=elasticity)
