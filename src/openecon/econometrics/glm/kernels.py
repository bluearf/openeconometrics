"""Likelihood kernels of the glm family on float64 tensors (no autograd, O(n) memory).

GLM objective
-------------
With prior weights ``w_i`` (user weights times binomial trials), unit
dispersion and the deviance ``D(b) = sum_i w_i d(y_i, mu_i)``, the objective
maximized is ``-D/2``, which equals the quasi-log-likelihood up to a constant.
Per observation, with ``r_i = (y_i - mu_i)/V(mu_i)``, ``m1 = dmu/deta``,
``q = m2/m1`` and ``V' = dV/dmu``:

    score factor      s_i = r_i m1               (gradient = X' (w s))
    observed weight   h_i = -m1^2/V + s_i (q - m1 V'/V)   (Hessian = X' diag(w h) X)
    expected weight   f_i = m1^2 / V              (Fisher information = X' diag(w f) X)

so a canonical link (``q = m1 V'/V``) has ``h = -f``. For binomial outcomes
the same quantities are written with the tail-stable ratios ``A = m1/mu`` and
``B = m1/(1-mu)`` of the link: ``s = y A - (1-y) B``, ``f = A B`` and
``h = -A B + s (q - (A - B))``, which stay finite where ``mu`` rounds to 0 or 1.

IRLS (Fisher scoring) solves the weighted least squares ``X b = z`` with
weights ``w f`` and working response ``z = eta - offset + s/f`` by QR and
halves the step while the deviance rises. It stops when the relative change
in the deviance is at most ``tol`` (Stata's ``irls`` rule) AND the remaining
Fisher step is negligible: scaled gradient ``g' I^-1 g <= tol`` and relative
step ``max_j |d_j| / max(1, |b_j|) <= tol`` (or the step has stopped
contracting below ``sqrt(tol)``: the rounding floor). The extra conditions
matter for non-canonical links, whose Fisher scoring converges only linearly
and whose Pearson statistic is not flat at the optimum.
Newton-Raphson with the observed Hessian is left to
``engines.optimize.maximize_newton`` (Stata's default ``ml``).

Negative binomial (``nbreg``)
-----------------------------
NB2, ``Var = mu + a mu^2``, parameters ``(b, ln a)``, ``m = 1/a``:

    l = lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1+a mu) + y ln(a mu)
    dl/deta = (y - mu)/(1 + a mu)
    dl/dln a = m [psi(m) - psi(y+m) + ln(1+a mu)] + (y - mu)/(1 + a mu)

NB1, ``Var = mu (1 + d)``, parameters ``(b, ln d)``, ``m = mu/d``,
``D = psi(y+m) - psi(m) - ln(1+d)``:

    l = lnG(y+m) - lnG(m) - lnG(y+1) - m ln(1+d) + y ln(d/(1+d))
    dl/deta = m D,   dl/dln d = -m D + (y - mu)/(1 + d)

Second derivatives are in the code; both are verified against numerical
derivatives in the tests.

Beta regression (``betareg``)
-----------------------------
``y ~ Beta(mu phi, (1-mu) phi)`` with ``mu = g^-1(x'b)`` and
``phi = s^-1(z'c)`` (Ferrari and Cribari-Neto 2004); with ``y* = logit y``,
``mu* = psi(mu phi) - psi((1-mu) phi)``:

    dl/dmu = phi (y* - mu*),   dl/dphi = mu (y* - mu*) + ln(1-y) - psi((1-mu)phi) + psi(phi)

and the chain rule through the two links gives the gradient and Hessian.

ppmlhdfe
--------
Poisson IRLS where every weighted least-squares step partials the fixed
effects out of the working response and the regressors (``absorb.demean``
with weights ``w mu``); the full linear predictor is recovered from the
working residual, ``eta = offset + z - (z~ - X~ b)``, which is the fitted
value of the dummy-variable regression by Frisch-Waugh-Lovell.

The iteration stops when the relative change in the deviance is at most
``tol`` AND no linear predictor moved by more than ``_DRIFT`` in that step.
The second condition is what separates convergence from separation: where a
regressor (or a combination with the fixed effects) predicts zero outcomes
perfectly, the affected ``eta_i`` fall by about one per iteration for ever
while their contribution to the deviance, ``2 w_i mu_i``, vanishes. Such a
run is stopped after ``_STALL`` flat-deviance iterations and reported with
the number of drifting zero-outcome observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.glm.families import Family, Link
from openecon.engines.absorb import demean
from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import poisson_deviance, validate_nb_precision
from openecon.engines.linalg import least_squares, weighted_crossprod

_EPS = torch.finfo(torch.float64).eps
_MAX_HALVINGS = 40
# ppmlhdfe: a linear predictor that still moves by more than _DRIFT per iteration once the
# deviance is flat is diverging (separation); _STALL such iterations end the run.
_DRIFT = 1e-2
_STALL = 5


def _check(x: Tensor, y: Tensor, prior: Tensor, offset: Tensor | None) -> None:
    if not isinstance(x, Tensor) or x.dtype != torch.float64 or x.ndim != 2:
        raise KernelError("invalid_design", "The design must be an n-by-k float64 tensor.")
    n = x.shape[0]
    for name, vector in (("outcome", y), ("weights", prior), ("offset", offset)):
        if vector is None:
            continue
        if not isinstance(vector, Tensor) or vector.dtype != torch.float64 \
                or vector.shape != (n,):
            raise KernelError("invalid_design", f"The {name} must be a float64 vector [n].")
        if not bool(torch.isfinite(vector).all()):
            raise KernelError("non_finite_values", f"The {name} must be finite.")
    if not bool(torch.isfinite(x).all()):
        raise KernelError("non_finite_values", "The design must be finite.")
    if bool((prior < 0).any()):
        raise KernelError("invalid_weights", "Weights must be nonnegative.")


@dataclass
class GlmState:
    beta: Tensor
    eta: Tensor
    mu: Tensor
    comp: Tensor | None        # 1 - mu for binomial outcomes
    deviance: float


@dataclass
class IrlsResult:
    state: GlmState
    iterations: int
    converged: bool
    message: str


class GlmObjective:
    """``-D(b)/2`` with analytic gradient and observed Hessian; IRLS and post-fit pieces.

    ``prior`` are the weights of the quasi-likelihood sums (user weights times
    trials); ``mult = prior / user`` (the trials) rescales per-observation
    score rows so that the analysis layer can apply user weights separately.
    """

    def __init__(self, x: Tensor, y: Tensor, prior: Tensor, offset: Tensor | None,
                 family: Family, link: Link, mult: Tensor | None = None):
        _check(x, y, prior, offset)
        # Fixed-dispersion NB must refuse unsafe gamma differences before the
        # coefficient iteration, rather than after an apparently successful fit.
        if family.name == "nbinomial":
            validate_nb_precision(y, 1 / family.k)
        self.x, self.y, self.prior, self.offset = x, y, prior, offset
        self.family, self.link = family, link
        self.mult = mult
        self.k = x.shape[1]

    # ---- states ------------------------------------------------------------------

    def predictor(self, beta: Tensor) -> Tensor:
        eta = self.x @ beta
        return eta if self.offset is None else eta + self.offset

    def state(self, beta: Tensor) -> GlmState | None:
        """Fitted pieces at ``beta``, or None when eta leaves the link's domain."""
        eta = self.predictor(beta)
        if not bool(self.link.valid(eta).all()):
            return None
        mu = self.link.inverse(eta)
        if not bool(self.family.valid_mu(mu).all()):
            return None
        comp = self.link.complement(eta, mu) if self.family.binomial else None
        if comp is not None and not bool((comp > 0).all()):
            return None
        deviance = float((self.prior * self.family.unit_deviance(self.y, mu, comp)).sum())
        if not math.isfinite(deviance):
            return None
        return GlmState(beta, eta, mu, comp, deviance)

    def pieces(self, state: GlmState) -> tuple[Tensor, Tensor, Tensor]:
        """Per-observation ``(s, h, f)`` without prior weights (see module docstring)."""
        eta, mu, y = state.eta, state.mu, self.y
        q = self.link.second_ratio(eta, mu)
        if self.family.binomial:
            a, b = self.link.ratios(eta, mu)
            s = y * a - (1 - y) * b
            return s, -a * b + s * (q - (a - b)), a * b
        m1 = self.link.derivative(eta, mu)
        v = self.family.variance(mu)
        s = (y - mu) / v * m1
        f = m1.square() / v
        h = -f + s * (q - m1 * self.family.variance_derivative(mu) / v)
        return s, h, f

    # ---- Newton objective ---------------------------------------------------------

    def value(self, beta: Tensor) -> Tensor:
        state = self.state(beta)
        return torch.tensor(-math.inf if state is None else -0.5 * state.deviance,
                            dtype=torch.float64)

    def __call__(self, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        state = self.state(beta)
        if state is None:
            k = self.k
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(k, dtype=torch.float64),
                    torch.zeros((k, k), dtype=torch.float64))
        s, h, _ = self.pieces(state)
        gradient = self.x.T @ (self.prior * s)
        hessian = weighted_crossprod(self.x, self.prior * h)
        return torch.tensor(-0.5 * state.deviance, dtype=torch.float64), gradient, hessian

    # ---- post-fit quantities --------------------------------------------------------

    def hessian(self, state: GlmState) -> Tensor:
        return weighted_crossprod(self.x, self.prior * self.pieces(state)[1])

    def fisher_information(self, state: GlmState) -> Tensor:
        return weighted_crossprod(self.x, self.prior * self.pieces(state)[2])

    def score_rows(self, state: GlmState) -> Tensor:
        """Unit-dispersion score rows ``x_i s_i mult_i`` (user weights not applied)."""
        s = self.pieces(state)[0]
        if self.mult is not None:
            s = s * self.mult
        return self.x * s[:, None]

    def pearson(self, state: GlmState) -> float:
        return float((self.prior * self.family.unit_pearson(self.y, state.mu, state.comp)).sum())

    def log_likelihood(self, state: GlmState, phi: float, user: Tensor) -> float:
        terms = self.family.log_likelihood(self.y, state.mu, phi, state.comp)
        return float((user * terms).sum())

    def gradient_max(self, state: GlmState) -> float:
        return float((self.x.T @ (self.prior * self.pieces(state)[0])).abs().max())

    # ---- starting values and IRLS -------------------------------------------------------

    def start(self, intercept: bool) -> Tensor:
        """Stata-style start: one weighted least-squares step from ``mu_0 = start_mu(y)``.

        Falls back to the link of the weighted mean of the start (intercept
        only) and then to zeros when the first step leaves the link's domain.
        """
        n, k = self.x.shape
        prior, family, link = self.prior, self.family, self.link
        mu0 = family.start_mu(self.y)
        candidates: list[Tensor] = []
        if bool(family.valid_mu(mu0).all()):
            eta0 = link.link(mu0)
            if bool(link.valid(eta0).all()):
                z = eta0 if self.offset is None else eta0 - self.offset
                weights = prior * link.derivative(eta0, mu0).square() / family.variance(mu0)
                if bool(torch.isfinite(weights).all()):
                    try:
                        fit = least_squares(self.x, z, weights, drop_collinear=True, tol=0.0)
                        candidates.append(fit.beta)
                    except KernelError:
                        pass
        if intercept and bool(prior.sum() > 0):
            mean = (prior * mu0).sum() / prior.sum()
            flat = torch.zeros(k, dtype=torch.float64)
            if bool(family.valid_mu(mean)):
                flat[0] = float(link.link(mean))
            candidates.append(flat)
        candidates.append(torch.zeros(k, dtype=torch.float64))
        for beta in candidates:
            if bool(torch.isfinite(beta).all()) and self.state(beta) is not None:
                return beta
        raise KernelError("invalid_start", "No starting values keep the linear predictor inside "
                          "the link's domain; choose another link or rescale the outcome.")

    def irls(self, beta: Tensor, *, tol: float, max_iter: int) -> IrlsResult:
        """Fisher scoring by weighted QR least squares with deviance step halving."""
        state = self.state(beta)
        if state is None:
            raise KernelError("invalid_start", "The starting values are outside the link domain.")
        message = (f"IRLS did not converge in {max_iter} iterations (Fisher scoring converges "
                   "only linearly for non-canonical links; optimizer='ml' is quadratic).")
        converged = settled = False
        steps, previous = 0, math.inf
        while True:
            s, _, f = self.pieces(state)
            weights = self.prior * f
            z = state.eta if self.offset is None else state.eta - self.offset
            z = z + torch.where(f > 0, s / torch.where(f > 0, f, torch.ones_like(f)), 0.0)
            if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(z).all()):
                raise KernelError("numerical_failure", "The IRLS working weights are not finite; "
                                  "the fit is diverging.")
            try:
                fit = least_squares(self.x, z, weights, drop_collinear=True, tol=0.0)
            except KernelError as exc:
                if exc.code != "singular_design":
                    raise
                fit = None
            if fit is None or fit.omitted:
                # The working weights of some observations have vanished (fitted means
                # running to the edge of the family's support): the iteration is diverging,
                # which the caller diagnoses from the last valid state.
                message = ("the weighted least-squares step of IRLS became rank deficient: "
                           "the working weights of some observations vanished.")
                break
            direction = fit.beta - state.beta
            # The deviance is flat at the optimum, so its change alone would stop a linearly
            # converging (non-canonical) iteration far too early. The remaining Fisher step
            # must also be negligible: g' I^-1 g (its squared length in standard-error
            # units) and its relative size, down to the rounding floor of the QR solve,
            # which shows as a step that no longer contracts.
            scaled = float((self.x.T @ (self.prior * s)) @ direction)
            relative = float((direction.abs() / state.beta.abs().clamp_min(1.0)).max())
            if settled and scaled <= tol and (
                    relative <= tol or (relative <= math.sqrt(tol) and relative >= previous)):
                converged = True
                message = "converged: relative deviance change, scaled gradient and step"
                break
            if steps >= max_iter:
                break
            previous = relative
            allowance = 8 * _EPS * max(1.0, state.deviance)
            fraction, accepted = 1.0, None
            for _ in range(_MAX_HALVINGS):
                candidate = self.state(state.beta + fraction * direction)
                if candidate is not None and candidate.deviance <= state.deviance + allowance:
                    accepted = candidate
                    break
                fraction *= 0.5
            if accepted is None:
                message = "IRLS step halving found no step that keeps the deviance from rising."
                break
            settled = abs(accepted.deviance - state.deviance) <= tol * (abs(accepted.deviance)
                                                                        + 1e-300)
            state = accepted
            steps += 1
        return IrlsResult(state, steps, converged, message)


# ---- negative binomial ---------------------------------------------------------------


class NegativeBinomialObjective:
    """Full NB2 (``form='mean'``) or NB1 (``form='constant'``) log likelihood in ``(b, ln a)``."""

    def __init__(self, x: Tensor, y: Tensor, prior: Tensor, offset: Tensor | None, form: str):
        _check(x, y, prior, offset)
        if form not in ("mean", "constant"):
            raise KernelError("invalid_option", "dispersion must be 'mean' or 'constant'.")
        validate_nb_precision(y)
        self.x, self.y, self.prior, self.offset, self.form = x, y, prior, offset, form
        self.k = x.shape[1]
        self.constant = -torch.lgamma(y + 1)
        self.reject_precision_trials = False
        self.precision_rejected = False

    def predictor(self, theta: Tensor) -> Tensor:
        eta = self.x @ theta[: self.k]
        return eta if self.offset is None else eta + self.offset

    def mean(self, theta: Tensor) -> Tensor:
        return torch.exp(self.predictor(theta))

    def check_precision(self, theta: Tensor) -> None:
        """Guard the NB gamma arguments, including the estimated ancillary shape."""
        log_a = float(theta[self.k])
        # An ordinary invalid link/domain trial remains an ordinary numerical
        # failure; precision_unsupported is reserved for finite gamma arguments.
        if not math.isfinite(log_a) or abs(log_a) > 700:
            return
        shape = (math.exp(-log_a) if self.form == "mean" else
                 self.mean(theta) * math.exp(-log_a))
        validate_nb_precision(self.y, shape)

    def _terms(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
        """Per-observation ``(l, g_eta, g_a, h_ee, h_ea, h_aa)`` before weighting."""
        mu = self.mean(theta)
        log_a = float(theta[self.k])
        if not bool(torch.isfinite(mu).all()) or not math.isfinite(log_a) or abs(log_a) > 700:
            return None
        try:
            # Reuse this trial's mean; NB1 must not repeat the n-by-k predictor
            # product merely to check its gamma arguments.
            shape = math.exp(-log_a) if self.form == "mean" else mu * math.exp(-log_a)
            validate_nb_precision(self.y, shape)
        except KernelError as exc:
            if exc.code != "precision_unsupported" or not self.reject_precision_trials:
                raise
            self.precision_rejected = True
            return None
        y = self.y
        if self.form == "mean":
            a, m = math.exp(log_a), math.exp(-log_a)
            ratio = 1 + a * mu
            log_ratio = torch.log1p(a * mu)
            value = (torch.lgamma(y + m) - math.lgamma(m) + self.constant - (y + m) * log_ratio
                     + torch.special.xlogy(y, a * mu))
            at_m = torch.full_like(y, m)
            c = torch.special.digamma(at_m) - torch.special.digamma(y + m) + log_ratio
            t = torch.special.polygamma(1, at_m) - torch.special.polygamma(1, y + m)
            b = (y - mu) / ratio
            g_eta, g_a = b, m * c + b
            h_ee = -mu * (1 + a * y) / ratio.square()
            h_ea = -a * mu * (y - mu) / ratio.square()
            h_aa = -m * c - m * m * t + mu / ratio + h_ea
        else:
            d = math.exp(log_a)
            m = mu / d
            log1pd = math.log1p(d)
            value = (torch.lgamma(y + m) - torch.lgamma(m) + self.constant - m * log1pd
                     + y * (log_a - log1pd))
            big = torch.special.digamma(y + m) - torch.special.digamma(m) - log1pd
            t = torch.special.polygamma(1, y + m) - torch.special.polygamma(1, m)
            curve = m * big + m.square() * t
            g_eta, g_a = m * big, -m * big + (y - mu) / (1 + d)
            h_ee = curve
            h_ea = -curve - mu / (1 + d)
            h_aa = curve + mu / (1 + d) - (y - mu) * d / (1 + d) ** 2
        return value, g_eta, g_a, h_ee, h_ea, h_aa

    def value(self, theta: Tensor) -> Tensor:
        terms = self._terms(theta)
        if terms is None:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.prior * terms[0]).sum()

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k = self.k
        terms = self._terms(theta)
        if terms is None:
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(k + 1, dtype=torch.float64),
                    torch.zeros((k + 1, k + 1), dtype=torch.float64))
        value, g_eta, g_a, h_ee, h_ea, h_aa = terms
        w = self.prior
        gradient = torch.cat([self.x.T @ (w * g_eta), (w * g_a).sum().reshape(1)])
        hessian = torch.empty((k + 1, k + 1), dtype=torch.float64)
        hessian[:k, :k] = weighted_crossprod(self.x, w * h_ee)
        cross = self.x.T @ (w * h_ea)
        hessian[:k, k] = cross
        hessian[k, :k] = cross
        hessian[k, k] = (w * h_aa).sum()
        return (w * value).sum(), gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        terms = self._terms(theta)
        if terms is None:
            raise KernelError("numerical_failure", "The negative binomial scores are not finite.")
        return torch.cat([self.x * terms[1][:, None], terms[2][:, None]], dim=1)


# ---- beta regression ----------------------------------------------------------------------


class ScaleLink:
    """Precision links of betareg: ``log`` (default), ``identity`` and ``sqrt``."""

    def __init__(self, name: str):
        if name not in ("log", "identity", "sqrt"):
            raise KernelError("invalid_link", "scale_link must be log, identity or sqrt.")
        self.name = name

    def inverse(self, eta: Tensor) -> Tensor:
        if self.name == "log":
            return torch.exp(eta)
        return eta if self.name == "identity" else eta.square()

    def link(self, phi: Tensor) -> Tensor:
        if self.name == "log":
            return torch.log(phi)
        return phi if self.name == "identity" else phi.sqrt()

    def derivatives(self, eta: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
        if self.name == "log":
            return phi, phi
        if self.name == "identity":
            return torch.ones_like(eta), torch.zeros_like(eta)
        return 2 * eta, torch.full_like(eta, 2.0)

    def valid(self, eta: Tensor) -> Tensor:
        finite = torch.isfinite(eta)
        return finite if self.name == "log" else finite & (eta > 0)


class BetaObjective:
    """Beta regression log likelihood in ``theta = (b, c)`` with analytic derivatives."""

    def __init__(self, x: Tensor, z: Tensor, y: Tensor, prior: Tensor, link: Link,
                 scale_link: ScaleLink):
        _check(x, y, prior, None)
        if not isinstance(z, Tensor) or z.dtype != torch.float64 or z.ndim != 2 \
                or z.shape[0] != x.shape[0]:
            raise KernelError("invalid_design", "The scale design must be float64 [n, q].")
        if not bool(((y > 0) & (y < 1)).all()):
            raise KernelError("invalid_outcome", "Beta regression needs 0 < y < 1.")
        self.x, self.z, self.y, self.prior = x, z, y, prior
        self.link, self.scale_link = link, scale_link
        self.k, self.q = x.shape[1], z.shape[1]
        self.log_y, self.log_1y = torch.log(y), torch.log1p(-y)
        self.ystar = self.log_y - self.log_1y

    def means(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
        eta1, eta2 = self.x @ theta[: self.k], self.z @ theta[self.k:]
        if not bool(self.link.valid(eta1).all()) or not bool(self.scale_link.valid(eta2).all()):
            return None
        mu, phi = self.link.inverse(eta1), self.scale_link.inverse(eta2)
        comp = self.link.complement(eta1, mu)
        if not bool((torch.isfinite(mu) & (mu > 0) & (comp > 0) & torch.isfinite(phi)
                     & (phi > 0)).all()):
            return None
        return eta1, eta2, mu, comp, phi

    def _terms(self, theta: Tensor):
        means = self.means(theta)
        if means is None:
            return None
        eta1, eta2, mu, comp, phi = means
        a, b = mu * phi, comp * phi
        value = (torch.lgamma(phi) - torch.lgamma(a) - torch.lgamma(b) + (a - 1) * self.log_y
                 + (b - 1) * self.log_1y)
        psi_a, psi_b = torch.special.digamma(a), torch.special.digamma(b)
        tri_a, tri_b = torch.special.polygamma(1, a), torch.special.polygamma(1, b)
        gap = self.ystar - psi_a + psi_b
        l_mu = phi * gap
        l_phi = mu * gap + self.log_1y - psi_b + torch.special.digamma(phi)
        l_mumu = -phi.square() * (tri_a + tri_b)
        l_muphi = gap - phi * (mu * tri_a - comp * tri_b)
        l_phiphi = -mu.square() * tri_a - comp.square() * tri_b + torch.special.polygamma(1, phi)
        m1 = self.link.derivative(eta1, mu)
        q1 = self.link.second_ratio(eta1, mu)
        s1, s2 = self.scale_link.derivatives(eta2, phi)
        g1, g2 = l_mu * m1, l_phi * s1
        h11 = m1 * (l_mumu * m1 + l_mu * q1)
        h22 = l_phiphi * s1.square() + l_phi * s2
        h12 = l_muphi * m1 * s1
        return value, g1, g2, h11, h12, h22, eta1, mu, phi

    def value(self, theta: Tensor) -> Tensor:
        terms = self._terms(theta)
        if terms is None:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.prior * terms[0]).sum()

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k, q = self.k, self.q
        terms = self._terms(theta)
        if terms is None:
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(k + q, dtype=torch.float64),
                    torch.zeros((k + q, k + q), dtype=torch.float64))
        value, g1, g2, h11, h12, h22, *_ = terms
        w = self.prior
        gradient = torch.cat([self.x.T @ (w * g1), self.z.T @ (w * g2)])
        hessian = torch.empty((k + q, k + q), dtype=torch.float64)
        hessian[:k, :k] = weighted_crossprod(self.x, w * h11)
        hessian[k:, k:] = weighted_crossprod(self.z, w * h22)
        cross = weighted_crossprod(self.x, w * h12, self.z)
        hessian[:k, k:] = cross
        hessian[k:, :k] = cross.T
        return (w * value).sum(), gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        terms = self._terms(theta)
        if terms is None:
            raise KernelError("numerical_failure", "The beta regression scores are not finite.")
        return torch.cat([self.x * terms[1][:, None], self.z * terms[2][:, None]], dim=1)

    def fitted(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(eta1, mu, phi)`` at theta (KernelError when outside the domain)."""
        means = self.means(theta)
        if means is None:
            raise KernelError("numerical_failure", "The beta regression means are not finite.")
        return means[0], means[2], means[4]


# ---- ppmlhdfe ------------------------------------------------------------------------------


@dataclass
class PpmlResult:
    beta: Tensor
    eta: Tensor
    mu: Tensor
    x_within: Tensor           # regressors partialled out at the final weights
    xtx_inv: Tensor            # (X~' W X~)^-1, W = prior * mu
    deviance: float
    iterations: int
    converged: bool
    message: str
    demeaning_sweeps: int
    separated: int = 0         # zero-outcome observations whose eta was still falling
    diverging: int = -1        # column of the regressor whose coefficient moved most


def _poisson_deviance(y: Tensor, mu: Tensor, prior: Tensor) -> float:
    return float((prior * poisson_deviance(y, mu)).sum())


@torch.no_grad()
def ppml_irls(x: Tensor, y: Tensor, prior: Tensor, offset: Tensor | None,
              dimensions: list[tuple[Tensor, int]], *, tol: float, max_iter: int,
              demean_tol: float, demean_max_iter: int) -> PpmlResult:
    """Poisson pseudo-ML with absorbed fixed effects by partialled-out IRLS.

    Starts from ``mu_0 = (y + ybar)/2`` (ppmlhdfe), iterates the weighted
    least squares of the module docstring with step halving on the deviance
    and stops when the relative change in the deviance is at most ``tol``
    and every linear predictor has settled (see the module docstring; a
    separated fit is returned with ``converged=False`` and ``separated > 0``).
    The returned ``x_within`` and ``xtx_inv`` are recomputed at the final
    weights so that sandwich covariances use the converged working design.
    """
    _check(x, y, prior, offset)
    if bool((y < 0).any()):
        raise KernelError("invalid_outcome", "Poisson outcomes must be nonnegative.")
    n, k = x.shape
    total = float(prior.sum())
    if total <= 0:
        raise KernelError("invalid_weights", "The weights must not all be zero.")
    mean = float((prior * y).sum()) / total
    mu = (y + mean) / 2
    eta = torch.log(mu)
    deviance = _poisson_deviance(y, mu, prior)
    beta = torch.zeros(k, dtype=torch.float64)
    base = torch.zeros_like(y) if offset is None else offset
    converged, sweeps, iteration = False, 0, 0
    stalled, separated, diverging = 0, 0, -1
    message = f"ppmlhdfe IRLS did not converge in {max_iter} iterations."

    def step(eta: Tensor, mu: Tensor):
        weights = prior * mu
        z = (eta - base) + (y - mu) / mu
        within = demean(torch.cat([z[:, None], x], dim=1), dimensions, weights, tol=demean_tol,
                        max_iter=demean_max_iter)
        z_within, x_within = within.values[:, 0], within.values[:, 1:]
        fit = least_squares(x_within, z_within, weights, drop_collinear=True, tol=0.0)
        if fit.omitted:
            raise KernelError("singular_design", "The partialled-out design became rank "
                              "deficient during IRLS; drop the affected regressors.")
        return z, z_within, x_within, fit, within.iterations

    for iteration in range(1, max_iter + 1):
        z, z_within, x_within, fit, used = step(eta, mu)
        sweeps += used
        eta_new = base + z - z_within + x_within @ fit.beta
        # The start (y + ybar)/2 is not a point of the model (it nearly interpolates y), so
        # its deviance is below anything the model can reach: the first step only has to
        # be finite. From then on the deviance of successive model iterates must not rise.
        limit = math.inf if iteration == 1 else deviance + 8 * _EPS * max(1.0, deviance)
        fraction, accepted = 1.0, None
        for _ in range(_MAX_HALVINGS):
            eta_c = eta + fraction * (eta_new - eta)
            mu_c = torch.exp(eta_c)
            finite = bool(torch.isfinite(mu_c).all())
            dev_c = _poisson_deviance(y, mu_c, prior) if finite else math.inf
            if math.isfinite(dev_c) and dev_c <= limit:
                accepted = (eta_c, mu_c, dev_c)
                break
            fraction *= 0.5
        if accepted is None:
            message = "ppmlhdfe step halving found no step that keeps the deviance from rising."
            break
        moved = fraction * (fit.beta - beta)
        beta = beta + moved
        change = abs(accepted[2] - deviance)
        drifting = (accepted[0] - eta).abs() > _DRIFT
        eta, mu, deviance = accepted
        if iteration > 1 and change <= tol * (abs(deviance) + 1e-300):
            if not bool(drifting.any()):
                converged = True
                message = "converged: relative change in deviance and linear predictor"
                break
            stalled += 1
            if stalled >= _STALL:
                separated = int((drifting & (y == 0)).sum())
                diverging = int(moved.abs().argmax()) if k and float(moved.abs().max()) > _DRIFT \
                    else -1
                message = ("the deviance is flat but the linear predictor of "
                           f"{int(drifting.sum())} observation(s) keeps falling")
                break
        else:
            stalled = 0
    _, _, x_within, fit, used = step(eta, mu)
    sweeps += used
    return PpmlResult(beta, eta, mu, x_within, fit.xtx_inv, deviance, iteration, converged,
                      message, sweeps, separated, diverging)
