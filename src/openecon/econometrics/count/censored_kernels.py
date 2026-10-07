"""Analytic censored Poisson/NB likelihoods on float64 Torch.

NB tails use a safeguarded incomplete-beta continued fraction with explicit
second-order forward derivatives, not autograd or a capped probability sum.
Poisson gamma tails use native special functions and log-domain rare-tail
fallbacks. Unconverged special-function evaluations fail explicitly.
"""
from __future__ import annotations

import math

import torch

from openecon.econometrics.count.kernels import NegBinDensity, PoissonDensity
from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import poisson_logmass

_TOL = 2e-14
_MAX_ITER = 10000
_NB_MAX_GAMMA_ARGUMENT = 1e7
_POISSON_MAX_CUMULATIVE_CUTOFF = 1e12


def _poisson_logpmf(y, eta):
    """Private compatibility alias for the shared stable count primitive."""
    return poisson_logmass(y, torch.exp(eta), eta=eta)


class _Jet:
    """Value and analytic first/second derivatives in at most two indices."""

    def __init__(self, value, gradient, hessian):
        self.v, self.g, self.h = value, gradient, hessian

    @classmethod
    def constant(cls, value, template):
        return cls(torch.as_tensor(value, dtype=torch.float64).expand_as(template.v),
                   torch.zeros_like(template.g), torch.zeros_like(template.h))

    @classmethod
    def variable(cls, value, position, dimensions):
        gradient = torch.zeros((*value.shape, dimensions), dtype=torch.float64)
        if dimensions:
            gradient[..., position] = 1.
        return cls(value, gradient,
                   torch.zeros((*value.shape, dimensions, dimensions), dtype=torch.float64))

    def _as(self, other):
        return other if isinstance(other, _Jet) else self.constant(other, self)

    def __add__(self, other):
        other = self._as(other)
        return _Jet(self.v + other.v, self.g + other.g, self.h + other.h)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.v, -self.g, -self.h)

    def __sub__(self, other):
        return self + -self._as(other)

    def __rsub__(self, other):
        return self._as(other) + -self

    def __mul__(self, other):
        other = self._as(other)
        return _Jet(self.v * other.v,
                    self.g * other.v[..., None] + other.g * self.v[..., None],
                    self.h * other.v[..., None, None] + other.h * self.v[..., None, None]
                    + self.g[..., :, None] * other.g[..., None, :]
                    + other.g[..., :, None] * self.g[..., None, :])

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = 1 / self.v
        relative = self.g / self.v[..., None]
        return _Jet(inverse, -relative * inverse[..., None],
                    (2 * relative[..., :, None] * relative[..., None, :]
                     - self.h / self.v[..., None, None]) * inverse[..., None, None])

    def __truediv__(self, other):
        return self * self._as(other).reciprocal()

    def __rtruediv__(self, other):
        return self._as(other) * self.reciprocal()

    def unary(self, value, first, second):
        return _Jet(value, first[..., None] * self.g,
                    first[..., None, None] * self.h
                    + second[..., None, None] * self.g[..., :, None] * self.g[..., None, :])

    def exp(self):
        value = torch.exp(self.v)
        return self.unary(value, value, value)

    def log(self):
        relative = self.g / self.v[..., None]
        return _Jet(torch.log(self.v), relative,
                    self.h / self.v[..., None, None] - relative[..., :, None] * relative[..., None, :])

    def lgamma(self):
        return self.unary(torch.lgamma(self.v), torch.special.digamma(self.v),
                          torch.special.polygamma(1, self.v))

    def logsigmoid(self):
        slope = torch.sigmoid(-self.v)
        return self.unary(torch.nn.functional.logsigmoid(self.v), slope,
                          -slope * torch.sigmoid(self.v))

    def log1mexp(self):
        value = torch.where(self.v < -math.log(2), torch.log1p(-torch.exp(self.v)),
                            torch.log(-torch.expm1(self.v)))
        ratio = 1 / torch.expm1(-self.v)
        return self.unary(value, -ratio, -ratio * (1 + ratio))

    def safeguard(self):
        small = self.v.abs() < 1e-30
        return _choose(small, self.constant(torch.where(self.v < 0, -1e-30, 1e-30), self), self)


def _choose(mask, a, b):
    return _Jet(torch.where(mask, a.v, b.v), torch.where(mask[..., None], a.g, b.g),
                torch.where(mask[..., None, None], a.h, b.h))


def _logadd(a, b):
    value = torch.logaddexp(a.v, b.v)
    p, q = torch.exp(a.v - value), torch.exp(b.v - value)
    gap = a.g - b.g
    return _Jet(value, p[..., None] * a.g + q[..., None] * b.g,
                p[..., None, None] * a.h + q[..., None, None] * b.h
                + (p * q)[..., None, None] * gap[..., :, None] * gap[..., None, :])


def _beta_fraction(a, b, x):
    one = a.constant(1., a)
    qab, qap, qam = a + b, a + 1, a - 1
    c = one
    d = (one - qab * x / qap).safeguard().reciprocal()
    h = d
    for iteration in range(1, _MAX_ITER + 1):
        m, m2 = iteration, 2 * iteration
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = (one + aa * d).safeguard().reciprocal()
        c = (one + aa / c).safeguard()
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = (one + aa * d).safeguard().reciprocal()
        c = (one + aa / c).safeguard()
        change = d * c
        h = h * change
        if bool(((change.v - 1).abs() < _TOL).all()) \
                and (not change.g.numel() or bool((change.g.abs() < 2e-12).all())) \
                and (not change.h.numel() or bool((change.h.abs() < 2e-10).all())):
            return h
        if not bool(torch.isfinite(h.v).all()):
            break
    raise KernelError("tail_nonconvergence", "The incomplete-beta continued fraction did not converge; no approximate tail was returned.")


def _nb_tails(k, eta, tau, form, derivatives):
    """Log Pr(Y<=k), log Pr(Y>k), with analytic eta/tau derivatives."""
    dimensions = 2 if derivatives else 0
    e = _Jet.variable(eta, 0, dimensions)
    t = _Jet.variable(torch.ones_like(eta) * tau, 1, dimensions)
    shape = (-t).exp() if form == "mean" else (e - t).exp()
    if bool((shape.v + k + 1 > _NB_MAX_GAMMA_ARGUMENT).any()):
        raise KernelError("precision_unsupported", "NB gamma arguments above 1e7 are outside the validated double-precision domain; no rounded tail was returned.")
    odds = e + t if form == "mean" else t
    log_p, log_q = (-odds).logsigmoid(), odds.logsigmoid()
    p, q = log_p.exp(), log_q.exp()
    other = shape.constant(k + 1, shape)
    direct = p.v < (shape.v + 1) / (shape.v + other.v + 2)
    a, b = _choose(direct, shape, other), _choose(direct, other, shape)
    x = _choose(direct, p, q)
    log_x, log_1x = _choose(direct, log_p, log_q), _choose(direct, log_q, log_p)
    fraction = _beta_fraction(a, b, x)
    small = (a + b).lgamma() - a.lgamma() - b.lgamma() + a * log_x + b * log_1x \
        + fraction.log() - a.log()
    if not bool((small.v < 0).all()):
        raise KernelError("numerical_failure", "The incomplete-beta tail lost precision at this parameter scale.")
    complement = small.log1mexp()
    return _choose(direct, small, complement), _choose(direct, complement, small)


def _poisson_rare_log(shape, mu, eta, lower):
    """Log-domain gamma P series or gamma Q continued fraction, explicitly converged."""
    prefactor = eta + _poisson_logpmf(shape - 1, eta)
    if lower:
        term = torch.ones_like(mu)
        total = term.clone()
        for iteration in range(1, _MAX_ITER + 1):
            term = term * mu / (shape + iteration)
            total = total + term
            if bool((term.abs() <= _TOL * total.abs()).all()):
                return prefactor - torch.log(shape) + torch.log(total)
    else:
        b = mu + 1 - shape
        c = torch.full_like(mu, 1e30)
        d = 1 / b
        h = d.clone()
        for iteration in range(1, _MAX_ITER + 1):
            an = -iteration * (iteration - shape)
            b = b + 2
            d = an * d + b
            c = b + an / c
            d = 1 / d
            change = d * c
            h = h * change
            if bool(((change - 1).abs() < _TOL).all()):
                return prefactor + torch.log(h)
    raise KernelError("tail_nonconvergence", "The log-domain Poisson tail did not converge; no capped sum was returned.")


def _poisson_tails(k, eta, derivatives):
    if bool((k > _POISSON_MAX_CUMULATIVE_CUTOFF).any()):
        raise KernelError("precision_unsupported", "Poisson cumulative cutoffs above 1e12 are outside the validated gamma-function domain; exact point masses have a separate precision limit.")
    shape, mu = k + 1, torch.exp(eta)
    cdf, survival = torch.special.gammaincc(shape, mu), torch.special.gammainc(shape, mu)
    log_cdf = torch.where(cdf < .5, torch.log(cdf), torch.log1p(-survival))
    log_sf = torch.where(survival < .5, torch.log(survival), torch.log1p(-cdf))
    for value, is_lower in ((log_cdf, False), (log_sf, True)):
        rare = ~torch.isfinite(value)
        if bool(rare.any()):
            value[rare] = _poisson_rare_log(shape[rare], mu[rare], eta[rare], is_lower)
    if not derivatives:
        blank = torch.zeros((*eta.shape, 0), dtype=torch.float64)
        h = torch.zeros((*eta.shape, 0, 0), dtype=torch.float64)
        return _Jet(log_cdf, blank, h), _Jet(log_sf, blank, h)
    boundary = eta + _poisson_logpmf(k, eta)
    left = -torch.exp(boundary - log_cdf)
    right = torch.exp(boundary - log_sf)
    def terms(value, slope):
        curvature = slope * ((k - mu) + 1) - slope.square()
        return _Jet(value, slope[..., None], curvature[..., None, None])
    return terms(log_cdf, left), terms(log_sf, right)


class CensoredPieces:
    """Event likelihood for inclusive integer ranges lower..upper (upper=inf allowed)."""

    def __init__(self, density, lower, upper):
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor) \
                or lower.dtype != torch.float64 or upper.dtype != torch.float64 \
                or lower.ndim != 1 or upper.shape != lower.shape or not len(lower):
            raise KernelError("invalid_censoring", "Count event bounds must be matching nonempty float64 vectors.")
        finite_upper = upper[torch.isfinite(upper)]
        if not bool(torch.isfinite(lower).all()) or bool(torch.isnan(upper).any()) \
                or bool((lower < 0).any()) or bool((upper < lower).any()) \
                or bool((lower != lower.round()).any()) or bool((finite_upper != finite_upper.round()).any()) \
                or bool((lower > 2**53).any()) or bool((finite_upper > 2**53).any()):
            raise KernelError("invalid_censoring", "Bounds must be ordered inclusive nonnegative integers <=2^53, with +inf permitted only above.")
        self.density, self.lower, self.upper = density, lower, upper
        self.size = density.size
        self.reject_precision_trials = False
        self.precision_rejected = False
        if isinstance(density, NegBinDensity) and (bool((lower >= _NB_MAX_GAMMA_ARGUMENT).any())
                                                   or bool((finite_upper >= _NB_MAX_GAMMA_ARGUMENT).any())):
            raise KernelError("precision_unsupported", "NB count endpoints at or above 1e7 exceed the validated gamma-function domain; use another representation or model.")
        if isinstance(density, PoissonDensity):
            # Short finite intervals are mixtures of exact point masses and
            # do not need gamma cutoffs. Other events require lo-1 and/or hi.
            cumulative = (lower != upper) & (torch.isinf(upper) | (upper - lower > 64))
            oversized = (lower - 1 > _POISSON_MAX_CUMULATIVE_CUTOFF) \
                | (torch.isfinite(upper) & (upper > _POISSON_MAX_CUMULATIVE_CUTOFF))
            if bool((cumulative & oversized).any()):
                raise KernelError("precision_unsupported", "This event needs a Poisson cumulative cutoff above the validated limit 1e12; exact point and short-interval masses have a separate count limit.")

    def check_precision(self, index=None):
        """Check the numerical region explicitly at starts and reported fits."""
        if index is not None and isinstance(self.density, NegBinDensity):
            shape = torch.exp(-index[1]) if self.density.form == "mean" else torch.exp(index[0] - index[1])
            endpoint = torch.where(torch.isfinite(self.upper), self.upper, self.lower)
            if bool((shape + endpoint + 1 > _NB_MAX_GAMMA_ARGUMENT).any()):
                raise KernelError("precision_unsupported", "NB shape plus count exceeds the validated gamma-function scale (1e7); no potentially inaccurate likelihood was returned.")

    def _tails(self, k, index, derivatives):
        if isinstance(self.density, PoissonDensity):
            return _poisson_tails(k, index[0], derivatives)
        return _nb_tails(k, *index, self.density.form, derivatives)

    def __call__(self, index, derivatives=True):
        if len(index) != self.size or not isinstance(index[0], torch.Tensor) \
                or index[0].dtype != torch.float64 or index[0].shape != self.lower.shape \
                or (self.size == 2 and (not isinstance(index[1], torch.Tensor)
                                        or index[1].dtype != torch.float64
                                        or index[1].shape not in (torch.Size([]), self.lower.shape))):
            raise KernelError("invalid_design", "Censored count indices must use float64 and match the event rows.")
        eta = index[0]
        if not bool(torch.isfinite(eta).all()) or not bool(torch.isfinite(torch.exp(eta)).all()):
            return None
        try:
            self.check_precision(index)
        except KernelError:
            if not self.reject_precision_trials:
                raise
            self.precision_rejected = True
            return None
        if isinstance(self.density, NegBinDensity) and (float(index[1].min()) < math.log(1e-9)
                                                       or float(index[1].max()) > 700):
            return None
        dimensions = self.size if derivatives else 0
        n = len(eta)
        result = _Jet(torch.zeros(n, dtype=torch.float64), torch.zeros((n, dimensions), dtype=torch.float64),
                      torch.zeros((n, dimensions, dimensions), dtype=torch.float64))
        whole = (self.lower == 0) & torch.isinf(self.upper)
        exact = self.lower == self.upper
        finite = torch.isfinite(self.upper) & ~exact
        narrow = finite & (self.upper - self.lower <= 64)
        # Point masses and short genuine intervals use exact log-sum-exp mixtures.
        direct = exact | narrow
        if bool(direct.any()):
            selected = [value[direct] if value.ndim else value for value in index]
            lo, hi = self.lower[direct], self.upper[direct]
            accumulated = self._density(lo, selected, derivatives)
            for step in range(1, int((hi - lo).max()) + 1):
                mass = self._density(lo + step, selected, derivatives)
                accumulated = _choose(lo + step <= hi, _logadd(accumulated, mass), accumulated)
            self._assign(result, direct, accumulated)
        remaining = ~whole & ~direct
        if bool(remaining.any()):
            selected = [value[remaining] if value.ndim else value for value in index]
            lo, hi = self.lower[remaining], self.upper[remaining]
            if bool((lo > 0).any()):
                safe_k = (lo - 1).clamp_min(0)
                below, above = self._tails(safe_k, selected, derivatives)
                at_zero = lo == 0
                below = _choose(at_zero, below.constant(-math.inf, below), below)
                above = _choose(at_zero, above.constant(0., above), above)
            else:
                template = _Jet.variable(selected[0], 0, dimensions)
                below, above = template.constant(-math.inf, template), template.constant(0., template)
            bounded = torch.isfinite(hi)
            value = above
            if bool(bounded.any()):
                high_cdf, high_sf = self._tails(torch.where(bounded, hi, 0.), selected, derivatives)
                use_cdf = high_cdf.v < -math.log(2)
                a, b = _choose(use_cdf, high_cdf, above), _choose(use_cdf, below, high_sf)
                interval = a + (b - a).log1mexp()
                value = _choose(bounded, interval, above)
            self._assign(result, remaining, value)
        if not bool(torch.isfinite(result.v).all()) or not bool(torch.isfinite(result.g).all()) \
                or not bool(torch.isfinite(result.h).all()):
            return None
        curves = [result.h[:, a, b] for a in range(dimensions) for b in range(a, dimensions)]
        return result.v, list(result.g.unbind(-1)) if derivatives else None, curves if derivatives else None

    def _density(self, y, index, derivatives):
        if isinstance(self.density, PoissonDensity):
            value = _poisson_logpmf(y, index[0])
            mu = torch.exp(index[0])
            scores, curves = ([y - mu], [-mu]) if derivatives else (None, None)
        else:
            value, scores, curves = self.density.terms(y, torch.lgamma(y + 1), index, derivatives)
        dimensions = self.size if derivatives else 0
        gradient = torch.zeros((len(y), dimensions), dtype=torch.float64)
        hessian = torch.zeros((len(y), dimensions, dimensions), dtype=torch.float64)
        if derivatives:
            gradient = torch.stack([torch.ones_like(y) * score for score in scores], -1)
            position = 0
            for a in range(dimensions):
                for b in range(a, dimensions):
                    hessian[:, a, b] = hessian[:, b, a] = curves[position]
                    position += 1
        return _Jet(value, gradient, hessian)

    @staticmethod
    def _assign(result, mask, value):
        result.v[mask], result.g[mask], result.h[mask] = value.v, value.g, value.h
