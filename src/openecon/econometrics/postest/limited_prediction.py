"""Saved normal limited-response means and their float64 parameter derivatives.

Limits are the resolved *fitted* limits, including outcome-derived min/max limits.
The interval-regression observation bounds are never used as prediction bounds.
Conditional moments use closed forms in the central region and normalized native
Torch quadrature in tails/narrow windows, avoiding inverse-Mills cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.limited.kernels import log_interval_probability

ESTIMATORS = frozenset({"tobit", "truncreg", "intreg"})
_LOG_NORMAL = .5 * math.log(2 * math.pi)
_NODES = 96
_BLOCK = 4096


def _error(code, message):
    raise AnalysisError(code, message)


class _CensoredMean(torch.autograd.Function):
    """Expose stable analytic derivatives instead of canceling CDF gradients."""

    @staticmethod
    def forward(ctx, mu, sigma, mean, first, scale_first):
        ctx.save_for_backward(first, scale_first)
        return mean.clone()

    @staticmethod
    def backward(ctx, gradient):
        first, scale_first = ctx.saved_tensors
        return gradient * first, (gradient * scale_first).sum(), None, None, None


class _CensoredSlope(torch.autograd.Function):
    """Stable mixed sigma derivative of the interval probability, near |z|=1."""

    @staticmethod
    def forward(ctx, mu, sigma, first, second, cross):
        ctx.save_for_backward(second, cross)
        return first.clone()

    @staticmethod
    def backward(ctx, gradient):
        second, cross = ctx.saved_tensors
        return gradient * second, (gradient * cross).sum(), None, None, None


@lru_cache(maxsize=1)
def _quadrature():
    # Golub-Welsch on the Legendre Jacobi matrix; no scientific-library runtime.
    # The first prediction may run in inference mode; cached nodes must still
    # be usable by a subsequent delta-gradient calculation with autograd.
    with torch.inference_mode(False):
        index = torch.arange(1, _NODES, dtype=torch.float64)
        diagonal = index / (4 * index.square() - 1).sqrt()
        values, vectors = torch.linalg.eigh(torch.diag(diagonal, 1) + torch.diag(diagonal, -1))
        return values.detach(), (2 * vectors[0].square()).detach()


@dataclass(frozen=True)
class LimitedNormal:
    estimator: str
    scale_index: int
    log_scale: bool
    lower: float | None
    upper: float | None

    def sigma(self, beta):
        value = beta[self.scale_index]
        sigma = value.exp() if self.log_scale else value
        if not bool(torch.isfinite(sigma)) or not bool(sigma > 0):
            _error("invalid_result", "Saved normal scale must be finite and strictly positive.")
        return sigma

    def definition(self, kind):
        if kind in {"xb", "latent", "stdp"} or self.estimator == "intreg":
            return "latent unconditional normal mean E[Y*|X]"
        if kind == "conditional" or self.estimator == "truncreg":
            return "latent normal mean conditional on the persisted fitted limits"
        return "expected recorded censored outcome E[clip(Y*, lower, upper)|X]"

    def evaluate(self, mu, beta, kind="response"):
        """Return mean, d/dmu, d/dsigma, d2/dmu2, d2/dmu dsigma."""
        if kind in {"xb", "latent", "stdp"} or self.estimator == "intreg":
            zero = mu * 0
            return mu, zero + 1, zero, zero, zero
        sigma = self.sigma(beta)
        if kind == "conditional" or self.estimator == "truncreg":
            return self._conditional(mu, sigma)
        return self._censored(mu, sigma)

    def _standard_bounds(self, mu, sigma):
        a = None if self.lower is None else (self.lower - mu) / sigma
        b = None if self.upper is None else (self.upper - mu) / sigma
        if any(not bool(torch.isfinite(value).all()) for value in (a, b) if value is not None):
            _error("prediction_domain", "Standardized normal prediction bounds exceed float64 range.")
        return a, b

    def _censored(self, mu, sigma):
        a, b = self._standard_bounds(mu, sigma)
        zero = torch.zeros_like(mu)
        # Saturated normal tails have no float64 contribution; limiting these
        # intermediates also prevents overflow in density/derivative products.
        aa = None if a is None else a.clamp(-40, 40)
        bb = None if b is None else b.clamp(-40, 40)
        pa = zero if aa is None else torch.exp(-.5 * aa.square() - _LOG_NORMAL)
        pb = zero if bb is None else torch.exp(-.5 * bb.square() - _LOG_NORMAL)
        if a is None:
            probability = torch.ones_like(mu) if b is None else torch.special.ndtr(bb)
        elif b is None:
            probability = torch.special.ndtr(-aa)
        else:
            # Preserve interval mass in either tail instead of subtracting CDFs.
            active = (a < 40) & (b > -40)
            probability = zero.clone()
            probability[active] = torch.exp(log_interval_probability(aa[active], bb[active]))
        mean = mu * probability + sigma * (pa - pb)
        if a is not None:
            mean = mean + self.lower * torch.special.ndtr(aa)
        if b is not None:
            mean = mean + self.upper * torch.special.ndtr(-bb)
        cross = (zero if aa is None else aa * pa) - (zero if bb is None else bb * pb)
        density_difference = pa - pb
        if a is not None and b is not None:
            width = (self.upper - self.lower) / sigma
            # In a narrow interval neither log-CDF subtraction nor differences
            # of endpoint densities retain relative precision. Integrate the
            # density and its derivative moments over the *physical* width.
            # Keep z=a+delta in split form: (z^2-1) cancels near z=+/-1,
            # whereas (a-1)(a+1)+2*a*delta+delta^2 retains the tiny increment.
            narrow = (width < .1) & (a < 40) & (b > -40)
            if bool(narrow.any()):
                mean = mean.clone()
                probability = probability.clone()
                density_difference = density_difference.clone()
                cross = cross.clone()
            for positions in narrow.nonzero().flatten().split(_BLOCK):
                if not len(positions):
                    continue
                nodes, weights = _quadrature()
                aa_narrow = a[positions, None]
                delta = width * (nodes + 1) / 2
                densities = torch.exp(-.5 * aa_narrow.square() - _LOG_NORMAL
                                      - aa_narrow * delta - .5 * delta.square())
                integrated = width / 2 * weights * densities
                p = integrated.sum(1)
                half_width = width / 2
                center = aa_narrow[:, 0] + half_width
                # Pair the endpoint densities around the interval center. The
                # sinh form preserves tiny differences and exact odd symmetry.
                s = (2 * torch.exp(-.5 * center.square() - _LOG_NORMAL - .5 * half_width.square())
                     * torch.sinh(center * half_width))
                c = (integrated * ((aa_narrow - 1) * (aa_narrow + 1)
                                  + 2 * aa_narrow * delta + delta.square())).sum(1)
                probability[positions], density_difference[positions], cross[positions] = p, s, c
                positive = nodes > 0
                shift = half_width * nodes[positive]
                centered_density = torch.exp(-.5 * center[:, None].square() - _LOG_NORMAL
                                             - .5 * shift.square())
                odd = (centered_density * (-2 * torch.sinh(center[:, None] * shift))
                       * shift * weights[positive]).sum(1) * half_width
                physical_half = (self.upper - self.lower) / 2
                tail_difference = -.5 * (torch.erf(a[positions] / math.sqrt(2))
                                         + torch.erf(b[positions] / math.sqrt(2)))
                if self.lower <= 0 <= self.upper:
                    local_mean = ((self.lower / 2 + self.upper / 2)
                                  + physical_half * tail_difference + sigma * odd)
                else:
                    local_mean = self.lower + physical_half * (1 + tail_difference) + sigma * odd
                mean[positions] = local_mean
        second, mixed = density_difference / sigma, cross / sigma
        if mu.requires_grad or sigma.requires_grad:
            probability = _CensoredSlope.apply(mu, sigma, probability, second, mixed)
            mean = _CensoredMean.apply(mu, sigma, mean, probability, density_difference)
        return mean, probability, density_difference, second, mixed

    def _conditional(self, mu, sigma):
        a, b = self._standard_bounds(mu, sigma)
        if a is None and b is None:
            zero = mu * 0
            return mu, zero + 1, zero, zero, zero
        if a is not None and b is not None:
            width = (self.upper - self.lower) / sigma
            if not bool(torch.isfinite(width)) or not bool(width >= torch.finfo(torch.float64).tiny):
                _error("prediction_precision", "The conditional interval width cannot retain relative float64 precision after standardization.")
        # Central third moments scale as 1/|z|^3 in a far one-sided tail.
        # Beyond this range their float64 underflow can corrupt the mixed
        # derivative even while its product with z remains representable.
        if any(bool((value.abs() > 1e100).any()) for value in (a, b) if value is not None):
            _error("prediction_precision", "Conditional normal moments beyond 1e100 standard deviations cannot retain the required float64 derivatives.")
        # Central, sufficiently wide windows have well-conditioned closed forms.
        regular = torch.ones(len(mu), dtype=torch.bool)
        if a is not None:
            regular &= a.abs() < 4
        if b is not None:
            regular &= b.abs() < 4
        if a is not None and b is not None:
            regular &= (b - a) > .5
        outputs = [torch.zeros_like(mu) for _ in range(5)]
        if bool(regular.any()):
            mm = mu[regular]
            aa = None if a is None else a[regular]
            bb = None if b is None else b[regular]
            zero = torch.zeros_like(mm)
            log_p = (torch.special.log_ndtr(bb) if aa is None else
                     torch.special.log_ndtr(-aa) if bb is None else
                     log_interval_probability(aa, bb))
            ra = zero if aa is None else torch.exp(-.5 * aa.square() - _LOG_NORMAL - log_p)
            rb = zero if bb is None else torch.exp(-.5 * bb.square() - _LOG_NORMAL - log_p)
            d = ra - rb
            second = 1 + (zero if aa is None else aa * ra) - (zero if bb is None else bb * rb)
            third = ((zero if aa is None else (aa.square() + 2) * ra)
                     - (zero if bb is None else (bb.square() + 2) * rb))
            fourth = (3 + (zero if aa is None else (aa.pow(3) + 3 * aa) * ra)
                      - (zero if bb is None else (bb.pow(3) + 3 * bb) * rb))
            variance = second - d.square()
            central3 = third - 3 * d * second + 2 * d.pow(3)
            central4 = fourth - 4 * d * third + 6 * d.square() * second - 3 * d.pow(4)
            values = (mm + sigma * d, variance, central3 + 2 * d * variance,
                      central3 / sigma,
                      (central4 - variance.square() + 2 * d * central3 - 2 * variance) / sigma)
            for output, value in zip(outputs, values, strict=True):
                output[regular] = value
        difficult = (~regular).nonzero().flatten()
        for positions in difficult.split(_BLOCK):
            if not len(positions):
                continue
            values = self._quadrature_moments(mu[positions], sigma)
            for output, value in zip(outputs, values, strict=True):
                output[positions] = value
        if not all(bool(torch.isfinite(value).all()) for value in outputs):
            _error("prediction_precision", "Conditional normal moments are not finite in float64.")
        if bool((outputs[1] < 0).any()):
            _error("prediction_precision", "Conditional normal variance could not be certified.")
        return tuple(outputs)

    def _quadrature_moments(self, mu, sigma):
        a, b = self._standard_bounds(mu, sigma)
        positive = torch.zeros(len(mu), dtype=torch.bool) if a is None else a >= 0
        negative = torch.zeros(len(mu), dtype=torch.bool) if b is None else b <= 0
        values = [torch.zeros_like(mu) for _ in range(5)]
        nodes, weights = _quadrature()
        for mask, direction in ((positive, 1), (negative, -1), (~positive & ~negative, 0)):
            if not bool(mask.any()):
                continue
            mm = mu[mask]
            if direction:
                edge = self.lower if direction == 1 else self.upper
                distance = direction * (edge - mm) / sigma
                scale = 1 / (1 + distance)
                width = (torch.full_like(mm, 40.) if self.lower is None or self.upper is None
                         else ((self.upper - self.lower) / sigma / scale).expand_as(mm).clamp(max=40))
                u = width[:, None] * (nodes + 1) / 2
                t = scale[:, None] * u
                log_w = -distance[:, None] * t - .5 * t.square()
                # Center around the endpoint, keeping tiny tail variation when
                # subtracting two enormous latent indexes would lose it.
                relative = direction * t
                anchor = torch.full_like(mm, edge)
                z_anchor = direction * distance
            else:
                low = torch.full_like(mm, -9.) if a is None else a[mask].clamp(min=-9.)
                high = torch.full_like(mm, 9.) if b is None else b[mask].clamp(max=9.)
                relative = (low + high)[:, None] / 2 + (high - low)[:, None] * nodes / 2
                log_w = -.5 * relative.square()
                anchor, z_anchor = mm, torch.zeros_like(mm)
            probability = torch.softmax(log_w + weights.log(), dim=1)
            average = (probability * relative).sum(1)
            centered = relative - average[:, None]
            variance = (probability * centered.square()).sum(1)
            third = (probability * centered.pow(3)).sum(1)
            fourth = (probability * centered.pow(4)).sum(1)
            d = z_anchor + average
            outputs = (anchor + sigma * average, variance, third + 2 * d * variance,
                       third / sigma,
                       (fourth - variance.square() + 2 * d * third - 2 * variance) / sigma)
            for target, value in zip(values, outputs, strict=True):
                target[mask] = value
        return tuple(values)


def saved_normal(result, terms):
    """Validate the persisted ancillary parameter and resolved limit contract."""
    estimator = result.spec.estimator
    if estimator not in ESTIMATORS:
        return None
    log_scale = estimator == "intreg"
    name = "/lnsigma" if log_scale else "/sigma"
    if terms.count(name) != 1:
        _error("invalid_result", "Saved normal regression needs exactly one fitted scale parameter.")
    lower = upper = None
    if estimator != "intreg":
        limits = result.extra.get("limits")
        if not isinstance(limits, dict) or set(limits) != {"lower", "upper"}:
            _error("invalid_result", "Resolved fitted normal limits are missing or malformed.")
        lower, upper = limits["lower"], limits["upper"]
        for value in (lower, upper):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                _error("invalid_result", "Saved normal limits must be finite numbers or None.")
        if lower is not None and upper is not None and not lower < upper:
            _error("invalid_result", "Saved normal lower limit must be below its upper limit.")
        for role, value, derived in (("ll", lower, "ll_at_min"), ("ul", upper, "ul_at_max")):
            requested = result.spec.options.get(role)
            if requested is not None and (isinstance(requested, bool) or not isinstance(requested, (int, float))
                                          or not math.isfinite(requested)):
                _error("invalid_result", "Saved requested normal limits must be finite numbers or None.")
            flag = result.spec.options.get(derived, False)
            if not isinstance(flag, bool):
                _error("invalid_result", "Saved outcome-derived normal limit flags must be Boolean.")
            if ((not flag and requested != value) or (flag and (requested is not None or value is None))):
                _error("invalid_result", "Resolved normal limits disagree with the saved fitted specification.")
    return LimitedNormal(estimator, terms.index(name), log_scale, lower, upper)
