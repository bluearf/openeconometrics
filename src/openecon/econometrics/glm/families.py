"""Exponential-family variance functions, link functions and log likelihoods (tensors only).

A generalized linear model relates the mean ``mu_i = E[y_i]`` to the linear
predictor ``eta_i = x_i'b + offset_i`` through a link ``g(mu) = eta`` and takes
``Var(y_i) = phi V(mu_i) / w_i`` with the family's variance function ``V``.
Everything a fitter needs is factored into two small objects:

* a :class:`Link` gives ``mu = g^-1(eta)``, ``m1 = dmu/deta``, the ratio
  ``m2/m1 = (d2mu/deta2)/(dmu/deta)`` and, for binomial outcomes, the
  complement ``1 - mu`` and the ratios ``m1/mu`` and ``m1/(1-mu)`` computed
  without cancellation in the tails (``1 - Phi(8)`` is ``Phi(-8)``, not
  ``1 - 1``), which keeps scores finite where fitted probabilities round to 0
  or 1;
* a :class:`Family` gives ``V(mu)``, ``V'(mu)``, the unit deviance
  ``d(y, mu)`` (so that ``D = sum_i w_i d_i``), the unit Pearson term
  ``(y - mu)^2 / V(mu)``, the full per-observation log likelihood at a given
  dispersion ``phi`` (the normalizing terms of the gamma and inverse Gaussian
  families included, as Stata's glm reports it) and the starting mean.

The binomial family works on the proportion scale: with ``n_i`` trials the
outcome ``y_i`` is replaced by ``y_i / n_i`` and the prior weight by
``w_i n_i``; the quasi-likelihood, the deviance and the Pearson statistic of
the count model are then exactly the weighted proportion-scale quantities, and
only the full log likelihood needs the counts (for the ``ln C(n, y)`` term).
Fractional outcomes (``fracreg``) use the same family without that term.

Formulas (working scale, ``k`` the fixed overdispersion of family nbinomial):

    family            V(mu)          unit deviance d(y, mu)
    gaussian          1              (y - mu)^2
    binomial          mu (1 - mu)    2 [y ln(y/mu) + (1 - y) ln((1-y)/(1-mu))]
    poisson           mu             2 [y ln(y/mu) - (y - mu)]
    gamma             mu^2           2 [-ln(y/mu) + (y - mu)/mu]
    inverse_gaussian  mu^3           (y - mu)^2 / (mu^2 y)
    nbinomial(k)      mu + k mu^2    2 [y ln(y/mu) - (y + 1/k) ln((1 + k y)/(1 + k mu))]

    link              mu(eta)                     m1 = dmu/deta          m2/m1
    identity          eta                         1                      0
    log               e^eta                       mu                     1
    logit             1/(1 + e^-eta)              mu (1 - mu)            1 - 2 mu
    probit            Phi(eta)                    phi(eta)               -eta
    cloglog           1 - exp(-e^eta)             e^eta exp(-e^eta)      1 - e^eta
    loglog            exp(-e^-eta)                e^-eta exp(-e^-eta)    e^-eta - 1
    power(a)          eta^(1/a)                   (1/a) eta^(1/a - 1)    (1/a - 1)/eta
    nbinomial(k)      e^eta / (k (1 - e^eta))     mu + k mu^2            1 + 2 k mu

``reciprocal`` is ``power(-1)``, ``inverse_squared`` is ``power(-2)``,
``power(0)`` is ``log`` and ``power(1)`` is ``identity`` (Stata's conventions).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import (
    poisson_deviance, poisson_logmass, validate_nb_precision,
)

_LOG_2PI = math.log(2 * math.pi)
_SQRT_2PI = math.sqrt(2 * math.pi)
FAMILY_NAMES = ("gaussian", "binomial", "poisson", "gamma", "inverse_gaussian", "nbinomial")
LINK_NAMES = ("identity", "log", "logit", "probit", "cloglog", "loglog", "power", "reciprocal",
              "inverse_squared", "nbinomial")
CANONICAL_LINK = {"gaussian": "identity", "binomial": "logit", "poisson": "log",
                  "gamma": "reciprocal", "inverse_gaussian": "inverse_squared",
                  "nbinomial": "log"}
# Links each family accepts: probabilities for binomial, positive means for the others.
PROBABILITY_LINKS = ("logit", "probit", "cloglog", "loglog", "log", "identity")
POSITIVE_LINKS = ("log", "identity", "power", "reciprocal", "inverse_squared")
FAMILY_LINKS = {
    "gaussian": ("identity", "log", "power", "reciprocal", "inverse_squared", "logit", "probit",
                 "cloglog", "loglog"),
    "binomial": PROBABILITY_LINKS, "poisson": POSITIVE_LINKS, "gamma": POSITIVE_LINKS,
    "inverse_gaussian": POSITIVE_LINKS, "nbinomial": (*POSITIVE_LINKS, "nbinomial"),
}


def _ndtr(t: Tensor) -> Tensor:
    """Standard normal cdf as ``erfc(-t / sqrt 2) / 2``.

    ``torch.special.ndtr`` evaluates ``(1 + erf) / 2`` and has no accuracy in
    the lower tail (2% off at -8, exactly 0 at -9); the erfc form is accurate
    to rounding down to the underflow threshold.
    """
    return 0.5 * torch.special.erfc(-t / math.sqrt(2))


def _mills(t: Tensor) -> Tensor:
    """phi(t) / Phi(t) without overflow in the lower tail (via the scaled erfc)."""
    negative = torch.minimum(t, torch.zeros_like(t))
    lower = math.sqrt(2 / math.pi) / torch.special.erfcx(-negative / math.sqrt(2))
    upper = torch.exp(-0.5 * t.square() - math.log(_SQRT_2PI) - torch.special.log_ndtr(t))
    return torch.where(t < 0, lower, upper)


class Link:
    """Base link: ``mu = inverse(eta)``; subclasses override the pieces they can do better."""

    name = ""

    def inverse(self, eta: Tensor) -> Tensor:
        raise NotImplementedError

    def link(self, mu: Tensor) -> Tensor:
        raise NotImplementedError

    def derivative(self, eta: Tensor, mu: Tensor) -> Tensor:
        """``dmu/deta``."""
        raise NotImplementedError

    def second_ratio(self, eta: Tensor, mu: Tensor) -> Tensor:
        """``(d2mu/deta2) / (dmu/deta)``."""
        raise NotImplementedError

    def complement(self, eta: Tensor, mu: Tensor) -> Tensor:
        """``1 - mu`` accurately (binomial outcomes)."""
        return 1 - mu

    def ratios(self, eta: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """``(m1/mu, m1/(1-mu))`` accurately (binomial outcomes)."""
        m1 = self.derivative(eta, mu)
        return m1 / mu, m1 / self.complement(eta, mu)

    def valid(self, eta: Tensor) -> Tensor:
        """Elementwise: eta lies inside the link's domain."""
        return torch.isfinite(eta)


class Identity(Link):
    name = "identity"

    def inverse(self, eta):
        return eta

    def link(self, mu):
        return mu

    def derivative(self, eta, mu):
        return torch.ones_like(eta)

    def second_ratio(self, eta, mu):
        return torch.zeros_like(eta)

    def ratios(self, eta, mu):
        return 1 / mu, 1 / (1 - mu)


class Log(Link):
    name = "log"

    def inverse(self, eta):
        return torch.exp(eta)

    def link(self, mu):
        return torch.log(mu)

    def derivative(self, eta, mu):
        return mu

    def second_ratio(self, eta, mu):
        return torch.ones_like(eta)

    def complement(self, eta, mu):
        return -torch.expm1(eta)

    def ratios(self, eta, mu):
        return torch.ones_like(eta), mu / (-torch.expm1(eta))


class Logit(Link):
    name = "logit"

    def inverse(self, eta):
        return torch.sigmoid(eta)

    def link(self, mu):
        return torch.logit(mu)

    def derivative(self, eta, mu):
        return mu * torch.sigmoid(-eta)

    def second_ratio(self, eta, mu):
        return 1 - 2 * mu

    def complement(self, eta, mu):
        return torch.sigmoid(-eta)

    def ratios(self, eta, mu):
        return torch.sigmoid(-eta), mu


class Probit(Link):
    name = "probit"

    def inverse(self, eta):
        return _ndtr(eta)

    def link(self, mu):
        return torch.special.ndtri(mu)

    def derivative(self, eta, mu):
        return torch.exp(-0.5 * eta.square()) / _SQRT_2PI

    def second_ratio(self, eta, mu):
        return -eta

    def complement(self, eta, mu):
        return _ndtr(-eta)

    def ratios(self, eta, mu):
        return _mills(eta), _mills(-eta)


class Cloglog(Link):
    name = "cloglog"

    def inverse(self, eta):
        return -torch.expm1(-torch.exp(eta))

    def link(self, mu):
        return torch.log(-torch.log1p(-mu))

    def derivative(self, eta, mu):
        return torch.exp(eta - torch.exp(eta))

    def second_ratio(self, eta, mu):
        return 1 - torch.exp(eta)

    def complement(self, eta, mu):
        return torch.exp(-torch.exp(eta))

    def ratios(self, eta, mu):
        return self.derivative(eta, mu) / mu, torch.exp(eta)


class Loglog(Link):
    name = "loglog"

    def inverse(self, eta):
        return torch.exp(-torch.exp(-eta))

    def link(self, mu):
        return -torch.log(-torch.log(mu))

    def derivative(self, eta, mu):
        return torch.exp(-eta) * mu

    def second_ratio(self, eta, mu):
        return torch.exp(-eta) - 1

    def complement(self, eta, mu):
        return -torch.expm1(-torch.exp(-eta))

    def ratios(self, eta, mu):
        return torch.exp(-eta), self.derivative(eta, mu) / self.complement(eta, mu)


class Power(Link):
    """``mu = eta^(1/a)`` for ``a != 0``; the predictor must be positive."""

    def __init__(self, exponent: float):
        if not math.isfinite(exponent) or exponent == 0:
            raise KernelError("invalid_link", "The power link exponent must be a nonzero number.")
        self.exponent = float(exponent)
        self.name = {-1.0: "reciprocal", -2.0: "inverse_squared"}.get(exponent,
                                                                     f"power({exponent:g})")

    def inverse(self, eta):
        return eta.pow(1 / self.exponent)

    def link(self, mu):
        return mu.pow(self.exponent)

    def derivative(self, eta, mu):
        return mu / (self.exponent * eta)

    def second_ratio(self, eta, mu):
        return (1 / self.exponent - 1) / eta

    def valid(self, eta):
        return torch.isfinite(eta) & (eta > 0)


class NegativeBinomialLink(Link):
    """Canonical negative binomial link ``eta = ln(k mu / (1 + k mu))``, ``eta < 0``."""

    name = "nbinomial"

    def __init__(self, k: float):
        self.k = float(k)

    def inverse(self, eta):
        return torch.exp(eta) / (self.k * (-torch.expm1(eta)))

    def link(self, mu):
        return torch.log(self.k * mu) - torch.log1p(self.k * mu)

    def derivative(self, eta, mu):
        return mu * (1 + self.k * mu)

    def second_ratio(self, eta, mu):
        return 1 + 2 * self.k * mu

    def valid(self, eta):
        return torch.isfinite(eta) & (eta < 0)


def make_link(name: str, *, power: float | None = None, k: float = 1.0) -> Link:
    """A link object by its Stata name (``power`` needs the exponent, ``nbinomial`` k)."""
    if name == "power":
        if power is None:
            raise KernelError("invalid_link", "The power link needs its exponent.")
        if power == 0:
            return Log()
        if power == 1:
            return Identity()
        return Power(power)
    if name == "reciprocal":
        return Power(-1.0)
    if name == "inverse_squared":
        return Power(-2.0)
    if name == "nbinomial":
        return NegativeBinomialLink(k)
    links = {"identity": Identity, "log": Log, "logit": Logit, "probit": Probit,
             "cloglog": Cloglog, "loglog": Loglog}
    if name not in links:
        raise KernelError("invalid_link", f"Unknown link '{name}'.")
    return links[name]()


class Family:
    """Base family on the working scale; ``phi`` is the dispersion of the full likelihood."""

    name = ""
    binomial = False
    default_scale = "x2"           # Stata: Pearson chi2 / df for gaussian, gamma, inverse Gaussian

    def variance(self, mu: Tensor) -> Tensor:
        raise NotImplementedError

    def variance_derivative(self, mu: Tensor) -> Tensor:
        raise NotImplementedError

    def unit_deviance(self, y: Tensor, mu: Tensor, comp: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def unit_pearson(self, y: Tensor, mu: Tensor, comp: Tensor | None = None) -> Tensor:
        return (y - mu).square() / self.variance(mu)

    def log_likelihood(self, y: Tensor, mu: Tensor, phi: float,
                       comp: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def start_mu(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def valid_mu(self, mu: Tensor) -> Tensor:
        return torch.isfinite(mu) & (mu > 0)

    @property
    def text(self) -> str:
        raise NotImplementedError


class Gaussian(Family):
    name, text = "gaussian", "V(mu) = 1"

    def variance(self, mu):
        return torch.ones_like(mu)

    def variance_derivative(self, mu):
        return torch.zeros_like(mu)

    def unit_deviance(self, y, mu, comp=None):
        return (y - mu).square()

    def log_likelihood(self, y, mu, phi, comp=None):
        return -0.5 * ((y - mu).square() / phi + math.log(phi) + _LOG_2PI)

    def start_mu(self, y):
        return y.clone()

    def valid_mu(self, mu):
        return torch.isfinite(mu)


class Binomial(Family):
    """Working scale: proportions ``y/n`` with prior weight ``w n``.

    ``trials`` holds ``n_i`` (ones for Bernoulli and fractional outcomes);
    ``combinatorial`` adds ``ln C(n, y)`` to the log likelihood (not for
    fractional outcomes, whose quasi-likelihood has no such term).
    """

    name, text, binomial, default_scale = "binomial", "V(mu) = mu (1 - mu)", True, 1.0

    def __init__(self, trials: Tensor | None = None, *, combinatorial: bool = True):
        self.trials = trials
        self.combinatorial = combinatorial

    def variance(self, mu):
        return mu * (1 - mu)

    def variance_derivative(self, mu):
        return 1 - 2 * mu

    def unit_deviance(self, y, mu, comp=None):
        comp = 1 - mu if comp is None else comp
        return 2 * (torch.special.xlogy(y, y / mu) + torch.special.xlogy(1 - y, (1 - y) / comp))

    def unit_pearson(self, y, mu, comp=None):
        comp = 1 - mu if comp is None else comp
        return (y - mu).square() / (mu * comp)

    def log_likelihood(self, y, mu, phi, comp=None):
        comp = 1 - mu if comp is None else comp
        n = torch.ones_like(y) if self.trials is None else self.trials
        count = y * n
        value = torch.special.xlogy(count, mu) + torch.special.xlogy(n - count, comp)
        if self.combinatorial:
            value = value + (torch.lgamma(n + 1) - torch.lgamma(count + 1)
                             - torch.lgamma(n - count + 1))
        return value

    def start_mu(self, y):
        n = torch.ones_like(y) if self.trials is None else self.trials
        return (y * n + 0.5) / (n + 1)

    def valid_mu(self, mu):
        # mu may round to 1 while the link's complement 1 - mu is still positive; the
        # fitter checks that complement separately.
        return torch.isfinite(mu) & (mu > 0) & (mu <= 1)


class Poisson(Family):
    name, text, default_scale = "poisson", "V(mu) = mu", 1.0

    def variance(self, mu):
        return mu

    def variance_derivative(self, mu):
        return torch.ones_like(mu)

    def unit_deviance(self, y, mu, comp=None):
        return poisson_deviance(y, mu)

    def log_likelihood(self, y, mu, phi, comp=None):
        return poisson_logmass(y, mu)

    def start_mu(self, y):
        return y + 0.1


class Gamma(Family):
    name, text = "gamma", "V(mu) = mu^2"

    def variance(self, mu):
        return mu.square()

    def variance_derivative(self, mu):
        return 2 * mu

    def unit_deviance(self, y, mu, comp=None):
        return 2 * (-torch.log(y / mu) + (y - mu) / mu)

    def log_likelihood(self, y, mu, phi, comp=None):
        inverse = 1 / phi
        return (-(y / mu + torch.log(mu)) * inverse + (inverse - 1) * torch.log(y)
                - math.log(phi) * inverse - math.lgamma(inverse))

    def start_mu(self, y):
        return y.clone()


class InverseGaussian(Family):
    name, text = "inverse_gaussian", "V(mu) = mu^3"

    def variance(self, mu):
        return mu.pow(3)

    def variance_derivative(self, mu):
        return 3 * mu.square()

    def unit_deviance(self, y, mu, comp=None):
        return (y - mu).square() / (mu.square() * y)

    def log_likelihood(self, y, mu, phi, comp=None):
        return -0.5 * (_LOG_2PI + math.log(phi) + 3 * torch.log(y)
                       + (y - mu).square() / (phi * mu.square() * y))

    def start_mu(self, y):
        return y.clone()


class NegativeBinomial(Family):
    """Negative binomial with FIXED overdispersion ``k`` (Stata ``family(nbinomial #)``)."""

    name, default_scale = "nbinomial", 1.0

    def __init__(self, k: float = 1.0):
        if not math.isfinite(k) or k <= 0:
            raise KernelError("invalid_option", "The negative binomial dispersion must be "
                              "a positive number.")
        self.k = float(k)

    @property
    def text(self) -> str:
        return f"V(mu) = mu + {self.k:g} mu^2"

    def variance(self, mu):
        return mu * (1 + self.k * mu)

    def variance_derivative(self, mu):
        return 1 + 2 * self.k * mu

    def unit_deviance(self, y, mu, comp=None):
        k = self.k
        return 2 * (torch.special.xlogy(y, y / mu)
                    - (y + 1 / k) * (torch.log1p(k * y) - torch.log1p(k * mu)))

    def log_likelihood(self, y, mu, phi, comp=None):
        k = self.k
        validate_nb_precision(y, 1 / k)
        return (torch.lgamma(y + 1 / k) - math.lgamma(1 / k) - torch.lgamma(y + 1)
                + torch.special.xlogy(y, k * mu) - (y + 1 / k) * torch.log1p(k * mu))

    def start_mu(self, y):
        return y + 0.1


def make_family(name: str, *, k: float = 1.0, trials: Tensor | None = None,
                combinatorial: bool = True) -> Family:
    """A family object by its Stata name."""
    if name == "binomial":
        return Binomial(trials, combinatorial=combinatorial)
    if name == "nbinomial":
        return NegativeBinomial(k)
    families = {"gaussian": Gaussian, "poisson": Poisson, "gamma": Gamma,
                "inverse_gaussian": InverseGaussian}
    if name not in families:
        raise KernelError("invalid_option", f"Unknown family '{name}'.")
    return families[name]()
