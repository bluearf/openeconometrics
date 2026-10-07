"""Single-index models of the treatment-effects estimators (tensor kernels).

The outcome and treatment models of ``teffects`` are index models
``E[y | x] = m(x'b)``. Every piece the stacked estimating equations need is
the product of a covariate row and a per-observation scalar, so the models
expose those scalars as tensors:

=========  ====================  ======================  ============================
model      mean m(eta)           ML residual r(y, eta)   dr/deta
=========  ====================  ======================  ============================
linear     eta                   y - eta                 -1
logit      L(eta)                y - L(eta)              -L(1 - L)
probit     Phi(eta)              y l(eta) - (1-y) l(-eta) -y v(eta) - (1-y) v(-eta)
poisson    exp(eta)              y - exp(eta)            -exp(eta)
=========  ====================  ======================  ============================

with ``l(t) = phi(t)/Phi(t)`` and ``v(t) = l(t)(l(t) + t)``. The score of the
(quasi-)likelihood is ``x r`` and its derivative ``x x' dr/deta``. For logit
and probit an outcome in [0, 1] is allowed: a fractional outcome gives the
Bernoulli quasi-likelihood (Stata's ``flogit``/``fprobit`` outcome models),
whose estimating equation has the same form.

Treatment models are the binary logit/probit (two levels) and the
multinomial logit (more levels, Stata's multivalued ``tmodel(logit)``).
``TreatmentFit`` stores, for ``S = L - 1`` non-control equations and ``L``
levels in the order control first:

* ``probabilities`` [n, L]: p_l(x);
* ``residual`` [n, S]: generalized residuals u_s, score of equation s = x u_s;
* ``curvature`` [n, S, S]: h_sr with d(x u_s)/d gamma_r' = x x' h_sr;
* ``slopes`` [n, L, S]: a_ls with d p_l / d gamma_s = a_ls x.

No autograd is used: every derivative is analytic and checked against
numerical derivatives in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete.common import maximize
from openecon.econometrics.discrete.kernels import MultinomialObjective, mills_ratio
from openecon.engines.contracts import KernelError

OUTCOME_MODELS = ("linear", "logit", "probit", "poisson")
TREATMENT_MODELS = ("logit", "probit")
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
# Fitted probabilities / means closer than this to their bound count as degenerate.
_DEGENERATE = 1e-10


def mean_pieces(model: str, eta: Tensor) -> tuple[Tensor, Tensor]:
    """The mean m(eta) and its derivative dm/deta."""
    if model == "linear":
        return eta, torch.ones_like(eta)
    if model == "logit":
        p = torch.sigmoid(eta)
        return p, p * (1 - p)
    if model == "probit":
        return torch.special.ndtr(eta), torch.exp(-0.5 * eta.square() - _LOG_SQRT_2PI)
    if model == "poisson":
        mu = torch.exp(eta)
        return mu, mu
    raise KernelError("invalid_model", f"Unknown index model '{model}'.")


def residual_pieces(model: str, y: Tensor, eta: Tensor) -> tuple[Tensor, Tensor]:
    """The ML residual r(y, eta) (score = x r) and its derivative dr/deta."""
    if model == "linear":
        return y - eta, -torch.ones_like(eta)
    if model == "logit":
        p = torch.sigmoid(eta)
        return y - p, -p * (1 - p)
    if model == "probit":
        _, upper, v_upper = mills_ratio(eta)
        _, lower, v_lower = mills_ratio(-eta)
        return y * upper - (1 - y) * lower, -y * v_upper - (1 - y) * v_lower
    if model == "poisson":
        mu = torch.exp(eta)
        return y - mu, -mu
    raise KernelError("invalid_model", f"Unknown index model '{model}'.")


def log_likelihood(model: str, y: Tensor, eta: Tensor) -> Tensor:
    """Per-observation (quasi-)log likelihood up to terms free of eta."""
    if model == "linear":
        return -0.5 * (y - eta).square()
    if model == "logit":
        return y * eta - torch.nn.functional.softplus(eta)
    if model == "probit":
        return y * torch.special.log_ndtr(eta) + (1 - y) * torch.special.log_ndtr(-eta)
    if model == "poisson":
        return y * eta - torch.exp(eta)
    raise KernelError("invalid_model", f"Unknown index model '{model}'.")


class IndexObjective:
    """Weighted (quasi-)log likelihood sum_i w_i l(y_i, x_i'b) with analytic derivatives."""

    def __init__(self, model: str, x: Tensor, y: Tensor, weights: Tensor):
        self.model, self.x, self.y, self.w = model, x, y, weights

    def value(self, theta: Tensor) -> Tensor:
        return (self.w * log_likelihood(self.model, self.y, self.x @ theta)).sum()

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self.x @ theta
        r, dr = residual_pieces(self.model, self.y, eta)
        value = (self.w * log_likelihood(self.model, self.y, eta)).sum()
        gradient = self.x.T @ (self.w * r)
        hessian = (self.x * (self.w * dr)[:, None]).T @ self.x
        return value, gradient, (hessian + hessian.T) / 2


def _start(model: str, y: Tensor, weights: Tensor, k: int) -> Tensor:
    """Constant-only starting values (column 0 is the constant of a centred design)."""
    start = torch.zeros(k, dtype=torch.float64)
    mean = float((weights * y).sum() / weights.sum())
    if model == "logit":
        start[0] = math.log(mean / (1 - mean))
    elif model == "probit":
        start[0] = float(torch.special.ndtri(torch.tensor(mean, dtype=torch.float64)))
    elif model == "poisson":
        start[0] = math.log(mean)
    return start


def fit_index(model: str, x: Tensor, y: Tensor, weights: Tensor, *, what: str) -> Tensor:
    """Maximum (quasi-)likelihood coefficients of one index model.

    ``x`` has the constant in column 0 and centred slopes; ``weights`` are the
    observation weights of the sums (user weights times any inverse-probability
    weight). Linear models are solved by QR least squares; the others by
    Newton-Raphson on the likelihood scaled by the mean weight. A monotone
    likelihood (outcome predicted perfectly) raises ``separation_detected``.
    """
    from openecon.engines.linalg import least_squares

    if model == "linear":
        try:
            fit = least_squares(x, y, weights, drop_collinear=False, tol=0.0)
        except KernelError as exc:
            raise AnalysisError(exc.code, f"The {what} could not be fitted: {exc}") from exc
        return fit.beta
    mean = float((weights * y).sum() / weights.sum())
    if model in {"logit", "probit"} and not 0 < mean < 1:
        raise AnalysisError("separation_detected", f"The {what} has an outcome that is always "
                            f"{0 if mean <= 0 else 1}; the {model} model is not identified.")
    if model == "poisson" and mean <= 0:
        raise AnalysisError("separation_detected", f"The {what} has an outcome that is always "
                            "zero; the Poisson model is not identified.")
    objective = IndexObjective(model, x, y, weights)
    scale = 1.0 / float(weights.mean())
    result = maximize(objective, _start(model, y, weights, x.shape[1]), what=what, scale=scale)
    if not result.converged:
        mu, _ = mean_pieces(model, x @ result.theta)
        degenerate = (mu < _DEGENERATE) | ((mu > 1 - _DEGENERATE) if model != "poisson"
                                           else torch.zeros_like(mu, dtype=torch.bool))
        if bool(degenerate.any()):
            raise AnalysisError("separation_detected", f"The {what} predicts "
                                f"{int(degenerate.sum())} observation(s) perfectly and its "
                                "likelihood keeps increasing, so finite estimates do not exist. "
                                "Remove the covariate that separates the outcome.")
        raise AnalysisError("nonconvergence", f"The {what} did not converge: "
                            f"{result.diagnostics.get('message')}")
    return result.theta


@dataclass
class TreatmentFit:
    model: str                 # "logit" or "probit" (multinomial logit when L > 2)
    gamma: Tensor              # [S, kt] coefficients of the non-control equations
    probabilities: Tensor      # [n, L], control first
    residual: Tensor           # [n, S]
    curvature: Tensor          # [n, S, S]
    slopes: Tensor             # [n, L, S]
    log_likelihood: float


def treatment_pieces(model: str, x: Tensor, codes: Tensor, levels: int, gamma: Tensor,
                     weights: Tensor | None = None) -> TreatmentFit:
    """Probabilities and derivative scalars of a treatment model at ``gamma`` [S, kt]."""
    n = x.shape[0]
    weights = torch.ones(n, dtype=torch.float64) if weights is None else weights
    if levels == 2:
        eta = x @ gamma[0]
        d = (codes == 1).to(torch.float64)
        p, f = mean_pieces(model, eta)
        r, dr = residual_pieces(model, d, eta)
        probabilities = torch.stack([1 - p, p], dim=1)
        if model == "probit":
            probabilities[:, 0] = torch.special.ndtr(-eta)
        slopes = torch.stack([-f, f], dim=1)[:, :, None]
        ll = float((weights * log_likelihood(model, d, eta)).sum())
        return TreatmentFit(model, gamma, probabilities, r[:, None], dr[:, None, None], slopes, ll)
    eta = torch.cat([torch.zeros((n, 1), dtype=torch.float64), x @ gamma.T], dim=1)
    log_p = eta - torch.logsumexp(eta, dim=1, keepdim=True)
    probabilities = torch.exp(log_p)
    onehot = torch.zeros((n, levels), dtype=torch.float64)
    onehot[torch.arange(n), codes] = 1.0
    others = probabilities[:, 1:]
    residual = onehot[:, 1:] - others
    eye = torch.eye(levels - 1, dtype=torch.float64)
    curvature = -others[:, :, None] * (eye[None] - others[:, None, :])
    full_eye = torch.eye(levels, dtype=torch.float64)[:, 1:]
    slopes = probabilities[:, :, None] * (full_eye[None] - others[:, None, :])
    ll = float((weights * log_p.gather(1, codes[:, None])[:, 0]).sum())
    return TreatmentFit("logit", gamma, probabilities, residual, curvature, slopes, ll)


def fit_treatment(model: str, x: Tensor, codes: Tensor, levels: int, weights: Tensor
                  ) -> TreatmentFit:
    """Maximum-likelihood treatment model: binary logit/probit or multinomial logit.

    ``codes`` are 0 for the control level and 1..L-1 for the others. A treatment
    predicted perfectly by the covariates is a violation of overlap.
    """
    kt = x.shape[1]
    if levels == 2:
        d = (codes == 1).to(torch.float64)
        try:
            theta = fit_index(model, x, d, weights, what="treatment model")
        except AnalysisError as exc:
            if exc.code != "separation_detected":
                raise
            raise AnalysisError("overlap_violation", "The treatment model predicts treatment "
                                "perfectly for some observations (propensity scores of 0 or 1), "
                                "so the overlap assumption fails. Remove or coarsen the covariate "
                                "that separates treated and control observations.") from exc
        return treatment_pieces(model, x, codes, levels, theta[None, :], weights)
    if model != "logit":
        raise AnalysisError("unsupported_model", "A treatment with more than two levels needs "
                            "tmodel='logit' (multinomial logit), as in Stata.")
    objective = MultinomialObjective(x, codes, levels, 0, weights)
    start = torch.zeros(objective.size, dtype=torch.float64)
    shares = torch.zeros(levels, dtype=torch.float64).index_add_(0, codes, weights)
    if bool((shares <= 0).any()):
        raise AnalysisError("invalid_treatment", "Every treatment level needs observations.")
    start.view(levels - 1, kt)[:, 0] = torch.log(shares[1:] / shares[0])
    result = maximize(objective, start, what="multinomial treatment model",
                      scale=1.0 / float(weights.mean()))
    fit = treatment_pieces(model, x, codes, levels, result.theta.reshape(levels - 1, kt), weights)
    if not result.converged:
        if bool((fit.probabilities < _DEGENERATE).any()):
            raise AnalysisError("overlap_violation", "The multinomial treatment model drives "
                                "some treatment probabilities to zero, so the overlap assumption "
                                "fails. Remove or coarsen the covariate that separates the levels.")
        raise AnalysisError("nonconvergence", "The multinomial treatment model did not converge: "
                            f"{result.diagnostics.get('message')}")
    return fit


def treatment_objective(model: str, x: Tensor, codes: Tensor, levels: int, weights: Tensor):
    """The objective maximized by ``fit_treatment`` (for derivative checks in tests)."""
    if levels == 2:
        return IndexObjective(model, x, (codes == 1).to(torch.float64), weights)
    return MultinomialObjective(x, codes, levels, 0, weights)
