"""Per-observation score contributions of fitted models, for ``suest``.

Results do not store their scores, so ``suest`` rebuilds them: the model's
design is reconstructed from ``result.spec`` and the data (same treatment
coding, same estimation rows, the reported terms only), and the log-likelihood
score of every observation and the Hessian are evaluated at the reported
estimates. Each provider returns a :class:`ModelScores`.

Supported score families (fweight/pweight/aweight likelihood semantics where the fit supports them):

- ``ols``: the Gaussian likelihood in ``(b, ln sigma^2)`` with the maximum
  likelihood variance ``sigma^2 = SSR/N``, as Stata's suest treats
  ``regress`` (equations ``mean`` and ``lnvar``):
  ``s_b = x u / sigma^2``, ``s_lnvar = (u^2/sigma^2 - 1) / 2``; the
  information is block diagonal with ``X'X / sigma^2`` and ``N/2``.
- ``logit``: ``s = (y - L(x'b)) x``, ``H = -sum L(1-L) x x'``.
- ``probit``: with ``q = 2y - 1`` and ``r = phi(q x'b) / Phi(q x'b)``:
  ``s = q r x``, ``H = -sum r (r + q x'b) x x'``.
- ``poisson`` (offset/exposure allowed): ``s = (y - mu) x``,
  ``H = -sum mu x x'``.
- ``cloglog`` and ``fracreg`` logit/probit: Bernoulli likelihood/QML analytic
  unit scores and observed Hessians, including fractional endpoints/interiors.
- ``ologit`` / ``oprobit`` and ``mlogit``: the family kernels' analytic scores
  and Hessians (``discrete.kernels``).

The score sum must vanish at the reported estimates. ``suest`` checks the
Newton decrement ``g' (-H)^-1 g``; a value above 1e-6 means the design could
not be reproduced (e.g. formula transforms) and raises ``suest_mismatch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _frame_hasher, _position_bytes
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, kernel_call
from openecon.econometrics.postest.common import coefficient_vector
from openecon.engines.optimize import information_inverse
from openecon.models import ResultBundle
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace, tensor_bytes

SUPPORTED = ("ols", "logit", "probit", "poisson", "ologit", "oprobit", "mlogit",
             "glm", "nbreg", "tobit", "intreg", "truncreg", "cloglog", "fracreg")
DECREMENT_TOLERANCE = 1e-6
BINOMIAL_COMMANDS = {"cloglog", "fracreg"}
# Admission bounds dense reconstruction work, independently of the byte budget.
MAX_BINOMIAL_SCORE_WORK = 100_000_000


def binomial_score_plan(n: int, p: int):
    work = n * p * p + p ** 3
    if work > MAX_BINOMIAL_SCORE_WORK:
        raise AnalysisError("suest_work_limit", "Saved binomial score reconstruction exceeds "
                            f"the {MAX_BINOMIAL_SCORE_WORK:,}-operation admission bound "
                            "(N*P^2 + P^3). Reduce the dense model dimensions.")
    return plan_workspace("saved binomial suest scores", {
        "design_score_copies": tensor_bytes((n, p)) * 4,
        "likelihood_vectors": tensor_bytes((n,)) * 24,
        "information_copies": tensor_bytes((p, p)) * 8,
    })


def _binomial_information_error():
    return AnalysisError("singular_information", "The saved binomial information is "
                         "singular at float64 precision in the reported coordinates. "
                         "Center/rescale the predictors and refit before suest.")


def _binomial_information_guard(objective, theta):
    """Assess observed information before a rounded Gram matrix hides its rank.

    Its square-root design has rows sqrt(-w_i h_i) x_i. Column normalization
    preserves predictor-unit and likelihood-weight scale invariance. Squared
    singular-value ratios are the normalized information's eigenvalue ratios;
    p*eps is the existing information solver's working-precision rank scale.
    """
    state = objective.state(theta)
    if state is None:
        raise AnalysisError("suest_mismatch", "GLM estimates leave the family/link domain.")
    curvature = -objective.prior * objective.pieces(state)[1]
    if not bool(torch.isfinite(curvature).all()) or bool((curvature < 0).any()):
        raise _binomial_information_error()
    factor = objective.x * curvature.sqrt()[:, None]
    norms = torch.linalg.vector_norm(factor, dim=0)
    if not len(norms) or not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
        raise _binomial_information_error()
    factor /= norms
    singular = torch.linalg.svdvals(factor)
    if (len(singular) != factor.shape[1] or not bool(torch.isfinite(singular).all())
            or float((singular[-1] / singular[0]).square())
            <= factor.shape[1] * torch.finfo(torch.float64).eps):
        raise _binomial_information_error()


@dataclass
class ModelScores:
    terms: list[str]                 # parameter names as the model reports them
    equations: list[str | None]      # equation labels as the model reports them
    params: Tensor                   # [p]
    scores: Tensor                   # [n_i, p] per-observation log-likelihood scores
    hessian: Tensor                  # [p, p] Hessian of the log likelihood
    rows: Tensor                     # int64 [n_i] data positions of the score rows

    def bread(self) -> Tensor:
        return kernel_call(information_inverse, -self.hessian)

    def decrement(self) -> float:
        gradient = self.scores.sum(dim=0)
        return float(gradient @ self.bread() @ gradient)


def _unsupported(result: ResultBundle, reason: str) -> AnalysisError:
    return AnalysisError("suest_unsupported", f"suest cannot rebuild the scores of the "
                         f"{result.spec.estimator} fit of {result.spec.outcome!r}: {reason}. "
                         f"Supported score providers: {', '.join(SUPPORTED)}.")


def _frame(result: ResultBundle, data: pd.DataFrame, positions: list[int]) -> ModelFrame:
    """The estimation sample of ``result`` (data rows ``positions``) rebuilt from the data."""
    allow_missing = []
    # Upper roles may be stored as one column name rather than a list.
    if result.spec.estimator == "intreg":
        from openecon.econometrics.registry import role_columns
        allow_missing = [result.spec.outcome, *role_columns(result.spec, "upper")]
    frame = ModelFrame(result.spec, data, allow_missing=allow_missing)
    wanted = torch.tensor(sorted(positions), dtype=torch.int64)
    current = torch.tensor(frame.positions, dtype=torch.int64)
    keep = torch.isin(current, wanted)
    frame.restrict(keep)
    if frame.n != len(wanted):
        raise AnalysisError("suest_mismatch", f"The estimation rows of the "
                            f"{result.spec.estimator} fit could not be rebuilt from the data.")
    return frame


def _columns(result: ResultBundle, frame: ModelFrame, terms: list[str], *,
             intercept: bool | None = None) -> Tensor:
    design = frame.design(intercept=intercept)
    index = {term: i for i, term in enumerate(design.terms)}
    absent = [term for term in terms if term not in index]
    if absent:
        raise _unsupported(result, f"the terms {', '.join(absent)} are not plain columns of the "
                           "data (formula transforms and lags are not supported; create them as "
                           "columns first)")
    return design.x[:, [index[term] for term in terms]]


def _rows(frame: ModelFrame) -> Tensor:
    return torch.tensor(frame.positions, dtype=torch.int64)


class GaussianObjective:
    """Gaussian log likelihood of a linear model in ``theta = (b, ln sigma^2)``."""

    def __init__(self, x: Tensor, y: Tensor, weights: Tensor | None = None):
        self.x, self.y = x, y
        self.w = torch.ones(len(y), dtype=torch.float64) if weights is None else weights

    def _pieces(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        return self.y - self.x @ theta[:-1], torch.exp(theta[-1])

    def score_rows(self, theta: Tensor) -> Tensor:
        resid, sigma2 = self._pieces(theta)
        return torch.cat([self.x * (resid / sigma2)[:, None],
                          (0.5 * (resid.square() / sigma2 - 1.0))[:, None]], dim=1)

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        resid, sigma2 = self._pieces(theta)
        k = self.x.shape[1]
        n = self.w.sum()
        ssr = (self.w * resid.square()).sum()
        value = -0.5 * (n * (math.log(2 * math.pi) + theta[-1]) + ssr / sigma2)
        hessian = torch.empty((k + 1, k + 1), dtype=torch.float64)
        hessian[:k, :k] = -(self.x.T @ (self.w[:, None] * self.x)) / sigma2
        hessian[:k, k] = -(self.x.T @ (self.w * resid)) / sigma2
        hessian[k, :k] = hessian[:k, k]
        hessian[k, k] = -0.5 * ssr / sigma2
        return value, (self.w[:, None] * self.score_rows(theta)).sum(dim=0), hessian


class LogitObjective:
    """Binary logit log likelihood ``sum y ln L + (1 - y) ln(1 - L)``, ``L = 1/(1 + e^-x'b)``."""

    def __init__(self, x: Tensor, y: Tensor, weights: Tensor | None = None):
        self.x, self.y = x, y
        self.w = torch.ones(len(y), dtype=torch.float64) if weights is None else weights

    def score_rows(self, theta: Tensor) -> Tensor:
        return self.x * (self.y - torch.sigmoid(self.x @ theta))[:, None]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self.x @ theta
        probability = torch.sigmoid(eta)
        value = (self.w * (self.y * eta - torch.nn.functional.softplus(eta))).sum()
        hessian = -(self.x * (self.w * probability * (1 - probability))[:, None]).T @ self.x
        return value, (self.w[:, None] * self.score_rows(theta)).sum(dim=0), hessian


class ProbitObjective:
    """Binary probit log likelihood ``sum ln Phi(q x'b)`` with ``q = 2y - 1``."""

    def __init__(self, x: Tensor, y: Tensor, weights: Tensor | None = None):
        self.x, self.sign = x, 2 * y - 1
        self.w = torch.ones(len(y), dtype=torch.float64) if weights is None else weights

    def _mills(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from openecon.econometrics.discrete.kernels import mills_ratio

        return mills_ratio(self.sign * (self.x @ theta))

    def score_rows(self, theta: Tensor) -> Tensor:
        return self.x * (self.sign * self._mills(theta)[1])[:, None]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_cdf, ratio, curvature = self._mills(theta)
        gradient = self.x.T @ (self.w * self.sign * ratio)
        return (self.w * log_cdf).sum(), gradient, -(self.x * (self.w * curvature)[:, None]).T @ self.x


class PoissonObjective:
    """Poisson log likelihood ``sum y eta - e^eta - ln y!`` with ``eta = x'b + offset``."""

    def __init__(self, x: Tensor, y: Tensor, offset: Tensor | None, weights: Tensor | None = None):
        self.x, self.y, self.offset = x, y, offset
        self.w = torch.ones(len(y), dtype=torch.float64) if weights is None else weights

    def _eta(self, theta: Tensor) -> Tensor:
        eta = self.x @ theta
        return eta if self.offset is None else eta + self.offset

    def score_rows(self, theta: Tensor) -> Tensor:
        return self.x * (self.y - torch.exp(self._eta(theta)))[:, None]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self._eta(theta)
        mu = torch.exp(eta)
        value = (self.w * (self.y * eta - mu - torch.lgamma(self.y + 1))).sum()
        gradient = self.x.T @ (self.w * (self.y - mu))
        return value, gradient, -(self.x * (self.w * mu)[:, None]).T @ self.x


def _evaluate(objective, theta: Tensor, terms: list[str], equations: list[str | None],
              frame: ModelFrame) -> ModelScores:
    _, _, hessian = kernel_call(objective, theta)
    scores = kernel_call(objective.score_rows, theta) * _weights(frame)[:, None]
    return ModelScores(terms, equations, theta, scores, hessian, _rows(frame))


def _weights(frame: ModelFrame) -> Tensor:
    from openecon.econometrics.discrete.common import likelihood_weights

    return likelihood_weights(frame).user


def _single(result: ResultBundle, data: pd.DataFrame,
            positions: list[int]) -> tuple[ModelFrame, Tensor, Tensor]:
    frame = _frame(result, data, positions)
    x = _columns(result, frame, [c.term for c in result.coefficients])
    return frame, x, frame.numeric(result.spec.outcome)


def ols_scores(result: ResultBundle, data: pd.DataFrame,
               positions: list[int]) -> ModelScores:
    frame, x, y = _single(result, data, positions)
    beta = coefficient_vector(result)
    w = _weights(frame)
    sigma2 = float((w * (y - x @ beta).square()).sum() / w.sum())
    if not sigma2 > 0:
        raise _unsupported(result, "the fit is exact (zero residual variance)")
    terms = [c.term for c in result.coefficients]
    theta = torch.cat([beta, torch.tensor([math.log(sigma2)], dtype=torch.float64)])
    return _evaluate(GaussianObjective(x, y, w), theta, [*terms, "/lnvar"],
                     ["mean"] * len(terms) + ["lnvar"], frame)


def _reported(result: ResultBundle) -> tuple[Tensor, list[str], list[str | None]]:
    return (coefficient_vector(result), [c.term for c in result.coefficients],
            [c.equation for c in result.coefficients])


def logit_scores(result: ResultBundle, data: pd.DataFrame,
                 positions: list[int]) -> ModelScores:
    frame, x, y = _single(result, data, positions)
    return _evaluate(LogitObjective(x, y, _weights(frame)), *_reported(result), frame)


def probit_scores(result: ResultBundle, data: pd.DataFrame,
                  positions: list[int]) -> ModelScores:
    frame, x, y = _single(result, data, positions)
    return _evaluate(ProbitObjective(x, y, _weights(frame)), *_reported(result), frame)


def poisson_scores(result: ResultBundle, data: pd.DataFrame,
                   positions: list[int]) -> ModelScores:
    from openecon.econometrics.glm.common import linear_offset

    frame, x, y = _single(result, data, positions)
    return _evaluate(PoissonObjective(x, y, linear_offset(frame), _weights(frame)), *_reported(result), frame)


def ordered_scores(result: ResultBundle, data: pd.DataFrame,
                   positions: list[int]) -> ModelScores:
    from openecon.econometrics.discrete.common import offset_column
    from openecon.econometrics.discrete.kernels import OrderedObjective
    from openecon.econometrics.discrete.ordered import ordered_outcome

    frame = _frame(result, data, positions)
    command = result.spec.estimator
    slopes = [c.term for c in result.coefficients if not c.term.startswith("/")]
    cuts = [c.term for c in result.coefficients if c.term.startswith("/cut")]
    if len(slopes) + len(cuts) != len(result.coefficients):
        raise _unsupported(result, "unexpected parameters")
    x = _columns(result, frame, slopes, intercept=False)
    codes, labels = ordered_outcome(frame, command)
    if len(labels) - 1 != len(cuts):
        raise AnalysisError("suest_mismatch", f"The outcome categories of the {command} fit "
                            "could not be rebuilt from the data.")
    link = "logit" if command == "ologit" else "probit"
    weights = _weights(frame)
    objective = kernel_call(OrderedObjective, x, codes, len(labels), weights,
                            offset_column(frame), link)
    return _evaluate(objective, *_reported(result), frame)


class MultinomialScores:
    """Score rows of the multinomial logit (``discrete.kernels.MultinomialObjective``).

    With ``p_ij`` the probability of non-base category j, the score of
    observation i for equation j is ``(1{y_i = j} - p_ij) x_i``; the value,
    gradient and Hessian come from the family kernel.
    """

    def __init__(self, x: Tensor, codes: Tensor, n_categories: int, base: int,
                 weights: Tensor | None = None):
        from openecon.econometrics.discrete.kernels import MultinomialObjective

        weights = torch.ones(len(codes), dtype=torch.float64) if weights is None else weights
        self.kernel = kernel_call(MultinomialObjective, x, codes, n_categories, base, weights)
        self.x, self.equations = x, n_categories - 1
        slot = torch.arange(n_categories)
        slot = torch.where(slot > base, slot - 1, slot)
        slot[base] = self.equations
        self.own = slot[codes]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.kernel(theta)

    def score_rows(self, theta: Tensor) -> Tensor:
        n, k = self.x.shape
        m = self.equations
        eta = torch.cat([self.x @ theta.reshape(m, k).T,
                         torch.zeros((n, 1), dtype=torch.float64)], dim=1)
        residual = -torch.softmax(eta, dim=1)[:, :m]
        chosen = self.own < m
        residual[torch.arange(n)[chosen], self.own[chosen]] += 1.0
        return (residual[:, :, None] * self.x[:, None, :]).reshape(n, m * k)


def mlogit_scores(result: ResultBundle, data: pd.DataFrame,
                  positions: list[int]) -> ModelScores:
    from openecon.econometrics.discrete.common import label_text
    from openecon.econometrics.discrete.mlogit import multinomial_outcome

    frame = _frame(result, data, positions)
    codes, labels = multinomial_outcome(frame, "mlogit")
    base = result.extra.get("base")
    names = [label_text(value) for value in labels]
    if base is None or label_text(base) not in names:
        raise AnalysisError("suest_mismatch", "The base outcome of the mlogit fit could not be "
                            "identified in the data.")
    equations = [name for name in names if name != label_text(base)]
    first = [c.term for c in result.coefficients if c.equation == equations[0]]
    base_terms = [term.split(":", 1)[1] for term in first]
    expected = [f"{equation}:{term}" for equation in equations for term in base_terms]
    if [c.term for c in result.coefficients] != expected:
        raise _unsupported(result, "the equations do not share one design")
    x = _columns(result, frame, base_terms)
    objective = MultinomialScores(x, codes, len(labels), names.index(label_text(base)), _weights(frame))
    return _evaluate(objective, *_reported(result), frame)


class GlmScores:
    """Unit-dispersion estimating scores; dispersion cancels from the sandwich."""

    def __init__(self, objective):
        self.objective = objective

    def __call__(self, theta):
        return self.objective(theta)

    def score_rows(self, theta):
        state = self.objective.state(theta)
        if state is None:
            raise AnalysisError("suest_mismatch", "GLM estimates leave the family/link domain.")
        return self.objective.score_rows(state)


def glm_scores(result, data, positions):
    from openecon.econometrics.glm.families import make_family, make_link
    from openecon.econometrics.glm.glm import _outcome, _settings
    from openecon.econometrics.glm.common import linear_offset
    from openecon.econometrics.glm.kernels import GlmObjective

    frame, x, _ = _single(result, data, positions)
    settings = _settings(frame)
    y, trials = _outcome(frame, settings["family"])
    family = make_family(settings["family"], k=settings["k"], trials=trials)
    link = make_link(settings["link"], power=settings["power"], k=settings["k"])
    prior = _weights(frame) if trials is None else _weights(frame) * trials
    objective = GlmObjective(x, y, prior, linear_offset(frame), family, link, trials)
    return _evaluate(GlmScores(objective), *_reported(result), frame)


def binomial_command_scores(result, data, positions):
    """Observed Bernoulli/QML derivatives, including fractional endpoints.

    Fractional probit requires y*log(Phi(eta)) + (1-y)*log(Phi(-eta));
    the binary probit's sign shortcut is invalid for interior responses.
    """
    from openecon.econometrics.glm.common import likelihood_weights, linear_offset
    from openecon.econometrics.glm.families import Binomial, make_link
    from openecon.econometrics.glm.kernels import GlmObjective

    kind = result.spec.estimator
    if kind == "cloglog" and result.spec.weight_type == "aweight":
        raise _unsupported(result, "cloglog does not fit analytic weights")
    allowed_roles = {"offset"} if kind == "cloglog" else set()
    if any(value and role not in allowed_roles for role, value in result.spec.columns.items()):
        raise _unsupported(result, "the saved columns include roles outside the native command")
    link = "cloglog" if kind == "cloglog" else result.spec.options.get("link", "logit")
    if link not in ({"cloglog"} if kind == "cloglog" else {"logit", "probit"}):
        raise _unsupported(result, "the saved link is outside the command's score domain")
    if (result.provenance.get("estimator") != kind or result.extra.get("link") != link
            or (kind == "fracreg" and result.extra.get("quasi_likelihood") is not True)):
        raise AnalysisError("suest_mismatch", "Saved estimator/link metadata do not describe "
                            "the fitted binomial command.")
    if result.provenance.get("solver_diagnostics", {}).get("converged") is not True:
        raise AnalysisError("suest_mismatch", "Saved binomial scores require a recorded "
                            "converged fit; refit historical results lacking this record.")
    if (not positions or positions != result.sample_positions or len(set(positions)) != len(positions)
            or any(isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= len(data)
                   for row in positions)):
        raise AnalysisError("suest_mismatch", "Saved binomial estimation positions are invalid "
                            "or do not match the requested sample.")
    binomial_score_plan(len(positions), len(result.coefficients))
    # Native binomial commands use every complete, positive-weight row. A
    # stationary subset can have the same coefficients while changing the
    # information and sandwich, so stationarity alone cannot verify this sample.
    frame = ModelFrame(result.spec, data)
    # Omitted columns and unused category levels still incur allocation/QR
    # work. Admit the exact unscreened width before either operation starts.
    binomial_score_plan(frame.n, frame.design_width())
    weights = likelihood_weights(frame)
    sample_hasher = _frame_hasher(frame.sample)
    sample_hasher.update(_position_bytes(frame.positions))
    if (frame.positions != positions or result.nobs != weights.nobs
            or result.nobs_original != len(frame.original)
            or result.dropped_rows != len(frame.original) - frame.n
            or result.provenance.get("sample_hash") != sample_hasher.hexdigest()):
        raise AnalysisError("suest_mismatch", "The saved binomial estimation sample, "
                            "counts or sample hash do not match the native fitted sample.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    binomial_score_plan(frame.n, len(design.terms))
    terms = [c.term for c in result.coefficients]
    if (len(terms) != len(set(terms)) or set(terms) != set(design.terms)
            or result.provenance.get("design_terms") != terms
            or result.provenance.get("categorical_encoding") != design.categories
            or result.provenance.get("omitted_terms") != frame.notes.get("omitted_terms", [])
            or any(c.equation is not None for c in result.coefficients)):
        raise AnalysisError("suest_mismatch", "The complete saved binomial design, categorical "
                            "coding or omitted terms could not be reproduced.")
    index = {term: i for i, term in enumerate(design.terms)}
    x = design.x[:, [index[term] for term in terms]]
    y = frame.numeric(result.spec.outcome)
    valid = (y == 0) | (y == 1) if kind == "cloglog" else (y >= 0) & (y <= 1)
    if not bool(valid.all()) or bool((y == y[0]).all()):
        raise AnalysisError("suest_mismatch", "The saved binomial outcome is outside the fitted "
                            "command's nonconstant response domain.")
    objective = GlmObjective(x, y, _weights(frame), linear_offset(frame),
                             Binomial(combinatorial=False), make_link(link))
    reported = _reported(result)
    _binomial_information_guard(objective, reported[0])
    return _evaluate(GlmScores(objective), *reported, frame)


def nbreg_scores(result, data, positions):
    from openecon.econometrics.glm.common import linear_offset
    from openecon.econometrics.glm.kernels import NegativeBinomialObjective

    frame = _frame(result, data, positions)
    x = _columns(result, frame, [c.term for c in result.coefficients if not c.term.startswith("/")])
    objective = NegativeBinomialObjective(x, frame.numeric(result.spec.outcome), _weights(frame),
                                         linear_offset(frame), frame.option("dispersion"))
    return _evaluate(objective, *_reported(result), frame)


class SigmaScores:
    """Exact change of coordinates from log sigma to reported positive sigma."""

    def __init__(self, objective):
        self.objective = objective

    def _theta(self, theta):
        if not float(theta[-1]) > 0:
            raise AnalysisError("suest_mismatch", "Reported sigma must be positive.")
        return torch.cat([theta[:-1], theta[-1:].log()])

    def score_rows(self, theta):
        rows = self.objective.score_rows(self._theta(theta)).clone()
        rows[:, -1] /= theta[-1]
        return rows

    def __call__(self, theta):
        value, gradient, hessian = self.objective(self._theta(theta))
        jacobian = torch.ones_like(theta)
        jacobian[-1] = 1 / theta[-1]
        converted = hessian * jacobian[:, None] * jacobian[None, :]
        converted[-1, -1] -= gradient[-1] / theta[-1].square()
        return value, gradient * jacobian, converted


def limited_scores(result, data, positions):
    from openecon.econometrics.discrete.common import likelihood_weights, offset_column
    from openecon.econometrics.limited.common import resolve_limits
    from openecon.econometrics.limited.censored import _bound, censoring_bounds
    from openecon.econometrics.limited.kernels import CensoredObjective, TruncatedObjective

    frame = _frame(result, data, positions)
    x = _columns(result, frame, [c.term for c in result.coefficients if not c.term.startswith("/")])
    w, offset = _weights(frame), offset_column(frame)
    if result.spec.estimator == "intreg":
        low, high = _bound(frame, result.spec.outcome), _bound(frame, frame.role("upper")[0])
        low = torch.where(torch.isnan(low), -torch.inf, low)
        high = torch.where(torch.isnan(high), torch.inf, high)
        objective = CensoredObjective(x, low, high, w, offset)
    else:
        y = frame.numeric(result.spec.outcome)
        ll, ul = resolve_limits(frame, y, result.spec.estimator)
        if result.spec.estimator == "tobit":
            low, high, *_ = censoring_bounds(y, ll, ul, likelihood_weights(frame), "tobit")
            objective = CensoredObjective(x, low, high, w, offset)
        else:
            objective = TruncatedObjective(x, y, w, ll, ul, offset)
    reported = objective if result.coefficients[-1].term == "/lnsigma" else SigmaScores(objective)
    return _evaluate(reported, *_reported(result), frame)


_PROVIDERS = {"ols": ols_scores, "logit": logit_scores, "probit": probit_scores,
              "poisson": poisson_scores, "ologit": ordered_scores, "oprobit": ordered_scores,
              "mlogit": mlogit_scores, "glm": glm_scores, "nbreg": nbreg_scores,
              "tobit": limited_scores, "intreg": limited_scores, "truncreg": limited_scores,
              "cloglog": binomial_command_scores, "fracreg": binomial_command_scores}


@resident_cpu
def model_scores(result: ResultBundle, data: pd.DataFrame, positions: list[int]) -> ModelScores:
    """Scores and Hessian of one fit at its reported estimates (see module docstring).

    ``positions`` are the fit's estimation rows in ``data`` (from ``matched_frame``).
    """
    spec = result.spec
    provider = _PROVIDERS.get(spec.estimator)
    if provider is None:
        raise _unsupported(result, "its scores are not available to suest yet")
    if spec.weights is not None and spec.weight_type not in {"fweight", "pweight", "aweight"}:
        raise _unsupported(result, "suest supports fweight, pweight and aweight only")
    scores = provider(result, data, positions)
    try:
        decrement = scores.decrement()
    except AnalysisError as error:
        if spec.estimator in BINOMIAL_COMMANDS and error.code == "singular_information":
            raise _binomial_information_error() from error
        raise
    if spec.weights and spec.weight_type == "pweight":
        # Match the scale-free convergence criterion used by likelihood fitting.
        decrement /= float(data[spec.weights].iloc[positions].mean())
    if not math.isfinite(decrement) or decrement > DECREMENT_TOLERANCE:
        raise AnalysisError("suest_mismatch", f"The score of the {spec.estimator} fit of "
                            f"{spec.outcome!r} does not vanish at its estimates (Newton decrement "
                            f"{decrement:.3g}); the model could not be reproduced from the data. "
                            "Pass the dataset used for fitting and a converged fit.")
    return scores
