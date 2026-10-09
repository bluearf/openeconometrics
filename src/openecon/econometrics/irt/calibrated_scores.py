"""Continuous person scores conditional on a caller's fixed IRT bank."""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace, workspace_budget_bytes

from . import calibrated as bank_tools

DTYPE = torch.float64
DEFAULT_WORK = 300_000_000
DEFAULT_BYTES = 128 * 1024**2
SOURCES = [
    "https://www.stata.com/manuals/irtirt2plpostestimation.pdf",
    "https://www.stata.com/manuals/irtirtgrm.pdf",
    "https://www.stata.com/manuals/irtirtpcm.pdf",
    "https://www.stata.com/manuals/irtirtnrm.pdf",
    "https://philchalmers.github.io/mirt/reference/fscores.html",
]


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _number(value, name, lower, upper, *, open_lower=False, open_upper=False):
    if type(value) not in (int, float):
        _error(f"{name} must be a finite real scalar.")
    try:
        admitted = math.isfinite(value)
    except OverflowError:
        admitted = False
    if (not admitted or value < lower or value > upper
            or (open_lower and value == lower) or (open_upper and value == upper)):
        _error(f"{name} is outside its admitted finite range.")
    return float(value)


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        _error(f"{name} must be an integer in [{lower}, {upper}].")
    return value


def _controls(theta_limit, tolerance, max_iter, max_work, max_bytes, device):
    if type(device) is not str or device != "cpu":
        _error("Known-bank person scoring supports device='cpu' only.", "unsupported_device")
    return {
        "theta_limit": _number(theta_limit, "theta_limit", 0.25, 12),
        "tolerance": _number(tolerance, "tolerance", 1e-12, 1e-5),
        "max_iter": _integer(max_iter, "max_iter", 1, 200),
        "max_work": _integer(max_work, "max_work", 1, DEFAULT_WORK),
        "max_bytes": _integer(max_bytes, "max_bytes", 1, DEFAULT_BYTES),
        "device": "cpu",
    }


def _concave(state):
    """Certify the admitted model geometry before selecting response columns."""
    for definition in state["definitions"]:
        family = definition["family"]
        if family == "binary":
            if definition["guessing"] != 0 or definition["upper"] != 1:
                _error("Continuous scores require 2PL binary items with guessing=0 and upper=1; "
                       "3PL/4PL likelihoods need not be concave.", "unsupported_model")
        elif family == "nrm":
            slopes = definition["slopes"]
            if all(value == slopes[0] for value in slopes):
                _error("Continuous scores require NRM items with nonconstant category slopes.",
                       "unsupported_model")
        elif family not in ("grm", "gpcm"):
            _error("Continuous scores admit only 2PL, GRM, GPCM and informative NRM items.",
                   "unsupported_model")


class _Work:
    def __init__(self, cost, maximum):
        self.cost = cost
        self.maximum = maximum
        self.used = 0
        self.evaluations = 0

    def charge(self):
        if self.used + self.cost > self.maximum:
            _error("Known-bank scoring evaluation budget is exhausted.", "resource_limit")
        self.used += self.cost
        self.evaluations += 1


@torch.inference_mode(False)
@torch.enable_grad()
def _evaluate(value, codes, state, prior, work):
    work.charge()
    theta = torch.tensor([value], dtype=DTYPE, device="cpu", requires_grad=True)
    logp = bank_tools._logp(state, theta)
    log_likelihood = theta[0] * 0.0
    for j, code in enumerate(codes):
        if code >= 0:
            log_likelihood = log_likelihood + logp[j][0, code]
    target = log_likelihood
    prior_information = 0.0
    if prior is not None:
        mean, sd = prior
        target = target - 0.5*((theta[0]-mean)/sd).square() - math.log(sd) - 0.5*math.log(2*math.pi)
        prior_information = 1/(sd*sd)
    gradient = torch.autograd.grad(target, theta, create_graph=True)[0][0]
    hessian = torch.autograd.grad(gradient, theta)[0][0]
    target_information = -float(hessian.detach())
    if prior is not None and all(code < 0 for code in codes):
        target_information = prior_information
    values = (float(target.detach()), float(log_likelihood.detach()), float(gradient.detach()),
              target_information, target_information-prior_information)
    if not all(math.isfinite(number) for number in values):
        _error("Person score objective, gradient or curvature is nonfinite.", "numerical_failure")
    if values[4] < -1e-10:
        _error("Admitted likelihood failed its concavity certificate.", "numerical_failure")
    # Subtraction of a large prior curvature can leave roundoff at zero.
    return values[:4] + (max(0.0, values[4]),)


def _solve(codes, position, state, controls, prior, work, trace):
    limit, tolerance = controls["theta_limit"], controls["tolerance"]
    lower, upper = -limit, limit
    evaluation = 0

    def evaluate(theta, iteration, stage):
        nonlocal evaluation
        result = _evaluate(theta, codes, state, prior, work)
        evaluation += 1
        trace.append([position, evaluation, iteration, stage, theta, result[0], result[1],
                      result[2], result[3], result[4], lower, upper])
        return result

    lo = evaluate(lower, 0, "lower_endpoint")
    hi = evaluate(upper, 0, "upper_endpoint")
    if lo[2] <= tolerance or hi[2] >= -tolerance:
        _error(f"Person at source position {position} has no certified interior score root "
               "within the admitted theta range; an extreme or boundary mode is refused.",
               "boundary_solution")
    margin = max(1e-9, 10*tolerance)
    if prior is not None and all(code < 0 for code in codes):
        value = prior[0]
        result = evaluate(value, 0, "exact_prior_mode")
        iteration = 0
    else:
        for iteration in range(1, controls["max_iter"]+1):
            value = (lower+upper)/2
            result = evaluate(value, iteration, "bisection")
            if abs(result[2]) <= tolerance:
                break
            if result[2] > 0:
                lower = value
            else:
                upper = value
        else:
            _error(f"Person at source position {position} did not reach the gradient tolerance "
                   "within max_iter.", "nonconvergence")
    if abs(value) >= limit-margin:
        _error(f"Person at source position {position} is numerically on the admitted theta boundary.",
               "boundary_solution")
    if abs(result[2]) > tolerance or result[3] <= 1e-12:
        _error(f"Person at source position {position} lacks finite positive score information "
               "or a stationary mode.", "numerical_failure")
    certificate = [position, lo[2], hi[2], abs(result[2]), result[3], True]
    return value, result, iteration, lower, upper, certificate


def _score(bank, data, controls, prior, level):
    budget = min(controls["max_bytes"], workspace_budget_bytes())
    state = bank_tools._bank(bank, max_bytes=budget)
    _concave(state)
    n = bank_tools._response_size(data, state, max_rows=1000)
    j = len(state["items"])
    categories = sum(definition["K"] for definition in state["definitions"])
    evaluation_cost = 64*categories
    planned_work = n*(controls["max_iter"]+3)*evaluation_cost
    if planned_work > controls["max_work"]:
        _error("Planned person-scoring probability/gradient/curvature work exceeds max_work "
               "before response selection.", "resource_limit")
    plan = plan_workspace("known-bank continuous person scores", {
        "named_response_selection_and_masks": 64*n*(j+2),
        "response_codes_and_person_results": 8*n*j+256*n,
        "one_person_probability_autograd_graph": 2048*(categories+j+1),
        "full_gradient_and_bracket_trace": 128*n*(controls["max_iter"]+3),
        "bounded_bank_and_summary_encoding": 4096*j+128*categories,
    }, budget_bytes=budget).record()
    responses, indices = bank_tools._responses(data, state, max_rows=1000, max_bytes=budget)
    work = _Work(evaluation_cost, controls["max_work"])
    people, samples, histories, certificates = [], [], [], []
    critical = None
    if level is not None:
        normal = torch.distributions.Normal(torch.zeros((), dtype=DTYPE, device="cpu"),
                                            torch.ones((), dtype=DTYPE, device="cpu"))
        upper_probability = 1-(1-level)/2
        critical = float(normal.icdf(torch.tensor(upper_probability, dtype=DTYPE, device="cpu")))
        if not math.isfinite(critical) or critical <= 0:
            _error("level must give a finite positive normal critical value.")
    for position in range(n):
        codes = responses[position].tolist()
        observed = sum(code >= 0 for code in codes)
        if prior is None and observed == 0:
            _error(f"MLE person at source position {position} has no observed items.",
                   "insufficient_sample")
        theta, result, iterations, lower, upper, certificate = _solve(
            codes, position, state, controls, prior, work, histories)
        sd = 1/math.sqrt(result[3])
        # The exact all-missing MAP prior is retained without inverse-curvature roundoff.
        if prior is not None and observed == 0:
            sd = prior[1]
        prefix = [position, indices[position], "mle" if prior is None else "map", theta, sd]
        if critical is not None:
            interval_lower, interval_upper = theta-critical*sd, theta+critical*sd
            if (not math.isfinite(interval_lower) or not math.isfinite(interval_upper)
                    or not interval_lower < theta < interval_upper):
                _error(f"Person at source position {position} has no distinct finite float64 "
                       "Wald interval endpoints at the requested level.", "numerical_failure")
            prefix += [interval_lower, interval_upper]
        people.append(prefix+[observed, j-observed, result[1], result[0], result[2],
                              result[4], result[3], iterations, lower, upper])
        samples.append([position, indices[position], observed, j-observed])
        certificates.append(certificate)
    columns = ["position", "index", "method", "theta", "standard_error" if prior is None else "laplace_sd"]
    if prior is None:
        columns += ["ci_lower", "ci_upper"]
    columns += ["observed_items", "missing_items", "log_likelihood", "objective", "gradient",
                "observed_information", "mode_information", "iterations", "bracket_lower", "bracket_upper"]
    method = "mle" if prior is None else "map"
    output = TableSet({
        "certificates": table(certificates, columns=["position", "lower_gradient", "upper_gradient",
                              "absolute_gradient", "mode_information", "interior"]),
        "people": table(people, columns=columns),
        "responses": table([[i, indices[i], *responses[i].tolist()] for i in range(n)],
                           columns=["position", "index", *state["items"]]),
        "sample": table(samples, columns=["position", "index", "observed_items", "missing_items"]),
        "trace": table(histories, columns=["position", "evaluation", "iteration", "stage", "theta",
                       "objective", "log_likelihood", "gradient", "mode_information",
                       "observed_information", "bracket_lower", "bracket_upper"]),
    }, title=f"Known-bank continuous person {method.upper()} scores", method=f"irt_score_{method}",
        bank_state=state, sample_positions=list(range(n)), controls=controls,
        level=level, prior=None if prior is None else {"mean": prior[0], "sd": prior[1]},
        uncertainty=("Conditional fixed-item-parameter observed-information asymptotic normal Wald "
                     "standard errors and intervals; no calibration uncertainty."
                     if prior is None else "Local inverse posterior-mode curvature Laplace SD; "
                     "not the exact posterior SD or a frequentist confidence interval."),
        boundary_policy="Refuse the entire call if any person lacks a certified interior stationary mode.",
        workspace_plan=plan, planned_work=planned_work, charged_work=work.used,
        objective_evaluations=work.evaluations, work_unit="64 times summed bank categories per objective/gradient/curvature evaluation",
        sources=SOURCES)
    saved_summary(output)
    return TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)


@resident_cpu
def irt_score_mle(bank: TableSet | str, *, data: Any, level: float = 0.95,
                  theta_limit: float = 12.0, tolerance: float = 1e-9, max_iter: int = 100,
                  max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                  device: str = "cpu") -> TableSet:
    """Conditional continuous MLE scores with observed-information normal Wald intervals.

    Fixed 2PL/GRM/GPCM/NRM banks only; missing items are skipped. Every person
    must have an interior stationary mode. Extreme, uninformative, boundary or
    unconverged patterns refuse the entire call. Item parameters are treated as
    known; normal intervals are asymptotic, without calibration uncertainty.
    """
    controls = _controls(theta_limit, tolerance, max_iter, max_work, max_bytes, device)
    level = _number(level, "level", 0, 1, open_lower=True, open_upper=True)
    lower_probability = (1-level)/2
    upper_probability = 1-lower_probability
    if not 0 < lower_probability < 0.5 < upper_probability < 1:
        _error("level must give distinct interior float64 equal-tail normal probabilities.")
    return _score(bank, data, controls, None, level)


@resident_cpu
def irt_score_map(bank: TableSet | str, *, data: Any, prior_mean: float = 0.0,
                  prior_sd: float = 1.0, theta_limit: float = 12.0,
                  tolerance: float = 1e-9, max_iter: int = 100, max_work: int = DEFAULT_WORK,
                  max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Normal-prior continuous MAP scores and local inverse-curvature Laplace SD.

    Fixed 2PL/GRM/GPCM/NRM banks only. The supplied normal mean and SD are known,
    and item parameters have no calibration uncertainty. Returned Laplace SD
    describes local posterior curvature, not exact posterior variance or a
    frequentist interval. Fully missing rows return the admitted prior mode/SD.
    """
    controls = _controls(theta_limit, tolerance, max_iter, max_work, max_bytes, device)
    prior = (_number(prior_mean, "prior_mean", -4, 4),
             _number(prior_sd, "prior_sd", 0.25, 3))
    return _score(bank, data, controls, prior, None)
