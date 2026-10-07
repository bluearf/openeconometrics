"""Coefficient inference from a persisted result, without refitting its model.

The public analysis dispatch retains fitted OLS methods (including private
contrast adjustments). These functions handle plain ResultBundle objects using
their recorded parameter vector, covariance and inference reference alone.
Restriction algebra and probability tails reuse OpenEconometrics's float64 Torch OLS
helpers; no third-party estimator is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fnmatch
import math
from numbers import Real
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.linear_ols.postestimation import _alpha, _wald
from openecon.models import ResultBundle


def _error(code: str, message: str):
    raise AnalysisError(code, message)


def _numeric(value: Any, *, code: str, ndim: int | None = None) -> torch.Tensor:
    try:
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            _error(code, "Provide finite real numbers, excluding booleans and complex values.")
        tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    except AnalysisError:
        raise
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError(code, "Provide finite real numeric values with the required shape.") from exc
    if (ndim is not None and tensor.ndim != ndim) or not bool(torch.isfinite(tensor).all()):
        _error(code, "Provide finite real numeric values with the required shape.")
    return tensor


def _scalar(value: Any, *, code: str) -> float:
    return float(_numeric(value, code=code, ndim=0))


@dataclass(frozen=True)
class _Parameters:
    beta: torch.Tensor
    covariance: torch.Tensor
    terms: tuple[str, ...]
    aliases: dict[str, tuple[int, ...]]
    df: float | None
    alpha: float

    def resolve(self, name: Any) -> int:
        if not isinstance(name, str) or not name:
            _error("invalid_restrictions", "Coefficient references must be nonempty names.")
        if name in self.terms:
            return self.terms.index(name)
        matches = self.aliases.get(name, ())
        if len(matches) > 1:
            _error("ambiguous_term", f"Coefficient '{name}' belongs to several equations; qualify its equation.")
        if not matches:
            _error("unknown_term", f"Unknown or omitted coefficient: {name}.")
        return matches[0]

    def contrast(self, mapping: Mapping) -> torch.Tensor:
        if not isinstance(mapping, Mapping) or not mapping:
            _error("invalid_restrictions", "Specify a nonempty mapping of coefficient names to weights.")
        vector = torch.zeros(len(self.terms), dtype=torch.float64)
        selected = set()
        for name, weight in mapping.items():
            index = self.resolve(name)
            if index in selected:
                _error("invalid_restrictions", "Reference each coefficient only once per restriction.")
            selected.add(index)
            vector[index] = _scalar(weight, code="invalid_restrictions")
        return vector


def _parameters(result: ResultBundle) -> _Parameters:
    if not isinstance(result, ResultBundle):
        _error("invalid_result", "Provide a fitted OpenEconometrics ResultBundle.")
    terms = tuple(item.term for item in result.coefficients)
    if (not terms or any(not isinstance(term, str) or not term.strip() for term in terms)
            or len(set(terms)) != len(terms)):
        _error("invalid_result", "Stored coefficients need distinct nonempty reporting names.")
    beta = _numeric([item.estimate for item in result.coefficients], code="invalid_result", ndim=1)
    covariance = _numeric(result.covariance_matrix, code="invalid_result", ndim=2)
    if covariance.shape != (len(terms), len(terms)):
        _error("invalid_result", "Stored coefficient and covariance dimensions disagree.")
    if not torch.allclose(covariance, covariance.T, rtol=1e-10, atol=0):
        _error("invalid_result", "The stored covariance must be symmetric.")
    covariance = covariance / 2 + covariance.T / 2
    diagonal = covariance.diagonal()
    if bool((diagonal < 0).any()):
        _error("invalid_result", "Stored covariance variances must be nonnegative.")
    positive = diagonal > 0
    if bool((covariance[~positive] != 0).any()):
        _error("invalid_result", "A zero-variance coefficient must have zero covariance with other coefficients.")
    if bool(positive.any()):
        block = covariance[positive][:, positive]
        scale = diagonal[positive].sqrt()
        correlation = block / scale[:, None] / scale[None, :]
        try:
            eigenvalues = torch.linalg.eigvalsh(correlation)
        except RuntimeError as exc:
            raise AnalysisError("invalid_result", "The stored covariance could not be validated.") from exc
        tolerance = 64 * torch.finfo(torch.float64).eps * len(eigenvalues) * max(1., float(eigenvalues[-1]))
        if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[0]) < -tolerance:
            _error("invalid_result", "The stored covariance must be positive semidefinite.")

    record = result.inference
    use_t = record.get("use_t")
    distribution = record.get("distribution")
    if use_t is None:
        if distribution not in {"t", "normal", "z"}:
            _error("invalid_inference", "The result must record t or normal coefficient inference.")
        use_t = distribution == "t"
    if not isinstance(use_t, bool) or (distribution is not None
            and distribution not in ({"t"} if use_t else {"normal", "z"})):
        _error("invalid_inference", "Recorded coefficient inference and use_t disagree.")
    df = None
    if use_t:
        df = record.get("df_inference", record.get("df_resid"))
        if isinstance(df, bool) or not isinstance(df, Real) or not math.isfinite(df) or df <= 0:
            _error("invalid_inference", "t inference requires recorded positive finite inference degrees of freedom.")
        df = float(df)
    alpha = _alpha(record.get("alpha", result.spec.alpha))
    # OLS contrast-specific designs cannot be reconstructed from b and V.
    if record.get("dfadjust") or record.get("hansen"):
        _error("estimation_state_unavailable", "Adjusted OLS contrasts require a fitted OLS result with its contrast state.")
    aliases: dict[str, list[int]] = {}
    for index, coefficient in enumerate(result.coefficients):
        equation, term = coefficient.equation, coefficient.term
        if equation is not None:
            if not isinstance(equation, str) or not equation.strip():
                _error("invalid_result", "Stored equation labels must be nonempty names.")
            # A colon can also denote an interaction in an unprefixed term.
            local = term[len(equation) + 1:] if term.startswith(equation + ":") else term
            for alias in (local, f"{equation}:{local}", f"[{equation}]{local}"):
                matches = aliases.setdefault(alias, [])
                if index not in matches:
                    matches.append(index)
    return _Parameters(beta, covariance, terms,
                       {name: tuple(indices) for name, indices in aliases.items()}, df, alpha)


def test(result: ResultBundle, restrictions, value=0) -> dict:
    """Test R b = value, using recorded t/F or normal/chi-squared inference.

    Restrictions are a coefficient name, a sequence of names, one weight
    mapping, a sequence of mappings, or a numeric vector/matrix in reporting
    coefficient order. Rows must be independent and have estimable covariance.
    """
    state = _parameters(result)
    if isinstance(restrictions, str):
        restrictions = [restrictions]
    if isinstance(restrictions, Mapping):
        matrix = state.contrast(restrictions)[None, :]
    elif (isinstance(restrictions, Sequence) and len(restrictions) > 0
          and all(isinstance(item, str) for item in restrictions)):
        indices = [state.resolve(name) for name in restrictions]
        if len(indices) != len(set(indices)):
            _error("invalid_restrictions", "Specify each tested coefficient only once.")
        matrix = torch.stack([state.contrast({name: 1}) for name in restrictions])
    elif (isinstance(restrictions, Sequence) and len(restrictions) > 0
          and all(isinstance(item, Mapping) for item in restrictions)):
        matrix = torch.stack([state.contrast(mapping) for mapping in restrictions])
    else:
        matrix = _numeric(restrictions, code="invalid_restrictions")
        if matrix.ndim == 1:
            matrix = matrix[None, :]
    if matrix.ndim != 2 or len(matrix) < 1 or matrix.shape[1] != len(state.terms):
        _error("invalid_restrictions", "Restrictions need independent rows and one column per fitted coefficient.")
    target = _numeric(value, code="invalid_restrictions")
    if target.ndim == 0:
        target = target.expand(len(matrix))
    if target.shape != (len(matrix),):
        _error("invalid_restrictions", "Provide one finite null value per restriction row.")
    return _wald(state.beta, state.covariance, matrix, target, state.df)


def testparm(result: ResultBundle, patterns) -> dict:
    """Joint zero test of exact names, wildcards, or a factor's dummy terms."""
    state = _parameters(result)
    if not isinstance(patterns, (str, list, tuple)):
        _error("invalid_restrictions", "Provide coefficient names or wildcard patterns.")
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    if not patterns or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        _error("invalid_restrictions", "Provide at least one coefficient name or wildcard pattern.")
    selected = set()
    for pattern in patterns:
        matched = []
        for index, term in enumerate(state.terms):
            names = [term, *(alias for alias, indices in state.aliases.items() if index in indices)]
            if any(name == pattern or fnmatch.fnmatchcase(name, pattern)
                   or name.startswith(pattern + "[") for name in names):
                matched.append(term)
        if not matched:
            _error("unknown_term", f"No fitted coefficients match '{pattern}'.")
        selected.update(matched)
    ordered = [term for term in state.terms if term in selected]
    output = test(result, ordered)
    output["terms"] = ordered
    return output


def _estimate(state: _Parameters, estimate: float, gradient: torch.Tensor, null: float) -> dict:
    variance = float(gradient @ state.covariance @ gradient)
    if not math.isfinite(variance) or variance <= 0 or not math.isfinite(estimate):
        _error("nonestimable_restriction", "The expression has no finite positive delta-method variance.")
    standard_error = math.sqrt(variance)
    statistic = (estimate - null) / standard_error
    if not math.isfinite(statistic):
        _error("invalid_inference", "The expression did not produce a finite inference statistic.")
    critical = critical_value(state.alpha, state.df)
    lower, upper = estimate - critical * standard_error, estimate + critical * standard_error
    if not math.isfinite(critical) or not math.isfinite(lower) or not math.isfinite(upper):
        _error("invalid_inference", "The expression did not produce finite confidence limits.")
    return {"estimate": estimate, "std_error": standard_error, "statistic": statistic,
            "distribution": "t" if state.df is not None else "normal", "df": state.df,
            "p_value": student_t_two_sided(statistic, state.df) if state.df is not None
            else math.erfc(abs(statistic) / math.sqrt(2)),
            "ci_low": lower, "ci_high": upper,
            "null": null, "alpha": state.alpha, "statistic_scale": 1.,
            "gradient": dict(zip(state.terms, gradient.tolist(), strict=True))}


def lincom(result: ResultBundle, mapping, constant=0) -> dict:
    """Estimate a named linear combination plus a constant; test it against zero."""
    state = _parameters(result)
    gradient = state.contrast(mapping)
    offset = _scalar(constant, code="invalid_restrictions")
    return _estimate(state, float(gradient @ state.beta) + offset, gradient, 0.)


class _CoefficientMap(Mapping):
    def __init__(self, state: _Parameters, values: torch.Tensor):
        self.state, self.values = state, values

    def __getitem__(self, name):
        return self.values[self.state.resolve(name)]

    def __iter__(self):
        return iter(self.state.terms)

    def __len__(self):
        return len(self.state.terms)


def nlcom(result: ResultBundle, function, null=0) -> dict:
    """Delta-method inference for one differentiable scalar Torch callable.

    The callable receives coefficient names mapped to Torch scalars. Strings
    are not evaluated. Autograd is limited to the small stored parameter vector;
    it does not retain observations or refit the estimator.
    """
    state = _parameters(result)
    if not callable(function):
        _error("invalid_nlcom", "nlcom requires a callable receiving named Torch coefficients.")
    null = _scalar(null, code="invalid_nlcom")
    try:
        with torch.inference_mode(False), torch.enable_grad():
            variable = state.beta.detach().clone().requires_grad_(True)
            expression = function(_CoefficientMap(state, variable))
            if (not isinstance(expression, torch.Tensor) or expression.ndim != 0
                    or expression.is_complex() or not bool(torch.isfinite(expression))):
                _error("invalid_nlcom", "Return one finite differentiable real Torch scalar.")
            gradient, = torch.autograd.grad(expression, variable)
            if not bool(torch.isfinite(gradient).all()):
                _error("invalid_nlcom", "The nonlinear derivative is not finite at the fitted coefficients.")
            estimate, gradient = float(expression.detach()), gradient.detach()
    except AnalysisError:
        raise
    except (RuntimeError, TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
        raise AnalysisError("invalid_nlcom", "The nonlinear expression is undefined or does not retain Torch derivatives.") from exc
    return _estimate(state, estimate, gradient, null)
