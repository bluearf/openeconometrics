"""Bounded smooth nonlinear SUR, native Gaussian profile ML and sealed replay.

A finite stationary local maximum is certified with the full physical observed
information for shared mean parameters and the unregularized residual covariance.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from statistics import NormalDist

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.quantile.formula import evaluate, parse
from openecon.engines.optimize import maximize_bfgs
from openecon.resources import plan_workspace, workspace_budget_bytes

DT = torch.float64
SCHEMA = "nonlinear_sur_v1"
SOLVER = "native CPU float64 Gaussian profile ML; deterministic multistart BFGS; full physical stationary local maximum"
NOTES = [
    "Common complete sample; globally shared named mean parameters; independent Gaussian residual vectors across rows for model-based OIM.",
    "Residual covariance is unregularized ML R'R/N. Full physical observed information includes nonlinear residual curvature and mean/covariance cross blocks.",
    "HC0 uses complete row joint scores; CR0 sums them within the declared cluster. Neither covariance has a degrees-of-freedom multiplier.",
    "Three deterministic starts certify a finite identified stationary local maximum, not a globally optimal solution.",
    "Asymptotic normal inference; positive residual variances use log-delta intervals without a zero-variance null test.",
    "Saved-state replay reparses the safe formula tapes and recomputes all moments without optimization. The checksum detects corruption, not authenticated provenance.",
]
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not math.isfinite(float(level)) or not 0 < level < 1:
        _error("level must be finite and strictly between zero and one.", "invalid_option")
    probability = (1+float(level))/2
    if not 0 < probability < 1:
        _error("level exceeds finite normal-inference precision.", "invalid_option")
    z = NormalDist().inv_cdf(probability)
    return float(level), z


def _parse_smooth(source):
    if not isinstance(source, str):
        _error("formula must be a safe expression string.", "invalid_formula")
    formula = parse(" ".join(source.split()))
    if len(formula.tape) > 128:
        _error("Each formula supports at most 128 safe tape nodes.", "resource_limit")
    constant = {}
    for i, node in enumerate(formula.tape):
        op = node[0]
        if op == "const":
            constant[i] = node[1]
        elif op == "neg" and node[1] in constant:
            constant[i] = -constant[node[1]]
        elif op == "fn" and node[1] not in {"exp", "ln", "log", "sqrt", "sin", "cos"}:
            _error("Only exp, log/ln, sqrt, sin and cos are supported smooth functions.", "unsupported_formula")
        elif op == "pow":
            exponent = constant.get(node[2])
            if exponent is None or not math.isfinite(exponent) or exponent != int(exponent) or abs(exponent) > 12:
                _error("Powers require an integer literal exponent in [-12,12].", "unsupported_formula")
    return formula


def _evaluate(formula, columns, theta, n):
    """Safe differentiable tape value, with theta in formula-local name order."""
    with torch.device("cpu"):
        return evaluate.__wrapped__(formula, columns, theta, n, jacobian=False)[0]


def _equations(equations, start):
    if not isinstance(equations, (list, tuple)) or not 2 <= len(equations) <= 4:
        _error("equations must contain 2..4 named smooth mean equations.", "invalid_spec")
    canonical, formulas, names, columns, declared = [], [], [], [], {}
    for item in equations:
        if not isinstance(item, Mapping) or not {"y", "formula"} <= set(item) or set(item)-{"y", "name", "formula"}:
            _error("Each equation requires y/formula and optional name only.", "invalid_spec")
        y, name = item["y"], item.get("name", item["y"])
        if any(not isinstance(v, str) or not _IDENT.fullmatch(v) or len(v) > 64 for v in (y, name)):
            _error("Outcome and equation names must be identifiers of at most 64 characters.", "invalid_spec")
        formula = _parse_smooth(item["formula"])
        canonical.append(dict(y=y, name=name, formula=formula.source))
        formulas.append(formula)
        for key in formula.parameters:
            if key not in names:
                names.append(key)
        for key in formula.columns:
            if key not in columns:
                columns.append(key)
        for key, value in formula.start.items():
            if key in declared and declared[key] != value:
                _error("Shared embedded parameter starts conflict across equations.", "invalid_start")
            declared[key] = value
    if len({e["name"] for e in canonical}) != len(canonical) or len({e["y"] for e in canonical}) != len(canonical):
        _error("Equation names and outcomes must each be distinct.", "invalid_spec")
    if {e["y"] for e in canonical} & set(columns):
        _error("An outcome cannot occur on any equation's right-hand side.", "endogenous_rhs")
    if len(names) > 12 or len(columns) > 16:
        _error("Fit support is at most 12 shared mean parameters and 16 numeric features.", "resource_limit")
    if start is not None:
        if not isinstance(start, Mapping) or set(start)-set(names):
            _error("start must map declared shared parameter names to finite values.", "invalid_start")
        for key, value in start.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or abs(value) > 1e12:
                _error("Starting values must be finite real scalars within 1e12.", "invalid_start")
            declared[key] = float(value)
    starts = {name: float(declared.get(name, 0.0)) for name in names}
    if any(not math.isfinite(v) or abs(v) > 1e12 for v in starts.values()):
        _error("Embedded starting values exceed finite 1e12 support.", "invalid_start")
    return canonical, formulas, names, columns, starts


def _options(covariance, cluster, tolerance, max_iterations, max_work, max_bytes):
    if not isinstance(covariance, str) or covariance not in {"oim", "hc0", "cr0"}:
        _error("covariance must be oim, hc0 or cr0.", "invalid_option")
    if (covariance == "cr0") != (cluster is not None):
        _error("Declare cluster exactly when covariance is cr0.", "invalid_option")
    if cluster is not None and (not isinstance(cluster, str) or not cluster):
        _error("cluster must name a grouping column.", "invalid_spec")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(float(tolerance)) or not 1e-12 <= tolerance <= 1e-5:
        _error("tolerance must be finite in [1e-12,1e-5].", "invalid_option")
    limits = ((max_iterations, 2000, "max_iterations"), (max_work, 2_000_000_000, "max_work"), (max_bytes, 2_000_000_000, "max_bytes"))
    for value, bound, name in limits:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= bound:
            _error(f"{name} must be an integer in [1,{bound}].", "invalid_option")
    return dict(covariance=covariance, cluster=cluster, tolerance=float(tolerance), max_iterations=int(max_iterations), max_work=int(max_work), max_bytes=int(max_bytes))


def _plan(n, m, q, features, tapes, options, *, budget_bytes=None):
    if not 2 <= m <= 4 or not 1 <= q <= 12 or n > 512 or features > 16 or max(tapes) > 128:
        _error("Resident computational support is N<=512, equations 2..4, mean parameters<=12, features<=16 and 128 nodes/equation.", "resource_limit")
    if n <= m or n*m <= q:
        _error("The complete sample is too small for a nonsingular residual covariance and identified means.", "insufficient_observations")
    k = q+m*(m+1)//2
    work = n*(sum(tapes)+m*m)*(q+1)*12 + n*(sum(tapes)+m*m)*(k+1)**2*4
    memory = 8*(n*(sum(tapes)+m*m)*(k+1)**2*3 + 16*k*k + n*(features+m+k))
    if work > options["max_work"]:
        _error("Initial derivative replay exceeds max_work before tensor allocation.", "work_budget")
    if memory > options["max_bytes"]:
        _error("Planned resident derivative workspace exceeds max_bytes before tensor allocation.", "workspace_limit")
    plan = plan_workspace("nonlinear_sur_joint_ml", {"complete_gaussian_derivative_workspace": memory},
                          budget_bytes=min(options["max_bytes"], workspace_budget_bytes()) if budget_bytes is None else budget_bytes).record()
    return plan | dict(planned_derivative_work=work, planned_bytes=memory, budget_work=options["max_work"], full_parameter_count=k)


def _index_record(index):
    if isinstance(index, pd.CategoricalIndex):
        _error("Categorical row indexes are outside the portable index contract.", "unsupported_input")
    labels = [encode(v) for v in index]
    if isinstance(index, pd.MultiIndex):
        return dict(labels=labels, names=[encode(v) for v in index.names], multi=True, dtype=None, range=None)
    return dict(labels=labels, names=[encode(index.name)], multi=False, dtype=str(index.dtype),
                range=[index.start, index.stop, index.step] if isinstance(index, pd.RangeIndex) else None)


def _index_restore(record, n):
    if not isinstance(record, dict) or set(record) != {"labels", "names", "multi", "dtype", "range"} or not isinstance(record["labels"], list) or len(record["labels"]) != n:
        _error("Saved row-index schema is incomplete.", "invalid_result")
    try:
        labels, names = [decode(v) for v in record["labels"]], [decode(v) for v in record["names"]]
        if any(encode(v) != c for v, c in zip(labels, record["labels"])) or any(encode(v) != c for v, c in zip(names, record["names"])):
            _error("Saved index codes are not canonical.", "invalid_result")
        if record["multi"] is True:
            index = pd.MultiIndex.from_tuples(labels, names=names)
        elif record["multi"] is False and len(names) == 1:
            if record["range"] is not None:
                index = pd.RangeIndex(*record["range"], name=names[0])
                if list(index) != labels:
                    _error("Saved range index disagrees with its labels.", "invalid_result")
            else:
                index = pd.Index(labels, dtype=record["dtype"], name=names[0], tupleize_cols=False)
        else:
            _error("Saved index kind is unsupported.", "invalid_result")
        if _index_record(index) != record:
            _error("Saved index cannot be reproduced canonically.", "invalid_result")
        return index
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved index codes are invalid.") from error


def _vector(value, n, name, index):
    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            _error("Selected Series must have exactly aligned row indexes.", "shape_mismatch")
        values = value.tolist()
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1 or value.requires_grad or value.is_complex():
            _error("Input tensors require real resident CPU vectors without gradients.", "unsupported_input")
        values = value.tolist()
    elif isinstance(value, (list, tuple)) or type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", 1) != 1:
            _error("Selected columns require one-dimensional vectors.", "unsupported_input")
        values = list(value)
    else:
        _error(f"{name} must be a resident numeric vector.", "unsupported_input")
    if len(values) != n:
        _error("All selected columns must have equal lengths.", "shape_mismatch")
    return values


def _numeric(values, name):
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        _error(f"{name} requires real numeric cells with no boolean or missing values.", "missing_values")
    if any(not math.isfinite(float(v)) for v in values):
        _error(f"{name} contains a missing or nonfinite cell; rows are never deleted.", "missing_values")
    if any(abs(v) > 1e12 for v in values):
        _error("Numeric inputs exceed the explicit 1e12 magnitude support.", "unsupported_input")
    return [float(v) for v in values]


@dataclass
class _Prepared:
    equations: list
    formulas: list
    mean_names: list
    columns: list
    starts: dict
    options: dict
    inputs: dict
    index: pd.Index
    index_record: dict
    x: dict
    y: torch.Tensor
    mean_scale: torch.Tensor
    parameter_names: list
    cluster_labels: list
    cluster_codes: list
    resources: dict

    @property
    def n(self):
        return len(self.index)

    @property
    def q(self):
        return len(self.mean_names)

    @property
    def m(self):
        return len(self.equations)

    @property
    def row_labels(self):
        return list(self.index)


def _prepare(data, equations, options, start=None, *, index_record=None):
    canonical, formulas, names, columns, starts = _equations(equations, start)
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            _error("Input column labels must be unique.", "invalid_spec")
        index, n = data.index.copy(), len(data)
    elif isinstance(data, Mapping):
        first = next(iter(data.values()), [])
        try:
            n = len(first)
        except TypeError:
            _error("data requires a frame or mapping of resident vectors.", "unsupported_input")
        series = [v for v in data.values() if isinstance(v, pd.Series)]
        index = series[0].index.copy() if series else pd.RangeIndex(n)
    else:
        _error("data requires a resident DataFrame or mapping of complete vectors.", "unsupported_input")
    resources = _plan(n, len(canonical), len(names), len(columns), [len(f.tape) for f in formulas], options)
    if index_record is not None:
        index = _index_restore(index_record, n)
    record = _index_record(index)
    selected = list(dict.fromkeys([e["y"] for e in canonical]+columns))
    inputs = {}
    for name in selected:
        if name not in data:
            _error(f"Missing selected column {name!r}.", "missing_column")
        inputs[name] = _numeric(_vector(data[name], n, name, index), name)
    cluster_labels, cluster_codes = [], []
    if options["cluster"]:
        key = options["cluster"]
        if key not in data:
            _error("Missing declared cluster column.", "missing_column")
        values = _vector(data[key], n, key, index)
        clean = []
        for value in values:
            if isinstance(value, Integral) and not isinstance(value, bool):
                value = int(value)
            elif not isinstance(value, str) or not value:
                _error("Cluster IDs must be nonmissing integers or nonempty strings.", "invalid_cluster")
            clean.append(value)
            typed = (type(value).__name__, value)
            existing = [(type(v).__name__, v) for v in cluster_labels]
            if typed not in existing:
                cluster_labels.append(value)
                existing.append(typed)
            cluster_codes.append(existing.index(typed))
        if len(cluster_labels) < 2:
            _error("CR0 requires at least two declared clusters.", "invalid_cluster")
        if key in inputs and inputs[key] != clean:
            _error("A cluster used as a numeric feature must preserve its numeric values.", "invalid_cluster")
        inputs[key] = clean
    aliases = [f"cov__{canonical[i]['name']}__{canonical[j]['name']}" for i in range(len(canonical)) for j in range(i+1)]
    if set(names) & set(aliases):
        _error("Mean parameter names collide with residual covariance aliases.", "invalid_spec")
    y = torch.tensor([[inputs[e["y"]][i] for e in canonical] for i in range(n)], dtype=DT, device="cpu")
    x = {name: torch.tensor(inputs[name], dtype=DT, device="cpu") for name in columns}
    p = _Prepared(canonical, formulas, names, columns, starts, options, inputs, index, record, x, y,
                  torch.ones(len(names), dtype=DT, device="cpu"), names+aliases, cluster_labels, cluster_codes, resources)
    initial = torch.tensor(list(starts.values()), dtype=DT, device="cpu")
    try:
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(lambda t: _means(t, p), initial, vectorize=True, strategy="forward-mode")
        output_scale = torch.sqrt((y*y).mean()).clamp_min(1e-12)
        scales = torch.sqrt((jac*jac).mean((0, 1)))/output_scale
        fallback = 1/(initial.abs()+1)
        p.mean_scale = torch.where(torch.isfinite(scales) & (scales > 1e-12), scales, fallback).clamp(1e-12, 1e12).detach()
    except (RuntimeError, OverflowError):
        # Domain-invalid starts are examined separately by the multistart protocol.
        pass
    return p


def _means(physical, p, x=None):
    columns = p.x if x is None else x
    n = p.n if x is None else (len(next(iter(columns.values()))) if columns else p.n)
    means = []
    for formula in p.formulas:
        local = torch.stack([physical[p.mean_names.index(name)] for name in formula.parameters])
        means.append(_evaluate(formula, columns, local, n))
    return torch.stack(means, dim=1)


def _sigma(physical, p):
    rows, offset = [], p.q
    for i in range(p.m):
        row = []
        for j in range(p.m):
            a, b = max(i, j), min(i, j)
            row.append(physical[offset+a*(a+1)//2+b])
        rows.append(torch.stack(row))
    return torch.stack(rows)


def _profile(physical, p):
    residual = p.y-_means(physical, p)
    sigma = residual.T@residual/p.n
    return residual, sigma


def _positive(sigma):
    if not bool(torch.isfinite(sigma).all()) or not bool((sigma.diag() > 0).all()):
        return False
    sd = sigma.diag().sqrt()
    normalized = sigma/sd[:, None]/sd[None, :]
    _, status = torch.linalg.cholesky_ex(normalized.detach())
    return int(status) == 0 and float(torch.linalg.eigvalsh(normalized.detach()).min()) > 1e-12


def _case_logs(physical, p):
    residual, sigma = p.y-_means(physical, p), _sigma(physical, p)
    sign, determinant = torch.linalg.slogdet(sigma)
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(determinant)) or float(sign.detach()) != 1:
        _error("Physical Gaussian likelihood is outside its finite covariance/domain support.", "numerical_failure")
    solved = torch.linalg.solve(sigma, residual.T).T
    return -.5*(p.m*math.log(2*math.pi)+determinant+(residual*solved).sum(1))


class _Objective:
    def __init__(self, p):
        self.p, self.used = p, 0

    def charge(self, multiplier):
        self.used += self.p.n*(sum(len(f.tape) for f in self.p.formulas)+self.p.m*self.p.m)*(self.p.q+1)*multiplier
        if self.used > self.p.options["max_work"]:
            _error("Actual derivative evaluations exceeded max_work.", "work_budget")

    def value(self, point):
        residual, sigma = _profile(point/self.p.mean_scale, self.p)
        if not bool(torch.isfinite(residual).all()) or not _positive(sigma):
            return point.sum()*0+point.new_tensor(-torch.inf)
        return -.5*self.p.n*(self.p.m*(math.log(2*math.pi)+1)+torch.linalg.slogdet(sigma)[1])

    def __call__(self, point):
        self.charge(2)
        with torch.enable_grad():
            x = point.detach().requires_grad_()
            value = self.value(x)
            gradient = torch.autograd.grad(value, x)[0]
        return value.detach(), gradient.detach()

    def hessian(self, point):
        self.charge(2*(self.p.q+1))
        with torch.enable_grad():
            return torch.autograd.functional.hessian(self.value, point, vectorize=True).detach()


def _inverse(information):
    if bool((information.diag() == 0).any()):
        _error("Full physical information has an unidentified zero-curvature parameter.", "rank_deficient")
    if not bool(torch.isfinite(information).all()) or not bool((information.diag() > 0).all()):
        _error("Full physical information is nonfinite or nonpositive.", "no_finite_mle")
    sd = information.diag().sqrt()
    normalized = information/sd[:, None]/sd[None, :]
    eigen = torch.linalg.eigvalsh((normalized+normalized.T)/2)
    if float(eigen[0]) <= 1e-10*float(eigen[-1]):
        _error("Full physical mean/covariance information is unidentified or ill-conditioned.", "rank_deficient")
    factor, status = torch.linalg.cholesky_ex(normalized)
    if int(status):
        _error("Full observed information must be positive definite.", "no_finite_mle")
    bread = torch.cholesky_inverse(factor)/sd[:, None]/sd[None, :]
    return (bread+bread.T)/2


def _moments(physical, p):
    if not _positive(_sigma(physical, p)):
        _error("Residual covariance must be finite positive definite without ridge.", "no_finite_mle")
    try:
        with torch.enable_grad():
            point = physical.detach().requires_grad_()
            values = _case_logs(point, p)
            gradient = torch.autograd.grad(values.sum(), point)[0]
            hessian = torch.autograd.functional.hessian(lambda t: _case_logs(t, p).sum(), physical, vectorize=True)
            scores = torch.autograd.functional.jacobian(lambda t: _case_logs(t, p), physical, vectorize=True, strategy="forward-mode")
    except RuntimeError as error:
        raise AnalysisError("numerical_failure", "Physical Gaussian derivatives exceed native numerical support.") from error
    information = -(hessian+hessian.T)/2
    if any(not bool(torch.isfinite(v).all()) for v in (values, gradient, information, scores)):
        _error("Full physical likelihood derivatives must be finite.", "numerical_failure")
    return dict(information=information.detach(), gradient=gradient.detach(), case_scores=scores.detach(), case_loglikelihood=values.detach(), log_likelihood=float(values.sum().detach()))


def _stationarity(physical, p, moments):
    bread = _inverse(moments["information"])
    step = bread@moments["gradient"]
    sigma = _sigma(physical, p)
    covariance_units = torch.stack([1/torch.sqrt(sigma[i, i]*sigma[j, j]) for i in range(p.m) for j in range(i+1)])
    units = torch.cat((p.mean_scale, covariance_units))
    norm_step = float((step*units).abs().max())
    decrement = float(moments["gradient"]@step)
    diagnostic = dict(normalized_parameter_step=norm_step, score_decrement=max(0., decrement),
                      physical_score_max=float(moments["gradient"].abs().max()), step_limit=max(20*p.options["tolerance"], 1e-8),
                      decrement_limit=max(100*p.options["tolerance"]**2, 1e-16)*p.n)
    stationary = math.isfinite(norm_step) and math.isfinite(decrement) and decrement >= -1e-12 and norm_step <= diagnostic["step_limit"] and decrement <= diagnostic["decrement_limit"]
    return diagnostic, bread, stationary


def _start_vectors(p):
    base = torch.tensor([p.starts[name] for name in p.mean_names], dtype=DT, device="cpu")*p.mean_scale
    offset = torch.tensor([.1 if j % 2 == 0 else -.1 for j in range(p.q)], dtype=DT, device="cpu")
    return [base, base+offset, base-offset]


def _fit(p):
    objective, attempts, accepted = _Objective(p), [], []
    for initial in _start_vectors(p):
        record = dict(start=(initial/p.mean_scale).tolist())
        try:
            fitted = kernel_call(maximize_bfgs, objective, initial, hessian_fn=objective.hessian, max_iter=p.options["max_iterations"],
                                 gradient_tol=min(1e-9, p.options["tolerance"]), scaled_gradient_tol=min(1e-12, p.options["tolerance"]**2), raise_on_failure=False)
            beta = fitted.theta.detach()/p.mean_scale
            _, sigma = _profile(beta, p)
            physical = torch.cat((beta, torch.stack([sigma[i, j] for i in range(p.m) for j in range(i+1)])))
            objective.charge(4*(len(physical)+1))
            moments = _moments(physical, p)
            diagnostic, _, stationary = _stationarity(physical, p, moments)
            if not stationary:
                _error("Profile candidate failed physical joint stationarity.", "nonconvergence")
            record.update(status="stationary_local", iterations=fitted.iterations, params=physical.tolist(), log_likelihood=moments["log_likelihood"], stationarity=diagnostic)
            accepted.append((moments["log_likelihood"], physical, len(attempts)))
        except RuntimeError as error:
            record.update(status="rejected", code="numerical_failure")
            if "out of memory" in str(error).lower():
                _error("Native workspace allocation failed.", "workspace_limit")
        except AnalysisError as error:
            if error.code in {"work_budget", "workspace_limit", "resource_limit"}:
                raise
            record.update(status="rejected", code=error.code if error.code in {"rank_deficient", "no_finite_mle", "nonconvergence", "numerical_failure", "invalid_start"} else "numerical_failure")
        attempts.append(record)
    if not accepted:
        code = "rank_deficient" if any(r["code"] == "rank_deficient" for r in attempts) else "no_finite_mle"
        _error("No deterministic start yielded finite unregularized identified Gaussian ML with physical stationarity.", code)
    _, physical, selected = max(accepted, key=lambda v: v[0])
    convergence = dict(starts=attempts, selected_start=selected, work_used=objective.used, max_work=p.options["max_work"], globally_certified=False, solver=SOLVER)
    return physical, convergence


def _replay(physical, p):
    moments = _moments(physical, p)
    diagnostic, bread, stationary = _stationarity(physical, p, moments)
    if not stationary:
        _error("Full physical Gaussian fit fails score/Newton-step stationarity.", "no_finite_mle")
    fitted = _means(physical, p).detach()
    residual = p.y-fitted
    sigma = _sigma(physical, p).detach()
    mle = residual.T@residual/p.n
    if not torch.allclose(sigma, mle, rtol=2e-10, atol=0):
        _error("Fitted residual covariance is not the unregularized profile ML covariance.", "no_finite_mle")
    cluster_scores = torch.zeros((len(p.cluster_labels), len(physical)), dtype=DT, device="cpu")
    if p.cluster_labels:
        cluster_scores.index_add_(0, torch.tensor(p.cluster_codes, dtype=torch.int64, device="cpu"), moments["case_scores"])
    scores = cluster_scores if p.options["covariance"] == "cr0" else moments["case_scores"]
    meat = scores.T@scores
    influence = scores@bread
    covariance = bread if p.options["covariance"] == "oim" else influence.T@influence
    covariance = (covariance+covariance.T)/2
    if not bool(torch.isfinite(covariance).all()) or bool((covariance.diag() < 0).any()):
        _error("Full joint covariance is outside finite float64 support.", "numerical_failure")
    return dict(params=physical.tolist(), parameter_names=p.parameter_names, sigma=sigma.tolist(), information=moments["information"].tolist(),
                bread=bread.tolist(), meat=meat.tolist(), covariance=covariance.tolist(), gradient=moments["gradient"].tolist(),
                case_scores=moments["case_scores"].tolist(), cluster_scores=cluster_scores.tolist(), case_loglikelihood=moments["case_loglikelihood"].tolist(),
                log_likelihood=moments["log_likelihood"], fitted=fitted.tolist(), residuals=residual.tolist(), mean_scale=p.mean_scale.tolist(), stationarity=diagnostic,
                n=p.n, n_equations=p.m, n_mean_parameters=p.q, n_clusters=len(p.cluster_labels), covariance_type=p.options["covariance"])


def _checksum(state):
    return hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "checksum"}, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _settings(p, level):
    return dict(**p.options, level=level, device="cpu", precision="float64", missing="raise", weights=None, n_input=p.n, n_retained=p.n, n_dropped=0,
                n_equations=p.m, mean_parameters=p.mean_names, parameter_names=p.parameter_names, features=p.columns,
                equations=p.equations, starts=p.starts, resources=p.resources, max_rows=512, max_equations=4, max_mean_parameters=12, max_features=16,
                max_tape_nodes=128, numeric_magnitude_bound=1e12, covariance_normalization="R'R/N ML; no ridge",
                inference="full physical observed information; normal; variance log-delta intervals; no variance zero-null test", inference_df=None,
                robust_covariance_may_be_singular=True, global_optimality_certified=False, complete_inputs_saved=True, complete_covariance_saved=True,
                restoration="safe tape reparse and complete numerical replay; no optimizer", excluded_scope=["weights", "missing-row deletion", "Dataset", "CUDA/MPS", "endogenous RHS", "structural FIML", "categories", "ridge", "global optimum certification"])


def _matrix(value, names):
    if isinstance(value, torch.Tensor):
        value = value.detach().tolist()
    marker = "parameter"
    while marker in names:
        marker = "_"+marker
    return table([[name, *row] for name, row in zip(names, value)], columns=[marker, *names])


def _assemble(state, p):
    fit, level = state["fit"], state["level"]
    _, zcrit = _confidence(level)
    diagonals = {p.q+i*(i+1)//2+i for i in range(p.m)}
    parameters = []
    for j, name in enumerate(p.parameter_names):
        estimate, variance = fit["params"][j], fit["covariance"][j][j]
        se = math.sqrt(variance)
        z = estimate/se if se and j not in diagonals else None
        lower, upper = (estimate-zcrit*se, estimate+zcrit*se) if se else (None, None)
        if j in diagonals and se:
            lowlog, highlog = math.log(estimate)-zcrit*se/estimate, math.log(estimate)+zcrit*se/estimate
            if highlog > math.log(float.fromhex('0x1.fffffffffffffp+1023')):
                _error("Positive-variance interval exceeds finite float64 support.", "numerical_failure")
            lower, upper = math.exp(lowlog), math.exp(highlog)
        if any(v is not None and not math.isfinite(v) for v in (z, lower, upper)):
            _error("Parameter inference exceeds finite float64 support.", "numerical_failure")
        kind = "mean" if j < p.q else "residual_variance" if j in diagonals else "residual_covariance"
        status = "log-delta normal; no zero-variance test" if j in diagonals and se else "asymptotic normal" if se else "unavailable: zero first-order variance"
        parameters.append([name, kind, estimate, se, z, math.erfc(abs(z)/math.sqrt(2)) if z is not None else None, lower, upper, status])
    cluster_marker = "cluster"
    while cluster_marker in p.parameter_names:
        cluster_marker = "_"+cluster_marker
    input_order = list(dict.fromkeys([e["y"] for e in p.equations]+p.columns+([p.options["cluster"]] if p.options["cluster"] else [])))
    summary_order = ("log_likelihood", "mean_scale", "stationarity", "n", "n_equations", "n_mean_parameters", "n_clusters", "covariance_type", "convergence", "solver")
    frames = dict(inputs=table({name: state["inputs"][name] for name in input_order}, index=p.index),
                  parameters=table(parameters, columns=["parameter", "kind", "estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper", "inference_status"]),
                  **{key: _matrix(fit[key], p.parameter_names) for key in ("information", "bread", "meat", "covariance")},
                  sigma=_matrix(fit["sigma"], [e["name"] for e in p.equations]),
                  fitted=table(fit["fitted"], columns=[e["name"] for e in p.equations], index=p.index),
                  residuals=table(fit["residuals"], columns=[e["name"] for e in p.equations], index=p.index),
                  case_scores=table(fit["case_scores"], columns=p.parameter_names, index=p.index),
                  cluster_scores=table([[label, *score] for label, score in zip(p.cluster_labels, fit["cluster_scores"])], columns=[cluster_marker, *p.parameter_names]),
                  case_likelihood=table(fit["case_loglikelihood"], columns=["log_likelihood"], index=p.index),
                  gradient=table([[name, score] for name, score in zip(p.parameter_names, fit["gradient"])], columns=["parameter", "score"]),
                  fit_summary=table([[key, json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)] for key in summary_order for value in [fit[key]]], columns=["setting", "json"]))
    # Assign immutable index objects after table() to preserve MultiIndex metadata.
    for key in ("inputs", "fitted", "residuals", "case_scores", "case_likelihood"):
        frames[key].index = p.index.copy()
    return TableSet(frames, title="Nonlinear seemingly unrelated regression", method="nlsur", contract=SCHEMA, settings=state["settings"], nonlinear_sur_state=state, notes=NOTES)


@procedure
def nlsur(data, equations, *, start=None, covariance="oim", cluster=None, level=.95, tolerance=1e-8,
          max_iterations=500, max_work=100_000_000, max_bytes=256_000_000, device="cpu"):
    """Fit shared smooth nonlinear Gaussian mean equations on a complete sample."""
    with torch.inference_mode(False):
        if device != "cpu":
            _error("Nonlinear SUR supports explicit native CPU float64 only.", "unsupported_device")
        options = _options(covariance, cluster, tolerance, max_iterations, max_work, max_bytes)
        level, _ = _confidence(level)
        p = _prepare(data, equations, options, start)
        physical, convergence = _fit(p)
        fit = _replay(physical, p) | dict(convergence=convergence, solver=SOLVER)
        state = dict(schema=SCHEMA, equations=p.equations, starts=p.starts, options=options, inputs=p.inputs, index=p.index_record,
                     tapes=[dict(parameters=list(f.parameters), columns=list(f.columns), tape=[list(v) for v in f.tape]) for f in p.formulas],
                     level=level, fit=fit, settings=_settings(p, level))
        state["checksum"] = _checksum(state)
        return _assemble(state, p)


def _same(saved, expected, name):
    if isinstance(expected, (list, float)):
        def finite(value):
            return all(finite(v) for v in value) if isinstance(value, list) else isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        if not finite(saved):
            _error(f"Saved {name} requires finite real numeric cells.", "invalid_result")
        try:
            value, actual = torch.tensor(saved, dtype=DT, device="cpu"), torch.tensor(expected, dtype=DT, device="cpu")
            if value.shape != actual.shape or not bool(torch.isfinite(value).all()):
                _error(f"Saved {name} has incomplete finite dimensions.", "invalid_result")
            if name in {"information", "bread", "meat", "covariance"}:
                scale = actual.diag().abs().sqrt()
                allowed = 2e-9*scale[:, None]*scale[None, :]
                match = bool(((actual-value).abs() <= allowed).all())
            else:
                match = bool(torch.allclose(value, actual, rtol=2e-9, atol=2e-12))
            if not match:
                _error(f"Saved {name} differs from complete physical replay.", "invalid_result")
        except (TypeError, ValueError, RuntimeError) as error:
            raise AnalysisError("invalid_result", f"Saved {name} is not a finite numeric payload.") from error
    elif type(saved) is not type(expected) or saved != expected:
        _error(f"Saved {name} differs from the canonical numerical protocol.", "invalid_result")


def _validate_inner(result):
    attrs = result.attrs if isinstance(result, TableSet) else result
    if not isinstance(attrs, Mapping) or set(attrs) != {"method", "contract", "settings", "nonlinear_sur_state", "notes"}:
        _error("Supply the complete nonlinear SUR TableSet or portable attrs.", "invalid_result")
    state = attrs["nonlinear_sur_state"]
    fields = {"schema", "equations", "starts", "options", "inputs", "index", "tapes", "level", "fit", "settings", "checksum"}
    if not isinstance(state, dict) or set(state) != fields or state["schema"] != SCHEMA or not isinstance(state["checksum"], str) or len(state["checksum"]) != 64 or not hmac.compare_digest(state["checksum"], _checksum(state)):
        _error("Saved nonlinear SUR schema/checksum is invalid.", "invalid_result")
    keys = {"covariance", "cluster", "tolerance", "max_iterations", "max_work", "max_bytes"}
    if not isinstance(state["options"], dict) or set(state["options"]) != keys:
        _error("Saved optimizer/resource options are incomplete.", "invalid_result")
    options = _options(**state["options"])
    if not isinstance(state["inputs"], dict) or not state["inputs"]:
        _error("Saved original input columns are incomplete.", "invalid_result")
    p = _prepare(state["inputs"], state["equations"], options, state["starts"], index_record=state["index"])
    if state["equations"] != p.equations or state["starts"] != p.starts or state["inputs"] != p.inputs:
        _error("Saved DSL, starts or complete inputs are not canonical.", "invalid_result")
    tapes = [dict(parameters=list(f.parameters), columns=list(f.columns), tape=[list(v) for v in f.tape]) for f in p.formulas]
    if state["tapes"] != tapes:
        _error("Saved DSL tapes differ from safe source reparse.", "invalid_result")
    fit = state["fit"]
    if not isinstance(fit, dict) or not isinstance(fit.get("params"), list) or len(fit["params"]) != len(p.parameter_names) or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)) for v in fit["params"]):
        _error("Saved physical parameter vector is incomplete.", "invalid_result")
    physical = torch.tensor(fit["params"], dtype=DT, device="cpu")
    replay = _replay(physical, p)
    if set(fit) != set(replay)|{"convergence", "solver"}:
        _error("Saved full moment schema is incomplete.", "invalid_result")
    for name, expected in replay.items():
        if name == "parameter_names":
            if fit[name] != expected:
                _error("Saved physical parameter aliases disagree.", "invalid_result")
        elif name == "stationarity":
            if not isinstance(fit[name], dict) or set(fit[name]) != set(expected):
                _error("Saved physical stationarity structure is invalid.", "invalid_result")
            for key, value in expected.items():
                _same(fit[name][key], value, key)
        else:
            _same(fit[name], expected, name)
    convergence = fit["convergence"]
    if not isinstance(convergence, dict) or set(convergence) != {"starts", "selected_start", "work_used", "max_work", "globally_certified", "solver"} or convergence["globally_certified"] is not False or convergence["solver"] != SOLVER or fit["solver"] != SOLVER:
        _error("Saved local multistart protocol is invalid.", "invalid_result")
    if type(convergence["work_used"]) is not int or not 0 < convergence["work_used"] <= options["max_work"] or convergence["max_work"] != options["max_work"]:
        _error("Saved derivative work accounting is invalid.", "invalid_result")
    records, starts = convergence["starts"], _start_vectors(p)
    if not isinstance(records, list) or len(records) != len(starts):
        _error("Saved deterministic starts are incomplete.", "invalid_result")
    accepted = []
    for i, (record, initial) in enumerate(zip(records, starts)):
        if not isinstance(record, dict) or record.get("start") != (initial/p.mean_scale).tolist():
            _error("Saved initial physical starts differ from the deterministic protocol.", "invalid_result")
        if record.get("status") == "rejected":
            if set(record) != {"start", "status", "code"} or record["code"] not in {"rank_deficient", "no_finite_mle", "nonconvergence", "numerical_failure", "invalid_start"}:
                _error("Saved rejected-start diagnostics are invalid.", "invalid_result")
            continue
        if set(record) != {"start", "status", "iterations", "params", "log_likelihood", "stationarity"} or record["status"] != "stationary_local" or type(record["iterations"]) is not int or not 0 <= record["iterations"] <= options["max_iterations"]+10:
            _error("Saved successful-start diagnostics are invalid.", "invalid_result")
        candidate = torch.tensor(record["params"], dtype=DT, device="cpu")
        if candidate.shape != physical.shape or not bool(torch.isfinite(candidate).all()):
            _error("Saved successful physical candidate is incomplete.", "invalid_result")
        moments = _moments(candidate, p)
        diagnostic, _, stationary = _stationarity(candidate, p, moments)
        if not stationary or record["stationarity"] != diagnostic:
            _error("Saved multistart candidate fails full physical replay.", "invalid_result")
        _same(record["log_likelihood"], moments["log_likelihood"], "candidate likelihood")
        accepted.append((moments["log_likelihood"], i))
    selected = convergence["selected_start"]
    if type(selected) is not int or not accepted or selected != max(accepted, key=lambda v: v[0])[1] or records[selected]["params"] != fit["params"]:
        _error("Saved selection is not the best certified declared candidate.", "invalid_result")
    level, _ = _confidence(state["level"])
    resources = state["settings"].get("resources") if isinstance(state["settings"], dict) else None
    if not isinstance(resources, dict) or type(resources.get("budget_bytes")) is not int or not 0 < resources["budget_bytes"] <= options["max_bytes"]:
        _error("Saved resident resource plan is invalid.", "invalid_result")
    p.resources = _plan(p.n, p.m, p.q, len(p.columns), [len(f.tape) for f in p.formulas], options, budget_bytes=resources["budget_bytes"])
    if state["settings"] != _settings(p, level):
        _error("Saved settings differ from the full computational/statistical contract.", "invalid_result")
    expected = _assemble(state, p)
    if dict(attrs) != expected.attrs:
        _error("Saved attrs differ from the complete sealed result.", "invalid_result")
    if isinstance(result, TableSet) and (result.title != expected.title or set(result) != set(expected) or any(not isinstance(result[key], pd.DataFrame) or not result[key].equals(expected[key]) or result[key].attrs != expected[key].attrs for key in expected)):
        _error("Saved public tables disagree with the complete state.", "invalid_result")
    return state, p


def _validated(result, level=None):
    """Strict safe reparse and physical replay; optimization is never invoked."""
    with torch.inference_mode(False):
        try:
            state, p = _validate_inner(result)
        except AnalysisError as error:
            if error.code == "invalid_result":
                raise
            raise AnalysisError("invalid_result", f"Saved nonlinear SUR replay rejected: {error}") from error
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OverflowError) as error:
            raise AnalysisError("invalid_result", "Saved nonlinear SUR requires a complete finite canonical payload.") from error
        if level is not None:
            updated = copy.deepcopy(state)
            updated["level"], _ = _confidence(level)
            updated["settings"] = _settings(p, updated["level"])
            updated["checksum"] = _checksum(updated)
            state = updated
        return state, p


@procedure
def nlsur_restore(result, *, level=None):
    """Restore complete physical Gaussian ML tables after replay, without refit."""
    state, p = _validated(result, level)
    return _assemble(state, p)
