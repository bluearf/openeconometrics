"""Normalized two/three-alternative Gaussian random-utility choice.

Only differences of Gaussian errors are identified.  The first declared
alternative is the reference, and the reference/second error-difference
variance is one. Three alternatives therefore have two covariance parameters.
CDF values use the existing native quadrature; derivatives use the Gaussian
boundary identities, never the adaptive quadrature's branch decisions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.discrete.bivariate import log_bvn_cdf
from openecon.econometrics.discrete.nested_logit import (
    _confidence, _error, _finite_matrix, _finite_vector, _identity, _indicator,
    _key, _list, _marker, _matches, _matrix, _numeric, _psd,
)
from openecon.econometrics.nonparametric.common import procedure
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.engines.optimize import maximize_bfgs
from openecon.resources import plan_workspace, workspace_budget_bytes

DT = torch.float64
SCHEMA = "multinomial_probit_v1"
STARTS = ((1.0, 0.0), (0.75, -0.35), (1.5, 0.35))
SD_MIN, SD_MAX, RHO_MAX = .05, 20.0, .98
QUADRATURE_ORDER = 96
MAX_STANDARDIZED = 16.0
SOLVER = "native Torch multistart BFGS; analytic Gaussian-CDF derivatives; physical stationary interior local ML"
NOTES = [
    "Two or three explicitly ordered typed alternatives. Reference error differences have first variance one; utility location is removed within each choice case.",
    "Three-alternative free covariance uses physical sd3 and rho3. The supported numerical interior is 0.05 < sd3 < 20 and abs(rho3) < 0.98; fixed covariance permits these endpoints. This is a computational bound, not the mathematical positive-definite domain.",
    "OIM contains the full physical coefficient/covariance Hessian and cross blocks. HC0 uses whole-case scores; CR0 sums these within respondents. No degrees-of-freedom multiplier or finite-cluster coverage guarantee is supplied.",
    "Three deterministic covariance starts are compared. Accepted solutions are finite, identified stationary local maxima, not a proof of a global maximum.",
    "Availability is conditioned on as declared. No probability renormalization, covariance ridge, implicit category coding, or missing-row deletion is used.",
    "Public Gaussian derivatives require absolute standardized and conditional thresholds at most 16. This keeps Gaussian boundary derivative factors away from float64 underflow before utility/attribute rescaling. Outside this computational derivative-accuracy domain the procedure refuses; probabilities are not clamped to the boundary.",
    "Portable restoration reparses complete retained inputs and replays all physical moments without optimization. The checksum detects corruption; it is not authentication.",
]


class _LogBivariateNormal(torch.autograd.Function):
    """Native accurate values with analytical first, second and higher chains.

    Saving the returned output (rather than a detached internal log probability)
    makes its analytical backward participate in double and triple backward.
    """
    @staticmethod
    def forward(ctx, a, b, rho):
        with torch.device("cpu"):
            value = log_bvn_cdf(a, b, rho)
        ctx.save_for_backward(a, b, rho, value)
        ctx.save_for_forward(a, b, rho, value)
        return value

    @staticmethod
    def backward(ctx, weight):
        a, b, rho, value = ctx.saved_tensors
        ga, gb, gr = _cdf_ratios(a, b, rho, value)
        return weight*ga, weight*gb, weight*gr

    @staticmethod
    def jvp(ctx, da, db, drho):
        a, b, rho, value = ctx.saved_tensors
        ga, gb, gr = _cdf_ratios(a, b, rho, value)
        return ga*da+gb*db+gr*drho


def _cdf_ratios(a, b, rho, value):
    complement = (1-rho)*(1+rho)
    sd = complement.sqrt()
    log_phi = -.5*math.log(2*math.pi)
    ga = torch.exp(log_phi-.5*a.square()+torch.special.log_ndtr((b-rho*a)/sd)-value)
    gb = torch.exp(log_phi-.5*b.square()+torch.special.log_ndtr((a-rho*b)/sd)-value)
    form = a.square()+(b-rho*a).square()/complement
    gr = torch.exp(-math.log(2*math.pi)-torch.log(sd)-.5*form-value)
    return ga, gb, gr


def _log_cdf(a, b, rho):
    a, b, rho = torch.broadcast_tensors(a, b, rho)
    return _LogBivariateNormal.apply(a, b, rho)


def _columns(chosen, x, case, alternative, available=None, cluster=None):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 8:
        _error("x must list 1..8 distinct numeric utility column names.", "invalid_spec")
    selected = [chosen, case, alternative, *x]+[v for v in (available, cluster) if v is not None]
    if any(not isinstance(v, str) or not v for v in selected) or len(set(selected)) != len(selected):
        _error("Selected columns must be distinct nonempty strings.", "invalid_spec")
    return dict(chosen=chosen, x=list(x), case=case, alternative=alternative, available=available, cluster=cluster)


def _options(vce="oim", max_iterations=300, tolerance=1e-9,
             max_work=2_000_000_000, max_bytes=256_000_000, quadrature_order=96):
    if not isinstance(vce, str) or vce not in {"oim", "hc0", "cr0"}:
        _error("vce must be exactly oim, hc0 or cr0.", "invalid_option")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or not 1 <= max_iterations <= 1000:
        _error("max_iterations must be an integer in [1,1000].", "invalid_option")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(float(tolerance)) or not 1e-12 <= tolerance <= 1e-5:
        _error("tolerance must be finite and in [1e-12,1e-5].", "invalid_option")
    for name, value, maximum in (("max_work", max_work, 2_000_000_000), ("max_bytes", max_bytes, 2_000_000_000)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
            _error(f"{name} must be an integer in [1,2e9].", "invalid_resource_budget")
    if type(quadrature_order) is not int or quadrature_order != QUADRATURE_ORDER:
        _error("The native conservative quadrature work order is fixed at 96.", "invalid_option")
    return dict(vce=vce, max_iterations=int(max_iterations), tolerance=float(tolerance),
                max_work=int(max_work), max_bytes=int(max_bytes), quadrature_order=QUADRATURE_ORDER)


def _catalogue(value):
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        _error("alternatives must explicitly order two or three typed alternative labels.", "invalid_spec")
    result = [_identity(v, "declared alternative") for v in value]
    if len({_key(v) for v in result}) != len(result):
        _error("Declared alternative labels must be unique without coercion.", "invalid_identifier")
    return result


def _fixed_covariance(value, j):
    if j == 2 and value is None:
        return [[1.0]]
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad or value.is_complex():
            _error("Fixed covariance requires a real resident CPU matrix without gradients.", "unsupported_input")
        value = value.tolist()
    elif type(value).__module__.startswith("numpy"):
        value = value.tolist()
    if not _finite_matrix(value, j-1, j-1):
        _error("covariance requires the complete finite normalized error-difference matrix.", "invalid_covariance")
    matrix = [[float(v) for v in row] for row in value]
    if matrix[0][0] != 1.0 or any(matrix[r][c] != matrix[c][r] for r in range(j-1) for c in range(j-1)):
        _error("Fixed covariance must be symmetric with reference/second difference variance exactly one.", "invalid_covariance")
    if j == 3:
        sd = math.sqrt(matrix[1][1]) if matrix[1][1] > 0 else 0.0
        rho = matrix[0][1]/sd if sd else math.inf
        if not SD_MIN <= sd <= SD_MAX or abs(rho) > RHO_MAX:
            _error("Fixed covariance lies outside sd3 [0.05,20], abs(rho3) <= 0.98 numerical support.", "invalid_covariance")
    return matrix


def _index_record(index):
    if isinstance(index, pd.CategoricalIndex):
        _error("Categorical row indexes are outside the portable index contract.", "unsupported_input")
    return dict(labels=[encode(v) for v in index], names=[encode(v) for v in index.names],
                multi=isinstance(index, pd.MultiIndex), dtype=None if isinstance(index, pd.MultiIndex) else str(index.dtype),
                range=[index.start, index.stop, index.step] if isinstance(index, pd.RangeIndex) else None)


def _index_restore(value, n):
    if not isinstance(value, dict) or set(value) != {"labels", "names", "multi", "dtype", "range"} or not isinstance(value["labels"], list) or len(value["labels"]) != n:
        _error("Saved row-index schema is incomplete.", "invalid_result")
    try:
        labels, names = [decode(v) for v in value["labels"]], [decode(v) for v in value["names"]]
        if value["multi"] is True:
            result = pd.MultiIndex.from_tuples(labels, names=names)
        elif value["multi"] is False and len(names) == 1:
            result = (pd.RangeIndex(*value["range"], name=names[0]) if value["range"] is not None
                      else pd.Index(labels, dtype=value["dtype"], name=names[0], tupleize_cols=False))
        else:
            _error("Saved row-index type is unsupported.", "invalid_result")
        if _index_record(result) != value:
            _error("Saved row identities cannot be restored canonically.", "invalid_result")
        return result
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved row-index codes are invalid.") from error


def _plan(n, cases, q, free, options, *, for_fit, budget_bytes=None):
    if not (2 if for_fit else 1) <= n <= (1536 if for_fit else 8192) or not 1 <= q <= 8 or not 1 <= cases <= (512 if for_fit else 4096):
        _error("Fit supports at most 512 cases/1536 rows; query preparation supports 4096 cases/8192 rows; 1..8 attributes.", "resource_limit")
    k = q+2*int(free)
    work = cases*(96+6*k)*8*(options["max_iterations"]*(3 if free else 1) if for_fit else k+1)
    if work > options["max_work"]:
        _error("Gaussian derivative work exceeds max_work before tensor allocation.", "work_budget")
    memory = 8*(n*(q*4+16)+cases*(96*20+k*k*24+k*16)+32*k*k)
    budget = min(options["max_bytes"], workspace_budget_bytes()) if budget_bytes is None else min(budget_bytes, options["max_bytes"])
    plan = plan_workspace("multinomial_probit_gaussian_choices", {"complete_inputs_gaussian_values_and_analytic_derivatives": memory}, budget_bytes=budget).record()
    return plan | dict(planned_work=work, max_work=options["max_work"], quadrature_order=96,
                       full_parameter_count=k, max_bytes=options["max_bytes"])


@dataclass
class Prepared:
    X: torch.Tensor
    Z: torch.Tensor
    scale: torch.Tensor
    chosen: list
    available: list
    case_labels: list
    case_rows: list
    alternative_labels: list
    alt_codes: list
    catalogue: list
    fixed_covariance: list | None
    parameters: list
    cluster_labels: list
    case_cluster_codes: list
    columns: dict
    options: dict
    inputs: dict
    resources: dict
    index: pd.Index
    available_rows: list
    selected_rows: list
    selected_competitors: list
    binary_cases: list
    triple_cases: list
    selected_matrix: torch.Tensor

    @property
    def n(self):
        return len(self.available)

    @property
    def q(self):
        return len(self.columns["x"])

    @property
    def free_covariance(self):
        return self.fixed_covariance is None

    @property
    def parameter_names(self):
        return self.parameters


def _prepare(data, columns, options, *, catalogue=None, fixed_covariance=None,
             require_chosen=True, for_fit=True, budget_bytes=None):
    catalogue = _catalogue(catalogue)
    fixed_covariance = _fixed_covariance(fixed_covariance, len(catalogue))
    free = fixed_covariance is None
    names = [columns["case"], columns["alternative"], *columns["x"]]
    if require_chosen:
        names.append(columns["chosen"])
    names += [columns[k] for k in ("available", "cluster") if columns[k] is not None and (k != "cluster" or for_fit)]
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique or any(name not in data.columns for name in names):
            _error("Selected input columns must be unique and present.", "invalid_spec")
        n, index = len(data), data.index.copy()
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Every selected input column must be present.", "invalid_spec")
        try:
            n = len(data[columns["case"]])
        except TypeError as error:
            raise AnalysisError("unsupported_input", "Input columns require resident positional vectors.") from error
        series = [data[name] for name in names if isinstance(data[name], pd.Series)]
        index = series[0].index.copy() if series else pd.RangeIndex(n)
        if any(not v.index.equals(index) for v in series):
            _error("Selected Series require complete identical row indexes.", "shape_mismatch")
    else:
        _error("Use a resident DataFrame or column mapping; Dataset is unsupported.", "unsupported_input")
    if not (2 if for_fit else 1) <= n <= (1536 if for_fit else 8192):
        _error("The complete resident row count exceeds the supported fit/query bound.", "resource_limit")
    values = {name: _list(data[name], n, name) for name in names}
    cases = [_identity(v, "case") for v in values[columns["case"]]]
    alternatives = [_identity(v, "alternative") for v in values[columns["alternative"]]]
    code = {_key(v): j for j, v in enumerate(catalogue)}
    if any(_key(v) not in code for v in alternatives):
        _error("Every input alternative must occur in the explicit fitted covariance catalogue.", "invalid_identifier")
    alt_codes = [code[_key(v)] for v in alternatives]
    chosen = _indicator(values[columns["chosen"]], "chosen") if require_chosen else [0]*n
    available = _indicator(values[columns["available"]], "available") if columns["available"] else [1]*n
    raw = [_numeric(values[name], name) for name in columns["x"]]
    clusters = [_identity(v, "cluster") for v in values[columns["cluster"]]] if columns["cluster"] and for_fit else None
    case_labels, case_rows, case_map = [], [], {}
    for row, label in enumerate(cases):
        if _key(label) not in case_map:
            case_map[_key(label)] = len(case_labels)
            case_labels.append(label)
            case_rows.append([])
        case_rows[case_map[_key(label)]].append(row)
    if for_fit and len(case_labels) < 2:
        _error("At least two complete choice cases are required.", "invalid_choice")
    resources = _plan(n, len(case_labels), len(raw), free, options, for_fit=for_fit, budget_bytes=budget_bytes)
    cluster_labels, cluster_map, case_cluster_codes = [], {}, []
    available_rows, selected_rows, competitors, binary, triple, matrices = [], [], [], [], [], []
    coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    for ci, rows in enumerate(case_rows):
        if len(set(alt_codes[i] for i in rows)) != len(rows):
            _error("Alternatives must be unique within every typed choice case.", "duplicate_alternative")
        active = [i for i in rows if available[i]]
        if not (2 if for_fit else 1) <= len(active) <= len(catalogue):
            _error("Fit cases require two/three available alternatives; queries may contain a structural singleton.", "invalid_choice")
        if any(chosen[i] and not available[i] for i in rows) or (require_chosen and sum(chosen[i] for i in rows) != 1):
            _error("Each case requires exactly one chosen available alternative.", "invalid_choice")
        selected = next((i for i in active if chosen[i]), active[0])
        others = [i for i in active if i != selected]
        selected_rows.append(selected)
        competitors.append(others)
        available_rows.extend(active)
        if len(others) == 1:
            binary.append(ci)
        elif len(others) == 2:
            triple.append(ci)
        base = coordinates[alt_codes[selected]]
        matrix = [[coordinates[alt_codes[i]][j]-base[j] for j in range(2)] for i in others]
        matrices.append(matrix+[[0.0, 0.0]]*(2-len(matrix)))
        if clusters is not None:
            if len({_key(clusters[i]) for i in rows}) != 1:
                _error("A respondent cluster must be constant within the entire case.", "split_case_cluster")
            label = clusters[rows[0]]
            if _key(label) not in cluster_map:
                cluster_map[_key(label)] = len(cluster_labels)
                cluster_labels.append(label)
            case_cluster_codes.append(cluster_map[_key(label)])
    if for_fit and options["vce"] == "cr0" and len(cluster_labels) < 2:
        _error("CR0 requires at least two complete respondent clusters.", "invalid_cluster")
    X = torch.tensor(list(zip(*raw)), dtype=DT, device="cpu")
    centered = torch.empty_like(X, device="cpu")
    for rows in case_rows:
        active = next(i for i in rows if available[i])
        centered[rows] = X[rows]-X[active]
    scale = centered[available_rows].square().mean(0).sqrt()
    if not for_fit:
        scale = torch.where(scale > 0, scale, torch.ones_like(scale, device="cpu"))
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        _error("A common within-case utility column is unidentified.", "rank_deficient")
    Z = centered/scale
    if for_fit:
        singular = torch.linalg.svdvals(Z[available_rows])
        if singular.numel() < len(raw) or float(singular[-1]) <= float(singular[0])*1e-10:
            _error("Within-case utility differences are rank deficient or ill-conditioned.", "rank_deficient")
    parameters = list(columns["x"])
    if free:
        if set(parameters) & {"sd3", "rho3"}:
            _error("Free covariance parameter names sd3/rho3 cannot also name utility columns.", "invalid_spec")
        parameters += ["sd3", "rho3"]
    inputs = dict(chosen=chosen, x=X.tolist(), case=cases, alternative=alternatives,
                  available=available, cluster=clusters, index=_index_record(index))
    return Prepared(X, Z, scale, chosen, available, case_labels, case_rows, alternatives, alt_codes,
                    catalogue, fixed_covariance, parameters, cluster_labels, case_cluster_codes,
                    columns, options, inputs, resources, index, available_rows, selected_rows,
                    competitors, binary, triple, torch.tensor(matrices, dtype=DT, device="cpu"))


def _gamma(theta, p):
    if p.fixed_covariance is not None:
        matrix = theta.new_tensor(p.fixed_covariance)
        if len(p.catalogue) == 2:
            matrix = torch.stack((torch.stack((matrix[0, 0], theta.sum()*0)),
                                  torch.stack((theta.sum()*0, theta.sum()*0))))
        return matrix
    sd, rho = theta[p.q], theta[p.q+1]
    return torch.stack((torch.stack((theta.new_tensor(1.0), sd*rho)), torch.stack((sd*rho, sd.square()))))


def _event_logs(theta, p, selected, competitors, matrix, binary, triple, X=None):
    X = p.X if X is None else X
    gamma = _gamma(theta, p)
    comparison = matrix@gamma@matrix.transpose(-1, -2)
    result = theta.new_zeros(len(selected))
    for indices, dimension in ((binary, 1), (triple, 2)):
        if not indices:
            continue
        own = [selected[i] for i in indices]
        other = [[competitors[i][j] for j in range(dimension)] for i in indices]
        other_indices = torch.tensor(other, dtype=torch.int64, device="cpu")
        deltas = (X[own, None, :]-X[other_indices])@theta[:p.q]
        variance = comparison[indices].diagonal(dim1=-2, dim2=-1)[:, :dimension]
        threshold = deltas/variance.sqrt()
        if not bool(torch.isfinite(threshold).all()) or bool((threshold.detach().abs() > MAX_STANDARDIZED).any()):
            _error("Standardized Gaussian thresholds exceed the absolute 16 computational derivative-accuracy domain.", "numerical_domain")
        if dimension == 1:
            logs = torch.special.log_ndtr(threshold[:, 0])
        else:
            correlation = comparison[indices, 0, 1]/(variance[:, 0]*variance[:, 1]).sqrt()
            complement = (1-correlation)*(1+correlation)
            a, b = threshold[:, 0], threshold[:, 1]
            conditional = torch.stack(((b-correlation*a)/complement.sqrt(), (a-correlation*b)/complement.sqrt()), dim=1)
            if not bool(torch.isfinite(conditional).all()) or bool((conditional.detach().abs() > MAX_STANDARDIZED).any()):
                _error("Conditional Gaussian thresholds exceed the absolute 16 computational derivative-accuracy domain.", "numerical_domain")
            logs = _log_cdf(threshold[:, 0], threshold[:, 1], correlation)
        result = result.index_add(0, torch.tensor(indices, dtype=torch.int64, device="cpu"), logs)
    return result


def _case_logs(theta, p):
    return _event_logs(theta, p, p.selected_rows, p.selected_competitors, p.selected_matrix,
                       p.binary_cases, p.triple_cases)


def _log_probabilities(theta, p, ci, X=None):
    """All available outcome logs; derivatives retain raw-cell utility chains."""
    rows = [i for i in p.case_rows[ci] if p.available[i]]
    coordinates = theta.new_tensor([[0., 0.], [1., 0.], [0., 1.]])
    competitors = [[i for i in rows if i != own] for own in rows]
    matrices = []
    for own, others in zip(rows, competitors):
        matrices.append(torch.stack([coordinates[p.alt_codes[i]]-coordinates[p.alt_codes[own]] for i in others])
                        if others else theta.new_zeros((0, 2)))
    matrices = torch.stack([torch.cat((v, theta.new_zeros((2-len(v), 2)))) for v in matrices])
    binary = list(range(len(rows))) if len(rows) == 2 else []
    triple = list(range(len(rows))) if len(rows) == 3 else []
    logs = _event_logs(theta, p, rows, competitors, matrices, binary, triple, X)
    if len(rows) == 1:
        logs = logs+theta.sum()*0+(p.X if X is None else X).sum()*0
    return dict(rows=rows, log_probability=logs, probability=logs.exp())


def _covariance_domain(theta, p, *, final=False):
    if not bool(torch.isfinite(theta).all()):
        return False
    if p.free_covariance:
        sd, rho = (float(v.detach()) for v in theta[p.q:])
        if final:
            if not SD_MIN < sd < SD_MAX or not abs(rho) < RHO_MAX:
                _error("Free difference covariance reaches the supported numerical interior boundary.", "boundary_solution")
        elif not 1e-4 < sd < 1e4 or not abs(rho) < 1-1e-8:
            return False
    gamma = _gamma(theta, p).detach()
    if len(p.catalogue) == 3:
        transforms = theta.new_tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [-1., 1.]], [[0., -1.], [1., -1.]]])
        comparisons = transforms@gamma@transforms.transpose(-1, -2)
        variance = comparisons.diagonal(dim1=-2, dim2=-1)
        complement = 1-comparisons[:, 0, 1].square()/(variance[:, 0]*variance[:, 1])
        if not bool((variance > 1e-8).all()) or not bool((complement > 1e-8).all()):
            if final:
                _error("An alternative comparison covariance exceeds supported Gaussian conditioning.", "boundary_solution")
            return False
    return True


class _Objective:
    def __init__(self, p):
        self.p, self.used = p, 0

    def charge(self, factor=1):
        self.used += len(self.p.case_labels)*(96+6*len(self.p.parameters))*8*factor
        if self.used > self.p.options["max_work"]:
            _error("Actual Gaussian derivative work exceeded max_work.", "work_budget")

    def physical(self, point):
        if self.p.free_covariance:
            return torch.cat((point[:self.p.q]/self.p.scale, point[self.p.q:self.p.q+1].exp(), point[self.p.q+1:].tanh()))
        return point/self.p.scale

    def value(self, point):
        theta = self.physical(point)
        if not _covariance_domain(theta, self.p):
            return point.sum()*0+point.new_tensor(-torch.inf)
        try:
            return _case_logs(theta, self.p).sum()
        except AnalysisError as error:
            if error.code != "numerical_domain":
                raise
            return point.sum()*0+point.new_tensor(-torch.inf)

    def __call__(self, point):
        self.charge()
        with torch.enable_grad():
            x = point.detach().requires_grad_()
            value = self.value(x)
            gradient = torch.autograd.grad(value, x)[0]
        return value.detach(), gradient.detach()

    def hessian(self, point):
        self.charge(len(point)+1)
        with torch.enable_grad():
            return torch.autograd.functional.hessian(self.value, point, vectorize=True).detach()


def _inverse(information):
    diagonal = information.diag()
    if not bool(torch.isfinite(information).all()) or not bool((diagonal > 0).all()):
        _error("Full physical observed information is nonfinite or nonpositive.", "no_finite_mle")
    scale = diagonal.sqrt()
    normalized = information/scale[:, None]/scale[None, :]
    eigen = torch.linalg.eigvalsh((normalized+normalized.T)/2)
    if float(eigen[0]) <= 1e-10*float(eigen[-1]):
        _error("Full utility/covariance information is unidentified or ill-conditioned.", "rank_deficient")
    chol, status = torch.linalg.cholesky_ex(normalized)
    if int(status):
        _error("Full physical observed information is not positive definite.", "no_finite_mle")
    inverse = torch.cholesky_inverse(chol)/scale[:, None]/scale[None, :]
    return (inverse+inverse.T)/2


def _moments(theta, p):
    try:
        with torch.enable_grad():
            point = theta.detach().requires_grad_()
            values = _case_logs(point, p)
            gradient = torch.autograd.grad(values.sum(), point)[0]
            hessian = torch.autograd.functional.hessian(lambda x: _case_logs(x, p).sum(), theta, vectorize=True)
            scores = torch.autograd.functional.jacobian(lambda x: _case_logs(x, p), theta, vectorize=True, strategy="forward-mode")
    except AnalysisError:
        raise
    except (RuntimeError, ValueError, OverflowError) as error:
        raise AnalysisError("numerical_failure", "Physical Gaussian likelihood derivatives could not be evaluated.") from error
    information = -(hessian+hessian.T)/2
    if any(not bool(torch.isfinite(v).all()) for v in (values, gradient, information, scores)):
        _error("Full physical Gaussian likelihood moments are nonfinite.", "numerical_failure")
    return dict(information=information.detach(), gradient=gradient.detach(), case_scores=scores.detach(),
                case_loglikelihood=values.detach(), log_likelihood=float(values.sum().detach()))


def _stationarity(theta, p, moments):
    _covariance_domain(theta, p, final=True)
    bread = _inverse(moments["information"])
    step = bread@moments["gradient"]
    units = torch.cat((p.scale, theta.new_tensor([1/max(float(theta[p.q]), SD_MIN), 1.]))) if p.free_covariance else p.scale
    norm = float((step*units).abs().max())
    decrement = float(moments["gradient"]@step)
    step_limit = max(20*p.options["tolerance"], 1e-8)
    decrement_limit = max(100*p.options["tolerance"]**2, 1e-16)*len(p.case_labels)
    diagnostic = dict(normalized_parameter_step=norm, score_decrement=max(0., decrement),
                      physical_score_max=float(moments["gradient"].abs().max()), step_limit=step_limit, decrement_limit=decrement_limit)
    if not math.isfinite(norm) or not math.isfinite(decrement) or decrement < -1e-12:
        _error("Physical stationarity diagnostics are nonfinite.", "no_finite_mle")
    return diagnostic, bread, norm <= step_limit and decrement <= decrement_limit


def _fit(p):
    objective, attempts, accepted = _Objective(p), [], []
    starts = STARTS if p.free_covariance else (None,)
    for start in starts:
        initial = torch.zeros(len(p.parameters), dtype=DT, device="cpu")
        if start is not None:
            initial[p.q:] = initial.new_tensor([math.log(start[0]), math.atanh(start[1])])
        try:
            with torch.device("cpu"):
                fit = kernel_call(maximize_bfgs, objective, initial, hessian_fn=objective.hessian,
                                  max_iter=p.options["max_iterations"], scaled_gradient_tol=min(1e-12, p.options["tolerance"]**2),
                                  raise_on_failure=False)
            theta = objective.physical(fit.theta).detach()
            moments = _moments(theta, p)
            diagnostic, _, stationary = _stationarity(theta, p, moments)
            if not stationary:
                _error("Joint ML fails actual physical Newton-step/score stationarity.", "nonconvergence")
            attempts.append(dict(start=list(start) if start else None, status="stationary_interior", iterations=fit.iterations,
                                 params=theta.tolist(), log_likelihood=moments["log_likelihood"], stationarity=diagnostic))
            accepted.append((moments["log_likelihood"], theta, len(attempts)-1))
        except AnalysisError as error:
            if error.code in {"work_budget", "resource_limit", "workspace_limit"}:
                raise
            attempts.append(dict(start=list(start) if start else None, status="rejected", code=error.code))
        except (RuntimeError, ValueError, OverflowError):
            attempts.append(dict(start=list(start) if start else None, status="rejected", code="numerical_failure"))
    if not accepted:
        code = ("boundary_solution" if any(v.get("code") == "boundary_solution" for v in attempts)
                else "numerical_domain" if any(v.get("code") == "numerical_domain" for v in attempts) else "no_finite_mle")
        _error("No start passed finite interior Gaussian ML, full identification and physical stationarity.", code)
    _, theta, selected = max(accepted, key=lambda item: item[0])
    return theta, dict(starts=attempts, selected_start=selected, work_used=objective.used, max_work=p.options["max_work"],
                       globally_certified=False, solver=SOLVER)


def _evaluate(theta, p):
    moments = _moments(theta, p)
    diagnostic, bread, stationary = _stationarity(theta, p, moments)
    if not stationary:
        _error("Final/saved parameters fail actual physical stationarity.", "no_finite_mle")
    clusters = torch.zeros((len(p.cluster_labels), len(theta)), dtype=DT, device="cpu")
    if p.cluster_labels:
        clusters.index_add_(0, torch.tensor(p.case_cluster_codes, dtype=torch.int64, device="cpu"), moments["case_scores"])
    scores = clusters if p.options["vce"] == "cr0" else moments["case_scores"]
    meat = scores.T@scores
    influence = scores@bread
    covariance = bread if p.options["vce"] == "oim" else influence.T@influence
    covariance = (covariance+covariance.T)/2
    _psd(covariance, "covariance", "numerical_failure")
    rows, probabilities, logs = [], [], []
    for ci in range(len(p.case_labels)):
        pieces = _log_probabilities(theta, p, ci)
        if not bool(torch.isfinite(pieces["log_probability"]).all()) or abs(float(pieces["probability"].sum())-1) > 2e-9:
            _error("Gaussian choice probabilities fail finite complete normalization without renormalization.", "numerical_failure")
        rows.extend(pieces["rows"])
        probabilities.extend(pieces["probability"].tolist())
        logs.extend(pieces["log_probability"].tolist())
    return dict(params=theta.tolist(), beta=theta[:p.q].tolist(), parameter_names=p.parameters,
                normalized_covariance=_gamma(theta, p)[:len(p.catalogue)-1, :len(p.catalogue)-1].tolist(),
                information=moments["information"].tolist(), bread=bread.tolist(), meat=meat.tolist(), covariance=covariance.tolist(),
                gradient=moments["gradient"].tolist(), case_scores=moments["case_scores"].tolist(), cluster_scores=clusters.tolist(),
                case_loglikelihood=moments["case_loglikelihood"].tolist(), log_likelihood=moments["log_likelihood"],
                probability_rows=rows, probabilities=probabilities, log_probabilities=logs, scale=p.scale.tolist(),
                stationarity=diagnostic, covariance_type=p.options["vce"], n=p.n, n_cases=len(p.case_labels),
                n_clusters=len(p.cluster_labels), n_alternatives=len(p.catalogue))


def _checksum(state):
    return hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "checksum"}, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _settings(p, level):
    return dict(columns=p.columns, **p.options, level=level, device="cpu", precision="float64", missing="raise", weights=None,
                alternatives=p.catalogue, reference=p.catalogue[0], scale_alternative=p.catalogue[1], difference_scale=1.0,
                n_input=p.n, n_retained=p.n, n_dropped=0, n_cases=len(p.case_labels), n_clusters=len(p.cluster_labels),
                covariance="full joint physical utility/difference-covariance OIM, whole-case HC0 or respondent CR0; no multiplier",
                covariance_free=p.free_covariance, fixed_covariance=p.fixed_covariance, inference_df=None, global_wald=None,
                free_sd3_interior=[SD_MIN, SD_MAX], free_rho3_absolute_limit=RHO_MAX,
                min_comparison_variance=1e-8, min_comparison_correlation_complement=1e-8,
                max_absolute_standardized_threshold=MAX_STANDARDIZED, max_absolute_conditional_threshold=MAX_STANDARDIZED,
                threshold_policy="refuse outside computational derivative-accuracy domain; no probability/threshold clamping",
                max_cases=512, max_rows=1536, max_attributes=8, max_global_alternatives=3, max_query_rows=8192,
                utility="shared explicit raw numeric columns; common case utility location removed", numeric_magnitude_bound=1e12,
                complete_inputs_saved=True, complete_covariance_saved=True, robust_covariance_may_be_singular=True,
                restoration="complete portable numerical replay and physical-stationarity certification; no optimizer or refit",
                resources=p.resources, excluded_scope=["weights", "implicit category encoding", "common intercept", "missing-row deletion", "Dataset",
                    "CUDA/MPS", "four or more alternatives", "simulation/GHK", "endogenous availability", "boundary inference", "vendor parity"])


def _assemble(state, p):
    fit, columns, inputs = state["fit"], state["columns"], state["inputs"]
    _, zcrit = _confidence(state["level"])
    parameters = []
    for j, (name, value) in enumerate(zip(p.parameters, fit["params"])):
        se = math.sqrt(fit["covariance"][j][j])
        z = value/se if se and (j < p.q or name == "rho3") else None
        lower, upper = (value-zcrit*se, value+zcrit*se) if se else (None, None)
        status = "asymptotic normal" if se else "unavailable: zero first-order variance"
        if j >= p.q and se:
            if name == "sd3":
                lo, hi = math.log(value)-zcrit*se/value, math.log(value)+zcrit*se/value
                if hi > math.log(float(torch.finfo(DT).max)):
                    _error("Covariance-parameter interval exceeds finite float64 support.", "numerical_failure")
                lower, upper = math.exp(lo), math.exp(hi)
            else:
                delta = zcrit*se/(1-value*value)
                lower, upper = math.tanh(math.atanh(value)-delta), math.tanh(math.atanh(value)+delta)
            status = "transformed asymptotic normal; no boundary test" if name == "sd3" else "Fisher-transformed asymptotic normal interval; rho zero-null normal"
        parameters.append([name, "coefficient" if j < p.q else "difference covariance", value, se, z,
                           math.erfc(abs(z)/math.sqrt(2)) if z is not None else None, lower, upper, status])
    names = [columns["case"], columns["alternative"], columns["chosen"], *columns["x"]]
    if columns["available"]:
        names.append(columns["available"])
    if columns["cluster"]:
        names.append(columns["cluster"])
    input_rows = []
    for i in range(p.n):
        row = [i, inputs["case"][i], inputs["alternative"][i], inputs["chosen"][i], *inputs["x"][i]]
        if columns["available"]:
            row.append(inputs["available"][i])
        if columns["cluster"]:
            row.append(inputs["cluster"][i])
        input_rows.append(row)
    complete = {row: [probability, log] for row, probability, log in zip(fit["probability_rows"], fit["probabilities"], fit["log_probabilities"])}
    probability_rows = [[inputs["case"][i], i, inputs["alternative"][i], bool(inputs["available"][i]), inputs["chosen"][i],
                         *complete.get(i, [0.0, None])] for i in range(p.n)]
    skip = {"params", "beta", "parameter_names", "normalized_covariance", "information", "bread", "meat", "covariance", "gradient", "case_scores", "cluster_scores", "case_loglikelihood", "probability_rows", "probabilities", "log_probabilities"}
    frames = dict(inputs=table(input_rows, columns=[_marker("row", names), *names]),
                  parameters=table(parameters, columns=["parameter", "kind", "estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper", "inference_status"]),
                  **{name: _matrix(fit[name], p.parameters) for name in ("information", "bread", "meat", "covariance")},
                  gradient=table([[name, value] for name, value in zip(p.parameters, fit["gradient"])], columns=["parameter", "score"]),
                  case_scores=table([[label, *row] for label, row in zip(p.case_labels, fit["case_scores"])], columns=[_marker("case", p.parameters), *p.parameters]),
                  cluster_scores=table([[label, *row] for label, row in zip(p.cluster_labels, fit["cluster_scores"])], columns=[_marker("cluster", p.parameters), *p.parameters]),
                  case_likelihood=table([[label, value, math.exp(value)] for label, value in zip(p.case_labels, fit["case_loglikelihood"])], columns=["case", "log_likelihood", "chosen_probability"]),
                  probabilities=table(probability_rows, columns=["case", "row", "alternative", "available", "chosen", "probability", "log_probability"]),
                  normalized_covariance=table([[label, *row] for label, row in zip(p.catalogue[1:], fit["normalized_covariance"])], columns=["difference_alternative", *[f"difference{j+1}" for j in range(len(p.catalogue)-1)]]),
                  fit_summary=table([[key, json.dumps(fit[key], sort_keys=True, ensure_ascii=False, allow_nan=False)] for key in sorted(fit) if key not in skip], columns=["setting", "json"]),
                  settings=table([[key, json.dumps(state["settings"][key], sort_keys=True, ensure_ascii=False, allow_nan=False)] for key in sorted(state["settings"])], columns=["setting", "json"]))
    frames["inputs"].index = p.index.copy()
    return TableSet(frames, title="Normalized multinomial probit", method="mprobit", contract=SCHEMA,
                    settings=state["settings"], mprobit_state=state, notes=NOTES)


@procedure
def mprobit(data, chosen, x, *, case, alternative, alternatives, available=None, cluster=None,
            covariance=None, vce="oim", level=.95, device="cpu", weights=None,
            max_iterations=300, tolerance=1e-9, max_work=2_000_000_000, max_bytes=256_000_000):
    """Fit bounded normalized Gaussian random-utility choice by joint local ML."""
    if device != "cpu":
        _error("Multinomial probit supports explicit native CPU float64 only.", "unsupported_device")
    if weights is not None:
        _error("Weights are outside the complete choice-case covariance contract.", "unsupported_weights")
    if (vce == "cr0") != (cluster is not None):
        _error("Declare respondent cluster exactly when vce is cr0.", "invalid_option")
    columns = _columns(chosen, x, case, alternative, available, cluster)
    options = _options(vce, max_iterations, tolerance, max_work, max_bytes)
    level, _ = _confidence(level)
    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        p = _prepare(data, columns, options, catalogue=alternatives, fixed_covariance=covariance)
        theta, convergence = _fit(p)
        fit = _evaluate(theta, p) | dict(convergence=convergence, solver=SOLVER)
        state = dict(schema=SCHEMA, columns=columns, options=options, level=level, inputs=p.inputs,
                     catalogue=p.catalogue, fixed_covariance=p.fixed_covariance, fit=fit, settings=_settings(p, level))
        state["checksum"] = _checksum(state)
        return _assemble(state, p)


def _prepared_from_state(state):
    columns, inputs = state["columns"], state["inputs"]
    data = {columns[key]: inputs[key] for key in ("chosen", "case", "alternative")}
    data.update({name: [row[j] for row in inputs["x"]] for j, name in enumerate(columns["x"])})
    if columns["available"]:
        data[columns["available"]] = inputs["available"]
    if columns["cluster"]:
        data[columns["cluster"]] = inputs["cluster"]
    frame = pd.DataFrame(data, index=_index_restore(inputs["index"], len(inputs["chosen"])))
    return _prepare(frame, columns, state["options"], catalogue=state["catalogue"], fixed_covariance=state["fixed_covariance"])


def _diagnostic_matches(saved, actual):
    if not isinstance(saved, dict) or set(saved) != set(actual):
        _error("Saved physical stationarity schema is incomplete.", "invalid_result")
    for key, value in actual.items():
        if isinstance(saved[key], bool) or not isinstance(saved[key], Real) or not math.isfinite(float(saved[key])) or abs(saved[key]-value) > 2e-10*max(1.0, abs(value)):
            _error("Saved physical stationarity differs from numerical replay.", "invalid_result")


def _validate_inner(result):
    attrs = result.attrs if isinstance(result, TableSet) else result
    if not isinstance(attrs, Mapping) or set(attrs) != {"method", "contract", "settings", "mprobit_state", "notes"}:
        _error("Supply the complete mprobit fit or its portable attrs.", "invalid_result")
    state = attrs["mprobit_state"]
    fields = {"schema", "columns", "options", "level", "inputs", "catalogue", "fixed_covariance", "fit", "settings", "checksum"}
    if not isinstance(state, dict) or set(state) != fields or state["schema"] != SCHEMA:
        _error("Saved multinomial-probit schema is unsupported.", "invalid_result")
    if not isinstance(state["checksum"], str) or len(state["checksum"]) != 64 or not hmac.compare_digest(state["checksum"], _checksum(state)):
        _error("Saved multinomial-probit checksum does not match.", "invalid_result")
    if not isinstance(state["columns"], dict) or set(state["columns"]) != {"chosen", "x", "case", "alternative", "available", "cluster"}:
        _error("Saved selected-column schema is incomplete.", "invalid_result")
    columns = _columns(**state["columns"])
    if not isinstance(state["options"], dict) or set(state["options"]) != {"vce", "max_iterations", "tolerance", "max_work", "max_bytes", "quadrature_order"}:
        _error("Saved resource/solver options are incomplete.", "invalid_result")
    options = _options(**state["options"])
    if (options["vce"] == "cr0") != (columns["cluster"] is not None):
        _error("Saved covariance and cluster declarations disagree.", "invalid_result")
    inputs = state["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"chosen", "x", "case", "alternative", "available", "cluster", "index"} or not isinstance(inputs["chosen"], list):
        _error("Saved full-input schema is incomplete.", "invalid_result")
    n, q = len(inputs["chosen"]), len(columns["x"])
    if not 2 <= n <= 1536 or not _finite_matrix(inputs["x"], n, q) or any(not isinstance(inputs[key], list) or len(inputs[key]) != n for key in ("case", "alternative", "available")):
        _error("Saved inputs have incomplete bounded dimensions.", "invalid_result")
    if (columns["cluster"] is None and inputs["cluster"] is not None) or (columns["cluster"] is not None and (not isinstance(inputs["cluster"], list) or len(inputs["cluster"]) != n)):
        _error("Saved cluster input is incomplete or undeclared.", "invalid_result")
    p = _prepared_from_state(state)
    if (json.dumps(inputs, sort_keys=True, ensure_ascii=False) != json.dumps(p.inputs, sort_keys=True, ensure_ascii=False)
            or state["catalogue"] != p.catalogue or state["fixed_covariance"] != p.fixed_covariance):
        _error("Saved geometry differs from canonical retained inputs.", "invalid_result")
    fit = state["fit"]
    fit_keys = {"params", "beta", "parameter_names", "normalized_covariance", "information", "bread", "meat", "covariance", "gradient", "case_scores", "cluster_scores", "case_loglikelihood", "log_likelihood", "probability_rows", "probabilities", "log_probabilities", "scale", "stationarity", "covariance_type", "n", "n_cases", "n_clusters", "n_alternatives", "convergence", "solver"}
    k = len(p.parameters)
    if not isinstance(fit, dict) or set(fit) != fit_keys or not _finite_vector(fit["params"], k):
        _error("Saved complete physical fit schema is unsupported.", "invalid_result")
    matrices = {name: (k, k) for name in ("information", "bread", "meat", "covariance")}
    matrices |= dict(normalized_covariance=(len(p.catalogue)-1, len(p.catalogue)-1), case_scores=(len(p.case_labels), k), cluster_scores=(len(p.cluster_labels), k))
    vectors = dict(beta=q, gradient=k, case_loglikelihood=len(p.case_labels), probabilities=len(p.available_rows), log_probabilities=len(p.available_rows), scale=q)
    if any(not _finite_matrix(fit[name], *shape) for name, shape in matrices.items()) or any(not _finite_vector(fit[name], count) for name, count in vectors.items()):
        _error("Saved full matrices/vectors require finite complete dimensions.", "invalid_result")
    if not isinstance(fit["probability_rows"], list) or any(type(v) is not int for v in fit["probability_rows"]):
        _error("Saved probability row identities are not canonical positions.", "invalid_result")
    if isinstance(fit["log_likelihood"], bool) or not isinstance(fit["log_likelihood"], Real) or not math.isfinite(float(fit["log_likelihood"])):
        _error("Saved log likelihood is nonfinite.", "invalid_result")
    for name in ("information", "bread", "meat", "covariance"):
        _psd(torch.tensor(fit[name], dtype=DT, device="cpu"), name, "invalid_result")
    replay = _evaluate(torch.tensor(fit["params"], dtype=DT, device="cpu"), p)
    for name, expected in replay.items():
        if name in matrices or name in vectors or name == "log_likelihood":
            _matches(fit[name], expected, name)
        elif name == "stationarity":
            _diagnostic_matches(fit[name], expected)
        elif fit[name] != expected or type(fit[name]) is not type(expected):
            _error(f"Saved {name} differs from complete numerical replay.", "invalid_result")
    convergence = fit["convergence"]
    keys = {"starts", "selected_start", "work_used", "max_work", "globally_certified", "solver"}
    starts = STARTS if p.free_covariance else (None,)
    if not isinstance(convergence, dict) or set(convergence) != keys or convergence["globally_certified"] is not False or convergence["solver"] != SOLVER or fit["solver"] != SOLVER:
        _error("Saved optimization protocol is unsupported.", "invalid_result")
    if type(convergence["work_used"]) is not int or not 0 < convergence["work_used"] <= options["max_work"] or convergence["max_work"] != options["max_work"]:
        _error("Saved actual work accounting is invalid.", "invalid_result")
    attempts = convergence["starts"]
    if not isinstance(attempts, list) or len(attempts) != len(starts):
        _error("Saved deterministic multistart sequence is incomplete.", "invalid_result")
    accepted = []
    for i, (record, start) in enumerate(zip(attempts, starts)):
        expected_start = list(start) if start else None
        if not isinstance(record, dict) or record.get("start") != expected_start:
            _error("Saved multistart declaration is invalid.", "invalid_result")
        if record.get("status") == "rejected":
            if set(record) != {"start", "status", "code"} or record["code"] not in {"rank_deficient", "no_finite_mle", "boundary_solution", "nonconvergence", "numerical_failure", "numerical_domain"}:
                _error("Saved rejected-start diagnostics are invalid.", "invalid_result")
            continue
        if set(record) != {"start", "status", "iterations", "params", "log_likelihood", "stationarity"} or record["status"] != "stationary_interior" or type(record["iterations"]) is not int or not 1 <= record["iterations"] <= options["max_iterations"]+10 or not _finite_vector(record["params"], k) or isinstance(record["log_likelihood"], bool) or not isinstance(record["log_likelihood"], Real) or not math.isfinite(float(record["log_likelihood"])):
            _error("Saved successful multistart diagnostics are incomplete.", "invalid_result")
        point = torch.tensor(record["params"], dtype=DT, device="cpu")
        moments = _moments(point, p)
        diagnostic, _, stationary = _stationarity(point, p, moments)
        if not stationary or abs(record["log_likelihood"]-moments["log_likelihood"]) > 2e-9*max(1.0, abs(moments["log_likelihood"])):
            _error("Saved accepted candidate fails physical replay.", "invalid_result")
        _diagnostic_matches(record["stationarity"], diagnostic)
        accepted.append((record["log_likelihood"], i))
    selected = convergence["selected_start"]
    if type(selected) is not int or not accepted or selected != max(accepted, key=lambda item: item[0])[1] or attempts[selected]["params"] != fit["params"]:
        _error("Saved selected parameters are not the best declared certified candidate.", "invalid_result")
    level, _ = _confidence(state["level"])
    resources = state["settings"].get("resources") if isinstance(state["settings"], dict) else None
    if not isinstance(resources, dict) or type(resources.get("budget_bytes")) is not int:
        _error("Saved resource metadata is incomplete.", "invalid_result")
    # Preparation already honored the current ambient limit. Recreate the
    # original declared plan for canonical output without changing that limit.
    p.resources = _plan(p.n, len(p.case_labels), p.q, p.free_covariance, options,
                        for_fit=True, budget_bytes=resources["budget_bytes"])
    if state["settings"] != _settings(p, level):
        _error("Saved settings differ from the bounded complete contract.", "invalid_result")
    expected = _assemble(state, p)
    if dict(attrs) != expected.attrs or (isinstance(result, TableSet) and (set(result) != set(expected) or any(not isinstance(result[name], pd.DataFrame) or not result[name].equals(expected[name]) for name in expected))):
        _error("Saved attrs/tables disagree with complete portable state.", "invalid_result")
    return state, p


def _validated(result, *, budget_bytes=None, max_work=None):
    """Strict complete scientific replay; optimization is never called."""
    try:
        if budget_bytes is not None or max_work is not None:
            attrs = result.attrs if isinstance(result, TableSet) else result
            state = attrs["mprobit_state"]
            options = _options(**state["options"])
            for label, value in (("budget_bytes", budget_bytes), ("max_work", max_work)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 2_000_000_000):
                    _error(f"{label} requires an integer in [1,2e9].", "invalid_resource_budget")
            inputs, columns = state["inputs"], state["columns"]
            cases = len({_key(_identity(v, "saved case")) for v in inputs["case"]})
            resources = _plan(len(inputs["case"]), cases, len(columns["x"]), state["fixed_covariance"] is None,
                              options, for_fit=False, budget_bytes=budget_bytes)
            work = resources["planned_work"]*(4 if state["fixed_covariance"] is None else 2)
            if max_work is not None and work > max_work:
                _error("Complete saved-fit numerical replay exceeds the caller's work budget before tensor allocation.", "work_budget")
        with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
            return _validate_inner(result)
    except AnalysisError as error:
        if error.code in {"invalid_result", "workspace_limit", "work_budget", "invalid_resource_budget"}:
            raise
        raise AnalysisError("invalid_result", "Saved multinomial-probit result fails its complete scientific contract.") from error
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, RuntimeError) as error:
        raise AnalysisError("invalid_result", "Saved multinomial-probit result is malformed or nonfinite.") from error


@procedure
def mprobit_restore(result, *, level=None):
    """Restore complete tables after numerical replay, without optimization."""
    state, p = _validated(result)
    if level is None or level == state["level"]:
        return _assemble(state, p)
    level, _ = _confidence(level)
    changed = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    changed["level"] = level
    changed["settings"]["level"] = level
    changed["checksum"] = _checksum(changed)
    return _assemble(changed, p)
