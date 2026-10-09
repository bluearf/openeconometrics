"""Bounded resident frequency and declared group-summary discriminant fits.

Sources: Stata's ``discrim lda``/``discrim qda`` manuals (sample covariance,
Gaussian density/posteriors and fweights), and IBM's DISCRIMINANT matrix-input
reference (group means, counts and covariance information). This API declares
sample covariances directly; it does not implement vendor matrix-file formats.
Frequency LOO means deletion of ONE expanded observation with fixed priors,
validated against literal row expansion/refitting, not vendor weighted-case LOO.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.resources import plan_workspace
from . import common as c
from .discrim import (_HINT, _classification, _priors,
                      _quadratic_scores, _softmax_rows, box_m)

MAX_ROWS = 1_000_000
MAX_GROUPS = 128
MAX_VARIABLES = 256
MAX_WORK = 2_000_000_000
EXACT_COUNT = 2**53


def _geometry(p: int, groups: int = 0, rows: int = 0, *, loo: bool = False) -> dict:
    if p > MAX_VARIABLES or groups > MAX_GROUPS or rows > MAX_ROWS:
        raise AnalysisError("workspace_limit", "Discriminant options support at most "
                            f"{MAX_VARIABLES} variables, {MAX_GROUPS} groups and "
                            f"{MAX_ROWS} physical rows.")
    work = (groups + 2) * p**3 + rows * (3*p*p + groups*p*p*(4+4*int(loo)))
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Discriminant moment/classification geometry "
                            "exceeds the work budget; reduce rows, groups or variables.")
    return plan_workspace("discriminant options", {
        "selected_rows_and_moments": rows * (p + 3) * 128,
        "classification_scores": rows * groups * 64,
        "group_covariance_workspace": 16 * groups * p*p * 8,
        "pooled_matrix_workspace": 64 * p*p * 8,
    }).record()


def _resident_rows(data: Any) -> int:
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_data", "Discriminant options require resident data; "
                            "Dataset inputs are not supported.")
    if isinstance(data, pd.DataFrame) or isinstance(data, (list, tuple)):
        return len(data)
    if isinstance(data, Mapping):
        lengths = []
        for value in data.values():
            if isinstance(value, Tensor) and value.device.type != "cpu":
                raise AnalysisError("unsupported_device", "Discriminant inputs require CPU float64.")
            try:
                lengths.append(len(value))
            except TypeError:
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    raise AnalysisError("invalid_data", "Resident mapping columns must be sized; iterators are not supported.")
                continue
        return max(lengths, default=0)
    return 0                         # c.source supplies the public invalid-data error.


def _project_columns(data: Any, names: list[str]) -> Any:
    # A resident mapping/record input may contain many unused columns. Project
    # those before pandas coercion, so admission bounds actual materialization.
    if isinstance(data, Mapping):
        return {name: data[name] for name in names if name in data}
    return data


def _project_records(data: Any, names: list[str]) -> Any:
    if isinstance(data, (list, tuple)) and all(isinstance(row, Mapping) for row in data):
        return [{name: row[name] for name in names if name in row} for row in data]
    return data


def _labels(values: Any) -> list[Any]:
    result = []
    for value in values:
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        if not isinstance(value, (str, bool, int, float)) or value is None \
                or (isinstance(value, float) and not math.isfinite(value)):
            raise AnalysisError("invalid_groups", "Summary groups must be finite JSON scalar labels.")
        result.append(c.label(value))
    if len(set(map(str, result))) != len(result) or len(set(result)) != len(result):
        raise AnalysisError("invalid_groups", "Group labels must be unique, including their text.")
    return result


def _matrix(value: Any, shape: tuple[int, int], what: str) -> Tensor:
    try:
        actual = tuple(value.shape) if hasattr(value, "shape") else (len(value), len(value[0]))
    except (TypeError, IndexError) as exc:
        raise AnalysisError("invalid_spec", f"{what} must be a numeric matrix.") from exc
    if actual != shape:
        raise AnalysisError("invalid_spec", f"{what} must have shape {shape}.")
    if isinstance(value, Tensor) and value.device.type != "cpu":
        raise AnalysisError("unsupported_device", "Discriminant inputs require CPU float64.")
    if (isinstance(value, Tensor) and value.is_complex()) \
            or getattr(getattr(value, "dtype", None), "kind", None) == "c":
        raise AnalysisError("invalid_matrix", f"{what} must be real-valued.")
    try:
        out = torch.as_tensor(value, dtype=c.FLOAT).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_spec", f"{what} must be a numeric matrix.") from exc
    if not bool(torch.isfinite(out).all()):
        raise AnalysisError("non_finite_values", f"{what} must be finite.")
    return out


def _counts(value: Any, labels: list[Any], *, minimum: int = 2) -> tuple[list[int], Tensor]:
    if isinstance(value, pd.Series):
        if value.index.has_duplicates or list(value.index) != labels:
            raise AnalysisError("invalid_spec", "Count labels must match mean-index order.")
        value = value.tolist()
    elif isinstance(value, Mapping):
        if set(value) != set(labels):
            raise AnalysisError("invalid_spec", "Count labels must match the group means.")
        value = [value[label] for label in labels]
    if isinstance(value, Tensor):
        if value.device.type != "cpu" or value.is_complex():
            raise AnalysisError("invalid_spec", "Counts require a CPU integer vector.")
        value = value.tolist()
    try:
        values = list(value)
    except TypeError as exc:
        raise AnalysisError("invalid_spec", "Supply one integer count per group.") from exc
    if len(values) != len(labels):
        raise AnalysisError("invalid_spec", "Supply one integer count per group.")
    counts = [c.check_count(v, "group count", minimum=minimum, maximum=EXACT_COUNT) for v in values]
    if sum(counts) > EXACT_COUNT:
        raise AnalysisError("invalid_spec", "Total count exceeds exact float64 integer precision.")
    return counts, torch.tensor(counts, dtype=c.FLOAT)


def _covariance(value: Any, names: list[str], count: int) -> Tensor:
    p = len(names)
    if isinstance(value, pd.DataFrame):
        if value.index.has_duplicates or value.columns.has_duplicates \
                or list(value.index) != names or list(value.columns) != names:
            raise AnalysisError("invalid_spec", "Covariance labels must match variable order.")
        c.require_numeric(value, names)
        value = value.to_numpy(dtype="float64")
    covariance = _matrix(value, (p, p), "Group covariance")
    scale = max(float(covariance.abs().max()), 1e-300)
    if scale > 1e300 / (count - 1):
        raise AnalysisError("non_finite_values", "Covariance and count exceed float64 moments; rescale.")
    diagonal = covariance.diagonal()
    if bool((diagonal < 0).any()):
        raise AnalysisError("invalid_matrix", "Group covariance must be positive semidefinite.")
    positive = diagonal > 0
    # PSD and sample rank must be invariant to measurement units. An unscaled
    # tolerance can hide an impossible covariance in a small-variance variable.
    if bool((covariance[~positive] != 0).any()) or bool((covariance[:, ~positive] != 0).any()):
        raise AnalysisError("invalid_matrix", "Zero group variance requires an entirely zero covariance row and column.")
    if bool(positive.any()):
        std = diagonal[positive].sqrt()
        correlation = covariance[positive][:, positive]/torch.outer(std, std)
        if not bool(torch.isfinite(correlation).all()):
            raise AnalysisError("invalid_matrix", "Group covariance is not positive semidefinite at float64 precision.")
        if float((correlation-correlation.T).abs().max()) > 1e-12:
            raise AnalysisError("invalid_matrix", "Group covariance must be symmetric; it is not repaired.")
        eigenvalues = torch.linalg.eigvalsh((correlation+correlation.T)/2)
        if float(eigenvalues.min()) < -1e-12*p:
            raise AnalysisError("invalid_matrix", "Group covariance must be positive semidefinite.")
        rank = int((eigenvalues > 8*p*torch.finfo(c.FLOAT).eps).sum())
    else:
        rank = 0
    if rank > count - 1:
        raise AnalysisError("invalid_matrix", "Group covariance rank exceeds count - 1; "
                            "these summaries cannot be sample covariances.")
    return (covariance+covariance.T)/2


def _hash(state: dict) -> str:
    try:
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_result", "Discriminant prediction state is not finite JSON.") from exc
    return hashlib.sha256(encoded.encode()).hexdigest()


def _weighted_classification(scores: Tensor, codes: Tensor, counts: Tensor,
                             labels: list[Any], weights: Tensor) -> tuple[pd.DataFrame, float]:
    if not bool(torch.isfinite(scores).all()):
        raise AnalysisError("numerical_failure", "Classification distances overflow float64; rescale inputs.")
    groups = len(labels)
    cells = torch.zeros(groups*groups, dtype=c.FLOAT).index_add_(
        0, codes*groups + scores.argmax(1), weights).reshape(groups, groups)
    correct = cells.diagonal()
    rows = torch.cat([cells, counts[:, None], (100*correct/counts)[:, None]], dim=1)
    out = c.frame(rows, columns=[*map(str, labels), "n", "percent_correct"], index=labels)
    for name in out.columns[:-1]:
        out[name] = out[name].astype("int64")
    out.index.name = "actual"
    return out, 100*float(correct.sum())/float(counts.sum())


def _centered_linear_scores(x: Tensor, means: Tensor, factor: Tensor,
                            df: float, log_prior: Tensor) -> Tensor:
    # Log odds against a nearby winning reference avoid both shared huge
    # squared distances and shared huge linear logits. A tournament selects a
    # row-specific reference using pairwise differences before full reporting.
    def difference(g: int, reference: Tensor) -> Tensor:
        point = torch.linalg.solve_triangular(factor, (x-means[reference]).T, upper=False).T
        change = torch.linalg.solve_triangular(factor, (means[g]-means[reference]).T, upper=False).T
        return df*(change*(point-.5*change)).sum(1)+log_prior[g]-log_prior[reference]
    winner = torch.zeros(len(x), dtype=torch.int64)
    for g in range(1, len(means)):
        winner = torch.where(difference(g, winner) > 0, g, winner)
    out = torch.empty((len(x), len(means)), dtype=c.FLOAT)
    for g in range(len(means)):
        out[:, g] = difference(g, winner)
    return out


def _linear_loo_scores(x: Tensor, means: Tensor, factor: Tensor, df: float,
                       log_prior: Tensor, codes: Tensor, counts: Tensor) -> Tensor:
    # Pairwise Gaussian log odds under W_minus = W - alpha*u*u'. Form deleted
    # mean differences directly, then apply Sherman--Morrison to a bilinear
    # contrast. No pair of huge squared distances/logits is ever subtracted.
    delta_own = x-means[codes]
    own = torch.linalg.solve_triangular(factor, delta_own.T, upper=False).T
    a = own.square().sum(1)
    inverse_remaining_count = 1/(counts[codes]-1)
    ratio = counts[codes]*inverse_remaining_count
    remaining = 1-ratio*a
    if bool((remaining <= 1e-12).any()):
        raise AnalysisError("singular_matrix", "Leave-one-out classification is undefined: "
                            "removing one expanded observation makes pooled covariance singular.")
    shift = inverse_remaining_count[:, None]*delta_own
    def difference(g: int, reference: Tensor) -> Tensor:
        is_reference = codes == reference
        delta = means[g]-means[reference] \
            - ((codes == g).to(c.FLOAT)-is_reference.to(c.FLOAT))[:, None]*shift
        midpoint = x-means[reference]+is_reference[:, None]*shift-.5*delta
        zd = torch.linalg.solve_triangular(factor, delta.T, upper=False).T
        zb = torch.linalg.solve_triangular(factor, midpoint.T, upper=False).T
        product = (zd*zb).sum(1)+ratio/remaining*(zd*own).sum(1)*(zb*own).sum(1)
        return df*product+log_prior[g]-log_prior[reference]
    winner = torch.zeros(len(x), dtype=torch.int64)
    for g in range(1, len(means)):
        winner = torch.where(difference(g, winner) > 0, g, winner)
    out = torch.empty((len(x), len(means)), dtype=c.FLOAT)
    for g in range(len(means)):
        out[:, g] = difference(g, winner)
    return out


def _resolved_quadratic_scores(x: Tensor, means: Tensor, factors: list[Tensor],
                               counts: Tensor, log_prior: Tensor,
                               covariances: list[Tensor], codes: Tensor | None = None) -> Tensor:
    # Identical declared covariance matrices have exactly the LDA log odds.
    # This special case is checked before Cholesky arithmetic can perturb equality.
    if codes is None and all(torch.equal(covariances[0], value) for value in covariances[1:]):
        common = c.positive_definite(covariances[0], "common QDA covariance", _HINT)
        return _centered_linear_scores(x, means, common, 1., log_prior)
    scores = _quadratic_scores(x, means, factors, counts, log_prior, codes)
    if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
        raise AnalysisError("numerical_failure", "QDA Gaussian scores exceed float64 precision; rescale inputs.")
    top = torch.topk(scores, 2, dim=1).values
    if not bool(torch.isfinite(top[:, 0]).all()):
        raise AnalysisError("numerical_failure", "Every group density underflows at the supplied QDA rows.")
    magnitude = top.abs().max(1).values
    error_bound = 128*torch.finfo(c.FLOAT).eps*(x.shape[1]+2)*magnitude
    # Refuse when roundoff could change a material posterior, not only when it
    # could change argmax. Beyond a log-odds gap of 36 plus the estimated error,
    # the losing posterior is below roughly 1e-15 and safely saturated.
    unresolved = torch.isfinite(top[:, 1]) & (error_bound > 1e-8) \
        & ((top[:, 0]-top[:, 1]) <= 36+error_bound)
    if bool(unresolved.any()):
        raise AnalysisError("numerical_failure", "QDA competing Gaussian logits are not resolved "
                            "at float64 precision; rescale/recenter inputs or use a better-conditioned domain.")
    relative = scores-top[:, :1]
    # A negative-infinite losing score is a resolved zero posterior, not a
    # missing row. -1000 has the same zero float64 exponential after centering.
    return torch.where(torch.isneginf(relative), torch.full_like(relative, -1000), relative)


def _fit(origin: Tensor, offsets: Tensor, group_sscp: list[Tensor], counts: Tensor,
         labels: list[Any], names: list[str], group: str, method: str, priors: Any,
         attrs: dict, *, x: Tensor | None = None, codes: Tensor | None = None,
         weights: Tensor | None = None, loo: bool = False) -> TableSet:
    groups, p = len(labels), len(names)
    n = sum(int(v) for v in counts.tolist())
    df1, df2 = groups - 1, n - groups
    if df2 < 1:
        raise AnalysisError("insufficient_observations", "Discriminant analysis needs more observations than groups.")
    location = (counts[:, None]*offsets).sum(0)/n
    location += (counts[:, None]*(offsets-location)).sum(0)/n
    means = offsets-location
    grand = origin+location
    within = torch.stack(group_sscp).sum(0)
    within = (within+within.T)/2
    between = means.T @ (counts[:, None]*means)
    total = within+between
    if not all(bool(torch.isfinite(v).all()) for v in (within, between, grand, total)):
        raise AnalysisError("numerical_failure", "Discriminant moments overflow float64; rescale inputs.")
    pooled = c.positive_definite(within, "pooled within-groups covariance matrix", _HINT)
    if isinstance(priors, Tensor) and priors.device.type != "cpu":
        raise AnalysisError("unsupported_device", "Discriminant priors require CPU float64.")
    if (isinstance(priors, Tensor) and priors.is_complex()) \
            or getattr(getattr(priors, "dtype", None), "kind", None) == "c":
        raise AnalysisError("invalid_option", "Discriminant priors must be real probabilities.")
    prior = _priors(priors, counts, labels)
    log_prior = torch.log(prior)
    notes = []
    factors = []
    for sscp in group_sscp:
        try:
            factors.append(c.positive_definite(sscp, "group covariance matrix", _HINT))
        except AnalysisError:
            factors = None
            break
    if method == "qda" and (factors is None or bool((counts <= p).any())):
        raise AnalysisError("singular_matrix", "QDA requires a positive definite sample covariance "
                            "and more observations than variables in every group.")
    covariances = [sscp/(float(count)-1) if count > 1 else torch.zeros_like(sscp)
                   for sscp, count in zip(group_sscp, counts, strict=True)]
    statistics = [[label, variable, int(counts[g]), float(origin[j]+offsets[g, j]),
                   float(covariances[g][j, j].clamp_min(0).sqrt()) if counts[g] > 1 else None]
                  for g, label in enumerate(labels) for j, variable in enumerate(names)]
    statistics += [["Total", variable, n, float(grand[j]), float((total[j, j]/(n-1)).sqrt())]
                   for j, variable in enumerate(names)]
    equality = []
    for j in range(p):
        lam = float(within[j, j]/total[j, j])
        f = (1-lam)/lam*df2/df1 if lam > 0 else None
        equality.append([lam, f, df1, df2, c.f_upper(f, df1, df2)])
    group_table = c.frame(torch.stack([counts, prior], dim=1), columns=["n", "prior"], index=labels)
    group_table["n"] = group_table["n"].astype("int64")
    if codes is not None:
        group_table["n_physical"] = torch.bincount(codes, minlength=groups).numpy()
    rows = [[label, variable, *(covariances[g][j].tolist() if counts[g] > 1 else [None]*p)]
            for g, label in enumerate(labels) for j, variable in enumerate(names)]
    tables = {
        "groups": group_table,
        "group_means": c.frame(origin+offsets, columns=names, index=labels),
        "group_statistics": c.frame(statistics, columns=["group", "variable", "n", "mean", "std_dev"]),
        "tests_of_equality": c.frame(equality, columns=["wilks_lambda", "statistic", "df1", "df2", "p_value"], index=names),
        "group_covariances": c.frame(rows, columns=["group", "variable", *names]),
        "pooled_covariance": c.frame(within/df2, columns=names, index=names),
        "total_covariance": c.frame(total/(n-1), columns=names, index=names),
    }
    if factors is not None and bool((counts > p).all()):
        log_dets = [2*float(torch.log(f.diagonal()).sum())-p*math.log(float(count)-1)
                    for f, count in zip(factors, counts, strict=True)]
        log_pooled = 2*float(torch.log(pooled.diagonal()).sum())-p*math.log(df2)
        box = box_m(log_dets, [int(v) for v in counts.tolist()], log_pooled, p)
        tables["box_m"] = c.frame([list(box.values())], columns=list(box), index=["box_m"])
        tables["log_determinants"] = c.frame([[v] for v in [*log_dets, log_pooled]],
            columns=["log_determinant"], index=[*labels, "Pooled within-groups"])
    else:
        notes.append("Box's M is unavailable: an individual sample covariance is singular.")
    if method == "lda":
        sw = within/df2
        half = torch.linalg.solve_triangular(pooled, between, upper=False)
        inner = torch.linalg.solve_triangular(pooled, half.T, upper=False)
        roots, vectors = torch.linalg.eigh((inner+inner.T)/2)
        q = min(p, groups-1)
        roots = roots.flip(0)[:q].clamp_min(0)
        raw_coef = torch.linalg.solve_triangular(pooled.T, vectors.flip(1)[:, :q], upper=True)*math.sqrt(df2)
        if not bool(torch.isfinite(roots).all()) or not bool(torch.isfinite(raw_coef).all()):
            raise AnalysisError("numerical_failure", "Canonical functions overflow float64; rescale inputs.")
        scale = sw.diagonal().sqrt()
        standardized = raw_coef*scale[:, None]
        pivot = standardized.abs().argmax(0)
        signs = torch.sign(standardized[pivot, torch.arange(q)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        raw_coef, standardized = raw_coef*signs, standardized*signs
        functions = c.numbered("Function", q)
        canonical, running = [], 0.0
        total_roots = float(roots.sum())
        multiplier = n-1-(p+groups)/2
        for k in range(q):
            running += float(roots[k])
            log_lambda = -float(torch.log1p(roots[k:]).sum())
            lam = math.exp(log_lambda)
            chi2 = -multiplier*log_lambda if multiplier > 0 else None
            df = (p-k)*(groups-k-1)
            canonical.append([float(roots[k]), 100*float(roots[k])/total_roots if total_roots else None,
                100*running/total_roots if total_roots else None,
                math.sqrt(float(roots[k])/(1+float(roots[k]))), lam, chi2, df, c.chi2_upper(chi2, df)])
        coefficients = torch.linalg.solve_triangular(pooled.T,
            torch.linalg.solve_triangular(pooled, (origin+offsets).T, upper=False), upper=True)*df2
        intercept = -.5*((origin+offsets).T*coefficients).sum(0)+log_prior
        if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(intercept).all()):
            raise AnalysisError("numerical_failure", "Raw-coordinate coefficients overflow; rescale inputs.")
        tables.update({
            "canonical_functions": c.frame(canonical, columns=["eigenvalue", "percent", "cumulative", "canonical_correlation", "wilks_lambda", "chi2", "df", "p_value"], index=functions),
            "standardized_coefficients": c.frame(standardized, columns=functions, index=names),
            "unstandardized_coefficients": c.frame(torch.cat([raw_coef, -(grand@raw_coef)[None, :]]), columns=functions, index=[*names, "Intercept"]),
            "structure_matrix": c.frame((sw@raw_coef)/scale[:, None], columns=functions, index=names),
            "centroids": c.frame(means@raw_coef, columns=functions, index=labels),
            "classification_functions": c.frame(torch.cat([coefficients, intercept[None, :]]), columns=list(map(str, labels)), index=[*names, "Intercept"]),
        })
    rate = rate_loo = None
    if x is not None:
        if loo and bool((counts < (2 if method == "lda" else p+2)).any()):
            raise AnalysisError("insufficient_observations", "Every group needs at least "
                                f"{2 if method == 'lda' else p+2} expanded observations for LOO.")
        scores = (_centered_linear_scores(x, offsets, pooled, df2, log_prior) if method == "lda" else
                  _resolved_quadratic_scores(x, offsets, factors, counts, log_prior, covariances))
        tables["classification_table"], rate = _weighted_classification(scores, codes, counts, labels, weights)
        physical = torch.bincount(codes, minlength=groups).to(c.FLOAT)
        tables["classification_table_physical"], _ = _classification(scores, codes, physical, labels)
        if loo:
            if method == "lda" and df2-1 < p:
                raise AnalysisError("insufficient_observations", "LDA LOO needs n - 1 - G >= p.")
            scores = (_linear_loo_scores(x, offsets, pooled, df2-1, log_prior, codes, counts) if method == "lda" else
                      _resolved_quadratic_scores(x, offsets, factors, counts, log_prior, covariances, codes))
            tables["classification_table_loo"], rate_loo = _weighted_classification(scores, codes, counts, labels, weights)
            tables["classification_table_loo_physical"], _ = _classification(scores, codes, physical, labels)
    else:
        notes.append("Training confusion, accuracy and LOO are unavailable from group summaries.")
    state = {"version": 1, "method": method, "variables": names, "groups": labels,
             "counts": [int(v) for v in counts.tolist()], "priors": prior.tolist(),
             "origin": origin.tolist(), "mean_offsets": offsets.tolist(),
             "sample_covariances": [v.tolist() if count > 1 else None
                                    for v, count in zip(covariances, counts, strict=True)]}
    attrs.update({"discriminant_state": state, "discriminant_state_sha256": _hash(state),
        "training_classification_available": x is not None, "loo_available": x is not None,
        "covariance_divisor": "group count - 1; pooled total count - group count",
        "inferential_assumptions": "Independent multivariate-normal observations within groups; LDA additionally assumes equal group covariance. These assumptions are declared, not empirically verified.",
        "precision": "float64", "device": "cpu", "prior_policy_loo": "fixed full-fit priors",
        "quadratic_precision_policy": "identical covariance uses stable pairwise linear odds; unresolved large competing quadratic logits are refused",
        "quadratic_logit_error_budget": 1e-8, "quadratic_saturation_log_gap": 36.,
        "loo_unit": "one expanded frequency observation" if x is not None else None})
    return TableSet(tables, title=f"{'Linear' if method == 'lda' else 'Quadratic'} discriminant analysis of {group}",
        procedure="discrim", n=n, method=method, group=group, variables=names, groups=labels,
        priors=prior.tolist(), percent_correct=rate, percent_correct_loo=rate_loo, notes=notes,
        sign_convention="largest absolute standardized coefficient of each function is positive", **attrs)


def frequency_discrim(data: Any, group: str, columns: list[str], *, method: str,
                     priors: Any, loo: bool, missing: str, weights: str) -> TableSet:
    group, weights = c.check_name(group, "group"), c.check_name(weights, "weights")
    names = c.name_list(columns, "columns", minimum=1)
    c.check_choice(method, "method", ("lda", "qda"))
    c.check_flag(loo, "loo")
    if len(set([group, weights, *names])) != len(names)+2:
        raise AnalysisError("invalid_spec", "Group, weight and variable roles must be distinct.")
    if set(names) & {"group", "variable", "Intercept"}:
        raise AnalysisError("invalid_spec", "Rename variables reserved by discriminant result tables: group, variable, Intercept.")
    used = [group, weights, *names]
    data = _project_columns(data, used)
    rows = _resident_rows(data)
    _geometry(len(names), rows=rows, loo=loo)
    data = _project_records(data, used)
    sample, _, dropped = c.select(data, used, numeric=[weights, *names], missing=missing)
    # Check native integer storage BEFORE conversion can round a large count.
    if bool((sample[weights] > EXACT_COUNT).any()):
        raise AnalysisError("invalid_weights", "Frequency weights exceed exact float64 integer precision.")
    w = c.column(sample, weights)
    if bool(((w < 0) | (w != w.round())).any()):
        raise AnalysisError("invalid_weights", "Frequency weights must be nonnegative exact integers.")
    count = sum(int(v) for v in w.tolist())
    if count > EXACT_COUNT:
        raise AnalysisError("invalid_weights", "Frequency weight total exceeds exact float64 integer precision.")
    complete = sample
    positive = w > 0
    zeros = int((~positive).sum())
    sample = sample.loc[positive.numpy()].reset_index(drop=True)
    if not len(sample):
        raise AnalysisError("empty_sample", "No positive-frequency observations remain.")
    # Bound the group census before constructing codes/labels for every level.
    seen = {}
    for value in sample[group].array:
        try:
            label = c.label(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_groups", "Groups need finite scalar labels.") from exc
        if isinstance(label, float) and not math.isfinite(label):
            raise AnalysisError("invalid_groups", "Groups need finite scalar labels.")
        text = str(label)
        if text in seen and label != seen[text]:
            raise AnalysisError("invalid_groups", "Group labels are not distinct when written as text.")
        seen[text] = label
        if len(seen) > MAX_GROUPS:
            raise AnalysisError("workspace_limit", f"Discriminant options support at most {MAX_GROUPS} groups.")
    codes, labels = c.group_codes(sample[group], group)
    if len(labels) < 2:
        raise AnalysisError("invalid_groups", "Discriminant analysis needs at least two positive-frequency groups.")
    if set(map(str, labels)) & {"n", "percent_correct"}:
        raise AnalysisError("invalid_groups", "Group labels n and percent_correct conflict with classification columns.")
    plan = _geometry(len(names), len(labels), rows, loo=loo)
    # Validate all complete feature rows, including zero-weight rows, consistently
    # with the existing frequency-moment API, then exclude zero rows numerically.
    raw = c.matrix(complete, names)[positive]
    w = w[positive]
    counts = torch.zeros(len(labels), dtype=c.FLOAT).index_add_(0, codes, w)
    origin = raw[0].clone()
    x = raw-origin
    offsets = torch.zeros((len(labels), len(names)), dtype=c.FLOAT).index_add_(0, codes, w[:, None]*x)/counts[:, None]
    offsets += torch.zeros_like(offsets).index_add_(0, codes, w[:, None]*(x-offsets[codes]))/counts[:, None]
    deviation = x-offsets[codes]
    order = torch.argsort(codes, stable=True)
    dev_sorted, w_sorted = deviation[order], w[order]
    physical = torch.bincount(codes, minlength=len(labels)).tolist()
    sscp, start = [], 0
    for size in physical:
        dev, weight = dev_sorted[start:start+size], w_sorted[start:start+size]
        block = dev.T @ (weight[:, None]*dev)
        sscp.append((block+block.T)/2)
        start += size
    attrs = {"n_missing": dropped, "missing": "listwise including weight", "weights": weights,
             "weight_type": "fweight", "weight_sum": count, "physical_rows": len(raw),
             "n_physical": len(raw), "n_zero_weight": zeros, "input_kind": "resident_frequency_data",
             "resource_plan": plan}
    return _fit(origin, offsets, sscp, counts, labels, names, group, method, priors, attrs,
                x=x, codes=codes, weights=w, loo=loo)


def summary_discrim(group_means: pd.DataFrame, group_covariances: Any, counts: Any, *,
                    method: str, priors: Any, columns: list[str] | None, group: str) -> TableSet:
    group = c.check_name(group, "group")
    c.check_choice(method, "method", ("lda", "qda"))
    if not isinstance(group_means, pd.DataFrame):
        raise AnalysisError("invalid_spec", "group_means must be a labelled DataFrame.")
    if group_means.columns.has_duplicates or group_means.index.has_duplicates:
        raise AnalysisError("invalid_spec", "Group means must have unique group and variable labels.")
    names = c.name_list(list(group_means.columns), "mean variables", minimum=1)
    if columns is not None and c.name_list(columns, "columns") != names:
        raise AnalysisError("invalid_spec", "columns must match mean variable order.")
    if set(names) & {"group", "variable", "Intercept"}:
        raise AnalysisError("invalid_spec", "Rename variables reserved by discriminant result tables: group, variable, Intercept.")
    plan = _geometry(len(names), len(group_means))
    if len(group_means) < 2:
        raise AnalysisError("invalid_groups", "Discriminant summary fitting needs at least two groups.")
    labels = _labels(group_means.index)
    if not isinstance(group_covariances, Mapping) or set(group_covariances) != set(labels):
        raise AnalysisError("invalid_spec", "group_covariances must map every mean group to its sample covariance.")
    count_list, count_tensor = _counts(counts, labels)
    c.require_numeric(group_means, names)
    raw_means = _matrix(group_means.to_numpy(dtype="float64"), (len(labels), len(names)), "Group means")
    if float(raw_means.abs().max()) > c.LARGEST:
        raise AnalysisError("non_finite_values", "Group means are too large for double-precision moments.")
    covariances = [_covariance(group_covariances[label], names, count)
                   for label, count in zip(labels, count_list, strict=True)]
    origin = raw_means[0].clone()
    offsets = raw_means-origin
    sscp = [covariance*(count-1) for covariance, count in zip(covariances, count_list, strict=True)]
    attrs = {"n_missing": 0, "missing": "not applicable: declared complete group summaries",
             "input_kind": "group_summary", "input_matrix": "sample_covariance", "resource_plan": plan,
             "training_means_supplied": True, "training_counts_supplied": True,
             "physical_rows": None, "training_accuracy_unavailable_reason": "Group summaries do not identify individual observations or classification errors."}
    return _fit(origin, offsets, sscp, count_tensor, labels, names, group, method, priors, attrs)


def predict_options(result: TableSet, data: Any) -> pd.DataFrame:
    state = result.attrs.get("discriminant_state")
    expected = {"version", "method", "variables", "groups", "counts", "priors", "origin", "mean_offsets", "sample_covariances"}
    if not isinstance(state, dict) or set(state) != expected or state.get("version") != 1 \
            or _hash(state) != result.attrs.get("discriminant_state_sha256"):
        raise AnalysisError("invalid_result", "Saved discriminant prediction state is missing or changed.")
    names = c.name_list(state["variables"], "saved variables")
    labels = _labels(state["groups"])
    method = c.check_choice(state["method"], "saved method", ("lda", "qda"))
    if names != result.attrs.get("variables") or labels != result.attrs.get("groups") \
            or method != result.attrs.get("method") or len(labels) < 2:
        raise AnalysisError("invalid_result", "Saved discriminant labels or method do not agree.")
    data = _project_columns(data, names)
    rows = _resident_rows(data)
    plan = _geometry(len(names), len(labels), rows)
    _, counts = _counts(state["counts"], labels, minimum=1)
    priors = _priors(state["priors"], counts, labels)
    origin = _matrix([state["origin"]], (1, len(names)), "Saved origin")[0]
    means = _matrix(state["mean_offsets"], (len(labels), len(names)), "Saved mean offsets")
    if not isinstance(state["sample_covariances"], list) or len(state["sample_covariances"]) != len(labels):
        raise AnalysisError("invalid_result", "Saved group covariance count does not agree.")
    covariances = []
    for value, count in zip(state["sample_covariances"], counts, strict=True):
        if count == 1:
            if value is not None or method != "lda":
                raise AnalysisError("invalid_result", "A singleton group has no sample covariance.")
            covariances.append(torch.zeros((len(names), len(names)), dtype=c.FLOAT))
        else:
            covariances.append(_covariance(value, names, int(count)))
    data = _project_records(data, names)
    frame = c.source(data)
    c.require_numeric(frame, names)
    frame = frame.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    x = c.matrix(frame.loc[keep], names)-origin
    if method == "lda":
        within = torch.stack([cov*(float(count)-1) for cov, count in zip(covariances, counts, strict=True)]).sum(0)
        pooled = c.positive_definite(within, "saved pooled covariance", _HINT)
        scores = _centered_linear_scores(x, means, pooled, float(counts.sum())-len(labels), torch.log(priors))
    else:
        factors = [c.positive_definite(cov*(float(count)-1), "saved group covariance", _HINT)
                   for cov, count in zip(covariances, counts, strict=True)]
        scores = _resolved_quadratic_scores(x, means, factors, counts, torch.log(priors), covariances)
    posterior = _softmax_rows(scores)
    if not bool(torch.isfinite(posterior).all()):
        raise AnalysisError("numerical_failure", "Prediction distances overflow float64; rescale inputs.")
    columns = [f"posterior_{label}" for label in labels]
    out = table(pd.DataFrame(index=frame.index, columns=["predicted", *columns], dtype=object))
    mask = keep.to_numpy()
    if bool(keep.any()):
        out.loc[mask, "predicted"] = [labels[i] for i in scores.argmax(1).tolist()]
    for j, name in enumerate(columns):
        values = pd.Series(float("nan"), index=frame.index, dtype="float64")
        values.loc[mask] = posterior[:, j].numpy()
        out[name] = values
    out.attrs.update({"resource_plan": plan, "prediction_origin": "saved centered discriminant state"})
    return out
