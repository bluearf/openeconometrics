"""Complete categorical factorial fits from exact frequency/cell moments.

Native weighted QR fits cell means; within-cell SSCP supplies the remaining
residual geometry. No observations are synthesized. Type III uses the same
sum-coded coefficient hypotheses and classical F conventions as stats.manova.

Primary derivations:
https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_syntax08.htm
https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/multivariate_ols.html
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import itertools
import math
from numbers import Real
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.multivariate import common as mc
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.inference import critical_value
from openecon.resources import plan_workspace
from . import common as c
from . import manova_options as mo
from .anova import ANOVA_COLUMNS, anova_rows
from .glm import build_terms, effect_coding, kron
from .manova import multivariate_tests

MAX_ROWS = 100_000
MAX_CELLS = 64
MAX_OUTCOMES = 32
MAX_FACTORS = 4
MAX_TARGETS = 256
MAX_WORK = 250_000_000
MAX_LABEL_BYTES = 4096
MAX_METADATA_BYTES = 1024**2
_SCHEMA = "openecon.factorial.moments.v1"
_BASIS = "raw sum-to-zero coded design; covariates in original coordinates"


def plan(p, *, rows=0, cells=1, k=1):
    if not 1 <= p <= MAX_OUTCOMES or not 1 <= cells <= MAX_CELLS \
            or not 1 <= k <= MAX_CELLS or rows > MAX_ROWS or k*p > MAX_TARGETS:
        raise AnalysisError("workspace_limit", "Factorial moment fits admit 100000 resident rows, "
                            "64 complete cells, 32 outcomes and 256 coefficient targets.")
    if rows*(p*p+8*p)+(cells+4)*p**3+8*cells*k*k+4*k**3 > MAX_WORK:
        raise AnalysisError("work_limit", "Factorial moment work exceeds 250 million operations.")
    return plan_workspace("complete factorial moment fit", {
        "resident_projection_and_moment_copies": rows*(p+MAX_FACTORS+4)*96,
        "cell_moments_and_portable_state": cells*(p*p+p+1)*96,
        "weighted_QR_and_coefficient_error_geometry": (cells*k+k*k+k*p+p*p)*128,
        "complete_coefficient_covariance_and_tables": (k*p)**2*96,
    }).record()


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (str, bool, int, float)) or value is None \
            or isinstance(value, float) and not math.isfinite(value):
        raise AnalysisError("invalid_groups", "Factor/subject labels must be finite JSON scalar identities.")
    if isinstance(value, str) and len(value) > MAX_LABEL_BYTES or len(encode(value).encode("utf-8")) > MAX_LABEL_BYTES:
        raise AnalysisError("workspace_limit", "One typed factor/subject label exceeds 4096 bytes.")
    return value


def metadata(names, levels=None, cells=None):
    """Admit labels before repeated table/state/digest materialization."""
    if any(len(name) > MAX_LABEL_BYTES or len(encode(name).encode("utf-8")) > MAX_LABEL_BYTES for name in names):
        raise AnalysisError("workspace_limit", "One declared role/outcome name exceeds 4096 bytes.")
    size = sum(len(encode(name).encode("utf-8")) for name in names)
    if levels is not None:
        size += sum(len(encode(_scalar(value)).encode("utf-8")) for values in levels.values() for value in values)
    if cells is not None:
        size += sum(len(encode(tuple(cell)).encode("utf-8")) for cell in cells)
    if size > MAX_METADATA_BYTES:
        raise AnalysisError("workspace_limit", "Declared typed model metadata exceeds one MiB.")


def factor_names(names):
    if any(name == "Intercept" or any(mark in name for mark in ("#", "[", "]")) for name in names):
        raise AnalysisError("invalid_spec", "Categorical factor names must exclude reserved coefficient/effect notation (Intercept, #, brackets).")


def portable_bound(moments, model):
    p, k, g = model.m, model.k, len(moments.cells)
    t = len(model.terms)
    outcome_bytes = sum(len(encode(name).encode("utf-8")) for name in moments.names)
    design_bytes = sum(len(encode(name).encode("utf-8")) for name in model.column_names)
    term_bytes = sum(len(encode(term.name).encode("utf-8")) for term in model.terms)
    cell_bytes = sum(len(encode(tuple(cell)).encode("utf-8")) for cell in moments.cells)
    rows = moments.provenance.get("n_input_rows", 0)
    subjects = sum(len(value.encode("utf-8")) for value in moments.provenance.get("subject_order", []))
    # Conservative JSON/table/state estimate, including repeated row labels and
    # preview/state copies. It bounds output bytes, separately from tensor/RSS.
    estimate = 256*((k*p)**2+g*p*p+t*p*p+k*k+4*k*p+4*p*p)
    estimate += 8*(2*p*(g+t)*outcome_bytes+p*p*term_bytes+3*p*design_bytes+4*cell_bytes)
    estimate += 256*rows+8*subjects
    if estimate > 32*1024**2:
        raise AnalysisError("workspace_limit", "Full moment tables and portable state exceed the conservative 32 MiB export domain.")
    return estimate


def resolved_variances(matrix, name, *, divisor=1):
    variance = matrix.diagonal()/divisor
    if bool(((matrix.diagonal() > 0) & (variance < torch.finfo(torch.float64).tiny)).any()):
        raise AnalysisError("unresolved_precision", f"{name} contains subnormal/underflowed variance; rescale the outcome domain.")


def _levels(values, *, minimum=2):
    labels, keys = [], set()
    for value in values:
        value = _scalar(value)
        key = encode(value)
        if key not in keys:
            keys.add(key)
            labels.append(value)
            if len(labels) > MAX_CELLS:
                raise AnalysisError("workspace_limit", "Factor metadata exceeds 64 levels/cells.")
    if len(labels) < minimum or len(set(labels)) != len(labels) \
            or len(set(map(str, labels))) != len(labels):
        raise AnalysisError("invalid_groups", "Factors need distinct observed levels without numeric/text aliases.")
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=lambda value: (type(value).__name__, str(value)))


def cells_from_frame(frame, factors):
    levels = {name: _levels(frame[name]) for name in factors}
    number = math.prod(len(value) for value in levels.values())
    if number > MAX_CELLS:
        raise AnalysisError("workspace_limit", "The complete crossed categorical design exceeds 64 cells.")
    cells = list(itertools.product(*(levels[name] for name in factors))) if factors else [()]
    metadata(factors, levels, cells)
    lookup = {encode(tuple(cell)): i for i, cell in enumerate(cells)}
    codes = [lookup[encode(tuple(_scalar(value) for value in row))]
             for row in frame[factors].itertuples(index=False, name=None)] if factors else [0]*len(frame)
    if set(codes) != set(range(len(cells))):
        raise AnalysisError("empty_cells", "Every declared crossed categorical cell must have positive-frequency observations.")
    return levels, cells, codes


def _resident(data, names, numeric, missing, *, p, complete=True):
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_data", "Moment frequency routes require bounded resident data.")
    if isinstance(data, Mapping):
        data = {name: data[name] for name in names if name in data}
        sizes = []
        for value in data.values():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise AnalysisError("unsupported_device", "Moment inputs require resident CPU values.")
            try:
                sizes.append(len(value))
            except TypeError:
                raise AnalysisError("invalid_data", "Resident columns must be sized; iterators are not admitted.") from None
        rows = max(sizes, default=0)
    elif isinstance(data, (pd.DataFrame, list, tuple)):
        rows = len(data)
    else:
        raise AnalysisError("invalid_data", "Supply resident rows or sized columns.")
    plan(p, rows=rows)
    if isinstance(data, (list, tuple)) and all(isinstance(row, Mapping) for row in data):
        data = [{name: row[name] for name in names if name in row} for row in data]
    if complete:
        frame, keep, dropped = mc.select(data, names, numeric=numeric, missing=missing)
    else:
        frame = mc.source(data)
        mc.require_numeric(frame, numeric)
        absent = [name for name in names if name not in frame.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        frame = frame.loc[:, names]
        keep = ~frame.isna().any(axis=1)
        dropped = int((~keep).sum())
        if dropped and missing == "raise":
            raise AnalysisError("missing_values", "Incomplete RM rows require explicit missing='drop_subject'.")
    for name in numeric:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            raise AnalysisError("invalid_matrix", "Boolean outcomes/frequencies are not admitted.")
    return frame, keep, dropped, rows


def frequencies(values, *, check_total=True):
    result = []
    for value in values:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) \
                or value < 0 or value > mo.MAX_COUNT or value != int(value):
            raise AnalysisError("invalid_weights", "Frequency counts must be nonnegative exact integers <=2**53.")
        result.append(int(value))
    if check_total and sum(result) > mo.MAX_COUNT:
        raise AnalysisError("invalid_weights", "Frequency total exceeds exact float64 integer precision.")
    return result


def reduce_rows(values, frequency, codes, cells):
    counts = [sum(w for w, code in zip(frequency, codes, strict=True) if code == g)
              for g in range(len(cells))]
    if any(count <= 0 for count in counts):
        raise AnalysisError("empty_cells", "All complete crossed cells require positive frequencies.")
    origin = values[0].clone()
    centred = values-origin
    offsets, scatters = [], []
    for g, count in enumerate(counts):
        indices = [i for i, code in enumerate(codes) if code == g]
        block = centred[indices]
        weights = torch.tensor([frequency[i] for i in indices], dtype=torch.float64)
        offset = (weights[:, None]*block).sum(0)/count
        offset += (weights[:, None]*(block-offset)).sum(0)/count
        deviation = block-offset
        scatter = deviation.T@(weights[:, None]*deviation)
        scatter = (scatter+scatter.T)/2
        mo._finite(offset, scatter)
        mo._covariance(scatter, "frequency cell SSCP", rank_limit=len(indices)-1)
        resolved_variances(scatter, "frequency sample covariance", divisor=max(1, count-1))
        offsets.append(offset)
        scatters.append(scatter)
    return origin, torch.stack(offsets), scatters, counts


def _ordered_index(index, factors, *, minimum_factors=0):
    if len(index) > MAX_CELLS:
        raise AnalysisError("workspace_limit", "Summary crossed cells exceed 64.")
    if index.has_duplicates:
        raise AnalysisError("invalid_spec", "Summary cell labels must be unique.")
    if factors:
        if not isinstance(index, pd.MultiIndex) or list(index.names) != factors:
            raise AnalysisError("invalid_spec", "Summary cells require a MultiIndex named in declared factor order.")
        if not minimum_factors <= len(factors) <= MAX_FACTORS:
            raise AnalysisError("invalid_spec", "The declared factor count exceeds its bounded scope.")
        cells = [tuple(_scalar(value) for value in cell) for cell in index.tolist()]
        levels = {name: _levels(cell[j] for cell in cells) for j, name in enumerate(factors)}
        expected = list(itertools.product(*(levels[name] for name in factors)))
        if cells != expected:
            raise AnalysisError("invalid_spec", "Summary cells must be the complete ordered Cartesian product of observed factor levels.")
    else:
        if len(index) != 1 or isinstance(index, pd.MultiIndex):
            raise AnalysisError("invalid_spec", "A no-between summary needs one explicitly labelled group row.")
        _scalar(index[0])
        levels, cells = {}, [()]
    if len(cells) > MAX_CELLS:
        raise AnalysisError("workspace_limit", "Summary crossed cells exceed 64.")
    metadata(factors, levels, cells)
    return levels, cells


def summary_moments(means, covariances, counts, cells, outcome_order):
    labels = list(means.index)
    if isinstance(counts, pd.Series):
        if counts.index.has_duplicates or list(counts.index) != labels:
            raise AnalysisError("invalid_spec", "Summary counts must follow cell mean order.")
        counts = counts.tolist()
    elif isinstance(counts, Mapping):
        if set(counts) != set(labels):
            raise AnalysisError("invalid_spec", "Counts must label every declared cell exactly once.")
        counts = [counts[label] for label in labels]
    if isinstance(counts, torch.Tensor):
        if counts.device.type != "cpu":
            raise AnalysisError("unsupported_device", "Counts require CPU values.")
        counts = counts.tolist()
    try:
        if len(counts) != len(cells):
            raise ValueError("count dimensions")
        counts = [mo._count(value, "declared cell count", minimum=2) for value in counts]
    except (TypeError, ValueError):
        raise AnalysisError("invalid_spec", "Supply one exact integer sample count >=2 per cell.") from None
    if sum(counts) > mo.MAX_COUNT:
        raise AnalysisError("invalid_spec", "Declared sample total exceeds 2**53.")
    if not isinstance(covariances, Mapping) or set(covariances) != set(labels):
        raise AnalysisError("invalid_spec", "Covariances must label every declared cell exactly once.")
    p = len(outcome_order)
    values = mo._matrix(means, (len(cells), p), "cell means")
    scatters = []
    for label, count in zip(labels, counts, strict=True):
        covariance = covariances[label]
        if isinstance(covariance, pd.DataFrame):
            if covariance.index.has_duplicates or covariance.columns.has_duplicates \
                    or list(covariance.index) != outcome_order or list(covariance.columns) != outcome_order:
                raise AnalysisError("invalid_spec", "Covariance rows/columns must match declared outcome/cell order.")
        value = mo._matrix(covariance, (p, p), "declared unbiased sample covariance")
        mo._covariance(value, "declared cell covariance", rank_limit=count-1)
        resolved_variances(value, "declared sample covariance")
        scatter = value*(count-1)
        mo._finite(scatter)
        scatters.append(scatter)
    origin = values[0].clone()
    return origin, values-origin, scatters, counts


@dataclass
class Moments:
    factors: list
    levels: dict
    cells: list
    names: list
    origin: torch.Tensor
    offsets: torch.Tensor
    scatters: list
    counts: list
    kind: str
    provenance: dict


class MomentModel:
    """Only the existing Type III/finish interface, fitted by bounded weighted QR."""

    def __init__(self, moments, interactions="full"):
        self.n, self.m = sum(moments.counts), len(moments.names)
        self.levels = moments.levels
        self.codings = {name: effect_coding(len(levels)) for name, levels in self.levels.items()}
        self.centres, self.covariates = {}, {}
        self.terms = build_terms(moments.factors, [], interactions)
        factor_names(moments.factors)
        self.columns, self.column_names, position = {}, [], 0
        for term in self.terms:
            width = math.prod(len(self.levels[name])-1 for name in term.factors)
            self.columns[term.name] = list(range(position, position+width))
            self.column_names.extend([term.name] if width == 1 else [f"{term.name}[{j+1}]" for j in range(width)])
            position += width
        self.k = position
        self.portable_bytes = portable_bound(moments, self)
        self.resource_plan = plan(self.m, cells=len(moments.cells), k=self.k)
        if self.n-self.k <= 0:
            raise AnalysisError("no_residual_df", "The frequency/cell design leaves no residual degrees of freedom.")
        level_codes = {name: {encode(level): i for i, level in enumerate(values)} for name, values in self.levels.items()}
        design = []
        for cell in moments.cells:
            cell_codes = {name: level_codes[name][encode(level)] for name, level in zip(moments.factors, cell, strict=True)}
            row = []
            for term in self.terms:
                part = torch.ones(1, dtype=torch.float64)
                for name in term.factors:
                    part = kron(part, self.codings[name][cell_codes[name]])
                row.extend(part.tolist())
            design.append(row)
        self.design = torch.tensor(design, dtype=torch.float64)
        self.shift = moments.origin.clone()
        count = torch.tensor(moments.counts, dtype=torch.float64)
        weighted = count.sqrt()[:, None]*self.design
        norms = torch.linalg.vector_norm(weighted, dim=0)
        spectrum = torch.linalg.svdvals(weighted/norms)
        if len(spectrum) != self.k or float(spectrum.min()) <= 1e-11*float(spectrum.max()):
            raise AnalysisError("collinear_design", "The declared weighted factorial design is rank deficient or unresolved.")
        q, r = torch.linalg.qr(weighted, mode="reduced")
        inverse = torch.linalg.solve_triangular(r, torch.eye(self.k, dtype=torch.float64), upper=True)
        bread = inverse@inverse.T
        self.beta = torch.linalg.solve_triangular(r, q.T@(count.sqrt()[:, None]*moments.offsets), upper=True)
        residual = moments.offsets-self.design@self.beta
        if self.k == len(moments.cells):
            # The saturated complete cell design interpolates the cell means
            # exactly; QR roundoff is not extra within-cell sampling variation.
            residual = torch.zeros_like(residual)
        self.error = torch.stack(moments.scatters).sum(0)+residual.T@(count[:, None]*residual)
        self.error = (self.error+self.error.T)/2
        self.full = SimpleNamespace(xtx_inv=(bread+bread.T)/2, beta=self.beta)
        self.uncentre = torch.eye(self.k, dtype=torch.float64)
        self.df_error, self.rank = self.n-self.k, self.k
        mo._finite(self.beta, self.error, bread)
        resolved_variances(self.error, "pooled residual covariance", divisor=self.df_error)
        mo._covariance(bread, "factorial design bread", positive=True)
        location = (count[:, None]*moments.offsets).sum(0)/self.n
        centred = moments.offsets-location
        corrected = torch.stack(moments.scatters).sum(0)+centred.T@(count[:, None]*centred)
        self.corrected_totals = corrected.diagonal()
        self.raw_totals = self.corrected_totals+self.n*(self.shift+location).square()
        mo._finite(self.corrected_totals, self.raw_totals)

    def hypothesis(self, term, ss_type=3):
        if ss_type != 3:
            raise AnalysisError("invalid_option", "Moment factorial fits support Type III hypotheses only.")
        columns = self.columns[term.name]
        beta = self.beta[columns].clone()
        if term.name == "Intercept":
            beta += self.shift
        bread = self.full.xtx_inv[columns][:, columns]
        result = beta.T@torch.linalg.solve(bread, beta)
        mo._finite(result)
        return (result+result.T)/2

    def df(self, term):
        return len(self.columns[term.name])


def moment_state(moments, model, *, extra=None):
    state = {"schema": _SCHEMA, "kind": moments.kind, "factors": moments.factors,
             "levels": moments.levels, "cells": [list(cell) for cell in moments.cells],
             "typed_cells": [encode(tuple(cell)) for cell in moments.cells],
             "outcomes": moments.names, "typed_outcomes": mo._typed_names(moments.names),
             "counts": moments.counts, "origin": moments.origin.tolist(),
             "mean_offsets": moments.offsets.tolist(), "cell_sscp": [value.tolist() for value in moments.scatters],
             "design": model.design.tolist(), "design_columns": model.column_names,
             "terms": [{"name": term.name, "factors": list(term.factors), "columns": model.columns[term.name]}
                       for term in model.terms], "provenance": moments.provenance, "extra": extra}
    state["sha256"] = mo._digest(state)
    return state


def _classical_tests(h, e, q, nu):
    rows = multivariate_tests(h, e, q, nu)
    if min(q, e.shape[0]) == 1:
        # All four rank-one F laws coincide exactly. Reuse the root-based
        # Hotelling law, avoiding V/(1-V) cancellation as Pillai approaches 1.
        reference = rows[2]
        for row in rows:
            row[2:6] = reference[2:6]
    return rows


def stabilize_saved_query(output):
    """Keep new moment followups in the same resolved/exact domain as fits."""
    df = output.attrs["df_resid"]
    error = torch.tensor(output["error_sscp"].to_numpy(), dtype=torch.float64)
    covariance = torch.tensor(output["target_covariance"].to_numpy(), dtype=torch.float64)
    resolved_variances(error, "projected residual covariance", divisor=df)
    if bool((covariance.diagonal() < torch.finfo(torch.float64).tiny).any()):
        raise AnalysisError("unresolved_precision", "Saved target covariance has subnormal/underflowed variance; rescale the hypothesis.")
    if min(output["L"].shape[0], output["M"].shape[1]) == 1:
        rows = output["multivariate"]
        fields = ["statistic", "df1", "df2", "p_value"]
        reference = rows.loc[rows.test == "hotelling", fields].iloc[0]
        rows.loc[:, fields] = [reference.to_list()]*len(rows)
        output.attrs["rank_one_f_convention"] = "all four exact rank-one F laws evaluated from the non-cancelling Hotelling root"


def _tables(moments, model, alpha):
    p, k, df = model.m, model.k, model.df_error
    mo._covariance(model.error, "pooled factorial residual SSCP", positive=True, rank_limit=df)
    if df < p or any(min(model.df(term), p) > 1 and df <= p for term in model.terms):
        raise AnalysisError("insufficient_observations", "Every factorial multivariate F law requires adequate residual df (nu>=p; nu>p for higher-rank effects).")
    multivariate, univariate, hypotheses, roots = [], [], [], []
    for term in model.terms:
        h = model.hypothesis(term)
        multivariate.extend([[term.name, *row] for row in _classical_tests(h, model.error, model.df(term), df)])
        hypotheses.extend([[term.name, moments.names[i], moments.names[j], float(h[i, j])]
                           for i in range(p) for j in range(p)])
        chol = torch.linalg.cholesky(model.error)
        half = torch.linalg.solve_triangular(chol, h, upper=False)
        half = torch.linalg.solve_triangular(chol, half.T, upper=False)
        values = torch.linalg.eigvalsh((half+half.T)/2).clamp_min(0).flip(0)
        roots.extend([[term.name, i+1, float(value)] for i, value in enumerate(values)])
    if bool((model.error.diagonal() <= 1e-24*model.corrected_totals).any()):
        raise AnalysisError("unresolved_precision", "Within-cell residual variation is unresolved against total outcome scale; rescale the model/domain.")
    for j, name in enumerate(moments.names):
        (rows, labels), _ = anova_rows(model, j, 3, None, totals=(float(model.raw_totals[j]), float(model.corrected_totals[j])))
        univariate.extend([[name, label, *row] for label, row in zip(labels, rows, strict=True)])
    beta = model.beta.clone()
    beta[0] += model.shift
    sigma = model.error/df
    covariance = torch.kron(model.full.xtx_inv.contiguous(), sigma.contiguous())
    mo._finite(covariance, beta)
    if bool((covariance.diagonal() < torch.finfo(torch.float64).tiny).any()):
        raise AnalysisError("unresolved_precision", "Coefficient covariance contains subnormal/underflowed variance; rescale the outcome domain.")
    se = covariance.diagonal().sqrt().reshape(k, p)
    critical = critical_value(alpha, df)
    targets = [f"target[{i+1}]" for i in range(k*p)]
    order = [[model.column_names[i], moments.names[j]] for i in range(k) for j in range(p)]
    statistic = beta/se
    mo._finite(statistic)
    estimates = [[model.column_names[i], moments.names[j], float(beta[i, j]), float(se[i, j]),
                  float(statistic[i, j]), df, c.t_two_sided(float(statistic[i, j]), df),
                  float(beta[i, j]-critical*se[i, j]), float(beta[i, j]+critical*se[i, j])]
                 for i in range(k) for j in range(p)]
    return {
        "multivariate": table(multivariate, columns=["effect", *mo._TEST_COLUMNS]),
        "univariate": table(univariate, columns=["outcome", "source", *ANOVA_COLUMNS]),
        "hypothesis_sscp": table(hypotheses, columns=["effect", "outcome", "other_outcome", "sscp"]),
        "error_sscp": table(model.error.tolist(), index=moments.names, columns=moments.names),
        "roots": table(roots, columns=["effect", "root", "eigenvalue"]),
        "coefficients": table(beta.tolist(), index=model.column_names, columns=moments.names),
        "bread": table(model.full.xtx_inv.tolist(), index=model.column_names, columns=model.column_names),
        "residual_covariance": table(sigma.tolist(), index=moments.names, columns=moments.names),
        "coefficient_covariance": table(covariance.tolist(), index=targets, columns=targets),
        "coefficient_estimates": table(estimates, columns=["design_column", "outcome", "estimate", "std_error", "statistic", "df", "p_value", "ci_low", "ci_high"]),
        "target_order": table([[target, *pair] for target, pair in zip(targets, order, strict=True)], columns=["target", "design_column", "outcome"]),
        "design": table(model.design.tolist(), index=[f"cell[{i+1}]" for i in range(len(moments.cells))], columns=model.column_names),
        "cell_means": table((moments.offsets+moments.origin).tolist(), index=[f"cell[{i+1}]" for i in range(len(moments.cells))], columns=moments.names),
        "cell_counts": table([[i+1, count] for i, count in enumerate(moments.counts)], columns=["cell", "n"]),
        "cell_labels": table([[i+1, name, value, encode(value)] for i, cell in enumerate(moments.cells)
                              for name, value in zip(moments.factors, cell, strict=True)], columns=["cell", "factor", "level", "typed_level"]),
        "cell_sscp": table([[g+1, moments.names[i], moments.names[j], float(block[i, j])]
                            for g, block in enumerate(moments.scatters) for i in range(p) for j in range(p)], columns=["cell", "outcome", "other_outcome", "sscp"]),
    }


def _finish(moments, interactions, alpha):
    model = MomentModel(moments, interactions)
    tables = _tables(moments, model, alpha)
    state = moment_state(moments, model)
    fit_state = mo.save_model_geometry(model, moments.names, dropped=moments.provenance["n_missing"],
                                      input_kind="resident_factorial_frequency" if moments.kind == "frequency" else "factorial_cell_summary")
    output = TableSet(tables, title="Complete categorical factorial MANOVA", procedure="manova_factorial" if moments.kind == "frequency" else "manova_factorial_summary",
                      n=model.n, n_missing=moments.provenance["n_missing"], df_resid=model.df_error,
                      outcomes=moments.names, factors=moments.factors, terms=[term.name for term in model.terms], ss_type=3,
                      alpha=alpha, precision="float64", device="cpu", covariance_order="row-major: design column then outcome",
                      pointwise_intervals=True, familywise_intervals=False, resource_plan=moments.provenance["resource_plan"],
                      factorial_moment_state=state, manova_contrast_state=fit_state, manova_contrast_available=True,
                      inference="fixed complete crossed categorical design; independent Gaussian observations with one common unrestricted outcome covariance; integer frequencies represent independent original observations",
                      input_kind=fit_state["input_kind"], sample=moments.provenance,
                      coding="sum-to-zero (effect) coding", hotelling_f_convention="existing native/R higher-rank approximation",
                      rank_one_f_convention="all four exact rank-one F laws evaluated from the non-cancelling Hotelling root",
                      coefficient_inference="marginal two-sided zero-null Student t tests; pointwise intervals",
                      estimated_portable_bytes=model.portable_bytes, portable_export_limit_bytes=32*1024**2)
    return saved_summary(output)


@resident_cpu
def manova_factorial(data: Any, y, factors, *, weights, weight_type="fweight", interactions="full",
                     missing="drop", alpha=0.05):
    """Complete categorical Type III MANOVA with exact integer observation counts.

    No continuous covariates, incomplete crossed cells, analytic/survey weights
    or selected-model uncertainty are admitted. Zero counts are validated then
    excluded; missingness is listwise across every used role. Saved full geometry
    supports manova_contrast and portable summary restoration without raw rows.
    """
    names, factors = c.name_list(y, "y"), c.name_list(factors, "factors", minimum=2)
    weights, alpha = c.check_name(weights, "weights"), c.check_alpha(alpha)
    c.check_choice(weight_type, "weight_type", ("fweight",))
    c.check_choice(missing, "missing", ("drop", "raise"))
    if len(factors) > MAX_FACTORS or len(set([*names, *factors, weights])) != len(names)+len(factors)+1:
        raise AnalysisError("invalid_spec", "Use 2..4 distinct categorical factor roles, outcomes and frequency column.")
    metadata([*names, *factors, weights])
    factor_names(factors)
    frame, keep, dropped, rows = _resident(data, [*names, *factors, weights], [*names, weights], missing, p=len(names))
    frequency = frequencies(frame[weights].array)
    raw = mc.matrix(frame, names)
    positive = [i for i, value in enumerate(frequency) if value > 0]
    if not positive:
        raise AnalysisError("empty_sample", "No positive-frequency observations remain.")
    selected = frame.iloc[positive]
    levels, cells, codes = cells_from_frame(selected, factors)
    resource_plan = plan(len(names), rows=rows, cells=len(cells), k=len(cells))
    origin, offsets, scatters, counts = reduce_rows(raw[positive], [frequency[i] for i in positive], codes, cells)
    provenance = {"n_missing": dropped, "n_input_rows": rows, "physical_rows": len(positive),
                  "n_zero_weight": len(frame)-len(positive), "weights": weights, "weight_type": "fweight",
                  "source_positions": [i for i, value in enumerate(keep.tolist()) if value], "frequencies": frequency,
                  "positive_complete_positions": positive, "cell_codes": codes, "sampling_unit": "independent original observation",
                  "resource_plan": resource_plan}
    return _finish(Moments(factors, levels, cells, names, origin, offsets, scatters, counts, "frequency", provenance), interactions, alpha)


@resident_cpu
def manova_factorial_summary(cell_means: pd.DataFrame, cell_covariances, counts, *, factors=None,
                             interactions="full", alpha=0.05):
    """Type III factorial MANOVA from complete ordered crossed cell moments.

    Means require named factor MultiIndex rows and outcome columns. Counts are
    integers >=2; covariances are unbiased n_cell-1 sample covariances, PSD with
    compatible sample rank. No synthetic observations/missing pattern are made.
    """
    alpha = c.check_alpha(alpha)
    if not isinstance(cell_means, pd.DataFrame) or cell_means.columns.has_duplicates:
        raise AnalysisError("invalid_spec", "Supply labelled complete-cell mean DataFrame.")
    factors = list(cell_means.index.names) if factors is None else factors
    factors = c.name_list(factors, "factors", minimum=2)
    names = c.name_list(list(cell_means.columns), "outcomes")
    if set(names)&set(factors):
        raise AnalysisError("invalid_spec", "Outcome and categorical factor names must be distinct.")
    metadata([*names, *factors])
    factor_names(factors)
    levels, cells = _ordered_index(cell_means.index, factors, minimum_factors=2)
    resource_plan = plan(len(names), cells=len(cells), k=len(cells))
    origin, offsets, scatters, counts = summary_moments(cell_means, cell_covariances, counts, cells, names)
    provenance = {"n_missing": 0, "physical_rows": None, "sampling_unit": "declared independent original observation",
                  "covariance_divisor": "cell count-1", "missing": "declared complete summaries", "resource_plan": resource_plan}
    return _finish(Moments(factors, levels, cells, names, origin, offsets, scatters, counts, "summary", provenance), interactions, alpha)


def validate_moments(state, *, rm=False):
    """Validate portable sufficient state; no numerical fit/table is replaced."""
    try:
        fields = {"schema", "kind", "factors", "levels", "cells", "typed_cells", "outcomes", "typed_outcomes",
                  "counts", "origin", "mean_offsets", "cell_sscp", "design", "design_columns", "terms",
                  "provenance", "extra", "sha256"}
        if not isinstance(state, dict) or set(state) != fields or state["schema"] != (
            "openecon.rm.moments.v1" if rm else _SCHEMA
        ) or state["kind"] not in {"frequency", "summary"}:
            raise ValueError("schema")
        factors = c.name_list(state["factors"], "saved factors", minimum=0 if rm else 2)
        names = c.name_list(state["outcomes"], "saved outcomes")
        metadata([*factors, *names])
        factor_names(factors)
        cells, levels = state["cells"], state["levels"]
        if len(factors) > MAX_FACTORS or not isinstance(levels, dict) or set(levels) != set(factors):
            raise ValueError("levels")
        for name in factors:
            if levels[name] != _levels(levels[name]):
                raise ValueError("level identity/order")
        expected_cells = [list(cell) for cell in itertools.product(*(levels[name] for name in factors))] if factors else [[]]
        if cells != expected_cells or state["typed_cells"] != [encode(tuple(cell)) for cell in cells] \
                or state["typed_outcomes"] != mo._typed_names(names):
            raise ValueError("typed complete cells/order")
        metadata(factors, levels, cells)
        if not rm and state["extra"] is not None:
            raise ValueError("unexpected extra fields")
        g, p, k = len(cells), len(names), len(state["design_columns"])
        plan(p, cells=g, k=k)
        provenance = state["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("provenance")
        summary_keys = {"n_missing", "sampling_unit", "covariance_divisor", "missing", "resource_plan"}
        summary_keys |= {"physical_subjects", "expanded_subjects"} if rm else {"physical_rows"}
        frequency_keys = {"n_missing", "n_input_rows", "weights", "weight_type", "frequencies", "positive_complete_positions", "sampling_unit", "resource_plan"}
        frequency_keys |= ({"physical_subjects", "positive_subjects", "n_dropped_subjects", "n_zero_weight_subjects", "expanded_subjects", "subject", "outcome", "subject_order", "dropped_subjects", "complete_subject_order", "profile_positions", "group_codes", "missing"} if rm else {"physical_rows", "n_zero_weight", "source_positions", "cell_codes"})
        if set(provenance) != (summary_keys if state["kind"] == "summary" else frequency_keys):
            raise ValueError("provenance fields")
        resource = provenance["resource_plan"]
        if not isinstance(resource, dict) or set(resource) != {"operation", "estimated_workspace_bytes", "budget_bytes", "buffers", "scope"} \
                or not isinstance(resource["buffers"], dict) or len(resource["buffers"]) > 8 \
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in resource["buffers"].values()):
            raise ValueError("resource receipt")
        if state["kind"] == "frequency":
            rows = mo._count(provenance["n_input_rows"], "saved physical input rows")
            plan(p, rows=rows, cells=g, k=k)
            maximum = 10000 if rm else rows
            arrays = ["frequencies", "positive_complete_positions", "group_codes" if rm else "cell_codes"]
            arrays += ["subject_order", "dropped_subjects", "complete_subject_order", "profile_positions"] if rm else ["source_positions"]
            if any(not isinstance(provenance[name], list) or len(provenance[name]) > maximum for name in arrays):
                raise ValueError("sample array bounds")
            if rm and any(not isinstance(value, str) or len(value) > MAX_LABEL_BYTES for name in ("subject_order", "dropped_subjects", "complete_subject_order") for value in provenance[name]):
                raise ValueError("subject metadata bounds")
            metadata([c.check_name(provenance[name], name) for name in (["weights", "subject", "outcome"] if rm else ["weights"])])
        if rm:
            extra = state["extra"]
            if not isinstance(extra, dict) or set(extra) != {"within", "within_levels", "within_cells", "typed_within_cells"}:
                raise ValueError("RM extra")
            within = c.name_list(extra["within"], "saved within")
            if not 1 <= len(within) <= MAX_FACTORS or not isinstance(extra["within_levels"], dict) or set(extra["within_levels"]) != set(within) \
                    or not isinstance(extra["within_cells"], list) or len(extra["within_cells"]) != p \
                    or not isinstance(extra["typed_within_cells"], list) or len(extra["typed_within_cells"]) != p:
                raise ValueError("RM dimensions")
            factor_names(within)
            for name in within:
                if not isinstance(extra["within_levels"][name], list) or len(extra["within_levels"][name]) > 32:
                    raise ValueError("RM level bounds")
            metadata(within, extra["within_levels"], extra["within_cells"])
        metadata(state["design_columns"])
        if mo._digest({key: value for key, value in state.items() if key != "sha256"}) != state["sha256"]:
            raise ValueError("digest")
        counts = [mo._count(value, "saved cell count", minimum=2 if state["kind"] == "summary" else 1)
                  for value in state["counts"]]
        if len(counts) != g or sum(counts) > mo.MAX_COUNT:
            raise ValueError("counts")
        origin = mo._vector(state["origin"], p, "saved cell origin")
        offsets = mo._matrix(state["mean_offsets"], (g, p), "saved cell mean offsets")
        if len(state["cell_sscp"]) != g:
            raise ValueError("scatter dimensions")
        scatters = []
        for value, count in zip(state["cell_sscp"], counts, strict=True):
            scatter = mo._matrix(value, (p, p), "saved cell SSCP")
            mo._covariance(scatter, "saved cell SSCP", rank_limit=count-1)
            resolved_variances(scatter, "saved sample covariance", divisor=max(1, count-1))
            scatters.append(scatter)
        terms = state["terms"]
        if not isinstance(terms, list) or not 1 <= len(terms) <= k \
                or any(not isinstance(term, dict) or set(term) != {"name", "factors", "columns"} for term in terms):
            raise ValueError("terms")
        expected_terms = build_terms(factors, [], [term["factors"] for term in terms if len(term["factors"]) > 1])
        if [term.name for term in expected_terms] != [term["name"] for term in terms]:
            raise ValueError("ordered terms")
        column_names, position = [], 0
        for term, record in zip(expected_terms, terms, strict=True):
            width = math.prod(len(levels[name])-1 for name in term.factors)
            if record["factors"] != list(term.factors) or record["columns"] != list(range(position, position+width)):
                raise ValueError("term columns")
            column_names.extend([term.name] if width == 1 else [f"{term.name}[{j+1}]" for j in range(width)])
            position += width
        if column_names != state["design_columns"] or position != k:
            raise ValueError("design identities")
        codings = {name: effect_coding(len(values)) for name, values in levels.items()}
        expected_design = []
        for cell in cells:
            code = {name: levels[name].index(value) for name, value in zip(factors, cell, strict=True)}
            row = []
            for term in expected_terms:
                part = torch.ones(1, dtype=torch.float64)
                for name in term.factors:
                    part = kron(part, codings[name][code[name]])
                row.extend(part.tolist())
            expected_design.append(row)
        design = mo._matrix(state["design"], (g, k), "saved categorical design")
        if not torch.equal(design, torch.tensor(expected_design, dtype=torch.float64)):
            raise ValueError("design/cells mismatch")
        provenance = state["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("sample provenance")
        mo._count(provenance["n_missing"], "saved missing rows", minimum=0)
        if state["kind"] == "summary":
            if provenance["n_missing"] != 0 or provenance.get("physical_subjects" if rm else "physical_rows") is not None \
                    or provenance.get("covariance_divisor") != ("group subject count-1" if rm else "cell count-1"):
                raise ValueError("declared summary provenance")
        else:
            rows = mo._count(provenance["n_input_rows"], "saved physical input rows")
            plan(p, rows=rows, cells=g, k=k)
            frequency = [mo._count(value, "saved frequency", minimum=0) for value in provenance["frequencies"]]
            positive = [i for i, value in enumerate(frequency) if value > 0]
            if provenance["positive_complete_positions"] != positive:
                raise ValueError("zero/positive frequency order")
            codes = provenance["group_codes" if rm else "cell_codes"]
            if len(codes) != len(positive) or any(type(code) is not int or not 0 <= code < g for code in codes) \
                    or [sum(frequency[i] for i, code in zip(positive, codes, strict=True) if code == j)
                        for j in range(g)] != counts:
                raise ValueError("group frequency alignment")
            if rm:
                subjects, complete, dropped = (provenance[name] for name in
                                              ("subject_order", "complete_subject_order", "dropped_subjects"))
                if len(subjects) > 10000 or len(set(subjects)) != len(subjects) \
                        or any(encode(_scalar(decode(value))) != value for value in subjects) \
                        or len(complete) != len(frequency) or len(set(complete)) != len(complete) \
                        or set(complete)&set(dropped) or set(complete)|set(dropped) != set(subjects) \
                        or complete != [value for value in subjects if value in set(complete)] \
                        or dropped != [value for value in subjects if value not in set(complete)] \
                        or provenance["physical_subjects"] != len(subjects) \
                        or provenance["positive_subjects"] != len(positive) \
                        or provenance["n_zero_weight_subjects"] != len(frequency)-len(positive) \
                        or provenance["n_dropped_subjects"] != len(dropped) \
                        or provenance["expanded_subjects"] != sum(counts):
                    raise ValueError("subject sample")
                positions = provenance["profile_positions"]
                if len(positions) != len(frequency) or any(len(row) != p for row in positions):
                    raise ValueError("profile positions")
                flat = [i for row in positions for i in row]
                if len(set(flat)) != len(flat) or any(type(i) is not int or not 0 <= i < rows for i in flat) \
                        or len(flat)+provenance["n_missing"] != rows:
                    raise ValueError("whole-subject deletion alignment")
            else:
                positions = provenance["source_positions"]
                if positions != sorted(set(positions)) or len(positions) != len(frequency) \
                        or any(type(i) is not int or not 0 <= i < rows for i in positions) \
                        or len(positions)+provenance["n_missing"] != rows \
                        or provenance["physical_rows"] != len(positive) \
                        or provenance["n_zero_weight"] != len(frequency)-len(positive):
                    raise ValueError("row/missing sample")
            for j, scatter in enumerate(scatters):
                mo._covariance(scatter, "saved physical cell covariance", rank_limit=sum(code == j for code in codes)-1)
        return Moments(factors, levels, [tuple(cell) for cell in cells], names, origin, offsets,
                       scatters, counts, state["kind"], provenance), design, column_names
    except (TypeError, KeyError, ValueError, AttributeError, IndexError, OverflowError):
        raise AnalysisError("invalid_state", "Saved complete factorial/RM moments, design, counts or sample identities are invalid.") from None


def validate_factorial_result(result, geometry):
    state = result.attrs.get("factorial_moment_state")
    moments, design, names = validate_moments(state)
    n, k = sum(moments.counts), len(names)
    expected_kind = "resident_factorial_frequency" if moments.kind == "frequency" else "factorial_cell_summary"
    if geometry["input_kind"] != expected_kind:
        raise AnalysisError("invalid_state", "Saved factorial input kind disagrees with its moment provenance.")
    if geometry["n"] != n or geometry["df_resid"] != n-k or geometry["n_missing"] != moments.provenance["n_missing"] \
            or geometry["design_columns"] != names or geometry["outcome_columns"] != moments.names \
            or geometry["outcome_origin"] != state["origin"]:
        raise AnalysisError("invalid_state", "Saved fitted geometry disagrees with declared complete-cell moments.")
    beta = mo._matrix(geometry["coefficient_offsets"], (k, len(moments.names)), "saved fitted offsets")
    bread = mo._matrix(geometry["bread"], (k, k), "saved bread")
    error = mo._matrix(geometry["residual_sscp"], (len(moments.names), len(moments.names)), "saved fitted residual SSCP")
    count = torch.tensor(moments.counts, dtype=torch.float64)
    normal = design.T@(count[:, None]*design)
    if not torch.allclose(normal@bread, torch.eye(k, dtype=torch.float64), rtol=1e-8, atol=1e-8):
        raise AnalysisError("invalid_state", "Saved bread disagrees with the declared weighted categorical design.")
    residual = moments.offsets-design@beta
    condition = design.T@(count[:, None]*residual)
    absolute = design.abs().T@(count[:, None]*(moments.offsets.abs()+design.abs()@beta.abs()))
    if bool((condition.abs() > 32768*torch.finfo(torch.float64).eps*absolute.clamp_min(1e-300)).any()):
        raise AnalysisError("invalid_state", "Saved coefficients disagree with the declared cell mean normal equations.")
    expected_error = torch.stack(moments.scatters).sum(0)
    if k != len(moments.cells):
        expected_error += residual.T@(count[:, None]*residual)
    scale = expected_error.diagonal().clamp_min(1e-300).sqrt()
    if not torch.allclose(error/torch.outer(scale, scale), expected_error/torch.outer(scale, scale), rtol=1e-10, atol=1e-10):
        raise AnalysisError("invalid_state", "Saved residual SSCP disagrees with declared cell sufficient moments.")
