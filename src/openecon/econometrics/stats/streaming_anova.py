"""Replay factorial ANOVA/ANCOVA with sum-to-zero coding and blocked TSQR.

The source is never collected. Factor levels/cells and the compressed QR are
bounded structural state; changing replay content is refused before inference.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.engines import linalg
from openecon.engines.streaming_ols import _TSQRTree
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import common as c
from .anova import ANOVA_COLUMNS, anova_rows, marginal_means
from .glm import LinearModel, _key, build_terms, check_sequential_order, effect_coding
from .replay import Moments, Replay


class ReplayLinearModel(LinearModel):
    def __init__(self, sample: Replay, y: str | list[str], factors: list[str], covariates: list[str],
                 interactions: Any, slopes: bool, *, allow_intercept: bool = False):
        self.terms = build_terms(factors, covariates, interactions, slopes)
        if not factors and not covariates and not allow_intercept:
            raise AnalysisError("invalid_spec", "Name at least one factor or covariate.")
        self._keys = {_key(term): term for term in self.terms}
        outcomes = [y] if isinstance(y, str) else list(y)
        numeric = [*outcomes, *covariates]
        state = Moments(len(numeric))
        discovered = {name: set() for name in factors}
        present = {term.name: set() for term in self.terms if len(term.factors) > 1}
        for frame in sample.batches():
            state.add(c.matrix(frame, numeric, check_scale=False))
            for name in factors:
                _, labels = c.group_codes(frame, name)
                discovered[name].update(labels)
                if len(discovered[name]) > 2000 or sum(len(str(v)) for v in discovered[name]) > 1024**2:
                    raise AnalysisError("design_too_large", "Factor-level metadata exceeds bounded factorial workspace.")
            for term in self.terms:
                if term.name in present:
                    rows = frame.loc[:, list(term.factors)].drop_duplicates()
                    for row in rows.itertuples(index=False, name=None):
                        present[term.name].add(tuple(c._json_scalar(v) for v in row))
                    if len(present[term.name]) > 2000:
                        raise AnalysisError("design_too_large", "Interaction-cell metadata exceeds bounded factorial workspace.")
        self.n, self.m = state.n, len(outcomes)
        self.shift = state.location[:self.m]
        self.levels = {}
        for name, values in discovered.items():
            categories = sample.category_schema[name]
            if categories is None:
                self.levels[name] = c.group_codes(pd.DataFrame({name: list(values)}), name)[1]
            else:
                order = {c._json_scalar(value): index for index, value in enumerate(categories)
                         if c._json_scalar(value) in values}
                self.levels[name] = sorted(values, key=order.__getitem__)
        self.centred = bool(covariates) and self._closed(self.terms)
        self.centres = {name: float(state.location[i+self.m]) if self.centred else 0.0
                        for i, name in enumerate(covariates)}
        self.covariates = {name: state.location[i+self.m:i+self.m+1] for i, name in enumerate(covariates)}
        self.codes = {}
        self.columns = {}
        position = 0
        for name, labels in self.levels.items():
            if len(labels) < 2:
                raise AnalysisError("single_level", f"Factor '{name}' has only one level in the sample.")
        for term in self.terms:
            width = 1
            for name in term.factors:
                width *= len(self.levels[name]) - 1
            if position + width > 2000:
                raise AnalysisError("design_too_large", "Factorial designs are limited to 2000 parameters.")
            self.columns[term.name] = list(range(position, position + width))
            position += width
        self.k = position
        if self.k > 2000:
            raise AnalysisError("design_too_large", "Factorial designs are limited to 2000 parameters.")
        if self.n - self.k < 1:
            raise AnalysisError("no_residual_df", "The factorial model has no residual degrees of freedom.")
        for term in self.terms:
            if term.name in present:
                size = 1
                for name in term.factors:
                    size *= len(self.levels[name])
                if len(present[term.name]) < size:
                    raise AnalysisError("empty_cells", f"{size-len(present[term.name])} interaction cells are unobserved; Type IV sums of squares are not implemented.")
        # Reserve geometry before coding matrices or QR factors
        # are allocated; source-reader buffers are outside this tensor estimate.
        geometry = 256 * (self.k + self.m)**2 + 2 * 1024**2
        rows = min(sample.rows, max(1, (workspace_budget_bytes() - geometry) // (64 * (self.k + len(numeric) + 1))))
        self.resource_plan = plan_workspace("chunked factorial ANOVA", {
            "factorial_geometry_and_QR": geometry,
            "encoded_design_blocks": rows * 64 * (self.k + len(numeric) + 1)})
        sample.rows = rows
        self.codings = {name: effect_coding(len(labels)) for name, labels in self.levels.items()}
        global_covariates = self.covariates
        tree = _TSQRTree()
        for frame in sample.batches():
            self.codes = {}
            for name in factors:
                codes, labels = c.group_codes(frame, name)
                locations = {label: index for index, label in enumerate(self.levels[name])}
                if any(label not in locations for label in labels):
                    raise AnalysisError("source_changed", "Factor levels changed during ANOVA replay.")
                indices = torch.tensor([locations[label] for label in labels], dtype=torch.int64)
                self.codes[name] = indices[codes]
            self.covariates = {name: c.values(frame, name, check_scale=False) for name in covariates}
            response = c.matrix(frame, outcomes, check_scale=False) - self.shift
            block = torch.cat((self._block(0, len(frame)), response), 1)
            if not bool(torch.isfinite(block).all()):
                raise AnalysisError("numerical_failure", "The factorial design exceeds finite float64 precision.")
            tree.add(torch.linalg.qr(block, mode="r").R)
        self.covariates, self.codes = global_covariates, {}
        self.factor = tree.finish()
        self.recentre, self.uncentre = self._transform(1.0), self._transform(-1.0)
        self.full = self._fit(list(range(self.k)))
        self.rank, self.df_error = self.k, self.n-self.k
        self.beta = self.full.beta if self.full.beta.ndim == 2 else self.full.beta[:, None]
        residual = self.full.resid if self.full.resid.ndim == 2 else self.full.resid[:, None]
        self.error = linalg.symmetrize(residual.T @ residual)
        self.raw_total = float(state.raw_ss[0])
        self.corrected_total = float(state.sscp[0, 0])
        self.raw_totals = state.raw_ss[:self.m].clone()
        self.corrected_totals = state.sscp.diagonal()[:self.m].clone()
        self.tsqr = {"tsqr_reduction_depth": tree.depth, "tsqr_peak_factors": tree.peak_factors}


def anova(data: Any, y: str, factors: list[str], covariates: list[str], *, interactions: Any,
          slopes: bool, ss_type: int, emmeans: list[Any] | None, adjust: str,
          alpha: float, missing: str) -> TableSet:
    sample = Replay(data, [y, *factors, *covariates], [y, *covariates], missing)
    model = ReplayLinearModel(sample, y, factors, covariates, interactions, slopes)
    if ss_type == 1:
        check_sequential_order(model.terms)
    # Use the same resident inference/finalizer, passing small sufficient totals
    # instead of an n-row outcome table.
    (rows, labels), summary = anova_rows(model, 0, ss_type, None,
                                        totals=(model.raw_total, model.corrected_total))
    tables = {"anova": c.frame(rows, columns=ANOVA_COLUMNS, index=labels)}
    if emmeans:
        tables.update(marginal_means(model, list(emmeans), factors, alpha, adjust))
    headline = tables["anova"].iloc[0]
    return TableSet(tables, title=f"Analysis of variance of {y} (Type {'I'*ss_type} sums of squares)",
                    statistic=headline["statistic"] if model.k > 1 else None,
                    p_value=headline["p_value"] if model.k > 1 else None, distribution="F", **summary,
                    n=model.n, n_missing=sample.dropped, ss_type=ss_type, alpha=alpha,
                    terms=[term.name for term in model.terms], levels=model.levels,
                    coding="sum-to-zero (effect) coding", missing="listwise", **sample.attrs(),
                    factorial_resource_plan=model.resource_plan.record(), **model.tsqr)
