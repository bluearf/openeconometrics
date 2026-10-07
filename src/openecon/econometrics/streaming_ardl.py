"""Disk-ordered ARDL with one full-source factor for lag-grid selection.

The reduced rows encode the original geometry; likelihoods, degrees of freedom,
covariance meats, response moments and lagged residual differences use the full
number of actual periods. No source-sized lag matrix is retained.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.resources import plan_workspace
from openecon.streaming_design import numeric_values
from . import registry
from .core import kernel_call
from .ordered_replay import OrderedReplay
from .streaming_linear import _Notes
from .streaming_var import _Facade, _head
from .tsmodels import ardl


class _ARDLReplay:
    def __init__(self, ordered):
        self.ordered, self.spec = ordered, ordered.spec
        self.notes = _Notes(self.spec)
        self.parameter_basis = None
        self.frame = SimpleNamespace(spec=self.spec, info=registry.get(self.spec.estimator),
                                     option=self.notes.option, role=lambda name: registry.role_columns(self.spec, name),
                                     warn=self.notes.warn, warnings=self.notes.warnings, notes={},
                                     drop_collinear=self.drop_collinear)

    def matrix(self, frame, terms):
        return torch.stack([numeric_values(frame[term], term) for term in terms], dim=1) if terms else torch.empty((len(frame), 0), dtype=torch.float64)

    def input_batches(self):
        return self.ordered.source.iter_batches(batch_rows=self.ordered.rows)

    def factory(self):
        carry, seen = None, 0
        for raw in self.input_batches():
            old = 0 if carry is None else len(carry)
            joined = pd.concat((carry, raw), ignore_index=True) if carry is not None else raw.reset_index(drop=True)
            first = max(self.largest, old)
            if first >= len(joined):
                seen += len(raw)
                carry = joined.tail(self.largest).copy() if self.largest else None
                continue
            output = joined.iloc[first:][[self.spec.outcome, self.ordered.positions_column]].copy().reset_index(drop=True)
            if self.trend != "none":
                output["Intercept"] = 1.
            if self.trend == "trend":
                output["trend"] = list(range(seen-old+first+1, seen+len(raw)+1))
            for lag in range(1, self.maxlags[0]+1):
                output[ardl._lag_name(self.spec.outcome, lag)] = joined[self.spec.outcome].iloc[first-lag:len(joined)-lag].to_numpy()
            for name, maximum in zip(self.names, self.maxlags[1:], strict=True):
                for lag in range(maximum+1):
                    output[ardl._lag_name(name, lag)] = joined[name].iloc[first-lag:len(joined)-lag].to_numpy()
            for name in self.exog:
                output[name] = joined[name].iloc[first:].to_numpy()
            output["__previous_y__"] = joined[self.spec.outcome].iloc[first-1:-1].to_numpy() if self.largest else 0.
            seen += len(raw)
            carry = joined.tail(self.largest).copy() if self.largest else None
            if len(output):
                yield output

    def prepare(self, names, exog, lags, maxlags, trend, ec, ic, builder):
        self.names, self.exog, self.maxlags, self.trend = names, exog, maxlags, trend
        self.largest = max(maxlags)
        self.n = self.ordered.count-self.largest
        if self.n <= int(trend != "none")+int(trend == "trend")+len(exog)+sum(maxlags)+len(names)+2:
            raise AnalysisError("insufficient_observations", "ARDL needs more periods for the requested maximum lags.")
        self.det_terms = (["Intercept"] if trend != "none" else [])+(["trend"] if trend == "trend" else [])
        all_terms = [*self.det_terms, *[ardl._lag_name(self.spec.outcome, lag) for lag in range(1, maxlags[0]+1)],
                     *[ardl._lag_name(name, lag) for name, maximum in zip(names, maxlags[1:], strict=True) for lag in range(maximum+1)], *exog]
        if len(all_terms) != len(set(all_terms)) or self.spec.outcome in all_terms or {"__previous_y__", self.ordered.positions_column}&set(all_terms):
            raise AnalysisError("duplicate_terms", "Generated ARDL lag names collide; rename the conflicting source column.")
        self.all_terms = all_terms
        width = len(all_terms)+3
        if width > 384:
            raise AnalysisError("model_too_wide", "ARDL maximum-lag factor exceeds384 columns.")
        fixed = 1024*width**2+8*1024**2
        self.ordered.rows = min(self.ordered.rows, max(1, (self.ordered.budget-fixed)//(160*width)))
        self.plan = plan_workspace("bounded ARDL maximum-lag joint factor", {
            "lag_rows_and_covariance_scores": 160*self.ordered.rows*width,
            "factor_and_covariance": 1024*width**2,
            "lag_carry": 256*(self.largest+1)*len(self.ordered.columns),
            "reader_and_SQL_cache": 8*1024**2,
        }, budget_bytes=self.ordered.budget)
        moments = _WeightedMoments(width, intercept=True)
        def joint(frame):
            return torch.cat((torch.ones((len(frame), 1), dtype=torch.float64), self.matrix(frame, [self.spec.outcome, "__previous_y__", *all_terms])), dim=1)
        for frame in self.factory():
            block = joint(frame)
            moments.add(block, torch.ones(len(block), dtype=torch.float64), False)
        means = moments.anchor+moments.magnitude*moments.mean
        scales = moments.magnitude*(moments.m2.value/moments.mass).clamp_min(0).sqrt()
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        means[0], scales[0] = 0., 1.
        tree = _TSQRTree()
        for frame in self.factory():
            tree.add(qr_factor((joint(frame)-means)/scales))
        factor = tree.finish()
        raw = factor*scales+factor[:, :1]*means
        self.ones, self.target, self.lagged_outcome_factor = raw[:, 0], raw[:, 1], raw[:, 2]
        self.master = raw[:, 3:]
        self.column_index = {name: i for i, name in enumerate(all_terms)}
        self.report = pd.concat(list(_head(self.factory(), min(self.n, 400))), ignore_index=True)
        self.positions = self.report.pop(self.ordered.positions_column).tolist()
        self.frame.__dict__.update(n=len(self.report), original=self.report, sample=self.report,
                                   positions=self.positions, _sorted_by=[self.spec.time] if self.spec.time else [],
                                   resource_plans=[self.plan.record()], numeric=lambda name: numeric_values(self.report[name], name))
        always = self.master[:, [self.column_index[name] for name in [*self.det_terms, *exog]]]
        level_blocks = [self.master[:, [self.column_index[ardl._lag_name(self.spec.outcome, lag)] for lag in range(1, maxlags[0]+1)]]]
        level_blocks += [self.master[:, [self.column_index[ardl._lag_name(name, lag)] for lag in range(maximum+1)]] for name, maximum in zip(names, maxlags[1:], strict=True)]
        selection = None
        if lags is None:
            ranges = [range(1, maxlags[0]+1)]+[range(maximum+1) for maximum in maxlags[1:]]
            if builder is None:
                yblock = level_blocks[0]
                blocks = [torch.cat((yblock[:, :1], yblock[:, :-1]-yblock[:, 1:]), dim=1)]
                blocks += [torch.cat((block[:, :1], block[:, :-1]-block[:, 1:]), dim=1) for block in level_blocks[1:]]
                lags, selection = ardl._select(self.target, always, blocks, ranges, ic, self.frame, nobs=self.n)
            else:
                lags, selection = ardl._select_constrained(self.target, always, level_blocks, ranges, ic, self.frame, builder, names, self.det_terms+exog, nobs=self.n)
        if ec and lags[0] < 1:
            raise AnalysisError("invalid_option", "Error-correction reporting needs at least one outcome lag.")
        terms, roles = list(self.det_terms), [("det", name, 0) for name in self.det_terms]
        for lag in range(1, lags[0]+1):
            terms.append(ardl._lag_name(self.spec.outcome, lag))
            roles.append(("y", None, lag))
        for j, name in enumerate(names):
            for lag in range(lags[j+1]+1):
                terms.append(ardl._lag_name(name, lag))
                roles.append(("x", j, lag))
        for name in exog:
            terms.append(name)
            roles.append(("exog", name, 0))
        return {"total": self.ordered.count, "n": self.n, "largest": self.largest,
                "lags": lags, "selection": selection, "terms": terms, "roles": roles,
                "y": self.target, "x": self.master[:, [self.column_index[name] for name in terms]],
                "det_terms": self.det_terms}

    def drop_collinear(self, design):
        x = design.x.clone()
        if design.intercept:
            x[:, 1:] -= self.ones[:, None]*((self.ones@x[:, 1:])/self.n)
        kept, omitted = kernel_call(collinear_columns, x)
        if omitted:
            names = [design.terms[i] for i in omitted]
            self.frame.notes.setdefault("omitted_terms", []).extend(names)
            self.notes.warn("Omitted because of collinearity: "+", ".join(names))
        return design.select(kept)

    def set_design(self, design):
        self.design = design

    def covariance(self, frame, *, x, resid, bread, n, k, df_resid, ssr):
        kind = "HC1" if self.spec.covariance == "robust" else self.spec.covariance
        info = {"covariance": self.spec.covariance, "df_inference": df_resid}
        if kind == "nonrobust":
            return bread*(ssr/df_resid), {**info, "correction": "classical: SSR/(N-K)"}
        basis = self.parameter_basis if self.parameter_basis is not None and x.shape[1] != self.design.x.shape[1] else torch.eye(len(self.design.terms), dtype=torch.float64)
        beta = kernel_call(least_squares, x, self.target, drop_collinear=False).beta
        meat = _CompensatedSum((k, k))
        for source in self.factory():
            values = self.matrix(source, self.design.terms)@basis
            residual = numeric_values(source[self.spec.outcome], self.spec.outcome)-values@beta
            if kind in {"HC2", "HC3"}:
                leverage = ((values@bread)*values).sum(1)
                if bool((leverage >= 1).any()):
                    raise AnalysisError("invalid_leverage", "HC2/HC3 needs every retained observation to have leverage below one.")
                residual /= (1-leverage).sqrt() if kind == "HC2" else 1-leverage
            score = values*residual[:, None]
            meat.add(score.T@score)
        factor = n/(n-k) if kind == "HC1" else 1.
        return bread@meat.value@bread*factor, {**info, "correction": kind+(": N/(N-K)" if kind == "HC1" else ""), "small_sample_correction": factor}

    def response_tss(self, ec, intercept):
        target = self.target-self.lagged_outcome_factor if ec else self.target
        if intercept:
            target = target-self.ones*float(self.ones@target/self.n)
        return float(target@target)

    def durbin_watson(self, beta):
        squares, differences = _CompensatedSum(()), _CompensatedSum(())
        previous = None
        for source in self.factory():
            residual = numeric_values(source[self.spec.outcome], self.spec.outcome)-self.matrix(source, self.design.terms)@beta
            squares.add(residual.square().sum())
            joined = residual if previous is None else torch.cat((previous, residual))
            differences.add(joined.diff().square().sum())
            previous = residual[-1:].clone()
        return float(differences.value/squares.value)

    def preview(self, beta, ec):
        observed = numeric_values(self.report[self.spec.outcome], self.spec.outcome)
        fitted = self.matrix(self.report, self.design.terms)@beta
        if ec:
            previous = numeric_values(self.report["__previous_y__"], "__previous_y__")
            observed, fitted = observed-previous, fitted-previous
        return fitted, observed

    def finalize(self, result):
        facade = _Facade(self.spec, self.ordered, self.report, self.positions, self.n, {}, {})
        result.provenance.update(facade.provenance())
        result.provenance.update({"solver": "global_tsqr_disk_ordered_ardl",
                                  "sample_order": "sorted by "+self.spec.time if self.spec.time else "input row order",
                                  "resource_plans": [self.plan.record(), self.ordered.plan.record()],
                                  "factor_rows_are_observations": False})
        result.nobs_original = self.ordered.original_count
        result.dropped_rows = self.ordered.original_count-self.n
        result.sample_positions = []
        return result


def fit_streaming_ardl(spec, source):
    with OrderedReplay(spec, source) as ordered:
        replay = _ARDLReplay(ordered)
        result = kernel_call(ardl._fit_ardl_frame, replay.frame, _replay=replay)
        ordered.verify_original()
        return result
