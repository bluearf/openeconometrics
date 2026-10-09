"""Bounded panel/absorbed 2SLS using native projections and actual-row scores.

Global QR factors represent linear geometry only. Residuals and every sandwich
or nonlinear diagnostic are replayed on actual retained observations. Multi-way
FE projection uses owned sequential scratch vectors, never an mmap/collection.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import copy
from types import SimpleNamespace
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.linalg import collinear_columns
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import f_sf
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import Design, kernel_call, wald_test
from .iv.common import Blocks, Setting, check_fit, check_pweights, estimate
from .iv.diagnostics import Diagnostics, chi2_entry, unavailable
from .iv.ivreghdfe import _controls, _WMATRIX
from .iv.xtivreg import _flag_constant
from .replay_sample import ReplayBatch, ReplaySample
from .streaming_hdfe import _Selection, _Vectors, _absorb
from .streaming_iv import _geometry_rows, _robust_diagnostics, _roles
from .streaming_iv_methods import c_test, estimate_replay, method_diagnostics
from .streaming_linear import (
    _CrossMoments, _GroupMeans, _Notes, _covariance, _finite, _period_keys,
    _result, _weights,
)


def _sample(spec, source, batch_rows, row_filter=None):
    endogenous, instruments = _roles(spec)
    sample = ReplaySample(spec, source, batch_rows=batch_rows, row_filter=row_filter)
    x = sample.add_design("regressors", [*spec.predictors, *endogenous], intercept=True)
    z = sample.add_design("instruments", [*spec.predictors, *instruments], intercept=True)
    sample.prepare()
    k1 = len([term for term in x.terms if term not in endogenous])
    if x.terms[:k1] != z.terms[:k1]:
        raise AnalysisError("singular_design", "Exogenous rank screens differ between regressor and instrument blocks.")
    return sample, x, z, k1


def _response(sample):
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(sample.spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), _weights(sample, batch), False)
    anchor, magnitude, center = float(moments.anchor[1]), float(moments.magnitude[1]), float(moments.mean[1])
    scale = magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
    if not math.isfinite(scale) or scale <= 0:
        raise AnalysisError("constant_outcome", "The IV outcome has no finite variation.")
    def values(batch):
        return ((batch.numeric(sample.spec.outcome)-anchor)/magnitude-center)*(magnitude/scale)
    return values, anchor+magnitude*center, scale


def _raw(batch, response, k1):
    return torch.cat((response(batch)[:, None], batch.designs["regressors"][:, 1:],
                      batch.designs["instruments"][:, k1:]), 1)


def _screen(sample, notes, factor, before, after, x, z, k1, intercept):
    """Left-to-right screens on small globally scaled transformed factors."""
    exog = list(range(k1-1))
    endogenous = list(range(k1-1, len(x.terms)-1))
    excluded = list(range(len(x.terms)-1, len(x.terms)-1+len(z.terms)-k1))
    terms = [*x.terms[1:], *z.terms[k1:]]
    offset = int(intercept)
    small = factor[:, offset:-1].clone()
    if intercept:
        c = factor[:, :1]
        small -= c*((c[:, 0]@small)/(c[:, 0]@c[:, 0]))
    gone = set((after <= 1e-13*before).nonzero().flatten().tolist())
    omitted_instruments = [term for term in z.predictors if term not in z.terms and term not in sample.spec.predictors]
    def reduce(candidates, base, instrument=False):
        alive = [i for i in candidates if i not in gone]
        matrix = small[:, [*base, *alive]]
        kept, _ = collinear_columns(matrix)
        chosen = [alive[i-len(base)] for i in kept if i >= len(base)]
        omitted = [i for i in candidates if i not in chosen]
        for i in omitted:
            if instrument:
                omitted_instruments.append(terms[i])
            else:
                sample.notes["omitted_terms"].append(terms[i])
                notes.warn(f"Omitted {terms[i]}: collinearity with the fixed effects or preceding regressors.")
        return chosen
    exog = reduce(exog, [])
    endogenous = reduce(endogenous, exog)
    excluded = reduce(excluded, exog, True)
    if not endogenous:
        raise AnalysisError("no_endogenous_regressors", "Every endogenous regressor was omitted after the FE/panel projection; use ordinary regression.")
    if len(excluded) < len(endogenous):
        raise AnalysisError("underidentified", "There are fewer usable excluded instruments than endogenous regressors after the FE/panel projection.")
    omitted_instruments = list(dict.fromkeys(omitted_instruments))
    sample.notes["omitted_terms"] = list(dict.fromkeys(term for term in sample.notes["omitted_terms"] if term not in omitted_instruments))
    if omitted_instruments:
        notes.warn("Instrument(s) omitted because of collinearity: "+", ".join(omitted_instruments)+".")
    selected_x = [*exog, *endogenous]
    selected_z = [*exog, *excluded]
    xterms = [*(["Intercept"] if intercept else []), *[terms[i] for i in selected_x]]
    zterms = [*(["Intercept"] if intercept else []), *[terms[i] for i in selected_z]]
    return selected_x, selected_z, len(exog)+offset, xterms, zterms, omitted_instruments


class _ActualRows:
    """Diagnostic facade that preserves real transformed rows and weights."""
    def __init__(self, base, factory, selected_x, selected_z, intercept, nobs=None):
        self.base, self.factory = base, factory
        self.selected_x, self.selected_z, self.intercept = selected_x, selected_z, intercept
        self.spec, self.nobs = base.spec, base.nobs if nobs is None else nobs
        self.weight_multiplier = base.weight_multiplier
        self.weight_max, self.weight_mean = base.weight_max, base.weight_mean
    def batches(self):
        for batch, values in self.factory():
            one = torch.ones((len(values), 1), dtype=torch.float64)
            xs, zs = values[:, [i+1 for i in self.selected_x]], values[:, [i+1 for i in self.selected_z]]
            batch.designs = {"regressors": torch.cat((one, xs), 1) if self.intercept else xs,
                            "instruments": torch.cat((one, zs), 1) if self.intercept else zs}
            batch.response_work = values[:, 0]
            yield batch


def _fit_geometry(factor, selected_x, selected_z, k1, xterms, zterms, notes, n, absorbed, intercept, omitted):
    offset = int(intercept)
    columns = [*([0] if intercept else []), *[i+offset for i in selected_z],
               *[i+offset for i in selected_x[k1-offset:]], factor.shape[1]-1]
    compressed, weights = _geometry_rows(factor[:, columns], intercept, n)
    n_instruments, q = len(zterms), len(xterms)-k1
    z, x2, y = compressed[:, :n_instruments], compressed[:, n_instruments:n_instruments+q], compressed[:, -1]
    x1 = z[:, :k1]
    designs = Blocks(Design(x1, xterms[:k1], {}, intercept),
                     Design(x2, xterms[k1:], {}, False), Design(z[:, k1:], zterms[k1:], {}, False),
                     x1, x2, z[:, k1:], omitted)
    setting = Setting(n, weights, "nonrobust", True, absorbed=absorbed)
    est = estimate(notes, setting, y, designs, method="2sls")
    return est, designs, setting, (x1, x2, z, y, weights)


def fit_streaming_fe_iv(spec, source, *, batch_rows=None):
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (torch.linalg.LinAlgError, RuntimeError) as error:
        raise AnalysisError("numerical_failure", "Native FE IV replay could not complete a finite linear algebra operation; rescale inputs or revise collinear terms.") from error


def _fit(spec, source, *, batch_rows=None):
    notes = _Notes(spec)
    if spec.estimator not in {"xtivreg", "ivreghdfe"}:
        raise AnalysisError("invalid_spec", "This replay path requires xtivreg or ivreghdfe.")
    panel = spec.estimator == "xtivreg"
    model = notes.option("model") if panel else "hdfe"
    if panel and (model not in {"fe", "be"} or notes.option("ec2sls")):
        raise AnalysisError("unsupported_streaming_method", "Native xtivreg replay supports FE and BE 2SLS; FD/RE/EC2SLS require separate verified replay kernels.")
    method = "2sls" if panel else notes.option("method")
    check_pweights(notes)
    dimensions = [spec.panel] if panel else list(spec.columns["absorb"])
    tolerance, max_iterations = (1e-10, 1) if panel else _controls(notes)
    cluster_names = ([spec.panel] if model == "fe" and spec.covariance == "robust" else
                     registry.cluster_columns(spec) if spec.covariance == "cluster" else [])
    with ExitStack() as stack:
        first, x, z, original_k1 = _sample(spec, source, batch_rows)
        selection = None
        dropped, metadata = 0, {}
        if not panel:
            first.plan_rows("absorbed IV singleton discovery", {"selection_SQLite_cache": 2*1024**2},
                            128*(len(dimensions)+len(x.terms)+len(z.terms))+512)
            selection = _Selection(len(dimensions), len(cluster_names))
            stack.callback(selection.close)
            selection.seed(first, dimensions, cluster_names, notes.option("drop_singletons"))
            dropped = first.nrows-selection.used
            sample, x, z, original_k1 = _sample(spec, source, batch_rows,
                         (lambda frame: selection.filter(frame, dimensions)) if dropped else None)
            if sample.baseline["data_hash"] != first.baseline["data_hash"] or sample.baseline["positions_hash"] != selection.position_hash():
                raise AnalysisError("source_changed", "The source or singleton-selected rows changed before absorbed IV replay.")
            levels, redundant, nested, total = selection.degrees_of_freedom()
            absorbed = total+int(all(nested))
            metadata = {"absorbed": [{"column": name, "levels": count, "redundant": lost, "nested": flag}
                        for name, count, lost, flag in zip(dimensions, levels, redundant, nested, strict=True)],
                        "constant_degree_of_freedom_added": all(nested), "singletons_dropped": dropped,
                        "tolerance": tolerance}
            if dropped:
                notes.warn(f"Dropped {dropped} singleton observation(s) alone in a level of an absorbed dimension.")
        else:
            sample = first
            absorbed = 0
        width = len(x.terms)+len(z.terms)-original_k1
        resource = sample.plan_rows("native FE IV projections, joint TSQR and diagnostic replay", {
            "initial_reporting_sample": first.reporting_bytes if not panel else 0,
            "initial_design_metadata": 128*(width+1)**2 if not panel else 0,
            "initial_category_metadata": first._category_bytes if not panel else 0,
            "selection_and_projection_SQLite_caches": (4 if not panel else 2)*1024**2,
            "cluster_accumulator_caches": 18*1024**2 if cluster_names else 0,
            "global_IV_geometry_covariance_diagnostics": 4096*(width+1)**2,
        }, 384*width+128*len(dimensions)+1024)
        response, y_mean, y_scale = _response(sample)
        before, after = _CompensatedSum((width-1,)), _CompensatedSum((width-1,))
        tss, tss_within = _CompensatedSum(()), _CompensatedSum(())
        store = states = None
        if panel:
            store = _GroupMeans(width, len(cluster_names), panel=True)
            stack.callback(store.close)
            for batch in sample.batches():
                values = _raw(batch, response, original_k1)
                store.add(encode_cluster_labels(batch.frame[spec.panel]), values, _weights(sample, batch),
                          [encode_cluster_labels(batch.frame[name]) for name in cluster_names],
                          _period_keys(batch.frame[spec.time]) if spec.time else None)
                before.add((values[:, 1:].square()*_weights(sample, batch)[:, None]).sum(0))
                tss.add((values[:, 0].square()*_weights(sample, batch)).sum())
            store.finish()
            structure = store.panel_structure()
            if model == "be" and not all(store.nested_flags):
                raise AnalysisError("cluster_varies_within_panel", "Cluster columns must be constant within each panel for the between estimator.")
            if model == "fe" and sample.nrows <= store.groups:
                raise AnalysisError("insufficient_observations", "Every panel has a single observation; within IV has no variation.")
            absorbed = store.groups-1 if model == "fe" else 0
            if model == "fe" and cluster_names and any(store.nested_flags):
                absorbed = 0
            def row_factory():
                if model == "be":
                    for values, _, _, labels in store.group_batches(sample.rows):
                        batch = ReplayBatch(pd.DataFrame(index=range(len(values))), torch.ones(len(values), dtype=torch.float64),
                                            torch.arange(len(values)), {})
                        batch.cluster_keys = labels
                        yield batch, values
                else:
                    for batch in sample.batches():
                        raw = _raw(batch, response, original_k1)
                        yield batch, raw-store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
            metadata.update({"model": model, "absorbed_effects": store.groups if model == "fe" else 0})
        else:
            states = _Vectors(sample, width, len(dimensions), selection.path.stat().st_size)
            stack.callback(states.close)
            with states.writer("raw") as writer:
                for batch in sample.batches():
                    values = _raw(batch, response, original_k1)
                    weights = _weights(sample, batch)
                    before.add((values[:, 1:].square()*weights[:, None]).sum(0))
                    tss.add((values[:, 0].square()*weights).sum())
                    states.write(writer, values)
            iterations, update, absorption_method = _absorb(states, sample, dimensions, tolerance, max_iterations)
            metadata.update({"iterations": iterations, "converged": True, "demeaning_method": absorption_method, "max_update": update})
            def row_factory():
                for batch, (values,) in states.batches(sample, "y"):
                    yield batch, values
        tree, pooled_tree = _TSQRTree(), _TSQRTree()
        between_outcome = _CrossMoments(1)
        for batch, values in row_factory():
            weights = _weights(sample, batch)
            after.add((values[:, 1:].square()*weights[:, None]).sum(0))
            tss_within.add((values[:, 0].square()*weights).sum())
            if model == "be":
                between_outcome.add(values[:, :1], weights)
            design = values[:, 1:]
            if panel:
                design = torch.cat((torch.ones((len(values), 1), dtype=torch.float64), design), 1)
            tree.add(torch.linalg.qr(torch.cat((design, values[:, :1]), 1)*weights.sqrt()[:, None], mode="r")[1])
            if model == "fe" and spec.covariance == "nonrobust":
                raw = _raw(batch, response, original_k1)
                pooled_tree.add(torch.linalg.qr(torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), raw[:, 1:], raw[:, :1]), 1)*weights.sqrt()[:, None], mode="r")[1])
        factor = tree.finish()
        if model == "be":
            before = after
            tss_within.value.copy_(between_outcome.m2.value[0, 0])
        n = store.groups if model == "be" else sample.nobs
        sx, sz, k1, terms, zterms, omitted = _screen(sample, notes, factor, before.value, after.value,
                                                    x, z, original_k1, panel)
        est, designs, setting, compressed = _fit_geometry(factor, sx, sz, k1, terms, zterms, notes, n, absorbed, panel, omitted)
        beta, bread = est.beta, est.tsls.bread
        k, q, n_instruments = len(terms), len(terms)-k1, len(zterms)
        df = n-k-absorbed
        if df <= 0:
            raise AnalysisError("insufficient_observations", "The FE IV equation has no residual degrees of freedom.")
        rows = _ActualRows(sample, row_factory, sx, sz, panel, n)
        scale_x = x.scales[x.kept][[i+1 for i in sx]]
        means_x = x.means[x.kept][[i+1 for i in sx]]
        transform = torch.diag(1/scale_x)
        if panel:
            transform = torch.diag(torch.cat((torch.ones(1, dtype=torch.float64), 1/scale_x)))
            transform[0, 1:] = -means_x/scale_x
        basis = torch.linalg.inv(transform)
        small = bool(notes.option("small"))
        instrument_scales = torch.cat((x.scales[x.kept][1:], z.scales[z.kept][original_k1:]))[sz]
        loading = None
        if not panel:
            est, bread, loading = estimate_replay(rows, notes, setting, compressed[3], designs,
                method=method, wmatrix=_WMATRIX[spec.covariance], igmm=False, center=False,
                response=lambda batch: batch.response_work, instrument_basis=torch.diag(instrument_scales))
            beta = est.beta
        def residuals():
            for batch in rows.batches():
                real_x, real_z = batch.designs["regressors"], batch.designs["instruments"]
                fitted_endog = real_z@est.proj.first.beta[:, :q]
                projected = (real_z@loading if loading is not None else
                    torch.cat((real_x[:, :k1], fitted_endog-((est.kappa or 1.)-1)*(real_x[:, k1:]-fitted_endog)), 1))
                batch.residual_override = batch.response_work-real_x@beta
                if panel and model != "be":
                    raw_x = sample.designs["regressors"]
                    # Reconstruct original-level fitted values, not the projected X2.
                    raw_work = raw_x.transform[:, [0, *[i+1 for i in sx]]]
                    # Use the source's prepared global scaling; no raw dense matrix.
                    original_x = sample.raw_design(batch, "regressors")
                    batch.prediction_response = y_mean+y_scale*(original_x@raw_work@beta)
                yield batch, projected, batch.response_work, (batch.numeric(spec.outcome) if model != "be" else
                                      y_mean+y_scale*batch.response_work), y_scale
        kind = "cluster" if model == "fe" and spec.covariance == "robust" else ("HC1" if spec.covariance == "robust" else spec.covariance)
        covariance, info, rss_work, predictions = _covariance(rows, residuals, beta, bread, df=df, k=k+absorbed,
                     notes=notes, score_basis=basis, kind=kind, cluster_columns=cluster_names,
                     collect_predictions=model != "be", small=True if panel else small,
                     group_factor=True if panel else small)
        if loading is not None and spec.covariance == _WMATRIX[spec.covariance]:
            covariance = bread*info.get("small_sample_correction", 1.)
            info["correction"] = "efficient GMM (X'Z S^-1 Z'X)^-1; "+info["correction"]
        rss, overall_tss, within_tss = rss_work*y_scale**2, float(tss.value)*y_scale**2, float(tss_within.value)*y_scale**2
        # FE nonrobust residual df always counts panel effects, even if the CR1
        # covariance would remove nested effects from its correction denominator.
        actual_df = n-k-(store.groups-1 if model == "fe" else absorbed)
        if model == "fe" and spec.covariance == "nonrobust":
            actual_df = df
        check_fit(rss, overall_tss if not panel else within_tss, actual_df,
                  overall_tss+sample.nobs*y_mean*y_mean,
                  resolution=(32*tolerance)**2*within_tss if not panel and len(dimensions)>1 else 0.)
        raw_beta = y_scale*(transform@beta)
        if panel:
            raw_beta[0] += y_mean
        raw_cov = y_scale**2*(transform@covariance@transform.T)
        _finite(raw_beta, raw_cov)
        reference_df = info["df_inference"] if small else None
        tests = {"model": wald_test(raw_beta, raw_cov, range(int(panel), k), df_resid=reference_df,
                  label="F test of the slopes" if small else "Wald chi2 test of the slopes")}
        if panel:
            metrics, predictions = _panel_finish(spec, sample, response, store, sx, beta, raw_beta,
                                  raw_cov, bread, y_scale, y_mean, actual_df, rss, within_tss, structure, predictions, notes)
            if model == "fe" and spec.covariance == "nonrobust" and store.groups>1:
                label = "F test that all u_i = 0"
                try:
                    pooled_est, _, _, _ = _fit_geometry(pooled_tree.finish(), sx, sz, k1, terms, zterms,
                                                       notes, n, 0, True, omitted)
                    statistic = max(0., (pooled_est.rss*y_scale**2-rss)/(store.groups-1))/(rss/actual_df)
                    tests["fixed_effects"] = {"statistic": statistic, "df": store.groups-1, "df2": actual_df,
                         "p_value": kernel_call(f_sf, statistic, store.groups-1, actual_df), "distribution": "F", "label": label}
                except AnalysisError:
                    tests["fixed_effects"] = {"statistic": None, "df": store.groups-1, "df2": actual_df,
                            "p_value": None, "distribution": "F", "label": label,
                            "note": "the pooled IV regression is not identified"}
        else:
            diagnostic_est = copy(est)
            if method == "gmm":
                diagnostic_est.method = "2sls"
            geometry = Diagnostics(notes, setting, compressed[3], designs, diagnostic_est).run()
            if method == "gmm":
                wmatrix = _WMATRIX[spec.covariance]
                label = ("Hansen's J test of overidentifying restrictions" if wmatrix != "unadjusted" else
                         "Sargan test of overidentifying restrictions (unadjusted weight matrix)")
                geometry.tests["hansen_j"] = (chi2_entry(est.gmm["j"], n_instruments-k1-q, label, wmatrix=wmatrix)
                     if n_instruments-k1>q else unavailable("Test of overidentifying restrictions", "exactly identified: there are no overidentifying restrictions"))
            tests.update(geometry.tests)
            first_stage = geometry.first_stage
            if spec.covariance != "nonrobust" and geometry.df_first >= q:
                xr = SimpleNamespace(terms=terms, scales=scale_x, kept=list(range(len(scale_x))))
                # The diagnostic scales are in coefficient order, with no FE constant.
                all_scales = torch.cat((x.scales[x.kept][1:], z.scales[z.kept][original_k1:]))
                zr = SimpleNamespace(terms=zterms, scales=all_scales[sz], kept=list(range(len(sz))))
                tests, first_stage = _robust_diagnostics(spec, rows, notes, designs, est, geometry,
                      *compressed, est.proj.first.beta, lambda batch: batch.response_work,
                      k1, q, n_instruments, n_instruments-k1, xr, zr, absorbed=absorbed)
            c_statistic = None
            if method == "gmm":
                wide_basis = torch.diag(torch.cat((instrument_scales, scale_x[k1:])))
                c_statistic = (unavailable("Test that the endogenous regressors are exogenous",
                                           "an endogenous regressor is fitted exactly by the instruments")
                           if geometry.exact_first_stage else
                           c_test(rows, notes, est, designs, setting, lambda batch: batch.response_work, wide_basis))
            tests = method_diagnostics(tests, geometry, method, c_statistic=c_statistic)
            tests["model"] = wald_test(raw_beta, raw_cov, range(k), df_resid=reference_df,
                      label="Model F test (slopes)" if small else "Wald chi2 test of the slopes")
            metadata.update({"method": method, "exogenous": terms[:k1], "first_stage": first_stage,
                             **({"kappa": est.kappa} if est.kappa is not None else {}),
                             **({"gmm": est.gmm} if est.gmm is not None else {})})
            r2 = 1-rss/overall_tss
            metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(n-1)/df,
                       "r_squared_within": 1-rss/within_tss,
                       "adjusted_r_squared_within": 1-(rss/df)/(within_tss/(n-absorbed)),
                       "rmse": math.sqrt(rss/(df if small else n)), "df_model": k,
                       "df_resid": df, "df_absorbed": absorbed, "n_singletons_dropped": dropped}
            if est.kappa is not None:
                metrics["kappa"] = est.kappa
            if est.gmm is not None:
                metrics["j"] = est.gmm["j"]
        metrics.update({"n_instruments": n_instruments-k1, "n_endogenous": q})
        metadata.update({"endogenous": terms[k1:], "instruments": zterms[k1:], "omitted_instruments": omitted, "ssr": rss})
        info.update({"covariance": spec.covariance, "nobs": n, "df_resid": actual_df,
                     "df_inference": reference_df, "small": small, "method": method,
                     "k_total": k+absorbed, "absorbed_degrees_of_freedom": absorbed,
                     "residual_definition": "observed minus fitted response"})
        if not panel:
            info["error_variance"] = "RSS/(N-K)" if small else "RSS/N"
        if model == "fe" and spec.covariance == "robust":
            info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        diagnostics = {"joint_TSQR_depth": tree.depth, "first_stage_condition_number": est.proj.first.condition_number,
                       "second_stage_condition_number": est.condition_number}
        if store:
            diagnostics.update(store.diagnostics)
        if states:
            diagnostics.update(states.diagnostics())
            diagnostics.update(selection.diagnostics())
        result = _result(sample, terms=terms, beta=raw_beta, covariance=raw_cov, info=info, metrics=metrics,
                  notes=notes, predictions=predictions, tests=tests, solver="native_FE_projection_joint_TSQR_2SLS_score_replay",
                  diagnostics=diagnostics, extra=metadata, resource=resource.record(), use_t=small,
                  title="Panel IV (2SLS) regression" if panel else f"IV ({method.upper()}) regression with absorbed fixed effects")
        if not panel or model == 'fe':
            def fitted_blocks():
                for batch,*_ in residuals():
                    yield batch.frame,batch.numeric(spec.outcome)-y_scale*batch.residual_override
            from .postest.group_state import capture_fixed_replay
            result.extra['group_state']=capture_fixed_replay(result,fitted_blocks)
        return result


def _panel_finish(spec, sample, response, store, selected, beta, raw_beta, raw_cov, bread,
                  y_scale, y_mean, df, rss, within_tss, structure, predictions, notes):
    model = notes.option("model")
    overall, within, effects, between = _CrossMoments(2), _CrossMoments(2), _CrossMoments(2), _CrossMoments(3)
    for means, _, _, _ in store.group_batches(sample.rows):
        xb = means[:, [i+1 for i in selected]]@beta[1:]
        between.add(torch.stack((xb, means[:, 0], means[:, 0]-xb), 1), torch.ones(len(means), dtype=torch.float64))
    if model == "be":
        predictions = []
    for batch in sample.batches():
        x = batch.designs["regressors"][:, [i+1 for i in selected]]
        y = response(batch)
        means = store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
        xb, xb_mean = x@beta[1:], means[:, [i+1 for i in selected]]@beta[1:]
        one = torch.ones(len(y), dtype=torch.float64)
        overall.add(torch.stack((xb, y), 1), one)
        within.add(torch.stack((xb-xb_mean, y-means[:, 0]), 1), one)
        effects.add(torch.stack((means[:, 0]-xb_mean, xb), 1), one)
        if model == "be":
            take = min(400-len(predictions), len(y))
            fitted = y_mean+y_scale*(beta[0]+xb)
            for position, observed, prediction in zip(batch.positions[:take].tolist(), batch.numeric(spec.outcome)[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row": position, "observed": observed, "fitted": prediction, "residual": observed-prediction})
    def squared(moments):
        var = moments.m2.value.diagonal()
        if min(float(var[0]), float(var[1])) <= 1e-28*max(float(var[0]), float(var[1])):
            return None
        return moments.squared_correlation()
    metrics = {"r_squared_within": 1-rss/within_tss if model == "fe" else squared(within),
               "r_squared_between": 1-rss/within_tss if model == "be" else squared(between),
               "r_squared_overall": squared(overall), "rmse": math.sqrt(rss/df), "df_resid": df, **structure}
    if model == "fe":
        sigma_u = y_scale*math.sqrt(max(0., float(between.m2.value[2, 2]))/(store.groups-1)) if store.groups>1 else None
        sigma_e = math.sqrt(rss/df)
        metrics.update({"sigma_u": sigma_u, "sigma_e": sigma_e,
                        "rho": None if sigma_u is None else sigma_u**2/(sigma_u**2+sigma_e**2),
                        "corr_u_xb": effects.correlation()})
        if spec.covariance != "nonrobust":
            # Global standardized centered regressors imply the raw intercept's
            # variance is xbar' V xbar under nested panel clustering.
            raw_transform_row = torch.cat((torch.ones(1, dtype=torch.float64),
                          -sample.designs["regressors"].means[sample.designs["regressors"].kept][[i+1 for i in selected]]/
                           sample.designs["regressors"].scales[sample.designs["regressors"].kept][[i+1 for i in selected]]))
            _flag_constant(notes, float(raw_cov[0, 0]), rss/df*float(raw_transform_row@bread@raw_transform_row))
    return metrics, predictions
