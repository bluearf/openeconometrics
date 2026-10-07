"""Native bounded replay for pooled, FD, ML and ordered panel covariance."""
from __future__ import annotations

import math
from contextlib import ExitStack

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import newey_west_lags
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_hac import HACAccumulator


class DriscollKraayAccumulator:
    """Cross-sectional score sums on disk, followed by complete ordered HAC."""
    def __init__(self, width, notes, scratch_directory, hac_budget):
        self.width, self.notes = width, notes
        self.hac_budget = hac_budget
        self.periods = ClusterAccumulator(width, scratch_directory=scratch_directory)
        self.datetime = None

    def add(self, scores, time):
        from .streaming_panelgls import _times
        periods, date = _times(time)
        if self.datetime is not None and self.datetime != date:
            raise AnalysisError("source_changed", "The DK time type changed between passes.")
        self.datetime = date
        keys = [(period+(1<<63)).to_bytes(8, "big") for period in periods]
        self.periods.add(keys, scores)

    def finish(self):
        try:
            _, count = self.periods.finish()
        except KernelError as error:
            if error.code == "insufficient_clusters":
                raise AnalysisError("insufficient_periods", "Driscoll-Kraay covariance needs at least two time periods.") from error
            raise
        lags = self.notes.option("lags")
        lags = newey_west_lags(count) if lags is None else int(lags)
        kernel = self.notes.option("kernel") or "bartlett"
        if kernel == "truncated":
            low, high = self.periods._connection.execute("SELECT MIN(key),MAX(key) FROM scores").fetchone()
            span = count-1 if self.datetime else int.from_bytes(high, "big")-int.from_bytes(low, "big")
            if lags >= span:
                raise AnalysisError("invalid_covariance", "A truncated Driscoll-Kraay window covering every period gives zero least-squares score covariance. Use a shorter lag window or another kernel.")
        with HACAccumulator(self.width, lags, kernel, budget_bytes=self.hac_budget) as accumulator:
            cursor = self.periods._connection.execute("SELECT key,total,correction FROM scores ORDER BY key")
            position = 0
            while records := cursor.fetchmany(4096):
                times = [int.from_bytes(row[0], "big")-(1<<63) for row in records]
                scores = torch.stack([self.periods._decode(row[1])+self.periods._decode(row[2]) for row in records])
                series = pd.Series(pd.to_datetime(times)) if self.datetime else pd.Series(times, dtype="int64")
                accumulator.add(scores, torch.arange(position, position+len(records)), time=series)
                position += len(records)
            matrix = accumulator.finish()
            info = {**accumulator.diagnostics, "periods": count, "lags": lags, "kernel": kernel,
                    "cross_section_aggregation": self.periods.diagnostics}
        return matrix, info

    def close(self):
        self.periods.close()


def fit_streaming_panel_options(spec, source, *, batch_rows=None):
    from .streaming_linear import _Notes
    from .panel.xtreg import _validate
    notes = _Notes(spec)
    model = notes.option("model")
    _validate(notes, model)
    if model == "pooled":
        return _pooled(spec, source, notes, batch_rows)
    if model == "fd":
        return _fd(spec, source, notes, batch_rows)
    if model == "mle":
        return _mle(spec, source, notes, batch_rows)
    raise AnalysisError("invalid_spec", "This native panel replay kernel requires pooled, FD or ML.")


def _pooled(spec, source, notes, batch_rows):
    from openecon.streaming_design import encode_cluster_labels
    from . import registry
    from .core import wald_test
    from .linear.common import check_pweights
    from .panel.common import require_residual_variation
    from .replay_sample import ReplaySample
    from .streaming_fe_iv import _response
    from .streaming_linear import _GroupMeans, _covariance, _factor, _finite, _period_keys, _result, _solve, _weights
    check_pweights(notes)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean", intercept=True)
    sample.prepare()
    k, n = len(design.terms), sample.nobs
    df = n-k
    if df <= 0:
        raise AnalysisError("insufficient_observations", "Pooled panel regression has no residual degrees of freedom.")
    clusters = ([spec.panel] if spec.covariance == "robust" else
                registry.cluster_columns(spec) if spec.covariance == "cluster" else [])
    resource = sample.plan_rows("pooled panel TSQR, disk metadata and ordered covariance", {
        "panel_metadata_SQLite_cache": 6*1024**2,
        "cluster_score_SQLite_caches": (2**len(clusters)-1)*6*1024**2,
        "small_global_factor_covariance_and_decode": 4096*(k+2)**2,
    }, 1024*(k+2))
    response, center, scale = _response(sample)
    with ExitStack() as stack:
        store = _GroupMeans(1, 0, panel=True)
        stack.callback(store.close)
        def factor_rows():
            for batch in sample.batches():
                yield batch.designs["mean"], response(batch), _weights(sample, batch)
        factor, diagnostics = _factor(factor_rows, k)
        beta, bread, condition = _solve(factor, k)
        def score_rows():
            for batch in sample.batches():
                store.add(encode_cluster_labels(batch.frame[spec.panel]),
                          torch.zeros((len(batch.frame), 1), dtype=torch.float64), _weights(sample, batch), [],
                          _period_keys(batch.frame[spec.time]) if spec.time else None)
                yield batch, batch.designs["mean"], response(batch), batch.numeric(spec.outcome), scale
        covariance, info, rss_work, predictions = _covariance(sample, score_rows, beta, bread,
            df=df, k=k, notes=notes, score_basis=torch.linalg.inv(design.transform),
            kind="cluster" if clusters else spec.covariance, cluster_columns=clusters)
        store.finish()
        structure = store.panel_structure(frequency=spec.weight_type == "fweight")
        rss, tss = rss_work*scale*scale, n*scale*scale
        require_residual_variation(rss, tss, tss+n*center*center, spec.outcome, "the pooled sample")
        raw_beta = scale*(design.transform@beta)
        raw_beta[0] += center
        covariance = scale*scale*(design.transform@covariance@design.transform.T)
        _finite(raw_beta, covariance)
        info.update({"covariance": spec.covariance, "df_resid": df})
        if spec.covariance == "robust":
            info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        r2 = 1-rss/tss
        metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(n-1)/df,
                   "rmse": math.sqrt(rss/df), "df_model": k-1, "df_resid": df, **structure}
        tests = {"model": wald_test(raw_beta, covariance, range(1, k),
                         df_resid=info["df_inference"], label="F test of the slopes")}
        return _result(sample, terms=design.terms, beta=raw_beta, covariance=covariance,
             info=info, metrics=metrics, notes=notes, predictions=predictions, tests=tests,
             solver="native_pooled_panel_TSQR_score_replay", diagnostics={**diagnostics, **store.diagnostics,
                         "condition_number": condition}, extra={"model": "pooled", "ssr": rss},
             resource=resource.record(), title="Pooled OLS regression", provenance_extra={"model": "pooled"})


def _fd(spec, source, notes, batch_rows):
    import hashlib
    from types import SimpleNamespace
    from openecon.engines.linalg import collinear_columns
    from . import registry
    from .core import wald_test
    from .linear.common import check_pweights
    from .panel.common import require_residual_variation
    from openecon.engines.streaming_ols import _CompensatedSum
    from .replay_sample import ReplaySample
    from .streaming_fd_iv import _Differences
    from .streaming_fe_iv import _response
    from .streaming_linear import _CrossMoments, _covariance, _factor, _finite, _result, _solve, _weights
    check_pweights(notes)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("regressors", intercept=True)
    sample.prepare()
    sample.designs["instruments"] = design
    k0 = len(design.terms)
    response, center, scale = _response(sample)
    clusters = ([spec.panel] if spec.covariance == "robust" else
                registry.cluster_columns(spec) if spec.covariance == "cluster" else [])
    resource = sample.plan_rows("first differences with disk ordering and global TSQR", {
        "ordered_source_snapshot_and_decode": 4*1024**2,
        "cluster_score_SQLite_caches": (2**len(clusters)-1)*6*1024**2,
        "small_global_factor_covariance": 4096*(k0+2)**2,
    }, 1024*(k0+2))
    with ExitStack() as stack:
        states = _Differences(sample, response, k0, clusters)
        stack.callback(states.close)
        moment = _CrossMoments(1)
        level_squares = _CompensatedSum(())
        def factor_rows():
            for batch, values in states.batches():
                weights = _weights(sample, batch)
                moment.add(values[:, :1], weights)
                # Dense FD judges cancellation against the retained outcome
                # levels, not differences after subtracting a grand mean.
                level_squares.add((weights*(batch.level_response_work+center/scale).square()).sum())
                yield torch.cat((torch.ones((len(values), 1), dtype=torch.float64), values[:, 1:]), 1), values[:, 0], weights
        factor, diagnostics = _factor(factor_rows, k0, require_more=False)
        screened = factor[:, :k0].clone()
        screened[:, 1:] -= screened[:, :1]*((screened[:, 0]@screened[:, 1:])/(screened[:, 0]@screened[:, 0]))
        kept, omitted = collinear_columns(screened)
        if not kept or kept[0] != 0:
            raise AnalysisError("singular_design", "The differenced regression requires an identified drift.")
        for index in omitted:
            if design.terms[index] not in sample.notes["omitted_terms"]:
                sample.notes["omitted_terms"].append(design.terms[index])
                notes.warn(f"Omitted because of collinearity after differencing: {design.terms[index]}.")
        k, n = len(kept), states.nobs
        df = n-k
        if df <= 0:
            raise AnalysisError("insufficient_observations", "First differences leave no residual degrees of freedom.")
        beta, bread, condition = _solve(torch.cat((factor[:, kept], factor[:, -1:]), 1), k)
        rows = SimpleNamespace(spec=spec, nobs=n, weight_mean=sample.weight_mean)
        def scores():
            for batch, values in states.batches():
                x = torch.cat((torch.ones((len(values), 1), dtype=torch.float64), values[:, 1:]), 1)[:, kept]
                yield batch, x, values[:, 0], scale*values[:, 0], scale
        scales = design.scales[design.kept][kept]
        transform = torch.diag(1/scales)
        covariance, info, rss, predictions = _covariance(rows, scores, beta, bread, df=df, k=k,
                notes=notes, kind="cluster" if clusters else "nonrobust", cluster_columns=clusters,
                score_basis=torch.linalg.inv(transform))
        tss = float(moment.m2.value[0, 0])
        require_residual_variation(rss, tss, float(level_squares.value), spec.outcome, "the first differences")
        raw_beta = scale*(transform@beta)
        covariance = scale*scale*(transform@covariance@transform.T)
        _finite(raw_beta, covariance)
        info.update({"covariance": spec.covariance, "df_resid": df,
                     "residual_definition": "differenced outcome minus fitted"})
        if spec.covariance == "robust":
            info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        r2 = 1-rss/tss
        metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(n-1)/df,
                   "rmse": scale*math.sqrt(rss/df), "df_model": k-1, "df_resid": df, **states.structure}
        tests = {"model": wald_test(raw_beta, covariance, range(1, k), df_resid=info["df_inference"],
                                    label="F test of the slopes")}
        notes.warn(f"Dropped {sample.nrows-states.kept} observation(s) without an observation in the previous period: first differences use consecutive periods.")
        for _ in sample.batches():
            pass
        result = _result(sample, terms=[design.terms[index] for index in kept], beta=raw_beta,
             covariance=covariance, info=info, metrics=metrics, notes=notes, predictions=predictions,
             tests=tests, solver="native_first_difference_TSQR_score_replay",
             diagnostics={**diagnostics, "condition_number": condition, "ordered_snapshot_passes": states.passes,
                          "maximum_snapshot_rows": states.peak_rows},
             extra={"model": "fd", "ssr": scale*scale*rss, "differenced_observations": states.kept,
                    "dropped_for_differencing": sample.nrows-states.kept}, resource=resource.record(),
             title="First-difference regression", provenance_extra={"model": "fd", "sample_position_count": states.kept,
                "sample_positions_hash": states.positions_hash, "sample_order": "panel key then exact time",
                "sample_hash": hashlib.sha256((sample.baseline["data_hash"]+states.positions_hash+"first_difference").encode()).hexdigest(),
                "scratch_disk_plan": states.disk_plan})
        result.nobs = n
        result.dropped_rows = sample.original_count-states.kept
        return result


class _DiskRandomEffectsLikelihood:
    """Exact Gaussian likelihood from a small within factor and disk means."""
    def __init__(self, sample, store, within, columns):
        self.sample, self.store, self.columns = sample, store, columns
        indices = [*columns, within.shape[1]-1]
        values = within[:, indices]
        self.gram = values.T@values
        self.width = len(columns)

    def value(self, theta):
        return self._evaluate(theta, False)[0]

    def __call__(self, theta):
        return self._evaluate(theta, True)

    def _evaluate(self, theta, derivatives):
        from openecon.engines.streaming_ols import _CompensatedSum
        beta, su, se = theta[:-2], torch.exp(2*theta[-2]), torch.exp(2*theta[-1])
        vector = torch.cat((-beta, torch.ones(1, dtype=torch.float64)))
        within = (vector@self.gram@vector).clamp_min(0)
        value = _CompensatedSum(())
        value.add(-.5*self.sample.nobs*math.log(2*math.pi)-.5*(self.sample.nobs-self.store.groups)*torch.log(se)-within/(2*se))
        if derivatives:
            width = self.width
            gradient = _CompensatedSum((width+2,))
            hessian = _CompensatedSum((width+2, width+2))
            within_xr = self.gram[:-1, -1]-self.gram[:-1, :-1]@beta
            initial_gradient = torch.cat((within_xr/se, torch.zeros(1, dtype=torch.float64),
                   (-(self.sample.nobs-self.store.groups)+within/se).reshape(1)))
            initial_hessian = torch.zeros((width+2, width+2), dtype=torch.float64)
            initial_hessian[:width, :width] = -self.gram[:-1, :-1]/se
            initial_hessian[:width, -1] = initial_hessian[-1, :width] = -2*within_xr/se
            initial_hessian[-1, -1] = -2*within/se
            gradient.add(initial_gradient)
            hessian.add(initial_hessian)
        for means, sizes, _, _ in self.store.group_batches(self.sample.rows):
            x, y = means[:, self.columns], means[:, -1]
            residual = y-x@beta
            sums = sizes*residual
            squares = sums.square()
            a = se+sizes*su
            inv, inv2, inv3 = a.reciprocal(), a.reciprocal().square(), a.reciprocal().pow(3)
            value.add((-.5*torch.log(a)-squares/(2*sizes*a)).sum())
            if not derivatives:
                continue
            gb = x.T@(sums*inv)
            gu = su*(-sizes*inv+squares*inv2).sum()
            ge = (-se*inv+se*squares/sizes*inv2).sum()
            gradient.add(torch.cat((gb, gu.reshape(1), ge.reshape(1))))
            h = torch.zeros((width+2, width+2), dtype=torch.float64)
            h[:width, :width] = -(x*(sizes*inv)[:, None]).T@x
            h[:width, -2] = h[-2, :width] = -2*su*(x.T@(sizes*sums*inv2))
            h[:width, -1] = h[-1, :width] = -2*se*(x.T@(sums*inv2))
            h[-2, -2] = 2*gu+2*su.square()*(sizes.square()*inv2-2*sizes*squares*inv3).sum()
            h[-1, -1] = (-2*se*inv+2*se.square()*inv2+2*se*squares/sizes*inv2-4*se.square()*squares/sizes*inv3).sum()
            h[-2, -1] = h[-1, -2] = 2*su*(se*sizes*inv2-2*se*squares*inv3).sum()
            hessian.add(h)
        if not derivatives:
            return (value.value,)
        return value.value, gradient.value, (hessian.value+hessian.value.T)/2


def _mle(spec, source, notes, batch_rows):
    from openecon.streaming_design import encode_cluster_labels
    from openecon.engines.distributions import chi2_sf
    from openecon.engines.inference import critical_value
    from openecon.engines.optimize import information_inverse
    from .core import information_criteria, kernel_call, lr_test
    from .panel.kernels import ols_log_likelihood
    from .panel.mle import _maximize
    from .replay_sample import ReplaySample
    from .streaming_fe_iv import _response
    from .streaming_linear import _GroupMeans, _factor, _finite, _period_keys, _result, _solve
    from .streaming_re import fit_streaming_re
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean", intercept=True)
    sample.prepare()
    p, n = len(design.terms), sample.nobs
    if n <= p+2:
        raise AnalysisError("insufficient_observations", "The random-effects likelihood needs more observations than parameters.")
    resource = sample.plan_rows("random-effects ML small within geometry and disk Gaussian panel moments", {
        "panel_moments_SQLite_cache": 6*1024**2,
        "small_joint_factor_and_information": 6144*(p+3)**2,
    }, 2048*(p+3))
    response, center, scale = _response(sample)
    re_spec = spec.model_copy(update={"options": {**spec.options, "model": "re"}})
    initial = fit_streaming_re(re_spec, source, batch_rows=batch_rows)
    raw_start = torch.tensor([entry.estimate for entry in initial.coefficients], dtype=torch.float64)
    raw_start[0] -= center
    beta_start = torch.linalg.solve(design.transform, raw_start/scale)
    sigma_e2 = (initial.metrics["sigma_e"]/scale)**2
    sigma_u2 = max((initial.metrics["sigma_u"]/scale)**2, 1e-2*sigma_e2)
    starting_sigma = torch.tensor([.5*math.log(sigma_u2), .5*math.log(sigma_e2)], dtype=torch.float64)
    with ExitStack() as stack:
        store = _GroupMeans(p+1, 0, panel=True)
        stack.callback(store.close)
        def pooled_rows():
            for batch in sample.batches():
                y, x = response(batch), batch.designs["mean"]
                store.add(encode_cluster_labels(batch.frame[spec.panel]), torch.cat((x, y[:, None]), 1),
                          batch.weights, [], _period_keys(batch.frame[spec.time]) if spec.time else None)
                yield x, y, batch.weights
        pooled, diagnostics = _factor(pooled_rows, p)
        store.finish()
        structure = store.panel_structure()
        def within_rows():
            for batch in sample.batches():
                values = torch.cat((batch.designs["mean"], response(batch)[:, None]), 1)
                values -= store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
                yield values[:, :p], values[:, p], batch.weights
        within, within_diagnostics = _factor(within_rows, p, require_more=False)
        like = _DiskRandomEffectsLikelihood(sample, store, within, list(range(p)))
        fitted = _maximize(like, torch.cat((beta_start, starting_sigma)), "random-effects")
        null = _DiskRandomEffectsLikelihood(sample, store, within, [0])
        restricted = _maximize(null, torch.cat((torch.zeros(1, dtype=torch.float64), fitted.theta[-2:])),
                               "constant-only random-effects")
        jacobian = torch.eye(p+2, dtype=torch.float64)
        jacobian[:p, :p] = scale*design.transform
        shift = torch.zeros(p+2, dtype=torch.float64)
        shift[0], shift[p:] = center, math.log(scale)
        theta = jacobian@fitted.theta+shift
        covariance_log = jacobian@kernel_call(information_inverse, -fitted.hessian)@jacobian.T
        su, se = float(theta[p].exp()), float(theta[p+1].exp())
        delta = torch.ones(p+2, dtype=torch.float64)
        delta[p], delta[p+1] = su, se
        covariance = covariance_log*delta[:, None]*delta
        params = torch.cat((theta[:p], torch.tensor([su, se], dtype=torch.float64)))
        _finite(params, covariance)
        offset = n*math.log(scale)
        ll = float(fitted.value)-offset
        pooled_beta, _, _ = _solve(pooled, p)
        pooled_ssr = float((pooled[:, -1]-pooled[:, :p]@pooled_beta).square().sum())
        ll_ols = kernel_call(ols_log_likelihood, pooled_ssr, n)-offset
        lr_sigma = max(0., 2*(ll-ll_ols))
        tests = {"model": lr_test(ll, float(restricted.value)-offset, p-1, label="LR chi2 test of the slopes"),
                 "sigma_u": {"statistic": lr_sigma, "df": 1, "p_value": .5*chi2_sf(lr_sigma, 1), "distribution": "chibar2",
                   "label": "LR test of sigma_u = 0 against pooled OLS (chibar2(01): chi2(1) tail halved)"}}
        metrics = {"sigma_u": su, "sigma_e": se, "rho": su*su/(su*su+se*se),
                   **information_criteria(ll, p+2, n), **structure}
        se_log = covariance_log.diagonal()[p:].sqrt()
        raw_scales = design.scales[design.kept][1:]
        extra = {"model": "mle", "ln_sigma": {"sigma_u": {"estimate": float(theta[p]), "std_error": float(se_log[0])},
                   "sigma_e": {"estimate": float(theta[p+1]), "std_error": float(se_log[1])}},
                 "null_log_likelihood": float(restricted.value)-offset,
                 "starting_values": {"sigma_u": scale*math.sqrt(sigma_u2), "sigma_e": scale*math.sqrt(sigma_e2),
                                     "method": "swamy_arora_gls"},
                 "conditioning": {"outcome_scale": scale, "regressor_scale_min": float(raw_scales.min()),
                                  "regressor_scale_max": float(raw_scales.max())}}
        predictions = []
        for batch in sample.batches():
            take = min(400-len(predictions), len(batch.frame))
            predicted = center+scale*(batch.designs["mean"]@fitted.theta[:p])
            observed = batch.numeric(spec.outcome)
            for row, actual, estimate in zip(batch.positions[:take].tolist(), observed[:take].tolist(), predicted[:take].tolist(), strict=True):
                predictions.append({"row": row, "observed": actual, "fitted": estimate, "residual": actual-estimate})
        info = {"covariance": "nonrobust", "df_inference": None, "df_resid": n-p,
                "correction": "observed information (OIM); /sigma_u and /sigma_e by the delta method from ln sigma, confidence intervals exp(ln sigma -/+ z se_ln)"}
        result = _result(sample, terms=[*design.terms, "/sigma_u", "/sigma_e"], beta=params, covariance=covariance,
            info=info, metrics=metrics, notes=notes, predictions=predictions, tests=tests,
            solver="native_random_effects_ML_disk_exact_Gaussian_moments", diagnostics={**diagnostics, **within_diagnostics,
                  **store.diagnostics, "converged": True, "iterations": fitted.iterations}, extra=extra,
            resource=resource.record(), title="Random-effects ML regression", use_t=False,
            provenance_extra={"model": "mle", "optimizer": {"method": fitted.method, "iterations": fitted.iterations,
                "converged": fitted.converged, "gradient_max": fitted.diagnostics.get("gradient_max"),
                "message": fitted.diagnostics.get("message")}})
        critical = critical_value(spec.alpha, None)
        for entry in result.coefficients[:p]:
            entry.equation = spec.outcome
        for position, sigma, error in ((p, su, float(se_log[0])), (p+1, se, float(se_log[1]))):
            entry = result.coefficients[position]
            entry.equation = None
            entry.ci_low, entry.ci_high = sigma*math.exp(-critical*error), sigma*math.exp(critical*error)
        return result
