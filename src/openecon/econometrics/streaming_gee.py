"""Native GEE with disk group sums and bounded row replays.

Independent/exchangeable and AR(1) use bounded row tensors and disk moments.
Patterned correlations use guarded T-by-T factors and one guarded panel solve;
no complete sample or G-by-P tensor is retained.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
import sqlite3

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import cholesky_solve
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .glm.families import FAMILY_LINKS, make_family, make_link
from .mixed.gee import DEFAULT_LINK, FAMILIES, _FIXED_SCALE, _MAX_ITER, _TOL
from .mixed.xt import _TITLES, _validate
from .replay_sample import ReplaySample
from .streaming_gee_ordered import ORDERED, PATTERNED, OrderedCorrelation, OrderedPanels
from .streaming_linear import _GroupMeans, _Notes, _finite, _result, _scratch_directory


def _offset(batch, spec):
    result = torch.zeros(len(batch.frame), dtype=torch.float64)
    for name in registry.role_columns(spec, "offset"):
        result += batch.numeric(name)
    for name in registry.role_columns(spec, "exposure"):
        value = batch.numeric(name)
        if bool((value <= 0).any()):
            raise AnalysisError("invalid_exposure", "Exposure values must be positive.")
        result += value.log()
    _finite(result)
    return result


class _Equations:
    def __init__(self, sample, design, family, link, response_mean, response_scale):
        self.sample, self.design = sample, design
        self.family, self.link = family, link
        self.mean, self.scale = response_mean, response_scale
    def outcome(self, batch):
        return (batch.numeric(self.sample.spec.outcome)-self.mean)/self.scale
    def at(self, batch, beta):
        x = batch.designs["mean"]
        y = self.outcome(batch)
        eta = x@beta+_offset(batch, self.sample.spec)/self.scale
        if not bool(self.link.valid(eta).all()):
            raise KernelError("invalid_mean", "The linear predictor left the domain of the GEE link.")
        mu = self.link.inverse(eta)
        if not bool(self.family.valid_mu(mu).all()):
            raise KernelError("invalid_mean", "The GEE fitted means left the family range.")
        residual = y-mu
        if self.family.name == "binomial":
            complement = self.link.complement(eta, mu)
            variance = mu*complement
            residual = torch.where(mu>.5, (y-1)+complement, residual)
        else:
            variance = self.family.variance(mu)
        if not bool((variance>0).all()):
            raise KernelError("invalid_mean", "A GEE fitted variance is not positive.")
        root = variance.sqrt()
        pearson = residual/root
        standardized = x*(self.link.derivative(eta, mu)/root)[:, None]
        _finite(mu, pearson, standardized)
        return y, mu, pearson, standardized
    def start(self):
        width = len(self.design.terms)
        information, score = _CompensatedSum((width, width)), _CompensatedSum((width,))
        for batch in self.sample.batches():
            x, y = batch.designs["mean"], self.outcome(batch)
            mu = self.family.start_mu(y)
            eta = self.link.link(mu)
            derivative = self.link.derivative(eta, mu)
            response = eta-_offset(batch, self.sample.spec)/self.scale+(y-mu)/derivative
            weight = derivative.square()/self.family.variance(mu)
            _finite(response, weight)
            information.add(x.T@(x*weight[:, None]))
            score.add(x.T@(weight*response))
        return cholesky_solve(information.value, score.value, code="singular_information", what="GEE initial information")


def _moments(eq, beta, kind, groups, longest, *, final=False):
    width = len(beta)
    information, score = _CompensatedSum((width, width)), _CompensatedSum((width,))
    pearson, deviance = _CompensatedSum(()), _CompensatedSum(())
    predictions, store = [], _GroupMeans(width+1, 0, panel=True) if kind == "exchangeable" else None
    accumulator = None
    try:
        if final:
            accumulator = ClusterAccumulator(width, scratch_directory=_scratch_directory())
        for batch in eq.sample.batches():
            y, mu, residual, x = eq.at(batch, beta)
            information.add(x.T@x)
            score.add(x.T@residual)
            pearson.add(residual.square().sum())
            deviance.add(eq.family.unit_deviance(y, mu).sum())
            if store:
                values = torch.cat((x, residual[:, None]), 1)
                store.add(encode_cluster_labels(batch.frame[eq.sample.spec.panel]), values,
                          torch.ones(len(x), dtype=torch.float64), [])
            elif final:
                accumulator.add(encode_cluster_labels(batch.frame[eq.sample.spec.panel]), x*residual[:, None])
            if final:
                take = min(400-len(predictions), len(mu))
                fitted = eq.mean+eq.scale*mu
                for pos, observed, expected in zip(batch.positions[:take].tolist(), batch.numeric(eq.sample.spec.outcome)[:take].tolist(), fitted[:take].tolist(), strict=True):
                    predictions.append({"row": pos, "observed": observed, "fitted": expected, "residual": observed-expected})
        if not bool(torch.isfinite(pearson.value)) or float(pearson.value)<=1e-24*eq.signal:
            raise KernelError("perfect_fit", "Every GEE Pearson residual is zero; the scale and working correlation are not identified.")
        alpha = 0.
        if store:
            store.finish()
            residual_sums, pairs = _CompensatedSum(()), _CompensatedSum(())
            for means, counts, _, _ in store.group_batches(eq.sample.rows):
                totals = means*counts[:, None]
                residual_sums.add(totals[:, width].square().sum())
                pairs.add((counts*(counts-1)).sum())
            if kind == "exchangeable":
                if float(pairs.value)<=0:
                    raise KernelError("insufficient_panel_length", "Exchangeable correlation needs panels with at least two observations.")
                alpha = float((residual_sums.value-pearson.value)/pairs.value/(pearson.value/eq.sample.nobs))
                if not -1/max(longest-1, 1)<alpha<1:
                    raise KernelError("working_correlation_not_pd", "The estimated exchangeable correlation is outside its positive-definite region.")
        meat = torch.zeros((width, width), dtype=torch.float64)
        if store:
            stable_information, stable_score = _CompensatedSum((width, width)), _CompensatedSum((width,))
            for means, counts, _, _ in store.group_batches(eq.sample.rows):
                mean_x, mean_r = means[:, :width], means[:, width]
                precision = counts/(1+(counts-1)*alpha)
                stable_information.add(mean_x.T@(mean_x*precision[:, None]))
                stable_score.add(mean_x.T@(precision*mean_r))
            # Between/within decomposition avoids subtracting two nearly equal
            # N-sized matrices when correlations and panel lengths are large.
            for batch in eq.sample.batches():
                _, _, residual, x = eq.at(batch, beta)
                keys = encode_cluster_labels(batch.frame[eq.sample.spec.panel])
                unique, codes = store._codes(keys)
                block, _, _ = store._load_block(unique)
                mass, means = block[codes, 0], block[codes, 1:]
                if bool((mass<=0).any()):
                    raise AnalysisError("source_changed", "A GEE group changed between moment replays.")
                within_x, within_r = x-means[:, :width], residual-means[:, width]
                stable_information.add(within_x.T@within_x/(1-alpha))
                stable_score.add(within_x.T@within_r/(1-alpha))
                if final:
                    contribution = within_x*within_r[:, None]/(1-alpha)+means[:, :width]*(means[:, width]/(1+(mass-1)*alpha))[:, None]
                    accumulator.add(keys, contribution)
            bread, gradient = stable_information.value, stable_score.value
        else:
            bread, gradient = information.value, score.value
        if final:
            meat, actual_groups = accumulator.finish()
            if actual_groups != groups:
                raise AnalysisError("source_changed", "The final GEE score groups differ from the fitted panels.")
        _finite(bread, gradient, meat)
        return bread, gradient, float(pearson.value), float(deviance.value), alpha, meat, predictions
    finally:
        if store:
            store.close()
        if accumulator:
            accumulator.close()


def fit_pa(spec, source, *, batch_rows=None):
    notes = _Notes(spec)
    families = {"xtlogit": ("logit", "binomial", "logit"), "xtprobit": ("probit", "binomial", "probit"),
                "xtpoisson": ("poisson", "poisson", "log")}
    if spec.estimator not in families or notes.option("model") != "pa":
        raise AnalysisError("invalid_spec", "Population-averaged replay requires an XT logit/probit/Poisson PA specification.")
    family, glm_family, link = families[spec.estimator]
    _validate(notes, family, "pa", spec.estimator)
    return _entry(spec, source, batch_rows=batch_rows, family=glm_family, link=link,
                  scale=None, nmp=False, title=_TITLES[(family, "pa")])


def fit_streaming_gee(spec, source, *, batch_rows=None):
    if spec.estimator != "xtgee":
        raise AnalysisError("invalid_spec", "GEE replay requires xtgee; PA wrappers use fit_pa.")
    notes = _Notes(spec)
    family = notes.option("family")
    if "nbk" in spec.options and family != "nbinomial":
        raise AnalysisError("invalid_spec", "nbk applies to family='nbinomial' only.")
    return _entry(spec, source, batch_rows=batch_rows, family=family, link=notes.option("link"),
                  scale=notes.option("scale"), nmp=notes.option("nmp"), title="GEE population-averaged model")


def _entry(spec, source, **kwargs):
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, **kwargs)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_information", "The GEE information is not finite positive definite.") from error
    except (OSError, sqlite3.Error) as error:
        raise _GroupMeans.storage_error() from error


def _fit(spec, source, *, batch_rows, family, link, scale, nmp, title):
    notes = _Notes(spec)
    corr = notes.option("corr")
    if corr not in {"independent", "exchangeable", *ORDERED}:
        raise AnalysisError("invalid_option", "Unknown GEE working correlation.")
    if corr in ORDERED and spec.time is None:
        raise AnalysisError("invalid_spec", f"corr='{corr}' needs a time column.")
    if spec.weights or spec.weight_type:
        raise AnalysisError("invalid_spec", "Native GEE and XT PA models do not support weights.")
    if spec.covariance not in {"nonrobust", "robust"}:
        raise AnalysisError("unsupported_covariance", "GEE supports nonrobust and robust panel covariance.")
    order = notes.option("corr_order")
    if type(order) is not int or order < 1:
        raise AnalysisError("invalid_spec", "corr_order must be a positive integer.")
    if corr not in {"stationary", "nonstationary"} and order != 1:
        raise AnalysisError("invalid_spec", "corr_order applies to stationary/nonstationary correlation only.")
    link = link or DEFAULT_LINK[family]
    if link not in FAMILY_LINKS[FAMILIES[family]]:
        raise AnalysisError("invalid_spec", "This link is not available for the selected GEE family.")
    if spec.panel in spec.predictors or spec.panel == spec.outcome:
        raise AnalysisError("invalid_spec", "The panel column must differ from the outcome and regressors.")
    if scale is not None and not ((isinstance(scale, str) and scale in {"x2", "dev"}) or (type(scale) in {int, float} and math.isfinite(scale) and scale>0)):
        raise AnalysisError("invalid_spec", "scale must be None, x2, dev or a finite positive number.")
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    with ExitStack() as stack:
        panels = None
        if corr in ORDERED:
            sample.plan_rows("GEE ordered metadata discovery", {
                "ordered_metadata_and_score_SQLite_caches": 12*1024**2,
                "parameter_factors": 512*(len(design.terms)+1)**2,
            }, 512*(len(design.terms)+1))
            panels = OrderedPanels(sample)
            stack.callback(panels.close)
            panels.seed(notes, corr, notes.option("force"))
            sample = panels.select(spec, source, batch_rows, notes, corr, notes.option("corr_order"))
            design = sample.designs["mean"]
        p, n = len(design.terms), sample.nobs
        if not p or n<=p:
            raise AnalysisError("insufficient_observations", "GEE needs a nonempty identified design and more observations than parameters.")
        resource = sample.plan_rows("native independent/exchangeable GEE group moment replay", {
            "panel_metadata_SQLite_cache": 2*1024**2,
            "group_moments_SQLite_cache_and_bulk_decode": 4*1024**2,
            "final_panel_score_accumulator_cache": 6*1024**2,
            "parameter_information_gradient_and_covariance": 512*(p+1)**2,
        }, 512*(p+1))
        moments = _WeightedMoments(2, intercept=True)
        groups = _GroupMeans(1, 0, panel=True)
        stack.callback(groups.close)
        for batch in sample.batches():
            y = batch.numeric(spec.outcome)
            bad = ((y<0)|(y>1)) if family == "binomial" else y<0 if family in {"poisson", "nbinomial"} else y<=0 if family in {"gamma", "igaussian"} else torch.zeros_like(y, dtype=torch.bool)
            if bool(bad.any()):
                raise AnalysisError("invalid_outcome", "The outcome is outside the selected GEE family range.")
            _offset(batch, spec)
            moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
            groups.add(encode_cluster_labels(batch.frame[spec.panel]), torch.zeros((len(y), 1), dtype=torch.float64),
                       torch.ones_like(y), [], encode_cluster_labels(batch.frame[spec.time]) if spec.time else None)
        groups.finish()
        structure = groups.panel_structure()
        longest = int(structure["t_max"])
        ordered = None
        if panels is not None:
            reservation = {
                "ordered_and_group_SQLite_caches": 12*1024**2,
                "parameter_information_gradient_and_covariance": 512*(p+1)**2,
            }
            if corr in PATTERNED:
                reservation.update({
                    "patterned_correlation_moments_masks_and_factor_generations": 160*longest**2,
                    "single_guarded_panel_design_and_solve": 128*longest*(p+1),
                })
            resource = sample.plan_rows("native ordered/patterned GEE replay", reservation, 512*(p+1))
            panels.initialize(p)
            ordered = OrderedCorrelation(panels, corr, notes.option("corr_order"))
        if groups.groups<2:
            raise AnalysisError("insufficient_groups", "GEE needs at least two panels.")
        variation = float(moments.m2.value[1]/moments.mass)
        if variation<=0:
            raise AnalysisError("constant_outcome", "The GEE outcome has no variation.")
        normalized = family == "gaussian" and link == "identity"
        response_scale = float(moments.magnitude[1])*math.sqrt(variation) if normalized else 1.
        response_mean = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1]) if normalized and spec.intercept else 0.
        eq = _Equations(sample, design, make_family(FAMILIES[family], k=notes.option("nbk") or 1.),
                        make_link(link), response_mean, response_scale)
        raw_mean = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1])
        eq.signal = n if normalized else n*(float(moments.magnitude[1])**2*variation+raw_mean**2)
        beta = eq.start()
        for stage in ("independent", corr):
            for iteration in range(1, _MAX_ITER+1):
                bread, gradient, _, _, _, _, _ = (ordered.moments(eq, beta) if stage in ORDERED
                                                     else _moments(eq, beta, stage, groups.groups, longest))
                step = cholesky_solve(bread, gradient, code="singular_information", what="GEE information")
                beta += step
                units = design.scales[design.kept]
                if float(((step/units).abs()/(beta/units).abs().clamp_min(1.)).max())<=_TOL:
                    break
            else:
                raise AnalysisError("nonconvergence", "The GEE scoring iterations did not converge.")
        bread, _, pearson, deviance, alpha, meat, predictions = (ordered.moments(eq, beta, final=True) if ordered
                                                               else _moments(eq, beta, corr, groups.groups, longest, final=True))
        inverse = information_inverse(bread)
        physical_pearson, physical_deviance = pearson*response_scale**2, deviance*response_scale**2
        divisor = n-p if nmp else n
        phi_hat = physical_pearson/divisor
        if scale is None:
            phi = 1. if family in _FIXED_SCALE else phi_hat
            scale_text = "fixed at 1" if family in _FIXED_SCALE else "Pearson chi2 / "+("(N - p)" if nmp else "N")
        elif scale in {"x2", "dev"}:
            phi = (physical_pearson if scale == "x2" else physical_deviance)/divisor
            scale_text = ("Pearson chi2" if scale == "x2" else "deviance")+" / "+("(N - p)" if nmp else "N")
        else:
            phi, scale_text = float(scale), "user-specified"
        if spec.covariance == "nonrobust":
            covariance = inverse*(phi/response_scale**2)
            correction = "conventional: phi (sum D'V^-1 D)^-1, phi "+scale_text
        else:
            covariance = inverse@meat@inverse*groups.groups/(groups.groups-1)
            correction = "semi-robust sandwich clustered on the panels: G/(G-1) B^-1 [sum_i s_i s_i'] B^-1"
        transform = response_scale*design.transform
        raw_beta, raw_covariance = transform@beta, transform@covariance@transform.T
        if spec.intercept:
            raw_beta[0] += response_mean
        _finite(raw_beta, raw_covariance)
        metrics = {"n_groups": groups.groups, "group_size_min": structure["t_min"],
                   "group_size_avg": structure["t_avg"], "group_size_max": structure["t_max"],
                   "scale": phi, "pearson_chi2": physical_pearson, "deviance": physical_deviance}
        extra = {"family": family, "link": link, "corr": corr, "scale_method": scale_text,
                 "pearson_dispersion": phi_hat, "iterations": iteration, "nmp": nmp}
        if corr in {"stationary", "nonstationary"}:
            extra["corr_order"] = notes.option("corr_order")
        if corr in {"exchangeable", "ar1", "stationary"}:
            extra["alpha"] = alpha.tolist() if ordered else [alpha]
        if longest<=50:
            matrix = ordered.full_matrix() if ordered else torch.eye(longest, dtype=torch.float64)
            if corr == "exchangeable":
                matrix = matrix*(1-alpha)+alpha
            extra["working_correlation"] = matrix.tolist()
        else:
            extra["working_correlation_note"] = f"the {longest}-by-{longest} working correlation is not stored (more than 50 periods)"
        tests = {"model": wald_test(raw_beta, raw_covariance, range(int(spec.intercept), p), label="Wald chi2 test of the slopes")}
        info = {"covariance": spec.covariance, "correction": correction, "df_inference": None, "df_resid": None}
        algorithm = ("owned SQL neighbours and tridiagonal precision; bounded row tensors, no whole panel" if corr == "ar1"
                     else "guarded T-by-T correlation factors and one guarded panel solve" if corr in PATTERNED
                     else "bounded row replays and Sherman-Morrison group moments; no full panel/correlation matrix")
        return _result(sample, terms=design.terms, beta=raw_beta, covariance=raw_covariance, info=info,
                       metrics=metrics, notes=notes, predictions=predictions, tests=tests,
                       solver="native_GEE_Fisher_scoring_ordered_panel_replay" if ordered else "native_GEE_Fisher_scoring_disk_group_sums", diagnostics={"converged": True, "iterations": iteration,
                                  "long_panel_algorithm": algorithm,
                                  "maximum_guarded_panel_rows": panels.maximum_panel_rows if panels else 0},
                       extra=extra, resource=resource.record(), title=title, use_t=False)
