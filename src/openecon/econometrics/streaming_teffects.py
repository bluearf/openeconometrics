"""Full-sample replayed RA/IPW/IPWRA/AIPW with native stacked covariance."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.streaming_design import encode_cluster_labels
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.models import ModelSpec, ResultBundle
from . import registry
from .core import ModelFrame, build_result
from .replay_sample import ReplayBatch, ReplaySample
from .streaming_likelihood import ReplayObjective, _binary_certificate, _option, _role
from .teffects.common import Treatment, coefficient_table
from .teffects.estimators import _OWNERS, _ROLE_OWNERS
from .teffects.index import IndexObjective, mean_pieces, residual_pieces, treatment_objective, treatment_pieces
from .teffects.stacked import _Equation, _Layout, _ipw_weights, _jacobian, _scores, METHOD_TITLES

_METHODS = frozenset({'ra', 'ipw', 'ipwra', 'aipw', 'nnmatch', 'psmatch'})
_TMODELS = {'ipw', 'ipwra', 'aipw', 'psmatch'}
_OMODELS = {'ra', 'ipwra', 'aipw'}


def supports(spec: ModelSpec) -> bool:
    return isinstance(spec, ModelSpec) and spec.estimator == 'teffects'


@dataclass
class Nuisance:
    treatment: Tensor | None
    outcomes: list[Tensor]


def _codes(sample: ReplaySample, batch: ReplayBatch) -> Tensor:
    codes = pd.Categorical(batch.frame[_role(sample.spec, 'treatment')[0]], categories=sample.notes['treatment_labels']).codes
    if bool((codes < 0).any()):
        raise AnalysisError('source_changed', 'A treatment label changed after global discovery.')
    return torch.as_tensor(codes.copy(), dtype=torch.int64)


def _weights(sample, batch):
    return batch.weights/sample.weight_mean if sample.spec.weight_type == 'pweight' else batch.weights


def _create(spec, source, batch_rows):
    method = _option(spec, 'method')
    if (_option(spec, 'tlevel') is not None or _role(spec, 'ematch') or
            _option(spec, 'matching_vce') != 'robust'):
        raise AnalysisError('streaming_options_unsupported', 'tlevel, exact matching and iid matching variance currently validate resident tables only.')
    if method not in _METHODS:
        raise AnalysisError('invalid_spec', 'Unknown treatment-effects method.')
    matching = method in {'nnmatch', 'psmatch'}
    if matching and (spec.covariance != 'robust' or spec.weights is not None):
        raise AnalysisError('unsupported_weights' if spec.weights is not None else 'unsupported_covariance', 'Matching needs unweighted observations and Abadie-Imbens robust covariance.')
    if matching and _option(spec, 'estimand') == 'pomeans':
        raise AnalysisError('unsupported_estimand', 'Matching estimates ate or atet.')
    if method == 'psmatch' and _option(spec, 'vce_neighbors') < 2:
        raise AnalysisError('invalid_spec', 'psmatch needs vce_neighbors >= 2 for its propensity-score covariance adjustment.')
    for option, owners in _OWNERS.items():
        if option in spec.options and method not in owners:
            raise AnalysisError('invalid_spec', f"Option '{option}' does not apply to method='{method}'.")
    for role, owners in _ROLE_OWNERS.items():
        if _role(spec, role) and method not in owners:
            raise AnalysisError('invalid_spec', f"Role '{role}' does not apply to method='{method}'.")
    treatment = _role(spec, 'treatment')[0]
    if treatment == spec.outcome or treatment in [*spec.predictors, *_role(spec, 'tx')]:
        raise AnalysisError('invalid_spec', 'Treatment must differ from the outcome and all model covariates.')
    if spec.covariance not in {'robust', 'cluster'}:
        raise AnalysisError('unsupported_covariance', 'Treatment effects need robust or clustered stacked covariance.')
    if spec.covariance == 'cluster' and len(registry.cluster_columns(spec)) != 1:
        raise AnalysisError('cluster_dimensions', 'Treatment effects support one cluster dimension.')
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    # Reserve the same bounded category metadata used by global design work,
    # including treatment labels that are not themselves covariates.
    sample._levels.setdefault(treatment, {})
    observed = {}
    for frame, _, _ in sample._raw(discovery=True):
        for value in frame[treatment].unique():
            sample._level(treatment, value, observed)
    try:
        declared = sample._declarations[treatment]
        labels = [v for v in declared[0] if v in observed.values()] if declared else sorted(observed.values())
    except TypeError as exc:
        raise AnalysisError('invalid_treatment', 'Treatment levels must use one sortable label type.') from exc
    if len(labels) < 2:
        raise AnalysisError('invalid_treatment', 'Treatment effects need at least two retained treatment levels.')
    from .discrete.common import label_text
    control = _option(spec, 'control')
    if control is not None:
        matches = [v for v in labels if v == control or label_text(v) == label_text(control)]
        if not matches:
            raise AnalysisError('invalid_treatment', 'The requested control is absent from retained treatment levels.')
        labels.remove(matches[0])
        labels.insert(0, matches[0])
    if matching and len(labels) != 2:
        raise AnalysisError('invalid_treatment', 'Matching needs binary treatment.')
    if _option(spec, 'estimand') == 'atet' and (len(labels) != 2 or method == 'aipw'):
        raise AnalysisError('unsupported_estimand', 'ATET needs binary treatment and ra/ipw/ipwra.')
    if len(labels) > 2 and method in _TMODELS and _option(spec, 'tmodel') != 'logit':
        raise AnalysisError('unsupported_model', 'Multivalued treatment requires multinomial logit.')
    sample.notes['treatment_labels'] = labels
    if method in _TMODELS:
        sample.add_design('treatment', _role(spec, 'tx') or spec.predictors, intercept=True, prefix='TME:')
    if method == 'nnmatch':
        sample.add_design('matching', spec.predictors, intercept=True)
    if matching and _role(spec, 'biasadj'):
        sample.add_design('bias', _role(spec, 'biasadj'), intercept=True)
    if method in _OMODELS:
        for i, label in enumerate(labels):
            sample.add_design(f'outcome{i}', spec.predictors, prefix=f'OME{label_text(label)}:',
                              selector=lambda frame, label=label: torch.as_tensor((frame[treatment] == label).to_numpy(), dtype=torch.bool))
    sample.prepare()
    if matching:
        name = 'matching' if method == 'nnmatch' else 'treatment'
        if len(sample.designs[name].terms) < 2:
            raise AnalysisError('invalid_spec', 'Matching needs a non-constant identified covariate.')
    levels = len(labels)
    width = sum(len(design.terms) for design in sample.designs.values())
    if method in _TMODELS:
        width += len(sample.designs['treatment'].terms)*(levels-2)
    size = width+levels
    if size > 384 or sample.nobs <= size:
        raise AnalysisError('model_too_wide', 'The replay stacked model needs at most384 parameters and more observations than parameters.')
    sample.plan_rows('replay treatment-effects stacked equations', {'stacked_derivatives_and_covariance': 128*size*size,
                     'cluster_sqlite_and_cache': 6*1024**2 if spec.covariance == 'cluster' else 0},
                     128*(size+levels*levels+sum(len(d.terms) for d in sample.designs.values())+4))
    sample.notes['stacked_size'] = size
    counts = torch.zeros(levels, dtype=torch.int64)
    shares = _CompensatedSum((levels,))
    omodel = _option(spec, 'omodel')
    for batch in sample.batches():
        codes = _codes(sample, batch)
        y = batch.numeric(spec.outcome)
        if not matching and omodel in {'logit', 'probit'} and bool(((y < 0)|(y > 1)).any()):
            raise AnalysisError('invalid_outcome', 'Fractional/binary outcome models need outcomes in[0,1].')
        if not matching and omodel == 'poisson' and bool((y < 0).any()):
            raise AnalysisError('invalid_outcome', 'Poisson outcome models need nonnegative outcomes.')
        counts += torch.bincount(codes, minlength=levels)
        shares.add(torch.bincount(codes, weights=_weights(sample, batch), minlength=levels))
    for i in range(levels):
        if method in _OMODELS and counts[i] <= len(sample.designs[f'outcome{i}'].terms):
            raise AnalysisError('insufficient_observations', 'Every treatment level needs more rows than its identified outcome parameters.')
    sample.notes.update({'treatment_counts': counts.tolist(), 'treatment_shares': shares.value})
    return sample


def _fit_objective(sample, builder, start, what):
    objective = ReplayObjective(sample, builder, len(start))
    scale = 1/sample.nrows
    run = optimize.maximize_newton(lambda theta: tuple(v*scale for v in objective(theta)), start,
                                  value_fn=lambda theta: objective.value(theta)*scale,
                                  max_iter=200, step_tol=1e-10, scaled_gradient_tol=1e-10, raise_on_failure=False)
    if not run.converged:
        raise AnalysisError('nonconvergence', f'The global {what} did not converge: {run.diagnostics.get("message")}')
    if not bool(torch.isfinite(run.theta).all()):
        raise AnalysisError('non_finite_result', f'The global {what} has no finite parameters.')
    return run.theta


def _treatment(sample, gamma, batch):
    if gamma is None:
        return None
    return treatment_pieces(_option(sample.spec, 'tmodel'), batch.designs['treatment'], _codes(sample, batch),
                            len(sample.notes['treatment_labels']), gamma, _weights(sample, batch))


def _nuisance(sample):
    spec, labels = sample.spec, sample.notes['treatment_labels']
    method, omodel, tmodel = (_option(spec, key) for key in ['method', 'omodel', 'tmodel'])
    levels, gamma, outcomes = len(labels), None, []
    if method in _TMODELS:
        kt = len(sample.designs['treatment'].terms)
        if levels == 2:
            try:
                _binary_certificate(sample, 'treatment', lambda b: (_codes(sample, b) == 1).to(torch.float64))
            except KernelError as exc:
                if exc.code == 'separation_detected':
                    raise AnalysisError('overlap_violation', 'The global treatment design separates treatment levels.') from exc
                raise
        def builder(batch):
            return treatment_objective(tmodel, batch.designs['treatment'], _codes(sample, batch), levels, _weights(sample, batch))
        start = torch.zeros(kt*(levels-1), dtype=torch.float64)
        shares = sample.notes['treatment_shares']
        if levels == 2 and tmodel == 'probit':
            start[0] = torch.special.ndtri(shares[1]/shares.sum())
        else:
            start.view(levels-1, kt)[:, 0] = torch.log(shares[1:]/shares[0])
        gamma = _fit_objective(sample, builder, start, 'treatment model').reshape(levels-1, kt)
        for batch in sample.batches():
            fit = _treatment(sample, gamma, batch)
            if bool((fit.probabilities < _option(spec, 'pstolerance')).any()):
                raise AnalysisError('overlap_violation', 'A global fitted propensity lies below pstolerance.')
    if method in _OMODELS:
        for level in range(levels):
            name = f'outcome{level}'
            k = len(sample.designs[name].terms)
            if not k:
                raise AnalysisError('invalid_spec', 'Every outcome equation needs an identified term.')
            if omodel in {'logit', 'probit'}:
                _binary_certificate(sample, name, lambda b: b.numeric(spec.outcome), lambda b, level=level: _codes(sample, b) == level)
            def builder(batch, level=level, name=name):
                chosen = _codes(sample, batch) == level
                w = _weights(sample, batch)
                if method == 'ipwra':
                    w = w*_ipw_weights(_treatment(sample, gamma, batch), _option(spec, 'estimand'), levels)[0][:, level]
                return IndexObjective(omodel, batch.designs[name][chosen], batch.numeric(spec.outcome)[chosen], w[chosen])
            total, response = _CompensatedSum(()), _CompensatedSum(())
            tree = _TSQRTree()
            for batch in sample.batches():
                obj = builder(batch)
                total.add(obj.w.sum())
                response.add(obj.w@obj.y)
                if omodel == 'linear' and len(obj.y):
                    tree.add(qr_factor(torch.cat((obj.x, obj.y[:, None]), 1)*obj.w.sqrt()[:, None]))
            mean = float(response.value/total.value)
            if omodel in {'logit', 'probit'} and not 0 < mean < 1 or omodel == 'poisson' and mean <= 0:
                raise AnalysisError('separation_detected', 'A nuisance outcome is constant at its identifying boundary.')
            if omodel == 'linear':
                factor = tree.finish()
                beta = least_squares(factor[:, :k], factor[:, k], drop_collinear=False).beta
            else:
                start = torch.zeros(k, dtype=torch.float64)
                if sample.designs[name].intercept:
                    start[0] = math.log(mean/(1-mean)) if omodel == 'logit' else (
                        torch.special.ndtri(torch.tensor(mean, dtype=torch.float64)) if omodel == 'probit' else math.log(mean))
                beta = _fit_objective(sample, builder, start, f'outcome model of treatment level{level}')
            outcomes.append(beta)
    return Nuisance(gamma, outcomes)


def _pieces(sample, nuisance, batch):
    spec = sample.spec
    method, estimand, omodel = (_option(spec, key) for key in ['method', 'estimand', 'omodel'])
    codes, levels = _codes(sample, batch), len(sample.notes['treatment_labels'])
    y, w = batch.numeric(spec.outcome), _weights(sample, batch)
    indicators = [(codes == level).to(torch.float64) for level in range(levels)]
    fit = _treatment(sample, nuisance.treatment, batch)
    if fit is None:
        omega, slopes = torch.ones((len(y), levels), dtype=torch.float64), None
    else:
        omega, slopes = _ipw_weights(fit, estimand, levels)
    equations = []
    for i, beta in enumerate(nuisance.outcomes):
        design = sample.designs[f'outcome{i}']
        x = batch.designs[f'outcome{i}']
        mu, dmu = mean_pieces(omodel, x@beta)
        r, dr = residual_pieces(omodel, y, x@beta)
        if not all(bool(torch.isfinite(v).all()) for v in [mu, dmu, r, dr]):
            raise AnalysisError('precision_unsupported', 'A nuisance outcome score is outside finite float64 precision.')
        equations.append(_Equation(design.terms, torch.zeros(len(beta), dtype=torch.float64), x, beta, mu, dmu, r, dr))
    layout = _Layout(batch.designs.get('treatment'), fit, equations, levels)
    selection = indicators[1] if estimand == 'atet' else torch.ones_like(y)
    return layout, y, w, indicators, omega, slopes, selection


def fit_streaming(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None) -> ResultBundle:
    """Global supported treatment-effect estimate without collecting the Dataset."""
    if not supports(spec):
        raise AnalysisError('streaming_unsupported', 'This replay adapter fits teffects.')
    try:
        with torch.no_grad(), torch.device('cpu'):
            sample = _create(spec, source, batch_rows)
            nuisance = _nuisance(sample)
            method, estimand = _option(spec, 'method'), _option(spec, 'estimand')
            if method in {'nnmatch', 'psmatch'}:
                from .streaming_matching import fit_matching
                return fit_matching(sample, nuisance)
            levels = len(sample.notes['treatment_labels'])
            numerator, denominator = _CompensatedSum((levels,)), _CompensatedSum((levels,))
            likelihood = _CompensatedSum(())
            minima, maxima = torch.ones(levels, dtype=torch.float64), torch.zeros(levels, dtype=torch.float64)
            propensity_means = _CompensatedSum((levels,))
            mass = _CompensatedSum(())
            for batch in sample.batches():
                layout, y, w, ind, omega, _, selection = _pieces(sample, nuisance, batch)
                if layout.fit is not None:
                    probabilities = layout.fit.probabilities
                    minima = torch.minimum(minima, probabilities.min(0).values)
                    maxima = torch.maximum(maxima, probabilities.max(0).values)
                    propensity_means.add(w@probabilities)
                    likelihood.add(torch.tensor(layout.fit.log_likelihood, dtype=torch.float64))
                    mass.add(w.sum())
                nums, dens = [], []
                for level in range(levels):
                    if method in {'ra', 'ipwra'}:
                        nums.append((w*selection)@layout.equations[level].mu)
                        dens.append((w*selection).sum())
                    elif method == 'ipw':
                        weights = w*ind[level]*omega[:, level]
                        nums.append(weights@y)
                        dens.append(weights.sum())
                    else:
                        augmented = ind[level]*omega[:, level]*(y-layout.equations[level].mu)+layout.equations[level].mu
                        nums.append(w@augmented)
                        dens.append(w.sum())
                numerator.add(torch.stack(nums))
                denominator.add(torch.stack(dens))
            tau = numerator.value/denominator.value
            size = sample.notes['stacked_size']
            derivative, meat = _CompensatedSum((size, size)), _CompensatedSum((size, size))
            accumulator = ClusterAccumulator(size) if spec.covariance == 'cluster' else None
            try:
                for batch in sample.batches():
                    layout, y, w, ind, omega, slopes, selection = _pieces(sample, nuisance, batch)
                    derivative.add(_jacobian(method, layout, y, w, ind, omega, slopes, tau, selection))
                    scores = _scores(method, layout, y, ind, omega, tau, selection, 0, len(y))
                    if accumulator is not None:
                        accumulator.add(encode_cluster_labels(batch.frame[registry.cluster_columns(spec)[0]]), scores*w[:, None])
                    else:
                        scaled = scores*(w.sqrt() if spec.weight_type == 'fweight' else w)[:, None]
                        meat.add(scaled.T@scaled)
                info = {'covariance': spec.covariance, 'small_sample_correction': 1., 'df_inference': None,
                        'variance_method': 'stacked estimating equations (M-estimation sandwich)',
                        'correction': 'replayed stacked sandwich; no small-sample factor'}
                matrix = meat.value
                if accumulator is not None:
                    matrix, groups = accumulator.finish()
                    info.update(accumulator.diagnostics)
                    info.update({'cluster_count': groups, 'cluster_column': registry.cluster_columns(spec)[0]})
                try:
                    half = torch.linalg.solve(derivative.value, matrix)
                    covariance = torch.linalg.solve(derivative.value, half.T)
                except RuntimeError as exc:
                    raise AnalysisError('singular_jacobian', 'The global stacked derivative is singular.') from exc
                covariance = (covariance+covariance.T)/2
            finally:
                if accumulator is not None:
                    accumulator.close()
            treatment = Treatment(_role(spec, 'treatment')[0], sample.notes['treatment_labels'], torch.empty(0, dtype=torch.int64), sample.notes['treatment_counts'])
            if estimand == 'pomeans':
                transform, terms, equations = torch.eye(levels, dtype=torch.float64), [treatment.mean_term('POmeans', i) for i in range(levels)], ['POmeans']*levels
            else:
                name = 'ATET' if estimand == 'atet' else 'ATE'
                transform = torch.zeros((levels, levels), dtype=torch.float64)
                for i in range(1, levels):
                    transform[i-1, i], transform[i-1, 0] = 1., -1.
                transform[-1, 0] = 1.
                terms = [treatment.effect_term(name, i) for i in range(1, levels)]+[treatment.mean_term('POmean', 0)]
                equations = [name]*(levels-1)+['POmean']
            auxiliary = {}
            if nuisance.treatment is not None:
                design = sample.designs['treatment']
                kt = len(design.terms)
                for i, beta in enumerate(nuisance.treatment):
                    block = slice(i*kt, (i+1)*kt)
                    auxiliary[f'TME{treatment.text(i+1)}'] = coefficient_table([t.removeprefix('TME:') for t in design.terms], design.transform@beta,
                                                                           design.transform@covariance[block, block]@design.transform.T)
            for i, beta in enumerate(nuisance.outcomes):
                design = sample.designs[f'outcome{i}']
                block = layout.beta[i]
                auxiliary[f'OME{treatment.text(i)}'] = coefficient_table([t.removeprefix(f'OME{treatment.text(i)}:') for t in design.terms], design.transform@beta,
                                                                       design.transform@covariance[block, block]@design.transform.T)
            metrics = {'n_levels': levels}
            if levels == 2:
                metrics.update({'n_control': treatment.counts[0], 'n_treated': treatment.counts[1]})
            extra: dict[str, Any] = {'method': method, 'estimand': estimand, 'treatment': treatment.column,
                  'levels': [treatment.text(i) for i in range(levels)], 'control': treatment.text(0),
                  'observations_by_level': {treatment.text(i): treatment.counts[i] for i in range(levels)},
                  'outcome_model': _option(spec, 'omodel') if method in _OMODELS else None,
                  'treatment_model': ('mlogit' if levels > 2 else _option(spec, 'tmodel')) if method in _TMODELS else None,
                  'auxiliary_equations': auxiliary, 'potential_outcome_means': {treatment.text(i): float(tau[i]) for i in range(levels)}}
            from .postest.causal_state import stacked_state
            blocks = []
            if nuisance.treatment is not None:
                design = sample.designs['treatment']
                kt = len(design.terms)
                for i, beta in enumerate(nuisance.treatment):
                    blocks.append((slice(i*kt,(i+1)*kt),f'TME{treatment.text(i+1)}',
                        [t.removeprefix('TME:') for t in design.terms],beta,design.transform))
            for i, beta in enumerate(nuisance.outcomes):
                design = sample.designs[f'outcome{i}']
                blocks.append((layout.beta[i],f'OME{treatment.text(i)}',
                    [t.removeprefix(f'OME{treatment.text(i)}:') for t in design.terms],beta,design.transform))
            extra['evaluation_state'] = stacked_state(size,covariance,tau,layout.tau,
                                                      [treatment.text(i) for i in range(levels)],blocks)
            if nuisance.treatment is not None:
                metrics['treatment_log_likelihood'] = float(likelihood.value)
                if levels == 2:
                    metrics.update({'propensity_min': float(minima[1]), 'propensity_max': float(maxima[1])})
                extra['treatment_covariates'] = [t.removeprefix('TME:') for t in sample.designs['treatment'].terms]
                extra['overlap'] = {'pstolerance': _option(spec, 'pstolerance'), 'summary_scope': 'global minimum, maximum and weighted mean by probability column; no exact quantiles',
                                    'propensity': {treatment.text(i): {'min': float(minima[i]), 'max': float(maxima[i]), 'mean': float(propensity_means.value[i]/mass.value)} for i in range(levels)}}
            result = build_result(ModelFrame(spec, sample.sample), terms=terms, params=transform@tau,
                                  covariance=transform@covariance[layout.tau:, layout.tau:]@transform.T,
                                  equations=equations, title=f'Treatment-effects estimation: {METHOD_TITLES[method]}',
                                  use_t=False, nobs=sample.nobs, inference=info, metrics=metrics, extra=extra,
                                  solver='torch_replayed_stacked_estimating_equations', categories=sample.categories,
                                  provenance={**sample.provenance(), 'prediction_sample': None, 'method': method, 'estimand': estimand,
                                              'parameters_in_stacked_system': size, 'outcome_model': extra['outcome_model'], 'treatment_model': extra['treatment_model']})
            result.nobs_original, result.dropped_rows = sample.original_count, sample.original_count-sample.nrows
            result.sample_positions = []
            return result
    except AnalysisError:
        raise
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
