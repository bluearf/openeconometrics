"""Restricted likelihoods and diagnostic reductions over the full replay sample."""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf, f_sf, normal_sf
from openecon.engines.inference import critical_value
from openecon.engines.streaming_ols import _CompensatedSum
from .core import lr_test, wald_test
from .discrete.common import pseudo_r_squared
from .limited.common import rho_record, sigma_record
from . import registry
from .streaming_likelihood import ReplayObjective, _adapter, _linear_start, _option, _role


class _Restricted:
    """Exact affine embedding of a restricted parameter into the existing kernel."""

    def __init__(self, native, base, free):
        self.native, self.base, self.free = native, base, free

    def full(self, value):
        theta = self.base.clone()
        theta[self.free] = value
        return theta

    def __call__(self, value):
        v, g, h = self.native(self.full(value))
        self.precision_rejected = bool(getattr(self.native, 'precision_rejected', False))
        return v, g[self.free], h[self.free[:, None], self.free]

    def value(self, value):
        answer = self.native.value(self.full(value))
        self.precision_rejected = bool(getattr(self.native, 'precision_rejected', False))
        return answer


def _fit(sample, builder, start):
    objective = ReplayObjective(sample, builder, len(start))
    scale = sample.optimization_scale
    run = optimize.maximize_newton(lambda value: tuple(v*scale for v in objective(value)), start,
                                  value_fn=lambda value: objective.value(value)*scale,
                                  max_iter=_option(sample.spec, 'max_iterations', 200),
                                  step_tol=_option(sample.spec, 'tolerance', 1e-10),
                                  scaled_gradient_tol=_option(sample.spec, 'tolerance', 1e-10),
                                  raise_on_failure=False)
    if not run.converged:
        raise AnalysisError('precision_unsupported' if objective.precision_rejected else 'nonconvergence',
                            'The full-sample restricted/comparison likelihood did not converge in its validated domain.')
    value = float(objective.value(run.theta))
    if not math.isfinite(value):
        raise AnalysisError('precision_unsupported', 'The final restricted likelihood is not finite.')
    return run.theta, value


def _restricted(sample, builder, theta, fixed: Sequence[int]):
    base = theta.clone()
    base[list(fixed)] = 0.
    free = torch.tensor([i for i in range(len(theta)) if i not in set(fixed)], dtype=torch.int64)
    if not len(free):
        value = float(ReplayObjective(sample, builder, len(theta)).value(base))
        if not math.isfinite(value):
            raise AnalysisError('precision_unsupported', 'The restricted fixed point is outside the likelihood domain.')
        return base, value
    reduced, value = _fit(sample, lambda batch: _Restricted(builder(batch), base, free), theta[free])
    base[free] = reduced
    return base, value


def _actual_density(sample, builder, theta, value):
    name = sample.spec.estimator
    if name in {'poisson', 'cloglog', 'fracreg'}:
        total = _CompensatedSum(())
        for batch in sample.batches():
            native = builder(batch).objective
            total.add(torch.tensor(native.log_likelihood(native.state(theta), 1., batch.weights), dtype=torch.float64))
        return float(total.value)
    if name == 'streg':
        total = _CompensatedSum(())
        for batch in sample.batches():
            failure = batch.numeric(_role(sample.spec, 'failure')[0]) if _role(sample.spec, 'failure') else torch.ones(len(batch.frame), dtype=torch.float64)
            total.add((batch.weights*failure*batch.numeric(sample.spec.outcome).log()).sum())
        return value+float(total.value)
    return value


def _comparison(sample, estimator, *, options=None):
    clone = copy.copy(sample)
    info = registry.get(estimator)
    valid_roles, valid_options = {r.name for r in info.roles}, {o.name for o in info.options}
    clone.spec = sample.spec.model_copy(update={'estimator': estimator,
       'columns': {key: value for key, value in sample.spec.columns.items() if key in valid_roles},
       'options': {**{key: value for key, value in sample.spec.options.items() if key in valid_options}, **(options or {})}})
    first_pass = clone.passes
    try:
        builder, start, _, _, _, _ = _adapter(clone)
        theta, value = _fit(clone, builder, start)
        return clone, builder, theta, _actual_density(clone, builder, theta, value)
    finally:
        sample.passes += clone.passes-first_pass
        sample.actual_numeric_peak_rows = max(sample.actual_numeric_peak_rows,
                                             clone.actual_numeric_peak_rows)


def _attempt(extra, name, call):
    try:
        answer = call()
    except (AnalysisError, KernelError) as exc:
        if exc.code not in {'precision_unsupported', 'nonconvergence', 'boundary_solution', 'numerical_failure', 'invalid_information', 'separation_detected'}:
            raise
        extra.setdefault('restricted_model_diagnostics', {})[name] = {'status': 'unavailable', 'reason': exc.code,
             'scope': 'full retained sample; unavailable reference is not replaced by a sampled estimate'}
        return None
    extra.setdefault('restricted_model_diagnostics', {})[name] = {'status': 'available', 'scope': 'full retained sample'}
    return answer


def _boundary(full, reference, label):
    statistic = max(0., 2*(full-reference))
    return {'statistic': statistic, 'df': 1, 'p_value': .5*chi2_sf(statistic, 1) if statistic else 1.,
            'distribution': 'chibar2', 'label': label, 'restricted_log_likelihood': reference}


def _row_density(native, theta):
    if hasattr(native, 'objective'):
        from openecon.engines.count_numeric import poisson_logmass
        obj = native.objective
        state = obj.state(theta)
        eta = obj.x@theta+(obj.offset if obj.offset is not None else 0.)
        return poisson_logmass(obj.y, state.mu, eta=eta)
    return native.observations(theta)


def _vuong(sample, full_builder, theta, alternative, extra_parameters):
    _, builder, other, _ = alternative
    n = sample.nobs
    total = _CompensatedSum(())
    for batch in sample.batches():
        difference = _row_density(full_builder(batch), theta)-_row_density(builder(batch), other)
        weight = batch.weights if sample.spec.weight_type == 'fweight' else torch.ones_like(difference)
        total.add(weight@difference)
    mean = float(total.value)/n
    variance = _CompensatedSum(())
    for batch in sample.batches():
        difference = _row_density(full_builder(batch), theta)-_row_density(builder(batch), other)
        weight = batch.weights if sample.spec.weight_type == 'fweight' else torch.ones_like(difference)
        variance.add(weight@(difference-mean).square())
    var = float(variance.value)/(n-1)
    if not var > 0:
        return None
    sd, root = math.sqrt(var), math.sqrt(n)
    z = root*mean/sd
    record = {'z': z, 'z_aic': root*(mean-extra_parameters/n)/sd,
              'z_bic': root*(mean-extra_parameters*math.log(n)/(2*n))/sd,
              'mean_difference': mean, 'sd_difference': sd, 'observations': n,
              'extra_parameters': extra_parameters, 'p_value_two_sided': 2*normal_sf(abs(z))}
    for key in ['z_aic', 'z_bic']:
        record['p_value_'+key[2:]] = normal_sf(record[key])
    return {'statistic': z, 'p_value': normal_sf(z), 'distribution': 'normal',
            'label': 'Vuong test against the model without inflation'}, record


def _first_stage(sample, extra):
    design, endogenous = sample.designs['instruments'], sample.designs['endogenous']
    k, kz = len(sample.designs['mean'].terms), len(design.terms)
    scales = endogenous.scales[endogenous.kept]
    centres = endogenous.means[endogenous.kept]/scales if design.intercept else torch.zeros_like(scales)
    summary = {}
    for j, name in enumerate(_role(sample.spec, 'endogenous')):
        def target(batch):
            return batch.designs['endogenous'][:, j]-centres[j]
        _, variance = _linear_start(sample, 'instruments', target)
        if k:
            _, restricted = _linear_start(sample, 'mean', target)
        else:
            restricted = None
        mass, spread = _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            mass.add(batch.weights.sum())
            spread.add(batch.weights@target(batch).square())
        ssr = variance*float(mass.value)
        ssr_r = restricted*float(mass.value) if restricted is not None else float(spread.value)
        tss = float(spread.value)
        r2 = 1-ssr/tss if tss > 0 else None
        df2, m = sample.nobs-kz, kz-k
        f = ((ssr_r-ssr)/m)/(ssr/df2) if df2 > 0 and ssr > 0 else None
        summary[name] = {'r_squared': r2, 'adjusted_r_squared': None if r2 is None or df2 <= 0 else 1-(1-r2)*(sample.nobs-int(design.intercept))/df2,
             'partial_r_squared': 1-ssr/ssr_r if ssr_r > 0 else None, 'f_statistic': f, 'f_df1': m,
             'f_df2': df2, 'f_p_value': None if f is None else f_sf(f, m, df2),
             'rmse': math.sqrt(ssr/df2)*float(scales[j]) if df2 > 0 else None}
    extra.update({'first_stage': summary, 'exogenous': sample.designs['mean'].terms,
                  'n_instruments': kz-k})


def augment(sample, builder, theta, terms, params, covariance, metrics, tests, extra):
    """Mutate only result dictionaries; every comparison uses the same global sample."""
    spec, name = sample.spec, sample.spec.estimator
    k = len(sample.designs['mean'].terms)
    slopes = [i for i, term in enumerate(terms[:k]) if not term.endswith('Intercept')]
    ll = metrics['log_likelihood']
    if name in {'heckman', 'heckprobit', 'zip', 'zinb', 'hurdle', 'intreg'}:
        counts = _CompensatedSum((4,))
        for batch in sample.batches():
            frequency = batch.weights if spec.weight_type == 'fweight' else torch.ones(len(batch.frame), dtype=torch.float64)
            if name == 'intreg':
                from .streaming_likelihood import _censored_bounds
                low, high = _censored_bounds(batch, spec)
                left, right = torch.isneginf(low), torch.isposinf(high)
                point = ~left & ~right & (low == high)
                interval = ~left & ~right & (low < high)
                masks = (left, point, right, interval)
            else:
                column = _role(spec, 'select')[0] if name in {'heckman', 'heckprobit'} else spec.outcome
                on = batch.numeric(column) > 0
                masks = (on, ~on, torch.zeros_like(on), torch.zeros_like(on))
            counts.add(torch.stack([frequency[mask].sum() for mask in masks]))
        values = [int(round(float(value))) for value in counts.value]
        if name == 'intreg':
            metrics.update(zip(('n_left_censored', 'n_uncensored', 'n_right_censored', 'n_interval'), values, strict=True))
            metrics.update(n_point=values[1], df_model=len(slopes))
        elif name in {'heckman', 'heckprobit'}:
            metrics.update(n_selected=values[0], n_censored=values[1])
        else:
            metrics['n_zero_observations'] = values[1]
    if name in {'tobit', 'ivtobit'}:
        metrics.update(zip(('n_left_censored', 'n_uncensored', 'n_right_censored'), sample.notes['censoring_counts'], strict=True))
    if name == 'truncreg':
        metrics['n_truncated'] = sample.notes['n_truncated']
        metrics['df_model'] = len(slopes)
    if name == 'streg':
        survival = sample.notes['survival']
        metrics.update({field: survival[field] for field in ('n_subjects', 'n_failures', 'time_at_risk')})
        metrics['df_model'] = len(slopes)
        extra['subject_validation'] = {'scope': 'all retained records, including subject intervals across source batches',
                                       'overlap_checked': survival['id_overlap_checked'],
                                       'scratch_bytes': survival['subject_scratch_bytes']}
    full_null = {'poisson', 'cloglog', 'fracreg', 'nbreg', 'ologit', 'oprobit', 'mlogit', 'streg',
                 'tobit', 'intreg', 'cpoisson', 'cnbreg', 'tpoisson', 'tnbreg', 'zip', 'zinb', 'gnbreg', 'hurdle', 'churdle'}
    if name == 'mlogit':
        slopes = [i for i, term in enumerate(terms) if not term.endswith('Intercept')]
    allowed = name not in {'tobit', 'intreg', 'cpoisson', 'cnbreg', 'tpoisson', 'tnbreg', 'zip', 'zinb', 'gnbreg', 'hurdle', 'churdle'} or spec.intercept
    if name in full_null and allowed:
        answer = _attempt(extra, 'null', lambda: _restricted(sample, builder, theta, slopes))
        null = None if answer is None else _actual_density(sample, builder, *answer)
        extra['null_log_likelihood'] = null
        if name not in {'intreg', 'streg'}:
            metrics['pseudo_r_squared'] = pseudo_r_squared(ll, null)
        lr_covariances = {'nonrobust', 'opg'} if name in {'tobit', 'intreg'} else {'nonrobust'}
        if null is not None and spec.covariance in lr_covariances and name != 'fracreg':
            tests['model'] = lr_test(ll, null, len(slopes), label='LR test against the full-sample restricted outcome equation')
        if null is not None and null > ll+1e-7*max(1., abs(ll)):
            raise AnalysisError('invalid_comparison', 'The restricted likelihood exceeds the reported full likelihood.')
    if name == 'biprobit':
        q = len(sample.designs['second'].terms)
        both = slopes+[k+i for i, term in enumerate(terms[k:k+q]) if not term.endswith('Intercept')]
        tests['model'] = wald_test(params, covariance, both, label='Wald chi2 of both outcome-equation slopes')
    if name in {'biprobit', 'heckman', 'heckprobit', 'hetprobit'}:
        if name == 'hetprobit':
            fixed, key = list(range(k, len(theta))), 'lnsigma'
            df = len(fixed)
        else:
            q = len(sample.designs['second' if name == 'biprobit' else 'selection'].terms)
            fixed, key, df = [k+q], 'rho', 1
        answer = _attempt(extra, key, lambda: _restricted(sample, builder, theta, fixed))
        if answer is not None:
            extra['probit_log_likelihood' if name == 'hetprobit' else 'comparison_log_likelihood'] = answer[1]
            if spec.covariance in {'nonrobust', 'opg'}:
                tests[key] = lr_test(ll, answer[1], df, label=f'LR test of {key}=0 using the full sample')
        if spec.covariance not in {'nonrobust', 'opg'}:
            tests[key] = wald_test(params, covariance, fixed, label=f'Wald test of {key}=0')
        if name != 'hetprobit':
            position = fixed[0]
            extra['rho'] = rho_record(float(params[position]), float(covariance[position, position]), spec.alpha)
            metrics['rho'] = extra['rho']['estimate']
    if name in {'ivprobit', 'ivtobit'}:
        exogeneity = [i for i, term in enumerate(terms) if term.startswith('/athrho') and term.endswith('_1')]
        tests['exogeneity'] = wald_test(params, covariance, exogeneity, label='Wald test of outcome-error/endogenous-error exogeneity')
        _first_stage(sample, extra)
        extra['correlations'], extra['standard_deviations'] = {}, {}
        for i, term in enumerate(terms):
            if term.startswith('/athrho'):
                extra['correlations'][term] = rho_record(float(params[i]), float(covariance[i, i]), spec.alpha)
            elif term.startswith('/lnsigma'):
                extra['standard_deviations'][term] = sigma_record(float(params[i]), float(covariance[i, i]), spec.alpha)
        if '/lnsigma1' in extra['standard_deviations']:
            metrics['sigma'] = extra['standard_deviations']['/lnsigma1']['estimate']
    if name == 'gnbreg':
        shape = sample.designs['dispersion']
        alpha_sum, weight_sum = _CompensatedSum(()), _CompensatedSum(())
        alpha_min, alpha_max = math.inf, -math.inf
        for batch in sample.batches():
            fitted_alpha = builder(batch).indices(theta)[1].exp()
            alpha_sum.add(batch.weights @ fitted_alpha)
            weight_sum.add(batch.weights.sum())
            alpha_min = min(alpha_min, float(fitted_alpha.min()))
            alpha_max = max(alpha_max, float(fitted_alpha.max()))
        extra['alpha'] = {'mean': float(alpha_sum.value / weight_sum.value),
                          'min': alpha_min, 'max': alpha_max,
                          'definition': "alpha_i = exp(z_i'd) over the estimation sample"}
        extra['variance_function'] = 'mu (1 + alpha_i mu)'
        indices = [k+i for i, term in enumerate(shape.terms) if not term.endswith('Intercept')]
        if indices and spec.covariance == 'nonrobust':
            comparison = _attempt(extra, 'constant_dispersion', lambda: _restricted(sample, builder, theta, indices))
            if comparison is not None:
                tests['lnalpha'] = lr_test(ll, comparison[1], len(indices), label='LR test of constant dispersion')
        elif indices:
            tests['lnalpha'] = wald_test(params, covariance, indices, label='Wald test of constant dispersion')
    if name in {'nbreg', 'cnbreg', 'tnbreg', 'zinb'} or name == 'hurdle' and _option(spec, 'dist') == 'nbinomial':
        if spec.covariance in {'nonrobust', 'opg'}:
            other_name = {'nbreg': 'poisson', 'cnbreg': 'cpoisson', 'tnbreg': 'tpoisson', 'zinb': 'zip', 'hurdle': 'hurdle'}[name]
            comparison = _attempt(extra, 'zero_dispersion', lambda: _comparison(sample, other_name, options={'dist': 'poisson'} if name == 'hurdle' else None))
            if comparison is not None:
                tests['alpha'] = _boundary(ll, comparison[3], 'LR test of zero dispersion (chibar2(01))')
    if name in {'zip', 'zinb'}:
        allowed = spec.weight_type in {None, 'fweight'} and spec.covariance in {'nonrobust', 'opg'}
        if allowed:
            comparison = _attempt(extra, 'without_inflation', lambda: _comparison(sample, 'poisson' if name == 'zip' else 'nbreg'))
            if comparison is not None:
                result = _vuong(sample, builder, theta, comparison, len(sample.designs['inflate'].terms))
                if result is not None:
                    tests['vuong'], extra['vuong'] = result
        else:
            extra['vuong_note'] = 'Vuong compares model likelihoods and is reported only with unweighted/frequency-weighted likelihood covariance.'
    if name == 'frontier' and spec.covariance in {'nonrobust', 'opg'}:
        _, variance = _linear_start(sample, 'mean', lambda b: b.numeric(spec.outcome))
        mass = _CompensatedSum(())
        for batch in sample.batches():
            mass.add(batch.weights.sum())
        reference = -.5*float(mass.value)*(math.log(2*math.pi*variance)+1)
        metrics['log_likelihood_ols'] = reference
        tests['sigma_u'] = _boundary(ll, reference, 'LR test of sigma_u=0 (chibar2(01))')
    if name == 'betareg':
        from .glm.families import make_link
        from .glm.kernels import ScaleLink
        link = make_link(_option(spec, 'link'))
        mass, sums = _CompensatedSum(()), _CompensatedSum((2,))
        low, high = math.inf, -math.inf
        for batch in sample.batches():
            native = builder(batch)
            eta, g = batch.designs['mean']@theta[:k], link.link(batch.numeric(spec.outcome))
            mass.add(batch.weights.sum())
            sums.add(torch.stack((batch.weights@eta, batch.weights@g)))
            phi = native.fitted(theta)[2]
            low, high = min(low, float(phi.min())), max(high, float(phi.max()))
        centers = sums.value/mass.value
        moments = _CompensatedSum((3,))
        for batch in sample.batches():
            eta = batch.designs['mean']@theta[:k]-centers[0]
            g = link.link(batch.numeric(spec.outcome))-centers[1]
            moments.add(torch.stack((batch.weights@eta.square(), batch.weights@g.square(), batch.weights@(eta*g))))
        a, b, cross = moments.value.tolist()
        metrics.update({'pseudo_r_squared': cross*cross/(a*b) if a*b > 0 else None, 'df_resid': sample.nobs-len(theta)})
        extra.update({'pseudo_r_squared_definition': "squared correlation between x'b and link(y)", 'precision_range': [low, high]})
        if len(sample.designs['precision'].terms) == 1:
            scale_link = ScaleLink(_option(spec, 'scale_link'))
            constant = params[k].reshape(1)
            fitted = scale_link.inverse(constant)
            estimate = float(fitted)
            derivative = abs(float(scale_link.derivatives(constant, fitted)[0]))
            se_link = math.sqrt(float(covariance[k, k]))
            z = critical_value(spec.alpha)
            bounds = ([float(scale_link.inverse(constant + sign*z*se_link)) for sign in (-1, 1)]
                      if scale_link.name == 'log' else
                      [estimate-z*derivative*se_link, estimate+z*derivative*se_link])
            extra['precision'] = {'estimate': estimate, 'std_error': derivative*se_link,
                                  'ci_low': bounds[0], 'ci_high': bounds[1],
                                  'method': 'delta method from the scale-equation constant'}
    if name == 'heckman':
        q = len(sample.designs['selection'].terms)
        a = k+q
        rho, sigma = math.tanh(float(params[a])), math.exp(float(params[a+1]))
        lam = rho*sigma
        gradient = torch.tensor([sigma*(1-rho*rho), lam], dtype=torch.float64)
        se = math.sqrt(max(0., float(gradient@covariance[a:a+2, a:a+2]@gradient)))
        metrics['lambda'] = lam
        extra['lambda'] = {'estimate': lam, 'std_error': se, 'method': 'delta method on athrho and lnsigma'}
    if name == 'poisson':
        df = sample.nobs-k
        metrics['df_resid'] = df
        for key, field in [('gof_deviance', 'deviance'), ('gof_pearson', 'pearson')]:
            tests[key] = {'statistic': metrics[field], 'df': df, 'p_value': chi2_sf(metrics[field], df) if df > 0 else None,
                          'distribution': 'chi2', 'label': field+' goodness of fit'}
    if name in {'tobit', 'truncreg', 'intreg', 'heckman', 'churdle'}:
        i = len(params)-1
        log_scale = name in {'intreg', 'heckman', 'churdle'}
        value = float(params[i])
        log_value = value if log_scale else math.log(value)
        log_var = float(covariance[i, i]) if log_scale else float(covariance[i, i])/(value*value)
        record = sigma_record(log_value, log_var, spec.alpha)
        metrics['sigma'] = record['estimate']
        if log_scale:
            extra['sigma'] = record
        else:
            extra['lnsigma'] = {'estimate': log_value, 'std_error': math.sqrt(log_var)}
            extra['variance'] = {'estimate': value*value, 'std_error': 2*value*math.sqrt(float(covariance[i, i])), 'method': 'delta method for sigma squared'}
    if name in {'nbreg', 'cnbreg', 'tnbreg', 'zinb'} or name == 'hurdle' and _option(spec, 'dist') == 'nbinomial':
        from .count.common import exponentiated
        field = 'delta' if _option(spec, 'dispersion', 'mean') == 'constant' else 'alpha'
        metrics[field] = math.exp(float(params[-1]))
        extra['alpha'] = exponentiated(float(params[-1]), math.sqrt(float(covariance[-1, -1])), spec.alpha, field)
    if name in {'glm', 'poisson', 'cloglog', 'fracreg'}:
        df = sample.nobs-k
        metrics['df_resid'] = df
        if name == 'glm':
            metrics.update({'dispersion_deviance': metrics['deviance']/df, 'dispersion_pearson': metrics['pearson']/df,
                            'aic_glm': metrics['aic']/sample.nobs, 'bic_glm': metrics['deviance']-df*math.log(sample.nobs)})
    extra['model_test_scope'] = 'Full-sample restricted likelihood and joint Wald comparisons follow each implemented native model convention; unavailable references are recorded explicitly.'
