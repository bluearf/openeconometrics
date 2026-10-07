"""Bacon diagnostics, BJS imputation and Sun-Abraham interaction weighting.

Resident balanced complete panels, absorbing binary adoption, unit/time FE,
no covariates or weights. Native QR with explicit conditional-design targets.
"""
from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, kernel_call, make_spec, table
from openecon.engines.linalg import least_squares


def load(frame):
    frame.sort_panel()
    units, n = frame.codes(frame.spec.panel)
    counts = torch.bincount(units, minlength=n)
    if n < 3 or not bool((counts == counts[0]).all()):
        raise AnalysisError('unbalanced_panel', 'Need at least three complete units on a balanced clock.')
    t = int(counts[0])
    clock = frame.time_index().reshape(n, t)
    if t < 3 or not bool((clock == clock[0]).all()) or not bool((clock[:, 1:] - clock[:, :-1] == 1).all()):
        raise AnalysisError('time_gaps', 'Every unit must have the same consecutive clock and at least three periods.')
    frame.workspace_plan('balanced causal panel inputs', {'numeric_panel': 96 * frame.n})
    y = frame.numeric(frame.spec.outcome).reshape(n, t)
    treatment = frame.numeric(frame.role('treatment')[0]).reshape(n, t)
    if not bool(((treatment == 0) | (treatment == 1)).all()) or bool((treatment[:, 1:] < treatment[:, :-1]).any()):
        raise AnalysisError('invalid_treatment', 'Treatment must be binary absorbing adoption within units.')
    if bool((treatment[:, 0] == 1).any()) or not bool(treatment.any()):
        raise AnalysisError('invalid_treatment', 'Always-treated units are unsupported; at least one adopting unit is required.')
    first = torch.where(treatment.bool(), torch.arange(t)[None, :], t).amin(dim=1)
    labels = frame.sample.groupby(frame.spec.panel, sort=False, observed=True)[frame.spec.panel].first().tolist()
    return y, treatment, first, labels, clock[0].tolist()


def _ols(x, y):
    return kernel_call(least_squares, x.contiguous(), y.contiguous(), drop_collinear=False)


def _fe(n, t):
    return torch.cat((torch.eye(n, dtype=torch.float64).repeat_interleave(t, dim=0),
                      torch.eye(t, dtype=torch.float64).repeat(n, 1)[:, 1:]), dim=1)


def fit_heterodid(spec, data):
    with torch.device('cpu'), torch.no_grad():
        return _fit(spec, data)


def _fit(spec, data):
    frame = ModelFrame(spec, data)
    y, d, first, labels, calendar = load(frame)
    n, t = y.shape
    cohorts = sorted(set(first[first < t].tolist()))
    if not bool((first == t).any()):
        raise AnalysisError('missing_controls', 'This domain requires never-treated comparison units.')
    lo, hi = frame.option('event_min'), frame.option('event_max')
    hi = t - 1 if hi is None else hi
    events = sorted({period - c for c in cohorts for period in range(t)
                     if lo <= period - c <= hi and period - c != -1})
    if not events:
        raise AnalysisError('empty_estimand', 'No requested non-reference relative periods are observed.')
    model = frame.option('model')
    width = n + t - 1 + (len(cohorts) * (t - 1) if model == 'sunab' else 0)
    work = 4 * frame.n * width ** 2
    if work > frame.option('max_work'):
        raise AnalysisError('work_budget', 'Declared unit/time/cohort QR work exceeds max_work.')
    frame.workspace_plan('heterogeneous DiD designs and joint inference', {
        'full_and_restricted_designs': 96 * frame.n * width,
        'inference_maps': 96 * frame.n * len(events),
        'coefficient_factors': 96 * width ** 2,
    })
    fe = _fe(n, t)
    target = torch.zeros((len(events), frame.n), dtype=torch.float64)
    support = []
    for j, event in enumerate(events):
        cells = [(i, c + event) for i, c in enumerate(first.tolist()) if c < t and 0 <= c + event < t]
        for i, period in cells:
            target[j, i * t + period] = 1 / len(cells)
        support.append({'event': event, 'n_units': len(cells), 'cohorts': [c for c in cohorts if 0 <= c + event < t]})
    flat_y = y.flatten()
    if model == 'bjs':
        if lo < 0:
            raise AnalysisError('unsupported_estimand', 'BJS currently estimates post-treatment effects; negative relative periods need a separately specified pretrend design.')
        if any(int((first == c).sum()) < 2 for c in cohorts):
            raise AnalysisError('insufficient_cohort', 'BJS cohort/time variance centering needs at least two units per treated cohort.')
        untreated = ~d.bool().flatten()
        fitted = _ols(fe[untreated], flat_y[untreated])
        untreated_prediction = fe @ fitted.beta
        beta = target @ (flat_y - untreated_prediction)
        # Linear map from all observed outcomes to every reported effect,
        # including uncertainty in the imputed unit/time coefficients.
        mapping = target.clone()
        mapping[:, untreated] -= target @ fe @ fitted.xtx_inv @ fe[untreated].T
        residual = flat_y - untreated_prediction
        for c in cohorts:
            rows = first == c
            for period in range(c, t):
                positions = rows.nonzero().flatten() * t + period
                residual[positions] -= residual[positions].mean()
        scores = (mapping * residual[None, :]).reshape(len(events), n, t).sum(dim=2).T
        vc = scores.T @ scores * n / (n - 1)
        correction = 'BJS imputation linear map including first-stage uncertainty; cohort/time-centered treated errors; unit CR1 G/(G-1)'
        detail = {'untreated_nobs': int(untreated.sum()), 'untreated_fe_coefficients': fitted.beta.tolist()}
    else:
        interaction, cell_terms = [], []
        for c in cohorts:
            for period in range(t):
                if period - c == -1:
                    continue
                column = torch.zeros((n, t), dtype=torch.float64)
                column[first == c, period] = 1
                interaction.append(column.flatten())
                cell_terms.append((c, period - c))
        design = torch.cat((fe, torch.stack(interaction, dim=1)), dim=1)
        fitted = _ols(design, flat_y)
        cross = design * fitted.resid[:, None]
        scores = cross.reshape(n, t, -1).sum(dim=1)
        all_v = fitted.xtx_inv @ (scores.T @ scores) @ fitted.xtx_inv * n / (n - 1) * (frame.n - 1) / (frame.n - design.shape[1])
        aggregation = torch.zeros((len(events), design.shape[1]), dtype=torch.float64)
        for j, event in enumerate(events):
            total = support[j]['n_units']
            for col, (c, relative) in enumerate(cell_terms):
                if relative == event:
                    aggregation[j, fe.shape[1] + col] = int((first == c).sum()) / total
        beta = aggregation @ fitted.beta
        vc = aggregation @ all_v @ aggregation.T
        correction = 'Sun-Abraham saturated cohort/event interactions; fixed observed cohort shares; full-design unit CR1'
        detail = {'cohort_event_terms': cell_terms, 'cohort_event_coefficients': fitted.beta[fe.shape[1]:].tolist(),
                  'cohort_event_covariance': all_v[fe.shape[1]:, fe.shape[1]:].tolist(),
                  'aggregation': aggregation[:, fe.shape[1]:].tolist()}
    return build_result(frame, terms=[f'event.{e}' for e in events], params=beta, covariance=vc,
        use_t=False, title='BJS imputation DiD' if model == 'bjs' else 'Sun-Abraham interaction-weighted event study',
        inference={'correction': correction, 'target': 'observed eligible unit mean by relative period; fixed cohort membership'},
        solver='native joint QR; explicit influence/aggregation maps',
        extra={'model': model, 'calendar': calendar, 'unit_labels': labels, 'adoption_period_index': first.tolist(),
               'event_support': support, 'reference_event': -1, 'fit_event_binning': 'none; all nonreference cohort/event cells included',
               'control': 'never-treated', 'assumptions': 'no anticipation and parallel untreated unit/time trends; independent units for inference',
               'planned_qr_work': work, **detail})


def heterodid(*, data: Any, y: str, treatment: str, panel: str, time: str,
              model: str = 'bjs', event_min: int = 0, event_max: int | None = None,
              missing: str = 'raise', max_work: int = 200_000_000, alpha: float = .05):
    from openecon.analysis import fit
    return fit(make_spec('heterodid', outcome=y, predictors=[], panel=panel, time=time,
                        columns={'treatment': treatment}, missing=missing, alpha=alpha,
                        options={'model': model, 'event_min': event_min, 'event_max': event_max, 'max_work': max_work}), data=data)


def bacon(data: Any, y: str, treatment: str, panel: str, time: str):
    """Unweighted, covariate-free Goodman-Bacon decomposition of balanced TWFE."""
    spec = make_spec('heterodid', outcome=y, predictors=[], panel=panel, time=time, columns={'treatment': treatment})
    frame = ModelFrame(spec, data)
    yy, d, first, labels, calendar = load(frame)
    n, t = yy.shape
    cohorts = sorted(set(first.tolist()))
    if len(cohorts) < 2:
        raise AnalysisError('unidentified_treatment', 'TWFE requires at least two adoption groups.')
    frame.workspace_plan('Bacon cohort comparisons', {'panel_inputs_and_comparisons': 192 * frame.n + 512 * len(cohorts) ** 2})
    def within(v):
        return v - v.mean(dim=0, keepdim=True) - v.mean(dim=1, keepdim=True) + v.mean()
    dtilde = within(d)
    denominator = float(dtilde.square().sum())
    if denominator <= 0:
        raise AnalysisError('unidentified_treatment', 'No residual treatment variation.')
    twfe = float((dtilde * within(yy)).sum()) / denominator
    rows = []
    for a in cohorts:
        if a == t:
            continue
        for b in cohorts:
            if a == b:
                continue
            periods = torch.arange(t) < b if a < b else torch.arange(t) >= b
            units = (first == a) | (first == b)
            ya, da = yy[units][:, periods], d[units][:, periods]
            va = within(da)
            sd = float(va.square().sum())
            if not sd > 0:
                continue
            estimate = float((va * within(ya)).sum()) / sd
            nk, nu = int((first == a).sum()), int((first == b).sum())
            # Global cohort shares and treatment fractions, restricted-window
            # 2x2 variances; time fractions enter the Bacon weight squared.
            if b == t:
                fraction = (t - a) / t
                raw_weight = nk * nu * fraction * (1 - fraction)
                kind = 'treated_vs_never'
            elif a < b:
                raw_weight = nk * nu * a * (b - a) / t ** 2
                kind = 'early_vs_late'
            else:
                raw_weight = nk * nu * (a - b) * (t - a) / t ** 2
                kind = 'late_vs_early'
            rows.append({'treated_cohort_index': a, 'comparison_cohort_index': b,
                         'comparison': kind, 'estimate': estimate, 'weight': raw_weight,
                         'window_period_indices': periods.nonzero().flatten().tolist()})
    total = sum(r['weight'] for r in rows)
    for row in rows:
        row['weight'] /= total
        row['contribution'] = row['weight'] * row['estimate']
    reconstructed = sum(r['contribution'] for r in rows)
    if abs(reconstructed - twfe) > 1e-8 * (1 + abs(twfe)):
        raise AnalysisError('decomposition_failure', '2x2 weights do not reconstruct the TWFE coefficient.')
    cells = [{'cohort_index': c, 'period_index': period,
              'weight': float(dtilde[first == c, period].sum()) / denominator}
             for c in cohorts if c < t for period in range(c, t)]
    return table(rows, columns=list(rows[0]), twfe=twfe, reconstructed_twfe=reconstructed,
                 causal_cell_weights=cells,
                 nobs=frame.n, unit_labels=labels, calendar=calendar,
                 contamination='late-vs-early comparisons use already-treated controls; nonnegative 2x2 weights do not imply nonnegative treatment-effect weights',
                 domain='balanced, complete, unweighted, no covariates or always-treated units')
