"""Outcome-history simplex synthetic control and synthetic DiD, native active-set QP."""
from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, make_spec
from openecon.econometrics.teffects.modern_did import load


class Work:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0

    def consume(self, work):
        self.used += work
        if self.used > self.maximum:
            raise AnalysisError('work_budget', 'Synthetic-control QP and placebo work exceeded max_work.')


def simplex_qp(a, b, penalty, work, *, tolerance=1e-10, max_iter=1000):
    """Feasible active-set equality QP; no clipping of an unconstrained solution."""
    n = a.shape[1]
    work.consume(a.shape[0] * n ** 2)
    h = a.T @ a + torch.eye(n, dtype=torch.float64) * penalty
    q = a.T @ b
    scale = max(1., float(h.abs().max()), float(q.abs().max()))
    h, q = h / scale, q / scale
    individual = h.diagonal() / 2 - q
    w = torch.zeros(n, dtype=torch.float64)
    w[int(individual.argmin())] = 1
    active = w > 0
    for iteration in range(max_iter):
        index = active.nonzero().flatten()
        k = len(index)
        work.consume((k + 1) ** 3)
        system = torch.zeros((k + 1, k + 1), dtype=torch.float64)
        system[:k, :k] = h[index][:, index]
        system[:k, k] = 1
        system[k, :k] = 1
        rhs = torch.cat((q[index], torch.ones(1, dtype=torch.float64)))
        optimum, info = torch.linalg.solve_ex(system, rhs)
        if int(info) or not bool(torch.isfinite(optimum).all()):
            raise AnalysisError('nonunique_simplex', 'Active simplex QP is singular; declare positive ridge regularization if needed.')
        candidate = torch.zeros_like(w)
        candidate[index] = optimum[:k]
        negative = candidate < -tolerance
        if bool(negative.any()):
            direction = candidate - w
            moving = direction < 0
            step = float((-w[moving] / direction[moving]).min())
            w += max(0., min(1., step)) * direction
            w[w.abs() < tolerance] = 0
            active = w > 0
            continue
        w = candidate.clamp_min(0)
        w /= w.sum()
        gradient = h @ w - q
        gap = float(w @ gradient - gradient.min())
        if gap <= tolerance:
            residual = a @ w - b
            return w, {'iterations': iteration + 1, 'simplex_sum': float(w.sum()),
                       'minimum_weight': float(w.min()), 'scaled_dual_gap': gap,
                       'objective': float(residual @ residual + penalty * (w @ w)), 'penalty': penalty}
        active[int(gradient.argmin())] = True
    raise AnalysisError('convergence_failure', 'Simplex active-set QP did not meet its declared dual-gap tolerance.')


def estimate(y, n0, t0, model, ridge, zeta_omega, zeta_lambda, work):
    n, t = y.shape
    control = y[:n0]
    treated = y[n0:].mean(dim=0)
    a, b = control[:, :t0].T, treated[:t0]
    noise = float((control[:, 1:t0] - control[:, :t0-1]).flatten().std())
    if model == 'sdid':
        a, b = a - a.mean(dim=0), b - b.mean()
        zo = ((n - n0) * (t - t0)) ** .25 * noise if zeta_omega is None else zeta_omega
        zl = 1e-6 * noise if zeta_lambda is None else zeta_lambda
        penalty = t0 * zo ** 2
    else:
        zo, zl, penalty = 0., 0., ridge
    omega, od = simplex_qp(a, b, penalty, work)
    if model == 'sdid':
        aa, bb = control[:, :t0], control[:, t0:].mean(dim=1)
        aa, bb = aa - aa.mean(dim=0), bb - bb.mean()
        lam, ld = simplex_qp(aa, bb, n0 * zl ** 2, work)
    else:
        lam, ld = torch.zeros(t0, dtype=torch.float64), None
    curve = treated - omega @ control
    shift = float(lam @ curve[:t0])
    att = float(curve[t0:].mean()) - shift
    return att, {'donor_weights': omega.tolist(), 'pre_period_weights': lam.tolist(),
                 'unit_qp': od, 'time_qp': ld, 'noise_scale': noise, 'zeta_omega': zo,
                 'zeta_lambda': zl, 'effect_curve': (curve - shift).tolist(),
                 'pre_rmspe': float(curve[:t0].square().mean().sqrt())}


def fit_synthcontrol(spec, data):
    with torch.device('cpu'), torch.no_grad():
        return _fit(spec, data)


def _fit(spec, data):
    frame = ModelFrame(spec, data)
    y, _, first, labels, calendar = load(frame)
    n, t = y.shape
    treated_times = first[first < t].unique()
    if len(treated_times) != 1:
        raise AnalysisError('unsupported_adoption', 'Synthetic control/DiD requires one shared adoption date.')
    t0 = int(treated_times[0])
    controls, treated = first == t, first < t
    n0, n1 = int(controls.sum()), int(treated.sum())
    if n0 < n1 + 1 or t0 < 2:
        raise AnalysisError('insufficient_donors', 'Need at least two pre-periods and more original donors than treated units for placebo inference.')
    reps = n0 if n1 == 1 else frame.option('placebo_reps')
    if n1 > 1 and reps < 2:
        raise AnalysisError('insufficient_placebos', 'Use at least two same-size placebo assignments.')
    frame.workspace_plan('synthetic QP and placebo states', {
        'numeric_panels_and_copies': 160 * frame.n,
        'simplex_qp_factors': 192 * (n ** 2 + t ** 2),
        'placebo_output': 128 * reps,
    })
    work = Work(frame.option('max_work'))
    order = torch.cat((controls.nonzero().flatten(), treated.nonzero().flatten()))
    kwargs = (frame.option('model'), frame.option('ridge'), frame.option('zeta_omega'), frame.option('zeta_lambda'), work)
    att, detail = estimate(y[order], n0, t0, *kwargs)
    if frame.option('model') == 'sdid':
        # Placebo assignments hold the fitted tuning parameters fixed, as in
        # the paper's Algorithm 4; do not reselect their noise multiplier.
        kwargs = (frame.option('model'), frame.option('ridge'), detail['zeta_omega'], detail['zeta_lambda'], work)
    generator = torch.Generator().manual_seed(frame.option('seed'))
    placebo, assignments = [], []
    original_donors = y[controls]
    for j in range(reps):
        draw = torch.tensor([j]) if n1 == 1 else torch.randperm(n0, generator=generator)[:n1]
        selected = torch.zeros(n0, dtype=torch.bool)
        selected[draw] = True
        placebo_order = torch.cat(((~selected).nonzero().flatten(), selected.nonzero().flatten()))
        effect, _ = estimate(original_donors[placebo_order], n0 - n1, t0, *kwargs)
        placebo.append(effect)
        assignments.append(draw.tolist())
    values = torch.tensor(placebo, dtype=torch.float64)
    variance = values.var(unbiased=False)
    return build_result(frame, terms=['ATT'], params=torch.tensor([att], dtype=torch.float64), covariance=variance.reshape(1, 1),
        use_t=False, title='Synthetic DiD' if frame.option('model') == 'sdid' else 'Synthetic control',
        solver='native feasible active-set simplex QP; profiled intercepts for SDID',
        inference={'correction': 'refitted original-control placebo dispersion, divisor B; normal approximation conditional on placebo exchangeability',
                   'target': 'mean treated post-period effect', 'inference_method': 'placebo'},
        tests={'placebo': {'p_value': (1 + int((values.abs() >= abs(att)).sum())) / (reps + 1),
                            'statistic': att, 'distribution': 'same-size original-control assignment distribution'}},
        extra={'model': frame.option('model'), 'calendar': calendar, 'adoption_period_index': t0,
               'donor_labels': [labels[i] for i in controls.nonzero().flatten().tolist()],
               'treated_labels': [labels[i] for i in treated.nonzero().flatten().tolist()],
               'placebo_effects': placebo, 'placebo_assignments_donor_indices': assignments,
               'placebo_population': 'original untreated units only; original treated units excluded from all placebo donor pools',
               'seed': frame.option('seed'), 'qp_work': work.used, 'max_work': work.maximum,
               'assumptions': 'stable untreated common factors and no anticipation; placebo exchangeability and comparable treated/donor noise for uncertainty', **detail})


def synthcontrol(*, data: Any, y: str, treatment: str, panel: str, time: str,
                 model: str = 'sc', ridge: float = 0., zeta_omega: float | None = None,
                 zeta_lambda: float | None = None, placebo_reps: int = 100,
                 seed: int = 0, missing: str = 'raise', max_work: int = 200_000_000, alpha: float = .05):
    from openecon.analysis import fit
    return fit(make_spec('synthcontrol', outcome=y, predictors=[], panel=panel, time=time,
        columns={'treatment': treatment}, missing=missing, alpha=alpha,
        options={'model': model, 'ridge': ridge, 'zeta_omega': zeta_omega, 'zeta_lambda': zeta_lambda,
                 'placebo_reps': placebo_reps, 'seed': seed, 'max_work': max_work}), data=data)
