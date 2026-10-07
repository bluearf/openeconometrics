"""Cox partial likelihood on float64 tensors: Breslow, Efron and exact (``exactp``) ties.

Notation: records i with entry ``t0_i``, exit ``t_i``, failure ``delta_i``,
weight ``w_i`` and risk score ``u_i = exp(eta_i)``, ``eta_i = x_i'b + offset_i``;
failure times ``tau_k`` with failure set ``D_k`` (``c_k`` records, weighted
count ``d_k``) and risk set ``R_k = {i: t0_i < tau_k <= t_i}`` in the stratum
of ``tau_k``. ``S0_k = sum_{R_k} w u``, ``S1_k = sum_{R_k} w u x``.

Breslow (Stata's default; weights allowed):

    ln L = sum_k [ sum_{i in D_k} w_i eta_i - d_k ln S0_k ]
    U    = sum_i w_i delta_i x_i - sum_k d_k S1_k / S0_k
    H    = -sum_k d_k [ S2_k / S0_k - S1_k S1_k' / S0_k^2 ]

Efron (no weights, as in Stata): the c_k tied failures see the denominators
``phi_kr = S0_k - (r / c_k) D0_k``, r = 0..c_k-1, with ``D0_k = sum_{D_k} u``:

    ln L = sum_k [ sum_{i in D_k} eta_i - sum_r ln phi_kr ].

Exact partial likelihood (``exactp``): a failure time with one failure
contributes its Breslow term; a tied failure time contributes the conditional
logit likelihood of its risk set given ``c_k`` failures, evaluated with the
discrete family's elementary-symmetric-function recursion
(``discrete.conditional.ConditionalLogitObjective``) on the expanded risk sets.

No per-time k-by-k matrix is formed: the S2 term is
``sum_k a_k S2_k = X' diag(w u A) X`` with ``A_i = sum_{k: i in R_k} a_k``
(``RiskSets.accumulate``), and the outer-product terms are
``S1' diag(.) S1`` products over the T failure times. Cost per iteration
O(n k^2) after one O(n log n) sort.

Residuals (Stata [ST] stcox postestimation; Therneau and Grambsch 2000):
Schoenfeld ``r_i = x_i - xbar_k`` for a failing record (Efron: xbar is the
average of the c_k Efron means), and the efficient score residuals of Lin and
Wei (1989)

    W_i = delta_i (x_i - xbar_{k(i)}) - u_i sum_{k: i in R_k} a_k (x_i - xbar_k)

(Efron: with the down-weighted risk-set membership ``1 - r/c_k`` of the tied
failures), which satisfy ``sum_i w_i W_i = U``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.survival.data import FLOAT, RiskSets
from openecon.engines.contracts import KernelError

# Largest number of (record, tied failure time) rows the exact partial likelihood expands.
EXACT_LIMIT = 5_000_000
# Largest sum over tied times of |R_k| min(c_k, |R_k| - c_k) (elementary-symmetric recursion).
EXACT_WORK_LIMIT = 2e7


class CoxObjective:
    """Partial log likelihood with analytic gradient and Hessian (``maximize_newton`` API).

    ``include`` ([T] bool) restricts the likelihood to some failure times (the
    exact method evaluates its tied times separately).
    """

    def __init__(self, x: Tensor, risk: RiskSets, weights: Tensor | None, offset: Tensor | None,
                 ties: str, include: Tensor | None = None):
        if ties not in {"breslow", "efron"}:
            raise KernelError("invalid_option", f"Unknown ties method {ties!r}.")
        self.x, self.risk, self.ties = x, risk, ties
        n = x.shape[0]
        self.w = torch.ones(n, dtype=FLOAT) if weights is None else weights
        self.offset = offset
        failing = risk.event_of >= 0
        if include is not None:
            failing = failing & include[risk.event_of.clamp_min(0)]
        self.failing = failing
        self.fail_float = failing.to(FLOAT)
        self.event_of = torch.where(failing, risk.event_of, torch.full_like(risk.event_of, -1))
        self.delta_w = self.w * self.fail_float
        self.d = risk.at_event(self.delta_w)                       # weighted failures per time
        self.g_const = (x * self.delta_w[:, None]).sum(0)
        counts = risk.at_event(self.fail_float).round().to(torch.int64)
        self.counts = counts
        if ties == "efron":
            times = torch.arange(risk.n_events)
            self.term_time = torch.repeat_interleave(times, counts)
            starts = torch.cumsum(counts, 0) - counts
            position = torch.arange(self.term_time.numel()) - starts[self.term_time]
            self.term_frac = position.to(FLOAT) / counts[self.term_time].to(FLOAT)

    # ---- pieces ---------------------------------------------------------------------

    def eta(self, beta: Tensor) -> Tensor:
        eta = self.x @ beta
        return eta if self.offset is None else eta + self.offset

    def _state(self, beta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shifted linear predictor (max 0), risk scores u, weighted v = w u and S0."""
        eta = self.eta(beta)
        eta = eta - eta.max()
        u = torch.exp(eta)
        v = self.w * u
        return eta, u, v, self.risk.at_risk(v)

    def _efron_phi(self, v: Tensor, s0: Tensor) -> tuple[Tensor, Tensor]:
        d0 = self.risk.at_event(v * self.fail_float)
        return s0[self.term_time] - self.term_frac * d0[self.term_time], d0

    def value(self, beta: Tensor) -> Tensor:
        eta, _, v, s0 = self._state(beta)
        if self.ties == "breslow":
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            return (self.delta_w * eta).sum() - (self.d * torch.log(safe)).sum()
        phi, _ = self._efron_phi(v, s0)
        return (self.delta_w * eta).sum() - torch.log(phi).sum()

    def __call__(self, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, risk = self.x, self.risk
        eta, _, v, s0 = self._state(beta)
        s1 = risk.at_risk(v[:, None] * x)                          # [T, k]
        if self.ties == "breslow":
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            value = (self.delta_w * eta).sum() - (self.d * torch.log(safe)).sum()
            a = self.d / safe
            weight = v * risk.accumulate(a)
            gradient = self.g_const - x.T @ weight
            hessian = -(x * weight[:, None]).T @ x + (s1 * (a / safe)[:, None]).T @ s1
            return value, gradient, (hessian + hessian.T) / 2
        phi, _ = self._efron_phi(v, s0)
        value = (self.delta_w * eta).sum() - torch.log(phi).sum()
        inverse = 1.0 / phi
        frac = self.term_frac
        total = self.risk.n_events

        def per_time(values: Tensor) -> Tensor:
            return torch.zeros(total, dtype=FLOAT).index_add_(0, self.term_time, values)

        a, b = per_time(inverse), per_time(frac * inverse)
        alpha2, beta2 = per_time(inverse.square()), per_time(frac * inverse.square())
        gamma2 = per_time(frac.square() * inverse.square())
        d1 = risk.at_event(v[:, None] * x * self.fail_float[:, None])
        b_rows = torch.where(self.failing, b[self.event_of.clamp_min(0)], torch.zeros_like(v))
        weight = v * risk.accumulate(a) - v * b_rows
        gradient = self.g_const - x.T @ weight
        cross = (s1 * beta2[:, None]).T @ d1
        hessian = (-(x * weight[:, None]).T @ x + (s1 * alpha2[:, None]).T @ s1
                   - cross - cross.T + (d1 * gamma2[:, None]).T @ d1)
        return value, gradient, (hessian + hessian.T) / 2

    # ---- residuals --------------------------------------------------------------------

    def residuals(self, beta: Tensor) -> tuple[Tensor, Tensor]:
        """Efficient score residuals W [n, k] (before weights) and per-time means [T, k].

        The Schoenfeld residual of failing record i is ``x_i - means[k(i)]``.
        """
        x, risk = self.x, self.risk
        _, u, v, s0 = self._state(beta)
        s1 = risk.at_risk(v[:, None] * x)
        fail = self.failing
        ev = self.event_of.clamp_min(0)
        if self.ties == "breslow":
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            means = s1 / safe[:, None]
            a = self.d / safe
            accumulated = risk.accumulate(a)
            pulled = risk.accumulate(a[:, None] * means)
            first = torch.where(fail[:, None], x - means[ev], torch.zeros_like(x))
            return first - u[:, None] * (x * accumulated[:, None] - pulled), means
        phi, _ = self._efron_phi(v, s0)
        d1 = risk.at_event(v[:, None] * x * self.fail_float[:, None])
        frac, terms = self.term_frac, self.term_time
        term_means = (s1[terms] - frac[:, None] * d1[terms]) / phi[:, None]
        total = risk.n_events

        def per_time(values: Tensor) -> Tensor:
            out = torch.zeros((total, *values.shape[1:]), dtype=FLOAT)
            return out.index_add_(0, terms, values)

        inverse = 1.0 / phi
        counts = self.counts.clamp_min(1).to(FLOAT)
        means = per_time(term_means) / counts[:, None]
        accumulated = risk.accumulate(per_time(inverse))
        pulled = risk.accumulate(per_time(inverse[:, None] * term_means))
        b = per_time(frac * inverse)
        b1 = per_time((frac * inverse)[:, None] * term_means)
        own = torch.where(fail[:, None], x * b[ev][:, None] - b1[ev], torch.zeros_like(x))
        first = torch.where(fail[:, None], x - means[ev], torch.zeros_like(x))
        return first - u[:, None] * (x * accumulated[:, None] - pulled) + u[:, None] * own, means

    def baseline(self, beta: Tensor, shift: float, iterations: int = 100
                 ) -> tuple[Tensor, Tensor]:
        """Baseline hazard increments and Kalbfleisch-Prentice log factors at covariates zero.

        ``shift`` is ``m'b``, the linear predictor of the centring point, so that the
        risk scores at covariates (and offset) zero are ``exp(eta_i + shift)`` for the
        centred design. Returns, per failure time:

        * the cumulative-hazard increment: Breslow ``d_k / S0_k``, or for Efron ties the
          analogue ``sum_{r<c_k} 1 / (S0_k - (r/c_k) D0_k)`` (Stata: after ``efron`` "all
          predictions are carried out using the Efron method");
        * ``ln alpha_k`` of Stata's ``predict basesurv`` (Kalbfleisch and Prentice 2002,
          eq. 4.34): ``alpha_k`` solves ``sum_{i in D_k} w_i u_i / (1 - alpha^{u_i}) =
          S0_k``. With ``psi = S0_k ln alpha`` and ``r_i = u_i / S0_k`` the equation
          ``sum w r / (1 - exp(r psi)) = 1`` is scale free; its left side is increasing
          and convex in ``psi < 0``, so Newton's method from Stata's starting value
          ``psi = -d_k`` (which lies right of the root) converges monotonically. A time
          at which everyone at risk fails has ``alpha_k = 0`` (``-inf``).
        """
        risk = self.risk
        eta = self.eta(beta)
        top = eta.max()
        u = torch.exp(eta - top)
        v = self.w * u
        s0 = risk.at_risk(v)
        log_scale = float(top) + shift              # scaled sums -> sums at covariates zero
        if self.ties == "efron":
            phi, _ = self._efron_phi(v, s0)
            per_time = torch.zeros(risk.n_events, dtype=FLOAT).index_add_(
                0, self.term_time, 1.0 / phi)
            increments = torch.exp(torch.log(per_time) - log_scale)
        else:
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            increments = torch.where(
                self.d > 0, torch.exp(torch.log(self.d.clamp_min(1e-300)) - torch.log(safe)
                                      - log_scale), torch.zeros_like(s0))
        fail = self.failing
        k = self.event_of[fail]
        r = u[fail] / s0[k]
        wf = self.w[fail]
        dead = torch.zeros(risk.n_events, dtype=FLOAT).index_add_(0, k, wf * r)
        survivors = dead < 1.0 - 1e-12              # someone at risk does not fail
        psi = torch.where(survivors, -self.d, -torch.ones_like(self.d))
        for _ in range(iterations):
            x = r * psi[k]
            q = r / (-torch.expm1(x))               # r / (1 - e^{r psi}), O(1/|psi|)
            g = torch.zeros_like(psi).index_add_(0, k, wf * q) - 1.0
            slope = torch.zeros_like(psi).index_add_(0, k, wf * q * q * torch.exp(x))
            step = torch.where(survivors, g / slope, torch.zeros_like(psi))
            psi = psi - step
            if bool((step.abs() <= 1e-15 * psi.abs()).all()):
                break
        log_alpha = psi * torch.exp(-(torch.log(s0) + log_scale))
        return increments, torch.where(survivors, log_alpha, torch.full_like(psi, -math.inf))

    def log_risk_sums(self, beta: Tensor) -> tuple[Tensor, Tensor]:
        """``ln S0_k`` of the unshifted linear predictor, and the weighted failures d_k."""
        eta = self.eta(beta)
        shift = eta.max()
        s0 = self.risk.at_risk(self.w * torch.exp(eta - shift))
        return torch.log(s0) + shift, self.d


class ExactObjective:
    """Exact partial likelihood: Breslow terms of single failures + conditional logits of ties.

    Tied failure time k becomes a conditional-logit group made of its risk set,
    with outcome 1 for the c_k failures. The expansion has ``sum_{tied k} |R_k|``
    rows; more than ``EXACT_LIMIT`` rows, or more than ``EXACT_WORK_LIMIT`` recursion
    steps, raise ``exact_too_large``.
    """

    def __init__(self, x: Tensor, risk: RiskSets, failure: Tensor, offset: Tensor | None):
        from openecon.econometrics.discrete.conditional import ConditionalLogitObjective

        counts = risk.at_event((risk.event_of >= 0).to(FLOAT)).round().to(torch.int64)
        at_risk = risk.at_risk(torch.ones(x.shape[0], dtype=FLOAT)).round().to(torch.int64)
        # A tied time at which everyone at risk fails has conditional likelihood 1.
        tied = (counts >= 2) & (at_risk > counts)
        self.single = CoxObjective(x, risk, None, offset, "breslow", include=counts < 2)
        self.tied_times = int(tied.sum())
        self.group = None
        if self.tied_times == 0:
            return
        width = torch.minimum(counts, at_risk - counts)[tied].to(FLOAT)
        work = float((at_risk[tied].to(FLOAT) * width).sum())
        if work > EXACT_WORK_LIMIT:
            raise KernelError(
                "exact_too_large",
                f"The exact partial likelihood of these heavily tied data needs about {work:.1e} "
                f"recursion steps per iteration (limit {EXACT_WORK_LIMIT:.0e}): its cost grows "
                "with the risk-set size times the number of tied failures. Use ties='efron', "
                "which approximates it closely.")
        tied_prefix = torch.zeros(risk.n_events + 1, dtype=torch.int64)
        tied_prefix[1:] = torch.cumsum(tied.to(torch.int64), 0)
        tied_index = tied.nonzero().flatten()
        low = tied_prefix[risk.entry_count]
        sizes = tied_prefix[risk.exit_count] - low
        total = int(sizes.sum())
        if total > EXACT_LIMIT:
            raise KernelError(
                "exact_too_large",
                f"The exact partial likelihood would expand the {self.tied_times} tied failure "
                f"times into {total:,} risk-set rows (limit {EXACT_LIMIT:,}). Use "
                "ties='efron', which approximates it closely, or coarsen fewer times.")
        rows = torch.repeat_interleave(torch.arange(x.shape[0]), sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(total) - starts[rows]
        group = low[rows] + local                                  # rank among tied times
        time = tied_index[group]
        chosen = ((risk.event_of[rows] == time) & (failure[rows] > 0)).to(FLOAT)
        self.group = ConditionalLogitObjective(
            x[rows], chosen, group, self.tied_times, torch.ones(self.tied_times, dtype=FLOAT),
            None if offset is None else offset[rows])
        self.expanded_rows = total

    def value(self, beta: Tensor) -> Tensor:
        value = self.single.value(beta)
        return value if self.group is None else value + self.group.value(beta)

    def __call__(self, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, gradient, hessian = self.single(beta)
        if self.group is not None:
            extra = self.group(beta)
            value, gradient, hessian = value + extra[0], gradient + extra[1], hessian + extra[2]
        return value, gradient, hessian
