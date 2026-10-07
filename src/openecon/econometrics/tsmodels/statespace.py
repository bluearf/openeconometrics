"""Kalman filter and smoother of unobserved-components models with exact diffuse initialization.

State-space form (univariate observation, time-invariant system):

    y_t - x_t'b = Z alpha_t + e_t,              e_t ~ N(0, h)
    alpha_(t+1) = T alpha_t + eta_t,             eta_t ~ N(0, Q),  Q diagonal
    alpha_1 ~ N(0, kappa P_inf + P_star),        kappa -> infinity

The trend (level, slope) and seasonal states are diffuse (P_inf = I on their
block); the stochastic cycle is stationary with P_star = var(cycle) / (1 - rho^2) I.

Filter. The exact diffuse recursions of Durbin and Koopman (2012, sections 5.2
and 7.2) run while P_inf is nonzero: with F_inf = Z P_inf Z', F_star = Z P_star Z' + h,

    F_inf > 0:  K0 = T P_inf Z' / F_inf,  K1 = T P_star Z' / F_inf - K0 F_star / F_inf,
                a <- T a + K0 v,  P_inf <- T P_inf T' - (T P_inf Z') K0',
                P_star <- T P_star T' - (T P_star Z') K0' - (T P_inf Z') K1' + Q,
                log-likelihood term -1/2 (log 2 pi + log F_inf);
    F_inf = 0:  K0 = T P_star Z' / F_star, a <- T a + K0 v, P_inf <- T P_inf T',
                P_star <- T P_star T' - F_star K0 K0' + Q,
                term -1/2 (log 2 pi + log F_star + v^2 / F_star);

then the ordinary filter (term -1/2 (log 2 pi + log F + v^2 / F)). This is the
exact diffuse log likelihood (DK eq. 7.4), the definition statsmodels uses with
``use_exact_diffuse=True``.

Batching. Everything is batched over B parameter vectors (rows of the system
tensors) and C data columns (the outcome and each regressor: the gains do not
depend on the data, so one pass gives the innovations of y and of every column
of X, from which b is concentrated out by GLS). Numerical derivatives evaluate
all perturbed parameter vectors in ONE pass, so the per-period Python overhead
is paid once.

Steady state. The covariance recursion does not depend on the data. Once P_t
stops changing (relative change below 1e-14 in two successive periods), the gain
K and the innovation variance F are constant and the remaining innovations obey
the time-invariant recursion a_(t+1) = L a_t + K y_t with L = T - K Z. They are
then computed without a time loop:

    v_(t0+s) = y_(t0+s) - (Z L^s) a_(t0) - sum_(i<s) (Z L^(s-1-i) K) y_(t0+i),

the impulse responses Z L^j by repeated squaring (stopping once they are below
1e-18 of their initial size) and the sum by FFT convolution. Models whose
deterministic components (a state with zero disturbance variance, e.g. the drift
of ``rwdrift``) never reach a steady state run the recursion to the end.

Smoother. The fixed-interval smoother uses r_(t-1) = Z' v_t / F_t + L_t' r_t,
alpha^_t = a_t + P_t r_(t-1), and the exact diffuse smoothing recursions
(DK section 5.3) for the diffuse periods; in the steady-state stretch the
backward sums are again FFT correlations with the same impulse responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

MODELS = {   # level, level variance free, slope, slope variance free, irregular
    "none": (False, False, False, False, False),
    "ntrend": (False, False, False, False, True),
    "dconstant": (True, False, False, False, True),
    "llevel": (True, True, False, False, True),
    "rwalk": (True, True, False, False, False),
    "dtrend": (True, False, True, False, True),
    "lldtrend": (True, True, True, False, True),
    "rwdrift": (True, True, True, False, False),
    "lltrend": (True, True, True, True, True),
    "strend": (True, False, True, True, True),
    "rtrend": (True, False, True, True, False),
}
_STEADY_TOL = 1e-14
_NEGLIGIBLE = 1e-18
_DIFFUSE_TOL = 1e-10


@dataclass(frozen=True)
class Structure:
    """Which components a UC model has, and which of its variances are fixed at zero."""

    model: str
    seasonal: int = 0
    cycle: bool = False
    fixed_zero: frozenset[str] = field(default_factory=frozenset)

    @property
    def flags(self) -> tuple[bool, bool, bool, bool, bool]:
        return MODELS[self.model]

    @property
    def m(self) -> int:
        level, _, slope, _, _ = self.flags
        return int(level) + int(slope) + max(self.seasonal - 1, 0) + 2 * int(self.cycle)

    @property
    def season_index(self) -> int:
        level, _, slope, _, _ = self.flags
        return int(level) + int(slope)

    @property
    def cycle_index(self) -> int:
        return self.season_index + max(self.seasonal - 1, 0)

    @property
    def z(self) -> Tensor:
        row = torch.zeros(self.m, dtype=torch.float64)
        if self.flags[0]:
            row[0] = 1.0
        if self.seasonal:
            row[self.season_index] = 1.0
        if self.cycle:
            row[self.cycle_index] = 1.0
        return row

    @property
    def diffuse(self) -> Tensor:
        row = torch.zeros(self.m, dtype=torch.float64)
        row[:self.cycle_index] = 1.0
        return row

    def all_parameters(self) -> list[str]:
        level, level_free, slope, slope_free, irregular = self.flags
        names = ["frequency", "damping"] if self.cycle else []
        names += ["var(level)"] if level and level_free else []
        names += ["var(slope)"] if slope and slope_free else []
        names += ["var(seasonal)"] if self.seasonal else []
        names += ["var(cycle)"] if self.cycle else []
        names += ["var(e)"] if irregular else []
        return names

    @property
    def parameters(self) -> list[str]:
        return [name for name in self.all_parameters() if name not in self.fixed_zero]

    def base_transition(self) -> Tensor:
        m, level, slope = self.m, self.flags[0], self.flags[2]
        t = torch.zeros((m, m), dtype=torch.float64)
        if level:
            t[0, 0] = 1.0
        if slope:
            t[0, 1] = 1.0
            t[1, 1] = 1.0
        if self.seasonal:
            s, j = self.seasonal - 1, self.season_index
            t[j, j:j + s] = -1.0
            for i in range(1, s):
                t[j + i, j + i - 1] = 1.0
        return t


@dataclass(frozen=True)
class Layout:
    """Observation row and diffuse indicator of a (reduced) state vector."""

    z: Tensor
    diffuse: Tensor


def deterministic_states(structure: Structure) -> list[int]:
    """Diffuse states that no disturbance ever reaches (zero variance and only fed by such states).

    Their initial values act as diffuse regression coefficients: with
    W_t = (Z T^(t-1))[:, D] the model is y_t = W_t d + (the reduced model without D), and the
    exact diffuse likelihood equals de Jong's diffuse likelihood of the reduced model with d
    concentrated out plus -1/2 log det(S_WW) (verified to rounding in the tests).
    """
    level, level_free, slope, slope_free, _ = structure.flags
    quiet = set()
    if level and (not level_free or "var(level)" in structure.fixed_zero):
        quiet.add(0)
    if slope and (not slope_free or "var(slope)" in structure.fixed_zero):
        quiet.add(1)
    if structure.seasonal and "var(seasonal)" in structure.fixed_zero:
        quiet.update(range(structure.season_index, structure.cycle_index))
    elif structure.seasonal:
        quiet.update(range(structure.season_index + 1, structure.cycle_index))
    transition = structure.base_transition()
    changed = True
    while changed:
        changed = False
        for i in sorted(quiet):
            feeders = [j for j in range(structure.m) if j != i and transition[i, j] != 0]
            if any(j not in quiet for j in feeders):
                quiet.discard(i)
                changed = True
    return sorted(quiet)


def deterministic_regressors(structure: Structure, states: list[int], n: int) -> Tensor:
    """W [n, d] with W_t = (Z T^t)[states], t = 0..n-1 (trend and seasonal blocks only)."""
    if not states:
        return torch.empty((n, 0), dtype=torch.float64)
    rows = _power_rows(structure.z[None, :], structure.base_transition()[None], n)[0]
    if rows.shape[0] < n:
        rows = torch.cat([rows, torch.zeros((n - rows.shape[0], structure.m), dtype=torch.float64)])
    return rows[:, states].contiguous()


def natural(structure: Structure, theta: Tensor, scale: float) -> dict[str, Tensor]:
    """Free parameters [B, p] (log variance / logit) -> natural values per name, each [B]."""
    values = {}
    for i, name in enumerate(structure.parameters):
        column = theta[:, i]
        if name == "frequency":
            values[name] = math.pi * torch.sigmoid(column)
        elif name == "damping":
            values[name] = torch.sigmoid(column)
        else:
            values[name] = scale * torch.exp(column)
    return values


def unconstrained(structure: Structure, values: dict[str, float], scale: float) -> Tensor:
    out = []
    for name in structure.parameters:
        value = values[name]
        if name == "frequency":
            p = value / math.pi
            out.append(math.log(p / (1.0 - p)))
        elif name == "damping":
            out.append(math.log(value / (1.0 - value)))
        else:
            out.append(math.log(value / scale))
    return torch.tensor(out, dtype=torch.float64)


def jacobian_diagonal(structure: Structure, theta: Tensor, scale: float) -> Tensor:
    """d natural / d theta for each free parameter (the transforms are elementwise)."""
    out = []
    for i, name in enumerate(structure.parameters):
        value = float(theta[i])
        if name in ("frequency", "damping"):
            s = 1.0 / (1.0 + math.exp(-value))
            out.append((math.pi if name == "frequency" else 1.0) * s * (1.0 - s))
        else:
            out.append(scale * math.exp(value))
    return torch.tensor(out, dtype=torch.float64)


def system(structure: Structure, values: dict[str, Tensor], batch: int):
    """(T [B,m,m], q [B,m] = diag Q, h [B], P_star_1 [B,m,m]) for natural parameter values."""
    m = structure.m
    zeros = torch.zeros(batch, dtype=torch.float64)
    get = lambda name: values.get(name, zeros)          # noqa: E731 - fixed-zero variances
    t = structure.base_transition().expand(batch, m, m).clone()
    q = torch.zeros((batch, m), dtype=torch.float64)
    p0 = torch.zeros((batch, m, m), dtype=torch.float64)
    level, level_free, slope, slope_free, irregular = structure.flags
    if level and level_free:
        q[:, 0] = get("var(level)")
    if slope and slope_free:
        q[:, 1] = get("var(slope)")
    if structure.seasonal:
        q[:, structure.season_index] = get("var(seasonal)")
    if structure.cycle:
        j = structure.cycle_index
        lam, rho = values["frequency"], values["damping"]
        c, s = rho * torch.cos(lam), rho * torch.sin(lam)
        t[:, j, j], t[:, j, j + 1], t[:, j + 1, j], t[:, j + 1, j + 1] = c, s, -s, c
        var = get("var(cycle)")
        q[:, j], q[:, j + 1] = var, var
        stationary = var / (1.0 - rho * rho)
        p0[:, j, j], p0[:, j + 1, j + 1] = stationary, stationary
    h = get("var(e)") if irregular else zeros
    return t, q, h, p0


def _power_rows(start: Tensor, matrix: Tensor, length: int) -> Tensor:
    """Rows start @ matrix^j for j = 0..length-1 by doubling ([B, J, m], J <= length).

    Stops early once a new block is negligible; the returned J may be shorter.
    """
    rows = start[:, None, :]
    power = matrix
    reference = float(start.abs().max()) or 1.0
    while rows.shape[1] < length:
        block = rows @ power
        rows = torch.cat([rows, block], dim=1)
        if float(block.abs().max()) <= _NEGLIGIBLE * reference:
            break
        power = power @ power
    return rows[:, :length]


def _convolve(a: Tensor, b: Tensor, size: int) -> Tensor:
    """First ``size`` terms of the linear convolution along the last dimension (broadcast)."""
    total = a.shape[-1] + b.shape[-1] - 1
    length = 1 << max(1, (total - 1).bit_length())
    out = torch.fft.irfft(torch.fft.rfft(a, length) * torch.fft.rfft(b, length), length)
    return out[..., :size]


@dataclass
class FilterResult:
    v: Tensor                # [B, C, n] innovations (undefined where diffuse_inf)
    f: Tensor                # [B, n] innovation variances (1 where diffuse_inf)
    diffuse_inf: Tensor      # [n] bool: periods contributing -1/2 log F_inf only
    log_f_inf: Tensor        # [n] log F_inf on those periods, 0 elsewhere
    steady_from: int         # first period of the vectorized steady-state stretch (n: none)
    final_state: Tensor | None = None       # [m] a_(n+1) (store mode, C = 1)
    final_covariance: Tensor | None = None  # [m, m] P_(n+1)
    records: list | None = None
    predicted: Tensor | None = None         # [n, m] a_t (store mode)
    steady: dict | None = None


def kalman(structure: Structure, t: Tensor, q: Tensor, h: Tensor, p0: Tensor, data: Tensor, *,
           store: bool = False) -> FilterResult:
    """Exact diffuse Kalman filter, batched over parameter sets and data columns.

    ``data`` is [C, n]; ``t, q, h, p0`` carry the batch dimension B. With
    ``store=True`` (B = C = 1) the per-period quantities the smoother needs are kept.
    """
    batch, m = t.shape[0], t.shape[1]
    columns, n = data.shape
    if m == 0:
        if not bool(torch.isfinite(h).all()) or not bool((h > 0).all()):
            raise KernelError("degenerate_model", "A prediction-error variance is not positive finite.")
        return FilterResult(data[None].expand(batch, -1, -1).clone(),
                            h[:, None].expand(batch, n).clone(),
                            torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.float64), n,
                            final_state=torch.empty(0, dtype=torch.float64) if store else None,
                            final_covariance=torch.empty((0, 0), dtype=torch.float64) if store else None,
                            records=[] if store else None,
                            predicted=torch.empty((n, 0), dtype=torch.float64) if store else None)
    z = structure.z
    zi = z.nonzero().flatten()
    if zi.numel() == 0:
        raise KernelError("invalid_model", "The model has no component in the observation equation.")
    tt = t.transpose(1, 2)
    qm = torch.diag_embed(q)
    a = torch.zeros((batch, columns, m), dtype=torch.float64)
    p_inf = torch.diag(structure.diffuse)                  # shared by the batch
    t_shared = t[0]
    p_star = p0.clone()
    v_out = torch.zeros((batch, columns, n), dtype=torch.float64)
    f_out = torch.ones((batch, n), dtype=torch.float64)
    diffuse_inf = torch.zeros(n, dtype=torch.bool)
    log_f_inf = torch.zeros(n, dtype=torch.float64)
    records: list = [] if store else None
    predicted = torch.zeros((n, m), dtype=torch.float64) if store else None
    diffuse_phase = bool(p_inf.abs().max() > 0)
    calm = 0
    step = 0
    steady_from = n
    gain = None
    variance = None
    while step < n and diffuse_phase:
        y = data[:, step]
        v = y - a[..., zi].sum(-1)                          # [B, C]
        if store:
            predicted[step] = a[0, 0]
        m_star = p_star[..., zi].sum(-1)                    # [B, m]
        f_star = m_star[:, zi].sum(-1) + h                  # [B]
        tm_star = (t @ m_star[..., None]).squeeze(-1)       # [B, m]
        m_inf = p_inf[:, zi].sum(-1)
        f_inf = float(m_inf[zi].sum())
        tm_inf = t_shared @ m_inf
        if f_inf > _DIFFUSE_TOL:
            k0 = tm_inf / f_inf
            k1 = tm_star / f_inf - k0[None] * (f_star / f_inf)[:, None]
            a = a @ tt + v[..., None] * k0
            new_inf = t_shared @ p_inf @ t_shared.T - torch.outer(tm_inf, k0)
            p_star_new = t @ p_star @ tt - tm_star[:, :, None] * k0[None, None, :] \
                - tm_inf[None, :, None] * k1[:, None, :] + qm
            diffuse_inf[step] = True
            log_f_inf[step] = math.log(f_inf)
            if store:
                records.append(("inf", p_star[0].clone(), p_inf.clone(), float(v[0, 0]),
                                f_inf, k0.clone(), k1[0].clone()))
        else:
            if not bool((f_star > 0).all()):
                raise KernelError("degenerate_model", "A prediction-error variance is zero.")
            k0 = tm_star / f_star[:, None]
            a = a @ tt + v[..., None] * k0[:, None, :]
            new_inf = t_shared @ p_inf @ t_shared.T
            p_star_new = t @ p_star @ tt + qm \
                - f_star[:, None, None] * k0[:, :, None] * k0[:, None, :]
            v_out[..., step], f_out[:, step] = v, f_star
            if store:
                records.append(("zero", p_star[0].clone(), p_inf.clone(), float(v[0, 0]),
                                float(f_star[0]), k0[0].clone(), None))
        p_inf = (new_inf + new_inf.T) / 2
        p_star = (p_star_new + p_star_new.transpose(1, 2)) / 2
        if float(p_inf.abs().max()) <= _DIFFUSE_TOL:
            p_inf = torch.zeros_like(p_inf)
            diffuse_phase = False
        step += 1
    # Ordinary filter: the few operations per period are fused (one T P product feeds both
    # T P Z' and T P T'); positivity and finiteness are checked once at the end, symmetry is
    # restored every 16 periods and the steady state is tested every 4 periods.
    zc = z.to(t.dtype)
    v_list, f_list = [], []
    first = step
    previous = None
    while step < n:
        v = data[:, step] - a @ zc                           # [B, C]
        if store:
            predicted[step] = a[0, 0]
        tp = t @ p_star
        tm_star = tp @ zc                                    # T P Z'  [B, m]
        f_star = (p_star @ zc) @ zc + h
        gain = tm_star / f_star[:, None]
        v_list.append(v)
        f_list.append(f_star)
        if store:
            records.append(("std", p_star[0].clone(), None, float(v[0, 0]), float(f_star[0]),
                            gain[0].clone(), None))
        a = torch.baddbmm(v[..., None] * gain[:, None, :], a, tt)
        new = torch.baddbmm(tp @ tt + qm, gain[:, :, None], tm_star[:, None, :], alpha=-1.0)
        step += 1
        if (step - first) % 16 == 0:
            new = (new + new.transpose(1, 2)) / 2
        if (step - first) % 4 == 3:
            previous = p_star
        elif previous is not None:
            change = float((new - p_star).abs().max())
            size = float(new.abs().max())
            calm = calm + 1 if change <= _STEADY_TOL * max(size, 1e-300) else 0
            previous = None
            if not math.isfinite(change):
                break
        p_star = new
        if calm >= 2 and step < n:
            steady_from = step
            variance = f_star
            break
    if v_list:
        v_out[..., first:step] = torch.stack(v_list, dim=-1)
        f_out[:, first:step] = torch.stack(f_list, dim=-1)
        if not bool(torch.isfinite(f_out[:, first:step]).all()) \
                or not bool((f_out[:, first:step] > 0).all()):
            raise KernelError("degenerate_model", "A prediction-error variance is zero or not "
                              "finite.")
    result = FilterResult(v_out, f_out, diffuse_inf, log_f_inf, steady_from, records=records,
                          predicted=predicted)
    if steady_from < n:
        _steady_remainder(structure, t, gain, variance, a, data, result, store, p_star)
    elif store:
        result.final_state = a[0, 0].clone()
        result.final_covariance = p_star[0].clone()
    return result


def _steady_remainder(structure: Structure, t: Tensor, gain: Tensor, variance: Tensor, a: Tensor,
                      data: Tensor, result: FilterResult, store: bool, p_star: Tensor) -> None:
    """Innovations of the time-invariant stretch without a time loop (see the module docstring)."""
    start = result.steady_from
    rest = data[:, start:]                                   # [C, R]
    length = rest.shape[1]
    z = structure.z
    batch, m = t.shape[0], t.shape[1]
    lmat = t - gain[:, :, None] * z[None, None, :]           # L = T - K Z  [B, m, m]
    h_rows = _power_rows(z.expand(batch, m), lmat, length)   # Z L^j        [B, J, m]
    g = (h_rows @ gain[:, :, None]).squeeze(-1)              # Z L^j K      [B, J]
    reach = h_rows.shape[1]
    homogeneous = torch.einsum("bjm,bcm->bcj", h_rows, a)    # [B, C, J]
    v = rest[None].expand(batch, -1, -1).clone()
    v[..., :reach] -= homogeneous
    if length > 1:
        response = _convolve(g[:, None, :], rest[None], length - 1)     # [B, C, R-1]
        v[..., 1:] -= response
    result.v[..., start:] = v
    result.f[:, start:] = variance[:, None]
    if store:
        # Predicted states a_(t0+s) = L^s a_(t0) + sum_(i<s) L^(s-1-i) K y_(t0+i), per component.
        lt = lmat.transpose(1, 2)
        homogeneous_rows = _power_rows(a[:, 0, :], lt, length + 1)       # (L^s a)'  [1, J', m]
        impulse = _power_rows(gain, lt, length)                         # (L^j K)'  [1, J, m]
        states = torch.zeros((length + 1, m), dtype=torch.float64)
        states[:homogeneous_rows.shape[1]] = homogeneous_rows[0]
        conv = _convolve(impulse[0].T, rest[0][None, :], length)        # [m, R]
        states[1:] += conv.T
        result.predicted[start:] = states[:length]
        result.final_state = states[length].clone()
        result.final_covariance = p_star[0].clone()
        result.steady = {"gain": gain[0].clone(), "variance": float(variance[0]),
                         "covariance": p_star[0].clone(), "h_rows": h_rows[0].clone()}


def smooth(structure: Structure, t: Tensor, q: Tensor, h: Tensor, p0: Tensor, y: Tensor) -> Tensor:
    """Smoothed states alpha^_t = E[alpha_t | y_1..y_n], [n, m] (one parameter set, one series)."""
    if structure.m == 0:
        return torch.empty((y.shape[0], 0), dtype=torch.float64)
    run = kalman(structure, t, q, h, p0, y[None, :], store=True)
    n, m = y.shape[0], structure.m
    z = structure.z
    tm = t[0]
    smoothed = torch.zeros((n, m), dtype=torch.float64)
    r = torch.zeros(m, dtype=torch.float64)
    start = run.steady_from
    if start < n:
        steady = run.steady
        u = run.v[0, 0, start:] / steady["variance"]                    # [R]
        rows = steady["h_rows"]                                         # Z L^j  [J, m]
        # r_(t-1) = sum_j (L')^j Z' u_(t+j): a correlation of u with the rows Z L^j.
        length = u.shape[0]
        reversed_u = u.flip(0)
        corr = _convolve(rows.T, reversed_u[None, :], length)           # [m, R]
        backward = corr.flip(1).T                                       # r_(t0+s-1), s = 0..R-1
        smoothed[start:] = run.predicted[start:] + backward @ steady["covariance"].T
        r = backward[0]
    r1 = None
    for step in range(min(start, n) - 1, -1, -1):
        kind, p_star, p_inf, v, f, k0, k1 = run.records[step]
        a = run.predicted[step]
        if kind == "std":
            r = z * (v / f) + tm.T @ r - z * float(k0 @ r)
            smoothed[step] = a + p_star @ r
            continue
        if r1 is None:
            r1 = torch.zeros(m, dtype=torch.float64)
        if kind == "zero":
            r0 = z * (v / f) + tm.T @ r - z * float(k0 @ r)
            r1 = tm.T @ r1
        else:
            r0 = tm.T @ r - z * float(k0 @ r)
            r1 = z * (v / f) + tm.T @ r1 - z * float(k0 @ r1) - z * float(k1 @ r)
        r = r0
        smoothed[step] = a + p_star @ r0 + p_inf @ r1
    return smoothed
