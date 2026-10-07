"""Float64 fused ARCH-family state from fixed trusted compiler source.

Only a finite lag ring and its analytic tangents survive numerical blocks.
Compilation does not depend on inspectable Python source in a packaged app.
"""
from functools import lru_cache

import torch

_SOURCE = r'''
def state_block(y: Tensor, x: Tensor, z: Tensor, theta: Tensor, state: Tensor,
                derivatives: Tensor, cursor: int, kind: int, mean_kind: int,
                mean_index: List[int], variance_index: List[int], constant_index: int,
                a_index: List[int], g_index: List[int], b_index: List[int],
                ar_index: List[int], ma_index: List[int], arch_lags: List[int],
                garch_lags: List[int], ar_lags: List[int], ma_lags: List[int],
                power_index: int, in_mean_index: int, signs: Tensor):
    n, k, longest = y.shape[0], theta.shape[0], state.shape[1]
    residual = torch.empty(n, dtype=torch.float64)
    disturbance = torch.empty(n, dtype=torch.float64)
    variance = torch.empty(n, dtype=torch.float64)
    d_residual = torch.empty((n, k), dtype=torch.float64)
    d_log_variance = torch.empty((n, k), dtype=torch.float64)
    eye = torch.eye(k, dtype=torch.float64)
    power = float(theta[power_index]) if power_index >= 0 else 2.
    for t in range(n):
        m = y[t]
        dm = torch.zeros(k, dtype=torch.float64)
        for j in range(len(mean_index)):
            m = m-x[t, j]*theta[mean_index[j]]
            dm[mean_index[j]] = -x[t, j]
        v = theta[constant_index]
        dv = eye[constant_index].clone()
        for j in range(len(variance_index)):
            v = v+z[t, j]*theta[variance_index[j]]
            dv[variance_index[j]] = z[t, j]
        if len(variance_index) and kind != 2:
            v = v.exp()
            dv = dv*v
        for j in range(len(a_index)):
            lag = (cursor-arch_lags[j]) % longest
            v = v+theta[a_index[j]]*state[0, lag]
            dv = dv+eye[a_index[j]]*state[0, lag]+theta[a_index[j]]*derivatives[0, lag]
            if len(g_index):
                v = v+theta[g_index[j]]*state[1, lag]
                dv = dv+eye[g_index[j]]*state[1, lag]+theta[g_index[j]]*derivatives[1, lag]
        for j in range(len(b_index)):
            lag = (cursor-garch_lags[j]) % longest
            v = v+theta[b_index[j]]*state[2, lag]
            dv = dv+eye[b_index[j]]*state[2, lag]+theta[b_index[j]]*derivatives[2, lag]
        if kind == 2:
            h, dlogh = v.exp(), dv
        elif kind == 3:
            h = v.pow(2./power)
            dlogh = (2./power)*dv/v
            dlogh[power_index] = dlogh[power_index]-(2./(power*power))*v.log()
        else:
            h, dlogh = v, dv/v
        if not bool(torch.isfinite(h)) or not float(h) > 0. or (kind != 2 and not float(v) > 0.):
            return residual, disturbance, variance, d_residual, d_log_variance, cursor, False
        u, du = m, dm
        if in_mean_index >= 0:
            if mean_kind == 1:
                gm, dg = h, h*dlogh
            elif mean_kind == 2:
                gm, dg = h.sqrt(), .5*h.sqrt()*dlogh
            else:
                gm, dg = h.log(), dlogh
            u, du = m-theta[in_mean_index]*gm, dm-theta[in_mean_index]*dg-eye[in_mean_index]*gm
        e, de = u, du.clone()
        for j in range(len(ar_index)):
            lag = (cursor-ar_lags[j]) % longest
            e = e-theta[ar_index[j]]*state[3, lag]
            de = de-theta[ar_index[j]]*derivatives[3, lag]-eye[ar_index[j]]*state[3, lag]
        for j in range(len(ma_index)):
            lag = (cursor-ma_lags[j]) % longest
            e = e-theta[ma_index[j]]*state[4, lag]
            de = de-theta[ma_index[j]]*derivatives[4, lag]-eye[ma_index[j]]*state[4, lag]
        if kind == 2:
            standard = e/h.sqrt()
            d_standard = de/h.sqrt()-.5*standard*dlogh
            sign = float(signs[t]) if signs.numel() else (-1. if float(standard) < 0. else 1.)
            news, dnews = standard, d_standard
            news2, dnews2 = sign*standard-.7978845608028654, sign*d_standard
        elif kind == 3:
            news = e.abs().pow(power)
            dnews = torch.zeros(k, dtype=torch.float64)
            if float(e) != 0.:
                dnews = power*e.abs().pow(power-1.)*e.sign()*de
                dnews[power_index] = dnews[power_index]+news*e.abs().log()
            news2, dnews2 = torch.zeros((), dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        else:
            news, dnews = e.square(), 2.*e*de
            news2 = news if float(e) < 0. else torch.zeros((), dtype=torch.float64)
            dnews2 = dnews if float(e) < 0. else torch.zeros(k, dtype=torch.float64)
        slot = cursor % longest
        state[0, slot], derivatives[0, slot] = news, dnews
        state[1, slot], derivatives[1, slot] = news2, dnews2
        state[2, slot], derivatives[2, slot] = v, dv
        state[3, slot], derivatives[3, slot] = u, du
        state[4, slot], derivatives[4, slot] = e, de
        residual[t], disturbance[t], variance[t] = e, u, h
        d_residual[t], d_log_variance[t] = de, dlogh
        cursor += 1
    return residual, disturbance, variance, d_residual, d_log_variance, cursor, True
'''


@lru_cache(maxsize=1)
def _kernel():
    return torch.jit.CompilationUnit(_SOURCE)


def state_block(*args):
    return _kernel().state_block(*args)
