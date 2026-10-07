"""Hamilton filter, Kim smoother and the analytic score of Markov-switching regressions.

Model (Stata's ``mswitch dr`` / ``mswitch ar``): with s_t in {1..k} a Markov chain with
transition probabilities p_ab = P(s_t = b | s_(t-1) = a),

    y_t = mu_t(s_t) + sum_(i in lags) phi_i(s_t) (y_(t-i) - mu_(t-i)(s_(t-i))) + e_t,
    mu_t(s) = x_t' beta_s + w_t' alpha,       e_t ~ N(0, sigma_(s_t)^2).

With autoregressive lags the density depends on S_t = (s_t, s_(t-1), ..., s_(t-P)), a
Markov chain on K = k^(P+1) "expanded" states with transition A[S', S] = p_(s'_0, s_0)
when the shared lags agree. The chain starts from its ergodic distribution
pi_exp(S) = pi(s_P) prod_l p_(s_l, s_(l-1)) at a latent period 0 before the first
observation (``p0(transition)`` in Stata), so the likelihood is

    L = 1' prod_t (diag(eta_t) A') pi_exp,      eta_t[S] = N(e_t(S); 0, sigma_(s_0)^2).

Hamilton filter without a per-period Python loop: periods are cut into blocks of
about sqrt(n); (1) the products of the block's matrices diag(eta_t) A' are formed
for all blocks at once, normalized at every step; (2) a short loop over blocks gives
the exact filtered probabilities at every block start; (3) all blocks are filtered in
parallel from those starts, giving xi_(t|t-1), xi_(t|t) and log f(y_t | Y_(t-1)).
The Kim smoother xi_(t|T) = xi_(t|t) * A (xi_(t+1|T) / xi_(t+1|t)) is linear in
xi_(t+1|T) and is evaluated by the same blocked scheme backwards. Densities are
scaled by their per-period maximum, so nothing underflows.

Score (Fisher's identity, exact): dll/dtheta = E[d log p(Y, S) / dtheta | Y], i.e.
sum_t sum_S xi_(t|T)(S) d log eta_t[S] + sum_(ab) N_ab d log p_ab + the initial term
with the derivative of the ergodic distribution, d pi' = pi' dP Z,
Z = (I - P + 1 pi')^-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError


@dataclass
class Layout:
    """Index bookkeeping of one Markov-switching specification."""

    states: int
    lags: list[int]                  # autoregressive lags (sorted, may be empty)
    switching: int                   # number of switching regressors
    common: int                      # number of common regressors
    arswitch: bool
    varswitch: bool

    @property
    def order(self) -> int:
        return max(self.lags, default=0)

    @property
    def expanded(self) -> int:
        return self.states ** (self.order + 1)

    def lag_state(self) -> Tensor:
        """[P+1, K] state at lag l of each expanded state (S = sum_l s_l k^l)."""
        k, codes = self.states, torch.arange(self.expanded)
        return torch.stack([(codes // k ** lag) % k for lag in range(self.order + 1)])

    def sizes(self) -> dict[str, int]:
        k, q = self.states, len(self.lags)
        return {"beta": k * self.switching, "alpha": self.common,
                "phi": (k if self.arswitch else 1) * q, "lnsigma": k if self.varswitch else 1,
                "logit": k * (k - 1)}

    def split(self, theta: Tensor) -> dict[str, Tensor]:
        out, start = {}, 0
        for name, size in self.sizes().items():
            out[name] = theta[start:start + size]
            start += size
        k = self.states
        out["beta"] = out["beta"].reshape(k, self.switching)
        out["phi"] = out["phi"].reshape(k if self.arswitch else 1, len(self.lags))
        out["logit"] = out["logit"].reshape(k, k - 1)
        return out

    @property
    def size(self) -> int:
        return sum(self.sizes().values())


def transition(logit: Tensor) -> Tensor:
    """p_ab = exp(q_ab) / (1 + sum_c exp(q_ac)), the last state the base."""
    full = torch.cat([logit, torch.zeros((logit.shape[0], 1), dtype=torch.float64)], dim=1)
    return torch.softmax(full, dim=1)


def ergodic(p: Tensor) -> tuple[Tensor, Tensor]:
    """Stationary distribution pi and the fundamental matrix Z = (I - P + 1 pi')^-1."""
    k = p.shape[0]
    system = torch.eye(k, dtype=torch.float64) - p.T
    system[-1] = 1.0
    rhs = torch.zeros(k, dtype=torch.float64)
    rhs[-1] = 1.0
    pi = torch.linalg.solve(system, rhs)
    fundamental = torch.linalg.inv(torch.eye(k, dtype=torch.float64) - p + pi[None, :])
    return pi, fundamental


def expanded_chain(layout: Layout, p: Tensor, pi: Tensor) -> tuple[Tensor, Tensor]:
    """Expanded transition A [K, K] and initial distribution pi_exp [K]."""
    lag = layout.lag_state()
    K, order = layout.expanded, layout.order
    newest_from, newest_to = lag[0][:, None], lag[0][None, :]
    consistent = torch.ones((K, K), dtype=torch.bool)
    for level in range(1, order + 1):
        consistent &= lag[level - 1][:, None] == lag[level][None, :]
    a = torch.where(consistent, p[newest_from, newest_to], torch.zeros((), dtype=torch.float64))
    start = pi[lag[order]]
    for level in range(order, 0, -1):
        start = start * p[lag[level], lag[level - 1]]
    return a, start


@dataclass
class Pieces:
    residual: Tensor     # [n, K] e_t(S)
    deviation: list      # per lag l = 0..P: [n, k] y_(t-l) - mu_(t-l)(s)
    log_density: Tensor  # [n, K]
    sigma: Tensor        # [k]


def densities(layout: Layout, theta: Tensor, y: Tensor, xs: Tensor, xc: Tensor) -> Pieces:
    """Residuals and log densities for every period t >= P and expanded state.

    ``y`` [N], ``xs`` [N, ks], ``xc`` [N, kc] hold ALL rows including the P presample rows.
    """
    parts = layout.split(theta)
    order, k = layout.order, layout.states
    n = y.shape[0] - order
    mean = xs @ parts["beta"].T if layout.switching else torch.zeros((y.shape[0], k),
                                                                   dtype=torch.float64)
    if layout.common:
        mean = mean + (xc @ parts["alpha"])[:, None]
    dev_all = y[:, None] - mean                                   # [N, k]
    deviation = [dev_all[order - lag:order - lag + n] for lag in range(order + 1)]
    lag_state = layout.lag_state()
    residual = deviation[0][:, lag_state[0]]
    phi = parts["phi"]
    for position, lag_ in enumerate(layout.lags):
        coefficient = phi[:, position][lag_state[0]] if layout.arswitch else phi[0, position]
        residual = residual - coefficient * deviation[lag_][:, lag_state[lag_]]
    lnsigma = parts["lnsigma"]
    sigma = torch.exp(lnsigma) if layout.varswitch else torch.exp(lnsigma).expand(k)
    state_sigma = sigma[lag_state[0]]
    log_density = -0.5 * math.log(2.0 * math.pi) - torch.log(state_sigma) \
        - 0.5 * (residual / state_sigma).square()
    return Pieces(residual, deviation, log_density, sigma)


def _block(n: int) -> int:
    return max(8, int(math.ceil(math.sqrt(n))))


def hamilton_filter(log_density: Tensor, a: Tensor, start: Tensor) -> dict[str, Tensor]:
    """Blocked Hamilton filter: predicted, filtered probabilities and log f(y_t | Y_(t-1))."""
    n, K = log_density.shape
    peak = log_density.max(dim=1).values
    eta = torch.exp(log_density - peak[:, None])
    size = _block(n)
    blocks = -(-n // size)
    padded = torch.ones((blocks * size, K), dtype=torch.float64)
    padded[:n] = eta
    grid = padded.reshape(blocks, size, K)
    at = a.T
    # Block products in the layout [row, block, column], so that each step is ONE matrix
    # product A' @ [K, blocks * K] instead of many tiny batched products.
    product = torch.eye(K, dtype=torch.float64)[:, None, :].expand(K, blocks, K).contiguous()
    eta_t = grid.permute(1, 2, 0)                                   # [size, K, blocks]
    for i in range(size):
        product = (at @ product.reshape(K, -1)).reshape(K, blocks, K) * eta_t[i][:, :, None]
        if i % 4 == 3 or i == size - 1:
            product = product / product.amax(dim=(0, 2), keepdim=True).clamp_min(1e-300)
    product = product.permute(1, 0, 2)                              # [blocks, K, K]
    starts = torch.empty((blocks, K), dtype=torch.float64)
    current = start / start.sum()
    for j in range(blocks):
        starts[j] = current
        current = product[j] @ current
        current = current / current.sum().clamp_min(1e-300)
    if not bool(torch.isfinite(starts).all()) or not bool((starts.sum(1) > 0.5).all()):
        raise KernelError("numerical_failure", "The Hamilton filter lost every state.")
    filtered = starts
    predicted_out = torch.empty((blocks, size, K), dtype=torch.float64)
    filtered_out = torch.empty((blocks, size, K), dtype=torch.float64)
    scale_out = torch.empty((blocks, size), dtype=torch.float64)
    for i in range(size):
        predicted = filtered @ a
        joint = grid[:, i] * predicted
        scale = joint.sum(dim=1)
        filtered = joint / scale[:, None].clamp_min(1e-300)
        predicted_out[:, i], filtered_out[:, i], scale_out[:, i] = predicted, filtered, scale
    scale = scale_out.reshape(-1)[:n]
    if not bool((scale > 0).all()):
        raise KernelError("numerical_failure", "An observation has zero likelihood under every "
                          "state.")
    return {"predicted": predicted_out.reshape(-1, K)[:n], "filtered": filtered_out.reshape(-1, K)[:n],
            "loglik": torch.log(scale) + peak}


def kim_smoother(filtered: Tensor, predicted: Tensor, a: Tensor) -> Tensor:
    """Smoothed probabilities xi_(t|T) by the blocked backward scheme."""
    n, K = filtered.shape
    ratio_rows = n - 1
    if ratio_rows == 0:
        return filtered.clone()
    size = _block(ratio_rows)
    blocks = -(-ratio_rows // size)
    # Backward step t <- t+1 (t = n-2 .. 0): xi_t = xi_(t|t) * (A (xi_(t+1) / xi_(t+1|t))).
    inverse = torch.where(predicted[1:] > 0, 1.0 / predicted[1:].clamp_min(1e-300),
                          torch.zeros((), dtype=torch.float64))            # [n-1, K]
    left = filtered[:-1].flip(0)                                          # step order
    right = inverse.flip(0)
    pad = blocks * size - ratio_rows
    if pad:
        left = torch.cat([left, torch.ones((pad, K), dtype=torch.float64)])
        right = torch.cat([right, torch.ones((pad, K), dtype=torch.float64)])
    left, right = left.reshape(blocks, size, K), right.reshape(blocks, size, K)
    product = torch.eye(K, dtype=torch.float64)[:, None, :].expand(K, blocks, K).contiguous()
    left_t, right_t = left.permute(1, 2, 0), right.permute(1, 2, 0)   # [size, K, blocks]
    for i in range(size):
        scaled = (product * right_t[i][:, :, None]).reshape(K, -1)
        product = (a @ scaled).reshape(K, blocks, K) * left_t[i][:, :, None]
        if i % 4 == 3 or i == size - 1:
            product = product / product.amax(dim=(0, 2), keepdim=True).clamp_min(1e-300)
    product = product.permute(1, 0, 2)                              # [blocks, K, K]
    starts = torch.empty((blocks, K), dtype=torch.float64)
    current = filtered[-1]
    for j in range(blocks):
        starts[j] = current
        current = product[j] @ current
        current = current / current.sum().clamp_min(1e-300)
    vector = starts
    out = torch.empty((blocks, size, K), dtype=torch.float64)
    for i in range(size):
        vector = left[:, i] * ((vector * right[:, i]) @ a.T)
        vector = vector / vector.sum(dim=1, keepdim=True).clamp_min(1e-300)
        out[:, i] = vector
    smoothed = torch.empty((n, K), dtype=torch.float64)
    smoothed[-1] = filtered[-1]
    smoothed[:-1] = out.reshape(-1, K)[:ratio_rows].flip(0)
    return smoothed


def score(layout: Layout, theta: Tensor, y: Tensor, xs: Tensor, xc: Tensor,
          *, need_smoothed: bool = False):
    """Log likelihood, its analytic gradient (Fisher identity) and optionally xi_(t|T)."""
    parts = layout.split(theta)
    k, order = layout.states, layout.order
    p = transition(parts["logit"])
    pi, fundamental = ergodic(p)
    a, start = expanded_chain(layout, p, pi)
    pieces = densities(layout, theta, y, xs, xc)
    run = hamilton_filter(pieces.log_density, a, start)
    loglik = run["loglik"].sum()
    smoothed = kim_smoother(run["filtered"], run["predicted"], a)
    lag_state = layout.lag_state()
    n = smoothed.shape[0]
    # Joint transition probabilities (latent period 0 included).
    ratio = torch.where(run["predicted"] > 0, smoothed / run["predicted"].clamp_min(1e-300),
                        torch.zeros((), dtype=torch.float64))
    previous = torch.cat([start[None, :], run["filtered"][:-1]])
    joint = a * (previous.T @ ratio)                                          # [K, K]
    initial = start * (a @ ratio[0])                                          # xi_(0|T)
    counts = torch.zeros((k, k), dtype=torch.float64)
    counts.index_put_((lag_state[0][:, None].expand(-1, layout.expanded),
                       lag_state[0][None, :].expand(layout.expanded, -1)), joint, accumulate=True)
    for level in range(1, order + 1):
        counts.index_put_((lag_state[level], lag_state[level - 1]), initial, accumulate=True)
    oldest = torch.zeros(k, dtype=torch.float64).index_add_(0, lag_state[order], initial)
    # Transition logits: d log p_ab / d q_ac = delta_bc - p_ac; ergodic term via Z.
    grad_logit = counts[:, :k - 1] - counts.sum(1, keepdim=True) * p[:, :k - 1]
    weight_pi = oldest / pi.clamp_min(1e-300)
    az = p @ fundamental
    for a_ in range(k):
        for c in range(k - 1):
            dpi = pi[a_] * p[a_, c] * (fundamental[c] - az[a_])
            grad_logit[a_, c] += float(weight_pi @ dpi)
    # Density parameters.
    sigma = pieces.sigma
    state_sigma = sigma[lag_state[0]]
    u = smoothed * pieces.residual / state_sigma.square()                     # [n, K]
    indicator = [torch.nn.functional.one_hot(lag_state[level], k).to(torch.float64)
                 for level in range(order + 1)]
    rows = slice(order, order + n)
    phi = parts["phi"]
    grad = {}
    if layout.switching:
        g = xs[rows].T @ (u @ indicator[0])                                   # [ks, k]
        for position, lag_ in enumerate(layout.lags):
            coefficient = phi[:, position][lag_state[0]] if layout.arswitch else phi[0, position]
            g = g - xs[order - lag_:order - lag_ + n].T @ ((u * coefficient) @ indicator[lag_])
        grad["beta"] = g.T.reshape(-1)
    if layout.common:
        g = xc[rows].T @ u.sum(1)
        for position, lag_ in enumerate(layout.lags):
            coefficient = phi[:, position][lag_state[0]] if layout.arswitch else phi[0, position]
            g = g - xc[order - lag_:order - lag_ + n].T @ (u * coefficient).sum(1)
        grad["alpha"] = g
    if layout.lags:
        values = []
        for position, lag_ in enumerate(layout.lags):
            term = u * pieces.deviation[lag_][:, lag_state[lag_]]                 # [n, K]
            values.append(term @ indicator[0] if layout.arswitch else term.sum(1, keepdim=True))
        grad["phi"] = torch.stack([value.sum(0) for value in values], dim=1).reshape(-1)
    standardized = smoothed * ((pieces.residual / state_sigma).square() - 1.0)
    grad["lnsigma"] = (standardized @ indicator[0]).sum(0) if layout.varswitch \
        else standardized.sum().reshape(1)
    grad["logit"] = grad_logit.reshape(-1)
    gradient = torch.cat([grad.get(name, torch.zeros(size, dtype=torch.float64))
                          for name, size in layout.sizes().items()])
    out = {"loglik": loglik, "gradient": gradient, "per_period": run["loglik"],
           "transition": p, "ergodic": pi}
    if need_smoothed:
        out.update(smoothed=smoothed, filtered=run["filtered"], predicted=run["predicted"])
    return out
