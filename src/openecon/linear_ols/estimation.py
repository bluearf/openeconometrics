"""Float64 OLS/WLS and covariance estimators implemented with native Torch.

The caller supplies the complete finite design, including its intercept.  The
solve centers/scales columns and uses reduced QR; collinearity omits the first
dependent term deterministically, retaining the intercept and original order.
Only n-by-k and k-by-k matrices are formed.  Weight and covariance conventions
follow https://www.stata.com/manuals/rregress.pdf (Methods and formulas).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
from typing import Sequence

import pandas as pd
import torch
from torch import Tensor

from openecon.engines.contracts import KernelError


_EPS = torch.finfo(torch.float64).eps
_WEIGHTS = {"aw", "fw", "pw", "iw"}
_COVARIANCES = {"nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster",
                "cluster_hc2", "cluster_hc3", "HAC", "bootstrap", "jackknife"}
_KERNELS = {"bartlett", "parzen", "quadratic_spectral", "truncated"}


@dataclass
class _Fit:
    x: Tensor
    z: Tensor
    q: Tensor
    r: Tensor
    transform: Tensor
    bridge: Tensor
    params: Tensor
    fitted: Tensor
    resid: Tensor
    bread: Tensor
    leverage: Tensor
    kept: list[int]
    constant: int | None
    condition: float
    normalization: _Normalization
    inverse_r: Tensor
    theta: Tensor


@dataclass
class _Normalization:
    anchor: Tensor
    scale: Tensor
    mean: Tensor
    rms: Tensor
    anchored: list[bool]
    constant: int | None

    def subset(self, kept):
        return _Normalization(*(value[kept] for value in (self.anchor, self.scale, self.mean, self.rms)),
                              [self.anchored[i] for i in kept],
                              kept.index(self.constant) if self.constant is not None else None)

    def encode(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=self.scale.dtype, device=self.scale.device)
        if x.ndim != 2 or x.shape[1] != len(self.scale):
            raise KernelError("invalid_design", "Prediction needs one encoded column per kept coefficient.")
        z = torch.empty_like(x)
        constant = x[:, self.constant] if self.constant is not None else None
        for i in range(x.shape[1]):
            if i == self.constant:
                z[:, i] = constant
            elif self.constant is not None:
                values = ((x[:, i] - self.anchor[i] * constant) / self.scale[i]
                          if self.anchored[i] else x[:, i] / self.scale[i])
                z[:, i] = (values - self.mean[i] * constant) / self.rms[i]
            else:
                z[:, i] = x[:, i] / self.scale[i] / self.rms[i]
        return z


class _PredictionState:
    """Evaluate prediction moments before the ill-conditioned raw-unit map."""

    def __init__(self, fit: _Fit, covariance: Tensor, raw_factor: Tensor | None = None):
        self.normalization, self.inverse_r, self.theta = fit.normalization, fit.inverse_r, fit.theta
        self.covariance = covariance
        self.raw_factor = raw_factor

    def variance(self, x: Tensor) -> Tensor:
        with torch.inference_mode():
            if self.raw_factor is not None:
                values = x.to(device=self.raw_factor.device, dtype=self.raw_factor.dtype) @ self.raw_factor
                return values.square().sum(dim=1).to(x.device)
            q = self.normalization.encode(x) @ self.inverse_r
            return (q @ self.covariance * q).sum(dim=1).to(x.device)

    def leverage(self, x: Tensor) -> Tensor:
        with torch.inference_mode():
            q = self.normalization.encode(x) @ self.inverse_r
            return q.square().sum(dim=1).to(x.device)

    def contrast_covariance(self, rows: Tensor) -> Tensor:
        with torch.inference_mode():
            if self.raw_factor is not None:
                values = rows.to(device=self.raw_factor.device, dtype=self.raw_factor.dtype) @ self.raw_factor
                return (values @ values.T).to(rows.device)
            q = self.normalization.encode(rows) @ self.inverse_r
            return (q @ self.covariance @ q.T).to(rows.device)

    def fitted(self, x: Tensor) -> Tensor:
        with torch.inference_mode():
            return (self.normalization.encode(x) @ self.theta).to(x.device)


class _ExactWeightSum:
    """Chunk-independent, correctly rounded positive float64 weight total."""

    def __init__(self):
        self._numerator = 0

    def add(self, weights: Tensor):
        for block in weights.detach().split(4096):
            for value in block.cpu().tolist():
                numerator, denominator = value.as_integer_ratio()
                self._numerator += numerator << (1074 - (denominator.bit_length() - 1))

    def total(self) -> float:
        try:
            return float(Fraction(self._numerator, 1 << 1074))
        except OverflowError as exc:
            raise KernelError("numerical_failure", "The sum of importance weights exceeds float64 precision.") from exc


def _weights(weights: Tensor | None, kind: str | None, n: int, x: Tensor,
             covariance: str) -> tuple[Tensor, int, str | None]:
    if weights is None:
        if kind is not None:
            raise KernelError("invalid_weights", "weight_type requires a weight vector.")
        return torch.ones(n, dtype=x.dtype, device=x.device), n, None
    if kind not in _WEIGHTS:
        raise KernelError("invalid_weights", "Choose aw, fw, pw or iw for weighted regression.")
    weights = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
    if weights.ndim != 1 or len(weights) != n or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise KernelError("invalid_weights", "Weights must be finite and positive after sample selection.")
    if kind == "fw":
        if bool((weights != weights.round()).any()) or bool((weights > 2**53).any()):
            raise KernelError("invalid_weights", "Frequency weights must be exactly representable positive integers.")
        effective = sum(int(value) for value in weights.detach().cpu().tolist())
        return weights, effective, kind
    if kind == "iw" and covariance == "nonrobust":
        accumulator = _ExactWeightSum()
        accumulator.add(weights)
        total = accumulator.total()
        if not math.isfinite(total):
            raise KernelError("numerical_failure", "The sum of importance weights exceeds float64 precision.")
        return weights, int(total), kind
    # Divide before summing to prevent overflow from large but finite weights.
    scaled = weights / weights.max()
    return scaled * (n / scaled.sum()), n, kind


def _intercept_index(x: Tensor, terms: list[str], intercept: bool) -> int | None:
    if not intercept:
        return None
    preferred = [i for i, term in enumerate(terms) if term in {"Intercept", "_cons", "const"}]
    order = preferred + [i for i in range(x.shape[1]) if i not in preferred]
    for i in order:
        if bool((x[:, i] == 1).all()):
            return i
    raise KernelError("invalid_design", "An intercept design must include a column of ones.")


def _normalized(x: Tensor, w: Tensor, constant: int | None) -> tuple[Tensor, Tensor, _Normalization]:
    n, k = x.shape
    unit = w / w.max()
    probabilities = unit / unit.sum()
    z = torch.empty_like(x)
    transform = torch.zeros((k, k), dtype=x.dtype, device=x.device)
    anchors = torch.zeros(k, dtype=x.dtype, device=x.device)
    scales, means, deviations = torch.ones_like(anchors), torch.zeros_like(anchors), torch.ones_like(anchors)
    anchored = [False] * k
    for i in range(k):
        if i == constant:
            z[:, i] = 1
            transform[i, i] = 1
            continue
        if constant is not None:
            anchor = x[0, i]
            diff = x[:, i] - anchor
            if bool(torch.isfinite(diff).all()):
                scale = diff.abs().max()
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                values = diff / scale
                mean = torch.dot(probabilities, values)
                centered = values - mean
                rms = torch.dot(probabilities, centered.square()).sqrt()
                if float(rms.item()) == 0:
                    z[:, i] = 0
                    continue
                z[:, i] = centered / rms
                anchors[i], scales[i], means[i], deviations[i] = anchor, scale, mean, rms
                anchored[i] = True
                transform[i, i] = 1 / (scale * rms)
                transform[constant, i] = -(anchor / scale + mean) / rms
            else:
                # Opposite signed extreme values can overflow subtraction.
                scale = x[:, i].abs().max()
                values = x[:, i] / scale
                mean = torch.dot(probabilities, values)
                centered = values - mean
                rms = torch.dot(probabilities, centered.square()).sqrt()
                z[:, i] = centered / rms
                scales[i], means[i], deviations[i] = scale, mean, rms
                transform[i, i] = 1 / (scale * rms)
                transform[constant, i] = -mean / rms
        else:
            scale = x[:, i].abs().max()
            if float(scale.item()) == 0:
                z[:, i] = 0
                continue
            values = x[:, i] / scale
            rms = torch.dot(probabilities, values.square()).sqrt()
            z[:, i] = values / rms
            scales[i], deviations[i] = scale, rms
            transform[i, i] = 1 / (scale * rms)
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(transform).all()):
        raise KernelError("numerical_failure", "Column units exceed float64 precision; rescale the input.")
    return z, transform, _Normalization(anchors, scales, means, deviations, anchored, constant)


def _independent_columns(z: Tensor, w: Tensor, constant: int | None) -> list[int]:
    _, r = torch.linalg.qr(z * w.sqrt()[:, None], mode="r")
    # Orthogonalize the small QR factor, never a second n-by-k matrix.  Double
    # reorthogonalization preserves term order without a pivot-dependent choice.
    order = ([constant] if constant is not None else []) + [i for i in range(z.shape[1]) if i != constant]
    vectors, kept = [], []
    largest = float(torch.linalg.vector_norm(r, dim=0).max().item())
    tolerance = _EPS * max(z.shape) * max(1., math.sqrt(z.shape[1])) * largest
    for i in order:
        vector = r[:, i].clone()
        for _ in range(2):
            for basis in vectors:
                vector -= basis * torch.dot(basis, vector)
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) > tolerance:
            vectors.append(vector / norm)
            kept.append(i)
    return sorted(kept)


def _fit(x: Tensor, y: Tensor, w: Tensor, terms: list[str], intercept: bool,
         *, omit: bool = True) -> _Fit:
    constant = _intercept_index(x, terms, intercept)
    z, transform, normalization = _normalized(x, w, constant)
    kept = _independent_columns(z, w, constant) if omit else list(range(x.shape[1]))
    if not kept:
        raise KernelError("singular_design", "The design has no estimable columns.")
    if len(x) < len(kept):
        raise KernelError("insufficient_observations", "The physical sample must identify the estimable design rank.")
    z = z[:, kept]
    transform = transform[kept][:, kept]
    q, r = torch.linalg.qr(z * w.sqrt()[:, None], mode="reduced")
    singular = torch.linalg.svdvals(r)
    if float(singular[-1].item()) <= _EPS * max(z.shape) * float(singular[0].item()):
        raise KernelError("singular_design", "The resampled design is rank deficient.")
    y_anchor = y[0] if constant is not None else torch.zeros((), dtype=y.dtype, device=y.device)
    centered = y - y_anchor
    theta = torch.linalg.solve_triangular(r, (q.T @ (centered * w.sqrt()))[:, None], upper=True)[:, 0]
    fitted_centered = z @ theta
    residual = centered - fitted_centered
    fitted = fitted_centered + y_anchor
    kept_constant = kept.index(constant) if constant is not None else None
    if kept_constant is not None:
        theta[kept_constant] += y_anchor
    inverse_r = torch.linalg.solve_triangular(r, torch.eye(len(kept), dtype=x.dtype, device=x.device), upper=True)
    bridge = transform @ inverse_r
    return _Fit(x[:, kept], z, q, r, transform, bridge, transform @ theta, fitted, residual,
                bridge @ bridge.T, q.square().sum(dim=1), kept, kept_constant,
                float((singular[0] / singular[-1]).item()), normalization.subset(kept), inverse_r, theta)


def _cluster_codes(clusters: list[Sequence] | None, n: int, device) -> list[Tensor]:
    if not isinstance(clusters, list) or not clusters:
        raise KernelError("invalid_clusters", "Provide one or more clustering variables.")
    result = []
    for labels in clusters:
        try:
            if len(labels) != n:
                raise ValueError("length")
            if isinstance(labels, Tensor):
                if labels.ndim != 1 or not bool(torch.isfinite(labels).all()):
                    raise ValueError("values")
                _, codes = torch.unique(labels.to(device=device), return_inverse=True)
            else:
                values = labels if isinstance(labels, (pd.Series, pd.Index)) else pd.Series(labels)
                codes, _ = pd.factorize(values, sort=False)
                if bool((codes < 0).any()):
                    raise ValueError("missing")
                codes = torch.as_tensor(codes, dtype=torch.int64, device=device)
            if int(codes.max().item()) < 1:
                raise KernelError("insufficient_clusters", "At least two groups are required per clustering variable.")
            result.append(codes)
        except KernelError:
            raise
        except (TypeError, ValueError) as exc:
            raise KernelError("invalid_clusters", "Cluster labels must be nonmissing scalar values, one per row.") from exc
    return result


def _group_totals(scores: Tensor, codes: Tensor) -> Tensor:
    count = int(codes.max().item()) + 1
    totals = torch.zeros((count, scores.shape[1]), dtype=scores.dtype, device=scores.device)
    totals.index_add_(0, codes, scores)
    return totals


def _group_indices(codes: Tensor):
    order = torch.argsort(codes, stable=True)
    start = 0
    for count in torch.bincount(codes).detach().cpu().tolist():
        yield order[start:start + count]
        start += count


class _ContrastAdjustment:
    """Design-only Satterthwaite moments; k-by-k scratch, no n-by-n hat matrix.

    Hansen (2025), https://users.ssc.wisc.edu/~bhansen/papers/jae_25.pdf,
    equation A6 specifies sqrt(trace/bread), not the dimensionally inconsistent
    squared trace printed in the current Stata manual's scale formula.
    """
    def __init__(self, fit: _Fit, w: Tensor, kind: str | None,
                 codes: Tensor | None, power: float, hansen: bool):
        self.q, self.bridge = fit.q, fit.bridge
        self.w, self.kind, self.codes = w, kind, codes
        self.power, self.hansen = power, hansen

    def __call__(self, gradient: Tensor) -> dict:
        with torch.inference_mode():
            gradient = torch.as_tensor(gradient, dtype=self.q.dtype, device=self.q.device)
            k = self.bridge.shape[0]
            if gradient.ndim != 1 or len(gradient) != k or not bool(torch.isfinite(gradient).all()):
                raise KernelError("invalid_contrast", "Adjusted inference needs one finite gradient per kept coefficient.")
            v = self.bridge.T @ gradient
            scale = v.abs().max()
            if float(scale.item()) == 0:
                return {"df": math.inf, "scale": 1.}
            v = v / scale
            bread = torch.dot(v, v)
            total_alpha = torch.zeros((), dtype=v.dtype, device=v.device)
            total_alpha2 = torch.zeros_like(total_alpha)
            total_norm = torch.zeros_like(total_alpha)
            total_cross = torch.zeros_like(total_alpha)
            bbt = torch.zeros((k, k), dtype=v.dtype, device=v.device)
            threshold = 100 * _EPS * k
            if self.codes is None:
                frequency = self.kind == "fw"
                units = sum(int(value) for value in self.w.detach().cpu().tolist()) if frequency else len(self.w)
                for start in range(0, len(self.w), 4096):
                    q = self.q[start:start + 4096]
                    multiplicity = self.w[start:start + 4096] if frequency else torch.ones(len(q), dtype=v.dtype, device=v.device)
                    if frequency:
                        q = q / multiplicity.sqrt()[:, None]
                    complement = 1 - q.square().sum(dim=1)
                    valid = complement > threshold
                    if self.hansen and not bool(valid.all()):
                        raise KernelError("unsupported_inference", "Hansen inference with unit-leverage observations requires a generalized jackknife implementation; ordinary HC3 is insufficient.")
                    inverse = torch.where(valid, complement.clamp_min(torch.finfo(v.dtype).tiny).pow(-self.power), 0)
                    a = (q @ v) * inverse
                    alpha = a.square()
                    b = q * a[:, None]
                    norm = b.square().sum(dim=1)
                    total_alpha += torch.dot(multiplicity, alpha)
                    total_alpha2 += torch.dot(multiplicity, alpha.square())
                    total_norm += torch.dot(multiplicity, norm)
                    total_cross += torch.dot(multiplicity, alpha * norm)
                    weighted_b = b * multiplicity.sqrt()[:, None]
                    bbt += weighted_b.T @ weighted_b
            else:
                units = int(self.codes.max().item()) + 1
                eye = torch.eye(k, dtype=v.dtype, device=v.device)
                for indices in _group_indices(self.codes):
                    q = self.q[indices]
                    gram = q.T @ q
                    values, vectors = torch.linalg.eigh(eye - (gram + gram.T) / 2)
                    valid = values > threshold
                    if self.hansen and not bool(valid.all()):
                        raise KernelError("unsupported_inference", "Hansen inference with singular cluster annihilators requires a generalized jackknife implementation; ordinary cluster HC3 is insufficient.")
                    inverse = torch.where(valid, values.clamp_min(torch.finfo(v.dtype).tiny).pow(-self.power), 0)
                    adjusted = vectors @ (inverse * (vectors.T @ v))
                    a = q @ adjusted
                    alpha = torch.dot(a, a)
                    b = q.T @ a
                    norm = torch.dot(b, b)
                    total_alpha += alpha
                    total_alpha2 += alpha.square()
                    total_norm += norm
                    total_cross += alpha * norm
                    bbt += torch.outer(b, b)
            trace = total_alpha - total_norm
            denominator = total_alpha2 - 2 * total_cross + bbt.square().sum()
            if float(trace.item()) <= 0 or float(denominator.item()) <= 0:
                raise KernelError("undefined_adjusted_inference", "The selected contrast has no identified adjusted reference variance.")
            df = float((trace.square() / denominator).item())
            coefficient_scale = float((trace / bread).sqrt().item()) if self.hansen else 1.
            if not math.isfinite(df) or not math.isfinite(coefficient_scale):
                raise KernelError("numerical_failure", "Adjusted inference moments exceed float64 precision.")
            return {"df": min(float(units), max(1., df)), "scale": coefficient_scale}


def _psd(variance: Tensor, warnings: list[str], *, return_factor: bool = False):
    variance = (variance + variance.T) / 2
    eigenvalues, vectors = torch.linalg.eigh(variance)
    factor = None
    if bool((eigenvalues < 0).any()):
        threshold = 100 * _EPS * len(eigenvalues) * max(float(eigenvalues.abs().max().item()), torch.finfo(variance.dtype).tiny)
        if float(eigenvalues.min().item()) < -threshold:
            warnings.append("Multiway covariance was not positive semidefinite; negative eigenvalues were set to zero (CGM correction).")
        variance = (vectors * eigenvalues.clamp_min(0)) @ vectors.T
        factor = vectors * eigenvalues.clamp_min(0).sqrt()
    return (variance, factor) if return_factor else variance


def _cluster_variance(fit: _Fit, w: Tensor, codes: list[Tensor], nobs: int,
                      covariance: str, warnings: list[str]) -> tuple[Tensor, dict]:
    scores = fit.q * (fit.resid * w.sqrt())[:, None]
    k = len(fit.params)
    marginal = [int(code.max().item()) + 1 for code in codes]
    metadata = {"cluster_counts": marginal, "cluster_count": min(marginal),
                "cluster_dimensions": len(codes), "df_adjustment": "G-1", "dfadjust": False,
                "hansen": False, "combination_counts": []}
    meat = torch.zeros((k, k), dtype=scores.dtype, device=scores.device)
    if covariance in {"cluster_hc2", "cluster_hc3"}:
        if len(codes) != 1:
            raise KernelError("unsupported_covariance", "Cluster HC2/HC3 currently require one clustering variable.")
        eye = torch.eye(k, dtype=scores.dtype, device=scores.device)
        totals = _group_totals(scores, codes[0])
        singular_groups = 0
        for group, indices in enumerate(_group_indices(codes[0])):
            rows = fit.q[indices]
            annihilator = eye - rows.T @ rows
            values, vectors = torch.linalg.eigh((annihilator + annihilator.T) / 2)
            valid = values > 100 * _EPS * k
            singular_groups += int(not bool(valid.all()))
            power = .5 if covariance == "cluster_hc2" else 1.
            inverse = torch.where(valid, values.clamp_min(torch.finfo(values.dtype).tiny).pow(-power), 0)
            adjusted = vectors @ (inverse * (vectors.T @ totals[group]))
            meat += torch.outer(adjusted, adjusted)
        metadata.update(small_sample_correction=1., partial_annihilator="Moore-Penrose low-rank inverse",
                        singular_cluster_directions=singular_groups)
        if singular_groups:
            warnings.append("Cluster HC2/HC3 used a Moore-Penrose inverse for unit-leverage directions.")
        warnings.append("Cluster HC2/HC3 inference uses G-1; Bell-McCaffrey dfadjust and Hansen corrections are not enabled.")
    else:
        for width in range(1, len(codes) + 1):
            for selected in combinations(range(len(codes)), width):
                if width == 1:
                    index = codes[selected[0]]
                else:
                    _, index = torch.unique(torch.stack([codes[i] for i in selected], dim=1),
                                            dim=0, return_inverse=True)
                count = int(index.max().item()) + 1
                totals = _group_totals(scores, index)
                correction = count / (count - 1) * (nobs - 1) / (nobs - k)
                meat += (1 if width % 2 else -1) * correction * (totals.T @ totals)
                metadata["combination_counts"].append({"dimensions": list(selected), "count": count,
                                                       "correction": correction})
        if len(codes) == 1:
            metadata["small_sample_correction"] = metadata["combination_counts"][0]["correction"]
    return meat, metadata


def _kernel_weights(distance: Tensor, bandwidth: int, kernel: str) -> Tensor:
    z = distance / bandwidth
    if kernel == "truncated":
        return (z < 1).to(distance.dtype)
    if kernel == "bartlett":
        return (1 - z).clamp_min(0)
    if kernel == "parzen":
        return torch.where(z <= .5, 1 - 6 * z.square() + 6 * z.pow(3), 2 * (1 - z).clamp_min(0).pow(3))
    theta = z * (6 * math.pi / 5)
    # A series near zero avoids catastrophic cancellation in sin(theta)/theta-cos(theta).
    small = theta.abs() < 1e-3
    series = 1 - theta.square() / 10 + theta.pow(4) / 280 - theta.pow(6) / 15120
    safe = torch.where(small, torch.ones_like(theta), theta)
    exact = 3 * (torch.sin(safe) / safe - torch.cos(safe)) / safe.square()
    return torch.where(small, series, exact)


def _automatic_lags(fit: _Fit, w: Tensor, times: Tensor, order: Tensor,
                    kernel: str, warnings: list[str]) -> tuple[int | float, dict]:
    """Newey-West (1994) plug-in selector, Stata rregress Methods and formulas."""
    if kernel == "truncated":
        raise KernelError("invalid_covariance_options", "Automatic HAC lags require Bartlett, Parzen or quadratic spectral.")
    n = len(w)
    q, power, constant = {"bartlett": (1, 2/9, 1.1447), "parzen": (2, 4/25, 2.6614),
                          "quadratic_spectral": (2, 2/25, 1.3221)}[kernel]
    pilot = min(int(20 * (n / 100)**power), int(times[-1].item()))
    slopes = [i for i in range(fit.x.shape[1]) if i != fit.constant]
    series = ((fit.x[:, slopes].sum(dim=1) * fit.resid * w)[order]
              if slopes else torch.zeros(n, dtype=w.dtype, device=w.device))
    sigma0 = float(torch.dot(series, series).item()) / n
    sum_zero, sum_order = sigma0, 0.
    for lag in range(1, pilot + 1):
        indices = torch.searchsorted(times, times + lag)
        valid = indices < n
        valid &= times[indices.clamp_max(n - 1)] == times + lag
        sigma = float(torch.dot(series[valid], series[indices[valid]]).item()) / n
        sum_zero += 2 * sigma
        sum_order += 2 * sigma * lag**q
    if not math.isfinite(sum_zero) or not math.isfinite(sum_order):
        raise KernelError("numerical_failure", "Automatic HAC score moments exceed float64 precision; rescale predictors.")
    if abs(sum_zero) <= 100 * _EPS * max(sigma0, torch.finfo(w.dtype).tiny):
        warnings.append("Automatic HAC lag selection had zero long-run pilot variance and selected zero lags.")
        selected = 0
    else:
        gamma = constant * abs(sum_order / sum_zero)**(2 / (2*q + 1))
        selected = min(gamma * n**(1 / (2*q + 1)), pilot)
        if kernel != "quadratic_spectral":
            selected = int(selected)
    return selected, {"lag_selection": "Newey-West 1994 plug-in", "automatic_lags": True,
                      "pilot_lags": pilot, "lag_selection_order": q, "lag_selection_constant": constant,
                      "lag_selection_time_gaps": True}


def _hac_variance(fit: _Fit, w: Tensor, time: Tensor | None, lags: int | str | None,
                  kernel: str, nobs: int, warnings: list[str]) -> tuple[Tensor, dict]:
    aliases = {"nwest": "bartlett", "gallant": "parzen", "quadraticspectral": "quadratic_spectral", "andrews": "quadratic_spectral"}
    kernel = aliases.get(kernel, kernel)
    if kernel not in _KERNELS:
        raise KernelError("invalid_covariance_options", "Choose Bartlett, Parzen or quadratic_spectral HAC.")
    n, k = fit.x.shape
    if time is None:
        raise KernelError("invalid_time", "HAC requires unique integer time units; missing periods must retain their gaps.")
    time = torch.as_tensor(time, device=fit.x.device)
    if time.ndim != 1 or len(time) != n or not bool(torch.isfinite(time).all()):
        raise KernelError("invalid_time", "HAC requires one finite integer time value per row.")
    if time.is_floating_point() and (bool((time != time.round()).any()) or bool((time.abs() > 2**53).any())):
        raise KernelError("invalid_time", "Convert time values to exactly representable integer units before HAC.")
    time = time.to(torch.int64)
    times, order = torch.sort(time)
    if bool((times[1:] == times[:-1]).any()):
        raise KernelError("invalid_time", "HAC time values must be unique.")
    span = int(times[-1].item()) - int(times[0].item())
    if span > 2**63 - 1:
        raise KernelError("invalid_time", "The time extent exceeds integer precision; choose a coarser time unit.")
    times = times - times[0]
    selection = {"automatic_lags": False, "lag_selection": "explicit" if lags is not None else "Stata N-2 default"}
    if lags == "auto":
        lags, selection = _automatic_lags(fit, w, times, order, kernel, warnings)
    elif lags is None:
        lags = n - 2
    elif type(lags) is not int or lags < 0:
        raise KernelError("invalid_covariance_options", "HAC lags must be a nonnegative integer.")
    scores = (fit.q * (fit.resid * w.sqrt())[:, None])[order]
    meat = scores.T @ scores
    bandwidth = lags + 1
    noncompact = kernel == "quadratic_spectral"
    if noncompact:
        warnings.append("Quadratic spectral HAC has noncompact support: all observed time differences are included; memory is bounded but work can be quadratic in rows.")
    if not noncompact and min(lags, span) <= n:
        # Efficient for normal short-lag time series; searchsorted preserves
        # real calendar gaps instead of shifting neighboring observed rows.
        for lag in range(1, min(lags, span) + 1):
            index = torch.searchsorted(times, times + lag)
            valid = index < n
            valid &= times[index.clamp_max(n - 1)] == times + lag
            cross = scores[valid].T @ scores[index[valid]]
            weight = float(_kernel_weights(torch.tensor(float(lag), dtype=scores.dtype, device=scores.device), bandwidth, kernel).item())
            meat += weight * (cross + cross.T)
    else:
        # Long/sparse calendars and noncompact kernels: bounded score blocks,
        # never an observation-by-observation covariance or distance matrix.
        for i in range(n - 1):
            end = n if noncompact else int(torch.searchsorted(times, times[i] + min(bandwidth, span + 1)).item())
            weighted = torch.zeros(k, dtype=scores.dtype, device=scores.device)
            for start in range(i + 1, end, 4096):
                stop = min(start + 4096, end)
                distance = (times[start:stop] - times[i]).to(scores.dtype)
                weighted += (_kernel_weights(distance, bandwidth, kernel)[:, None] * scores[start:stop]).sum(dim=0)
            cross = torch.outer(scores[i], weighted)
            meat += cross + cross.T
    correction = nobs / (nobs - k)
    variance = fit.bridge @ (meat * correction) @ fit.bridge.T
    eigenvalues = torch.linalg.eigvalsh((variance + variance.T) / 2)
    threshold = 100 * _EPS * k * max(float(eigenvalues.abs().max().item()), torch.finfo(w.dtype).tiny)
    positive_semidefinite = float(eigenvalues.min().item()) >= -threshold
    if not positive_semidefinite:
        warnings.append("HAC covariance is not positive semidefinite; the truncated kernel may require Bartlett, Parzen or quadratic spectral instead.")
    return meat * correction, {
        "kernel": kernel, "lags": lags, "bandwidth": bandwidth, "time_gaps": True,
        "small_sample_correction": correction, "noncompact_support": noncompact,
        "positive_semidefinite": positive_semidefinite, **selection}


def _frequency_draw(weights: Tensor, total: int, generator: torch.Generator) -> Tensor:
    """Exact multinomial counts via conditional binomials, without expanded rows."""
    if total > 2**53:
        raise KernelError("invalid_resampling_options", "Frequency bootstrap counts exceed exact float64 random-count precision.")
    remaining_count = torch.tensor(float(total), dtype=weights.dtype, device=weights.device)
    remaining_weight = weights.sum()
    result = torch.zeros_like(weights)
    for i in range(len(weights) - 1):
        probability = (weights[i] / remaining_weight).clamp(0, 1)
        result[i] = torch.binomial(remaining_count, probability, generator=generator)
        remaining_count -= result[i]
        remaining_weight -= weights[i]
    result[-1] = remaining_count
    return result


def _bootstrap(fit: _Fit, y: Tensor, w: Tensor, kind: str | None, nobs: int,
               clusters: list[Sequence] | None, terms: list[str], reps: int, seed: int,
               warnings: list[str]) -> tuple[Tensor, dict]:
    if type(reps) is not int or reps < 2 or type(seed) is not int or not 0 <= seed < 2**63:
        raise KernelError("invalid_resampling_options", "Bootstrap needs at least two repetitions and a nonnegative integer seed.")
    codes = _cluster_codes(clusters, len(y), fit.x.device) if clusters is not None else []
    if len(codes) > 1:
        raise KernelError("unsupported_covariance", "Bootstrap supports pairs or one-way cluster resampling; multiway resampling requires an explicit scheme.")
    generator = torch.Generator(device=fit.x.device).manual_seed(seed)
    k, success, failed = len(fit.params), 0, 0
    mean = torch.zeros(k, dtype=w.dtype, device=w.device)
    cross = torch.zeros((k, k), dtype=w.dtype, device=w.device)
    units = int(codes[0].max().item()) + 1 if codes else len(y)
    for _ in range(reps):
        if codes:
            count = torch.bincount(torch.randint(units, (units,), generator=generator, device=w.device), minlength=units)
            draw = w * count[codes[0]]
        elif kind == "fw":
            draw = _frequency_draw(w, nobs, generator)
        else:
            count = torch.bincount(torch.randint(units, (units,), generator=generator, device=w.device), minlength=units)
            draw = w * count
        selected = draw > 0
        try:
            sampled = _fit(fit.z[selected], y[selected], draw[selected], terms,
                           fit.constant is not None, omit=False)
        except KernelError as exc:
            if exc.code not in {"singular_design", "insufficient_observations"}:
                raise
            failed += 1
            continue
        success += 1
        parameters = fit.r @ sampled.params
        difference = parameters - mean
        mean += difference / success
        cross += torch.outer(difference, parameters - mean)
    if success < 2:
        raise KernelError("insufficient_bootstrap_replications", "Fewer than two bootstrap samples retain the original design rank.")
    if failed:
        warnings.append(f"{failed} bootstrap repetitions had insufficient observations or rank and were excluded; covariance uses {success} successful repetitions.")
    return cross / (success - 1), {"scheme": "cluster" if codes else "pairs", "reps": reps,
                                  "successful_reps": success, "failed_reps": failed, "seed": seed,
                                  "resampling_mean": fit.bridge @ mean, "random_engine": "torch.Generator",
                                  "random_count_algorithm": "exact conditional-binomial multinomial" if kind == "fw" and not codes else "torch.randint with bincount",
                                  "cluster_count": units if codes else None,
                                  "frequency_count_resampling": bool(kind == "fw" and not codes)}


def _jackknife(fit: _Fit, w: Tensor, kind: str | None, nobs: int,
               clusters: list[Sequence] | None) -> tuple[Tensor, dict]:
    codes = _cluster_codes(clusters, len(w), fit.x.device) if clusters is not None else []
    if len(codes) > 1:
        raise KernelError("unsupported_covariance", "Jackknife supports row or one-way cluster deletion.")
    k = len(fit.params)
    if codes:
        units = int(codes[0].max().item()) + 1
        totals = _group_totals(fit.q * (fit.resid * w.sqrt())[:, None], codes[0])
        deleted = torch.empty_like(totals)
        eye = torch.eye(k, dtype=w.dtype, device=w.device)
        for group, indices in enumerate(_group_indices(codes[0])):
            q = fit.q[indices]
            annihilator = eye - q.T @ q
            if float(torch.linalg.eigvalsh(annihilator).min().item()) <= 100 * _EPS * k:
                raise KernelError("jackknife_rank_deficient", "Deleting a cluster removes an identified coefficient; jackknife covariance is undefined.")
            deleted[group] = torch.linalg.solve(annihilator, totals[group])
        multiplicity = torch.ones(units, dtype=w.dtype, device=w.device)
    else:
        units = nobs if kind == "fw" else len(w)
        multiplicity = w if kind == "fw" else torch.ones_like(w)
        complement = 1 - (fit.leverage / w if kind == "fw" else fit.leverage)
        if bool((complement <= 100 * _EPS * k).any()):
            raise KernelError("jackknife_rank_deficient", "Deleting an observation removes an identified coefficient; jackknife covariance is undefined.")
        scalar = fit.resid / w.sqrt() if kind == "fw" else fit.resid * w.sqrt()
        deleted = fit.q * (scalar / complement)[:, None]
    average = (deleted * multiplicity[:, None]).sum(dim=0) / units
    centered = (deleted - average) * multiplicity.sqrt()[:, None]
    meat = centered.T @ centered * ((units - 1) / units)
    return meat, {"scheme": "cluster" if codes else "delete-one",
        "reps": units, "successful_reps": units, "failed_reps": 0,
        "resampling_units": units, "bias_estimate": -(units - 1) * (fit.bridge @ average),
        "frequency_count_resampling": bool(kind == "fw" and not codes),
        "cluster_count": units if codes else None}


def estimate(x: Tensor, y: Tensor, *, terms: list[str], intercept: bool = True,
             covariance: str = "nonrobust", weights: Tensor | None = None,
             weight_type: str | None = None, clusters: list[Sequence] | None = None,
             time: Tensor | None = None, lags: int | str | None = None, kernel: str = "bartlett",
             reps: int = 199, seed: int = 0, dfadjust: bool = False, hansen: bool = False) -> dict:
    """Estimate a finite dense design, retaining all tensor results on its device.

    aw/pw and robust iw normalize to physical rows; fw counts and classical iw
    determine the effective sample size.  pw defaults to HC1.  Bootstrap is a
    normal-approximation pairs/cluster VCE, while jackknife uses unit-count t
    inference. CR2/CR3 default to G-1; optional dfadjust/Hansen use per-contrast
    design moments. Singular Hansen cases require a different generalized
    jackknife estimator and are rejected explicitly.
    """
    if covariance == "robust":
        covariance = "HC1"
    weight_type = {"aweight": "aw", "fweight": "fw", "pweight": "pw", "iweight": "iw",
                   "aweights": "aw", "fweights": "fw", "pweights": "pw", "iweights": "iw"}.get(weight_type, weight_type)
    if isinstance(covariance, str) and covariance.lower() == "hac":
        covariance = "HAC"
    if covariance not in _COVARIANCES:
        raise KernelError("unsupported_covariance", f"Unsupported OLS covariance: {covariance}.")
    if type(dfadjust) is not bool or type(hansen) is not bool:
        raise KernelError("invalid_covariance_options", "dfadjust and hansen must be boolean controls.")
    if dfadjust and covariance not in {"HC2", "HC3", "cluster_hc2", "cluster_hc3"}:
        raise KernelError("unsupported_inference", "dfadjust requires HC2/HC3 or one-way cluster HC2/HC3.")
    if hansen and covariance not in {"HC3", "cluster_hc3"}:
        raise KernelError("unsupported_inference", "Hansen scaling requires HC3 or one-way cluster HC3.")
    if not isinstance(x, Tensor) or not isinstance(y, Tensor) or x.device != y.device:
        raise KernelError("invalid_design", "Supply Torch design and outcome tensors on the same device.")
    if x.device.type not in {"cpu", "cuda"}:
        raise KernelError("unsupported_device", "Float64 OLS supports CPU and CUDA devices.")
    x, y = x.detach().to(dtype=torch.float64), y.detach().to(dtype=torch.float64)
    if (x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2 or x.shape[1] < 1
            or not isinstance(terms, list) or len(terms) != x.shape[1]
            or any(not isinstance(term, str) for term in terms) or len(set(terms)) != len(terms)):
        raise KernelError("invalid_design", "Expected a complete n-by-k design, outcome and unique term labels.")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise KernelError("non_finite_values", "The estimation sample must contain finite values.")
    warnings = []
    if weight_type == "pw" and covariance == "nonrobust":
        covariance = "HC1"
        warnings.append("Sampling weights use HC1 robust covariance by default.")
    with torch.inference_mode():
        w, nobs, kind = _weights(weights, weight_type, len(y), x, covariance)
        fit = _fit(x, y, w, terms, intercept)
        k = len(fit.params)
        if nobs <= k:
            raise KernelError("insufficient_observations", "Effective sample size must exceed the estimable design rank.")
        residual_df = nobs - k
        ss_resid = float(torch.dot(w, fit.resid.square()).item())
        if not math.isfinite(ss_resid):
            raise KernelError("numerical_failure", "Weighted residual variance exceeds finite float64 precision.")
        sigma2 = ss_resid / residual_df
        df_inference = residual_df
        information = {"covariance": covariance, "distribution": "t", "dfadjust": False, "hansen": False,
                       "correction": "Residual mean square uses N-k effective degrees of freedom."}
        if covariance == "nonrobust":
            basis_covariance = torch.eye(k, dtype=x.dtype, device=x.device) * sigma2
        elif covariance in {"HC0", "HC1", "HC2", "HC3"}:
            scalar = fit.resid if kind == "fw" else fit.resid * w.sqrt()
            if covariance in {"HC2", "HC3"}:
                complement = 1 - (fit.leverage / w if kind == "fw" else fit.leverage)
                valid = complement > 100 * _EPS * k
                power = .5 if covariance == "HC2" else 1.
                scalar = torch.where(valid, scalar / complement.clamp_min(torch.finfo(w.dtype).tiny).pow(power), 0)
                if not bool(valid.all()):
                    warnings.append("Unit-leverage observations were excluded from HC2/HC3 covariance using the Moore-Penrose convention.")
            scores = fit.q * scalar[:, None]
            correction = nobs / residual_df if covariance == "HC1" else 1.
            basis_covariance = (scores.T @ scores) * correction
            information["small_sample_correction"] = correction
            information["correction"] = f"{covariance} with factor {correction:.12g}; frequency weights use expanded-row leverage."
        elif covariance.startswith("cluster"):
            codes = _cluster_codes(clusters, len(y), x.device)
            basis_covariance, detail = _cluster_variance(fit, w, codes, nobs, covariance, warnings)
            information.update(detail)
            df_inference = min(detail["cluster_counts"]) - 1
            information["correction"] = ("CGM inclusion-exclusion with G/(G-1)*(N-1)/(N-k) for each intersection." if covariance == "cluster" else "Low-rank Moore-Penrose cluster leverage adjustment; G-1 inference without dfadjust/Hansen.")
        elif covariance == "HAC":
            basis_covariance, detail = _hac_variance(fit, w, time, lags, kernel, nobs, warnings)
            information.update(detail)
            information["correction"] = "HAC score covariance multiplied by N/(N-k); calendar gaps retained."
        elif covariance == "bootstrap":
            basis_covariance, detail = _bootstrap(fit, y, w, kind, nobs, clusters,
                                          [terms[i] for i in fit.kept], reps, seed, warnings)
            information.update(detail)
            information["distribution"] = "normal"
            information["correction"] = "Sample covariance across successful bootstrap estimates with B-1 divisor."
            df_inference = math.inf
        else:
            basis_covariance, detail = _jackknife(fit, w, kind, nobs, clusters)
            information.update(detail)
            df_inference = detail["resampling_units"] - 1
            information["correction"] = "Delete-unit covariance centered on the delete-unit mean, multiplied by (M-1)/M."
        basis_covariance = (basis_covariance + basis_covariance.T) / 2
        variance = fit.bridge @ basis_covariance @ fit.bridge.T
        raw_factor = None
        if covariance == "cluster" and len(codes) > 1:
            variance, raw_factor = _psd(variance, warnings, return_factor=True)
            if raw_factor is not None:
                information["psd_projection"] = "raw coefficient coordinates; not basis invariant"
        variance = (variance + variance.T) / 2
        if not bool(torch.isfinite(fit.params).all()) or not bool(torch.isfinite(variance).all()):
            raise KernelError("numerical_failure", "OLS produced non-finite coefficients or covariance.")
        adjustment = None
        if dfadjust or hansen:
            adjusted_codes = _cluster_codes(clusters, len(y), x.device)[0] if covariance.startswith("cluster") else None
            adjustment = _ContrastAdjustment(fit, w, kind, adjusted_codes,
                                             .5 if covariance.endswith("2") else 1., hansen)
            controls = [adjustment(row) for row in torch.eye(k, dtype=x.dtype, device=x.device)]
            information.update(dfadjust=True, hansen=hansen,
                               coefficient_df=[row["df"] for row in controls],
                               coefficient_scale=[row["scale"] for row in controls],
                               df_adjustment="Bell-McCaffrey/Satterthwaite contrast-specific",
                               scaling_semantics="scale multiplies t statistics and divides confidence interval width",
                               adjustment_source="Hansen 2025 A6/A7" if hansen else "Stata rregress adjusted degrees of freedom")
            warnings = [warning for warning in warnings if "Bell-McCaffrey dfadjust and Hansen corrections are not enabled" not in warning]
            warnings.append("Adjusted degrees of freedom apply to individual contrasts; joint model tests use conventional residual/cluster degrees of freedom.")
        probability = w / w.max()
        probability /= probability.sum()
        centered_y = y - y[0]
        mean_y_offset = torch.dot(probability, centered_y)
        deviations = centered_y - mean_y_offset if intercept else y
        ss_total = float(torch.dot(w, deviations.square()).item())
        r2 = 1 - ss_resid / ss_total if ss_total > 0 else math.nan
        r2_adj = 1 - (1 - r2) * (nobs - int(intercept)) / residual_df
        model_df = k - int(fit.constant is not None)
        # Gaussian model likelihood; sampling/importance weights are explicitly
        # a pseudo likelihood and do not imply a full complex-survey design.
        ll = -.5 * nobs * (math.log(2 * math.pi) + 1 + math.log(ss_resid / nobs)) if ss_resid > 0 else math.inf
        if kind == "aw":
            ll += .5 * float(w.log().sum().item())
        likelihood = "frequency_expanded_gaussian" if kind == "fw" else (
            "gaussian_wls" if kind in {None, "aw"} else "weighted_gaussian_pseudo_likelihood")
        slopes = [i for i in range(k) if i != fit.constant]
        f_value = math.nan
        f_df_num = model_df
        if slopes:
            coefficient = fit.params[slopes]
            slope_variance = variance[slopes][:, slopes]
            deviations = slope_variance.diagonal().clamp_min(0).sqrt()
            positive = deviations > 0
            if bool(positive.any()):
                deviations = deviations[positive]
                correlation = slope_variance[positive][:, positive] / deviations[:, None] / deviations[None, :]
                standardized = coefficient[positive] / deviations
                f_df_num = int(torch.linalg.matrix_rank(correlation).item())
                if f_df_num:
                    f_value = float((standardized @ torch.linalg.pinv(correlation, hermitian=True) @ standardized / f_df_num).item())
        information.update(df=df_inference, standard_errors=variance.diagonal().clamp_min(0).sqrt(),
                           model_test="F" if math.isfinite(df_inference) else "Wald chi-square",
                           f_statistic=f_value, f_df_num=f_df_num, f_df_denom=df_inference)
        metrics = {"nobs": nobs, "physical_nobs": len(y), "effective_nobs": nobs, "nparams": k,
                   "df_model": model_df, "df_resid": residual_df, "sigma2": sigma2,
                   "ss_resid": ss_resid, "ss_total": ss_total, "ss_model": ss_total - ss_resid,
                   "r_squared": r2, "r_squared_adj": r2_adj, "rmse": math.sqrt(sigma2),
                   "mean_y": float((y[0] + mean_y_offset).item()), "condition_number": fit.condition,
                   "log_likelihood": ll, "likelihood_convention": likelihood, "weight_type": kind,
                   "sum_weights": float(w.sum().item()), "covariance": covariance,
                   "f_statistic": f_value, "f_df_num": f_df_num, "f_df_denom": df_inference}
        omitted = [term for i, term in enumerate(terms) if i not in fit.kept]
        if omitted:
            warnings.append("Collinear terms omitted: " + ", ".join(omitted))
        prediction = _PredictionState(fit, basis_covariance, raw_factor)
        return {"params": fit.params, "covariance": variance, "x": fit.x, "y": y,
                "weights": w, "kept_indices": fit.kept, "omitted_terms": omitted,
                "terms": [terms[i] for i in fit.kept],
                "resid": fit.resid, "fitted": fit.fitted, "bread": fit.bread,
                "leverage": fit.leverage, "nobs": nobs, "df_resid": residual_df,
                "df_inference": df_inference, "sigma2": sigma2, "metrics": metrics,
                "inference": information, "warnings": warnings,
                "contrast_inference": adjustment,
                "prediction_variance": prediction.variance,
                "prediction_leverage": prediction.leverage,
                "prediction_fitted": prediction.fitted,
                "contrast_covariance": prediction.contrast_covariance}
