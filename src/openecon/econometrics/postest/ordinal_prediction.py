"""Saved ordered/multinomial probability contracts, evaluated by native Torch.

Ordered thresholds are reported directly; logistic scale and probit variance
are normalized, not extra estimated parameters. Multinomial blocks include
every non-base equation and retain the full saved cross-equation covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.postest.inference import _numeric
from openecon.engines.inference import critical_value
from openecon.frame import as_frame

ESTIMATORS = frozenset({"ologit", "oprobit", "mlogit"})
# Eight-node Gauss-Legendre integrates smooth normal density in narrow windows.
_NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.525532409916329,
    -0.1834346424956498,
    0.1834346424956498,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975363,
)
_WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.362683783378362,
    0.362683783378362,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


def _error(code, message):
    raise AnalysisError(code, message)


def _same(left, right):
    left, right = _label_scalar(left), _label_scalar(right)
    # bool is a distinct fitted label, never a synonym for numeric 0/1.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _label_scalar(value):
    # A fitted pandas/NumPy scalar is a legitimate selector; arrays and arbitrary
    # user objects with an item() method are not category labels.
    if pd.api.types.is_scalar(value) and type(value).__module__.split(".")[0] == "numpy":
        value = value.item()
    return value


def _exp_times(log_weight, *factors):
    """Multiply exponential weights after scaling each individual component.

    An underflowed density may become representable after multiplying by a
    large slope/design value. Preserve that product without an intermediate
    zero (or an overflowing product of several otherwise finite factors).
    """
    normalized = torch.ones_like(log_weight)
    exponent = log_weight
    zero = torch.zeros_like(log_weight, dtype=torch.bool)
    for factor in factors:
        # A vector maximum can erase a tiny component before another factor
        # rescales it. Component-wise logs retain arbitrary design ranges.
        scale = factor.detach().abs()
        zero = zero | (scale == 0)
        scale = torch.where(scale == 0, 1.0, scale)
        exponent = exponent + scale.log()
        normalized = normalized * (factor / scale)
    # Preserve the ordinary zero-factor gradient when the exponent is finite;
    # avoid a spurious inf*0 only where the other factors already overflow.
    exponent = torch.where(
        zero & (exponent > math.log(torch.finfo(torch.float64).max)), 0.0, exponent
    )
    return exponent.exp() * normalized


def category_standard_error(gradient, covariance):
    """A scaled quadratic form retaining SEs whose squared variance underflows."""
    if not bool(torch.isfinite(gradient).all()):
        _error("invalid_inference", "Category prediction gradients must be finite.")
    # Remove fixed parameters first. Their (possibly enormous) derivatives
    # cannot normalize away a tiny derivative for an uncertain parameter.
    positive = covariance.diagonal() > 0
    if not bool(positive.any()):
        return torch.zeros(gradient.shape[:-1], dtype=torch.float64)
    selected = gradient[..., positive]
    scales = covariance.diagonal()[positive].sqrt()
    normalized_cov = covariance[positive][:, positive] / scales[:, None] / scales[None, :]
    logarithms = selected.abs().log() + scales.log()
    maximum = logarithms.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isneginf(maximum), 0.0, maximum)
    normalized = selected.sign() * (logarithms - maximum).exp()
    variance = torch.einsum("...k,kl,...l->...", normalized, normalized_cov, normalized)
    absolute = torch.einsum(
        "...k,kl,...l->...", normalized.abs(), normalized_cov.abs(), normalized.abs()
    )
    if not bool(torch.isfinite(variance).all()) or bool(
        (variance < -64 * torch.finfo(torch.float64).eps * absolute).any()
    ):
        _error(
            "invalid_inference", "Category prediction variance is nonfinite or materially negative."
        )
    tiny = torch.isfinite(logarithms) & (logarithms - maximum < math.log(math.ulp(0.0)) / 2)
    if bool(
        (
            tiny.any(dim=-1) & (variance.abs() <= 64 * torch.finfo(torch.float64).eps * absolute)
        ).any()
    ):
        _error(
            "prediction_precision",
            "A nearly singular covariance projection and extreme gradient scales require higher precision.",
        )
    standard_error = (maximum.squeeze(-1) + variance.clamp_min(0).log() / 2).exp()
    if not bool(torch.isfinite(standard_error).all()):
        _error(
            "invalid_inference", "Category prediction standard errors exceed finite float64 range."
        )
    return standard_error


def _label_text(value):
    # Match the saved fitter's reporting convention without importing fitting
    # kernels/optimizers merely to reconstruct a reported equation name.
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.15g}" if isinstance(value, float) else str(value)


@dataclass(frozen=True)
class CategoryResponse:
    estimator: str
    labels: tuple
    blocks: tuple[tuple[int, ...], ...]
    local_terms: tuple[str, ...]
    cut_indices: tuple[int, ...]
    base: int | None

    def selection(self, outcome, kind):
        if self.estimator != "mlogit" and kind in {"xb", "stdp"}:
            if outcome is not None:
                _error(
                    "invalid_prediction_outcome",
                    "The ordered latent index is common to all outcomes; omit outcome.",
                )
            return (0,)
        if outcome is None:
            return tuple(range(len(self.labels)))
        outcome = _label_scalar(outcome)
        if not isinstance(outcome, (str, int, float, bool)) or (
            isinstance(outcome, float) and not math.isfinite(outcome)
        ):
            _error(
                "invalid_prediction_outcome",
                "outcome must be an actual finite fitted category label.",
            )
        matches = [i for i, label in enumerate(self.labels) if _same(label, outcome)]
        if len(matches) != 1:
            _error("invalid_prediction_outcome", "outcome is not a unique fitted category label.")
        return tuple(matches)

    def indexes(self, design, beta):
        if self.estimator != "mlogit":
            return (design.x @ beta + design.deterministic)[:, None]
        # The base index is fixed zero. Each other block uses identical encoded
        # predictor values but its own (possibly permuted) parameter positions.
        return torch.stack(
            [
                design.x[:, list(block)] @ beta[list(block)]
                if block
                else beta[0] * torch.zeros(len(design.x), dtype=torch.float64)
                for block in self.blocks
            ],
            dim=1,
        )

    def log_probabilities(self, design, beta):
        eta = self.indexes(design, beta)
        if not bool(torch.isfinite(eta).all()):
            _error(
                "non_finite_prediction", "The fitted category indexes exceed finite float64 range."
            )
        # Subtract a differentiable winning index, then log1p the OTHER mass.
        # logsumexp/softmax's 1-P diagonal rounds to zero when P rounds to one.
        winner = eta.argmax(dim=1, keepdim=True)
        relative = eta - eta.gather(1, winner)
        if not bool(torch.isfinite(relative).all()):
            _error("prediction_precision", "Multinomial index differences exceed float64 range.")
        other = torch.ones_like(relative, dtype=torch.bool).scatter(1, winner, False)
        remainder = torch.where(other, relative.exp(), 0.0).sum(dim=1, keepdim=True)
        return relative - torch.log1p(remainder)

    def log_density(self, z):
        if self.estimator == "oprobit":
            # Beyond this range the normal log density is below -1,125,000;
            # no product of the at most three finite gradient factors can
            # recover it. Avoid overflow in z squared, not a logistic cutoff.
            z = z.clamp(-1500, 1500)
            return -z.square() / 2 - math.log(2 * math.pi) / 2
        return -torch.nn.functional.softplus(z) - torch.nn.functional.softplus(-z)

    def curvature(self, z):
        return -z.clamp(-1500, 1500) if self.estimator == "oprobit" else -torch.tanh(z / 2)

    def ordered_edges(self, design, beta):
        eta = self.indexes(design, beta)
        cuts = beta[list(self.cut_indices)]
        if not bool(torch.isfinite(eta).all()):
            _error(
                "non_finite_prediction", "The fitted category indexes exceed finite float64 range."
            )
        edges = cuts[None, :] - eta
        if not bool(torch.isfinite(edges).all()) or not bool(
            torch.isfinite(cuts[1:] - cuts[:-1]).all()
        ):
            _error(
                "prediction_precision",
                "Ordered threshold/index differences exceed finite float64 range.",
            )
        return cuts, edges

    def density_product(self, z, *factors):
        return _exp_times(self.log_density(z)[..., None], *factors)

    def density_difference(self, a, width, *factors, curvature=False):
        """(f(a)-f(b)), or (f'(a)-f'(b)), times supplied vectors."""
        b = a + width
        if float(width.detach()) < 0.1:
            nodes = torch.tensor(_NODES, dtype=torch.float64)
            weights = torch.tensor(_WEIGHTS, dtype=torch.float64) / 2
            z = a[:, None] + width * (nodes + 1) / 2
            if curvature:
                multiplier = (
                    1 - z.clamp(-1500, 1500).square()
                    if self.estimator == "oprobit"
                    else 0.5 - 1.5 * torch.tanh(z / 2).square()
                )
            else:
                multiplier = (
                    z.clamp(-1500, 1500) if self.estimator == "oprobit" else torch.tanh(z / 2)
                )
            integrated = self.density_product(
                z, multiplier[..., None], *(factor[:, None, :] for factor in factors)
            )
            return width * (integrated * weights[None, :, None]).sum(dim=1)
        if self.estimator == "oprobit":
            # Difference of squared endpoints evaluated with the physical
            # width preserves an almost symmetric interval's small gradient.
            ratio = -width * (a + width / 2)
        else:
            ratio = self.log_density(b) - self.log_density(a)
            central = (a.abs() < 20) & (b.abs() < 20)
            safe_a, safe_b = a.clamp(-20, 20), b.clamp(-20, 20)
            change = (
                2
                * torch.sinh((safe_a + safe_b) / 4)
                * torch.sinh(width.clamp_max(40) / 4)
                / torch.cosh(safe_a / 2)
            )
            # The near-symmetric central formula avoids subtracting equal
            # log densities. In asymmetric tails the ordinary logs are stable.
            ratio = torch.where(central, -2 * torch.log1p(change), ratio)
        lower = self.density_product(a, (-torch.expm1(ratio.clamp_max(0)))[:, None], *factors)
        upper = self.density_product(b, torch.expm1((-ratio).clamp_max(0))[:, None], *factors)
        difference = torch.where((ratio <= 0)[:, None], lower, upper)
        if not curvature:
            return difference
        change = (
            width.expand_as(a)
            if self.estimator == "oprobit"
            else 2 * self.middle_probability(a, width)
        )
        answer = difference * self.curvature(a)[:, None] + self.density_product(
            b, change[:, None], *factors
        )
        if self.estimator == "oprobit":
            outside = (a.abs() > 1500) | (b.abs() > 1500)
            direct = self.density_product(
                a, self.curvature(a)[:, None], *factors
            ) - self.density_product(b, self.curvature(b)[:, None], *factors)
            answer = torch.where(outside[:, None], direct, answer)
        return answer

    def middle_probability(self, a, width):
        b = a + width
        if self.estimator == "ologit":
            return (
                torch.nn.functional.logsigmoid(b)
                + torch.nn.functional.logsigmoid(-a)
                + torch.log(-torch.expm1(-width))
            ).exp()

        def cdf(z):
            return 0.5 * torch.erfc(-z / math.sqrt(2))

        return torch.where(a >= 0, cdf(-a) - cdf(-b), cdf(b) - cdf(a))

    def probabilities(self, design, beta):
        eta = self.indexes(design, beta)
        if not bool(torch.isfinite(eta).all()):
            _error(
                "non_finite_prediction", "The fitted category indexes exceed finite float64 range."
            )
        if self.estimator == "mlogit":
            return self.log_probabilities(design, beta).exp()
        cuts = beta[list(self.cut_indices)]
        edges = cuts[None, :] - eta
        if not bool(torch.isfinite(edges).all()) or not bool(
            torch.isfinite(cuts[1:] - cuts[:-1]).all()
        ):
            _error(
                "prediction_precision",
                "Ordered threshold/index subtraction exceeds finite float64 range.",
            )
        if self.estimator == "ologit":

            def cdf(z):
                return torch.where(z <= 0, torch.sigmoid(z), 1 - torch.sigmoid(-z))

            first, last = cdf(edges[:, :1]), cdf(-edges[:, -1:])
            a, b = edges[:, :-1], edges[:, 1:]
            width = cuts[1:] - cuts[:-1]
            middle = (
                torch.nn.functional.logsigmoid(b)
                + torch.nn.functional.logsigmoid(-a)
                + torch.log(-torch.expm1(-width))
            ).exp()
        else:

            def cdf(z):
                return 0.5 * torch.erfc(-z / math.sqrt(2))

            first, last = cdf(edges[:, :1]), cdf(-edges[:, -1:])
            a, b = edges[:, :-1], edges[:, 1:]
            middle = torch.where(a >= 0, cdf(-a) - cdf(-b), cdf(b) - cdf(a))
        # Use the physical threshold difference, not rounded shifted edges.
        # Integrated density keeps location/threshold second derivatives even
        # for narrow symmetric logistic windows whose endpoint gradients cancel.
        width = cuts[1:] - cuts[:-1]
        narrow = width < 0.1
        if bool(narrow.any()):
            nodes = torch.tensor(_NODES, dtype=torch.float64)
            weights = torch.tensor(_WEIGHTS, dtype=torch.float64) / 2
            z = a[:, narrow, None] + width[narrow][None, :, None] * (nodes + 1) / 2
            if self.estimator == "oprobit":
                density = self.log_density(z).exp()
            else:
                safe = z.clamp(-20, 20)
                central = 1 / (4 * torch.cosh(safe / 2).square())
                tails = (-torch.nn.functional.softplus(z) - torch.nn.functional.softplus(-z)).exp()
                density = torch.where(z.abs() < 20, central, tails)
            middle = middle.clone()
            middle[:, narrow] = width[narrow][None, :] * (density @ weights)
        return torch.cat((first, middle, last), dim=1)

    def values(self, design, beta, kind):
        return self.indexes(design, beta) if kind == "xb" else self.probabilities(design, beta)

    def difference(self, alternative, baseline, beta, kind):
        if kind == "xb":
            return self.indexes(alternative, beta) - self.indexes(baseline, beta)

        def logs(design):
            if self.estimator == "mlogit":
                return self.log_probabilities(design, beta)
            probabilities = self.probabilities(design, beta)
            edges = beta[list(self.cut_indices)][None, :] - self.indexes(design, beta)
            if self.estimator == "ologit":
                first = -torch.nn.functional.softplus(-edges[:, :1])
                last = -torch.nn.functional.softplus(edges[:, -1:])
            else:

                def logcdf(z):
                    return torch.where(
                        z <= 0,
                        torch.special.log_ndtr(z),
                        torch.log1p(-0.5 * torch.erfc(z / math.sqrt(2))),
                    )

                first, last = logcdf(edges[:, :1]), logcdf(-edges[:, -1:])
            return torch.cat((first, probabilities[:, 1:-1].log(), last), dim=1)

        a, b = logs(alternative), logs(baseline)
        both_zero = torch.isneginf(a) & torch.isneginf(b)
        a, b = torch.where(both_zero, 0.0, a), torch.where(both_zero, 0.0, b)
        result = torch.where(
            a >= b,
            a.exp() * -torch.expm1((b - a).clamp_max(0)),
            b.exp() * torch.expm1((a - b).clamp_max(0)),
        )
        return torch.where(both_zero, 0.0, result)

    def slopes(self, beta, variable):
        if variable not in self.local_terms:
            return beta[0] * torch.zeros(len(self.blocks), dtype=torch.float64)
        local = self.local_terms.index(variable)
        return torch.stack([beta[block[local]] if block else beta[0] * 0 for block in self.blocks])

    def effects(self, design, beta, variable, kind):
        slope = self.slopes(beta, variable)
        if kind == "xb":
            return slope[None, :].expand(len(design.x), -1)
        if self.estimator == "mlogit":
            logs = self.log_probabilities(design, beta)
            outputs = []
            for j in range(len(self.labels)):
                total = beta[0] * torch.zeros(len(design.x), dtype=torch.float64)
                for k in range(len(self.labels)):
                    if j != k:
                        difference = torch.ones_like(logs[:, :1]) * (slope[j] - slope[k])
                        total = (
                            total + _exp_times((logs[:, j] + logs[:, k])[:, None], difference)[:, 0]
                        )
                outputs.append(total)
            return torch.stack(outputs, dim=1)
        cuts, edges = self.ordered_edges(design, beta)
        factor = torch.ones((len(design.x), 1), dtype=torch.float64) * slope[0]
        values = [-self.density_product(edges[:, 0], factor)[:, 0]]
        values.extend(
            self.density_difference(edges[:, j], cuts[j + 1] - cuts[j], factor)[:, 0]
            for j in range(len(cuts) - 1)
        )
        values.append(self.density_product(edges[:, -1], factor)[:, 0])
        return torch.stack(values, dim=1)

    def jacobian(self, design, beta, kind, variable=None):
        """Full parameter gradients without underflowed intermediate densities.

        These are analytic derivatives in the reported parameterization, not
        finite differences. They include every threshold and equation block.
        """
        n, parameters = design.x.shape
        if kind == "derivative_xb":
            units = []
            for block in self.blocks:
                unit = torch.zeros_like(design.x)
                if block and variable in self.local_terms:
                    unit[:, block[self.local_terms.index(variable)]] = 1
                units.append(unit)
            return torch.stack(units, dim=1)
        if self.estimator == "mlogit":
            indexes = []
            for block in self.blocks:
                matrix = torch.zeros_like(design.x)
                matrix[:, list(block)] = design.x[:, list(block)]
                indexes.append(matrix)
            if kind == "xb":
                return torch.stack(indexes, dim=1)
            logs = self.log_probabilities(design, beta)
            log_other = [
                torch.logsumexp(logs[:, [k for k in range(len(self.labels)) if k != j]], dim=1)
                for j in range(len(self.labels))
            ]
            gradients = []
            for j in range(len(self.labels)):
                probability_gradient = torch.zeros_like(design.x)
                for k, block in enumerate(self.blocks):
                    if not block:
                        continue
                    # Own-equation 1-P is the other-category mass, never a
                    # rounded subtraction from a dominant probability.
                    weight = log_other[j] if j == k else logs[:, k]
                    sign = 1 if j == k else -1
                    probability_gradient[:, list(block)] = sign * _exp_times(
                        (logs[:, j] + weight)[:, None], indexes[k][:, list(block)]
                    )
                gradients.append(probability_gradient)
            if kind == "response":
                return torch.stack(gradients, dim=1)
            slope = self.slopes(beta, variable)
            eta = self.indexes(design, beta)
            contrasts = []
            for r in range(len(self.labels)):
                others = [t for t in range(len(self.labels)) if t != r]
                anchor_local = eta[:, others].argmax(dim=1)
                anchor = torch.tensor(others, dtype=torch.long)[anchor_local]
                reference = eta.gather(1, anchor[:, None])[:, 0]
                reference_log = logs.gather(1, anchor[:, None])[:, 0]
                positive = reference >= eta[:, r]
                factor = torch.where(
                    positive,
                    -torch.expm1((eta[:, r] - reference).clamp_max(0)),
                    torch.expm1((reference - eta[:, r]).clamp_max(0)),
                )
                leading_log = torch.where(positive, reference_log, logs[:, r])
                include = (torch.arange(len(self.labels))[None, :] != r) & (
                    torch.arange(len(self.labels))[None, :] != anchor[:, None]
                )
                tail_log = torch.logsumexp(torch.where(include, logs, -float("inf")), dim=1)
                # 1-2P_r = (P_largest_other-P_r) + other remaining mass.
                # The first signed difference uses index differences; a tiny
                # third category survives even if the two leaders equal .5.
                contrasts.append((leading_log, factor, tail_log))
            slope_units = []
            for block in self.blocks:
                unit = torch.zeros_like(design.x)
                if block and variable in self.local_terms:
                    unit[:, block[self.local_terms.index(variable)]] = 1
                slope_units.append(unit)
            outputs = []
            for j in range(len(self.labels)):
                gradient = torch.zeros_like(design.x)
                for k in range(len(self.labels)):
                    if j == k:
                        continue
                    weight = (logs[:, j] + logs[:, k])[:, None]
                    difference = torch.ones((n, 1), dtype=torch.float64) * (slope[j] - slope[k])
                    for r, block in enumerate(self.blocks):
                        if not block:
                            continue
                        matrix = indexes[r][:, list(block)]
                        if r in {j, k}:
                            leading_log, factor, tail_log = contrasts[r]
                            contribution = _exp_times(
                                weight + leading_log[:, None], difference, factor[:, None], matrix
                            )
                            contribution = contribution + _exp_times(
                                weight + tail_log[:, None], difference, matrix
                            )
                        else:
                            contribution = -2 * _exp_times(
                                weight + logs[:, r, None], difference, matrix
                            )
                        gradient[:, list(block)] = gradient[:, list(block)] + contribution
                    gradient = gradient + _exp_times(weight, slope_units[j] - slope_units[k])
                outputs.append(gradient)
            return torch.stack(outputs, dim=1)
        if kind == "xb":
            return design.x[:, None, :]
        cuts, edges = self.ordered_edges(design, beta)

        def unit(position):
            value = torch.zeros_like(design.x)
            value[:, position] = 1
            return value

        units = [unit(position) for position in self.cut_indices]
        if kind == "response":
            first = self.density_product(edges[:, 0], units[0] - design.x)
            last = self.density_product(edges[:, -1], design.x - units[-1])
            middle = [
                self.density_difference(edges[:, j], cuts[j + 1] - cuts[j], design.x)
                + self.density_product(edges[:, j + 1], units[j + 1])
                - self.density_product(edges[:, j], units[j])
                for j in range(len(cuts) - 1)
            ]
        else:
            slope = self.slopes(beta, variable)[0]
            factor = torch.ones((n, 1), dtype=torch.float64) * slope
            slope_unit = (
                unit(self.blocks[0][self.local_terms.index(variable)])
                if variable in self.local_terms
                else torch.zeros_like(design.x)
            )
            first = -self.density_product(edges[:, 0], slope_unit) - self.density_product(
                edges[:, 0], factor, self.curvature(edges[:, 0])[:, None], units[0] - design.x
            )
            last = self.density_product(edges[:, -1], slope_unit) + self.density_product(
                edges[:, -1], factor, self.curvature(edges[:, -1])[:, None], units[-1] - design.x
            )
            middle = [
                self.density_difference(edges[:, j], cuts[j + 1] - cuts[j], slope_unit)
                - self.density_difference(
                    edges[:, j], cuts[j + 1] - cuts[j], factor, design.x, curvature=True
                )
                + self.density_product(
                    edges[:, j], factor, self.curvature(edges[:, j])[:, None], units[j]
                )
                - self.density_product(
                    edges[:, j + 1], factor, self.curvature(edges[:, j + 1])[:, None], units[j + 1]
                )
                for j in range(len(cuts) - 1)
            ]
        return torch.stack([first, *middle, last], dim=1)


def saved_categories(result, state, features):
    """Validate complete saved threshold/equation state and design accounting."""
    estimator = result.spec.estimator
    if estimator not in ESTIMATORS:
        return None, features, None
    labels = result.extra.get("categories")
    limit = 300 if estimator == "mlogit" else 500
    if (
        not isinstance(labels, list)
        or not 2 <= len(labels) <= limit
        or any(
            not isinstance(v, (str, bool, int, float))
            or isinstance(v, float)
            and not math.isfinite(v)
            for v in labels
        )
    ):
        _error("invalid_result", "Saved outcome categories are invalid or missing.")
    try:
        pd.Categorical([], categories=labels)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            "invalid_result", "Saved outcome categories must be distinct valid labels."
        ) from exc
    if isinstance(result.metrics.get("n_categories"), bool) or result.metrics.get(
        "n_categories"
    ) != len(labels):
        _error("invalid_result", "Saved category count differs from fitted categories.")
    counts = _numeric(result.extra.get("category_counts"), code="invalid_result", ndim=1)
    if counts.shape != (len(labels),) or bool((counts <= 0).any()):
        _error("invalid_result", "Saved category frequencies must be positive and complete.")
    omitted = result.provenance.get("omitted_terms", [])
    if (
        not isinstance(omitted, list)
        or any(not isinstance(v, str) for v in omitted)
        or len(set(omitted)) != len(omitted)
        or any(v not in features for v in omitted)
    ):
        _error("invalid_result", "Saved design omissions are invalid.")
    local_terms = tuple(term for term in features if term not in omitted)
    if not local_terms and estimator == "mlogit":
        _error("invalid_result", "Saved multinomial model has no retained design terms.")
    if estimator != "mlogit":
        if result.spec.intercept is not False or result.extra.get("link") != (
            "logit" if estimator == "ologit" else "probit"
        ):
            _error("invalid_result", "Saved ordered link or intercept normalization is invalid.")
        cuts = tuple(f"/cut{i}" for i in range(1, len(labels)))
        if set(state.terms) != set(local_terms) | set(cuts):
            _error(
                "invalid_result", "Saved ordered slope/cutpoint terms are incomplete or unexpected."
            )
        for coefficient in result.coefficients:
            if coefficient.equation != (None if coefficient.term in cuts else result.spec.outcome):
                _error(
                    "invalid_result",
                    "Saved ordered coefficient equations disagree with the design.",
                )
        positions = tuple(state.terms.index(name) for name in cuts)
        retained = _numeric(result.extra.get("cutpoints"), code="invalid_result", ndim=1)
        values = state.beta[list(positions)]
        if (
            retained.shape != values.shape
            or not torch.equal(retained, values)
            or bool((values[1:] <= values[:-1]).any())
        ):
            _error("invalid_result", "Saved ordered cutpoints are stale, unordered or incomplete.")
        order = result.extra.get("category_order")
        if (
            not isinstance(order, str)
            or order not in {"ordered Categorical", "sorted numeric values"}
            or (
                order == "sorted numeric values"
                and (
                    any(isinstance(x, str) for x in labels)
                    or any(a >= b for a, b in zip(labels, labels[1:]))
                )
            )
        ):
            _error("invalid_result", "Saved ordered outcome order is invalid.")
        block = tuple(state.terms.index(term) for term in local_terms)
        return (
            CategoryResponse(estimator, tuple(labels), (block,), local_terms, positions, None),
            features,
            set(local_terms),
        )
    names = [_label_text(label) for label in labels]
    if len(set(names)) != len(names):
        _error("invalid_result", "Saved multinomial equation labels collide.")
    matching = [i for i, label in enumerate(labels) if _same(label, result.extra.get("base"))]
    if len(matching) != 1:
        _error("invalid_result", "Saved multinomial base category is invalid.")
    base = matching[0]
    requested = result.spec.options.get("base")
    if (
        requested is not None
        and not _same(requested, labels[base])
        or requested is None
        and base != int(counts.argmax())
    ):
        _error("invalid_result", "Saved base category differs from the fitted base selection.")
    if result.extra.get("equations") != [name for i, name in enumerate(names) if i != base]:
        _error("invalid_result", "Saved multinomial equation order is stale or incomplete.")
    mapped, blocks = {}, []
    for i, name in enumerate(names):
        terms = [] if i == base else [f"{name}:{term}" for term in local_terms]
        blocks.append(tuple(state.terms.index(term) for term in terms if term in state.terms))
        mapped.update(
            {f"{name}:{term}": features[term] for term in local_terms} if i != base else {}
        )
    if set(mapped) != set(state.terms):
        _error(
            "invalid_result", "Saved multinomial coefficient blocks are incomplete or unexpected."
        )
    if any(
        coefficient.equation
        != next(
            names[i]
            for i, block in enumerate(blocks)
            if state.terms.index(coefficient.term) in block
        )
        for coefficient in result.coefficients
    ):
        _error(
            "invalid_result", "Saved multinomial coefficient equations disagree with their terms."
        )
    return (
        CategoryResponse(estimator, tuple(labels), tuple(blocks), local_terms, (), base),
        mapped,
        set(mapped),
    )


def predict_categories(
    model, design, original, positions, kind, outcome, significance, interval, term
):
    choice = model.choice
    selected = choice.selection(outcome, kind)
    if kind == "derivative":
        if term not in model.predictors or term in model.categories:
            _error("invalid_prediction_term", "Choose one continuous original predictor.")
        if term == model.offset:
            _error(
                "unsupported_margins_transform",
                "A predictor also used as offset needs an explicit intervention contract.",
            )
    if interval is not None or kind == "stdp":
        # The helper builds every outcome before selecting columns. Include
        # live gradients/intermediates, rather than just returned columns.
        categories = 1 if choice.estimator != "mlogit" and kind == "stdp" else len(choice.labels)
        width = max(8, 4 * categories) * len(model.state.terms)
        if 8 * len(design.x) * max(1, width) > _MAX_DESIGN_BYTES:
            _error(
                "prediction_memory_limit",
                "Category prediction gradients exceed 256 MiB; predict smaller batches.",
            )

    def evaluate(beta):
        values = (
            choice.effects(design, beta, term, "response")
            if kind == "derivative"
            else choice.values(design, beta, "xb" if kind == "stdp" else kind)
        )
        return values[:, list(selected)]

    with torch.inference_mode(False), torch.enable_grad():
        from openecon.econometrics.postest.prediction import _Design

        design = _Design(
            design.x.detach().clone(),
            design.deterministic.detach().clone(),
            design.scale.detach().clone(),
        )
        beta = model.state.beta.detach().clone()
        value = evaluate(beta)
        jac = (
            choice.jacobian(design, beta, "xb" if kind == "stdp" else kind, term)[:, list(selected)]
            if interval is not None or kind == "stdp"
            else None
        )
    value = value.detach()
    columns = {}
    if jac is not None:
        jac = jac.detach()
        se = category_standard_error(jac, model.state.covariance)
        if kind == "stdp":
            value = se
        else:
            critical = critical_value(significance, model.state.df)
            columns.update(
                std_error=se, ci_low=value - critical * se, ci_high=value + critical * se
            )
    label = f"dydx[{term}]" if kind == "derivative" else kind
    columns = {label: value, **columns}
    common = choice.estimator != "mlogit" and kind in {"xb", "stdp"}
    wide = outcome is None and not common
    output, mapping = {}, {}
    for name, values in columns.items():
        if not bool(torch.isfinite(values).all()):
            _error("non_finite_prediction", "Category predictions or intervals are nonfinite.")
        for j, position in enumerate(selected):
            column = (
                f"{name}[{json.dumps(choice.labels[position], ensure_ascii=False)}]"
                if wide
                else name
            )
            scattered = torch.full((len(original),), float("nan"), dtype=torch.float64)
            scattered[positions] = values[:, j]
            output[column] = scattered.tolist()
            if not common:
                mapping[column] = choice.labels[position]
    result = as_frame(pd.DataFrame(output, index=original.index))
    present = set(positions.tolist())
    result.attrs.update(
        kind=kind,
        estimator=model.result.spec.estimator,
        precision="float64",
        outcome_labels=list(choice.labels),
        outcome=choice.labels[selected[0]] if outcome is not None else None,
        outcome_columns=mapping,
        base_outcome=choice.labels[choice.base] if choice.base is not None else None,
        missing_row_positions=[i for i in range(len(original)) if i not in present],
        interval_method="pointwise delta method" if interval else None,
        response_definition="fitted-category probability"
        if kind not in {"xb", "stdp"}
        else "category versus base log odds"
        if choice.estimator == "mlogit"
        else "common latent ordered index",
    )
    return result
