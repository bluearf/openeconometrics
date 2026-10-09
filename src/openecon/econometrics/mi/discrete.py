"""Native discrete FCS with stated proper priors and finite exact-target MH.

Every conditional update targets its declared posterior with a symmetric
proposal. A finite number of transitions does not establish stationarity or
compatibility of the collection of conditional models.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real

import torch
from pydantic import model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import MIResult, admit, check_seed, make_result
from openecon.resources import plan_workspace


_DTYPE = torch.float64
_MAX_DIMENSION = 128
_MAX_UPDATES = 100_000
_MAX_COUNT = 1_000_000


def _integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise AnalysisError("invalid_spec", f"{name} must be an integer in [{lower}, {upper}].")
    return int(value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("invalid_spec", f"{name} must be a finite positive real number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or not 0 < number * number < math.inf:
        raise AnalysisError("invalid_spec", f"{name} must have a finite positive square.")
    return number


def _finite(value, context):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{context} is not finite.")


def _negative_infinity(reference):
    return reference.new_full((), -math.inf)


def _poisson_log_posterior(x, y, beta, prior_scale):
    """Poisson log-link likelihood plus N(0, scale² I), without y! constants.

    Exponential overflow gives a rejected negative-infinite target, never a
    clipped rate or an exception that changes the proposal distribution.
    """
    eta = x @ beta
    rates = torch.exp(eta)
    target = (y * eta - rates).sum() - beta.square().sum() / (2 * prior_scale ** 2)
    return target if bool(torch.isfinite(target)) else _negative_infinity(beta)


def _ordinal_cutpoints(theta, design_dimension, n_categories):
    first = theta[design_dimension:design_dimension + 1]
    gaps = torch.exp(theta[design_dimension + 1:])
    if theta.numel() != design_dimension + n_categories - 1:
        raise AnalysisError("invalid_spec", "Ordinal posterior coordinates have the wrong dimension.")
    return torch.cat((first, first + torch.cumsum(gaps, dim=0))), gaps


def _ordinal_log_probabilities(x, theta, n_categories):
    """Ordered-logit PMF, stable even when ordinary CDF subtraction cancels."""
    d = x.shape[1]
    cutpoints, gaps = _ordinal_cutpoints(theta, d, n_categories)
    eta = x @ theta[:d]
    if (not bool(torch.isfinite(eta).all()) or not bool(torch.isfinite(cutpoints).all())
            or not bool((gaps > 0).all()) or not bool((cutpoints[1:] > cutpoints[:-1]).all())):
        return theta.new_full((x.shape[0], n_categories), -math.inf)
    shifted = cutpoints.unsqueeze(0) - eta.unsqueeze(1)
    # sigmoid(b)-sigmoid(a) = sigmoid(b)*sigmoid(-a)*(1-exp(a-b)).
    # Using the positive gaps directly avoids subtracting nearly equal CDFs.
    middle = (-torch.nn.functional.softplus(-shifted[:, 1:])
              - torch.nn.functional.softplus(shifted[:, :-1])
              + torch.log(-torch.expm1(-gaps)).unsqueeze(0))
    return torch.cat((-torch.nn.functional.softplus(-shifted[:, :1]), middle,
                      -torch.nn.functional.softplus(shifted[:, -1:])), dim=1)


def _ordinal_log_posterior(x, y, theta, prior_scale, n_categories):
    log_probability = _ordinal_log_probabilities(x, theta, n_categories)
    target = log_probability.gather(1, y[:, None]).sum() - theta.square().sum() / (
        2 * prior_scale ** 2
    )
    return target if bool(torch.isfinite(target)) else _negative_infinity(theta)


def _multinomial_log_probabilities(x, theta, n_categories):
    coefficients = theta.reshape(x.shape[1], n_categories - 1)
    logits = torch.cat((x.new_zeros((x.shape[0], 1)), x @ coefficients), dim=1)
    return logits - torch.logsumexp(logits, dim=1, keepdim=True)


def _multinomial_log_posterior(x, y, theta, prior_scale, n_categories):
    log_probability = _multinomial_log_probabilities(x, theta, n_categories)
    target = log_probability.gather(1, y[:, None]).sum() - theta.square().sum() / (
        2 * prior_scale ** 2
    )
    return target if bool(torch.isfinite(target)) else _negative_infinity(theta)


def _mh_draw(log_posterior, dimension, generator, *, prior_scale, proposal_scale,
             burn, steps, initial=None):
    """Fixed isotropic symmetric random walk in the declared prior coordinates."""
    dimension = _integer(dimension, "posterior dimension", 1, _MAX_DIMENSION)
    burn = _integer(burn, "MH burn", 0, 100_000)
    steps = _integer(steps, "MH steps", 1, 100_000)
    _positive(prior_scale, "prior_scale")
    proposal_scale = _positive(proposal_scale, "proposal_scale")
    if initial is not None and (not isinstance(initial, torch.Tensor)
                               or initial.shape != (dimension,)
                               or initial.dtype != _DTYPE or initial.device.type != "cpu"):
        raise AnalysisError("invalid_spec", "Initial posterior coordinates must be a CPU float64 vector.")
    state = (torch.zeros(dimension, dtype=_DTYPE, device="cpu")
             if initial is None else initial.detach().clone())
    target = log_posterior(state)
    _finite(target, "The initial conditional posterior")
    accepted_burn = accepted_sampling = rejected_nonfinite = 0
    for transition in range(burn + steps):
        candidate = state + proposal_scale * torch.randn(
            dimension, dtype=_DTYPE, device="cpu", generator=generator
        )
        candidate_target = log_posterior(candidate)
        log_uniform = torch.rand((), dtype=_DTYPE, device="cpu", generator=generator).log()
        if not bool(torch.isfinite(candidate_target)):
            rejected_nonfinite += 1
        elif bool(log_uniform < candidate_target - target):
            state, target = candidate, candidate_target
            if transition < burn:
                accepted_burn += 1
            else:
                accepted_sampling += 1
    return state, {
        "burn_proposals": burn, "sampling_proposals": steps,
        "accepted_burn": accepted_burn, "accepted_sampling": accepted_sampling,
        "acceptance_rate": (accepted_burn + accepted_sampling) / (burn + steps),
        "rejected_nonfinite": rejected_nonfinite, "final_log_posterior": float(target),
    }


def _poisson_draw(x, y, generator, *, prior_scale, proposal_scale, burn, steps, initial=None):
    """Return beta and MH diagnostics; predictive drawing belongs to the caller."""
    if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
            or x.ndim != 2 or y.shape != (x.shape[0],) or not 1 <= x.shape[0] <= 10_000
            or not 1 <= x.shape[1] <= _MAX_DIMENSION
            or x.dtype != _DTYPE or y.dtype != _DTYPE
            or x.device.type != "cpu" or y.device.type != "cpu"):
        raise AnalysisError("invalid_spec", "Poisson X and y must be bounded CPU float64 arrays.")
    _finite(x, "The Poisson design")
    _finite(y, "The Poisson observed counts")
    if not bool(((y >= 0) & (y <= _MAX_COUNT) & (y == y.floor())).all()):
        raise AnalysisError("invalid_count_outcome", "Poisson observations must be integer counts in [0, 1,000,000].")
    prior_scale = _positive(prior_scale, "prior_scale")
    return _mh_draw(lambda beta: _poisson_log_posterior(x, y, beta, prior_scale),
                    x.shape[1], generator, prior_scale=prior_scale,
                    proposal_scale=proposal_scale, burn=burn, steps=steps, initial=initial)


def _predictors(names, targets, requested):
    if requested is None:
        return {target: tuple(name for name in names if name != target) for target in targets}
    if not isinstance(requested, Mapping) or set(requested) != set(targets):
        raise AnalysisError("invalid_spec", "predictors must map exactly the incomplete columns.")
    result = {}
    for target in targets:
        selected = requested[target]
        if (isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence)
                or any(not isinstance(name, str) for name in selected)):
            raise AnalysisError("invalid_spec", f"predictors[{target!r}] must be a sequence of column names.")
        selected = tuple(selected)
        if len(set(selected)) != len(selected) or target in selected or any(name not in names for name in selected):
            raise AnalysisError("invalid_spec", "Predictors must be distinct other selected columns.")
        result[target] = selected
    return result


def _category_spec(targets, methods, categories):
    categorical = tuple(target for target in targets if methods[target] != "poisson")
    if categories is None and not categorical:
        return {}
    if not isinstance(categories, Mapping) or set(categories) != set(categorical):
        raise AnalysisError("invalid_categories", "categories must declare exactly the ordinal and multinomial targets.")
    result = {}
    for target in categorical:
        codes = categories[target]
        if isinstance(codes, (str, bytes)) or not isinstance(codes, Sequence) or not 3 <= len(codes) <= 8:
            raise AnalysisError("invalid_categories", "Declare 3..8 numeric category codes in their fixed order.")
        if any(isinstance(code, bool) or not isinstance(code, Real) or not math.isfinite(float(code))
               or abs(float(code)) > 1e140 for code in codes):
            raise AnalysisError("invalid_categories", "Category codes must be finite real numbers, excluding booleans.")
        codes = tuple(float(code) for code in codes)
        if len(set(codes)) != len(codes):
            raise AnalysisError("invalid_categories", "Category codes must be distinct.")
        result[target] = codes
    return result


def _rates(eta, target):
    rates = torch.exp(eta)
    if not bool(torch.isfinite(rates).all()) or bool((rates > _MAX_COUNT).any()):
        raise AnalysisError("prediction_limit", f"Poisson predicted rates for {target!r} must be finite and <=1,000,000; no clipping is used.")
    return rates


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


def _close(actual, expected):
    return _number(actual) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)


def _checked_rows(rows, count, dimension, context):
    _require(isinstance(rows, (list, tuple)) and len(rows) == count,
             f"Discrete saved {context} has the wrong row count.")
    _require(all(isinstance(row, (list, tuple)) and len(row) == dimension
                 and all(_number(value) for value in row) for row in rows),
             f"Discrete saved {context} must contain finite vectors of the declared dimension.")


class MIDiscreteResult(MIResult):
    """Full discrete FCS state with operation-specific restore and copy checks."""

    @model_validator(mode="after")
    def _validate_discrete_state(self):
        meta = self.metadata
        _require(self.method == "mi_chained" and meta.get("api") == "mi_discrete"
                 and meta.get("fcs") is True, "Discrete result operation identity is invalid.")
        names = self.columns
        n, p, m = len(self.original), len(names), len(self.completed_matrices)
        targets = tuple(name for j, name in enumerate(names)
                        if any(row[j] is None for row in self.original))
        methods, predictors, categories = meta.get("methods"), meta.get("predictors"), meta.get("categories")
        _require(isinstance(methods, Mapping) and set(methods) == set(targets)
                 and all(value in ("poisson", "ordinal", "multinomial") for value in methods.values()),
                 "Discrete methods must identify every incomplete target.")
        try:
            predictor_spec = _predictors(names, targets, predictors)
            codes = _category_spec(targets, methods, categories)
        except AnalysisError as exc:
            raise ValueError("Invalid discrete predictor/category contracts.") from exc
        _require(meta.get("visit_order") == targets, "Discrete update order disagrees with selected targets.")
        burn, iterations = meta.get("burn"), meta.get("iterations")
        _require(type(burn) is int and 0 <= burn <= 1000 and type(iterations) is int
                 and iterations >= 1 and burn + iterations <= 1000,
                 "Discrete sweep counts are invalid.")
        cycles = burn + iterations
        _require(meta.get("cycles_per_chain") == cycles and m * cycles * len(targets) <= _MAX_UPDATES,
                 "Discrete trace length exceeds its declared sweep contract.")
        prior, sampler = meta.get("prior", {}), meta.get("sampler", {})
        _require(isinstance(prior, Mapping) and isinstance(sampler, Mapping),
                 "Discrete prior and sampler declarations must be mappings.")
        scale = prior.get("scale")
        _require(prior.get("family") == "independent Gaussian" and _number(prior.get("mean"))
                 and prior["mean"] == 0.0 and _number(scale) and float(scale) > 0
                 and 0 < float(scale) * float(scale) < math.inf,
                 "Discrete results require the stated proper Gaussian prior.")
        _require(prior.get("ordinal") == "no-intercept raw coefficients, first cutpoint, and log positive gaps; prior in these coordinates; no Jacobian"
                 and prior.get("multinomial") == "raw nonbaseline coefficients including intercept; baseline logit fixed at zero"
                 and prior.get("poisson") == "raw coefficients including intercept",
                 "Discrete prior coordinates are invalid.")
        mh_burn, mh_steps = sampler.get("burn_per_update"), sampler.get("steps_per_update")
        _require(type(mh_burn) is int and 0 <= mh_burn <= 100_000 and type(mh_steps) is int
                 and 1 <= mh_steps <= 100_000 and _number(sampler.get("proposal_scale"))
                 and float(sampler["proposal_scale"]) > 0
                 and 0 < float(sampler["proposal_scale"]) * float(sampler["proposal_scale"]) < math.inf
                 and sampler.get("kernel") == "symmetric isotropic Gaussian random-walk Metropolis"
                 and sampler.get("exact_target") is True and sampler.get("stationarity_claim") is False
                 and sampler.get("nonfinite_proposals") == "rejected without clipping",
                 "Discrete finite-MH sampler declaration is invalid.")
        convergence = meta.get("convergence", {})
        _require(isinstance(convergence, Mapping) and set(convergence) == {"assessed", "converged"}
                 and all(value is False for value in convergence.values())
                 and meta.get("convergence_claim") is False
                 and meta.get("conditional_compatibility_assessed") is False,
                 "Finite discrete FCS cannot claim convergence or conditional compatibility.")
        _require(meta.get("max_observed_count") == _MAX_COUNT
                 and meta.get("max_predicted_poisson_rate") == _MAX_COUNT,
                 "Discrete Poisson bounds are invalid.")
        specs = {}
        work = final_cells = trace_bytes = 0
        max_dimension = max_design = max_categories = 1
        for target in targets:
            j = names.index(target)
            positions = tuple(i for i, row in enumerate(self.original) if row[j] is None)
            observed_positions = tuple(i for i, row in enumerate(self.original) if row[j] is not None)
            observed = tuple(self.original[i][j] for i in observed_positions)
            _require(bool(observed), "Every discrete target needs observed outcomes.")
            method = methods[target]
            k = 1 if method == "poisson" else len(codes[target])
            d = len(predictor_spec[target]) + (method != "ordinal")
            dimension = d if method == "poisson" else d + k - 1 if method == "ordinal" else d * (k - 1)
            _require(1 <= dimension <= _MAX_DIMENSION, "Discrete posterior dimension exceeds its bound.")
            if method == "poisson":
                _require(all(0 <= value <= _MAX_COUNT and value == math.floor(value) for value in observed),
                         "Discrete original Poisson observations are not bounded integer counts.")
            else:
                _require(set(observed) == set(codes[target]), "Every declared category must be observed.")
            for matrix in self.completed_matrices:
                fills = tuple(matrix[i][j] for i in positions)
                _require(all(value >= 0 and value == math.floor(value) for value in fills)
                         if method == "poisson" else all(value in codes[target] for value in fills),
                         "Discrete completed target values violate their declared support.")
            n_missing, n_observed = len(positions), len(observed_positions)
            work += ((mh_burn + mh_steps) * (n_observed * (dimension + 4 * k) + 4 * dimension)
                     + n * (d + 4 * k) + dimension)
            final_cells += 4 * dimension + n * (d + 1) + n_missing * k + 2 * k
            trace_bytes += 4096 + 160 * dimension
            max_dimension, max_design, max_categories = (max(max_dimension, dimension),
                                                        max(max_design, d), max(max_categories, k))
            specs[target] = (j, positions, observed_positions, observed, d, k, dimension)
        projected = m * (cycles * work + n * p)
        _require(type(meta.get("projected_work")) is int and meta["projected_work"] == projected
                 and projected <= meta["max_work"], "Discrete saved work estimate disagrees with its dimensions.")
        expected_buffers = {
            "admitted panels and saved state": meta["resource_plan"]["estimated_workspace_bytes"],
            "conditional design and probability workspace": 8 * n * (4 * max_design + 8 * max_categories),
            "posterior vectors and proposal workspace": 8 * 12 * max_dimension,
            "MH traces and serialized checksum state": m * cycles * trace_bytes,
            "last parameters and prediction state with serialization": m * final_cells * 384,
        }
        plan = meta.get("chained_resource_plan", {})
        _require(plan.get("operation") == "discrete chained multiple imputation"
                 and plan.get("buffers") == expected_buffers,
                 "Discrete state/workspace buffers disagree with the declared model dimensions.")
        plan_workspace("discrete result validation", expected_buffers)
        chains = meta.get("chain_diagnostics")
        _require(isinstance(chains, (tuple, list)) and len(chains) == m,
                 "Discrete diagnostics must contain each imputation.")
        for chain_index, chain in enumerate(chains):
            _require(isinstance(chain, Mapping) and type(chain.get("imputation")) is int and chain["imputation"] == chain_index + 1
                     and type(chain.get("seed")) is int and chain["seed"] == meta["imputation_seeds"][chain_index],
                     "Discrete diagnostics disagree with imputation identity.")
            trace, models = chain.get("trace"), chain.get("last_models")
            _require(isinstance(trace, (tuple, list)) and len(trace) == cycles
                     and isinstance(models, Mapping) and set(models) == set(targets),
                     "Discrete diagnostics do not contain the declared updates/models.")
            for cycle, entry in enumerate(trace, 1):
                _require(isinstance(entry, Mapping) and type(entry.get("cycle")) is int and entry["cycle"] == cycle
                         and entry.get("phase") == ("burn" if cycle <= burn else "sampling")
                         and isinstance(entry.get("variables"), Mapping) and set(entry["variables"]) == set(targets),
                         "Discrete trace visit order/count is invalid.")
                for target, diagnostic in entry["variables"].items():
                    _require(isinstance(diagnostic, Mapping), "Discrete variable diagnostics must be mappings.")
                    counts = (diagnostic.get("accepted_burn"), diagnostic.get("accepted_sampling"),
                              diagnostic.get("rejected_nonfinite"))
                    _require(diagnostic.get("method") == methods[target]
                             and type(diagnostic.get("burn_proposals")) is int and diagnostic["burn_proposals"] == mh_burn
                             and type(diagnostic.get("sampling_proposals")) is int and diagnostic["sampling_proposals"] == mh_steps
                             and all(type(value) is int and value >= 0 for value in counts)
                             and counts[0] <= mh_burn and counts[1] <= mh_steps
                             and sum(counts) <= mh_burn + mh_steps
                             and _close(diagnostic.get("acceptance_rate"), (counts[0] + counts[1]) / (mh_burn + mh_steps))
                             and all(_number(diagnostic.get(key)) for key in (
                                 "final_log_posterior", "imputed_mean", "imputed_variance"))
                             and diagnostic["imputed_variance"] >= 0,
                             "Discrete MH/predictive diagnostics are incoherent.")
            for target, model in models.items():
                _require(isinstance(model, Mapping), "Discrete last models must be mappings.")
                j, positions, observed_positions, observed, d, k, dimension = specs[target]
                method = methods[target]
                _require(model.get("method") == method and model.get("predictors") == predictor_spec[target]
                         and model.get("intercept") is (method != "ordinal")
                         and type(model.get("posterior_dimension")) is int and model["posterior_dimension"] == dimension
                         and model.get("missing_row_positions") == positions
                         and model.get("categories") == (None if method == "poisson" else codes[target])
                         and ((_number(model.get("baseline_category")) and model["baseline_category"] == codes[target][0])
                              if method == "multinomial" else model.get("baseline_category") is None)
                         and model.get("prediction_at") == "target update in final sweep, before subsequent target updates",
                         "Discrete last-model identities are inconsistent.")
                observed_design, missing_design = model.get("observed_design"), model.get("missing_design")
                _checked_rows(observed_design, len(observed_positions), d, "observed design")
                _checked_rows(missing_design, len(positions), d, "missing design")
                for row_positions, design_rows in ((observed_positions, observed_design), (positions, missing_design)):
                    for row_position, design_row in zip(row_positions, design_rows):
                        offset = int(method != "ordinal")
                        _require(not offset or design_row[0] == 1.0, "Discrete design intercept is invalid.")
                        for column, value in zip(predictor_spec[target], design_row[offset:]):
                            original = self.original[row_position][names.index(column)]
                            if original is not None:
                                _require(value == original, "Discrete conditional design changed an observed predictor.")
                            elif targets.index(column) < targets.index(target):
                                _require(value == self.completed_matrices[chain_index][row_position][names.index(column)],
                                         "Discrete conditional design disagrees with earlier final-sweep updates.")
                coordinates = model.get("coefficients") if method == "poisson" else model.get("coordinates")
                _require(isinstance(coordinates, (tuple, list)) and len(coordinates) == dimension
                         and all(_number(value) for value in coordinates), "Discrete posterior coordinates are invalid.")
                theta = torch.tensor(coordinates, dtype=_DTYPE, device="cpu")
                x_observed = torch.tensor(observed_design, dtype=_DTYPE, device="cpu").reshape(len(observed), d)
                x_missing = torch.tensor(missing_design, dtype=_DTYPE, device="cpu").reshape(len(positions), d)
                if method == "poisson":
                    y = torch.tensor(observed, dtype=_DTYPE, device="cpu")
                    log_target = _poisson_log_posterior(x_observed, y, theta, scale)
                    means = model.get("missing_means")
                    expected = _rates(x_missing @ theta, target).tolist()
                    _require(isinstance(means, (tuple, list)) and len(means) == len(positions)
                             and all(_close(value, reference) for value, reference in zip(means, expected)),
                             "Discrete Poisson means disagree with saved posterior coordinates/design.")
                else:
                    y = torch.tensor([codes[target].index(value) for value in observed], dtype=torch.int64, device="cpu")
                    log_target = (_ordinal_log_posterior if method == "ordinal" else _multinomial_log_posterior)(
                        x_observed, y, theta, scale, k
                    )
                    probabilities = model.get("missing_probabilities")
                    _checked_rows(probabilities, len(positions), k, "missing probabilities")
                    expected = torch.exp((_ordinal_log_probabilities if method == "ordinal"
                                          else _multinomial_log_probabilities)(x_missing, theta, k)).tolist()
                    _require(all(all(value >= 0 and _close(value, reference) for value, reference in zip(row, reference_row))
                                 and _close(sum(row), 1.0) for row, reference_row in zip(probabilities, expected)),
                             "Discrete category probabilities disagree with saved coordinates/design.")
                    if method == "ordinal":
                        cuts, _ = _ordinal_cutpoints(theta, d, k)
                        _require(model.get("coefficients") == tuple(coordinates[:d])
                                 and model.get("log_gaps") == tuple(coordinates[d + 1:])
                                 and len(model.get("cutpoints", ())) == k - 1
                                 and all(_close(value, reference) for value, reference in zip(model["cutpoints"], cuts.tolist())),
                                 "Discrete ordinal cutpoints disagree with their coordinates.")
                    else:
                        _require(model.get("coefficients") == tuple(tuple(row) for row in theta.reshape(d, k - 1).tolist()),
                                 "Discrete multinomial coefficients disagree with their fixed-baseline coordinates.")
                final_trace = trace[-1]["variables"][target]
                _require(_close(final_trace["final_log_posterior"], float(log_target)),
                         "Discrete saved final posterior density is inconsistent.")
                fills = [self.completed_matrices[chain_index][i][j] for i in positions]
                mean = sum(fills) / len(fills)
                variance = sum((value - mean) ** 2 for value in fills) / len(fills)
                _require(_close(final_trace["imputed_mean"], mean) and _close(final_trace["imputed_variance"], variance),
                         "Discrete final imputation summaries disagree with completed state.")
        return self


def mi_discrete(data, columns, *, methods, categories=None, m=5, seed=0, burn=10,
                iterations=5, predictors=None, prior_scale=2.5, proposal_scale=0.5,
                mh_burn=100, mh_steps=100, max_work=100_000_000):
    """Impute counts and declared categorical codes with discrete FCS.

    ``methods`` maps exactly the incomplete selected columns to ``poisson``,
    ``ordinal`` or ``multinomial``. ``categories`` maps each categorical target
    to 3..8 numeric codes: ordinal order and multinomial baseline are explicit.
    Each independent chain retains its panel after ``burn + iterations`` sweeps.
    Every update uses the stated proper-prior posterior and finite symmetric MH;
    convergence, stationarity and joint-model compatibility remain unassessed.
    """
    check_seed(seed)
    m = _integer(m, "m", 1, 100)
    burn = _integer(burn, "burn", 0, 1000)
    iterations = _integer(iterations, "iterations", 1, 1000)
    if burn + iterations > 1000:
        raise AnalysisError("resource_limit", "burn + iterations must not exceed 1,000.")
    mh_burn = _integer(mh_burn, "mh_burn", 0, 100_000)
    mh_steps = _integer(mh_steps, "mh_steps", 1, 100_000)
    prior_scale = _positive(prior_scale, "prior_scale")
    proposal_scale = _positive(proposal_scale, "proposal_scale")
    max_work = _integer(max_work, "max_work", 1, 10_000_000_000)
    frame, values, missing, admission = admit(
        data, columns, m=m, iterations=burn + iterations, max_work=max_work
    )
    names = tuple(frame.columns)
    targets = tuple(name for j, name in enumerate(names) if bool(missing[:, j].any()))
    if not targets:
        raise AnalysisError("no_missing_values", "mi_discrete needs an incomplete selected column.")
    if not isinstance(methods, Mapping) or set(methods) != set(targets):
        raise AnalysisError("invalid_spec", "methods must specify exactly the incomplete selected columns.")
    if any(not isinstance(methods[target], str) or methods[target] not in (
            "poisson", "ordinal", "multinomial") for target in targets):
        raise AnalysisError("invalid_spec", "Discrete methods are 'poisson', 'ordinal' and 'multinomial'.")
    codes = _category_spec(targets, methods, categories)
    predictor_spec = _predictors(names, targets, predictors)
    n, p = values.shape
    cycles = burn + iterations
    if m * cycles * len(targets) > _MAX_UPDATES:
        raise AnalysisError("resource_limit", "Discrete FCS traces exceed 100,000 variable updates.")
    specifications = {}
    work_per_cycle = final_cells = trace_per_cycle = 0
    max_dimension = max_design = max_categories = 1
    for target in targets:
        j = names.index(target)
        mask = missing[:, j]
        n_missing = int(mask.sum())
        n_observed = n - n_missing
        if n_observed == 0:
            raise AnalysisError("all_missing", f"{target!r} has no observed outcomes.")
        observed = values[~mask, j]
        method = methods[target]
        k = len(codes[target]) if method != "poisson" else 1
        d = len(predictor_spec[target]) + (method != "ordinal")
        dimension = d if method == "poisson" else d + k - 1 if method == "ordinal" else d * (k - 1)
        _integer(dimension, "posterior dimension", 1, _MAX_DIMENSION)
        if method == "poisson":
            if not bool(((observed >= 0) & (observed <= _MAX_COUNT) & (observed == observed.floor())).all()):
                raise AnalysisError("invalid_count_outcome", f"Observed {target!r} must be integer counts in [0, 1,000,000].")
        elif set(observed.tolist()) != set(codes[target]):
            raise AnalysisError("invalid_categories", f"All and only declared levels of {target!r} must be observed.")
        work_per_cycle += ((mh_burn + mh_steps) * (n_observed * (dimension + 4 * k) + 4 * dimension)
                           + n * (d + 4 * k) + dimension)
        final_cells += 4 * dimension + n * (d + 1) + n_missing * k + 2 * k
        trace_per_cycle += 4096 + 160 * dimension
        max_dimension, max_design, max_categories = (max(max_dimension, dimension),
                                                    max(max_design, d), max(max_categories, k))
        specifications[target] = (j, mask, tuple(names.index(name) for name in predictor_spec[target]), d, k, dimension)
    projected_work = m * (cycles * work_per_cycle + n * p)
    if projected_work > max_work:
        raise AnalysisError("resource_limit", f"Discrete FCS work {projected_work:,} exceeds max_work={max_work:,}.")
    # Before designs, posterior vectors, prediction matrices, traces, and their
    # immutable JSON/checksum copies. Shared admission already bounded its panel.
    plan = plan_workspace("discrete chained multiple imputation", {
        "admitted panels and saved state": admission["resource_plan"]["estimated_workspace_bytes"],
        "conditional design and probability workspace": 8 * n * (4 * max_design + 8 * max_categories),
        "posterior vectors and proposal workspace": 8 * 12 * max_dimension,
        "MH traces and serialized checksum state": m * cycles * trace_per_cycle,
        "last parameters and prediction state with serialization": m * final_cells * 384,
    })
    completed = []
    chains = []
    seeds = [(int(seed) + chain) % (2 ** 63) for chain in range(m)]
    for imputation, chain_seed in enumerate(seeds, 1):
        generator = torch.Generator(device="cpu").manual_seed(chain_seed)
        filled = values.clone()
        for target in targets:
            j, mask, *_ = specifications[target]
            observed = values[~mask, j]
            positions = torch.randint(observed.numel(), (int(mask.sum()),), device="cpu", generator=generator)
            filled[mask, j] = observed[positions]
        states, trace, final_models = {}, [], {}
        for cycle in range(1, cycles + 1):
            updates = {}
            for target in targets:
                j, mask, indices, d, k, dimension = specifications[target]
                method = methods[target]
                selected = filled[:, indices]
                design = (selected if method == "ordinal" else torch.cat((
                    torch.ones((n, 1), dtype=_DTYPE, device="cpu"), selected), dim=1))
                _finite(design, f"Predictor design for {target!r}")
                x_observed, x_missing = design[~mask], design[mask]
                observed = values[~mask, j]
                if method == "poisson":
                    theta, diagnostic = _poisson_draw(
                        x_observed, observed, generator, prior_scale=prior_scale,
                        proposal_scale=proposal_scale, burn=mh_burn, steps=mh_steps,
                        initial=states.get(target)
                    )
                    means = _rates(design @ theta, target)[mask]
                    draw = torch.poisson(means, generator=generator)
                    model = {"coefficients": theta.tolist(), "missing_means": means.tolist()}
                else:
                    category_tensor = torch.tensor(codes[target], dtype=_DTYPE, device="cpu")
                    y = (observed[:, None] == category_tensor[None, :]).to(torch.int64).argmax(1)
                    log_target = (_ordinal_log_posterior if method == "ordinal" else _multinomial_log_posterior)
                    theta, diagnostic = _mh_draw(
                        lambda state: log_target(x_observed, y, state, prior_scale, k),
                        dimension, generator, prior_scale=prior_scale, proposal_scale=proposal_scale,
                        burn=mh_burn, steps=mh_steps, initial=states.get(target)
                    )
                    log_probabilities = (_ordinal_log_probabilities if method == "ordinal"
                                         else _multinomial_log_probabilities)(x_missing, theta, k)
                    probabilities = torch.exp(log_probabilities)
                    _finite(probabilities, f"Category probabilities for {target!r}")
                    if bool((probabilities.sum(1) <= 0).any()):
                        raise AnalysisError("numerical_failure", "Categorical prediction has no positive mass.")
                    choice = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
                    draw = category_tensor[choice]
                    model = {"coordinates": theta.tolist(), "missing_probabilities": probabilities.tolist()}
                    if method == "ordinal":
                        cutpoints, _ = _ordinal_cutpoints(theta, d, k)
                        model.update({"coefficients": theta[:d].tolist(), "cutpoints": cutpoints.tolist(),
                                      "log_gaps": theta[d + 1:].tolist()})
                    else:
                        model["coefficients"] = theta.reshape(d, k - 1).tolist()
                _finite(draw, f"Imputed {target!r}")
                filled[mask, j] = draw
                states[target] = theta
                updates[target] = {"method": method, **diagnostic,
                                   "imputed_mean": float(draw.mean()),
                                   "imputed_variance": float(draw.var(unbiased=False))}
                if cycle == cycles:
                    final_models[target] = {
                        "method": method, "predictors": list(predictor_spec[target]),
                        "intercept": method != "ordinal", "posterior_dimension": dimension,
                        "missing_row_positions": torch.nonzero(mask, as_tuple=False)[:, 0].tolist(),
                        "categories": list(codes[target]) if method != "poisson" else None,
                        "baseline_category": codes[target][0] if method == "multinomial" else None,
                        "prediction_at": "target update in final sweep, before subsequent target updates",
                        "observed_design": x_observed.tolist(), "missing_design": x_missing.tolist(),
                        **model,
                    }
            trace.append({"cycle": cycle, "phase": "burn" if cycle <= burn else "sampling", "variables": updates})
        completed.append(filled)
        chains.append({"imputation": imputation, "seed": chain_seed, "trace": trace, "last_models": final_models})
    metadata = {
        **admission, "api": "mi_discrete", "fcs": True,
        "methods": {target: methods[target] for target in targets},
        "categories": {target: list(value) for target, value in codes.items()},
        "predictors": {target: list(predictor_spec[target]) for target in targets},
        "visit_order": list(targets), "burn": burn, "iterations": iterations,
        "cycles_per_chain": cycles, "initialization": "independent random observed donor per missing cell",
        "prior": {"family": "independent Gaussian", "mean": 0.0, "scale": prior_scale,
                  "poisson": "raw coefficients including intercept",
                  "multinomial": "raw nonbaseline coefficients including intercept; baseline logit fixed at zero",
                  "ordinal": "no-intercept raw coefficients, first cutpoint, and log positive gaps; prior in these coordinates; no Jacobian"},
        "sampler": {"kernel": "symmetric isotropic Gaussian random-walk Metropolis",
                    "exact_target": True, "stationarity_claim": False,
                    "proposal_scale": proposal_scale, "burn_per_update": mh_burn,
                    "steps_per_update": mh_steps, "nonfinite_proposals": "rejected without clipping"},
        "convergence": {"assessed": False, "converged": False}, "convergence_claim": False,
        "conditional_compatibility_assessed": False, "max_observed_count": _MAX_COUNT,
        "max_predicted_poisson_rate": _MAX_COUNT, "projected_work": projected_work,
        "chained_resource_plan": plan.record(), "chain_diagnostics": chains,
    }
    result = make_result("mi_chained", frame, values, missing, completed, seed, metadata,
                         imputation_seeds=seeds)
    return MIDiscreteResult.model_validate(result.model_dump(mode="python"))
