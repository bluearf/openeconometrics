"""Conditional recursive NARDL multiplier bootstrap, using float64 Torch.

The iid residual scheme follows the single-equation algorithm in Appendix A of
Brun-Aguerre, Fuertes and Greenwood-Nimmo (2017), doi:10.1111/rssa.12213.
Rademacher wild innovations are an explicitly conditional extension; neither
scheme is a cointegration/bounds calibration or a simultaneous confidence band.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import _frame_hasher, _position_bytes
from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.tsmodels.ardl import _case, _constraint_basis, _lag_name
from openecon.econometrics.tsmodels.common import lag, ordered_frame
from openecon.models import ResultBundle

_NATIVE_RECURSION = None
_RECURSION_SOURCE = '''
def recursive_ar(initial: Tensor, forcing: Tensor, phi: Tensor, hold: int) -> Tensor:
    values = torch.empty((forcing.size(0), initial.size(0)), dtype=torch.float64, device="cpu")
    values[:, :hold] = initial[:hold]
    for t in range(hold, initial.size(0)):
        value = forcing[:, t - hold]
        for i in range(phi.numel()):
            value = value + phi[i] * values[:, t - i - 1]
        values[:, t] = value
    return values
'''


def _native_values(y: Tensor, forcing: Tensor, phi: Tensor, holdback: int) -> Tensor:
    """Compile a fixed trusted filter into the bundled TorchScript interpreter.

    No inspection of Python source, external compiler, SciPy or model-specific
    generated code is required. The time loop runs in native TorchScript, with
    every replicate processed by one vector operation at each lag/time step.
    """
    global _NATIVE_RECURSION
    if _NATIVE_RECURSION is None:
        _NATIVE_RECURSION = torch.jit.CompilationUnit(_RECURSION_SOURCE)
    return _NATIVE_RECURSION.recursive_ar(y, forcing, phi, holdback)


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or maximum is not None and value > maximum):
        bound = f"from {minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise AnalysisError("invalid_option", f"{name} must be an integer {bound}.")
    return value


def _validate_reporting(result: ResultBundle, beta: Tensor, basis: Tensor, terms: list[str],
                        orders: dict[str, int], names: list[str]) -> None:
    """A saved public coefficient table must agree with its saved levels state."""
    covariance = torch.tensor(result.extra["levels_covariance"], dtype=torch.float64)
    outcome, case = result.spec.outcome, result.extra["case"]
    k = len(terms)
    unit = torch.eye(k, dtype=torch.float64)
    ec = bool(result.spec.options["ec"])
    if result.extra["ec"] != ec:
        raise ValueError("Saved error-correction reporting differs from the specification")
    if not ec:
        labels, estimates, jacobian = terms, beta, unit
    else:
        y_index = [terms.index(_lag_name(outcome, i)) for i in range(1, orders[outcome] + 1)]
        denominator = 1 - beta[y_index].sum()
        labels, values, jac = [], [], []

        def add(label, value, gradient):
            labels.append(label)
            values.append(value)
            jac.append(gradient)

        gradient = unit[y_index].sum(0)
        add(f"ADJ:{_lag_name(outcome, 1)}", -denominator, gradient)
        lr = list(names)
        if case in (2, 4):
            lr.insert(0, "Intercept" if case == 2 else "trend")
        for name in lr:
            indices = ([terms.index(name)] if name in {"Intercept", "trend"} else
                       [terms.index(_lag_name(name, i)) for i in range(orders[name] + 1)])
            total = beta[indices].sum()
            gradient = unit[indices].sum(0) / denominator
            gradient[y_index] = total / denominator.square()
            add(f"LR:{name}", total / denominator, gradient)
        for i in range(1, orders[outcome]):
            indices = y_index[i:]
            add(f"SR:{_lag_name(outcome, i, difference=True)}", -beta[indices].sum(),
                -unit[indices].sum(0))
        for name in names:
            for i in range(orders[name]):
                indices = [terms.index(_lag_name(name, j)) for j in range(i + 1, orders[name] + 1)]
                add(f"SR:{_lag_name(name, i, difference=True)}", -beta[indices].sum(),
                    -unit[indices].sum(0))
        static = {"Intercept", "trend", *result.spec.columns.get("exog", [])}
        restricted = "Intercept" if case == 2 else "trend" if case == 4 else None
        for index, name in enumerate(terms):
            if name in static and name != restricted:
                add(f"SR:{name}", beta[index], unit[index])
        estimates, jacobian = torch.stack(values), torch.stack(jac)
    if result.extra.get("constraints") is not None:
        fixed = torch.linalg.vector_norm(jacobian @ basis, dim=1) <= (
            1e-12 * torch.linalg.vector_norm(jacobian, dim=1).clamp_min(1))
        keep = ~fixed
        fixed_values = {label: float(estimate) for label, estimate, is_fixed
                        in zip(labels, estimates, fixed, strict=True) if bool(is_fixed)}
        recorded = result.extra.get("constrained_terms", {})
        if set(recorded) != set(fixed_values) or any(not math.isclose(recorded[label], value,
                rel_tol=1e-9, abs_tol=1e-11) for label, value in fixed_values.items()):
            raise ValueError("Saved fixed coefficient terms differ")
        labels = [label for label, kept in zip(labels, keep, strict=True) if bool(kept)]
        estimates, jacobian = estimates[keep], jacobian[keep]
    reported = torch.tensor([coefficient.estimate for coefficient in result.coefficients],
                            dtype=torch.float64)
    public_cov = torch.tensor(result.covariance_matrix, dtype=torch.float64)
    expected_cov = jacobian @ covariance @ jacobian.T
    if ([coefficient.term for coefficient in result.coefficients] != labels
            or reported.shape != estimates.shape or public_cov.shape != expected_cov.shape
            or not bool(torch.allclose(reported, estimates, rtol=1e-9, atol=1e-11))
            or not bool(torch.allclose(public_cov, expected_cov, rtol=1e-8, atol=1e-11))):
        raise ValueError("Saved public coefficient/covariance table differs from levels state")


def _source(result: ResultBundle, data: Any):
    """Reconstruct the exact levels design, sample and free parameter space.

    Results retain bounded predictions, not full residuals. Original data are
    therefore mandatory, with hashes checked before any bootstrap generation.
    """
    from openecon.econometrics.tsmodels.nardl import _expand_orders, _symmetry_rows

    if data is None:
        raise AnalysisError("missing_data", "Bootstrap multipliers require data= with the original "
                            "model-input rows; a saved result does not retain all residuals.")
    frame, _ = ordered_frame(result.spec, data, what="NARDL recursive bootstrap")
    if (frame.n < 2 or len(frame.original) != result.nobs_original
            or list(frame.original.columns) != result.provenance.get("input_columns")
            or _frame_hasher(frame.original).hexdigest() != result.provenance.get("data_hash")):
        raise AnalysisError("data_mismatch", "Bootstrap data must match the fitted model-input "
                            "columns, dtypes and original row order exactly.")
    try:
        spec, extra = result.spec, result.extra
        original = list(spec.predictors)
        asymmetric = spec.options.get("asymmetric")
        asymmetric = original if asymmetric is None else list(asymmetric)
        pairs = {name: [f"{name}_positive", f"{name}_negative"]
                 for name in original if name in asymmetric}
        if extra["asymmetric_predictors"] != pairs:
            raise ValueError("Partial-sum names differ from the original specification")
        names, values = [], {}
        for name in original:
            numeric = frame.numeric(name)
            if name in pairs:
                difference = torch.diff(numeric, prepend=numeric[:1])
                positive, negative = pairs[name]
                values[positive] = difference.clamp_min(0).cumsum(0)
                values[negative] = difference.clamp_max(0).cumsum(0)
                names.extend(pairs[name])
            else:
                values[name] = numeric
                names.append(name)
        outcome, orders = spec.outcome, extra["lags"]
        if set(orders) != {outcome, *names} or extra["expanded_predictors"] != names:
            raise ValueError("Saved lag order names differ from the original specification")
        selected = [orders[outcome], *[orders[name] for name in names]]
        given = _expand_orders(spec.options.get("lags"), original, set(asymmetric), name="lags")
        maxima = _expand_orders(spec.options.get("maxlags"), original, set(asymmetric), name="maxlags")
        if given is not None:
            if selected != given or extra.get("lag_selection") is not None:
                raise ValueError("Saved explicit lag orders differ from the original specification")
            holdback = max(given)
        else:
            if not maxima:
                raise ValueError("Original maximum lag orders are absent")
            maxima = maxima * len(selected) if len(maxima) == 1 else maxima
            if (len(maxima) != len(selected) or any(order > maximum for order, maximum
                    in zip(selected, maxima, strict=True))
                    or extra["lag_selection"]["selected"] != selected):
                raise ValueError("Saved selected lag orders differ from the lag search")
            holdback = max(maxima)
        if (frame.positions[holdback:] != result.sample_positions
                or frame.n - holdback != result.nobs
                or extra["partial_sum_origin"]["row"] != frame.positions[0]):
            raise ValueError("Saved estimation sample or partial-sum origin differs")
        y = frame.numeric(outcome)
        trend, restricted = spec.options["trend"], bool(spec.options["restricted"])
        if (extra["trend"] != trend or extra["restricted"] != restricted
                or extra["case"] != _case(trend, restricted)):
            raise ValueError("Saved deterministic case differs")
        columns = {"Intercept": torch.ones(frame.n, dtype=torch.float64)}
        if trend == "trend":
            columns["trend"] = torch.arange(1, frame.n + 1, dtype=torch.float64)
        y_terms = [_lag_name(outcome, i) for i in range(1, orders[outcome] + 1)]
        columns.update({name: lag(y, i) for i, name in enumerate(y_terms, start=1)})
        for name in names:
            columns.update({_lag_name(name, i): lag(values[name], i)
                            for i in range(orders[name] + 1)})
        columns.update({name: frame.numeric(name) for name in frame.role("exog")})
        terms = list(extra["levels_coefficients"])
        omitted = set(result.provenance.get("omitted_terms", []))
        if terms != [name for name in columns if name not in omitted]:
            raise ValueError("Saved levels terms differ from the fitted design")
        x = torch.stack([columns[term] for term in terms], dim=1)[holdback:]
        beta = torch.tensor(list(extra["levels_coefficients"].values()), dtype=torch.float64)
        imposed = {kind: list(spec.options.get(f"{kind}_symmetric") or [])
                   for kind in ("long_run", "short_run")}
        if any(extra["imposed_symmetry"][kind] != subset for kind, subset in imposed.items()):
            raise ValueError("Saved imposed symmetry differs")
        rows = _symmetry_rows(terms, pairs, orders)
        restrictions = [rows[name][0] for name in imposed["long_run"]]
        restrictions += [row for name in imposed["short_run"] for row in rows[name][1]]
        matrix = torch.stack(restrictions) if restrictions else torch.empty(
            (0, len(terms)), dtype=torch.float64)
        basis, independent = _constraint_basis(matrix, len(terms))
        saved = extra.get("constraints")
        if saved is not None:
            retained = torch.tensor(saved["parameter_basis"], dtype=torch.float64)
            recorded_matrix = torch.tensor(saved["matrix"], dtype=torch.float64).reshape(
                -1, len(terms))
            free_coefficients = torch.tensor(saved["free_coefficients"], dtype=torch.float64)
            if (not bool(torch.allclose(basis @ basis.T, retained @ retained.T,
                                       rtol=1e-9, atol=1e-11))
                    or saved["terms"] != terms or saved["rank"] != len(independent)
                    or saved["free_parameters"] != basis.shape[1]
                    or recorded_matrix.shape != independent.shape
                    or not bool(torch.allclose(recorded_matrix, independent, rtol=1e-9, atol=1e-11))
                    or free_coefficients.shape != (basis.shape[1],)
                    or not bool(torch.allclose(retained @ free_coefficients, beta,
                                               rtol=1e-9, atol=1e-11))
                    or saved["values"] != [0.0] * len(independent)):
                raise ValueError("Saved constraint subspace differs")
            basis = retained
        elif restrictions:
            raise ValueError("Saved constraints are absent")
        if not bool(torch.allclose(basis @ (basis.T @ beta), beta, rtol=1e-9, atol=1e-11)):
            raise ValueError("Saved parameters violate imposed restrictions")
        if result.metrics["df_resid"] != result.nobs - basis.shape[1]:
            raise ValueError("Saved residual degrees of freedom differ")
        observed = y[holdback:]
        optimal_beta = _refit(x[None, :, :], observed[None, :], basis)[0]
        if not bool(torch.allclose(optimal_beta, beta, rtol=1e-8, atol=1e-10)):
            raise ValueError("Saved levels coefficients do not refit the original estimation sample")
        _validate_reporting(result, beta, basis, terms, orders, names)
        fitted = x @ beta
        residuals = observed - fitted
        ssr = float(residuals.square().sum())
        if (not math.isfinite(ssr) or not math.isclose(ssr, result.metrics["ssr"],
                                                     rel_tol=1e-8, abs_tol=1e-12)):
            raise ValueError("Saved residual fit differs from the original data")
        # This includes the original hold-back (which can exceed selected lags).
        frame.restrict(torch.arange(frame.n) >= holdback)
        sample_hasher = _frame_hasher(frame.sample)
        sample_hasher.update(_position_bytes(frame.positions))
        if sample_hasher.hexdigest() != result.provenance.get("sample_hash"):
            raise ValueError("Saved estimation sample hash differs")
        if not all(bool(torch.isfinite(item).all()) for item in (x, y, residuals)):
            raise ValueError("Nonfinite reconstructed design")
        return y, x, beta, basis, terms, orders, pairs, holdback, residuals
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError("invalid_result", "Saved NARDL design, sample or restrictions are "
                            "incomplete or inconsistent with the original data.") from exc


def _refit(x: Tensor, y: Tensor, basis: Tensor) -> Tensor:
    """Batched, column-scaled Householder QR in the constraint null space."""
    free = x @ basis
    scale = free.abs().amax(dim=1)
    if not bool(torch.isfinite(free).all()) or bool((scale == 0).any()):
        raise AnalysisError("bootstrap_failure", "A bootstrap refit has a nonfinite or zero design column.")
    normalized = free / scale[:, None, :]
    q, r = torch.linalg.qr(normalized, mode="reduced")
    singular = torch.linalg.svdvals(r)
    threshold = torch.finfo(torch.float64).eps * max(free.shape[1:]) * singular[:, :1]
    if not bool(torch.isfinite(singular).all()) or bool((singular[:, -1:] <= threshold).any()):
        raise AnalysisError("bootstrap_failure", "A bootstrap free-parameter design is rank deficient; "
                            "no replicates are discarded or replaced.")
    yscale = y.abs().amax(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    target = torch.bmm(q.transpose(1, 2), (y / yscale[:, None])[:, :, None])
    coefficients = torch.linalg.solve_triangular(r, target, upper=True).squeeze(-1)
    coefficients = (coefficients / scale) * yscale[:, None]
    beta = coefficients @ basis.T
    if not bool(torch.isfinite(beta).all()):
        raise AnalysisError("bootstrap_failure", "A bootstrap coefficient is nonfinite.")
    return beta


def _paths(beta: Tensor, terms: list[str], outcome: str, orders: dict[str, int],
           pairs: dict[str, list[str]], steps: int) -> Tensor:
    names = [name for pair in pairs.values() for name in pair]
    phi = [terms.index(_lag_name(outcome, i)) for i in range(1, orders[outcome] + 1)]
    response = torch.zeros((len(beta), steps + 1, len(names)), dtype=torch.float64)
    for j, name in enumerate(names):
        indices = [terms.index(_lag_name(name, i)) for i in range(orders[name] + 1)]
        sums = beta[:, indices].cumsum(1)
        response[:, :, j] = sums[:, torch.arange(steps + 1).clamp_max(orders[name])]
    for horizon in range(steps + 1):
        for i, index in enumerate(phi, start=1):
            if horizon >= i:
                response[:, horizon] += beta[:, index, None] * response[:, horizon - i]
    positive, negative = response[:, :, 0::2], response[:, :, 1::2]
    return torch.stack((positive, negative, positive - negative), dim=-1)


def _generate(y: Tensor, x: Tensor, beta: Tensor, terms: list[str], outcome: str,
              p: int, holdback: int, errors: Tensor,
              diagnostics: dict[str, Any] | None = None) -> tuple[Tensor, Tensor]:
    """Generate y* with a certified scalar prefix or a native direct AR filter.

    Squaring a higher-order companion matrix can be unstable even with roots
    strictly inside the unit circle (nonnormal/repeated-root dynamics). AR(p>1)
    therefore uses the native direct recurrence. Scalar AR(1) uses prefix
    doubling with an equation-residual check and a direct native fallback.
    """
    indices = [terms.index(_lag_name(outcome, i)) for i in range(1, p + 1)]
    static = x.clone()
    static[:, indices] = 0
    mean = static @ beta
    forcing, phi = mean[None, :] + errors, beta[indices]
    if p > 1:
        values = _native_values(y, forcing, phi, holdback)
    else:
        states = forcing.clone()
        states[:, 0] += phi[0] * y[holdback - 1]
        stride, power = 1, phi[0]
        while stride < len(x):
            increment = states[:, :-stride] * power
            states[:, stride:] += increment
            stride *= 2
            if stride < len(x):
                power = power * power
        values = torch.empty((len(errors), len(y)), dtype=torch.float64)
        values[:, :holdback] = y[:holdback]
        values[:, holdback:] = states
        previous = values[:, holdback - 1:-1]
        discrepancy = values[:, holdback:] - (forcing + phi[0] * previous)
        scale = values[:, holdback:].abs() + forcing.abs() + phi[0].abs() * previous.abs()
        tolerance = 128 * torch.finfo(torch.float64).eps * scale.clamp_min(
            torch.finfo(torch.float64).tiny)
        if (not bool(torch.isfinite(values).all())
                or bool((discrepancy.abs() > tolerance).any())):
            values = _native_values(y, forcing, phi, holdback)
            if diagnostics is not None:
                diagnostics["scalar_prefix_fallback_batches"] += 1
    design = x.expand(len(errors), -1, -1).clone()
    for i, index in enumerate(indices, start=1):
        design[:, :, index] = values[:, holdback - i:len(y) - i]
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("bootstrap_failure", "A recursive bootstrap outcome overflowed; "
                            "no replicates are discarded or replaced.")
    return design, values[:, holdback:]


def bootstrap_multipliers(result: ResultBundle, *, data: Any, steps: int, alpha: float | None,
                          method: str, replications: int, seed: int, batch_size: int):
    from openecon.econometrics.tsmodels.nardl import (
        _structural_difference_zeros, nardl_multipliers,
    )

    replications = _integer(replications, "replications", 20)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    batch_size = _integer(batch_size, "batch_size", 1)
    # Reuse saved-parameter, covariance, alpha and horizon validation. The delta
    # point estimates are identical; only uncertainty is replaced below.
    answer = nardl_multipliers(result, steps=steps, alpha=alpha)
    if not answer.attrs["autoregressive_stable"]:
        raise AnalysisError("unstable_bootstrap", "Recursive multiplier bootstrap requires a "
                            "stable fitted conditional autoregression; no unit-root calibration "
                            "is implemented.")
    alpha = result.spec.alpha if alpha is None else float(alpha)
    y, x, beta, basis, terms, orders, pairs, holdback, residuals = _source(result, data)
    n, k = x.shape
    path_bytes = replications * (steps + 1) * len(pairs) * 3 * 8
    # Quantiles may sort/copy paths and maintain integer indices. Reserve three
    # path-sized blocks, rather than budgeting only the retained draws.
    path_workspace = 3 * path_bytes
    if path_workspace >= _MAX_DESIGN_BYTES:
        raise AnalysisError("bootstrap_capacity", "Requested bootstrap multiplier paths exceed "
                            "the 256 MiB inference workspace; reduce replications or horizons.")
    # Bound working batches independently of total replications. A generated
    # design, free design, QR factors and outcome paths dominate live storage.
    per_replication = 8 * (4 * n * k + 3 * n * orders[result.spec.outcome]
                           + 2 * len(y) + (steps + 1) * 3 * len(pairs))
    baseline = 8 * (x.numel() + y.numel() + beta.numel() + basis.numel())
    available = _MAX_DESIGN_BYTES - path_workspace - baseline
    if per_replication > available:
        raise AnalysisError("bootstrap_capacity", "One NARDL bootstrap design exceeds the 256 "
                            "MiB inference workspace after reserving multiplier paths; reduce "
                            "replications/horizons. A streaming NARDL refit is not implemented.")
    effective_batch = min(batch_size, replications, available // per_replication)
    innovations = (residuals - residuals.mean()) * math.sqrt(n / (n - basis.shape[1]))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, steps + 1, len(pairs), 3), dtype=torch.float64)
    fixed_contrasts = [_structural_difference_zeros(basis, terms, pair, orders, steps)
                       for pair in pairs.values()]
    unstable = 0
    recursion = {"algorithm": "native_direct_ar" if orders[result.spec.outcome] > 1
                 else "certified_scalar_affine_prefix_with_native_fallback",
                 "time_complexity": "O(N p), batched replicates" if orders[result.spec.outcome] > 1
                 else "O(N log N) scalar work; O(N) native fallback",
                 "python_time_row_loop": False,
                 "scalar_prefix_fallback_batches": 0,
                 "scalar_prefix_certification": "128 eps times local absolute equation scale",
                 "native_implementation": "fixed trusted torch.jit.CompilationUnit source; no external compiler"}
    try:
        for start in range(0, replications, effective_batch):
            count = min(effective_batch, replications - start)
            if method == "residual":
                indices = torch.randint(n, (count, n), generator=generator)
                errors = innovations[indices]
            else:
                signs = torch.randint(2, (count, n), generator=generator) * 2 - 1
                errors = innovations[None, :] * signs
            try:
                design, observed = _generate(y, x, beta, terms, result.spec.outcome,
                                             orders[result.spec.outcome], holdback, errors, recursion)
                fitted_beta = _refit(design, observed, basis)
            except AnalysisError as exc:
                raise AnalysisError(exc.code, f"Replicates {start + 1}..{start + count}: {exc}") from exc
            paths = _paths(fitted_beta, terms, result.spec.outcome, orders, pairs, steps)
            for j, fixed in enumerate(fixed_contrasts):
                paths[:, fixed, j, 2] = 0.0
            if not bool(torch.isfinite(paths).all()):
                raise AnalysisError("bootstrap_failure", "A bootstrap multiplier path overflowed.")
            draws[start:start + count] = paths
            p = orders[result.spec.outcome]
            companion = torch.zeros((count, p, p), dtype=torch.float64)
            companion[:, 0] = fitted_beta[:, [terms.index(_lag_name(result.spec.outcome, i))
                                             for i in range(1, p + 1)]]
            if p > 1:
                companion[:, 1:, :-1] = torch.eye(p - 1, dtype=torch.float64)
            unstable += int((torch.linalg.eigvals(companion).abs().amax(dim=1) >= 1).sum())
        percentiles = torch.quantile(draws, torch.tensor([alpha / 2, 1 - alpha / 2],
                                                       dtype=torch.float64), dim=0)
        errors = draws.std(dim=0, correction=1)
    except RuntimeError as exc:
        raise AnalysisError("bootstrap_failure", "Torch could not complete the recursive "
                            "bootstrap; no partial confidence intervals are returned.") from exc
    if not bool(torch.isfinite(percentiles).all()) or not bool(torch.isfinite(errors).all()):
        raise AnalysisError("bootstrap_failure", "Bootstrap uncertainty overflowed; no partial "
                            "confidence intervals are returned.")
    for j, variable in enumerate(pairs):
        rows = answer.variable == variable
        for i, name in enumerate(("positive", "negative", "difference")):
            low, high, se = percentiles[0, :, j, i].clone(), percentiles[1, :, j, i].clone(), errors[:, j, i].clone()
            if name == "difference":
                fixed = fixed_contrasts[j]
                low[fixed], high[fixed], se[fixed] = 0.0, 0.0, 0.0
            answer.loc[rows, f"{name}_ci_low"] = low.tolist()
            answer.loc[rows, f"{name}_ci_high"] = high.tolist()
            answer.loc[rows, f"{name}_std_error"] = se.tolist()
    warnings = ["Intervals condition on observed predictors and initial outcome values, selected "
                "lag orders and imposed symmetry; no selection or regressor-feedback uncertainty "
                "is included.", "Pointwise percentile intervals are not simultaneous bands or "
                "NARDL bounds/cointegration critical values.",
                "Predictor exogeneity and innovation assumptions require separate assessment; "
                "this bootstrap has no general coverage validation for endogenous regressors "
                "or unit-root/local-to-unity asymptotics."]
    assumption = ("iid homoskedastic innovations independent of the conditioned predictor path"
                  if method == "residual" else "serially uncorrelated martingale-difference "
                  "innovations with conditional heteroskedasticity, exogenous conditioned predictors")
    if replications < 999:
        warnings.append("Fewer than 999 replicates give coarse Monte Carlo tail quantiles; "
                        "increase replications for publication inference.")
    if unstable:
        warnings.append(f"{unstable} refitted autoregressions are unstable; all finite-horizon "
                        "draws are retained, without a stable-root filter or long-run interpretation.")
    answer.attrs.update(
        inference=f"pointwise percentile recursive {method} bootstrap, conditional on selected lag orders",
        bootstrap={"method": method, "replications": replications, "completed": replications,
                   "seed": seed, "rng": "local CPU torch.Generator; integer draws in replicate order",
                   "batch_size_requested": batch_size, "batch_size_used": effective_batch,
                   "solver": "batched scaled Householder QR in the original constraint null space",
                   "outcome_generation": recursion,
                   "hold_back": holdback, "lags": orders, "case": result.extra["case"],
                   "constraints_imposed": result.extra["imposed_symmetry"],
                   "residual_centering": "subtract original estimation residual mean",
                   "residual_scale": "sqrt(n / (n - free_parameters))",
                   "innovation_weights": "Rademacher {-1,+1}" if method == "wild" else None,
                   "interval": "pointwise equal-tailed percentile, linear quantile interpolation",
                   "standard_error": "sample standard deviation of bootstrap multipliers (B-1 denominator)",
                   "invalid_draw_policy": "fail the entire call; never discard or replace replicates",
                   "structural_zero_policy": "only proved homogeneous distributed-coefficient "
                   "identities; zero delta variance never removes empirical uncertainty",
                   "unstable_refits_retained": unstable, "precision": "float64", "device": "cpu",
                   "data_hash": result.provenance["data_hash"], "sample_hash": result.provenance["sample_hash"],
                   "assumptions": assumption, "joint_regressor_resampling": False,
                   "selection_repeated": False, "simultaneous": False,
                   "unit_root_or_cointegration_calibration": False,
                   "general_nardl_coverage_validated": False},
        warnings=warnings, notes=[f"{100 * (1 - alpha):g}% pointwise percentile {method} bootstrap; "
                                f"{replications} replications, seed {seed}.", *warnings],
    )
    return answer
