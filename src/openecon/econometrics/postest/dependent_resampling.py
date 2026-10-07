"""Bounded resampling contracts: pairs blocks, fixed-design residuals and cluster wild.

Dependent schemes currently validate unweighted OLS on explicit columns. They
do not authorize iid resampling of ARIMA, state-space or lag-generated models.
BCa is a delete-unit acceleration for iid pairs (rows or whole clusters).
Studentization uses each refit's recorded covariance, never a common bootstrap SE.
"""

from collections import Counter
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.econometrics.postest.common import (
    coefficient_vector,
    matched_frame,
    rebuild,
    refit_spec,
    refitter,
    result_covariance,
)
from openecon.econometrics.postest.draws import (
    Clusters,
    Strata,
    bootstrap_covariance,
    bootstrap_intervals,
    draw_units,
    stata_percentile,
)
from openecon.engines.distributions import normal_cdf, normal_ppf

SCHEMES = ("iid", "moving_block", "residual", "residual_block", "wild_cluster")
MAX_ENTRIES = 2_000_000


def block_indices(n, length, generator):
    """Non-circular, overlapping moving blocks, concatenated to n observations."""
    starts = torch.randint(n - length + 1, (math.ceil(n / length),), generator=generator)
    return (starts[:, None] + torch.arange(length)[None, :]).flatten()[:n]


def multipliers(units, reps, seed, kind="rademacher"):
    if (
        isinstance(units, bool)
        or not isinstance(units, int)
        or units < 2
        or isinstance(reps, bool)
        or not isinstance(reps, int)
        or reps < 2
    ):
        raise AnalysisError(
            "invalid_spec", "Multiplier inference needs at least two units and draws."
        )
    if units * reps > MAX_ENTRIES:
        raise AnalysisError(
            "bootstrap_too_large", "Multiplier matrix exceeds the explicit 2,000,000-entry budget."
        )
    generator = torch.Generator().manual_seed(seed)
    if kind == "rademacher":
        return (torch.randint(2, (reps, units), generator=generator) * 2 - 1).to(torch.float64)
    if kind == "normal":
        return torch.randn((reps, units), generator=generator, dtype=torch.float64)
    if kind == "webb":
        values = torch.tensor(
            [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)]
        )
        return values[torch.randint(6, (reps, units), generator=generator)].to(torch.float64)
    raise AnalysisError("invalid_spec", "Multiplier kind must be rademacher, normal or webb.")


def advanced_intervals(
    draws, observed, alpha, kind, *, errors=None, original_errors=None, jack=None
):
    p = draws.shape[1]
    if kind in {"percentile", "bc"}:
        return bootstrap_intervals(draws, observed, alpha, kind), {}
    if kind == "normal":
        se = bootstrap_covariance(draws).diagonal().clamp_min(0).sqrt()
        z = normal_ppf(1 - alpha / 2)
        return list(
            zip((observed - z * se).tolist(), (observed + z * se).tolist(), strict=True)
        ), {}
    if kind == "studentized":
        if (
            errors is None
            or original_errors is None
            or bool((errors <= 0).any())
            or bool((original_errors <= 0).any())
        ):
            raise AnalysisError(
                "bootstrap_failed",
                "Studentized intervals require positive finite SEs for every parameter and draw.",
            )
        pivots = (draws - observed) / errors
        intervals = []
        for j in range(p):
            ordered = pivots[:, j].sort().values
            intervals.append(
                (
                    float(
                        observed[j] - stata_percentile(ordered, 1 - alpha / 2) * original_errors[j]
                    ),
                    float(observed[j] - stata_percentile(ordered, alpha / 2) * original_errors[j]),
                )
            )
        return intervals, {"studentization": "each refit's recorded covariance"}
    if kind != "bca" or jack is None:
        raise AnalysisError(
            "invalid_spec", "Unknown interval type or unavailable BCa delete-unit acceleration."
        )
    influence = jack.mean(dim=0) - jack
    denominator = 6 * influence.square().sum(dim=0).pow(1.5)
    acceleration = torch.where(
        denominator > 0, influence.pow(3).sum(dim=0) / denominator, torch.zeros(p)
    )
    intervals, adjusted = [], []
    for j in range(p):
        share = float((draws[:, j] <= observed[j]).to(torch.float64).mean())
        if not 0 < share < 1:
            intervals.append((None, None))
            adjusted.append(None)
            continue
        z0 = normal_ppf(share)
        probabilities = []
        for tail in [alpha / 2, 1 - alpha / 2]:
            z = z0 + normal_ppf(tail)
            d = 1 - float(acceleration[j]) * z
            if d <= 0:
                raise AnalysisError(
                    "bootstrap_failed", "BCa acceleration has a singular tail transformation."
                )
            probabilities.append(normal_cdf(z0 + z / d))
        ordered = draws[:, j].sort().values
        intervals.append(tuple(stata_percentile(ordered, q) for q in probabilities))
        adjusted.append(probabilities)
    return intervals, {
        "acceleration": acceleration.tolist(),
        "adjusted_probabilities": adjusted,
        "acceleration_method": "delete one resampling unit",
        "jackknife_refits": len(jack),
    }


def advanced_bootstrap(
    result,
    data,
    *,
    reps,
    seed,
    cluster,
    strata,
    size,
    alpha,
    ci,
    scheme,
    block_length,
    time,
    null,
    wild,
    max_refits,
):
    from openecon.econometrics.postest.resampling import (
        FAILURE_SHARE,
        _Population,
        _alpha_spec,
        _refuse_unsupported,
        _require_variation,
        _seed,
    )

    if scheme not in SCHEMES or ci not in {"normal", "percentile", "bc", "bca", "studentized"}:
        raise AnalysisError("invalid_spec", "Unknown bootstrap scheme or interval type.")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 2:
        raise AnalysisError("invalid_spec", "reps must be an integer of at least two.")
    if isinstance(max_refits, bool) or not isinstance(max_refits, int) or max_refits < 2:
        raise AnalysisError("invalid_spec", "max_refits must be an integer of at least two.")
    if strata is not None or size is not None or result.spec.weights is not None:
        raise AnalysisError(
            "bootstrap_unsupported",
            "Advanced schemes currently require unweighted, unstratified full-size draws.",
        )
    spec, seed_value = _alpha_spec(result, alpha), _seed(seed)
    if result_covariance(result) in {"bootstrap", "jackknife"} or spec.covariance in {
        "bootstrap",
        "jackknife",
    }:
        raise AnalysisError(
            "bootstrap_unsupported",
            "The fit already uses a resampling covariance; refit with its conventional covariance first.",
        )
    if scheme == "iid":
        _refuse_unsupported(result, "bootstrap")
        if block_length is not None or time is not None or null is not None or wild != "rademacher":
            raise AnalysisError("invalid_spec", "iid pairs do not accept dependence options.")
    elif (
        spec.estimator != "ols"
        or spec.panel is not None
        or any(term not in spec.predictors for term in spec.options.get("terms", spec.predictors))
    ):
        raise AnalysisError(
            "bootstrap_unsupported",
            "Dependent resampling currently validates OLS with explicit columns and no panel/formula transforms.",
        )
    if scheme != "iid" and ci == "bca":
        raise AnalysisError(
            "bootstrap_unsupported",
            "BCa acceleration is validated for iid rows/whole-cluster pairs only.",
        )
    if null is not None and (scheme != "wild_cluster" or ci != "normal"):
        raise AnalysisError(
            "invalid_spec",
            "Imposed-null tests require wild_cluster with ci='normal'; test-inverted confidence sets are not supplied.",
        )
    if scheme == "wild_cluster" and spec.time is not None:
        raise AnalysisError(
            "bootstrap_unsupported",
            "Wild cluster draws validate independent OLS clusters; use explicit blocks for a declared series.",
        )
    population = _Population(result, data, cluster, None, "bootstrap")
    base = population.base.copy()
    full, positions = matched_frame(result, data)
    if time is None:
        time = spec.time
    if scheme in {"moving_block", "residual_block"}:
        if (
            time is None
            or time not in full.columns
            or bool(full.iloc[positions][time].isna().any())
        ):
            raise AnalysisError(
                "invalid_spec", "Block draws require a complete explicit time column."
            )
        if cluster is not None:
            raise AnalysisError(
                "bootstrap_unsupported",
                "Blocks currently describe one regularly spaced series, without panel/cluster mixing.",
            )
        if time in spec.predictors:
            raise AnalysisError(
                "bootstrap_unsupported", "A block time index must not also be a predictor."
            )
        # Use data positions, preserving duplicate dataframe index labels safely.
        order = full.iloc[positions][time].to_numpy().argsort(kind="stable")
        base = base.iloc[order].reset_index(drop=True)
        times = pd.Series(full.iloc[positions][time].to_numpy()[order])
        delta = times.diff().dropna()
        if (
            not len(delta)
            or bool(
                (
                    delta <= pd.Timedelta(0)
                    if pd.api.types.is_datetime64_any_dtype(times)
                    else delta <= 0
                ).any()
            )
            or delta.nunique() != 1
        ):
            raise AnalysisError(
                "invalid_time",
                "Block resampling requires distinct, regularly spaced retained time observations.",
            )
        if (
            isinstance(block_length, bool)
            or not isinstance(block_length, int)
            or not 1 < block_length < len(base)
        ):
            raise AnalysisError(
                "invalid_spec", "block_length must be an integer between 2 and N-1."
            )
    elif block_length is not None or (
        time is not None and scheme not in {"residual", "wild_cluster"}
    ):
        raise AnalysisError("invalid_spec", "A block length requires a block scheme.")
    if scheme == "residual" and time is not None:
        raise AnalysisError(
            "bootstrap_unsupported",
            "iid residual draws assume independent errors; use residual_block for a series.",
        )
    if scheme not in {"iid", "wild_cluster"} and cluster is not None:
        raise AnalysisError(
            "invalid_spec", "cluster applies to iid whole-cluster pairs or wild_cluster."
        )
    if scheme != "wild_cluster" and wild != "rademacher":
        raise AnalysisError("invalid_spec", "wild weights require wild_cluster.")
    terms = [c.term for c in result.coefficients]
    observed = coefficient_vector(result)
    units = population.n_clusters if population.cluster_codes is not None else population.n
    jack_refits = units if ci == "bca" else 0
    if reps + jack_refits > max_refits or reps * len(terms) > MAX_ENTRIES:
        raise AnalysisError(
            "bootstrap_too_large",
            "Requested draws plus acceleration refits exceed the explicit bootstrap budget.",
        )
    refit = refitter(refit_spec(result))
    generator = torch.Generator().manual_seed(seed_value)
    clusters = (
        Clusters.build(population.cluster_codes, units)
        if population.cluster_codes is not None
        else None
    )
    table = Strata.build(torch.zeros(units, dtype=torch.int64), 1)
    fitted = resid = null_beta = None
    if scheme in {"residual", "residual_block", "wild_cluster"}:
        model_frame = ModelFrame(spec, base)
        from openecon.econometrics.postest.scores import _columns

        x = _columns(result, model_frame, terms)
        y = model_frame.numeric(spec.outcome)
        beta = observed.clone()
        if null is not None:
            if not isinstance(null, dict) or not null or any(k not in terms for k in null):
                raise AnalysisError(
                    "invalid_spec", "null must map reported coefficients to finite null values."
                )
            target = torch.tensor(list(null.values()), dtype=torch.float64)
            if not bool(torch.isfinite(target).all()):
                raise AnalysisError("invalid_spec", "Null values must be finite.")
            r = torch.eye(len(terms), dtype=torch.float64)[[terms.index(k) for k in null]]
            inv = torch.linalg.inv(x.T @ x)
            beta -= inv @ r.T @ torch.linalg.solve(r @ inv @ r.T, r @ beta - target)
        null_beta = beta
        fitted = x @ beta
        resid = y - fitted
        if scheme != "wild_cluster":
            resid -= resid.mean()
    wild_draws = None
    if scheme == "wild_cluster":
        if clusters is None:
            raise AnalysisError(
                "invalid_spec", "wild_cluster requires cluster= with at least two groups."
            )
        wild_draws = multipliers(units, reps, seed_value, wild)
        if (ci == "studentized" or null is not None) and (
            spec.covariance != "cluster" or spec.cluster != cluster
        ):
            raise AnalysisError(
                "bootstrap_unsupported",
                "Wild studentization requires the original OLS covariance clustered on the same column.",
            )
    draws, errors, reasons = [], [], Counter()
    for i in range(reps):
        if scheme == "iid":
            drawn = draw_units(table, table.count, generator)
            rows, slot = clusters.expand(drawn) if clusters else (drawn, None)
            sample = base.iloc[rows.numpy()].reset_index(drop=True)
            if clusters:
                population.relabel(sample, rows, slot)
        elif scheme == "moving_block":
            rows = block_indices(len(base), block_length, generator)
            sample = base.iloc[rows.numpy()].reset_index(drop=True)
            if spec.time is not None:
                # The resampled block sequence has a fresh, regular synthetic timeline.
                sample[spec.time] = base[spec.time].to_numpy()
        else:
            sample = base.copy()
            if scheme == "wild_cluster":
                sampled_resid = resid * wild_draws[i, population.cluster_codes]
            else:
                rows = (
                    block_indices(len(base), block_length, generator)
                    if scheme == "residual_block"
                    else torch.randint(len(base), (len(base),), generator=generator)
                )
                sampled_resid = resid[rows]
            sample[spec.outcome] = (fitted + sampled_resid).numpy()
        try:
            fit = refit(sample)
            if [c.term for c in fit.coefficients] != terms:
                raise AnalysisError(
                    "different_terms", "A replicate changes the reported parameter domain."
                )
            values = coefficient_vector(fit)
            se = torch.tensor(
                [math.nan if c.std_error is None else c.std_error for c in fit.coefficients],
                dtype=torch.float64,
            )
            if not bool(torch.isfinite(values).all()) or (
                (ci == "studentized" or null is not None)
                and (not bool(torch.isfinite(se).all()) or bool((se <= 0).any()))
            ):
                raise AnalysisError(
                    "invalid_replicate",
                    "Nonfinite estimates or nonpositive replicate standard errors.",
                )
        except (AnalysisError, ArithmeticError, RuntimeError, ValueError) as exc:
            reasons[getattr(exc, "code", type(exc).__name__)] += 1
        else:
            draws.append(values)
            errors.append(se)
    failed = reps - len(draws)
    if len(draws) < 2 or failed > reps * FAILURE_SHARE:
        raise AnalysisError(
            "bootstrap_failed", f"{failed}/{reps} draws failed (maximum 10%): {dict(reasons)}."
        )
    values = torch.stack(draws)
    _require_variation(values, observed, terms)
    jack = []
    for unit in range(jack_refits):
        keep = population.cluster_codes != unit if clusters else torch.arange(len(base)) != unit
        try:
            fit = refit(base.iloc[keep.numpy()].reset_index(drop=True))
            if [c.term for c in fit.coefficients] != terms:
                raise AnalysisError("different_terms", "Jackknife parameter domain changed.")
            value = coefficient_vector(fit)
            if not bool(torch.isfinite(value).all()):
                raise AnalysisError("invalid_replicate", "Jackknife estimates must be finite.")
            jack.append(value)
        except (AnalysisError, ArithmeticError, RuntimeError, ValueError) as exc:
            raise AnalysisError(
                "bootstrap_failed", "Every BCa delete-unit fit must succeed."
            ) from exc
    covariance = bootstrap_covariance(values)
    original_errors = torch.tensor(
        [math.nan if c.std_error is None else c.std_error for c in result.coefficients],
        dtype=torch.float64,
    )
    if (ci == "studentized" or null is not None) and (
        not bool(torch.isfinite(original_errors).all()) or bool((original_errors <= 0).any())
    ):
        raise AnalysisError(
            "bootstrap_failed",
            "Bootstrap-t inference requires finite positive original standard errors.",
        )
    pairs, details = advanced_intervals(
        values,
        observed,
        spec.alpha,
        ci,
        errors=torch.stack(errors),
        original_errors=original_errors,
        jack=torch.stack(jack) if jack else None,
    )
    record = dict(
        scheme=scheme,
        seed=seed_value,
        reps=reps,
        reps_completed=len(draws),
        failed_replicates=failed,
        failure_reasons=dict(reasons),
        block_length=block_length,
        time_column=time,
        resampling_unit="clusters"
        if clusters
        else "time blocks"
        if "block" in scheme
        else "observations",
        cluster_column=cluster,
        cluster_count=units if clusters else None,
        residual_model="restricted" if null else "unrestricted",
        null=null,
        wild=wild if wild_draws is not None else None,
        max_refits=max_refits,
        failure_policy="fail above 10%; report all discarded draws",
    )
    extra = {
        "bootstrap": record,
        "bootstrap_ci": {
            "type": ci,
            "confidence_level": 1 - spec.alpha,
            "intervals": {
                term: dict(ci_low=low, ci_high=high)
                for term, (low, high) in zip(terms, pairs, strict=True)
            },
            **details,
        },
    }
    if null is not None:
        # Restricted-bootstrap-t two-sided coefficient tests. No inversion is implied.
        pivots = (values - null_beta) / torch.stack(errors)
        tests = {
            term: {
                "null": target,
                "p_value": float(
                    (
                        pivots[:, terms.index(term)].abs()
                        >= abs(
                            float(
                                (observed[terms.index(term)] - target)
                                / original_errors[terms.index(term)]
                            )
                        )
                    )
                    .double()
                    .mean()
                ),
            }
            for term, target in null.items()
        }
        extra["bootstrap_test"] = {
            "type": "restricted wild bootstrap-t",
            "tests": tests,
            "confidence_set_inversion": False,
        }
    warnings = [*population.notes]
    if failed:
        warnings.append(f"{failed}/{reps} failed draws discarded: {dict(reasons)}.")
    if clusters and units < 30:
        warnings.append(
            f"Only {units} clusters; bootstrap accuracy remains sample- and design-dependent."
        )
    undefined = [
        term
        for term, bounds in extra["bootstrap_ci"]["intervals"].items()
        if bounds["ci_low"] is None
    ]
    if undefined:
        warnings.append(
            f"Bootstrap bias correction is undefined for {', '.join(undefined)}; all draws lie on one side of the estimate."
        )
    return rebuild(
        result,
        covariance,
        method="bootstrap",
        use_t=False,
        df_inference=None,
        correction="covariance of successful replicates, divisor R-1",
        inference=record,
        provenance={
            "postestimation": {
                "method": "bootstrap",
                **record,
                "original_result_id": result.id,
                "original_covariance": result_covariance(result),
                "generator": "local torch.Generator",
            }
        },
        extra=extra,
        tests={},
        warnings=warnings,
        spec=spec,
        title=f"{result.title or spec.estimator} ({scheme} bootstrap)",
    )
