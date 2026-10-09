"""Restricted wild-cluster linear inference with explicit design and draw state."""

from __future__ import annotations

import math
from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.core import ModelFrame, TableSet, kernel_call, table
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.engines.linalg import least_squares
from openecon.resources import plan_workspace
from openecon.econometrics.resident_cpu import resident_cpu


def _count(value, name, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not minimum <= value <= maximum
    ):
        raise AnalysisError("invalid_option", f"{name} must be an integer in {minimum}..{maximum}.")
    return int(value)


def _geometry(result, data, cluster):
    result = require_result(result, "wild-cluster result")
    spec = result.spec
    if (
        spec.estimator not in {"ols", "areg", "reghdfe", "xtreg"}
        or spec.weights
        or spec.categorical
    ):
        raise AnalysisError(
            "unsupported_spec", "Use unweighted numeric OLS/absorbed FE/panel FE results."
        )
    if spec.estimator == "xtreg" and spec.options.get("model", "fe") != "fe":
        raise AnalysisError("unsupported_spec", "Only panel fixed effects are supported.")
    if spec.options.get("formula") or spec.options.get("terms", spec.predictors) != spec.predictors:
        raise AnalysisError(
            "unsupported_spec", "Use explicit numeric predictors without formula transformations."
        )
    original, positions = matched_frame(result, data)
    sample = original.iloc[positions].copy()
    if cluster is None:
        cluster = spec.cluster or spec.panel
    if not isinstance(cluster, str) or cluster not in sample or sample[cluster].isna().any():
        raise AnalysisError("invalid_cluster", "Declare one complete independent cluster column.")
    frame = ModelFrame(spec, sample)
    if frame.n > 8192 or frame.design_width() > 256:
        raise AnalysisError(
            "work_budget_exceeded", "Wild geometry requires N<=8192 and K_full<=256."
        )
    design = frame.design()
    x, terms = design.x, list(design.terms)
    codes, labels = pd.factorize(sample[cluster], sort=False)
    groups = len(labels)
    if groups < 2:
        raise AnalysisError(
            "insufficient_clusters", "At least two independent clusters are required."
        )
    fe = []
    if spec.estimator == "xtreg":
        fe = [spec.panel]
    elif spec.estimator in {"areg", "reghdfe"}:
        raw = spec.columns.get("absorb")
        fe = [raw] if isinstance(raw, str) else list(raw or [])
    for name in fe:
        if name not in sample or sample[name].isna().any():
            raise AnalysisError("invalid_groups", "Fixed-effect identity must be complete.")
        fc, fl = pd.factorize(sample[name], sort=False)
        # Each FE group must lie within an independent bootstrap cluster.
        for i in range(len(fl)):
            if len(set(codes[fc == i])) != 1:
                raise AnalysisError(
                    "invalid_groups", "Fixed-effect groups must nest within bootstrap clusters."
                )
        if not spec.intercept:
            raise AnalysisError(
                "unsupported_spec", "Absorbed designs require an explicit intercept in this route."
            )
        if len(fl) > 256:
            raise AnalysisError(
                "work_budget_exceeded",
                "Explicit FE geometry permits at most 256 levels per effect.",
            )
        proposed = x.shape[1] + len(fl) - 1
        if proposed > 256 or len(sample) <= proposed:
            raise AnalysisError(
                "work_budget_exceeded",
                "Explicit FE geometry exceeds the full-rank parameter budget.",
            )
        frame.workspace_plan(
            "wild FE dummy admission",
            {"full_design_and_copy": len(sample) * proposed * 40, "dummy_eye": len(fl) ** 2 * 8},
        )
        dummy = torch.eye(len(fl), dtype=torch.float64)[torch.tensor(fc, dtype=torch.int64), 1:]
        x = torch.cat((x, dummy), 1)
        terms.extend(f"nuisance:{name}:{j}" for j in range(1, len(fl)))
    n, k = x.shape
    if n > 8192 or k > 256 or n <= k or n * k * k > 100_000_000:
        raise AnalysisError(
            "work_budget_exceeded",
            "Wild inference needs N<=8192, K<=256, N>K and bounded explicit geometry.",
        )
    frame.workspace_plan("restricted wild geometry", {"design": n * k * 40, "bread": k * k * 32})
    y = frame.numeric(spec.outcome)
    fit = kernel_call(least_squares, x, y, drop_collinear=False)
    return dict(
        x=x,
        y=y,
        beta=fit.beta,
        bread=fit.xtx_inv,
        codes=torch.tensor(codes, dtype=torch.int64),
        groups=groups,
        labels=labels.tolist(),
        terms=terms,
        positions=positions,
        fe=fe,
        cluster=cluster,
        result_id=result.id,
        n=n,
        k=k,
    )


def _covariance(g, residual):
    scores = g["x"] * residual[:, None]
    sums = scores.new_zeros((g["groups"], g["k"])).index_add(0, g["codes"], scores)
    correction = g["groups"] / (g["groups"] - 1) * (g["n"] - 1) / (g["n"] - g["k"])
    return g["bread"] @ (sums.T @ sums) @ g["bread"] * correction


def _restrictions(g, null):
    if not isinstance(null, dict) or not null or len(null) > 24:
        raise AnalysisError("invalid_restrictions", "Supply 1..24 term-to-null-value restrictions.")
    r = torch.zeros((len(null), g["k"]), dtype=torch.float64)
    values = []
    for i, (term, value) in enumerate(null.items()):
        if (
            term not in g["terms"]
            or term.startswith("nuisance:")
            or (g["fe"] and term == "Intercept")
        ):
            raise AnalysisError(
                "invalid_restrictions",
                "Test reported slopes; FE intercepts/nuisance coordinates are unsupported.",
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise AnalysisError("invalid_restrictions", "Null values must be finite real numbers.")
        r[i, g["terms"].index(term)] = 1.0
        values.append(float(value))
    if g["groups"] <= len(null):
        raise AnalysisError(
            "insufficient_clusters",
            "There must be more independent clusters than joint restrictions.",
        )
    return r, torch.tensor(values, dtype=torch.float64)


def _weights(groups, reps, seed, wild, enumerate_all):
    seed = _count(seed, "seed", 0, 2**63 - 1)
    if wild not in {"rademacher", "mammen", "webb"} or not isinstance(enumerate_all, bool):
        raise AnalysisError(
            "invalid_option", "Use rademacher/mammen/webb and boolean enumerate_all."
        )
    if enumerate_all:
        if wild != "rademacher" or groups > 12:
            raise AnalysisError(
                "enumeration_limit", "Complete enumeration requires Rademacher and G<=12."
            )
        values = torch.arange(2**groups, dtype=torch.int64)[:, None]
        return (((values >> torch.arange(groups, dtype=torch.int64)) & 1) * 2 - 1).to(torch.float64)
    reps = _count(reps, "reps", 49, 100_000)
    if reps * groups > 2_000_000:
        raise AnalysisError("work_budget_exceeded", "Multiplier storage exceeds 2,000,000 entries.")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    if wild == "rademacher":
        return (torch.randint(2, (reps, groups), generator=rng) * 2 - 1).to(torch.float64)
    if wild == "webb":
        choices = torch.tensor(
            [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)],
            dtype=torch.float64,
        )
        return choices[torch.randint(6, (reps, groups), generator=rng)]
    lower, upper = (1 - math.sqrt(5)) / 2, (1 + math.sqrt(5)) / 2
    prob = (math.sqrt(5) + 1) / (2 * math.sqrt(5))
    draws = torch.rand((reps, groups), generator=rng, dtype=torch.float64)
    return torch.where(draws < prob, lower, upper)


def _admit_draws(g, reps, enumerate_all, wild, max_work, grid=1):
    if not isinstance(enumerate_all, bool) or wild not in {"rademacher", "mammen", "webb"}:
        raise AnalysisError(
            "invalid_option", "Declare a supported multiplier and boolean enumeration."
        )
    if enumerate_all:
        if wild != "rademacher" or g["groups"] > 12:
            raise AnalysisError(
                "enumeration_limit", "Complete enumeration requires Rademacher and G<=12."
            )
        count = 2 ** g["groups"]
    else:
        count = _count(reps, "reps", 49, 100_000)
    work = grid * (count + 1) * g["n"] * g["k"] ** 2
    if count * g["groups"] > 2_000_000 or work > max_work:
        raise AnalysisError(
            "work_budget_exceeded", "Wild refits/multipliers exceed the declared work budget."
        )
    plan = plan_workspace(
        "restricted wild draws and studentization",
        {
            "multipliers": count * g["groups"] * 32,
            "design_and_scores": g["n"] * g["k"] * 64,
            "bread_and_full_covariance": g["k"] ** 2 * 64,
            "cluster_scores": g["groups"] * g["k"] * 32,
        },
    )
    return work, plan.record()


def _stat(delta, covariance):
    try:
        chol = torch.linalg.cholesky(covariance)
        value = float(delta @ torch.cholesky_solve(delta[:, None], chol).flatten())
    except RuntimeError as exc:
        raise AnalysisError(
            "singular_covariance", "Restricted studentizing covariance is not positive definite."
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise AnalysisError(
            "invalid_statistic", "Restricted Wald statistic is not finite/nonnegative."
        )
    return value


def _test(g, null, weights, enumerate_all):
    r, target = _restrictions(g, null)
    delta = r @ g["beta"] - target
    unrestricted = g["y"] - g["x"] @ g["beta"]
    v = _covariance(g, unrestricted)
    observed = _stat(delta, r @ v @ r.T)
    rb = r @ g["bread"] @ r.T
    restricted = g["beta"] - g["bread"] @ r.T @ torch.linalg.solve(rb, delta)
    resid = g["y"] - g["x"] @ restricted
    statistics = []
    for draw in weights:
        noise = resid * draw[g["codes"]]
        change = g["bread"] @ (g["x"].T @ noise)
        b = restricted + change
        error = noise - g["x"] @ change
        statistics.append(_stat(r @ b - target, r @ _covariance(g, error) @ r.T))
    count = sum(value >= observed * (1 - 1e-12) for value in statistics)
    probability = count / len(statistics) if enumerate_all else (count + 1) / (len(statistics) + 1)
    return dict(
        statistic=observed,
        p_value=probability,
        df=len(null),
        covariance_matrix=v.tolist(),
        restricted_parameters=restricted.tolist(),
        bootstrap_statistics=statistics,
        exceedances=count,
        mc_std_error=None
        if enumerate_all
        else math.sqrt(probability * (1 - probability) / len(statistics)),
    )


@resident_cpu
def wild_cluster_test(
    result,
    *,
    data,
    null,
    cluster=None,
    reps=999,
    seed=0,
    wild="rademacher",
    enumerate_all=False,
    max_work=100_000_000,
):
    """Null-imposed CR1-studentized wild-cluster Wald test of linear coefficients.

    Numeric unweighted resident OLS/connected absorbed FE/panel FE, one
    independent cluster dimension, nested FE and bounded explicit dummy design.
    Every draw recomputes its residual CR1 covariance; failed draws raise.
    Complete Rademacher enumeration is conditional on the restricted fitted
    sample, not an exact population/randomization test. MC uses plus-one p.
    """
    max_work = _count(max_work, "max_work", 1, 10**12)
    g = _geometry(result, data, cluster)
    _restrictions(g, null)
    work, plan = _admit_draws(g, reps, enumerate_all, wild, max_work)
    weights = _weights(g["groups"], reps, seed, wild, enumerate_all)
    value = _test(g, null, weights, enumerate_all)
    return saved_summary(
        TableSet(
            {
                "test": table(
                    [{k: value[k] for k in ("statistic", "p_value", "df", "mc_std_error")}]
                )
            },
            title="Restricted wild-cluster inference",
            **value,
            null=null,
            source_result_id=g["result_id"],
            sample_positions=g["positions"],
            terms=g["terms"],
            cluster=g["cluster"],
            cluster_labels=g["labels"],
            cluster_sizes=torch.bincount(g["codes"]).tolist(),
            draws=len(weights),
            seed=seed,
            multiplier=wild,
            enumeration=enumerate_all,
            multipliers=weights.tolist(),
            work=work,
            resource_plan=plan,
            correction="CR1: G/(G-1)*(N-1)/(N-K_full); complete explicit FE rank",
            p_convention="complete conditional enumeration"
            if enumerate_all
            else "plus-one Monte Carlo",
            tail_tolerance="relative 1e-12 equality guard",
            failed_draws=0,
            assumptions="Independent clusters and valid linear mean design; finite-G accuracy is design-dependent; no structural/assignment claim.",
            source="https://doi.org/10.1016/j.jeconom.2022.04.001",
        )
    )


@resident_cpu
def wild_cluster_confidence_set(
    result,
    *,
    data,
    term,
    grid,
    alpha=0.05,
    cluster=None,
    reps=999,
    seed=0,
    wild="rademacher",
    enumerate_all=False,
    max_work=100_000_000,
):
    """Invert scalar restricted tests on a declared finite increasing grid.

    Returns accepted grid points/components and rejection boundary brackets.
    It does not interpolate, silently convexify disconnected sets or assert
    unbounded/full real-line coverage beyond tested endpoints. Draws are shared
    across nulls; grid resolution and unknown outside tails are explicit.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must be in (0,1).")
    if not isinstance(grid, (list, tuple)) or not 2 <= len(grid) <= 201:
        raise AnalysisError("invalid_grid", "Supply 2..201 ordered finite grid values.")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in grid
    ) or any(a >= b for a, b in zip(grid, grid[1:])):
        raise AnalysisError("invalid_grid", "The grid must be finite and strictly increasing.")
    g = _geometry(result, data, cluster)
    _restrictions(g, {term: 0.0})
    max_work = _count(max_work, "max_work", 1, 10**12)
    work, plan = _admit_draws(g, reps, enumerate_all, wild, max_work, grid=len(grid))
    weights = _weights(g["groups"], reps, seed, wild, enumerate_all)
    rows = []
    for value in grid:
        out = _test(g, {term: value}, weights, enumerate_all)
        rows.append(dict(null=value, p_value=out["p_value"], accepted=out["p_value"] > alpha))
    components = []
    start = None
    for i, row in enumerate(rows + [dict(accepted=False)]):
        if row["accepted"] and start is None:
            start = i
        if not row["accepted"] and start is not None:
            stop = i - 1
            components.append(
                dict(
                    lower_grid=grid[start],
                    upper_grid=grid[stop],
                    left_rejected_grid=grid[start - 1] if start else None,
                    right_rejected_grid=grid[i] if i < len(grid) else None,
                )
            )
            start = None
    return saved_summary(
        TableSet(
            {"grid": table(rows), "components": table(components)},
            title="Wild-cluster finite-grid confidence set",
            term=term,
            alpha=alpha,
            seed=seed,
            draws=len(weights),
            enumeration=enumerate_all,
            source_result_id=g["result_id"],
            sample_positions=g["positions"],
            work=work,
            resource_plan=plan,
            maximum_grid_gap=max(b - a for a, b in zip(grid, grid[1:])),
            outside_grid="unknown; no tail extrapolation",
            interval_convention="accepted tested points; component endpoints and rejected boundary brackets; no interpolation",
            shared_null_multipliers=True,
        )
    )
