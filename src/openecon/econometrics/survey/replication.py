"""Single-stage PSU replication; no row-bootstrap or ordinary jackknife fallback."""

from __future__ import annotations

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from .common import FLOAT, SurveyResult, allocation, digest, finite, number, prepare, result

MAX_REPLICATES = 4096
MAX_CELLS = 8_000_000
MAX_WORK = 50_000_000
MAX_BALANCE_WORK = 50_000_000


def integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise AnalysisError("survey_replicate_budget", f"{name} must be an integer in {lo}..{hi}.")
    return value


def budget(target, count, supplied=False):
    integer(count, "replicate count", 1, MAX_REPLICATES)
    n, k = len(target.weights), len(target.labels)
    if n * count > MAX_CELLS or n * k * count > MAX_WORK:
        raise AnalysisError(
            "survey_replicate_budget",
            "Complete replicate cells/work exceed the explicit bounded plan.",
        )
    return allocation(n, k, target.design.validation.n_psu, count, supplied).record()


def _supplied_preflight(target, data):
    """Admit resident weight shape and storage before numeric tensor copies."""
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError(
            "invalid_survey_replicates",
            "Supplied replicate weights need a resident DataFrame with explicit unique column IDs.",
        )
    if (
        len(data) != len(target.weights)
        or data.columns.has_duplicates
        or any(not isinstance(v, str) or not v or len(v) > 200 for v in data.columns)
    ):
        raise AnalysisError(
            "invalid_survey_replicates",
            "Weights require exact physical row order and distinct bounded replicate IDs.",
        )
    budget(target, len(data.columns), True)
    if (
        sum(int(data[c].memory_usage(index=False, deep=True)) for c in data.columns)
        > target.design.max_memory_mb * 1024**2
    ):
        raise AnalysisError(
            "survey_replicate_budget",
            "Supplied weight columns exceed the declaration's resident admission budget.",
        )
    return len(data.columns)


def supplied(target, data):
    _supplied_preflight(target, data)
    values = torch.tensor(
        [
            [number(v, "replicate weight", 0.0) for v in row]
            for row in data.itertuples(index=False, name=None)
        ],
        dtype=FLOAT,
    ).T
    factors = values / target.weights
    finite(factors, "Replicate PSU factors")
    # Compare all PSU factors in O(N*R); no full-row scan per PSU.
    first = {}
    for i, p in enumerate(target.groups.tolist()):
        first.setdefault(p, i)
    psu_factors = factors[:, list(first.values())]
    if not torch.allclose(factors, psu_factors[:, target.groups], atol=1e-12, rtol=1e-12):
        raise AnalysisError(
            "incompatible_survey_replicates",
            "Replicate factors must be constant within each first-stage PSU.",
        )
    certainty = [
        p for h, psus in enumerate(target.strata_groups) if not target.fpc[h] for p in psus
    ]
    if certainty and not torch.allclose(
        psu_factors[:, certainty],
        torch.ones_like(psu_factors[:, certainty]),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AnalysisError(
            "incompatible_survey_replicates",
            "Census PSU weights must remain unchanged in every replica.",
        )
    return values, list(data.columns), digest(values.tolist())


def evaluate_all(target, weights, ids):
    estimates, failures = [], []
    for row, identity in zip(weights, ids):
        try:
            estimates.append(target.evaluate(row)[0])
        except AnalysisError as exc:
            failures.append({"id": identity, "code": exc.code})
    if failures:
        raise AnalysisError(
            "survey_replicate_failure",
            "Complete replicate inference failed; no replica was omitted: " + str(failures),
        )
    return finite(torch.stack(estimates), "Replicate estimates")


def finish(
    target,
    method,
    weights,
    ids,
    multipliers,
    *,
    centering,
    alpha,
    null,
    df=None,
    metadata=None,
    strata=None,
):
    if not isinstance(centering, str) or centering not in {
        "original",
        "replicate_mean",
        "stratum_mean",
    }:
        raise AnalysisError(
            "invalid_survey_replicates",
            "Declare original, replicate_mean or jackknife stratum_mean centering.",
        )
    estimate = target.evaluate(target.weights)[0]
    replicas = evaluate_all(target, weights, ids)
    count = len(ids)
    if centering == "original":
        difference = replicas - estimate
    elif centering == "replicate_mean":
        difference = replicas - replicas.mean(0)
    else:
        if strata is None:
            raise AnalysisError(
                "invalid_survey_replicates",
                "stratum_mean is only supported by stratified jackknife.",
            )
        difference = torch.empty_like(replicas)
        for h in sorted(set(strata)):
            positions = [i for i, value in enumerate(strata) if value == h]
            difference[positions] = replicas[positions] - replicas[positions].mean(0)
    covariance = finite(
        (difference.T * torch.tensor(multipliers, dtype=FLOAT)) @ difference, "Replicate covariance"
    )
    df = (
        min(target.design.validation.design_df, count - 1)
        if df is None
        else integer(df, "replicate df", 1, count - 1)
    )
    record = {
        "replicate_ids": ids,
        "replicate_count": count,
        "replicate_estimates": replicas.tolist(),
        "variance_multipliers": multipliers,
        "centering": centering,
        "failed_replicates": [],
        "replicate_df_convention": "minimum of complete-design df and R-1 unless explicitly declared",
        "workspace": budget(target, count),
        **(metadata or {}),
    }
    return result(
        target, method, estimate, covariance, alpha=alpha, null=null, df=df, metadata=record
    )


def hadamard_signs(size, columns):
    # Sylvester entries are (-1)**popcount(row & column). Only selected
    # nonconstant columns are materialized, never an R-by-R square.
    return torch.tensor(
        [
            [(-1.0) ** ((row & column).bit_count()) for column in range(1, columns + 1)]
            for row in range(size)
        ],
        dtype=FLOAT,
    )


def plan_brr(
    admitted, *, replicates=None, replicate_weights=None, centering="original", rho=0.0
):
    """Plan balanced first-stage weights from already admitted complete geometry.

    The caller supplies its original target or regression sample. No outcomes,
    domains or missing rows are read or selected a second time here.
    """
    rho = number(rho, "Fay rho", 0.0, 1.0 - 1e-6)
    if not isinstance(centering, str) or centering not in {"original", "replicate_mean"}:
        raise AnalysisError(
            "invalid_survey_replicates", "BRR/Fay centering must be original or replicate_mean."
        )
    stochastic = [h for h, factor in enumerate(admitted.fpc) if factor]
    if any(len(admitted.strata_groups[h]) != 2 for h in stochastic):
        raise AnalysisError(
            "unsupported_survey_brr", "Each noncensus stratum must contain exactly two PSUs."
        )
    if any(admitted.fpc[h] != 1.0 for h in stochastic):
        raise AnalysisError(
            "unsupported_survey_brr",
            "Noncensus FPC BRR requires a separate finite-population method; no FPC is ignored.",
        )
    if not stochastic:
        raise AnalysisError(
            "unsupported_survey_brr", "A census has zero design variance; use Taylor inference."
        )
    if replicate_weights is not None:
        if replicates is not None:
            raise AnalysisError(
                "invalid_survey_replicates",
                "Supply weights or a generated replicate count, not both.",
            )
        count = _supplied_preflight(admitted, replicate_weights)
        if count <= len(stochastic) or count * len(stochastic) ** 2 > MAX_BALANCE_WORK:
            raise AnalysisError(
                "survey_replicate_budget",
                "Supplied balanced BRR requires R > stochastic strata and bounded R*strata^2 validation work.",
            )
        weights, ids, weight_hash = supplied(admitted, replicate_weights)
        signs = []
        for h in stochastic:
            first, second = admitted.strata_groups[h]
            a = (weights[:, admitted.groups == first] / admitted.weights[admitted.groups == first])[
                :, 0
            ]
            b = (
                weights[:, admitted.groups == second] / admitted.weights[admitted.groups == second]
            )[:, 0]
            valid = (
                torch.isclose(a, torch.full_like(a, rho), atol=1e-12, rtol=1e-12)
                & torch.isclose(b, torch.full_like(b, 2.0 - rho), atol=1e-12, rtol=1e-12)
            ) | (
                torch.isclose(b, torch.full_like(b, rho), atol=1e-12, rtol=1e-12)
                & torch.isclose(a, torch.full_like(a, 2.0 - rho), atol=1e-12, rtol=1e-12)
            )
            if not bool(valid.all()):
                raise AnalysisError(
                    "incompatible_survey_replicates",
                    "Supplied BRR/Fay factors do not match declared PSU pairs/rho.",
                )
            signs.append(torch.where(a > 1.0, 1.0, -1.0).to(FLOAT))
        signs = torch.stack(signs, 1)
        balanced = torch.equal(signs.sum(0), torch.zeros(len(stochastic), dtype=FLOAT))
        # Check column dot products in bounded vectors; never form H-by-H Gram storage.
        for i in range(1, len(stochastic)):
            balanced = balanced and bool(((signs[:, :i].T @ signs[:, i]) == 0).all())
        if not balanced:
            raise AnalysisError(
                "unbalanced_survey_brr",
                "Supplied BRR/Fay selections must have zero-mean orthogonal stratum signs.",
            )
        source = "supplied balanced PSU weights"
    else:
        count = (
            1 << len(stochastic).bit_length()
            if replicates is None
            else integer(replicates, "replicates", 2, MAX_REPLICATES)
        )
        if count & (count - 1) or count <= len(stochastic):
            raise AnalysisError(
                "unsupported_survey_brr",
                "Generated BRR uses a power-of-two Sylvester order strictly exceeding stochastic strata.",
            )
        budget(admitted, count, True)
        signs = hadamard_signs(count, len(stochastic))
        weights = admitted.weights.expand(count, -1).clone()
        for column, h in enumerate(stochastic):
            first, second = admitted.strata_groups[h]
            weights[:, admitted.groups == first] *= (1.0 + (1.0 - rho) * signs[:, column])[:, None]
            weights[:, admitted.groups == second] *= (1.0 - (1.0 - rho) * signs[:, column])[:, None]
        ids = [f"sylvester-{i}" for i in range(count)]
        weight_hash = digest(weights.tolist())
        source = "generated deterministic Sylvester Hadamard, constant column excluded"
    multiplier = 1.0 / (count * (1.0 - rho) ** 2)
    return {
        "weights": weights,
        "ids": ids,
        "multipliers": [multiplier] * count,
        "centering": centering,
        "df": min(admitted.design.validation.design_df, count - 1),
        "strata": None,
        "metadata": {
            "fay_rho": rho,
            "scale": multiplier,
            "rscales": [1.0] * count,
            "replicate_weight_sha256": weight_hash,
            "generation": source,
            "balanced_signs_sha256": digest(signs.tolist()),
            "seed": None,
        },
    }


def _finish_plan(admitted, method, plan, *, alpha, null):
    # Zero df is an admitted default for an all-census supplied bootstrap. It
    # must not be reinterpreted as an explicitly requested user df of zero.
    return finish(
        admitted, method, plan["weights"], plan["ids"], plan["multipliers"],
        centering=plan["centering"], alpha=alpha, null=null,
        df=plan["df"] or None, metadata=plan["metadata"], strata=plan["strata"],
    )


def brr(
    data,
    design,
    outcomes,
    *,
    target="mean",
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    replicates=None,
    replicate_weights=None,
    centering="original",
    rho=0.0,
    method="brr",
):
    admitted = prepare(
        data, design, target, outcomes, denominators=denominators,
        categories=categories, domain=domain, missing=missing,
    )
    plan = plan_brr(
        admitted, replicates=replicates, replicate_weights=replicate_weights,
        centering=centering, rho=rho,
    )
    return _finish_plan(admitted, method, plan, alpha=alpha, null=null)


def survey_brr(
    data,
    design,
    outcomes,
    *,
    target="mean",
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    replicates=None,
    replicate_weights=None,
    centering="original",
) -> SurveyResult:
    """Balanced two-PSU BRR for one-stage targets; supplied balance is verified."""
    return brr(
        data,
        design,
        outcomes,
        target=target,
        denominators=denominators,
        categories=categories,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        replicates=replicates,
        replicate_weights=replicate_weights,
        centering=centering,
    )


def survey_fay(
    data,
    design,
    outcomes,
    *,
    rho=0.5,
    target="mean",
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    replicates=None,
    replicate_weights=None,
    centering="original",
) -> SurveyResult:
    """Fay BRR with explicit rho and 1/[R*(1-rho)^2] variance multiplier."""
    return brr(
        data,
        design,
        outcomes,
        target=target,
        denominators=denominators,
        categories=categories,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        replicates=replicates,
        replicate_weights=replicate_weights,
        centering=centering,
        rho=rho,
        method="fay",
    )


def plan_jackknife(admitted, *, centering="original"):
    """Plan stratum delete-one weights, retaining complete PSU/FPC geometry."""
    if not isinstance(centering, str) or centering not in {"original", "stratum_mean"}:
        raise AnalysisError(
            "invalid_survey_replicates", "Jackknife centering must be original or stratum_mean."
        )
    count = sum(len(p) for p, f in zip(admitted.strata_groups, admitted.fpc) if f)
    if not count:
        raise AnalysisError(
            "unsupported_survey_jackknife",
            "A census has zero design variance; use Taylor inference.",
        )
    budget(admitted, count, True)
    weights, ids, multipliers, strata_ids = [], [], [], []
    for h, (psus, fpc) in enumerate(zip(admitted.strata_groups, admitted.fpc)):
        if not fpc:
            continue
        block = torch.tensor([int(v) in psus for v in admitted.groups], dtype=torch.bool)
        for p in psus:
            row = admitted.weights.clone()
            row[block] *= len(psus) / (len(psus) - 1)
            row[admitted.groups == p] = 0.0
            weights.append(row)
            ids.append(f"stratum-{h}-delete-psu-{p}")
            multipliers.append(fpc * (len(psus) - 1) / len(psus))
            strata_ids.append(h)
    weights = torch.stack(weights)
    return {
        "weights": weights,
        "ids": ids,
        "multipliers": multipliers,
        "centering": centering,
        "df": min(admitted.design.validation.design_df, count - 1),
        "strata": strata_ids,
        "metadata": {
            "replicate_stratum_indices": strata_ids,
            "fpc_multipliers": admitted.fpc,
            "scale": 1.0,
            "rscales": multipliers,
            "replicate_weight_sha256": digest(weights.tolist()),
            "generation": "whole-PSU stratified delete-one; census PSUs unchanged",
        },
    }


def survey_jackknife(
    data,
    design,
    outcomes,
    *,
    target="mean",
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    centering="original",
) -> SurveyResult:
    """Stratum-aware delete-one PSU jackknife, including stratum FPC and census handling."""
    if not isinstance(centering, str) or centering not in {"original", "stratum_mean"}:
        raise AnalysisError(
            "invalid_survey_replicates", "Jackknife centering must be original or stratum_mean."
        )
    admitted = prepare(
        data, design, target, outcomes, denominators=denominators,
        categories=categories, domain=domain, missing=missing,
    )
    return _finish_plan(
        admitted, "jackknife", plan_jackknife(admitted, centering=centering),
        alpha=alpha, null=null,
    )


def plan_bootstrap(
    admitted, replicate_weights, *, justification, scale, rscales=None,
    df=None, centering="original",
):
    """Plan supplied first-stage weights with declared scaling and provenance."""
    if not isinstance(justification, str) or not 1 <= len(justification.strip()) <= 2000:
        raise AnalysisError(
            "invalid_survey_replicates",
            "Declare the design-bootstrap method/provenance in justification.",
        )
    if not isinstance(centering, str) or centering not in {"original", "replicate_mean"}:
        raise AnalysisError(
            "invalid_survey_replicates", "Bootstrap centering must be original or replicate_mean."
        )
    count = _supplied_preflight(admitted, replicate_weights)
    if count < 2:
        raise AnalysisError(
            "invalid_survey_replicates", "Bootstrap needs at least two supplied replicates."
        )
    scale = number(scale, "scale", 1e-15, 1e15)
    if rscales is None:
        rscales = [1.0] * count
    if not isinstance(rscales, (list, tuple)) or len(rscales) != count:
        raise AnalysisError(
            "invalid_survey_replicates", "Declare one rscale per complete replicate."
        )
    rscales = [number(v, "rscale", 1e-15, 1e15) for v in rscales]
    resolved_df = (
        min(admitted.design.validation.design_df, count - 1)
        if df is None else integer(df, "replicate df", 1, count - 1)
    )
    weights, ids, weight_hash = supplied(admitted, replicate_weights)
    return {
        "weights": weights,
        "ids": ids,
        "multipliers": [scale * v for v in rscales],
        "centering": centering,
        "df": resolved_df,
        "strata": None,
        "metadata": {
            "scale": scale,
            "rscales": rscales,
            "justification": justification,
            "replicate_weight_sha256": weight_hash,
            "generation": "supplied first-stage PSU bootstrap weights; caller's sampling provenance is not authenticated",
        },
    }


def survey_bootstrap(
    data,
    design,
    outcomes,
    replicate_weights,
    *,
    justification,
    scale,
    rscales=None,
    df=None,
    target="mean",
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    centering="original",
) -> SurveyResult:
    """Supplied design-bootstrap weights, declared variance scale/df/centering; no row resampling."""
    if not isinstance(justification, str) or not 1 <= len(justification.strip()) <= 2000:
        raise AnalysisError(
            "invalid_survey_replicates",
            "Declare the design-bootstrap method/provenance in justification.",
        )
    if not isinstance(centering, str) or centering not in {"original", "replicate_mean"}:
        raise AnalysisError(
            "invalid_survey_replicates", "Bootstrap centering must be original or replicate_mean."
        )
    admitted = prepare(
        data, design, target, outcomes, denominators=denominators,
        categories=categories, domain=domain, missing=missing,
    )
    plan = plan_bootstrap(
        admitted, replicate_weights, justification=justification, scale=scale,
        rscales=rscales, df=df, centering=centering,
    )
    return _finish_plan(admitted, "bootstrap", plan, alpha=alpha, null=null)
