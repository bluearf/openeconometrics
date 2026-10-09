"""Complete coefficient refits with admitted single-stage PSU replicate weights.

The physical sample, survey declaration, and coefficient order are admitted
once. Replicas change only weights and positive-weight support. No Taylor
covariance or replacement survey declaration is constructed for a replica.
"""

from __future__ import annotations

from dataclasses import replace
import json

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from .common import FLOAT, finite, number
from .regression import _inference_options, _options, _work_budget, fit_sample
from .regression_common import MAX_WORK, SurveyRegressionResult, admit, build_result, cpu_call
from .replication import (
    MAX_REPLICATES, _supplied_preflight, budget as replica_budget, integer,
    plan_bootstrap, plan_brr, plan_jackknife,
)

_DF_CONVENTION = "minimum of complete-design df and R-1 unless explicitly declared"
_SAFETY_CODES = {"survey_regression_budget", "survey_replicate_budget", "workspace_limit"}


class _WorkLedger:
    """Reserve cumulative work before each bounded numerical operation."""

    def __init__(self, planned):
        self.planned = planned
        self.used = 0

    def reserve(self, units):
        if type(units) is not int or units < 0:
            raise AnalysisError("invalid_resource_budget", "Work reservations must be nonnegative integers.")
        if self.used + units > MAX_WORK:
            error = AnalysisError(
                "survey_regression_budget",
                "Cumulative original and replicate fits, including every step-halving evaluation, "
                "exceed 50 million work units; execution stopped before the next operation.",
            )
            error.work_record = self.record()
            raise error
        self.used += units

    def record(self):
        return {
            "planned_work_units": self.planned,
            "actual_work_units": self.used,
            "max_work_units": MAX_WORK,
            "accounting": "planner and covariance work + complete-design fit overhead + "
            "bounded separation allowance + every actual score/Hessian evaluation",
            "preflight_iteration_allowance": "one candidate and one score per declared Newton step; "
            "additional halving evaluations are reserved dynamically before execution",
        }


def _method_options(method, *, replicates, replicate_weights, centering, rho,
                    justification, scale, rscales, df):
    if not isinstance(method, str) or method not in {"brr", "fay", "jackknife", "bootstrap"}:
        raise AnalysisError(
            "invalid_survey_replicates", "method must be brr, fay, jackknife, or bootstrap."
        )
    if method != "fay" and rho is not None:
        raise AnalysisError("invalid_survey_replicates", "rho is an option only for method='fay'.")
    if method != "bootstrap" and any(v is not None for v in (justification, scale, rscales, df)):
        raise AnalysisError(
            "invalid_survey_replicates", "justification, scale, rscales, and df are bootstrap options."
        )
    if method == "jackknife" and (replicates is not None or replicate_weights is not None):
        raise AnalysisError(
            "invalid_survey_replicates", "Jackknife requires the complete generated delete-one PSU plan."
        )
    if method == "bootstrap" and (replicates is not None or replicate_weights is None):
        raise AnalysisError(
            "invalid_survey_replicates", "Bootstrap requires supplied weights and no generated count."
        )
    if method in {"brr", "fay"} and replicates is not None and replicate_weights is not None:
        raise AnalysisError("invalid_survey_replicates", "Supply weights or a generated count, not both.")
    allowed = {"original", "stratum_mean"} if method == "jackknife" else {"original", "replicate_mean"}
    if not isinstance(centering, str) or centering not in allowed:
        raise AnalysisError("invalid_survey_replicates", f"Invalid centering for {method}.")
    return number(0.5 if rho is None else rho, "Fay rho", 0, 1 - 1e-6) if method == "fay" else 0.0


def _preflight(sample, family, method, max_iter, replicates, replicate_weights, *, target_trace=None):
    """Plan all fits and retained support before a full replica weight allocation."""
    stochastic = [h for h, correction in enumerate(sample.fpc) if correction]
    if method in {"brr", "fay"}:
        if not stochastic or any(len(sample.strata_groups[h]) != 2 for h in stochastic):
            raise AnalysisError("unsupported_survey_brr", "BRR/Fay requires noncensus paired-PSU strata.")
        if any(sample.fpc[h] != 1.0 for h in stochastic):
            raise AnalysisError("unsupported_survey_brr", "BRR/Fay cannot ignore noncensus FPC.")
    if replicate_weights is not None:
        count = _supplied_preflight(sample, replicate_weights)
    elif method == "jackknife":
        count = sum(len(sample.strata_groups[h]) for h in stochastic)
        if not count:
            raise AnalysisError("unsupported_survey_jackknife", "An all-census design needs Taylor inference.")
    else:
        count = 1 << len(stochastic).bit_length() if replicates is None else replicates
        integer(count, "replicates", 2, MAX_REPLICATES)
        if count & (count - 1) or count <= len(stochastic):
            raise AnalysisError(
                "unsupported_survey_brr", "Generated BRR needs a power-of-two order exceeding stochastic strata."
            )
    integer(count, "replicate count", 2, MAX_REPLICATES)
    if method in {"brr", "fay"} and count <= len(stochastic):
        raise AnalysisError("survey_replicate_budget", "Balanced BRR requires more replicates than strata.")
    replica_budget(sample, count, True)
    n, width = len(sample.weights), len(sample.labels)
    overhead, diagnostic, evaluation = _work_budget(sample, family, max_iter)
    # The original support bounds every replica's active fit work. Separation
    # reservations include full bounded LP/replay work even for easy cases.
    evaluations = 1 if family == "linear" else 2 * max_iter + 1
    fit_work = (count + 1) * (overhead + diagnostic + evaluations * evaluation)
    planner_work = n * count * 8 + n * len(stochastic) * 4 + count * width * width * 4
    if method in {"brr", "fay"}:
        planner_work += count * len(stochastic) ** 2
    planned = fit_work + planner_work
    target_workspace = {}
    if target_trace is not None:
        target_work, target_workspace = target_trace.preflight(sample, count)
        planned += target_work
    if planned > MAX_WORK:
        raise AnalysisError(
            "survey_regression_budget",
            "The complete original plus replicate fit plan exceeds 50 million work units before "
            "replica allocation; reduce dimensions, replicate count, or declared iterations.",
        )
    workspace_buffers = {
        "replica weights, factors, admission and hashes": n * count * 192,
        "all replica physical support and serialization": n * count * 96,
        "original and current fit, scores and design": n * (width + 6) * 320,
        "coefficient replicas, convergence and saved replay": count * (width * 192 + 4096),
        "PSU scores, covariance and bounded separation workspace":
            sample.design.validation.n_psu * width * 384 + width**2 * 8192,
    }
    workspace_buffers.update(target_workspace)
    workspace = plan_workspace(
        "complete survey coefficient replication",
        workspace_buffers,
        budget_bytes=(workspace_budget_bytes() if target_trace is None else
                      min(workspace_budget_bytes(), sample.design.max_memory_mb * 1024**2)),
    ).record()
    return count, planner_work, workspace, _WorkLedger(planned)


def _failures_error(failures, *, target=False):
    error = AnalysisError(
        "survey_replicate_failure",
        ("Complete empirical target replication failed; no replica was omitted: " if target else
         "Complete coefficient replication failed; no replica was omitted: ")
        + json.dumps(failures, ensure_ascii=True, allow_nan=False),
    )
    error.failed_replicates = failures
    return error


def _covariance(estimates, original, plan):
    replicas = torch.stack(estimates)
    if plan["centering"] == "original":
        difference = replicas - original
    elif plan["centering"] == "replicate_mean":
        difference = replicas - replicas.mean(0)
    else:
        difference = torch.empty_like(replicas)
        for h in sorted(set(plan["strata"])):
            indices = [r for r, stratum in enumerate(plan["strata"]) if stratum == h]
            difference[indices] = replicas[indices] - replicas[indices].mean(0)
    multipliers = torch.tensor(plan["multipliers"], dtype=FLOAT)
    covariance = finite((difference.T * multipliers) @ difference, "Coefficient replica covariance")
    return replicas, (covariance + covariance.T) * 0.5


@cpu_call
def _fit_replicate(
    data, design, outcome, regressors, *, family, method, intercept, domain, missing,
    alpha, null, max_iter, tolerance, replicates, replicate_weights, centering, rho,
    justification, scale, rscales, df, _target_trace=None,
):
    resolved_rho = _method_options(
        method, replicates=replicates, replicate_weights=replicate_weights,
        centering=centering, rho=rho, justification=justification, scale=scale, rscales=rscales, df=df,
    )
    max_iter, tolerance = _options(max_iter, tolerance)
    sample = admit(data, design, outcome, regressors, intercept=intercept, domain=domain, missing=missing)
    alpha, null = _inference_options(sample, alpha, null)
    count, planner_work, workspace, ledger = _preflight(
        sample, family, method, max_iter, replicates, replicate_weights, target_trace=_target_trace
    )
    ledger.reserve(planner_work)
    if method in {"brr", "fay"}:
        plan = plan_brr(sample, replicates=replicates, replicate_weights=replicate_weights,
                        centering=centering, rho=resolved_rho)
    elif method == "jackknife":
        plan = plan_jackknife(sample, centering=centering)
    else:
        plan = plan_bootstrap(sample, replicate_weights, justification=justification, scale=scale,
                              rscales=rscales, df=df, centering=centering)
    if plan["weights"].shape != (count, len(sample.weights)) or len(plan["ids"]) != count:
        raise AnalysisError("invalid_survey_replicates", "The complete planned replica dimensions changed.")
    if _target_trace is not None:
        _target_trace.prepare(sample, plan, ledger)
    base = fit_sample(sample, family, max_iter=max_iter, tolerance=tolerance, budget=ledger)
    if _target_trace is not None:
        _target_trace.evaluate(sample, base, replicate_id=None, ledger=ledger)
    estimates, convergence, failures = [], [], []
    for r, identity in enumerate(plan["ids"]):
        weights = plan["weights"][r]
        selected = sample.selected & (weights > 0)
        n_used = int(selected.sum())
        replica = replace(sample, weights=weights, selected=selected)
        stage = "fit"
        try:
            fitted = fit_sample(replica, family, max_iter=max_iter, tolerance=tolerance, budget=ledger)
            if _target_trace is not None:
                stage = "target"
                _target_trace.evaluate(replica, fitted, replicate_id=identity, ledger=ledger)
        except AnalysisError as error:
            if error.code in _SAFETY_CODES:
                error.failed_replicates = failures
                error.unattempted_replicate_ids = plan["ids"][r:]
                error.work_record = ledger.record()
                raise
            failure = {"replicate_id": identity, "code": error.code,
                       "reason": str(error), "n_used": n_used}
            if _target_trace is not None:
                failure["stage"] = stage
            failures.append(failure)
            continue
        estimates.append(fitted["beta"])
        convergence.append({**fitted["convergence"], "replicate_id": identity, "n_used": n_used,
                            "sample_positions": torch.where(selected)[0].tolist()})
    if failures:
        error = _failures_error(failures, target=_target_trace is not None)
        error.work_record = ledger.record()
        raise error
    coefficient_replicas, covariance = _covariance(estimates, base["beta"], plan)
    record = {
        **plan["metadata"],
        "method": method, "replicate_count": count, "replicate_ids": plan["ids"],
        "replicate_estimates": coefficient_replicas.tolist(), "replicate_convergence": convergence,
        "variance_multipliers": plan["multipliers"], "centering": plan["centering"],
        "replicate_stratum_indices": plan["strata"], "replicate_df_convention": _DF_CONVENTION,
        "df_explicit": df is not None, "failed_replicates": [],
        "workspace": workspace, "work": ledger.record(),
    }
    if method in {"brr", "fay"}:
        signs = []
        for h, correction in enumerate(sample.fpc):
            if not correction:
                continue
            first = int(torch.where(sample.groups == sample.strata_groups[h][0])[0][0])
            signs.append(torch.where(plan["weights"][:, first] > sample.weights[first], 1.0, -1.0))
        record["balanced_signs"] = torch.stack(signs, dim=1).tolist()
    result = build_result(sample, family, **base, alpha=alpha, null=null,
                          covariance=covariance, df=plan["df"], replication=record)
    if _target_trace is not None:
        return _target_trace.finish(result, plan, workspace, ledger)
    return result


def survey_regress_replicate(
    data, design, outcome, regressors, *, method, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9, replicates=None, replicate_weights=None,
    centering="original", rho=None, justification=None, scale=None, rscales=None, df=None,
) -> SurveyRegressionResult:
    """Weighted linear coefficients with complete BRR/Fay/PSU-jackknife/bootstrap refits.

    Fay defaults to rho=0.5; other methods reject rho. Bootstrap requires
    supplied whole-PSU weights, justification, and variance scale. Every
    admitted replica must identify the same coefficient vector. All failures
    are reported without deleting replicas. Work and resident workspace are
    planned before allocating full replica weights.
    """
    return _fit_replicate(
        data, design, outcome, regressors, family="linear", method=method, intercept=intercept,
        domain=domain, missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
        replicates=replicates, replicate_weights=replicate_weights, centering=centering, rho=rho,
        justification=justification, scale=scale, rscales=rscales, df=df,
    )


def survey_logit_replicate(
    data, design, outcome, regressors, *, method, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9, replicates=None, replicate_weights=None,
    centering="original", rho=None, justification=None, scale=None, rscales=None, df=None,
) -> SurveyRegressionResult:
    """Binary logit coefficients with complete design-weighted coefficient replicas.

    Exact 0/1 responses and a native separation certificate are required for
    the original fit and every replica. No penalty, rank repair, Taylor
    covariance on individual replicas, or row-bootstrap fallback is used.
    Centering, variance scales, retained supports, and all converged refits
    remain in the saved result's replication record.
    """
    return _fit_replicate(
        data, design, outcome, regressors, family="logit", method=method, intercept=intercept,
        domain=domain, missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
        replicates=replicates, replicate_weights=replicate_weights, centering=centering, rho=rho,
        justification=justification, scale=scale, rscales=rscales, df=df,
    )


def survey_probit_replicate(
    data, design, outcome, regressors, *, method, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9, replicates=None, replicate_weights=None,
    centering="original", rho=None, justification=None, scale=None, rscales=None, df=None,
) -> SurveyRegressionResult:
    """Binary probit with complete BRR/Fay/PSU-jackknife/bootstrap coefficient refits.

    Each fit uses the exact observed Bernoulli score sensitivity and requires
    exact 0/1 responses, full rank, a native separation certificate, and
    accepted convergence. Replicas retain the original physical sample and
    coefficient order, with zero-weight rows deselected. The covariance uses
    all refitted coefficient vectors and the declared method's centering and
    multipliers; no individual replica's Taylor covariance is constructed.

    Fay defaults to rho=0.5; other methods reject rho. Bootstrap requires
    supplied whole-PSU weights, justification, and variance scale. Complete
    work and workspace are planned before replica allocation, and failures
    are reported without deleting replicas or returning a partial result.
    """
    return _fit_replicate(
        data, design, outcome, regressors, family="probit", method=method, intercept=intercept,
        domain=domain, missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
        replicates=replicates, replicate_weights=replicate_weights, centering=centering, rho=rho,
        justification=justification, scale=scale, rscales=rscales, df=df,
    )


def survey_poisson_replicate(
    data, design, outcome, regressors, *, method, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9, replicates=None, replicate_weights=None,
    centering="original", rho=None, justification=None, scale=None, rscales=None, df=None,
) -> SurveyRegressionResult:
    """Count Poisson with complete design-weighted coefficient replicas.

    Responses must be exact nonnegative counts at most 2^53. The original and
    every positive-weight replica use the native guarded Poisson score and
    sensitivity, including a bounded general zero-count separation check.
    Overflow, rank loss, separation, and nonconvergence refuse a result; all
    ordinary failed replicas are reported with their IDs and reasons.

    BRR, Fay, stratified delete-one PSU jackknife, and supplied design bootstrap
    preserve the admitted physical sample and coefficient order. Fay defaults
    to rho=0.5. Bootstrap requires whole-PSU weights, declared provenance, and
    variance scale. Saved state retains complete coefficients, supports,
    convergence, centering, multipliers, and cumulative work/workspace proof.
    Offsets and exposures are outside this bounded count API.
    """
    return _fit_replicate(
        data, design, outcome, regressors, family="poisson", method=method, intercept=intercept,
        domain=domain, missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
        replicates=replicates, replicate_weights=replicate_weights, centering=centering, rho=rho,
        justification=justification, scale=scale, rscales=rscales, df=df,
    )
