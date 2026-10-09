"""Expanding OLS by updated augmented QR rather than repeated full-sample QR."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, build_result, table
from openecon.econometrics.tsworkflows.common import ordered
from openecon.econometrics.tsworkflows.workflows import _integer
from openecon.models import ModelSpec


def recursive_ols(spec, *, data, minimum, step=1, max_fits=100, on_error="raise"):
    """Forward expanding numeric iid OLS using an augmented [X,y] QR update.

    Reuses R plus only newly admitted rows. IID covariance is RSS/(N-K)*inv(X'X),
    with t(N-K) inference and complete per-origin ResultBundles. Rank or perfect
    fits raise, or remain explicit failed windows. Resident float64 CPU inputs;
    weights, categorical terms and robust/cluster covariance use recursive().
    """
    if (
        not isinstance(spec, ModelSpec)
        or spec.estimator != "ols"
        or spec.weights
        or spec.categorical
        or spec.panel
        or spec.columns
        or spec.covariance != "nonrobust"
    ):
        raise AnalysisError(
            "unsupported_spec",
            "recursive_ols supports unweighted numeric single-series iid OLS specs.",
        )
    if on_error not in {"raise", "record"}:
        raise AnalysisError("invalid_option", "on_error must be raise or record.")
    frame = ModelFrame(spec, data)
    ordered(frame)
    minimum = _integer(minimum, "minimum", 2, frame.n)
    step = _integer(step, "step", 1, frame.n)
    max_fits = _integer(max_fits, "max_fits", 1, 1000)
    stops = list(range(minimum, frame.n + 1, step))
    if len(stops) > max_fits:
        raise AnalysisError("search_budget", "recursive_ols exceeds max_fits.")
    design = frame.design()
    k = len(design.terms)
    if k > 64 or frame.n * k * k > 100_000_000 or len(stops) * (k + 1) ** 3 > 100_000_000:
        raise AnalysisError(
            "work_budget_exceeded", "Recursive QR exceeds its width/operation bounds."
        )
    frame.workspace_plan(
        "recursive augmented QR",
        {"joint": frame.n * (k + 1) * 16, "records": len(stops) * k * k * 24},
    )
    joint = torch.cat((design.x, frame.numeric(spec.outcome)[:, None]), 1)
    r = None
    admitted = 0
    history, rows = [], []
    for stop in stops:
        combined = joint[admitted:stop] if r is None else torch.cat((r, joint[admitted:stop]), 0)
        _, r = torch.linalg.qr(combined, mode="reduced")
        admitted = stop
        record = {
            "origin": stop - 1,
            "start": 0,
            "stop_exclusive": stop,
            "input_positions": frame.positions[:stop],
        }
        try:
            if stop <= k or r.shape[0] <= k:
                raise AnalysisError(
                    "insufficient_observations",
                    "Recursive OLS leaves no residual degrees of freedom.",
                )
            rx = r[:k, :k]
            scaled = rx / torch.linalg.vector_norm(design.x[:stop], dim=0).clamp_min(
                torch.finfo(torch.float64).tiny
            )
            if int(torch.linalg.matrix_rank(scaled)) != k:
                raise AnalysisError(
                    "singular_design", "Recursive OLS design lacks full rank at this origin."
                )
            beta = torch.linalg.solve_triangular(rx, r[:k, k, None], upper=True).flatten()
            rss = float(r[k, k].square())
            total = float(joint[:stop, k].square().sum())
            if rss <= 1e-24 * max(total, 1):
                raise AnalysisError(
                    "perfect_fit", "Recursive OLS needs estimable residual variation."
                )
            ri = torch.linalg.solve_triangular(rx, torch.eye(k, dtype=torch.float64), upper=True)
            v = ri @ ri.T * rss / (stop - k)
            part = ModelFrame(spec, frame.sample.iloc[:stop].copy())
            result = build_result(
                part,
                terms=design.terms,
                params=beta,
                covariance=v,
                title="Recursive iid OLS",
                use_t=True,
                df_inference=stop - k,
                df_resid=stop - k,
                metrics={"rss": rss, "rmse": (rss / (stop - k)) ** 0.5, "df_resid": stop - k},
                solver="incremental augmented Householder QR",
                inference={
                    "covariance": "nonrobust",
                    "df_inference": stop - k,
                    "correction": "RSS/(N-K)",
                },
                solver_diagnostics={
                    "qr_rows": len(combined),
                    "new_rows": step if stop != minimum else minimum,
                },
            )
            record.update(
                status="ok",
                result_id=result.id,
                nobs=stop,
                sample_positions=frame.positions[:stop],
                coefficients=[c.model_dump(mode="json") for c in result.coefficients],
                covariance_matrix=result.covariance_matrix,
                inference=result.inference,
                metrics=result.metrics,
                result=result.model_dump(mode="json"),
            )
            rows.extend(
                {
                    "origin": stop - 1,
                    "term": c.term,
                    "estimate": c.estimate,
                    "std_error": c.std_error,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                }
                for c in result.coefficients
            )
        except AnalysisError as exc:
            if on_error == "raise":
                raise
            record.update(status="failed", error_code=exc.code, error=str(exc))
        history.append(record)
    return TableSet(
        {"coefficients": table(rows)},
        title="Incremental recursive OLS",
        spec=spec.model_dump(mode="json"),
        origins=history,
        minimum=minimum,
        step=step,
        sample_positions=frame.positions,
        lookahead="Only rows admitted through each origin enter augmented QR.",
        uncertainty="iid t(N-K), no robust or selection uncertainty",
    )
