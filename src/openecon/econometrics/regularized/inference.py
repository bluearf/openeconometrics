"""Single-treatment, IID partially linear orthogonal inference.

Only the treatment parameter is inferential. Nuisance parameters never get
selected-variable OLS confidence intervals. DML implements the pooled DML2
partialling-out score with honest outer folds and train-only nuisance tuning.
"""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.resources import tensor_bytes
from .kernels import fit_penalized, folds, solver_work, work_guard


def _convenience(
    name,
    data,
    outcome,
    treatment,
    controls,
    *,
    covariance="HC0",
    missing="raise",
    intercept=True,
    alpha=0.05,
    **options,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=outcome,
            predictors=column_list(controls, "controls"),
            columns={"treatment": treatment},
            intercept=intercept,
            covariance=covariance,
            missing=missing,
            alpha=alpha,
            options=options,
        ),
        data=data,
    )


def postdouble(*, data, y, treatment, x, **options):
    """Post-double-selection PLR effect with union of both Lasso selections."""
    return _convenience("postdouble", data, y, treatment, x, **options)


def partiallingout(*, data, y, treatment, x, **options):
    """Orthogonal partialling-out PLR, without cross-fitting (sparsity conditions)."""
    return _convenience("partiallingout", data, y, treatment, x, **options)


def dmlplr(*, data, y, treatment, x, **options):
    """Honest cross-fitted DML2 for a single PLR treatment effect."""
    return _convenience("dmlplr", data, y, treatment, x, **options)


def _infer(spec, data):
    treatment = spec.columns["treatment"]
    if treatment == spec.outcome or treatment in spec.predictors:
        raise AnalysisError(
            "invalid_roles", "Outcome, treatment and controls must be distinct columns."
        )
    frame = ModelFrame(spec, data)
    n, p = frame.n, len(spec.predictors)
    o = {opt.name: frame.option(opt.name) for opt in frame.info.options}
    nuisance = o.pop("nuisance")
    ratio = 1.0 if nuisance == "lasso" else 0.0 if nuisance == "ridge" else o["l1_ratio"]
    if spec.estimator == "postdouble" and nuisance != "lasso":
        raise AnalysisError(
            "unsupported_nuisance", "Post-double-selection uses Lasso selection in both equations."
        )
    if n < 10:
        raise AnalysisError(
            "insufficient_sample",
            "Orthogonal PLR inference needs at least ten complete observations.",
        )
    outer = o["folds"] if spec.estimator == "dmlplr" else 1
    frame.workspace_plan(
        "orthogonal PLR nuisance estimation and influence state",
        {
            "design_and_nuisance_fit_copies": tensor_bytes((n, p + 8), itemsize=128),
            "nuisance_gram_projection_factors": tensor_bytes((p + 1, p + 1), itemsize=96)
            if ratio == 0 or spec.estimator == "postdouble"
            else 0,
            "folds_and_saved_nuisance_state": tensor_bytes((n, outer + 12), itemsize=64),
            "nuisance_paths_and_loadings": tensor_bytes((outer * 200, p + 8), itemsize=64),
        },
    )
    tuning_multiplier = (
        (o["folds"] + 1)
        * (len(o["lambda_path"]) if o["lambda_path"] is not None else o["n_lambdas"])
        if o["selection"] == "cv"
        else o["plugin_iterations"]
        if o["selection"] == "plugin"
        else max(1, len(o["lambda_path"] or []))
    )
    work_guard(
        2 * outer * solver_work(n, p, o["max_iterations"], ratio) * tuning_multiplier,
        o["max_work"],
        "orthogonal PLR nuisance fits",
    )
    x, y, d = frame.matrix(spec.predictors), frame.numeric(spec.outcome), frame.numeric(treatment)
    lhat, mhat = torch.empty_like(y), torch.empty_like(d)
    nuisance_records = []
    selected = None
    rank_controls = int(spec.intercept)
    assignment = folds(n, outer, o["seed"]) if outer > 1 else torch.zeros(n, dtype=torch.int64)
    for fold in range(outer):
        train = torch.ones(n, dtype=torch.bool) if outer == 1 else assignment != fold
        test = torch.ones(n, dtype=torch.bool) if outer == 1 else assignment == fold
        yfit = fit_penalized(
            x[train],
            y[train],
            ratio=ratio,
            intercept=spec.intercept,
            options=o,
            seed=o["seed"] + 2 * fold + 1,
        )
        dfit = fit_penalized(
            x[train],
            d[train],
            ratio=ratio,
            intercept=spec.intercept,
            options=o,
            seed=o["seed"] + 2 * fold + 2,
        )
        lhat[test] = x[test] @ yfit["coefficient"] + yfit["constant"]
        mhat[test] = x[test] @ dfit["coefficient"] + dfit["constant"]
        nuisance_records.append(
            {
                "fold": fold,
                "train_positions": [frame.positions[i] for i in train.nonzero().flatten().tolist()],
                "test_positions": [frame.positions[i] for i in test.nonzero().flatten().tolist()],
                "outcome_model": yfit["state"],
                "treatment_model": dfit["state"],
            }
        )
        if spec.estimator == "postdouble":
            selected_y = yfit["coefficient"] != 0
            selected_d = dfit["coefficient"] != 0
            selected = (selected_y | selected_d).nonzero().flatten().tolist()
            control_design = x[:, selected]
            if spec.intercept:
                control_design = torch.cat(
                    (torch.ones(n, 1, dtype=torch.float64), control_design), dim=1
                )
            rank_controls = (
                int(torch.linalg.matrix_rank(control_design)) if control_design.shape[1] else 0
            )
            if rank_controls >= n - 1:
                raise AnalysisError(
                    "saturated_selection",
                    "Double-selection controls leave no residual degrees of freedom for the treatment effect.",
                )
            if rank_controls:
                projection = torch.linalg.lstsq(
                    control_design, torch.stack((y, d), dim=1), driver="gelsd"
                ).solution
                lhat, mhat = (control_design @ projection).unbind(1)
            else:
                lhat, mhat = torch.zeros_like(y), torch.zeros_like(d)
            nuisance_records[-1].update(
                {
                    "selected_outcome": [
                        spec.predictors[i] for i in selected_y.nonzero().flatten().tolist()
                    ],
                    "selected_treatment": [
                        spec.predictors[i] for i in selected_d.nonzero().flatten().tolist()
                    ],
                    "selected_union": [spec.predictors[i] for i in selected],
                    "projection_rank": rank_controls,
                }
            )
    residual_y, residual_d = y - lhat, d - mhat
    jacobian = residual_d.square().mean()
    threshold = 1e-12 * max(float(d.square().mean()), 1.0)
    if float(jacobian) <= threshold:
        raise AnalysisError(
            "unidentified_treatment", "Treatment has insufficient variation after nuisance removal."
        )
    theta = torch.dot(residual_d, residual_y) / residual_d.square().sum()
    residual = residual_y - theta * residual_d
    score = residual_d * residual
    influence = score / jacobian
    dof = n - rank_controls - 1 if spec.estimator == "postdouble" else n - 1
    correction = n / dof if spec.covariance == "HC1" else 1.0
    variance = influence.square().mean() / n * correction
    if not bool(torch.isfinite(variance)) or float(variance) <= 0:
        raise AnalysisError(
            "invalid_covariance", "The estimated orthogonal score has no positive finite variance."
        )
    # Sensitivity summaries are diagnostics, not a finite-sample proof of the
    # nuisance convergence and approximate-sparsity assumptions.
    sensitivity_l = -(x * residual_d[:, None]).mean(0)
    sensitivity_m = (x * (theta * residual_d - residual)[:, None]).mean(0)
    assumptions = (
        "IID observations and finite score moments; E[u|D,X]=0, positive E[(D-E[D|X])^2]. "
        "DML needs L2 nuisance consistency and product error o(n^-1/2); in-sample partialling-out/double-selection additionally require approximate sparsity and valid Lasso selection rates. "
        "No causal interpretation without the PLR identification assumptions."
    )
    warning = "Nuisance and identification assumptions are required for orthogonal asymptotic inference; numerical score diagnostics do not certify them."
    if spec.estimator != "dmlplr" and o["selection"] == "cv":
        frame.warn(
            "CV penalties optimize prediction; uniform sparse post-selection inference is not guaranteed by CV alone."
        )
    return build_result(
        frame,
        terms=[treatment],
        params=theta.reshape(1),
        covariance=variance.reshape(1, 1),
        use_t=False,
        fitted=lhat + theta * residual_d,
        metrics={
            "orthogonal_residual_mse": float(residual.square().mean()),
            "treatment_residual_second_moment": float(jacobian),
        },
        solver="orthogonal PLR residual score; DML2 pooled cross-fit"
        if outer > 1
        else "orthogonal PLR residual score",
        solver_diagnostics={"score_mean": float(score.mean()), "nuisance_models_converged": True},
        inference={
            "target": "PLR treatment effect",
            "covariance": spec.covariance,
            "correction": "estimated-nuisance orthogonal influence sandwich: mean(IF^2)/N",
            "small_sample_correction": correction,
            "df_resid": dof,
            "cross_fitted": outer > 1,
            "nuisance_uncertainty": "orthogonal score with estimated nuisance residuals; first-order nuisance influence eliminated under stated rate conditions",
        },
        extra={
            "target": "PLR treatment effect",
            "method": spec.estimator,
            "score_definition": "(D-mhat(X))*(Y-lhat(X)-theta*(D-mhat(X)))",
            "jacobian": float(jacobian),
            "score": score.tolist(),
            "influence": influence.tolist(),
            "nuisance_outcome_predictions": lhat.tolist(),
            "nuisance_treatment_predictions": mhat.tolist(),
            "fold_assignments": assignment.tolist(),
            "folds": outer,
            "seed": o["seed"],
            "fold_records": nuisance_records,
            "cross_fitted": outer > 1,
            "selected_controls": None
            if selected is None
            else [spec.predictors[i] for i in selected],
            "nuisance_score_sensitivity": {
                "outcome_max": float(sensitivity_l.abs().max()),
                "treatment_max": float(sensitivity_m.abs().max()),
            },
            "notes": [
                assumptions,
                "Only the treatment effect is an inferential target; nuisance coefficients have no reported confidence intervals.",
            ],
        },
        warnings=[warning],
    )


def fit_postdouble(spec, data):
    return _infer(spec, data)


def fit_partiallingout(spec, data):
    return _infer(spec, data)


def fit_dmlplr(spec, data):
    return _infer(spec, data)
