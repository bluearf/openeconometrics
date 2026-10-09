"""Natural-effect contrasts with a continuous mediator and exposure interaction."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.econometrics.decomposition.mediation import _equation, CAUSAL_ASSUMPTIONS


def fit_mediation_interaction(spec, data):
    frame = ModelFrame(spec, data)
    if len(spec.predictors) != 1:
        raise AnalysisError("invalid_spec", "One exposure is required.")
    mediator = frame.role("mediator")[0]
    controls = frame.role("controls")
    names = [spec.outcome, spec.predictors[0], mediator, *controls]
    if len(names) != len(set(names)):
        raise AnalysisError(
            "overlapping_roles", "Exposure, mediator, outcome and controls must be distinct."
        )
    assumptions = frame.option("assumptions")
    if frame.option("interpretation") == "causal" and not CAUSAL_ASSUMPTIONS <= set(assumptions):
        raise AnalysisError(
            "causal_assumptions_required",
            "Declare all four causal mediation identification assumptions explicitly.",
        )
    a0, a1 = frame.option("treatment0"), frame.option("treatment1")
    if a0 == a1:
        raise AnalysisError("invalid_contrast", "Exposure levels must differ.")
    n = frame.n
    p = len(controls)
    if p > 24 or n * (p + 4) ** 2 > 50_000_000:
        raise AnalysisError(
            "work_budget_exceeded", "Interaction mediation exceeds its resident regression budget."
        )
    frame.workspace_plan(
        "interaction mediation joint scores",
        {"designs": n * (p + 4) * 32, "joint": (2 * p + 6) ** 2 * 32},
    )
    a, m, y = (
        frame.numeric(spec.predictors[0]),
        frame.numeric(mediator),
        frame.numeric(spec.outcome),
    )
    c = frame.matrix(controls)
    xm = torch.cat((torch.ones((n, 1), dtype=torch.float64), a[:, None], c), 1)
    xy = torch.cat(
        (torch.ones((n, 1), dtype=torch.float64), a[:, None], m[:, None], (a * m)[:, None], c), 1
    )
    bm, im, _ = _equation(xm, m, None, "linear")
    by, iy, _ = _equation(xy, y, None, "linear")
    beta = torch.cat((bm, by))
    influence = torch.cat((im, iy), 1)
    if spec.covariance == "cluster":
        codes, g = frame.cluster_dimensions()[0]
        if g < 3:
            raise AnalysisError(
                "insufficient_clusters", "Mediation needs at least three independent clusters."
            )
        sums = influence.new_zeros((g, len(beta))).index_add(0, codes, influence)
        joint = sums.T @ sums * g / (g - 1) * (n - 1) / (n - max(xm.shape[1], xy.shape[1]))
        df = g - 1
    else:
        # Separate equation finite-N factors also scale their cross covariance.
        scale = torch.tensor(
            [
                *([(n / (n - xm.shape[1])) ** 0.5] * len(bm)),
                *([(n / (n - xy.shape[1])) ** 0.5] * len(by)),
            ],
            dtype=torch.float64,
        )
        scaled = influence * scale
        joint = scaled.T @ scaled
        df = n - max(xm.shape[1], xy.shape[1])
    average = c.mean(0)

    def targets(parameters):
        mb, yb = parameters[: len(bm)], parameters[len(bm) :]

        def potential(exposure, mediator_exposure):
            mean_m = mb[0] + mb[1] * mediator_exposure + mb[2:] @ average
            return yb[0] + yb[1] * exposure + (yb[2] + yb[3] * exposure) * mean_m + yb[4:] @ average

        y00, y10, y01, y11 = (
            potential(a0, a0),
            potential(a1, a0),
            potential(a0, a1),
            potential(a1, a1),
        )
        return torch.stack((y10 - y00, y11 - y01, y01 - y00, y11 - y10, y11 - y00))

    with torch.enable_grad():
        derivative = torch.autograd.functional.jacobian(targets, beta)
    values = targets(beta)
    v = derivative @ joint @ derivative.T
    return build_result(
        frame,
        terms=["pure_direct", "total_direct", "pure_indirect", "total_indirect", "total"],
        params=values,
        covariance=v,
        title="Mediation with exposure--mediator interaction",
        use_t=True,
        df_inference=df,
        df_resid=df,
        solver="two QR equations, joint scores and counterfactual delta map",
        inference={
            "covariance": spec.covariance,
            "df_inference": df,
            "target": "natural effects standardized to retained controls; controls treated as fixed",
        },
        extra={
            "interpretation": frame.option("interpretation"),
            "assumptions": assumptions,
            "assumptions_verified": False,
            "treatment0": a0,
            "treatment1": a1,
            "controls_mean": average.tolist(),
            "equation_parameters": beta.tolist(),
            "equation_terms": [
                *[f"M:{name}" for name in ["Intercept", spec.predictors[0], *controls]],
                *[
                    f"Y:{name}"
                    for name in [
                        "Intercept",
                        spec.predictors[0],
                        mediator,
                        f"{spec.predictors[0]}*{mediator}",
                        *controls,
                    ]
                ],
            ],
            "equation_covariance": joint.tolist(),
            "target_jacobian": derivative.tolist(),
            "identities": "total=pure_direct+total_indirect=total_direct+pure_indirect",
        },
    )


def mediation_interaction(
    *,
    data,
    y,
    x,
    mediator,
    controls=None,
    treatment0=0.0,
    treatment1=1.0,
    interpretation="associational",
    assumptions=None,
    cluster=None,
    missing="raise",
    alpha=0.05,
):
    """Five natural-effect contrasts with a linear continuous mediator/outcome.

    Outcome includes exposure*mediator. Full cross-equation HC1 or CR1 scores
    and a joint delta map retain both direct/indirect decompositions. Causal
    interpretation requires explicit assumptions; defaults are associational.
    Unweighted numeric CPU resident inputs; nonlinear outcomes use mediation().
    """
    from openecon.analysis import fit

    return fit(
        make_spec(
            "mediation_interaction",
            outcome=y,
            predictors=column_list(x, "x"),
            columns={"mediator": mediator, "controls": column_list(controls, "controls")},
            covariance="cluster" if cluster is not None else "HC1",
            cluster=cluster,
            missing=missing,
            alpha=alpha,
            options={
                "treatment0": treatment0,
                "treatment1": treatment1,
                "interpretation": interpretation,
                "assumptions": assumptions,
            },
        ),
        data=data,
    )
