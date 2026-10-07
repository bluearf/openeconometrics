"""Observed-variable linear paths and Gaussian-mediator natural-effect targets."""

from __future__ import annotations

import math
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import ls
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec
from openecon.engines.covariance import meat_cluster
from openecon.engines.linalg import cholesky_inverse

CAUSAL_ASSUMPTIONS = {
    "consistency",
    "positivity",
    "sequential_ignorability",
    "no_exposure_induced_mediator_outcome_confounding",
}


def _design(a, mediators, controls, w, stage, serial, moderate):
    n = len(a)
    pieces = [torch.ones((n, 1), dtype=torch.float64), a[:, None]]
    indices = {"a": 1, "previous": [], "b": [], "aw": None, "bw": []}
    if stage == "outcome":
        for j in range(mediators.shape[1]):
            indices["b"].append(len(pieces))
            pieces.append(mediators[:, j : j + 1])
    elif serial:
        for j in range(stage):
            indices["previous"].append(len(pieces))
            pieces.append(mediators[:, j : j + 1])
    pieces.extend(controls[:, j : j + 1] for j in range(controls.shape[1]))
    if w is not None:
        pieces.append(w[:, None])
        if (stage == "outcome" and "direct" in moderate) or (
            stage != "outcome" and "a" in moderate
        ):
            indices["aw"] = len(pieces)
            pieces.append((a * w)[:, None])
        if stage == "outcome" and "b" in moderate:
            for j in range(mediators.shape[1]):
                indices["bw"].append(len(pieces))
                pieces.append((mediators[:, j] * w)[:, None])
    return torch.cat(pieces, dim=1), indices


def _equation(x, y, weight, kind):
    if kind == "linear":
        if weight is None:
            fit = ls(x, y)
        else:
            # Rank and residual uncertainty use the same weighted design.
            fit = ls(x * weight.sqrt()[:, None], y * weight.sqrt())
        residual = y - x @ fit.beta
        influence = (x * (residual * (weight if weight is not None else 1))[:, None]) @ fit.xtx_inv
        return fit.beta, influence, residual
    from openecon.analysis import fit

    columns = [f"c{i}" for i in range(1, x.shape[1])]
    model = fit(
        make_spec(kind, outcome="out", predictors=columns, intercept=True, covariance="nonrobust"),
        data={"out": y.numpy(), **{name: x[:, i + 1].numpy() for i, name in enumerate(columns)}},
    )
    beta = torch.tensor([c.estimate for c in model.coefficients], dtype=torch.float64)
    bread = torch.tensor(model.covariance_matrix, dtype=torch.float64)
    eta = x @ beta
    if kind == "logit":
        residual = y - torch.sigmoid(eta)
        score = residual
    else:
        residual = y - torch.special.ndtr(eta)
        logphi = -0.5 * eta.square() - 0.5 * math.log(2 * math.pi)
        score = torch.where(
            y > 0,
            torch.exp(logphi - torch.special.log_ndtr(eta)),
            -torch.exp(logphi - torch.special.log_ndtr(-eta)),
        )
    return beta, (x * score[:, None]) @ bread, residual


def fit_mediation(spec, data):
    frame = ModelFrame(spec, data)
    if len(spec.predictors) != 1:
        raise AnalysisError("invalid_spec", "Mediation needs one exposure column in x.")
    names, moderator = frame.role("mediators"), frame.role("moderator")
    if not 1 <= len(names) <= 8:
        raise AnalysisError("invalid_spec", "Use 1..8 observed continuous mediators.")
    serial, kind = frame.option("serial"), frame.option("outcome_model")
    if kind != "linear" and (serial or spec.weights):
        raise AnalysisError(
            "unsupported_target",
            "Nonlinear natural effects support unweighted parallel continuous Gaussian mediators; serial or weighted nonlinear targets are not silently approximated.",
        )
    interpretation, assumptions = frame.option("interpretation"), frame.option("assumptions")
    if interpretation == "causal" and not CAUSAL_ASSUMPTIONS.issubset(assumptions):
        raise AnalysisError(
            "causal_assumptions_required",
            "Causal interpretation requires explicit consistency, positivity, sequential_ignorability and no_exposure_induced_mediator_outcome_confounding declarations.",
        )
    moderate = frame.option("moderate")
    if set(moderate) - {"a", "b", "direct"}:
        raise AnalysisError("invalid_option", "moderate can contain a, b and direct.")
    a0, a1 = frame.option("treatment0"), frame.option("treatment1")
    if a0 == a1:
        raise AnalysisError("invalid_contrast", "The exposure contrast must be nonzero.")
    a, m = frame.numeric(spec.predictors[0]), frame.matrix(names)
    controls, y = frame.matrix(frame.role("controls")), frame.numeric(spec.outcome)
    if kind == "linear" and len(torch.unique(y)) == 2:
        frame.warn(
            "A binary outcome under the linear model targets a linear probability path; select outcome_model='logit' for integrated probability effects."
        )
    w = frame.numeric(moderator[0]) if moderator else None
    at = frame.option("at")
    if w is None and at is not None:
        raise AnalysisError("invalid_option", "at requires a moderator column.")
    at = at if at is not None else [float(w.mean()) if w is not None else 0.0]
    if not at or len(at) > 20:
        raise AnalysisError("invalid_option", "Use 1..20 moderator evaluation values.")
    weight = frame.weights()
    designs, indices, betas, influences, residuals, slices = [], [], [], [], [], []
    offset = 0
    for stage in [*range(len(names)), "outcome"]:
        x, ix = _design(a, m, controls, w, stage, serial, moderate)
        beta, impact, resid = _equation(
            x,
            y if stage == "outcome" else m[:, stage],
            weight,
            kind if stage == "outcome" else "linear",
        )
        if spec.covariance == "HC1":
            impact *= (frame.n / (frame.n - x.shape[1])) ** 0.5
        designs.append(x)
        indices.append(ix)
        betas.append(beta)
        influences.append(impact)
        residuals.append(resid)
        slices.append(slice(offset, offset + len(beta)))
        offset += len(beta)
    paths = []
    if serial:
        for j in range(len(names)):
            sequences = [[j]]
            for earlier in paths:
                if earlier[-1] < j:
                    sequences.append([*earlier, j])
            paths.extend(sequences)
    else:
        paths = [[j] for j in range(len(names))]
    parameters = torch.cat(betas)
    extra = {
        "interpretation": interpretation,
        "assumptions": assumptions,
        "causal_assumptions_checked": False,
        "assumption_note": "Declarations do not establish unmeasured-confounding assumptions from data.",
        "mediators": names,
        "serial": serial,
        "moderator": moderator[0] if moderator else None,
        "at": at,
        "moderate": moderate if moderator else [],
        "exposure_contrast": [a0, a1],
        "outcome_model": kind,
        "raw_equation_coefficients": [b.tolist() for b in betas],
        "joint_covariance": "joint per-observation equation influence; all cross-equation blocks retained",
    }
    if kind != "linear":
        residual = torch.stack(residuals[:-1], dim=1)
        sigma = residual.T @ residual / frame.n
        kernel_call(cholesky_inverse, sigma, what="joint mediator residual covariance")
        vech = [(i, j) for i in range(len(names)) for j in range(i + 1)]
        parameters = torch.cat((parameters, torch.stack([sigma[i, j] for i, j in vech])))
        influences.append(
            torch.stack(
                [(residual[:, i] * residual[:, j] - sigma[i, j]) / frame.n for i, j in vech], dim=1
            )
            * (frame.n / (frame.n - 1)) ** 0.5
        )
        draws = frame.option("integration_draws")
        if kind == "logit" and draws % 2:
            raise AnalysisError(
                "invalid_option",
                "integration_draws must be even for the declared antithetic normal protocol.",
            )
        frame.workspace_plan(
            "nonlinear mediation integration and Jacobian",
            {
                "normal_draws_and_counterfactuals": 64
                * (draws if kind == "logit" else 1)
                * frame.n
                * len(names),
                "joint_covariance": 64 * len(parameters) ** 2,
            },
        )
        noise = None
        if kind == "logit":
            generator = torch.Generator().manual_seed(frame.option("seed"))
            noise = torch.randn(
                (draws // 2, frame.n, len(names)), generator=generator, dtype=torch.float64
            )
            noise = torch.cat((noise, -noise))

        def effects(theta, value):
            coeffs = [theta[s] for s in slices]
            covariance = torch.zeros_like(sigma)
            for index, (i, j) in enumerate(vech):
                covariance[i, j] = covariance[j, i] = theta[offset + index]
            errors = (
                noise @ torch.linalg.cholesky(covariance).T
                if kind == "logit"
                else torch.zeros((1, frame.n, len(names)), dtype=torch.float64)
            )
            wc = torch.full_like(a, value) if w is not None else None
            means = []
            for exposure in (a0, a1):
                ac = torch.full_like(a, exposure)
                means.append(
                    torch.stack(
                        [
                            _design(ac, m, controls, wc, j, False, moderate)[0] @ coeffs[j]
                            for j in range(len(names))
                        ],
                        dim=1,
                    )[None, :, :]
                    + errors
                )

            def probability(exposure, mediator):
                ac = torch.full_like(a, exposure)
                values = []
                for sample in mediator:
                    design, _ = _design(ac, sample, controls, wc, "outcome", False, moderate)
                    if kind == "logit":
                        values.append(torch.sigmoid(design @ coeffs[-1]).mean())
                    else:
                        iy = indices[-1]
                        bm = coeffs[-1][iy["b"]]
                        if iy["bw"]:
                            bm = bm + value * coeffs[-1][iy["bw"]]
                        scale = (1 + bm @ covariance @ bm).sqrt()
                        values.append(torch.special.ndtr((design @ coeffs[-1]) / scale).mean())
                return torch.stack(values).mean()

            base = probability(a0, means[0])
            counterfactual = probability(a1, means[0])
            total = probability(a1, means[1]) - base
            direct = counterfactual - base
            components = []
            switched = means[0]
            previous = counterfactual
            for j in range(len(names)):
                switched = torch.cat((means[1][:, :, : j + 1], means[0][:, :, j + 1 :]), dim=2)
                current = probability(a1, switched)
                components.append(current - previous)
                previous = current
            return torch.stack([direct, torch.stack(components).sum(), total, *components])

        paths = [[j] for j in range(len(names))]
        extra.update(
            target="population average natural effects on probability scale; order-dependent telescoping mediator-switch components",
            integration_draws=draws if kind == "logit" else 0,
            integration_seed=frame.option("seed") if kind == "logit" else None,
            integration_method="antithetic joint-Gaussian Monte Carlo"
            if kind == "logit"
            else "analytic Gaussian-probit mixture",
            mediator_distribution="joint Gaussian residuals; fitted covariance retained as nuisance parameters",
        )
    else:

        def effects(theta, value):
            coeffs = [theta[s] for s in slices]
            iy = indices[-1]
            direct = coeffs[-1][iy["a"]] + (
                value * coeffs[-1][iy["aw"]] if iy["aw"] is not None else 0
            )
            components = []
            for path in paths:
                j = path[0]
                aindex = indices[j]
                effect = coeffs[j][aindex["a"]] + (
                    value * coeffs[j][aindex["aw"]] if aindex["aw"] is not None else 0
                )
                for before, after in zip(path[:-1], path[1:]):
                    effect = effect * coeffs[after][indices[after]["previous"][before]]
                last = path[-1]
                effect = effect * (
                    coeffs[-1][iy["b"][last]]
                    + (value * coeffs[-1][iy["bw"][last]] if iy["bw"] else 0)
                )
                components.append(effect * (a1 - a0))
            indirect = torch.stack(components).sum()
            direct = direct * (a1 - a0)
            return torch.stack([direct, indirect, direct + indirect, *components])

        extra["target"] = "conditional linear direct/path/total effects in outcome units"
    influence = torch.cat(influences, dim=1)
    if spec.covariance == "cluster":
        groups, count = frame.codes(spec.cluster)
        if count < 2:
            raise AnalysisError(
                "insufficient_clusters", "Cluster mediation requires at least two groups."
            )
        correction = (
            count / (count - 1) * (frame.n - 1) / (frame.n - max(x.shape[1] for x in designs))
        )
        joint = kernel_call(meat_cluster, influence, groups, count) * correction
        use_t, df = True, count - 1
    else:
        joint = influence.T @ influence
        use_t, df = False, None

    def targets(theta):
        return torch.cat([effects(theta, value) for value in at])

    estimates = targets(parameters)
    jac = torch.autograd.functional.jacobian(targets, parameters)
    covariance = jac @ joint @ jac.T
    labels = [
        "direct",
        "indirect",
        "total",
        *["path:" + ">".join(names[j] for j in path) for path in paths],
    ]
    terms = [f"at={value}:{label}" if moderator else label for value in at for label in labels]
    extra.update(
        raw_parameter_vector=parameters.tolist(),
        raw_parameter_covariance=joint.tolist(),
        raw_equation_slices=[[s.start, s.stop] for s in slices],
        paths=paths,
    )
    return build_result(
        frame,
        terms=terms,
        params=estimates,
        covariance=covariance,
        use_t=use_t,
        df_inference=df,
        categories={},
        solver="float64_equations_joint_influence_delta",
        extra=extra,
        inference={
            "correction": "joint equation delta method; mediator variance nuisance included for nonlinear targets"
        },
    )


def mediation(
    *,
    data,
    y,
    x,
    mediators,
    controls=None,
    moderator=None,
    serial=False,
    at=None,
    moderate=None,
    treatment0=0.0,
    treatment1=1.0,
    interpretation="associational",
    assumptions=None,
    outcome_model="linear",
    integration_draws=128,
    seed=0,
    covariance=None,
    cluster=None,
    weights=None,
    missing="raise",
    alpha=0.05,
):
    """Observed-variable mediation; causal interpretation requires explicit assumptions."""
    from openecon.analysis import fit

    return fit(
        make_spec(
            "mediation",
            outcome=y,
            predictors=column_list(x, "x"),
            columns={
                "mediators": column_list(mediators, "mediators"),
                "controls": column_list(controls, "controls"),
                "moderator": moderator,
            },
            options={
                "serial": serial,
                "at": at,
                "moderate": moderate,
                "treatment0": treatment0,
                "treatment1": treatment1,
                "interpretation": interpretation,
                "assumptions": assumptions,
                "outcome_model": outcome_model,
                "integration_draws": integration_draws,
                "seed": seed,
            },
            covariance=covariance,
            cluster=cluster,
            weights=weights,
            weight_type="aweight",
            missing=missing,
            alpha=alpha,
        ),
        data=data,
    )
