"""Hainmueller entropy balancing, author CEM ATT weights and balance diagnostics.

References: https://web.mit.edu/~jhainm/www/Paper/eb.pdf ;
https://gking.harvard.edu/cem/ ; https://doi.org/10.1002/sim.3697 .
Weights balance observed pre-treatment features; balance alone is not causal identification.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.resources import plan_workspace
from . import common as m


def _columns(x, treatment, other=()):
    names = c.name_list(x, "x", minimum=1)
    if len(names) > 16 or treatment in names or set(names) & set(other):
        raise AnalysisError(
            "invalid_roles",
            "Select 1..16 distinct pre-treatment covariate columns and separate treatment/weight roles.",
        )
    return names


def _matrix(selected, names):
    return torch.stack([m.tensor(selected, name) for name in names], dim=1)


def _positive(selected, name):
    raw = (
        torch.ones(len(selected), dtype=torch.float64, device="cpu")
        if name is None
        else m.tensor(selected, name)
    )
    if bool((raw <= 0).any()):
        raise AnalysisError(
            "invalid_weights",
            "Base entropy weights must be strictly positive; no zero exclusion or clipping is performed.",
        )
    return raw


@m.procedure
def ebalance(
    data,
    treatment: str,
    x: list[str],
    *,
    base_weights: str | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Entropy-balance controls to the treated population's prespecified numeric moments.

    KL-minimizes normalized donor weights relative to positive base_weights;
    x may include prespecified powers/interactions. All control weights are
    strictly positive and both reported arm sums equal the treated sample size.
    A full-rank strict-interior solution is required. No trimming, clipping,
    effect coefficient or estimated-weight standard error is implied.
    """
    m.options(device, weights, max_work)
    names = _columns(x, treatment, [base_weights] if base_weights else [])
    if base_weights == treatment:
        raise AnalysisError("invalid_roles", "Base weight and treatment roles must differ.")
    c.check_number(tolerance, "tolerance", minimum=1e-12, maximum=1e-3)
    c.check_count(max_iterations, "max_iterations", maximum=10000)
    columns = [treatment, *names, *([base_weights] if base_weights else [])]
    selected, metadata = m.sample(
        data,
        columns,
        numeric=columns,
        missing=missing,
        max_work=max_work,
        cost=max_iterations * len(names) ** 2 * 36,
    )
    t, xx, base = (
        m.binary(selected, treatment),
        _matrix(selected, names),
        _positive(selected, base_weights),
    )
    treated, controls = t == 1, t == 0
    nt, nc, p = int(treated.sum()), int(controls.sum()), len(names)
    if nt < 2 or nc <= p:
        raise AnalysisError(
            "insufficient_support",
            "Entropy balancing requires two treated rows and more controls than moment columns.",
        )
    plan_workspace(
        "entropy dual and complete weight state",
        {
            "standardized_moments": 64 * len(selected) * p,
            "hessian_and_factorizations": 128 * p * p,
            "weights_and_diagnostics": 128 * len(selected),
        },
    )
    # Stable normalization retains global base-weight scaling invariance.
    qt = base[treated] / base[treated].max()
    qt = qt / qt.sum()
    qc = base[controls] / base[controls].max()
    qc = qc / qc.sum()
    if not bool(torch.isfinite(qt).all() & torch.isfinite(qc).all()) or bool(
        (qt <= 0).any() | (qc <= 0).any()
    ):
        raise AnalysisError(
            "numerical_failure",
            "Positive base-weight normalization underflowed; no target/donor unit is silently excluded.",
        )
    target = (qt[:, None] * xx[treated]).sum(0)
    donor = xx[controls]
    center = (qc[:, None] * donor).sum(0)
    scale = (qc[:, None] * (donor - center).square()).sum(0).sqrt()
    if bool((scale <= 0).any()):
        raise AnalysisError(
            "rank_deficient_constraints",
            "Entropy constraints must have positive donor variability.",
        )
    z = (donor - target) / scale
    augmented = torch.cat((torch.ones((nc, 1), dtype=torch.float64, device="cpu"), z), dim=1)
    if int(torch.linalg.matrix_rank(augmented)) != p + 1:
        raise AnalysisError(
            "rank_deficient_constraints",
            "Moment columns are redundant on controls; declare independent constraints.",
        )
    if bool(((target <= donor.min(0).values) | (target >= donor.max(0).values)).any()):
        raise AnalysisError(
            "infeasible_balance",
            "A strictly positive solution needs treated moments strictly inside donor ranges.",
        )
    lam = torch.zeros(p, dtype=torch.float64, device="cpu")
    logq = qc.log()
    trace = []

    def evaluate(value):
        logits = logq + z @ value
        objective = torch.logsumexp(logits, 0)
        ww = logits.softmax(0)
        gradient = (ww[:, None] * z).sum(0)
        centered = z - gradient
        hessian = centered.T @ (ww[:, None] * centered)
        return objective, ww, gradient, hessian

    converged = False
    for iteration in range(max_iterations):
        objective, ww, gradient, hessian = evaluate(lam)
        residual = float(gradient.abs().max())
        trace.append([iteration, float(objective), residual, float(ww.min()), float(ww.max())])
        if residual <= tolerance:
            converged = True
            break
        if not bool(torch.isfinite(hessian).all()) or int(torch.linalg.matrix_rank(hessian)) != p:
            raise AnalysisError(
                "infeasible_balance",
                "Entropy dual lost positive-definite support before convergence.",
            )
        direction = torch.linalg.solve(hessian, gradient)
        decrease = float(gradient @ direction)
        step = 1.0
        accepted = False
        for _ in range(32):
            proposal = lam - step * direction
            value = evaluate(proposal)[0]
            if torch.isfinite(value) and float(value) <= float(objective) - 1e-4 * step * decrease:
                lam = proposal
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise AnalysisError(
                "balance_nonconvergence",
                "Entropy line search failed; no approximate weights are returned.",
            )
    if not converged or not bool(torch.isfinite(ww).all()) or bool((ww <= 0).any()):
        raise AnalysisError(
            "balance_nonconvergence",
            "Entropy balancing did not reach the declared tolerance with strictly positive weights.",
        )
    attained = (ww[:, None] * donor).sum(0)
    final_residual = (attained - target) / scale
    if float(final_residual.abs().max()) > tolerance * 1.05:
        raise AnalysisError(
            "balance_nonconvergence", "Reported moments fail the declared normalized tolerance."
        )
    # A small dual gradient alone can accept a boundary/outside-hull target.
    # Certify a numerically interior primal point by correcting its equality
    # residual and checking a declared positive probability margin. The entropy
    # weights remain unchanged; this point is a separate feasibility certificate.
    aa = augmented.T
    rhs = torch.zeros(p + 1, dtype=torch.float64, device="cpu")
    rhs[0] = 1
    certificate = ww + aa.T @ torch.linalg.solve(aa @ aa.T, rhs - aa @ ww)
    margin = 128 * torch.finfo(torch.float64).eps * (p + 1)
    certificate_tolerance = (
        64 * torch.finfo(torch.float64).eps * (p + 1) * max(1.0, float(aa.abs().max()))
    )
    certificate_error = float((aa @ certificate - rhs).abs().max())
    if (
        not bool(torch.isfinite(certificate).all())
        or float(certificate.min()) <= margin
        or certificate_error > certificate_tolerance
    ):
        raise AnalysisError(
            "infeasible_balance",
            "The target has no verified primal interior point at the declared numeric probability margin; boundary/weak-support weights are refused.",
        )
    report_weights = torch.empty(len(selected), dtype=torch.float64, device="cpu")
    report_weights[treated], report_weights[controls] = qt * nt, ww * nt
    rows = [
        [pos, int(t[i]), float(report_weights[i])] for i, pos in enumerate(metadata["positions"])
    ]
    tables = {
        "weights": m.frame(rows, columns=["position", "treatment", "weight"]),
        "moments": m.frame(
            [
                [names[j], float(target[j]), float(attained[j]), float(final_residual[j])]
                for j in range(p)
            ],
            columns=["moment", "target", "attained", "scaled_residual"],
        ),
        "convergence": m.frame(
            trace,
            columns=[
                "iteration",
                "dual_objective",
                "max_scaled_residual",
                "min_control_weight",
                "max_control_weight",
            ],
        ),
    }
    state = dict(
        target_population="observed treated, optionally positive base-weighted; ATT balancing only",
        treatment=t.tolist(),
        moments=xx.tolist(),
        base_weights=base.tolist(),
        weights=report_weights.tolist(),
        treated_target=target.tolist(),
        dual=lam.tolist(),
        control_probability_weights=ww.tolist(),
        treated_probability_weights=qt.tolist(),
        primal_interior_certificate=dict(
            probabilities=certificate.tolist(),
            minimum_probability_margin=margin,
            equality_residual=certificate_error,
            equality_tolerance=certificate_tolerance,
        ),
        covariance=None,
        inference="not applicable: balancing weights, not an effect estimator",
        convergence=dict(
            converged=True,
            iterations=len(trace),
            trace=trace,
            max_scaled_residual=float(final_residual.abs().max()),
        ),
        control_ess=float(1 / ww.square().sum()),
        treated_ess=float(1 / qt.square().sum()),
        assumptions="prespecified pre-treatment moments and positive donor support; observed balance does not verify unconfoundedness",
    )
    return m.result(
        "ebalance",
        tables,
        metadata,
        dict(
            treatment=treatment,
            x=names,
            base_weights=base_weights,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        state,
        notes=[state["inference"], state["assumptions"]],
    )


@m.procedure
def cem(
    data,
    treatment: str,
    x: list[str],
    *,
    cutpoints: dict,
    categorical: list[str] | None = None,
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Prespecified coarsened exact matching with retained-treated ATT weights.

    Numeric x use explicit sorted cuts (x==cut enters the higher bin); declared
    categorical columns match exactly. Only strata with both arms are retained.
    Treated weights=1 and donor weights=n_t(stratum)/n_c(stratum); unmatched=0.
    The target is retained treated rows, not automatically the original ATT.
    """
    m.options(device, weights, max_work)
    names = c.name_list(x, "x", minimum=0)
    cats = [] if categorical is None else c.name_list(categorical, "categorical", minimum=0)
    allx = _columns([*names, *cats], treatment)
    if not isinstance(cutpoints, dict) or set(cutpoints) != set(names):
        raise AnalysisError(
            "invalid_cutpoints", "Declare cutpoints for every numeric x and no other column."
        )
    cuts = {}
    for name in names:
        values = cutpoints[name]
        if not isinstance(values, (list, tuple)) or len(values) > 1000:
            raise AnalysisError(
                "invalid_cutpoints",
                "Cuts must be an explicit finite sorted list, at most 1000 per column.",
            )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise AnalysisError("invalid_cutpoints", "Numeric cuts must be finite numbers.")
        values = list(map(float, values))
        if any(a >= b for a, b in zip(values, values[1:])):
            raise AnalysisError(
                "invalid_cutpoints", "Numeric cuts must be strictly increasing and unique."
            )
        cuts[name] = values
    selected, metadata = m.sample(
        data,
        [treatment, *allx],
        numeric=[treatment, *names],
        missing=missing,
        max_work=max_work,
        cost=max(len(allx), sum(len(v) for v in cuts.values()) + 1),
    )
    t = m.binary(selected, treatment)
    codes = [
        torch.bucketize(
            m.tensor(selected, name),
            torch.tensor(cuts[name], dtype=torch.float64, device="cpu"),
            right=True,
        ).tolist()
        for name in names
    ]
    labels = []
    for name in cats:
        values = []
        for value in selected[name]:
            if not pd.api.types.is_scalar(value) or not isinstance(
                c.label(value), (str, int, float, bool)
            ):
                raise AnalysisError(
                    "invalid_category", "Exact-match category labels must be finite scalar values."
                )
            label = c.label(value)
            if isinstance(label, float) and not math.isfinite(label):
                raise AnalysisError(
                    "invalid_category", "Exact-match category labels must be finite."
                )
            if hasattr(value, "item") and callable(value.item):
                value = value.item()
            values.append((type(value).__name__, value))
        labels.append(values)
    strata, keys, memberships, key_ids = {}, [], [], {}
    for i in range(len(selected)):
        key = tuple([code[i] for code in codes] + [label[i] for label in labels])
        if key not in strata:
            strata[key] = [[], []]
            key_ids[key] = len(keys)
            keys.append(key)
        strata[key][int(t[i])].append(i)
        memberships.append(key_ids[key])
    ww = torch.zeros(len(selected), dtype=torch.float64, device="cpu")
    counts = []
    for sid, key in enumerate(keys):
        control, treated = strata[key]
        retained = bool(control and treated)
        if retained:
            ww[treated] = 1
            ww[control] = len(treated) / len(control)
        counts.append([sid, len(control), len(treated), retained])
    kept = ww > 0
    if not bool(kept.any()):
        raise AnalysisError(
            "no_common_support", "No prespecified coarsened stratum contains both treatment arms."
        )
    retained_positions = [metadata["positions"][i] for i in range(len(selected)) if kept[i]]
    excluded_positions = [metadata["positions"][i] for i in range(len(selected)) if not kept[i]]
    state = dict(
        target_population="retained treated within prespecified common-support coarsened strata",
        treatment=t.tolist(),
        weights=ww.tolist(),
        stratum_keys=[list(key) for key in keys],
        category_key_rule="typed scalar identity: bool, integer, float and text remain distinct",
        stratum_membership=memberships,
        retained_positions=retained_positions,
        excluded_positions=excluded_positions,
        retained_treated=int(((t == 1) & kept).sum()),
        original_complete_treated=int((t == 1).sum()),
        covariance=None,
        inference="not applicable: preprocessing weights, not an effect estimator",
        assumptions="prespecified pre-treatment bins/categories; no automatic causal identification or original-population ATT",
    )
    tables = {
        "weights": m.frame(
            [
                [metadata["positions"][i], int(t[i]), memberships[i], bool(kept[i]), float(ww[i])]
                for i in range(len(selected))
            ],
            columns=["position", "treatment", "stratum", "retained", "weight"],
        ),
        "strata": m.frame(counts, columns=["stratum", "control_n", "treated_n", "retained"]),
    }
    return m.result(
        "cem",
        tables,
        metadata,
        dict(treatment=treatment, x=names, categorical=cats, cutpoints=cuts),
        state,
        notes=[state["inference"], state["assumptions"]],
    )


@m.procedure
def balance(
    data,
    treatment: str,
    x: list[str],
    *,
    balance_weights: str | None = None,
    level: float = 0.95,
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Before/after means, fixed-scale SMD, variance ratio, ECDF distance and ESS.

    balance_weights are finite nonnegative prespecified analysis weights; zeros
    explicitly remove rows only from the after-weight target. Inference describes
    independent-row weighted mean differences with weights held fixed, not an
    entropy/CEM estimated-weight effect uncertainty or causal identification test.
    """
    m.options(device, weights, max_work, level)
    names = _columns(x, treatment, [balance_weights] if balance_weights else [])
    if balance_weights == treatment:
        raise AnalysisError("invalid_roles", "Treatment and balance weight roles must differ.")
    columns = [treatment, *names, *([balance_weights] if balance_weights else [])]
    selected, metadata = m.sample(
        data,
        columns,
        numeric=columns,
        missing=missing,
        max_work=max_work,
        cost=len(names) ** 2 + 32 * len(names),
    )
    t, xx = m.binary(selected, treatment), _matrix(selected, names)
    ww = (
        torch.ones(len(selected), dtype=torch.float64, device="cpu")
        if balance_weights is None
        else m.tensor(selected, balance_weights)
    )
    if bool((ww < 0).any()):
        raise AnalysisError(
            "invalid_weights",
            "Balance weights must be finite nonnegative; zeros are recorded as after-weight exclusions.",
        )
    p = len(names)
    plan_workspace(
        "weighted balance covariance and ECDF state",
        {
            "moments_and_scores": 96 * len(selected) * p,
            "full_covariance": 64 * p * p,
            "sorted_ecdf_buffers": 128 * len(selected),
        },
    )
    means, variances, before_means, before_variances, ess, pieces = [], [], [], [], [], []
    for group in (0, 1):
        take = t == group
        values, raw = xx[take], ww[take]
        admitted = raw > 0
        if len(values) < 2 or int(admitted.sum()) < 2:
            raise AnalysisError(
                "insufficient_support",
                "Each arm needs at least two original and positive-after-weight observations.",
            )
        scaled = raw / raw.max()
        probability = scaled / scaled.sum()
        if not bool(torch.isfinite(probability).all()) or bool(
            ((raw > 0) & (probability <= 0)).any()
        ):
            raise AnalysisError(
                "numerical_failure",
                "A positive balance weight underflowed normalization; no positive unit is silently excluded.",
            )
        mean = (probability[:, None] * values).sum(0)
        centered = values - mean
        # Fixed-weight HC1 contribution, n+/(n+-1); weights are not re-estimated.
        scores = probability[:, None] * centered
        covariance = scores.T @ scores
        if bool(((covariance.diagonal() == 0) & (centered[admitted] != 0).any(0)).any()):
            raise AnalysisError(
                "numerical_failure",
                "Nonconstant fixed-weight arm variance underflowed float64; a zero-uncertainty diagnostic is not returned.",
            )
        covariance *= int(admitted.sum()) / (int(admitted.sum()) - 1)
        pieces.append(covariance)
        means.append(mean)
        variances.append((probability.sqrt()[:, None] * centered).square().sum(0))
        before_means.append(values.mean(0))
        before_variances.append(values.var(0, unbiased=True))
        ess.append(float(1 / probability.square().sum()))
    difference = means[1] - means[0]
    cov = pieces[0] + pieces[1]
    se = cov.diagonal().clamp_min(0).sqrt()
    critical = critical_value(1 - level, None)
    pooled = ((before_variances[0] + before_variances[1]) / 2).sqrt()
    rows = []
    ecdf_state = {}
    for j, name in enumerate(names):
        values, order = torch.sort(xx[:, j])
        group0, group1 = t[order] == 0, t[order] == 1
        raw = ww[order]
        f0 = torch.cumsum(raw * group0, 0) / raw[group0].sum()
        f1 = torch.cumsum(raw * group1, 0) / raw[group1].sum()
        # Evaluate only the last member of each tie, i.e. actual CDF jumps.
        ends = torch.cat((values[1:] != values[:-1], torch.tensor([True], device="cpu")))
        ecdf_distance = float((f1[ends] - f0[ends]).abs().max())
        ecdf_state[name] = dict(
            support=values[ends].tolist(), control=f0[ends].tolist(), treated=f1[ends].tolist()
        )
        statistic = float(difference[j] / se[j]) if se[j] > 0 else None
        pvalue = (
            float(
                two_sided_p_values(
                    torch.tensor([statistic], dtype=torch.float64, device="cpu"), None
                )[0]
            )
            if statistic is not None
            else None
        )
        rows.append(
            [
                name,
                float(before_means[0][j]),
                float(before_means[1][j]),
                float(means[0][j]),
                float(means[1][j]),
                float((before_means[1][j] - before_means[0][j]) / pooled[j])
                if pooled[j] > 0
                else None,
                float(difference[j] / pooled[j]) if pooled[j] > 0 else None,
                float(variances[1][j] / variances[0][j]) if variances[0][j] > 0 else None,
                ecdf_distance,
                float(difference[j]),
                float(se[j]),
                statistic,
                pvalue,
                float(difference[j] - critical * se[j]),
                float(difference[j] + critical * se[j]),
            ]
        )
    tables = {
        "balance": m.frame(
            rows,
            columns=[
                "covariate",
                "before_control",
                "before_treated",
                "after_control",
                "after_treated",
                "smd_before",
                "smd_after",
                "variance_ratio_after",
                "ecdf_distance_after",
                "mean_difference",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
            ],
        ),
        "covariance": m.frame(cov.tolist(), index=names, columns=names),
        "effective_sample": m.frame(
            [[g, int((t == g).sum()), int(((t == g) & (ww > 0)).sum()), ess[g]] for g in (0, 1)],
            columns=["treatment", "n_before", "n_positive_weight", "ess_after"],
        ),
    }
    state = dict(
        treatment=t.tolist(),
        covariates=xx.tolist(),
        weights=ww.tolist(),
        covariance=cov.tolist(),
        arm_covariances=[v.tolist() for v in pieces],
        ecdf=ecdf_state,
        inference=dict(
            distribution="normal",
            df=None,
            level=level,
            covariance="fixed-weight independent-row HC1 arm means; not estimated-weight uncertainty",
        ),
        assumptions="pre-treatment observed balance diagnostics; weights held fixed; no causal identification from balance or its p-value",
        zero_weight_positions=[
            metadata["positions"][i] for i in range(len(selected)) if ww[i] == 0
        ],
    )
    return m.result(
        "balance",
        tables,
        metadata,
        dict(treatment=treatment, x=names, balance_weights=balance_weights, level=level),
        state,
        notes=[state["assumptions"], state["inference"]["covariance"]],
    )
