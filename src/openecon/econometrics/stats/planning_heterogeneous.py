"""Prespecified heterogeneous allocations; no observed data or fitted power.

Welch power uses a Satterthwaite/noncentral-t approximation. Fixed one-way
ANOVA uses the exact noncentral-F law. Cluster mean power uses the exact
Gaussian law only when the complete exchangeable covariance is known.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import check_choice, check_number, procedure
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import _gauss_legendre, normal_isf
from openecon.resources import plan_workspace

from .planning import MAX_N as MAX_PARTICIPANTS
from .planning import _count, _normal_power, _number, _options, _result
from .power_designs import _f_cut
from .power_distributions import MAX_NONCENTRALITY, f_power, t_cut, t_power

MAX_GROUP_N = 20_000
MAX_GROUPS = 30
MAX_TOTAL_ANOVA_N = 20_000
MAX_CLUSTERS = 10_000


def _vector(value: Any, name: str, maximum: int, *, minimum: int = 1) -> list:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise AnalysisError(
            "invalid_option", f"{name} must be a list/tuple of at least {minimum} values."
        )
    if len(value) > maximum:
        raise AnalysisError("resource_limit", f"{name} exceeds the {maximum}-entry budget.")
    return list(value)


def _admit(method: str, rows: int, law_buffers: int = 65_536) -> dict:
    return plan_workspace(
        method, {"law_buffers": law_buffers, "geometry_and_saved_state": 4096 + rows * 512}
    ).record()


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or (positive and value <= 0):
        raise AnalysisError(
            "numerical_failure", f"{name} is unresolved in float64 for this design."
        )
    return value


def _small_df_t_tail(cut: float, delta: float, df: float, order: int) -> float:
    """Chi-radius integral with r=R*u^8 smoothing its fractional zero endpoint.

    For df>=1 the transformed density vanishes as u^(8*df-1). The omitted
    radius tail above sqrt(df)+12 has mass below 1e-24 for 1<=df<8.
    Raw-radius panels and conditional-normal transition panels are retained.
    """
    root, upper = math.sqrt(df), math.sqrt(df) + 12
    panels = math.ceil(upper / 0.5)
    radii = [upper * i / panels for i in range(panels + 1)]
    center, scale = delta * root / cut, root / abs(cut)
    radii += [center + i * scale for i in range(-12, 13, 2) if 0 < center + i * scale < upper]
    edges = sorted({(radius / upper) ** 0.125 for radius in radii})
    with torch.device("cpu"), torch.no_grad():
        nodes, weights = _gauss_legendre(order)
        left = torch.tensor(edges[:-1], dtype=torch.float64, device="cpu")
        half = (torch.tensor(edges[1:], dtype=torch.float64, device="cpu") - left) / 2
        u = left[:, None] + half[:, None] * (nodes + 1)
        radius = upper * u**8
        log_density_jacobian = (
            (1 - df / 2) * math.log(2)
            - math.lgamma(df / 2)
            + (df - 1) * torch.log(radius)
            - radius.square() / 2
            + math.log(8 * upper)
            + 7 * torch.log(u)
        )
        tail = 0.5 * torch.erfc((cut * radius / root - delta) / math.sqrt(2))
        return float(((torch.exp(log_density_jacobian) * tail) @ weights) @ half)


def _welch_t_power(delta: float, df: float, alpha: float, alternative: str) -> float:
    if df >= 8:
        return t_power(delta, df, alpha, alternative)
    if not math.isfinite(delta) or abs(delta) > 64:
        raise AnalysisError("resource_limit", "Welch noncentrality exceeds [-64,64].")
    cut = t_cut(alpha, alternative, df)
    signed = -delta if alternative == "lower" else delta

    def evaluate(order: int) -> float:
        result = _small_df_t_tail(cut, signed, df, order)
        if alternative == "two-sided":
            result += _small_df_t_tail(cut, -delta, df, order)
        return result

    previous = evaluate(16)
    for order in (32, 64, 128):
        value = evaluate(order)
        if math.isfinite(value) and abs(value - previous) <= 2e-11:
            if not -2e-12 <= value <= 1 + 2e-12:
                raise AnalysisError("numerical_failure", "Invalid Welch noncentral-t probability.")
            return min(1.0, max(0.0, value))
        previous = value
    raise AnalysisError("numerical_failure", "Welch low-df quadrature did not meet its error gate.")


def _output(
    method: str, plan: dict, scenarios: list[dict], geometry: list[dict], **metadata: Any
) -> TableSet:
    metadata.setdefault(
        "scenario_definition", "fixed allocation, null / half / prespecified signed effect"
    )
    output = _result(
        method,
        dict(solve_for="power", **plan),
        scenarios,
        **metadata,
    )
    output.attrs["solver"] = "fixed prespecified design; no sample-size/effect inversion"
    output.attrs["input"] = "prespecified finite scalars and explicit allocation assumptions"
    output["allocation"] = table(geometry)
    # Complete JSON stores mixed numeric row matrices as floats. Use the same
    # numeric display types before export so restored LaTeX is identical; exact
    # integer allocation identities remain in the plan and metadata.
    for name in ("scenarios", "allocation"):
        output[name] = output[name].astype(float)
    output = saved_summary(output)
    # Summary JSON has canonical sorted table keys. Match its display order.
    return TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)


@procedure
def power_welch(
    effect: float,
    *,
    sd1: float,
    sd2: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TableSet:
    """Approximate Welch power for explicit independent normal group sizes.

    SD1/SD2 are prospective population assumptions. The analysis estimates both
    variances and its df from the samples. Replacing those random quantities by
    population-based Satterthwaite df is an approximation, not exact Welch power.
    Effect is (population mean2-mean1) minus the prespecified null difference.
    """
    alpha, alternative, _ = _options(alpha, alternative, None)
    effect = _number(effect, "effect")
    sd1, sd2 = _number(sd1, "sd1", positive=True), _number(sd2, "sd2", positive=True)
    n1, n2 = _count(n1, "n1", 2, MAX_GROUP_N), _count(n2, "n2", 2, MAX_GROUP_N)
    resource = _admit("power_welch", 2, 2 * 1024**2)
    a, b = sd1 / math.sqrt(n1), sd2 / math.sqrt(n2)
    se = _finite(math.hypot(a, b), "Design standard error", positive=True)
    df = 1 / ((a / se) ** 4 / (n1 - 1) + (b / se) ** 4 / (n2 - 1))
    cut = t_cut(alpha, alternative, df)

    def row(multiplier: float) -> dict:
        difference = multiplier * effect
        delta = _finite(difference / se, "Noncentrality")
        return dict(
            effect_multiplier=multiplier,
            effect=difference,
            n1=n1,
            n2=n2,
            total_n=n1 + n2,
            design_standard_error=se,
            df=df,
            noncentrality=delta,
            critical_value=cut,
            power=_welch_t_power(delta, df, alpha, alternative),
        )

    # Evaluate the requested design before constructing its auxiliary scenarios.
    plan = row(1.0)
    scenarios = [row(multiplier) for multiplier in (0.0, 0.5, 1.0)]
    geometry = [
        dict(group=1, n=n1, sd=sd1, design_mean_variance=a * a),
        dict(group=2, n=n2, sd=sd2, design_mean_variance=b * b),
    ]
    return _output(
        "power_welch",
        dict(**plan, sd1=sd1, sd2=sd2, alpha=alpha, alternative=alternative),
        scenarios,
        geometry,
        inference="Satterthwaite df / noncentral-t approximation; not exact Welch-test power",
        sampling="independent iid normal groups; analysis estimates both variances and df",
        effect_definition="(mean2-mean1)-null_difference",
        df_method="population Satterthwaite",
        covariance_known_at_analysis=False,
        approximate=True,
        sd1=sd1,
        sd2=sd2,
        n1=n1,
        n2=n2,
        effect=effect,
        alpha=alpha,
        alternative=alternative,
        probability_budget="|noncentrality|<=64; df<=40000; error gate<=2e-11; df<8 radius endpoint smoothing with orders16/32/64/128",
        max_n_per_group=MAX_GROUP_N,
        resource_plan=resource,
        source="https://www.stata.com/manuals13/psspowertwomeans.pdf; Methods and formulas",
    )


@procedure
def power_unbalanced_anova(
    means: list[float], *, sizes: list[int], sd: float, alpha: float = 0.05
) -> TableSet:
    """Exact fixed one-way Gaussian F power at explicit integer allocations.

    Common SD is prespecified for planning but estimated by pooled within-group
    variance at analysis. This tests equality of all group means. It is neither
    heteroskedastic/Welch ANOVA nor random-effects/repeated-measures ANOVA.
    """
    alpha, _, _ = _options(alpha, "upper", None)
    values = _vector(means, "means", MAX_GROUPS, minimum=2)
    counts = _vector(sizes, "sizes", MAX_GROUPS, minimum=2)
    if len(values) != len(counts):
        raise AnalysisError(
            "invalid_option", "Supply exactly one integer size for each group mean."
        )
    values = [_number(value, f"means[{i}]") for i, value in enumerate(values)]
    counts = [_count(value, f"sizes[{i}]", 1, MAX_TOTAL_ANOVA_N) for i, value in enumerate(counts)]
    n, groups = sum(counts), len(counts)
    if n > MAX_TOTAL_ANOVA_N:
        raise AnalysisError("resource_limit", "Total ANOVA allocation exceeds 20000 observations.")
    if n <= groups:
        raise AnalysisError(
            "invalid_option", "ANOVA needs positive pooled within-group degrees of freedom."
        )
    sd = _number(sd, "sd", positive=True)
    resource = _admit("power_unbalanced_anova", groups)
    anchor = values[0]
    shift = _finite(
        math.fsum(count / n * (value - anchor) for count, value in zip(counts, values)),
        "Weighted mean shift",
    )
    grand = _finite(anchor + shift, "Weighted grand mean")
    centered = [_finite((value - anchor) - shift, "Centered group mean") for value in values]
    contributions = []
    for count, value in zip(counts, centered):
        standardized = _finite(value / sd, "Standardized group mean")
        if abs(standardized) > math.sqrt(MAX_NONCENTRALITY / count):
            raise AnalysisError("resource_limit", "ANOVA noncentrality exceeds 4096.")
        contributions.append(count * standardized * standardized)
    nc = _finite(math.fsum(contributions), "Noncentrality")
    if nc > MAX_NONCENTRALITY:
        raise AnalysisError("resource_limit", "ANOVA noncentrality exceeds 4096.")
    df1, df2 = groups - 1, n - groups
    cut = _f_cut(alpha, df1, df2)

    def row(multiplier: float) -> dict:
        scaled_nc = multiplier * multiplier * nc
        return dict(
            effect_multiplier=multiplier,
            total_n=n,
            groups=groups,
            df1=df1,
            df2=df2,
            noncentrality=scaled_nc,
            critical_value=cut,
            power=f_power(cut, df1, df2, scaled_nc),
        )

    plan = row(1.0)
    geometry = [
        dict(
            group=i + 1,
            n=count,
            mean=value,
            grand_mean=grand,
            centered_mean=center,
            allocation_fraction=count / n,
            noncentrality_contribution=contribution,
        )
        for i, (count, value, center, contribution) in enumerate(
            zip(counts, values, centered, contributions)
        )
    ]
    return _output(
        "power_unbalanced_anova",
        dict(**plan, sd=sd, grand_mean=grand, alpha=alpha),
        [row(multiplier) for multiplier in (0.0, 0.5, 1.0)],
        geometry,
        scenario_definition="fixed integer allocation, centered mean deviations multiplied by 0/.5/1",
        inference="exact noncentral-F rejection probability under fixed homoskedastic Gaussian design",
        sampling="independent normal observations; fixed group means and common variance",
        covariance_known_at_analysis=False,
        approximate=False,
        means=values,
        sizes=counts,
        sd=sd,
        alpha=alpha,
        weighted_grand_mean=grand,
        probability_budget="lambda<=4096; df<=40000; Poisson omitted mass<=2e-14",
        max_total_n=MAX_TOTAL_ANOVA_N,
        max_groups=MAX_GROUPS,
        resource_plan=resource,
        source="https://www.stata.com/manuals13/psspoweroneway.pdf; fixed one-way F methods",
    )


def _arm(
    sizes: list[int], sd: float, icc: float, weighting: str, arm: int
) -> tuple[dict, list[dict]]:
    n, groups = sum(sizes), len(sizes)
    rows = []
    for i, size in enumerate(sizes):
        weight = size / n if weighting == "participant" else 1 / groups
        variance = sd * sd * (icc + (1 - icc) / size)
        contribution = weight * weight * variance
        independent_contribution = weight * weight * sd * sd / size
        rows.append(
            dict(
                arm=arm,
                cluster=i + 1,
                size=size,
                weight=weight,
                cluster_mean_variance=variance,
                variance_contribution=contribution,
                independent_variance_contribution=independent_contribution,
            )
        )
    variance = _finite(
        math.fsum(row["variance_contribution"] for row in rows), "Arm mean variance", positive=True
    )
    independent = _finite(
        math.fsum(row["independent_variance_contribution"] for row in rows),
        "Independent arm variance",
        positive=True,
    )
    return dict(
        n=n,
        clusters=groups,
        variance=variance,
        independent_variance=independent,
        design_effect=variance / independent,
    ), rows


@procedure
def power_unequal_cluster_mean(
    effect: float,
    *,
    sd1: float,
    sd2: float,
    sizes1: list[int],
    sizes2: list[int],
    icc1: float,
    icc2: float,
    weighting: str = "participant",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TableSet:
    """Exact Gaussian two-arm mean power for known exchangeable covariance.

    Clusters are mutually independent; cluster sizes and both arm SD/ICC values
    are fixed and known at analysis. 'participant' targets pooled participant
    means; 'cluster' weights each cluster mean equally. No cluster-t, estimated
    ICC uncertainty, optimal mixed-model weighting or random allocation is used.
    """
    alpha, alternative, _ = _options(alpha, alternative, None)
    effect = _number(effect, "effect")
    sd1, sd2 = _number(sd1, "sd1", positive=True), _number(sd2, "sd2", positive=True)
    weighting = check_choice(weighting, "weighting", ("participant", "cluster"))
    icc1, icc2 = check_number(icc1, "icc1"), check_number(icc2, "icc2")
    if not 0 <= icc1 <= 1 or not 0 <= icc2 <= 1:
        raise AnalysisError("invalid_option", "Both prespecified ICCs must be in [0,1].")
    vectors = [_vector(sizes1, "sizes1", MAX_CLUSTERS), _vector(sizes2, "sizes2", MAX_CLUSTERS)]
    sizes = [
        [_count(size, f"sizes{arm + 1}[{i}]", 1, MAX_PARTICIPANTS) for i, size in enumerate(vector)]
        for arm, vector in enumerate(vectors)
    ]
    if any(sum(vector) > MAX_PARTICIPANTS for vector in sizes):
        raise AnalysisError("resource_limit", "Each arm is limited to 10000000 participants.")
    resource = _admit("power_unequal_cluster_mean", sum(map(len, sizes)))
    first, first_rows = _arm(sizes[0], sd1, icc1, weighting, 1)
    second, second_rows = _arm(sizes[1], sd2, icc2, weighting, 2)
    variance = _finite(
        first["variance"] + second["variance"], "Mean-difference variance", positive=True
    )
    se = math.sqrt(variance)
    cut = normal_isf(alpha / 2 if alternative == "two-sided" else alpha)

    def row(multiplier: float) -> dict:
        difference = multiplier * effect
        delta = _finite(difference / se, "Normal shift")
        return dict(
            effect_multiplier=multiplier,
            effect=difference,
            n1=first["n"],
            n2=second["n"],
            total_n=first["n"] + second["n"],
            clusters1=first["clusters"],
            clusters2=second["clusters"],
            design_standard_error=se,
            difference_variance=variance,
            normal_shift=delta,
            critical_value=cut,
            power=_normal_power(delta, alpha, alternative),
        )

    plan = row(1.0)
    return _output(
        "power_unequal_cluster_mean",
        dict(
            **plan,
            weighting=weighting,
            alpha=alpha,
            alternative=alternative,
            variance1=first["variance"],
            variance2=second["variance"],
            design_effect1=first["design_effect"],
            design_effect2=second["design_effect"],
        ),
        [row(multiplier) for multiplier in (0.0, 0.5, 1.0)],
        first_rows + second_rows,
        inference="exact Gaussian rejection probability with known exchangeable covariance",
        sampling="independent Gaussian clusters; fixed sizes, common arm mean, known SD/ICC",
        effect_definition="(arm2 mean-arm1 mean)-null_difference",
        approximate=False,
        covariance_known_at_analysis=True,
        weighting=weighting,
        sizes1=sizes[0],
        sizes2=sizes[1],
        sd1=sd1,
        sd2=sd2,
        icc1=icc1,
        icc2=icc2,
        effect=effect,
        alpha=alpha,
        alternative=alternative,
        known_mean_covariance=[[first["variance"], 0.0], [0.0, second["variance"]]],
        independent_mean_variances=[first["independent_variance"], second["independent_variance"]],
        design_effect_reference="same declared mean weights with ICC=0; not mixed-model relative efficiency",
        max_clusters_per_arm=MAX_CLUSTERS,
        max_participants_per_arm=MAX_PARTICIPANTS,
        resource_plan=resource,
        source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4382318/; Gaussian random-intercept model equation (1); direct known-covariance weighted-mean calculation",
    )
