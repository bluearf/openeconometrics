"""One-sample distribution tests: ksmirnov, runtest, bitest, prtest, chi2gof.

``ksmirnov`` also covers the two-sample Kolmogorov-Smirnov test (``by=``).
Empirical distribution functions come from one sort; the reference
distributions of the Kolmogorov statistics live in ``exact.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric import exact as exact_kernels
from openecon.econometrics.nonparametric.common import (
    FLOAT, binary_codes, binomial_log_pmf, binomial_lower, binomial_upper, check_alpha,
    check_choice, check_flag, check_number, check_probability, group_codes, label,
    normal_two_sided, numeric, procedure, select, two_groups, weight_column, weighted_count,
)
from openecon.engines.distributions import chi2_sf, f_isf, normal_cdf, normal_isf, normal_sf

DISTRIBUTIONS = ("normal", "uniform", "exponential", "poisson")
EXACT_ONE_SAMPLE_N = 1000
EXACT_TWO_SAMPLE_PRODUCT = 10_000
EXACT_RUNS_N = 50


# ---- Kolmogorov-Smirnov ------------------------------------------------------------------


def lilliefors_p_value(d: float, n: int) -> float:
    """Lilliefors-corrected p of the normal KS statistic with estimated mean and variance.

    Dallal and Wilkinson (1986): for 5 <= n <= 100 and p <= 0.1

        p = exp(-7.01256 D^2 (n + 2.78019) + 2.99587 D sqrt(n + 2.78019) - 0.122119
                + 0.974598 / sqrt(n) + 1.67997 / n);

    for n > 100 the statistic is rescaled, D (n / 100)^0.49, and n set to 100.
    When that formula gives p > 0.1 (outside its range) the p-value is taken
    from Stephens' (1974) modified statistic (sqrt(n) - 0.01 + 0.85 / sqrt(n)) D
    through the piecewise polynomial used by R's ``nortest::lillie.test``.
    """
    scaled, size = (d * (n / 100.0) ** 0.49, 100.0) if n > 100 else (d, float(n))
    p = math.exp(-7.01256 * scaled * scaled * (size + 2.78019)
                 + 2.99587 * scaled * math.sqrt(size + 2.78019) - 0.122119
                 + 0.974598 / math.sqrt(size) + 1.67997 / size)
    if p <= 0.1:
        return p
    k = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * d
    if k <= 0.302:
        return 1.0
    if k <= 0.5:
        p = 2.76773 - 19.828315 * k + 80.709644 * k ** 2 - 138.55152 * k ** 3 + 81.218052 * k ** 4
    elif k <= 0.9:
        p = (-4.901232 + 40.662806 * k - 97.490286 * k ** 2 + 94.029866 * k ** 3
             - 32.355711 * k ** 4)
    elif k <= 1.31:
        p = (6.198765 - 19.558097 * k + 23.186922 * k ** 2 - 12.234627 * k ** 3
             + 2.423045 * k ** 4)
    else:
        p = 0.0
    return min(1.0, max(0.0, p))


def _parameters(distribution: str, params: Any, x: Tensor) -> tuple[list[float], bool]:
    """(parameters, estimated?) of the reference distribution."""
    expected = {"normal": 2, "uniform": 2, "exponential": 1, "poisson": 1}[distribution]
    if params is None:
        n = x.numel()
        if distribution == "normal":
            if n < 2:
                raise AnalysisError("too_few_observations", "Estimating the normal parameters "
                                    "needs at least 2 observations.")
            centred = x - x.mean()
            scale = float(centred.abs().max())          # scaled so the squares cannot underflow
            deviation = scale * float((centred / scale).std(unbiased=True)) if scale > 0 else 0.0
            values = [float(x.mean()), deviation]
        elif distribution == "uniform":
            values = [float(x.min()), float(x.max())]
        else:
            values = [float(x.mean())]
        estimated = True
    else:
        if isinstance(params, (int, float)) and not isinstance(params, bool):
            params = [params]
        if not isinstance(params, (list, tuple)) or len(params) != expected:
            names = {"normal": "(mean, sd)", "uniform": "(low, high)", "exponential": "(mean,)",
                     "poisson": "(mean,)"}[distribution]
            raise AnalysisError("invalid_option", f"params for the {distribution} distribution "
                                f"must be {names}.")
        values = [check_number(value, "params") for value in params]
        estimated = False
    if distribution == "normal" and not values[1] > 0:
        raise AnalysisError("no_variation" if estimated else "invalid_option",
                            "The normal standard deviation must be positive"
                            + (": the column is constant." if estimated else "."))
    if distribution == "uniform" and not values[1] > values[0]:
        raise AnalysisError("no_variation" if estimated else "invalid_option",
                            "The uniform range must have low < high"
                            + (": the column is constant." if estimated else "."))
    if distribution in ("exponential", "poisson") and not values[0] > 0:
        raise AnalysisError("invalid_values" if estimated else "invalid_option",
                            f"The {distribution} mean must be positive.")
    return values, estimated


def _one_sample_differences(x: Tensor, distribution: str, values: list[float]) -> tuple[float,
                                                                                         float]:
    """(D+, D-) >= 0: the largest excess of the empirical cdf over F and of F over it."""
    n = x.numel()
    ordered = torch.sort(x).values
    if distribution == "poisson":
        if bool((ordered < 0).any()) or bool((ordered != ordered.round()).any()):
            raise AnalysisError("invalid_values", "The Poisson test needs nonnegative integer "
                                "values.")
        distinct, counts = torch.unique_consecutive(ordered, return_counts=True)
        after = counts.cumsum(0).to(FLOAT) / n
        before = after - counts.to(FLOAT) / n
        mean = torch.tensor(values[0], dtype=FLOAT)
        cdf = torch.special.gammaincc(distinct + 1.0, mean)
        cdf_before = torch.where(distinct > 0, torch.special.gammaincc(distinct.clamp_min(1.0),
                                                                       mean),
                                 torch.zeros_like(distinct))
        return float((after - cdf).max().clamp_min(0.0)), \
            float((cdf_before - before).max().clamp_min(0.0))
    if distribution == "normal":
        cdf = torch.special.ndtr((ordered - values[0]) / values[1])
    elif distribution == "uniform":
        cdf = ((ordered - values[0]) / (values[1] - values[0])).clamp(0.0, 1.0)
    else:
        cdf = -torch.expm1(-ordered.clamp_min(0.0) / values[0])
    position = torch.arange(1, n + 1, dtype=FLOAT)
    return float((position / n - cdf).max().clamp_min(0.0)), \
        float((cdf - (position - 1.0) / n).max().clamp_min(0.0))


def _two_sample(x: Tensor, codes: Tensor, n1: int, n2: int) -> tuple[int, int, bool]:
    """(max(i n2 - j n1), max(j n1 - i n2), any ties) over the pooled distinct values."""
    values, order = torch.sort(x, stable=True)
    first = (codes[order] == 0).to(torch.int64).cumsum(0)
    position = torch.arange(1, x.numel() + 1, dtype=torch.int64)
    last = torch.ones(x.numel(), dtype=torch.bool)
    last[:-1] = values[1:] != values[:-1]
    i, total = first[last], position[last]
    lattice = i * n2 - (total - i) * n1
    return max(int(lattice.max()), 0), max(int((-lattice).max()), 0), bool((~last).any())


@procedure
def ksmirnov(data: Any, y: str, *, by: str | None = None, distribution: str = "normal",
             params: Any = None, exact: bool | None = None, missing: str = "drop"):
    """Kolmogorov-Smirnov test: one sample against a distribution, or two samples.

    One sample (``by=None``). With F the hypothesized cdf and F_n the
    empirical cdf of the n sorted values x_(1) <= ... <= x_(n),

        D+ = max_i (i/n - F(x_(i))),   D- = max_i (F(x_(i)) - (i-1)/n),   D = max(D+, D-).

    ``distribution`` is "normal", "uniform", "exponential" or "poisson";
    ``params`` are (mean, sd), (low, high), (mean,) and (mean,). With
    ``params=None`` they are estimated as SPSS does: sample mean and standard
    deviation (n - 1), sample minimum and maximum, sample mean. For the
    Poisson distribution the differences are taken on both sides of every
    jump, D+ = max_v (F_n(v) - F(v)), D- = max_v (F(v - 1) - F_n(v-)).

    p-values:

    * asymptotic: the Kolmogorov distribution at sqrt(n) D (SPSS's
      "Kolmogorov-Smirnov Z", Stata's combined K-S); one-sided
      exp(-2 n D+-^2) for D+ and D-.
    * exact (continuous distribution with given parameters): P(D_n >= D) by
      Marsaglia-Tsang-Wang; ``exact=None`` computes it for n <= 1000.
    * normal with estimated parameters: the Lilliefors-corrected p (Dallal and
      Wilkinson 1986), which SPSS reports as "Lilliefors Significance
      Correction"; the asymptotic p is then far too large.
    * other distributions with estimated parameters: only the asymptotic p,
      which is conservative (noted in ``attrs["notes"]``).

    Two samples (``by=`` a column with two groups). D+ = max (F_1 - F_2),
    D- = max (F_2 - F_1), D the larger; asymptotic p from the Kolmogorov
    distribution at D sqrt(n1 n2 / (n1 + n2)); exact p by counting lattice
    paths when there are no ties (``exact=None``: n1 n2 <= 10,000).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column.
    by : optional grouping column with exactly two values (two-sample test).
    distribution : "normal", "uniform", "exponential" or "poisson" (one sample).
    params : parameters of the distribution, or None to estimate them.
    exact : None (the size rules above), True or False.
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    One table with rows positive, negative, combined and columns statistic,
    p_value (asymptotic). The negative difference is reported with a minus
    sign (SPSS "Most Extreme Differences"). ``attrs``: n (or n1, n2),
    statistic (D), z (the scaled statistic), p_value (the recommended one: the
    Lilliefors p, else the exact p when computed, else the asymptotic p),
    p_asymptotic, p_exact, p_lilliefors, distribution, params, estimated.

    Stata: ``ksmirnov y = normal((y - m) / s)``, ``ksmirnov y, by(group)
    [exact]``. SPSS: ``NPAR TESTS /K-S(NORMAL)=y``, ``/K-S=y BY group(a b)``,
    ``EXAMINE ... /PLOT NPPLOT`` (Lilliefors).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [0.1, 0.4, 0.7, 1.3, 2.1, 2.8, 3.3, 4.9]}
    >>> result = oe.ksmirnov(data, "x", distribution="exponential", params=[2.0])
    >>> round(result.attrs["statistic"], 4)
    0.1501
    """
    if not isinstance(y, str) or (by is not None and not isinstance(by, str)):
        raise AnalysisError("invalid_spec", "y and by must each be the name of one column.")
    exact = check_flag(exact, "exact", optional=True)
    check_choice(distribution, "distribution", DISTRIBUTIONS)
    frame, dropped = select(data, [y] if by is None else [y, by], missing=missing)
    x = numeric(frame, y)
    notes: list[str] = []
    if by is not None:
        if params is not None:
            raise AnalysisError("invalid_option", "params applies to the one-sample test; do "
                                "not combine it with by=.")
        codes, levels = two_groups(frame, by, "The two-sample Kolmogorov-Smirnov test")
        n1 = int((codes == 0).sum())
        n2 = x.numel() - n1
        h_plus, h_minus, ties = _two_sample(x, codes, n1, n2)
        scale = float(n1) * n2
        d_plus, d_minus = h_plus / scale, h_minus / scale
        factor = math.sqrt(scale / (n1 + n2))
        attrs: dict[str, Any] = {"n": n1 + n2, "n1": n1, "n2": n2, "groups": levels,
                                 "ties": ties}
        use_exact = exact if exact is not None else (not ties and scale <= EXACT_TWO_SAMPLE_PRODUCT)
        p_exact = exact_kernels.smirnov_exact(n1, n2, max(h_plus, h_minus)) if use_exact else None
        if use_exact and ties:
            notes.append("Ties between the samples: the exact p-value assumes continuous "
                         "data and is conservative.")
        title = f"Two-sample Kolmogorov-Smirnov test: {y} by {by}"
    else:
        values, estimated = _parameters(distribution, params, x)
        d_plus, d_minus = _one_sample_differences(x, distribution, values)
        n = x.numel()
        factor = math.sqrt(n)
        attrs = {"n": n, "distribution": distribution, "params": values, "estimated": estimated}
        continuous = distribution != "poisson"
        if exact and (estimated or not continuous):
            raise AnalysisError("exact_unavailable", "The exact Kolmogorov distribution holds "
                                "only for a continuous distribution with given parameters; "
                                "pass params=... or exact=False.")
        use_exact = exact if exact is not None else (continuous and not estimated
                                                     and n <= EXACT_ONE_SAMPLE_N)
        p_exact = exact_kernels.kolmogorov_sf_exact(n, max(d_plus, d_minus)) if use_exact else None
        if estimated and distribution == "normal":
            if n >= 5:
                attrs["p_lilliefors"] = lilliefors_p_value(max(d_plus, d_minus), n)
            else:
                notes.append("The Lilliefors correction needs at least 5 observations.")
        elif estimated:
            notes.append("Parameters were estimated from the sample: the asymptotic p-value "
                         "is conservative (too large).")
        if not continuous:
            notes.append("The Kolmogorov distribution is conservative for a discrete "
                         "distribution.")
        title = f"One-sample Kolmogorov-Smirnov test: {y} vs {distribution}"
    d = max(d_plus, d_minus)
    p_asymptotic = exact_kernels.kolmogorov_sf(factor * d)
    rows = [[d_plus, math.exp(-2.0 * (factor * d_plus) ** 2)],
            [-d_minus, math.exp(-2.0 * (factor * d_minus) ** 2)], [d, p_asymptotic]]
    attrs.update({"statistic": d, "d_plus": d_plus, "d_minus": -d_minus, "z": factor * d,
                  "p_asymptotic": p_asymptotic, "exact": p_exact is not None})
    if p_exact is not None:
        attrs["p_exact"] = p_exact
    attrs["p_value"] = attrs.get("p_lilliefors", p_exact if p_exact is not None else p_asymptotic)
    attrs.update({"n_dropped": dropped, "label": title})
    if notes:
        attrs["notes"] = notes
    return table(rows, columns=["statistic", "p_value"],
                 index=["positive", "negative", "combined"], **attrs)


# ---- runs test ---------------------------------------------------------------------------


@procedure
def runtest(data: Any, y: str, *, threshold: Any = "median", ties: str = "above",
            continuity: bool | None = None, exact: bool | None = None, missing: str = "drop"):
    """Wald-Wolfowitz runs test for randomness of the order of a series.

    Observations (in row order) are split at ``threshold`` ("median", "mean"
    or a number) into values above and below; a run is a maximal stretch on
    one side. With n1, n2 the two counts, N = n1 + n2 and R the number of runs,

        E(R) = 2 n1 n2 / N + 1,   Var(R) = 2 n1 n2 (2 n1 n2 - N) / (N^2 (N - 1)),
        z = (R - E(R)) / sqrt(Var(R)).

    Values equal to the threshold (``ties``) are counted as above ("above":
    x >= cut, the SPSS rule), as below ("below": x <= cut, the Stata rule) or
    left out ("drop", Stata's ``drop`` option).

    ``continuity`` moves |R - E(R)| by 1/2 towards zero. ``None`` applies it
    when N < 50, as SPSS does; Stata applies it only on request.

    Exact test: the distribution of R given n1 and n2 is known in closed
    form; the exact two-sided p is P(|R - E| >= |r - E|) (``exact=None``
    computes it for N <= 50).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records, in series order.
    y : numeric column.
    threshold : "median", "mean" or a number.
    ties : "above", "below" or "drop" (values equal to the threshold).
    continuity : None (applied when N < 50), True or False.
    exact : None (computed when N <= 50), True or False.
    missing : "drop" (missing values are removed and the remaining values are
        treated as consecutive; counted in ``n_dropped``) or "raise".

    Returns
    -------
    One table (row "runs"): n, n_below, n_above, runs, expected, z, p_value.
    ``attrs`` adds threshold, variance, continuity, p_exact, p_exact_lower
    (P(R <= r): too few runs, positive dependence) and p_exact_upper.

    Stata: ``runtest y [, threshold(#) mean continuity drop]``. SPSS: ``NPAR
    TESTS /RUNS(MEDIAN)=y``.

    Example
    -------
    >>> import openecon as oe
    >>> result = oe.runtest({"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, "y")
    >>> int(result.attrs["runs"]), round(result.attrs["p_exact"], 4)
    (2, 0.0159)
    """
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    check_choice(ties, "ties", ("above", "below", "drop"))
    continuity = check_flag(continuity, "continuity", optional=True)
    exact = check_flag(exact, "exact", optional=True)
    frame, dropped = select(data, [y], missing=missing)
    x = numeric(frame, y)
    if isinstance(threshold, str):
        check_choice(threshold, "threshold", ("median", "mean"))
        if threshold == "mean":
            cut = float(x.mean())
        else:
            ordered = torch.sort(x).values
            half = x.numel() // 2
            cut = float(ordered[half]) if x.numel() % 2 else \
                0.5 * float(ordered[half - 1] + ordered[half])
    else:
        cut = check_number(threshold, "threshold")
    if ties == "drop":
        x = x[x != cut]
    above = x >= cut if ties == "above" else x > cut
    n_above = int(above.sum())
    n_below = x.numel() - n_above
    total = n_above + n_below
    if n_above == 0 or n_below == 0:
        raise AnalysisError("no_variation", "All values fall on one side of the threshold "
                            f"{cut:g}, so the runs test is undefined. Choose another threshold "
                            "or ties= rule.")
    runs = 1 + int((above[1:] != above[:-1]).sum())
    product = 2.0 * n_above * n_below
    expected = product / total + 1.0
    variance = product * (product - total) / (total * total * (total - 1.0))
    if variance <= 0.0:
        raise AnalysisError("too_few_observations", "The runs test needs at least 3 "
                            "observations with both sides of the threshold present.")
    use_continuity = continuity if continuity is not None else total < 50
    gap = runs - expected
    if use_continuity:
        gap = math.copysign(max(0.0, abs(gap) - 0.5), gap)
    z = gap / math.sqrt(variance)
    attrs: dict[str, Any] = {"n": total, "n_below": n_below, "n_above": n_above, "runs": runs,
                             "expected": expected, "variance": variance, "z": z,
                             "p_value": normal_two_sided(z), "threshold": cut, "ties": ties,
                             "continuity": bool(use_continuity)}
    use_exact = exact if exact is not None else total <= EXACT_RUNS_N
    if use_exact:
        support, pmf = exact_kernels.runs_distribution(n_above, n_below)
        distance = abs(runs - expected)
        far = (support - expected).abs() >= distance - 1e-9
        attrs.update({"p_exact": min(1.0, float(pmf[far].sum())),
                      "p_exact_lower": min(1.0, float(pmf[support <= runs].sum())),
                      "p_exact_upper": min(1.0, float(pmf[support >= runs].sum()))})
    attrs.update({"exact": bool(use_exact), "n_dropped": dropped, "label": "Runs test"})
    return table([[total, n_below, n_above, runs, expected, z, attrs["p_value"]]],
                 columns=["n", "n_below", "n_above", "runs", "expected", "z", "p_value"],
                 index=["runs"], **attrs)


# ---- binomial and proportions ------------------------------------------------------------


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence limits of a binomial proportion, via F quantiles."""
    low = 0.0
    if k > 0:
        f = f_isf(1.0 - alpha / 2.0, 2.0 * k, 2.0 * (n - k + 1))
        low = k * f / (n - k + 1 + k * f)
    high = 1.0
    if k < n:
        f = f_isf(alpha / 2.0, 2.0 * (k + 1), 2.0 * (n - k))
        high = (k + 1) * f / (n - k + (k + 1) * f)
    return low, high


@procedure
def bitest(data: Any, y: str, *, p: float = 0.5, positive: Any = None, alpha: float = 0.05,
           missing: str = "drop"):
    """Exact binomial test of a proportion with the Clopper-Pearson interval.

    ``y`` is a binary column; k is the number of rows at the event level (the
    larger of the two values, 1 for 0/1 data, or ``positive``). Under
    H0: P(event) = p, k ~ Binomial(n, p).

    * ``p_lower`` = P(K <= k), ``p_upper`` = P(K >= k).
    * ``p_value`` (two-sided) = the total probability of the outcomes that are
      no more probable than k (Stata ``bitest``, R, SciPy). For p = 1/2 this
      equals SPSS's 2 * min(p_lower, p_upper).
    * The confidence interval is Clopper-Pearson's exact interval (Stata
      ``ci proportions``, default exact).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : binary column.
    p : hypothesized probability of the event, strictly between 0 and 1.
    positive : the value counted as the event (default: the larger value).
    alpha : 1 - confidence level of the interval.
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    One table (row = y): n, successes, proportion, expected (n p), p_value,
    ci_low, ci_high. ``attrs`` adds p_lower, p_upper, p, alpha, positive.

    Stata: ``bitest y == 0.3``. SPSS: ``NPAR TESTS /BINOMIAL(0.3)=y``.

    Example
    -------
    >>> import openecon as oe
    >>> result = oe.bitest({"y": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]}, "y", p=0.5)
    >>> round(result.attrs["p_value"], 4)
    0.1094
    """
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one binary column.")
    p = check_probability(p, "p")
    alpha = check_alpha(alpha)
    frame, dropped = select(data, [y], missing=missing)
    codes, levels = binary_codes(frame, y, positive)
    n = codes.numel()
    k = int(codes.sum())
    log_pmf = binomial_log_pmf(n, p)
    two_sided = float(torch.exp(log_pmf[log_pmf <= log_pmf[k] + 1e-7]).sum())
    low, high = clopper_pearson(k, n, alpha)
    attrs = {"n": n, "successes": k, "proportion": k / n, "p": p,
             "p_value": min(1.0, two_sided), "p_lower": binomial_lower(k, n, p),
             "p_upper": binomial_upper(k, n, p), "ci_low": low, "ci_high": high, "alpha": alpha,
             "positive": levels[1], "n_dropped": dropped, "label": "Exact binomial test"}
    return table([[n, k, k / n, n * p, attrs["p_value"], low, high]],
                 columns=["n", "successes", "proportion", "expected", "p_value", "ci_low",
                          "ci_high"], index=[y], **attrs)


@procedure
def prtest(data: Any, y: str, *, p: float | None = None, by: str | None = None,
           positive: Any = None, alpha: float = 0.05, missing: str = "drop"):
    """Large-sample z tests of proportions (Stata's ``prtest``).

    One sample (``by=None``): H0: P(event) = p (default 0.5),

        z = (phat - p) / sqrt(p (1 - p) / n),

    with the Wald interval phat +- z_(alpha/2) sqrt(phat (1 - phat) / n).

    Two samples (``by=`` a column with two groups): H0: p1 = p2,

        z = (phat1 - phat2) / sqrt(pbar (1 - pbar) (1/n1 + 1/n2)),  pbar pooled,

    and the interval of the difference uses the unpooled standard error
    sqrt(phat1 (1 - phat1)/n1 + phat2 (1 - phat2)/n2).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : binary column.
    p : hypothesized proportion of the one-sample test (default 0.5).
    by : grouping column with exactly two values (two-sample test).
    positive : the value counted as the event (default: the larger value).
    alpha : 1 - confidence level of the intervals.
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    One table with a row per sample (and "diff" for two samples): n,
    proportion, std_error, ci_low, ci_high, z, p_value (the test is shown on
    the tested row). ``attrs``: z, p_value (two-sided), p_lower (Pr(Z < z)),
    p_upper (Pr(Z > z)), std_error_null, positive.

    Stata: ``prtest y == 0.5``, ``prtest y, by(group)``. SPSS: ``PROPORTIONS``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1], "g": list("aaaaabbbbb")}
    >>> round(oe.prtest(data, "y", by="g").attrs["z"], 4)
    0.0
    """
    if not isinstance(y, str) or (by is not None and not isinstance(by, str)):
        raise AnalysisError("invalid_spec", "y and by must each be the name of one column.")
    alpha = check_alpha(alpha)
    critical = normal_isf(alpha / 2.0)
    frame, dropped = select(data, [y] if by is None else [y, by], missing=missing)
    codes, levels = binary_codes(frame, y, positive)
    if by is None:
        p0 = 0.5 if p is None else check_probability(p, "p")
        n = codes.numel()
        phat = float(codes.sum()) / n
        se_null = math.sqrt(p0 * (1.0 - p0) / n)
        z = (phat - p0) / se_null
        se = math.sqrt(phat * (1.0 - phat) / n)
        rows, index = [[n, phat, se, phat - critical * se, phat + critical * se, z]], [y]
        attrs: dict[str, Any] = {"n": n, "proportion": phat, "p": p0}
        title = f"One-sample test of proportion: {y} = {p0:g}"
    else:
        if p is not None:
            raise AnalysisError("invalid_option", "p applies to the one-sample test; the "
                                "two-sample test compares the groups of by=.")
        groups, group_levels = two_groups(frame, by, "The two-sample test of proportions")
        sizes = torch.bincount(groups, minlength=2).to(FLOAT)
        events = torch.zeros(2, dtype=FLOAT).index_add_(0, groups, codes.to(FLOAT))
        n1, n2 = float(sizes[0]), float(sizes[1])
        p1, p2 = float(events[0]) / n1, float(events[1]) / n2
        pooled = float(events.sum()) / (n1 + n2)
        se_null = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
        if se_null <= 0.0:
            raise AnalysisError("no_variation", f"'{y}' takes a single value in both groups, so "
                                "the test of proportions is undefined.")
        difference = p1 - p2
        z = difference / se_null
        se1, se2 = math.sqrt(p1 * (1.0 - p1) / n1), math.sqrt(p2 * (1.0 - p2) / n2)
        se = math.sqrt(se1 * se1 + se2 * se2)
        rows = [[n1, p1, se1, p1 - critical * se1, p1 + critical * se1, None],
                [n2, p2, se2, p2 - critical * se2, p2 + critical * se2, None],
                [n1 + n2, difference, se, difference - critical * se, difference + critical * se,
                 z]]
        index = [str(group_levels[0]), str(group_levels[1]), "diff"]
        attrs = {"n": int(n1 + n2), "n1": int(n1), "n2": int(n2), "groups": group_levels,
                 "difference": difference, "pooled": pooled}
        title = f"Two-sample test of proportions: {y} by {by}"
    attrs.update({"z": z, "p_value": normal_two_sided(z), "p_lower": normal_cdf(z),
                  "p_upper": normal_sf(z), "std_error_null": se_null, "alpha": alpha,
                  "positive": levels[1], "n_dropped": dropped, "label": title})
    rows = [[*line, None if line[5] is None else attrs["p_value"]] for line in rows]
    return table(rows, columns=["n", "proportion", "std_error", "ci_low", "ci_high", "z",
                                "p_value"], index=index, **attrs)


# ---- chi-square goodness of fit ----------------------------------------------------------


@procedure
def chi2gof(data: Any, y: str, *, expected: Any = None, weights: str | None = None,
            missing: str = "drop"):
    """Chi-square goodness-of-fit test of the frequencies of one categorical column.

    With O_j the observed count of category j and E_j the expected count,

        chi2 = sum_j (O_j - E_j)^2 / E_j        ~ chi2(k - 1).

    ``expected`` is None (all categories equally likely, SPSS's default), a
    list of k positive numbers in the sorted order of the categories, or a
    mapping {category: number}; the numbers are relative (proportions or
    counts) and are rescaled to sum to N. A mapping may name categories that
    do not occur in the data (observed count 0). ``weights`` is an optional
    column of frequency weights.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : categorical column (any scalar type).
    expected : None, a list of k positive numbers or a mapping (see above).
    weights : optional column of nonnegative frequency weights.
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``frequencies`` (observed, expected, residual,
    std_residual = (O - E)/sqrt(E) per category) and ``tests`` (row chi2:
    statistic, df, p_value). ``attrs``: n, k, statistic, df, p_value,
    min_expected, cells_below_5 and a note when expected counts are small.

    SPSS: ``NPAR TESTS /CHISQUARE=y /EXPECTED=...``. Stata: community command
    ``csgof`` (or ``tabulate`` + manual calculation).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"die": [1, 2, 3, 4, 5, 6, 6, 6, 6, 2, 3, 6]}
    >>> round(oe.chi2gof(data, "die").attrs["statistic"], 3)
    6.0
    """
    if not isinstance(y, str) or (weights is not None and not isinstance(weights, str)):
        raise AnalysisError("invalid_spec", "y and weights must each be the name of one column.")
    frame, dropped = select(data, [y] if weights is None else [y, weights], missing=missing)
    codes, levels = group_codes(frame[y])
    observed = weighted_count(codes, len(levels), weight_column(frame, weights))
    if isinstance(expected, dict):
        wanted = {str(label(key)): value for key, value in expected.items()}
        absent = [str(level) for level in levels if str(level) not in wanted]
        if absent:
            raise AnalysisError("invalid_option", "expected has no entry for the observed "
                                f"categories: {', '.join(absent[:10])}.")
        extra = [key for key in expected if str(label(key)) not in set(map(str, levels))]
        levels = levels + [label(key) for key in extra]
        observed = torch.cat([observed, torch.zeros(len(extra), dtype=FLOAT)])
        shares = [check_number(wanted[str(level)], "expected") for level in levels]
    elif expected is None:
        shares = [1.0] * len(levels)
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(levels):
            raise AnalysisError("invalid_option", f"expected must have one entry per category "
                                f"({len(levels)}: {', '.join(map(str, levels[:10]))}).")
        shares = [check_number(value, "expected") for value in expected]
    else:
        raise AnalysisError("invalid_option", "expected must be None, a list of numbers or a "
                            "mapping {category: number}.")
    k = len(levels)
    if k < 2:
        raise AnalysisError("no_variation", f"'{y}' has a single category, so there is nothing "
                            "to test.")
    if any(value <= 0 for value in shares):
        raise AnalysisError("invalid_option", "Expected frequencies must be positive.")
    total = float(observed.sum())
    share = torch.tensor(shares, dtype=FLOAT)
    fitted = share / share.sum() * total
    residual = observed - fitted
    statistic = float((residual * residual / fitted).sum())
    df = k - 1
    small = int((fitted < 5).sum())
    attrs: dict[str, Any] = {"n": total if weights is not None else int(total), "k": k,
                             "statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df),
                             "min_expected": float(fitted.min()), "cells_below_5": small,
                             "distribution": "chi2", "n_dropped": dropped,
                             "label": "Chi-square goodness-of-fit test"}
    if small:
        attrs["notes"] = [f"{small} cell(s) ({100.0 * small / k:.1f}%) have expected "
                          f"frequencies less than 5. The minimum expected frequency is "
                          f"{float(fitted.min()):.3g}."]
    frequencies = table({"observed": observed.tolist(), "expected": fitted.tolist(),
                         "residual": residual.tolist(),
                         "std_residual": (residual / fitted.sqrt()).tolist()},
                        index=[str(level) for level in levels])
    tests = table([[statistic, df, attrs["p_value"]]], columns=["statistic", "df", "p_value"],
                  index=["chi2"])
    return TableSet({"frequencies": frequencies, "tests": tests},
                    title=f"Chi-square goodness of fit: {y}", **attrs)
