"""Discriminant analysis: ``oe.discrim`` and ``oe.discrim_predict``.

SPSS ``DISCRIMINANT``; Stata ``discrim lda`` / ``discrim qda`` / ``candisc`` with
``estat classtable`` (resubstitution and leave-one-out).

Notation. G groups with n_g observations (n in total), p variables, group means
m_g, W = sum_g sum_i (x_i - m_g)(x_i - m_g)' the pooled within-groups SSCP
matrix, S_w = W / (n - G), T the total SSCP matrix and B = T - W.

Canonical discriminant functions. The eigenvalues l_1 >= ... >= l_q,
q = min(p, G - 1), of W^{-1}B and eigenvectors v_k scaled to v_k' S_w v_k = 1
(computed from the symmetric matrix L^{-1} B L^{-T}, W = L L', never from an
explicit inverse) give the unstandardized coefficients v_k with constant
-v_k' xbar, the standardized coefficients v_jk sqrt(S_w,jj), the structure matrix
(pooled within-groups correlations between variables and functions), the
canonical correlations sqrt(l_k / (1 + l_k)) and Bartlett's chi-square tests of
functions k..q: -(n - 1 - (p + G)/2) ln prod_{j >= k} 1/(1 + l_j) with
(p - k + 1)(G - k) degrees of freedom.

Classification. Linear (pooled covariance): an observation is assigned to the
group with the largest Fisher classification function
c_g(x) = x' S_w^{-1} m_g - m_g' S_w^{-1} m_g / 2 + ln prior_g; posterior
probabilities are proportional to exp(c_g(x)). Quadratic (separate covariances
S_g): c_g(x) = -ln|S_g| / 2 - (x - m_g)' S_g^{-1} (x - m_g) / 2 + ln prior_g.

Leave-one-out classification removes each observation from the estimation of its
own group's mean and of the covariance matrix. It is computed in closed form
from rank-one (Sherman-Morrison) updates of the whitened distances, in O(n G p),
without refitting.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.multivariate import common as c

_HINT = ("a variable is constant within every group or a linear combination of the others, or "
         "there are too few observations. Remove the redundant variable.")


def _priors(priors: Any, counts: Tensor, labels: list[Any]) -> Tensor:
    groups = len(labels)
    if isinstance(priors, str):
        c.check_choice(priors, "priors", ("equal", "proportional"))
        if priors == "equal":
            return torch.full((groups,), 1.0 / groups, dtype=c.FLOAT)
        return counts / counts.sum()
    try:
        values = torch.as_tensor(priors, dtype=c.FLOAT).flatten()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_option", "priors must be 'equal', 'proportional' or a list "
                            "of one probability per group.") from exc
    if values.numel() != groups or not bool(torch.isfinite(values).all()) \
            or bool((values <= 0).any()) or abs(float(values.sum()) - 1.0) > 1e-8:
        raise AnalysisError("invalid_option", f"priors must list {groups} positive probabilities "
                            f"that sum to 1, in the order of the groups: "
                            f"{', '.join(map(str, labels))}.")
    return values / values.sum()


def _group_blocks(x: Tensor, codes: Tensor, groups: int) -> list[Tensor]:
    """The rows of each group (one sort, then slices)."""
    order = torch.argsort(codes, stable=True)
    counts = torch.bincount(codes, minlength=groups).tolist()
    ordered = x[order]
    blocks, start = [], 0
    for count in counts:
        blocks.append(ordered[start:start + count])
        start += count
    return blocks


def box_m(log_dets: list[float], counts: list[int], log_pooled: float, p: int
          ) -> dict[str, float | None]:
    """Box's M test of equal group covariance matrices (Box 1949).

    M = (n - G) ln|S_w| - sum (n_g - 1) ln|S_g|; chi2 = M (1 - c1) with
    df1 = p (p + 1)(G - 1)/2 and the F approximation with df2 = (df1 + 2)/|c2 - c1^2|
    (the statistic SPSS prints), where
    c1 = [sum 1/(n_g - 1) - 1/(n - G)] (2p^2 + 3p - 1) / (6 (p + 1)(G - 1)) and
    c2 = [sum 1/(n_g - 1)^2 - 1/(n - G)^2] (p - 1)(p + 2) / (6 (G - 1)).
    """
    groups = len(counts)
    total = sum(counts)
    df = [count - 1.0 for count in counts]
    statistic = max((total - groups) * log_pooled
                    - sum(d * value for d, value in zip(df, log_dets, strict=True)), 0.0)
    c1 = (sum(1.0 / d for d in df) - 1.0 / (total - groups)) * (2 * p * p + 3 * p - 1) \
        / (6.0 * (p + 1) * (groups - 1))
    c2 = (sum(1.0 / (d * d) for d in df) - 1.0 / (total - groups) ** 2) * (p - 1) * (p + 2) \
        / (6.0 * (groups - 1))
    df1 = p * (p + 1) * (groups - 1) / 2.0
    chi2 = statistic * (1.0 - c1)
    f = df2 = None
    gap = c2 - c1 * c1
    if gap > 0:
        df2 = (df1 + 2.0) / gap
        f = statistic * (1.0 - c1 - df1 / df2) / df1
    elif gap < 0:
        df2 = (df1 + 2.0) / -gap
        bound = df2 / (1.0 - c1 + 2.0 / df2)
        f = df2 * statistic / (df1 * (bound - statistic)) if bound > statistic else None
    return {"statistic": statistic, "f": f, "df1": df1, "df2": df2,
            "p_value": c.f_upper(f, df1, df2) if df2 else None, "chi2": chi2,
            "chi2_p_value": c.chi2_upper(chi2, df1)}


def _softmax_rows(scores: Tensor) -> Tensor:
    shifted = scores - scores.max(1, keepdim=True).values
    weights = torch.exp(shifted)
    return weights / weights.sum(1, keepdim=True)


def _linear_scores(x: Tensor, means: Tensor, factor: Tensor, df: float, log_prior: Tensor,
                   codes: Tensor | None = None, counts: Tensor | None = None) -> Tensor:
    """[n, G] scores -D^2/2 + ln prior under the pooled covariance W / df.

    With ``codes`` the scores are leave-one-out: observation i is removed from its
    group mean and from W (df is then n - 1 - G).
    """
    zx = torch.linalg.solve_triangular(factor, x.T, upper=False).T
    zm = torch.linalg.solve_triangular(factor, means.T, upper=False).T
    diff_sq = (zx.square().sum(1)[:, None] - 2.0 * (zx @ zm.T) + zm.square().sum(1)).clamp_min(0)
    if codes is None:
        return -0.5 * df * diff_sq + log_prior
    own = zx - zm[codes]
    a = own.square().sum(1)                                     # d' W^{-1} d
    ratio = (counts / (counts - 1.0))[codes]                    # n_g / (n_g - 1)
    remaining = 1.0 - ratio * a
    if bool((remaining <= 1e-12).any()):
        raise AnalysisError("singular_matrix", "Leave-one-out classification is undefined: "
                            "removing an observation makes the pooled covariance matrix "
                            "singular. Use more observations or fewer variables.")
    # cross_ih = (x_i - m_h)' W^{-1} (x_i - m_g(i))
    cross = (zx.square().sum(1)[:, None] - zx @ zm.T - (zx * zm[codes]).sum(1)[:, None]
             + (zm @ zm.T)[codes])
    distance = diff_sq + ratio[:, None] * cross.square() / remaining[:, None]
    own_distance = ratio.square() * a / remaining
    distance[torch.arange(x.shape[0]), codes] = own_distance
    return -0.5 * df * distance + log_prior


def _quadratic_scores(x: Tensor, means: Tensor, factors: list[Tensor], counts: Tensor,
                      log_prior: Tensor, codes: Tensor | None = None) -> Tensor:
    """[n, G] scores -ln|S_g|/2 - D_g^2/2 + ln prior with S_g = W_g / (n_g - 1).

    With ``codes`` the own-group term of each observation is leave-one-out.
    """
    n, p = x.shape
    out = torch.empty((n, len(factors)), dtype=x.dtype)
    for g, factor in enumerate(factors):
        ng = float(counts[g])
        z = torch.linalg.solve_triangular(factor, (x - means[g]).T, upper=False).T
        a = z.square().sum(1)
        log_det_w = 2.0 * float(torch.log(factor.diagonal()).sum())
        score = -0.5 * (log_det_w - p * math.log(ng - 1.0)) - 0.5 * (ng - 1.0) * a
        if codes is not None:
            ratio = ng / (ng - 1.0)
            remaining = 1.0 - ratio * a
            mine = codes == g
            if bool((remaining[mine] <= 1e-12).any()):
                raise AnalysisError("singular_matrix", "Leave-one-out classification is "
                                    "undefined: removing an observation makes a group "
                                    "covariance matrix singular.")
            safe = torch.where(mine, remaining, torch.ones_like(remaining))
            loo = -0.5 * (log_det_w + torch.log(safe) - p * math.log(ng - 2.0)) \
                - 0.5 * (ng - 2.0) * ratio * ratio * a / safe
            score = torch.where(mine, loo, score)
        out[:, g] = score + log_prior[g]
    return out


def _classification(scores: Tensor, codes: Tensor, counts: Tensor, labels: list[Any]
                    ) -> tuple[pd.DataFrame, float]:
    groups = len(labels)
    predicted = scores.argmax(1)
    cells = torch.bincount(codes * groups + predicted, minlength=groups * groups)
    cells = cells.reshape(groups, groups).to(c.FLOAT)
    correct = cells.diagonal()
    rows = torch.cat([cells, counts[:, None], (100.0 * correct / counts)[:, None]], dim=1)
    rate = 100.0 * float(correct.sum()) / float(counts.sum())
    out = c.frame(rows, columns=[*map(str, labels), "n", "percent_correct"], index=labels)
    for name in out.columns[:-1]:
        out[name] = out[name].astype("int64")
    out.index.name = "actual"
    return out, rate


@c.procedure
def discrim(data: Any, group: str, columns: list[str], *, method: str = "lda",
            priors: Any = "equal", loo: bool = False, missing: str = "drop",
            weights: str | None = None, weight_type: str = "fweight") -> TableSet:
    """Linear and quadratic discriminant analysis with canonical functions.

    ``method="lda"`` assumes a common covariance matrix (the pooled within-groups
    S_w) and reports Fisher's canonical discriminant functions and linear
    classification functions; ``method="qda"`` classifies with the separate group
    covariance matrices. See the module documentation for the formulas.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    group : the grouping column (any scalar labels; at least two groups).
    columns : one or more numeric discriminating variables.
    method : ``"lda"`` (default) or ``"qda"``.
    priors : prior probabilities: ``"equal"`` (default, as SPSS and Stata),
        ``"proportional"`` (group shares of the sample; SPSS ``/PRIORS SIZE``) or a
        list of probabilities in the order of the sorted group labels.
    loo : also report the leave-one-out classification table (SPSS "cross-validated",
        Stata ``estat classtable, loo``).
    missing : ``"drop"`` (listwise deletion over the group and the variables) or
        ``"raise"``.
    weights : optional resident-data frequency-weight column. Exact nonnegative
        integers represent repeated observations without expanding rows. ``loo``
        removes one repeated copy, with the full-fit priors held fixed.
    weight_type : only ``"fweight"`` is supported. Other weight domains and
        weighted Dataset inputs are refused.

    Returns
    -------
    TableSet with

    * ``groups``: ``n`` and ``prior`` of each group; ``group_means``: the group means;
    * ``group_statistics``: n, mean and standard deviation of every variable by group
      and in total;
    * ``tests_of_equality``: for each variable Wilks' lambda w_jj / t_jj and the
      one-way ANOVA F with G - 1 and n - G degrees of freedom;
    * ``box_m``: Box's M test of equal covariance matrices, with ``log_determinants``
      (omitted, with a note, when a group covariance matrix is singular);
    * ``classification_table`` (and ``classification_table_loo``): actual groups in
      rows, predicted groups in columns, ``n`` and ``percent_correct``;

    and for ``method="lda"``

    * ``canonical_functions``: ``eigenvalue``, ``percent``, ``cumulative``,
      ``canonical_correlation`` and the test of functions k through q:
      ``wilks_lambda``, ``chi2``, ``df``, ``p_value``;
    * ``unstandardized_coefficients`` (with the ``Intercept``),
      ``standardized_coefficients``, ``structure_matrix`` and ``centroids`` (group
      means of the functions);
    * ``classification_functions``: Fisher's linear classification functions, one
      column per group, with the ``Intercept`` (includes ln prior);

    and for ``method="qda"``: ``group_covariances`` (used by ``oe.discrim_predict``).

    ``attrs``: ``n``, ``n_missing``, ``method``, ``group``, ``variables``, ``groups``
    (labels), ``priors``, ``percent_correct``, ``percent_correct_loo``, ``notes``.

    Sign convention: each canonical function is signed so that its standardized
    coefficient of largest absolute value is positive.

    Equivalent commands: SPSS ``DISCRIMINANT /GROUPS=g(1 3) /VARIABLES=x1 x2
    /STATISTICS=MEAN STDDEV UNIVF BOXM COEFF RAW TABLE CROSSVALID``; Stata
    ``discrim lda x1 x2, group(g)``, ``estat classtable, loo``, ``candisc``;
    ``discrim qda x1 x2, group(g)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"g": ["a"] * 5 + ["b"] * 5,
    ...         "x": [1.0, 1.5, 0.8, 1.2, 1.9, 3.1, 2.8, 3.5, 2.6, 3.3],
    ...         "y": [2.0, 2.4, 1.7, 2.9, 2.2, 1.0, 0.7, 1.4, 0.6, 1.1]}
    >>> result = oe.discrim(data, "g", ["x", "y"])
    >>> result.attrs["percent_correct"]
    100.0
    """
    c.check_choice(weight_type, "weight_type", ("fweight",))
    if weights is not None:
        from .discrim_options import frequency_discrim
        return frequency_discrim(data, group, columns, method=method, priors=priors,
                                 loo=loo, missing=missing, weights=weights)
    group = c.check_name(group, "group")
    names = c.name_list(columns, "columns", minimum=1)
    c.check_choice(method, "method", ("lda", "qda"))
    c.check_flag(loo, "loo")
    sample, _, dropped = c.select(data, [group, *names], numeric=names, missing=missing)
    codes, labels = c.group_codes(sample[group], group)
    groups = len(labels)
    if groups < 2:
        raise AnalysisError("invalid_groups", f"Discriminant analysis needs at least two groups, "
                            f"but '{group}' has {groups}.")
    raw = c.matrix(sample, names)
    n, p = raw.shape
    grand, total_sscp, x = c.moments(raw, names)
    counts = torch.bincount(codes, minlength=groups).to(c.FLOAT)
    means = torch.zeros((groups, p), dtype=c.FLOAT).index_add_(0, codes, x) / counts[:, None]
    deviation = x - means[codes]
    means = means + torch.zeros((groups, p), dtype=c.FLOAT).index_add_(0, codes, deviation) \
        / counts[:, None]
    deviation = x - means[codes]
    within = deviation.T @ deviation
    within = (within + within.T) / 2
    if n - groups < 1:
        raise AnalysisError("insufficient_observations", "Discriminant analysis needs more "
                            "observations than groups.")
    prior = _priors(priors, counts, labels)
    log_prior = torch.log(prior)
    if loo and bool((counts < (2 if method == "lda" else p + 2)).any()):
        raise AnalysisError("insufficient_observations", "Leave-one-out classification needs at "
                            f"least {2 if method == 'lda' else p + 2} observations in every "
                            "group.")
    notes: list[str] = []
    text = [str(item) for item in labels]

    # Descriptive tables.
    group_std = (torch.zeros((groups, p), dtype=c.FLOAT).index_add_(0, codes, deviation.square())
                 / (counts - 1.0).clamp_min(1.0)[:, None]).sqrt()
    statistics = []
    for g, name in enumerate(labels):
        for j, variable in enumerate(names):
            statistics.append([name, variable, int(counts[g]), float(means[g, j] + grand[j]),
                               float(group_std[g, j]) if counts[g] > 1 else None])
    total_std = (total_sscp.diagonal() / (n - 1)).sqrt()
    statistics += [["Total", variable, n, float(grand[j]), float(total_std[j])]
                   for j, variable in enumerate(names)]
    wilks = within.diagonal() / total_sscp.diagonal()
    df1, df2 = groups - 1, n - groups
    equality = []
    for j in range(p):
        lam = float(wilks[j])
        f = (1.0 - lam) / lam * df2 / df1 if lam > 0 else None
        equality.append([lam, f, df1, df2, c.f_upper(f, df1, df2)])
    group_table = c.frame(torch.stack([counts, prior], dim=1), columns=["n", "prior"],
                          index=labels)
    group_table["n"] = group_table["n"].astype("int64")
    tables = {
        "groups": group_table,
        "group_means": c.frame(means + grand, columns=names, index=labels),
        "group_statistics": c.frame(statistics, columns=["group", "variable", "n", "mean",
                                                         "std_dev"]),
        "tests_of_equality": c.frame(equality, columns=["wilks_lambda", "statistic", "df1",
                                                        "df2", "p_value"], index=names),
    }

    # Group covariance matrices (QDA and Box's M).
    blocks = _group_blocks(deviation, codes, groups)
    group_sscp = [block.T @ block for block in blocks]
    factors: list[Tensor] | None = []
    for sscp in group_sscp:
        try:
            factors.append(c.positive_definite(sscp, "group covariance matrix", _HINT))
        except AnalysisError:
            factors = None
            break
    if method == "qda" and factors is None:
        raise AnalysisError("singular_matrix", "A group covariance matrix is singular: "
                            "quadratic discriminant analysis needs more observations than "
                            "variables in every group and no redundant variable. Use "
                            "method='lda' or remove variables.")
    pooled_factor = c.positive_definite(within, "pooled within-groups covariance matrix", _HINT)
    if factors is not None and bool((counts > p).all()):
        log_dets = [2.0 * float(torch.log(f.diagonal()).sum()) - p * math.log(float(nc) - 1.0)
                    for f, nc in zip(factors, counts, strict=True)]
        log_pooled = 2.0 * float(torch.log(pooled_factor.diagonal()).sum()) \
            - p * math.log(n - groups)
        box = box_m(log_dets, [int(v) for v in counts.tolist()], log_pooled, p)
        tables["box_m"] = c.frame([list(box.values())], columns=list(box), index=["box_m"])
        tables["log_determinants"] = c.frame(
            [[value] for value in [*log_dets, log_pooled]], columns=["log_determinant"],
            index=[*labels, "Pooled within-groups"])
    else:
        notes.append("Box's M is not computed: a group covariance matrix is singular.")

    if method == "lda":
        sw = within / df2
        between = total_sscp - within
        half = torch.linalg.solve_triangular(pooled_factor, between, upper=False)
        inner = torch.linalg.solve_triangular(pooled_factor, half.T, upper=False)
        roots, vectors = torch.linalg.eigh((inner + inner.T) / 2)
        q = min(p, groups - 1)
        roots = roots.flip(0)[:q].clamp_min(0.0)
        raw_coef = torch.linalg.solve_triangular(pooled_factor.T, vectors.flip(1)[:, :q],
                                                 upper=True) * math.sqrt(df2)
        scale = sw.diagonal().sqrt()
        standardized = raw_coef * scale[:, None]
        pivot = standardized.abs().argmax(0)
        sign = torch.sign(standardized[pivot, torch.arange(q)])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        raw_coef, standardized = raw_coef * sign, standardized * sign
        functions = c.numbered("Function", q)
        log_terms = torch.log1p(roots)
        canonical, running = [], 0.0
        multiplier = n - 1.0 - (p + groups) / 2.0
        total_roots = float(roots.sum())
        for k in range(q):
            running += float(roots[k])
            lam = math.exp(-float(log_terms[k:].sum()))
            chi2 = -multiplier * math.log(lam) if multiplier > 0 and lam > 0 else None
            df = (p - k) * (groups - k - 1)
            canonical.append([
                float(roots[k]), 100.0 * float(roots[k]) / total_roots if total_roots else None,
                100.0 * running / total_roots if total_roots else None,
                math.sqrt(float(roots[k]) / (1.0 + float(roots[k]))), lam, chi2, df,
                c.chi2_upper(chi2, df)])
        constant = -(grand @ raw_coef)
        coefficients = torch.linalg.solve_triangular(
            pooled_factor.T, torch.linalg.solve_triangular(
                pooled_factor, (means + grand).T, upper=False), upper=True) * df2
        intercept = -0.5 * ((means + grand).T * coefficients).sum(0) + log_prior
        tables.update({
            "canonical_functions": c.frame(canonical, columns=[
                "eigenvalue", "percent", "cumulative", "canonical_correlation",
                "wilks_lambda", "chi2", "df", "p_value"], index=functions),
            "standardized_coefficients": c.frame(standardized, columns=functions, index=names),
            "structure_matrix": c.frame((sw @ raw_coef) / scale[:, None], columns=functions,
                                        index=names),
            "unstandardized_coefficients": c.frame(
                torch.cat([raw_coef, constant[None, :]]), columns=functions,
                index=[*names, "Intercept"]),
            "centroids": c.frame(means @ raw_coef, columns=functions, index=labels),
            "classification_functions": c.frame(
                torch.cat([coefficients, intercept[None, :]]), columns=text,
                index=[*names, "Intercept"]),
        })
        scores = _linear_scores(x, means, pooled_factor, df2, log_prior)
    else:
        rows = []
        for g, name in enumerate(labels):
            covariance = group_sscp[g] / (float(counts[g]) - 1.0)
            rows += [[name, variable, *covariance[j].tolist()]
                     for j, variable in enumerate(names)]
        tables["group_covariances"] = c.frame(rows, columns=["group", "variable", *names])
        scores = _quadratic_scores(x, means, factors, counts, log_prior)

    tables["classification_table"], rate = _classification(scores, codes, counts, labels)
    rate_loo = None
    if loo:
        if method == "lda":
            if n - 1 - groups < p:
                raise AnalysisError("insufficient_observations", "Leave-one-out classification "
                                    "needs n - 1 - G >= p.")
            loo_scores = _linear_scores(x, means, pooled_factor, df2 - 1.0, log_prior, codes,
                                        counts)
        else:
            loo_scores = _quadratic_scores(x, means, factors, counts, log_prior, codes)
        tables["classification_table_loo"], rate_loo = _classification(
            loo_scores, codes, counts, labels)
    title = "Linear" if method == "lda" else "Quadratic"
    return TableSet(
        tables, title=f"{title} discriminant analysis of {group} on {', '.join(names)}",
        procedure="discrim", n=n, n_missing=dropped, method=method, group=group,
        variables=names, groups=labels, priors=prior.tolist(), percent_correct=rate,
        percent_correct_loo=rate_loo, notes=notes, missing="listwise",
        sign_convention="largest absolute standardized coefficient of each function is positive")


@c.procedure
def discrim_summary(group_means: pd.DataFrame, group_covariances: Any, counts: Any, *,
                    method: str = "lda", priors: Any = "equal",
                    columns: list[str] | None = None, group: str = "group") -> TableSet:
    """Fit LDA/QDA from declared group means, sample covariances and integer counts.

    ``group_means`` is a DataFrame with unique group labels in its index and
    ordered variable names in its columns. ``group_covariances`` maps those labels
    to covariance matrices (labelled matrices must use the same variable order).
    Each covariance uses divisor ``count - 1``. ``counts`` is an aligned Series,
    a mapping by group label, or a positional integer vector in mean-index order.
    Counts and their sum must not exceed 2**53. Summary inputs are checked for
    covariance symmetry, positive semidefiniteness and feasible sample rank.

    LDA requires positive definite pooled within-group covariance; individual
    groups may be singular. QDA requires positive definite covariance in every
    group. Gaussian inference follows the same declared sample assumptions as
    ``discrim``. No training observations are constructed: training accuracy,
    confusion tables and leave-one-out classification are unavailable. Saved
    results can classify resident new rows through ``discrim_predict``.
    """
    from .discrim_options import summary_discrim
    return summary_discrim(group_means, group_covariances, counts, method=method,
                           priors=priors, columns=columns, group=group)


@c.procedure
def discrim_predict(result: TableSet, data: Any) -> pd.DataFrame:
    """Predicted group and posterior probabilities for the rows of ``data``.

    Parameters
    ----------
    result : the TableSet returned by ``oe.discrim``.
    data : rows to classify; must contain the discriminating variables. Rows with a
        missing value get missing predictions.

    Returns
    -------
    A table indexed like ``data`` with ``predicted`` (the group label with the
    largest posterior probability; the first such group on ties) and one column
    ``posterior_<label>`` per group. Posterior probabilities are
    prior_g f_g(x) / sum_h prior_h f_h(x) with normal densities f_g (common
    covariance for LDA, group covariances for QDA) and the priors of the analysis.

    Equivalent commands: Stata ``predict, classification`` / ``predict p*, pr``; SPSS
    ``/SAVE CLASS PROBS``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"g": ["a"] * 5 + ["b"] * 5,
    ...         "x": [1.0, 1.5, 0.8, 1.2, 1.9, 3.1, 2.8, 3.5, 2.6, 3.3],
    ...         "y": [2.0, 2.4, 1.7, 2.9, 2.2, 1.0, 0.7, 1.4, 0.6, 1.1]}
    >>> result = oe.discrim(data, "g", ["x", "y"])
    >>> list(oe.discrim_predict(result, {"x": [1.1, 3.0], "y": [2.1, 0.9]})["predicted"])
    ['a', 'b']
    """
    c.check_result(result, "discrim", "discrim_predict")
    if "discriminant_state" in result.attrs or result.attrs.get("input_kind") in {
            "resident_frequency_data", "group_summary"}:
        from .discrim_options import predict_options
        return predict_options(result, data)
    names = list(result.attrs["variables"])
    labels = list(result["groups"].index)
    frame = c.source(data)
    c.require_numeric(frame, names)
    frame = frame.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    x = c.matrix(frame.loc[keep], names)
    log_prior = torch.log(torch.as_tensor(
        result["groups"]["prior"].to_numpy(dtype="float64").copy()))
    means = torch.as_tensor(result["group_means"].loc[:, names].to_numpy(dtype="float64").copy())
    if result.attrs["method"] == "lda":
        functions = result["classification_functions"]
        weights = torch.as_tensor(functions.loc[names].to_numpy(dtype="float64").copy())
        constant = torch.as_tensor(functions.loc["Intercept"].to_numpy(dtype="float64").copy())
        scores = x @ weights + constant
    else:
        counts = torch.as_tensor(result["groups"]["n"].to_numpy(dtype="float64").copy())
        covariances = result["group_covariances"]
        p = len(names)
        factors = []
        for g in range(len(labels)):
            block = covariances.iloc[g * p:(g + 1) * p].loc[:, names].to_numpy(dtype="float64")
            sscp = torch.as_tensor(block.copy()) * (float(counts[g]) - 1.0)
            factors.append(torch.linalg.cholesky((sscp + sscp.T) / 2))
        scores = _quadratic_scores(x, means, factors, counts, log_prior)
    posterior = _softmax_rows(scores)
    predicted = scores.argmax(1).tolist()
    columns = [f"posterior_{item}" for item in labels]
    out = pd.DataFrame(index=keep.index, columns=["predicted", *columns], dtype=object)
    mask = keep.to_numpy()
    out.loc[mask, "predicted"] = pd.Series([labels[i] for i in predicted], dtype=object).to_numpy()
    out = table(out)
    for j, name in enumerate(columns):
        values = pd.Series(float("nan"), index=keep.index, dtype="float64")
        values.loc[mask] = posterior[:, j].numpy()
        out[name] = values
    return out
