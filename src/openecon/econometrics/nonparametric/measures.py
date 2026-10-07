"""Measures of association for a two-way table of counts (float64 tensors).

Every measure theta is a function of the cell counts f_ij that is homogeneous
of degree zero (it depends on the proportions only). For such a function the
delta method gives the asymptotic standard error that does not assume
independence -- "ASE1" in the SPSS CROSSTABS output -- as

    ASE1(theta)^2 = sum_ij f_ij (d theta / d f_ij)^2 ,

because sum_ij f_ij d theta / d f_ij = 0 by Euler's theorem. The functions
below evaluate the analytic partial derivatives; this reproduces the closed
forms printed in the SPSS Statistics Algorithms (CROSSTABS) and SAS PROC FREQ
documentation (Goodman and Kruskal 1972; Brown and Benedetti 1977; Fleiss,
Cohen and Everitt 1969). The standard error under independence ("ASE0") is
used only for the approximate T = theta / ASE0.

Notation: f [R, C] counts, r_i row totals, c_j column totals, W the total.
C_ij (D_ij) is the number of observations concordant (discordant) with cell
(i, j); P = sum f_ij C_ij and Q = sum f_ij D_ij are twice the numbers of
concordant and discordant pairs. All of this is O(R C) through cumulative
sums; nothing is computed per observation.

Each function returns ``{name: (value, ase1, t, p)}`` rows; entries that are
not defined are None.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.distributions import chi2_sf, normal_isf, t_sf
from openecon.econometrics.nonparametric.common import FLOAT, normal_two_sided

Row = tuple[float | None, float | None, float | None, float | None]


def _lower_left(counts: Tensor) -> Tensor:
    """sum of counts strictly above and strictly to the left of every cell."""
    padded = torch.zeros((counts.shape[0] + 1, counts.shape[1] + 1), dtype=FLOAT)
    padded[1:, 1:] = counts.cumsum(0).cumsum(1)
    return padded[:-1, :-1]


def concordance(counts: Tensor) -> tuple[Tensor, Tensor]:
    """(C_ij, D_ij): counts concordant and discordant with each cell."""
    flipped_columns = counts.flip(1)
    concordant = _lower_left(counts) + _lower_left(counts.flip(0, 1)).flip(0, 1)
    discordant = _lower_left(flipped_columns).flip(1) + _lower_left(counts.flip(0)).flip(0)
    return concordant, discordant


def _ase(counts: Tensor, gradient: Tensor) -> float:
    return math.sqrt(max(0.0, float((counts * gradient * gradient).sum())))


def _row(value: float, ase1: float | None, ase0: float | None) -> Row:
    if ase0 is None or not ase0 > 0.0:
        return value, ase1, None, None
    t = value / ase0
    return value, ase1, t, normal_two_sided(t)


def ordinal_measures(counts: Tensor) -> dict[str, Row]:
    """gamma, Kendall's tau-b and tau-c, Somers' d (symmetric, row and column dependent).

    tau-b = (P - Q) / sqrt(D_r D_c), tau-c = q (P - Q) / (W^2 (q - 1)) with
    q = min(R, C), gamma = (P - Q) / (P + Q), d(C|R) = (P - Q) / D_r,
    d(R|C) = (P - Q) / D_c, symmetric d = 2 (P - Q) / (D_r + D_c), where
    D_r = W^2 - sum r_i^2 and D_c = W^2 - sum c_j^2. Under independence
    Var(P - Q) is estimated by 4 [sum f_ij (C_ij - D_ij)^2 - (P - Q)^2 / W].
    """
    total = float(counts.sum())
    rows, columns = counts.sum(1), counts.sum(0)
    concordant, discordant = concordance(counts)
    p, q = float((counts * concordant).sum()), float((counts * discordant).sum())
    d_r = total * total - float((rows * rows).sum())
    d_c = total * total - float((columns * columns).sum())
    difference = concordant - discordant
    squares = float((counts * difference * difference).sum())
    spread = squares - (p - q) ** 2 / total
    null = math.sqrt(spread) if spread > 1e-12 * squares else 0.0     # 0: no null variance
    out: dict[str, Row] = {}
    not_row, not_column = (total - rows)[:, None], (total - columns)[None, :]
    if p + q > 0:
        gradient = 4.0 * (q * concordant - p * discordant) / (p + q) ** 2
        out["gamma"] = _row((p - q) / (p + q), _ase(counts, gradient), 2.0 * null / (p + q))
    if d_r > 0 and d_c > 0:
        root = math.sqrt(d_r * d_c)
        tau_b = (p - q) / root
        gradient = (2.0 * root * difference - tau_b * (not_row * d_c + not_column * d_r)) \
            / (d_r * d_c)
        out["kendall_tau_b"] = _row(tau_b, _ase(counts, gradient), 2.0 * null / root)
        m = min(counts.shape)
        scale = 2.0 * m / ((m - 1.0) * total * total)
        out["kendall_tau_c"] = _row(scale * (p - q) / 2.0, scale * null, scale * null)
        both = d_r + d_c
        gradient = 4.0 * (difference * both - (p - q) * (not_row + not_column)) / both ** 2
        out["somers_d_symmetric"] = _row(2.0 * (p - q) / both, _ase(counts, gradient),
                                         4.0 * null / both)
        gradient = 2.0 * (d_c * difference - (p - q) * not_column) / d_c ** 2
        out["somers_d_row"] = _row((p - q) / d_c, _ase(counts, gradient), 2.0 * null / d_c)
        gradient = 2.0 * (d_r * difference - (p - q) * not_row) / d_r ** 2
        out["somers_d_column"] = _row((p - q) / d_r, _ase(counts, gradient), 2.0 * null / d_r)
    return out


def _lambda_column(counts: Tensor) -> tuple[Row, Tensor, Tensor, float, float]:
    """lambda with the column variable dependent, and its pieces for the symmetric form."""
    total = float(counts.sum())
    columns = counts.sum(0)
    modal_column = int(torch.argmax(columns))
    largest = float(columns[modal_column])
    row_modes = torch.argmax(counts, dim=1)
    is_row_mode = torch.zeros_like(counts)
    is_row_mode[torch.arange(counts.shape[0]), row_modes] = 1.0
    is_modal_column = torch.zeros_like(counts)
    is_modal_column[:, modal_column] = 1.0
    explained = float((counts * is_row_mode).sum())
    rest = total - largest
    if rest <= 0:
        return (None, None, None, None), is_row_mode, is_modal_column, explained, largest
    value = (explained - largest) / rest
    gradient = ((is_row_mode - is_modal_column) * rest
                - (explained - largest) * (1.0 - is_modal_column)) / rest ** 2
    spread = float((counts * (is_row_mode - is_modal_column) ** 2).sum()) \
        - (explained - largest) ** 2 / total
    ase0 = math.sqrt(max(0.0, spread)) / rest
    return _row(value, _ase(counts, gradient), ase0), is_row_mode, is_modal_column, explained, \
        largest


def nominal_measures(counts: Tensor, chi2: float, likelihood_ratio: float) -> dict[str, Row]:
    """phi, Cramer's V, contingency coefficient, lambda, Goodman-Kruskal tau, uncertainty.

    phi = sqrt(chi2 / W) (signed for a 2x2 table, where it is the Pearson
    correlation), V = sqrt(chi2 / (W (q - 1))), C = sqrt(chi2 / (chi2 + W));
    their p-value is that of Pearson's chi2.

    lambda(C|R) = (sum_i max_j f_ij - max_j c_j) / (W - max_j c_j); Goodman
    and Kruskal's tau(C|R) = (W sum_ij f_ij^2 / r_i - sum_j c_j^2) / (W^2 -
    sum_j c_j^2), tested through (W - 1)(C - 1) tau ~ chi2((R-1)(C-1)); the
    uncertainty coefficient U(C|R) = (H(R) + H(C) - H(RC)) / H(C), H the
    entropy, with the p-value of the likelihood-ratio chi2.
    "_column" means the column variable is the dependent one.
    """
    total = float(counts.sum())
    r, c = counts.shape
    rows, columns = counts.sum(1), counts.sum(0)
    df = (r - 1) * (c - 1)
    p_chi2 = chi2_sf(chi2, df) if df > 0 else None
    out: dict[str, Row] = {}
    phi = math.sqrt(chi2 / total)
    if r == 2 and c == 2:
        determinant = float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0])
        phi = math.copysign(phi, determinant) if determinant != 0 else 0.0
    out["phi"] = (phi, None, None, p_chi2)
    out["cramers_v"] = (math.sqrt(chi2 / (total * (min(r, c) - 1.0))), None, None, p_chi2)
    out["contingency_coefficient"] = (math.sqrt(chi2 / (chi2 + total)), None, None, p_chi2)

    column_row, a_c, b_c, explained_c, largest_c = _lambda_column(counts)
    row_row, a_r, b_r, explained_r, largest_r = _lambda_column(counts.T)
    a_r, b_r = a_r.T, b_r.T
    denominator = 2.0 * total - largest_c - largest_r
    if denominator > 0:
        numerator = explained_c + explained_r - largest_c - largest_r
        d_a, d_b = a_c + a_r, b_c + b_r
        gradient = ((d_a - d_b) * denominator - numerator * (2.0 - d_b)) / denominator ** 2
        spread = float((counts * (d_a - d_b) ** 2).sum()) - numerator ** 2 / total
        out["lambda_symmetric"] = _row(numerator / denominator, _ase(counts, gradient),
                                       math.sqrt(max(0.0, spread)) / denominator)
    if row_row[0] is not None:
        out["lambda_row"] = row_row
    if column_row[0] is not None:
        out["lambda_column"] = column_row

    for name, table, dependent in (("goodman_kruskal_tau_row", counts.T, r),
                                   ("goodman_kruskal_tau_column", counts, c)):
        given, target = table.sum(1), table.sum(0)
        delta = total * total - float((target * target).sum())
        if delta <= 0:
            continue
        within = (table * table / given[:, None]).sum(1)              # sum_j f_ij^2 / r_i
        nu = total * float(within.sum()) - float((target * target).sum())
        d_nu = float(within.sum()) + total * (2.0 * table / given[:, None]
                                              - (within / given)[:, None]) - 2.0 * target[None, :]
        d_delta = 2.0 * total - 2.0 * target[None, :]
        gradient = (d_nu * delta - nu * d_delta) / delta ** 2
        value = nu / delta
        p = chi2_sf((total - 1.0) * (dependent - 1.0) * value, df) if df > 0 else None
        out[name] = (value, _ase(table, gradient), None, p)

    def entropy(v: Tensor) -> float:
        return -float(torch.xlogy(v / total, v / total).sum())

    h_r, h_c, h_rc = entropy(rows), entropy(columns), entropy(counts)
    mutual = h_r + h_c - h_rc
    positive = counts > 0
    safe = torch.where(positive, counts, torch.ones_like(counts))
    log_ratio = torch.log(rows[:, None] * columns[None, :] / (total * safe))
    d_mutual = torch.where(positive, -(log_ratio + mutual) / total, torch.zeros_like(counts))
    d_hr = -(torch.log(rows / total) + h_r)[:, None] / total
    d_hc = -(torch.log(columns / total) + h_c)[None, :] / total
    squares = float((counts * torch.where(positive, log_ratio,
                                          torch.zeros_like(counts)) ** 2).sum())
    spread = squares - total * mutual * mutual
    null = math.sqrt(spread) if spread > 1e-12 * squares else 0.0     # 0: perfect association
    p_lr = chi2_sf(likelihood_ratio, df) if df > 0 else None

    def uncertainty(denominator: float, d_denominator: Tensor, factor: float) -> Row:
        value = factor * mutual / denominator
        gradient = factor * (d_mutual * denominator - mutual * d_denominator) / denominator ** 2
        ase0 = factor * null / (total * denominator)
        t = value / ase0 if ase0 > 0 else None
        return value, _ase(counts, gradient), t, p_lr

    if h_r + h_c > 0:
        out["uncertainty_symmetric"] = uncertainty(h_r + h_c, d_hr + d_hc, 2.0)
    if h_r > 0:
        out["uncertainty_row"] = uncertainty(h_r, d_hr.expand_as(counts), 1.0)
    if h_c > 0:
        out["uncertainty_column"] = uncertainty(h_c, d_hc.expand_as(counts), 1.0)
    return out


def _t_row(value: float, ase1: float, total: float) -> Row:
    if total <= 2 or abs(value) >= 1.0:
        return value, ase1, None, None
    t = value * math.sqrt((total - 2.0) / (1.0 - value * value))
    return value, ase1, t, min(1.0, 2.0 * t_sf(abs(t), total - 2.0))


def correlations(counts: Tensor, row_scores: Tensor, column_scores: Tensor) -> dict[str, Row]:
    """Pearson's r for the given scores and Spearman's rho (midrank scores).

    Both are tested with t = r sqrt((W - 2) / (1 - r^2)) on W - 2 degrees of
    freedom. The Spearman ASE1 accounts for the dependence of the rank scores
    on the margins (Brown and Benedetti 1977).
    """
    total = float(counts.sum())
    rows, columns = counts.sum(1), counts.sum(0)
    out: dict[str, Row] = {}

    def pieces(x: Tensor, y: Tensor):
        x = x - float((rows * x).sum()) / total
        y = y - float((columns * y).sum()) / total
        sxx, syy = float((rows * x * x).sum()), float((columns * y * y).sum())
        if sxx <= 0 or syy <= 0:
            return None
        root = math.sqrt(sxx * syy)
        r = float((x[:, None] * counts * y[None, :]).sum()) / root
        fixed = x[:, None] * y[None, :] / root \
            - 0.5 * r * ((x * x)[:, None] / sxx + (y * y)[None, :] / syy)
        return x, y, sxx, syy, root, r, fixed

    pearson = pieces(row_scores, column_scores)
    if pearson is not None:
        out["pearson_r"] = _t_row(pearson[5], _ase(counts, pearson[6]), total)
    row_ranks = rows.cumsum(0) - rows / 2.0
    column_ranks = columns.cumsum(0) - columns / 2.0
    spearman = pieces(row_ranks, column_ranks)
    if spearman is not None:
        x, y, sxx, syy, root, r, fixed = spearman
        g_x = (counts * y[None, :]).sum(1) / root - r * rows * x / sxx
        g_y = (counts * x[:, None]).sum(0) / root - r * columns * y / syy
        after_x = g_x.flip(0).cumsum(0).flip(0) - g_x / 2.0          # sum_{i > a} g + g_a / 2
        after_y = g_y.flip(0).cumsum(0).flip(0) - g_y / 2.0
        gradient = fixed + after_x[:, None] + after_y[None, :]
        out["spearman_rho"] = _t_row(r, _ase(counts, gradient), total)
    return out


def kappa(counts: Tensor, weighting: str = "none") -> Row:
    """Cohen's kappa of a square table; ``weighting`` is "none", "linear" or "quadratic".

    kappa = (p_o - p_e) / (1 - p_e) with p_o = sum w_ij f_ij / W and
    p_e = sum w_ij r_i c_j / W^2; agreement weights w_ij = 1 - |i - j| / (K - 1)
    (linear) or 1 - (i - j)^2 / (K - 1)^2 (quadratic). ASE1 is the delta-method
    error; z = kappa / ASE0 with the null error of Fleiss, Cohen and Everitt
    (1969).
    """
    k = counts.shape[0]
    total = float(counts.sum())
    rows, columns = counts.sum(1), counts.sum(0)
    index = torch.arange(k, dtype=FLOAT)
    distance = (index[:, None] - index[None, :]).abs() / max(k - 1.0, 1.0)
    if weighting == "linear":
        w = 1.0 - distance
    elif weighting == "quadratic":
        w = 1.0 - distance * distance
    else:
        w = torch.eye(k, dtype=FLOAT)
    agree = float((w * counts).sum())
    chance = float((w * torch.outer(rows, columns)).sum())
    delta = total * total - chance
    if delta <= 0:
        return None, None, None, None
    nu = total * agree - chance
    margin = (w @ columns)[:, None] + (rows @ w)[None, :]
    gradient = ((agree + total * w - margin) * delta - nu * (2.0 * total - margin)) / delta ** 2
    p_e = chance / (total * total)
    mean_weights = margin / total
    spread = float((torch.outer(rows, columns) / (total * total) * (w - mean_weights) ** 2).sum())
    ase0 = math.sqrt(max(0.0, spread - p_e * p_e) / total) / (1.0 - p_e)
    return _row(nu / delta, _ase(counts, gradient), ase0)


def risk_2x2(counts: Tensor, alpha: float) -> dict[str, tuple[float, float, float]]:
    """Odds ratio and the two column-wise relative risks of a 2x2 table with log intervals.

    OR = ad / (bc), se(ln OR) = sqrt(1/a + 1/b + 1/c + 1/d) (Woolf). For column
    j the relative risk is the row-1 share of that column's outcome over the
    row-2 share: RR_1 = [a / (a + b)] / [c / (c + d)] with
    se(ln RR_1) = sqrt(b / (a (a + b)) + d / (c (c + d))). Entries with a zero
    cell in the formula are omitted.
    """
    a, b = float(counts[0, 0]), float(counts[0, 1])
    c, d = float(counts[1, 0]), float(counts[1, 1])
    z = normal_isf(alpha / 2.0)
    out: dict[str, tuple[float, float, float]] = {}

    def interval(value: float, se: float) -> tuple[float, float, float]:
        return value, value * math.exp(-z * se), value * math.exp(z * se)

    if min(a, b, c, d) > 0:
        out["odds_ratio"] = interval(a * d / (b * c), math.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    if a > 0 and c > 0:
        out["risk_ratio_column_1"] = interval((a / (a + b)) / (c / (c + d)),
                                              math.sqrt(b / (a * (a + b)) + d / (c * (c + d))))
    if b > 0 and d > 0:
        out["risk_ratio_column_2"] = interval((b / (a + b)) / (d / (c + d)),
                                              math.sqrt(a / (b * (a + b)) + c / (d * (c + d))))
    return out


def mantel_haenszel(strata: Tensor, alpha: float) -> dict[str, object]:
    """Cochran-Mantel-Haenszel analysis of K 2x2 tables ([K, 2, 2] counts).

    * Common odds ratio psi = sum(a d / n) / sum(b c / n) with the
      Robins-Breslow-Greenland variance of ln psi, its log-scale interval and
      z = ln psi / se (the "Asymp. Sig." of SPSS's common odds ratio table).
    * Test of conditional independence: with E(a_k) = r1 c1 / n and
      Var(a_k) = r1 r2 c1 c2 / (n^2 (n - 1)),
      CMH = (sum (a_k - E))^2 / sum Var, also with |.| - 1/2 (the
      "Mantel-Haenszel" line of SPSS); Cochran's statistic uses the binomial
      variance r1 r2 c1 c2 / n^3. Each is chi2(1).
    * Homogeneity of the odds ratios: Breslow-Day chi2 = sum (a_k - A_k)^2 /
      V_k on K - 1 df, A_k the fitted count under the common odds ratio psi
      and V_k its variance; Tarone's correction subtracts
      (sum (a_k - A_k))^2 / sum V_k.

    Strata with a zero margin carry no information and are left out.
    """
    a, b, c, d = strata[:, 0, 0], strata[:, 0, 1], strata[:, 1, 0], strata[:, 1, 1]
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    keep = (r1 > 0) & (r2 > 0) & (c1 > 0) & (c2 > 0) & (n > 1)
    used = int(keep.sum())
    out: dict[str, object] = {"strata": int(strata.shape[0]), "strata_used": used}
    if used == 0:
        return out
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]
    r1, r2, c1, c2 = r1[keep], r2[keep], c1[keep], c2[keep]
    expected = r1 * c1 / n
    product = r1 * r2 * c1 * c2
    variance = float((product / (n * n * (n - 1.0))).sum())
    binomial = float((product / n ** 3).sum())
    excess = float((a - expected).sum())
    if variance > 0:
        plain = excess ** 2 / variance
        corrected = max(0.0, abs(excess) - 0.5) ** 2 / variance
        cochran = excess ** 2 / binomial
        out["tests"] = {"cmh": (plain, 1, chi2_sf(plain, 1)),
                        "cmh_continuity": (corrected, 1, chi2_sf(corrected, 1)),
                        "cochran": (cochran, 1, chi2_sf(cochran, 1))}
    big_r, big_s = float((a * d / n).sum()), float((b * c / n).sum())
    if big_r > 0 and big_s > 0:
        psi = big_r / big_s
        p_k, q_k = (a + d) / n, (b + c) / n
        r_k, s_k = a * d / n, b * c / n
        log_variance = float((p_k * r_k).sum()) / (2.0 * big_r ** 2) \
            + float((p_k * s_k + q_k * r_k).sum()) / (2.0 * big_r * big_s) \
            + float((q_k * s_k).sum()) / (2.0 * big_s ** 2)
        se = math.sqrt(log_variance)
        z = normal_isf(alpha / 2.0)
        wald = math.log(psi) / se if se > 0 else None
        out["odds_ratio"] = (psi, se, psi * math.exp(-z * se), psi * math.exp(z * se), wald,
                             normal_two_sided(wald) if wald is not None else None)
        # Fitted (1, 1) cell under the common odds ratio: the root of
        # (1 - psi) A^2 + [n - r1 - c1 + psi (r1 + c1)] A - psi r1 c1 = 0 inside its range.
        quadratic = 1.0 - psi
        linear = n - r1 - c1 + psi * (r1 + c1)
        constant = -psi * r1 * c1
        if abs(quadratic) < 1e-12:
            fitted = -constant / linear
        else:
            root = torch.sqrt((linear * linear - 4.0 * quadratic * constant).clamp_min(0.0))
            # the numerically stable form of the root that lies in [max(0, r1+c1-n), min(r1, c1)]
            fitted = 2.0 * (-constant) / (linear + root)
        cells = torch.stack([fitted, r1 - fitted, c1 - fitted, n - r1 - c1 + fitted])
        if used > 1 and bool((cells > 0).all()):
            v = 1.0 / (1.0 / cells).sum(0)
            breslow_day = float(((a - fitted) ** 2 / v).sum())
            tarone = max(0.0, breslow_day - float((a - fitted).sum()) ** 2 / float(v.sum()))
            tests = out.setdefault("tests", {})
            tests["breslow_day"] = (breslow_day, used - 1, chi2_sf(breslow_day, used - 1))
            tests["tarone"] = (tarone, used - 1, chi2_sf(tarone, used - 1))
    return out
