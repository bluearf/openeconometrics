"""Independent oracles for the model-building family (oe.stepwise, oe.collin, oe.curvefit,
oe.tabstat).

Every expected number in this file is derived from first principles, never from the
implementation's formulas: stepwise selection is replayed by refitting every candidate model
with NumPy least squares (SVD) and auxiliary regressions for every tolerance (no sweep
operator, no correlation matrix); final models are checked against statsmodels; collinearity
diagnostics against an eigendecomposition of X'X; curve models against statsmodels OLS on the
transformed variables and, for ill-conditioned polynomials, against EXACT rational arithmetic;
summary statistics against explicit Python loops over expanded data, SciPy and statsmodels.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import openecon as oe

# ==== stepwise ==========================================================================


def _wls(y: np.ndarray, x: np.ndarray, w: np.ndarray | None) -> float:
    """Residual sum of squares of y on [1, x] by weighted least squares (NumPy SVD)."""
    design = np.column_stack([np.ones(len(y)), x]) if x.size else np.ones((len(y), 1))
    root = np.ones(len(y)) if w is None else np.sqrt(w)
    weighted = design * root[:, None]
    # Equilibrate the columns first: lstsq's default cutoff would otherwise discard a
    # column measured in units 1e-14 times smaller than another (e.g. x * 1e-6 next to 3e8).
    norms = np.sqrt((weighted ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    beta, *_ = np.linalg.lstsq(weighted / norms, y * root, rcond=None)
    return float(((root * y - (weighted / norms) @ beta) ** 2).sum())


class Replay:
    """SPSS / Stata stepwise selection replayed by brute-force refitting.

    ``terms`` maps a term to its columns (one column, or the indicators of a categorical
    predictor). Sums of squares follow Stata: analytic weights are normalized to sum to the
    number of rows; frequency weights count each row w times (n = sum w).
    """

    def __init__(self, frame, y, terms, *, weights=None, frequency=False, categorical=()):
        self.y = frame[y].to_numpy(float)
        self.names = list(terms)
        self.columns: dict[str, np.ndarray] = {}
        self.labels: dict[str, list[str]] = {}
        for term in terms:
            if term in categorical:
                levels = sorted(frame[term].unique())
                self.columns[term] = np.column_stack(
                    [(frame[term] == level).to_numpy(float) for level in levels[1:]])
                self.labels[term] = [f"{term}[{level}]" for level in levels[1:]]
            else:
                self.columns[term] = frame[[term]].to_numpy(float)
                self.labels[term] = [term]
        self.w = None if weights is None else frame[weights].to_numpy(float)
        rows = len(self.y)
        self.n = float(self.w.sum()) if frequency else float(rows)
        self.norm = 1.0 if (self.w is None or frequency) else rows / self.w.sum()
        ww = np.ones(rows) if self.w is None else self.w
        mean = (ww * self.y).sum() / ww.sum()
        self.tss = float((ww * (self.y - mean) ** 2).sum()) * self.norm

    def design(self, model):
        if not model:
            return np.empty((len(self.y), 0))
        return np.column_stack([self.columns[t] for t in model])

    def sse(self, model):
        return _wls(self.y, self.design(model), self.w) * self.norm

    def k(self, model):
        return sum(self.columns[t].shape[1] for t in model)

    def tolerances(self, x):
        """1 - R^2 of every column of x regressed on the other columns (and a constant)."""
        out = []
        ww = np.ones(len(self.y)) if self.w is None else self.w
        for j in range(x.shape[1]):
            target = x[:, j]
            mean = (ww * target).sum() / ww.sum()
            total = (ww * (target - mean) ** 2).sum()
            out.append(_wls(target, np.delete(x, j, axis=1), self.w) / total)
        return np.array(out)

    def entry_tolerance(self, model, term):
        """(tolerance of the term's columns, minimum tolerance of the augmented model)."""
        x = self.design([*model, term])
        tol = self.tolerances(x)
        width = self.columns[term].shape[1]
        return float(tol[-width:].min()), float(tol.min())

    def enter_test(self, model, term):
        q = self.columns[term].shape[1]
        sse0, sse1 = self.sse(model), self.sse([*model, term])
        df2 = self.n - 1 - self.k(model) - q
        f = ((sse0 - sse1) / q) / (sse1 / df2)
        return f, q, df2, stats.f.sf(f, q, df2)

    def remove_test(self, model, term):
        q = self.columns[term].shape[1]
        sse1 = self.sse(model)
        sse0 = self.sse([t for t in model if t != term])
        df2 = self.n - 1 - self.k(model)
        f = ((sse0 - sse1) / q) / (sse1 / df2)
        return f, q, df2, stats.f.sf(f, q, df2)

    def summary(self, model):
        """(R^2, adjusted R^2, rmse, Stata AIC, Stata BIC) of a model."""
        k, n = self.k(model), self.n
        sse = self.sse(model)
        r2 = 1 - sse / self.tss
        ll = -0.5 * n * (math.log(2 * math.pi) + math.log(sse / n) + 1)
        return (r2, 1 - (1 - r2) * (n - 1) / (n - 1 - k), math.sqrt(sse / (n - 1 - k)),
                -2 * ll + 2 * (k + 1), -2 * ll + math.log(n) * (k + 1))

    def eligible(self, model, term, tolerance):
        if self.n - 1 - self.k(model) - self.columns[term].shape[1] <= 0:
            return False
        tol, low = self.entry_tolerance(model, term)
        return tol >= tolerance and low >= tolerance

    def search(self, method="stepwise", criterion="pvalue", p_enter=0.05, p_remove=0.10,
               forced=(), tolerance=1e-4):
        forced = list(forced)
        model: list[str] = []
        for term in forced + ([t for t in self.names if t not in forced]
                              if method == "backward" else []):
            if not model or self.eligible(model, term, tolerance):
                model.append(term)
        start = list(model)
        steps = []
        penalty = 2.0 if criterion == "aic" else math.log(self.n)

        def value(m):
            return self.n * math.log(self.sse(m) / self.n) + penalty * self.k(m)

        for _ in range(100):
            removable = [t for t in model if t not in forced] if method != "forward" else []
            enterable = [t for t in self.names if t not in model
                         and self.eligible(model, t, tolerance)] if method != "backward" else []
            if criterion == "pvalue":
                if removable:
                    tests = [(self.remove_test(model, t), t) for t in removable]
                    (f, q, df2, p), term = max(tests, key=lambda item: (item[0][3], -item[0][0]))
                    if p > p_remove:
                        model.remove(term)
                        steps.append(("removed", term, f, q, df2, p, list(model)))
                        continue
                if enterable:
                    tests = [(self.enter_test(model, t), t) for t in enterable]
                    (f, q, df2, p), term = min(tests, key=lambda item: (item[0][3], -item[0][0]))
                    if p < p_enter:
                        model.append(term)
                        steps.append(("entered", term, f, q, df2, p, list(model)))
                        continue
                break
            moves = [(value([u for u in model if u != t]), "removed", t) for t in removable]
            moves += [(value([*model, t]), "entered", t) for t in enterable]
            if not moves:
                break
            best, action, term = min(moves, key=lambda item: item[0])
            if not best < value(model) - 1e-9:
                break
            test = self.enter_test(model, term) if action == "entered" \
                else self.remove_test(model, term)
            model = [*model, term] if action == "entered" else [u for u in model if u != term]
            steps.append((action, term, *test, list(model)))
        return start, steps, model


def _statsmodels(y, design, w):
    """statsmodels OLS / WLS of y on design = [1, X], reparametrized for accuracy.

    The fit uses [1, (X - mean) / scale] and maps the estimates and their unscaled
    covariance back to the raw parametrization with the exact linear map T (raw = T c).
    Without it statsmodels' pinv loses digits (or drops columns) for a regressor such as
    -40 + 1e-6 z, nearly parallel to the constant; checked against exact rational
    arithmetic.
    """
    ww = np.ones(len(y)) if w is None else w
    x = design[:, 1:]
    mean = (ww[:, None] * x).sum(axis=0) / ww.sum()
    scale = np.sqrt((ww[:, None] * (x - mean) ** 2).sum(axis=0))
    centred = np.column_stack([np.ones(len(y)), (x - mean) / scale])
    fit = sm.OLS(y, centred).fit() if w is None else sm.WLS(y, centred, weights=w).fit()
    t = np.diag(np.concatenate([[1.0], 1.0 / scale]))
    t[0, 1:] = -mean / scale
    cov = t @ np.asarray(fit.normalized_cov_params) @ t.T
    params = t @ fit.params
    return fit, params, cov, np.sqrt(np.diag(cov) * fit.scale)


def _check_stepwise(result, replay: Replay, start, steps, model, *, alpha=0.05,
                    tolerance=1e-4):
    """Compare every table of oe.stepwise with the brute-force replay."""
    assert result.attrs["selected"] == model
    table = result["steps"]
    rows = table[table["action"] != "start"]
    assert list(rows["action"]) == [s[0] for s in steps]
    assert list(rows["term"]) == [s[1] for s in steps]
    assert list(rows["step"]) == list(range(1, len(steps) + 1))
    if steps:
        np.testing.assert_allclose(rows["statistic"], [s[2] for s in steps], rtol=1e-7)
        np.testing.assert_allclose(rows["df1"], [s[3] for s in steps])
        np.testing.assert_allclose(rows["df2"], [s[4] for s in steps])
        np.testing.assert_allclose(rows["p_value"], [s[5] for s in steps], rtol=1e-6,
                                   atol=1e-300)
        fits = np.array([replay.summary(s[6]) for s in steps])
        np.testing.assert_allclose(rows[["r_squared", "adjusted_r_squared", "rmse", "aic",
                                         "bic"]].to_numpy(float), fits, rtol=1e-8)
        before = [replay.summary(start)[0] if start else 0.0, *fits[:-1, 0]]
        np.testing.assert_allclose(rows["r_squared_change"], fits[:, 0] - np.array(before),
                                   rtol=1e-6, atol=1e-10)
    if start:
        first = table.iloc[0]
        assert first["action"] == "start" and first["step"] == 0
        k = replay.k(start)
        r2 = replay.summary(start)[0]
        f = (r2 / k) / ((1 - r2) / (replay.n - 1 - k))
        assert first["statistic"] == pytest.approx(f, rel=1e-8)
        assert first["df1"] == k and first["df2"] == pytest.approx(replay.n - 1 - k)
        assert first["p_value"] == pytest.approx(stats.f.sf(f, k, replay.n - 1 - k), rel=1e-6)
    else:
        assert "start" not in set(table["action"])

    # Final model against statsmodels.
    x = replay.design(model)
    design = np.column_stack([np.ones(len(replay.y)), x])
    labels = ["Intercept", *(label for t in model for label in replay.labels[t])]
    coefficients = result["coefficients"]
    assert list(coefficients.index) == labels
    n, k = replay.n, x.shape[1]
    fit, params, unscaled, bse = _statsmodels(replay.y, design, replay.w)
    df_resid = n - 1 - k
    sse = fit.ssr * replay.norm
    s2 = sse / df_resid
    # Covariance with the right residual df (frequency weights: n = sum of weights).
    cov = unscaled * s2 / replay.norm
    se = np.sqrt(np.diag(cov))
    t = params / se
    crit = stats.t.isf(alpha / 2, df_resid)
    np.testing.assert_allclose(coefficients["b"], params, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(coefficients["std_error"], se, rtol=1e-8)
    np.testing.assert_allclose(coefficients["t"], t, rtol=1e-8)
    np.testing.assert_allclose(coefficients["p_value"], 2 * stats.t.sf(np.abs(t), df_resid),
                               rtol=1e-6, atol=1e-300)
    np.testing.assert_allclose(coefficients["ci_low"], params - crit * se, rtol=1e-8,
                               atol=1e-10)
    np.testing.assert_allclose(coefficients["ci_high"], params + crit * se, rtol=1e-8,
                               atol=1e-10)
    if replay.w is None and n == len(replay.y):
        np.testing.assert_allclose(coefficients["std_error"], bse, rtol=1e-8)
        # Slope p-values do not depend on the reparametrization (the intercept's does).
        np.testing.assert_allclose(coefficients["p_value"].iloc[1:], fit.pvalues[1:],
                                   rtol=1e-6, atol=1e-300)
    ww = np.ones(len(replay.y)) if replay.w is None else replay.w

    def wsd(v):
        return math.sqrt((ww * (v - (ww * v).sum() / ww.sum()) ** 2).sum())

    if k:
        beta = [params[j + 1] * wsd(x[:, j]) / wsd(replay.y) for j in range(k)]
        np.testing.assert_allclose(coefficients["beta"].iloc[1:], beta, rtol=1e-8, atol=1e-12)
        tol = replay.tolerances(x)
        np.testing.assert_allclose(coefficients["tolerance"].iloc[1:], tol, rtol=1e-7)
        np.testing.assert_allclose(coefficients["vif"].iloc[1:], 1 / tol, rtol=1e-7)
    assert np.isnan(coefficients.loc["Intercept", "beta"])

    # Excluded terms: refit with each candidate added.
    excluded = result["excluded"]
    outside = [term for term in replay.names if term not in model]
    assert list(excluded.index) == outside
    sse0 = replay.sse(model)
    for term in outside:
        row = excluded.loc[term]
        q = replay.columns[term].shape[1]
        assert row["df1"] == q
        tol, low = replay.entry_tolerance(model, term)
        assert row["tolerance"] == pytest.approx(tol, rel=1e-6, abs=1e-12)
        assert row["min_tolerance"] == pytest.approx(low, rel=1e-6, abs=1e-12)
        if tol < 1e-10:
            continue
        f, _, df2, p = replay.enter_test(model, term)
        assert row["df2"] == pytest.approx(df2)
        assert row["statistic"] == pytest.approx(f, rel=1e-6)
        assert row["p_value"] == pytest.approx(p, rel=1e-5, abs=1e-300)
        partial_r2 = (sse0 - replay.sse([*model, term])) / sse0
        if q == 1:
            bigger = np.column_stack([design, replay.columns[term]])
            augmented = _statsmodels(replay.y, bigger, replay.w)[1]
            sign = np.sign(augmented[-1])
            # SPSS: t of the candidate = signed root of its F-to-enter; Beta In = standardized
            # coefficient it would get; partial correlation of y and x given the model.
            assert row["t"] == pytest.approx(sign * math.sqrt(f), rel=1e-6)
            assert row["beta_in"] == pytest.approx(
                augmented[-1] * wsd(replay.columns[term][:, 0]) / wsd(replay.y),
                rel=1e-6)
            assert row["partial_correlation"] == pytest.approx(sign * math.sqrt(partial_r2),
                                                               rel=1e-6)
        else:
            assert np.isnan(row["beta_in"]) and np.isnan(row["t"])
            assert row["partial_correlation"] == pytest.approx(math.sqrt(partial_r2), rel=1e-6)

    # ANOVA of the final model and the attributes.
    anova = result["anova"]
    regression_ss = replay.tss - sse
    np.testing.assert_allclose(anova["ss"], [regression_ss, sse, replay.tss], rtol=1e-8)
    np.testing.assert_allclose(anova["df"], [k, df_resid, n - 1])
    if k:
        f = (regression_ss / k) / s2
        assert anova.loc["regression", "statistic"] == pytest.approx(f, rel=1e-8)
        assert anova.loc["regression", "p_value"] == pytest.approx(
            stats.f.sf(f, k, df_resid), rel=1e-6, abs=1e-300)
        assert result.attrs["statistic"] == pytest.approx(f, rel=1e-8)
    r2, adjusted, rmse, aic, bic = replay.summary(model)
    attrs = result.attrs
    assert attrs["r_squared"] == pytest.approx(r2, rel=1e-9, abs=1e-12)
    assert attrs["adjusted_r_squared"] == pytest.approx(adjusted, rel=1e-8, abs=1e-12)
    assert attrs["rmse"] == pytest.approx(rmse, rel=1e-9)
    assert attrs["aic"] == pytest.approx(aic, rel=1e-10)
    assert attrs["bic"] == pytest.approx(bic, rel=1e-10)
    assert attrs["df1"] == k and attrs["df2"] == pytest.approx(df_resid)
    assert attrs["n"] == pytest.approx(n)


def _correlated(seed=0, n=160):
    """Correlated candidates; x1 is a proxy that is redundant once x2 and x3 are in."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 6))
    frame = pd.DataFrame({
        "x2": z[:, 0], "x3": z[:, 1],
        "x1": z[:, 0] + z[:, 1] + 0.45 * z[:, 2],
        "x4": 0.5 * z[:, 0] + z[:, 3],
        "x5": z[:, 4], "x6": 0.3 * z[:, 1] + z[:, 5]})
    frame["y"] = 1.5 + frame["x2"] + 0.9 * frame["x3"] + 0.25 * frame["x5"] \
        + 0.8 * rng.standard_normal(n)
    frame["g"] = rng.choice(["k", "m", "p", "s"], n)
    frame["y"] += frame["g"].map({"k": 0.0, "m": 0.6, "p": -0.3, "s": 0.2})
    frame["w"] = rng.uniform(0.2, 4.0, n)
    frame["f"] = rng.integers(1, 5, n)
    return frame


CANDIDATES = ["x1", "x2", "x3", "x4", "x5", "x6"]


@pytest.mark.parametrize("method", ["stepwise", "forward", "backward"])
@pytest.mark.parametrize("criterion", ["pvalue", "aic", "bic"])
@pytest.mark.parametrize("seed", [0, 5])
def test_stepwise_matches_brute_force_replay(method, criterion, seed):
    frame = _correlated(seed)
    replay = Replay(frame, "y", CANDIDATES)
    start, steps, model = replay.search(method, criterion)
    result = oe.stepwise(frame, "y", CANDIDATES, method=method, criterion=criterion)
    _check_stepwise(result, replay, start, steps, model)


def test_stepwise_path_with_a_removal_matches_the_replay():
    frame = _correlated(0)
    replay = Replay(frame, "y", CANDIDATES)
    start, steps, model = replay.search("stepwise", "pvalue", 0.15, 0.15)
    assert "removed" in [s[0] for s in steps]           # the design exercises a removal
    result = oe.stepwise(frame, "y", CANDIDATES, p_enter=0.15, p_remove=0.15)
    _check_stepwise(result, replay, start, steps, model)


@pytest.mark.parametrize("method", ["stepwise", "backward"])
def test_stepwise_analytic_weights_match_weighted_replay(method):
    frame = _correlated(2)
    replay = Replay(frame, "y", CANDIDATES, weights="w")
    start, steps, model = replay.search(method, "pvalue", 0.2, 0.25)
    result = oe.stepwise(frame, "y", CANDIDATES, method=method, p_enter=0.2, p_remove=0.25,
                         weights="w", alpha=0.1)
    _check_stepwise(result, replay, start, steps, model, alpha=0.1)
    # Analytic weights are scale free: every number is unchanged by w -> 1000 w.
    scaled = oe.stepwise(frame.assign(w=frame["w"] * 1000.0), "y", CANDIDATES, method=method,
                         p_enter=0.2, p_remove=0.25, weights="w", alpha=0.1)
    for name in ("steps", "coefficients", "excluded", "anova"):
        np.testing.assert_allclose(scaled[name].select_dtypes("number").to_numpy(float),
                                   result[name].select_dtypes("number").to_numpy(float),
                                   rtol=1e-9, atol=1e-12)


def test_stepwise_frequency_weights_equal_expanded_rows():
    frame = _correlated(3)
    replay = Replay(frame, "y", CANDIDATES, weights="f", frequency=True)
    start, steps, model = replay.search("stepwise", "bic")
    result = oe.stepwise(frame, "y", CANDIDATES, criterion="bic", weights="f",
                         weight_type="fweight")
    _check_stepwise(result, replay, start, steps, model)
    expanded = frame.loc[frame.index.repeat(frame["f"])].reset_index(drop=True)
    plain = oe.stepwise(expanded, "y", CANDIDATES, criterion="bic")
    for name in ("steps", "coefficients", "excluded", "anova"):
        np.testing.assert_allclose(plain[name].select_dtypes("number").to_numpy(float),
                                   result[name].select_dtypes("number").to_numpy(float),
                                   rtol=1e-8, atol=1e-12)
    assert plain.attrs["n"] == result.attrs["n"] == int(frame["f"].sum())


def test_categorical_blocks_and_forced_terms_match_the_replay():
    frame = _correlated(4)
    terms = ["x4", "g", *[t for t in CANDIDATES if t != "x4"]]
    replay = Replay(frame, "y", terms, categorical=("g",))
    start, steps, model = replay.search("stepwise", "pvalue", 0.05, 0.10, forced=["x4"])
    assert "g" in model
    result = oe.stepwise(frame, "y", [t for t in terms if t != "x4"], forced=["x4"],
                         categorical=["g"])
    _check_stepwise(result, replay, start, steps, model)
    # The block F-to-enter of g equals statsmodels' joint F test of its three indicators.
    entered = result["steps"].set_index("term").loc["g"]
    before = steps[[s[1] for s in steps].index("g")][6][:-1]
    small = sm.OLS(replay.y, sm.add_constant(replay.design(before))).fit()
    large = sm.OLS(replay.y, sm.add_constant(replay.design([*before, "g"]))).fit()
    f, p, df = large.compare_f_test(small)
    assert entered["statistic"] == pytest.approx(f, rel=1e-8) and entered["df1"] == df == 3
    assert entered["p_value"] == pytest.approx(p, rel=1e-6)


def test_backward_with_a_categorical_block_and_weights():
    frame = _correlated(6)
    terms = ["g", *CANDIDATES]
    replay = Replay(frame, "y", terms, categorical=("g",), weights="w")
    start, steps, model = replay.search("backward", "pvalue", 0.05, 0.10)
    result = oe.stepwise(frame, "y", terms, method="backward", categorical=["g"], weights="w")
    _check_stepwise(result, replay, start, steps, model)


@pytest.mark.parametrize("criterion", ["aic", "pvalue"])
def test_backward_with_forced_terms_and_analytic_weights(criterion):
    frame = _correlated(10)
    terms = ["x5", *[t for t in CANDIDATES if t != "x5"]]
    replay = Replay(frame, "y", terms, weights="w")
    start, steps, model = replay.search("backward", criterion, 0.05, 0.10, forced=["x5"])
    result = oe.stepwise(frame, "y", [t for t in CANDIDATES if t != "x5"], forced=["x5"],
                         method="backward", criterion=criterion, weights="w")
    _check_stepwise(result, replay, start, steps, model)
    assert model[0] == "x5" and result["steps"].iloc[0]["term"] == "(all terms)"


def test_hald_cement_with_spss_defaults_matches_the_replay():
    hald = pd.DataFrame({
        "x1": [7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10],
        "x2": [26, 29, 56, 31, 52, 55, 71, 31, 54, 47, 40, 66, 68],
        "x3": [6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8],
        "x4": [60, 52, 20, 47, 33, 22, 6, 44, 22, 26, 34, 12, 12],
        "y": [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3,
              109.4]})
    xs = ["x1", "x2", "x3", "x4"]
    replay = Replay(hald, "y", xs)
    for method in ("stepwise", "forward", "backward"):
        for pe, pr in ((0.05, 0.10), (0.15, 0.15)):
            start, steps, model = replay.search(method, "pvalue", pe, pr)
            result = oe.stepwise(hald, "y", xs, method=method, p_enter=pe, p_remove=pr)
            _check_stepwise(result, replay, start, steps, model)
    # Draper & Smith: the stepwise path at 0.15 is x4, x1, x2, then x4 leaves.
    path = oe.stepwise(hald, "y", xs, p_enter=0.15, p_remove=0.15)["steps"]
    assert list(zip(path["action"], path["term"], strict=True)) == [
        ("entered", "x4"), ("entered", "x1"), ("entered", "x2"), ("removed", "x4")]


def test_missing_rows_are_dropped_listwise_like_a_complete_case_replay():
    frame = _correlated(7)
    frame.loc[[3, 17, 40], "x5"] = np.nan
    frame.loc[[9], "y"] = np.nan
    frame.loc[[17, 60], "w"] = np.nan
    result = oe.stepwise(frame, "y", CANDIDATES, weights="w", missing="drop")
    complete = frame.dropna(subset=["y", *CANDIDATES, "w"]).reset_index(drop=True)
    replay = Replay(complete, "y", CANDIDATES, weights="w")
    start, steps, model = replay.search()
    _check_stepwise(result, replay, start, steps, model)
    assert result.attrs["n_missing"] == 5 and result.attrs["n"] == len(complete)


def test_collinear_candidate_is_kept_out_by_the_tolerance_test():
    # x2 and x3 are forced, so the exact combination 'dup' can never tie with one of them.
    frame = _correlated(8)
    frame["dup"] = frame["x2"] - 2.0 * frame["x3"]               # exact combination
    frame["near"] = frame["x5"] + 0.02 * np.random.default_rng(1).standard_normal(len(frame))
    xs = ["x1", "x4", "x5", "x6", "dup", "near"]
    terms = ["x2", "x3", *xs]
    replay = Replay(frame, "y", terms)
    start, steps, model = replay.search("forward", "pvalue", 0.5, 0.6, forced=["x2", "x3"])
    result = oe.stepwise(frame, "y", xs, forced=["x2", "x3"], method="forward", p_enter=0.5,
                         p_remove=0.6)
    _check_stepwise(result, replay, start, steps, model)
    assert "dup" not in model
    assert result["excluded"].loc["dup", "tolerance"] < 1e-10
    assert np.isnan(result["excluded"].loc["dup", "statistic"])
    # A stricter tolerance keeps the near duplicate of x5 out once x5 is in (and vice versa).
    for tolerance in (1e-4, 0.01):
        strict = oe.stepwise(frame, "y", xs, forced=["x2", "x3"], method="stepwise",
                             p_enter=0.5, p_remove=0.6, tolerance=tolerance)
        again = replay.search("stepwise", "pvalue", 0.5, 0.6, forced=["x2", "x3"],
                              tolerance=tolerance)
        _check_stepwise(strict, replay, *again, tolerance=tolerance)
    assert not ("x5" in again[2] and "near" in again[2])


def test_tiny_sample_matches_the_replay():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.standard_normal((7, 4)), columns=["a", "b", "c", "d"])
    frame["y"] = 2 * frame["a"] - frame["b"] + 0.3 * rng.standard_normal(7)
    replay = Replay(frame, "y", ["a", "b", "c", "d"])
    for method in ("stepwise", "forward", "backward"):
        start, steps, model = replay.search(method, "pvalue", 0.2, 0.3)
        result = oe.stepwise(frame, "y", ["a", "b", "c", "d"], method=method, p_enter=0.2,
                             p_remove=0.3)
        _check_stepwise(result, replay, start, steps, model)


def test_invariance_to_scale_offset_and_row_order():
    frame = _correlated(9)
    base = oe.stepwise(frame, "y", CANDIDATES, p_enter=0.1, p_remove=0.2)
    changed = frame.copy()
    changed["x2"] = changed["x2"] * 1e7 + 3e8
    changed["x5"] = changed["x5"] * 1e-6 - 40.0
    changed["y"] = changed["y"] * 1e4 + 1e9
    changed = changed.sample(frac=1.0, random_state=3).reset_index(drop=True)
    moved = oe.stepwise(changed, "y", CANDIDATES, p_enter=0.1, p_remove=0.2)
    assert moved.attrs["selected"] == base.attrs["selected"]
    for name in ("statistic", "p_value", "r_squared", "adjusted_r_squared"):
        np.testing.assert_allclose(moved["steps"][name], base["steps"][name], rtol=1e-8)
    a, b = moved["coefficients"], base["coefficients"]
    for name in ("beta", "t", "p_value", "tolerance", "vif"):
        np.testing.assert_allclose(a[name].iloc[1:], b[name].iloc[1:], rtol=1e-7)
    factor = {"x2": 1e-7 * 1e4, "x5": 1e6 * 1e4}
    for term in a.index[1:]:
        assert a.loc[term, "b"] == pytest.approx(b.loc[term, "b"] * factor.get(term, 1e4),
                                                 rel=1e-7)
    # The replay on the transformed data agrees as well.
    replay = Replay(changed, "y", CANDIDATES)
    _check_stepwise(moved, replay, *replay.search("stepwise", "pvalue", 0.1, 0.2))


@pytest.mark.parametrize("noise", [1e-4, 1e-5, 1e-7])
def test_near_perfect_fits_keep_full_precision(noise):
    """Regression: the residual of the final model was 1 - r'R^-1 r, which cancels as
    R^2 -> 1 (SE error 2e-5 at 1 - R^2 = 1e-9) and refused 1 - R^2 = 1e-13 as an 'exact
    linear combination'. It is now recomputed from the data."""
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["a", "b", "c"])
    frame["y"] = 2 + 3 * frame["a"] - frame["b"] + noise * rng.standard_normal(1000)
    result = oe.stepwise(frame, "y", ["a", "b", "c"])
    assert result.attrs["selected"] == ["a", "b"]
    fit = sm.OLS(frame["y"], sm.add_constant(frame[["a", "b"]])).fit()
    coefficients = result["coefficients"]
    np.testing.assert_allclose(coefficients["b"], fit.params, rtol=1e-9)
    np.testing.assert_allclose(coefficients["std_error"], fit.bse, rtol=1e-8)
    assert result["anova"].loc["residual", "ss"] == pytest.approx(fit.ssr, rel=1e-8)
    assert 1 - result.attrs["r_squared"] == pytest.approx(1 - fit.rsquared, rel=1e-6)
    assert result.attrs["rmse"] == pytest.approx(math.sqrt(fit.scale), rel=1e-8)
    assert result.attrs["aic"] == pytest.approx(fit.aic, rel=1e-10)
    last = result["steps"].iloc[-1]
    assert last["rmse"] == pytest.approx(result.attrs["rmse"], rel=1e-12)
    assert last["bic"] == pytest.approx(fit.bic, rel=1e-10)
    exhausted = any("rounding error" in note for note in result.attrs["notes"])
    assert exhausted == (noise == 1e-7)
    if exhausted:                  # 1 - R^2 = 1e-15: step statistics are withdrawn
        assert np.isnan(last["statistic"]) and np.isnan(last["p_value"])
        assert np.isnan(result["excluded"].loc["c", "statistic"])
        assert result["excluded"].loc["c", "tolerance"] > 0.99


def test_exact_fits_are_refused_with_the_terms_named():
    rng = np.random.default_rng(32)
    frame = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    for method in ("stepwise", "backward"):
        with pytest.raises(oe.AnalysisError) as error:
            oe.stepwise(frame.assign(y=2 * frame["a"] - frame["c"] + 1), "y", ["a", "b", "c"],
                        method=method)
        assert error.value.code == "perfect_fit" and "a" in str(error.value)


# ==== collin ============================================================================


def _bkw(x: np.ndarray, intercept: bool):
    """Belsley-Kuh-Welsch from an eigendecomposition of the equilibrated X'X."""
    design = np.column_stack([np.ones(len(x)), x]) if intercept else x
    scaled = design / np.sqrt((design ** 2).sum(axis=0))
    values, vectors = np.linalg.eigh(scaled.T @ scaled)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    phi = vectors ** 2 / values                     # [coefficient k, dimension j]
    proportions = (phi / phi.sum(axis=1, keepdims=True)).T
    return values, np.sqrt(values[0] / values), proportions


def _vif(x: np.ndarray, intercept: bool):
    """1 / (1 - R^2) of each column on the others: centred with a constant, else uncentred."""
    out = []
    for j in range(x.shape[1]):
        others = np.delete(x, j, axis=1)
        if intercept:
            others = np.column_stack([np.ones(len(x)), others])
        target = x[:, j]
        beta, *_ = np.linalg.lstsq(others, target, rcond=None) if others.size \
            else (np.zeros(0), None, None, None)
        resid = target - (others @ beta if others.size else 0.0)
        total = ((target - target.mean()) ** 2).sum() if intercept else (target ** 2).sum()
        out.append(total / (resid ** 2).sum())
    return np.array(out)


def _collin_frame(seed=0, n=90):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 5))
    return pd.DataFrame({"a": z[:, 0], "b": 0.9 * z[:, 0] + 0.3 * z[:, 1],
                         "c": z[:, 2] + 2.0, "d": 0.5 * z[:, 1] - 0.5 * z[:, 2] + 0.2 * z[:, 3],
                         "e": rng.integers(0, 2, n).astype(float)})


@pytest.mark.parametrize("intercept", [True, False])
def test_collin_matches_eigendecomposition_and_auxiliary_regressions(intercept):
    frame = _collin_frame()
    names = list(frame.columns)
    x = frame.to_numpy(float)
    result = oe.collin(frame, names, intercept=intercept)
    vif = _vif(x, intercept)
    table = result["vif"]
    np.testing.assert_allclose(table["vif"], vif, rtol=1e-9)
    np.testing.assert_allclose(table["tolerance"], 1 / vif, rtol=1e-9)
    np.testing.assert_allclose(table["r_squared"], 1 - 1 / vif, rtol=1e-9)
    values, indices, proportions = _bkw(x, intercept)
    condition = result["condition"]
    np.testing.assert_allclose(condition["eigenvalue"], values, rtol=1e-9)
    np.testing.assert_allclose(condition["condition_index"], indices, rtol=1e-9)
    terms = (["Intercept"] if intercept else []) + names
    np.testing.assert_allclose(condition[terms].to_numpy(float), proportions, rtol=1e-7,
                               atol=1e-12)
    assert result.attrs["condition_number"] == pytest.approx(indices[-1], rel=1e-9)
    assert result.attrs["mean_vif"] == pytest.approx(vif.mean(), rel=1e-9)
    assert result.attrs["max_vif"] == pytest.approx(vif.max(), rel=1e-9)
    assert result.attrs["n"] == len(frame)
    # statsmodels' variance_inflation_factor on the same exog.
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    exog = sm.add_constant(x) if intercept else x
    offset = 1 if intercept else 0
    reference = [variance_inflation_factor(exog, j + offset) for j in range(len(names))]
    np.testing.assert_allclose(table["vif"], reference, rtol=1e-8)


def test_collin_longley_is_badly_conditioned_but_exact():
    from statsmodels.datasets import longley
    frame = longley.load_pandas().exog
    names = list(frame.columns)
    x = frame.to_numpy(float)
    result = oe.collin(frame, names)
    np.testing.assert_allclose(result["vif"]["vif"], _vif(x, True), rtol=1e-6)
    values, indices, proportions = _bkw(x, True)
    np.testing.assert_allclose(result["condition"]["condition_index"], indices, rtol=1e-6)
    np.testing.assert_allclose(result["condition"][["Intercept", *names]].to_numpy(float),
                               proportions, rtol=1e-4, atol=1e-9)


def test_collin_invariances_and_missing_rows():
    frame = _collin_frame(1)
    base = oe.collin(frame, list(frame.columns))
    # Rescaling a column leaves VIFs and (equilibrated) condition indices unchanged.
    scaled = oe.collin(frame.assign(a=frame["a"] * 1e6, d=frame["d"] * 1e-5),
                       list(frame.columns))
    for name in ("vif", "condition"):
        np.testing.assert_allclose(scaled[name].to_numpy(float), base[name].to_numpy(float),
                                   rtol=1e-8, atol=1e-12)
    # Centred VIFs do not depend on a column's location (the condition table does).
    shifted = oe.collin(frame.assign(b=frame["b"] + 1e6), list(frame.columns))
    np.testing.assert_allclose(shifted["vif"].to_numpy(float), base["vif"].to_numpy(float),
                               rtol=1e-7)
    permuted = oe.collin(frame.sample(frac=1.0, random_state=0), list(frame.columns))
    np.testing.assert_allclose(permuted["vif"].to_numpy(float), base["vif"].to_numpy(float),
                               rtol=1e-10)
    holes = frame.copy()
    holes.loc[[2, 5, 11], "c"] = np.nan
    holes.loc[[5, 30], "e"] = np.nan
    result = oe.collin(holes, list(frame.columns))
    complete = holes.dropna()
    np.testing.assert_allclose(result["vif"]["vif"], _vif(complete.to_numpy(float), True),
                               rtol=1e-9)
    assert result.attrs["n"] == len(complete) and result.attrs["n_missing"] == 4


# ==== curvefit ==========================================================================


def _curve_oracle(y, x, model, intercept=True, upper=None):
    """statsmodels OLS on SPSS's transformed variables, back-transformed to SPSS's b's."""
    lny = np.log(y) if (y > 0).all() else None
    with np.errstate(divide="ignore", invalid="ignore"):     # unused invalid transforms
        table = _curve_table(y, x, lny, upper)
    target, regressors = table[model]
    design = np.column_stack(regressors)
    if intercept:
        design = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(target, design).fit()
    params = list(fit.params) if intercept else [None, *fit.params]
    if model in ("compound", "power", "exponential", "logistic") and intercept:
        params[0] = math.exp(params[0])
    if model in ("compound", "logistic"):
        params[1] = math.exp(params[1])
    return fit, params


def _curve_table(y, x, lny, upper):
    """(transformed outcome, transformed regressors) of every SPSS CURVEFIT model."""
    return {
        "linear": (y, [x]), "logarithmic": (y, [np.log(x)]), "inverse": (y, [1 / x]),
        "quadratic": (y, [x, x ** 2]), "cubic": (y, [x, x ** 2, x ** 3]),
        "compound": (lny, [x]), "power": (lny, [np.log(x)]), "s": (lny, [1 / x]),
        "growth": (lny, [x]), "exponential": (lny, [x]),
        "logistic": (np.log(1 / y - (0 if upper is None else 1 / upper)), [x]),
    }


CURVES = ["linear", "logarithmic", "inverse", "quadratic", "cubic", "compound", "power", "s",
          "growth", "exponential", "logistic"]


@pytest.mark.parametrize("intercept", [True, False])
def test_curvefit_matches_transformed_ols(intercept):
    rng = np.random.default_rng(21)
    x = rng.uniform(0.5, 6.0, 60)
    y = 3.0 * np.exp(0.3 * x) * np.exp(0.15 * rng.standard_normal(60))
    frame = pd.DataFrame({"x": x, "y": y})
    result = oe.curvefit(frame, "y", "x", intercept=intercept, upper_bound=None)
    summary = result["summary"]
    for model in CURVES:
        fit, params = _curve_oracle(y, x, model, intercept)
        row = summary.loc[model]
        # statsmodels reports the uncentred R^2 and F for a model without a constant.
        assert row["r_squared"] == pytest.approx(fit.rsquared, rel=1e-9), model
        assert row["statistic"] == pytest.approx(fit.fvalue, rel=1e-8), model
        assert row["p_value"] == pytest.approx(fit.f_pvalue, rel=1e-6, abs=1e-300), model
        assert row["df1"] == fit.df_model and row["df2"] == fit.df_resid
        for j, value in enumerate(params):
            if value is None:
                assert np.isnan(row[f"b{j}"])
            else:
                assert row[f"b{j}"] == pytest.approx(value, rel=1e-8, abs=1e-12), (model, j)
        for j in range(len(params), 4):
            assert np.isnan(row[f"b{j}"])
    # The fitted grid evaluates the published equations (b0 = 0 or 1 without a constant).
    g = np.linspace(x.min(), x.max(), 400)
    b = summary[["b0", "b1", "b2", "b3"]].to_numpy(float)
    additive = 0.0 if not intercept else None
    curves = {
        "linear": lambda c: c[0] + c[1] * g, "logarithmic": lambda c: c[0] + c[1] * np.log(g),
        "inverse": lambda c: c[0] + c[1] / g,
        "quadratic": lambda c: c[0] + c[1] * g + c[2] * g ** 2,
        "cubic": lambda c: c[0] + c[1] * g + c[2] * g ** 2 + c[3] * g ** 3,
        "compound": lambda c: c[0] * c[1] ** g, "power": lambda c: c[0] * g ** c[1],
        "s": lambda c: np.exp(c[0] + c[1] / g), "growth": lambda c: np.exp(c[0] + c[1] * g),
        "exponential": lambda c: c[0] * np.exp(c[1] * g),
        "logistic": lambda c: 1 / (c[0] * c[1] ** g),
    }
    multiplicative = {"compound", "power", "exponential", "logistic"}
    for i, model in enumerate(CURVES):
        c = b[i].copy()
        if additive is not None:
            c[0] = 1.0 if model in multiplicative else additive
        np.testing.assert_allclose(result["fitted"][model], curves[model](c), rtol=1e-9)


def test_curvefit_logistic_upper_bound_and_fitted_grid():
    rng = np.random.default_rng(22)
    x = np.linspace(-3, 3, 40)
    y = 1 / (1 / 10 + 0.5 * 0.4 ** x) * np.exp(0.03 * rng.standard_normal(40))
    frame = pd.DataFrame({"x": x, "y": y})
    upper = 1.2 * y.max()
    result = oe.curvefit(frame, "y", "x", upper_bound=upper)
    fit, params = _curve_oracle(y, x, "logistic", True, upper)
    row = result["summary"].loc["logistic"]
    assert row["r_squared"] == pytest.approx(fit.rsquared, rel=1e-9)
    assert [row["b0"], row["b1"]] == pytest.approx(params, rel=1e-9)
    # The fitted grid evaluates each published equation at 400 evenly spaced x values.
    grid = result["fitted"]
    assert len(grid) == 400
    g = np.linspace(x.min(), x.max(), 400)
    np.testing.assert_allclose(grid["x"], g, rtol=1e-12, atol=1e-12)
    s = result["summary"]
    equations = {
        "linear": lambda b: b[0] + b[1] * g,
        "logarithmic": None, "power": None,
        "inverse": lambda b: b[0] + b[1] / g,
        "s": lambda b: np.exp(b[0] + b[1] / g),
        "quadratic": lambda b: b[0] + b[1] * g + b[2] * g ** 2,
        "cubic": lambda b: b[0] + b[1] * g + b[2] * g ** 2 + b[3] * g ** 3,
        "compound": lambda b: b[0] * b[1] ** g,
        "growth": lambda b: np.exp(b[0] + b[1] * g),
        "exponential": lambda b: b[0] * np.exp(b[1] * g),
        "logistic": lambda b: 1 / (1 / upper + b[0] * b[1] ** g),
    }
    for model, curve in equations.items():
        if curve is None:                # x <= 0 occurs: the ln(x) models are skipped
            assert model in result.attrs["skipped"] and model not in grid.columns
            continue
        b = s.loc[model, ["b0", "b1", "b2", "b3"]].to_numpy(float)
        with np.errstate(over="ignore"):
            expected = curve(b)
        finite = np.isfinite(expected)    # exp(b0 + b1/x) overflows next to x = 0
        np.testing.assert_allclose(grid[model].to_numpy(float)[finite], expected[finite],
                                   rtol=1e-8)
        assert grid[model][~finite].isna().all()


def test_cubic_in_calendar_years_matches_exact_rational_arithmetic():
    """The raw cubic in years has a condition number near 1e20; solve it exactly."""
    years = list(range(1961, 1991))
    rng = np.random.default_rng(23)
    y = [round(float(v), 3) for v in 50 + 0.8 * (np.array(years) - 1975)
         - 0.02 * (np.array(years) - 1975) ** 2 + 0.002 * (np.array(years) - 1975) ** 3
         + rng.standard_normal(30)]
    frame = pd.DataFrame({"year": [float(v) for v in years], "y": y})
    result = oe.curvefit(frame, "y", "year", models=["linear", "quadratic", "cubic"])
    for model, degree in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
        xs = [Fraction(v) for v in years]
        ys = [Fraction(v).limit_denominator(10 ** 6) for v in y]
        p = degree + 1
        a = [[sum(xi ** (i + j) for xi in xs) for j in range(p)] for i in range(p)]
        b = [sum(yi * xi ** i for xi, yi in zip(xs, ys, strict=True)) for i in range(p)]
        for col in range(p):                            # exact Gauss-Jordan elimination
            pivot = next(r for r in range(col, p) if a[r][col] != 0)
            a[col], a[pivot], b[col], b[pivot] = a[pivot], a[col], b[pivot], b[col]
            for r in range(p):
                if r != col and a[r][col] != 0:
                    factor = a[r][col] / a[col][col]
                    a[r] = [u - factor * v for u, v in zip(a[r], a[col], strict=True)]
                    b[r] -= factor * b[col]
        coef = [b[i] / a[i][i] for i in range(p)]
        fitted = [sum(coef[i] * xi ** i for i in range(p)) for xi in xs]
        mean = sum(ys) / len(ys)
        sse = sum((yi - fi) ** 2 for yi, fi in zip(ys, fitted, strict=True))
        sst = sum((yi - mean) ** 2 for yi in ys)
        row = result["summary"].loc[model]
        assert row["r_squared"] == pytest.approx(float(1 - sse / sst), rel=1e-10)
        for j in range(p):
            assert row[f"b{j}"] == pytest.approx(float(coef[j]), rel=1e-6), (model, j)


def test_curvefit_domain_skips_and_missing_rows():
    rng = np.random.default_rng(24)
    x = rng.uniform(-2, 5, 50)
    x[0] = 0.0
    y = 2 + x + rng.standard_normal(50)
    y[:3] = -abs(y[:3])
    frame = pd.DataFrame({"x": x, "y": y})
    frame.loc[[7, 8], "y"] = np.nan
    result = oe.curvefit(frame, "y", "x")
    complete = frame.dropna()
    assert result.attrs["n"] == len(complete) and result.attrs["n_missing"] == 2
    fitted = result.attrs["fitted_models"]
    assert fitted == ["linear", "quadratic", "cubic"]
    for model in fitted:
        fit, params = _curve_oracle(complete["y"].to_numpy(), complete["x"].to_numpy(), model)
        assert result["summary"].loc[model, "r_squared"] == pytest.approx(fit.rsquared,
                                                                          rel=1e-10)
    for model in set(CURVES) - set(fitted):
        assert np.isnan(result["summary"].loc[model, "r_squared"])
        assert any(note.startswith(f"{model}:") for note in result.attrs["notes"])


# ==== tabstat ===========================================================================


def _stata_percentile(x, w, p):
    """Stata's (weighted) percentile, written as an explicit loop over the sorted values."""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    total = ws.sum()
    target = total * p / 100
    running = 0.0
    for i in range(len(xs)):
        running += ws[i]
        if abs(running - target) <= 1e-9 * total and i + 1 < len(xs):
            return 0.5 * (xs[i] + xs[i + 1])
        if running > target:
            return xs[i]
    return xs[-1]


def _grouped_median(x, w):
    """Class-interval median: classes halfway between neighbouring distinct values."""
    values = np.unique(x)
    freq = np.array([w[x == v].sum() for v in values])
    half = freq.sum() / 2
    if len(values) == 1:
        return values[0]
    cum = 0.0
    for i, v in enumerate(values):
        lower_gap = values[i] - values[i - 1] if i > 0 else values[1] - values[0]
        upper_gap = values[i + 1] - values[i] if i + 1 < len(values) else lower_gap
        if cum + freq[i] >= half - 1e-9 * freq.sum():
            low, high = v - lower_gap / 2, v + upper_gap / 2
            return low + (half - cum) / freq[i] * (high - low)
        cum += freq[i]
    raise AssertionError


def _expected_row(x, w, kind):
    """Every tabstat statistic for one cell, from first principles."""
    w = np.ones(len(x)) if w is None else w
    total = w.sum()
    n = total if kind == "fweight" else float(len(x))
    v = w * n / total                      # Stata: weights normalized to sum to n
    mean = (w * x).sum() / total
    var = (v * (x - mean) ** 2).sum() / (n - 1)
    m = {r: (w * (x - mean) ** r).sum() / total for r in (2, 3, 4)}
    g1, g2 = m[3] / m[2] ** 1.5, m[4] / m[2] ** 2
    out = {
        "n": n, "mean": mean, "sum": (w * x).sum(), "min": x.min(), "max": x.max(),
        "range": x.max() - x.min(), "std_dev": math.sqrt(var), "variance": var,
        "cv": math.sqrt(var) / mean, "std_error": math.sqrt(var / n),
        "skewness": g1, "kurtosis": g2,
        "se_skewness": math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))),
        "median": _stata_percentile(x, w, 50),
        "iqr": _stata_percentile(x, w, 75) - _stata_percentile(x, w, 25),
        "p10": _stata_percentile(x, w, 10), "p2.5": _stata_percentile(x, w, 2.5),
        "grouped_median": _grouped_median(x, w),
    }
    out["se_kurtosis"] = math.sqrt(4 * (n * n - 1) * out["se_skewness"] ** 2
                                   / ((n - 3) * (n + 5)))
    if (x > 0).all():
        out["geometric_mean"] = math.exp((w * np.log(x)).sum() / total)
        out["harmonic_mean"] = total / (w / x).sum()
    return out


TAB_STATS = ["n", "mean", "sum", "min", "max", "range", "sd", "variance", "cv", "semean",
             "skewness", "kurtosis", "se_skewness", "se_kurtosis", "median", "iqr", "p10",
             "p2.5", "gmedian", "harmonic", "geometric"]
TAB_COLUMNS = ["n", "mean", "sum", "min", "max", "range", "std_dev", "variance", "cv",
               "std_error", "skewness", "kurtosis", "se_skewness", "se_kurtosis", "median",
               "iqr", "p10", "p2.5", "grouped_median", "harmonic_mean", "geometric_mean"]


def _tab_frame(seed=0, n=150):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "x": np.round(rng.gamma(2.0, 3.0, n), 1) + 0.1,
        "z": rng.integers(1, 9, n).astype(float),
        "g": rng.choice(["b", "a", "c"], n, p=[0.5, 0.3, 0.2]),
        "w": rng.uniform(0.3, 3.0, n), "f": rng.integers(1, 5, n)})
    frame.loc[[4, 9], "x"] = np.nan
    return frame


@pytest.mark.parametrize("kind", [None, "aweight", "fweight"])
def test_tabstat_matches_first_principles(kind):
    frame = _tab_frame()
    column = {"aweight": "w", "fweight": "f"}.get(kind)
    table = oe.tabstat(frame, ["x", "z"], by="g", stats=TAB_STATS, weights=column,
                       weight_type=kind)
    for name in ("x", "z"):
        block = table[table["variable"] == name].set_index("g")
        for label, group in [*sorted(frame.groupby("g")), ("Total", frame)]:
            use = group[name].notna()
            x = group.loc[use, name].to_numpy(float)
            w = None if column is None else group.loc[use, column].to_numpy(float)
            expected = _expected_row(x, w, kind or "fweight")
            for key in TAB_COLUMNS:
                got = block.loc[label, key]
                assert got == pytest.approx(expected[key], rel=1e-9, abs=1e-12), \
                    (name, label, key)


def test_tabstat_frequency_weights_equal_expanded_rows_and_scipy():
    frame = _tab_frame(1).dropna()
    weighted = oe.tabstat(frame, ["x"], by="g", stats=TAB_STATS, weights="f",
                          weight_type="fweight", moments="spss")
    expanded = frame.loc[frame.index.repeat(frame["f"])]
    plain = oe.tabstat(expanded, ["x"], by="g", stats=TAB_STATS, moments="spss")
    np.testing.assert_allclose(weighted.select_dtypes("number").to_numpy(float),
                               plain.select_dtypes("number").to_numpy(float), rtol=1e-10)
    for label, group in expanded.groupby("g"):
        x = group["x"].to_numpy()
        row = plain[plain["g"] == label].iloc[0]
        assert row["skewness"] == pytest.approx(stats.skew(x, bias=False), rel=1e-9)
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False), rel=1e-9)
        assert row["median"] == pytest.approx(np.median(x), rel=1e-12)
        assert row["harmonic_mean"] == pytest.approx(stats.hmean(x), rel=1e-10)
        assert row["geometric_mean"] == pytest.approx(stats.gmean(x), rel=1e-10)
        assert row["std_dev"] == pytest.approx(np.std(x, ddof=1), rel=1e-10)


def test_tabstat_spss_moments_with_analytic_weights_use_the_row_count():
    frame = _tab_frame(3).dropna()
    table = oe.tabstat(frame, ["x"], stats=["skewness", "kurtosis"], weights="w",
                       moments="spss")
    x, w = frame["x"].to_numpy(), frame["w"].to_numpy()
    mean = (w * x).sum() / w.sum()
    m2, m3, m4 = ((w * (x - mean) ** r).sum() / w.sum() for r in (2, 3, 4))
    n = len(x)
    g1, g2 = m3 / m2 ** 1.5, m4 / m2 ** 2
    big_g1 = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    big_g2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 - 3 * (n - 1))
    assert table.loc["x", "skewness"] == pytest.approx(big_g1, rel=1e-10)
    assert table.loc["x", "kurtosis"] == pytest.approx(big_g2, rel=1e-10)


@pytest.mark.parametrize("kind", [None, "aweight", "fweight"])
def test_tabstat_anova_matches_scipy_and_statsmodels(kind):
    frame = _tab_frame(2)
    column = {"aweight": "w", "fweight": "f"}.get(kind)
    result = oe.tabstat(frame, ["x"], by="g", weights=column, weight_type=kind, anova=True)
    data = frame.dropna(subset=["x"])
    if kind == "fweight":
        data = data.loc[data.index.repeat(data["f"])]
    anova = result["anova"].set_index("source")
    if kind == "aweight":
        # WLS on group indicators, weights normalized to sum to n (Stata oneway [aw=]).
        v = data["w"].to_numpy() * len(data) / data["w"].sum()
        y = data["x"].to_numpy()
        dummies = pd.get_dummies(data["g"]).to_numpy(float)
        within = sm.WLS(y, dummies, weights=v).fit().ssr
        total = sm.WLS(y, np.ones((len(y), 1)), weights=v).fit().ssr
        between = total - within
        f = (between / 2) / (within / (len(y) - 3))
        p = stats.f.sf(f, 2, len(y) - 3)
    else:
        groups = [g["x"].to_numpy() for _, g in data.groupby("g")]
        f, p = stats.f_oneway(*groups)
        y = data["x"].to_numpy()
        total = ((y - y.mean()) ** 2).sum()
        within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        between = total - within
    assert anova.loc["between", "ss"] == pytest.approx(between, rel=1e-9)
    assert anova.loc["within", "ss"] == pytest.approx(within, rel=1e-10)
    assert anova.loc["total", "ss"] == pytest.approx(total, rel=1e-10)
    assert anova.loc["between", "df"] == 2 and anova.loc["within", "df"] == len(y) - 3
    assert anova.loc["between", "statistic"] == pytest.approx(f, rel=1e-9)
    assert anova.loc["between", "p_value"] == pytest.approx(p, rel=1e-7)
    eta2 = between / total
    assoc = result["association"].loc["x"]
    assert assoc["eta_squared"] == pytest.approx(eta2, rel=1e-9)
    assert assoc["eta"] == pytest.approx(math.sqrt(eta2), rel=1e-9)
