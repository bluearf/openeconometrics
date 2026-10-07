"""Independent oracles for tobit, truncreg and intreg (verification stage).

Nothing here shares code or formulas with the implementation or with the
implementer's tests. Every likelihood is written from the model definition in
a parameterization that differs from both the working one (``ln sigma``) and,
where possible, the reported one, so the delta-method mapping is verified too:

* tobit: Olsen's ``(g, h) = (b / sigma, 1 / sigma)``;
* truncreg: ``(b, sigma^2)``;
* intreg: ``(b, sigma)``.

The oracle maximizes by Newton-Raphson on complex-step scores (exact to
rounding) and a Richardson-extrapolated central-difference Hessian of that
gradient; the covariance estimators are rebuilt from those pieces with Stata's
``ml`` conventions and mapped to the reported parameters with the Jacobian of
the reparameterization. The toolkit at the top is shared by the other
``test_econ_limited_oracle_*`` files.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special, stats

import openecon as oe
from openecon.analysis import AnalysisError

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
KINDS = ("nonrobust", "opg", "robust", "cluster")


# ---- numerical toolkit ---------------------------------------------------------------


def jacobian_cs(function, theta, step=1e-30):
    """Complex-step Jacobian [m, k] of a vector function of ``theta``."""
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        point = theta.astype(complex)
        point[j] += 1j * step
        columns.append(np.atleast_1d(function(point)).imag / step)
    return np.column_stack(columns)


def gradient(obs, theta, w):
    return w @ jacobian_cs(obs, theta)


def hessian(obs, theta, w, extrapolate=True):
    """Hessian of ``sum w obs``: central differences of the complex-step gradient.

    ``extrapolate`` adds a second step size and Richardson extrapolation
    (error of order h^4); the plain difference is enough for Newton directions.
    """
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    out = np.empty((k, k))
    for j in range(k):
        h = 1e-4 * max(1.0, abs(theta[j]))
        estimates = []
        for size in (h, h / 2) if extrapolate else (h / 4,):
            up, down = theta.copy(), theta.copy()
            up[j] += size
            down[j] -= size
            estimates.append((gradient(obs, up, w) - gradient(obs, down, w)) / (2 * size))
        out[:, j] = (4 * estimates[1] - estimates[0]) / 3 if extrapolate else estimates[0]
    return (out + out.T) / 2


def newton(obs, start, w, iterations=200):
    """Maximize ``sum w obs(theta)`` by damped Newton steps; returns (theta, value)."""
    theta = np.asarray(start, dtype=float)

    def value(point):
        with np.errstate(all="ignore"):
            total = float(w @ obs(point).real)
        return total if np.isfinite(total) else -np.inf

    current = value(theta)
    assert np.isfinite(current)
    for _ in range(iterations):
        slope = gradient(obs, theta, w)
        curvature = -hessian(obs, theta, w, extrapolate=False)
        values, vectors = np.linalg.eigh(curvature)
        values = np.maximum(np.abs(values), 1e-8 * np.abs(values).max())
        step = vectors @ ((vectors.T @ slope) / values)
        size = 1.0
        while size > 1e-12 and not value(theta + size * step) >= current - 1e-13 * abs(current):
            size /= 2
        theta = theta + size * step
        current = value(theta)
        if np.max(np.abs(size * step) / np.maximum(1.0, np.abs(theta))) < 1e-12:
            break
    slope = gradient(obs, theta, w)
    assert slope @ np.linalg.solve(-hessian(obs, theta, w, extrapolate=False), slope) < 1e-14
    return theta, current


def likelihood_pieces(obs, theta, w):
    """Score rows and the inverse negative Hessian shared by every covariance estimator."""
    return jacobian_cs(obs, theta), np.linalg.inv(-hessian(obs, theta, w))


def stata_covariance(obs, theta, kind, *, w=None, weight_type=None, groups=None,
                     pieces=None):
    """Stata's -ml- covariance of a (weighted) maximum likelihood estimate.

    ``w`` are the weights as they multiply the log likelihood. OIM: ``(-H)^-1``.
    OPG: ``(sum w s s')^-1`` (weights enter linearly, as frequencies do).
    Robust: ``N/(N-1) (-H)^-1 M (-H)^-1`` with ``M = sum f s s'`` under
    fweights (``N = sum f``) and ``sum w^2 s s'`` otherwise (``N`` = rows).
    Cluster: ``G/(G-1)`` and ``M`` the outer product of the cluster sums of ``w s``.
    """
    n = len(obs(np.asarray(theta, dtype=float)))
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    scores, bread = pieces or likelihood_pieces(obs, theta, w)
    if kind == "nonrobust":
        return bread
    if kind == "opg":
        return np.linalg.inv((scores * w[:, None]).T @ scores)
    if kind == "robust":
        if weight_type == "fweight":
            nobs, meat = w.sum(), (scores * w[:, None]).T @ scores
        else:
            nobs, meat = n, (scores * w[:, None]).T @ (scores * w[:, None])
        return nobs / (nobs - 1) * bread @ meat @ bread
    codes = pd.factorize(np.asarray(groups))[0]
    count = codes.max() + 1
    sums = np.zeros((count, scores.shape[1]))
    np.add.at(sums, codes, scores * w[:, None])
    return count / (count - 1) * bread @ (sums.T @ sums) @ bread


def likelihood_weights(frame, column, weight_type):
    """Weights as they enter the likelihood, and N as Stata counts it."""
    if column is None:
        return np.ones(len(frame)), len(frame)
    raw = frame[column].to_numpy(dtype=float)
    if weight_type == "fweight":
        return raw, int(raw.sum())
    if weight_type == "aweight":
        return raw * len(raw) / raw.sum(), len(raw)
    return raw, len(raw)


def dummies(frame, columns, categorical=(), intercept=True):
    """Design matrix and term names with treatment coding (first sorted level omitted)."""
    blocks, names = [], []
    if intercept:
        blocks.append(np.ones(len(frame)))
        names.append("Intercept")
    for column in columns:
        if column in categorical:
            levels = sorted(frame[column].dropna().unique().tolist())
            for level in levels[1:]:
                blocks.append((frame[column] == level).to_numpy(dtype=float))
                names.append(f"{column}[{level}]")
        else:
            blocks.append(frame[column].to_numpy(dtype=float))
            names.append(column)
    return np.column_stack(blocks), names


def table(result):
    """Coefficient table of a result as arrays."""
    rows = result.coefficients
    return {name: np.array([getattr(row, name) for row in rows])
            for name in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")}


def check_table(result, params, covariance, *, df=None, alpha=0.05, rtol=1e-6, atol=1e-9):
    """Estimates, covariance, statistics, p-values and intervals of a result."""
    got = table(result)
    errors = np.sqrt(np.diag(covariance))
    assert_allclose(got["estimate"], params, rtol=rtol, atol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), covariance, rtol=20 * rtol, atol=atol)
    assert_allclose(got["std_error"], errors, rtol=20 * rtol)
    statistics = params / errors
    assert_allclose(got["statistic"], statistics, rtol=20 * rtol, atol=1e-7)
    if df is None:
        p_values = 2 * stats.norm.sf(np.abs(statistics))
        critical = stats.norm.ppf(1 - alpha / 2)
    else:
        p_values = 2 * stats.t.sf(np.abs(statistics), df)
        critical = stats.t.ppf(1 - alpha / 2, df)
    assert_allclose(got["p_value"], p_values, rtol=1e-4, atol=1e-12)
    assert_allclose(got["ci_low"], params - critical * errors, rtol=20 * rtol, atol=1e-7)
    assert_allclose(got["ci_high"], params + critical * errors, rtol=20 * rtol, atol=1e-7)
    assert result.inference["distribution"] == ("normal" if df is None else "t")
    assert result.inference["df_inference"] == df


def check_test(entry, statistic, df, *, df2=None, rtol=1e-6):
    """A tests[...] entry against chi2(df) or F(df, df2)."""
    assert_allclose(entry["statistic"], statistic, rtol=rtol)
    assert entry["df"] == df
    if df2 is None:
        assert entry["distribution"] == "chi2"
        assert_allclose(entry["p_value"], stats.chi2.sf(statistic, df), rtol=1e-5, atol=1e-300)
    else:
        assert entry["distribution"] == "F" and entry["df2"] == df2
        assert_allclose(entry["p_value"], stats.f.sf(statistic, df, df2), rtol=1e-5, atol=1e-300)


def wald(params, covariance, indices):
    indices = list(indices)
    block = covariance[np.ix_(indices, indices)]
    return float(params[indices] @ np.linalg.solve(block, params[indices]))


def options_for(kind, cluster="firm"):
    return {"cluster": cluster} if kind == "cluster" else {"covariance": kind}


# ---- independent likelihoods ---------------------------------------------------------


def tobit_olsen(x, y, ll, ul, offset=None):
    """Per-observation tobit log likelihood in Olsen's ``(g, h) = (b/sigma, 1/sigma)``."""
    k = x.shape[1]
    shift = 0.0 if offset is None else offset
    left = np.zeros(len(y), dtype=bool) if ll is None else y <= ll
    right = np.zeros(len(y), dtype=bool) if ul is None else y >= ul
    low = 0.0 if ll is None else ll
    high = 0.0 if ul is None else ul

    def obs(theta):
        index, h = x @ theta[:k], theta[k]
        exact = np.log(h) - 0.5 * (h * (y - shift) - index) ** 2 - LOG_SQRT_2PI
        below = np.log(special.ndtr(h * (low - shift) - index))
        above = np.log(special.ndtr(index - h * (high - shift)))
        return np.where(left, below, np.where(right, above, exact))

    def reported(theta):
        return np.concatenate([theta[:k] / theta[k], [1 / theta[k]]])

    return obs, reported


def truncreg_variance(x, y, ll, ul, offset=None):
    """Per-observation truncated-normal log likelihood in ``(b, sigma^2)``."""
    k = x.shape[1]
    shift = 0.0 if offset is None else offset

    def obs(theta):
        mean, sigma = x @ theta[:k] + shift, np.sqrt(theta[k])
        density = -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma) - LOG_SQRT_2PI
        top = 1.0 if ul is None else special.ndtr((ul - mean) / sigma)
        bottom = 0.0 if ll is None else special.ndtr((ll - mean) / sigma)
        return density - np.log(top - bottom)

    def reported(theta):
        return np.concatenate([theta[:k], [np.sqrt(theta[k])]])

    return obs, reported


def intreg_sigma(x, low, high, offset=None):
    """Per-observation interval-regression log likelihood in ``(b, sigma)``."""
    k = x.shape[1]
    shift = 0.0 if offset is None else offset
    open_low, open_high = np.isnan(low), np.isnan(high)
    point = ~open_low & ~open_high & (low == high)
    a, b = np.where(open_low, 0.0, low), np.where(open_high, 0.0, high)

    def obs(theta):
        mean, sigma = x @ theta[:k] + shift, theta[k]
        density = -0.5 * ((a - mean) / sigma) ** 2 - np.log(sigma) - LOG_SQRT_2PI
        top = np.where(open_high, 1.0, special.ndtr((b - mean) / sigma))
        bottom = np.where(open_low, 0.0, special.ndtr((a - mean) / sigma))
        return np.where(point, density, np.log(np.where(point, 1.0, top - bottom)))

    def reported(theta):
        return np.concatenate([theta[:k], [np.log(theta[k])]])

    return obs, reported


def solve(obs, reported, start, kind, *, w=None, weight_type=None, groups=None):
    """Oracle estimates in the reported parameterization: (params, covariance, ll)."""
    weights = np.ones(len(obs(np.asarray(start, dtype=float)))) if w is None else w
    theta, value = newton(obs, start, weights)
    internal = stata_covariance(obs, theta, kind, w=weights, weight_type=weight_type,
                                groups=groups)
    jacobian = jacobian_cs(reported, theta)
    return reported(theta), jacobian @ internal @ jacobian.T, value


def ols_start(x, y, w=None):
    w = np.ones(len(y)) if w is None else w
    root = np.sqrt(w)
    beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    resid = y - x @ beta
    return beta, float(np.sqrt(w @ resid ** 2 / w.sum()))


# ---- data ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9173)
    n = 420
    sizes = rng.integers(2, 20, size=60)
    firm = np.repeat(np.arange(len(sizes)), sizes)[:n]
    firm = np.concatenate([firm, np.full(n - len(firm), len(sizes))])
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.exponential(size=n),
        "region": rng.choice(["north", "south", "west"], size=n, p=[0.5, 0.3, 0.2]),
        "firm": firm, "fw": rng.integers(1, 5, size=n).astype(float),
        "aw": rng.gamma(2.0, 1.0, size=n) + 0.1, "off": rng.normal(scale=0.4, size=n),
    })
    effect = frame.region.map({"north": 0.0, "south": 0.5, "west": -0.7}).to_numpy()
    shock = rng.normal(size=n) + 0.5 * rng.normal(size=firm.max() + 1)[firm]
    frame["latent"] = 1.0 + 0.8 * frame.x1 - 0.5 * frame.x2 + effect + 1.4 * shock
    frame["y"] = frame.latent.clip(0.0, 3.0)
    frame["lo"] = np.floor(frame.latent * 2) / 2
    frame["hi"] = frame.lo + 0.5
    frame.loc[frame.latent < -1.0, ["lo", "hi"]] = [np.nan, -1.0]
    frame.loc[frame.latent > 3.5, ["lo", "hi"]] = [3.5, np.nan]
    exact = np.arange(n) % 4 == 1
    frame.loc[exact, "lo"] = frame.latent[exact]
    frame.loc[exact, "hi"] = frame.latent[exact]
    return frame


X = ["x1", "x2", "region"]
CAT = ["region"]


# ---- tobit ---------------------------------------------------------------------------


def tobit_oracle(frame, kind, *, ll, ul, weights=None, weight_type=None, offset=None,
                 columns=X, intercept=True):
    x, names = dummies(frame, columns, CAT, intercept)
    y = frame.y.to_numpy()
    w, nobs = likelihood_weights(frame, weights, weight_type)
    shift = None if offset is None else frame[offset].to_numpy()
    obs, reported = tobit_olsen(x, y, ll, ul, shift)
    beta, sigma = ols_start(x, y - (0 if shift is None else shift), w)
    params, covariance, value = solve(obs, reported, np.r_[beta / sigma, 1 / sigma], kind, w=w,
                                      weight_type=weight_type, groups=frame.firm)
    null = None
    if intercept:
        ones = np.ones((len(y), 1))
        obs0, _ = tobit_olsen(ones, y, ll, ul, shift)
        null = newton(obs0, np.r_[y.mean() / y.std(), 1 / y.std()], w)[1]
    return names, params, covariance, value, null, nobs, w


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "iweight", "pweight"])
def test_tobit_every_covariance_and_weight_type(data, kind, weighting):
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        with pytest.raises(AnalysisError) as caught:
            oe.tobit(data=data, y="y", x=X, categorical=CAT, ll=0, ul=3, covariance=kind,
                     weights="aw", weight_type="pweight")
        assert caught.value.code == "unsupported_covariance"
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.tobit(data=data, y="y", x=X, categorical=CAT, ll=0, ul=3,
                      **options_for(kind), **extra)
    names, params, covariance, value, null, nobs, w = tobit_oracle(
        data, kind, ll=0.0, ul=3.0, weights=column, weight_type=weighting)
    k = len(names)
    df_model = k - 1
    df = nobs - df_model
    assert [c.term for c in result.coefficients] == [*names, "/sigma"]
    assert [c.equation for c in result.coefficients] == ["y"] * k + [None]
    check_table(result, params, covariance, df=df)
    metrics = result.metrics
    assert result.nobs == nobs and metrics["df_model"] == df_model and metrics["df_resid"] == df
    assert_allclose(metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(metrics["sigma"], params[k], rtol=1e-7)
    assert_allclose(metrics["pseudo_r_squared"], 1 - value / null, rtol=1e-8)
    assert_allclose(metrics["aic"], -2 * value + 2 * (k + 1), rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * value + np.log(nobs) * (k + 1), rtol=1e-10)
    y = data.y.to_numpy()
    counts = (w if weighting == "fweight" else np.ones(len(y)))
    assert metrics["n_left_censored"] == int(counts[y <= 0].sum())
    assert metrics["n_right_censored"] == int(counts[y >= 3].sum())
    assert metrics["n_uncensored"] == nobs - metrics["n_left_censored"] \
        - metrics["n_right_censored"]
    if kind in ("nonrobust", "opg"):
        check_test(result.tests["model"], 2 * (value - null), df_model)
    else:
        statistic = wald(params, covariance, range(1, k)) / df_model
        check_test(result.tests["model"], statistic, df_model, df2=df, rtol=1e-5)
    variance = result.extra["variance"]
    assert_allclose(variance["estimate"], params[k] ** 2, rtol=1e-7)
    assert_allclose(variance["std_error"], 2 * params[k] * np.sqrt(covariance[k, k]), rtol=1e-5)
    assert result.provenance["stata_parity_validated"] is False


def test_tobit_one_limit_offset_no_constant_and_alpha(data):
    result = oe.tobit(data=data, y="y", x=["x1", "x2"], ul=3, offset="off", intercept=False,
                      covariance="robust", alpha=0.1)
    names, params, covariance, value, null, nobs, _ = tobit_oracle(
        data, "robust", ll=None, ul=3.0, offset="off", columns=["x1", "x2"], intercept=False)
    assert null is None and names == ["x1", "x2"]
    check_table(result, params, covariance, df=nobs - 2, alpha=0.1)
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert result.metrics["pseudo_r_squared"] is None and result.metrics["n_left_censored"] == 0
    check_test(result.tests["model"], wald(params, covariance, [0, 1]) / 2, 2, df2=nobs - 2,
               rtol=1e-5)
    # With a constant, the offset also enters the constant-only comparison model.
    result = oe.tobit(data=data, y="y", x=["x1"], ll=0, offset="off")
    names, params, covariance, value, null, nobs, _ = tobit_oracle(
        data, "nonrobust", ll=0.0, ul=None, offset="off", columns=["x1"])
    check_table(result, params, covariance, df=nobs - 1)
    check_test(result.tests["model"], 2 * (value - null), 1)
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-10)


def test_tobit_limits_at_the_sample_extremes(data):
    low, high = float(data.y.min()), float(data.y.max())
    assert (low, high) == (0.0, 3.0)
    at_extremes = oe.tobit(data=data, y="y", x=["x1"], ll="min", ul="max")
    explicit = oe.tobit(data=data, y="y", x=["x1"], ll=0, ul=3)
    assert_allclose(table(at_extremes)["estimate"], table(explicit)["estimate"], rtol=1e-12)
    assert at_extremes.extra["limits"] == {"lower": 0.0, "upper": 3.0}


# ---- truncreg ------------------------------------------------------------------------


def truncreg_oracle(frame, kind, *, ll, ul, weights=None, weight_type=None, offset=None):
    y = frame.latent.to_numpy()
    keep = np.ones(len(y), dtype=bool)
    if ll is not None:
        keep &= y > ll
    if ul is not None:
        keep &= y < ul
    kept = frame[keep]
    x, names = dummies(kept, X, CAT)
    w, nobs = likelihood_weights(kept, weights, weight_type)
    shift = None if offset is None else kept[offset].to_numpy()
    obs, reported = truncreg_variance(x, y[keep], ll, ul, shift)
    beta, sigma = ols_start(x, y[keep] - (0 if shift is None else shift), w)
    params, covariance, value = solve(obs, reported, np.r_[beta, sigma ** 2], kind, w=w,
                                      weight_type=weight_type, groups=kept.firm)
    if weight_type == "fweight":
        truncated = int(frame.loc[~keep, weights].sum())
    else:
        truncated = int((~keep).sum())
    return names, params, covariance, value, nobs, truncated


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("limits", [(0.0, None), (None, 3.0), (-0.5, 3.5)])
def test_truncreg_every_covariance(data, kind, limits):
    ll, ul = limits
    result = oe.truncreg(data=data, y="latent", x=X, categorical=CAT, ll=ll, ul=ul,
                         **options_for(kind))
    names, params, covariance, value, nobs, truncated = truncreg_oracle(
        data, kind, ll=ll, ul=ul)
    k = len(names)
    assert [c.term for c in result.coefficients] == [*names, "/sigma"]
    check_table(result, params, covariance)
    assert result.nobs == nobs and result.metrics["n_truncated"] == truncated
    assert result.dropped_rows == truncated and len(result.sample_positions) == nobs
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(result.metrics["sigma"], params[k], rtol=1e-7)
    assert_allclose(result.metrics["aic"], -2 * value + 2 * (k + 1), rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * value + np.log(nobs) * (k + 1), rtol=1e-10)
    check_test(result.tests["model"], wald(params, covariance, range(1, k)), k - 1, rtol=1e-5)
    assert "pseudo_r_squared" not in result.metrics
    assert any("truncat" in warning for warning in result.warnings)


@pytest.mark.parametrize("weighting,kind", [
    ("fweight", "nonrobust"), ("fweight", "opg"), ("fweight", "robust"), ("fweight", "cluster"),
    ("aweight", "nonrobust"), ("aweight", "opg"), ("aweight", "robust"),
    ("iweight", "nonrobust"), ("iweight", "opg"), ("pweight", "robust"), ("pweight", "cluster"),
])
def test_truncreg_weights_and_offset(data, weighting, kind):
    column = "fw" if weighting == "fweight" else "aw"
    result = oe.truncreg(data=data, y="latent", x=X, categorical=CAT, ll=0, offset="off",
                         weights=column, weight_type=weighting, **options_for(kind))
    names, params, covariance, value, nobs, truncated = truncreg_oracle(
        data, kind, ll=0.0, ul=None, weights=column, weight_type=weighting, offset="off")
    check_table(result, params, covariance)
    assert result.nobs == nobs and result.metrics["n_truncated"] == truncated
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * value + np.log(nobs) * (len(names) + 1),
                    rtol=1e-10)


# ---- intreg --------------------------------------------------------------------------


def intreg_oracle(frame, kind, *, weights=None, weight_type=None, offset=None,
                  columns=X, intercept=True):
    x, names = dummies(frame, columns, CAT, intercept)
    low, high = frame.lo.to_numpy(), frame.hi.to_numpy()
    w, nobs = likelihood_weights(frame, weights, weight_type)
    shift = None if offset is None else frame[offset].to_numpy()
    obs, reported = intreg_sigma(x, low, high, shift)
    middle = np.where(np.isnan(low), high, np.where(np.isnan(high), low, (low + high) / 2))
    beta, sigma = ols_start(x, middle - (0 if shift is None else shift), w)
    params, covariance, value = solve(obs, reported, np.r_[beta, sigma], kind, w=w,
                                      weight_type=weight_type, groups=frame.firm)
    null = None
    if intercept:
        obs0, _ = intreg_sigma(np.ones((len(low), 1)), low, high, shift)
        null = newton(obs0, np.r_[middle.mean(), middle.std()], w)[1]
    return names, params, covariance, value, null, nobs, w


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "iweight", "pweight"])
def test_intreg_every_covariance_and_weight_type(data, kind, weighting):
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.intreg(data=data, y_low="lo", y_high="hi", x=X, categorical=CAT,
                       **options_for(kind), **extra)
    names, params, covariance, value, null, nobs, w = intreg_oracle(
        data, kind, weights=column, weight_type=weighting)
    k = len(names)
    assert [c.term for c in result.coefficients] == [*names, "/lnsigma"]
    assert [c.equation for c in result.coefficients] == ["lo"] * k + [None]
    check_table(result, params, covariance)
    metrics = result.metrics
    assert result.nobs == nobs and metrics["df_model"] == k - 1
    assert_allclose(metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(metrics["aic"], -2 * value + 2 * (k + 1), rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * value + np.log(nobs) * (k + 1), rtol=1e-10)
    low, high = data.lo.to_numpy(), data.hi.to_numpy()
    counts = w if weighting == "fweight" else np.ones(len(low))
    assert metrics["n_left_censored"] == int(counts[np.isnan(low)].sum())
    assert metrics["n_right_censored"] == int(counts[np.isnan(high)].sum())
    assert metrics["n_uncensored"] == int(counts[low == high].sum())
    assert metrics["n_interval"] == int(counts[low < high].sum())
    assert metrics["n_left_censored"] + metrics["n_right_censored"] + metrics["n_uncensored"] \
        + metrics["n_interval"] == nobs
    if kind in ("nonrobust", "opg"):
        check_test(result.tests["model"], 2 * (value - null), k - 1)
    else:
        check_test(result.tests["model"], wald(params, covariance, range(1, k)), k - 1,
                   rtol=1e-5)
    # sigma: exp(lnsigma), delta-method standard error, interval = exp of the lnsigma interval.
    sigma, error = np.exp(params[k]), np.sqrt(covariance[k, k])
    record = result.extra["sigma"]
    z = stats.norm.ppf(0.975)
    assert_allclose(metrics["sigma"], sigma, rtol=1e-7)
    assert_allclose([record["estimate"], record["std_error"], record["ci_low"],
                     record["ci_high"]],
                    [sigma, sigma * error, np.exp(params[k] - z * error),
                     np.exp(params[k] + z * error)], rtol=1e-5)


def test_intreg_offset_and_no_constant(data):
    result = oe.intreg(data=data, y_low="lo", y_high="hi", x=["x1", "x2"], offset="off",
                       intercept=False, covariance="opg")
    names, params, covariance, value, null, _, _ = intreg_oracle(
        data, "opg", offset="off", columns=["x1", "x2"], intercept=False)
    check_table(result, params, covariance)
    # Without a constant there is no constant-only comparison model: Wald test.
    check_test(result.tests["model"], wald(params, covariance, [0, 1]), 2, rtol=1e-5)
    result = oe.intreg(data=data, y_low="lo", y_high="hi", x=["x1"], offset="off")
    names, params, covariance, value, null, _, _ = intreg_oracle(
        data, "nonrobust", offset="off", columns=["x1"])
    check_table(result, params, covariance)
    check_test(result.tests["model"], 2 * (value - null), 1)


# ---- exact equivalences and invariances ----------------------------------------------


def estimates(result):
    return table(result)["estimate"]


def covariance_of(result):
    return np.array(result.covariance_matrix)


def test_without_limits_all_three_commands_are_normal_ml(data):
    """No censoring / truncation / intervals: OLS coefficients, sigma^2 = SSR/N, V = inverse
    information diag(sigma^2 (X'X)^-1, sigma^2 / (2N))."""
    x, _ = dummies(data, ["x1", "x2"])
    y = data.latent.to_numpy()
    n = len(y)
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    sigma2 = ((y - x @ beta) ** 2).sum() / n
    bread = sigma2 * np.linalg.inv(x.T @ x)
    value = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    tobit = oe.tobit(data=data, y="latent", x=["x1", "x2"])
    trunc = oe.truncreg(data=data, y="latent", x=["x1", "x2"])
    points = data.assign(top=data.latent)
    interval = oe.intreg(data=points, y_low="latent", y_high="top", x=["x1", "x2"])
    for result in (tobit, trunc):
        assert_allclose(estimates(result), np.r_[beta, np.sqrt(sigma2)], rtol=1e-9)
        assert_allclose(covariance_of(result)[:3, :3], bread, rtol=1e-8)
        assert_allclose(covariance_of(result)[3, 3], sigma2 / (2 * n), rtol=1e-8)
        assert_allclose(covariance_of(result)[3, :3], 0, atol=1e-12)
        assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-12)
    assert_allclose(estimates(interval), np.r_[beta, 0.5 * np.log(sigma2)], rtol=1e-9)
    assert_allclose(covariance_of(interval)[:3, :3], bread, rtol=1e-8)
    assert_allclose(covariance_of(interval)[3, 3], 1 / (2 * n), rtol=1e-8)
    assert tobit.metrics["n_uncensored"] == n and trunc.metrics["n_truncated"] == 0
    assert interval.metrics["n_uncensored"] == n and interval.metrics["n_interval"] == 0
    # Robust tobit without censoring: the sandwich of the normal likelihood, whose slope
    # block is N/(N-1) (X'X)^-1 X' diag(e^2) X (X'X)^-1 (HC0 times N/(N-1)).
    robust = oe.tobit(data=data, y="latent", x=["x1", "x2"], covariance="robust")
    resid = y - x @ beta
    inverse = np.linalg.inv(x.T @ x)
    white = inverse @ (x.T * resid ** 2) @ x @ inverse
    assert_allclose(covariance_of(robust)[:3, :3], n / (n - 1) * white, rtol=1e-8)


def test_intreg_reproduces_tobit_on_equivalent_bounds(data):
    y = data.y.to_numpy()
    bounds = data.assign(a=np.where(y <= 0, np.nan, y), b=np.where(y >= 3, np.nan, y))
    for kind in KINDS:
        tobit = oe.tobit(data=data, y="y", x=X, categorical=CAT, ll=0, ul=3,
                         **options_for(kind))
        interval = oe.intreg(data=bounds, y_low="a", y_high="b", x=X, categorical=CAT,
                             **options_for(kind))
        k = len(tobit.coefficients) - 1
        sigma = estimates(tobit)[k]
        assert_allclose(estimates(interval)[:k], estimates(tobit)[:k], rtol=1e-8)
        assert_allclose(np.exp(estimates(interval)[k]), sigma, rtol=1e-9)
        jacobian = np.diag(np.r_[np.ones(k), sigma])
        assert_allclose(jacobian @ covariance_of(interval) @ jacobian, covariance_of(tobit),
                        rtol=1e-6, atol=1e-12)
        assert_allclose(interval.metrics["log_likelihood"], tobit.metrics["log_likelihood"],
                        rtol=1e-12)
        for name in ("n_left_censored", "n_uncensored", "n_right_censored"):
            assert interval.metrics[name] == tobit.metrics[name]


def fits(frame, **options):
    """One call per censored / truncated command on the same data."""
    return {
        "tobit": oe.tobit(data=frame, y="y", x=X, categorical=CAT, ll=0, ul=3, **options),
        "truncreg": oe.truncreg(data=frame, y="latent", x=X, categorical=CAT, ll=-0.5, ul=3.5,
                                **options),
        "intreg": oe.intreg(data=frame, y_low="lo", y_high="hi", x=X, categorical=CAT,
                            **options),
    }


@pytest.mark.parametrize("kind", KINDS)
def test_frequency_weights_equal_replicated_rows(data, kind):
    replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    weighted = fits(data, weights="fw", weight_type="fweight", **options_for(kind))
    expanded = fits(replicated, **options_for(kind))
    for name in weighted:
        left, right = weighted[name], expanded[name]
        assert left.nobs == right.nobs
        assert_allclose(estimates(left), estimates(right), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance_of(left), covariance_of(right), rtol=1e-7, atol=1e-12)
        assert_allclose(table(left)["p_value"], table(right)["p_value"], rtol=1e-6, atol=1e-14)
        for key, value in right.metrics.items():
            assert_allclose(left.metrics[key], value, rtol=1e-9), (name, key)
        assert_allclose(left.tests["model"]["statistic"], right.tests["model"]["statistic"],
                        rtol=1e-7)
        assert left.tests["model"]["df"] == right.tests["model"]["df"]
        assert left.tests["model"].get("df2") == right.tests["model"].get("df2")


@pytest.mark.parametrize("kind", ["nonrobust", "opg"])
def test_integer_importance_weights_equal_frequency_weights(data, kind):
    """Regression test: opg is linear in the weights, as the information is."""
    frequency = fits(data, weights="fw", weight_type="fweight", covariance=kind)
    importance = fits(data, weights="fw", weight_type="iweight", covariance=kind)
    for name in frequency:
        assert_allclose(estimates(importance[name]), estimates(frequency[name]), rtol=1e-9)
        assert_allclose(covariance_of(importance[name]), covariance_of(frequency[name]),
                        rtol=1e-8, atol=1e-13)
        assert importance[name].nobs < frequency[name].nobs       # rows versus sum of weights


def test_analytic_weights_do_not_depend_on_their_unit(data):
    scaled = data.assign(aw=data.aw * 1e4)
    for kind in ("nonrobust", "opg", "robust"):
        base = fits(data, weights="aw", weight_type="aweight", covariance=kind)
        other = fits(scaled, weights="aw", weight_type="aweight", covariance=kind)
        for name in base:
            assert_allclose(estimates(other[name]), estimates(base[name]), rtol=1e-9)
            assert_allclose(covariance_of(other[name]), covariance_of(base[name]), rtol=1e-8,
                            atol=1e-13)


def test_row_order_and_regressor_units_do_not_matter(data):
    shuffled = data.sample(frac=1.0, random_state=5).reset_index(drop=True)
    rescaled = data.assign(x1=data.x1 * 1e4, x2=data.x2 * 1e-3)
    base = fits(data, cluster="firm")
    scale = {"x1": 1e-4, "x2": 1e3}
    for name, result in base.items():
        permuted = fits(shuffled, cluster="firm")[name]
        assert_allclose(estimates(permuted), estimates(result), rtol=1e-9)
        assert_allclose(covariance_of(permuted), covariance_of(result), rtol=1e-7, atol=1e-13)
        other = fits(rescaled, cluster="firm")[name]
        factor = np.array([scale.get(c.term, 1.0) for c in result.coefficients])
        assert_allclose(estimates(other), estimates(result) * factor, rtol=1e-7)
        assert_allclose(covariance_of(other), covariance_of(result) * np.outer(factor, factor),
                        rtol=1e-6, atol=1e-16)
        assert_allclose(other.metrics["log_likelihood"], result.metrics["log_likelihood"],
                        rtol=1e-10)


def test_outcome_units_scale_coefficients_and_sigma(data):
    """y -> c y, limits -> c limits: b and sigma scale by c; lnsigma shifts by ln c."""
    c = 1e3
    scaled = data.assign(y=data.y * c, latent=data.latent * c, lo=data.lo * c, hi=data.hi * c)
    tobit = oe.tobit(data=data, y="y", x=["x1", "x2"], ll=0, ul=3, covariance="robust")
    big = oe.tobit(data=scaled, y="y", x=["x1", "x2"], ll=0, ul=3 * c, covariance="robust")
    assert_allclose(estimates(big), c * estimates(tobit), rtol=1e-8)
    assert_allclose(covariance_of(big), c * c * covariance_of(tobit), rtol=1e-7)
    assert_allclose(table(big)["p_value"], table(tobit)["p_value"], rtol=1e-6, atol=1e-14)
    assert_allclose(big.tests["model"]["statistic"], tobit.tests["model"]["statistic"],
                    rtol=1e-7)
    trunc = oe.truncreg(data=data, y="latent", x=["x1", "x2"], ll=0)
    big = oe.truncreg(data=scaled, y="latent", x=["x1", "x2"], ll=0)
    assert_allclose(estimates(big), c * estimates(trunc), rtol=1e-7)
    interval = oe.intreg(data=data, y_low="lo", y_high="hi", x=["x1", "x2"])
    big = oe.intreg(data=scaled, y_low="lo", y_high="hi", x=["x1", "x2"])
    assert_allclose(estimates(big)[:3], c * estimates(interval)[:3], rtol=1e-7)
    assert_allclose(estimates(big)[3], estimates(interval)[3] + np.log(c), rtol=1e-9)
    assert_allclose(covariance_of(big)[3, 3], covariance_of(interval)[3, 3], rtol=1e-7)
    for tiny in (1e-8, 1e8):
        moved = data.assign(y=data.y * tiny)
        result = oe.tobit(data=moved, y="y", x=["x1", "x2"], ll=0, ul=3 * tiny)
        assert_allclose(estimates(result), tiny * estimates(
            oe.tobit(data=data, y="y", x=["x1", "x2"], ll=0, ul=3)), rtol=1e-7)


def test_missing_values_collinear_and_categorical_columns(data):
    holes = data.copy()
    holes.loc[[3, 40, 77], "x1"] = np.nan
    holes.loc[[5, 40], "y"] = np.nan
    holes.loc[[9], "fw"] = np.nan
    holes["twice"] = 2 * holes.x1 - 1                     # collinear with the constant and x1
    with pytest.raises(AnalysisError) as caught:
        oe.tobit(data=holes, y="y", x=["x1", "twice", "x2"], ll=0)
    assert caught.value.code == "missing_values"
    result = oe.tobit(data=holes, y="y", x=["x1", "twice", "x2", "region"], categorical=CAT,
                      ll=0, ul=3, missing="drop", weights="fw", weight_type="fweight")
    clean = holes.dropna(subset=["x1", "y", "fw"]).reset_index(drop=True)
    expected = oe.tobit(data=clean, y="y", x=X, categorical=CAT, ll=0, ul=3, weights="fw",
                        weight_type="fweight")
    assert [c.term for c in result.coefficients] == [c.term for c in expected.coefficients]
    assert_allclose(estimates(result), estimates(expected), rtol=1e-9)
    assert_allclose(covariance_of(result), covariance_of(expected), rtol=1e-8)
    assert result.provenance["omitted_terms"] == ["twice"]
    assert result.dropped_rows == 5 and result.nobs == int(clean.fw.sum())
    assert result.sample_positions == [i for i in range(len(holes)) if i not in (3, 5, 9, 40, 77)]
    assert any("missing" in w for w in result.warnings)
    assert any("twice" in w for w in result.warnings)
    assert result.provenance["categorical_encoding"]["region"]["reference"] == "north"
