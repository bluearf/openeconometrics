"""Independent oracles for the count family (verification pass).

Nothing here uses the implementation's formulas or the implementer's test
helpers. Every likelihood is written from the textbook density with explicit
``loggamma`` algebra on NumPy arrays; its per-observation scores come from
complex-step differentiation (exact to rounding), its Hessian from central
differences of those scores, and its maximum from a Newton iteration of our
own. Covariances (OIM, OPG, Huber-White, one- and two-way cluster) are coded
from their definitions with Stata's ``ml`` factors, and p-values and intervals
from ``scipy.stats.norm``.

The design is deliberately awkward: unbalanced clusters, a regressor in units
of 1e4 around 1e6 (the oracle fits its standardized version and maps back with
explicit linear algebra), a three-level categorical regressor, an exposure, a
collinear column and rows with missing values (``missing='drop'``).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, special, stats

import openecon as oe

X = ["x1", "big", "cat", "twin"]            # twin = 2 x1 - 3 is collinear and must be omitted
TERMS = ["Intercept", "x1", "big", "cat[b]", "cat[c]"]
BIG_MEAN, BIG_SCALE = 1e6, 1e4


# ---- data ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(90210)
    n = 640
    x1 = rng.normal(size=n)
    std = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    z1 = rng.normal(size=n)
    expo = rng.uniform(0.5, 2.0, size=n)
    sizes = rng.geometric(0.08, size=200)
    firm = np.repeat(np.arange(200), sizes)[:n]                   # unbalanced clusters
    frame = pd.DataFrame({
        "x1": x1, "big": BIG_MEAN + BIG_SCALE * std, "cat": cat, "twin": 2 * x1 - 3, "z1": z1,
        "expo": expo, "firm": firm, "region": rng.integers(0, 60, size=n),
        "fw": rng.integers(1, 4, size=n).astype(float), "rw": rng.uniform(0.3, 2.5, size=n),
    })
    eta = 0.7 + 0.35 * x1 - 0.25 * std + 0.3 * (cat == "b") - 0.2 * (cat == "c")
    mu = expo * np.exp(eta)
    excess = rng.uniform(size=n) < special.expit(-0.5 + 0.8 * z1)
    alpha = 0.6
    negbin = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    poisson = rng.poisson(mu)
    frame["zip_y"] = np.where(excess, 0, poisson).astype(float)
    frame["zinb_y"] = np.where(excess, 0, negbin).astype(float)
    frame["pois_y"] = poisson.astype(float)
    frame["nb_y"] = negbin.astype(float)
    varying = np.exp(-1.0 + 0.7 * z1)
    frame["gnb_y"] = rng.negative_binomial(1 / varying, 1 / (1 + varying * mu)).astype(float)
    frame["limit"] = rng.integers(0, 3, size=n).astype(float)
    takes_part = 0.3 + 0.7 * z1 + rng.normal(size=n) > 0
    frame["amount"] = np.where(
        takes_part, np.exp(0.8 + 0.4 * x1 - 0.2 * std + 0.5 * rng.normal(size=n)), 0.0)
    latent = 2.5 + 0.8 * x1 - 0.5 * std + 1.4 * rng.normal(size=n)
    frame["hours"] = np.where(takes_part & (latent > 0), latent, 0.0)
    frame.loc[rng.choice(n, size=18, replace=False), "x1"] = np.nan
    frame.loc[rng.choice(n, size=9, replace=False), "z1"] = np.nan
    return frame


def arrays(frame, need=("x1", "z1")):
    """Complete cases and the oracle's design (``big`` standardized) for a frame."""
    used = frame.dropna(subset=list(need)).reset_index(drop=True)
    std = (used.big.to_numpy() - BIG_MEAN) / BIG_SCALE
    x = np.column_stack([np.ones(len(used)), used.x1, std, used.cat == "b", used.cat == "c"])
    z = np.column_stack([np.ones(len(used)), used.z1])
    return used, x.astype(float), z


def back_map(size, equations=(0,)):
    """T with theta = T theta_std: undo the standardization of ``big`` (column 2)."""
    transform = np.eye(size)
    for start in equations:
        transform[start + 2, start + 2] = 1 / BIG_SCALE
        transform[start, start + 2] = -BIG_MEAN / BIG_SCALE
    return transform


# ---- densities (complex-step safe) ---------------------------------------------------------


def log_poisson(y, mu):
    return y * np.log(mu) - mu - special.gammaln(y + 1)


def log_nb2(y, mu, alpha):
    """Negative binomial with Var = mu (1 + alpha mu)."""
    m = 1 / alpha
    return (special.loggamma(y + m) - special.loggamma(m) - special.gammaln(y + 1)
            + m * np.log(m / (m + mu)) + y * np.log(mu / (m + mu)))


def log_nb1(y, mu, delta):
    """Negative binomial with Var = mu (1 + delta)."""
    m = mu / delta
    return (special.loggamma(y + m) - special.loggamma(m) - special.gammaln(y + 1)
            - m * np.log(1 + delta) + y * np.log(delta / (1 + delta)))


def cdf(link, index):
    if link == "logit":
        return 1 / (1 + np.exp(-index))
    if link == "probit":
        return special.ndtr(index)
    return 1 - np.exp(-np.exp(index))


def log_count(kind, y, mu, tail):
    if kind == "poisson":
        return log_poisson(y, mu)
    return log_nb2(y, mu, np.exp(tail)) if kind == "mean" else log_nb1(y, mu, np.exp(tail))


def truncated(kind, y, mu, tail, limit):
    """ln Pr(y | y > limit) with the lower sum written out term by term."""
    limit = np.broadcast_to(np.asarray(limit, dtype=float), y.shape)
    below = 0
    for j in range(int(limit.max()) + 1):
        term = np.exp(log_count(kind, np.full(y.shape, float(j)), mu, tail))
        below = below + np.where(limit >= j, term, 0)
    return log_count(kind, y, mu, tail) - np.log(1 - below)


# ---- numerical machinery -------------------------------------------------------------------


def score_rows(loglik, theta):
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        point = theta.astype(complex)
        point[j] += 1e-30j
        columns.append(np.imag(loglik(point)) / 1e-30)
    return np.column_stack(columns)


def hessian(loglik, theta, weights):
    theta = np.asarray(theta, dtype=float)
    out = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        step = 1e-5 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        out[:, j] = (weights @ score_rows(loglik, up) - weights @ score_rows(loglik, down)) \
            / (2 * step)
    return (out + out.T) / 2


def maximize(loglik, start, weights=None):
    """(theta, log likelihood, Hessian, score rows) at the maximum of sum w l_i."""
    start = np.asarray(start, dtype=float)
    weights = np.ones(len(np.real(loglik(start)))) if weights is None else weights
    scale = weights.mean()
    first = optimize.minimize(
        lambda t: -(weights @ np.real(loglik(t))) / scale, start,
        jac=lambda t: -(weights @ score_rows(loglik, t)) / scale, method="BFGS",
        options={"gtol": 1e-7, "maxiter": 3000})
    theta = first.x
    for _ in range(60):
        step = np.linalg.solve(-hessian(loglik, theta, weights),
                               weights @ score_rows(loglik, theta))
        theta = theta + step
        if np.max(np.abs(step) / np.maximum(1.0, np.abs(theta))) < 1e-12:
            break
    else:
        raise AssertionError("the oracle's Newton iteration did not converge")
    return (theta, float(weights @ np.real(loglik(theta))), hessian(loglik, theta, weights),
            score_rows(loglik, theta))


def cluster_sums(rows, labels):
    labels = np.asarray(labels)
    return np.stack([rows[labels == label].sum(axis=0) for label in pd.unique(labels)])


def oracle_covariance(kind, hess, rows, *, weights=None, frequency=False, groups=None):
    """Stata's ml covariance conventions written out.

    ``rows`` are unweighted score rows; ``weights`` the likelihood weights
    (None: unweighted); ``frequency`` marks them as replication counts.
    """
    n = len(rows)
    w = np.ones(n) if weights is None else weights
    bread = np.linalg.inv(-hess)
    if kind == "nonrobust":
        return bread
    if kind == "opg":
        return np.linalg.inv((rows * w[:, None]).T @ rows)
    weighted = rows * w[:, None]
    if kind == "robust":
        total = w.sum() if frequency else n
        meat = (weighted.T @ rows) if frequency else (weighted.T @ weighted)
        return total / (total - 1) * bread @ meat @ bread
    if kind == "cluster":
        sums = cluster_sums(weighted, groups)
        count = len(sums)
        return count / (count - 1) * bread @ (sums.T @ sums) @ bread
    first, second = groups
    both = pd.Series(list(zip(first, second, strict=True))).factorize()[0]
    meat = 0
    smallest = math.inf
    for labels, sign in ((first, 1), (second, 1), (both, -1)):
        sums = cluster_sums(weighted, labels)
        meat = meat + sign * sums.T @ sums
        if sign > 0:
            smallest = min(smallest, len(sums))
    assert np.linalg.eigvalsh(meat).min() > 0          # no PSD repair is involved in the check
    return smallest / (smallest - 1) * bread @ meat @ bread


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def same_matrix(actual, desired, rtol):
    """Entrywise agreement relative to sqrt(v_ii v_jj) (exact zeros stay comparable)."""
    desired = np.asarray(desired)
    scale = np.sqrt(np.outer(np.diag(desired), np.diag(desired)))
    assert np.max(np.abs(np.asarray(actual) - desired) / scale) < rtol


def check_fit(result, theta, value, covariance, *, level=0.05, rtol=1e-7, cov_rtol=5e-6):
    """Coefficients, covariance, standard errors, z, p-values and intervals."""
    assert_allclose(estimates(result), theta, rtol=rtol, atol=1e-9)
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    same_matrix(result.covariance_matrix, covariance, cov_rtol)
    errors = np.sqrt(np.diag(covariance))
    critical = stats.norm.ppf(1 - level / 2)
    for i, c in enumerate(result.coefficients):
        assert_allclose(c.std_error, errors[i], rtol=cov_rtol)
        assert_allclose(c.statistic, theta[i] / errors[i], rtol=cov_rtol, atol=1e-7)
        assert_allclose(c.p_value, 2 * stats.norm.sf(abs(theta[i] / errors[i])),
                        rtol=2e-4, atol=1e-300)
        assert_allclose([c.ci_low, c.ci_high],
                        [theta[i] - critical * errors[i], theta[i] + critical * errors[i]],
                        rtol=cov_rtol, atol=1e-7)
    assert result.inference["distribution"] == "normal" and result.inference["use_t"] is False


SCENARIOS = [
    # label, estimator keywords, weight column, weight type, covariance kind
    ("oim", {}, None, None, "nonrobust"),
    ("opg", {"covariance": "opg"}, None, None, "opg"),
    ("robust", {"covariance": "robust"}, None, None, "robust"),
    ("cluster", {"cluster": "firm"}, None, None, "cluster"),
    ("twoway", {"cluster": ["firm", "region"]}, None, None, "twoway"),
    ("fweight", {}, "fw", "fweight", "nonrobust"),
    ("fweight opg", {"covariance": "opg"}, "fw", "fweight", "opg"),
    ("fweight robust", {"covariance": "robust"}, "fw", "fweight", "robust"),
    ("fweight cluster", {"cluster": "firm"}, "fw", "fweight", "cluster"),
    ("aweight", {}, "rw", "aweight", "nonrobust"),
    ("aweight opg", {"covariance": "opg"}, "rw", "aweight", "opg"),
    ("aweight robust", {"covariance": "robust"}, "rw", "aweight", "robust"),
    ("iweight", {}, "rw", "iweight", "nonrobust"),
    ("pweight", {}, "rw", "pweight", "robust"),          # the convenience default is robust
    ("pweight cluster", {"cluster": "firm"}, "rw", "pweight", "cluster"),
]


def run_scenarios(fit, loglik, start, used, transform, *, scenarios=SCENARIOS, level=0.05):
    """Compare ``fit(**keywords)`` with the oracle for every covariance and weight type.

    Returns the unweighted oracle solution ``(theta_std, log likelihood)`` and
    the unweighted result for further checks.
    """
    n = len(used)
    cache = {}
    plain = None
    for label, keywords, column, kind, covariance in scenarios:
        weights = None
        if column is not None:
            weights = used[column].to_numpy()
            if kind == "aweight":
                weights = weights * n / weights.sum()
        key = (column, kind == "aweight")
        if key not in cache:
            cache[key] = maximize(loglik, start if not cache else cache[(None, False)][0],
                                  weights)
        theta, value, hess, rows = cache[key]
        groups = None
        if covariance == "cluster":
            groups = used.firm.to_numpy()
        elif covariance == "twoway":
            groups = (used.firm.to_numpy(), used.region.to_numpy())
        expected = oracle_covariance(covariance, hess, rows, weights=weights,
                                     frequency=kind == "fweight", groups=groups)
        options = dict(keywords)
        if column is not None:
            options.update(weights=column, weight_type=kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit(alpha=level, **options)
        try:
            check_fit(result, transform @ theta, value, transform @ expected @ transform.T,
                      level=level)
        except AssertionError as error:
            raise AssertionError(f"scenario '{label}': {error}") from error
        assert result.spec.covariance == ("cluster" if "cluster" in keywords else
                                          "robust" if covariance == "robust" else covariance)
        assert result.nobs == (int(used.fw.sum()) if kind == "fweight" else n)
        assert result.provenance["omitted_terms"] == ["twin"]
        assert result.provenance["stata_parity_validated"] is False
        assert result.dropped_rows == result.nobs_original - n
        if label == "oim":
            plain = result
            assert "observed information" in result.inference["correction"]
        if label == "robust":
            assert "N/(N-1)" in result.inference["correction"]
            assert_allclose(result.inference["small_sample_correction"], n / (n - 1))
        if label == "cluster":
            count = used.firm.nunique()
            assert "G/(G-1)" in result.inference["correction"]
            assert result.inference["cluster_count"] == count
            assert_allclose(result.inference["small_sample_correction"], count / (count - 1))
        if label == "twoway":
            smallest = min(used.firm.nunique(), used.region.nunique())
            assert result.inference["cluster_count"] == smallest
            assert_allclose(result.inference["small_sample_correction"],
                            smallest / (smallest - 1))
            assert not any("positive semidefinite" in text for text in result.warnings)
    theta, value = cache[(None, False)][:2]
    return theta, value, plain


def information_criteria(result, value, parameters, n):
    assert_allclose(result.metrics["aic"], -2 * value + 2 * parameters, rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * value + parameters * math.log(n), rtol=1e-10)
    assert result.inference["n_parameters"] == parameters
    assert result.inference["df_resid"] == n - parameters


def lr_matches(test, value, restricted, df):
    statistic = 2 * (value - restricted)
    assert test["distribution"] == "chi2" and test["df"] == df
    assert test["label"].startswith("LR chi2")
    assert_allclose(test["statistic"], statistic, rtol=1e-7)
    assert_allclose(test["p_value"], stats.chi2.sf(statistic, df), rtol=1e-5, atol=1e-300)


def chibar_matches(test, value, restricted):
    statistic = 2 * (value - restricted)
    assert test["distribution"] == "chibar2" and test["df"] == 1
    assert_allclose(test["statistic"], statistic, rtol=1e-7)
    assert_allclose(test["p_value"], 0.5 * stats.chi2.sf(statistic, 1), rtol=1e-5, atol=1e-300)


def wald_matches(test, result, positions):
    beta = estimates(result)[positions]
    block = np.array(result.covariance_matrix)[np.ix_(positions, positions)]
    statistic = beta @ np.linalg.solve(block, beta)
    assert test["distribution"] == "chi2" and test["df"] == len(positions)
    assert test["label"].startswith("Wald chi2")
    assert_allclose(test["statistic"], statistic, rtol=1e-8)
    assert_allclose(test["p_value"], stats.chi2.sf(statistic, len(positions)), rtol=1e-5,
                    atol=1e-300)


def exponentiated_matches(record, result, position, level=0.05):
    """A parameter estimated in logs: delta-method SE and exponentiated interval."""
    c = result.coefficients[position]
    critical = stats.norm.ppf(1 - level / 2)
    assert_allclose(record["estimate"], math.exp(c.estimate), rtol=1e-12)
    assert_allclose(record["std_error"], math.exp(c.estimate) * c.std_error, rtol=1e-12)
    assert_allclose([record["ci_low"], record["ci_high"]],
                    [math.exp(c.estimate - critical * c.std_error),
                     math.exp(c.estimate + critical * c.std_error)], rtol=1e-10)


# ---- zip / zinb ----------------------------------------------------------------------------


def zero_inflated(y, x, z, offset, link, negbin):
    k, q = x.shape[1], z.shape[1]

    def loglik(theta):
        mu = np.exp(x @ theta[:k] + offset)
        inflate = cdf(link, z @ theta[k:k + q])
        base = log_nb2(y, mu, np.exp(theta[k + q])) if negbin else log_poisson(y, mu)
        return np.where(y == 0, np.log(inflate + (1 - inflate) * np.exp(base)),
                        np.log(1 - inflate) + base)

    return loglik


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("negbin", [False, True])
def test_zero_inflated_models_against_the_oracle(data, link, negbin):
    used, x, z = arrays(data)
    n, k = len(used), x.shape[1]
    column = "zinb_y" if negbin else "zip_y"
    y, offset = used[column].to_numpy(), np.log(used.expo.to_numpy())
    loglik = zero_inflated(y, x, z, offset, link, negbin)
    start = np.r_[math.log(y.mean()), np.zeros(k - 1), 0.0, 0.0, [-0.5] * negbin]
    transform = back_map(len(start))
    estimator = oe.zinb if negbin else oe.zip

    def fit(**options):
        return estimator(data=data, y=column, x=X, inflate=["z1"], categorical=["cat"],
                         inflate_link=link, exposure="expo", missing="drop", **options)

    theta, value, result = run_scenarios(fit, loglik, start, used, transform, level=0.1)
    assert [c.term for c in result.coefficients] == TERMS + [
        "inflate:Intercept", "inflate:z1"] + ["/lnalpha"] * negbin
    assert [c.equation for c in result.coefficients] == [column] * k + ["inflate"] * 2 + (
        [None] if negbin else [])
    information_criteria(result, value, len(theta), n)
    assert result.metrics["n_zero_observations"] == (y == 0).sum()

    # LR chi2(k - 1): the comparison model keeps the inflation equation (and alpha).
    null = zero_inflated(y, x[:, :1], z, offset, link, negbin)
    restricted = maximize(null, np.r_[theta[0], theta[k:]])[1]
    lr_matches(result.tests["model"], value, restricted, k - 1)
    assert_allclose(result.extra["null_log_likelihood"], restricted, rtol=1e-10)

    # Vuong (1989) against the model without inflation, each at its own MLE.
    if negbin:
        def plain(t):
            return log_nb2(y, np.exp(x @ t[:k] + offset), np.exp(t[k]))
        plain_theta, plain_value = maximize(plain, np.r_[theta[:k], -0.3])[:2]
    else:
        def plain(t):
            return log_poisson(y, np.exp(x @ t + offset))
        plain_theta, plain_value = maximize(plain, theta[:k])[:2]
    m = np.real(loglik(theta)) - np.real(plain(plain_theta))
    sd = m.std(ddof=1)
    vuong = math.sqrt(n) * m.mean() / sd
    assert_allclose(result.tests["vuong"]["statistic"], vuong, rtol=1e-7)
    assert_allclose(result.tests["vuong"]["p_value"], stats.norm.sf(vuong), rtol=1e-5)
    record = result.extra["vuong"]
    assert_allclose(record["z_aic"], math.sqrt(n) * (m.mean() - 2 / n) / sd, rtol=1e-7)
    assert_allclose(record["z_bic"], math.sqrt(n) * (m.mean() - 2 * math.log(n) / (2 * n)) / sd,
                    rtol=1e-7)
    assert_allclose(record["p_value_two_sided"], 2 * stats.norm.sf(abs(vuong)), rtol=1e-5)
    assert record["observations"] == n and record["extra_parameters"] == 2
    name = "nbreg_log_likelihood" if negbin else "poisson_log_likelihood"
    assert_allclose(result.extra[name], plain_value, rtol=1e-10)

    if negbin:
        # LR test of alpha = 0 against zip: chibar2(01), half the chi2(1) tail.
        zip_value = maximize(zero_inflated(y, x, z, offset, link, False), theta[:k + 2])[1]
        chibar_matches(result.tests["alpha"], value, zip_value)
        assert_allclose(result.metrics["alpha"], math.exp(theta[-1]), rtol=1e-7)
        exponentiated_matches(result.extra["alpha"], result, len(theta) - 1, level=0.1)

    # Robust fits report the Wald test of the count slopes and no likelihood-based tests.
    robust = fit(covariance="robust")
    wald_matches(robust.tests["model"], robust, [1, 2, 3, 4])
    assert "vuong" not in robust.tests and "alpha" not in robust.tests

    # Chart sample: E[y] = (1 - F) mu at the estimates.
    fitted = (1 - cdf(link, z @ theta[k:k + 2])) * np.exp(x @ theta[:k] + offset)
    rows = data.dropna(subset=["x1", "z1"]).index.to_numpy()
    lookup = {row: i for i, row in enumerate(rows)}
    for point in result.predictions[:25]:
        i = lookup[point["row"]]
        assert_allclose([point["observed"], point["fitted"]], [y[i], fitted[i]], rtol=1e-7)


def test_zero_inflated_frequency_weighted_tests_equal_expanded_data(data):
    """Vuong, LR and alpha tests under fweights are those of the duplicated rows."""
    used = data.dropna(subset=["x1", "z1"]).reset_index(drop=True)
    expanded = used.loc[used.index.repeat(used.fw.astype(int))].reset_index(drop=True)
    common = dict(y="zinb_y", x=["x1", "cat"], inflate=["z1"], categorical=["cat"],
                  exposure="expo")
    weighted = oe.zinb(data=used, weights="fw", weight_type="fweight", **common)
    repeated = oe.zinb(data=expanded, **common)
    assert weighted.nobs == repeated.nobs == int(used.fw.sum())
    assert_allclose(estimates(weighted), estimates(repeated), rtol=1e-9)
    assert_allclose(weighted.covariance_matrix, repeated.covariance_matrix, rtol=1e-7)
    for name in ("model", "alpha", "vuong"):
        assert_allclose(weighted.tests[name]["statistic"], repeated.tests[name]["statistic"],
                        rtol=1e-8)
    for name in ("z", "z_aic", "z_bic", "mean_difference", "sd_difference"):
        assert_allclose(weighted.extra["vuong"][name], repeated.extra["vuong"][name], rtol=1e-8)
    for name in ("log_likelihood", "aic", "bic", "n_zero_observations", "alpha"):
        assert_allclose(weighted.metrics[name], repeated.metrics[name], rtol=1e-10)


# ---- tpoisson / tnbreg ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["poisson", "mean", "constant"])
@pytest.mark.parametrize("limit", [0, 2, "limit"])
def test_truncated_models_against_the_oracle(data, kind, limit):
    column = "pois_y" if kind == "poisson" else "nb_y"
    bound = data.limit if limit == "limit" else limit
    sample = data[data[column] > bound]
    used, x, _ = arrays(sample, need=["x1"])              # z1 is not part of these models
    n, k = len(used), x.shape[1]
    y, offset = used[column].to_numpy(), np.log(used.expo.to_numpy())
    cut = used.limit.to_numpy() if limit == "limit" else limit

    def loglik(theta, design=x):
        size = design.shape[1]
        tail = theta[size] if kind != "poisson" else None
        return truncated(kind, y, np.exp(design @ theta[:size] + offset), tail, cut)

    start = np.r_[math.log(y.mean()), np.zeros(k - 1), [-0.5] * (kind != "poisson")]
    transform = back_map(len(start))

    def fit(**options):
        if kind == "poisson":
            return oe.tpoisson(data=sample, y=column, x=X, categorical=["cat"], ll=limit,
                               exposure="expo", missing="drop", **options)
        return oe.tnbreg(data=sample, y=column, x=X, categorical=["cat"], ll=limit,
                         dispersion=kind, exposure="expo", missing="drop", **options)

    scenarios = SCENARIOS if limit != 2 else SCENARIOS[:4] + SCENARIOS[5:6] + SCENARIOS[-2:-1]
    theta, value, result = run_scenarios(fit, loglik, start, used, transform,
                                         scenarios=scenarios)
    ancillary = {"poisson": [], "mean": ["/lnalpha"], "constant": ["/lndelta"]}[kind]
    assert [c.term for c in result.coefficients] == TERMS + ancillary
    information_criteria(result, value, len(theta), n)

    # LR chi2 against the constant-only model (free dispersion) and McFadden's R2.
    restricted = maximize(lambda t: loglik(t, x[:, :1]), np.r_[theta[0], theta[k:]])[1]
    lr_matches(result.tests["model"], value, restricted, k - 1)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - value / restricted, rtol=1e-8)
    assert_allclose(result.extra["null_log_likelihood"], restricted, rtol=1e-10)

    if kind != "poisson":
        poisson_value = maximize(
            lambda t: truncated("poisson", y, np.exp(x @ t + offset), None, cut), theta[:k])[1]
        chibar_matches(result.tests["alpha"], value, poisson_value)
        name = "alpha" if kind == "mean" else "delta"
        assert_allclose(result.metrics[name], math.exp(theta[-1]), rtol=1e-7)
        assert result.extra["alpha"]["parameter"] == name
        exponentiated_matches(result.extra["alpha"], result, len(theta) - 1)

    robust = fit(covariance="robust")
    wald_matches(robust.tests["model"], robust, [1, 2, 3, 4])
    assert "alpha" not in robust.tests
    assert_allclose(robust.metrics["pseudo_r_squared"], 1 - value / restricted, rtol=1e-8)

    # Chart sample: the conditional mean E[y | y > ll], summed from the definition.
    mu = np.exp(x @ theta[:k] + offset)
    tail = theta[k] if kind != "poisson" else None
    limits = np.broadcast_to(np.asarray(cut, dtype=float), y.shape)
    mass = partial = 0
    for j in range(int(limits.max()) + 1):
        term = np.where(limits >= j,
                        np.exp(log_count(kind, np.full(y.shape, float(j)), mu, tail)), 0)
        mass, partial = mass + term, partial + j * term
    conditional = (mu - partial) / (1 - mass)
    lookup = {row: i for i, row in enumerate(sample.dropna(subset=["x1"]).index)}
    for point in result.predictions[:25]:
        i = lookup[sample.index[point["row"]]]
        assert_allclose(point["fitted"], conditional[i], rtol=1e-6)


# ---- hurdle --------------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
@pytest.mark.parametrize("negbin", [False, True])
def test_count_hurdle_against_the_oracle(data, link, negbin):
    used, x, z = arrays(data)
    n, k = len(used), x.shape[1]
    y, offset = used.zinb_y.to_numpy(), np.log(used.expo.to_numpy())
    positive = y > 0
    safe = np.where(positive, y, 1.0)
    kind = "mean" if negbin else "poisson"

    def loglik(theta, design=x):
        size = design.shape[1]
        mu = np.exp(design @ theta[:size] + offset)
        participate = cdf(link, z @ theta[size:size + 2])
        tail = theta[size + 2] if negbin else None
        return np.where(positive, np.log(participate) + truncated(kind, safe, mu, tail, 0),
                        np.log(1 - participate))

    start = np.r_[math.log(y[positive].mean()), np.zeros(k - 1), 0.0, 0.0, [-0.5] * negbin]
    transform = back_map(len(start))

    def fit(**options):
        return oe.hurdle(data=data, y="zinb_y", x=X, select_x=["z1"], categorical=["cat"],
                         dist="nbinomial" if negbin else "poisson", zero_link=link,
                         exposure="expo", missing="drop", **options)

    scenarios = SCENARIOS if link == "logit" else SCENARIOS[:4] + SCENARIOS[7:8] + SCENARIOS[13:14]
    theta, value, result = run_scenarios(fit, loglik, start, used, transform,
                                         scenarios=scenarios)
    assert [c.term for c in result.coefficients] == TERMS + [
        "select:Intercept", "select:z1"] + ["/lnalpha"] * negbin
    assert [c.equation for c in result.coefficients] == ["zinb_y"] * k + ["select"] * 2 + (
        [None] if negbin else [])
    information_criteria(result, value, len(theta), n)
    assert result.metrics["n_zero_observations"] == (~positive).sum()

    restricted = maximize(lambda t: loglik(t, x[:, :1]), np.r_[theta[0], theta[k:]])[1]
    lr_matches(result.tests["model"], value, restricted, k - 1)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - value / restricted, rtol=1e-8)
    if negbin:
        def poisson_hurdle(t):
            mu = np.exp(x @ t[:k] + offset)
            return np.where(positive, np.log(cdf(link, z @ t[k:])) + truncated(
                "poisson", safe, mu, None, 0), np.log(1 - cdf(link, z @ t[k:])))
        chibar_matches(result.tests["alpha"], value, maximize(poisson_hurdle, theta[:k + 2])[1])
        assert_allclose(result.metrics["alpha"], math.exp(theta[-1]), rtol=1e-7)
        exponentiated_matches(result.extra["alpha"], result, len(theta) - 1)

    # Chart sample: E[y] = Pr(y > 0) mu / (1 - f(0)).
    mu = np.exp(x @ theta[:k] + offset)
    zero = np.exp(log_count(kind, np.zeros(n), mu, theta[-1] if negbin else None))
    fitted = cdf(link, z @ theta[k:k + 2]) * mu / (1 - zero)
    lookup = {row: i for i, row in enumerate(data.dropna(subset=["x1", "z1"]).index)}
    for point in result.predictions[:25]:
        assert_allclose(point["fitted"], fitted[lookup[point["row"]]], rtol=1e-6)


# ---- churdle -------------------------------------------------------------------------------


@pytest.mark.parametrize(("model", "limit"), [("exponential", 0.0), ("exponential", 1.2),
                                              ("linear", 0.0), ("linear", 0.8)])
@pytest.mark.parametrize("link", ["probit", "logit"])
def test_cragg_hurdle_against_the_oracle(data, model, limit, link):
    used, x, z = arrays(data)
    x = x[:, :3]                                           # Intercept, x1, big
    n, k = len(used), 3
    column = "amount" if model == "exponential" else "hours"
    y = used[column].to_numpy()
    above = y > limit
    safe = np.where(above, y, limit + 1.0)

    def loglik(theta, design=x):
        size = design.shape[1]
        xb, log_sigma = design @ theta[:size], theta[size + 2]
        sigma = np.exp(log_sigma)
        participate = cdf(link, z @ theta[size:size + 2])
        if model == "linear":
            density = (-0.5 * ((safe - xb) / sigma) ** 2 - 0.5 * math.log(2 * math.pi)
                       - log_sigma - np.log(special.ndtr((xb - limit) / sigma)))
        else:
            density = (-0.5 * ((np.log(safe) - xb) / sigma) ** 2 - 0.5 * math.log(2 * math.pi)
                       - log_sigma - np.log(safe))
            if limit > 0:
                density = density - np.log(special.ndtr((xb - math.log(limit)) / sigma))
        return np.where(above, np.log(participate) + density, np.log(1 - participate))

    target = np.log(y[above]) if model == "exponential" else y[above]
    start = np.r_[target.mean(), 0.0, 0.0, 0.0, 0.0, math.log(target.std())]
    transform = back_map(len(start))

    def fit(**options):
        return oe.churdle(data=data, y=column, x=["x1", "big", "twin"], select_x=["z1"],
                          model=model, ll=limit, select_link=link, missing="drop", **options)

    scenarios = SCENARIOS if (link == "probit" and limit == 0) else \
        SCENARIOS[:4] + SCENARIOS[5:6] + SCENARIOS[13:14]
    theta, value, result = run_scenarios(fit, loglik, start, used, transform,
                                         scenarios=scenarios)
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "big", "select:Intercept", "select:z1", "/lnsigma"]
    assert [c.equation for c in result.coefficients] == [column] * 3 + ["select"] * 2 + [None]
    information_criteria(result, value, 6, n)
    assert result.extra["n_bounded_observations"] == (~above).sum()

    restricted = maximize(lambda t: loglik(t, x[:, :1]), np.r_[theta[0], theta[k:]])[1]
    lr_matches(result.tests["model"], value, restricted, k - 1)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - value / restricted, rtol=1e-8)
    assert_allclose(result.metrics["sigma"], math.exp(theta[-1]), rtol=1e-7)
    exponentiated_matches(result.extra["sigma"], result, 5)
    robust = fit(covariance="robust")
    wald_matches(robust.tests["model"], robust, [1, 2])

    # Chart sample: E[y] = (1 - P) ll + P E[y | y > ll], by quadrature of the density.
    sigma = math.exp(theta[-1])
    participate = cdf(link, z @ theta[k:k + 2])
    lookup = {row: i for i, row in enumerate(data.dropna(subset=["x1", "z1"]).index)}
    from scipy import integrate

    for point in result.predictions[:6]:
        i = lookup[point["row"]]
        xb = x[i] @ theta[:k]
        if model == "linear":
            mass = stats.norm.sf((limit - xb) / sigma)
            mean = integrate.quad(lambda v: v * stats.norm.pdf((v - xb) / sigma) / sigma,
                                  limit, xb + 12 * sigma)[0] / mass
        else:
            low = math.log(limit) if limit > 0 else xb - 12 * sigma
            mass = stats.norm.sf((low - xb) / sigma)
            mean = integrate.quad(lambda v: math.exp(v) * stats.norm.pdf((v - xb) / sigma)
                                  / sigma, low, xb + 14 * sigma)[0] / mass
        expected = (1 - participate[i]) * limit + participate[i] * mean
        assert_allclose(point["fitted"], expected, rtol=1e-6)


# ---- gnbreg --------------------------------------------------------------------------------


def test_generalized_negative_binomial_against_the_oracle(data):
    used, x, z = arrays(data)
    n, k = len(used), x.shape[1]
    y, offset = used.gnb_y.to_numpy(), np.log(used.expo.to_numpy())

    def loglik(theta, design=x, shape=z):
        size = design.shape[1]
        return log_nb2(y, np.exp(design @ theta[:size] + offset), np.exp(shape @ theta[size:]))

    start = np.r_[math.log(y.mean()), np.zeros(k - 1), -0.5, 0.0]
    transform = back_map(len(start))

    def fit(**options):
        return oe.gnbreg(data=data, y="gnb_y", x=X, lnalpha=["z1"], categorical=["cat"],
                         exposure="expo", missing="drop", **options)

    theta, value, result = run_scenarios(fit, loglik, start, used, transform)
    assert [c.term for c in result.coefficients] == TERMS + ["lnalpha:Intercept", "lnalpha:z1"]
    assert [c.equation for c in result.coefficients] == ["gnb_y"] * k + ["lnalpha"] * 2
    information_criteria(result, value, k + 2, n)

    restricted = maximize(lambda t: loglik(t, x[:, :1]), np.r_[theta[0], theta[k:]])[1]
    lr_matches(result.tests["model"], value, restricted, k - 1)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - value / restricted, rtol=1e-8)
    constant = maximize(lambda t: loglik(t, x, z[:, :1]), theta[:k + 1])[1]
    lr_matches(result.tests["lnalpha"], value, constant, 1)
    assert_allclose(result.extra["nbreg_log_likelihood"], constant, rtol=1e-10)
    fitted_alpha = np.exp(z @ theta[k:])
    assert_allclose([result.extra["alpha"][name] for name in ("mean", "min", "max")],
                    [fitted_alpha.mean(), fitted_alpha.min(), fitted_alpha.max()], rtol=1e-6)
    robust = fit(covariance="robust")
    wald_matches(robust.tests["model"], robust, [1, 2, 3, 4])
    wald_matches(robust.tests["lnalpha"], robust, [k + 1])
    mu = np.exp(x @ theta[:k] + offset)
    lookup = {row: i for i, row in enumerate(data.dropna(subset=["x1", "z1"]).index)}
    for point in result.predictions[:25]:
        assert_allclose(point["fitted"], mu[lookup[point["row"]]], rtol=1e-7)


# ---- likelihood kernels at arbitrary points ------------------------------------------------


def _kernel_cases():
    """(name, pieces factory, designs, offsets, independent likelihood, theta)."""
    import torch

    from openecon.econometrics.count import kernels

    rng = np.random.default_rng(31337)
    n = 90
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, size=n)])
    z = np.column_stack([np.ones(n), rng.normal(size=n)])
    offset = rng.normal(scale=0.2, size=n)
    y = rng.negative_binomial(1.5, 0.35, size=n).astype(float)
    y[::7] = 0
    limit = rng.integers(0, 4, size=n).astype(float)
    above = y + limit + 1                                      # outcomes beyond the limits
    amounts = np.exp(rng.normal(0.5, 0.6, size=n)) + 0.7
    tensor = lambda a: torch.tensor(a, dtype=torch.float64)    # noqa: E731
    xt, zt, ot = tensor(x), tensor(z), tensor(offset)
    beta, gamma = np.array([0.4, 0.3, -0.5]), np.array([-0.3, 0.6])
    cases = []
    for link in ("logit", "probit"):
        for negbin in (False, True):
            density = kernels.NegBinDensity() if negbin else kernels.PoissonDensity()
            cases.append((
                f"zero-inflated {link} negbin={negbin}",
                kernels.ZeroInflatedPieces(density, tensor(y), link),
                [xt, zt] + [None] * negbin, [ot, None] + [None] * negbin,
                zero_inflated(y, x, z, offset, link, negbin),
                np.r_[beta, gamma, [-0.4] * negbin]))
    for kind in ("poisson", "mean", "constant"):
        for cut in (0, 3, limit):
            density = kernels.PoissonDensity() if kind == "poisson" else \
                kernels.NegBinDensity(kind)
            bound = tensor(cut) if isinstance(cut, np.ndarray) else cut
            cases.append((
                f"truncated {kind} limit={'column' if isinstance(cut, np.ndarray) else cut}",
                kernels.TruncatedPieces(density, tensor(above), bound),
                [xt] + [None] * (kind != "poisson"), [ot] + [None] * (kind != "poisson"),
                lambda t, kind=kind, cut=cut: truncated(
                    kind, above, np.exp(x @ t[:3] + offset), t[3] if kind != "poisson" else None,
                    cut),
                np.r_[beta + [0.9, 0, 0], [-0.7] * (kind != "poisson")]))
    cases.append((
        "generalized negative binomial",
        kernels.CountPieces(kernels.NegBinDensity("mean"), tensor(y)), [xt, zt], [ot, None],
        lambda t: log_nb2(y, np.exp(x @ t[:3] + offset), np.exp(z @ t[3:])),
        np.r_[beta, -0.6, 0.5]))
    for link in ("logit", "probit", "cloglog"):
        cases.append((
            f"binary {link}", kernels.BinaryPieces(tensor(y) > 0, link), [zt], [None],
            lambda t, link=link: np.where(y > 0, np.log(cdf(link, z @ t)),
                                          np.log(1 - cdf(link, z @ t))),
            gamma))
    for cut in (None, 0.4):
        cases.append((
            f"truncated normal limit={cut}",
            kernels.TruncatedNormalPieces(tensor(amounts), cut, tensor(-np.log(amounts))),
            [xt, None], [None, None],
            lambda t, cut=cut: (-0.5 * ((amounts - x @ t[:3]) / np.exp(t[3])) ** 2
                                - 0.5 * math.log(2 * math.pi) - t[3] - np.log(amounts)
                                - (0 if cut is None else np.log(
                                    special.ndtr((x @ t[:3] - cut) / np.exp(t[3]))))),
            np.r_[1.2, 0.2, -0.3, -0.2]))
    return cases


def test_kernel_values_scores_and_hessians_match_independent_likelihoods():
    import torch

    from openecon.econometrics.count.kernels import IndexObjective
    from openecon.engines.optimize import check_derivatives

    rng = np.random.default_rng(5)
    cases = _kernel_cases()
    assert len(cases) == 19
    for name, pieces, designs, offsets, loglik, theta in cases:
        n = len(np.real(loglik(theta)))
        weights = rng.uniform(0.5, 2.0, size=n)
        objective = IndexObjective(pieces, designs, torch.tensor(weights), offsets)
        point = torch.tensor(theta, dtype=torch.float64)
        value, gradient, hess = objective(point)
        rows = score_rows(loglik, theta)
        assert_allclose(float(value), weights @ np.real(loglik(theta)), rtol=1e-12, err_msg=name)
        assert_allclose(objective.observations(point).numpy(), np.real(loglik(theta)),
                        rtol=1e-10, atol=1e-12, err_msg=name)
        assert_allclose(gradient.numpy(), weights @ rows, rtol=1e-9, atol=1e-9, err_msg=name)
        assert_allclose(objective.score_rows(point).numpy(), rows, rtol=1e-8, atol=1e-10,
                        err_msg=name)
        expected = hessian(loglik, theta, weights)
        assert_allclose(hess.numpy(), expected, rtol=2e-6, atol=2e-6 * np.abs(expected).max(),
                        err_msg=name)
        report = check_derivatives(objective, point)
        assert report["gradient_max_rel_error"] < 1e-7, (name, report)
        assert report["hessian_max_rel_error"] < 1e-6, (name, report)
        assert report["hessian_asymmetry"] < 1e-9 * max(1.0, float(hess.abs().max())), name


def test_zero_inflation_vanishes_into_the_plain_count_model():
    """With F(z'g) -> 0 the zero-inflated likelihood is the plain count likelihood."""
    import torch

    from openecon.econometrics.count import kernels

    rng = np.random.default_rng(8)
    y = rng.poisson(2.0, size=50).astype(float)
    eta = rng.normal(0.5, 0.3, size=50)
    index = [torch.tensor(eta), torch.full((50,), -45.0, dtype=torch.float64)]
    for link in ("logit", "probit"):
        if link == "probit":
            index[1] = torch.full((50,), -12.0, dtype=torch.float64)
        pieces = kernels.ZeroInflatedPieces(kernels.PoissonDensity(), torch.tensor(y), link)
        assert_allclose(pieces(index, False)[0].numpy(), log_poisson(y, np.exp(eta)),
                        rtol=1e-12, atol=1e-12)


# ---- statsmodels where the identical model exists ------------------------------------------


def test_statsmodels_fits_agree_where_the_model_is_the_same(data):
    from statsmodels.discrete.count_model import ZeroInflatedPoisson
    from statsmodels.discrete.truncated_model import HurdleCountModel, TruncatedLFPoisson
    from statsmodels.tools.numdiff import approx_hess

    used, x, z = arrays(data)
    design = pd.DataFrame(x, columns=["const", "x1", "std", "b", "c"])
    frame = pd.concat([used, design[["std", "b", "c"]]], axis=1)
    names = ["x1", "std", "b", "c"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for link in ("logit", "probit"):
            reference = ZeroInflatedPoisson(
                used.zip_y, x, exog_infl=z, inflation=link, offset=np.log(used.expo)).fit(
                method="newton", maxiter=200, disp=0, tol=1e-12)
            ours = oe.zip(data=frame, y="zip_y", x=names, inflate=["z1"], inflate_link=link,
                          exposure="expo")
            # statsmodels orders the inflation parameters first.
            assert_allclose(estimates(ours), np.r_[reference.params[2:], reference.params[:2]],
                            rtol=1e-6, atol=1e-8)
            assert_allclose(ours.metrics["log_likelihood"], reference.llf, rtol=1e-10)
            assert_allclose(ours.metrics["aic"], reference.aic, rtol=1e-10)
            assert_allclose(ours.metrics["bic"], reference.bic, rtol=1e-10)
            # statsmodels 0.14's analytic ZIP Hessian leaves the inflation-count cross block
            # at zero, so its cov_params() is not the inverse information; its own likelihood
            # differentiated numerically is.
            order = [2, 3, 4, 5, 6, 0, 1]
            information = -approx_hess(np.asarray(reference.params), reference.model.loglike)
            same_matrix(ours.covariance_matrix,
                        np.linalg.inv(information)[np.ix_(order, order)], 2e-4)
        positive = frame[frame.pois_y > 2].reset_index(drop=True)
        xp = np.column_stack([np.ones(len(positive)), positive[names]])
        reference = TruncatedLFPoisson(positive.pois_y, xp, truncation=2,
                                       offset=np.log(positive.expo)).fit(
            method="newton", maxiter=200, disp=0, tol=1e-12)
        ours = oe.tpoisson(data=positive, y="pois_y", x=names, ll=2, exposure="expo")
        assert_allclose(estimates(ours), reference.params, rtol=1e-7)
        assert_allclose(ours.metrics["log_likelihood"], reference.llf, rtol=1e-10)
        same_matrix(ours.covariance_matrix, reference.cov_params(), 1e-5)
        # statsmodels' HC0 sandwich times N/(N-1) is Stata's vce(robust).
        sandwich = TruncatedLFPoisson(positive.pois_y, xp, truncation=2,
                                      offset=np.log(positive.expo)).fit(
            method="newton", maxiter=200, disp=0, tol=1e-12, cov_type="HC0")
        robust = oe.tpoisson(data=positive, y="pois_y", x=names, ll=2, exposure="expo",
                             covariance="robust")
        count = len(positive)
        same_matrix(robust.covariance_matrix,
                    np.asarray(sandwich.cov_params()) * count / (count - 1), 1e-5)
        hurdle = HurdleCountModel(frame.zinb_y, x[:, :2], dist="poisson",
                                  zerodist="poisson").fit(disp=0, maxiter=300)
        ours = oe.hurdle(data=frame, y="zinb_y", x=["x1"], zero_link="cloglog")
        assert_allclose(ours.metrics["log_likelihood"], hurdle.llf, rtol=1e-8)
        # (statsmodels stops its two fits at a looser tolerance than ours.)
        assert_allclose(estimates(ours)[:2], hurdle.results_count.params, rtol=2e-4)
        # Mullahy's Poisson zero hurdle is the complementary log-log participation model.
        assert_allclose(estimates(ours)[2:], hurdle.results_zero.params, rtol=2e-4)


# ---- equivalences between estimators --------------------------------------------------------


def _complete(data):
    return data.dropna(subset=["x1", "z1"]).reset_index(drop=True)


def test_gnbreg_with_a_constant_dispersion_is_nbreg(data):
    used = _complete(data)
    common = dict(data=used, y="gnb_y", x=["x1", "cat"], categorical=["cat"], exposure="expo")
    for options in ({}, {"covariance": "robust"}, {"cluster": "firm"},
                    {"weights": "fw", "weight_type": "fweight"}):
        general = oe.gnbreg(**common, **options)
        plain = oe.nbreg(**common, **options)
        assert general.coefficients[-1].term == "lnalpha:Intercept"
        assert_allclose(estimates(general), estimates(plain), rtol=1e-7, atol=1e-9)
        same_matrix(general.covariance_matrix, plain.covariance_matrix, 1e-6)
        assert_allclose(general.metrics["log_likelihood"], plain.metrics["log_likelihood"],
                        rtol=1e-11)
        assert "lnalpha" not in general.tests
        assert_allclose(general.tests["model"]["statistic"], plain.tests["model"]["statistic"],
                        rtol=1e-6)


def test_hurdle_is_a_binary_model_plus_a_truncated_count_model(data):
    used = _complete(data)
    used = used.assign(any=(used.zinb_y > 0).astype(float))
    positive = used[used.zinb_y > 0]
    for dist, link in (("poisson", "logit"), ("nbinomial", "probit"), ("poisson", "cloglog")):
        joint = oe.hurdle(data=used, y="zinb_y", x=["x1", "cat"], select_x=["z1", "x1"],
                          categorical=["cat"], dist=dist, zero_link=link, exposure="expo")
        binary = {"logit": oe.logit, "probit": oe.probit, "cloglog": oe.cloglog}[link](
            data=used, y="any", x=["z1", "x1"])
        count = (oe.tnbreg if dist == "nbinomial" else oe.tpoisson)(
            data=positive, y="zinb_y", x=["x1", "cat"], categorical=["cat"], exposure="expo")
        values = {c.term: (c.estimate, c.std_error) for c in joint.coefficients}
        for c in binary.coefficients:
            assert_allclose(values[f"select:{c.term}"], (c.estimate, c.std_error), rtol=2e-6)
        for c in count.coefficients:
            assert_allclose(values[c.term], (c.estimate, c.std_error), rtol=1e-7)
        assert_allclose(
            joint.metrics["log_likelihood"],
            binary.metrics["log_likelihood"] + count.metrics["log_likelihood"], rtol=1e-10)
        assert_allclose(joint.tests["model"]["statistic"], count.tests["model"]["statistic"],
                        rtol=1e-8)
        # The two blocks of the conventional covariance are uncorrelated.
        covariance = np.array(joint.covariance_matrix)
        assert np.abs(covariance[:4, 4:7]).max() == 0.0
    # select_x defaults to the regressors of the count equation, categoricals included.
    default = oe.hurdle(data=used, y="zinb_y", x=["x1", "cat"], categorical=["cat"])
    explicit = oe.hurdle(data=used, y="zinb_y", x=["x1", "cat"], select_x=["x1", "cat"],
                         categorical=["cat"])
    assert [c.term for c in default.coefficients][4:] == [
        "select:Intercept", "select:x1", "select:cat[b]", "select:cat[c]"]
    assert_allclose(estimates(default), estimates(explicit), rtol=1e-12)
    assert set(default.provenance["categorical_encoding"]) == {"cat"}


def test_lognormal_hurdle_is_probit_plus_least_squares_of_log_outcomes(data):
    used = _complete(data)
    joint = oe.churdle(data=used, y="amount", x=["x1", "cat"], select_x=["z1"],
                       categorical=["cat"], model="exponential", ll=0)
    above = used[used.amount > 0]
    design = np.column_stack([np.ones(len(above)), above.x1, above.cat == "b",
                              above.cat == "c"]).astype(float)
    target = np.log(above.amount.to_numpy())
    beta, ssr = np.linalg.lstsq(design, target, rcond=None)[:2]
    sigma2 = float(ssr[0]) / len(above)                    # the ML variance divides by n
    assert_allclose(estimates(joint)[:4], beta, rtol=1e-9)
    assert_allclose(joint.metrics["sigma"], math.sqrt(sigma2), rtol=1e-9)
    assert_allclose(np.array(joint.covariance_matrix)[:4, :4],
                    sigma2 * np.linalg.inv(design.T @ design), rtol=1e-7)
    assert_allclose(joint.coefficients[-1].std_error, math.sqrt(1 / (2 * len(above))),
                    rtol=1e-8)                             # Var(ln sigma) = 1 / (2 n)
    probit = oe.probit(data=used.assign(d=(used.amount > 0).astype(float)), y="d", x=["z1"])
    assert_allclose(estimates(joint)[4:6], estimates(probit), rtol=1e-6)
    lognormal = (-0.5 * len(above) * (math.log(2 * math.pi * sigma2) + 1) - target.sum())
    assert_allclose(joint.metrics["log_likelihood"],
                    probit.metrics["log_likelihood"] + lognormal, rtol=1e-10)


def test_truncation_column_scalar_and_dummy_coding_equivalences(data):
    used = _complete(data)
    sample = used[used.nb_y > 1].assign(one=1.0, b=lambda f: (f.cat == "b").astype(float),
                                        c=lambda f: (f.cat == "c").astype(float))
    for estimator in (oe.tpoisson, oe.tnbreg):
        scalar = estimator(data=sample, y="nb_y", x=["x1", "cat"], categorical=["cat"], ll=1)
        column = estimator(data=sample, y="nb_y", x=["x1", "cat"], categorical=["cat"],
                           ll="one")
        dummies = estimator(data=sample, y="nb_y", x=["x1", "b", "c"], ll=1)
        for other in (column, dummies):
            assert_allclose(estimates(other), estimates(scalar), rtol=1e-9)
            assert_allclose(other.covariance_matrix, scalar.covariance_matrix, rtol=1e-7)
            assert_allclose(other.metrics["log_likelihood"], scalar.metrics["log_likelihood"],
                            rtol=1e-12)
        assert column.extra["truncation_column"] == "one"
        assert scalar.extra["truncation_point"] == 1
    # exposure is an offset of ln(exposure)
    logged = used.assign(lnexpo=np.log(used.expo))
    first = oe.zip(data=logged, y="zip_y", x=["x1"], inflate=["z1"], exposure="expo")
    second = oe.zip(data=logged, y="zip_y", x=["x1"], inflate=["z1"], offset="lnexpo")
    assert_allclose(estimates(first), estimates(second), rtol=1e-12)


# ---- invariances of the whole pipeline -----------------------------------------------------


def _flatten(result):
    out = {"nobs": float(result.nobs)}
    for c in result.coefficients:
        out[f"b:{c.term}"], out[f"se:{c.term}"] = c.estimate, c.std_error
    out.update({f"metric:{name}": value for name, value in result.metrics.items()})
    for name, test in result.tests.items():
        out[f"test:{name}"], out[f"p:{name}"] = test["statistic"], test["p_value"]
    for name, value in result.extra.items():
        if isinstance(value, float):
            out[f"extra:{name}"] = value
        elif isinstance(value, dict):
            out.update({f"extra:{name}.{key}": item for key, item in value.items()
                        if isinstance(item, float)})
    return out


def _same(first, second, rtol, skip=()):
    assert first.keys() == second.keys()
    for name, value in first.items():
        if any(part in name for part in skip) or value is None:
            continue
        assert_allclose(second[name], value, rtol=rtol, atol=1e-200, err_msg=name)


def _models(frame):
    """One call per estimator on complete rows of ``frame`` (keyword overrides allowed)."""
    return {
        "zip": lambda f=frame, **k: oe.zip(
            data=f, y="zip_y", x=["x1", "cat"], inflate=["z1"], categorical=["cat"], **k),
        "zinb": lambda f=frame, **k: oe.zinb(
            data=f, y="zinb_y", x=["x1", "cat"], inflate=["z1"], categorical=["cat"], **k),
        "tpoisson": lambda f=frame, **k: oe.tpoisson(
            data=f[f.pois_y > f.limit], y="pois_y", x=["x1", "cat"], categorical=["cat"],
            ll="limit", **k),
        "tnbreg": lambda f=frame, **k: oe.tnbreg(
            data=f[f.nb_y > 0], y="nb_y", x=["x1", "cat"], categorical=["cat"],
            dispersion="constant", **k),
        "churdle": lambda f=frame, **k: oe.churdle(
            data=f, y="hours", x=["x1", "cat"], select_x=["z1"], categorical=["cat"],
            model="linear", ll=0.5, **k),
        "hurdle": lambda f=frame, **k: oe.hurdle(
            data=f, y="zinb_y", x=["x1", "cat"], select_x=["z1"], categorical=["cat"],
            dist="nbinomial", **k),
        "gnbreg": lambda f=frame, **k: oe.gnbreg(
            data=f, y="gnb_y", x=["x1", "cat"], lnalpha=["z1"], categorical=["cat"], **k),
    }


@pytest.mark.parametrize("name", ["zip", "zinb", "tpoisson", "tnbreg", "churdle", "hurdle",
                                  "gnbreg"])
def test_pipeline_invariances(data, name):
    used = _complete(data)
    fit = _models(used)[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. Frequency weights are duplicated rows, for every covariance estimator.
        expanded = used.loc[used.index.repeat(used.fw.astype(int))].reset_index(drop=True)
        for options in ({}, {"covariance": "opg"}, {"covariance": "robust"}, {"cluster": "firm"},
                        {"cluster": ["firm", "region"]}):
            weighted = fit(weights="fw", weight_type="fweight", **options)
            _same(_flatten(fit(expanded, **options)), _flatten(weighted), 2e-8)
        # 2. Row order is irrelevant (also for cluster sums).
        base, clustered = fit(), fit(cluster="firm")
        shuffled = used.sample(frac=1.0, random_state=7).reset_index(drop=True)
        _same(_flatten(base), _flatten(fit(shuffled)), 1e-8)
        _same(_flatten(clustered), _flatten(fit(shuffled, cluster="firm")), 1e-8)
        # 3. Affine rescaling of a regressor rescales its coefficient and nothing else.
        moved = fit(used.assign(x1=used.x1 * 250.0 - 4000.0))
        _same(_flatten(base), _flatten(moved), 1e-7,
              skip=("b:Intercept", "se:Intercept", "b:x1", "se:x1"))
        ratio = {c.term: c for c in moved.coefficients}["x1"]
        original = {c.term: c for c in base.coefficients}["x1"]
        assert_allclose([ratio.estimate * 250, ratio.std_error * 250],
                        [original.estimate, original.std_error], rtol=1e-7)
        # 4. The unit of analytic, importance and sampling weights does not move estimates.
        scaled = used.assign(rw=used.rw * 1e4)
        _same(_flatten(fit(weights="rw", weight_type="aweight")),
              _flatten(fit(scaled, weights="rw", weight_type="aweight")), 1e-8)
        sampling, rescaled = (fit(f, weights="rw", weight_type="pweight") for f in (used, scaled))
        assert_allclose(estimates(rescaled), estimates(sampling), rtol=1e-8)
        assert_allclose(rescaled.covariance_matrix, sampling.covariance_matrix, rtol=1e-7)
        importance = fit(scaled, weights="rw", weight_type="iweight")
        assert_allclose(estimates(importance), estimates(sampling), rtol=1e-8)
        assert_allclose(np.array(importance.covariance_matrix) * 1e4,
                        fit(weights="rw", weight_type="iweight").covariance_matrix, rtol=1e-7)
        # 5. Serialization keeps every number.
        restored = type(base).model_validate_json(base.model_dump_json())
        assert restored.coefficients == base.coefficients and restored.tests == base.tests


def test_small_samples_constant_only_and_no_constant_models(data):
    used = _complete(data).iloc[:70]
    y, offset = used.zip_y.to_numpy(), 0.0
    x = np.column_stack([np.ones(len(used)), used.x1])
    z = np.ones((len(used), 1))
    # zip with inflate(_cons), n = 70
    loglik = zero_inflated(y, x, z, offset, "logit", False)
    theta, value, hess, _ = maximize(loglik, [0.5, 0.0, -0.5])
    result = oe.zip(data=used, y="zip_y", x=["x1"], inflate=[])
    check_fit(result, theta, value, np.linalg.inv(-hess))
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "inflate:Intercept"]
    # constant-only count equation: the model test has no restrictions
    constant = oe.zip(data=used, y="zip_y", x=[], inflate=[])
    theta0, value0, hess0, _ = maximize(zero_inflated(y, x[:, :1], z, offset, "logit", False),
                                        [0.5, -0.5])
    check_fit(constant, theta0, value0, np.linalg.inv(-hess0))
    assert constant.tests["model"]["df"] == 0 and constant.tests["model"]["p_value"] is None
    lr_matches(result.tests["model"], value, value0, 1)
    # no constant: Wald test of every slope, no comparison model
    slope = zero_inflated(y, x[:, 1:], z, offset, "logit", False)
    theta1, value1, hess1, _ = maximize(slope, [0.1, 0.0])
    bare = oe.zip(data=used, y="zip_y", x=["x1"], inflate=[], intercept=False)
    check_fit(bare, theta1, value1, np.linalg.inv(-hess1))
    wald_matches(bare.tests["model"], bare, [0])
    assert bare.extra["null_log_likelihood"] is None
    positive = used[used.nb_y > 0]
    truncated_fit = oe.tnbreg(data=positive, y="nb_y", x=["x1"], intercept=False)
    assert truncated_fit.metrics["pseudo_r_squared"] is None
    assert truncated_fit.tests["model"]["label"].startswith("Wald")
    yp, xp = positive.nb_y.to_numpy(), positive.x1.to_numpy()
    theta2, value2, hess2, _ = maximize(
        lambda t: truncated("mean", yp, np.exp(xp * t[0]), t[1], 0), [0.1, -0.5])
    check_fit(truncated_fit, theta2, value2, np.linalg.inv(-hess2))
