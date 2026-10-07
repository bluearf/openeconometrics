"""Independent likelihood derivatives, joint sandwiches and weight invariances."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.scores import model_scores
from openecon.models import ResultBundle
from test_econ_glm_oracle import num_jac


def data():
    rng = np.random.default_rng(710)
    n = 220
    x = rng.normal(size=n)
    latent = 0.4 + 0.7 * x + rng.normal(size=n)
    mu = np.exp(0.2 + 0.25 * x)
    return pd.DataFrame(
        dict(
            x=x,
            y=latent,
            cens=np.maximum(latent, 0),
            pos=rng.gamma(3, mu / 3),
            nb=rng.negative_binomial(2, 2 / (2 + mu)),
            lo=latent - 0.2,
            hi=latent + 0.3,
            g=np.arange(n) % 20,
            f=rng.integers(1, 4, n),
            w=rng.uniform(0.3, 2, n),
        )
    )


@pytest.mark.parametrize("kind", ["glm", "nbreg", "tobit", "intreg", "truncreg"])
def test_new_providers_against_independent_row_likelihood_derivatives(kind):
    df = data()
    kwargs = dict(data=df, x=["x"], weights="f", weight_type="fweight")
    if kind == "glm":
        fit = oe.glm(y="pos", family="gamma", link="log", **kwargs)
    elif kind == "nbreg":
        fit = oe.nbreg(y="nb", **kwargs)
    elif kind == "tobit":
        fit = oe.tobit(y="cens", ll=0, **kwargs)
    elif kind == "intreg":
        fit = oe.intreg(y_low="lo", y_high="hi", **kwargs)
    else:
        fit = oe.truncreg(y="y", ll=0, **kwargs)
    rows = fit.sample_positions
    sub = df.iloc[rows]
    x = np.column_stack([np.ones(len(sub)), sub.x])
    y = sub[fit.spec.outcome].to_numpy()
    w = sub.f.to_numpy()
    theta = np.array([c.estimate for c in fit.coefficients])

    def ll(t):
        eta = x @ t[:2]
        if kind == "glm":
            # Gamma unit-dispersion quasi likelihood; constants drop from derivatives.
            return -eta - y / np.exp(eta)
        if kind == "nbreg":
            shape = np.exp(-t[-1])
            mu = np.exp(eta)
            return (
                special.gammaln(y + shape)
                - special.gammaln(shape)
                - special.gammaln(y + 1)
                + shape * np.log(shape / (shape + mu))
                + y * np.log(mu / (shape + mu))
            )
        sigma = np.exp(t[-1]) if kind == "intreg" else t[-1]
        z = (y - eta) / sigma
        normal = -np.log(sigma) - z * z / 2 - np.log(2 * np.pi) / 2
        if kind == "tobit":
            return np.where(y == 0, special.log_ndtr(-eta / sigma), normal)
        if kind == "intreg":
            return np.log(
                special.ndtr((sub.hi.to_numpy() - eta) / sigma)
                - special.ndtr((sub.lo.to_numpy() - eta) / sigma)
            )
        return normal - special.log_ndtr(eta / sigma)

    piece = model_scores(fit, df, rows)
    expected_scores = num_jac(ll, theta) * w[:, None]
    expected_hessian = num_jac(lambda t: w @ num_jac(ll, t), theta)
    assert_allclose(piece.scores, expected_scores, rtol=2e-6, atol=2e-7)
    assert_allclose(piece.hessian, expected_hessian, rtol=5e-5, atol=2e-5)
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    joint = oe.suest(restored, restored, data=df, names=["a", "b"])
    bread = np.linalg.inv(-expected_hessian)
    raw = expected_scores / w[:, None]
    v = bread @ ((raw * w[:, None]).T @ raw) @ bread * w.sum() / (w.sum() - 1)
    p = len(theta)
    assert_allclose(np.asarray(joint.covariance_matrix)[:p, p:], v, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("cluster", [None, "g"])
def test_fweights_joint_full_matrix_equals_literal_row_duplication(cluster):
    df = data()
    weighted = [
        oe.ols(data=df, y=y, x=["x"], weights="f", weight_type="fweight") for y in ["y", "pos"]
    ]
    expanded = df.loc[df.index.repeat(df.f)].reset_index(drop=True)
    plain = [oe.ols(data=expanded, y=y, x=["x"]) for y in ["y", "pos"]]
    a = oe.suest(*weighted, data=df, names=["a", "b"], cluster=cluster)
    b = oe.suest(*plain, data=expanded, names=["a", "b"], cluster=cluster)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-10, atol=1e-12)
    assert a.nobs == b.nobs == df.f.sum()


@pytest.mark.parametrize("weight_type", ["aweight", "pweight"])
@pytest.mark.parametrize("cluster", [None, "g"])
def test_weight_scale_invariance_and_different_estimation_samples(weight_type, cluster):
    df = data()
    df.loc[::7, "pos"] = np.nan

    def joint(frame):
        fits = [
            oe.ols(
                data=frame,
                y=y,
                x=["x"],
                missing="drop",
                covariance="robust",
                weights="w",
                weight_type=weight_type,
            )
            for y in ["y", "pos"]
        ]
        return oe.suest(*fits, data=frame, names=["a", "b"], cluster=cluster)

    a, b = joint(df), joint(df.assign(w=df.w * 37))
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-9, atol=1e-12)
    # Direct weighted least-squares slope sandwich, including off-diagonal overlap.
    full = np.column_stack([np.ones(len(df)), df.x])
    blocks, scores = [], []
    for y in ["y", "pos"]:
        used = df[y].notna().to_numpy()
        x, target, w = full[used], df.loc[used, y].to_numpy(), df.loc[used, "w"].to_numpy()
        beta = np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * target))
        s = np.zeros((len(df), 2))
        s[used] = x * (w * (target - x @ beta))[:, None]
        blocks.append(np.linalg.inv(x.T @ (w[:, None] * x)))
        scores.append(s)
    s = np.column_stack(scores)
    if cluster:
        summed = np.zeros((20, 4))
        np.add.at(summed, df.g, s)
        meat = summed.T @ summed * 20 / 19
    else:
        meat = s.T @ s * len(df) / (len(df) - 1)
    bread = np.zeros((4, 4))
    bread[:2, :2], bread[2:, 2:] = blocks
    indices = [0, 1, 3, 4]
    assert_allclose(
        np.asarray(a.covariance_matrix)[np.ix_(indices, indices)],
        bread @ meat @ bread,
        rtol=1e-9,
        atol=1e-12,
    )
    test = oe.test(a, {"a:x": 1, "b:x": -1})
    assert np.isfinite(test["statistic"])


def test_inconsistent_weight_values_and_tampered_likelihood_fit_rejected():
    df = data().assign(other=lambda d: d.w * 2)
    a = oe.ols(data=df, y="y", x=["x"], weights="w", weight_type="pweight", covariance="robust")
    b = oe.ols(
        data=df, y="pos", x=["x"], weights="other", weight_type="pweight", covariance="robust"
    )
    with pytest.raises(AnalysisError, match="same weight values"):
        oe.suest(a, b, data=df)
    b = oe.nbreg(data=df, y="nb", x=["x"])
    bad = b.model_copy(
        update={
            "coefficients": [
                c.model_copy(update={"estimate": c.estimate + 0.1}) for c in b.coefficients
            ]
        }
    )
    with pytest.raises(AnalysisError) as exc:
        oe.suest(b, bad, data=df)
    assert exc.value.code == "suest_mismatch"


def test_intreg_open_and_exact_bounds_use_the_saved_estimation_rows():
    df = data()
    df.loc[:29, "lo"] = np.nan
    df.loc[30:59, "hi"] = np.nan
    df.loc[60:89, "hi"] = df.loc[60:89, "lo"]
    fit = oe.intreg(data=df, y_low="lo", y_high="hi", x=["x"])
    rows = fit.sample_positions
    sub = df.iloc[rows]
    x = np.column_stack([np.ones(len(sub)), sub.x])
    lo, hi = sub.lo.to_numpy(), sub.hi.to_numpy()

    def likelihood(theta):
        mu, sigma = x @ theta[:2], np.exp(theta[-1])
        value = np.empty(len(sub))
        left, right = np.isnan(lo), np.isnan(hi)
        exact = lo == hi
        interval = ~(left | right | exact)
        value[left] = special.log_ndtr((hi[left] - mu[left]) / sigma)
        value[right] = special.log_ndtr((mu[right] - lo[right]) / sigma)
        z = (lo[exact] - mu[exact]) / sigma
        value[exact] = -np.log(sigma) - z * z / 2 - np.log(2 * np.pi) / 2
        value[interval] = np.log(
            special.ndtr((hi[interval] - mu[interval]) / sigma)
            - special.ndtr((lo[interval] - mu[interval]) / sigma)
        )
        return value

    theta = np.array([c.estimate for c in fit.coefficients])
    piece = model_scores(fit, df, rows)
    assert_allclose(piece.scores, num_jac(likelihood, theta), rtol=2e-6, atol=2e-7)
    assert oe.suest(fit, fit, data=df, names=["a", "b"]).sample_positions == rows
