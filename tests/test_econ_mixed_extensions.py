"""MARKET-152 mixed-model extensions against independent references.

* Kenward-Roger / Satterthwaite contrasts: lme4 + lmerTest + pbkrtest fixtures
  (``scripts/generate_mixed_extension_oracles.py``) and an explicit dense NumPy
  implementation of the KR (1997) formulas evaluated at the same estimates.
* Random effects plus structured residuals: nlme::lme fixtures (corAR1,
  corCompSymm, varIdent, pdDiag/pdSymm) and a dense NumPy likelihood whose
  finite-difference Hessian gives the full covariance.
* Ordinal GLMM: ordinal::clmm fixtures (adaptive GHQ, nAGQ 3 and 10) and an
  independent NumPy/SciPy quadrature likelihood (scalar mode search).
* Censored GLMM (Tobit): explicit NumPy/SciPy quadrature likelihood maximized by
  SciPy, and the exact Gaussian random-intercept likelihood without censoring.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, stats
from scipy.special import expit, logsumexp, roots_hermite

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed.small_sample import combined_ddf, mixed_contrast

FIXTURES = Path(__file__).parent / "fixtures" / "mixed_extensions"


@lru_cache(maxsize=None)
def fixture(stage: str) -> dict:
    return json.loads((FIXTURES / f"{stage}.json").read_text())


@lru_cache(maxsize=None)
def kr_fit(name: str):
    case = fixture("kr")["cases"][name]
    data = pd.DataFrame(case["data"])
    options = dict(case["spec"]["openecon"])
    return data, oe.mixed(data=data, **options), case


def contrast_frame(rows, terms):
    return pd.DataFrame([dict(zip(terms, row)) for row in rows],
                        index=[f"r{i}" for i in range(len(rows))])


# ---- Kenward-Roger / Satterthwaite: lme4, lmerTest, pbkrtest ------------------------------


@pytest.mark.parametrize("name", ["sleepstudy_reml", "unbalanced_reml", "unbalanced_ml",
                                  "nested_reml"])
def test_fit_matches_lme4(name):
    _, fitted, case = kr_fit(name)
    oracle = case["oracle"]
    p = len(oracle["fixef"])
    beta = [c.estimate for c in fitted.coefficients[:p]]
    assert_allclose(beta, oracle["fixef"], rtol=2e-6, atol=1e-6)
    assert_allclose(np.asarray(fitted.covariance_matrix)[:p, :p], oracle["vcov"], rtol=5e-6)
    assert fitted.metrics["log_likelihood"] == pytest.approx(oracle["logLik"], abs=2e-6)


@pytest.mark.parametrize("name", ["sleepstudy_reml", "unbalanced_reml", "unbalanced_ml",
                                  "nested_reml"])
def test_contrasts_match_lmertest_and_pbkrtest(name):
    data, fitted, case = kr_fit(name)
    oracle = case["oracle"]
    terms = [c.term for c in fitted.coefficients[:len(oracle["fixef"])]]
    methods = ["satterthwaite"] + (["kenward-roger"] if case["spec"]["reml"] else [])
    for spec_index, reference in enumerate(oracle["contrasts"]):
        rows = case["spec"]["contrasts"]["L"][spec_index]
        rhs = case["spec"]["contrasts"]["rhs"][spec_index]
        frame = contrast_frame(rows, terms)
        for method in methods:
            key = method.replace("-", "_")
            out = mixed_contrast(fitted, data=data, contrast=frame, null=rhs, ddf=method)
            table = out["contrasts"]
            for i, row in enumerate(reference["rows"]):
                expected = row[key]
                got = table.iloc[i]
                assert got.estimate == pytest.approx(expected["estimate"], rel=2e-6, abs=1e-6)
                assert got.std_error == pytest.approx(expected["se"], rel=1e-6)
                assert got.df == pytest.approx(expected["df"], rel=2e-6)
                assert got.t == pytest.approx(expected["t"], rel=2e-6, abs=1e-7)
                assert got.p_value == pytest.approx(expected["p"], rel=1e-4, abs=1e-12)
            joint = out["joint_test"].iloc[0]
            expected = reference["joint"][key]
            assert joint.numerator_df == expected["ndf"]
            assert joint.denominator_df == pytest.approx(expected["ddf"], rel=2e-6)
            assert joint.statistic == pytest.approx(expected["F"], rel=2e-6)
            assert joint.p_value == pytest.approx(expected["p"], rel=1e-4, abs=1e-14)
            if method == "kenward-roger":
                assert joint.scale == pytest.approx(expected["scale"], rel=1e-5)
                assert joint.unscaled_statistic == pytest.approx(expected["F_unscaled"], rel=1e-5)
                assert_allclose(out["fixed_covariance"].to_numpy(), oracle["vcov_adjusted"],
                                rtol=3e-6)


# ---- independent dense NumPy KR / Satterthwaite (structures lme4 cannot express) ----------


def _structured_data(seed=152):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 9, size=16)
    g = np.repeat(np.arange(len(sizes)), sizes)
    n = len(g)
    x1, x2 = rng.normal(size=(2, n))
    u = rng.multivariate_normal([0, 0, 0], [[.5, .2, .1], [.2, .5, .15], [.1, .15, .9]],
                                size=len(sizes))
    y = 1 + .5 * x1 - .3 * x2 + u[g, 0] * x1 + u[g, 1] * x2 + u[g, 2] + rng.normal(0, .8, n)
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y})


def _dense_pieces(frame, fit, random):
    """Full N x N derivative matrices G_r in the reported linear coordinates."""
    n = len(frame)
    names = [c.term for c in fit.coefficients]
    p = names.index(next(t for t in names if t.startswith("/")))
    s = np.array([c.estimate for c in fit.coefficients[p:]])
    z = np.column_stack([*[frame[c].to_numpy() for c in random], np.ones(n)])
    low = frame["g"].to_numpy()
    same = (low[:, None] == low[None, :]).astype(float)
    kind = fit.extra["covstructure"]
    q = z.shape[1]
    outer = {(a, b): same * (np.outer(z[:, a], z[:, b])
                             + (np.outer(z[:, b], z[:, a]) if a != b else 0))
             for a in range(q) for b in range(q)}
    blocks = []
    if kind == "identity":
        blocks.append(sum(outer[a, a] for a in range(q)))
    elif kind == "exchangeable":
        blocks.append(sum(outer[a, a] for a in range(q)))
        blocks.append(sum(outer[a, b] for a in range(q) for b in range(a)))
    elif kind == "independent":
        blocks.extend(outer[a, a] for a in range(q))
    else:
        blocks.extend(outer[a, a] for a in range(q))
        blocks.extend(outer[b, a] for a in range(q) for b in range(a + 1, q))
    blocks.append(np.eye(n))
    return s, np.array(blocks), p


def _numpy_kr(x, y, s, blocks):
    v = np.tensordot(s, blocks, 1)
    vi = np.linalg.inv(v)
    phi = np.linalg.inv(x.T @ vi @ x)
    beta = phi @ x.T @ vi @ y
    proj = vi - vi @ x @ phi @ x.T @ vi
    k = len(s)
    pr = [-x.T @ vi @ g @ vi @ x for g in blocks]
    info = np.array([[np.trace(proj @ blocks[i] @ proj @ blocks[j]) / 2 for j in range(k)]
                     for i in range(k)])
    w = np.linalg.inv(info)
    middle = sum(w[i, j] * (x.T @ vi @ blocks[i] @ vi @ blocks[j] @ vi @ x - pr[i] @ phi @ pr[j])
                 for i in range(k) for j in range(k))
    return beta, phi, phi + 2 * phi @ middle @ phi, pr, w


def _numpy_kr_df(lmat, phi, pr, w):
    """Kenward and Roger (1997, section 4) moments written from the paper's definitions."""
    q = lmat.shape[0]
    theta = lmat.T @ np.linalg.inv(lmat @ phi @ lmat.T) @ lmat
    u = [theta @ phi @ p @ phi for p in pr]
    k = len(u)
    a1 = sum(w[i, j] * np.trace(u[i]) * np.trace(u[j]) for i in range(k) for j in range(k))
    a2 = sum(w[i, j] * np.trace(u[i] @ u[j]) for i in range(k) for j in range(k))
    b = (a1 + 6 * a2) / (2 * q)
    g = ((q + 1) * a1 - (q + 4) * a2) / ((q + 2) * a2)
    c1, c2, c3 = np.array([g, q - g, q + 2 - g]) / (3 * q + 2 * (1 - g))
    e_star = 1 / (1 - a2 / q)
    v_star = 2 / q * (1 + c1 * b) / ((1 - c2 * b) ** 2 * (1 - c3 * b))
    rho = v_star / (2 * e_star ** 2)
    m = 4 + (q + 2) / (q * rho - 1)
    return m, m / (e_star * (m - 2))


def _numpy_restricted(x, y, s, blocks):
    v = np.tensordot(s, blocks, 1)
    vi = np.linalg.inv(v)
    phi = np.linalg.inv(x.T @ vi @ x)
    r = y - x @ (phi @ x.T @ vi @ y)
    value = -.5 * (np.linalg.slogdet(v)[1] + r @ vi @ r + np.linalg.slogdet(x.T @ vi @ x)[1])
    return value, phi


@pytest.mark.parametrize("structure,random", [("exchangeable", ["x1", "x2"]),
                                              ("identity", ["x1"]),
                                              ("unstructured", ["x1", "x2"])])
def test_kr_and_satterthwaite_match_dense_numpy_formulas(structure, random):
    frame = _structured_data()
    fit = oe.mixed(data=frame, y="y", x=["x1", "x2"], group="g", random=random,
                   covstructure=structure, method="reml")
    s, blocks, p = _dense_pieces(frame, fit, random)
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    y = frame.y.to_numpy()
    beta, phi, adjusted, pr, w = _numpy_kr(x, y, s, blocks)
    assert_allclose([c.estimate for c in fit.coefficients[:p]], beta, rtol=1e-9, atol=1e-10)
    lmat = np.array([[0.0, 1, 0], [0, 1, 1]])
    out = mixed_contrast(fit, data=frame,
                         contrast=pd.DataFrame(lmat, columns=["Intercept", "x1", "x2"]),
                         null=[.2, 0.0], ddf="kenward-roger")
    assert_allclose(out["fixed_covariance"].to_numpy(), adjusted, rtol=1e-9, atol=1e-13)
    assert_allclose(out["variance_parameter_covariance"].to_numpy(), w, rtol=1e-8)
    m, scale = _numpy_kr_df(lmat, phi, pr, w)
    joint = out["joint_test"].iloc[0]
    assert joint.denominator_df == pytest.approx(m, rel=1e-9)
    assert joint.scale == pytest.approx(scale, rel=1e-9)
    delta = lmat @ beta - np.array([.2, 0])
    wald = delta @ np.linalg.solve(lmat @ adjusted @ lmat.T, delta) / 2
    assert joint.statistic == pytest.approx(scale * wald, rel=1e-8)
    assert joint.p_value == pytest.approx(stats.f.sf(scale * wald, 2, m), rel=1e-7)
    for i in range(2):
        row_df, _ = _numpy_kr_df(lmat[i:i + 1], phi, pr, w)
        assert out["contrasts"].df.iloc[i] == pytest.approx(row_df, rel=1e-9)
    # Satterthwaite: finite-difference observed information and gradient in linear s.
    step = 1e-4 * np.maximum(np.abs(s), 1e-2)
    k = len(s)
    eye = np.eye(k) * step

    def value(point):
        return _numpy_restricted(x, y, point, blocks)[0]

    hessian = np.array([[(value(s + eye[i] + eye[j]) - value(s + eye[i] - eye[j])
                          - value(s - eye[i] + eye[j]) + value(s - eye[i] - eye[j]))
                         / (4 * step[i] * step[j]) for j in range(k)] for i in range(k)])
    covariance = np.linalg.inv(-hessian)
    sat = mixed_contrast(fit, data=frame, contrast={"x1": 1, "x2": 1}, ddf="satterthwaite")
    vec = np.array([0, 1, 1])

    def variance(point):
        return vec @ _numpy_restricted(x, y, point, blocks)[1] @ vec

    gradient = np.array([(variance(s + eye[i]) - variance(s - eye[i])) / (2 * step[i])
                         for i in range(k)])
    df = 2 * (vec @ phi @ vec) ** 2 / (gradient @ covariance @ gradient)
    assert sat["contrasts"].df.iloc[0] == pytest.approx(df, rel=2e-5)
    assert_allclose(sat["variance_parameter_covariance"].to_numpy(), covariance, rtol=2e-4,
                    atol=1e-8)


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_satterthwaite_scalar_agrees_with_existing_log_scale_procedure(method):
    data = _structured_data()
    fitted = oe.mixed(data=data, y="y", x=["x1", "x2"], group="g", random=["x1"],
                      covstructure="unstructured", method=method)
    for contrast in ({"x1": 1}, {"Intercept": .3, "x2": 1}):
        old = oe.mixed_satterthwaite(fitted, data=data, contrast=contrast)["contrast"].iloc[0]
        new = oe.mixed_contrast(fitted, data=data, contrast=contrast,
                                ddf="satterthwaite")["contrasts"].iloc[0]
        assert new.df == pytest.approx(old.df, rel=2e-5)
        assert new.std_error == pytest.approx(old.std_error, rel=1e-10)


def test_combined_denominator_df_rule():
    assert combined_ddf([7.5]) == 7.5
    assert combined_ddf([5.0, 5.0 + 1e-10]) == pytest.approx(5.0)
    assert combined_ddf([1.5, 30.0]) == 2.0
    nu = [6.0, 11.0, 40.0]
    e = sum(v / (v - 2) for v in nu)
    assert combined_ddf(nu) == pytest.approx(2 * e / (e - 3))


def test_contrast_failure_contract_and_round_trip():
    data, fitted, _ = kr_fit("sleepstudy_reml")
    data_ml, fitted_ml, _ = kr_fit("unbalanced_ml")
    with pytest.raises(AnalysisError) as error:
        mixed_contrast(fitted_ml, data=data_ml, contrast={"x": 1}, ddf="kenward-roger")
    assert error.value.code == "unsupported_spec"
    bad = [({"Days": 1}, {"ddf": "containment"}, "invalid_option"),
           ({"Nope": 1}, {}, "invalid_contrast"),
           ({"Days": 0}, {}, "invalid_contrast"),
           ([{"Days": 1}, {"Days": 2}], {}, "invalid_contrast"),
           ({"Days": True}, {}, "invalid_contrast"),
           ({"Days": 1}, {"null": [1, 2]}, "invalid_null"),
           ({"Days": 1}, {"alpha": 1.5}, "invalid_option"),
           ({"Days": 1}, {"max_work": 10}, "work_budget")]
    for contrast, options, code in bad:
        with pytest.raises(AnalysisError) as error:
            mixed_contrast(fitted, data=data, contrast=contrast, **options)
        assert error.value.code == code, (contrast, options)
    with pytest.raises(AnalysisError) as error:
        mixed_contrast(fitted, data=data.iloc[::-1], contrast={"Days": 1})
    assert error.value.code == "data_mismatch"
    tampered = fitted.model_copy(deep=True)
    tampered.coefficients[-1].estimate *= 1.05
    with pytest.raises(AnalysisError) as error:
        mixed_contrast(tampered, data=data, contrast={"Days": 1})
    assert error.value.code == "invalid_result_state"
    robust = oe.mixed(data=data, y="Reaction", x=["Days"], group="Subject", covariance="robust")
    with pytest.raises(AnalysisError) as error:
        mixed_contrast(robust, data=data, contrast={"Days": 1}, ddf="satterthwaite")
    assert error.value.code == "unsupported_spec"
    holes = data.copy()
    holes.loc[[3, 40], "Reaction"] = np.nan
    dropped = oe.mixed(data=holes, y="Reaction", x=["Days"], group="Subject", random=["Days"],
                       covstructure="unstructured", method="reml", missing="drop")
    complete = holes.dropna().reset_index(drop=True)
    refit = oe.mixed(data=complete, y="Reaction", x=["Days"], group="Subject", random=["Days"],
                     covstructure="unstructured", method="reml")
    first = oe.mixed_contrast(dropped, data=holes, contrast={"Days": 1})
    second = oe.mixed_contrast(refit, data=complete, contrast={"Days": 1})
    assert first.attrs["source_sample_positions"] == [i for i in range(len(data))
                                                      if i not in (3, 40)]
    assert_allclose(first["contrasts"][["estimate", "std_error", "df"]].to_numpy(),
                    second["contrasts"][["estimate", "std_error", "df"]].to_numpy(), rtol=1e-6)
    restored = type(fitted).model_validate_json(fitted.model_dump_json())
    out = oe.mixed_contrast(restored, data=data, contrast=[{"Days": 1}, {"Intercept": 1}])
    assert out.attrs["refitted"] is False and out.attrs["ddf_method"] == "kenward-roger"
    again = oe.restore_summary(oe.summary_state(out))
    for name in out:
        pd.testing.assert_frame_equal(again[name], out[name], check_dtype=False, check_exact=True)
    assert "Kenward-Roger" in str(out) and "tabular" in out.to_latex()


# ---- random effects plus structured residual covariance: nlme::lme ------------------------


@lru_cache(maxsize=None)
def lme_fit(name: str):
    case = fixture("lme")["cases"][name]
    data = pd.DataFrame(case["data"])
    return data, oe.mixedresid(data=data, **case["spec"]["openecon"]), case


def _dense_lme_loglik(frame, options, beta, psi, reml):
    """Independent dense Gaussian (restricted) log likelihood in natural coordinates."""
    x = np.column_stack([np.ones(len(frame)), *[frame[c] for c in options["x"]]])
    y = frame[options["y"]].to_numpy()
    random = options.get("random", [])
    z = np.column_stack([*[frame[c] for c in random], np.ones(len(frame))])
    q = z.shape[1]
    g = np.zeros((q, q))
    g[np.diag_indices(q)] = psi[:q]
    position = q
    if options.get("covstructure") == "unstructured":
        for i in range(q):
            for j in range(i):
                g[i, j] = g[j, i] = psi[position]
                position += 1
    stratum = options.get("stratum")
    if stratum:
        labels = sorted(frame[stratum].unique())
        variance = np.array([psi[position + labels.index(v)] for v in frame[stratum]])
        position += len(labels)
    else:
        variance = np.full(len(frame), psi[position])
        position += 1
    rho = psi[position] if options["residual"] != "independent" else 0.0
    group = frame[options["group"]].to_numpy()
    cell = frame[options["within"]].to_numpy() if options.get("within") else group
    information, score, quad, logdet = np.zeros((x.shape[1],) * 2), np.zeros(x.shape[1]), 0.0, 0.0
    for label in pd.unique(group):
        rows = np.flatnonzero(group == label)
        sd = np.sqrt(variance[rows])
        same = cell[rows][:, None] == cell[rows][None, :]
        if options["residual"] == "ar1":
            t = frame[options["occasion"]].to_numpy()[rows]
            corr = rho ** np.abs(t[:, None] - t[None, :])
        elif options["residual"] == "exchangeable":
            corr = np.where(same, rho, 0.0) + (1 - rho) * np.eye(len(rows))
        else:
            corr = np.eye(len(rows))
        v = z[rows] @ g @ z[rows].T + sd[:, None] * corr * sd[None, :]
        vi = np.linalg.inv(v)
        information += x[rows].T @ vi @ x[rows]
        score += x[rows].T @ vi @ y[rows]
        logdet += np.linalg.slogdet(v)[1]
        if beta is not None:
            r = y[rows] - x[rows] @ beta
            quad += r @ vi @ r
    n, p = x.shape
    if beta is None:
        b = np.linalg.solve(information, score)
        return _dense_lme_loglik(frame, options, b, psi, reml) - (
            0.5 * np.linalg.slogdet(information)[1] - 0.5 * p * np.log(2 * np.pi) if reml else 0)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + quad)


def _fd_hessian(function, point, relative=2e-4):
    step = relative * np.maximum(np.abs(point), 1e-2)
    k = len(point)
    eye = np.eye(k) * step
    return np.array([[(function(point + eye[i] + eye[j]) - function(point + eye[i] - eye[j])
                       - function(point - eye[i] + eye[j]) + function(point - eye[i] - eye[j]))
                      / (4 * step[i] * step[j]) for j in range(k)] for i in range(k)])


LME_CASES = ["ar1_ri_varident_reml", "ar1_ri_varident_ml", "ar1_slope_unstructured_ml",
             "diag_slope_varident_reml", "cs_nested_reml"]


@pytest.mark.parametrize("name", LME_CASES)
def test_structured_residual_fit_matches_nlme(name):
    _, fitted, case = lme_fit(name)
    oracle, options = case["oracle"], case["spec"]["openecon"]
    p = len(oracle["fixef"])
    estimates = {c.term: c.estimate for c in fitted.coefficients}
    assert_allclose([c.estimate for c in fitted.coefficients[:p]], oracle["fixef"], rtol=1e-6,
                    atol=1e-7)
    assert fitted.metrics["log_likelihood"] == pytest.approx(oracle["logLik"], abs=1e-8)
    # nlme reports the conditional GLS fixed covariance for ML and REML alike.
    assert_allclose(fitted.extra["conditional_gls_covariance"], oracle["vcov"], rtol=5e-6)
    if options["method"] == "reml":
        assert_allclose(np.asarray(fitted.covariance_matrix)[:p, :p], oracle["vcov"], rtol=5e-6)
    names = ["x", "_cons"] if options.get("random") else ["_cons"]
    nlme_names = ["x", "(Intercept)"] if options.get("random") else ["(Intercept)"]
    order = [oracle["random_names"].index(n) for n in nlme_names]
    reference = np.atleast_2d(oracle["random_covariance"])[np.ix_(order, order)]
    assert_allclose(fitted.extra["G"], reference, rtol=5e-6, atol=1e-7)
    group = options["group"]
    for i, effect in enumerate(names):
        assert estimates[f"/var({effect}[{group}])"] == pytest.approx(reference[i, i], rel=2e-6)
    sigma2 = oracle["sigma"] ** 2
    if options.get("stratum"):
        for label, ratio in oracle["varStruct"].items():
            key = f"/var(Residual[{options['stratum']}={label}])"
            assert estimates[key] == pytest.approx(sigma2 * ratio ** 2, rel=2e-6)
    else:
        assert estimates["/var(Residual)"] == pytest.approx(sigma2, rel=2e-6)
    if options["residual"] != "independent":
        assert estimates["/rho(Residual)"] == pytest.approx(np.atleast_1d(oracle["corStruct"])[0], abs=2e-6)


@pytest.mark.parametrize("name", ["ar1_ri_varident_reml", "ar1_ri_varident_ml"])
def test_variance_parameter_covariance_matches_nlme_apvar(name):
    """nlme apVar (fdHess) in its coordinates: log sd, 2 atanh(rho), log delta, log sigma."""
    _, fitted, case = lme_fit(name)
    oracle = case["oracle"]
    # ours: (ln sd_u, ln sigma_a, ln delta_b, atanh rho); nlme's reference stratum is "b".
    assert list(oracle["varStruct"]) == ["a", "b"] and oracle["varStruct"]["b"] == 1
    linear = np.array([[1, 0, 0, 0], [0, 0, 0, 2], [0, 0, -1, 0], [0, 1, 1, 0]], dtype=float)
    ours = linear @ np.asarray(fitted.extra["likelihood_covariance"]) @ linear.T
    # apVar is nlme's fdHess finite-difference approximation (measured deviation at most
    # 0.52% on the correlation scale); the exact check is the dense NumPy Hessian below.
    reference = np.asarray(oracle["apVar"])
    scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
    assert np.max(np.abs(ours - reference) / scale) < 1e-2


@pytest.mark.parametrize("name", LME_CASES)
def test_structured_residual_full_covariance_against_dense_numpy_hessian(name):
    data, fitted, case = lme_fit(name)
    options = case["spec"]["openecon"]
    p = len(case["oracle"]["fixef"])
    estimates = np.array([c.estimate for c in fitted.coefficients])
    beta, psi = estimates[:p], estimates[p:]
    reml = options["method"] == "reml"
    covariance = np.asarray(fitted.covariance_matrix)
    assert fitted.metrics["log_likelihood"] == pytest.approx(
        _dense_lme_loglik(data, options, None if reml else beta, psi, reml), abs=1e-9)
    if reml:
        hessian = _fd_hessian(lambda point: _dense_lme_loglik(data, options, None, point, True),
                              psi)
        assert_allclose(covariance[p:, p:], np.linalg.inv(-hessian), rtol=2e-4, atol=1e-9)
        assert_allclose(covariance[:p, p:], 0.0, atol=0)
    else:
        joint = np.concatenate([beta, psi])
        hessian = _fd_hessian(lambda point: _dense_lme_loglik(data, options, point[:p], point[p:],
                                                              False), joint)
        assert_allclose(covariance, np.linalg.inv(-hessian), rtol=2e-4, atol=1e-9)
        # The variance block equals the inverse profile information (Schur complement).
        profile = _fd_hessian(lambda point: _dense_lme_loglik(data, options, None, point, False),
                              psi)
        assert_allclose(covariance[p:, p:], np.linalg.inv(-profile), rtol=2e-4, atol=1e-9)


def test_structured_residual_failure_contract_missing_rows_and_round_trip():
    data, fitted, case = lme_fit("ar1_ri_varident_reml")
    base = dict(y="y", x=["x", "t"], group="g")

    def code(frame=data, **options):
        with pytest.raises(AnalysisError) as error:
            oe.mixedresid(data=frame, **{**base, **options})
        return error.value.code

    assert code(residual="ar1") == "invalid_spec"                         # no occasion
    assert code(residual="exchangeable", occasion="t") == "invalid_spec"  # occasion unused
    assert code(residual="independent", within="arm") == "invalid_spec"
    assert code(residual="exchangeable") == "unidentified_covariance"     # RI + CS same blocks
    assert code(data.assign(g2=1), residual="independent", stratum="g2") == "invalid_spec"
    duplicated = data.copy()
    duplicated.loc[1, "t"] = duplicated.loc[0, "t"]
    assert code(duplicated, residual="ar1", occasion="t") == "duplicate_occasion"
    fractional = data.assign(t=data.t + 0.5 * (data.index == 3))
    assert code(fractional, residual="ar1", occasion="t") == "invalid_occasion"
    even = data.assign(t2=2 * data.t)
    assert code(even, residual="ar1", occasion="t2") == "unsupported_occasion_grid"
    assert code(residual="ar1", occasion="t", max_work=1) == "work_budget"
    flat = data.assign(y=1.0 + 0.5 * data.x + 0.2 * data.t
                       + np.random.default_rng(3).normal(0, 0.7, len(data)))
    assert code(flat, residual="ar1", occasion="t") == "boundary_solution"  # no group effect
    # Missing outcome rows are dropped explicitly and match the fit on the complete rows.
    holes = data.copy()
    holes.loc[[2, 17], "y"] = np.nan
    options = dict(case["spec"]["openecon"])
    dropped = oe.mixedresid(data=holes, missing="drop", **options)
    complete = oe.mixedresid(data=holes.dropna().reset_index(drop=True), **options)
    assert dropped.nobs == len(data) - 2
    assert_allclose([c.estimate for c in dropped.coefficients],
                    [c.estimate for c in complete.coefficients], rtol=1e-8, atol=1e-10)
    with pytest.raises(AnalysisError):
        oe.mixedresid(data=holes, **options)
    restored = type(fitted).model_validate_json(fitted.model_dump_json())
    assert restored == fitted
    text = fitted.summary()
    assert "/rho(Residual)" in text and "/var(Residual[arm=b])" in text
    assert "tabular" in fitted.to_latex()
    assert fitted.provenance["stata_parity_validated"] is False


# ---- independent NumPy/SciPy random-intercept quadrature ----------------------------------


def numpy_quadrature(row_logf, codes, sigma, points, adaptive):
    """ln int prod_i f_i(sigma v) phi(v) dv per group; Brent mode, Richardson curvature."""
    nodes, weights = roots_hermite(points)
    total = 0.0
    for label in np.unique(codes):
        rows = np.flatnonzero(codes == label)

        def h(v, rows=rows):
            v = np.atleast_1d(v)
            return row_logf(rows, sigma * v[None, :]).sum(0) + stats.norm.logpdf(v)

        if adaptive:
            mode = optimize.minimize_scalar(lambda v: -h(v)[0], bracket=(-1.0, 1.0),
                                            options={"xtol": 1e-13}).x
            second = [-(h(mode + e)[0] - 2 * h(mode)[0] + h(mode - e)[0]) / e ** 2
                      for e in (2e-3, 1e-3)]
            scale = np.sqrt(2 / ((4 * second[1] - second[0]) / 3))
        else:
            mode, scale = 0.0, np.sqrt(2)
        abscissae = mode + scale * nodes
        total += np.log(scale) + logsumexp(np.log(weights) + nodes ** 2 + h(abscissae))
    return total


def ordered_row_logf(frame, link, terms, beta, cuts):
    cdf = expit if link == "logit" else stats.norm.cdf
    eta = frame[terms].to_numpy() @ beta
    levels = np.sort(frame.y.unique())
    code = np.searchsorted(levels, frame.y.to_numpy())
    bounds = np.concatenate([[-np.inf], cuts, [np.inf]])

    def row_logf(rows, shift):
        e = eta[rows][:, None] + shift
        upper = bounds[code[rows] + 1][:, None] - e
        lower = bounds[code[rows]][:, None] - e
        return np.log(cdf(upper) - cdf(lower))

    return row_logf


def ordered_numpy_loglik(frame, link, points, adaptive, params):
    beta, cuts, variance = params[:2], params[2:-1], params[-1]
    row_logf = ordered_row_logf(frame, link, ["x", "z"], beta, cuts)
    return numpy_quadrature(row_logf, frame.g.to_numpy(), np.sqrt(variance), points, adaptive)


@lru_cache(maxsize=None)
def clmm_fit(name):
    case = fixture("clmm")["cases"][name]
    data = pd.DataFrame(case["data"])
    function = oe.meologit if case["spec"]["link"] == "logit" else oe.meoprobit
    return data, function(data=data, **case["spec"]["openecon"]), case


@pytest.mark.parametrize("name", ["ologit_agq10", "oprobit_agq10", "ologit_agq3"])
def test_ordinal_glmm_matches_clmm(name):
    _, fitted, case = clmm_fit(name)
    oracle = case["oracle"]
    clmm = dict(zip(oracle["coefficient_names"], oracle["coefficients"]))
    ours = [c.estimate for c in fitted.coefficients]
    expected = [clmm["x"], clmm["z"], clmm["1|2"], clmm["2|3"], clmm["3|4"],
                oracle["random_variance"]]
    loose = name.endswith("agq3")         # clmm stops with max|gradient| 3e-5 there
    assert_allclose(ours, expected, rtol=0, atol=1e-5 if loose else 5e-8)
    assert fitted.metrics["log_likelihood"] == pytest.approx(oracle["logLik"], abs=1e-9)
    assert fitted.metrics["log_likelihood"] >= oracle["logLik"] - 1e-10
    # clmm order (cuts, b, ln sd) -> ours (b, cuts, var): permutation and d var/d ln sd.
    jacobian = np.zeros((6, 6))
    for ours_index, clmm_index in enumerate([3, 4, 0, 1, 2]):
        jacobian[ours_index, clmm_index] = 1.0
    jacobian[5, 5] = 2 * oracle["random_variance"]
    reference = jacobian @ np.asarray(oracle["vcov"]) @ jacobian.T
    # clmm's covariance is a numerical (Richardson) Hessian: compare on the correlation scale.
    scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
    deviation = np.abs(np.asarray(fitted.covariance_matrix) - reference) / scale
    assert deviation.max() < (5e-4 if loose else 2e-5)
    assert_allclose(np.diag(fitted.covariance_matrix), np.diag(reference),
                    rtol=5e-4 if loose else 5e-6)


def test_ordinal_glmm_against_independent_numpy_quadrature():
    data, fitted, _ = clmm_fit("ologit_agq10")
    params = np.array([c.estimate for c in fitted.coefficients])
    value = ordered_numpy_loglik(data, "logit", 10, True, params)
    assert fitted.metrics["log_likelihood"] == pytest.approx(value, abs=1e-8)
    step = 1e-5
    gradient = [(ordered_numpy_loglik(data, "logit", 10, True, params + e)
                 - ordered_numpy_loglik(data, "logit", 10, True, params - e)) / (2 * step)
                for e in np.eye(6) * step]
    assert np.max(np.abs(gradient)) < 2e-5
    # Non-adaptive (Stata ghermite) rule: value, stationarity and full covariance.
    plain = oe.meologit(data=data, y="y", x=["x", "z"], group="g", intpoints=20,
                        intmethod="ghermite")
    params = np.array([c.estimate for c in plain.coefficients])

    def numpy_value(point):
        return ordered_numpy_loglik(data, "logit", 20, False, point)

    assert plain.metrics["log_likelihood"] == pytest.approx(numpy_value(params), abs=1e-9)
    hessian = _fd_hessian(numpy_value, params, relative=1e-4)
    assert_allclose(plain.covariance_matrix, np.linalg.inv(-hessian), rtol=1e-4, atol=1e-9)
    check = fitted.extra["quadrature_check"]
    assert abs(check["difference"]) < 1e-6 and check["doubled_intpoints"] == 20


def test_ordinal_glmm_failure_contract_and_round_trip():
    data, fitted, _ = clmm_fit("oprobit_agq10")
    with pytest.raises(AnalysisError) as error:
        oe.meoprobit(data=data.assign(y=pd.Categorical(data.y)), y="y", x=["x"], group="g")
    assert error.value.code == "invalid_ordered_outcome"
    with pytest.raises(AnalysisError) as error:
        oe.meologit(data=data, y="y", x=["x"], group="g", intpoints=0)
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.meologit(data=data.assign(g=data.g % 2), y="y", x=["x"], group="g")
    assert error.value.code == "insufficient_groups"
    rng = np.random.default_rng(4)
    pooled = data.assign(y=np.searchsorted([-1.0, 0.3, 1.5], 0.8 * data.x
                                           + rng.logistic(size=len(data))) + 1)
    with pytest.raises(AnalysisError) as error:
        oe.meologit(data=pooled, y="y", x=["x"], group="g")
    assert error.value.code == "boundary_solution"
    with pytest.raises(AnalysisError) as error:
        oe.meologit(data=data, y="y", x=["x"], group="g", max_work=10)
    assert error.value.code == "work_budget"
    restored = type(fitted).model_validate_json(fitted.model_dump_json())
    assert restored == fitted
    assert "/cut3" in fitted.summary() and "tabular" in fitted.to_latex()
    assert fitted.extra["intmethod"] == "mcaghermite"


# ---- censored (Tobit) GLMM ---------------------------------------------------------------


@lru_cache(maxsize=None)
def tobit_data():
    rng = np.random.default_rng(152_006)
    g = np.repeat(np.arange(40), rng.integers(4, 10, size=40))
    x = rng.normal(size=len(g))
    latent = 0.5 + 1.0 * x + rng.normal(0, 0.8, 40)[g] + rng.normal(0, 1.0, len(g))
    return pd.DataFrame({"g": g, "x": x, "latent": latent, "y": np.clip(latent, 0.0, 2.5)})


def tobit_numpy_loglik(frame, params, points, adaptive, ll=0.0, ul=2.5):
    beta, var_u, var_e = params[:2], params[2], params[3]
    eta = beta[0] + beta[1] * frame.x.to_numpy()
    y = frame.y.to_numpy()
    s = np.sqrt(var_e)

    def row_logf(rows, shift):
        e = eta[rows][:, None] + shift
        yy = y[rows][:, None]
        return np.where(yy <= ll, stats.norm.logcdf((ll - e) / s),
                        np.where(yy >= ul, stats.norm.logsf((ul - e) / s),
                                 stats.norm.logpdf(yy, e, s)))

    return numpy_quadrature(row_logf, frame.g.to_numpy(), np.sqrt(var_u), points, adaptive)


def test_tobit_glmm_against_scipy_maximized_quadrature_likelihood():
    data = tobit_data()
    fitted = oe.metobit(data=data, y="y", x=["x"], group="g", ll=0.0, ul=2.5)
    params = np.array([c.estimate for c in fitted.coefficients])
    assert fitted.metrics["n_left_censored"] == int((data.y <= 0).sum())
    assert fitted.metrics["n_right_censored"] == int((data.y >= 2.5).sum())
    assert fitted.metrics["log_likelihood"] == pytest.approx(
        tobit_numpy_loglik(data, params, 12, True), abs=1e-8)
    # SciPy maximizes an explicit 60-point non-adaptive likelihood in log variances.

    def negative(point):
        return -tobit_numpy_loglik(data, np.r_[point[:2], np.exp(point[2:])], 60, False)

    start = np.r_[params[:2] + 0.05, np.log(params[2:]) - 0.1]
    solution = optimize.minimize(negative, start, method="BFGS", options={"gtol": 1e-9})
    reference = np.r_[solution.x[:2], np.exp(solution.x[2:])]
    assert_allclose(params, reference, rtol=2e-5, atol=2e-6)
    hessian = _fd_hessian(lambda point: tobit_numpy_loglik(data, point, 60, False), reference,
                          relative=1e-4)
    assert_allclose(fitted.covariance_matrix, np.linalg.inv(-hessian), rtol=1e-3, atol=1e-8)
    plain = oe.metobit(data=data, y="y", x=["x"], group="g", ll=0.0, ul=2.5, intpoints=40,
                       intmethod="ghermite")
    plain_params = np.array([c.estimate for c in plain.coefficients])
    assert plain.metrics["log_likelihood"] == pytest.approx(
        tobit_numpy_loglik(data, plain_params, 40, False), abs=1e-9)
    hessian = _fd_hessian(lambda point: tobit_numpy_loglik(data, point, 40, False), plain_params,
                          relative=1e-4)
    assert_allclose(plain.covariance_matrix, np.linalg.inv(-hessian), rtol=1e-4, atol=1e-9)


def test_tobit_glmm_without_censoring_is_the_exact_gaussian_random_intercept_model():
    data = tobit_data().assign(y=lambda d: d.latent)
    fitted = oe.metobit(data=data, y="y", x=["x"], group="g", ll=-50.0, intpoints=1)
    gaussian = oe.mixed(data=data, y="y", x=["x"], group="g")
    assert fitted.metrics["n_left_censored"] == 0
    assert_allclose([c.estimate for c in fitted.coefficients],
                    [c.estimate for c in gaussian.coefficients], rtol=2e-7, atol=1e-8)
    assert fitted.metrics["log_likelihood"] == pytest.approx(
        gaussian.metrics["log_likelihood"], abs=1e-8)
    x = np.column_stack([np.ones(len(data)), data.x])
    y, groups = data.y.to_numpy(), data.g.to_numpy()

    def dense(point):
        beta, var_u, var_e = point[:2], point[2], point[3]
        value = 0.0
        for label in np.unique(groups):
            rows = groups == label
            v = var_u + var_e * np.eye(rows.sum())
            r = y[rows] - x[rows] @ beta
            value += -0.5 * (rows.sum() * np.log(2 * np.pi) + np.linalg.slogdet(v)[1]
                             + r @ np.linalg.solve(v, r))
        return value

    params = np.array([c.estimate for c in fitted.coefficients])
    assert fitted.metrics["log_likelihood"] == pytest.approx(dense(params), abs=1e-9)
    hessian = _fd_hessian(dense, params, relative=1e-4)
    assert_allclose(fitted.covariance_matrix, np.linalg.inv(-hessian), rtol=1e-4, atol=1e-10)


def test_tobit_glmm_failure_contract_and_round_trip():
    data = tobit_data()
    cases = [({}, "invalid_spec"), ({"ll": 1.0, "ul": 0.5}, "invalid_option"),
             ({"ll": 2.4, "ul": 2.45}, "insufficient_observations"),
             ({"ll": 0.0, "max_work": 5}, "work_budget"),
             ({"ll": 0.0, "intmethod": "laplace"}, "invalid_spec")]
    for options, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.metobit(data=data, y="y", x=["x"], group="g", **options)
        assert error.value.code == code, options
    rng = np.random.default_rng(9)
    pooled = data.assign(y=np.clip(0.5 + data.x + rng.normal(size=len(data)), 0.0, None))
    with pytest.raises(AnalysisError) as error:
        oe.metobit(data=pooled, y="y", x=["x"], group="g", ll=0.0)
    assert error.value.code == "boundary_solution"
    fitted = oe.metobit(data=data, y="y", x=["x"], group="g", ll=0.0, ul=2.5)
    restored = type(fitted).model_validate_json(fitted.model_dump_json())
    assert restored == fitted
    text = fitted.summary()
    assert "/var(_cons[g])" in text and "/var(Residual)" in text
    assert "tabular" in fitted.to_latex()
    assert fitted.provenance["stata_parity_validated"] is False
