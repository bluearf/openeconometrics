"""Independent sufficient-moment and separable Gaussian inference oracles."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats
from statsmodels.multivariate.manova import MANOVA

from openecon.dataset import Dataset
from openecon.econometrics.stats.manova import manova
from openecon.econometrics.stats.manova_options import (
    manova_contrast, manova_oneway, manova_summary,
)
from openecon.econometrics.summary_state import restore_summary, summary_state


def dgp(seed=411, p=3):
    rng = np.random.default_rng(seed)
    groups = np.repeat(["control", "dose1", "dose2"], [19, 27, 32])
    n = len(groups)
    x = rng.normal(4, 1.7, n)
    coding = np.array([[1, 0], [0, 1], [-1, -1]])
    code = coding[pd.Categorical(groups, categories=sorted(set(groups))).codes]
    design = np.column_stack([np.ones(n), x, code])
    coefficients = rng.normal(size=(4, p))
    a = rng.normal(size=(p, p))
    covariance = a @ a.T + np.eye(p)
    values = design @ coefficients + rng.multivariate_normal(np.zeros(p), covariance, n)
    frame = pd.DataFrame(values, columns=[f"y{i+1}" for i in range(p)])
    frame["group"], frame["x"], frame["w"] = groups, x, rng.integers(0, 5, n)
    return frame, design


def classical_oracle(h, e, q, v):
    p = len(e)
    roots = np.maximum(linalg.eigvalsh(h, e), 0)[::-1][:min(q, p)]
    s = min(q, p)
    m, n = (abs(p-q)-1)/2, (v-p-1)/2
    pillai = np.sum(roots/(1+roots))
    wilks = np.prod(1/(1+roots))
    hotelling, roy = roots.sum(), roots[0]
    t = math.sqrt((p*p*q*q-4)/(p*p+q*q-5)) if p*p+q*q-5 > 0 else 1
    df = [(s*(2*m+s+1), s*(2*n+s+1)),
          (p*q, (v-(p-q+1)/2)*t-(p*q-2)/2),
          (s*(2*m+s+1), 2*(s*n+1)), (max(p, q), v-max(p, q)+q)]
    f = [df[0][1]*pillai/(df[0][0]*(s-pillai)),
         (1-wilks**(1/t))/wilks**(1/t)*df[1][1]/df[1][0],
         df[2][1]*hotelling/(s*s*(2*m+s+1)), roy*df[3][1]/df[3][0]]
    return np.array([[value, statistic, d1, d2, stats.f.sf(statistic, d1, d2)]
                     for value, statistic, (d1, d2) in zip(
                         [pillai, wilks, hotelling, roy], f, df, strict=True)])


def assert_joint(output, beta, bread, e, df, left, m, null, alpha=.05):
    target = left @ beta @ m
    a, projected = left @ bread @ left.T, m.T @ e @ m
    delta = target-null
    h = delta.T @ np.linalg.solve(a, delta)
    covariance = np.kron(a, projected/df)
    se = np.sqrt(np.diag(covariance))
    t = delta.ravel()/se
    estimates = output["estimates"]
    np.testing.assert_allclose(estimates["estimate"], target.ravel(), rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(estimates["std_error"], se, rtol=2e-10)
    np.testing.assert_allclose(estimates["statistic"], t, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(estimates["p_value"], 2*stats.t.sf(abs(t), df), rtol=2e-9, atol=2e-13)
    np.testing.assert_allclose(estimates["ci_low"], target.ravel()-stats.t.ppf(1-alpha/2, df)*se, rtol=2e-10)
    np.testing.assert_allclose(estimates["ci_high"], target.ravel()+stats.t.ppf(1-alpha/2, df)*se, rtol=2e-10)
    np.testing.assert_allclose(output["target_covariance"], covariance, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(output["hypothesis_sscp"], h, rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(output["error_sscp"], projected, rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(output["roots"]["eigenvalue"], np.maximum(linalg.eigvalsh(h, projected), 0)[::-1], atol=2e-11)
    np.testing.assert_allclose(output["multivariate"][["value", "statistic", "df1", "df2", "p_value"]],
                               classical_oracle(h, projected, len(left), df), rtol=2e-9, atol=2e-11)
    assert estimates["df"].tolist() == [df]*target.size
    assert not output.attrs["familywise_intervals"]


@pytest.mark.parametrize("seed,p", [(411, 3), (819, 4)])
def test_frequency_literal_expansion_summary_full_covariance_and_tests(seed, p):
    frame, _ = dgp(seed, p)
    names = [f"y{i+1}" for i in range(p)]
    weighted = manova_oneway(frame, names, "group", weights="w", alpha=.1)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    groups = sorted(set(expanded.group))
    values = expanded[names].to_numpy()
    counts = [int((expanded.group == g).sum()) for g in groups]
    means = np.array([values[expanded.group == g].mean(0) for g in groups])
    scatters = []
    covariances = {}
    for g in groups:
        y = values[expanded.group == g]
        residuals = y-y.mean(0)
        scatter = residuals.T @ residuals
        scatters.append(scatter)
        covariances[g] = pd.DataFrame(scatter/(len(y)-1), index=names, columns=names)
    e = sum(scatters)
    grand = np.average(means, axis=0, weights=counts)
    h = (means-grand).T @ (np.array(counts)[:, None]*(means-grand))
    df = len(expanded)-len(groups)
    np.testing.assert_allclose(weighted["coefficients"], means, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(weighted["error_sscp"], e, rtol=2e-13)
    np.testing.assert_allclose(weighted["hypothesis_sscp"], h, rtol=2e-13)
    np.testing.assert_allclose(weighted["coefficient_covariance"], np.kron(np.diag(1/np.array(counts)), e/df), rtol=2e-13)
    np.testing.assert_allclose(weighted["multivariate"][["value", "statistic", "df1", "df2", "p_value"]],
                               classical_oracle(h, e, 2, df), rtol=2e-9, atol=2e-11)
    assert weighted.attrs["n"] == sum(counts)
    summary = manova_summary(pd.DataFrame(means, index=groups, columns=names), covariances, counts, alpha=.1)
    for key in ("multivariate", "means", "univariate", "coefficients", "error_sscp", "hypothesis_sscp",
                "coefficient_covariance", "residual_covariance", "roots", "bread"):
        a, b = weighted[key], summary[key]
        numeric = a.select_dtypes(include="number").columns
        np.testing.assert_allclose(a[numeric], b[numeric], rtol=2e-10, atol=2e-11)
    left = np.array([[-1., 1, 0], [-1, 0, 1]])
    m = np.eye(p)[:, :2]+.2
    null = np.array([[.2, -.1], [.3, .7]])
    for source in (weighted, summary, restore_summary(summary_state(weighted)), restore_summary(summary_state(summary))):
        assert_joint(manova_contrast(source, left, M=m, null=null, alpha=.1), means,
                     np.diag(1/np.array(counts)), e, df, left, m, null, .1)


@pytest.mark.parametrize("seed,p", [(918, 3), (1231, 4)])
def test_raw_coordinate_mancova_nonzero_null_statsmodels_and_dataset(seed, p):
    frame, design = dgp(seed, p)
    names = [f"y{i+1}" for i in range(p)]
    y = frame[names].to_numpy()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    bread = np.linalg.inv(design.T @ design)
    residuals = y-design@beta
    e, df = residuals.T@residuals, len(frame)-design.shape[1]
    left = np.array([[0, .2, 1, -.5], [1, -.4, 0, .7]])
    m = np.column_stack([np.arange(1, p+1)/p, np.linspace(-1, 1, p)])
    null = np.array([[.4, -.3], [.1, -.2]])
    sources = [manova(frame, names, ["group"], covariates=["x"]),
               manova(Dataset.from_batches(lambda: (frame.iloc[i:i+13] for i in range(0, len(frame), 13)),
                                          frame.columns, row_count=len(frame)), names, ["group"], covariates=["x"])]
    sm = MANOVA(y, design).mv_test([("joint", left, m, null)]).results["joint"]
    for fit in sources:
        state = fit.attrs["manova_contrast_state"]
        assert state["design_columns"] == ["Intercept", "x", "group[1]", "group[2]"]
        np.testing.assert_allclose(state["coefficients"], beta, rtol=2e-11, atol=1e-12)
        np.testing.assert_allclose(state["bread"], bread, rtol=2e-11, atol=1e-12)
        for source in (fit, restore_summary(summary_state(fit))):
            output = manova_contrast(source, left, M=m, null=null)
            assert_joint(output, beta, bread, e, df, left, m, null)
            np.testing.assert_allclose(output["hypothesis_sscp"], sm["H"], rtol=2e-10, atol=2e-11)
            np.testing.assert_allclose(output["error_sscp"], sm["E"], rtol=2e-10)
            reference = sm["stat"].loc[["Pillai's trace", "Wilks' lambda", "Hotelling-Lawley trace", "Roy's greatest root"]]
            np.testing.assert_allclose(output["multivariate"].value, reference["Value"].astype(float), rtol=2e-10)
            np.testing.assert_allclose(output["multivariate"].iloc[[0, 1, 3]].statistic,
                                       reference.iloc[[0, 1, 3]]["F Value"].astype(float), rtol=2e-10)
    # Refitting transformed outcomes independently yields the same H/E and t law.
    transformed = frame.drop(columns=names).copy()
    transformed[["z1", "z2"]] = y@m
    fit_z = manova(transformed, ["z1", "z2"], ["group"], covariates=["x"])
    original = manova_contrast(sources[0], left, M=m, null=null)
    refit = manova_contrast(fit_z, left, null=null)
    for key in ("estimates", "hypothesis_sscp", "error_sscp", "target_covariance", "multivariate"):
        np.testing.assert_allclose(original[key].select_dtypes(include="number"), refit[key].select_dtypes(include="number"), rtol=3e-10, atol=3e-11)


def test_affine_null_equals_saved_estimate_zero_hypothesis_has_valid_probability_one():
    frame, _ = dgp()
    fit = manova(frame, ["y1", "y2", "y3"], ["group"], covariates=["x"])
    left, m = [[0., 0, 1, 0]], [[1.], [-1], [0]]
    first = manova_contrast(fit, left, M=m)
    target = first["estimates"].estimate.iloc[0]
    output = manova_contrast(fit, left, M=m, null=[[target]])
    assert output["estimates"].statistic.iloc[0] == 0
    assert output["estimates"].p_value.iloc[0] == 1
    assert output["hypothesis_sscp"].iloc[0, 0] == 0
    np.testing.assert_array_equal(output["multivariate"].statistic, np.zeros(4))
    np.testing.assert_array_equal(output["multivariate"].p_value, np.ones(4))


def test_huge_common_outcome_offset_uses_saved_centered_coefficients_and_exact_m_weights():
    frame, _ = dgp(715)
    names = ["y1", "y2", "y3"]
    frame[names] += 1e12
    fit = manova(frame, names, ["group"], covariates=["x"])
    left, m = np.array([[1., 0, 0, 0]]), np.array([[-1.], [1.], [0.]])
    transformed = frame.assign(z=frame.y2-frame.y1)
    reference = manova(transformed, ["z"], ["group"], covariates=["x"])
    a, b = manova_contrast(fit, left, M=m), manova_contrast(reference, left)
    np.testing.assert_allclose(a["estimates"].estimate, b["estimates"].estimate, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(a["target_covariance"], b["target_covariance"], rtol=1e-10)
    # A represented nonzero sum must contribute its huge level; do not silently
    # canonicalize the caller's coefficients into a different hypothesis.
    near = np.array([[-1.], [1+5e-13], [0.]])
    output = manova_contrast(fit, left, M=near)
    state = fit.attrs["manova_contrast_state"]
    origin = np.array(state["outcome_origin"])
    offset = np.array(state["coefficient_offsets"])[0]
    expected = math.fsum(float(a)*float(b) for a, b in zip(offset, near[:, 0], strict=True))
    expected += math.fsum(float(a)*float(b) for a, b in zip(origin-origin[0], near[:, 0], strict=True))
    expected += origin[0]*math.fsum(near[:, 0])
    assert output["estimates"].estimate.iloc[0] == pytest.approx(expected, abs=1e-11)
    assert output["M"].iloc[1, 0] == 1+5e-13


def test_oneway_changes_outcome_units_without_changing_multivariate_inference():
    frame, _ = dgp()
    names = ["y1", "y2", "y3"]
    normal = manova_oneway(frame, names, "group", weights="w")
    scaled = frame.copy()
    scaled[names] *= [1e-9, 1e7, 3]
    output = manova_oneway(scaled, names, "group", weights="w")
    np.testing.assert_allclose(output["multivariate"].select_dtypes(include="number"),
                               normal["multivariate"].select_dtypes(include="number"), rtol=3e-10, atol=3e-11)
