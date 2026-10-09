"""Independent dense/literal-replication factorial moment references."""
import itertools

import numpy as np
import pandas as pd
import patsy
import pytest
from scipy import linalg, stats
from statsmodels.multivariate.manova import MANOVA

from openecon.econometrics.stats.manova import manova
from openecon.econometrics.stats.manova_factorial import manova_factorial, manova_factorial_summary
from openecon.econometrics.stats.manova_options import manova_contrast
from openecon.econometrics.summary_state import restore_summary, summary_state


def domain(seed=1967, process=False):
    rng = np.random.default_rng(seed)
    rows = []
    covariance = np.array([[1., .35, -.2], [.35, 1.6, .4], [-.2, .4, 1.4]])
    for a, b in itertools.product(["A", "B"], ["early", "mid", "late"]):
        for _ in range(9+2*int(a == "B")+3*int(b == "mid")):
            z = rng.multivariate_normal(np.zeros(3), covariance)
            z += np.array([.4, -.2, .3])*int(a == "B") + np.array([.1, .5, -.3])*["early", "mid", "late"].index(b)
            z += np.array([-.3, .2, .5])*int(a == "B" and b == "late")
            if process:
                z = 80+z*np.array([.005, .008, .003])
            rows.append([a, b, *z, int(rng.integers(0, 5))])
    frame = pd.DataFrame(rows, columns=["A", "B", "y1", "y2", "y3", "w"])
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    keys = list(itertools.product(sorted(frame.A.unique()), sorted(frame.B.unique())))
    means, covariances, counts = [], {}, []
    for key in keys:
        y = expanded.loc[(expanded.A == key[0]) & (expanded.B == key[1]), ["y1", "y2", "y3"]].to_numpy()
        means.append(y.mean(0))
        covariances[key] = np.cov(y, rowvar=False, ddof=1)
        counts.append(len(y))
    summary = pd.DataFrame(means, index=pd.MultiIndex.from_tuples(keys, names=["A", "B"]), columns=["y1", "y2", "y3"])
    return frame, expanded, summary, covariances, counts


@pytest.mark.parametrize("process", [False, True])
@pytest.mark.parametrize("interactions", ["full", "none"])
def test_weighted_summary_literal_dense_type_three_and_statsmodels(process, interactions):
    frame, expanded, summary, covariances, counts = domain(1662 if process else 1967, process)
    names = ["y1", "y2", "y3"]
    weighted = manova_factorial(frame, names, ["A", "B"], weights="w", interactions=interactions, alpha=.1)
    declared = manova_factorial_summary(summary, covariances, counts, interactions=interactions, alpha=.1)
    raw = manova(expanded, names, ["A", "B"], interactions=interactions)
    formula = "C(A, Sum)*C(B, Sum)" if interactions == "full" else "C(A, Sum)+C(B, Sum)"
    design = patsy.dmatrix(formula, expanded)
    x, y = np.asarray(design), expanded[names].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@beta
    e = residual.T@residual
    bread = np.linalg.inv(x.T@x)
    df = len(y)-x.shape[1]
    # Process means around 80 imply harmless 1e-12 cancellation in the dense
    # reference; the production fit centers first and preserves tiny effects.
    tolerance = 2e-9 if process else 3e-11
    for fit in (weighted, declared):
        np.testing.assert_allclose(fit["coefficients"], beta, rtol=tolerance, atol=2e-12)
        np.testing.assert_allclose(fit["bread"], bread, rtol=3e-11, atol=2e-13)
        np.testing.assert_allclose(fit["error_sscp"], e, rtol=tolerance, atol=2e-13)
        np.testing.assert_allclose(fit["coefficient_covariance"], np.kron(bread, e/df), rtol=tolerance, atol=2e-13)
        se = np.sqrt(np.kron(bread, e/df).diagonal())
        np.testing.assert_allclose(fit["coefficient_estimates"].std_error, se, rtol=tolerance)
        np.testing.assert_allclose(fit["coefficient_estimates"].statistic, beta.ravel()/se, rtol=tolerance, atol=2e-9)
        np.testing.assert_allclose(fit["coefficient_estimates"].p_value, 2*stats.t.sf(abs(beta.ravel()/se), df), rtol=2e-7, atol=2e-13)
        np.testing.assert_allclose(fit["coefficient_estimates"].ci_low, beta.ravel()-stats.t.ppf(.95, df)*se, rtol=tolerance, atol=2e-12)
        np.testing.assert_allclose(fit["coefficient_estimates"].ci_high, beta.ravel()+stats.t.ppf(.95, df)*se, rtol=tolerance, atol=2e-12)
        selected = fit["multivariate"].effect != "Intercept" if process else np.ones(len(fit["multivariate"]), dtype=bool)
        np.testing.assert_allclose(fit["multivariate"].loc[selected].select_dtypes(include="number"), raw["multivariate"].loc[selected].select_dtypes(include="number"), rtol=3e-8, atol=3e-9)
        np.testing.assert_allclose(fit["univariate"].select_dtypes(include="number"), raw["univariate"].select_dtypes(include="number"), rtol=3e-8, atol=3e-9)
        for term, block in design.design_info.term_name_slices.items():
            indices = list(range(block.start, block.stop))
            selector = np.eye(x.shape[1])[indices]
            h = (selector@beta).T@np.linalg.solve(selector@bread@selector.T, selector@beta)
            native_name = term.replace("C(A, Sum)", "A").replace("C(B, Sum)", "B").replace(":", "#")
            actual = fit["hypothesis_sscp"].loc[fit["hypothesis_sscp"].effect == native_name, "sscp"].to_numpy().reshape(3, 3)
            np.testing.assert_allclose(actual, h, rtol=3e-8, atol=3e-9)
            np.testing.assert_allclose(fit["roots"].loc[fit["roots"].effect == native_name, "eigenvalue"],
                                       np.maximum(linalg.eigvalsh(h, e), 0)[::-1], rtol=3e-8, atol=3e-7)
            # Location-invariant factor terms use an equivalent centered
            # response refit so the external oracle preserves the tiny units.
            sm_y = y-y[0] if process and native_name != "Intercept" else y
            sm = MANOVA(sm_y, x).mv_test([(native_name, selector)]).results[native_name]
            native = fit["multivariate"].loc[fit["multivariate"].effect == native_name].set_index("test")
            if process and native_name == "Intercept":
                # Statsmodels reconstructs E^-1 H roots from eigenvalues of
                # (E+H)^-1 H here, losing digits when those approach one.
                root = np.trace(np.linalg.solve(e, h))
                np.testing.assert_allclose(native.value.to_numpy(float), [root/(1+root), 1/(1+root), root, root], rtol=3e-11, atol=1e-14)
                np.testing.assert_allclose(native.statistic, root*(df-2)/3, rtol=3e-11)
                continue
            for test, label in [("pillai", "Pillai's trace"), ("wilks", "Wilks' lambda"),
                                ("hotelling", "Hotelling-Lawley trace"), ("roy", "Roy's greatest root")]:
                np.testing.assert_allclose(native.loc[test, "value"], float(sm["stat"].loc[label, "Value"]), rtol=3e-8, atol=3e-7)
                if test != "hotelling" or len(indices) == 1:
                    np.testing.assert_allclose(native.loc[test, ["statistic", "df1", "df2", "p_value"]].to_numpy(float),
                                               sm["stat"].loc["Hotelling-Lawley trace" if len(indices) == 1 else label, ["F Value", "Num DF", "Den DF", "Pr > F"]].to_numpy(float), rtol=3e-8, atol=3e-7)
    assert weighted.attrs["n"] == sum(counts)
    assert declared.attrs["sample"]["physical_rows"] is None
    np.testing.assert_allclose(weighted["multivariate"].select_dtypes(include="number"),
                               declared["multivariate"].select_dtypes(include="number"), rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("process", [False, True])
def test_saved_factorial_affine_joint_full_covariance_and_json_requery(process):
    frame, expanded, summary, covariances, counts = domain(7162 if process else 9167, process)
    for fit in (manova_factorial(frame, ["y1", "y2", "y3"], ["A", "B"], weights="w"),
                manova_factorial_summary(summary, covariances, counts)):
        state = fit.attrs["manova_contrast_state"]
        beta, bread, e = (np.array(state[key]) for key in ("coefficients", "bread", "residual_sscp"))
        left = np.eye(len(beta))[1:3]
        transform = np.array([[-1., 0], [1., -1], [0., 1]])
        null = np.array([[.001, -.002], [.003, -.001]]) if process else np.array([[.1, -.2], [.2, -.1]])
        df = fit.attrs["df_resid"]
        target = left@beta@transform
        covariance = np.kron(left@bread@left.T, transform.T@e@transform/df)
        delta = target-null
        h = delta.T@np.linalg.solve(left@bread@left.T, delta)
        for source in (fit, restore_summary(summary_state(fit))):
            output = manova_contrast(source, left, M=transform, null=null)
            np.testing.assert_allclose(output["estimates"].estimate, target.ravel(), rtol=3e-8, atol=3e-11)
            np.testing.assert_allclose(output["target_covariance"], covariance, rtol=3e-8, atol=3e-13)
            np.testing.assert_allclose(output["hypothesis_sscp"], h, rtol=3e-8, atol=3e-11)
            np.testing.assert_allclose(output["estimates"].p_value, 2*stats.t.sf(abs(delta.ravel()/np.sqrt(covariance.diagonal())), df), rtol=3e-8)
            restored = restore_summary(summary_state(output))
            again = manova_contrast(restored, left, M=transform, null=null)
            pd.testing.assert_frame_equal(output["estimates"], again["estimates"])
            assert summary_state(restore_summary(summary_state(fit))) == summary_state(fit)
            reloaded = restore_summary(summary_state(fit))
            for key in fit:
                pd.testing.assert_frame_equal(fit[key], reloaded[key])
                assert reloaded[key].to_latex() == fit[key].to_latex()


def test_huge_common_outcome_level_remains_location_invariant_for_factor_effects():
    frame, _, _, _, _ = domain()
    names = ["y1", "y2", "y3"]
    frame[names] += 1e12
    fit = manova_factorial(frame, names, ["A", "B"], weights="w")
    shifted = frame.copy()
    shifted[names] -= 1e12
    reference = manova_factorial(shifted, names, ["A", "B"], weights="w")
    actual = manova_contrast(fit, [[0., 1, 0, 0, 0, 0]], M=[[-1.], [1.], [0.]])
    expected = manova_contrast(reference, [[0., 1, 0, 0, 0, 0]], M=[[-1.], [1.], [0.]])
    np.testing.assert_allclose(actual["estimates"].select_dtypes(include="number"), expected["estimates"].select_dtypes(include="number"), rtol=3e-10, atol=3e-11)


@pytest.mark.parametrize("level", [80., 1e12])
def test_large_intercept_rank_one_exact_f_uses_direct_root_without_cancellation(level):
    frame, expanded, _, _, _ = domain(1662, process=True)
    names = ["y1", "y2", "y3"]
    frame[names] += level-80
    expanded[names] += level-80
    fit = manova_factorial(frame, names, ["A", "B"], weights="w")
    h = fit["hypothesis_sscp"].loc[fit["hypothesis_sscp"].effect == "Intercept", "sscp"].to_numpy().reshape(3, 3)
    e = fit["error_sscp"].to_numpy()
    nu = fit.attrs["df_resid"]
    root = np.trace(np.linalg.solve(e, h))
    f = root*(nu-2)/3
    rows = fit["multivariate"].loc[fit["multivariate"].effect == "Intercept"]
    np.testing.assert_allclose(rows.statistic, f, rtol=3e-11)
    np.testing.assert_allclose(rows.df1, 3)
    np.testing.assert_allclose(rows.df2, nu-2)
    np.testing.assert_allclose(rows.p_value, stats.f.sf(f, 3, nu-2), atol=1e-15)
    assert rows.f_type.tolist() == ["exact"]*4
    for source in [fit, restore_summary(summary_state(fit))]:
        query = manova_contrast(source, np.eye(6)[:1])
        np.testing.assert_allclose(query["multivariate"].statistic, f, rtol=3e-11)
        np.testing.assert_allclose(query["multivariate"].df1, 3)
        np.testing.assert_allclose(query["multivariate"].df2, nu-2)
        np.testing.assert_allclose(query["multivariate"].p_value, stats.f.sf(f, 3, nu-2), atol=1e-15)
        requery = manova_contrast(restore_summary(summary_state(query)), np.eye(6)[:1])
        pd.testing.assert_frame_equal(query["multivariate"], requery["multivariate"])
    legacy = manova(expanded, names, ["A", "B"])["multivariate"]
    old = legacy.loc[(legacy.effect == "Intercept") & (legacy.test == "pillai"), "statistic"].iloc[0]
    assert not np.isfinite(old) or abs(old/f-1) > 1e-8
