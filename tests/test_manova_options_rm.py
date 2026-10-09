"""Original-cell RM covariance and scalar/Hotelling compatibility oracles."""
import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats.manova_options import rm_mtest
from openecon.econometrics.stats.rm_anova import rm_anova
from openecon.econometrics.stats.rm_contrast import rm_contrast
from openecon.econometrics.summary_state import restore_summary, summary_state


def rm_dgp(seed=818, between=True):
    rng = np.random.default_rng(seed)
    groups = np.repeat(["control", "dose1", "dose2"], [17, 21, 28])
    n = len(groups)
    coding = np.array([[1, 0], [0, 1], [-1, -1]])[pd.Categorical(groups, categories=sorted(set(groups))).codes]
    design = np.column_stack([np.ones(n), coding]) if between else np.ones((n, 1))
    a = rng.normal(size=(4, 4))
    covariance = a@a.T + np.eye(4)
    beta = rng.normal(size=(design.shape[1], 4))
    values = design@beta + rng.multivariate_normal(np.zeros(4), covariance, n)
    frame = pd.DataFrame([(i, g, a, b, values[i, 2*a+b])
                          for i, g in enumerate(groups) for a in range(2) for b in range(2)],
                         columns=["id", "group", "A", "B", "y"])
    fit = rm_anova(frame, "y", "id", ["A", "B"], between=["group"] if between else [])
    return fit, design, values


@pytest.mark.parametrize("seed", [818, 1182])
def test_joint_unequal_between_by_crossed_within_full_gaussian_covariance(seed):
    fit, design, values = rm_dgp(seed)
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    residuals = values-design@beta
    df = len(values)-design.shape[1]
    bread, e = np.linalg.inv(design.T@design), residuals.T@residuals
    left = np.array([[0., 1, -.5], [1., .3, .4]])
    m = np.array([[-1., -.5], [1., -.5], [-1., .5], [1., .5]])
    null = np.array([[.2, -.1], [.1, .3]])
    delta = left@beta@m-null
    a, projected = left@bread@left.T, m.T@e@m
    h = delta.T@np.linalg.solve(a, delta)
    covariance = np.kron(a, projected/df)
    se = np.sqrt(covariance.diagonal())
    for source in (fit, restore_summary(summary_state(fit))):
        output = rm_mtest(source, left, M=m, null=null, alpha=.1)
        np.testing.assert_allclose(output["estimates"].estimate, (left@beta@m).ravel(), rtol=2e-10, atol=1e-11)
        np.testing.assert_allclose(output["estimates"].std_error, se, rtol=2e-10)
        np.testing.assert_allclose(output["estimates"].p_value, 2*stats.t.sf(abs(delta.ravel()/se), df), rtol=2e-9)
        np.testing.assert_allclose(output["estimates"].ci_low, (left@beta@m).ravel()-stats.t.ppf(.95, df)*se, rtol=2e-10)
        np.testing.assert_allclose(output["estimates"].ci_high, (left@beta@m).ravel()+stats.t.ppf(.95, df)*se, rtol=2e-10)
        np.testing.assert_allclose(output["hypothesis_sscp"], h, rtol=2e-10, atol=1e-11)
        np.testing.assert_allclose(output["error_sscp"], projected, rtol=2e-10)
        np.testing.assert_allclose(output["target_covariance"], covariance, rtol=2e-10, atol=1e-11)
        np.testing.assert_allclose(output["roots"].eigenvalue, linalg.eigvalsh(h, projected)[::-1], atol=2e-11)
        assert output.attrs["cell_order"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
        assert output.attrs["sphericity_required"] is False
        again = rm_mtest(restore_summary(summary_state(output)), left, M=m, null=null, alpha=.1)
        pd.testing.assert_frame_equal(output["estimates"], again["estimates"])
        pd.testing.assert_frame_equal(output["target_covariance"], again["target_covariance"])


@pytest.mark.parametrize("seed", [412, 661])
def test_scalar_joint_equals_existing_rm_contrast_and_f_equals_t_squared(seed):
    fit, _, _ = rm_dgp(seed)
    left, m, null = [1., -.3, .7], [-1., 1., -.5, .5], .35
    scalar = rm_contrast(fit, m, between_contrast=left, null=null, alpha=.1)["contrast"]
    joint = rm_mtest(fit, [left], M=np.array(m)[:, None], null=[[null]], alpha=.1)
    np.testing.assert_allclose(joint["estimates"][scalar.columns].to_numpy(float), scalar.to_numpy(float), rtol=2e-10)
    np.testing.assert_allclose(joint["multivariate"].statistic, scalar.statistic.iloc[0]**2, rtol=2e-10)
    np.testing.assert_allclose(joint["multivariate"].p_value, scalar.p_value.iloc[0], rtol=2e-9)
    assert joint["multivariate"].f_type.tolist() == ["exact"]*4


@pytest.mark.parametrize("between", [False, True])
def test_one_between_hypothesis_multiple_within_has_exact_hotelling_law(between):
    fit, design, values = rm_dgp(545, between)
    left = np.zeros((1, design.shape[1]))
    left[0, 0] = 1
    m = np.array([[-1., -.5], [1., -.5], [-1., .5], [1., .5]])
    output = rm_mtest(fit, left, M=m)
    df = len(values)-design.shape[1]
    h, e = output["hypothesis_sscp"].to_numpy(), output["error_sscp"].to_numpy()
    root = np.trace(np.linalg.solve(e, h))
    f = (df-2+1)/2*root
    np.testing.assert_allclose(output["multivariate"].statistic, f, rtol=2e-10)
    np.testing.assert_allclose(output["multivariate"].df1, 2)
    np.testing.assert_allclose(output["multivariate"].df2, df-1)
    np.testing.assert_allclose(output["multivariate"].p_value, stats.f.sf(f, 2, df-1), rtol=2e-9)


def test_rm_refuses_cancellation_dominated_original_cell_projection():
    rng = np.random.default_rng(491)
    subjects = rng.normal(size=50)*1e8
    values = subjects[:, None]+rng.normal(size=(50, 3))
    frame = pd.DataFrame([(i, t, values[i, t]) for i in range(50) for t in range(3)], columns=["id", "t", "y"])
    fit = rm_anova(frame, "y", "id", ["t"])
    with pytest.raises(AnalysisError) as caught:
        rm_mtest(fit, [[1.]], M=[[-1.], [1.], [0.]])
    assert caught.value.code == "unresolved_precision"


def test_rm_refuses_unresolved_already_rounded_original_cell_coefficients():
    fit, design, values = rm_dgp(618)
    values += 1e12
    groups = np.repeat(["control", "dose1", "dose2"], [17, 21, 28])
    frame = pd.DataFrame([(i, g, a, b, values[i, 2*a+b])
                          for i, g in enumerate(groups) for a in range(2) for b in range(2)],
                         columns=["id", "group", "A", "B", "y"])
    fit = rm_anova(frame, "y", "id", ["A", "B"], between=["group"])
    with pytest.raises(AnalysisError) as caught:
        rm_mtest(fit, [[1., 0, 0]], M=[[1.], [-1.], [0.], [0.]])
    assert caught.value.code == "unresolved_precision"
