"""Independent complete-profile replication and Gaussian moment RM references."""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats
from statsmodels.stats.anova import AnovaRM

from openecon.econometrics.stats.manova_options import rm_mtest
from openecon.econometrics.stats.rm_anova import rm_anova
from openecon.econometrics.stats.rm_moments import rm_anova_fweight, rm_anova_summary
from openecon.econometrics.summary_state import restore_summary, summary_state


def domain(seed=1234, process=False, between=True):
    rng = np.random.default_rng(seed)
    cells = list(itertools.product(range(2), range(3)))
    groups = ["control", "dose1", "dose2"] if between else ["pooled"]
    a = rng.normal(size=(6, 6))
    covariance = a@a.T/5+np.eye(6)
    profiles, labels, frequencies, rows = [], [], [], []
    for g, group in enumerate(groups):
        for _ in range(15+5*g):
            mean = np.array([.1*a+.3*b+.2*a*b+.3*g*(-1)**b for a, b in cells])
            values = rng.multivariate_normal(mean, covariance)
            if process:
                values = 20+values*.003
            subject = len(profiles)
            weight = int(rng.integers(0, 4))
            profiles.append(values)
            labels.append(group)
            frequencies.append(weight)
            rows.extend([[subject, group, a, b, values[j], weight] for j, (a, b) in enumerate(cells)])
    frame = pd.DataFrame(rows, columns=["id", "group", "A", "B", "y", "w"])
    expanded, expanded_values, expanded_groups = [], [], []
    for i, (values, group, weight) in enumerate(zip(profiles, labels, frequencies, strict=True)):
        for _ in range(weight):
            subject = len(expanded_values)
            expanded_values.append(values)
            expanded_groups.append(group)
            expanded.extend([[subject, group, a, b, values[j]] for j, (a, b) in enumerate(cells)])
    long = pd.DataFrame(expanded, columns=["id", "group", "A", "B", "y"])
    values = np.asarray(expanded_values)
    means, covariances, counts = [], {}, []
    axis = pd.MultiIndex.from_tuples(cells, names=["A", "B"])
    for group in groups:
        block = values[np.array(expanded_groups) == group]
        means.append(block.mean(0))
        covariances[group] = pd.DataFrame(np.cov(block, rowvar=False), index=axis, columns=axis)
        counts.append(len(block))
    summary = pd.DataFrame(means, columns=axis, index=pd.Index(groups, name="group" if between else None))
    coding = np.eye(len(groups))[:-1].T if between else np.empty((1, 0))
    if between:
        coding[-1] = -1
    design = np.column_stack([np.ones(len(values)), coding[[groups.index(group) for group in expanded_groups]]])
    return frame, long, summary, covariances, counts, values, design


def independent_transforms():
    # Any orthonormal basis of each effect subspace gives the same SS/roots.
    a, b = linalg.helmert(2).T, linalg.helmert(3).T
    one_a, one_b = np.ones((2, 1))/np.sqrt(2), np.ones((3, 1))/np.sqrt(3)
    return {"A": np.kron(a, one_b), "B": np.kron(one_a, b), "A#B": np.kron(a, b)}


@pytest.mark.parametrize("process", [False, True])
def test_rm_whole_frequency_summary_replication_and_independent_ss_epsilons(process):
    frame, long, means, covariances, counts, y, x = domain(1134 if process else 1234, process)
    weighted = rm_anova_fweight(frame, "y", "id", ["A", "B"], between=["group"], weights="w", alpha=.1)
    declared = rm_anova_summary(means, covariances, counts, within=["A", "B"], between=["group"], alpha=.1)
    legacy = rm_anova(long, "y", "id", ["A", "B"], between=["group"], alpha=.1)
    origin = y[0]
    beta = np.linalg.lstsq(x, y-origin, rcond=None)[0]
    beta[0] += origin
    bread = np.linalg.inv(x.T@x)
    residual = (y-origin)-x@(beta-np.outer(np.eye(x.shape[1])[0], origin))
    e, nu = residual.T@residual, len(y)-x.shape[1]
    tolerance = 2e-9 if process else 3e-11
    for fit in (weighted, declared):
        for key in ["within", "between", "sphericity", "descriptives"]:
            np.testing.assert_allclose(fit[key].select_dtypes(include="number"), legacy[key].select_dtypes(include="number"), rtol=tolerance, atol=2e-11)
        state = fit.attrs["rm_contrast_state"]
        np.testing.assert_allclose(state["coefficients"], beta, rtol=tolerance, atol=2e-12)
        np.testing.assert_allclose(state["bread"], bread, rtol=3e-11)
        np.testing.assert_allclose(state["residual_cell_covariance"], e/nu, rtol=tolerance)
        np.testing.assert_allclose(fit["original_coefficient_covariance"], np.kron(bread, e/nu), rtol=tolerance)
        for label, m in independent_transforms().items():
            projected = m.T@e@m
            d = m.shape[1]
            roots = np.linalg.eigvalsh(projected)
            gg = roots.sum()**2/(d*(roots**2).sum())
            hf = min(1, (len(y)*d*gg-2)/(d*(nu-d*gg)))
            sphere = fit["sphericity"].loc[label]
            np.testing.assert_allclose(sphere[["epsilon_gg", "epsilon_hf", "epsilon_lb"]].to_numpy(float), [gg, hf, 1/d], rtol=tolerance)
            if d > 1:
                log_w = np.linalg.slogdet(projected)[1]-d*np.log(np.trace(projected)/d)
                chi = -(nu-(2*d*d+d+2)/(6*d))*log_w
                df = d*(d+1)/2-1
                np.testing.assert_allclose(sphere[["mauchly_w", "chi2", "df", "p_value"]].to_numpy(float), [np.exp(log_w), chi, df, stats.chi2.sf(chi, df)], rtol=tolerance)
            for term, indices in [("Intercept", [0]), ("group", [1, 2])]:
                selected = beta[indices]@m
                h = selected.T@np.linalg.solve(bread[np.ix_(indices, indices)], selected)
                source = label if term == "Intercept" else label+"#group"
                for correction, eps in [("sphericity_assumed", 1), ("greenhouse_geisser", gg), ("huynh_feldt", hf), ("lower_bound", 1/d)]:
                    row = fit["within"].loc[(fit["within"].source == source) & (fit["within"].correction == correction)].iloc[0]
                    f = np.trace(h)/len(indices)/(np.trace(projected)/nu)
                    np.testing.assert_allclose(row[["ss", "df", "statistic", "p_value"]].to_numpy(float), [np.trace(h), d*len(indices)*eps, f, stats.f.sf(f, d*len(indices)*eps, d*nu*eps)], rtol=tolerance, atol=2e-11)
    assert weighted.attrs["n_subjects"] == sum(counts)
    assert weighted.attrs["sample"]["physical_subjects"] == frame.id.nunique()
    assert declared.attrs["sample"]["physical_subjects"] is None
    assert weighted.attrs["resource_plan"]["buffers"]["wide_profile_and_transform_copies"] > 0


@pytest.mark.parametrize("process", [False, True])
def test_rm_full_saved_joint_covariance_hotelling_t_ci_and_json(process):
    frame, _, means, covariances, counts, y, x = domain(765 if process else 876, process)
    for fit in (rm_anova_fweight(frame, "y", "id", ["A", "B"], weights="w", between=["group"]),
                rm_anova_summary(means, covariances, counts, within=["A", "B"], between=["group"])):
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y-x@beta
        e, bread, nu = residual.T@residual, np.linalg.inv(x.T@x), len(y)-x.shape[1]
        left = np.array([[0., 1, .3], [0., -.4, 1]])
        m = np.array([[-1., -.5], [1., -.5], [0., 0], [-1., .5], [1., .5], [0., 0]])
        null = np.array([[.2, -.1], [-.1, .2]])*(.003 if process else 1)
        target = left@beta@m
        delta = target-null
        a, projected = left@bread@left.T, m.T@e@m
        h = delta.T@np.linalg.solve(a, delta)
        cov = np.kron(a, projected/nu)
        se = np.sqrt(cov.diagonal())
        for source in (fit, restore_summary(summary_state(fit))):
            output = rm_mtest(source, left, M=m, null=null, alpha=.1)
            np.testing.assert_allclose(output["estimates"].estimate, target.ravel(), atol=2e-12, rtol=2e-9)
            np.testing.assert_allclose(output["target_covariance"], cov, rtol=2e-9)
            np.testing.assert_allclose(output["hypothesis_sscp"], h, rtol=2e-9, atol=2e-11)
            np.testing.assert_allclose(output["estimates"].p_value, 2*stats.t.sf(abs(delta.ravel()/se), nu), rtol=2e-9)
            np.testing.assert_allclose(output["estimates"].ci_low, target.ravel()-stats.t.ppf(.95, nu)*se, atol=2e-12, rtol=2e-9)
            pd.testing.assert_frame_equal(output["estimates"], rm_mtest(restore_summary(summary_state(output)), left, M=m, null=null, alpha=.1)["estimates"])
            assert summary_state(restore_summary(summary_state(fit))) == summary_state(fit)
        single = rm_mtest(fit, left[:1], M=m)
        h = single["hypothesis_sscp"].to_numpy()
        root = np.trace(np.linalg.solve(projected, h))
        f = root*(nu-1)/2
        np.testing.assert_allclose(single["multivariate"].statistic, f, rtol=2e-9)
        np.testing.assert_allclose(single["multivariate"].p_value, stats.f.sf(f, 2, nu-1), rtol=2e-9)


def test_no_between_complete_rm_matches_independent_statsmodels_anovarm():
    frame, long, means, covariances, counts, _, _ = domain(382, between=False)
    sm = AnovaRM(long, "y", "id", within=["A", "B"]).fit().anova_table
    for fit in (rm_anova_fweight(frame, "y", "id", ["A", "B"], weights="w"),
                rm_anova_summary(means, covariances, counts, within=["A", "B"])):
        for name in ["A", "B", "A#B"]:
            row = fit["within"].loc[(fit["within"].source == name) & (fit["within"].correction == "sphericity_assumed")].iloc[0]
            reference = sm.loc[name.replace("#", ":")]
            np.testing.assert_allclose(row[["statistic", "df", "p_value"]].to_numpy(float), reference[["F Value", "Num DF", "Pr > F"]].to_numpy(float), rtol=3e-10)


def test_single_factor_named_axes_are_equivalent_to_multiindex():
    frame, _, _, _, _, _, _ = domain(492, between=False)
    frame = frame.loc[frame.A == 0].copy()
    fit = rm_anova_fweight(frame, "y", "id", ["B"], weights="w")
    positive = frame.loc[frame.w > 0]
    profiles = positive.pivot(index="id", columns="B", values="y")
    frequencies = positive.groupby("id").w.first()
    expanded = profiles.loc[profiles.index.repeat(frequencies)].to_numpy()
    axis = pd.Index([0, 1, 2], name="B")
    means = pd.DataFrame([expanded.mean(0)], columns=axis, index=["pooled"])
    covariance = pd.DataFrame(np.cov(expanded, rowvar=False), index=axis, columns=axis)
    declared = rm_anova_summary(means, {"pooled": covariance}, [len(expanded)], within=["B"])
    np.testing.assert_allclose(declared["within"].select_dtypes(include="number"), fit["within"].select_dtypes(include="number"), rtol=3e-10, atol=1e-12)


def test_huge_rm_non_cancelling_mean_saved_rank_one_f_is_exact():
    frame = domain(846, process=True)[0]
    frame["y"] += 1e12-20
    fit = rm_anova_fweight(frame, "y", "id", ["A", "B"], weights="w", between=["group"])
    left = [[1., 0, 0]]
    transform = np.ones((6, 1))/6
    for source in [fit, restore_summary(summary_state(fit))]:
        query = rm_mtest(source, left, M=transform)
        h = query["hypothesis_sscp"].to_numpy()
        e = query["error_sscp"].to_numpy()
        nu = query.attrs["df_resid"]
        f = float(np.trace(np.linalg.solve(e, h)))*nu
        np.testing.assert_allclose(query["multivariate"].statistic, f, rtol=3e-11)
        np.testing.assert_allclose(query["multivariate"].df1, 1)
        np.testing.assert_allclose(query["multivariate"].df2, nu)
        np.testing.assert_allclose(query["multivariate"].p_value, stats.f.sf(f, 1, nu), atol=1e-15)
        requery = rm_mtest(restore_summary(summary_state(query)), left, M=transform)
        pd.testing.assert_frame_equal(query["multivariate"], requery["multivariate"])
