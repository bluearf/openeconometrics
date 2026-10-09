"""Full-matrix analytical MI oracle and restored joint inference contracts."""

import json

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import chi2, f, t
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


def fixture(m=8):
    rng = np.random.default_rng(7631)
    q = rng.normal(size=(m, 3)) + [1.0, -0.5, 2.0]
    u = []
    for _ in range(m):
        a = rng.normal(size=(3, 3))
        u.append(a @ a.T + np.diag([1.0, 2.0, 3.0]))
    return q, np.array(u)


@pytest.mark.parametrize("complete_df", [None, 60.0])
def test_full_matrix_rubin_marginal_reference(complete_df):
    q, u = fixture()
    p = oe.mi_pool(
        q,
        u,
        terms=["a", "b", "c"],
        complete_df=complete_df,
        imputation_description="Declared common parameter with justified eight draws",
    )
    mean = q.mean(0)
    within = u.mean(0)
    between = np.cov(q.T, ddof=1)
    total = within + 1.125 * between
    riv = 1.125 * between.diagonal() / within.diagonal()
    old = 7 * (1 + 1 / riv) ** 2
    df = old if complete_df is None else 1 / (1 / old + 1 / (60 * 61 / 63 / (1 + riv)))
    se = np.sqrt(total.diagonal())
    actual = p.table
    assert_allclose(p.covariance, total, rtol=1e-13)
    assert_allclose(actual.estimate, mean)
    assert_allclose(actual.std_error, se)
    assert_allclose(actual.df, df)
    assert_allclose(actual.p_value, 2 * t.sf(abs(mean / se), df), rtol=2e-12)
    assert_allclose(actual.ci_low, mean - t.isf(0.025, df) * se, rtol=2e-12)
    assert_allclose(actual.fraction_missing_information, (riv + 2 / (df + 3)) / (1 + riv))
    assert_allclose(p.tables["between_covariance"], between)
    restored = oe.MIPoolResult.model_validate_json(p.model_dump_json())
    assert restored == p
    assert "tabular" in str(restored.to_latex())
    state = json.loads(p.model_dump_json())
    state["estimates"][0][0] += 0.01
    with pytest.raises(ValueError, match="integrity"):
        oe.MIPoolResult.model_validate(state)


def oracle(q, u, r, null, finite_df=None):
    """Independent NumPy moment equations with SciPy reference F tail."""
    transformed = q @ r.T - null
    v = np.mean(np.array([r @ s @ r.T for s in u]), 0)
    between = np.cov(transformed.T, ddof=1)
    between = np.atleast_2d(between)
    k, m = len(null), len(q)
    riv = (1 + 1 / m) * np.trace(np.linalg.solve(v, between)) / k
    stat = transformed.mean(0) @ np.linalg.solve(v, transformed.mean(0)) / (k * (1 + riv))
    degrees = k * (m - 1)
    if finite_df is None:
        df = (
            4 + (degrees - 4) * (1 + (1 - 2 / degrees) / riv) ** 2
            if degrees > 4
            else (m - 1) * (k + 1) * (1 + 1 / riv) ** 2 / 2
        )
    else:
        # Reiter original Eq 2, evaluated as separate summands (author PDF).
        a = riv * degrees / (degrees - 2)
        vs = finite_df * (finite_df + 1) / (finite_df + 3)
        h2, h4 = vs - 2 * (1 + a), vs - 4 * (1 + a)
        summands = [
            1 / h4,
            a * a * h2 / ((degrees - 4) * (1 + a) ** 2 * h4),
            8 * a * a * h2 / ((degrees - 4) * (1 + a) * h4**2),
            4 * a * a / ((degrees - 4) * (1 + a) * h4),
            4 * a * a / ((degrees - 4) * h4 * h2),
            16 * a * a * h2 / ((degrees - 4) * h4**3),
            8 * a * a / ((degrees - 4) * h4**2),
        ]
        df = 4 + 1 / sum(summands)
    return stat, k, df, f.sf(stat, k, df), riv


@pytest.mark.parametrize(
    "m,k,complete_df", [(3, 1, None), (3, 2, None), (8, 2, None), (8, 2, 100.0), (15, 3, 250.0)]
)
def test_d1_restriction_and_both_finite_imputation_branches(m, k, complete_df):
    q, u = fixture(m)
    r = np.eye(3)[:k]
    r[0] = [1.0, 0.2, -0.4]
    null = np.linspace(-0.3, 0.2, k)
    pool = oe.mi_pool(
        q, u, complete_df=complete_df, imputation_description="Shared normal-limit estimand"
    )
    result = oe.mi_test(pool, r, null)
    expected = oracle(q, u, r, null, complete_df)
    actual = result.table
    assert_allclose(
        actual.iloc[0][["statistic", "df1", "df2", "p_value", "relative_increase_variance"]].astype(
            float
        ),
        expected,
        rtol=3e-12,
    )
    assert oe.MIJointResult.model_validate_json(result.model_dump_json()) == result
    # Change restriction basis: joint null and p-value must be unchanged.
    transform = np.diag(np.arange(1, k + 1))
    other = oe.mi_test(pool, transform @ r, transform @ null)
    assert_allclose(other.table.statistic, actual.statistic)
    assert_allclose(other.table.p_value, actual.p_value)


def test_complete_data_zero_between_limit_and_null():
    q = [[1.0, 2.0]] * 8
    u = [[[2.0, 0.8], [0.8, 3.0]]] * 8
    pool = oe.mi_pool(
        q, u, imputation_description="Identical fits with zero imputation uncertainty"
    )
    joint = oe.mi_test(pool)
    mahal = np.array([1.0, 2.0]) @ np.linalg.solve(u[0], [1.0, 2.0])
    assert joint.table.df2.iloc[0] is None
    assert_allclose(joint.table.p_value, [chi2.sf(mahal, 2)])
    assert_allclose(pool.table.std_error, np.sqrt([2.0, 3.0]))
    assert_allclose(oe.mi_test(pool, values=[1.0, 2.0]).table.p_value, [1.0])
    finite = oe.mi_pool(
        q, u, complete_df=100, imputation_description="Zero between covariance, finite complete df"
    )
    assert_allclose(oe.mi_test(finite).table.df2, [100 * 101 / 103])


def test_rejections_budget_dtype_convergence_and_incompatible_state():
    q, u = fixture()
    for changes in (
        dict(estimates=q.astype(complex)),
        dict(estimates=q.astype(bool)),
        dict(covariances=u * 0.0),
        dict(estimates=q[:1], covariances=u[:1]),
    ):
        args = dict(estimates=q, covariances=u, imputation_description="commonestimand")
        args.update(changes)
        with pytest.raises(AnalysisError):
            oe.mi_pool(**args)
    for df in (True, 0, float("inf"), float("nan")):
        with pytest.raises(AnalysisError):
            oe.mi_pool(q, u, complete_df=df, imputation_description="commonestimand")
    with pytest.raises(AnalysisError):
        oe.mi_pool(q, u, terms=[True, "b", "c"], imputation_description="commonestimand")
    pool = oe.mi_pool(q, u, complete_df=30, imputation_description="commonestimand")
    with pytest.raises(AnalysisError):
        oe.mi_test(pool, [[1, 0, 0], [2, 0, 0]])
    low = oe.mi_pool(q[:2], u[:2], complete_df=30, imputation_description="commonestimand")
    with pytest.raises(AnalysisError, match="Reiter"):
        oe.mi_test(low)
    high = oe.mi_pool(q * 100, u, complete_df=30, imputation_description="commonestimand")
    with pytest.raises(AnalysisError, match="positive-moment"):
        oe.mi_test(high)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.mi_pool(
            np.zeros((100, 32)),
            np.tile(np.eye(32), (100, 1, 1)),
            imputation_description="declaredestimate",
        )
    if torch.backends.mps.is_available():
        with pytest.raises(AnalysisError, match="CPU"):
            oe.mi_pool(
                torch.tensor(q, device="mps", dtype=torch.float32),
                u,
                imputation_description="commonestimand",
            )


def test_restored_fitted_results_pooling_preserves_sample_and_rejects_mismatch():
    import pandas as pd

    rng = np.random.default_rng(592)
    data = pd.DataFrame({"x": rng.normal(size=70), "y": rng.normal(size=70)})
    fits = [
        oe.ols(data=data.assign(y=data.y + rng.normal(scale=0.2, size=70)), y="y", x=["x"])
        for _ in range(8)
    ]
    pool = oe.mi_pool(fits, imputation_description="Declared eight imputed analyses, common sample")
    assert pool.metadata["source_results"][0]["id"] == fits[0].id
    assert pool.complete_df == 68
    assert oe.mi_test(pool, [[0.0, 1.0]]).table.df1.iloc[0] == 1
    fits[-1] = oe.ols(data=data.iloc[:-1], y="y", x=["x"])
    with pytest.raises(AnalysisError, match="identical"):
        oe.mi_pool(fits, imputation_description="incompatible sample")


def test_before_coercion_and_copy_shortcuts_reject_invalid_scientific_state():
    from openecon.econometrics.mi.joint import digest

    q, u = fixture()
    pool = oe.mi_pool(q, u, complete_df=100, imputation_description="Declared common estimand")
    state = pool.model_dump(mode="json")
    normalized = dict(state)
    normalized["estimates"] = [list(row) for row in state["estimates"]]
    normalized["estimates"][0][0] = 1.0
    normalized["integrity_sha256"] = digest(
        {k: v for k, v in normalized.items() if k != "integrity_sha256"}
    )
    normalized["estimates"][0][0] = True
    with pytest.raises((ValueError, AnalysisError), match="numeric"):
        oe.MIPoolResult.model_validate(normalized)
    normalized = pool.model_dump(mode="json")
    normalized["complete_df"] = 1.0
    normalized["integrity_sha256"] = digest(
        {k: v for k, v in normalized.items() if k != "integrity_sha256"}
    )
    normalized["complete_df"] = True
    with pytest.raises(ValueError, match="boolean"):
        oe.MIPoolResult.model_validate(normalized)
    with pytest.raises(ValueError):
        pool.model_copy(update={"complete_df": -1})
    with pytest.raises(ValueError):
        oe.mi_test(pool).model_copy(update={"values": (100.0, 200.0, 300.0)})
    assert pool.model_copy(deep=True) == pool


def test_declared_cpu_double_d1_ignores_ambient_default_device_and_dtype():
    q, u = fixture()
    old_dtype = torch.get_default_dtype()
    torch.set_default_device("meta")
    torch.set_default_dtype(torch.float32)
    try:
        pool = oe.mi_pool(q, u, imputation_description="Declared common estimates, CPU only")
        result = oe.mi_test(pool)
        assert result.table.df1.iloc[0] == 3
        assert result.table.p_value.iloc[0] > 0
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(old_dtype)
