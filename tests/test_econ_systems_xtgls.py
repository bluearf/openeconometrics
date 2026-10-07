"""Independent oracles for xtgls (panel FGLS) and xtpcse (panel-corrected standard errors).

Every estimate is rebuilt in NumPy with explicit N x N matrices on small
panels: Prais-Winsten transformation matrices, Omega = Sigma kron I_T (or its
unbalanced analogue built pair by pair), GLS through Omega^-1, and the
Beck-Katz sandwich (X'X)^-1 X'Omega X (X'X)^-1.
"""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError


def make_panel(seed=5, m=5, periods=20):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(m), periods)
    t = np.tile(np.arange(1990, 1990 + periods), m)
    x1, x2 = rng.normal(size=(2, m * periods))
    shocks = rng.multivariate_normal(np.zeros(m), np.eye(m) * 0.5 + 0.5, size=periods)
    e = np.zeros((periods, m))
    rho = np.linspace(0.2, 0.6, m)
    for s in range(periods):
        e[s] = shocks[s] + (rho * e[s - 1] if s else 0)
    scale = np.linspace(0.5, 2.0, m)
    e = (e * scale).T.reshape(-1)
    frame = pd.DataFrame({"id": ids, "t": t, "x1": x1, "x2": x2})
    frame["y"] = 1 + 2 * x1 - 0.5 * x2 + e
    return frame.sample(frac=1, random_state=1).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def arrays(frame):
    frame = frame.sort_values(["id", "t"]).reset_index(drop=True)
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    return frame, x, frame.y.to_numpy(), frame.id.to_numpy()


def est(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


def rho_of(e, ids, kind="regress", k=3):
    out = []
    for i in np.unique(ids):
        r = e[ids == i]
        cur, lag = r[1:], r[:-1]
        size = len(r)
        if kind == "regress":
            value = cur @ lag / (lag @ lag)
        elif kind == "freg":
            value = cur @ lag / (cur @ cur)
        elif kind in ("tscorr", "theil"):
            value = cur @ lag / (r @ r)
            if kind == "theil":
                value *= (size - k) / size
        else:
            dw = 1 - ((cur - lag) ** 2).sum() / (r @ r) / 2
            value = dw if kind == "dw" else (dw * size ** 2 + k ** 2) / (size ** 2 - k ** 2)
        out.append(value)
    return np.array(out)


def pw_matrix(rho, size):
    p = np.eye(size)
    p[0, 0] = np.sqrt(1 - rho ** 2)
    for s in range(1, size):
        p[s, s - 1] = -rho
    return p


def transform(x, y, ids, rho):
    blocks = [pw_matrix(r, (ids == i).sum()) for i, r in zip(np.unique(ids), rho)]
    p = np.zeros((len(y), len(y)))
    start = 0
    for b in blocks:
        size = len(b)
        p[start:start + size, start:start + size] = b
        start += size
    return p @ x, p @ y


def gls_oracle(frame, panels, corr="independent", rhotype="regress"):
    frame, x, y, ids = arrays(frame)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    m, periods = len(np.unique(ids)), len(y) // len(np.unique(ids))
    if corr != "independent":
        rho = rho_of(y - x @ b, ids, rhotype)
        if corr == "ar1":
            rho = np.full(m, rho.mean())
        x, y = transform(x, y, ids, rho)
        b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    if panels == "iid":
        omega = np.eye(len(y)) * (e @ e / len(y))
    elif panels == "heteroskedastic":
        omega = np.diag([(e[ids == i] ** 2).mean() for i in ids])
    else:
        grid = e.reshape(m, periods)
        omega = np.kron(grid @ grid.T / periods, np.eye(periods))
    inv = np.linalg.inv(omega)
    v = np.linalg.inv(x.T @ inv @ x)
    return v @ x.T @ inv @ y, v


@pytest.mark.parametrize("panels", ["iid", "heteroskedastic", "correlated"])
@pytest.mark.parametrize("corr", ["independent", "ar1", "psar1"])
def test_xtgls_matches_explicit_omega(panel, panels, corr):
    beta, v = gls_oracle(panel, panels, corr)
    result = oe.xtgls(data=panel, y="y", x=["x1", "x2"], panel="id", time="t", panels=panels,
                      corr=corr)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-8)
    assert result.inference["use_t"] is False
    assert result.metrics["n_groups"] == 5 and result.metrics["n_periods"] == 20
    if corr == "ar1":
        frame, x, y, ids = arrays(panel)
        e = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        assert_allclose(result.metrics["rho"], rho_of(e, ids).mean(), rtol=1e-10)


@pytest.mark.parametrize("rhotype", ["freg", "tscorr", "dw", "theil", "nagar"])
def test_rho_estimators(panel, rhotype):
    frame, x, y, ids = arrays(panel)
    e = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    result = oe.xtgls(data=panel, y="y", x=["x1", "x2"], panel="id", time="t", corr="psar1",
                      rhotype=rhotype)
    assert_allclose(result.extra["rho"], rho_of(e, ids, rhotype), rtol=1e-10)
    beta, _ = gls_oracle(panel, "iid", "psar1", rhotype)
    assert_allclose(est(result)[0], beta, rtol=1e-9)


def test_xtgls_iterated_and_likelihood(panel):
    frame, x, y, ids = arrays(panel)
    m, periods = 5, 20
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(1000):
        e = y - x @ b
        grid = e.reshape(m, periods)
        sigma = grid @ grid.T / periods
        inv = np.kron(np.linalg.inv(sigma), np.eye(periods))
        v = np.linalg.inv(x.T @ inv @ x)
        new = v @ x.T @ inv @ y
        if np.abs(new - b).max() < 1e-13:
            break
        b = new
    result = oe.xtgls(data=panel, y="y", x=["x1", "x2"], panel="id", time="t",
                      panels="correlated", igls=True, tolerance=1e-12)
    assert_allclose(est(result)[0], new, rtol=1e-8)
    assert_allclose(est(result)[1], np.sqrt(np.diag(v)), rtol=1e-7)
    e = y - x @ new
    grid = e.reshape(m, periods)
    ll = -0.5 * (len(y) * (np.log(2 * np.pi) + 1)
                 + periods * np.log(np.linalg.det(grid @ grid.T / periods)))
    assert_allclose(result.metrics["log_likelihood"], ll, rtol=1e-9)
    het = oe.xtgls(data=panel, y="y", x=["x1", "x2"], panel="id", time="t",
                   panels="heteroskedastic", igls=True, tolerance=1e-12)
    e = y - x @ est(het)[0]
    variances = np.array([(e[ids == i] ** 2).mean() for i in range(m)])
    assert_allclose(het.metrics["log_likelihood"],
                    -0.5 * (periods * (np.log(2 * np.pi * variances) + 1)).sum(), rtol=1e-9)
    assert het.extra["iterations"] > 2


def pcse_oracle(frame, *, corr="independent", pairwise=False, hetonly=False, independent=False):
    frame, x, y, ids = arrays(frame)
    t = frame.t.to_numpy()
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    panels = np.unique(ids)
    if corr != "independent":
        rho = rho_of(y - x @ b, ids)
        if corr == "ar1":
            sizes = np.array([(ids == i).sum() for i in panels])
            rho = np.full(len(panels), ((sizes - 1) * rho).sum() / (sizes - 1).sum())
        x, y = transform(x, y, ids, rho)
        b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    common = [s for s in np.unique(t) if all(((ids == i) & (t == s)).any() for i in panels)]
    n = len(y)
    sigma = np.zeros((len(panels), len(panels)))
    for a, i in enumerate(panels):
        for c, j in enumerate(panels):
            if pairwise:
                shared = np.intersect1d(t[ids == i], t[ids == j])
            else:
                shared = np.array(common)
            ei = np.array([e[(ids == i) & (t == s)][0] for s in shared])
            ej = np.array([e[(ids == j) & (t == s)][0] for s in shared])
            sigma[a, c] = ei @ ej / len(shared) if len(shared) else 0.0
    if hetonly:
        if pairwise:
            sigma = np.diag([(e[ids == i] ** 2).mean() for i in panels])
        else:
            sigma = np.diag(np.diag(sigma))
    if independent:
        sigma = np.eye(len(panels)) * (e @ e / n)
    omega = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            if t[r] == t[c]:
                omega[r, c] = sigma[ids[r], ids[c]]
    bread = np.linalg.inv(x.T @ x)
    return b, bread @ x.T @ omega @ x @ bread, 1 - e @ e / ((y - y.mean()) ** 2).sum()


@pytest.mark.parametrize("options", [{}, {"correlation": "ar1"}, {"correlation": "psar1"},
                                     {"hetonly": True}, {"independent": True}])
def test_xtpcse_balanced(panel, options):
    corr = options.get("correlation", "independent")
    beta, v, r2 = pcse_oracle(panel, corr=corr, hetonly=options.get("hetonly", False),
                              independent=options.get("independent", False))
    result = oe.xtpcse(data=panel, y="y", x=["x1", "x2"], panel="id", time="t", **options)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-8)
    assert_allclose(result.metrics["r_squared"], r2, rtol=1e-10)
    assert result.spec.covariance == "robust"


@pytest.mark.parametrize("options", [{}, {"pairwise": True}, {"hetonly": True},
                                     {"hetonly": True, "pairwise": True}])
def test_xtpcse_unbalanced(panel, options):
    rng = np.random.default_rng(9)
    frame = panel.drop(index=rng.choice(len(panel), size=12, replace=False)).reset_index(drop=True)
    beta, v, _ = pcse_oracle(frame, pairwise=options.get("pairwise", False),
                             hetonly=options.get("hetonly", False))
    result = oe.xtpcse(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", **options)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-8)


def test_xtpcse_casewise_cheaper_path_when_few_common_periods(panel):
    frame = panel[~((panel.id == 0) & (panel.t > 1995))].reset_index(drop=True)
    beta, v, _ = pcse_oracle(frame)
    result = oe.xtpcse(data=frame, y="y", x=["x1", "x2"], panel="id", time="t")
    assert result.extra["common_periods"] == 6
    assert_allclose(est(result)[1], np.sqrt(np.diag(v)), rtol=1e-8)


def test_errors(panel):
    unbalanced = panel.iloc[3:].reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=unbalanced, y="y", x=["x1"], panel="id", time="t", panels="correlated")
    assert error.value.code == "unbalanced_panel"
    gaps = panel[panel.t != 1995].reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=gaps, y="y", x=["x1"], panel="id", time="t", corr="ar1")
    assert error.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as error:
        oe.xtpcse(data=gaps, y="y", x=["x1"], panel="id", time="t", correlation="psar1")
    assert error.value.code == "time_gaps"
    short = panel[panel.t < 1993].reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=short, y="y", x=["x1"], panel="id", time="t", panels="correlated")
    assert error.value.code == "insufficient_periods"
    with pytest.raises(AnalysisError) as error:
        oe.xtpcse(data=panel, y="y", x=["x1"], panel="id", time="t", hetonly=True,
                  independent=True)
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=panel, y="y", x=["x1"], panel="id", time="t", panels="spatial")
    assert error.value.code == "invalid_spec"
    trend = panel.assign(y=np.where(panel.id == 0, panel.t - 1990.0, panel.y))
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=trend, y="y", x=["x1"], panel="id", time="t", corr="psar1")
    assert error.value.code == "invalid_rho"
    disjoint = panel[~((panel.id == 0) & (panel.t < 2000)) & ~((panel.id == 1) & (panel.t >= 2000))]
    with pytest.raises(AnalysisError) as error:
        oe.xtpcse(data=disjoint.reset_index(drop=True), y="y", x=["x1"], panel="id", time="t")
    assert error.value.code == "no_common_periods"
    repeated = pd.concat([panel, panel.head(1)], ignore_index=True)
    with pytest.raises(AnalysisError) as error:
        oe.xtgls(data=repeated, y="y", x=["x1"], panel="id", time="t")
    assert error.value.code == "repeated_time_values"


def test_round_trip_and_rendering(panel):
    result = oe.xtgls(data=panel, y="y", x=["x1", "x2"], panel="id", time="t",
                      panels="heteroskedastic", corr="ar1")
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["extra"]["panels"] == "heteroskedastic"
    assert "Wald chi2" in result.summary() and "x1" in result.to_latex()
    pcse = oe.xtpcse(data=panel, y="y", x=["x1", "x2"], panel="id", time="t")
    assert type(pcse).model_validate_json(pcse.model_dump_json()) == pcse
    missing = panel.copy()
    missing.loc[0, "x2"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.xtpcse(data=missing, y="y", x=["x1", "x2"], panel="id", time="t")
    assert error.value.code == "missing_values"
    dropped = oe.xtpcse(data=missing, y="y", x=["x1", "x2"], panel="id", time="t",
                        missing="drop", pairwise=True)
    assert dropped.nobs == len(panel) - 1
    collinear = oe.xtgls(data=panel.assign(dup=3 * panel.x1), y="y", x=["x1", "dup"],
                         panel="id", time="t")
    assert collinear.provenance["omitted_terms"] == ["dup"]
    assert oe.xtgls.__doc__ and oe.xtpcse.__doc__
