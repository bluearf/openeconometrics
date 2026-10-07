"""Independent NumPy horizon, structural impact and generalized FEVD references."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def fixture(n=140):
    rng = np.random.default_rng(902)
    z = rng.normal(size=n)
    shock = z + 0.6 * rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.4 * shock[t] + 0.2 * shock[t - 1] + rng.normal()
    return pd.DataFrame(dict(y=y, shock=shock, z=z, t=np.arange(n)))


@pytest.mark.parametrize(
    "iv,cumulative", [(False, False), (True, False), (False, True), (True, True)]
)
def test_horizon_coefficients_full_hac_and_sample_alignment(iv, cumulative):
    df = fixture()
    n = len(df)
    hs = 4
    scores = np.zeros((n, hs + 1))
    betas = []
    for h in range(hs + 1):
        rows = np.arange(1, n - h)
        x = np.column_stack(
            [
                np.ones(len(rows)),
                df.shock.to_numpy()[rows],
                df.y.to_numpy()[rows - 1],
                df.shock.to_numpy()[rows - 1],
            ]
        )
        target = df.y.to_numpy()[rows + h] - (df.y.to_numpy()[rows - 1] if cumulative else 0)
        if iv:
            z = x.copy()
            z[:, 1] = df.z.to_numpy()[rows]
            p = z @ np.linalg.inv(z.T @ z) @ z.T @ x
            bread = np.linalg.inv(p.T @ p)
            beta = bread @ p.T @ target
            influence = (p @ bread[:, 1]) * (target - x @ beta)
        else:
            beta = np.linalg.lstsq(x, target, rcond=None)[0]
            bread = np.linalg.inv(x.T @ x)
            influence = (x @ bread[:, 1]) * (target - x @ beta)
        scores[rows, h] = influence * np.sqrt(len(rows) / (len(rows) - x.shape[1]))
        betas.append(beta[1])
    v = scores.T @ scores
    for lag in range(1, hs + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        v += (1 - lag / (hs + 1)) * (gamma + gamma.T)
    result = getattr(oe, "lpiv" if iv else "lp")(
        data=df,
        y="y",
        x=["shock"],
        instruments=["z"] if iv else None,
        time="t",
        horizons=hs,
        cumulative=cumulative,
    )
    np.testing.assert_allclose([c.estimate for c in result.coefficients], betas, atol=2e-9)
    np.testing.assert_allclose(result.covariance_matrix, v, rtol=2e-7, atol=2e-9)
    assert result.extra["nobs_by_horizon"] == list(range(n - 1, n - hs - 2, -1))
    assert result.extra["equations"][4]["sample_positions"] == list(range(1, n - 4))
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra


def test_panel_lp_unit_fixed_effects_and_joint_cluster_covariance():
    df = pd.concat(
        [fixture(90 + i * 4).assign(unit=i, y=lambda d: d.y + i * 0.7) for i in range(6)],
        ignore_index=True,
    )
    hs = 3
    influence = np.zeros((len(df), hs + 1))
    betas = []
    for h in range(hs + 1):
        designs, targets, positions = [], [], []
        for unit, g in df.groupby("unit", sort=True):
            t = np.arange(1, len(g) - h)
            rows = g.index.to_numpy()[t]
            dummy = np.column_stack([np.full(len(t), int(unit == i)) for i in range(1, 6)])
            x = np.column_stack(
                [
                    np.ones(len(t)),
                    g.shock.to_numpy()[t],
                    g.y.to_numpy()[t - 1],
                    g.shock.to_numpy()[t - 1],
                    dummy,
                ]
            )
            designs.append(x)
            targets.append(g.y.to_numpy()[t + h])
            positions.extend(rows.tolist())
        x, y = np.vstack(designs), np.concatenate(targets)
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        bread = np.linalg.inv(x.T @ x)
        influence[positions, h] = (
            (x @ bread[:, 1]) * (y - x @ b) * np.sqrt(6 / 5 * (len(y) - 1) / (len(y) - x.shape[1]))
        )
        betas.append(b[1])
    sums = np.vstack([influence[df.unit == i].sum(0) for i in range(6)])
    result = oe.panel_lp(data=df, y="y", x=["shock"], panel="unit", time="t", horizons=hs)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], betas, atol=2e-9)
    np.testing.assert_allclose(result.covariance_matrix, sums.T @ sums, rtol=2e-7, atol=2e-9)
    assert result.inference["df_inference"] == 5


@pytest.mark.parametrize("long_run", [False, True])
def test_published_macro_recursive_short_and_long_run_impacts_irfs(long_run):
    df = pd.read_csv(Path(__file__).parent / "fixtures/structural-macro.csv")
    names = ["growth", "inflation"]
    restrictions = [[None, 0], [None, None]]
    result = oe.svar(
        data=df,
        y=names,
        time="t",
        long_run=restrictions if long_run else None,
        short_run=None if long_run else restrictions,
    )
    ref = VAR(df[names]).fit(1)
    sigma = ref.sigma_u_mle.to_numpy()
    a = ref.coefs[0]
    f = np.linalg.inv(np.eye(2) - a)
    impact = (
        np.linalg.solve(f, np.linalg.cholesky(f @ sigma @ f.T))
        if long_run
        else np.linalg.cholesky(sigma)
    )
    np.testing.assert_allclose(result.extra["structural"]["impact"], impact, rtol=2e-7, atol=2e-8)
    responses = oe.svar_irf(result, steps=3, draws=20, seed=93)
    for h in range(4):
        expected = np.linalg.matrix_power(a, h) @ impact
        np.testing.assert_allclose(
            responses[responses.horizon == h].irf.to_numpy().reshape(2, 2),
            expected,
            rtol=2e-7,
            atol=2e-8,
        )
    again = oe.svar_irf(result, steps=3, draws=20, seed=93)
    np.testing.assert_array_equal(
        responses[["irf", "ci_low", "ci_high"]], again[["irf", "ci_low", "ci_high"]]
    )
    assert (responses.ci_high >= responses.ci_low).all()
    restricted = (
        (responses.horizon == 0)
        & (responses.response == "growth")
        & (responses.shock == "inflation")
    )
    if not long_run:
        np.testing.assert_allclose(responses.loc[restricted, ["ci_low", "ci_high"]], 0, atol=1e-8)
        assert (responses.loc[~restricted, "ci_high"] > responses.loc[~restricted, "ci_low"]).all()


def test_generalized_fevd_published_formula_normalization_and_order_invariance():
    df = pd.read_csv(Path(__file__).parent / "fixtures/structural-macro.csv")
    model = oe.var(data=df, y=["growth", "inflation"], time="t", lags=1, lm_lags=0, irf_steps=1)
    ref = VAR(df[["growth", "inflation"]]).fit(1)
    a, sigma = ref.coefs[0], ref.sigma_u_mle.to_numpy()
    phi = np.array([np.linalg.matrix_power(a, h) for h in range(10)])
    raw = (phi @ sigma) ** 2
    raw = (
        raw.sum(0)
        / sigma.diagonal()[None, :]
        / np.einsum("hij,jk,hik->i", phi, sigma, phi)[:, None]
    )
    norm = raw / raw.sum(1)[:, None]
    result = oe.connectedness(model, horizon=10)
    np.testing.assert_allclose(
        result["fevd"].normalized_percent.to_numpy().reshape(2, 2), norm * 100, atol=2e-8
    )
    assert result.attrs["total_percent"] == pytest.approx((norm.sum() - np.trace(norm)) * 50)
    assert result["directional"].net_percent.sum() == pytest.approx(0, abs=1e-10)
    reversed_result = oe.connectedness(
        oe.var(data=df, y=["inflation", "growth"], time="t", lags=1, lm_lags=0, irf_steps=1),
        horizon=10,
    )
    assert result.attrs["total_percent"] == pytest.approx(reversed_result.attrs["total_percent"])
    assert "\\toprule" in result.to_latex()


def test_sign_restriction_admissibility_and_seed_protocol():
    df = fixture()
    names = ["y", "shock"]
    signs = [
        {"horizon": 0, "response": "y", "shock": "y", "sign": 1},
        {"horizon": 1, "response": "y", "shock": "y", "sign": 1},
    ]
    result = oe.svar(data=df, y=names, time="t", signs=signs, accepted=12, draws=500, seed=9)
    again = oe.svar(data=df, y=names, time="t", signs=signs, accepted=12, draws=500, seed=9)
    assert result.extra["structural"] == again.extra["structural"]
    bands = oe.svar_irf(result, steps=2, draws=20)
    assert "not sampling confidence" in bands.attrs["interval_method"]
    assert (
        bands[(bands.shock == "y") & (bands.response == "y") & (bands.horizon <= 1)].ci_low.min()
        > 0
    )


@pytest.mark.parametrize(
    "case,code",
    [
        ("unidentified", "unidentified_svar"),
        ("bad_restriction", "invalid_restrictions"),
        ("instrument", "singular_design"),
        ("gap", "time_gaps"),
        ("bandwidth", "invalid_bandwidth"),
    ],
)
def test_identification_calendar_and_overlap_guards(case, code):
    df = fixture()
    with pytest.raises(AnalysisError) as error:
        if case == "unidentified":
            oe.svar(data=df, y=["y", "shock"], time="t")
        elif case == "bad_restriction":
            oe.svar(data=df, y=["y", "shock"], time="t", short_run=[[None, 1], [None, None]])
        elif case == "instrument":
            oe.lpiv(data=df.assign(z=1.0), y="y", x=["shock"], instruments=["z"], time="t")
        elif case == "gap":
            oe.lp(data=df.drop(index=10), y="y", x=["shock"], time="t")
        else:
            oe.lp(data=df, y="y", x=["shock"], time="t", horizons=4, bandwidth=1)
    assert error.value.code == code


@pytest.mark.parametrize("long_run", [False, True])
def test_published_macro_interval_independent_gaussian_wishart_algebra(long_run):
    import torch  # shared RNG protocol only; oracle algebra is NumPy

    df = pd.read_csv(Path(__file__).parent / "fixtures/structural-macro.csv")
    names = ["growth", "inflation"]
    restrictions = [[None, 0], [None, None]]
    result = oe.svar(
        data=df,
        y=names,
        time="t",
        **({"long_run": restrictions} if long_run else {"short_run": restrictions}),
    )
    values = df[names].to_numpy()
    z = np.column_stack([values[:-1], np.ones(len(values) - 1)])
    original = np.linalg.lstsq(z, values[1:], rcond=None)[0].T
    residual = values[1:] - z @ original.T
    sigma = residual.T @ residual / len(z)
    covariance = np.kron(sigma, np.linalg.inv(z.T @ z))
    np.testing.assert_allclose(result.covariance_matrix, covariance, rtol=2e-9, atol=2e-10)
    root = np.linalg.cholesky(covariance)
    p = np.linalg.cholesky(sigma)
    generator = torch.Generator().manual_seed(93)
    samples = []
    for _ in range(20):
        beta = (
            original.ravel()
            + root @ torch.randn(6, generator=generator, dtype=torch.float64).numpy()
        )
        a = beta.reshape(2, 3)[:, :2]
        errors = torch.randn((len(z), 2), generator=generator, dtype=torch.float64).numpy() @ p.T
        sd = errors.T @ errors / len(z)
        if long_run:
            f = np.linalg.inv(np.eye(2) - a)
            impact = np.linalg.solve(f, np.linalg.cholesky(f @ sd @ f.T))
        else:
            impact = np.linalg.cholesky(sd)
        samples.append([np.linalg.matrix_power(a, h) @ impact for h in range(4)])
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    actual = oe.svar_irf(result, steps=3, draws=20, seed=93)
    np.testing.assert_allclose(actual.ci_low, low.ravel(), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(actual.ci_high, high.ravel(), rtol=2e-7, atol=2e-8)


def test_svar_original_row_hash_and_prediction_alignment_after_sort_and_leading_drop():
    df = fixture().sample(frac=1, random_state=8).reset_index(drop=True)
    df.loc[df.t == 0, "y"] = np.nan
    result = oe.svar(
        data=df, y=["y", "shock"], time="t", missing="drop", short_run=[[None, 0], [None, None]]
    )
    expected = (
        df.index[df.t >= 2].to_numpy()[np.argsort(df.loc[df.t >= 2, "t"].to_numpy())].tolist()
    )
    assert result.sample_positions == expected
    assert result.nobs_original == len(df) and result.dropped_rows == 2
    for point in result.predictions:
        assert point["observed"] == pytest.approx(df.y.iloc[point["row"]])
    assert result.provenance["sample_order"] == "sorted by t"
    assert ResultBundle.model_validate_json(result.model_dump_json()).sample_positions == expected


def test_lpiv_perfect_first_stage_is_identified():
    df = fixture().assign(z=lambda d: d.shock)
    ordinary = oe.lp(data=df, y="y", x=["shock"], time="t", horizons=2)
    instrumented = oe.lpiv(data=df, y="y", x=["shock"], instruments=["z"], time="t", horizons=2)
    np.testing.assert_allclose(
        [c.estimate for c in instrumented.coefficients],
        [c.estimate for c in ordinary.coefficients],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        instrumented.covariance_matrix, ordinary.covariance_matrix, atol=1e-9
    )
