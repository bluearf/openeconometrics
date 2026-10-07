import numpy as np
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from test_econ_teffects import oracle as stacked_oracle
from test_econ_teffects_matching import make_data, sets


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
@pytest.mark.parametrize("target", [1, 2])
def test_multivalued_atet_entire_joint_sandwich_against_numerical_jacobian(method, target):
    df = make_data(n=350)
    rng = np.random.default_rng(9)
    df["d"] = rng.choice(3, len(df), p=[0.4, 0.35, 0.25])
    df["g"] = np.arange(len(df)) % 35
    result = oe.teffects(
        data=df,
        y="y",
        treatment="d",
        x=["x1", "x2"],
        method=method,
        estimand="atet",
        tlevel=target,
        covariance="cluster",
        cluster="g",
    )
    est, se, theta, full_cov = stacked_oracle(
        df, method, estimand="atet", cluster="g", target=target
    )
    transform = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    cov = transform @ full_cov[-3:, -3:] @ transform.T
    assert_allclose([c.estimate for c in result.coefficients], est, rtol=1e-7, atol=1e-9)
    assert_allclose(result.covariance_matrix, cov, rtol=2e-6, atol=1e-9)
    assert result.extra["target_population"]["treatment_level"] == target
    assert (
        ResultBundle.model_validate_json(result.model_dump_json()).covariance_matrix
        == result.covariance_matrix
    )


@pytest.mark.parametrize("kind", ["robust", "iid"])
@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_exact_strata_ties_and_matching_variance_against_brute_force(kind, estimand):
    df = make_data(n=180)
    df["cell"] = np.arange(len(df)) % 3
    coords = df[["k1", "k2"]].to_numpy()
    y = df.y.to_numpy()
    d = df.d.to_numpy()
    cells = df.cell.to_numpy()
    n = len(df)
    usage = np.zeros(n)
    usage2 = np.zeros(n)
    imputed = np.zeros((n, 2))
    imputed[np.arange(n), d] = y
    pairs = {}
    directions = [1] if estimand == "atet" else [0, 1]
    for level in directions:
        for cell in range(3):
            q = np.flatnonzero((d == level) & (cells == cell))
            pool = np.flatnonzero((d != level) & (cells == cell))
            mask, _ = sets(coords[q], coords[pool], 1)
            count = mask.sum(1)
            usage[pool] += (mask / count[:, None]).sum(0)
            usage2[pool] += (mask / count[:, None] ** 2).sum(0)
            imputed[q, 1 - level] = mask @ y[pool] / count
            pairs[level, cell] = q, pool, mask
    effect = imputed[:, 1] - imputed[:, 0]
    target = np.flatnonzero(d == 1) if estimand == "atet" else np.arange(n)
    tau = effect[target].mean()
    sigma = np.zeros(n)
    if kind == "iid":
        square = np.zeros(n)
        for (level, cell), (q, pool, mask) in pairs.items():
            differences = (2 * level - 1) * (y[q, None] - y[pool][None, :]) - tau
            square[q] = (mask * differences**2).sum(1) / mask.sum(1)
        sigma[:] = square[target].mean() / 2
    else:
        for level in [0, 1]:
            for cell in range(3):
                g = np.flatnonzero((d == level) & (cells == cell))
                mask, _ = sets(coords[g], coords[g], 2, own=True)
                count = mask.sum(1)
                sigma[g] = count / (count + 1) * (y[g] - mask @ y[g] / count) ** 2
    if estimand == "ate":
        variance = ((effect - tau) ** 2).sum() + ((usage**2 + 2 * usage - usage2) * sigma).sum()
    else:
        control = d == 0
        variance = ((effect[target] - tau) ** 2).sum() + (
            (usage[control] ** 2 - usage2[control]) * sigma[control]
        ).sum()
    variance /= len(target) ** 2
    fit = oe.teffects(
        data=df,
        y="y",
        treatment="d",
        x=["k1", "k2"],
        ematch=["cell"],
        method="nnmatch",
        estimand=estimand,
        metric="euclidean",
        matching_vce=kind,
    )
    assert_allclose(fit.coefficients[0].estimate, tau, rtol=1e-11)
    assert_allclose(fit.covariance_matrix, [[variance]], rtol=1e-10)
    assert fit.extra["exact_cells"] == 3 and fit.extra["matches"]["units_with_ties"] > 0


@pytest.mark.parametrize("method", ["nnmatch", "psmatch"])
def test_matching_frequency_weights_literal_duplication_and_sample_identity(method):
    df = make_data(n=150).assign(f=np.random.default_rng(10).integers(1, 4, 150))
    expanded = df.loc[df.index.repeat(df.f)].reset_index(drop=True)
    a = oe.teffects(
        data=df,
        y="y",
        treatment="d",
        x=["x1", "x2"],
        method=method,
        weights="f",
        weight_type="fweight",
    )
    b = oe.teffects(data=expanded, y="y", treatment="d", x=["x1", "x2"], method=method)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-11)
    assert (
        a.spec.weights == "f"
        and a.nobs == int(df.f.sum())
        and a.sample_positions == list(range(len(df)))
    )
    assert a.provenance["data_hash"] != b.provenance["data_hash"]


def test_exact_support_and_frequency_budget_fail_without_dropping_rows():
    df = make_data(n=150).assign(cell=lambda d: d.d, f=10000)
    with pytest.raises(AnalysisError) as exc:
        oe.teffects(data=df, y="y", treatment="d", x=["x1"], method="nnmatch", ematch=["cell"])
    assert exc.value.code == "exact_match_support"
    with pytest.raises(AnalysisError) as exc:
        oe.teffects(
            data=df,
            y="y",
            treatment="d",
            x=["x1"],
            method="nnmatch",
            weights="f",
            weight_type="fweight",
        )
    assert exc.value.code == "matching_too_large"


@pytest.mark.parametrize(
    "options,columns",
    [
        ({"method": "aipw", "estimand": "atet"}, {}),
        ({"method": "ra", "estimand": "atet", "tlevel": 2}, {}),
        ({"method": "nnmatch", "matching_vce": "iid"}, {}),
        ({"method": "nnmatch"}, {"ematch": ["x1"]}),
    ],
)
def test_resident_options_do_not_auto_route_to_unsupported_replay(options, columns):
    from openecon.models import ModelSpec
    from openecon.econometrics.streaming_registry import supports_spec

    spec = ModelSpec(
        estimator="teffects",
        outcome="y",
        predictors=["x1"],
        covariance="robust",
        columns={"treatment": "d", **columns},
        options=options,
    )
    assert not supports_spec(spec)
