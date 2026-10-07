"""Independent equation/moment references and published research/data examples."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
import statsmodels.api as sm
from statsmodels.stats.oaxaca import OaxacaBlinder

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def fixture(n=180):
    rng = np.random.default_rng(184)
    a, w, control = rng.normal(size=(3, n))
    m1 = 0.5 * a + 0.2 * a * w + 0.3 * control + rng.normal(size=n)
    m2 = 0.2 * a + 0.3 * m1 + 0.4 * control + rng.normal(size=n)
    y = (
        0.2 * a
        + 0.1 * a * w
        + 0.7 * m1
        + 0.1 * m1 * w
        + 0.3 * m2
        + 0.2 * control
        + rng.normal(size=n)
    )
    return pd.DataFrame(
        dict(a=a, w=w, c=control, m1=m1, m2=m2, y=y, g=np.arange(n) % 12, weight=1 + rng.random(n))
    )


@pytest.mark.parametrize(
    "serial,moderator,cluster,weights",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, True, False),
        (True, False, False, True),
    ],
)
def test_linear_parallel_serial_moderated_joint_delta(serial, moderator, cluster, weights):
    df = fixture()
    a, m1, m2, y, c, w = [df[col].to_numpy() for col in ["a", "m1", "m2", "y", "c", "w"]]
    one = np.ones(len(df))
    x1 = np.column_stack([one, a, c, *([w, a * w] if moderator else [])])
    x2 = np.column_stack([one, a, *([m1] if serial else []), c, *([w, a * w] if moderator else [])])
    xy = np.column_stack([one, a, m1, m2, c, *([w, a * w, m1 * w, m2 * w] if moderator else [])])
    weight = df.weight.to_numpy() if weights else one
    coeffs, impacts = [], []
    for x, target in [(x1, m1), (x2, m2), (xy, y)]:
        beta = np.linalg.solve(x.T @ (weight[:, None] * x), x.T @ (weight * target))
        bread = np.linalg.inv(x.T @ (weight[:, None] * x))
        impact = (x * (weight * (target - x @ beta))[:, None]) @ bread
        if not cluster:
            impact *= np.sqrt(len(df) / (len(df) - x.shape[1]))
        coeffs.append(beta)
        impacts.append(impact)
    combined = np.column_stack(impacts)
    if cluster:
        sums = np.vstack([combined[df.g == i].sum(0) for i in range(12)])
        joint = sums.T @ sums * 12 / 11 * (len(df) - 1) / (len(df) - xy.shape[1])
    else:
        joint = combined.T @ combined
    initial = np.concatenate(coeffs)
    sizes = [len(b) for b in coeffs]
    splits = np.cumsum(sizes)[:-1]

    def targets(theta, value):
        b1, b2, by = np.split(theta, splits)
        aa1 = b1[1] + (value * b1[-1] if moderator else 0)
        aa2 = b2[1] + (value * b2[-1] if moderator else 0)
        bb1 = by[2] + (value * by[-2] if moderator else 0)
        bb2 = by[3] + (value * by[-1] if moderator else 0)
        direct = by[1] + (value * by[-3] if moderator else 0)
        paths = [aa1 * bb1, aa2 * bb2] + ([aa1 * b2[2] * bb2] if serial else [])
        return np.r_[direct, sum(paths), direct + sum(paths), paths]

    values = [-1.0, 1.0] if moderator else [0.0]

    def all_targets(theta):
        return np.concatenate([targets(theta, v) for v in values])

    eps = 1e-5
    jac = np.column_stack(
        [
            (
                all_targets(initial + eps * np.eye(len(initial))[i])
                - all_targets(initial - eps * np.eye(len(initial))[i])
            )
            / (2 * eps)
            for i in range(len(initial))
        ]
    )
    result = oe.mediation(
        data=df,
        y="y",
        x=["a"],
        mediators=["m1", "m2"],
        controls=["c"],
        serial=serial,
        moderator="w" if moderator else None,
        at=values if moderator else None,
        covariance="cluster" if cluster else "HC1",
        cluster="g" if cluster else None,
        weights="weight" if weights else None,
    )
    np.testing.assert_allclose(
        [v.estimate for v in result.coefficients], all_targets(initial), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(result.covariance_matrix, jac @ joint @ jac.T, rtol=2e-6, atol=2e-8)
    assert result.extra["interpretation"] == "associational"
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra
    assert "\\toprule" in result.to_latex()


def test_published_framing_probit_natural_effects_exact_gaussian_reference():
    df = pd.read_csv(Path(__file__).parent / "fixtures/mediation-framing.csv")
    encoded = pd.get_dummies(df[["age", "educ", "gender", "income"]], drop_first=True, dtype=float)
    controls = list(encoded.columns)
    df = pd.concat([df.drop(columns=["age", "educ", "gender", "income"]), encoded], axis=1)
    result = oe.mediation(
        data=df,
        y="cong_mesg",
        x=["treat"],
        mediators=["emo"],
        controls=controls,
        outcome_model="probit",
    )
    xm = np.column_stack([np.ones(len(df)), df.treat, df[controls]])
    xy = np.column_stack([np.ones(len(df)), df.treat, df.emo, df[controls]])
    bm = np.linalg.lstsq(xm, df.emo, rcond=None)[0]
    by = sm.Probit(df.cong_mesg, xy).fit(disp=0, tol=1e-12).params.to_numpy()
    variance = np.mean((df.emo - xm @ bm) ** 2)

    def probability(a, mediated_a):
        xc = xm.copy()
        xc[:, 1] = mediated_a
        yc = xy.copy()
        yc[:, 1] = a
        yc[:, 2] = xc @ bm
        return np.mean(ndtr((yc @ by) / np.sqrt(1 + by[2] ** 2 * variance)))

    base, cross, total = probability(0, 0), probability(1, 0), probability(1, 1)
    expected = [cross - base, total - cross, total - base, total - cross]
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], expected, rtol=2e-6, atol=2e-8
    )
    assert result.extra["integration_draws"] == 0
    assert result.nobs == len(df)


def test_logit_integrated_probability_effects_and_seed_replay():
    df = fixture(100)
    rng = np.random.default_rng(99)
    df.y = rng.binomial(1, 1 / (1 + np.exp(-(0.2 * df.a + 0.7 * df.m1))))
    result = oe.mediation(
        data=df,
        y="y",
        x=["a"],
        mediators=["m1"],
        controls=["c"],
        outcome_model="logit",
        integration_draws=20,
        seed=28,
    )
    again = oe.mediation(
        data=df,
        y="y",
        x=["a"],
        mediators=["m1"],
        controls=["c"],
        outcome_model="logit",
        integration_draws=20,
        seed=28,
    )
    assert (
        result.coefficients == again.coefficients
        and result.covariance_matrix == again.covariance_matrix
    )
    assert result.coefficients[2].estimate == pytest.approx(
        result.coefficients[0].estimate + result.coefficients[1].estimate
    )
    assert "probability scale" in result.extra["target"]


@pytest.mark.parametrize("reference", ["pooled", "neumark", "a", "b", "reimers", "cotton"])
def test_published_ccard_two_fold_independent_reference(reference):
    df = pd.read_csv(Path(__file__).parent / "fixtures/oaxaca-ccard.csv")
    x = ["AGE", "INCOME", "INCOMESQ"]
    result = oe.oaxaca(
        data=df,
        y="AVGEXP",
        x=x,
        group="OWNRENT",
        groups=[1.0, 0.0],
        reference=reference,
        reps=60,
        seed=89,
    )
    oracle = OaxacaBlinder(df.AVGEXP, df[[*x, "OWNRENT"]], "OWNRENT", hasconst=False, swap=True)
    kind = {"a": "self_submitted", "b": "self_submitted"}.get(reference, reference)
    if reference in ("a", "b", "cotton"):
        ra, rb = df[df.OWNRENT == 1.0], df[df.OWNRENT == 0.0]
        ba = np.linalg.lstsq(sm.add_constant(ra[x]), ra.AVGEXP, rcond=None)[0]
        bb = np.linalg.lstsq(sm.add_constant(rb[x]), rb.AVGEXP, rcond=None)[0]
        ma, mb = np.r_[1, ra[x].mean()], np.r_[1, rb[x].mean()]
        benchmark = (
            ba
            if reference == "a"
            else bb
            if reference == "b"
            else len(ra) / (len(ra) + len(rb)) * ba + len(rb) / (len(ra) + len(rb)) * bb
        )
        ref = [
            (ma - mb) @ benchmark,
            ma @ (ba - benchmark) + mb @ (benchmark - bb),
            ma @ ba - mb @ bb,
        ]
    else:
        # The independent library spells this option 'nuemark'. Its Cotton
        # branch keeps unswapped group counts, so Cotton is checked above by
        # its published size-weighting formula in declared [a,b] order.
        output = oracle.two_fold(two_fold_type="nuemark" if kind == "neumark" else kind).params
        ref = [output[1], output[0], output[2]]
    np.testing.assert_allclose([c.estimate for c in result.coefficients], ref, atol=2e-8)
    if reference == "pooled":
        np.testing.assert_allclose(ref, [130.80954, 27.94091, 158.75044], atol=2e-5)
    details = oe.oaxaca_details(result)
    for c in result.coefficients[:-1]:
        assert details[details.component == c.term].estimate.sum() == pytest.approx(c.estimate)


def test_three_fold_published_ccard_and_reproducible_uncertainty():
    df = pd.read_csv(Path(__file__).parent / "fixtures/oaxaca-ccard.csv")
    result = oe.oaxaca(
        data=df,
        y="AVGEXP",
        x=["AGE", "INCOME", "INCOMESQ"],
        group="OWNRENT",
        groups=[1.0, 0.0],
        fold=3,
        reference="b",
        reps=80,
        seed=82,
    )
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients],
        [321.74824, 75.45371, -238.45151, 158.75044],
        atol=2e-5,
    )
    again = oe.oaxaca(
        data=df,
        y="AVGEXP",
        x=["AGE", "INCOME", "INCOMESQ"],
        group="OWNRENT",
        groups=[1.0, 0.0],
        fold=3,
        reference="b",
        reps=80,
        seed=82,
    )
    assert (
        result.coefficients == again.coefficients
        and result.covariance_matrix == again.covariance_matrix
    )
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra
    assert "\\toprule" in oe.oaxaca_details(result).to_latex()


def test_categorical_normalization_weights_and_reference_invariance():
    df = fixture(260)
    df["group"] = np.arange(len(df)) % 2
    df["category"] = pd.Categorical(np.arange(len(df)) % 3, categories=[0, 1, 2])
    a = oe.oaxaca(
        data=df,
        y="y",
        x=["a", "category"],
        group="group",
        categorical=["category"],
        weights="weight",
        reps=60,
        seed=31,
    )
    df.category = df.category.cat.reorder_categories([2, 0, 1])
    b = oe.oaxaca(
        data=df,
        y="y",
        x=["a", "category"],
        group="group",
        categorical=["category"],
        weights="weight",
        reps=60,
        seed=31,
    )
    np.testing.assert_allclose(
        [c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], atol=2e-8
    )
    da, db = (
        oe.oaxaca_details(a).sort_values(["component", "term"]),
        oe.oaxaca_details(b).sort_values(["component", "term"]),
    )
    np.testing.assert_allclose(da.estimate, db.estimate, atol=2e-8)
    np.testing.assert_allclose(da.std_error, db.std_error, atol=2e-8)


def test_explicit_causal_and_nonlinear_target_guards():
    df = fixture()
    with pytest.raises(AnalysisError) as error:
        oe.mediation(data=df, y="y", x=["a"], mediators=["m1"], interpretation="causal")
    assert error.value.code == "causal_assumptions_required"
    with pytest.raises(AnalysisError) as error:
        oe.mediation(
            data=df, y="y", x=["a"], mediators=["m1", "m2"], serial=True, outcome_model="logit"
        )
    assert error.value.code == "unsupported_target"


def test_probit_full_joint_nuisance_covariance_and_delta_independent_equations():
    from scipy.special import expit

    df = fixture(140)
    df.y = np.random.default_rng(73).binomial(1, expit(0.2 * df.a + 0.5 * df.m1 + 0.2 * df.c))
    xm = np.column_stack([np.ones(len(df)), df.a, df.c])
    xy = np.column_stack([np.ones(len(df)), df.a, df.m1, df.c])
    bm = np.linalg.lstsq(xm, df.m1, rcond=None)[0]
    residual = df.m1.to_numpy() - xm @ bm
    binary = sm.Probit(df.y, xy).fit(disp=0, tol=1e-12)
    by = np.asarray(binary.params)
    variance = np.mean(residual**2)
    im = (
        (xm * residual[:, None])
        @ np.linalg.inv(xm.T @ xm)
        * np.sqrt(len(df) / (len(df) - xm.shape[1]))
    )
    iy = (
        binary.model.score_obs(by)
        @ np.linalg.inv(-binary.model.hessian(by))
        * np.sqrt(len(df) / (len(df) - xy.shape[1]))
    )
    iv = (residual**2 - variance)[:, None] / len(df) * np.sqrt(len(df) / (len(df) - 1))
    influence = np.column_stack([im, iy, iv])
    joint = influence.T @ influence
    theta = np.r_[bm, by, variance]

    def target(t):
        b, m, v = t[:3], t[3:7], t[-1]
        probabilities = []
        for exposure, mediated in [(0, 0), (1, 0), (1, 1)]:
            xx = xm.copy()
            xx[:, 1] = mediated
            yy = xy.copy()
            yy[:, 1] = exposure
            yy[:, 2] = xx @ b
            probabilities.append(np.mean(ndtr((yy @ m) / np.sqrt(1 + m[2] ** 2 * v))))
        a, b, c = probabilities
        return np.array([b - a, c - b, c - a, c - b])

    step = 1e-5
    jac = np.column_stack(
        [
            (target(theta + step * d) - target(theta - step * d)) / (2 * step)
            for d in np.eye(len(theta))
        ]
    )
    result = oe.mediation(
        data=df, y="y", x=["a"], mediators=["m1"], controls=["c"], outcome_model="probit"
    )
    np.testing.assert_allclose(
        result.extra["raw_parameter_covariance"], joint, rtol=2e-6, atol=2e-8
    )
    np.testing.assert_allclose(result.covariance_matrix, jac @ joint @ jac.T, rtol=2e-6, atol=2e-8)
    assert result.extra["integration_method"] == "analytic Gaussian-probit mixture"
