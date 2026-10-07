"""Independent conditional sums, Gaussian integrals and joint FE/BLUP oracles."""

import copy
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import expit, logsumexp
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_fprime

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle


def source(frame):
    return Dataset.from_batches(
        lambda: (frame.iloc[i : i + 3].copy() for i in range(0, len(frame), 3)),
        list(frame),
        row_count=len(frame),
    )


def rows(output):
    return pd.concat(list(output.iter_batches(batch_rows=5)))


def saved(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


def beta(result):
    return np.array([c.estimate for c in result.coefficients])


@pytest.fixture(scope="module")
def conditional():
    rng = np.random.default_rng(452)
    d = pd.DataFrame(
        {
            "g": np.repeat(np.arange(25), 6),
            "x": rng.normal(size=150),
            "z": rng.normal(size=150),
            "off": rng.normal(size=150) * 0.2,
        }
    )
    y = []
    for _, block in d.groupby("g"):
        m = 2 if block.g.iloc[0] % 2 else 4
        subsets = list(combinations(range(6), m))
        scores = np.array(
            [
                (block.x.to_numpy()[list(s)] * 0.6 + block.off.to_numpy()[list(s)]).sum()
                for s in subsets
            ]
        )
        selected = subsets[rng.choice(len(subsets), p=np.exp(scores - logsumexp(scores)))]
        y.extend([int(i in selected) for i in range(6)])
    d["y"] = y
    d.index = pd.MultiIndex.from_arrays(
        [np.tile(["a", "a", "c"], 50), np.arange(150) % 7], names=["label", "position"]
    )
    return d, saved(oe.clogit(data=d, y="y", x=["x", "z"], group="g", offset="off"))


def conditional_oracle(block, b):
    x = block[["x", "z"]].to_numpy()
    subsets = np.array(
        [
            np.isin(np.arange(len(block)), s)
            for s in combinations(range(len(block)), int(block.y.sum()))
        ],
        dtype=float,
    )
    score = subsets @ (x @ b + block.off.to_numpy())
    return np.exp(score - logsumexp(score)) @ subsets


def test_complete_groups_cross_batches_exact_success_sums_full_covariance_and_index(conditional):
    d, r = conditional
    d = d.sample(frac=1, random_state=8)
    before = copy.deepcopy(r.model_dump())
    actual = rows(oe.predict(r, source(d), interval="mean", batch_rows=4))
    b = beta(r)
    expected = np.zeros(len(d))
    se = np.zeros(len(d))
    for indices in d.groupby("g", sort=False).indices.values():
        block = d.iloc[indices]

        def fun(point):
            return conditional_oracle(block, point)

        expected[indices] = fun(b)
        jac = approx_fprime(b, fun, epsilon=1e-6, centered=True)
        se[indices] = np.sqrt(np.einsum("ij,jk,ik->i", jac, r.covariance_matrix, jac))
        assert_allclose(actual.response.iloc[indices].sum(), block.y.sum(), atol=2e-13)
    assert actual.index.equals(d.index)
    assert_allclose(actual.response, expected, atol=2e-13)
    assert_allclose(actual.std_error, se, rtol=2e-8, atol=2e-10)
    assert r.model_dump() == before
    fitted = saved(oe.clogit(data=source(d), y="y", x=["x", "z"], group="g", offset="off"))
    assert_allclose(rows(oe.predict(fitted, source(d), batch_rows=7)).response, expected, atol=2e-8)


def test_conditional_incomplete_changed_unknown_historical_and_argument_contracts(conditional):
    d, r = conditional
    for changed, code in [
        (d.iloc[:-1], "incomplete_group"),
        (d.assign(x=d.x + 0.01), "incomplete_group"),
        (d.assign(g=d.g + 100), "unknown_group"),
    ]:
        with pytest.raises(AnalysisError) as error:
            oe.predict(r, source(changed), batch_rows=4)
        assert error.value.code == code
    historical = r.model_copy(deep=True)
    historical.extra.pop("group_state")
    with pytest.raises(AnalysisError, match="historical"):
        oe.predict(historical, source(d))
    with pytest.raises(AnalysisError):
        oe.predict(r, d, random_effects={"_cons": 0})
    with pytest.raises(AnalysisError):
        oe.predict(None, d)


@pytest.fixture(scope="module")
def continuous():
    rng = np.random.default_rng(140)
    g = np.repeat(np.arange(25), 6)
    x = rng.normal(size=len(g))
    y = 1 + 0.7 * x + rng.normal(size=25)[g] + rng.normal(size=len(g)) * 0.6
    d = pd.DataFrame({"g": g, "x": x, "y": y, "t": np.tile(np.arange(6), 25)})
    d.index = pd.Index(np.arange(len(d)) % 9, name="duplicate")
    return d


@pytest.mark.parametrize("name", ["areg", "reghdfe", "xtreg"])
@pytest.mark.parametrize("streamed", [False, True])
def test_saved_fixed_response_and_joint_ols_information(continuous, name, streamed):
    d = continuous
    kwargs = (
        {"panel": "g", "time": "t"}
        if name == "xtreg"
        else {"absorb": "g" if name == "areg" else ["g"]}
    )
    r = saved(getattr(oe, name)(data=source(d) if streamed else d, y="y", x=["x"], **kwargs))
    actual = rows(oe.predict(r, source(d), interval="mean", batch_rows=4))
    f = pd.get_dummies(d.g, dtype=float).to_numpy()
    x = np.column_stack((d.x, f))
    b = np.linalg.lstsq(x, d.y, rcond=None)[0]
    variance = np.linalg.inv(x.T @ x) * np.sum((d.y - x @ b) ** 2) / (len(d) - x.shape[1])
    se = np.sqrt(np.einsum("ij,jk,ik->i", x, variance, x))
    assert actual.index.equals(d.index)
    assert_allclose(actual.response, x @ b, atol=3e-8)
    assert_allclose(actual.std_error, se, rtol=3e-8, atol=2e-10)
    with pytest.raises(AnalysisError, match="absent"):
        oe.predict(r, d.assign(g=100))
    historical = r.model_copy(deep=True)
    historical.extra.pop("group_state")
    with pytest.raises(AnalysisError):
        oe.predict(historical, d)


def test_ppml_saved_fe_sum_full_joint_information_resident_and_replay():
    rng = np.random.default_rng(64)
    g = np.repeat(np.arange(15), 10)
    x = rng.normal(size=len(g))
    off = rng.normal(size=len(g)) * 0.2
    d = pd.DataFrame(
        {
            "g": g,
            "x": x,
            "off": off,
            "y": rng.poisson(np.exp(0.8 + 0.25 * x + rng.normal(size=15)[g] * 0.3 + off)),
        }
    )
    for input_data in [d, source(d)]:
        r = saved(
            oe.ppmlhdfe(
                data=input_data, y="y", x=["x"], absorb=["g"], offset="off", covariance="nonrobust"
            )
        )
        actual = rows(oe.predict(r, source(d), interval="mean", batch_rows=7))
        xfull = np.column_stack((d.x, pd.get_dummies(d.g, dtype=float)))
        mu = actual.response.to_numpy()
        cov = np.linalg.inv(xfull.T @ (mu[:, None] * xfull))
        expected = mu * np.sqrt(np.einsum("ij,jk,ik->i", xfull, cov, xfull))
        assert_allclose(actual.std_error, expected, rtol=1e-7, atol=2e-8)
        assert_allclose(d.groupby("g").y.sum(), pd.Series(mu).groupby(d.g).sum(), atol=2e-7)
        assert np.max(np.abs(mu - np.exp(d.x * beta(r)[0] + off))) > 0.1


@pytest.fixture(scope="module")
def binary():
    rng = np.random.default_rng(653)
    g = np.repeat(np.arange(40), 6)
    x = rng.normal(size=len(g))
    u = rng.normal(size=40) * 1.1
    d = pd.DataFrame({"g": g, "x": x, "y": rng.binomial(1, expit(0.3 + 0.6 * x + u[g]))})
    r = saved(oe.melogit(data=d, y="y", x=["x"], group="g"))
    return d, r


def normal_oracle(d, b, posterior=False):
    eta = np.column_stack((np.ones(len(d)), d.x)) @ b[:2]
    sd = np.sqrt(b[-1])

    def mass(v):
        p = expit(eta + sd * v)
        return (
            norm.pdf(v) * np.exp((d.y * np.log(p) + (1 - d.y) * np.log1p(-p)).sum())
            if posterior
            else norm.pdf(v)
        )

    denominator = quad(mass, -10, 10, epsabs=1e-12)[0]
    return np.array(
        [
            quad(lambda v: expit(e + sd * v) * mass(v), -10, 10, epsabs=1e-12)[0] / denominator
            for e in eta
        ]
    )


def test_normal_population_explicit_and_complete_posterior_full_variance_delta(binary):
    d, r = binary
    b = beta(r)
    # Integration includes saved variance, not just the fixed mean coefficients.
    pop = oe.predict(r, d.head(7), interval="mean")

    def fun(point):
        return normal_oracle(d.head(7), point)

    jac = approx_fprime(b, fun, epsilon=1e-5, centered=True)
    assert_allclose(pop.response, fun(b), atol=1e-9)
    assert_allclose(
        pop.std_error, np.sqrt(np.einsum("ij,jk,ik->i", jac, r.covariance_matrix, jac)), rtol=2e-7
    )
    explicit = oe.predict(
        r, d.head(7), target="conditional", random_effects={"_cons": 0.7}, interval="mean"
    )
    assert_allclose(explicit.response, expit(b[0] + b[1] * d.x.head(7) + 0.7), atol=1e-14)
    block = d[d.g == 2]
    posterior = rows(
        oe.predict(r, source(block), target="posterior", interval="mean", batch_rows=2)
    )

    def fun(point):
        return normal_oracle(block, point, True)

    jac = approx_fprime(b, fun, epsilon=1e-5, centered=True)
    assert_allclose(posterior.response, fun(b), atol=1e-8)
    assert_allclose(
        posterior.std_error,
        np.sqrt(np.einsum("ij,jk,ik->i", jac, r.covariance_matrix, jac)),
        rtol=2e-6,
        atol=2e-8,
    )
    assert np.max(abs(posterior.response.to_numpy() - normal_oracle(block, b))) > 0.01
    dense = oe.margins(r, "x", data=d)
    batched = oe.margins(r, "x", data=source(d), batch_rows=3)
    assert_allclose(
        batched[["estimate", "std_error"]], dense[["estimate", "std_error"]], atol=2e-12
    )


def test_dataset_blups_independent_dense_and_index(continuous):
    d = continuous
    for input_data in [d, source(d)]:
        r = saved(oe.mixed(data=input_data, y="y", x=["x"], group="g", method="ml"))
        actual = rows(oe.mixed_predict(r, source(d), kind="fitted", batch_rows=4))
        coefficients = {c.term: c.estimate for c in r.coefficients}
        var = coefficients["/var(_cons[g])"]
        residual = coefficients["/var(Residual)"]
        x = np.column_stack((np.ones(len(d)), d.x))
        mean = x @ beta(r)[:2]
        expected = mean.copy()
        for indices in d.groupby("g").indices.values():
            v = np.eye(len(indices)) * residual + var
            u = var * np.linalg.solve(v, d.y.to_numpy()[indices] - mean[indices]).sum()
            expected[indices] += u
        assert actual.index.equals(d.index)
        assert_allclose(actual.fitted, expected, atol=3e-9)
        effects = rows(oe.mixed_predict(r, source(d), kind="reffects", batch_rows=4))
        assert len(effects) == 25
        assert effects.group_identity.nunique() == 25


def test_nested_blups_keep_reused_child_identity():
    rng = np.random.default_rng(75)
    top = np.repeat(np.arange(8), 12)
    low = np.tile(np.repeat(np.arange(3), 4), 8)
    x = rng.normal(size=len(top))
    d = pd.DataFrame(
        {
            "top": top,
            "low": low,
            "x": x,
            "y": 1
            + 0.5 * x
            + rng.normal(size=8)[top] * 1.5
            + rng.normal(size=24)[3 * top + low]
            + rng.normal(size=len(top)) * 0.5,
        }
    )
    r = saved(oe.mixed(data=d, y="y", x=["x"], group=["top", "low"], method="ml"))
    actual = rows(oe.mixed_predict(r, source(d), kind="fitted", batch_rows=5))
    resident = oe.mixed_predict(r, d, kind="fitted")
    assert_allclose(actual.fitted, resident.fitted, atol=2e-12)
    effects = rows(oe.mixed_predict(r, source(d), kind="reffects", batch_rows=5))
    assert len(effects) == 32
    assert effects.top_group_identity.nunique() == 8


def test_owned_state_export_checksum_budgets_source_change_and_cleanup(
    conditional, tmp_path, monkeypatch
):
    from openecon.econometrics.postest import group_state

    d, r = conditional
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(scratch))
    monkeypatch.setenv("OPENECON_GROUP_STATE_DIRECTORY", str(tmp_path / "state"))
    monkeypatch.setattr(group_state, "_INLINE", 1)
    disk = saved(oe.clogit(data=d, y="y", x=["x", "z"], group="g", offset="off"))
    assert not list(scratch.iterdir())

    moved = oe.export_group_state(disk, tmp_path / "portable.sqlite3")
    assert_allclose(
        rows(oe.predict(moved, source(d))).response, oe.predict(r, d).response, atol=1e-12
    )
    with open(moved.extra["group_state"]["path"], "ab") as out:
        out.write(b"changed")
    with pytest.raises(AnalysisError, match="checksum"):
        oe.predict(moved, source(d))
    for kwargs, code in [
        ({"max_group_rows": 5}, "group_prediction_budget"),
        ({"max_disk_bytes": 1}, "prediction_disk_budget"),
    ]:
        with pytest.raises(AnalysisError) as error:
            oe.predict(r, source(d), **kwargs)
        assert error.value.code == code
        assert not list(scratch.iterdir())
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        current = d.copy()
        if calls > 1:
            current.x += 0.01
        yield current

    changed = Dataset.from_batches(factory, list(d), row_count=len(d))
    with pytest.raises(AnalysisError, match="changed"):
        oe.predict(r, changed)
    assert not list(scratch.iterdir())


def test_missing_row_index_and_initialization_failure_cleanup(conditional, tmp_path, monkeypatch):
    import sqlite3

    d, _ = conditional
    extra = d.iloc[[0]].copy()
    extra["x"] = np.nan
    combined = pd.concat([extra, d])
    r = oe.clogit(data=combined, y="y", x=["x", "z"], group="g", offset="off", missing="drop")
    actual = rows(oe.predict(r, source(combined), interval="mean", batch_rows=3))
    assert actual.iloc[0].isna().all()
    assert actual.index.equals(combined.index)
    assert len(actual) == len(combined)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(scratch))

    def failed_connect(*args, **kwargs):
        raise OSError("Injected owned SQLite setup failure")

    monkeypatch.setattr(sqlite3, "connect", failed_connect)
    with pytest.raises(OSError, match="Injected"):
        oe.predict(r, source(combined))
    assert not list(scratch.iterdir())
    with pytest.raises(OSError, match="Injected"):
        oe.clogit(data=combined, y="y", x=["x", "z"], group="g", offset="off", missing="drop")
    assert not list(scratch.iterdir())


def test_panel_re_log_variance_population_has_separate_target(binary):
    d, _ = binary
    r = saved(oe.xtlogit(data=source(d), y="y", x=["x"], panel="g", model="re"))
    b = beta(r)

    def fun(point):
        reported = point.copy()
        reported[-1] = np.exp(point[-1])
        return normal_oracle(d.head(7), reported)

    jac = approx_fprime(b, fun, epsilon=1e-5, centered=True)
    actual = oe.predict(r, d.head(7), interval="mean")
    assert_allclose(actual.response, fun(b), atol=1e-9)
    assert_allclose(
        actual.std_error,
        np.sqrt(np.einsum("ij,jk,ik->i", jac, r.covariance_matrix, jac)),
        rtol=2e-7,
    )
    changed = r.model_copy(deep=True)
    changed.extra["model"] = "pa"
    with pytest.raises(AnalysisError, match="disagree"):
        oe.predict(changed, d.head(7))
