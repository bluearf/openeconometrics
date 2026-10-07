"""Independent full-risk, density and full-covariance survival target oracles."""

import gc
import numpy as np
import pandas as pd
import pytest
from scipy import stats, special

import openecon as oe
from openecon.models import ResultBundle
from openecon.analysis_contracts import AnalysisError
from test_econ_streaming_cox import data, spec, fit_streaming_cox, Dataset
from test_econ_survival_streg import make_data, logdensity


def collect(dataset):
    return pd.concat(list(dataset.iter_batches(batch_rows=13)))


def baseline_oracle(frame, beta, ties):
    records = []
    x = frame[["x", "z"]].to_numpy()
    risks = np.exp(x @ beta + frame.off.to_numpy())
    for s in sorted(frame.s.unique()):
        cumulative, gradient = 0.0, np.zeros(2)
        for t in sorted(frame.loc[(frame.s == s) & (frame.d == 1), "t"].unique()):
            risk = (frame.s == s) & (frame.t0 < t) & (frame.t >= t)
            dead = (frame.s == s) & (frame.t == t) & (frame.d == 1)
            mass, first = risks[risk].sum(), (risks[risk, None] * x[risk]).sum(0)
            d = int(dead.sum())
            if ties == "efron":
                denominator = mass - np.arange(d) / d * risks[dead].sum()
                numerator = first - np.arange(d)[:, None] / d * (risks[dead, None] * x[dead]).sum(0)
                jump = (1 / denominator).sum()
                gradient -= (numerator / denominator[:, None] ** 2).sum(0)
            else:
                jump = d / mass
                gradient -= d * first / mass**2
            cumulative += jump
            records.append((s, t, cumulative, *gradient))
    return np.array(records)


@pytest.mark.parametrize("weight_type", ["fweight", "iweight", "pweight"])
def test_weighted_baseline_permutation_support_and_cleanup(weight_type, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    d = data(n=90)
    r = fit_streaming_cox(spec("breslow", "robust", weight_type), Dataset.from_frame(d))
    beta = np.array([c.estimate for c in r.coefficients])
    w = d.f.to_numpy() if weight_type == "fweight" else d.w.to_numpy()
    r.coefficients.reverse()
    r.covariance_matrix = np.array(r.covariance_matrix)[::-1, ::-1].tolist()
    baseline = oe.cox_baseline(r, Dataset.from_frame(d), batch_rows=7)
    out = collect(baseline)
    risks = w * np.exp(d[["x", "z"]].to_numpy() @ beta + d.off.to_numpy())
    wanted = []
    for s in sorted(d.s.unique()):
        cumulative = 0.0
        for t in sorted(d.loc[(d.s == s) & (d.d == 1), "t"].unique()):
            mask = (d.s == s) & (d.t0 < t) & (d.t >= t)
            dead = (d.s == s) & (d.t == t) & (d.d == 1)
            cumulative += w[dead].sum() / risks[mask].sum()
            wanted.append(cumulative)
    np.testing.assert_allclose(out.cumulative_hazard, wanted, rtol=1e-10)
    with pytest.raises(AnalysisError, match="stratum"):
        oe.survival_predict(
            r,
            Dataset.from_frame(d.iloc[:1].assign(s=99)),
            target="survival",
            time=1,
            baseline=baseline,
        )
    with pytest.raises(AnalysisError, match="last"):
        oe.survival_predict(
            r, Dataset.from_frame(d.iloc[:1]), target="survival", time=1e6, baseline=baseline
        )
    r.id += "-different"
    with pytest.raises(AnalysisError, match="full baseline"):
        oe.survival_predict(
            r, Dataset.from_frame(d.iloc[:1]), target="survival", time=1, baseline=baseline
        )
    del baseline
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_cox_source_changes_between_replays_and_tvc_path_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    d = data(n=60)
    r = fit_streaming_cox(spec(), Dataset.from_frame(d))
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        current = d.copy()
        if calls > 1:
            current.loc[5, "x"] += 0.1
        yield current

    with pytest.raises(AnalysisError, match="changed"):
        oe.cox_baseline(r, Dataset.from_batches(factory, d.columns), batch_rows=11)
    r.spec.columns["tvc"] = ["x"]
    with pytest.raises(AnalysisError, match="path"):
        oe.cox_baseline(r, Dataset.from_frame(d))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ties", ["breslow", "efron", "exactp"])
def test_full_baseline_restore_risk_masks_gradients_and_disk_cleanup(ties, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data(n=58)
    requested = spec(ties)
    requested.missing = "drop"
    result = fit_streaming_cox(requested, Dataset.from_frame(frame), batch_rows=17)
    result = ResultBundle.model_validate_json(result.model_dump_json())
    baseline = oe.cox_baseline(result, Dataset.from_frame(frame), batch_rows=9)
    actual = collect(baseline)
    wanted = baseline_oracle(frame, np.array([c.estimate for c in result.coefficients]), ties)
    np.testing.assert_allclose(
        actual[["stratum", "time", "cumulative_hazard", "gradient[0]", "gradient[1]"]],
        wanted,
        rtol=2e-10,
        atol=1e-11,
    )
    evaluation = frame.iloc[:5].copy()
    evaluation.index = pd.Index(["same", "same", None, 2, "2"])
    evaluation.loc[evaluation.index.isna(), "x"] = np.nan
    prediction = oe.survival_predict(
        result,
        Dataset.from_frame(evaluation),
        target="survival",
        time=1,
        baseline=baseline,
        interval="mean",
        batch_rows=2,
    )
    output = collect(prediction)
    assert output.index.tolist() == evaluation.index.tolist()
    assert output.iloc[2].isna().all()
    for i in [0, 1, 3, 4]:
        row = evaluation.iloc[i]
        matching = wanted[(wanted[:, 0] == row.s) & (wanted[:, 1] <= 1)]
        h = matching[-1, 2] if len(matching) else 0.0
        dh = matching[-1, 3:] if len(matching) else np.zeros(2)
        x = row[["x", "z"]].to_numpy(dtype=float)
        risk = np.exp(x @ np.array([c.estimate for c in result.coefficients]) + row.off)
        survivor = np.exp(-h * risk)
        jac = -survivor * risk * (dh + h * x)
        assert output.iloc[i].response == pytest.approx(survivor)
        assert output.iloc[i].std_error == pytest.approx(
            np.sqrt(jac @ np.array(result.covariance_matrix) @ jac), rel=1e-9
        )
    assert (
        "counting-process variance excluded"
        in prediction.metadata["analysis"]["response_definition"]
    )
    del prediction, baseline
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_all_failure_times_not_preview_and_source_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data(n=510, entry=False)
    frame.t = np.arange(1.0, len(frame) + 1)
    frame.d = 1.0
    requested = spec()
    requested.columns.pop("strata")
    result = fit_streaming_cox(requested, Dataset.from_frame(frame), batch_rows=87)
    assert result.extra["baseline"]["thinned"]
    output = oe.stcurve(result, Dataset.from_frame(frame), batch_rows=37)
    assert len(collect(output)) == 510
    assert output.metadata["analysis"]["full_step_function"]
    del output
    changed = frame.copy()
    changed.loc[420, "z"] += 0.1
    with pytest.raises(AnalysisError, match="differ"):
        oe.cox_baseline(result, Dataset.from_frame(changed), batch_rows=37)
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.fixture(
    scope="module",
    params=[
        ("exponential", "ph"),
        ("exponential", "aft"),
        ("weibull", "ph"),
        ("weibull", "aft"),
        ("gompertz", "ph"),
        ("lognormal", "aft"),
        ("loglogistic", "aft"),
        ("ggamma", "aft"),
    ],
)
def parametric(request):
    dist, metric = request.param
    d = make_data(n=240)
    r = oe.streg(
        data=d,
        time="t",
        failure="d",
        x=["x"],
        distribution=dist,
        metric=metric,
        ancillary=["z"] if dist not in {"exponential", "ggamma"} else None,
    )
    return r, d


def oracle(r, d, target, t=0.8, p=0.4, theta=None):
    b = {c.term: c.estimate for c in r.coefficients} if theta is None else theta
    dist, metric = r.extra["distribution"], r.extra["metric"]
    mu = b["Intercept"] + b["x"] * d.x.to_numpy()
    ancillary = {
        "weibull": "ln_p",
        "gompertz": "gamma",
        "lognormal": "lnsigma",
        "loglogistic": "lngamma",
        "ggamma": "lnsigma",
    }.get(dist)
    a = (
        None
        if ancillary is None
        else b.get("/" + ancillary, b.get(ancillary + ":Intercept", 0))
        + b.get(ancillary + ":z", 0) * d.z.to_numpy()
    )
    if dist == "ggamma":
        kappa, sigma = b["/kappa"], np.exp(a)
        g = kappa**-2
        u = g * np.exp(kappa * (np.log(t) - mu) / sigma)
        logs = np.log(special.gammaincc(g, u) if kappa > 0 else special.gammainc(g, u))
        logf = (
            np.log(abs(kappa)) - np.log(sigma) - np.log(t) + g * np.log(u) - u - special.gammaln(g)
        )
        if target == "quantile":
            inverse = special.gammaincinv(g, p) if kappa > 0 else special.gammainccinv(g, p)
            return np.exp(mu + sigma / kappa * np.log(inverse / g))
    else:
        logf, logs = logdensity(dist, metric, np.array(t), mu, a)
    if target == "survival":
        return np.exp(logs)
    if target == "hazard":
        return np.exp(logf - logs)
    if target == "cumulative_hazard":
        return -logs
    if dist in {"exponential", "weibull"}:
        shape = 1 if a is None else np.exp(a)
        scale = np.exp(-mu / shape if metric == "ph" else mu)
        return stats.weibull_min.ppf(p, shape, scale=scale)
    if dist == "lognormal":
        return stats.lognorm.ppf(p, np.exp(a), scale=np.exp(mu))
    if dist == "loglogistic":
        return stats.fisk.ppf(p, np.exp(-a), scale=np.exp(mu))
    return np.log1p(a * (-np.log1p(-p)) * np.exp(-mu)) / a


@pytest.mark.parametrize("target", ["survival", "hazard", "cumulative_hazard", "quantile"])
def test_parametric_full_ancillary_values_delta_restore_and_bounded_output(
    parametric, target, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    r, d = parametric
    r = ResultBundle.model_validate_json(r.model_dump_json())
    d = d.iloc[:9].copy()
    kwargs = {"time": 0.8} if target != "quantile" else {"quantile": 0.4}
    disk = oe.survival_predict(
        r, Dataset.from_frame(d), target=target, interval="mean", batch_rows=3, **kwargs
    )
    actual = collect(disk)
    wanted = oracle(r, d, target)
    np.testing.assert_allclose(actual.response, wanted, rtol=1e-8, atol=1e-9)
    b = {c.term: c.estimate for c in r.coefficients}
    jac = []
    for term in b:
        plus, minus = dict(b), dict(b)
        plus[term] += 1e-4
        minus[term] -= 1e-4
        jac.append((oracle(r, d, target, theta=plus) - oracle(r, d, target, theta=minus)) / 2e-4)
    jac = np.array(jac).T
    expected = np.sqrt(np.einsum("nk,kl,nl->n", jac, np.array(r.covariance_matrix), jac))
    np.testing.assert_allclose(actual.std_error, expected, rtol=2e-5, atol=2e-8)
    del disk
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_undefined_targets_and_missing_state(parametric):
    r, d = parametric
    with pytest.raises(AnalysisError):
        oe.survival_predict(r, d, target="survival", time=0)
    with pytest.raises(AnalysisError):
        oe.survival_predict(r, d, target="quantile", quantile=1)
    damaged = ResultBundle.model_validate_json(r.model_dump_json())
    damaged.extra.pop("metric")
    with pytest.raises(AnalysisError):
        oe.survival_predict(damaged, d, target="survival", time=1)
    if r.extra["distribution"] == "loglogistic":
        r = ResultBundle.model_validate_json(r.model_dump_json())
        for c in r.coefficients:
            if c.term in {"/lngamma", "lngamma:Intercept"}:
                c.estimate = 0.1
            if c.term == "lngamma:z":
                c.estimate = 0.0
        with pytest.raises(AnalysisError, match="mean exists"):
            oe.survival_predict(r, d, target="mean")
