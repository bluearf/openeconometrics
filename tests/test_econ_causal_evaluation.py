"""Full joint nuisance oracle, identified effect and safe explicit GMM targets."""

import gc
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import openecon as oe
from openecon.models import ResultBundle
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from test_econ_teffects import make_data, oracle


def collect(source):
    return pd.concat(list(source.iter_batches(batch_rows=17)))


@pytest.mark.parametrize("method", ["ra", "ipwra", "aipw"])
def test_resident_and_physical_saved_categorical_nuisance_standardization(method, tmp_path):
    from test_econ_streaming_teffects import specification
    frame = make_data(n=250)
    frame["cat"] = pd.Categorical(np.arange(len(frame)) % 3, categories=[0, 1, 2, 3])
    path = tmp_path / "causal.parquet"
    frame.to_parquet(path, index=False)
    parsed = pd.read_parquet(path)
    spec = specification(method, mixed=True)
    reference = registry.load_entry(registry.get("teffects"))(spec, parsed)
    physical = oe.fit(spec, data=oe.scan(path))
    evaluation = parsed.dropna().head(17)
    outputs = []
    for result in (reference, physical):
        restored = ResultBundle.model_validate_json(result.model_dump_json())
        assert "cat" in restored.provenance["categorical_encoding"]
        output = oe.causal_evaluate(restored, oe.Dataset.from_frame(evaluation),
            target="standardized_outcome", treatment="1", population="fixed_evaluation")
        try:
            outputs.append(collect(output).select_dtypes("number"))
        finally:
            output.close()
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_full_joint_nuisance_covariance_and_potential_outcome_identity(
    streamed, method, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    d = make_data(n=250)
    from test_econ_streaming_teffects import specification, fit_streaming

    spec = specification(method)
    r = (
        fit_streaming(spec, oe.Dataset.from_frame(d), batch_rows=19)
        if streamed
        else registry.load_entry(registry.get("teffects"))(spec, d)
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    est, se, theta, cov = oracle(d, method)
    record = r.extra["evaluation_state"]
    np.testing.assert_allclose(record["parameters"], theta, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(record["covariance"], cov, rtol=3e-4, atol=2e-7)
    output = oe.causal_evaluate(r, target="population_effect", treatment="1")
    actual = collect(output).iloc[0]
    assert actual.estimate == pytest.approx(est[0], rel=2e-6)
    assert actual.std_error == pytest.approx(se[0], rel=2e-5)
    pom = collect(oe.causal_evaluate(r, target="population_potential_outcome", treatment="1")).iloc[
        0
    ]
    gradient = np.ones(2)
    assert pom.estimate == pytest.approx(sum(est))
    assert pom.std_error == pytest.approx(
        np.sqrt(gradient @ np.array(r.covariance_matrix) @ gradient), rel=1e-8
    )
    assert output.metadata["analysis"]["population"] == "estimation"
    del output
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("omodel", ["linear", "logit", "probit", "poisson"])
@pytest.mark.parametrize("rows", [1, 41])
def test_fixed_population_global_standardization_full_delta_and_missing_alignment(omodel, rows):
    d = make_data(n=320)
    y = "yb" if omodel in {"probit", "logit"} else "yc" if omodel == "poisson" else "y"
    r = oe.teffects(
        data=oe.Dataset.from_frame(d),
        y=y,
        treatment="d",
        x=["x1", "x2"],
        method="aipw",
        omodel=omodel,
        missing="drop",
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    evaluation = d.iloc[:13].copy()
    evaluation.index = pd.Index(["same", "same", None, 3, "3", *range(8)])
    evaluation.iloc[2, evaluation.columns.get_loc("x1")] = np.nan
    out = collect(
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(evaluation),
            target="outcome_mean",
            population="fixed_evaluation",
            treatment="1",
            batch_rows=rows,
        )
    )
    assert out.index.tolist() == evaluation.index.tolist()
    assert out.iloc[2].isna().all()
    record = r.extra["evaluation_state"]
    beta = np.array(record["parameters"])
    cov = np.array(record["covariance"])
    positions = [record["terms"].index("OME1:" + term) for term in ["Intercept", "x1", "x2"]]
    x = np.c_[np.ones(12), evaluation.dropna(subset=["x1", "x2"])[["x1", "x2"]].to_numpy()]
    eta = x @ beta[positions]
    mean = {
        "linear": lambda e: e,
        "logit": lambda e: stats.logistic.cdf(e),
        "probit": stats.norm.cdf,
        "poisson": np.exp,
    }[omodel](eta)
    first = {
        "linear": lambda e: np.ones_like(e),
        "logit": stats.logistic.pdf,
        "probit": stats.norm.pdf,
        "poisson": np.exp,
    }[omodel](eta)
    np.testing.assert_allclose(out.dropna().response, mean, rtol=1e-9)
    jac = np.zeros((len(mean), len(beta)))
    jac[:, positions] = first[:, None] * x
    wanted = np.sqrt(np.einsum("nk,kl,nl->n", jac, cov, jac))
    np.testing.assert_allclose(out.dropna().std_error, wanted, rtol=1e-9)
    population = oe.causal_evaluate(
        r,
        oe.Dataset.from_frame(evaluation),
        target="standardized_outcome",
        population="fixed_evaluation",
        treatment="1",
        batch_rows=rows,
    )
    actual = collect(population).iloc[0]
    assert actual.estimate == pytest.approx(mean.mean(), rel=1e-9)
    assert actual.std_error == pytest.approx(np.sqrt(jac.mean(0) @ cov @ jac.mean(0)), rel=1e-9)
    assert population.metadata["analysis"]["source"]["full_source_collected"] is False


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("fuzzy", [False, True])
def test_rd_cutoff_side_effect_uncertainty_and_local_support(streamed, fuzzy):
    from test_econ_streaming_rd import fixture, spec, fit_streaming

    d = fixture(380)
    s = spec(fuzzy=fuzzy, h=[0.65, 0.7], b=[0.8, 0.9], vce="hc1")
    r = (
        fit_streaming(s, oe.Dataset.from_frame(d), batch_rows=31)
        if streamed
        else registry.load_entry(registry.get("rdrobust"))(s, d)
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    for correction, i in [("conventional", 0), ("bias_corrected", 1), ("robust", 2)]:
        out = collect(
            oe.causal_evaluate(r, target="cutoff_effect", point=0, correction=correction)
        ).iloc[0]
        assert out.estimate == r.coefficients[i].estimate
        assert out.std_error == pytest.approx(r.coefficients[i].std_error)
    for side in ["left", "right"]:
        out = collect(oe.causal_evaluate(r, target="cutoff_side", side=side, point=0)).iloc[0]
        assert out.estimate == r.extra["side_estimates"][side]["bias_corrected"]
        assert out.std_error == pytest.approx(
            np.sqrt(r.extra["cutoff_evaluation"]["sides"][side]["robust_variance"])
        )
        assert out.main_bandwidth == r.metrics["h_" + side]
    with pytest.raises(AnalysisError, match="cutoff"):
        oe.causal_evaluate(r, target="cutoff_side", side="left", point=-0.01)
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(r, target="cutoff_side", point=0)


@pytest.mark.parametrize("name", ["didregress", "eventstudy", "csdid"])
def test_saved_identified_did_and_joint_share_aggregations(name):
    if name == "csdid":
        from test_econ_streaming_csdid import fixture, spec, fit_streaming

        d = fixture(65)
        r = fit_streaming(spec(), oe.Dataset.from_frame(d), batch_rows=17)
        r = ResultBundle.model_validate_json(r.model_dump_json())
        for target in ["simple", "dynamic", "calendar"]:
            actual = collect(oe.causal_evaluate(r, target=target))
            expected = pd.DataFrame([r.extra[target]] if target == "simple" else r.extra[target])
            np.testing.assert_allclose(
                actual[["estimate", "std_error"]], expected[["estimate", "std_error"]]
            )
        group = r.extra["group"][0]["label"]
        assert len(collect(oe.causal_evaluate(r, target="group", cohort=group))) == 1
        with pytest.raises(AnalysisError):
            oe.causal_evaluate(r, target="group", cohort=999)
    else:
        from test_econ_streaming_did import spec, fit_streaming, make_panel

        d = make_panel(groups=24, staggered=name == "eventstudy")
        r = fit_streaming(spec(name), oe.Dataset.from_frame(d), batch_rows=13)
        r = ResultBundle.model_validate_json(r.model_dump_json())
        if name == "didregress":
            actual = collect(oe.causal_evaluate(r, target="population_effect")).iloc[0]
            assert actual.estimate == r.coefficients[0].estimate
            assert actual.std_error == pytest.approx(r.coefficients[0].std_error)
        else:
            event = next(
                row
                for row in r.extra["event_table"]
                if row["estimate"] is not None and not row["reference"]
            )
            actual = collect(
                oe.causal_evaluate(r, target="event_effect", event=event["relative_time"])
            ).iloc[0]
            assert actual.estimate == event["estimate"]
            assert actual.std_error == pytest.approx(event["std_error"])
            with pytest.raises(AnalysisError):
                oe.causal_evaluate(r, target="event_effect", event=999)
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(
            r, oe.Dataset.from_frame(d), target="outcome_mean", population="fixed_evaluation"
        )


def test_explicit_gmm_function_saved_covariance_global_population_and_undefined_response():
    from test_econ_systems_gmm import make_iv, LINEAR, INSTR

    d = make_iv(n=180)
    r = oe.gmm(
        data=oe.Dataset.from_frame(d), moments=LINEAR, instruments=INSTR, winitial="unadjusted"
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    evaluation = d.iloc[:15]
    function = "{b0}+{b1}*x1+{b2}*x2"
    x = np.c_[np.ones(len(evaluation)), evaluation.x1, evaluation.x2]
    beta = np.array([c.estimate for c in r.coefficients])
    cov = np.array(r.covariance_matrix)
    out = collect(
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(evaluation),
            target="outcome_mean",
            population="fixed_evaluation",
            outcome_function=function,
            batch_rows=4,
        )
    )
    np.testing.assert_allclose(out.response, x @ beta)
    np.testing.assert_allclose(out.std_error, np.sqrt(np.einsum("nk,kl,nl->n", x, cov, x)))
    pop = collect(
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(evaluation),
            target="standardized_outcome",
            population="fixed_evaluation",
            outcome_function=function,
            batch_rows=4,
        )
    ).iloc[0]
    assert pop.estimate == pytest.approx(x.mean(0) @ beta)
    assert pop.std_error == pytest.approx(np.sqrt(x.mean(0) @ cov @ x.mean(0)))
    with pytest.raises(AnalysisError):
        oe.predict(r, evaluation)
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(evaluation),
            target="outcome_mean",
            population="fixed_evaluation",
        )
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(evaluation),
            target="outcome_mean",
            population="fixed_evaluation",
            outcome_function="__import__('os').system('true')",
        )


def test_old_state_support_missing_population_and_corrupted_joint_covariance():
    d = make_data(n=230)
    r = oe.teffects(
        data=oe.Dataset.from_frame(d), y="y", treatment="d", x=["x1", "x2"], method="aipw"
    )
    with pytest.raises(AnalysisError, match="selectors"):
        oe.causal_evaluate(r, target="population_effect", treatment="1", cohort=3)
    r.extra.pop("evaluation_state")
    with pytest.raises(AnalysisError, match="no refit"):
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(d),
            target="outcome_mean",
            population="fixed_evaluation",
            treatment="1",
        )
    r = oe.teffects(
        data=oe.Dataset.from_frame(d), y="y", treatment="d", x=["x1", "x2"], method="aipw"
    )
    hostile = d.iloc[:2].assign(x1=1e4, x2=-1e4)
    with pytest.raises(AnalysisError, match="overlap"):
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(hostile),
            target="outcome_mean",
            population="fixed_evaluation",
            treatment="1",
        )
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(r, target="population_effect", treatment="2")
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(
            r, oe.Dataset.from_frame(d), target="population_effect", population="new", treatment="1"
        )
    r.extra["evaluation_state"]["covariance"][0][0] = -1
    with pytest.raises(AnalysisError):
        oe.causal_evaluate(
            r,
            oe.Dataset.from_frame(d),
            target="outcome_mean",
            population="fixed_evaluation",
            treatment="1",
        )
