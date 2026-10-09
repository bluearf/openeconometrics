"""Scientific hand case and complete no-refit/integrity contract checks."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.decomposition import binary
from openecon.econometrics.decomposition.mediation import CAUSAL_ASSUMPTIONS
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def data(kind="gaussian", seed=5041, n=160):
    r = np.random.default_rng(seed)
    a, c = r.binomial(1, .5, n), r.normal(size=n)
    m = r.binomial(1, 1/(1+np.exp(-(-.3+.6*a+.3*c))))
    eta = -.2+.6*a+.8*m+.2*a*m+.25*c
    if kind == "gaussian":
        y = eta+r.normal(size=n)
    elif kind == "poisson":
        y = r.poisson(np.exp(eta))
    else:
        y = r.binomial(1, 1/(1+np.exp(-eta)))
    return dict(y=y, a=a, m=m, c=c)


def fit(kind="gaussian", link="logit", **kwargs):
    options = dict(outcome_model=kind, mediator_link=link)
    options.update(kwargs)
    return binary.mediation_binary(data=data(kind), y="y", treatment="a", mediator="m",
                                   controls=["c"], **options)


def attrs(result):
    return json.loads(json.dumps(result.attrs, sort_keys=True, ensure_ascii=False, allow_nan=False))


def seal(saved):
    state = saved["binary_mediation_state"]
    state["checksum"] = hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "checksum"},
                                                 sort_keys=True, ensure_ascii=False, allow_nan=False,
                                                 separators=(",", ":")).encode()).hexdigest()
    return saved


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_saturated_gaussian_hand_case(link):
    # Both exposure strata have 64 observations. The mediator rates are exactly
    # 1/4 and 3/4; the four conditional outcome means are 1,3,2,6.
    a, m, y = [], [], []
    for av, mv, count, mean in [(0, 0, 48, 1), (0, 1, 16, 2), (1, 0, 16, 3), (1, 1, 48, 6)]:
        a.extend([av]*count)
        m.extend([mv]*count)
        y.extend([mean-1, mean+1]*(count//2))
    result = binary.mediation_binary(data=dict(y=y, a=a, m=m), y="y", treatment="a", mediator="m",
                                     mediator_link=link, covariance="OIM")
    np.testing.assert_allclose(result["means"]["estimate"], [1.25, 3.75, 1.75, 5.25], atol=1e-10)
    np.testing.assert_allclose(result["effects"]["estimate"], [2.5, 3.5, .5, 1.5, 4], atol=1e-10)
    covariance = result["effects_covariance"].iloc[:, 1:].to_numpy(float)
    # Two exact contrast identities force two null directions, rather than a
    # five-parameter regular joint Wald distribution.
    np.testing.assert_allclose(np.array([[-1, 0, 0, -1, 1], [0, -1, -1, 0, 1]])@covariance, 0, atol=1e-12)
    assert np.linalg.matrix_rank(covariance, tol=1e-10) <= 3
    assert result.attrs["settings"]["global_effect_wald"] is None


@pytest.mark.parametrize("kind", ["gaussian", "logit", "probit", "poisson"])
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_canonical_json_replays_all_tables_attrs_and_latex_without_fit(kind, link, monkeypatch):
    result = fit(kind, link)
    def forbidden(*args, **kwargs):
        raise AssertionError("Restoration must not optimize or refit")
    monkeypatch.setattr(binary, "fit_joint", forbidden)
    from openecon.econometrics.decomposition import binary_kernels
    monkeypatch.setattr(binary_kernels, "fit_joint", forbidden)
    restored = binary.mediation_binary_restore(attrs(result))
    assert len(restored) == 14
    assert restored.attrs == result.attrs
    assert restored.to_latex() == result.to_latex()
    assert all(restored[name].equals(result[name]) for name in result)
    assert "p_value" not in result["means"] and "z" not in result["means"]
    if kind == "gaussian":
        assert pd.isna(result["parameters"].iloc[-1]["p_value"])


def test_level_change_retains_every_saved_fit_and_target_number(monkeypatch):
    result = fit()
    monkeypatch.setattr(binary, "fit_joint", lambda *args, **kwargs: pytest.fail("refit"))
    restored = binary.mediation_binary_restore(attrs(result), level=.8)
    for name in result:
        if name not in ("parameters", "means", "effects"):
            assert restored[name].equals(result[name])
        else:
            assert restored[name].drop(columns=["ci_lower", "ci_upper"]).equals(result[name].drop(columns=["ci_lower", "ci_upper"]))
    assert (restored["effects"]["ci_upper"]-restored["effects"]["ci_lower"]).lt(
        result["effects"]["ci_upper"]-result["effects"]["ci_lower"]).all()
    original_state = result.attrs["binary_mediation_state"]
    changed_state = restored.attrs["binary_mediation_state"]
    assert original_state["fit"] == changed_state["fit"]
    assert original_state["derived"] == changed_state["derived"]
    assert binary.mediation_binary_restore(restored).attrs == restored.attrs


@pytest.mark.parametrize("field", ["theta", "scores", "information", "bread", "covariance", "row_loglikelihood"])
def test_resealed_numeric_state_tampering_is_rejected(field):
    saved = attrs(fit())
    array = saved["binary_mediation_state"]["fit"][field]
    if field == "theta":
        array[0] += .1
    else:
        array[0][0] += .1
    with pytest.raises(AnalysisError):
        binary.mediation_binary_restore(seal(saved))


@pytest.mark.parametrize("field,value", [("solver", "fallback"), ("n", 99), ("parameter_names", ["invented"]),
                                         ("parameter_slices", {}), ("estimated_work", 0), ("convergence", {}),
                                         ("equation_log_likelihood", [0, 0]), ("control_scale", [0])])
def test_resealed_metadata_tampering_is_rejected(field, value):
    saved = attrs(fit())
    saved["binary_mediation_state"]["fit"][field] = value
    with pytest.raises(AnalysisError):
        binary.mediation_binary_restore(seal(saved))


def test_fully_consistent_nonstationary_fit_cannot_be_restored():
    saved = attrs(fit())
    state, options = saved["binary_mediation_state"], saved["binary_mediation_state"]["options"]
    inputs = state["inputs"]
    y, m, a, c = [torch.tensor(inputs[key], dtype=torch.float64, device="cpu") for key in ("y", "m", "a", "c")]
    theta = torch.tensor(state["fit"]["theta"], dtype=torch.float64, device="cpu")
    theta[1] += .2
    replay = binary.evaluate_joint(theta, y, m, a, c, mediator_link=options["mediator_link"],
                                   outcome_model=options["outcome_model"], interaction=options["interaction"],
                                   covariance=options["covariance"])
    state["fit"].update(replay)
    for i, name in enumerate(("mediator", "outcome")):
        state["fit"]["equations"][name]["log_likelihood"] = replay["equation_log_likelihood"][i]
    state["derived"] = binary._derived(theta, torch.tensor(replay["covariance"], dtype=torch.float64, device="cpu"),
                                       y, m, a, c, options)
    with pytest.raises(AnalysisError, match="stationar|Newton|fit gate"):
        binary.mediation_binary_restore(seal(saved))


@pytest.mark.parametrize("field", ["covariance", "target_covariance", "natural_covariance"])
def test_tiny_outcome_units_do_not_allow_an_absolute_covariance_tolerance(field):
    source = data()
    source["y"] *= 1e-8
    result = binary.mediation_binary(data=source, y="y", treatment="a", mediator="m", controls=["c"])
    saved = attrs(result)
    state = saved["binary_mediation_state"]
    if field == "covariance":
        # Keep the rest of the state mutually consistent with the forged
        # covariance; only complete likelihood replay can reject this attack.
        state["fit"]["covariance"][4][4] += 1e-12
        inputs, options = state["inputs"], state["options"]
        y, m, a, c = [torch.tensor(inputs[key], dtype=torch.float64, device="cpu") for key in ("y", "m", "a", "c")]
        theta = torch.tensor(state["fit"]["theta"], dtype=torch.float64, device="cpu")
        state["derived"] = binary._derived(theta, torch.tensor(state["fit"]["covariance"], dtype=torch.float64, device="cpu"),
                                           y, m, a, c, options)
    elif field == "target_covariance":
        state["derived"]["covariance"][0][0] += 1e-12
    else:
        state["derived"]["natural_covariance"][4][4] += 1e-12
    with pytest.raises(AnalysisError, match="replay"):
        binary.mediation_binary_restore(seal(saved))
    restored = binary.mediation_binary_restore(attrs(result))
    assert restored["effects_covariance"].equals(result["effects_covariance"])


def test_tables_and_outer_attrs_must_match_sealed_state():
    result = fit()
    result["effects"].loc[0, "estimate"] += .1
    with pytest.raises(AnalysisError, match="tables"):
        binary.mediation_binary_restore(result)
    saved = attrs(fit())
    saved["settings"]["n"] = 99
    with pytest.raises(AnalysisError, match="attrs"):
        binary.mediation_binary_restore(saved)


def test_checksum_shape_schema_and_resource_claims():
    for mutate in (
        lambda state: state.update(checksum="0"*64),
        lambda state: state["fit"].update(unexpected="field"),
        lambda state: state["inputs"]["c"].pop(),
        lambda state: state["derived"]["jacobian"].pop(),
        lambda state: state["settings"]["resources"].update(estimated_workspace_bytes=0),
    ):
        saved = attrs(fit())
        mutate(saved["binary_mediation_state"])
        with pytest.raises(AnalysisError):
            binary.mediation_binary_restore(seal(saved) if saved["binary_mediation_state"]["checksum"] != "0"*64 else saved)


def test_unavailable_first_order_inference_is_not_point_certainty():
    # An exact zero first derivative permits a nonzero higher-order product
    # effect distribution. First-order delta variance cannot justify [0,0].
    result = binary._inference_rows(["PNIE"], [0.0], [[0.0]], .95, label="effect")
    assert result.iloc[0]["std_error"] == 0
    assert result.iloc[0]["ci_lower"] is None and result.iloc[0]["ci_upper"] is None
    assert result.iloc[0]["z"] is None and result.iloc[0]["p_value"] is None
    assert result.iloc[0]["inference_status"].startswith("unavailable")


@pytest.mark.parametrize("option,value", [("device", "meta"), ("weights", [1]), ("cluster", "g"),
                                          ("interaction", 1), ("covariance", "HC1"), ("outcome_model", "linear"),
                                          ("mediator_link", "cloglog"), ("max_iterations", True),
                                          ("max_iterations", 1001), ("tolerance", float("nan")), ("level", 1)])
def test_explicit_unsupported_options(option, value):
    with pytest.raises(AnalysisError):
        fit(**{option: value})


@pytest.mark.parametrize("column,value", [("a", 2), ("m", -1), ("y", float("nan")),
                                          ("c", None), ("c", True), ("c", 10**1000)])
def test_missing_and_unsupported_coding_never_drop_or_recode(column, value):
    source = {name: values.tolist() for name, values in data().items()}
    source[column][0] = value
    with pytest.raises(AnalysisError):
        binary.mediation_binary(data=source, y="y", treatment="a", mediator="m", controls=["c"])


def test_binary_y_requires_exact_zero_one_and_counts_require_integers():
    source = data("logit")
    source["y"] += 2
    with pytest.raises(AnalysisError, match="0 and 1"):
        binary.mediation_binary(data=source, y="y", treatment="a", mediator="m", outcome_model="logit")
    source = data("poisson")
    source["y"] = source["y"].astype(float)
    source["y"][0] = .5
    with pytest.raises(AnalysisError, match="integers"):
        binary.mediation_binary(data=source, y="y", treatment="a", mediator="m", outcome_model="poisson")


def test_causal_interpretation_requires_all_declarations():
    with pytest.raises(AnalysisError, match="Causal interpretation"):
        fit(interpretation="causal")
    result = fit(interpretation="causal", assumptions=sorted(CAUSAL_ASSUMPTIONS))
    assert binary.mediation_binary_restore(attrs(result)).attrs == result.attrs


def test_workspace_gate_precedes_numeric_materialization(monkeypatch):
    source = data(n=512)
    for j in range(5):
        source[f"c{j}"] = np.random.default_rng(j).normal(size=512)
    monkeypatch.setattr(binary, "_vector", lambda *args, **kwargs: pytest.fail("Materialized before budget check"))
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        binary.mediation_binary(data=source, y="y", treatment="a", mediator="m", controls=[f"c{j}" for j in range(5)])


def test_row_label_collision_and_nonmutation():
    source = pd.DataFrame(data()).rename(columns={"y": "row", "c": "_row"})
    before = source.copy(deep=True)
    result = binary.mediation_binary(data=source, y="row", treatment="a", mediator="m", controls=["_row"])
    pd.testing.assert_frame_equal(source, before)
    assert result["inputs"].columns.is_unique
    assert result["inputs"].columns[0] == "__row"
    result["inputs"].iloc[0, 1] = 99
    pd.testing.assert_frame_equal(source, before)


def test_global_meta_and_float32_do_not_change_cpu_float64_contract():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = fit("probit", "probit")
        restored = binary.mediation_binary_restore(attrs(result))
        assert restored.attrs == result.attrs
        assert restored["effects"].equals(result["effects"])
        assert result.attrs["settings"]["precision"] == "float64"
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(original_dtype)
