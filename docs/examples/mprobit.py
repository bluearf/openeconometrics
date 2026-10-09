"""Synthetic bounded multinomial probit, complete output and saved-query replay."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import pandas as pd
import torch

import openecon as oe


def synthetic_choices(seed=706, cases=128, alternatives=3):
    generator = random.Random(seed)
    records = []
    labels = ("base", "second", "third")[:alternatives]
    for case in range(cases):
        x = [generator.uniform(1., 5.) for _ in labels]
        a, b = generator.gauss(0., 1.), generator.gauss(0., 1.)
        errors = [0., a, 1.2*(.15*a+math.sqrt(1-.15**2)*b)][:alternatives]
        chosen = max(range(alternatives), key=lambda j: .65*x[j]+errors[j])
        for j, alternative in enumerate(labels):
            records.append(dict(case=case, alternative=alternative, x=x[j],
                                chosen=int(chosen == j), available=True, cluster=case//4))
    return pd.DataFrame(records)


def query_data():
    return pd.DataFrame([dict(case=case, alternative=alternative, x=x, available=available)
                        for case, values, available in [
                            ("query-A", (2., 3., 4.), (True, True, True)),
                            ("query-B", (2.5, 4., 3.), (True, True, False)),
                            ("query-C", (3., 2., 3.5), (False, True, True)),
                        ] for alternative, x, available in zip(("base", "second", "third"), values, available)])


def tables(result):
    return {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
            for key, frame in result.items()}


def payload(result):
    return dict(attrs=result.attrs, tables=tables(result), latex=result.to_latex())


def assert_close(actual, expected, tolerance=2e-10):
    assert len(actual) == len(expected)
    assert all(math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance*1e-2)
               for a, b in zip(actual, expected, strict=True))


def verify_binary_reduction(fit):
    """Binary CDF/effect/elasticity and all physical beta Jacobian, without helpers."""
    state = fit.attrs["mprobit_state"]
    beta = state["fit"]["params"][0]
    data = pd.DataFrame([dict(case="binary", alternative="base", x=2., available=True),
                         dict(case="binary", alternative="second", x=3., available=True)])
    prediction = oe.mprobit_predict(fit, data)
    probability = .5*math.erfc(-beta/math.sqrt(2))
    density = math.exp(-beta*beta/2)/math.sqrt(2*math.pi)
    assert_close(prediction["predictions"].estimate, [1-probability, probability])
    assert_close(prediction["jacobian"].iloc[:, 1].tolist(), [-density, density])
    effect = oe.mprobit_margins(fit, data, targets=[
        dict(outcome="second", changed="second", attribute="x")])
    elasticity = oe.mprobit_margins(fit, data, targets=[
        dict(outcome="second", changed="second", attribute="x")], elasticity=True)
    assert_close(effect["effects"].estimate, [beta*density])
    assert_close(effect["jacobian"].iloc[:, 1].tolist(), [density*(1-beta*beta)])
    expected_elasticity = 3*beta*density/probability
    expected_derivative = 3*density*(1-beta*beta)/probability-3*beta*density*density/probability**2
    assert_close(elasticity["elasticities"].estimate, [expected_elasticity])
    assert_close(elasticity["jacobian"].iloc[:, 1].tolist(), [expected_derivative])
    variance = state["fit"]["covariance"][0][0]
    assert_close(prediction["covariance"].iloc[:, 1:].values.ravel(),
                 [density*density*variance, -density*density*variance,
                  -density*density*variance, density*density*variance])


def verify_saved_queries(results, query):
    prediction = results["probabilities"]["predictions"]
    for case in query.case.drop_duplicates():
        selected = prediction[prediction.case == case]
        assert math.isclose(math.fsum(selected.estimate), 1., rel_tol=2e-8, abs_tol=2e-9)
    unavailable = prediction[((prediction.case == "query-B") & (prediction.alternative == "third"))
                             | ((prediction.case == "query-C") & (prediction.alternative == "base"))]
    assert unavailable.estimate.tolist() == [0., 0.]
    logs = results["log_probabilities"]["predictions"]
    positive = prediction[prediction.estimate > 0]
    assert_close(logs.estimate, [math.log(value) for value in positive.estimate])
    # Every common utility shift changes all alternative cells equally.
    effects = results["effects"]["effects"]
    for case in query.case.drop_duplicates():
        for changed in query.loc[(query.case == case) & query.available, "alternative"]:
            selected = effects[(effects.case == case) & (effects.changed == changed)]
            assert abs(math.fsum(selected.estimate)) < 2e-8
    for key, source_key in (("average_effects", "effects"),
                             ("average_elasticities", "elasticities")):
        result = results[key]
        for row in result["averages"].itertuples(index=False):
            per_case = result[source_key][result[source_key].margin == row.target]
            expected = math.fsum(per_case.estimate*per_case.normalized_weight)
            assert_close([row.estimate], [expected])
            assert row.eligible_cases == len(per_case)
        J = torch.tensor(result["jacobian"].iloc[:, 1:].values.tolist(), dtype=torch.float64)
        covariance = torch.tensor(results["cr0"].attrs["mprobit_state"]["fit"]["covariance"],
                                  dtype=torch.float64)
        actual = torch.tensor(result["covariance"].iloc[:, 1:].values.tolist(), dtype=torch.float64)
        assert torch.allclose(actual, J@covariance@J.T, rtol=2e-10, atol=2e-13)
        assert float(actual[:len(result[source_key]), len(result[source_key]):].abs().sum()) > 0
    assert len(effects) == 17
    assert len(results["average_effects"]["averages"]) == 9
    assert len(results["average_effects"]["support"]) == 27
    assert results["average_effects"]["averages"].eligible_cases.min() == 1


def run_acceptance(directory=None, show=True):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = synthetic_choices()
        results = {}
        for vce in ("oim", "hc0", "cr0"):
            fit = oe.mprobit(data, "chosen", ["x"], case="case", alternative="alternative",
                             alternatives=["base", "second", "third"],
                             available="available", cluster="cluster" if vce == "cr0" else None,
                             vce=vce)
            saved = json.loads(json.dumps(payload(fit), sort_keys=True, allow_nan=False))
            assert payload(oe.mprobit_restore(saved["attrs"])) == saved
            results[vce] = fit
        assert results["oim"].attrs["mprobit_state"]["fit"]["params"] == results["cr0"].attrs["mprobit_state"]["fit"]["params"]
        saved_fit = json.loads(json.dumps(results["cr0"].attrs, sort_keys=True, allow_nan=False))
        results["restored"] = oe.mprobit_restore(saved_fit)
        binary = synthetic_choices(alternatives=2)
        results["fixed"] = oe.mprobit(binary, "chosen", ["x"], case="case", alternative="alternative",
                                      alternatives=["base", "second"],
                                      available="available", covariance=[[1.]])
        verify_binary_reduction(results["fixed"])
        query = query_data()
        probability_targets = [dict(case=row.case, alternative=row.alternative)
                               for row in query.itertuples(index=False)]
        calls = {
            "probabilities": lambda fit: oe.mprobit_predict(fit, query, targets=probability_targets),
            "log_probabilities": lambda fit: oe.mprobit_predict(fit, query, kind="log_probability"),
            "effects": lambda fit: oe.mprobit_margins(fit, query, attribute="x"),
            "elasticities": lambda fit: oe.mprobit_margins(fit, query, attribute="x", elasticity=True),
            "average_effects": lambda fit: oe.mprobit_margins(fit, query, attribute="x", average=True,
                                                            weights=[1., 2., 3.]),
            "average_elasticities": lambda fit: oe.mprobit_margins(fit, query, attribute="x", elasticity=True,
                                                                 average=True, weights=[1., 2., 3.]),
        }
        for name, function in calls.items():
            results[name] = function(results["cr0"])
            assert payload(function(saved_fit)) == payload(results[name])
            assert results[name].attrs["settings"]["no_optimizer"] is True
        verify_saved_queries(results, query)
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        proof = {}
        fit_names = ("oim", "hc0", "cr0", "restored", "fixed")
        for name, result in results.items():
            content = json.dumps(payload(result), sort_keys=True, allow_nan=False)
            if directory is not None:
                (directory/(name+".json")).write_text(content, encoding="utf-8")
            keys = (["parameters"] if name in fit_names else ["predictions"]
                    if name in ("probabilities", "log_probabilities") else ["averages"]
                    if name.startswith("average_") else ["elasticities"] if name == "elasticities"
                    else ["effects"])
            proof[name] = dict(sha256=hashlib.sha256(content.encode()).hexdigest(),
                               table_order=list(result), table_rows={key: len(frame) for key, frame in result.items()},
                               displayed_keys=keys,
                               displayed_latex_sha256={key: hashlib.sha256(str(result[key].to_latex()).encode()).hexdigest()
                                                       for key in keys})
            if show:
                for key in keys:
                    if "display" in globals():
                        globals()["display"](result[key])
                    else:
                        print(name+" — "+key)
                        print(result[key].to_string(index=False))
        forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
        assert not forbidden
        receipt = dict(methods=proof, frozen=bool(getattr(sys, "frozen", False)),
                       all_eight_scopes_verified=True, full_state_replay_verified=True,
                       analytical_binary_probability_effect_elasticity_verified=True,
                       full_physical_joint_delta_verified=True, availability_and_complete_support_verified=True,
                       fixed_weight_case_average_verified=True,
                       saved_tables=sum(len(result) for result in results.values()),
                       displayed_tables=11 if show else 0, third_party_estimation_imports=forbidden)
        print("MPROBIT_ACCEPTANCE_OK "+json.dumps(receipt, sort_keys=True, allow_nan=False))
        return results, receipt
    finally:
        torch.set_num_threads(previous_threads)


if globals().get("MPROBIT_LIBRARY_ONLY") is not True:
    run_acceptance(globals().get("MPROBIT_RESULT_DIRECTORY"), globals().get("MPROBIT_DISPLAY", True))
