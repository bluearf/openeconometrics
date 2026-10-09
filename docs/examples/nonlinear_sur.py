"""Synthetic shared-parameter nonlinear SUR, complete save/replay and joint deltas."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import pandas as pd
import torch

import openecon as oe


def synthetic_system(seed=679, rows=192):
    generator = random.Random(seed)
    records = []
    for group in range(rows//6):
        common = (generator.gauss(0, 1), generator.gauss(0, 1))
        for _ in range(6):
            x, z = generator.uniform(-1.5, 1.5), generator.uniform(-1.5, 1.5)
            a, b = (math.sqrt(.4)*common[j]+math.sqrt(.6)*generator.gauss(0, 1) for j in range(2))
            e1, e2 = .7*a, (.2/.7)*a+math.sqrt(.81-(.2/.7)**2)*b
            records.append(dict(x=x, z=z, y1=2+.3*x+1.2*math.exp(.3)*x*x+e1,
                                y2=2+.3*z+1.2*math.exp(-.3)*z*z+e2, cluster=group))
    return pd.DataFrame(records, index=[f"sample-{i:03d}" for i in range(rows)])


def tables(result):
    return {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
            for key, frame in result.items()}


def payload(result):
    return dict(attrs=result.attrs, tables=tables(result), latex=result.to_latex())


def verify_saved_targets(results, query):
    """Closed Gaussian conditioning and analytic derivatives, independent of queries."""
    state = results["cr0"].attrs["nonlinear_sur_state"]
    names, params = state["fit"]["parameter_names"], state["fit"]["params"]
    b0, b1, amp, rate = (params[names.index(name)] for name in ("b0", "b1", "amp", "rate"))
    sigma = state["fit"]["sigma"]
    loading = sigma[1][0] / sigma[0][0]
    residual_variance = sigma[1][1] - sigma[1][0]**2/sigma[0][0]
    means, conditional, effects, elasticities, conditional_effects = [], [], [], [], []
    conditional_jacobian = []
    for x, z, observed in query[["x", "z", "y1"]].itertuples(index=False, name=None):
        first = b0+b1*x+amp*math.exp(rate)*x*x
        second = b0+b1*z+amp*math.exp(-rate)*z*z
        dx, dz = b1+2*amp*math.exp(rate)*x, b1+2*amp*math.exp(-rate)*z
        means.extend([first, second])
        conditional.append(second+loading*(observed-first))
        effects.extend([dx, 0., 0., dz])
        elasticities.extend([dx*x/first, 0., 0., dz*z/second])
        conditional_effects.extend([-loading*dx, dz])
        gradient = [0.] * len(names)
        for name, first_partial, second_partial in [
            ("b0", 1., 1.), ("b1", x, z),
            ("amp", math.exp(rate)*x*x, math.exp(-rate)*z*z),
            ("rate", amp*math.exp(rate)*x*x, -amp*math.exp(-rate)*z*z),
        ]:
            gradient[names.index(name)] = second_partial-loading*first_partial
        gradient[names.index("cov__first__first")] = -loading*(observed-first)/sigma[0][0]
        gradient[names.index("cov__second__first")] = (observed-first)/sigma[0][0]
        conditional_jacobian.append(gradient)
    variance_gradient = [0.] * len(names)
    variance_gradient[names.index("cov__first__first")] = loading**2
    variance_gradient[names.index("cov__second__first")] = -2*loading
    variance_gradient[names.index("cov__second__second")] = 1.
    conditional_jacobian.append(variance_gradient)

    def close(actual, expected):
        assert len(actual) == len(expected)
        assert all(math.isclose(float(a), float(b), rel_tol=2e-11, abs_tol=2e-13)
                   for a, b in zip(actual, expected, strict=True))

    close(results["means"]["means"].estimate, means)
    close(results["conditional"]["means"].estimate, conditional)
    close(results["conditional"]["residual_covariance"].estimate, [residual_variance])
    close(results["effects"]["effects"].estimate, effects)
    close(results["elasticities"]["elasticities"].estimate, elasticities)
    close(results["conditional_effects"]["effects"].estimate, conditional_effects)
    # The positive query features share fixed weights 1:2:3. Reduce each
    # equation/feature pair before inference, including its structural zeros.
    for label, values, width in (("effects", effects, 4), ("elasticities", elasticities, 4),
                                 ("conditional_effects", conditional_effects, 2)):
        expected = [sum((row+1)*values[row*width+j] for row in range(3))/6 for j in range(width)]
        close(results[label]["averages"].estimate, expected)
        assert len(results[label]["effects" if label != "elasticities" else "elasticities"]) == 3*width
    gradient = torch.tensor(conditional_jacobian, dtype=torch.float64, device="cpu")
    covariance = torch.tensor(state["fit"]["covariance"], dtype=torch.float64, device="cpu")
    actual_gradient = torch.tensor(results["conditional"]["parameter_jacobian"].iloc[:, 1:].values.tolist(),
                                   dtype=torch.float64, device="cpu")
    actual_covariance = torch.tensor(results["conditional"]["target_covariance"].iloc[:, 1:].values.tolist(),
                                     dtype=torch.float64, device="cpu")
    assert torch.allclose(actual_gradient, gradient, rtol=2e-11, atol=2e-13)
    assert torch.allclose(actual_covariance, gradient@covariance@gradient.T, rtol=2e-10, atol=2e-13)
    assert abs(float(actual_gradient[:3, names.index("cov__second__first")].abs().sum())) > 0
    assert float(actual_covariance[:3, -1].abs().sum()) > 0
    close(results["covariance_contrast"]["contrasts"].estimate, [residual_variance])
    # Exact two-restriction Wald quadratic, with the cross covariance retained.
    slope, asymmetry = names.index("b1"), names.index("rate")
    v11, v12, v22 = float(covariance[slope, slope]), 2*float(covariance[slope, asymmetry]), 4*float(covariance[asymmetry, asymmetry])
    expected_wald = (v22*b1*b1-2*v12*b1*(2*rate)+v11*(2*rate)**2)/(v11*v22-v12*v12)
    assert math.isclose(results["contrasts"]["wald"].iloc[0].statistic, expected_wald, rel_tol=2e-11)
    assert results["contrasts"]["wald"].iloc[0].df == 2
    assert results["conditional"]["means"].index.equals(query.index)
    assert results["effects"]["effects"].index.equals(query.index.repeat(4))


def run_acceptance(directory=None, show=True):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        frame = synthetic_system()
        equations = [dict(y="y1", name="first", formula="{b0=2}+{b1=.3}*x+{amp=1.2}*exp({rate=.3})*x^2"),
                     dict(y="y2", name="second", formula="{b0=2}+{b1=.3}*z+{amp=1.2}*exp(-{rate=.3})*z^2")]
        results, proof = {}, {}
        for covariance in ("oim", "hc0", "cr0"):
            result = oe.nlsur(frame, equations, covariance=covariance,
                              cluster="cluster" if covariance == "cr0" else None)
            saved = json.loads(json.dumps(payload(result), sort_keys=True, allow_nan=False))
            assert payload(oe.nlsur_restore(saved["attrs"])) == saved
            assert len(result) == 14
            results[covariance] = result
        assert results["oim"].attrs["nonlinear_sur_state"]["fit"]["params"] == results["cr0"].attrs["nonlinear_sur_state"]["fit"]["params"]
        saved_fit = json.loads(json.dumps(results["cr0"].attrs, sort_keys=True, allow_nan=False))
        results["restored"] = oe.nlsur_restore(saved_fit)
        query = pd.DataFrame(dict(x=[.2,.6,1.1], z=[.35,.55,.95], y1=[2.3,2.9,4.1]),
                             index=["query-A","query-B","query-C"])
        calls = {
            "means": lambda fit: oe.nlsur_predict(fit, query),
            "conditional": lambda fit: oe.nlsur_predict(fit, query, given=["first"]),
            "effects": lambda fit: oe.nlsur_margins(fit, query, x=["x","z"], weights=[1.,2.,3.]),
            "elasticities": lambda fit: oe.nlsur_margins(fit, query, x=["x","z"], scale="elasticity", weights=[1.,2.,3.]),
            "conditional_effects": lambda fit: oe.nlsur_margins(fit, query, x=["x","z"], given=["first"], weights=[1.,2.,3.]),
            "contrasts": lambda fit: oe.nlsur_contrast(fit, {"slope":"{b1}", "asymmetry":"2*{rate}"}, null={"slope":0.,"asymmetry":0.}),
            "covariance_contrast": lambda fit: oe.nlsur_contrast(fit, {"conditional_variance":"{cov__second__second}-{cov__second__first}^2/{cov__first__first}"}, null={"conditional_variance":.5}),
        }
        for name, function in calls.items():
            results[name] = function(results["cr0"])
            assert payload(function(saved_fit)) == payload(results[name])
            assert results[name].attrs["settings"]["refit"] is False
            assert results[name].attrs["settings"]["no_optimizer"] is True
        verify_saved_targets(results, query)
        assert sum(len(result) for result in results.values()) == 86
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            content = json.dumps(payload(result), sort_keys=True, allow_nan=False)
            if directory is not None:
                (directory/(name+".json")).write_text(content, encoding="utf-8")
            keys = ["parameters"] if name in ("oim","hc0","cr0","restored") else ["means"] if name in ("means","conditional") else ["elasticities"] if name == "elasticities" else ["effects"] if name in ("effects","conditional_effects") else ["contrasts"]
            proof[name] = dict(sha256=hashlib.sha256(content.encode()).hexdigest(),
                               table_rows={key:len(value) for key,value in result.items()},
                               table_order=list(result), displayed_keys=keys,
                               displayed_latex_sha256={key:hashlib.sha256(str(result[key].to_latex()).encode()).hexdigest() for key in keys})
            if show:
                for key in keys:
                    if "display" in globals():
                        globals()["display"](result[key])
                    else:
                        print(name+" — "+key)
                        print(result[key].to_string(index=False))
        forbidden = [name for name in ("scipy","statsmodels","linearmodels") if name in sys.modules]
        assert not forbidden
        receipt = dict(methods=proof, frozen=bool(getattr(sys,"frozen",False)),
                       all_eight_scopes_verified=True, full_state_replay_verified=True,
                       closed_form_conditional_mean_and_covariance_verified=True,
                       complete_conditional_jacobian_and_cross_covariance_verified=True,
                       continuous_effect_elasticity_and_fixed_weight_averages_verified=True,
                       joint_wald_cross_covariance_verified=True,
                       saved_tables=86, displayed_tables=11 if show else 0,
                       third_party_estimation_imports=forbidden)
        print("NONLINEAR_SUR_ACCEPTANCE_OK "+json.dumps(receipt,sort_keys=True,allow_nan=False))
        return results, receipt
    finally:
        torch.set_num_threads(previous_threads)


if globals().get("NONLINEAR_SUR_LIBRARY_ONLY") is not True:
    run_acceptance(globals().get("NONLINEAR_SUR_RESULT_DIRECTORY"),globals().get("NONLINEAR_SUR_DISPLAY",True))
