"""Synthetic two-level nested choice and complete saved-result acceptance.

The choice population is synthetic. Fixed weights standardize postestimation
only; they do not weight the fitted likelihood. No licensed vendor acceptance
or global maximum claim is implied by this example.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import pandas as pd
import torch

import openecon as oe


def synthetic_choices(seed=608, cases=320, offset=0):
    generator = random.Random(seed)
    beta, dissimilarity = (-.65, .48), (.46, .67)
    records = []
    for case in range(cases):
        rows = []
        for alternative in range(5):
            rows.append(dict(choice=offset+case, respondent=case//4,
                             alternative="ABCDE"[alternative], nest="A" if alternative < 3 else "B",
                             chosen=0, available=not (alternative == 2 and case % 4 == 0),
                             cost=.5+4*generator.random(), quality=.5+3*generator.random()))
        utilities = [sum(value*row[name] for value, name in zip(beta, ("cost", "quality"))) for row in rows]
        conditional, inclusive = {}, []
        for nest, lam in zip(("A", "B"), dissimilarity):
            members = [i for i, row in enumerate(rows) if row["nest"] == nest and row["available"]]
            maximum = max(utilities[i]/lam for i in members)
            total = sum(math.exp(utilities[i]/lam-maximum) for i in members)
            inclusive.append(lam*(maximum+math.log(total)))
            for i in members:
                conditional[i] = math.exp(utilities[i]/lam-maximum)/total
        maximum = max(inclusive)
        mass = [math.exp(value-maximum) for value in inclusive]
        mass = [value/sum(mass) for value in mass]
        uniform, cumulative, selected = generator.random(), 0., None
        for i, row in enumerate(rows):
            if row["available"]:
                cumulative += mass[(0 if row["nest"] == "A" else 1)]*conditional[i]
                if selected is None and uniform < cumulative:
                    selected = i
        rows[selected if selected is not None else 4]["chosen"] = 1
        records.extend(rows)
    return pd.DataFrame(records)


def tables(result):
    return {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
            for key, frame in result.items()}


def payload(result):
    return dict(attrs=result.attrs, tables=tables(result), latex=result.to_latex())


def run_acceptance(directory=None, show=True):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        frame = synthetic_choices()
        results, proof = {}, {}
        for vce in ("oim", "hc0", "cr0"):
            result = oe.nlogit(frame, "chosen", ["cost", "quality"], case="choice",
                               alternative="alternative", nest="nest", available="available", vce=vce,
                               cluster="respondent" if vce == "cr0" else None)
            saved = json.loads(json.dumps(payload(result), sort_keys=True, allow_nan=False))
            assert payload(oe.nlogit_restore(saved["attrs"])) == saved
            results[vce] = result
        choices = synthetic_choices(seed=609, cases=3, offset=1000).drop(columns=["chosen", "respondent"])
        results["probabilities"] = oe.nlogit_predict(results["cr0"], data=choices)
        component_targets = [
            dict(case=1000, alternative="A", kind="conditional"),
            dict(case=1000, alternative="A", kind="log_conditional"),
            dict(case=1000, alternative="A", kind="log_probability"),
            dict(case=1000, nest="A", kind="nest_probability"),
            dict(case=1000, nest="A", kind="log_nest_probability"),
            dict(case=1001, nest="B", kind="nest_probability"),
        ]
        results["components"] = oe.nlogit_predict(results["cr0"], data=choices, targets=component_targets)
        margin_targets = [dict(outcome="A", changed="A", attribute="cost"),
                          dict(outcome="B", changed="A", attribute="cost"),
                          dict(outcome="D", changed="A", attribute="cost"),
                          dict(outcome="C", changed="B", attribute="quality")]
        weights = {1000: 1., 1001: 2., 1002: 3.}
        for name, kind in (("effects", "effect"), ("elasticities", "elasticity")):
            results[name] = oe.nlogit_margins(results["cr0"], data=choices, kind=kind,
                                             targets=margin_targets, case_weights=weights)
        saved_fit = json.loads(json.dumps(results["cr0"].attrs, sort_keys=True, allow_nan=False))
        assert payload(oe.nlogit_predict(saved_fit, data=choices)) == payload(results["probabilities"])
        assert payload(oe.nlogit_predict(saved_fit, data=choices, targets=component_targets)) == payload(results["components"])
        for name, kind in (("effects", "effect"), ("elasticities", "elasticity")):
            assert payload(oe.nlogit_margins(saved_fit, data=choices, kind=kind,
                            targets=margin_targets, case_weights=weights)) == payload(results[name])
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            content = json.dumps(payload(result), sort_keys=True, allow_nan=False)
            if directory is not None:
                (directory/(name+".json")).write_text(content, encoding="utf-8")
            keys = ["parameters"] if name in ("oim", "hc0", "cr0") else ["predictions"] if name in ("probabilities", "components") else ["margins"]
            proof[name] = dict(sha256=hashlib.sha256(content.encode()).hexdigest(),
                               table_rows={key: len(value) for key, value in result.items()}, table_order=list(result), displayed_keys=keys,
                               displayed_latex_sha256={key: hashlib.sha256(str(result[key].to_latex()).encode()).hexdigest() for key in keys})
            if show:
                for key in keys:
                    if "display" in globals():
                        globals()["display"](result[key])
                    else:
                        print(name + " — " + key)
                        print(result[key].to_string(index=False))
        forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
        assert not forbidden
        receipt = dict(methods=proof, frozen=bool(getattr(sys, "frozen", False)),
                       all_eight_scopes_verified=True, full_state_replay_verified=True,
                       saved_tables=sum(len(result) for result in results.values()),
                       displayed_tables=7 if show else 0, third_party_estimation_imports=forbidden)
        print("NESTED_LOGIT_ACCEPTANCE_OK "+json.dumps(receipt, sort_keys=True, allow_nan=False))
        return results, receipt
    finally:
        torch.set_num_threads(previous_threads)


if globals().get("NESTED_LOGIT_LIBRARY_ONLY") is not True:
    run_acceptance(globals().get("NESTED_LOGIT_RESULT_DIRECTORY"), globals().get("NESTED_LOGIT_DISPLAY", True))
