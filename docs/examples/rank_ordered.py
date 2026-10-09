"""Synthetic ranking, robust inference and portable postestimation acceptance.

All input is synthetic. Save complete tables, attributes and LaTeX when
RANK_ORDERED_RESULT_DIRECTORY is set; display selected complete small tables.
"""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

import openecon as oe


def synthetic_rankings(seed=545, cases=100):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for case in range(cases):
        size = 3 + case % 2
        price = 1 + 3*torch.rand(size, generator=generator, dtype=torch.float64, device="cpu")
        quality = torch.randn(size, generator=generator, dtype=torch.float64, device="cpu")
        available = [True]*size
        if case % 5 == 0:
            available[-1] = False
        uniforms = torch.rand(size, generator=generator, dtype=torch.float64, device="cpu")
        utilities = -.55*price+.7*quality-torch.log(-torch.log(uniforms))
        order = sorted((j for j in range(size) if available[j]), key=lambda j: -float(utilities[j]))
        prefix = order if case % 3 == 0 else order[:2]
        ranks = {j: rank for rank, j in enumerate(prefix, 1)}
        for j in range(size):
            rows.append(dict(choice=case, respondent=case//2, alternative="ABCD"[j],
                             rank=ranks.get(j, 0), available=available[j],
                             price=float(price[j]), quality=float(quality[j])))
    return pd.DataFrame(rows)


def tables(result):
    return {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
            for key, frame in result.items()}


def payload(result):
    return dict(attrs=result.attrs, tables=tables(result), latex=result.to_latex())


previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    frame = synthetic_rankings()
    results, proof = {}, {}
    for vce in ("oim", "hc0", "cr0"):
        result = oe.rologit(frame, "rank", ["price", "quality"], case="choice",
                            alternative="alternative", available="available", vce=vce,
                            cluster="respondent" if vce == "cr0" else None)
        saved = json.loads(json.dumps(payload(result), sort_keys=True, allow_nan=False))
        restored = oe.rologit_restore(saved["attrs"])
        assert payload(restored) == saved
        results[vce] = result
    # New keys are intentionally absent from the fitted cases. Unranked rows
    # remain in the stage denominator. Query matrices fit the explicit budget.
    ranked = synthetic_rankings(seed=547, cases=2)
    ranked["choice"] += 1000
    choices = ranked.drop(columns=["rank", "respondent"])
    results["first"] = oe.rologit_predict(results["hc0"], data=choices, mode="first")
    results["stages"] = oe.rologit_predict(results["hc0"], data=ranked, mode="stages")
    margin_targets = [(1, "A", "A", "price"), (1, "B", "A", "price"),
                      (1, "A", "B", "price"), (1, "B", "B", "price")]
    for name, kind in (("effects", "effect"), ("elasticities", "elasticity")):
        results[name] = oe.rologit_margins(results["cr0"], data=choices, kind=kind,
                                         targets=margin_targets, case_weights={1000: 1., 1001: 2.})
    # Querying a canonical saved fit must reproduce all tables, including every
    # covariance/Jacobian block, with no optimizer call or new stochastic draws.
    saved_fit = json.loads(json.dumps(results["hc0"].attrs, sort_keys=True, allow_nan=False))
    assert payload(oe.rologit_predict(saved_fit, data=choices)) == payload(results["first"])
    assert payload(oe.rologit_predict(saved_fit, data=ranked, mode="stages")) == payload(results["stages"])
    saved_cluster = json.loads(json.dumps(results["cr0"].attrs, sort_keys=True, allow_nan=False))
    for name, kind in (("effects", "effect"), ("elasticities", "elasticity")):
        assert payload(oe.rologit_margins(saved_cluster, data=choices, kind=kind,
                        targets=margin_targets, case_weights={1000: 1., 1001: 2.})) == payload(results[name])
    directory = globals().get("RANK_ORDERED_RESULT_DIRECTORY")
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    displayed = 0
    for name, result in results.items():
        content = json.dumps(payload(result), sort_keys=True, allow_nan=False)
        if directory is not None:
            (directory/(name+".json")).write_text(content, encoding="utf-8")
        keys = ["parameters"] if name in ("oim", "hc0", "cr0") else ["predictions"] if name in ("first", "stages") else ["margins"]
        proof[name] = dict(sha256=hashlib.sha256(content.encode()).hexdigest(),
                           table_rows={key: len(value) for key, value in result.items()}, displayed_keys=keys,
                           displayed_latex_sha256={key: hashlib.sha256(str(result[key].to_latex()).encode()).hexdigest() for key in keys})
        for key in keys:
            if "display" in globals():
                globals()["display"](result[key])
            else:
                print(name + " — " + key)
                print(result[key].to_string(index=False))
            displayed += 1
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    print("RANK_ORDERED_ACCEPTANCE_OK "+json.dumps(dict(
        methods=proof, frozen=bool(getattr(sys, "frozen", False)),
        all_four_scopes_verified=True, full_state_replay_verified=True,
        saved_tables=sum(len(result) for result in results.values()), displayed_tables=displayed,
        third_party_estimation_imports=forbidden), sort_keys=True, allow_nan=False))
finally:
    torch.set_num_threads(previous_threads)
