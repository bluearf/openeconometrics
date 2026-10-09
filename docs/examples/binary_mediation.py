"""Eight synthetic binary mediator models, complete saved state and no-fit replay.

All data are synthetic. Display sixteen complete parameter/effect tables;
save all fourteen tables per model, their attributes and full LaTeX exports.
Set BINARY_MEDIATION_RESULT_DIRECTORY to retain the eight JSON files.
"""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

import openecon as oe


METHODS = (
    "logit_gaussian", "logit_logit", "logit_probit", "logit_poisson",
    "probit_gaussian", "probit_logit", "probit_probit", "probit_poisson",
)
DOMAIN_LABELS = (
    "Training, certification and synthetic earnings",
    "Reminders, attendance and synthetic completion probability",
    "Advising, application and synthetic admission probability",
    "Outreach, engagement and synthetic service visits",
    "Coaching, participation and synthetic performance",
    "Information, uptake and synthetic retention probability",
    "Support, enrollment and synthetic graduation probability",
    "Scheduling, access and synthetic event counts",
)
EXPECTED_TABLES = {
    "inputs", "parameters", "information", "bread", "scores",
    "model_loglikelihood", "covariance", "counterfactual_rows", "means",
    "means_covariance", "effects", "effects_covariance", "delta_jacobian",
    "fit_summary",
}


def table_payload(result):
    return {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
            for key, frame in result.items()}


def synthetic_data(seed, link, outcome, interaction):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = 240
    baseline = torch.randn(n, generator=generator, dtype=torch.float64, device="cpu")
    treatment = (torch.rand(n, generator=generator, dtype=torch.float64, device="cpu") < .5).to(torch.float64)
    mediator_eta = -.3+.65*treatment+.25*baseline
    mediator_p = (torch.sigmoid(mediator_eta) if link == "logit"
                  else .5*(1+torch.erf(mediator_eta/2**.5)))
    mediator = torch.bernoulli(mediator_p, generator=generator)
    outcome_eta = -.15+.35*treatment+.55*mediator+.2*baseline
    if interaction:
        outcome_eta = outcome_eta-.2*treatment*mediator
    if outcome == "gaussian":
        y = outcome_eta+.7*torch.randn(n, generator=generator, dtype=torch.float64, device="cpu")
    elif outcome == "poisson":
        y = torch.poisson(torch.exp(outcome_eta), generator=generator)
    else:
        outcome_p = (torch.sigmoid(outcome_eta) if outcome == "logit"
                     else .5*(1+torch.erf(outcome_eta/2**.5)))
        y = torch.bernoulli(outcome_p, generator=generator)
    return pd.DataFrame({"outcome": y.tolist(), "treatment": treatment.tolist(),
                         "mediator": mediator.tolist(), "baseline": baseline.tolist()})


previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    proof, encoded, results = {}, {}, {}
    displayed_tables = 0
    for index, name in enumerate(METHODS):
        link, outcome = name.split("_")
        interaction = True
        data = synthetic_data(1800+index, link, outcome, interaction)
        result = oe.mediation_binary(
            data=data, y="outcome", treatment="treatment", mediator="mediator", controls=["baseline"],
            mediator_link=link, outcome_model=outcome,
            interaction=interaction, covariance="HC0",
        )
        assert result.attrs["contract"] == "binary_mediation_v1"
        assert set(result) == EXPECTED_TABLES
        payload = {"attrs": result.attrs, "tables": table_payload(result), "latex": result.to_latex()}
        content = json.dumps(payload, sort_keys=True, allow_nan=False)
        saved = json.loads(content)
        # This public helper validates complete saved state and never refits.
        restored = oe.mediation_binary_restore(saved["attrs"])
        assert restored.attrs == saved["attrs"]
        assert table_payload(restored) == saved["tables"]
        assert restored.to_latex() == saved["latex"]
        for covariance_key in ("covariance", "means_covariance", "effects_covariance"):
            assert restored[covariance_key].equals(result[covariance_key])
        assert len(result["inputs"]) == len(data)
        assert len(result["effects"]) == 5 and len(result["means"]) == 4
        encoded[name], results[name] = content, result
        proof[name] = {
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "table_rows": {key: len(frame) for key, frame in result.items()},
            "displayed_keys": ["parameters", "effects"],
            "interaction": interaction,
        }
        for key in ("parameters", "effects"):
            if "display" in globals():
                globals()["display"](result[key])
            else:
                print(DOMAIN_LABELS[index]+" — "+key)
                print(result[key].to_string(index=False))
            displayed_tables += 1
    if "BINARY_MEDIATION_RESULT_DIRECTORY" in globals():
        directory = Path(globals()["BINARY_MEDIATION_RESULT_DIRECTORY"])
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in encoded.items():
            (directory/(name+".json")).write_text(content, encoding="utf-8")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden and displayed_tables == 16
    print("BINARY_MEDIATION_ACCEPTANCE_OK "+json.dumps({
        "methods": proof,
        "frozen": bool(getattr(sys, "frozen", False)),
        "all_eight_contracts_verified": True,
        "full_state_replay_verified": True,
        "saved_tables": sum(len(result) for result in results.values()),
        "displayed_tables": displayed_tables,
        "third_party_estimation_imports": forbidden,
    }, sort_keys=True, allow_nan=False))
finally:
    torch.set_num_threads(previous_threads)
