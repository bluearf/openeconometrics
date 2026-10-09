"""Eight predeclared finite-grid, multiarm and individual-effect targets.

Synthetic randomized assignment is declared here, not inferred from columns.
Every complete typed table and scientific certificate is saved and restored;
the eight rendered tables are representative summaries.
"""
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import torch
import openecon as oe

NAMEORDER = ["randomization_confidence_set", "paired_randomization_confidence_set",
             "cluster_randomization_confidence_set", "bernoulli_neyman_ate",
             "multiarm_neyman_ate", "stratified_multiarm_neyman_ate",
             "treatment_effect_cdf_bounds", "treatment_effect_quantile_bounds"]
REPRESENTATIVE_TABLES = dict.fromkeys(NAMEORDER, "effects")
REPRESENTATIVE_TABLES.update(randomization_confidence_set="profile",
    paired_randomization_confidence_set="profile", cluster_randomization_confidence_set="profile",
    treatment_effect_cdf_bounds="bounds", treatment_effect_quantile_bounds="bounds")
results = {}
rng_before = torch.random.get_rng_state().clone()
experiment = pd.DataFrame({"y": [0., 2., 1., 4., 3., 7.], "d": [0, 1]*3,
                           "pair": ["A", "A", "B", "B", "C", "C"]},
                          index=[f"subject-{i}" for i in range(6)])
results[NAMEORDER[0]] = oe.randomization_confidence_set(
    experiment, "y", "d", candidates=[-2., 0., 1., 2., 4.], design="complete_randomized")
results[NAMEORDER[1]] = oe.paired_randomization_confidence_set(
    experiment, "y", "d", "pair", candidates=[-2., 0., 1., 2., 4.], design="paired_randomized")
clusters = pd.DataFrame({"y": [0., 1., 3., 2., 5., 4.], "d": [0, 0, 1, 0, 1, 1],
                         "cluster": ["A", "A", "B", "C", "D", "D"]})
results[NAMEORDER[2]] = oe.cluster_randomization_confidence_set(
    clusters, "y", "d", "cluster", candidates=[-2., 0., 1., 2., 4.], design="cluster_randomized")
bernoulli = pd.DataFrame({"y": [0., 2., -1., 3., 1., 4.], "d": [0, 1, 1, 0, 1, 0],
                          "probability": [.2, .35, .5, .65, .8, .4]})
results[NAMEORDER[3]] = oe.bernoulli_neyman_ate(
    bernoulli, "y", "d", "probability", design="bernoulli_randomized")
multiarm = pd.DataFrame({"y": [0., 2., 1., 4., 3., 7.],
                        "arm": ["control", "control", "low", "low", "high", "high"]})
contrasts = {"low_vs_control": [-1., 1., 0.], "high_vs_control": [-1., 0., 1.],
             "high_vs_low": [0., -1., 1.]}
results[NAMEORDER[4]] = oe.multiarm_neyman_ate(
    multiarm, "y", "arm", arms=["control", "low", "high"], design="complete_randomized",
    contrasts=contrasts)
stratified = pd.concat([multiarm.assign(stratum="A"),
                        multiarm.assign(y=multiarm.y + 1., stratum="B")], ignore_index=True)
results[NAMEORDER[5]] = oe.stratified_multiarm_neyman_ate(
    stratified, "y", "arm", "stratum", arms=["control", "low", "high"],
    design="stratified_randomized", contrasts=contrasts)
discrete = pd.DataFrame({"y": [0., 1., 2., 0., 1., 2.], "d": [0, 0, 0, 1, 1, 1]})
results[NAMEORDER[6]] = oe.treatment_effect_cdf_bounds(
    discrete, "y", "d", support=[0., 1., 2.], thresholds=[-1., 0., 1.], design="randomized")
results[NAMEORDER[7]] = oe.treatment_effect_quantile_bounds(
    discrete, "y", "d", support=[0., 1., 2.], quantiles=[.25, .5, .75], design="randomized")
assert list(results) == NAMEORDER
assert torch.equal(torch.random.get_rng_state(), rng_before)
artifact_directory = Path(globals().get("ARTIFACT_DIRECTORY") or tempfile.mkdtemp(prefix="openecon-causal-confidence-eight-"))
artifact_directory.mkdir(parents=True, exist_ok=True)
show = globals().get("display", lambda table: print(table.to_string(index=False)))
artifact_sha256s, table_counts = {}, {}
for name, output in results.items():
    path = artifact_directory / (name + ".json")
    artifact = oe.causal_design_save(output, path)
    restored = oe.causal_design_load(path)
    assert oe.causal_design_save(restored) == artifact
    assert json.loads(path.read_text()) == artifact
    assert output.attrs == restored.attrs
    assert list(output) == list(restored)
    for key in output:
        pd.testing.assert_frame_equal(output[key], restored[key], check_exact=True)
    assert output.attrs["stata_parity_validated"] is False
    artifact_sha256s[name], table_counts[name] = artifact["sha256"], len(output)
    show(output[REPRESENTATIVE_TABLES[name]])
print("CAUSAL_CONFIDENCE_EIGHT_OK " + json.dumps(dict(
    nameorder=NAMEORDER, artifact_sha256s=artifact_sha256s, full_roundtrip_equal=True,
    frozen=bool(getattr(sys, "frozen", False)), table_counts=table_counts), sort_keys=True))
