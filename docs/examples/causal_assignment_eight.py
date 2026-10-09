"""Eight design-based assignment/ATE and population-bound methods, synthetic data.

Every declaration is part of the example design; labels alone cannot establish
randomization, IID sampling or monotone selection. All scientific tables and
state are saved and restored exactly; the eight visible tables are summaries.
"""
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import torch
import openecon as oe

NAMEORDER = ["cluster_randomization", "bernoulli_randomization", "neyman_ate",
             "stratified_neyman_ate", "cluster_neyman_ate", "paired_neyman_ate",
             "manski_ate_inference", "stratified_lee_bounds"]
REPRESENTATIVE_TABLES = dict.fromkeys(NAMEORDER, "effects")
REPRESENTATIVE_TABLES.update(cluster_randomization="test", bernoulli_randomization="test",
                              manski_ate_inference="bounds", stratified_lee_bounds="bounds")
results = {}
rng_before = torch.random.get_rng_state().clone()
# Unequal cluster sizes; fixed original N under six-cluster assignment.
clusters = pd.DataFrame({
    "y": [1., 3., 5., 2., 4., 7., 0., 2., 4., 1., 6., 8.],
    "d": [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    "cluster": ["A", "B", "B", "C", "C", "C", "D", "E", "E", "F", "F", "F"],
}, index=[f"cluster-unit-{i}" for i in range(12)])
results["cluster_randomization"] = oe.cluster_randomization(
    clusters, "y", "d", "cluster", design="cluster_randomized", null_effect=.25)
bernoulli = pd.DataFrame({"y": [0., 2., -1., 3., 1., 4.], "d": [0, 1, 1, 0, 1, 0],
                          "probability": [.2, .35, .5, .65, .8, .4]})
results["bernoulli_randomization"] = oe.bernoulli_randomization(
    bernoulli, "y", "d", "probability", design="bernoulli_randomized", null_effect=.5)
experiment = pd.DataFrame({"y": [0., 2., 1., 4., 3., 7., 1., 5., 3., 8., 4., 6.],
                           "d": [0, 1]*6, "stratum": ["A"]*6+["B"]*6,
                           "pair": [f"pair-{i//2}" for i in range(12)]},
                          index=[f"subject-{i}" for i in range(12)])
results["neyman_ate"] = oe.neyman_ate(experiment, "y", "d", design="complete_randomized")
results["stratified_neyman_ate"] = oe.stratified_neyman_ate(
    experiment, "y", "d", "stratum", design="stratified_randomized")
results["cluster_neyman_ate"] = oe.cluster_neyman_ate(
    clusters, "y", "d", "cluster", design="cluster_randomized")
results["paired_neyman_ate"] = oe.paired_neyman_ate(
    experiment, "y", "d", "pair", design="paired_randomized")
bounded = pd.DataFrame({"y": [.1, .3, .6, .8, .2, .9, .4, .7], "d": [0, 0, 0, 1, 0, 1, 1, 1]})
results["manski_ate_inference"] = oe.manski_ate_inference(
    bounded, "y", "d", lower=0., upper=1., sampling="iid")
# Different treatment fractions across cells; selected control counts alone
# do not define the always-selected target's covariate distribution.
lee = pd.DataFrame({
    "stratum": ["A"]*8 + ["B"]*12,
    "d": [0]*4 + [1]*4 + [0]*8 + [1]*4,
    "s": [1,1,0,0,1,1,1,0] + [1,1,1,1,0,0,0,0,1,1,1,0],
    "y": [0.,2.,None,None,1.,2.,3.,None] + [1.,2.,3.,4.,None,None,None,None,2.,4.,6.,None],
}, index=[f"selection-{i}" for i in range(20)])
results["stratified_lee_bounds"] = oe.stratified_lee_bounds(
    lee, "y", "d", "s", strata="stratum", levels=["A", "B"],
    design="randomized_within_strata", monotonicity="increasing")
assert list(results) == NAMEORDER
assert torch.equal(torch.random.get_rng_state(), rng_before)
artifact_directory = Path(globals().get("ARTIFACT_DIRECTORY") or tempfile.mkdtemp(prefix="openecon-causal-assignment-eight-"))
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
print("CAUSAL_ASSIGNMENT_EIGHT_OK " + json.dumps(dict(
    nameorder=NAMEORDER, artifact_sha256s=artifact_sha256s, full_roundtrip_equal=True,
    frozen=bool(getattr(sys, "frozen", False)), table_counts=table_counts), sort_keys=True))
