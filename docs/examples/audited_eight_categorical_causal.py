"""Complete synthetic categorical, scored-conjoint and causal-design replay.

Call run(destination) from the native code panel or an installed SDK. Every
result writes its complete typed JSON artifact and concatenated LaTeX, then
reopens both. The fixed query/assignment targets are declarations, not inferred
causal identification or population market shares. No external estimator runs.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
import tempfile

import pandas as pd
import torch

import openecon as oe


def fixtures():
    """Create only synthetic resident inputs using a private CPU generator."""
    with torch.device("cpu"):
        generator = torch.Generator(device="cpu").manual_seed(75860)
        people = pd.DataFrame(
            list(product(["red", "green", "blue"], [0, 1, 2], ["left", "right"], range(4))),
            columns=["a", "b", "c", "replicate"],
        )
        people.index = [f"synthetic-person-{i:03d}" for i in range(len(people))]
        people["x"] = torch.randn(len(people), dtype=torch.float64, generator=generator).tolist()
        noise = torch.randn(len(people), dtype=torch.float64, generator=generator).tolist()
        people["y"] = [
            3 + {"red": 1.5, "green": -.4, "blue": -1.1}[a] + .7*b - .4*x + .15*e
            for a, b, x, e in zip(people.a, people.b, people.x, noise, strict=True)
        ]
        people.loc[people.index[5], "a"] = None
        cells = pd.DataFrame(list(product(range(2), range(3), range(2))), columns=["a", "b", "c"])
        cells["count"] = [21, 13, 35, 20, 18, 31, 27, 30, 17, 16, 25, 0]
        cells["structural"] = [False]*11 + [True]
        points = torch.tensor(
            [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., .4], [.3, 2.]], dtype=torch.float64,
        )
        names = [f"synthetic-object-{i}" for i in range(len(points))]
        distances = pd.DataFrame(torch.cdist(points, points).square().tolist(), index=names, columns=names)

        n = 120
        experiment = pd.DataFrame({
            "x": torch.randn(n, dtype=torch.float64, generator=generator).tolist(),
            "baseline": torch.randn(n, dtype=torch.float64, generator=generator).tolist(),
            "treated": torch.randint(0, 2, (n,), generator=generator).tolist(),
        }, index=[f"synthetic-unit-{i:03d}" for i in range(n)])
        errors = torch.randn(n, dtype=torch.float64, generator=generator).tolist()
        experiment["outcome"] = [
            1 + 1.2*t + .4*x + .2*b + e
            for t, x, b, e in zip(experiment.treated, experiment.x, experiment.baseline, errors, strict=True)
        ]
        survival = experiment.loc[:, ["treated"]].copy()
        event = torch.exp(1 + .25*torch.tensor(experiment.treated.tolist())
                          + .5*torch.randn(n, dtype=torch.float64, generator=generator))
        censor = 1.5 + 4*torch.rand(n, dtype=torch.float64, generator=generator)
        survival["time"], survival["event"] = torch.minimum(event, censor).tolist(), (event <= censor).to(torch.int64).tolist()
        paired = pd.DataFrame({
            "pair": [f"synthetic-pair-{i}" for i in range(6) for _ in range(2)],
            "treated": [1, 0]*6,
            "outcome": [2., 1., 1.2, 1.4, 3., 1.5, 2.2, 1.1, 1.8, 1.8, 2.5, 1.7],
        }, index=[f"synthetic-paired-unit-{i}" for i in range(12)])
    return people, cells, distances, experiment, paired, survival


def analyze():
    """Fit each native core stage and retain all saved-map/profile artifacts."""
    people, cells, distances, experiment, paired, survival = fixtures()
    inputs = {name: value.copy(deep=True) for name, value in (
        ("people", people), ("cells", cells), ("dissimilarities", distances),
        ("experiment", experiment), ("paired", paired), ("survival", survival),
    )}
    results = {}
    results["categorical/nominal"] = oe.catreg_nominal(
        people, "y", ["a", "b", "x"], scales={"a": "nominal", "b": "nominal", "x": "numeric"},
        n_starts=2, seed=17,
    )
    results["categorical/ordinal"] = oe.catreg_ordinal(
        people, "y", ["b", "x"], scales={"b": "ordinal", "x": "numeric"}, orders={"b": [0, 1, 2]},
        n_starts=2, seed=17,
    )
    results["categorical/catpca"] = oe.catpca(
        people, ["a", "b", "c", "x"], components=2,
        scales={"a": "nominal", "b": "ordinal", "c": "nominal", "x": "numeric"},
        orders={"b": [0, 1, 2]}, n_starts=2, seed=17, tol=1e-6,
    )
    results["categorical/mca"] = oe.mca(people, ["a", "b", "c"], n_components=2)
    results["categorical/overals"] = oe.overals(
        people, [["a", "b"], ["c", "x"]], n_components=2,
        scales={"a": "multiple_nominal", "b": "ordinal", "c": "nominal", "x": "numeric"},
        orders={"b": [0, 1, 2]}, n_starts=2, seed=17, tol=1e-6,
    )
    results["categorical/nonmetric_mds"] = oe.mds_nonmetric(
        distances, n_components=2, ties="secondary", zero="include", n_starts=2,
        seed=83, max_iter=1000, tol=1e-6,
    )
    levels = {"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]}
    results["categorical/ipf"] = oe.loglinear_ipf(
        cells, ["a", "b", "c"], "count", levels=levels,
        margins=[["a", "b"], ["b", "c"]], structural="structural",
    )
    design = [[float(i == j) for j in range(11)] for i in range(12)]
    results["categorical/general_ml"] = oe.loglinear_ml(
        cells, ["a", "b", "c"], "count", levels=levels, structural="structural",
        design=design, terms=[f"active-cell-{i}" for i in range(11)],
    )
    for name in ("nominal", "ordinal", "catpca", "mca"):
        stored = oe.restore_summary(oe.summary_state(results["categorical/"+name]))
        function = oe.catreg_predict if name in ("nominal", "ordinal") else oe.catpca_predict if name == "catpca" else oe.mca_project
        results["categorical/"+name+"_projection"] = function(stored, people, missing="drop")
    results["categorical/nested_lr"] = oe.loglinear_compare(
        oe.restore_summary(oe.summary_state(results["categorical/ipf"])),
        oe.restore_summary(oe.summary_state(results["categorical/general_ml"])),
    )

    attributes = {"brand": ["a", "b"], "price": [9, 15, 24, 39], "quality": ["basic", "premium"]}
    full = oe.conjoint_plan(attributes)
    profiles = full["profiles"]
    holdout_positions = [0, 7, 10, 13]
    held = profiles.iloc[holdout_positions].copy()
    training = profiles.drop(profiles.index[holdout_positions]).copy()
    training_ids = set(training.profile_id)
    responses, validation = [], []
    for person, (brand_effect, linear, curvature, quality_effect) in zip(
        ["person-z", "person-a", "person-m"], [(1., -.03, -.01, .6), (-.6, .1, -.005, .2), (.4, -.05, .002, 1.)], strict=True,
    ):
        for i, (card, brand, price, quality) in enumerate(profiles.itertuples(index=False, name=None)):
            score = 12 + brand_effect*(1 if brand == "a" else -1) + linear*price + curvature*price**2
            score += quality_effect*(1 if quality == "premium" else -1) + .08*((i*7) % 5 - 2)
            (responses if card in training_ids else validation).append([person, card, score])
    responses = pd.DataFrame(responses[::-1], columns=["subject", "profile_id", "score"])
    validation = pd.DataFrame(validation, columns=responses.columns)
    results["conjoint/full_plan"] = full
    fraction = oe.conjoint_orthogonal({a: [-1, 1] for a in "ABCD"}, list("ABC"), {"D": list("ABC")})
    results["conjoint/regular_fraction"] = fraction
    results["conjoint/design_diagnostics"] = oe.conjoint_diagnostics(fraction)
    fit = oe.conjoint_fit(training, responses, attributes, factors={"price": "ideal"})
    results["conjoint/fit"] = fit
    restored = oe.conjoint_load(oe.conjoint_save(fit))
    results["conjoint/predictions"] = oe.conjoint_predict(restored, held)
    results["conjoint/holdout"] = oe.conjoint_holdout(restored, held, validation)
    results["conjoint/importance"] = oe.conjoint_importance(restored)
    results["conjoint/logit_shares"] = oe.conjoint_simulate(restored, held, method="logit", temperature=1.25)
    inputs.update({"conjoint_training": training, "conjoint_holdout": held, "conjoint_responses": responses,
                   "conjoint_validation": validation})

    entropy = oe.ebalance(experiment, "treated", ["baseline", "x"])
    results["causal_design/ebalance"] = entropy
    results["causal_design/cem"] = oe.cem(
        experiment, "treated", ["baseline", "x"], cutpoints={"baseline": [-.5, .5], "x": [-.5, .5]},
    )
    diagnostic = experiment.copy()
    weights = entropy["weights"].set_index("position").weight
    diagnostic["balance_weight"] = [float(weights.loc[i]) for i in range(len(diagnostic))]
    results["causal_design/balance"] = oe.balance(diagnostic, "treated", ["baseline", "x"], balance_weights="balance_weight")
    results["causal_design/rosenbaum_bounds"] = oe.rosenbaum_bounds(paired, "outcome", "treated", "pair", gammas=[1., 1.5, 2.])
    results["causal_design/paired_randomization"] = oe.paired_randomization(
        paired, "outcome", "treated", "pair", design="paired_randomized", method="exact",
    )
    results["causal_design/treatment_cdf"] = oe.treatment_cdf(
        experiment, "outcome", "treated", design="randomized", thresholds=[-1., 0., 1., 2., 3., 4.],
    )
    results["causal_design/treatment_quantile"] = oe.treatment_quantile(
        experiment, "outcome", "treated", design="randomized", quantiles=[.25, .5, .75], reps=199, seed=1729,
    )
    results["causal_design/treatment_rmst"] = oe.treatment_rmst(
        survival, "time", "event", "treated", design="randomized", tau=2.,
    )
    for name, frame in (("people", people), ("cells", cells), ("dissimilarities", distances),
                        ("experiment", experiment), ("paired", paired), ("survival", survival)):
        pd.testing.assert_frame_equal(frame, inputs[name], check_exact=True)
    return results, inputs


def run(destination):
    """Persist and reopen all 29 full result artifacts and their exact LaTeX."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results, inputs = analyze()
    record = {"issues": [58, 60, 75], "result_count": len(results), "files": {},
              "all_complete_roundtrips_equal": True, "all_latex_roundtrips_equal": True,
              "device": "cpu", "precision": "float64", "vendor_parity_validated": False,
              "new_frozen_or_native_claimed": False, "results": {}}
    for name, result in results.items():
        path = destination / (name+".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("categorical/"):
            text = oe.summary_state(result)
            path.write_text(text, encoding="utf-8")
            restored = oe.restore_summary(path.read_text(encoding="utf-8"))
            assert oe.summary_state(restored) == text
        else:
            save, load = (oe.conjoint_save, oe.conjoint_load) if name.startswith("conjoint/") else (oe.causal_design_save, oe.causal_design_load)
            artifact = save(result, path)
            restored = load(path)
            assert save(restored) == artifact == json.loads(path.read_text(encoding="utf-8"))
        assert restored.attrs == result.attrs
        assert list(restored) == list(result)
        for table_name in result:
            pd.testing.assert_frame_equal(restored[table_name], result[table_name], check_exact=True)
        latex = result.to_latex()
        assert restored.to_latex() == latex
        tex = path.with_suffix(".tex")
        tex.write_text(latex, encoding="utf-8")
        assert tex.read_text(encoding="utf-8") == latex
        record["results"][name] = {"tables": {k: len(result[k]) for k in result},
                                   "state_replayed": True, "table_dtypes_replayed": True}
    input_path = destination / "inputs.json"
    input_path.write_text(json.dumps({name: json.loads(frame.to_json(orient="table")) for name, frame in inputs.items()},
                                    indent=2, allow_nan=False), encoding="utf-8")
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "receipt.json":
            record["files"][str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    target = globals().get("ARTIFACT_DIRECTORY", tempfile.mkdtemp(prefix="openecon-audited-categorical-causal-"))
    print("AUDITED_CATEGORICAL_CAUSAL_OK "+json.dumps(run(target), sort_keys=True, allow_nan=False))
