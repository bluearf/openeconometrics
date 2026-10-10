"""Save every complete conjoint bootstrap result for source/installed-wheel comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import pandas as pd
import torch

import openecon as oe


def capture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    attributes = {"zbrand":["a","b"], "price":[10,20,30,40], "aquality":["basic","premium"]}
    profiles = oe.conjoint_plan(attributes)["profiles"]
    plan = pd.concat([profiles]*4, ignore_index=True)
    plan.profile_id = [f"card-{i}" for i in range(len(plan))]
    rng = random.Random(805812)
    rows = []
    for i, who in enumerate([17, "17", "z", 5.5]):
        for card, brand, price, quality in plan.itertuples(index=False, name=None):
            score = (16+i*.2+(.7+i*.5)*(1 if brand == "a" else -1)+(-.02+i*.003)*price
                     +(-.001+i*.0001)*price**2+(.3-i*.15)*(1 if quality == "basic" else -1)
                     +rng.gauss(0,.3))
            rows.append([who, card, score])
    data = pd.DataFrame(rows, columns=["subject","profile_id","score"], index=[f"row-{i}" for i in range(len(rows))])
    result = oe.conjoint_fit(plan, data, attributes, factors={"price":"ideal"})
    payloads, files = {}, []

    def save(name, output):
        artifact = oe.conjoint_save(output)
        restored = oe.conjoint_load(artifact)
        assert oe.conjoint_save(restored) == artifact
        latex = output.to_latex()
        assert restored.to_latex() == latex
        raw = (json.dumps(artifact, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()
        (directory/(name+".json")).write_bytes(raw)
        (directory/(name+".tex")).write_text(latex, encoding="utf-8")
        payloads[name] = artifact
        files.append({"name":name, "artifact_sha256":artifact["sha256"], "file_sha256":hashlib.sha256(raw).hexdigest(),
                      "latex_sha256":hashlib.sha256(latex.encode()).hexdigest(),
                      "tables":len(output), "rows":{key:len(frame) for key,frame in output.items()}})
        return restored

    restored = save("source-fit", result)
    for method in ("residual","rademacher","mammen","pairs"):
        save(method, oe.conjoint_bootstrap_fit(restored, method=method, replications=199, seed=812))
    save("importance", oe.conjoint_bootstrap_importance(restored, replications=199, seed=812))
    for method in ("first_choice","btl","logit"):
        save(method, oe.conjoint_bootstrap_shares(restored, plan, method=method, replications=199, seed=812))
    fixtures = {"attributes":attributes, "plan":plan.astype(object).to_dict("split"), "responses":data.astype(object).to_dict("split")}
    (directory/"fixtures.json").write_text(json.dumps(fixtures, sort_keys=True, allow_nan=False)+"\n")
    (directory/"complete-results.json").write_text(json.dumps(payloads, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n")
    report = {"package_path":oe.__file__, "artifact_count":len(files), "complete_table_count":sum(item["tables"] for item in files),
              "complete_artifacts":files, "complete_json_and_latex_replay":True,
              "scope":"source/installed-wheel CPU SDK; no frozen/desktop/public release claim"}
    (directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("package_path","artifact_count","complete_table_count")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    capture(args.directory)


if __name__ == "__main__":
    main()
