"""Capture complete native conjoint uncertainty artifacts for source/wheel comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd
import torch

import openecon as oe


def capture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    attributes = {"brand": ["a", "b"], "price": [10, 20, 30, 40], "quality": ["basic", "premium"]}
    plan = oe.conjoint_plan(attributes)
    rng = random.Random(785792)
    rows = []
    for subject in range(8):
        for i, (card, brand, price, quality) in enumerate(plan["profiles"].itertuples(index=False, name=None)):
            score = (12+.2*subject + (.5+.1*subject)*(1 if brand == "a" else -1)
                     + (-.03+.002*subject)*price-.002*price**2
                     + .4*(1 if quality == "basic" else -1)
                     + rng.gauss(0, .2+.01*price)+.1*math.sin(i))
            rows.append([subject, card, score, i//2])
    responses = pd.DataFrame(rows, columns=["subject", "profile_id", "score", "session"])
    methods = ("nonrobust", "hc0", "hc1", "hc2", "hc3", "cr0", "cr1")
    files, payloads = [], {}

    def save(name, result):
        payload = oe.conjoint_save(result)
        restored = oe.conjoint_load(payload)
        assert oe.conjoint_save(restored) == payload
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        target = directory/(name+".json")
        target.write_bytes(encoded+b"\n")
        latex = result.to_latex()
        (directory/(name+".tex")).write_text(latex, encoding="utf-8")
        assert restored.to_latex() == latex
        files.append({"name": name, "artifact_sha256": payload["sha256"], "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "latex_sha256": hashlib.sha256(latex.encode()).hexdigest(), "tables": len(result),
                      "rows": {key: len(frame) for key, frame in result.items()}})
        payloads[name] = payload
        return restored

    save("plan", plan)
    for method in methods:
        result = oe.conjoint_fit(plan, responses, factors={"price": "ideal"}, covariance=method,
                                **({"cluster": "session"} if method.startswith("cr") else {}))
        restored = save(method+"-fit", result)
        save(method+"-predictions", oe.conjoint_predict(restored, plan))
        save(method+"-respondent-mean", oe.conjoint_group_mean(restored))
        save(method+"-contrasts", oe.conjoint_contrast(restored, {
            "price shift": {"price:linear":10, "price:quadratic":300},
            "brand-quality": {'brand:"a"':2, 'quality:"basic"':-.5},
        }, null={"price shift":-.1, "brand-quality":.7}))
    report = {"package_path": oe.__file__, "complete_artifacts": files,
              "artifact_count": len(files), "covariance_methods": methods,
              "refit_free_json_and_latex_replay": True, "scope": "resident CPU SDK; no installed UI or public release claim"}
    (directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (directory/"complete-results.json").write_text(json.dumps(payloads, sort_keys=True, separators=(",", ":"))+"\n")
    print(json.dumps({"artifact_count":len(files), "complete_tables":sum(f["tables"] for f in files),
                      "report":str(directory/"report.json"), "package_path":oe.__file__}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    capture(args.directory)


if __name__ == "__main__":
    main()
