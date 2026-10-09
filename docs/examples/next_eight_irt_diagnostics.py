"""Runnable synthetic IRT diagnostics and complete portable-state readback.

Use from the repository with PYTHONPATH=src:packages/openecon-charts/src.
No vendor execution, external estimator or desktop acceptance is asserted.
"""

from pathlib import Path
import argparse

import torch
import openecon as oe


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(20261007)
    theta = torch.randn(600, dtype=torch.float64, device="cpu", generator=generator)
    probability = torch.sigmoid(theta[:, None] - torch.tensor([-.8, -.2, .3, .9], dtype=torch.float64, device="cpu"))
    responses = (torch.rand((600, 4), dtype=torch.float64, device="cpu", generator=generator) < probability).long()
    frame = oe.DataFrame(responses.tolist(), columns=["a", "b", "c", "d"])
    model = oe.irt_rasch(data=frame, items=["a", "b", "c", "d"])
    fit = oe.irt_fit_diagnostics(model, data=frame)
    # The matching score is selected before analysis. All four items are
    # included here; a distinct prespecified matching test is also possible.
    frame["match"] = responses.sum(1).tolist()
    frame["group"] = ["reference"] * 300 + ["focal"] * 300
    dif = oe.irt_mh_dif(frame, items=["a", "b"], group="group", reference="reference",
                       focal="focal", match="match", continuity=True)
    for name, result in (("fit", fit), ("dif", dif)):
        path = output / (name + ".json")
        path.write_text(result.to_json())
        restored = oe.irt_diagnostics_restore(path.read_text())
        assert restored.to_json() == result.to_json()
        assert restored.to_latex() == result.to_latex()
        for table in result:
            assert restored[table].equals(result[table])
    return fit, dif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fit, dif = run(args.output)
    print(fit["items"])
    print(fit["pairs"])
    print(dif["dif"])
