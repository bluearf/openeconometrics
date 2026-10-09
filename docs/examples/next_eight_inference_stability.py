"""Native saved summaries and actual complete JSON file readback."""

from pathlib import Path
import sys

import pandas as pd
import torch

import openecon as oe
from openecon.models import ResultBundle


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(431)
    n = 180
    shock = torch.randn(n, dtype=torch.float64, generator=generator)
    noise = torch.randn(n, dtype=torch.float64, generator=generator)
    frame = pd.DataFrame({"t": list(range(n)), "shock": shock.tolist(),
                          "y": (.4*shock+noise).tolist()})
    lp = oe.lp(data=frame, y="y", x=["shock"], time="t", horizons=4)
    lp_path = destination / "lp-result.json"
    lp_path.write_text(lp.model_dump_json())
    restored = ResultBundle.model_validate_json(lp_path.read_text())
    bands = oe.trajectory_bands(restored, draws=5000,
        assumptions="Prespecified horizons, exogenous shock, sufficient weak-dependence HAC CLT")
    stability = oe.ols_cusum(frame, "y", ["shock"], time="t", replications=199)
    groups, size = 16, 8
    x = torch.linspace(-1, 1, size, dtype=torch.float64).repeat(groups)
    random = torch.randn(groups, dtype=torch.float64, generator=generator).repeat_interleave(size)
    errors = torch.randn(groups*size, dtype=torch.float64, generator=generator)
    mixed_frame = pd.DataFrame({"group": torch.arange(groups).repeat_interleave(size).tolist(),
                                "x": x.tolist(), "y": (2*random+.7*x+errors).tolist()})
    mixed = oe.mixed(data=mixed_frame, y="y", x=["x"], group="group", method="reml")
    mixed_path = destination / "mixed-result.json"
    mixed_path.write_text(mixed.model_dump_json())
    scalar = oe.mixed_satterthwaite(ResultBundle.model_validate_json(mixed_path.read_text()),
                                   data=mixed_frame, contrast={"x": 1})
    sizes = {}
    for name, output in {"trajectory": bands, "ols-cusum": stability, "mixed-contrast": scalar}.items():
        path = destination / f"{name}.json"
        path.write_text(oe.summary_state(output))
        replay = oe.restore_summary(path.read_text())
        assert oe.summary_state(replay) == path.read_text()
        assert replay.attrs["device"] == "cpu"
        (destination / f"{name}.tex").write_text(replay.to_latex())
        sizes[name] = {key: len(table) for key, table in replay.items()}
    return sizes


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/openecon-next-eight-inference"))
