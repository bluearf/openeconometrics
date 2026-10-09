"""Eight fixed-calibration IRT stages, with complete source/runtime payloads."""

from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
ISSUES = tuple(f"MARKET-{i}" for i in range(473, 481))


def save(result, stem, output_dir):
    import openecon as oe

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored.to_latex() == result.to_latex(), stem + " complete LaTeX differs"
    for key, frame in result.items():
        assert restored[key].equals(frame), stem + " table differs: " + key
    (output / (stem + ".json")).write_text(state)
    (output / (stem + ".tex")).write_text(result.to_latex())
    return restored


def run_stage(stage, input_dir, output_dir):
    import openecon as oe
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    data = pd.read_csv(Path(input_dir) / "responses.csv", index_col="person")
    if stage == 0:
        result = oe.irt_bank_binary(["b1", "b2", "b3", "b4"], [1., .8, 1.2, .9],
                                   [-1., -.3, .4, 1.], guessing=[.1, 0., .2, .05],
                                   upper=[.9, 1., .95, .98])
        restored = save(result, "binary-bank", output_dir)
        assert oe.summary_state(oe.irt_bank_restore(restored)) == oe.summary_state(result)
    elif stage == 1 or stage >= 4:
        bank = oe.irt_bank_polytomous(["g", "p", "n"], family=["grm", "gpcm", "nrm"],
                                     thresholds=[[-1., 1.], [.7, -.4], None],
                                     discrimination=[1.1, .9, None],
                                     slopes=[None, None, [0., .8, -.5]],
                                     intercepts=[None, None, [0., .4, -.2]],
                                     scores=[[0, 1, 2], [0, 1, 2], [0, 2, 2]])
        if stage == 1:
            result = bank
            restored = save(result, "mixed-bank", output_dir)
            assert oe.summary_state(oe.irt_bank_restore(restored)) == oe.summary_state(result)
        else:
            bank = oe.irt_bank_restore(oe.restore_summary(oe.summary_state(bank)))
            posterior = oe.irt_posterior(bank, data=data, support=[-2., -.5, .8, 2.],
                                         masses=[.1, .2, .4, .3])
            posterior = oe.irt_posterior_restore(oe.restore_summary(oe.summary_state(posterior)))
            if stage == 4:
                result = posterior
                save(result, "finite-posterior", output_dir)
            elif stage == 5:
                result = oe.irt_plausible_values(posterior, draws=19, seed=83)
                save(result, "posterior-draws", output_dir)
            elif stage == 6:
                result = oe.irt_predictive(posterior, items=["n", "g", "p"])
                save(result, "replicate-prediction", output_dir)
            else:
                result = oe.irt_test_score_distribution(posterior, items=["n", "g", "p"], level=.95)
                save(result, "replicate-score-pmf", output_dir)
    elif stage in (2, 3):
        bank = oe.irt_bank_binary(["b1", "b2", "b3", "b4"], [1., 1., 1., 1.],
                                  [0., 0., 0., 0.])
        if stage == 2:
            result = oe.irt_score_mle(bank, data=data.iloc[:8])
            save(result, "conditional-mle", output_dir)
        else:
            result = oe.irt_score_map(bank, data=data, prior_mean=.2, prior_sd=1.1)
            save(result, "conditional-map", output_dir)
    else:
        raise ValueError("Unknown stage")
    return result


def native_script(stage, input_dir, output_dir):
    common = "import sys\nfrom pathlib import Path\nimport openecon as oe\nassert getattr(sys,'frozen',False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\nassert oe.capabilities()['irt']['stata_parity_validated'] is False\n"
    common += inspect.getsource(save) + "\n" + inspect.getsource(run_stage) + "\n"
    common += f"result=run_stage({stage},{str(input_dir)!r},{str(output_dir)!r})\n"
    common += 'for name,frame in result.items():\n    if name!="settings":\n        display(frame)\n'
    common += f"print('IRT_CALIBRATED_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/fixtures/irt-calibrated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import openecon as oe
    assert Path(oe.__file__).is_relative_to(ROOT / "src")
    for stage, issue in enumerate(ISSUES):
        result = run_stage(stage, args.input, args.output)
        print(issue, result.attrs.get("method", result.attrs.get("kind")))
    receipt = dict(source_sdk=str(Path(oe.__file__).resolve()), stages=list(ISSUES), passed=8,
                   files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "receipt.json"})
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
