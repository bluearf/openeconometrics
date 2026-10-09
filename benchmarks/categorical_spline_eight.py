"""Eight bounded categorical spline/weighted-option delivery stages."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
ISSUES = tuple(f"MARKET-{i}" for i in range(691, 699))
KNOTS = {"x": [0., 1., 2., 3.], "z": [0., 1., 2., 3.]}
QUERY_POSITIONS = [0, 9, 18, 27, 36, 54, 72, 99]


def save(result, stem, output_dir):
    import openecon as oe

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored.to_latex() == result.to_latex(), stem + " complete LaTeX differs"
    assert list(restored) == list(result), stem + " table order differs"
    for key, frame in result.items():
        assert restored[key].equals(frame), stem + " complete table differs: " + key
    (output / (stem + ".json")).write_text(state, encoding="utf-8")
    (output / (stem + ".tex")).write_text(result.to_latex(), encoding="utf-8")
    return restored


def _ramps(data, knots):
    import numpy as np

    return np.column_stack([
        np.clip((data[name].to_numpy()-left)/(right-left), 0, 1)
        for name, values in knots.items() for left, right in zip(values, values[1:])
    ])


def _cone_solution(matrix, target):
    """Enumerate all faces of this fixture's small nonnegative cone."""
    import numpy as np

    width = matrix.shape[1]
    best = None
    for mask in range(1 << width):
        columns = [j for j in range(width) if mask & (1 << j)]
        coefficient = np.zeros(width)
        if columns:
            coefficient[columns] = np.linalg.lstsq(matrix[:, columns], target, rcond=None)[0]
        if np.any(coefficient < -1e-10):
            continue
        coefficient = np.maximum(coefficient, 0.)
        loss = np.sum((target-matrix@coefficient)**2)
        if best is None or loss < best[0]:
            best = (loss, coefficient)
    assert best is not None
    return best


def _bootstrap_oracle(data, queries, counts, ordinal):
    import numpy as np

    name = "b" if ordinal else "a"
    levels = ["low", "mid", "high"] if ordinal else ["red", "blue", "green"]
    if ordinal:
        codes = np.array([levels.index(value) for value in data[name]])
        query_codes = np.array([levels.index(value) for value in queries[name]])
        blocks = [np.ones(len(data)), data.x.to_numpy(), codes >= 1, codes >= 2]
        query_blocks = [np.ones(len(queries)), queries.x.to_numpy(), query_codes >= 1, query_codes >= 2]
    else:
        blocks = [np.ones(len(data)), data.x.to_numpy()] + [(data[name] == value).to_numpy() for value in levels[1:]]
        query_blocks = [np.ones(len(queries)), queries.x.to_numpy()] + [(queries[name] == value).to_numpy() for value in levels[1:]]
    design, query_design = np.column_stack(blocks), np.column_stack(query_blocks)
    root = np.sqrt(np.asarray(counts, dtype=float))
    target = data.y.to_numpy()*root
    matrix = design*root[:, None]
    if not ordinal:
        coefficient = np.linalg.lstsq(matrix, target, rcond=None)[0]
    else:
        # The intercept and numeric effect are free. Enumerate both ordered
        # directions and every face of the two category-increment cone.
        candidates = []
        for sign in (-1., 1.):
            signed = matrix*np.array([1., 1., sign, sign])
            for mask in range(4):
                columns = [0, 1]+[2+j for j in range(2) if mask & (1 << j)]
                value = np.zeros(4)
                value[columns] = np.linalg.lstsq(signed[:, columns], target, rcond=None)[0]
                if np.any(value[2:] < -1e-10):
                    continue
                coefficient0 = value*np.array([1., 1., sign, sign])
                candidates.append((np.sum((target-matrix@coefficient0)**2), coefficient0))
        assert candidates
        coefficient = min(candidates, key=lambda pair: pair[0])[1]
    return query_design@coefficient


def _spline_correlation(data, monotone):
    """Independent two-variable canonical correlations over all cone faces."""
    import numpy as np

    full = _ramps(data, KNOTS)
    bases = [full[:, :3], full[:, 3:]]
    bases = [matrix-matrix.mean(0) for matrix in bases]
    masks = range(1, 8) if monotone else [7]
    best = 0.
    for first in masks:
        columns1 = [j for j in range(3) if first & (1 << j)]
        q1, r1 = np.linalg.qr(bases[0][:, columns1], mode="reduced")
        for second in masks:
            columns2 = [j for j in range(3) if second & (1 << j)]
            q2, r2 = np.linalg.qr(bases[1][:, columns2], mode="reduced")
            left, roots, right = np.linalg.svd(q1.T@q2, full_matrices=False)
            for j, correlation in enumerate(roots):
                first_coefficient = np.linalg.solve(r1, left[:, j])
                second_coefficient = np.linalg.solve(r2, right[j, :])
                if monotone and any(not (np.all(value >= -1e-10) or np.all(value <= 1e-10))
                                    for value in (first_coefficient, second_coefficient)):
                    continue
                best = max(best, float(correlation))
    return best


def check_stage(stage, result, data, queries):
    """Independent fixture oracles; no SciPy or licensed reference runtime."""
    import itertools
    import numpy as np

    if stage in (0, 1):
        pattern = result["pattern"].to_numpy(dtype=float)
        structure = result["structure"].to_numpy(dtype=float)
        phi = result["component_correlations"].to_numpy(dtype=float)
        transform = result["transformation"].to_numpy(dtype=float)
        original = result["original_loadings"].to_numpy(dtype=float)
        scores = result["scores"].iloc[:, 1:].to_numpy(dtype=float)
        counts = data.w.to_numpy(dtype=float)
        np.testing.assert_allclose(pattern, original@transform, atol=2e-8)
        np.testing.assert_allclose(structure, pattern@phi, atol=2e-8)
        np.testing.assert_allclose(scores.T@(counts[:, None]*scores)/counts.sum(), phi, atol=2e-8)
        np.testing.assert_allclose(scores@pattern.T, result["reconstruction"].iloc[:, 1:], atol=2e-8)
        if stage == 0:
            np.testing.assert_allclose(phi, np.eye(2), atol=2e-8)
        else:
            np.testing.assert_allclose(np.diag(phi), np.ones(2), atol=2e-8)
        centers = result["category_centroids"]
        for _, row in centers.iterrows():
            selected = data[row.variable] == row.category
            expected = np.average(scores[selected], weights=counts[selected], axis=0)
            np.testing.assert_allclose(row.iloc[4:].to_numpy(dtype=float), expected, atol=2e-8)
    elif stage in (2, 3):
        state = result.attrs["bootstrap_state"]
        ordinal = stage == 3
        sampled = state["draw_counts"]
        total = int(data.w.sum())
        assert len(sampled) == 19 and all(sum(counts) == total for counts in sampled)
        assert all(type(value) is int and value >= 0 for counts in sampled for value in counts)
        assert result.attrs["inference_available"] and bool(result["draw_status"].success.all())
        expected = np.array([_bootstrap_oracle(data, queries, counts, ordinal) for counts in sampled])
        point = _bootstrap_oracle(data, queries, data.w.to_list(), ordinal)
        np.testing.assert_allclose(result["baseline_predictions"].fitted, point, atol=2e-6, rtol=2e-7)
        np.testing.assert_allclose(result["draw_predictions"].iloc[:, 1:], expected, atol=2e-6, rtol=2e-7)
        np.testing.assert_allclose(result["prediction_covariance"], np.cov(expected, rowvar=False, ddof=1), atol=2e-6)
        quantiles = np.quantile(expected, [.025, .975], axis=0, method="linear")
        np.testing.assert_allclose(result["percentile_intervals"][["percentile_lower", "percentile_upper"]].to_numpy().T,
                                   quantiles, atol=2e-6, rtol=2e-7)
    elif stage in (4, 5):
        ramps = _ramps(data, KNOTS)
        if stage == 4:
            design = np.column_stack([np.ones(len(data)), ramps])
            fitted = design@np.linalg.lstsq(design, data.y.to_numpy(), rcond=None)[0]
        else:
            centered = ramps-ramps.mean(0)
            target = data.y.to_numpy()-data.y.mean()
            candidates = []
            for signs in itertools.product((-1., 1.), repeat=2):
                direction = np.repeat(signs, 3)
                loss, positive = _cone_solution(centered*direction, target)
                candidates.append((loss, positive*direction))
            coefficient = min(candidates, key=lambda pair: pair[0])[1]
            fitted = data.y.mean()+centered@coefficient
            for name in KNOTS:
                assert np.all(np.diff(result["spline_knots"].query("variable == @name").quantification) >= -1e-10)
        np.testing.assert_allclose(result["fitted"].fitted, fitted, atol=2e-8, rtol=2e-8)
        transformed = result["transformed"].iloc[:, 1:].to_numpy(dtype=float)
        np.testing.assert_allclose(transformed.mean(0), 0., atol=2e-8)
        np.testing.assert_allclose((transformed**2).mean(0), 1., atol=2e-8)
    elif stage in (6, 7):
        transformed = result["transformed"].iloc[:, 1:].to_numpy(dtype=float)
        scores = result["scores"].iloc[:, 1:].to_numpy(dtype=float)
        loadings = result["loadings"].to_numpy(dtype=float)
        np.testing.assert_allclose(transformed.mean(0), 0., atol=2e-8)
        np.testing.assert_allclose((transformed**2).mean(0), 1., atol=2e-8)
        np.testing.assert_allclose(scores.mean(0), 0., atol=2e-8)
        np.testing.assert_allclose(scores.T@scores/len(scores), np.eye(1), atol=2e-8)
        np.testing.assert_allclose(loadings, transformed.T@scores/len(scores), atol=2e-8)
        loss = np.sum((transformed-scores@loadings.T)**2)/len(scores)
        np.testing.assert_allclose(result["fit"].reconstruction_loss.iloc[0], loss, atol=2e-8)
        np.testing.assert_allclose(loss, 1-_spline_correlation(data, stage == 7), atol=2e-6)
        if stage == 7:
            assert np.all(result["splines"].coefficient.to_numpy() >= -1e-10)


def run_stage(stage, input_dir, output_dir):
    import openecon as oe
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    data = pd.read_csv(Path(input_dir) / "people.csv")
    assert len(data) == 100 and list(data.columns) == ["x", "z", "a", "b", "w", "y"]
    queries = data.iloc[QUERY_POSITIONS].reset_index(drop=True)
    if stage in (0, 1):
        source = oe.catpca_fweight(data, ["a", "b", "x", "z"], components=2, frequency="w",
            scales={"a": "nominal", "b": "ordinal", "x": "numeric", "z": "numeric"},
            orders={"b": ["low", "mid", "high"]}, n_starts=2, maxiter=300, tol=1e-8, seed=0)
        stem = "weighted-varimax" if stage == 0 else "weighted-promax"
        restored_source = save(source, stem+"-source", output_dir)
        method = oe.catpca_varimax_fweight if stage == 0 else oe.catpca_promax_fweight
        result = method(restored_source, max_iter=1000, tol=1e-10)
        restored = save(result, stem, output_dir)
        save(oe.catpca_rotated_predict_fweight(restored, queries), stem+"-projection", output_dir)
    elif stage in (2, 3):
        options = dict(frequency="w", queries=queries, reps=19, n_starts=2, maxiter=100, tol=1e-8, seed=83)
        if stage == 2:
            result = oe.catreg_nominal_fweight_bootstrap(data, "y", ["a", "x"],
                scales={"a": "nominal", "x": "numeric"}, **options)
            stem = "frequency-nominal-bootstrap"
        else:
            result = oe.catreg_ordinal_fweight_bootstrap(data, "y", ["b", "x"],
                scales={"b": "ordinal", "x": "numeric"}, orders={"b": ["low", "mid", "high"]}, **options)
            stem = "frequency-ordinal-bootstrap"
        restored = save(result, stem, output_dir)
        verified = oe.catreg_fweight_bootstrap_restore(restored)
        assert oe.summary_state(verified) == oe.summary_state(result)
    elif stage in (4, 5):
        method = oe.catreg_spline if stage == 4 else oe.catreg_mspline
        stem = "spline-nominal-catreg" if stage == 4 else "spline-monotone-catreg"
        result = method(data, "y", ["x", "z"], knots=KNOTS)
        restored = save(result, stem, output_dir)
        save(oe.catreg_spline_predict(restored, queries), stem+"-prediction", output_dir)
    elif stage in (6, 7):
        method = oe.catpca_spline if stage == 6 else oe.catpca_mspline
        stem = "spline-nominal-catpca" if stage == 6 else "spline-monotone-catpca"
        result = method(data, ["x", "z"], knots=KNOTS, components=1,
                        n_starts=2, seed=83, maxiter=400, tol=1e-8)
        restored = save(result, stem, output_dir)
        save(oe.catpca_spline_predict(restored, queries), stem+"-projection", output_dir)
    else:
        raise ValueError("Unknown stage")
    check_stage(stage, result, data, queries)
    return result


def native_script(stage, input_dir, output_dir):
    common = "import sys\nfrom pathlib import Path\nimport openecon as oe\nimport importlib.util\n"
    common += "assert getattr(sys, 'frozen', False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\n"
    common += "assert importlib.util.find_spec('scipy') is None\nassert importlib.util.find_spec('statsmodels') is None\n"
    common += "assert oe.capabilities()['categorical_spline_weighted_options']['stata_parity_validated'] is False\n"
    common += f"KNOTS = {KNOTS!r}\nQUERY_POSITIONS = {QUERY_POSITIONS!r}\n"
    for function in (save, _ramps, _cone_solution, _bootstrap_oracle, _spline_correlation, check_stage, run_stage):
        common += inspect.getsource(function) + "\n"
    common += f"result = run_stage({stage}, {str(input_dir)!r}, {str(output_dir)!r})\n"
    common += "for name, frame in result.items():\n    display(frame)\n"
    common += f"print('CATEGORICAL_SPLINE_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/fixtures/categorical_spline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import openecon as oe

    assert Path(oe.__file__).is_relative_to(ROOT / "src")
    for stage, issue in enumerate(ISSUES):
        result = run_stage(stage, args.input, args.output)
        print(issue, result.attrs.get("method"), flush=True)
    receipt = dict(source_sdk=str(Path(oe.__file__).resolve()), stages=list(ISSUES), passed=8,
                   input_sha256=hashlib.sha256((args.input/"people.csv").read_bytes()).hexdigest(),
                   independent_fixture_oracles=True, licensed_vendor_run=False,
                   files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                          if p.is_file() and p.name != "receipt.json"})
    (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
