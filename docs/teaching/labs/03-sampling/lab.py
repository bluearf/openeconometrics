"""Original Monte Carlo sampling laboratory with a known lognormal population."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import openecon as oe


SEED = 3032026
REPLICATIONS = 1200
SAMPLE_SIZES = (10, 40, 160)
LOG_MEAN = 1.0
LOG_SD = 0.8


def run_lab(output_dir=None, display_callback=None):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    population_mean = math.exp(LOG_MEAN + LOG_SD**2 / 2)
    population_sd = math.sqrt((math.exp(LOG_SD**2) - 1) * math.exp(2 * LOG_MEAN + LOG_SD**2))
    columns, rows, panels, checks = {}, [], [], {}
    for n in SAMPLE_SIZES:
        draws = torch.exp(
            LOG_MEAN
            + LOG_SD * torch.randn(REPLICATIONS, n, generator=generator, dtype=torch.float64)
        )
        means = draws.mean(dim=1)
        true_se = population_sd / math.sqrt(n)
        standardized = (means - population_mean) / true_se
        estimated_se = draws.std(dim=1, correction=1) / math.sqrt(n)
        row = {
            "sample_size": n,
            "replications": REPLICATIONS,
            "average_sample_mean": float(means.mean()),
            "empirical_sd_of_means": float(means.std(correction=1)),
            "theoretical_se": true_se,
            "average_estimated_se": float(estimated_se.mean()),
            "known_sigma_normal_interval_coverage": float(
                (standardized.abs() <= 1.959963984540054).double().mean()
            ),
            "estimated_sigma_normal_interval_coverage": float(
                ((means - population_mean).abs() <= 1.959963984540054 * estimated_se)
                .double()
                .mean()
            ),
            "standardized_mean": float(standardized.mean()),
            "standardized_sd": float(standardized.std(correction=1)),
        }
        rows.append(row)
        columns[f"means_n{n}"] = means.tolist()
        counts, edges = torch.histogram(standardized, bins=28)
        panels.append(
            {
                "kind": "bar",
                "title": f"Standardized sample means, n={n}",
                "xlabel": "(Sample mean − population mean) / theoretical SE",
                "ylabel": "Simulated samples per bin",
                "series": [
                    {
                        "label": f"n={n}",
                        "x": ((edges[1:] + edges[:-1]) / 2).tolist(),
                        "y": counts.tolist(),
                    }
                ],
            }
        )
        checks[f"n{n}_histogram_count"] = int(counts.sum()) == REPLICATIONS
        checks[f"n{n}_mean_within_five_monte_carlo_se"] = abs(
            row["average_sample_mean"] - population_mean
        ) < 5 * true_se / math.sqrt(REPLICATIONS)
        checks[f"n{n}_sd_close_to_analytic_se"] = (
            abs(row["empirical_sd_of_means"] / true_se - 1) < 0.15
        )
    native = oe.describe(oe.DataFrame(columns), list(columns), stats=["n", "mean", "std_dev"])
    for row in rows:
        name = f"means_n{row['sample_size']}"
        checks[f"{name}_native_descriptives"] = (
            abs(float(native.loc[name, "mean"]) - row["average_sample_mean"]) < 1e-12
            and abs(float(native.loc[name, "std_dev"]) - row["empirical_sd_of_means"]) < 1e-12
        )
    checks["se_square_root_law"] = (
        abs(rows[0]["theoretical_se"] / rows[2]["theoretical_se"] - 4) < 1e-12
    )
    if not all(checks.values()):
        raise AssertionError(checks)
    summary = {
        "population_mean": population_mean,
        "population_sd": population_sd,
        "replications": REPLICATIONS,
    }
    for row in rows:
        for key, value in row.items():
            if key not in ("sample_size", "replications"):
                summary[f"n{row['sample_size']}_{key}"] = value
    result = {
        "metadata": {
            "lab": "03-sampling",
            "title": "What Changes Across Random Samples?",
            "seed": SEED,
            "synthetic": True,
            "population": "lognormal(log mean=1, log SD=0.8)",
            "independent_samples": True,
        },
        "summary": summary,
        "models": {},
        "chart_data": {"panels": panels, "sampling_summary": rows},
        "checks": checks,
    }
    table = oe.DataFrame(rows)
    latex = table[
        [
            "sample_size",
            "average_sample_mean",
            "empirical_sd_of_means",
            "theoretical_se",
            "known_sigma_normal_interval_coverage",
            "estimated_sigma_normal_interval_coverage",
        ]
    ].to_latex(
        index=False, precision=4, caption="Repeated sampling from an original lognormal population"
    )
    if display_callback is not None:
        display_callback(table)
        for n in SAMPLE_SIZES:
            values = [
                (value - population_mean) / (population_sd / math.sqrt(n))
                for value in columns[f"means_n{n}"]
            ]
            display_callback(
                oe.plot.hist(
                    data={"standardized_mean": values},
                    x="standardized_mean",
                    bins=28,
                    title=f"Standardized sample means: n={n}",
                )
            )
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (target / "table.tex").write_text(str(latex), encoding="utf-8")
        if json.loads((target / "reference.json").read_text()) != result:
            raise AssertionError("Explicit export readback differs.")
    print(
        f"Known population mean {population_mean:.6f}, SD {population_sd:.6f}; {REPLICATIONS} independent samples at each n."
    )
    print(
        table[
            ["sample_size", "average_sample_mean", "empirical_sd_of_means", "theoretical_se"]
        ].to_string(index=False)
    )
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
