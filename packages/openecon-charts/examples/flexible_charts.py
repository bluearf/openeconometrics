"""Generate nine offline charts and their matching LaTeX sources.

Run with an optional output directory:
    python flexible_charts.py /tmp/openecon-chart-examples
Only the standard library and openecon_charts are required.
"""
from pathlib import Path
import sys

import openecon_charts as charts


def examples():
    numeric = {"income": [10, 100, 1000], "spending": [8, 60, 450]}
    categorical = {"quarter": ["Q1", "Q2", "Q3"],
                   "online": [12, 18, 15], "retail": [8, 11, 13]}
    palette = ["#163d68", "#6c8eaf"]
    return {
        "scatter": charts.scatter(
            data=numeric, x="income", y="spending", title="Income and spending",
            x_scale="log", x_label="Income", y_label="Spending",
            xlim=(5, 2000), ylim=(0, 500), color=palette[0],
            point_size=4, opacity=.65, width=900, height=500, grid=True,
        ),
        "line": charts.line(
            data={"time": [1, 2, 3, 4, 5], "value": [2, 4, None, 8, 10]},
            x="time", y="value", title="Missing values remain gaps",
            line_width=2.5, point_size=3, color=palette[0], grid=False,
        ),
        "histogram": charts.hist(
            data={"value": [1, 1, 2, 3, 4, 4, 4, 5]}, x="value", bins=4,
            title="All finite observations counted", color=palette[0],
            x_label="Value", y_label="Count", opacity=.85,
        ),
        "coefficients": charts.coefficients(
            {"coefficients": [
                {"term": "Education", "estimate": 1.2, "ci_low": .8, "ci_high": 1.6},
                {"term": "Experience", "estimate": .3, "ci_low": -.1, "ci_high": .7},
            ]}, title="Estimates and confidence intervals", palette=palette,
            xlim=(-.5, 2), point_size=4, line_width=2,
        ),
        "bar": charts.bar(
            data=categorical, x="quarter", y=["online", "retail"],
            title="Grouped sales", unit="EUR thousands", palette=palette,
            y_label="Sales", legend_position="top", height=400,
        ),
        "barh": charts.barh(
            data=categorical, x="quarter", y="online", title="Online sales",
            x_label="Sales", y_label="Quarter", xlim=(0, 20),
            color=palette[0], legend=False,
        ),
        "area": charts.area(
            data={"period": ["A", "B", "C", "D"], "balance": [2, None, -1, 3]},
            x="period", y="balance", title="Signed values and missing gaps",
            color=palette[0], line_width=2, point_size=3,
        ),
        "stacked_bar": charts.stacked_bar(
            data=categorical, x="quarter", y=["online", "retail"],
            title="Complete sales components", unit="EUR thousands",
            palette=palette, legend_position="top",
        ),
        "donut": charts.donut(
            data={"region": ["North", "South", "West"], "value": [12, 18, 9]},
            labels="region", values="value", title="Regional composition",
            palette=["#163d68", "#6c8eaf", "#a8b8cb"],
            legend_position="bottom", height=400,
        ),
    }


def main():
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "chart-examples")
    destination.mkdir(parents=True, exist_ok=True)
    for name, chart in examples().items():
        chart.save_html(destination / f"{name}.html")
        # The manuscript copy uses page-friendly physical dimensions.
        paper = chart.with_options(width=600, height=360)
        paper.to_latex(destination / f"{name}.tex", standalone=True)
    print(f"Saved nine HTML charts and nine LaTeX sources to {destination}")


if __name__ == "__main__":
    main()
