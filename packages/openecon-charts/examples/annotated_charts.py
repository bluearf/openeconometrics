"""Export nine annotated charts without additional Python dependencies.

Run: python annotated_charts.py /tmp/openecon-annotated-charts
"""
from pathlib import Path
import sys

from flexible_charts import examples as base_examples


def examples():
    charts = base_examples()
    charts["scatter"] = (
        charts["scatter"].vline(100, text="Income threshold", dash="dotted")
        .hspan(50, 100, text="Reference range", opacity=.12)
        .annotate("Observed value", x=100, y=60, arrow=True, dx=26, dy=-36)
    )
    charts["line"] = (
        charts["line"].vspan(1.5, 2.5, text="Policy period")
        .hline(5, text="Target", color="#6c8eaf")
        .annotate("Source: survey", x=.03, y=.92, coords="axes", font_size=10)
    )
    charts["histogram"] = charts["histogram"].vline(3, text="Threshold")
    charts["coefficients"] = (
        charts["coefficients"].vline(0, text="Null effect")
        .annotate("Education effect", x=1.2, y="Education", arrow=True, dx=18, dy=-24)
    )
    charts["bar"] = (
        charts["bar"].hline(20, text="Target")
        .annotate("Q2 sales", x="Q2", y=18, arrow=True, dx=16, dy=-28)
    )
    charts["barh"] = (
        charts["barh"].vline(15, text="Target")
        .annotate("Q2 sales", x=18, y="Q2", arrow=True, dx=-54, dy=-28)
    )
    charts["area"] = charts["area"].hspan(-.5, .5, text="Neutral range")
    charts["stacked_bar"] = charts["stacked_bar"].hline(25, text="Total target")
    charts["donut"] = charts["donut"].annotate(
        "Source: regional survey", x=.03, y=.95, coords="axes", font_size=10,
    )
    return charts


def main():
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "annotated-charts")
    destination.mkdir(parents=True, exist_ok=True)
    for name, chart in examples().items():
        chart.save_html(destination / f"{name}.html")
        chart.with_options(width=600, height=400).to_latex(
            destination / f"{name}.tex", standalone=True,
        )
    print(f"Saved nine annotated HTML charts and LaTeX sources to {destination}")


if __name__ == "__main__":
    main()
