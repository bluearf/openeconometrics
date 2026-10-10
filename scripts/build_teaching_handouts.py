"""Build offline teaching handouts from reviewed Markdown and numerical references.

Optional tools: docs/teaching/requirements-build.txt and web's locked KaTeX.
This script renders saved references; verify_teaching_labs.py executes the labs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import textwrap

from teaching_content import assert_student_content

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "teaching"
INK = "#142c43"
BLUE = "#326b9e"
GOLD = "#b37926"
CSS = """
:root {color-scheme:light; --ink:#142c43; --muted:#566577; --blue:#326b9e;}
* {box-sizing:border-box} body {margin:0;background:#f6f7f8;color:var(--ink);
 font:17px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
a {color:#225c8f;text-underline-offset:.2em} a:hover {color:#9c6822}
.shell {display:grid;grid-template-columns:235px minmax(0,900px);gap:55px;
 max-width:1280px;margin:0 auto;padding:40px 32px 100px;}
aside {position:sticky;top:28px;align-self:start;font-size:13px;line-height:1.5;}
aside a {display:block;margin:12px 0;text-decoration:none} aside .wordmark {
 font-weight:750;font-size:16px;letter-spacing:-.03em;margin-bottom:28px}
aside small {color:var(--muted)} .toc ul {padding-left:16px;list-style:none}
.toc li a {font-size:12px;margin:8px 0} .toc>ul {padding:0}
main {min-width:0;background:white;padding:56px 65px;border:1px solid #e2e7ec;
 overflow-wrap:anywhere;}
.eyebrow {text-transform:uppercase;letter-spacing:.17em;font-size:11px;
 font-weight:700;color:var(--blue);margin:0 0 18px}
h1 {font-size:43px;line-height:1.14;font-weight:700;letter-spacing:-.035em;
 margin:0 0 28px} h2 {font-size:25px;line-height:1.3;letter-spacing:-.02em;
 margin:48px 0 18px;padding-top:12px;border-top:1px solid #e4e9ef}
h3 {font-size:19px;line-height:1.4;margin-top:28px} p {margin:15px 0}
strong {font-weight:650} blockquote {margin:24px 0;padding:3px 20px;
 border-left:3px solid #326b9e;background:#f5f8fb;color:#344e65;}
pre {background:#f3f6f9;border:1px solid #e1e7ee;padding:18px 20px;
 font-size:12px;line-height:1.6;overflow-x:auto;tab-size:4;}
code {font-family:'SFMono-Regular',Consolas,monospace;font-size:.83em;}
pre code {font-size:inherit} :not(pre)>code {background:#f0f3f6;padding:2px 4px;}
table {width:100%;border-collapse:collapse;font-size:13px;line-height:1.55;
 margin:24px 0} th {background:#edf2f7;text-align:left;font-weight:650;}
th,td {padding:10px 12px;border-bottom:1px solid #dce3eb;vertical-align:top;}
img {max-width:100%;height:auto;display:block;margin:24px auto;}
.math-display {overflow-x:auto;margin:22px 0;text-align:center;}
.katex {font-size:1.05em}.katex-display {margin:.6em 0}
.lesson-links {display:flex;gap:16px;flex-wrap:wrap;font-size:12px;
 border-bottom:1px solid #e4e9ef;padding:0 0 20px;margin-bottom:30px}
.card {padding:26px 0;border-top:1px solid #dce3eb} .card h2 {border:0;
 padding:0;margin:8px 0;font-size:25px}.card p {margin:8px 0;color:var(--muted)}
.number {color:var(--blue);font-size:12px;font-weight:700;letter-spacing:.12em;}
footer {margin-top:55px;padding-top:20px;border-top:1px solid #dce3eb;
 font-size:12px;color:var(--muted)} .book .lesson {break-before:page;}
.book .cover {min-height:640px;display:flex;flex-direction:column;justify-content:center}
.book .cover h1 {font-size:52px}.book .cover .lede {font-size:23px;line-height:1.5;}
@media(max-width:1000px){.shell{display:block;max-width:880px;padding:18px;}
 aside{position:static;padding:12px 5px}.toc{display:none}aside a{display:inline-block;
 margin:4px 12px 12px 0}main{padding:38px}h1{font-size:36px}}
@media screen and (max-width:600px){body{font-size:16px}.shell{padding:0}main{padding:28px 20px;
 border:0}aside{padding:18px 20px;background:white}h1{font-size:32px}h2{font-size:23px}
 table{display:block;overflow-x:auto;overflow-wrap:normal}th,td{min-width:85px}
 td code{white-space:nowrap}pre{padding:14px;font-size:11px}
 .math-display{text-align:left}.math-display .katex-display{
 width:max-content;min-width:100%;text-align:left}}
@page {size:A4;margin:17mm 17mm 18mm;}
@media print {body{background:white;font-size:10.5pt;line-height:1.52}
 .shell{display:block;padding:0;max-width:none}aside,.lesson-links{display:none}
 main{border:0;padding:0}h1{font-size:29pt;margin-bottom:18pt}
 h2{font-size:17pt;margin-top:25pt;padding-top:8pt}h3{font-size:13pt}
 h1,h2,h3{break-after:avoid}p,li{orphans:3;widows:3}ul{break-inside:avoid}pre{white-space:pre-wrap;
 overflow-wrap:anywhere;font-size:8pt;line-height:1.45;padding:10pt;break-inside:avoid;}
 table{font-size:8.5pt;break-inside:avoid}thead{display:table-header-group}tr{break-inside:avoid}
 blockquote,img,.math-display{break-inside:avoid}img{max-height:110mm}
 a{color:inherit;text-decoration:none}.katex{font-size:1em}
 .math-display{break-before:avoid}footer{display:none}.cover footer{display:block}
 .book .cover{height:235mm;min-height:0}.book .cover h1{font-size:38pt}
 .book .cover .lede{font-size:17pt}.book .cover footer{margin-top:35pt}
 .instructor{font-size:10pt;line-height:1.42}.instructor p{margin:7px 0}
 .instructor h2{font-size:16pt;margin-top:16pt}.instructor h3{margin-top:18px}
 .instructor-preface table{break-inside:auto}}
"""
JS = """
document.querySelectorAll('[data-math]').forEach(el => {
 try {katex.render(el.textContent,el,{displayMode:el.dataset.math==='display',
  throwOnError:true,trust:false,strict:'error'});}
 catch(error){el.classList.add('math-error');el.title=error.message;}
});
document.documentElement.dataset.mathReady='true';
"""


def render_markdown(source: str) -> tuple[str, str]:
    """Protect TeX from Markdown emphasis/backslash handling, excluding code."""
    protected = {}

    def protect(match):
        expression = match.group(0)
        display = expression.startswith("$$")
        tex = expression[2:-2] if display else expression[1:-1]
        token = f"MATHPLACEHOLDER{len(protected):05d}END"
        tag = "div" if display else "span"
        mode = "display" if display else "inline"
        protected[token] = (
            f'<{tag} class="math-{mode}" data-math="{mode}">{html.escape(tex.strip())}</{tag}>'
        )
        return token

    chunks = re.split(r"(```[\s\S]*?```|`[^`\n]*`)", source)
    for index in range(0, len(chunks), 2):
        chunks[index] = re.sub(
            r"\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\$)[^$\n]+?\$", protect, chunks[index]
        )
    parser = markdown.Markdown(
        extensions=["extra", "toc", "sane_lists"], extension_configs={"toc": {"toc_depth": "2-3"}}
    )
    body = parser.convert("".join(chunks))
    for token, replacement in protected.items():
        body = body.replace(f"<p>{token}</p>", replacement).replace(token, replacement)
    return body, parser.toc


def page(title, body, *, asset_prefix="", toc="", book=False, audience="student"):
    collection = "../../index.html" if asset_prefix.startswith("../../") else "index.html"
    nav = f'<a href="{collection}">Course collection</a>'
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)} · OpenEconometrics</title>"
        f'<link rel="stylesheet" href="{asset_prefix}assets/katex/katex.min.css">'
        f'<style>{CSS}</style></head><body class="{"book " if book else ""}{audience}">'
        '<div class="shell"><aside><div class="wordmark">OpenEconometrics<br>Teaching Labs</div>'
        f"{nav}<small>English<br>Original synthetic data<br>{audience.title()} edition</small>"
        f"{toc}</aside><main>{body}</main></div>"
        f'<script src="{asset_prefix}assets/katex/katex.min.js"></script>'
        f"<script>{JS}</script></body></html>"
    )


def configure_plots():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#b8c5d2",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "openecon-teaching",
            "svg.fonttype": "none",
        }
    )


def generic_figure(data):
    panels = data.get("panels") or [data["figure"]]
    if not 1 <= len(panels) <= 3:
        raise ValueError("A lesson figure needs one to three panels.")
    count = len(panels)
    fig, axes = plt.subplots(
        1, count, figsize=(9 if count > 1 else 8, 4.1), squeeze=False, layout="constrained"
    )
    colors = [BLUE, INK, GOLD, "#697e42"]
    for ax, panel in zip(axes.flat, panels):
        kind = panel["kind"]
        series_list = panel["series"]
        for index, series in enumerate(series_list):
            x, y = series["x"], series.get("y")
            label = series.get("label")
            color = colors[index % len(colors)]
            style = "--" if index % 2 else "-"
            if kind == "scatter":
                ax.scatter(x, y, s=12, color=color, alpha=0.45, edgecolors="none", label=label)
            elif kind == "line":
                ax.plot(x, y, color=color, lw=1.7, linestyle=style, label=label)
                if "low" in series:
                    ax.fill_between(x, series["low"], series["high"], color=color, alpha=0.13)
            elif kind == "interval":
                if any(isinstance(value, str) for value in y):
                    positions = list(range(len(y)))
                    ax.set_yticks(positions, [textwrap.fill(str(value), 20) for value in y])
                    errors = [
                        [v - low for v, low in zip(x, series["low"])],
                        [high - v for v, high in zip(x, series["high"])],
                    ]
                    ax.errorbar(
                        x,
                        positions,
                        xerr=errors,
                        color=color,
                        fmt="o",
                        capsize=3,
                        markersize=4,
                        linewidth=1.4,
                        label=label,
                    )
                    continue
                if any(isinstance(value, str) for value in x):
                    positions = list(range(len(x)))
                    ax.set_xticks(positions, [textwrap.fill(str(value), 18) for value in x])
                else:
                    positions = x
                if "low" in series:
                    errors = [
                        [v - low for v, low in zip(y, series["low"])],
                        [high - v for v, high in zip(y, series["high"])],
                    ]
                    ax.errorbar(
                        positions,
                        y,
                        yerr=errors,
                        color=color,
                        fmt="o",
                        capsize=3,
                        markersize=4,
                        linewidth=1.4,
                        label=label,
                    )
                else:
                    ax.plot(positions, y, "o", color=color, label=label)
            elif kind in {"bar", "hist"} and y is not None:
                if kind == "hist":
                    numeric = list(map(float, x))
                    width = (
                        min((b - a for a, b in zip(numeric, numeric[1:]) if b > a), default=1) * 0.9
                    )
                    ax.bar(numeric, y, width=width, color=color, alpha=0.75, label=label)
                else:
                    width = 0.8 / len(series_list)
                    positions = [
                        i + (index - (len(series_list) - 1) / 2) * width for i in range(len(x))
                    ]
                    ax.bar(positions, y, width=width, color=color, label=label)
                    ax.set_xticks(range(len(x)), [textwrap.fill(str(value), 16) for value in x])
                if all(value >= 0 for value in y):
                    ax.set_ylim(bottom=0)
            elif kind == "hist":
                ax.hist(x, bins=panel.get("bins", 20), color=color, alpha=0.65, label=label)
            else:
                raise ValueError(f"Unsupported lesson figure kind: {kind}")
        ax.set(
            xlabel=panel["xlabel"],
            ylabel=panel["ylabel"],
            title=textwrap.fill(panel.get("title", ""), 42 if count > 1 else 70),
        )
        if "ylim" in panel:
            ax.set_ylim(panel["ylim"])
        if "xlim" in panel:
            ax.set_xlim(panel["xlim"])
        if panel.get("zero_line"):
            ax.axhline(0, color="#8c9baa", lw=1)
        if len(series_list) > 1:
            if panel.get("legend") == "above":
                ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01))
                ax.set_title(ax.get_title(), pad=16 + 12 * len(series_list))
            else:
                ax.legend(frameon=False, fontsize=8)
        if count > 1:
            ax.tick_params(labelsize=8)
    return fig, list(axes.flat)


def make_figure(lab, reference):
    data = reference["chart_data"]
    if "figure" in data or "panels" in data:
        fig, axes = generic_figure(data)
    elif lab["number"] == 4:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        rows, line, residual = data["scatter"], data["fitted_line"], data["residuals"]
        axes[0].scatter(
            [r["class_size"] for r in rows],
            [r["test_score"] for r in rows],
            s=11,
            color=BLUE,
            alpha=0.45,
            edgecolors="none",
        )
        axes[0].plot(
            [r["class_size"] for r in line],
            [r["test_score"] for r in line],
            color=INK,
            linewidth=1.8,
            label="OLS fitted mean",
        )
        axes[0].set(
            xlabel="Students per class",
            ylabel="Test score (points)",
            title="The fitted association",
        )
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].scatter(
            [r["class_size"] for r in residual],
            [r["residual"] for r in residual],
            s=11,
            color=BLUE,
            alpha=0.45,
            edgecolors="none",
        )
        axes[1].axhline(0, color=INK, lw=1)
        axes[1].set(
            xlabel="Students per class", ylabel="OLS residual (points)", title="Residual variation"
        )
    elif lab["number"] == 6:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        rows = data["education_coefficients"]
        for index, row in enumerate(rows):
            axes[0].errorbar(
                row["estimate"],
                index,
                xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
                fmt="o" if index == 0 else "s",
                color=BLUE if index == 0 else INK,
                capsize=4,
                markersize=6,
                linewidth=1.6,
            )
        axes[0].set(
            yticks=[0, 1],
            yticklabels=["Education only", "With experience"],
            xlabel="Log earnings per year of education",
            title="Same 588 workers; 95% HC1 intervals",
        )
        axes[0].invert_yaxis()
        axes[0].axvline(0, color="#a5b3c1", lw=1, linestyle="--")
        partial = data["partial_regression"]
        axes[1].scatter(
            [r["residual_education"] for r in partial],
            [r["residual_log_earnings"] for r in partial],
            s=9,
            color=BLUE,
            alpha=0.35,
            edgecolors="none",
        )
        xs = [
            min(r["residual_education"] for r in partial),
            max(r["residual_education"] for r in partial),
        ]
        slope = reference["summary"]["adjusted_education_coefficient"]
        axes[1].plot(xs, [slope * x for x in xs], color=INK, lw=1.8)
        axes[1].set(
            xlabel="Education after removing experience (years)",
            ylabel="Log earnings after removing experience",
            title="The partial relationship (FWL)",
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        rows = data["group_means"]
        years = [r["year"] for r in rows]
        for field, label, color, marker in [
            ("control", "Never treated", BLUE, "o"),
            ("treated", "Treated", INK, "s"),
        ]:
            axes[0].plot(
                years,
                [r[field] for r in rows],
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
                label=label,
            )
        axes[0].set(
            xlabel="Year", ylabel="Employment rate (%)", title="Group means in the synthetic panel"
        )
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].plot(
            years, [r["gap"] for r in rows], color=INK, marker="s", lw=1.8, label="Baseline outcome"
        )
        axes[1].plot(
            years,
            [r["gap"] for r in data["confounded_group_means"]],
            color=GOLD,
            marker="o",
            lw=1.8,
            linestyle="--",
            label="With unrelated post-policy shock",
        )
        axes[1].set(
            xlabel="Year",
            ylabel="Treated minus control (pp)",
            title="A concurrent shock changes attribution",
        )
        axes[1].legend(frameon=False, fontsize=8, loc="upper left")
        for ax in axes:
            ax.axvline(2017.5, color="#7c8995", linestyle=":", lw=1.4)
            ax.set_xticks(years)
            ax.tick_params(axis="x", labelsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#e8edf2", lw=0.6)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(10)
    path = SOURCE / "labs" / lab["slug"] / "figure.svg"
    fig.savefig(path, metadata={"Date": None, "Creator": "OpenEconometrics Teaching Labs"})
    plt.close(fig)
    # Matplotlib writes spaces before SVG path newlines; keep source diffs clean.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    svg = path.read_text().replace(
        "</defs>", "<style>text{font-family:Arial,Helvetica,sans-serif!important}</style></defs>", 1
    )
    path.write_text(svg)
    return path


def copy_math_assets(destination, katex):
    assets = destination / "assets" / "katex"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ["katex.min.css", "katex.min.js"]:
        shutil.copy2(katex / name, assets / name)
    shutil.copytree(katex / "fonts", assets / "fonts", dirs_exist_ok=True)
    if (katex.parent / "LICENSE").exists():
        shutil.copy2(katex.parent / "LICENSE", assets / "LICENSE")


def reading_links(body):
    return body.replace(
        'href="../../../', 'href="https://github.com/bluearf/openecon/blob/main/docs/'
    )


def book_cover(labs, audience, student_title="Econometrics in practice"):
    instructor = audience == "instructor"
    title = "Instructor guide" if instructor else student_title
    subtitle = (
        "Teaching notes, worked answers and reproducibility"
        if instructor
        else "Twenty labs with OpenEconometrics"
    )
    intro = (
        "Companion material for teaching the course. This edition contains exercise "
        "answers and verification notes and is distributed separately from student materials."
        if instructor
        else "Analytical questions, mathematical explanations, runnable Python, computed results "
        "and exercises. Each lab connects a mathematical idea to a complete worked example."
    )
    links = "".join(
        f'<li><a href="#lab-{lab["number"]}">{lab["number"]:02d}. '
        f"{html.escape(lab['title'])}</a></li>"
        for lab in labs
    )
    return (
        f'<section class="cover"><p class="eyebrow">OpenEconometrics Teaching Labs</p>'
        f'<h1>{title}</h1><p class="lede">{subtitle}</p><p>{intro}</p>'
        "<footer>English edition · Original synthetic data<br>"
        "Original exercises and explanations; no textbook exercises are reproduced.</footer></section>"
        f'<section class="lesson"><h1>Contents</h1><ol class="contents">{links}</ol></section>'
    )


def build(destination, katex, selected=None):
    catalog = json.loads((SOURCE / "catalog.json").read_text())
    labs = [lab for lab in catalog["labs"] if selected is None or lab["number"] in selected]
    if selected is None and len(labs) != catalog.get("expected_labs", 20):
        raise ValueError("The complete course requires exactly twenty labs.")
    configure_plots()
    editions = {audience: destination / audience for audience in ["student", "instructor"]}
    for target in editions.values():
        target.mkdir(parents=True, exist_ok=True)
        copy_math_assets(target, katex)
        for name in ["LICENSE", "NOTICE"]:
            shutil.copy2(ROOT / name, target / name)
    sections = {audience: [] for audience in editions}
    cards = {audience: [] for audience in editions}
    provenance = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE / "catalog.json", SOURCE / "README.md", SOURCE / "INSTRUCTORS.md"]
    }
    shared_instructor_text = (SOURCE / "INSTRUCTORS.md").read_text()
    shared_instructor_text = re.sub(
        r"\]\(instructors/([^)]*)\.md\)", r"](labs/\1/INSTRUCTOR.md)", shared_instructor_text
    )
    (editions["instructor"] / "INSTRUCTORS.md").write_text(shared_instructor_text)
    shutil.copy2(SOURCE / "TASKS.md", editions["instructor"] / "TASKS.md")
    for lab in labs:
        source = SOURCE / "labs" / lab["slug"]
        instructor_source = SOURCE / "instructors" / f"{lab['slug']}.md"
        for path in [
            source / "README.md",
            source / "lab.py",
            source / "reference.json",
            source / "table.tex",
            instructor_source,
        ]:
            if not path.is_file():
                raise ValueError(f"Missing course source: {path}")
        student_text = (source / "README.md").read_text()
        assert_student_content(student_text, lab["slug"])
        reference = json.loads((source / "reference.json").read_text())
        data_file = lab.get("data_file")
        generator = SOURCE / "instructors/generators" / f"{lab['slug']}.py"
        if data_file and not (source / data_file).is_file():
            raise ValueError(f"Missing prepared workbook: {source / data_file}")
        make_figure(lab, reference)
        for path in list(source.glob("*")) + [instructor_source]:
            if path.is_file():
                provenance[str(path.relative_to(ROOT))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if data_file:
            provenance[str(generator.relative_to(ROOT))] = hashlib.sha256(
                generator.read_bytes()
            ).hexdigest()
        for audience, edition in editions.items():
            target = edition / "labs" / lab["slug"]
            target.mkdir(parents=True, exist_ok=True)
            names = ["lab.py", "table.tex", "figure.svg", "README.md"]
            if data_file:
                names.append(data_file)
            if audience == "instructor":
                names.append("reference.json")
            for name in names:
                shutil.copy2(source / name, target / name)
            text = student_text
            if audience == "instructor":
                text = instructor_source.read_text()
                if data_file:
                    shutil.copy2(generator, target / "generator.py")
                    text = text.replace(f"generators/{lab['slug']}.py", "generator.py")
                    text = text.replace("../instructors/generator.py", "generator.py")
                # Instructor source sits in docs/teaching/instructors; its data
                # links become local companions beside the exported lesson.
                text = text.replace(f"../labs/{lab['slug']}/", "")
                text = text.replace(
                    "](../../", "](https://github.com/bluearf/openecon/blob/main/docs/"
                )
                (target / "INSTRUCTOR.md").write_text(text)
            body, toc = render_markdown(text)
            body = reading_links(body)
            header = (
                f'<p class="eyebrow">Lab {lab["number"]:02d} · {audience.title()} edition</p>'
                '<nav class="lesson-links"><a href="lab.py" download>Python lab</a>'
                '<a href="table.tex" download>LaTeX table</a>'
            )
            if data_file:
                header += f'<a href="{data_file}" download>Excel dataset</a>'
            if audience == "instructor":
                header += '<a href="reference.json">Numerical reference</a><a href="README.md">Student source</a>'
            header += "</nav>"
            footer = f"<footer>OpenEconometrics Teaching Labs · {audience.title()} edition · Original synthetic data</footer>"
            (target / "index.html").write_text(
                page(
                    lab["title"],
                    header + body + footer,
                    asset_prefix="../../",
                    toc=toc,
                    audience=audience,
                )
            )
            book_body = body.replace('src="figure.svg"', f'src="labs/{lab["slug"]}/figure.svg"')
            book_body = re.sub(
                r'href="(lab\.py|reference\.json|table\.tex|README\.md|generator\.py|[^"/]+\.xlsx)"',
                lambda m: f'href="labs/{lab["slug"]}/{m[1]}"',
                book_body,
            )
            book_body = re.sub(
                r'id="([^"]+)"', lambda m: f'id="lab{lab["number"]}-{m[1]}"', book_body
            )
            sections[audience].append(
                f'<section class="lesson" id="lab-{lab["number"]}">'
                f'<p class="eyebrow">Lab {lab["number"]:02d}</p>{book_body}{footer}</section>'
            )
            cards[audience].append(
                f'<section class="card"><span class="number">LAB {lab["number"]:02d}</span>'
                f'<h2><a href="labs/{lab["slug"]}/index.html">{html.escape(lab["title"])}</a></h2>'
                f"<p>{html.escape(lab['scope'])}</p></section>"
            )
    for audience, edition in editions.items():
        startup = (
            "# OpenEconometrics Teaching Labs\n\nExtract this entire ZIP, then open "
            "`index.html` in a browser for the offline course. Keep the folder structure intact.\n\n"
            "Each chapter's `labs/` folder contains its Python analysis, LaTeX table and figure. "
            "For Labs 01–02 and 04–20, first import the chapter's named Excel workbook into OpenEconometrics "
            "without renaming it. Its first sheet contains the analysis data and its second sheet defines the variables. "
            "Then open the chapter's complete `lab.py` in an empty Python document and press Run. "
            "If a workbook has been imported several times with the same name, the script uses the most recent import. "
            "Lab 03 creates repeated random samples as part of the sampling exercise. "
            "For ordinary Python, open that chapter folder and run `python lab.py` in an environment with OpenEconometrics installed. "
            "Repository-relative commands in the readings apply to a source checkout. "
            "The PDF is the reading edition; the HTML collection provides navigation to all companion files.\n"
        )
        if SOURCE != ROOT / "docs/teaching":
            startup = (
                f"# {catalog['title']}\n\nExtract the entire ZIP and open `index.html`. "
                "Keep the folder structure intact. Each chapter includes its named Excel workbook, "
                "Python analysis, figure and LaTeX table. Import the workbook into OpenEconometrics "
                "without renaming it, then open the complete `lab.py` in an empty Python document "
                "and run the whole file. For ordinary Python, keep the workbook beside the script. "
                'For another input location, call `run_lab(data_path="/path/to/workbook.xlsx")`. '
                "The PDF is the reading edition; HTML links open all companion files. "
                "All observations are original synthetic teaching data.\n"
            )
        if audience == "instructor":
            startup += "\nThis edition also contains worked answers, full numerical references, original input generators and reproduction notes. Each prepared-data chapter includes `generator.py` for instructor inspection.\n"
        (edition / "START-HERE.md").write_text(startup)
        if audience == "student":
            title = "Econometrics in practice"
            intro = (
                "<h1>Economic questions.<br>Working analyses.<br>Clear interpretations.</h1>"
                "<p>Twenty English labs connect the mathematical idea to a complete analysis in "
                "OpenEconometrics. Import the chapter's named Excel workbook, run its Python file, inspect the result, "
                "and work through the exercises.</p>"
                "<p>Nineteen chapters include fixed Excel observations and a variable dictionary. "
                "Lab 03 generates repeated samples for its sampling exercise. All data are original teaching simulations.</p>"
            )
            if SOURCE != ROOT / "docs/teaching":
                title = catalog["student_title"]
                intro = (
                    f"<h1>{html.escape(title)}</h1><p>Twenty English labs develop the concepts, "
                    "mathematics and interpretation of complete examples in OpenEconometrics. "
                    "Import the named Excel workbook, run the paired Python file, inspect the "
                    "computed result and work through the exercises.</p><p>Every chapter supplies "
                    "fixed observations and a variable dictionary. Simulations resample those "
                    "observations only when resampling is part of the topic.</p>"
                )
        else:
            title = "Instructor guide"
            intro = (
                "<h1>Instructor guide</h1><p>Teaching notes, worked exercise answers and "
                "reproduction evidence for the twenty labs. Distribute the separate student "
                "edition to the class.</p>"
            )
        (edition / "index.html").write_text(
            page(
                title,
                '<p class="eyebrow">OpenEconometrics Teaching Labs</p>'
                + intro
                + '<p><a href="course-book.html">Open the complete edition</a></p>'
                + "".join(cards[audience]),
                audience=audience,
            )
        )
        preface = ""
        if audience == "instructor":
            # The print cover already supplies the chapter contents. Keep the
            # shared guide's teaching/evidence instructions without a second
            # twenty-row chapter index in the combined book.
            book_preface = re.sub(
                r"## Chapter guides[\s\S]*?(?=## Reproduce)", "", shared_instructor_text
            )
            shared_body, _ = render_markdown(book_preface)
            shared_body = re.sub(r'id="([^"]+)"', lambda m: f'id="guide-{m[1]}"', shared_body)
            preface = f'<section class="lesson instructor-preface">{shared_body}</section>'
        (edition / "course-book.html").write_text(
            page(
                title,
                book_cover(labs, audience, catalog.get("student_title", "Econometrics in practice"))
                + preface
                + "".join(sections[audience]),
                book=True,
                audience=audience,
            )
        )
    tables = "\n\\clearpage\n".join(
        (SOURCE / "labs" / lab["slug"] / "table.tex").read_text() for lab in labs
    )
    (editions["instructor"] / "regression-tables.tex").write_text(
        "\\documentclass[11pt]{article}\n\\usepackage[margin=22mm]{geometry}\n"
        "\\usepackage{booktabs,adjustbox,amsmath,amssymb,array}\n\\begin{document}\n"
        "\\section*{OpenEconometrics Teaching Labs: executed tables}\n"
        "Original synthetic teaching data.\n" + tables + "\n\\end{document}\n"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "index.html").write_text(
        page(
            "Teaching Labs review",
            '<p class="eyebrow">OpenEconometrics Teaching Labs</p><h1>Course editions</h1>'
            '<p><a href="student/index.html">Student edition</a>: explanations, code, results and exercises.</p>'
            '<p><a href="instructor/index.html">Instructor edition</a>: teaching notes, answers and reproduction evidence.</p>',
            asset_prefix="student/",
        )
    )
    report = {
        "labs": len(labs),
        "canonical_source_sha256": provenance,
        "student_excluded_files": ["reference.json", "INSTRUCTOR.md", "regression-tables.tex"],
        "math": "local KaTeX; trust=false; strict=error",
        "figures": "verified reference.json chart_data; no refitting in renderer",
    }
    (destination / "build-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Built {len(labs)} lessons in separate student and instructor editions: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "teaching")
    parser.add_argument(
        "--course", choices=["statistics", "microeconomics", "advanced-econometrics"]
    )
    parser.add_argument(
        "--katex-dir", type=Path, default=ROOT / "web" / "node_modules" / "katex" / "dist"
    )
    parser.add_argument(
        "--labs",
        type=int,
        nargs="+",
        help="Build selected lessons during editing; default: all twenty",
    )
    args = parser.parse_args()
    if args.course:
        SOURCE = ROOT / "docs/teaching" / args.course
        if args.output == ROOT / "output/teaching":
            args.output = args.output / args.course
    if not (args.katex_dir / "katex.min.js").is_file():
        parser.error("KaTeX not found. Run npm --prefix web ci or provide --katex-dir.")
    build(args.output.resolve(), args.katex_dir.resolve(), args.labs)
