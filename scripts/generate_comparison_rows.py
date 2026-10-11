#!/usr/bin/env python3
"""Index the retained vendor research rows without inferring API equivalence.

This is a documentation/content check. It imports no estimator or runtime.
Exact reviewed label links are opt-in in comparison-row-contracts.json. Remaining
rows retain their original text and are explicitly queued for option review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/econometrics/comparison-row-contracts.json"
OUTPUT = ROOT / "docs/econometrics/comparison-rows.md"
DETAILS = ROOT / "docs/econometrics/comparison-rows"
STATUS = "comparison-status-2026-10-11.md"
URL = re.compile(r"https?://[^\s<>\)\]]+")
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
BULLET = re.compile(r"^\s*[-*]\s+(.+)$")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cells(line: str) -> list[str]:
    """Split source cells, retaining escaped and literal inline-code pipes.

    Some original reports contain an unescaped `mean|constant` option. Preserve
    that source syntax as one logical cell rather than dropping the whole row.
    """
    line = URL.sub(lambda match: match[0].replace("|", "\\|"), line)
    parts = []
    start = 0
    code = False
    for index, character in enumerate(line):
        if character == "`" and (index == 0 or line[index - 1] != "\\"):
            code = not code
        if character == "|" and not code and (index == 0 or line[index - 1] != "\\"):
            parts.append(line[start:index].strip())
            start = index + 1
    parts.append(line[start:].strip())
    return parts[1:-1]


def structural_path(headings: dict[int, str]) -> str:
    # Takeaway/Cited Findings/Inferences/Gaps repeat throughout these reports.
    # H2 and H4 identify inventory sections; H3 stays in the displayed context.
    return " / ".join(headings[level] for level in sorted(headings) if level != 3)


def extract(source: dict) -> list[dict]:
    path = ROOT / source["path"]
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: dict[int, str] = {}
    result: list[dict] = []
    occurrences: Counter = Counter()
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading[1])
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = heading[2]
        table = (
            line.startswith("|") and index + 1 < len(lines) and SEPARATOR.match(lines[index + 1])
        )
        bullet = BULLET.match(line)
        if table:
            header = cells(line)
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                row = cells(lines[index])
                if len(row) != len(header):
                    raise ValueError(f"{source['path']}:{index + 1}: inconsistent table width")
                add_row(
                    result,
                    occurrences,
                    source,
                    headings,
                    index + 1,
                    "table",
                    lines[index],
                    header,
                    row,
                )
                index += 1
            continue
        if bullet:
            # A multi-line bullet is one occurrence, including its continuation.
            start = index
            parts = [line]
            while (
                index + 1 < len(lines)
                and re.match(r"^\s{2,}\S", lines[index + 1])
                and not BULLET.match(lines[index + 1])
            ):
                index += 1
                parts.append(lines[index])
            add_row(
                result,
                occurrences,
                source,
                headings,
                start + 1,
                "bullet",
                " ".join(part.strip() for part in parts),
                [],
                [bullet[1]],
            )
        index += 1
    return result


def add_row(
    result: list,
    occurrences: Counter,
    source: dict,
    headings: dict,
    number: int,
    kind: str,
    text: str,
    header: list,
    row: list,
) -> None:
    section = structural_path(headings)
    identity = "\n".join((source["id"], section, kind, text))
    occurrences[identity] += 1
    result.append(
        {
            "id": f"{source['id']}-{digest(identity)[:12]}-{occurrences[identity]}",
            "line": number,
            "kind": kind,
            "heading": " / ".join(headings[level] for level in sorted(headings)),
            "section": section,
            "text": text,
            "header": header,
            "cells": row,
            "label": row[1] if kind == "table" and row[0].isdigit() else row[0],
            "content_sha256": digest(text),
        }
    )


def route(source: dict, row: dict) -> list[str]:
    if row["section"] not in source["reviewed_sections"]:
        raise ValueError(
            f"unreviewed comparison section in {source['id']}: {row['section']}; "
            "review its family routing before registering it in comparison-row-contracts.json"
        )
    matched = []
    for rule in source.get("section_routes", []):
        if any(part.startswith(rule["heading_prefix"]) for part in row["section"].split(" / ")):
            matched.append(rule)
    if len(matched) > 1:
        raise ValueError(f"ambiguous section rules for {row['id']}")
    return (matched[0] if matched else source)["families"]


def category(source: dict, row: dict) -> str:
    if row["kind"] == "bullet":
        # All bullets remain reachable; prose claims do not become estimators.
        h3 = row["heading"].split(" / ")
        if any(part in {"Inferences", "Gaps", "Takeaway"} for part in h3):
            return "research-note"
        if "`" in row["text"] and any(
            prefix in row["section"] for prefix in source.get("method_bullet_sections", [])
        ):
            return "method-group"
        return "source-note"
    first_header = row["header"][0].lower()
    if first_header in {
        "anahtar",
        "doc set",
        "version (date)",
        "sürüm",
        "theme",
        "tema",
        "library",
    }:
        return "bibliography-or-history"
    return "comparison"


def exact_link(config: dict, source: dict, row: dict) -> dict | None:
    found = [
        entry
        for entry in config["reviewed_api_links"]
        if entry["source"] == source["id"]
        and entry["label"] == row["label"]
        and row["kind"] == "table"
    ]
    if len(found) > 1:
        raise ValueError(f"duplicate exact-label link for {row['id']}")
    return found[0] if found else None


def references(source: dict, row: dict, all_rows: list) -> list[str]:
    direct = URL.findall(row["text"])
    # Resolve Stata's declared bibliography keys, preserving row-level provenance.
    for candidate in all_rows:
        if candidate["kind"] == "table" and candidate["header"][0] == "Anahtar":
            key = candidate["cells"][0]
            if re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", row["text"]):
                direct.extend(URL.findall(candidate["text"]))
    return sorted(set(direct))


def escape(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ")


def issue_link(identifier: int) -> str:
    return f"[MARKET-{identifier}](https://linear.app/bluearf/issue/MARKET-{identifier})"


def repository_text(path: str) -> str:
    """Read a local file or its HEAD body in a sparse managed checkout."""
    local = (ROOT / path).resolve()
    relative = local.relative_to(ROOT).as_posix()
    if local.is_file():
        return local.read_text(encoding="utf-8")
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"missing local/HEAD documentation file {path}")
    return result.stdout


def validate_config(config: dict) -> None:
    if config["schema"] != 1:
        raise ValueError("unsupported comparison contract schema")
    catalogue = json.loads((ROOT / "docs/econometrics/capabilities.generated.json").read_text())
    public_names = set(catalogue["estimators"]) | set(catalogue["auxiliary_exports"])
    status = repository_text(f"docs/econometrics/{STATUS}")
    status_anchors = set(re.findall(r'id=["\']([^"\']+)', status))
    missing_anchors = set(config["families"]) - status_anchors
    if missing_anchors:
        raise ValueError(f"missing method/status anchors: {sorted(missing_anchors)}")
    source_ids = [source["id"] for source in config["sources"]]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate source id")
    for source in config["sources"]:
        if not (ROOT / source["path"]).is_file():
            raise ValueError(f"missing source {source['path']}")
        for rule in [source, *source.get("section_routes", [])]:
            if not rule["families"] or any(
                family not in config["families"] for family in rule["families"]
            ):
                raise ValueError(f"unknown/missing section family in {source['id']}")
    for entry in config["reviewed_api_links"]:
        if entry["source"] not in source_ids or entry["family"] not in config["families"]:
            raise ValueError("unknown reviewed-link source or family")
        for api in entry["apis"]:
            if api not in public_names:
                raise ValueError(f"reviewed API absent from committed generated catalogue: {api}")
        repository_text(str(Path("docs/econometrics") / entry["guide"]))


def generate(config: dict) -> dict[Path, str]:
    validate_config(config)
    sources = [(source, extract(source)) for source in config["sources"]]
    seen: set[str] = set()
    counts: Counter = Counter()
    for source, rows in sources:
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"duplicate row identity: {row['id']}")
            seen.add(row["id"])
            row["category"] = category(source, row)
            row["families"] = route(source, row)
            row["api_link"] = exact_link(config, source, row)
            row["vendor_sources"] = references(source, row, rows)
            row["review_state"] = (
                "needs-review"
                if row["category"] in {"comparison", "method-group"}
                else "context-only"
            )
            counts[row["category"]] += 1
            if row["category"] in {"comparison", "method-group"}:
                counts["reviewed-api-link" if row["api_link"] else "api-null"] += 1
    for entry in config["reviewed_api_links"]:
        if not any(
            source["id"] == entry["source"] and any(row["label"] == entry["label"] for row in rows)
            for source, rows in sources
        ):
            raise ValueError(
                f"reviewed exact label disappeared: {entry['source']} {entry['label']}"
            )

    out = [
        "# Vendor comparison row index and remaining option review",
        "",
        "Generated by `scripts/generate_comparison_rows.py`; run with `--check` to detect missing, changed or stale indexed rows. Inputs and exact-label links are declared in [the row contracts](comparison-row-contracts.json). No estimator or numerical test runs during this documentation check.",
        "",
        f"Review date: **{config['review_date']}**. Baseline source: `{config['source_baseline']}`. Research reports retain their own original dates, wording and vendor evidence. Document changes in this PR are identified by the complete input SHA256 values below; they are not retroactively claimed as files at the baseline commit.",
        "",
        "This index preserves every Markdown table data row and every unordered source bullet in the eleven research inputs, including repeated command occurrences and administrative/history/source notes. Plain prose is source context, not a separately enumerated method. Table headers/separators and newly added reconciliation notices are not method rows. The counts are documentation occurrences, never distinct estimators or a vendor-parity denominator.",
        "",
        f"**{counts['comparison']} comparison table rows**, **{counts['method-group']} method-list bullets**, **{counts['bibliography-or-history']} bibliography/history table rows**, **{counts['source-note']} source-note bullets**, **{counts['research-note']} inference/gap/research-note bullets**. Of comparison rows and method-list bullets, **{counts['reviewed-api-link']} have a reviewed bounded API link** and **{counts['api-null']} retain `API=null`**. Every method/option row still requires its own complete option/evidence review before [MARKET-832](https://linear.app/bluearf/issue/MARKET-832) can close.",
        "",
        "An exact-label API link establishes a documented bounded route only. It does not assert that every vendor option, default, weight law or scientific result agrees. `API=null` means no direct route has been asserted in this index; it does not mean the method is absent. Section routing is a review destination, not algorithm matching. Remaining rows are explicit review work under MARKET-832 and the listed canonical family issues. Existing completed children are not reopened by this index.",
        "",
        "Each family link resolves to the [dated method/status contracts](comparison-status-2026-10-11.md): sample and missingness; weights; VCE/covariance; inference/diagnostics; postestimation; saved-result replay; excluded options; canonical remaining issue; source pin and dated evidence. The separate evidence fields are **source**, **independent scientific reference**, **wheel/frozen**, **installed UI**, **hosted Windows**, **persistence** and **public release**. This row index establishes **none** of the six execution/delivery layers by itself. A guide or registry entry is not an execution receipt. Use each dated receipt's actual source and options, and record unknown layers explicitly during row review.",
        "",
        "Licensed vendor executable comparisons are not required. Independent analytical, author/development reference and synthetic evidence are accepted within their actual scope. Nonzero optimizer convergence remains a failure. CUDA and Developer ID/notarization are deferred; hosted Windows and ad-hoc Mac internal testing follow [AGENTS.md](../../AGENTS.md). No fresh full numerical matrix is requested for this document index.",
        "",
        "## Input coverage",
        "",
        "| Input / generated row detail | Product/version and original date | Role | Indexed occurrences | SHA256 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for source, rows in sources:
        link = "../../" + quote(source["path"], safe="/")
        out.append(
            f"| [{source['id']}]({link}) / [row detail](comparison-rows/{source['id']}.md) | {source['product']} · {source['version']} · {source['recorded_date']} | {source['role']} | {len(rows)} | `{digest((ROOT / source['path']).read_text())}` |"
        )
    out.extend(
        [
            "",
            "## Field conventions",
            "",
            "Row IDs bind the source ID, structural heading, complete retained row text and repeated-occurrence number. They do not depend on line numbers; line links are regenerated after edits. Vendor URLs come from that row or its explicitly named Stata bibliography key. If a row has no direct URL, the source line and its surrounding references remain visible and the vendor-evidence review is pending. Do not replace a missing row-level citation with an invented verification claim.",
            "",
            "A source's product label supplies the original file scope. Where a table combines built-ins, add-ins, beta features or separate products, its exact retained cells decide the actual row classification. In particular, SPSS Statistics ≠ Amos, Stata official ≠ community packages and EViews 14 GA ≠ EViews 15 beta. Mixed-source rows require review; the file-level label does not promote them to built-ins.",
            "",
        ]
    )
    rendered = {OUTPUT: "\n".join(out)}
    for source, rows in sources:
        page = [
            f"# {source['id']}: vendor research row detail",
            "",
            "Generated by `scripts/generate_comparison_rows.py`. [Scope, counts and remaining review](../comparison-rows.md) · [row contracts](../comparison-row-contracts.json).",
            "",
            "Every `needs-review` row still requires complete option/evidence reconciliation under [MARKET-832](https://linear.app/bluearf/issue/MARKET-832), including rows with a bounded API link. `context-only` rows preserve bibliography, history and research notes; they are not counted as methods. `API=null` asserts no direct equivalence and does not mean absence. See the linked family contract for sample/missing/weights/VCE/inference/diagnostics/postestimation/replay and the distinct source/scientific/wheel-frozen/installed/Windows/persistence/public-release evidence fields. This index does not establish those execution/delivery layers.",
            "",
        ]
        page.extend(
            [
                f"{source['product']} · {source['version']} · original report date {source['recorded_date']}. {source['role']}. Full original row text follows its line link; display labels are not used to infer undocumented APIs.",
                "",
                "| Row ID / source line | Original label or bullet | Row type / review state | Review contract / canonical issue | Bounded API link / direct vendor evidence |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            source_link = "../../../" + quote(source["path"], safe="/") + f"#L{row['line']}"
            family_ids = [row["api_link"]["family"]] if row["api_link"] else row["families"]
            contracts = "; ".join(
                f"[{family}](../{STATUS}#{family}) "
                + "/".join(issue_link(issue) for issue in config["families"][family]["issues"])
                for family in family_ids
            )
            if row["category"] in {"comparison", "method-group"}:
                contracts += "; " + issue_link(832) + " option review"
            else:
                contracts = "Source context; not counted as a method."
            linked = row["api_link"]
            api = "`API=null`; no direct equivalence asserted"
            if linked:
                api = (
                    ", ".join(f"`oe.{name}`" for name in linked["apis"])
                    + f" · [{escape(linked['scope'])}](../{quote(linked['guide'])}) · option review pending"
                )
            vendor = (
                ", ".join(
                    f"[vendor {number}]({url.replace('|', '%7C')})"
                    for number, url in enumerate(row["vendor_sources"], 1)
                )
                or "vendor citation review pending; inspect source context"
            )
            label = row["label"] if row["kind"] == "table" else row["cells"][0]
            # The exact full text is retained by the source link and identity hash.
            page.append(
                f'| <a id="{row["id"]}"></a>[`{row["id"]}`]({source_link}) | {escape(label)} | {row["category"]} / **{row["review_state"]}** | {contracts} | {api}; {vendor} |'
            )
        page.append("")
        rendered[DETAILS / f"{source['id']}.md"] = "\n".join(page)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="require the committed row index to match every declared input",
    )
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    try:
        rendered = generate(config)
    except ValueError as exc:
        print(f"Comparison content check failed: {exc}")
        return 1
    expected_details = set(rendered) - {OUTPUT}
    actual_details = {path for path in DETAILS.rglob("*") if path.is_file()}
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, content in rendered.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        extra = sorted(str(path.relative_to(ROOT)) for path in actual_details - expected_details)
        if stale or extra:
            print(
                f"Comparison row output is stale/missing: {stale}; unexpected detail files: {extra}. Run scripts/generate_comparison_rows.py."
            )
            return 1
        print("Comparison row index matches every declared research input and exact API link.")
        return 0
    DETAILS.mkdir(parents=True, exist_ok=True)
    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")
    for stale in actual_details - expected_details:
        if stale.suffix == ".md":
            stale.unlink()
    print(f"Wrote summary and {len(expected_details)} source detail pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
