"""Check local course links and package separate student/instructor editions.

Run after building HTML and printing each edition to its own PDF. This command
normalizes generated PDF links and writes two reviewed ZIP files.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject


ROOT = Path(__file__).resolve().parents[1]


def portable_pdf_links(folder, pdf_name, audience):
    """Replace preview-server URLs with paths inside the extracted edition."""
    pdf = folder / pdf_name
    reader = PdfReader(pdf)
    writer = PdfWriter(clone_from=reader)
    checked = changed = 0
    for page in writer.pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if "/URI" not in action:
                continue
            uri = str(action["/URI"])
            parsed = urlsplit(uri)
            relative = parsed.path
            if parsed.hostname in {"localhost", "127.0.0.1"}:
                prefix = f"/{audience}/"
                if prefix not in relative:
                    raise ValueError(f"Cross-edition preview URL in {pdf}: {uri}")
                relative = relative.split(prefix, 1)[1]
                action[NameObject("/URI")] = TextStringObject(relative)
                changed += 1
            elif parsed.scheme == "file":
                target = Path(unquote(relative)).resolve()
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError(f"PDF file link leaves its edition: {uri}")
                relative = target.relative_to(folder.resolve()).as_posix()
                action[NameObject("/URI")] = TextStringObject(relative)
                changed += 1
            elif parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (folder / unquote(relative)).resolve()
            if not target.is_relative_to(folder.resolve()) or not target.is_file():
                raise ValueError(f"Broken local PDF link in {pdf}: {relative}")
            checked += 1
    if changed:
        temporary = pdf.with_suffix(".portable.pdf")
        writer.write(temporary)
        if len(PdfReader(temporary).pages) != len(reader.pages):
            raise ValueError(f"PDF link rewrite changed the page count: {pdf}")
        temporary.replace(pdf)
    return checked


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def verify_edition(folder, labs, pdf_name, audience, source=None):
    pdf = folder / pdf_name
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError(f"Print this edition to its own PDF first: {pdf}")
    for lab in labs:
        chapter = folder / "labs" / lab["slug"]
        names = ["index.html", "README.md", "lab.py", "table.tex", "figure.svg"]
        if lab.get("data_file"):
            names.append(lab["data_file"])
            canonical = (source or ROOT / "docs/teaching") / "labs" / lab["slug"] / lab["data_file"]
            if (chapter / lab["data_file"]).read_bytes() != canonical.read_bytes():
                raise ValueError(f"Packaged workbook differs from the supplied source: {chapter}")
        if audience == "instructor":
            names += ["INSTRUCTOR.md", "reference.json"]
            if lab.get("data_file"):
                names.append("generator.py")
        canonical_folder = (source or ROOT / "docs/teaching") / "labs" / lab["slug"]
        identical = ["README.md", "lab.py", "table.tex", "figure.svg"]
        if audience == "instructor":
            identical.append("reference.json")
        for name in identical:
            if (chapter / name).read_bytes() != (canonical_folder / name).read_bytes():
                raise ValueError(f"Packaged source differs from verified source: {chapter / name}")
        for name in names:
            if not (chapter / name).is_file():
                raise ValueError(f"Missing packaged chapter file: {chapter / name}")
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    for path in files:
        if audience == "student" and (
            path.suffix == ".json"
            or path.name == "INSTRUCTOR.md"
            or "instructor" in path.name.lower()
            or path.name == "generator.py"
        ):
            raise ValueError(f"Instructor material in student edition: {path}")
        if path.suffix == ".html":
            parser = LocalLinks()
            parser.feed(path.read_text())
            for link in parser.links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(folder.resolve()) or not target.is_file():
                    raise ValueError(f"Broken or cross-edition local link in {path}: {link}")
        if audience == "student" and path.suffix in {".md", ".html"}:
            text = path.read_text()
            for pattern in [
                r"(?im)^#{1,6} .*instructor",
                r"(?im)^#{1,6} .*worked answers",
                r"(?im)^#{1,6}\s+(?:\d+[.)]\s+)?(?:reproduce(?:\s|$)|reproduction(?:\s|$)|reproducibility\s+(?:evidence|notes))",
                r"(?i)no previous .*experience is required",
                r"(?i)(?:duration|estimated time|takes|allow)\s*[:=]?\s*\d+(?:\s*[–-]\s*\d+)?\s+minutes\b",
            ]:
                if re.search(pattern, text):
                    raise ValueError(f"Excluded student content in {path}: {pattern}")
    return files


def package(destination, course=None):
    source = ROOT / "docs/teaching"
    if course:
        source = source / course
    catalog = json.loads((source / "catalog.json").read_text())
    if len(catalog["labs"]) != 20:
        raise ValueError("The delivery requires all twenty chapters.")
    receipts = {}
    for audience, pdf_name in [
        ("student", f"{course or 'econometrics'}-labs.pdf"),
        ("instructor", "instructor-guide.pdf"),
    ]:
        folder = destination / audience
        pdf_links = portable_pdf_links(folder, pdf_name, audience)
        files = verify_edition(folder, catalog["labs"], pdf_name, audience, source)
        archive = destination / f"{course or 'econometrics'}-{audience}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(
                    path,
                    arcname=str(
                        Path(f"{course or 'econometrics'}-{audience}") / path.relative_to(folder)
                    ),
                )
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None or len(bundle.namelist()) != len(files):
                raise ValueError(f"Archive readback failed: {archive}")
            if audience == "student" and any(
                name.endswith("reference.json") or "INSTRUCTOR.md" in name
                for name in bundle.namelist()
            ):
                raise ValueError("Student archive contains instructor evidence or answers.")
        receipts[audience] = {
            "archive": archive.name,
            "files": len(files),
            "chapters": 20,
            "prepared_workbooks": sum(bool(lab.get("data_file")) for lab in catalog["labs"]),
            "local_links": "passed",
            "portable_pdf_local_links": pdf_links,
            "archive_readback": "passed",
            "pdf": pdf_name,
            "instructor_material_excluded": audience == "student",
        }
    (destination / "delivery-verification.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/teaching")
    parser.add_argument(
        "--course", choices=["statistics", "microeconomics", "advanced-econometrics"]
    )
    args = parser.parse_args()
    if args.course and args.output == ROOT / "output/teaching":
        args.output = args.output / args.course
    package(args.output.resolve(), args.course)
