"""Read-only bytecode/static-asset equality check for a locally built QA Mac bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import CodeType

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def normalized(code):
    return code.replace(co_filename="<bundle>", co_consts=tuple(
        normalized(v) if isinstance(v, CodeType) else v for v in code.co_consts))


def verify(app):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    checked, excluded = [], []
    for source in (ROOT / "src/openecon", ROOT / "packages/openecon-charts/src/openecon_charts"):
        for path in sorted(source.rglob("*.py")):
            module = ".".join(path.relative_to(source.parent).with_suffix("").parts).removesuffix(".__init__")
            if module not in archive.toc:
                excluded.append(module)
                continue
            expected = compile(path.read_text(), str(path), "exec", dont_inherit=True)
            if normalized(expected) != normalized(archive.extract(module)):
                raise RuntimeError(f"Frozen source differs: {module}")
            checked.append({"module": module, "sha256": digest(path)})
    forbidden = [name for name in archive.toc if name.split(".")[0] in {
        "scipy", "statsmodels", "arch", "linearmodels", "sklearn"}]
    assert not forbidden, forbidden
    static = runtime.parent / "_internal/openecon/static"
    files = []
    for path in sorted((ROOT / "src/openecon/static").rglob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT / "src/openecon/static")
            bundled = static / relative
            assert bundled.is_file() and digest(path) == digest(bundled), relative
            files.append({"file": str(relative), "sha256": digest(path), "bytes": path.stat().st_size})
    return {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "owned_modules": len(checked), "checked": checked, "excluded": excluded,
            "forbidden_in_bundle": forbidden, "runtime_sha256": digest(runtime), "qa_ui_assets": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(args.app.resolve(strict=True))
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"modules_matched": receipt["owned_modules"], "ui_assets_matched": len(receipt["qa_ui_assets"])}))
