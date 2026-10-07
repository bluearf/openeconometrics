"""Explicit, synchronous project package installation from workspace Python."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import io
from pathlib import Path
import re
import shlex
import tokenize

from openecon.project_packages import PackageError, canonical_name, canonical_version

_installer = ContextVar("openecon_script_installer", default=None)


def parse_requirements(requirements: tuple[str, ...] | list[str]) -> list[dict]:
    """Accept plain PyPI names or exact pins, never shell or installer options."""
    if not isinstance(requirements, (tuple, list)) or not requirements or len(requirements) > 100:
        raise PackageError("INVALID_PACKAGE", "Specify between 1 and 100 package names.")
    result, seen = [], set()
    for requirement in requirements:
        if not isinstance(requirement, str):
            raise PackageError("INVALID_PACKAGE", "Package requirements must be strings.")
        parts = requirement.split("==")
        if len(parts) > 2:
            raise PackageError("INVALID_PACKAGE", "Use a package name or one exact name==version pin.")
        name = canonical_name(parts[0])
        version = canonical_version(parts[1]) if len(parts) == 2 else None
        if name in seen:
            raise PackageError("INVALID_PACKAGE", "Specify each package only once.")
        seen.add(name)
        result.append({"name": name, "version": version})
    return result


@contextmanager
def _use_installer(callback):
    token = _installer.set(callback)
    try:
        yield
    finally:
        _installer.reset(token)


def _read_requirements_file(path: str | Path) -> list[str]:
    """Read bounded local requirements, including relative -r includes."""
    if not isinstance(path, (str, Path)):
        raise PackageError("INVALID_PACKAGE", "Specify a local requirements file.")
    entries, seen, total = [], set(), 0

    def read(candidate: Path, depth: int):
        nonlocal total
        candidate = candidate.resolve()
        if candidate in seen or depth > 8 or len(seen) >= 20:
            raise PackageError("INVALID_PACKAGE", "Requirements includes are cyclic or too deep.")
        seen.add(candidate)
        try:
            if not candidate.is_file() or candidate.stat().st_size > 256 * 1024:
                raise PackageError("INVALID_PACKAGE", "Requirements files must be local files under 256 KiB.")
            content = candidate.read_bytes()
            total += len(content)
            if total > 256 * 1024:
                raise PackageError("INVALID_PACKAGE", "Requirements files exceed the 256 KiB limit.")
            text = content.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise PackageError("INVALID_PACKAGE", "The local requirements file could not be read.") from exc
        for line in text.splitlines():
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r ", "--requirement ", "--requirement=")):
                try:
                    words = shlex.split(line)
                except ValueError as exc:
                    raise PackageError("INVALID_PACKAGE", "Invalid requirements include.") from exc
                if len(words) == 1 and words[0].startswith("--requirement="):
                    nested = words[0].split("=", 1)[1]
                elif len(words) == 2:
                    nested = words[1]
                else:
                    raise PackageError("INVALID_PACKAGE", "Specify one local requirements include.")
                read(candidate.parent / nested, depth + 1)
            else:
                entries.append(line)
                if len(entries) > 100:
                    raise PackageError("INVALID_PACKAGE", "Specify at most 100 package requirements.")

    read(Path(path), 0)
    return entries


def install(*requirements: str, installer: str = "pip",
            requirements_file: str | Path | None = None, upgrade: bool = False,
            source_options: dict | None = None) -> None:
    """Install project libraries, then continue the same Python script.

    Example: ``oe.install("humanize", "polars==1.44.2")``. Names are resolved
    through the desktop's bundled installer and saved as exact project pins.
    """
    from openecon.package_requirements import parse_specifications
    if not isinstance(installer, str) or installer not in {"pip", "uv"} or type(upgrade) is not bool:
        raise PackageError("INVALID_PACKAGE", "Use pip or uv, with a boolean upgrade option.")
    values = list(requirements)
    if requirements_file is not None:
        values.extend(_read_requirements_file(requirements_file))
    if not values:
        raise PackageError("INVALID_PACKAGE", "Specify a package or a nonempty requirements file.")
    requested, specifications = parse_specifications(values)
    if source_options is not None:
        from openecon.package_sources import validate_sources
        source_options = validate_sources(source_options, {row["name"] for row in requested})
    callback = _installer.get()
    if callback is None:
        raise PackageError(
            "PROJECT_INSTALL_UNAVAILABLE",
            "Run package installation in an OpenEconometrics desktop project.",
        )
    if not requested:
        print("No package requirements apply to this computer.")
        return
    try:
        simple = parse_requirements(values) == requested
    except PackageError:
        simple = False
    options = {}
    if installer != "pip":
        options["installer"] = installer
    if not simple:
        options["specifications"] = specifications
    if upgrade:
        options["upgrade"] = True
    if source_options:
        options["source_options"] = source_options
    receipt = callback(requested, **options)
    installed = ", ".join(f"{row['name']}=={row['version']}" for row in receipt["installed"])
    print(f"Packages ready: {installed}")


def _parse_install_command(installer: str, words: list[str]) -> dict:
    if installer == "uv" and words[:2] == ["pip", "install"]:
        action = ["pip", "install"]
    elif installer == "uv" and words[:1] == ["add"]:
        action = ["add"]
    elif installer == "pip" and words[:1] == ["install"]:
        action = ["install"]
    else:
        raise PackageError("INVALID_PACKAGE", "Use pip install, uv pip install or uv add.")
    if words[:len(action)] != action:
        raise PackageError("INVALID_PACKAGE", "Use pip install, uv pip install or uv add.")
    packages, file, upgrade = [], None, False
    source_options={}
    index = len(action)
    while index < len(words):
        value = words[index]
        if value in {"--upgrade", "-U"}:
            upgrade = True
        elif value == "--index-url" or value.startswith("--index-url="):
            if "index_url" in source_options:
                raise PackageError("INVALID_SOURCE", "Specify one --index-url.")
            if value=="--index-url":
                index+=1
                if index>=len(words):
                    raise PackageError("INVALID_SOURCE", "Specify a HTTPS index URL.")
                source_options["index_url"]=words[index]
            else:
                source_options["index_url"]=value.split("=",1)[1]
        elif value in {"-r", "--requirement"}:
            index += 1
            if index >= len(words) or file is not None:
                raise PackageError("INVALID_PACKAGE", "Specify one local requirements file after -r.")
            file = words[index]
        elif value.startswith("--requirement=") or value.startswith("-r"):
            if file is not None:
                raise PackageError("INVALID_PACKAGE", "Specify one local requirements file.")
            file = value.split("=", 1)[1] if value.startswith("--requirement=") else value[2:]
            if not file:
                raise PackageError("INVALID_PACKAGE", "Specify a local requirements file.")
        elif value.startswith("-"):
            raise PackageError("UNSUPPORTED_INSTALL_OPTION", "This workspace supports package names, ranges, extras, -r and --upgrade.")
        else:
            packages.append(value)
        index += 1
    if not packages and file is None:
        raise PackageError("INVALID_PACKAGE", "Specify a package or a local requirements file.")
    from openecon.package_requirements import parse_specifications
    parse_specifications(packages)
    result={"requirements": packages, "installer": installer, "requirements_file": file, "upgrade": upgrade}
    if source_options:
        from openecon.package_sources import validate_sources
        result["source_options"]=validate_sources(source_options,set())
    return result


def install_command(installer: str, words: list[str]) -> None:
    options = _parse_install_command(installer, words)
    packages = options.pop("requirements")
    install(*packages, **options)


def rewrite_install_commands(code: str) -> str:
    """Translate %pip/%uv and !pip/!uv installs, preserving Python strings."""
    protected = set()
    command_lines = {index for index, line in enumerate(code.splitlines(), start=1)
                     if re.match(r"\s*[!%](?:pip|uv)[ \t]+", line)}
    interpolation_start = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            kind = tokenize.tok_name[token.type]
            if kind.endswith("STRING_START"):
                interpolation_start.append(token.start[0])
            elif kind.endswith("STRING_END") and interpolation_start:
                start = interpolation_start.pop()
                protected.update(range(start, token.end[0] + 1))
            elif token.type == tokenize.STRING:
                # Quotes are valid command arguments. Only real Python literals
                # protect a line; a multiline Python string protects its body.
                if token.start[0] != token.end[0] or token.start[0] not in command_lines:
                    protected.update(range(token.start[0], token.end[0] + 1))
    except (tokenize.TokenError, IndentationError):
        # Python's parser will report malformed code before anything executes.
        if interpolation_start:
            protected.update(range(min(interpolation_start), len(code.splitlines()) + 1))
    lines = code.splitlines(keepends=True)
    for index, line in enumerate(lines):
        text = line.lstrip(" \t")
        match = re.match(r"[!%](pip|uv)(?:[ \t]+)(.*)", text)
        if index + 1 in protected or match is None:
            continue
        try:
            words = shlex.split(match.group(2), comments=True)
        except ValueError as exc:
            raise PackageError("INVALID_PACKAGE", "Invalid %pip install command.") from exc
        installer = match.group(1)
        options = _parse_install_command(installer, words)
        indent = line[:len(line) - len(text)]
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        # Local files are opened when this line executes, respecting conditions.
        if installer == "pip" and options["requirements_file"] is None and not options["upgrade"]:
            arguments = ", ".join(repr(word) for word in options["requirements"])
            call = f".install({arguments})"
        else:
            call = f".install_command({installer!r}, {words!r})"
        lines[index] = (indent + "__import__('openecon.script_packages', fromlist=['install'])"
                        + call + newline)
    return "".join(lines)
