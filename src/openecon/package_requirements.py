"""Bounded, inert PyPI requirement specifications shared by local and cloud code."""
from __future__ import annotations

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def parse_specifications(values, *, evaluate_markers: bool = True) -> tuple[list[dict], list[str]]:
    from openecon.project_packages import PackageError, canonical_name, canonical_version

    if not isinstance(values, (tuple, list)) or len(values) > 100:
        raise PackageError("INVALID_PACKAGE", "Specify at most 100 package requirements.")
    if sum(len(value.encode("utf-8")) for value in values if isinstance(value, str)) > 32 * 1024:
        raise PackageError("INVALID_PACKAGE", "Package specifications exceed the 32 KiB limit.")
    rows, normalized, seen = [], [], set()
    for value in values:
        if (not isinstance(value, str) or not 1 <= len(value) <= 2000
                or value != value.strip() or any(ord(char) < 32 for char in value)):
            raise PackageError("INVALID_PACKAGE", "Package requirements must be single-line strings.")
        try:
            requirement = Requirement(value)
        except (InvalidRequirement, RecursionError) as exc:
            raise PackageError("INVALID_PACKAGE", "Use a valid PyPI requirement, version range or extra.") from exc
        if requirement.url is not None:
            raise PackageError("UNSUPPORTED_PACKAGE_SOURCE", "Direct URLs, Git sources and editable builds are not supported by this wheel installer.")
        name = canonical_name(requirement.name)
        if name in seen:
            raise PackageError("INVALID_PACKAGE", "Specify each package only once.")
        seen.add(name)
        requirement.name = name
        requirement.extras = {canonicalize_name(extra) for extra in requirement.extras}
        version = None
        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==" and "*" not in specifiers[0].version:
            version = canonical_version(specifiers[0].version)
            requirement.specifier = type(requirement.specifier)("==" + version)
        if any(specifier.operator == "===" for specifier in specifiers):
            raise PackageError("INVALID_PACKAGE", "Package versions must follow PEP 440.")
        normalized.append(str(requirement))
        if not evaluate_markers or requirement.marker is None or requirement.marker.evaluate():
            rows.append({"name": name, "version": version})
    return rows, normalized


def satisfies(specification: str, version: str) -> bool:
    return Requirement(specification).specifier.contains(Version(version), prereleases=True)


def validate_specification_pins(specifications, requirements: list[dict], core: dict) -> list[str]:
    """Validate syntax/pins without applying the control server's OS markers."""
    from openecon.project_packages import PackageError

    rows, normalized = parse_specifications(specifications, evaluate_markers=False)
    by_name = dict(zip((row["name"] for row in rows), normalized, strict=True))
    if any(row["name"] not in by_name or not satisfies(by_name[row["name"]], row["version"])
           for row in requirements):
        raise PackageError("INVALID_MANIFEST", "Requested specifications must match the locked root versions.")
    declared = {row["name"] for row in requirements} | set(core)
    # Inactive platform requirements can have no installed root on this host.
    for row, specification in zip(rows, normalized, strict=True):
        if row["name"] in core and not satisfies(specification, core[row["name"]]):
            raise PackageError("INCOMPATIBLE_CORE", "A specification would replace OpenEconometrics's fixed core.")
        if row["name"] not in declared and Requirement(specification).marker is None:
            raise PackageError("INVALID_MANIFEST", "Every unconditional specification needs a locked root.")
    return sorted(normalized, key=lambda value: canonicalize_name(Requirement(value).name))
