"""Read-only Mach-O minimum-system and relocated application payload contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import struct


def version(value: int) -> str:
    return ".".join(str(x) for x in (value >> 16, (value >> 8) & 255, value & 255))


def macho_minimums(data: bytes) -> list[dict]:
    """Inspect LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX in thin and fat files."""
    if data[:4] in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        wide = data[:4] == b"\xca\xfe\xba\xbf"
        count = struct.unpack_from(">I", data, 4)[0]
        if count > 64:
            raise ValueError("Unbounded universal binary header")
        rows = []
        for index in range(count):
            pos = 8 + index * (32 if wide else 20)
            offset, size = struct.unpack_from(">QQ" if wide else ">II", data, pos + 8)
            if offset + size > len(data):
                raise ValueError("Truncated universal binary")
            rows.extend(macho_minimums(data[offset:offset + size]))
        return rows
    magics = {b"\xcf\xfa\xed\xfe": ("<", 32), b"\xce\xfa\xed\xfe": ("<", 28),
              b"\xfe\xed\xfa\xcf": (">", 32), b"\xfe\xed\xfa\xce": (">", 28)}
    if data[:4] not in magics:
        return []
    endian, pos = magics[data[:4]]
    cpu, commands, command_bytes = struct.unpack_from(endian + "I8xII", data, 4)
    boundary = pos + command_bytes
    if commands > 4096 or boundary > len(data):
        raise ValueError("Invalid Mach-O load-command header")
    rows = []
    for _ in range(commands):
        command, size = struct.unpack_from(endian + "II", data, pos)
        if size < 8 or pos + size > boundary:
            raise ValueError("Invalid Mach-O load-command length")
        if command == 0x32:  # LC_BUILD_VERSION
            if size < 24:
                raise ValueError("Truncated LC_BUILD_VERSION")
            platform, minimum, sdk = struct.unpack_from(endian + "III", data, pos + 8)
            rows.append({"cpu": cpu, "platform": platform,
                         "minimum": version(minimum), "sdk": version(sdk)})
        elif command == 0x24:  # LC_VERSION_MIN_MACOSX
            if size < 16:
                raise ValueError("Truncated LC_VERSION_MIN_MACOSX")
            minimum, sdk = struct.unpack_from(endian + "II", data, pos + 8)
            rows.append({"cpu": cpu, "platform": 1,
                         "minimum": version(minimum), "sdk": version(sdk)})
        pos += size
    return rows


def audit(app: Path) -> dict:
    app = app.resolve(strict=True)
    metadata = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    files, aliases, binaries = {}, {}, {}
    for path in sorted(app.rglob("*")):
        relative = path.relative_to(app).as_posix()
        if path.is_symlink():
            aliases[relative] = os.readlink(path)
            if Path(aliases[relative]).is_absolute():
                raise ValueError("Application aliases must use relative paths")
            # Every loader alias must remain inside the bundle after relocation.
            path.resolve(strict=True).relative_to(app)
        elif path.is_file():
            data = path.read_bytes()
            files[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            minimums = macho_minimums(data)
            if minimums:
                binaries[relative] = minimums
    minimum = metadata["LSMinimumSystemVersion"]
    limit = tuple(map(int, minimum.split("."))) + (0,) * (3 - len(minimum.split(".")))
    violations = [name for name, rows in binaries.items() if any(
        row["platform"] == 1 and tuple(map(int, row["minimum"].split("."))) > limit
        for row in rows)]
    canonical = json.dumps({"files": files, "aliases": aliases}, sort_keys=True).encode()
    return {"identifier": metadata["CFBundleIdentifier"],
            "version": metadata["CFBundleShortVersionString"],
            "minimum_macos": minimum, "payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "regular_files": len(files), "aliases": len(aliases),
            "macho_files": len(binaries), "minimum_os_violations": violations,
            "binaries": binaries}
