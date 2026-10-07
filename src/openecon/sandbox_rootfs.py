"""Image-build-only construction of the clean sandbox lower filesystem.

No runtime file from /tmp, /run, /proc, /home or /app is copied. Hard links keep
the CPU Torch runtime in one image layer; sandbox --write supplies a separate
copy-on-write overlay, never write access to these lower inodes.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path("/opt/openecon-sandbox-rootfs")


def link_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, destination, copy_function=os.link, symlinks=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)


def build() -> None:
    if os.geteuid() != 0 or ROOT.exists():
        raise RuntimeError("Build the clean rootfs exactly once as image-build root.")
    ROOT.mkdir(parents=True)
    # Runtime libraries and Python only. No launcher, host credentials, package
    # manager databases, application checkout, or parent state enter the rootfs.
    for path in ("/usr/lib", "/usr/local/lib", "/opt/venv"):
        link_tree(Path(path), ROOT / path.lstrip("/"))
    for path in ("/lib", "/lib64", "/usr/lib64"):
        source = Path(path)
        if source.exists() or source.is_symlink():
            link_tree(source, ROOT / path.lstrip("/"))
    for path in ("/usr/local/bin/python", "/usr/local/bin/python3",
                 f"/usr/local/bin/python{sys.version_info.major}.{sys.version_info.minor}",
                 "/usr/bin/dash", "/usr/bin/sh", "/usr/bin/env"):
        link_tree(Path(path), ROOT / path.lstrip("/"))
    (ROOT / "bin").symlink_to("usr/bin")
    (ROOT / "etc/ssl").mkdir(parents=True)
    # Official Python images resolve libpython from /usr/local/lib through the
    # image's loader cache. All referenced runtime library paths were copied.
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"):
        source = Path(path)
        if source.exists():
            link_tree(source, ROOT / path.lstrip("/"))
    # The Debian certificates directory links to /usr/share/ca-certificates.
    link_tree(Path("/etc/ssl/certs"), ROOT / "etc/ssl/certs")
    link_tree(Path("/usr/share/ca-certificates"), ROOT / "usr/share/ca-certificates")
    (ROOT / "etc/passwd").write_text("root:x:0:0:root:/:/bin/sh\nopenecon:x:10001:10001::/tmp:/bin/sh\n")
    (ROOT / "etc/group").write_text("root:x:0:\nopenecon:x:10001:\n")
    (ROOT / "etc/nsswitch.conf").write_text("passwd: files\ngroup: files\nhosts: files dns\n")
    (ROOT / "etc/hosts").write_text(
        "127.0.0.1 localhost\n::1 localhost\n169.254.169.254 metadata.google.internal\n")
    # Custom rootfs launch does not supply resolv.conf. These fixed public
    # resolvers were verified inside the managed guest with active capabilities
    # cleared; no builder/parent DNS settings or mounts enter the guest.
    (ROOT / "etc/resolv.conf").write_text(
        "nameserver 8.8.8.8\nnameserver 8.8.4.4\noptions timeout:1 attempts:2\n")
    for name in ("tmp", "run", "proc", "dev", "home"):
        (ROOT / name).mkdir()
    site = ROOT / f"opt/venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    for item in site.iterdir():
        normalized = item.name.lower().replace("-", "_")
        if normalized.startswith(("google", "firebase", "grpc", "proto_plus", "protobuf")):
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    package = site / "openecon"
    # Keep team_job's data-transfer adapter, but omit every privileged control
    # endpoint/client and the broker itself from the child runtime.
    for item in package.glob("team_*"):
        if item.name not in {"team_job.py", "team_sandbox_worker.py"}:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    for name in ("cloud.py", "sandbox_rootfs.py", "server.py", "cli.py"):
        (package / name).unlink(missing_ok=True)
    # Compiled copies must not reintroduce the omitted control modules.
    for item in package.rglob("__pycache__"):
        shutil.rmtree(item)
    (ROOT / ".openecon-clean-rootfs").write_text("openecon-clean-rootfs-v1\n")
    for directory, dirs, files in os.walk(ROOT):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
        for name in dirs:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o555)
    os.chmod(ROOT, 0o555)
    # A namespace-local upper layer backs all writes. The empty lower /tmp needs
    # this mode for both root-started and UID-10001-started sandbox launchers.
    # The trusted broker never writes under ROOT; it checks this directory empty.
    os.chmod(ROOT / "tmp", 0o1777)


if __name__ == "__main__":
    build()
