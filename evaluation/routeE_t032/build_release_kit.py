#!/usr/bin/env python3
"""Build sanitized allowlisted release kit for T-032 Route E preflight."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP_NAME = "routeE-t032-kit-v1.zip"
ALLOWED_SUFFIXES = {".py", ".sh", ".md"}
EXCLUDED_FILES = {ZIP_NAME, f"{ZIP_NAME}.sha256", "SHA256SUMS"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_kit() -> tuple[Path, str, int]:
    files = [
        p
        for p in sorted(HERE.iterdir())
        if p.is_file()
        and p.name not in EXCLUDED_FILES
        and p.suffix in ALLOWED_SUFFIXES
        and not p.name.startswith(".")
    ]
    sums_lines = [f"{sha256_file(p)}  {p.name}" for p in files]
    sums_path = HERE / "SHA256SUMS"
    sums_path.write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    zip_path = HERE / ZIP_NAME
    all_for_zip = files + [sums_path]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in all_for_zip:
            info = zipfile.ZipInfo(f"routeE_t032/{p.name}", date_time=(2026, 11, 27, 0, 0, 0))
            info.external_attr = 0o755 << 16 if p.suffix == ".sh" else 0o644 << 16
            zf.writestr(info, p.read_bytes())

    z_sha = sha256_file(zip_path)
    z_size = zip_path.stat().st_size
    (HERE / f"{ZIP_NAME}.sha256").write_text(f"{z_sha}  {ZIP_NAME}\n", encoding="utf-8")
    return zip_path, z_sha, z_size


if __name__ == "__main__":
    zp, sha, sz = build_kit()
    print(f"BUILT {zp.name} bytes={sz} sha256={sha}")
