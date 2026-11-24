#!/usr/bin/env python3
"""Download and SHA-256 verify eth-nlped/Eduardo-27b (commit 3bbf17e) on Pro 5000.

Downloads from public hf-mirror without any HF token or proxy into RAM disk
(/dev/shm/zhujie_a_group/Eduardo-27b-3bbf17e by default), resuming partial shard
downloads and verifying exact byte size and SHA-256 of all 9 model files.
Stops immediately if authentication/login is required.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "eth-nlped/Eduardo-27b"
REVISION = "3bbf17e44287eadbfda1fa09bb73a1bade407ff2"
DEFAULT_MIRROR_BASE = "https://hf-mirror.com"
DEFAULT_DEST = Path("/dev/shm/zhujie_a_group/Eduardo-27b-3bbf17e")

# Exact sizes and SHA-256 digests pinned to HuggingFace commit 3bbf17e44287eadbfda1fa09bb73a1bade407ff2.
# Includes the 6 files in T-022 TUTOR_LOCK plus the 3 required safetensors/config companion files.
MODEL_FILES: dict[str, tuple[int, str]] = {
    "chat_template.jinja": (9460, "2fbf099c2eb9284b19cc6188139e6f8aa840a93f4f9fcc937b3b89ae9b4fa837"),
    "config.json": (3763, "ddeb906bee82e415882d04790e045a961503ac86ec08e107010dffc18cd8c83b"),
    "generation_config.json": (213, "5264f0a3e636da545410862c0076c51e2efbb3ebe18a1d129d738d7c68f7c6a6"),
    "model-00001-of-00002.safetensors": (49972377736, "626abc50901718cd92703c081963ff16f42dff5ec45133d76f201999ec7ac4b4"),
    "model-00002-of-00002.safetensors": (4741230952, "7be58da8a22afb664ebdfdc9226c2ae30790ccd2a74ff076f8cfff92174d6cbf"),
    "model.safetensors.index.json": (111110, "0cc3f04468ef7cbee9fae18b0080428b6aa1f5da0d8cf3b104327e6824ca198f"),
    "processor_config.json": (1191, "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"),
    "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1165, "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"),
}

LOCKED_SIX_FILES = (
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "MISSING"
    got_size = path.stat().st_size
    if got_size != expected_size:
        return False, f"SIZE_MISMATCH(got={got_size},exp={expected_size})"
    got_sha = sha256_file(path)
    if got_sha != expected_sha256:
        return False, f"SHA256_MISMATCH(got={got_sha},exp={expected_sha256})"
    return True, "OK"


def verify_model_dir(dest: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for fname, (exp_size, exp_sha) in MODEL_FILES.items():
        ok, status = verify_file(dest / fname, exp_size, exp_sha)
        results[fname] = status
        if not ok:
            raise ValueError(f"Model file verification failed for {fname}: {status}")
    return results


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
    ):
        env.pop(k, None)
    return env


def _probe_public_access(url: str) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "eduardo-27b-eval/1.0"})
    try:
        with opener.open(req, timeout=30) as resp:
            if resp.status in (401, 403):
                raise PermissionError(f"Public access denied (HTTP {resp.status}); login/token required")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(f"Public access denied (HTTP {exc.code}); login/token required") from exc
        raise


def _download_with_urllib(url: str, part_path: Path) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"User-Agent": "eduardo-27b-eval/1.0"}
    mode = "wb"
    if part_path.is_file() and part_path.stat().st_size > 0:
        headers["Range"] = f"bytes={part_path.stat().st_size}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=180) as resp, part_path.open(mode) as out:
            if resp.status in (401, 403):
                raise PermissionError(f"Public access denied (HTTP {resp.status}); login/token required")
            while True:
                buf = resp.read(4 * 1024 * 1024)
                if not buf:
                    break
                out.write(buf)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(f"Public access denied (HTTP {exc.code}); login/token required") from exc
        if exc.code == 416:
            # Range not satisfiable (partial file >= remote size); let caller verify/re-download cleanly.
            return
        raise


def download_one_file(url: str, dest_path: Path, expected_size: int, expected_sha256: str) -> str:
    ok, _ = verify_file(dest_path, expected_size, expected_sha256)
    if ok:
        return "SKIP_VERIFIED"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(f".{dest_path.name}.part")

    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "--noproxy", "*",
            "-fSL",
            "--connect-timeout", "20",
            "--retry", "5",
            "--retry-delay", "5",
            "-C", "-",
            "-o", str(part_path),
            url,
        ]
        proc = subprocess.run(cmd, env=_clean_env(), capture_output=True, text=True)
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip()[-300:]
            if "401" in err_tail or "403" in err_tail:
                part_path.unlink(missing_ok=True)
                raise PermissionError(f"curl rejected public download ({err_tail}); login/token required")
            # Retry without -C - if server rejected byte range (curl exit 33).
            if proc.returncode == 33:
                part_path.unlink(missing_ok=True)
                cmd_no_resume = [c for c in cmd if c not in ("-C", "-")]
                proc2 = subprocess.run(cmd_no_resume, env=_clean_env(), capture_output=True, text=True)
                if proc2.returncode != 0:
                    raise RuntimeError(f"curl failed for {dest_path.name} rc={proc2.returncode}: {(proc2.stderr or '').strip()[-300:]}")
            else:
                raise RuntimeError(f"curl failed for {dest_path.name} rc={proc.returncode}: {err_tail}")
    else:
        _download_with_urllib(url, part_path)

    ok_part, status_part = verify_file(part_path, expected_size, expected_sha256)
    if not ok_part:
        part_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file failed integrity check for {dest_path.name}: {status_part}")

    os.replace(part_path, dest_path)
    return "DOWNLOADED_VERIFIED"


def download_and_verify_model(
    dest: Path,
    *,
    mirror_base: str = DEFAULT_MIRROR_BASE,
    verify_only: bool = False,
    min_free_gib: float = 55.0,
) -> dict[str, str]:
    if verify_only:
        return verify_model_dir(dest)

    dest.mkdir(parents=True, exist_ok=True)

    # Calculate remaining bytes needed so re-runs don't fail disk checks when already downloaded.
    missing_bytes = 0
    for fname, (exp_size, exp_sha) in MODEL_FILES.items():
        ok, _ = verify_file(dest / fname, exp_size, exp_sha)
        if not ok:
            missing_bytes += exp_size

    if missing_bytes > 0:
        free_bytes = shutil.disk_usage(dest).free
        req_bytes = max(missing_bytes + 2 * (1024 ** 3), int(min_free_gib * (1024 ** 3)) if missing_bytes > 40 * (1024 ** 3) else 0)
        if free_bytes < req_bytes:
            raise OSError(
                f"Insufficient space at {dest}: free={free_bytes / (1024**3):.2f}GiB < required={req_bytes / (1024**3):.2f}GiB"
            )

    base = mirror_base.rstrip("/")
    statuses: dict[str, str] = {}
    # Check small config.json first so login/403 problems fail fast before downloading shards.
    ordered_files = ["config.json"] + [f for f in sorted(MODEL_FILES) if f != "config.json"]
    for fname in ordered_files:
        exp_size, exp_sha = MODEL_FILES[fname]
        url = f"{base}/{REPO_ID}/resolve/{REVISION}/{fname}"
        print(f"[download_27b] Checking {fname} ({exp_size} bytes)...", flush=True)
        st = download_one_file(url, dest / fname, exp_size, exp_sha)
        statuses[fname] = st
        print(f"[download_27b] {fname}: {st} sha256={exp_sha}", flush=True)

    verify_model_dir(dest)
    print(f"[download_27b] ALL_VERIFIED files={len(MODEL_FILES)} dest={dest}", flush=True)
    return statuses


def main() -> int:
    ap = argparse.ArgumentParser(description="Download & verify Eduardo-27b (3bbf17e) without HF token.")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--mirror-base", default=DEFAULT_MIRROR_BASE)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--min-free-gib", type=float, default=55.0)
    args = ap.parse_args()

    try:
        download_and_verify_model(
            args.dest,
            mirror_base=args.mirror_base,
            verify_only=args.verify_only,
            min_free_gib=args.min_free_gib,
        )
    except PermissionError as exc:
        print(f"ERROR: login/token required ({exc}); stop and report", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
