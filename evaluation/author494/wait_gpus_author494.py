#!/usr/bin/env python3
"""Read-only 4-GPU idle waiter for A100 author-494 evaluation.

Polls nvidia-smi without touching any GPU memory or other users' processes.
Requires 4 GPUs each with >= min_free_mib free, util == 0, and 0 compute apps
across two snapshots separated by confirm_seconds.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def parse_gpu_csv(gpu_csv: str, apps_csv: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi GPU and compute-app CSV outputs into per-GPU records."""
    gpus: dict[int, dict[str, Any]] = {}
    uuid_to_idx: dict[str, int] = {}
    for raw in gpu_csv.strip().splitlines():
        if not raw.strip():
            continue
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {raw!r}")
        idx = int(parts[0])
        uuid = parts[1]
        free_mib = int(parts[2])
        total_mib = int(parts[3])
        util_pct = int(parts[4])
        gpus[idx] = {
            "index": idx,
            "uuid": uuid,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "util_pct": util_pct,
            "apps": [],
        }
        uuid_to_idx[uuid] = idx

    for raw in apps_csv.strip().splitlines():
        if not raw.strip():
            continue
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid_s, mem_s = parts
        if uuid in uuid_to_idx:
            gpus[uuid_to_idx[uuid]]["apps"].append((pid_s, mem_s))

    return [gpus[i] for i in sorted(gpus)]


def is_gpu_eligible(gpu: dict[str, Any], min_free_mib: int = 78000) -> bool:
    return (
        gpu["total_mib"] >= min_free_mib
        and gpu["free_mib"] >= min_free_mib
        and gpu["util_pct"] == 0
        and len(gpu["apps"]) == 0
    )


def select_confirmed_four(
    snap1: list[dict[str, Any]],
    snap2: list[dict[str, Any]],
    *,
    min_free_mib: int = 78000,
) -> list[int] | None:
    """Return 4 GPU indices eligible with matching UUIDs in both snapshots, or None."""
    by_idx1 = {g["index"]: g for g in snap1}
    by_idx2 = {g["index"]: g for g in snap2}
    eligible: list[int] = []
    for idx in sorted(by_idx2):
        g1 = by_idx1.get(idx)
        g2 = by_idx2[idx]
        if g1 is None or g1["uuid"] != g2["uuid"]:
            continue
        if is_gpu_eligible(g1, min_free_mib) and is_gpu_eligible(g2, min_free_mib):
            eligible.append(idx)
    if len(eligible) < 4:
        return None
    return eligible[:4]


def snapshot_nvidia_smi() -> list[dict[str, Any]]:
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    apps_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    return parse_gpu_csv(gpu_csv, apps_csv)


def port_is_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def wait_for_four_gpus(
    *,
    status_path: Path,
    log_path: Path,
    port: int,
    min_free_mib: int = 78000,
    poll_seconds: int = 60,
    confirm_seconds: int = 5,
    snapshot_fn: Callable[[], list[dict[str, Any]]] = snapshot_nvidia_smi,
    port_fn: Callable[[int], bool] = port_is_free,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_polls: int = 0,
) -> tuple[int, list[int]]:
    """Poll until 4 idle GPUs are confirmed across two snapshots and port is free."""
    polls = 0
    while True:
        polls += 1
        snap1 = snapshot_fn()
        elig1 = [g["index"] for g in snap1 if is_gpu_eligible(g, min_free_mib)]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        if len(elig1) >= 4:
            sleep_fn(confirm_seconds)
            snap2 = snapshot_fn()
            chosen = select_confirmed_four(snap1, snap2, min_free_mib=min_free_mib)
            if chosen is not None and port_fn(port):
                tutor_gpu = chosen[0]
                student_gpus = chosen[1:4]
                _write_json(
                    status_path,
                    {
                        "state": "ACQUIRED",
                        "polls": polls,
                        "tutor_gpu": tutor_gpu,
                        "student_gpus": student_gpus,
                        "selected_gpus": chosen,
                        "port": port,
                        "updated_at": ts,
                    },
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{ts}] ACQUIRED poll={polls} tutor={tutor_gpu} student={student_gpus} port={port}\n"
                    )
                return tutor_gpu, student_gpus

        free_summary = {g["index"]: g["free_mib"] for g in snap1}
        _write_json(
            status_path,
            {
                "state": "WAITING",
                "polls": polls,
                "eligible_gpus": elig1,
                "eligible_count": len(elig1),
                "free_mib_by_gpu": free_summary,
                "updated_at": ts,
            },
        )
        if polls == 1 or polls % 10 == 0 or len(elig1) >= 4:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"[{ts}] WAITING poll={polls} eligible={elig1} ({len(elig1)}/4) free_mib={free_summary}\n"
                )
        if max_polls and polls >= max_polls:
            raise TimeoutError(f"Timed out after {polls} polls waiting for 4 GPUs")
        sleep_fn(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for 4 idle A100 GPUs (read-only).")
    ap.add_argument("--status-file", required=True, type=Path)
    ap.add_argument("--log-file", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18961)
    ap.add_argument("--min-free-mib", type=int, default=78000)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--confirm-seconds", type=int, default=5)
    args = ap.parse_args()

    tutor_gpu, student_gpus = wait_for_four_gpus(
        status_path=args.status_file,
        log_path=args.log_file,
        port=args.port,
        min_free_mib=args.min_free_mib,
        poll_seconds=args.poll_seconds,
        confirm_seconds=args.confirm_seconds,
    )
    print(f"{tutor_gpu} {','.join(map(str, student_gpus))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
