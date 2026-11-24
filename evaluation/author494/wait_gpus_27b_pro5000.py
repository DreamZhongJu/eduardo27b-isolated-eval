#!/usr/bin/env python3
"""Read-only Pro 5000 GPU idle waiter and Tutor-GPU handoff holder for Eduardo-27B eval.

Polls nvidia-smi without touching any GPU memory or other users' processes.
Requires GPUs each with >= min_free_mib free (default 70000 MiB on 73415 MiB
RTX PRO 5000), 0 compute apps, and util < 10 (except GPU 6 idle sensor quirk
where used_mib < 500 and 0 compute apps are verified) across two snapshots
separated by confirm_seconds. Prefers GPUs with util < 10 first.

Supports flexible capacity acquisition (e.g., --num-gpus 5 --fallback-min-gpus 4):
uses 5 GPUs (TP=2 Tutor + DP=3 Student) when >=5 are idle, or falls back to
4 GPUs (TP=2 Tutor + DP=2 Student) when only 4 are idle.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
import socket
import subprocess
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
            "used_mib": max(0, total_mib - free_mib),
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


def is_gpu_eligible(
    gpu: dict[str, Any],
    min_free_mib: int = 70000,
    *,
    allow_util_quirk_gpus: set[int] | None = None,
) -> bool:
    """Return True if a Pro 5000 GPU is idle with enough free memory."""
    quirk_gpus = {6} if allow_util_quirk_gpus is None else allow_util_quirk_gpus
    if gpu["total_mib"] < min_free_mib or gpu["free_mib"] < min_free_mib:
        return False
    if len(gpu["apps"]) != 0:
        return False
    if gpu["index"] in quirk_gpus:
        used_mib = gpu.get("used_mib", max(0, gpu["total_mib"] - gpu["free_mib"]))
        return used_mib < 500
    return gpu["util_pct"] < 10


def _confirmed_eligible_ordered(
    snap1: list[dict[str, Any]],
    snap2: list[dict[str, Any]],
    *,
    exclude: set[int] | None = None,
    min_free_mib: int = 70000,
    allow_util_quirk_gpus: set[int] | None = None,
) -> list[int]:
    excl = exclude or set()
    by_idx1 = {g["index"]: g for g in snap1}
    by_idx2 = {g["index"]: g for g in snap2}
    preferred: list[int] = []
    quirk_fallback: list[int] = []
    for idx in sorted(by_idx2):
        if idx in excl:
            continue
        g1 = by_idx1.get(idx)
        g2 = by_idx2.get(idx)
        if g1 is None or g1["uuid"] != g2["uuid"]:
            continue
        if is_gpu_eligible(g1, min_free_mib, allow_util_quirk_gpus=allow_util_quirk_gpus) and is_gpu_eligible(
            g2, min_free_mib, allow_util_quirk_gpus=allow_util_quirk_gpus
        ):
            if g2["util_pct"] < 10:
                preferred.append(idx)
            else:
                quirk_fallback.append(idx)
    return preferred + quirk_fallback


def select_confirmed_gpus(
    snap1: list[dict[str, Any]],
    snap2: list[dict[str, Any]],
    *,
    num_gpus: int = 5,
    fallback_min_gpus: int = 0,
    min_free_mib: int = 70000,
    allow_util_quirk_gpus: set[int] | None = None,
) -> list[int] | None:
    """Return up to num_gpus (at least fallback_min_gpus if >0) confirmed idle GPUs, or None."""
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if fallback_min_gpus < 0 or fallback_min_gpus > num_gpus:
        raise ValueError("fallback_min_gpus must be in [0, num_gpus]")
    eligible = _confirmed_eligible_ordered(
        snap1,
        snap2,
        min_free_mib=min_free_mib,
        allow_util_quirk_gpus=allow_util_quirk_gpus,
    )
    if len(eligible) >= num_gpus:
        return eligible[:num_gpus]
    if fallback_min_gpus > 0 and len(eligible) >= fallback_min_gpus:
        return eligible
    return None


def find_replacement_gpus(
    snap1: list[dict[str, Any]],
    snap2: list[dict[str, Any]],
    *,
    exclude: set[int],
    count: int = 1,
    min_free_mib: int = 70000,
    allow_util_quirk_gpus: set[int] | None = None,
) -> list[int] | None:
    """Return count idle GPU indices not in exclude confirmed across both snapshots, or None."""
    if count <= 0:
        raise ValueError("count must be positive")
    cands = _confirmed_eligible_ordered(
        snap1,
        snap2,
        exclude=exclude,
        min_free_mib=min_free_mib,
        allow_util_quirk_gpus=allow_util_quirk_gpus,
    )
    if len(cands) < count:
        return None
    return cands[:count]


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


def wait_for_gpus(
    *,
    status_path: Path,
    log_path: Path,
    port: int,
    num_gpus: int = 5,
    fallback_min_gpus: int = 0,
    min_free_mib: int = 70000,
    poll_seconds: int = 60,
    confirm_seconds: int = 5,
    snapshot_fn: Callable[[], list[dict[str, Any]]] = snapshot_nvidia_smi,
    port_fn: Callable[[int], bool] = port_is_free,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_polls: int = 0,
) -> list[int]:
    """Poll until target (or fallback_min_gpus) idle GPUs are confirmed and port is free."""
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if fallback_min_gpus < 0 or fallback_min_gpus > num_gpus:
        raise ValueError("fallback_min_gpus must be in [0, num_gpus]")
    required_min = fallback_min_gpus if fallback_min_gpus > 0 else num_gpus
    polls = 0
    while True:
        polls += 1
        snap1 = snapshot_fn()
        elig1 = [g["index"] for g in snap1 if is_gpu_eligible(g, min_free_mib)]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        if len(elig1) >= required_min:
            sleep_fn(confirm_seconds)
            snap2 = snapshot_fn()
            chosen = select_confirmed_gpus(
                snap1,
                snap2,
                num_gpus=num_gpus,
                fallback_min_gpus=fallback_min_gpus,
                min_free_mib=min_free_mib,
            )
            if chosen is not None and port_fn(port):
                _write_json(
                    status_path,
                    {
                        "state": "ACQUIRED",
                        "polls": polls,
                        "num_gpus_target": num_gpus,
                        "fallback_min_gpus": fallback_min_gpus,
                        "num_gpus": len(chosen),
                        "selected_gpus": chosen,
                        "port": port,
                        "updated_at": ts,
                    },
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{ts}] ACQUIRED poll={polls} target={num_gpus} acquired={len(chosen)} "
                        f"selected={chosen} port={port}\n"
                    )
                return chosen

        free_summary = {g["index"]: g["free_mib"] for g in snap1}
        _write_json(
            status_path,
            {
                "state": "WAITING",
                "polls": polls,
                "num_gpus_target": num_gpus,
                "fallback_min_gpus": fallback_min_gpus,
                "eligible_gpus": elig1,
                "eligible_count": len(elig1),
                "free_mib_by_gpu": free_summary,
                "updated_at": ts,
            },
        )
        if polls == 1 or polls % 10 == 0 or len(elig1) >= required_min:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"[{ts}] WAITING poll={polls} eligible={elig1} ({len(elig1)}/{required_min}..{num_gpus}) "
                    f"free_mib={free_summary}\n"
                )
        if max_polls and polls >= max_polls:
            raise TimeoutError(f"Timed out after {polls} polls waiting for {required_min}..{num_gpus} GPUs")
        sleep_fn(poll_seconds)


def _set_pdeathsig() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def hold_tutor_gpu(min_free_mib: int = 70000) -> int:
    """Hold the visible CUDA device 0 during student startup to prevent GPU race."""
    _set_pdeathsig()
    import torch

    torch.cuda.init()
    free, _total = torch.cuda.mem_get_info(0)
    mib = 1024 * 1024
    if free < min_free_mib * mib:
        print(f"HOLD_FAILED free_mib={free // mib}", flush=True)
        return 2
    amount = min(free - 2048 * mib, 68 * 1024 * mib)
    _buf = torch.empty(amount, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    print(f"RESERVED mib={int(torch.cuda.memory_allocated(0)) // mib}", flush=True)

    stop = False

    def _on_sig(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)
    while not stop:
        time.sleep(0.5)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for idle Pro 5000 GPUs (read-only) or hold Tutor GPU.")
    ap.add_argument("--hold", action="store_true", help="Hold CUDA_VISIBLE_DEVICES=0 until SIGTERM")
    ap.add_argument("--find-replacement", action="store_true", help="Find idle GPU(s) not in --exclude")
    ap.add_argument("--count", type=int, default=1, help="Number of replacement GPUs to find")
    ap.add_argument("--exclude", default="", help="Comma-separated GPU indices to exclude")
    ap.add_argument("--status-file", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--port", type=int, default=18971)
    ap.add_argument("--num-gpus", type=int, default=5)
    ap.add_argument("--fallback-min-gpus", type=int, default=0)
    ap.add_argument("--min-free-mib", type=int, default=70000)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--confirm-seconds", type=int, default=5)
    args = ap.parse_args()

    if args.hold:
        return hold_tutor_gpu(min_free_mib=args.min_free_mib)

    if args.find_replacement:
        exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
        snap1 = snapshot_nvidia_smi()
        time.sleep(args.confirm_seconds)
        snap2 = snapshot_nvidia_smi()
        reps = find_replacement_gpus(
            snap1,
            snap2,
            exclude=exclude,
            count=args.count,
            min_free_mib=args.min_free_mib,
        )
        if reps is None:
            return 1
        print(",".join(map(str, reps)))
        return 0

    if args.status_file is None or args.log_file is None:
        ap.error("--status-file and --log-file are required when waiting")

    chosen = wait_for_gpus(
        status_path=args.status_file,
        log_path=args.log_file,
        port=args.port,
        num_gpus=args.num_gpus,
        fallback_min_gpus=args.fallback_min_gpus,
        min_free_mib=args.min_free_mib,
        poll_seconds=args.poll_seconds,
        confirm_seconds=args.confirm_seconds,
    )
    print(",".join(map(str, chosen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
