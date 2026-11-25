#!/usr/bin/env python3
"""Read-only Pro 5000 GPU idle waiter for T-023 Stage 2 (3-4 GPUs: Judge TP=2 + Student DP=1..2).

Follows T-020 / T-022 & 资源账本.md rules:
- Never touches GPUs used by T-022 or other processes.
- Requires free_mib >= min_free_mib (default 70000 on 73415 MiB RTX PRO 5000),
  0 active compute apps, and util_pct < 10 — EXCEPT GPU 6 (idle sensor quirk
  showing util=100% when idle), where used_mib < 500 and 0 compute apps qualify.
- Prefers non-quirk GPUs (util < 10) first.
- Confirms eligibility across two snapshots separated by confirm_seconds (default 5s).
- Requests --num-gpus 4 --fallback-min-gpus 3 by default:
    4 idle GPUs -> judge_gpus (2, TP=2) + student_gpus (2, DP=2)
    3 idle GPUs -> judge_gpus (2, TP=2) + student_gpus (1, DP=1)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def parse_gpu_csv(gpu_csv: str, apps_csv: str) -> List[Dict[str, Any]]:
    gpus: Dict[int, Dict[str, Any]] = {}
    uuid_to_idx: Dict[str, int] = {}
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
    gpu: Dict[str, Any],
    min_free_mib: int = 70000,
    *,
    allow_util_quirk_gpus: Optional[Set[int]] = None,
) -> bool:
    quirk_gpus = {6} if allow_util_quirk_gpus is None else allow_util_quirk_gpus
    if gpu["total_mib"] < min_free_mib or gpu["free_mib"] < min_free_mib:
        return False
    if len(gpu["apps"]) != 0:
        return False
    if gpu["index"] in quirk_gpus:
        used_mib = gpu.get("used_mib", max(0, gpu["total_mib"] - gpu["free_mib"]))
        return used_mib < 500
    return gpu["util_pct"] < 10


def select_confirmed_gpus(
    snap1: List[Dict[str, Any]],
    snap2: List[Dict[str, Any]],
    *,
    num_gpus: int = 4,
    fallback_min_gpus: int = 3,
    min_free_mib: int = 70000,
    allow_util_quirk_gpus: Optional[Set[int]] = None,
) -> Optional[List[int]]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if fallback_min_gpus < 0 or fallback_min_gpus > num_gpus:
        raise ValueError("fallback_min_gpus must be in [0, num_gpus]")
    by1 = {g["index"]: g for g in snap1}
    by2 = {g["index"]: g for g in snap2}
    preferred: List[int] = []
    quirk_fallback: List[int] = []
    for idx in sorted(by2):
        g1 = by1.get(idx)
        g2 = by2.get(idx)
        if g1 is None or g1["uuid"] != g2["uuid"]:
            continue
        if is_gpu_eligible(g1, min_free_mib, allow_util_quirk_gpus=allow_util_quirk_gpus) and is_gpu_eligible(
            g2, min_free_mib, allow_util_quirk_gpus=allow_util_quirk_gpus
        ):
            if g2["util_pct"] < 10:
                preferred.append(idx)
            else:
                quirk_fallback.append(idx)
    eligible = preferred + quirk_fallback
    if len(eligible) >= num_gpus:
        return eligible[:num_gpus]
    if fallback_min_gpus > 0 and len(eligible) >= fallback_min_gpus:
        return eligible
    return None


def split_judge_and_student_gpus(gpus: List[int]) -> Tuple[List[int], List[int]]:
    """Assign 2 GPUs to Judge (TP=2) and remaining 1-2 GPUs to Student (DP=1..2)."""
    if len(gpus) < 3:
        raise ValueError(f"Need at least 3 GPUs (Judge TP=2 + Student DP>=1), got {gpus}")
    return gpus[:2], gpus[2:]


def query_nvidia_smi() -> List[Dict[str, Any]]:
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return parse_gpu_csv(gpu_csv, apps_csv)


def wait_for_gpus(
    *,
    num_gpus: int = 4,
    fallback_min_gpus: int = 3,
    min_free_mib: int = 70000,
    confirm_seconds: float = 5.0,
    poll_seconds: float = 60.0,
    query_fn: Callable[[], List[Dict[str, Any]]] = query_nvidia_smi,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        snap1 = query_fn()
        cand1 = select_confirmed_gpus(
            snap1, snap1,
            num_gpus=num_gpus,
            fallback_min_gpus=fallback_min_gpus,
            min_free_mib=min_free_mib,
        )
        if cand1 is not None:
            if confirm_seconds > 0:
                sleep_fn(confirm_seconds)
            snap2 = query_fn()
            selected = select_confirmed_gpus(
                snap1, snap2,
                num_gpus=num_gpus,
                fallback_min_gpus=fallback_min_gpus,
                min_free_mib=min_free_mib,
            )
            if selected is not None:
                judge_gpus, student_gpus = split_judge_and_student_gpus(selected)
                return {
                    "selected_gpus": selected,
                    "judge_gpus": judge_gpus,
                    "student_gpus": student_gpus,
                    "judge_tp": len(judge_gpus),
                    "student_dp": len(student_gpus),
                    "attempts": attempt,
                }
        idle_now = [g["index"] for g in snap1 if is_gpu_eligible(g, min_free_mib)]
        print(
            f"[gpu_wait] attempt={attempt} idle={idle_now} (need {fallback_min_gpus}..{num_gpus} GPUs with free>={min_free_mib}MiB); sleeping {poll_seconds:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        sleep_fn(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for 3-4 idle Pro 5000 GPUs for T-023 Stage 2.")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--fallback-min-gpus", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=70000)
    parser.add_argument("--confirm-seconds", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    res = wait_for_gpus(
        num_gpus=args.num_gpus,
        fallback_min_gpus=args.fallback_min_gpus,
        min_free_mib=args.min_free_mib,
        confirm_seconds=args.confirm_seconds,
        poll_seconds=args.poll_seconds,
    )
    txt = json.dumps(res, ensure_ascii=False, sort_keys=True)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
