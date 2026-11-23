#!/usr/bin/env python3
"""Read-only GPU waiter + Eduardo-27B single-GPU (and optional TP=2) vLLM load probe.

Tests whether Eduardo-27B can load in env-candidate-vllm0191 with the author script's
settings (gpu_memory_utilization=0.85, max_model_len=20000, trust_remote_code=True,
seed=42) on 1 idle A100 GPU.

Never kills other users' processes, never prints prompt/output strings, and never
prints raw exception messages (only sanitized failure category, class name, and
traceback file/line/func).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import wait_gpus_author494 as wg  # noqa: E402

_OOM_HINTS = (
    "out of memory",
    "outofmemoryerror",
    "free memory on device",
    "available memory for the cache blocks",
    "kv cache",
    "max_model_len",
    "gpu memory utilization",
    "cuda error: out of memory",
)


def classify_probe_failure(exc_name: str, exc_msg: str) -> str:
    lowered = f"{exc_name} {exc_msg}".lower()
    if any(h in lowered for h in _OOM_HINTS):
        return "OOM_OR_KV_CAPACITY"
    return "ENV_INCOMPATIBLE"


def _run_load_worker(model_path: str, tp: int, visible_gpus: str) -> tuple[bool, str, str, list[str]]:
    """Run vllm.LLM init + 1 synthetic prompt in an isolated child process."""
    code = r"""
import gc, json, sys, traceback
from vllm import LLM, SamplingParams
model_path, tp = sys.argv[1], int(sys.argv[2])
try:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.85,
        max_model_len=20000,
        trust_remote_code=True,
        seed=42,
    )
    outputs = llm.generate(["1 + 1 ="], SamplingParams(temperature=0.0, max_tokens=4))
    n_toks = len(outputs[0].outputs[0].token_ids) if outputs and outputs[0].outputs else 0
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(json.dumps({"ok": True, "tokens": n_toks}))
except BaseException as exc:
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f'File "{f.filename}", line {f.lineno}, in {f.name}' for f in tb[-15:]]
    print(json.dumps({
        "ok": False,
        "exc_class": type(exc).__name__,
        "exc_msg": str(exc),
        "frames": frames,
    }))
    sys.exit(1)
"""
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=visible_gpus,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_NO_USAGE_STATS="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code, model_path, str(tp)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                doc = json.loads(ln)
                if doc.get("ok"):
                    return True, "OK", "", []
                cat = classify_probe_failure(doc.get("exc_class", ""), doc.get("exc_msg", ""))
                return False, cat, doc.get("exc_class", "Exception"), doc.get("frames", [])
            except Exception:
                pass
    return False, "ENV_INCOMPATIBLE", f"SubprocessExit({proc.returncode})", []


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Eduardo-27B vLLM loading on 1 (or 2) idle A100 GPUs.")
    ap.add_argument("--model", default="/workspace/models/Eduardo-27b-3bbf17e")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--status-dir", type=Path, required=True)
    ap.add_argument("--min-free-mib", type=int, default=78000)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--allow-tp2-fallback", action="store_true")
    args = ap.parse_args()

    args.status_dir.mkdir(parents=True, exist_ok=True)
    chosen1 = wg.wait_for_gpus(
        status_path=args.status_dir / "probe_tp1_wait_status.json",
        log_path=args.status_dir / "probe_tp1_wait.log",
        port=18961,
        num_gpus=1,
        min_free_mib=args.min_free_mib,
        poll_seconds=args.poll_seconds,
        port_fn=lambda _p: True,
    )
    print(f"[27b_probe] Probing Eduardo-27B TP=1 on idle GPU {chosen1[0]}...", flush=True)
    ok1, cat1, exc1, frames1 = _run_load_worker(args.model, 1, str(chosen1[0]))
    result: dict[str, Any] = {
        "tp1_gpu": chosen1[0],
        "tp1_load_ok": ok1,
        "tp1_status": cat1,
        "tp1_exc_class": exc1,
        "tp1_frames": frames1,
    }
    if ok1:
        result["recommended_tp"] = 1
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[27b_probe] tp=1 status=OK recommended_tp=1", flush=True)
        return 0

    print(f"[27b_probe] tp=1 status={cat1} exc={exc1}", flush=True)
    for fr in frames1:
        print(f"  {fr}", flush=True)

    if cat1 != "OOM_OR_KV_CAPACITY" or not args.allow_tp2_fallback:
        result["recommended_tp"] = None
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2

    chosen2 = wg.wait_for_gpus(
        status_path=args.status_dir / "probe_tp2_wait_status.json",
        log_path=args.status_dir / "probe_tp2_wait.log",
        port=18961,
        num_gpus=2,
        min_free_mib=args.min_free_mib,
        poll_seconds=args.poll_seconds,
        port_fn=lambda _p: True,
    )
    print(f"[27b_probe] Probing Eduardo-27B TP=2 on idle GPUs {chosen2}...", flush=True)
    ok2, cat2, exc2, frames2 = _run_load_worker(args.model, 2, ",".join(map(str, chosen2)))
    result.update({
        "tp2_gpus": chosen2,
        "tp2_load_ok": ok2,
        "tp2_status": cat2,
        "tp2_exc_class": exc2,
        "tp2_frames": frames2,
        "recommended_tp": 2 if ok2 else None,
    })
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if ok2:
        print(f"[27b_probe] tp=2 status=OK recommended_tp=2", flush=True)
        return 0
    print(f"[27b_probe] tp=2 status={cat2} exc={exc2}", flush=True)
    for fr in frames2:
        print(f"  {fr}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
