#!/usr/bin/env python3
"""Launcher wrapper for unmodified upstream eval/run_env_test_eval.py with Eduardo-27B.

Leaves all upstream EduardoRL files untouched (zero-diff verified), verifies the
Tutor model config dtype is bfloat16, wraps the Agnes client returned by
make_judge_client (and shared eval_judge_client) with AgnesWindowClient, and
writes sanitized aggregate window statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agnes_window import (  # noqa: E402
    AgnesWindowClient,
    AsyncCongestionWindow,
    INNER_MAX_RETRIES,
)

window_logger = logging.getLogger("eduardo27b.agnes_window")


def _locate_author_repo() -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    candidates.extend([
        Path("/workspace/A_group_repro_20261117/source/EduardoRL"),
        HERE.parents[2] / "EduardoRL_A组源码_20261117",
    ])
    for cand in candidates:
        if (cand / "eval" / "run_env_test_eval.py").is_file():
            return cand
    raise SystemExit("ERROR: EduardoRL upstream repo not found; set EDUARDO_REPO")


def _verify_tutor_bfloat16(model_path: str) -> str:
    p = Path(model_path)
    cfg_path = p / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"ERROR: Tutor config.json not found at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "")
    if dtype != "bfloat16":
        raise SystemExit(f"ERROR: Expected Tutor config dtype='bfloat16', got {dtype!r}")
    return dtype


def _strip_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--agnes-stats-out", default=None)
    return ap.parse_known_args(argv)


def _extract_flag_value(argv: list[str], flag: str, default: str) -> str:
    for idx, tok in enumerate(argv):
        if tok == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


def install_agnes_window_hooks(
    upstream_eval: Any,
    *,
    agnes_api_base: str,
    agnes_api_key_env: str,
    outer_num_retries: int,
) -> AgnesWindowClient:
    """Monkeypatch upstream_eval judge client factories in memory (never touching files)."""
    shared_window = AsyncCongestionWindow()
    holder: dict[str, AgnesWindowClient] = {}

    orig_make_judge_client = upstream_eval.make_judge_client
    orig_get_async_client = upstream_eval.get_async_openai_client

    def _wrap_client(raw_client: Any) -> AgnesWindowClient:
        if "client" not in holder:
            holder["client"] = AgnesWindowClient(
                raw_client,
                window=shared_window,
                inner_max_retries=INNER_MAX_RETRIES,
            )
            window_logger.info(
                "[27b_agnes_window] STARTUP %s",
                holder["client"].startup_summary(outer_num_retries=outer_num_retries),
            )
        return holder["client"]

    def wrapped_make_judge_client(args: Any) -> tuple[Any, str, dict[str, Any]]:
        client, model, extra_body = orig_make_judge_client(args)
        if getattr(args, "gemini", False):
            return _wrap_client(client), model, extra_body
        return client, model, extra_body

    def wrapped_get_async_openai_client(api_base: str, api_key_env: str, timeout: float = 300.0) -> Any:
        raw_client = orig_get_async_client(api_base, api_key_env, timeout=timeout)
        if api_base == agnes_api_base and api_key_env == agnes_api_key_env:
            return _wrap_client(raw_client)
        return raw_client

    upstream_eval.make_judge_client = wrapped_make_judge_client
    upstream_eval.get_async_openai_client = wrapped_get_async_openai_client
    # Also ensure placeholder exists before score_records runs so startup/finally never crash.
    return holder.setdefault(
        "placeholder",
        AgnesWindowClient(None, window=shared_window, inner_max_retries=INNER_MAX_RETRIES),
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_args, pass_argv = _strip_wrapper_args(raw_argv)

    repo = _locate_author_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    model_val = _extract_flag_value(pass_argv, "--model", "")
    if model_val:
        dtype = _verify_tutor_bfloat16(model_val)
        window_logger.info("[27b_agnes_window] TUTOR_CONFIG dtype=%s", dtype)

    agnes_api_base = _extract_flag_value(
        pass_argv, "--gemini-api-base", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    agnes_api_key_env = _extract_flag_value(pass_argv, "--gemini-api-key-env", "GEMINI_API_KEY")
    outer_num_retries = int(_extract_flag_value(pass_argv, "--num-retries", "3"))

    import eval.run_env_test_eval as upstream_eval  # noqa: E402

    placeholder = install_agnes_window_hooks(
        upstream_eval,
        agnes_api_base=agnes_api_base,
        agnes_api_key_env=agnes_api_key_env,
        outer_num_retries=outer_num_retries,
    )
    print(
        f"[27b_agnes_window] STARTUP {placeholder.startup_summary(outer_num_retries=outer_num_retries)}",
        flush=True,
    )

    old_argv = sys.argv
    try:
        sys.argv = ["eval.run_env_test_eval", *pass_argv]
        upstream_eval.main()
    finally:
        sys.argv = old_argv
        stats = placeholder.stats(outer_num_retries=outer_num_retries)
        summary_line = (
            f"[27b_agnes_window] SUMMARY logical_calls={stats['logical_calls']} "
            f"http_attempts={stats['http_attempts']} success_calls={stats['success_calls']} "
            f"transient_retries={stats['transient_retries']} "
            f"ambiguous_failures={stats['ambiguous_failures']} "
            f"non_retriable_failures={stats['non_retriable_failures']} "
            f"max_window_reached={stats['max_limit_reached']} "
            f"final_window={stats['final_limit']} "
            f"window_drops={stats['window_drops']} "
            f"congestion_events={stats['congestion_events']} "
            f"http_429_events={stats['http_429_events']}"
        )
        window_logger.info("%s", summary_line)
        print(summary_line, flush=True)
        if wrapper_args.agnes_stats_out:
            out_p = Path(wrapper_args.agnes_stats_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
