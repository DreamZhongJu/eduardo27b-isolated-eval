#!/usr/bin/env python3
"""Launcher wrapper for unmodified upstream eval/run_env_test_eval.py with Eduardo-27B.

Leaves all upstream EduardoRL files untouched (zero-diff verified), verifies the
Tutor model config dtype is bfloat16, wraps:
1. Tutor (make_teacher / LocalVLLMTeacher) -> ReplayTeacherWrapper
2. Agnes judge (make_judge_client & shared get_async_openai_client) -> ReplayOpenAIClient(AgnesWindowClient(...))
3. Student (get_async_openai_client) -> ReplayOpenAIClient

Provides atomic per-call persistence, --resume deterministic replay, strict
manifest binding, and AIMD congestion window (1..16) for Agnes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
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
from eval_hardening import (  # noqa: E402
    ReplayOpenAIClient,
    ReplayStore,
    ReplayTeacherWrapper,
    build_replay_manifest,
)

window_logger = logging.getLogger("eduardo27b.agnes_window")
replay_logger = logging.getLogger("eduardo27b.replay")


def _locate_author_repo() -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    if os.environ.get("PRO5000_BASE"):
        candidates.append(Path(os.environ["PRO5000_BASE"]) / "source" / "EduardoRL")
    candidates.extend([
        Path.home() / "bzhang" / "a_group_pro5000" / "source" / "EduardoRL",
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
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--replay-stats-out", default=None)
    ap.add_argument("--student-model-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_known_args(argv)


def _extract_flag_value(argv: list[str], flag: str, default: str) -> str:
    for idx, tok in enumerate(argv):
        if tok == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


def _resolve_git_commit(repo: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except OSError:
        pass
    return "UNKNOWN"


def install_hardening_hooks(
    upstream_eval: Any,
    upstream_middleturn: Any,
    *,
    store: ReplayStore,
    agnes_api_base: str,
    agnes_api_key_env: str,
    student_api_base: str,
    student_api_key_env: str,
    outer_num_retries: int,
) -> AgnesWindowClient:
    """Monkeypatch upstream_eval teacher & client factories in memory (never touching files)."""
    shared_window = AsyncCongestionWindow()
    agnes_holder: dict[str, Any] = {}
    student_holder: dict[str, Any] = {}

    orig_make_teacher = upstream_eval.make_teacher
    orig_make_judge_client = upstream_eval.make_judge_client
    orig_get_async_client = upstream_eval.get_async_openai_client

    def _wrap_agnes_client(raw_client: Any) -> ReplayOpenAIClient:
        if "replay_client" not in agnes_holder:
            win_client = AgnesWindowClient(
                raw_client,
                window=shared_window,
                inner_max_retries=INNER_MAX_RETRIES,
            )
            agnes_holder["window_client"] = win_client
            agnes_holder["replay_client"] = ReplayOpenAIClient(
                win_client,
                store=store,
                api_base=agnes_api_base,
                role="judge",
            )
            window_logger.info(
                "[27b_agnes_window] STARTUP %s",
                win_client.startup_summary(outer_num_retries=outer_num_retries),
            )
        return agnes_holder["replay_client"]

    def _wrap_student_client(raw_client: Any) -> ReplayOpenAIClient:
        if "replay_client" not in student_holder:
            student_holder["replay_client"] = ReplayOpenAIClient(
                raw_client,
                store=store,
                api_base=student_api_base,
                role="student",
            )
        return student_holder["replay_client"]

    def wrapped_make_teacher(args: Any) -> Any:
        if getattr(args, "use_gemini", False) or getattr(args, "use_openrouter", False):
            return orig_make_teacher(args)
        return ReplayTeacherWrapper(
            args,
            teacher_factory=orig_make_teacher,
            store=store,
            render_prompts_fn=upstream_middleturn._render_teacher_prompts,
            verify_consistency_fn=upstream_middleturn._verify_prompt_consistency,
            log_first_sample_fn=upstream_middleturn._log_first_thinking_sample,
            thinking_text_fn=upstream_middleturn.thinking_text,
            describe_thinking_fn=upstream_middleturn.describe_teacher_thinking,
        )

    def wrapped_make_judge_client(args: Any) -> tuple[Any, str, dict[str, Any]]:
        client, model, extra_body = orig_make_judge_client(args)
        if getattr(args, "gemini", False):
            return _wrap_agnes_client(client), model, extra_body
        return client, model, extra_body

    def wrapped_get_async_openai_client(api_base: str, api_key_env: str, timeout: float = 300.0) -> Any:
        raw_client = orig_get_async_client(api_base, api_key_env, timeout=timeout)
        if api_base == agnes_api_base and api_key_env == agnes_api_key_env:
            return _wrap_agnes_client(raw_client)
        if api_base == student_api_base and api_key_env == student_api_key_env:
            return _wrap_student_client(raw_client)
        return raw_client

    upstream_eval.make_teacher = wrapped_make_teacher
    upstream_eval.make_judge_client = wrapped_make_judge_client
    upstream_eval.get_async_openai_client = wrapped_get_async_openai_client
    # Keep middleturn module's client factory aligned in case OpenRouter/API teacher is used.
    upstream_middleturn.get_async_openai_client = wrapped_get_async_openai_client

    return agnes_holder.setdefault(
        "placeholder",
        AgnesWindowClient(None, window=shared_window, inner_max_retries=INNER_MAX_RETRIES),
    )


def install_agnes_window_hooks(
    upstream_eval: Any,
    *,
    agnes_api_base: str,
    agnes_api_key_env: str,
    outer_num_retries: int,
) -> AgnesWindowClient:
    """Backwards-compatible window-only hook helper (used when no ReplayStore is requested)."""
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

    import eval.run_env_test_eval as upstream_eval  # noqa: E402
    import eval.run_middleturn_eval as upstream_middleturn  # noqa: E402

    old_argv = sys.argv
    try:
        sys.argv = ["eval.run_env_test_eval", *pass_argv]
        parsed_eval_args = upstream_eval.parse_args()
    finally:
        sys.argv = old_argv

    out_dir = Path(
        parsed_eval_args.output_dir
        or (Path(os.environ["WORK_DIR"]) / "eval_results"
            if os.environ.get("WORK_DIR") else Path("eval_results"))
    )
    replay_dir = Path(wrapper_args.replay_dir) if wrapper_args.replay_dir else (out_dir / "replay_cache")

    eval_args_manifest = {
        k: v for k, v in vars(parsed_eval_args).items()
        if k not in ("output_dir", "conversations")
    }
    wrapper_files = [
        HERE / "agnes_window.py",
        HERE / "eval_hardening.py",
        HERE / "run_eval_27b_author494.py",
    ]
    manifest = build_replay_manifest(
        upstream_repo=repo,
        upstream_commit=_resolve_git_commit(repo),
        wrapper_files=wrapper_files,
        wrapper_base_dir=HERE,
        tutor_model_path=parsed_eval_args.model,
        student_model_path=wrapper_args.student_model_dir,
        test_parquet_path=parsed_eval_args.test_parquet,
        eval_args=eval_args_manifest,
    )

    store = ReplayStore(replay_dir, manifest=manifest, resume=wrapper_args.resume)
    outer_num_retries = int(parsed_eval_args.num_retries)
    placeholder = install_hardening_hooks(
        upstream_eval,
        upstream_middleturn,
        store=store,
        agnes_api_base=parsed_eval_args.gemini_api_base,
        agnes_api_key_env=parsed_eval_args.gemini_api_key_env,
        student_api_base=parsed_eval_args.student_api_base,
        student_api_key_env=parsed_eval_args.student_api_key_env,
        outer_num_retries=outer_num_retries,
    )

    startup_win = f"[27b_agnes_window] STARTUP {placeholder.startup_summary(outer_num_retries=outer_num_retries)}"
    startup_rep = (
        f"[27b_replay] STARTUP resume={wrapper_args.resume} "
        f"loaded_records={store.loaded_records} dropped_corrupt_tail={store.dropped_corrupt_tail}"
    )
    print(startup_win, flush=True)
    print(startup_rep, flush=True)
    replay_logger.info("%s", startup_rep)

    try:
        sys.argv = ["eval.run_env_test_eval", *pass_argv]
        upstream_eval.main()
    finally:
        sys.argv = old_argv
        stats = placeholder.stats(outer_num_retries=outer_num_retries)
        rstats = store.stats()
        store.close()

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
        replay_line = (
            f"[27b_replay] SUMMARY resume={wrapper_args.resume} "
            f"loaded_records={rstats['loaded_records']} "
            f"dropped_corrupt_tail={rstats['dropped_corrupt_tail']} "
            f"replayed_hits={rstats['total_hits']} "
            f"live_calls={rstats['total_live']} "
            f"newly_recorded={rstats['total_recorded']} "
            f"hits_tutor={rstats['hits_by_role']['tutor']} "
            f"hits_student={rstats['hits_by_role']['student']} "
            f"hits_judge={rstats['hits_by_role']['judge']} "
            f"live_tutor={rstats['live_by_role']['tutor']} "
            f"live_student={rstats['live_by_role']['student']} "
            f"live_judge={rstats['live_by_role']['judge']}"
        )
        window_logger.info("%s", summary_line)
        replay_logger.info("%s", replay_line)
        print(summary_line, flush=True)
        print(replay_line, flush=True)

        if wrapper_args.agnes_stats_out:
            out_p = Path(wrapper_args.agnes_stats_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if wrapper_args.replay_stats_out:
            rp = Path(wrapper_args.replay_stats_out)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(rstats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
