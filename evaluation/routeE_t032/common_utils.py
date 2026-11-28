#!/usr/bin/env python3
"""Shared utilities for T-032 Route E preflight (P1-P4).

Includes:
- Exact dependency & file-hash verification (bf16, math-verify==0.8.0, heretic).
- Thinking tag stripping & deterministic math grading fallback.
- Vectorized paired bootstrap confidence intervals (10,000 draws).
- Pro 5000 idle GPU detection respecting GPU 6 sensor rules and total <= 8 GPUs.
- Safe aggregate-only machine summary printer (never emits raw text/answers).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Locked hashes & versions
# ---------------------------------------------------------------------------

EXACT_GRADING_VERSIONS: dict[str, str] = {
    "math-verify": "0.8.0",
    "latex2sympy2-extended": "1.10.2",
}
OPTIONAL_GRADING_VERSIONS: dict[str, str] = {
    "latex2sympy2": "1.5.4",
}

BASE_4B_LOCK: dict[str, tuple[str, str]] = {
    "chat_template.jinja": ("git_sha1", "a585dec894e63da457d9440ec6aa7caa16d20860"),
    "config.json": ("git_sha1", "557d961b205319c6a7da5f757f565b69b3967b7d"),
    "merges.txt": ("git_sha1", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "model.safetensors-00001-of-00002.safetensors": (
        "sha256",
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    ),
    "model.safetensors-00002-of-00002.safetensors": (
        "sha256",
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
    ),
    "model.safetensors.index.json": ("git_sha1", "fddda6039f7c1d17260c9e923b8a72fd025d9a86"),
    "preprocessor_config.json": ("git_sha1", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "tokenizer.json": (
        "sha256",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": ("git_sha1", "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1"),
    "video_preprocessor_config.json": ("git_sha1", "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "vocab.json": ("git_sha1", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}

STEP100_LOCK: dict[str, str] = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "6d320dedc3a50ea3eb4bf7ab503fed018f21784f001c61ff9de9a7ed411e69f0",
    "generation_config.json": "944b9abc4ec388933cc00c757a7b794683929e5a02e5123853892a5750645689",
    "model.safetensors": "35435951b689199681a59cc9667cda29e75025f575cd8758bb8320e2f11d88e8",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}

# T-026 Route A step_100 export (runs/t026_stage3_formal_20261126_140508/export_step100_sha256.txt)
ROUTE_A_STEP100_LOCK: dict[str, str] = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "6d320dedc3a50ea3eb4bf7ab503fed018f21784f001c61ff9de9a7ed411e69f0",
    "generation_config.json": "944b9abc4ec388933cc00c757a7b794683929e5a02e5123853892a5750645689",
    "model.safetensors": "9411088e35efc7106fa0f7ed2d9cc3990bfebb4180faa6f3bab2dce7501a96dd",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}

# Author 494-problem test.parquet (same file as T-018 / T-020 / T-022 / T-033)
TEST_494_PARQUET_SHA256 = "338dc73b005b2c8036cc23959061007d7339c24d1988800b62543e1c8224385c"

STUDENT_8B_LOCK: dict[str, str] = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}

MATH500_REV = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_FILES: dict[str, tuple[str, int, str]] = {
    "test.jsonl": ("git_sha1", 446564, "2376b9a194b46c0790e197c91b7249e5f88ac09b"),
}

GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILES: dict[str, tuple[str, int, str]] = {
    "main/train-00000-of-00001.parquet": (
        "sha256",
        2306545,
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    ),
    "main/test-00000-of-00001.parquet": (
        "sha256",
        419088,
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
}

AUTHOR_DATASET_REV = "5d7529f06359f31b692c92634f5cc11c8b7953af"
AUTHOR_DATASET_FILES: dict[str, str] = {
    "data/train-00000-of-00001.parquet": "5171c9891b502f9792b14ac2b15684ce85b8ceb2593689ed3e88b9f349b25d61",
    "data/test-00000-of-00001.parquet": "109b45cf350cb70414a294ff999d4f545b06e9df91e02ec1a0a3b30783c46ec0",
}

HERETIC_COMMIT = "d89c7a2d5a970f7ec634cd300f2bf893dd8048d4"
HERETIC_TAR_SHA256 = "3d5af87cdb90e5f504ca11eedecb771dcef95b0a51c7adbbb65197d6c683d487"
HERETIC_RAW_TAR_URL = f"https://github.com/p-e-w/heretic/archive/{HERETIC_COMMIT}.tar.gz"
GH_MIRROR_PREFIXES = (
    "https://gh-proxy.org/",
    "https://ghfast.top/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
    "",
)

TEACHER_SYSTEM_PROMPT = (
    "You are tutoring a student through a math problem. Your goal is that they\n"
    "understand deeply enough to solve a similar problem on their own afterwards.\n"
    "The student should always show you their attempt first.\n"
    "Never state the final answer before the student has reached it themselves.\n"
    "Do not name the method or key formula before the student has made an attempt\n"
    '(proposed a method, wrote a step, stated a guess, or described a dead end —\n'
    '"how do I do this?" is not an attempt). If the student cannot produce any\n'
    "attempt after genuinely trying, you may demonstrate the key idea on a\n"
    "different example and have them apply it to this one.\n"
    "The student should do the cognitive work — the decisions, not just the\n"
    "calculations: let them choose the method and the next step. Students remember what they choose,\n"
    "work out, and put into their own words far better than what they are told.\n"
    "Everything you say must be mathematically correct.\n"
    "Even if the student is already able to solve the problem, help them deepen\n"
    "and improve their solution so they gain the strongest possible conceptual\n"
    "and transfer understanding.\n"
    "Use your private thinking (the student never sees it) to reason about\n"
    "where the student's mistake is and which pedagogical move suits the situation.\n"
    "When the student has demonstrated understanding and no question of yours is\n"
    "left unanswered, end your final message with <end_of_conversation> on its own line.\n"
    "Problem: {problem}"
)

STUDENT_SYSTEM_PROMPT = (
    "You are a student working through a math problem with a tutor.\n"
    "Problem: {problem}\n"
    "Think step by step, show your genuine attempts and reasoning, and ask when stuck."
)

STUDENT_OPENING_TEMPLATE = "This is the problem I am trying to understand: {problem}"

STUDENT_FINAL_PROMPT = (
    "The conversation with the teacher has ended.\n"
    "It is time you create a step by step complete solution such that it can be graded without the past conversation.\n"
    "Always format your final answer in perfect LaTeX \\boxed{final_answer} format."
)

STUDENT_FINAL_TRANSFER_PROMPT = (
    "The conversation with the teacher has ended. You are now tested on a NEW problem, "
    "to check whether you can apply what you just learned on your own.\n\n"
    "New problem:\n{transfer_problem}\n\n"
    "It is time you create a step by step complete solution such that it can be graded without the past conversation.\n"
    "Always format your final answer in perfect LaTeX \\boxed{final_answer} format."
)


# ---------------------------------------------------------------------------
# Hashing & atomic file I/O
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha1_file(path: Path) -> str:
    data = Path(path).read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("utf-8"))
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any, mode: int = 0o644) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, data, mode=mode)


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]], mode: int = 0o644) -> None:
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    atomic_write_text(path, lines, mode=mode)


def verify_directory_lock(
    model_dir: Path,
    lock_map: dict[str, tuple[str, str] | str],
) -> tuple[bool, list[str]]:
    model_dir = Path(model_dir)
    errors: list[str] = []
    if not model_dir.is_dir():
        return False, [f"MISSING_DIR:{model_dir}"]
    for fname, spec in sorted(lock_map.items()):
        p = model_dir / fname
        if not p.is_file():
            errors.append(f"MISSING_FILE:{fname}")
            continue
        if isinstance(spec, tuple):
            algo, exp = spec
            got = sha256_file(p) if algo == "sha256" else git_sha1_file(p)
        else:
            exp = spec
            got = sha256_file(p)
        if got != exp:
            errors.append(f"HASH_MISMATCH:{fname}:got={got}:exp={exp}")
    return len(errors) == 0, errors


def check_grading_packages(require_exact: bool = True) -> dict[str, str]:
    """Check installed math-verify & latex2sympy2 versions."""
    found: dict[str, str] = {}
    mismatches: list[str] = []
    for pkg, exp in EXACT_GRADING_VERSIONS.items():
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT_INSTALLED"
        found[pkg] = ver
        if ver != exp:
            mismatches.append(f"{pkg}: got={ver} expected={exp}")
    for pkg, exp in OPTIONAL_GRADING_VERSIONS.items():
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT_INSTALLED"
        found[pkg] = ver
        if ver not in ("NOT_INSTALLED", exp):
            mismatches.append(f"{pkg}: got={ver} expected={exp}")
    if require_exact and mismatches:
        raise RuntimeError(
            "Grading package version mismatch (Worker Guideline #10): "
            + "; ".join(mismatches)
        )
    return found


def verify_model_bfloat16(model_dir: Path) -> str:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    dtype = str(
        cfg.get("torch_dtype")
        or cfg.get("dtype")
        or text_cfg.get("torch_dtype")
        or text_cfg.get("dtype")
        or ""
    )
    if dtype != "bfloat16":
        raise RuntimeError(f"Expected bfloat16 in {cfg_path}, got {dtype!r}")
    return dtype


# ---------------------------------------------------------------------------
# Thinking & teacher control stripping (aligned with EduardoRL)
# ---------------------------------------------------------------------------

_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
_EOC_TOKEN = "<end_of_conversation>"


def split_thinking(text: str, assume_thinking: bool = True) -> tuple[str, str, bool]:
    """Return ``(reasoning, visible_reply, unclosed_thinking)``.

    Mirrors ``eduardo/utils.py:strip_teacher_controls`` + ``run_middleturn_eval.py:split_thinking``:
    - Balanced ``<think>...</think>`` blocks are extracted to reasoning.
    - When ``assume_thinking=True`` (Qwen3.5/3.6 prefill template), a tag-free
      turn means thinking overran the per-turn token cap, so ``visible_reply=""``
      and ``unclosed_thinking=True``.
    - Orphan ``</think>`` splits ``REASONING</think>VISIBLE``.
    - Orphan ``<think>`` drops everything after it (unclosed reasoning).
    """
    if not text:
        return "", "", False
    if assume_thinking and "</think>" not in text and "<think>" not in text:
        return text.strip(), "", True

    reasoning_parts = [m.group(1) for m in _THINK_CAPTURE_RE.finditer(text)]
    visible = _THINK_PAIR_RE.sub("", text)
    unclosed = False
    if "</think>" in visible:
        head, visible = visible.split("</think>", 1)
        reasoning_parts.append(head.split("<think>", 1)[-1])
    if "<think>" in visible:
        visible, tail = visible.split("<think>", 1)
        reasoning_parts.append(tail)
        unclosed = True
    visible = visible.replace(_EOC_TOKEN, "").strip()
    reasoning = "\n".join(p.strip() for p in reasoning_parts if p.strip())
    return reasoning, visible, unclosed


def strip_teacher_controls(text: str, assume_thinking: bool = True) -> str:
    return split_thinking(text, assume_thinking=assume_thinking)[1]


# ---------------------------------------------------------------------------
# Math answer extraction & grading
# ---------------------------------------------------------------------------


def last_boxed_only_string(string: str) -> str | None:
    if not string:
        return None
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx
    depth = 0
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[idx : i + 1]
        i += 1
    return None


def remove_boxed(s: str | None) -> str:
    if not s:
        return ""
    left = r"\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1].strip()
    return ""


_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*ANSWER\s*:\s*(.+?)\s*(?:\n|$)", flags=re.IGNORECASE)
_LAST_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def extract_answer_text(text: str, mode: str = "math500") -> str:
    """Extract final answer string from visible math response."""
    if not text:
        return ""
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        ans = matches[-1].strip()
        boxed = last_boxed_only_string(ans)
        if boxed:
            return remove_boxed(boxed)
        return ans.strip().rstrip(".")
    boxed = last_boxed_only_string(text[-300:]) or last_boxed_only_string(text)
    if boxed:
        return remove_boxed(boxed)
    if mode == "gsm8k":
        if "####" in text:
            return text.split("####")[-1].strip()
        nums = _LAST_NUMBER_RE.findall(text[-150:])
        if nums:
            return nums[-1].replace(",", "").strip()
    return ""


def normalize_math_expr(expr: str) -> str:
    s = (expr or "").strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\,", "").replace("\\ ", "").replace(",", "")
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace("−", "-").strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def _try_parse_rational(expr: str) -> Fraction | float | None:
    s = normalize_math_expr(expr)
    if not s:
        return None
    # Handle \frac{a}{b} or -\frac{a}{b}
    frac_m = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s.replace(" ", ""))
    if frac_m:
        sign = -1 if frac_m.group(1) == "-" else 1
        num = int(frac_m.group(2))
        den = int(frac_m.group(3))
        if den != 0:
            return Fraction(sign * num, den)
    # Handle plain integer / decimal / a/b
    plain = s.replace(" ", "")
    if re.fullmatch(r"[+-]?\d+", plain):
        return Fraction(int(plain), 1)
    if re.fullmatch(r"[+-]?\d+/\d+", plain):
        num_s, den_s = plain.split("/", 1)
        den = int(den_s)
        if den != 0:
            return Fraction(int(num_s), den)
    if re.fullmatch(r"[+-]?\d+\.\d+", plain):
        try:
            return Fraction(plain)
        except ValueError:
            return float(plain)
    return None


class _TimeoutAlarm:
    def __init__(self, seconds: int = 5):
        self.seconds = seconds
        self._prev = None

    def __enter__(self):
        self._prev = signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, *args):
        signal.alarm(0)
        if self._prev is not None:
            signal.signal(signal.SIGALRM, self._prev)

    @staticmethod
    def _handle(signum, frame):
        raise TimeoutError("math_verify timeout")


def math_values_equal(
    pred: str,
    gold: str,
    *,
    require_math_verify: bool = False,
) -> bool:
    p_norm = normalize_math_expr(pred)
    g_norm = normalize_math_expr(gold)
    if not p_norm or not g_norm:
        return False
    if p_norm == g_norm:
        return True

    if require_math_verify:
        check_grading_packages(require_exact=True)

    try:
        from math_verify import parse as mv_parse, verify as mv_verify

        try:
            with _TimeoutAlarm(seconds=5):
                return bool(mv_verify(mv_parse(g_norm), mv_parse(p_norm)))
        except Exception:
            return False
    except ImportError:
        if require_math_verify:
            raise
        vp = _try_parse_rational(p_norm)
        vg = _try_parse_rational(g_norm)
        if vp is not None and vg is not None:
            if isinstance(vp, Fraction) and isinstance(vg, Fraction):
                return vp == vg
            return abs(float(vp) - float(vg)) <= 1e-9
        return p_norm.replace(" ", "") == g_norm.replace(" ", "")


def grade_math_response(
    response_text: str,
    ground_truth: str,
    *,
    mode: str = "math500",
    assume_thinking: bool = True,
    require_math_verify: bool = False,
) -> dict[str, Any]:
    _, visible, unclosed = split_thinking(response_text, assume_thinking=assume_thinking)
    if not visible:
        return {
            "correct": 0,
            "pred": "",
            "incorrect_format": 1,
            "unclosed_thinking": int(unclosed),
        }
    pred = extract_answer_text(visible, mode=mode)
    if not pred:
        return {
            "correct": 0,
            "pred": "",
            "incorrect_format": 1,
            "unclosed_thinking": int(unclosed),
        }
    ok = math_values_equal(pred, ground_truth, require_math_verify=require_math_verify)
    return {
        "correct": int(ok),
        "pred": pred,
        "incorrect_format": 0,
        "unclosed_thinking": int(unclosed),
    }


# ---------------------------------------------------------------------------
# Statistics & bootstrap CIs
# ---------------------------------------------------------------------------


def normalized_gain(pre: float, post: float) -> float:
    if post > pre:
        return (post - pre) / (1.0 - pre) if pre < 1.0 else 0.0
    return post - pre


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    chunk_size = max(1, min(draws, 5_000_000 // max(1, n)))
    boots = np.empty(draws, dtype=np.float64)
    offset = 0
    while offset < draws:
        cur = min(chunk_size, draws - offset)
        idx = rng.integers(0, n, size=(cur, n))
        boots[offset : offset + cur] = arr[idx].mean(axis=1)
        offset += cur
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def paired_bootstrap_diff_ci(
    base_values: Sequence[float],
    target_values: Sequence[float],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    base_arr = np.asarray(base_values, dtype=np.float64)
    target_arr = np.asarray(target_values, dtype=np.float64)
    if len(base_arr) != len(target_arr):
        raise ValueError(f"Length mismatch: {len(base_arr)} vs {len(target_arr)}")
    n = len(base_arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array")
    diff_arr = target_arr - base_arr
    rng = np.random.default_rng(seed)
    chunk_size = max(1, min(draws, 5_000_000 // max(1, n)))
    boot_base = np.empty(draws, dtype=np.float64)
    boot_target = np.empty(draws, dtype=np.float64)
    boot_diff = np.empty(draws, dtype=np.float64)
    offset = 0
    while offset < draws:
        cur = min(chunk_size, draws - offset)
        idx = rng.integers(0, n, size=(cur, n))
        boot_base[offset : offset + cur] = base_arr[idx].mean(axis=1)
        boot_target[offset : offset + cur] = target_arr[idx].mean(axis=1)
        boot_diff[offset : offset + cur] = diff_arr[idx].mean(axis=1)
        offset += cur
    b_lo, b_hi = np.percentile(boot_base, [2.5, 97.5])
    t_lo, t_hi = np.percentile(boot_target, [2.5, 97.5])
    d_lo, d_hi = np.percentile(boot_diff, [2.5, 97.5])
    return {
        "n": int(n),
        "base_mean": float(base_arr.mean()),
        "base_ci_low": float(b_lo),
        "base_ci_high": float(b_hi),
        "target_mean": float(target_arr.mean()),
        "target_ci_low": float(t_lo),
        "target_ci_high": float(t_hi),
        "diff_mean": float(diff_arr.mean()),
        "diff_ci_low": float(d_lo),
        "diff_ci_high": float(d_hi),
        "p_diff_gt_0": float((boot_diff > 0.0).mean()),
    }


def format_ci(mean: float, lo: float, hi: float, pct: bool = False) -> str:
    if pct:
        return f"{mean * 100:.2f}% [{lo * 100:.2f}%, {hi * 100:.2f}%]"
    return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# Pro 5000 GPU detection & downloads
# ---------------------------------------------------------------------------


def _gpu_snapshot() -> dict[int, dict[str, Any]]:
    gpu_out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    app_out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid_to_idx: dict[str, int] = {}
    gpus: dict[int, dict[str, Any]] = {}
    for line in gpu_out.strip().splitlines():
        idx_s, uuid, free_s, total_s, util_s = [x.strip() for x in line.split(",")]
        idx = int(idx_s)
        total_mib = int(total_s)
        free_mib = int(free_s)
        uuid_to_idx[uuid] = idx
        gpus[idx] = {
            "idx": idx,
            "uuid": uuid,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "used_mib": max(0, total_mib - free_mib),
            "util": int(util_s),
            "apps": [],
        }
    if app_out:
        for line in app_out.splitlines():
            uuid, pid, mem = [x.strip() for x in line.split(",")]
            idx = uuid_to_idx.get(uuid)
            if idx is not None:
                gpus[idx]["apps"].append((pid, mem))
    return gpus


def detect_idle_gpus(
    *,
    min_free_mib: int = 70000,
    gpu6_max_used_mib: int = 500,
    snap_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Detect idle GPUs on Pro 5000 following 组织/资源账本.md rules.

    GPU 6 sensor reports 100% util when idle, so it is treated as idle iff
    used_mib < 500 and apps == [].
    """
    s1 = _gpu_snapshot()
    time.sleep(snap_interval_s)
    s2 = _gpu_snapshot()
    idle: list[int] = []
    busy: list[int] = []
    for idx in sorted(s2):
        g1, g2 = s1[idx], s2[idx]
        if g1["uuid"] != g2["uuid"] or g1["apps"] or g2["apps"]:
            busy.append(idx)
            continue
        if g1["free_mib"] < min_free_mib or g2["free_mib"] < min_free_mib:
            busy.append(idx)
            continue
        if idx == 6:
            if g1["used_mib"] < gpu6_max_used_mib and g2["used_mib"] < gpu6_max_used_mib:
                idle.append(idx)
            else:
                busy.append(idx)
        elif g1["util"] < 10 and g2["util"] < 10:
            idle.append(idx)
        else:
            busy.append(idx)
    return {
        "total_gpus": len(s2),
        "idle_gpus": idle,
        "busy_gpus": busy,
    }


def download_verified_file(
    urls: Sequence[str],
    dest: Path,
    *,
    algo: str,
    expected_hash: str,
    timeout_s: int = 180,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        got = sha256_file(dest) if algo == "sha256" else git_sha1_file(dest)
        if got == expected_hash:
            return dest
        dest.unlink()

    last_err: Exception | None = None
    for url in urls:
        tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "routeE-t032/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp, tmp.open("wb") as out:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
            got = sha256_file(tmp) if algo == "sha256" else git_sha1_file(tmp)
            if got != expected_hash:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Hash mismatch for {url}: got={got} exp={expected_hash}")
            os.replace(tmp, dest)
            return dest
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            last_err = exc
    raise RuntimeError(f"Failed to download {dest.name}: {last_err}")


def print_machine_summary(fields: dict[str, Any]) -> None:
    """Emit standardized key: value summary block at the end of stdout."""
    for key, val in fields.items():
        clean_key = re.sub(r"\s+", "_", str(key).strip())
        if isinstance(val, float):
            print(f"{clean_key}: {val:.6f}")
        else:
            print(f"{clean_key}: {val}")
