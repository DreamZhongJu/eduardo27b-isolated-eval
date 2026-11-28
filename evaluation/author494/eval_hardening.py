#!/usr/bin/env python3
"""Reusable idempotent record-and-replay hardening layer for author494 evaluations.

Provides atomic per-call persistence and deterministic --resume replay across:
1. LocalVLLMTeacher (in-process vLLM tutor calls, lazy-loaded on cache miss)
2. Student OpenAI-compatible client (dialogue turns + final attempts)
3. Agnes judge OpenAI-compatible client (layered outside AgnesWindowClient)

No author files, prompts, sampling settings, or verdict parsing are modified.
Raw problems, dialogues, and responses are written ONLY inside the station-local
replay cache directory and are never printed to stdout or logs.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from agnes_window import is_valid_completion_envelope

logger = logging.getLogger("eduardo27b.replay")

MANIFEST_SCHEMA = "author494-replay-manifest-v1"
RECORD_SCHEMA = "author494-replay-v1"


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_model_dir_fingerprint(model_path: str | Path | None) -> dict[str, str] | str:
    """Compute sorted relative-path SHA-256 map for a local model directory."""
    if not model_path:
        return "UNSPECIFIED"
    p = Path(model_path)
    if not p.is_dir():
        return f"NON_DIR:{model_path}"
    out: dict[str, str] = {}
    for child in sorted(p.iterdir()):
        if child.is_file() and not child.name.startswith("."):
            out[child.name] = sha256_file(child)
    return out


def compute_files_fingerprint(paths: list[Path], base_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(paths):
        if p.is_file():
            out[p.relative_to(base_dir).as_posix()] = sha256_file(p)
    return out


def hash_request(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _record_integrity_digest(rec_without_digest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(rec_without_digest)).hexdigest()


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


class ReplayStore:
    """Crash-safe, append-only per-call record/replay store with strict manifest binding."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        manifest: dict[str, Any],
        resume: bool,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "replay_manifest.json"
        self.records_path = self.cache_dir / "replay_records.jsonl"
        self.lock_path = self.cache_dir / ".replay.lock"

        self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as exc:
            os.close(self._lock_fd)
            raise RuntimeError(f"Another process holds lock on {self.cache_dir}") from exc

        self._write_lock = threading.Lock()
        self._cursor_lock = threading.Lock()
        self._cached_by_hash: dict[str, list[dict[str, Any]]] = {}
        self._read_cursor: dict[str, int] = {}
        self.dropped_corrupt_tail = 0
        self.loaded_records = 0

        self.hits_by_role: dict[str, int] = {"tutor": 0, "student": 0, "judge": 0}
        self.live_by_role: dict[str, int] = {"tutor": 0, "student": 0, "judge": 0}
        self.recorded_by_role: dict[str, int] = {"tutor": 0, "student": 0, "judge": 0}

        try:
            if resume:
                if not self.manifest_path.is_file():
                    raise ValueError(f"Cannot --resume: missing manifest at {self.manifest_path}")
                saved_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if saved_manifest != manifest:
                    diffs = [
                        k for k in sorted(set(saved_manifest) | set(manifest))
                        if saved_manifest.get(k) != manifest.get(k)
                    ]
                    raise ValueError(f"Replay manifest mismatch on fields={diffs}; refusing --resume")
                self._load_and_repair_records()
            else:
                if self.manifest_path.exists() or (self.records_path.exists() and self.records_path.stat().st_size > 0):
                    raise ValueError(
                        f"Replay cache already exists at {self.cache_dir}; pass --resume or use a new output dir"
                    )
                _atomic_write_json(self.manifest_path, manifest)
                if not self.records_path.exists():
                    fd = os.open(self.records_path, os.O_WRONLY | os.O_CREAT, 0o600)
                    os.close(fd)
        except Exception:
            self.close()
            raise

    def _load_and_repair_records(self) -> None:
        if not self.records_path.is_file():
            return
        raw_bytes = self.records_path.read_bytes()
        if not raw_bytes:
            return
        raw_lines = raw_bytes.splitlines(keepends=True)
        valid_end_offset = 0
        seen_seq: dict[str, int] = {}
        loaded: dict[str, list[dict[str, Any]]] = {}

        for idx, line_bytes in enumerate(raw_lines):
            is_last = idx == len(raw_lines) - 1
            bad = False
            parsed: dict[str, Any] | None = None
            if not line_bytes.endswith(b"\n"):
                bad = True
            else:
                try:
                    parsed = json.loads(line_bytes.decode("utf-8"))
                    if not isinstance(parsed, dict) or parsed.get("schema") != RECORD_SCHEMA:
                        bad = True
                    else:
                        digest = parsed.get("record_sha256")
                        body = {k: v for k, v in parsed.items() if k != "record_sha256"}
                        if not isinstance(digest, str) or digest != _record_integrity_digest(body):
                            bad = True
                        rhash = parsed.get("request_hash")
                        seq_idx = parsed.get("seq_idx")
                        if not isinstance(rhash, str) or isinstance(seq_idx, bool) or not isinstance(seq_idx, int):
                            bad = True
                        elif seq_idx != seen_seq.get(rhash, 0):
                            bad = True
                except Exception:
                    bad = True

            if bad:
                if is_last:
                    # Truncate corrupted/incomplete trailing record and re-send it on resume.
                    fd = os.open(self.records_path, os.O_RDWR)
                    try:
                        os.ftruncate(fd, valid_end_offset)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    self.dropped_corrupt_tail += 1
                    break
                raise ValueError(f"Corrupted non-trailing replay record at line {idx + 1}")

            assert parsed is not None
            rhash = parsed["request_hash"]
            seen_seq[rhash] = seen_seq.get(rhash, 0) + 1
            loaded.setdefault(rhash, []).append(parsed)
            valid_end_offset += len(line_bytes)

        self._cached_by_hash = loaded
        self.loaded_records = sum(len(v) for v in loaded.values())

    def has_records_for_kind(self, kind: str) -> bool:
        return any(
            rec.get("kind") == kind
            for recs in self._cached_by_hash.values()
            for rec in recs
        )

    def acquire_slot(self, request_hash: str, *, role: str) -> tuple[bool, dict[str, Any] | None]:
        """Claim the next deterministic occurrence slot for request_hash.

        Returns (True, record) on cache hit, or (False, None) on cache miss.
        """
        with self._cursor_lock:
            occ = self._read_cursor.get(request_hash, 0)
            self._read_cursor[request_hash] = occ + 1
            cached_list = self._cached_by_hash.get(request_hash)
            if cached_list is not None and occ < len(cached_list):
                self.hits_by_role[role] = self.hits_by_role.get(role, 0) + 1
                return True, cached_list[occ]
            self.live_by_role[role] = self.live_by_role.get(role, 0) + 1
            return False, None

    def record_success(
        self,
        *,
        request_hash: str,
        kind: str,
        role: str,
        response_payload: Any,
    ) -> None:
        """Atomically append one valid completed model call to replay_records.jsonl."""
        with self._write_lock:
            seq_idx = len(self._cached_by_hash.get(request_hash, []))
            body = {
                "schema": RECORD_SCHEMA,
                "request_hash": request_hash,
                "seq_idx": seq_idx,
                "kind": kind,
                "role": role,
                "response": response_payload,
            }
            rec = {**body, "record_sha256": _record_integrity_digest(body)}
            line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            fd = os.open(self.records_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                written = 0
                while written < len(line):
                    n = os.write(fd, line[written:])
                    if n <= 0:
                        raise OSError("Short write appending replay record")
                    written += n
                os.fsync(fd)
            finally:
                os.close(fd)
            self._cached_by_hash.setdefault(request_hash, []).append(rec)
            self.recorded_by_role[role] = self.recorded_by_role.get(role, 0) + 1

    def stats(self) -> dict[str, Any]:
        total_hits = sum(self.hits_by_role.values())
        total_live = sum(self.live_by_role.values())
        total_recorded = sum(self.recorded_by_role.values())
        return {
            "loaded_records": self.loaded_records,
            "dropped_corrupt_tail": self.dropped_corrupt_tail,
            "total_hits": total_hits,
            "total_live": total_live,
            "total_recorded": total_recorded,
            "hits_by_role": dict(self.hits_by_role),
            "live_by_role": dict(self.live_by_role),
            "recorded_by_role": dict(self.recorded_by_role),
        }

    def close(self) -> None:
        fd = getattr(self, "_lock_fd", None)
        if fd is not None:
            self._lock_fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


class _ReplayCompletionsProxy:
    def __init__(self, wrapper: "ReplayOpenAIClient") -> None:
        self._wrapper = wrapper

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await self._wrapper._create(*args, **kwargs)


class _ReplayChatProxy:
    def __init__(self, wrapper: "ReplayOpenAIClient") -> None:
        self.completions = _ReplayCompletionsProxy(wrapper)


class ReplayOpenAIClient:
    """AsyncOpenAI-compatible client wrapper adding idempotent record & --resume replay."""

    def __init__(
        self,
        inner_client: Any,
        *,
        store: ReplayStore,
        api_base: str,
        role: str,
    ) -> None:
        self._inner = inner_client
        self._store = store
        self._api_base = api_base
        self._role = role
        self.chat = _ReplayChatProxy(self)

    def _request_hash(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        payload = {
            "kind": "openai_chat",
            "endpoint": self._api_base,
            "args": list(args),
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "max_tokens": kwargs.get("max_tokens"),
            "n": kwargs.get("n", 1),
            "seed": kwargs.get("seed"),
            "extra_body": kwargs.get("extra_body"),
            "other_kwargs": {
                k: v for k, v in kwargs.items()
                if k not in ("model", "messages", "temperature", "top_p", "max_tokens", "n", "seed", "extra_body")
            },
        }
        return hash_request(payload)

    @staticmethod
    def _serialize_response(resp: Any) -> dict[str, Any]:
        choice0 = resp.choices[0]
        return {
            "id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None),
            "content": choice0.message.content,
            "finish_reason": getattr(choice0, "finish_reason", None),
        }

    @staticmethod
    def _deserialize_response(data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            id=data.get("id"),
            model=data.get("model"),
            choices=[
                SimpleNamespace(
                    finish_reason=data.get("finish_reason"),
                    message=SimpleNamespace(content=data.get("content")),
                )
            ],
        )

    async def _create(self, *args: Any, **kwargs: Any) -> Any:
        rhash = self._request_hash(args, kwargs)
        hit, rec = self._store.acquire_slot(rhash, role=self._role)
        if hit and rec is not None:
            return self._deserialize_response(rec["response"])

        resp = await self._inner.chat.completions.create(*args, **kwargs)
        if not is_valid_completion_envelope(resp):
            # Incomplete/ambiguous response must NOT be persisted.
            raise RuntimeError("Incomplete completion envelope")

        self._store.record_success(
            request_hash=rhash,
            kind="openai_chat",
            role=self._role,
            response_payload=self._serialize_response(resp),
        )
        return resp

    async def close(self) -> None:
        close_fn = getattr(self._inner, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                await res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ReplayTeacherWrapper:
    """Wraps LocalVLLMTeacher with per-prompt record/replay and lazy GPU loading on --resume."""

    def __init__(
        self,
        args: Any,
        *,
        teacher_factory: Callable[[Any], Any],
        store: ReplayStore,
        render_prompts_fn: Callable[[Any, list[Any], bool], list[str]],
        verify_consistency_fn: Callable[[str, Any], None],
        log_first_sample_fn: Callable[[list[str]], None],
        thinking_text_fn: Callable[[str], str],
        describe_thinking_fn: Callable[[Any], str],
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._args = args
        self._teacher_factory = teacher_factory
        self._store = store
        self._render_prompts = render_prompts_fn
        self._verify_consistency = verify_consistency_fn
        self._log_first_sample = log_first_sample_fn
        self._thinking_text = thinking_text_fn
        self._describe_thinking = describe_thinking_fn
        self.enable_thinking = bool(getattr(args, "enable_thinking", False))
        self._verified = False
        self._inner: Any = None

        if self._store.has_records_for_kind("tutor_vllm"):
            # Resume mode with cached Tutor records: keep GPU idle unless a cache miss occurs.
            if tokenizer_loader is not None:
                self.tokenizer = tokenizer_loader(args.model)
            else:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            logger.info(
                "ReplayTeacherWrapper: loaded tokenizer on CPU; deferring vLLM engine init until cache miss"
            )
        else:
            self._ensure_inner()

    def _ensure_inner(self) -> Any:
        if self._inner is None:
            self._inner = self._teacher_factory(self._args)
            self.tokenizer = getattr(self._inner, "tokenizer", None)
        return self._inner

    def _prompt_hash(self, prompt: str) -> str:
        args = self._args
        payload = {
            "kind": "tutor_vllm",
            "endpoint": "local_vllm",
            "model": str(args.model),
            "prompt": prompt,
            "temperature": float(args.teacher_temperature),
            "top_p": float(args.teacher_top_p),
            "max_tokens": int(args.max_tokens_per_turn),
            "n": 1,
            "seed": getattr(args, "engine_seed", getattr(args, "seed", None)),
            "enable_thinking": bool(args.enable_thinking),
            "max_model_len": int(args.max_model_len),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
        }
        return hash_request(payload)

    async def generate(self, conversations: list[Any]) -> list[str]:
        prompts = self._render_prompts(self.tokenizer, conversations, self.enable_thinking)
        if not self._verified and prompts:
            self._verify_consistency(prompts[0], conversations[0])
            has_prefill = "<think>" in prompts[0][-200:]
            logger.info(
                "Teacher thinking: enable_thinking=%s, rendered prompt %s a <think> prefill%s",
                self.enable_thinking,
                "HAS" if has_prefill else "has NO",
                "" if has_prefill == self.enable_thinking
                else " — MISMATCH: this template appears to ignore enable_thinking",
            )
            self._verified = True

        results: list[str | None] = [None] * len(prompts)
        missed_indices: list[int] = []
        missed_convs: list[Any] = []
        missed_hashes: list[str] = []

        for idx, (conv, prompt) in enumerate(zip(conversations, prompts)):
            rhash = self._prompt_hash(prompt)
            hit, rec = self._store.acquire_slot(rhash, role="tutor")
            if hit and rec is not None:
                results[idx] = str(rec["response"])
            else:
                missed_indices.append(idx)
                missed_convs.append(conv)
                missed_hashes.append(rhash)

        if missed_convs:
            inner = self._ensure_inner()
            # Keep inner's _verified aligned so it doesn't duplicate verification logs.
            if hasattr(inner, "_verified"):
                inner._verified = True
            gen_texts = await inner.generate(missed_convs)
            if len(gen_texts) != len(missed_convs):
                raise RuntimeError("Teacher generate returned mismatched batch length")
            for idx, rhash, text in zip(missed_indices, missed_hashes, gen_texts):
                if not isinstance(text, str):
                    raise RuntimeError("Teacher generate returned non-string text")
                self._store.record_success(
                    request_hash=rhash,
                    kind="tutor_vllm",
                    role="tutor",
                    response_payload=text,
                )
                results[idx] = text

        final_texts = [t if t is not None else "" for t in results]
        self._log_first_sample(final_texts)
        return final_texts

    def count_tokens(self, conv: Any) -> int:
        if self._inner is not None:
            return self._inner.count_tokens(conv)
        return sum(len(self.tokenizer.encode(m["content"])) for m in conv.messages)

    def count_text_tokens(self, text: str) -> int:
        if self._inner is not None:
            return self._inner.count_text_tokens(text)
        return len(self.tokenizer.encode(text)) if text else 0

    def thinking_tokens(self, index: int, text: str) -> tuple[int, str]:
        if self._inner is not None:
            return self._inner.thinking_tokens(index, text)
        return self.count_text_tokens(self._thinking_text(text)), "tokenizer"

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None


def build_replay_manifest(
    *,
    upstream_repo: Path,
    upstream_commit: str,
    wrapper_files: list[Path],
    wrapper_base_dir: Path,
    tutor_model_path: str,
    student_model_path: str | None,
    test_parquet_path: str,
    eval_args: dict[str, Any],
) -> dict[str, Any]:
    """Construct strict binding manifest preventing mismatched --resume runs."""
    upstream_files = sorted(
        list((upstream_repo / "eval").rglob("*.py"))
        + list((upstream_repo / "eduardo").rglob("*.py"))
        + list((upstream_repo / "eduardo" / "configs" / "prompts").glob("*.txt"))
    )
    tpq = Path(test_parquet_path)
    return {
        "schema": MANIFEST_SCHEMA,
        "upstream_commit": upstream_commit,
        "upstream_files_sha256": compute_files_fingerprint(upstream_files, upstream_repo),
        "wrapper_files_sha256": compute_files_fingerprint(wrapper_files, wrapper_base_dir),
        "tutor_model_sha256": compute_model_dir_fingerprint(tutor_model_path),
        "student_model_sha256": compute_model_dir_fingerprint(student_model_path),
        "test_parquet_sha256": sha256_file(tpq) if tpq.is_file() else f"MISSING:{test_parquet_path}",
        "eval_args": eval_args,
    }
