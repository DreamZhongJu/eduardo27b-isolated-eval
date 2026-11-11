"""Primitive for mutually exclusive, source-scoped work.

This module is not connected to the running evaluator or supervisor. The
caller must check the existing source state *under* the acquired lock and keep
the claim until generation and its durable completion/held marker finish.
Do not delete lock files while any worker may use them. All workers need to
share a filesystem with functioning flock semantics and the same lock root.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


@dataclass
class Claim:
    """Held advisory lock; use with a context manager or call release()."""

    row: dict
    _fd: int

    def release(self) -> None:
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self) -> "Claim":
        if self._fd < 0:
            raise RuntimeError("released claim")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def _lock_name(stage: str, source_id: str) -> str:
    if stage not in ("EASY", "MEDIUM", "HARD"):
        raise ValueError("invalid stage")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("invalid source_id")
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return f"{stage}-{digest}.lock"


def claim_first(root: Path, stage: str, rows: Iterable[dict],
                eligible: Callable[[dict], bool]) -> Optional[Claim]:
    """Try sources in dataset order, returning the first eligible free claim.

    `eligible` executes under the lock and must verify authoritative durable
    source state, accepting only new or safely recoverable source-only work.
    Existing partial, held, invalid or complete records must not be regenerated.
    Locks are advisory, so every writer for the same output tree must use them.
    """
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid lock directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for row in rows:
        name = _lock_name(stage, row.get("source_id"))
        fd = os.open(root / name, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("invalid lock file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            if not eligible(row):
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                continue
            return Claim(row=row, _fd=fd)
        except BaseException:
            os.close(fd)
            raise
    return None
