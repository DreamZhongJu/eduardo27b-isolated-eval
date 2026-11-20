"""Inspect only task-owned logs. Never return author log text or model content."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat


@dataclass(frozen=True)
class Marker:
    name: str
    pattern: re.Pattern
    stop: bool


# The order makes an exhausted fallback one event even if both rules match.
MARKERS = (
    Marker('重试耗尽', re.compile(rb'gave up after\b.*?\bretr(?:y|ies)\b', re.I), True),
    Marker('请求回退', re.compile(rb'(?:gave up after|non-retriable)\b.*?using fallback', re.I), True),
    Marker('学生交互硬失败', re.compile(rb'EduardoInteraction: student call failed hard', re.I), True),
    Marker('parse', re.compile(rb'Failed to parse judge response, defaulting to REJECT', re.I), False),
    Marker('recovery', re.compile(rb'Judge JSON malformed, recovered decision via regex', re.I), False),
)
HTTP = re.compile(rb'\[C_HTTP \d+\.\d{6}\] "POST /v1/chat/completions HTTP/[\d.]+" \d{3}\b')
WORKER_FILE = re.compile(r'worker-.+\.(?:out|err)(?:\.\d+)?\Z')
COMPRESSED = re.compile(r'worker-.+\.(?:out|err)(?:\.\d+)?\.(?:gz|bz2|xz|zip|zst)\Z')


class MonitoringFailure(RuntimeError):
    """Fixed diagnostic label and optional numeric evidence."""

    def __init__(self, label, counts=None):
        super().__init__(label)
        self.counts = counts


def classify(line):
    # A parse warning can include an excerpt from the judge after the actual
    # marker. Treat the first event on the line as the logger's classification.
    found = [(match.start(), marker) for marker in MARKERS
             if (match := marker.pattern.search(line)) is not None]
    return min(found, key=lambda item: item[0])[1] if found else None


def reject_terminal(counts):
    for marker in MARKERS:
        if marker.stop and counts[marker.name]:
            raise MonitoringFailure(marker.name)


def excess_parse_failures(parse, requests):
    """Evaluate a strict threshold with exact integer arithmetic."""
    return 100 * parse > requests


@dataclass(frozen=True)
class LogEnd:
    path: Path
    size: int
    kind: str


def identity(info):
    return info.st_dev, info.st_ino


class RunMonitor:
    """Uses source worker logs for vote counts, including rotated inodes once.

    The driver snapshots a phase immediately before and after its work. The
    controller maintains a separate cursor for live and final stop checks.
    Trainer log mirrors are useful for detecting a terminal marker, but they
    are never added to source-worker vote counts.
    """

    def __init__(self, work, raytmp):
        self.work = Path(work)
        self.raytmp = Path(raytmp)
        self.cursors = {}

    def _files(self, *, phase):
        yield ('judge' if phase else 'trainer'), self.work / ('judge.log' if phase else 'trainer.log')
        alias = self.raytmp / 'session_latest' / 'logs'
        if not alias.is_dir():
            return
        try:
            folder = alias.resolve(strict=True)
            if folder.parent.parent != self.raytmp.resolve(strict=True):
                raise MonitoringFailure('Ray worker 日志不属于本次运行')
            for path in sorted(folder.iterdir()):
                if COMPRESSED.fullmatch(path.name):
                    raise MonitoringFailure('本次 worker 日志压缩不可检查')
                if WORKER_FILE.fullmatch(path.name):
                    yield 'worker', path
        except OSError:
            raise MonitoringFailure('本次 worker 日志不可读') from None

    def _ends(self, *, phase):
        ends = {}
        for kind, path in self._files(phase=phase):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue  # A concurrent rotation can remove this name.
            except OSError:
                raise MonitoringFailure('本任务日志不可读') from None
            if not stat.S_ISREG(info.st_mode):
                raise MonitoringFailure('本任务日志路径异常')
            key = identity(info)
            existing = ends.get(key)
            if existing is not None:
                if existing.kind != kind:
                    raise MonitoringFailure('本任务日志身份冲突')
                continue
            ends[key] = LogEnd(path, info.st_size, kind)
        return ends

    @staticmethod
    def _verify_span(before, after):
        if any(key not in after or after[key].size < old.size
               or after[key].kind != old.kind for key, old in before.items()):
            raise MonitoringFailure('本任务日志轮转覆盖不完整')

    def _read(self, key, end, start, *, phase, tail=b'', final=False):
        counts, data = Counter(), tail
        remaining = end.size - start
        if remaining:
            try:
                with end.path.open('rb') as stream:
                    info = os.fstat(stream.fileno())
                    if identity(info) != key or info.st_size < end.size:
                        raise MonitoringFailure('读取期间本任务日志轮转')
                    stream.seek(start)
                    while remaining:
                        part = stream.read(min(remaining, 1024 * 1024))
                        if not part:
                            raise MonitoringFailure('本任务日志未读完')
                        remaining -= len(part)
                        lines = (data + part).split(b'\n')
                        data = lines.pop()
                        for line in lines:
                            self._count(line, end.kind, counts, phase=phase)
            except OSError:
                raise MonitoringFailure('本任务日志不可读') from None
        if final and data:
            self._count(data, end.kind, counts, phase=phase)
            data = b''
        return counts, data

    @staticmethod
    def _count(line, kind, counts, *, phase):
        if kind == 'judge':
            counts['requests'] += int(HTTP.search(line) is not None)
            return
        marker = classify(line)
        if marker is not None and (phase or marker.stop):
            counts[marker.name] += 1

    def poll(self, *, final=False):
        """Inspect growth since the preceding poll, with no telemetry export."""
        ends = self._ends(phase=False)
        before = {key: end for key, (end, _) in self.cursors.items()}
        self._verify_span(before, ends)
        counts = Counter()
        for key, end in ends.items():
            prior, tail = self.cursors.get(key, (LogEnd(end.path, 0, end.kind), b''))
            found, tail = self._read(key, end, prior.size, phase=False, tail=tail, final=final)
            counts.update(found)
            reject_terminal(counts)
            self.cursors[key] = (end, tail)
        return counts

    def finish(self):
        return self.poll(final=True)

    def begin(self):
        """Capture each source log's byte position before train/validation calls."""
        ends = self._ends(phase=True)
        if not any(end.kind == 'judge' for end in ends.values()):
            raise MonitoringFailure('裁判 HTTP 日志缺失')
        return ends

    def end(self, snapshot, *, enforce=True):
        """Return source-only numeric vote counts for a completed phase."""
        ends = self._ends(phase=True)
        self._verify_span(snapshot, ends)
        if not any(end.kind == 'worker' and end.size for end in ends.values()):
            raise MonitoringFailure('本次 worker 日志无覆盖')
        counts = Counter()
        for key, end in ends.items():
            start = snapshot[key].size if key in snapshot else 0
            found, _ = self._read(key, end, start, phase=True, final=True)
            counts.update(found)
        reject_terminal(counts)
        summary = {name: counts[name] for name in ('requests', 'parse', 'recovery')}
        if enforce and excess_parse_failures(summary['parse'], summary['requests']):
            raise MonitoringFailure('本步裁判解析失败超过请求的 1%', summary)
        return summary
