"""Read only the four B result directories, the B logs, and filesystem usage."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


KEYS = ("actor/grad_norm", "actor/pg_loss", "actor/lr")
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
HEADER = re.compile(r"(?<!\S)step:(\d+)\s+-\s")
GLOBAL = re.compile(r"(?:^|\s-\s)training/global_step:([0-9]+)(?=\s|$)")
SCALAR = rf"(?:{NUMBER}|[+-]?(?:inf|nan))"
VALUE = re.compile(r"(?:^|\s-\s)(" + "|".join(KEYS) + r"):(?:(?:np|numpy)\.float(?:16|32|64)\(("
                   + SCALAR + r")\)|(" + SCALAR + r"))(?=\s|$)", re.I)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
UNKNOWN = "未知"


def metrics(path):
    seen_steps = set()
    values = defaultdict(set)
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                text = ANSI.sub("", line)
                headers, global_steps = HEADER.findall(text), GLOBAL.findall(text)
                if len(headers) != 1 or global_steps != headers:
                    continue
                step = int(headers[0])
                seen_steps.add(step)
                if step == 1:
                    for key, wrapped, bare in VALUE.findall(text):
                        values[key.lower()].add(float(wrapped or bare))
    except OSError:
        return set(), {}
    return seen_steps, {key: next(iter(items)) for key, items in values.items() if len(items) == 1}


def gate_count(work, checkpoints, steps):
    # The author assigns ts with time.time() when recording each completed reward.
    # Single-node synchronous B saves data.pt after all rewards of the respective step.
    # Cross-check both actual save boundaries, cardinality, and the two-source grouping.
    if steps != {1, 2}:
        return UNKNOWN
    records, seen = [], set()
    try:
        for path in (work / "eduardo_conversations").glob("conversations_*.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    uid, gate = row["uuid"], row["gated"]
                    if (not isinstance(uid, str) or not uid or uid in seen
                            or type(gate) not in (bool, int) or gate not in (0, 1)
                            or row["note"] != "ok" or row["judge_pass"] != 1 - gate):
                        return UNKNOWN
                    seen.add(uid)
                    problem = row["problem"]
                    if not isinstance(problem, str) or not problem.strip():
                        return UNKNOWN
                    group = hashlib.sha256(problem.encode("utf-8")).digest()
                    ts = row["ts"]
                    if type(ts) not in (float, int) or not math.isfinite(ts) or ts <= 0:
                        return UNKNOWN
                    records.append((ts, group, 1 - gate))
    except (UnicodeError, OSError, ValueError, TypeError, KeyError):
        return UNKNOWN
    if len(records) != 16:
        return UNKNOWN
    passed = sum(row[2] for row in records)
    possible = f"未知（{max(0, passed - 8)}–{min(8, passed)} / 8）"
    try:
        first, second = ((ckpt / "data.pt").stat().st_mtime for ckpt in checkpoints)
    except OSError:
        return possible
    before = [row for row in records if row[0] < first]
    after = [row for row in records if first < row[0] < second]
    groups = [Counter(row[1] for row in batch) for batch in (before, after)]
    if any(sorted(count.values()) != [4, 4] for count in groups) or groups[0].keys() & groups[1].keys():
        return possible
    return f"{sum(row[2] for row in before)} / 8"


def disk_lines(workspace, folders):
    env = dict(os.environ, LC_ALL="C")
    df = subprocess.run(["df", "-h", "--output=size,used,avail,pcent,target", "--", str(workspace)],
                        capture_output=True, text=True, env=env)
    print("df -h（总量 已用 可用 占比 挂载点）", df.stdout.splitlines()[-1] if df.returncode == 0 else UNKNOWN)
    labels = ("第1步检查点", "第2步检查点", "第1步导出", "第2步导出")
    # GNU du counts allocated blocks and de-duplicates hard links across these four arguments.
    proc = subprocess.run(["du", "-B1", "-s", "-c", "--", *map(str, folders)],
                          capture_output=True, text=True, env=env)
    rows = {}
    for line in proc.stdout.splitlines():
        count, sep, path = line.partition("\t")
        if sep and count.isdecimal():
            rows[path] = int(count)
    print("| B 磁盘占用 | GiB |")
    print("| --- | ---: |")
    for label, folder in zip(labels, folders):
        value = rows.get(str(folder))
        print(f"| {label} | {value / (2**30):.2f} |" if proc.returncode == 0 and value is not None
              else f"| {label} | 未知 |")
    if proc.returncode == 0 and all(str(folder) in rows for folder in folders) and "total" in rows:
        print(f"| 合计（实际磁盘块去重） | {rows['total'] / (2**30):.2f} |")
    else:
        print("| 合计（实际磁盘块去重） | 未知 |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/A_group_repro_20261117"))
    root = parser.parse_args().root.resolve()
    work = root / "pilots/b_native_b09d2fb"
    ckpts = [work / f"verl_ckpts/b-functional/global_step_{step}" for step in (1, 2)]
    exports = [root / f"b-accept-b09d2fb-v1/checkpoints/step_{step}" for step in (1, 2)]
    steps, numbers = metrics(work / "trainer.log")
    print("| B 第1步 | 数值 |")
    print("| --- | ---: |")
    for key in KEYS:
        value = numbers.get(key)
        print(f"| {key} | {value:.12g} |" if value is not None else f"| {key} | 未知 |")
    print(f"| 第1步门槛放行 | {gate_count(work, ckpts, steps)} |")
    disk_lines(root.parent, ckpts + exports)


if __name__ == "__main__":
    main()
