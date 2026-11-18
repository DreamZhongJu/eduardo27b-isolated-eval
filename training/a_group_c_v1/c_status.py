"""Read the C controller records and emit Markdown tables to stdout."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

if not Path(__file__).with_name("status.py").is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a_group_b_native_v1"))
import status as base


ROLES = {"trainer": "训练器", "student": "学生", "judge": "裁判", "cpu_merge": "CPU 合并"}
UNKNOWN = "未知"


def integer(value, *, minimum=None):
    return type(value) is int and (minimum is None or value >= minimum)


def controller_state(work, pid, proc=Path("/proc")):
    if work.parent.name != "pilots" or not integer(pid, minimum=1):
        return UNKNOWN
    folder = proc / str(pid)
    try:
        args = (folder / "cmdline").read_bytes().split(b"\0")
        script = os.fsencode(Path(__file__).resolve().with_name("c_stage.py"))
        if script not in args:
            return "未发现匹配控制器"
        args = args[args.index(script) + 1:]
        if b"--run" not in args or b"--name" not in args:
            return "未发现匹配控制器"
        pos = args.index(b"--name") + 1
        if pos >= len(args) or args[pos] != os.fsencode(work.name):
            return "未发现匹配控制器"
        text = (folder / "stat").read_text()
        fields = text.rpartition(")")[2].split()
        if not fields:
            return UNKNOWN
        return "已退出" if fields[0] in ("Z", "X", "x") else "运行中"
    except (FileNotFoundError, ProcessLookupError):
        return "未发现匹配控制器"
    except (UnicodeError, OSError):
        return UNKNOWN


def exit_time(ns):
    if not integer(ns, minimum=1):
        return UNKNOWN
    try:
        sec, frac = divmod(ns, 10**9)
        date = datetime.fromtimestamp(sec).astimezone()
        return f"{date:%Y-%m-%d %H:%M:%S}.{frac:09d} {date:%z}"
    except (OverflowError, OSError, ValueError):
        return UNKNOWN


def exit_report(work, state):
    records = {role: base.read_obj(work / f"exit-{role}.json") for role in ROLES}
    # observed_ns is when the entry wrapper observed the child exit. Missing receipts
    # have no rank, and receipts sharing a timestamp share a rank.
    times = sorted({rec["observed_ns"] for rec in records.values()
                    if exit_time(rec.get("observed_ns")) != UNKNOWN})
    sequence = {ts: i for i, ts in enumerate(times, start=1)}
    rows = []
    for role, rec in records.items():
        ns = rec.get("observed_ns")
        order = sequence.get(ns) if integer(ns) else None
        code = rec.get("exit_code")
        requested = rec.get("requested")
        codes = state.get("role_exit_codes") or {}
        processes = state.get("role_processes") or {}
        guard_code = codes.get(role)
        alive = processes.get(role)
        rows.append((order or "—", ROLES[role],
                     "运行" if alive is True else ("退出" if alive is False else UNKNOWN),
                     guard_code if integer(guard_code) else UNKNOWN,
                     code if integer(code) else UNKNOWN, exit_time(ns),
                     "是" if requested is True else ("否" if requested is False else UNKNOWN)))
    rows.sort(key=lambda row: row[0] if isinstance(row[0], int) else len(ROLES) + 1)
    print("\n各角色退出记录（按观察到的子进程退出时间排序，同一时间并列）")
    base.table(("观察顺序", "角色", "守护上次状态", "守护退出码", "子进程退出码", "观察时间", "已请求停止"), rows)


def gpu_report(work, state):
    bindings = state.get("bindings") or {}
    peaks = base.read_obj(work / "gpu_peaks.json")
    assignments = {}
    for role, name in (("tutor", "Tutor"), ("student", "学生"), ("judge", "裁判")):
        devices = bindings.get(role)
        if not isinstance(devices, list):
            continue
        for device in devices:
            if integer(device, minimum=0):
                assignments.setdefault(device, []).append(name)
    for key in peaks:
        if key.isascii() and key.isdecimal():
            assignments.setdefault(int(key), [])
    rows = []
    for device, names in sorted(assignments.items()):
        value = peaks.get(str(device))
        rows.append((device, " / ".join(names) or UNKNOWN, value if integer(value, minimum=0) else UNKNOWN))
    print("\n已交接 GPU 与本任务显存峰值（每 10 秒采样）")
    base.table(("GPU", "角色", "峰值 MiB"), rows or [("—", "尚无记录", UNKNOWN)])


def report(work):
    work = Path(work)
    base.report(work, stage_label="C", include_finish=False)
    state = base.read_obj(work / "orchestration.json")
    print("\nC 控制器（本容器当前进程）")
    base.table(("角色", "进程状态"), [("控制器", controller_state(work, state.get("controller_pid")))])
    gpu_report(work, state)
    exit_report(work, state)
    acc = base.read_obj(work / "c_acceptance.json")
    verified = acc.get("merge_verified")
    stats = [
        ("自动 CPU 合并与权重读取", "通过" if verified is True else ("未通过或未完成" if verified is False else UNKNOWN)),
        ("已核验导出分片数", acc.get("merge_shards") if integer(acc.get("merge_shards"), minimum=0) else UNKNOWN),
        ("已核验张量键数", acc.get("merge_keys") if integer(acc.get("merge_keys"), minimum=0) else UNKNOWN),
        ("C 统计验收退出码", acc.get("metrics_exit") if integer(acc.get("metrics_exit")) else UNKNOWN),
    ]
    phases = {e.get("phase"): e.get("time") for e in state.get("events", [])}
    def duration(start, end):
        if all(isinstance(x, (int, float)) for x in (start, end)) and end >= start:
            return f"{end-start:.2f}"
        return UNKNOWN
    # Includes scheduling wait; the handoff event table separates model startup.
    stats.append(("凑卡至训练启动（含等待，秒）", duration(phases.get("凑卡"), phases.get("训练中"))))
    merge_s = acc.get("merge_seconds")
    stats.append(("CPU 合并与张量抽读（秒）", f"{merge_s:.2f}" if isinstance(merge_s, (int, float)) else UNKNOWN))
    print("\nC 收尾验收")
    base.table(("项目", "记录"), stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    report(parser.parse_args().work)
