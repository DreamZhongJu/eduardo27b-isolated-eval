"""Pure planning for source-level scaling; does not inspect or start GPUs.

This module is intentionally not connected to a live supervisor. The caller
must acquire GPUs safely and atomically claim sources before executing a plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

STAGES = ("easy", "medium", "hard")
MAX_EXPERIMENT_GPUS = 6


@dataclass(frozen=True)
class Teacher:
    gpu: int
    stage: str | None
    busy: bool


def plan(teachers, student_gpus, free_gpus, pending, max_teachers=5):
    """Return desired teachers while preserving all teachers with in-flight work.

    `pending` counts only *unclaimed* eligible sources. `free_gpus` must have
    been checked and reserved by the caller; it is not an occupancy detector.
    All GPUs already used by this experiment must appear in either `teachers`
    or `student_gpus`; pass every candidate that is not already used in
    `free_gpus`. The six-GPU cap applies to all existing and newly selected
    teacher/student GPUs combined. This function cannot find other processes,
    reserve devices, make work claims, or enforce the cap for nonparticipants.
    A plan is advisory; execution must enforce the cap again under a run lock.
    """
    teachers = tuple(teachers)
    students = tuple(student_gpus)
    free = tuple(free_gpus)
    if (not students or type(max_teachers) is not int or not 1 <= max_teachers < MAX_EXPERIMENT_GPUS
            or len(teachers) > max_teachers or set(pending) != set(STAGES)
            or any(type(pending[s]) is not int or pending[s] < 0 for s in STAGES)):
        raise ValueError("invalid capacity or pending sources")
    if len(teachers) + len(students) > MAX_EXPERIMENT_GPUS:
        raise ValueError("experiment exceeds six-GPU limit")
    if (any(not isinstance(t, Teacher) or type(t.busy) is not bool
            or (t.stage not in STAGES and t.stage is not None)
            or (t.busy and t.stage is None) for t in teachers)):
        raise ValueError("invalid teacher")
    ids = [t.gpu for t in teachers] + list(students) + list(free)
    if any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("GPUs must be distinct nonnegative integers")

    # Keep in-flight work pinned; idle teacher services may be reassigned.
    desired = {t.gpu: t.stage for t in teachers if t.busy}
    assigned = {s: sum(t.busy and t.stage == s for t in teachers) for s in STAGES}
    remaining = dict(pending)
    idle = sorted(t.gpu for t in teachers if not t.busy)
    room = min(max_teachers - len(teachers), MAX_EXPERIMENT_GPUS - len(teachers) - len(students))
    candidates = idle + sorted(free)[:room]
    for gpu in candidates:
        eligible = [s for s in STAGES if remaining[s] > 0]
        if not eligible:
            if gpu in idle:
                desired[gpu] = None
            continue
        # Balance by unclaimed backlog per existing/new worker. Reserve one
        # source in the projection; real claiming must use a separate atomic lock.
        stage = max(eligible, key=lambda s: (Fraction(remaining[s], assigned[s] + 1),
                                              -STAGES.index(s)))
        desired[gpu] = stage
        assigned[stage] += 1
        remaining[stage] -= 1
    return desired
