"""Memory budget calculation for RTX PRO 5000 72GB (Stage 1 Step 6)."""
from __future__ import annotations

from dataclasses import dataclass
import math

MIB = 1024 ** 2
GIB = 1024 ** 3
MODEL_PARAMETERS = 4_659_865_088
VOCAB = 248_320
HIDDEN = 2560
LAYERS = 32
MAX_SEQUENCE = 10240
TUTOR_FRACTION = 0.58
JUDGE_FRACTION = 0.86
STUDENT_FRACTION = 0.40
RUNTIME_RESERVE_GIB = 8.0
UNMEASURED_ALLOWANCE_GIB = 10.0


@dataclass(frozen=True)
class TutorBudget:
    device_gib: float
    optimizer_offload: bool
    master_gib: float
    model_bf16_gib: float
    adam_gib: float
    logits_2x_gib: float
    saved_hidden_gib: float
    update_gib: float
    rollout_gib: float
    weight_sync_gib: float

    @property
    def peak_gib(self) -> float:
        return max(self.update_gib, self.rollout_gib, self.weight_sync_gib)

    @property
    def margin_gib(self) -> float:
        return self.device_gib - self.peak_gib

    @property
    def limit_95_gib(self) -> float:
        return 0.95 * self.device_gib

    @property
    def meets_8gib_reserve(self) -> bool:
        return self.margin_gib >= RUNTIME_RESERVE_GIB

    @property
    def within_95_pct(self) -> bool:
        return self.peak_gib <= self.limit_95_gib

    def as_dict(self) -> dict[str, float | bool]:
        return {
            'device_gib': round(self.device_gib, 3),
            'optimizer_offload': self.optimizer_offload,
            'master_gib': round(self.master_gib, 3),
            'model_bf16_gib': round(self.model_bf16_gib, 3),
            'adam_gib': round(self.adam_gib, 3),
            'logits_2x_gib': round(self.logits_2x_gib, 3),
            'saved_hidden_gib': round(self.saved_hidden_gib, 3),
            'update_gib': round(self.update_gib, 3),
            'rollout_gib': round(self.rollout_gib, 3),
            'weight_sync_gib': round(self.weight_sync_gib, 3),
            'peak_gib': round(self.peak_gib, 3),
            'margin_gib': round(self.margin_gib, 3),
            'limit_95_gib': round(self.limit_95_gib, 3),
            'meets_8gib_reserve': self.meets_8gib_reserve,
            'within_95_pct': self.within_95_pct,
        }


def estimate_tutor_4gpu(device_gib: float, optimizer_offload: bool) -> TutorBudget:
    if not math.isfinite(device_gib) or device_gib <= 0:
        raise ValueError('显存总量异常')
    master = MODEL_PARAMETERS * 4 / (4 * GIB)
    model_bf16 = MODEL_PARAMETERS * 2 / GIB
    adam = 2 * master
    logits_2x = 2 * (MAX_SEQUENCE * VOCAB * 4 / GIB)
    saved_hidden = MAX_SEQUENCE * HIDDEN * LAYERS * 2 / GIB
    update = master + master + adam + model_bf16 + logits_2x + saved_hidden + UNMEASURED_ALLOWANCE_GIB
    rollout = master + (0.0 if optimizer_offload else adam) + TUTOR_FRACTION * device_gib + UNMEASURED_ALLOWANCE_GIB
    weight_sync = master + (0.0 if optimizer_offload else adam) + 2 * model_bf16 + 6.0 + UNMEASURED_ALLOWANCE_GIB
    return TutorBudget(
        device_gib=device_gib,
        optimizer_offload=optimizer_offload,
        master_gib=master,
        model_bf16_gib=model_bf16,
        adam_gib=adam,
        logits_2x_gib=logits_2x,
        saved_hidden_gib=saved_hidden,
        update_gib=update,
        rollout_gib=rollout,
        weight_sync_gib=weight_sync,
    )


def summarize_pro5000(device_mib: float = 73415.0) -> dict[str, dict[str, float | bool]]:
    device_gib = device_mib * MIB / GIB
    return {
        'offload_false': estimate_tutor_4gpu(device_gib, False).as_dict(),
        'offload_true': estimate_tutor_4gpu(device_gib, True).as_dict(),
    }
