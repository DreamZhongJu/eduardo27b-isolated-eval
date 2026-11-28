#!/usr/bin/env python3
"""P4 E0: 6 pluggable/toggleable anti-leakage modification methods across 2 bases.

Methods implemented (all reversible/toggleable at inference time):
1. `ara_r50` & `ara_r16`: Arbitrary-Rank Ablation using official `p-e-w/heretic`
   (`ara_loss`, `mean_distances_to_knn`, `Parameters`) + L-BFGS + row-norm preservation
   + low-rank SVD projection into PEFT LoRA.
2. `dir_ablation`: Mean-difference residual orthogonalized ablation (Arditi / MPOA),
   rank 1..k selected on contrastive heldout split, exported as PEFT LoRA.
3. `dir_addition`: PIVOT-style residual steering vector addition via toggleable forward hooks.
4. `dpo_lora_r32`: DPO-trained PEFT LoRA (r=32) on the exact same contrastive fit split.
5. `random_ablation`: Rank-matched (r=50) random orthonormal subspace ablation baseline (PEFT LoRA).
6. `prompt_only`: Weight-free single-sentence anti-leakage system prompt instruction.

Hyperparameters for every method are selected strictly on the 20% contrastive heldout
split (never on val.parquet or test.parquet).
Includes mandatory 20-prompt greedy token-for-token exactness verification when unplugged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor

from common_utils import atomic_write_json, print_machine_summary

ANTI_LEAKAGE_PROMPT_SENTENCE = (
    "CRITICAL REMINDER: Never state the final answer or solve the key mathematical steps "
    "for the student; only guide them with questions and brief hints."
)

ALL_E0_METHODS: tuple[str, ...] = (
    "ara_r50",
    "ara_r16",
    "dir_ablation",
    "dir_addition",
    "dpo_lora_r32",
    "random_ablation",
    "prompt_only",
)


# ---------------------------------------------------------------------------
# Official p-e-w/heretic ARA loader (compatible with pro5000-fast / cu04)
# ---------------------------------------------------------------------------


def load_heretic_ara_primitives(heretic_root: Path) -> dict[str, Any]:
    """Import `ara_loss`, `mean_distances_to_knn`, and `Parameters` from p-e-w/heretic.

    If optional CLI/quantization packages (`optuna`, `bitsandbytes`, `rich`) are
    absent in the runtime environment, lightweight stubs are installed only for
    module import; the math/L-BFGS objective itself runs purely in PyTorch.
    """
    ara_path = Path(heretic_root) / "src" / "heretic" / "modifiers" / "ara.py"
    if not ara_path.is_file():
        raise FileNotFoundError(f"heretic ara.py not found at {ara_path}")

    saved_modules: dict[str, Any] = {}
    stubs_needed: dict[str, Any] = {}

    if "bitsandbytes.functional" not in sys.modules:
        try:
            import bitsandbytes.functional  # noqa: F401
        except ImportError:
            bnb_mod = types.ModuleType("bitsandbytes")
            bnb_f = types.ModuleType("bitsandbytes.functional")
            bnb_f.dequantize_4bit = lambda *a, **kw: None  # type: ignore[attr-defined]
            bnb_mod.functional = bnb_f  # type: ignore[attr-defined]
            stubs_needed["bitsandbytes"] = bnb_mod
            stubs_needed["bitsandbytes.functional"] = bnb_f

    if "optuna" not in sys.modules:
        try:
            import optuna  # noqa: F401
        except ImportError:
            opt_mod = types.ModuleType("optuna")
            opt_mod.Trial = Any  # type: ignore[attr-defined]
            stubs_needed["optuna"] = opt_mod

    if "heretic" not in sys.modules:
        h_pkg = types.ModuleType("heretic")
        h_cfg = types.ModuleType("heretic.config")
        h_cfg.DatasetSpecification = Any  # type: ignore[attr-defined]
        h_cfg.SingleDatasetSpecification = lambda **kw: kw  # type: ignore[attr-defined]
        h_mod = types.ModuleType("heretic.modifier")
        h_mod.Context = Any  # type: ignore[attr-defined]

        class _Serializable:
            pass

        class _Modifier:
            def __class_getitem__(cls, item):
                return cls

        h_mod.Modifier = _Modifier  # type: ignore[attr-defined]
        h_mod.Serializable = _Serializable  # type: ignore[attr-defined]
        h_utl = types.ModuleType("heretic.utils")
        h_utl.format_dataset_specification = lambda x: str(x)  # type: ignore[attr-defined]
        h_utl.print = print  # type: ignore[attr-defined]
        stubs_needed.update(
            {
                "heretic": h_pkg,
                "heretic.config": h_cfg,
                "heretic.modifier": h_mod,
                "heretic.utils": h_utl,
            }
        )

    for k, v in stubs_needed.items():
        if k in sys.modules:
            saved_modules[k] = sys.modules[k]
        sys.modules[k] = v

    try:
        spec = importlib.util.spec_from_file_location("_heretic_ara_official", ara_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load spec for {ara_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {
            "ara_loss": mod.ara_loss,
            "mean_distances_to_knn": mod.mean_distances_to_knn,
            "Parameters": mod.Parameters,
            "ara_path": str(ara_path),
        }
    finally:
        for k in stubs_needed:
            if k in saved_modules:
                sys.modules[k] = saved_modules[k]
            else:
                sys.modules.pop(k, None)


# ---------------------------------------------------------------------------
# Architecture-agnostic layer & module discovery (Qwen3.5 hybrid + standard)
# ---------------------------------------------------------------------------


def unwrap_base_model(model: nn.Module) -> nn.Module:
    if isinstance(model, PeftModel):
        return model.base_model.model
    return model


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    m = unwrap_base_model(model)
    # Multimodal / Qwen3.5 conditional generation
    for path in (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        cur: Any = m
        ok = True
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, nn.ModuleList):
            return cur
    raise AttributeError("Unable to locate transformer layers ModuleList on model")


def get_layer_target_modules(layer: nn.Module) -> dict[str, list[nn.Module]]:
    """Mirror heretic/src/heretic/model.py:get_layer_modules for Qwen3.5 + standard LMs."""
    modules: dict[str, list[nn.Module]] = {}

    def try_add(comp: str, mod: Any) -> None:
        if isinstance(mod, nn.Module):
            modules.setdefault(comp, []).append(mod)

    self_attn = getattr(layer, "self_attn", None)
    if self_attn is not None:
        try_add("attn.o_proj", getattr(self_attn, "o_proj", None))
    linear_attn = getattr(layer, "linear_attn", None)
    if linear_attn is not None:
        try_add("attn.o_proj", getattr(linear_attn, "out_proj", None))
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        try_add("mlp.down_proj", getattr(mlp, "down_proj", None))

    if not modules:
        raise AssertionError("No modifiable modules found in transformer layer")
    return modules


def collect_target_module_names(model: nn.Module) -> list[str]:
    base = unwrap_base_model(model)
    id_to_name = {id(mod): name for name, mod in base.named_modules()}
    names: set[str] = set()
    for layer in get_transformer_layers(base):
        for mods in get_layer_target_modules(layer).values():
            for mod in mods:
                if id(mod) in id_to_name:
                    names.add(id_to_name[id(mod)])
    return sorted(names)


# ---------------------------------------------------------------------------
# Chat encoding & contrastive feature extraction
# ---------------------------------------------------------------------------


def encode_context_and_reply(
    tokenizer: Any,
    context_messages: Sequence[dict[str, str]],
    reply_text: str,
) -> tuple[list[int], int]:
    """Encode ``context_messages + reply_text`` and return ``(full_ids, reply_start_idx)``."""
    msgs = [dict(m) for m in context_messages]
    full_msgs = msgs + [{"role": "assistant", "content": reply_text}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            prefix_str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full_str = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except TypeError:
            prefix_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full_str = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prefix_ids = list(tokenizer.encode(prefix_str, add_special_tokens=False))
        full_ids = list(tokenizer.encode(full_str, add_special_tokens=False))
        return full_ids, min(len(prefix_ids), len(full_ids))

    # Fallback for simple synthetic tokenizers without chat_template
    prefix_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant: "
    full_text = prefix_text + reply_text
    prefix_ids = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    full_ids = list(tokenizer.encode(full_text, add_special_tokens=False))
    return full_ids, min(len(prefix_ids), len(full_ids))


def compute_reply_logprob_sum(
    model: nn.Module,
    input_ids: Tensor,
    reply_start: int,
) -> Tensor:
    """Compute sum of log-probabilities over reply tokens ``[reply_start, seq_len)``."""
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    start_pos = max(0, reply_start - 1)
    return tok_lp[:, start_pos:].sum(dim=-1)


@torch.no_grad()
def evaluate_heldout_contrastive_objective(
    model: nn.Module,
    tokenizer: Any,
    heldout_pairs: Sequence[dict[str, Any]],
    *,
    ref_good_lp: Sequence[float] | None = None,
    max_eval_pairs: int = 128,
) -> dict[str, float]:
    """Evaluate modification strictly on the 20% contrastive heldout split.

    Score = mean((lp_bad_drop) - penalty * |good_nll_drift|):
    - Higher `margin_gain` = model shifts logprob toward `good_reply` over `bad_reply`.
    - Lower `good_nll_drift` = preserves good behavior distribution.
    """
    device = next(model.parameters()).device
    subset = list(heldout_pairs[:max_eval_pairs])
    margins: list[float] = []
    good_lps: list[float] = []
    bad_lps: list[float] = []
    for p in subset:
        g_ids, g_start = encode_context_and_reply(tokenizer, p["context_messages"], p["good_reply"])
        b_ids, b_start = encode_context_and_reply(tokenizer, p["context_messages"], p["bad_reply"])
        g_t = torch.tensor([g_ids], dtype=torch.long, device=device)
        b_t = torch.tensor([b_ids], dtype=torch.long, device=device)
        g_lp = float(compute_reply_logprob_sum(model, g_t, g_start).item()) / max(1, len(g_ids) - g_start)
        b_lp = float(compute_reply_logprob_sum(model, b_t, b_start).item()) / max(1, len(b_ids) - b_start)
        good_lps.append(g_lp)
        bad_lps.append(b_lp)
        margins.append(g_lp - b_lp)

    good_drift = 0.0
    if ref_good_lp is not None and len(ref_good_lp) == len(good_lps):
        good_drift = sum(abs(a - b) for a, b in zip(good_lps, ref_good_lp)) / max(1, len(good_lps))

    mean_margin = sum(margins) / max(1, len(margins))
    return {
        "mean_margin": float(mean_margin),
        "mean_good_lp": float(sum(good_lps) / max(1, len(good_lps))),
        "mean_bad_lp": float(sum(bad_lps) / max(1, len(bad_lps))),
        "good_nll_drift": float(good_drift),
        "objective": float(mean_margin - 0.5 * good_drift),
        "good_lps": good_lps,
    }


@torch.no_grad()
def extract_contrastive_activations(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[dict[str, Any]],
    *,
    max_pairs: int = 400,
) -> dict[str, Any]:
    """Collect reply-token-pooled layer residuals & target module I/O in float32 on CPU."""
    base = unwrap_base_model(model)
    layers = get_transformer_layers(base)
    n_layers = len(layers)
    device = next(base.parameters()).device

    def collect_side(reply_key: str) -> tuple[Tensor, list[dict[str, dict[int, tuple[Tensor, Tensor]]]]]:
        all_res: list[Tensor] = []
        io_accum: list[dict[str, dict[int, tuple[list[Tensor], list[Tensor]]]]] = [
            {} for _ in range(n_layers)
        ]
        cur_span = [0]

        def make_hook(l_idx: int, comp: str, m_idx: int):
            def hook(mod: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
                s = cur_span[0]
                x = inputs[0][0, s:, :].float().mean(dim=0).detach().cpu()
                y_t = output[0] if isinstance(output, tuple) else output
                y = y_t[0, s:, :].float().mean(dim=0).detach().cpu()
                slot = io_accum[l_idx].setdefault(comp, {}).setdefault(m_idx, ([], []))
                slot[0].append(x)
                slot[1].append(y)

            return hook

        handles = []
        for l_idx, layer in enumerate(layers):
            for comp, mods in get_layer_target_modules(layer).items():
                for m_idx, mod in enumerate(mods):
                    handles.append(mod.register_forward_hook(make_hook(l_idx, comp, m_idx)))

        try:
            for p in pairs[:max_pairs]:
                ids, r_start = encode_context_and_reply(tokenizer, p["context_messages"], p[reply_key])
                cur_span[0] = min(r_start, max(0, len(ids) - 1))
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                out = base(input_ids=inp, output_hidden_states=True, use_cache=False)
                # Hidden states: tuple of (L + 1) tensors of shape (1, seq, D)
                hs_pooled = torch.stack(
                    [h[0, cur_span[0] :, :].float().mean(dim=0).cpu() for h in out.hidden_states],
                    dim=0,
                )
                all_res.append(hs_pooled)
        finally:
            for h in handles:
                h.remove()

        res_tensor = torch.stack(all_res, dim=0)  # (N, L+1, D)
        io_final: list[dict[str, dict[int, tuple[Tensor, Tensor]]]] = []
        for l_idx in range(n_layers):
            layer_map: dict[str, dict[int, tuple[Tensor, Tensor]]] = {}
            for comp, m_map in io_accum[l_idx].items():
                layer_map[comp] = {}
                for m_idx, (xs, ys) in m_map.items():
                    layer_map[comp][m_idx] = (torch.stack(xs, dim=0), torch.stack(ys, dim=0))
            io_final.append(layer_map)
        return res_tensor, io_final

    good_res, good_io = collect_side("good_reply")
    bad_res, bad_io = collect_side("bad_reply")
    return {
        "n_layers": n_layers,
        "good_res": good_res,
        "bad_res": bad_res,
        "good_io": good_io,
        "bad_io": bad_io,
    }


# ---------------------------------------------------------------------------
# Method implementations & PEFT LoRA helpers
# ---------------------------------------------------------------------------


def _attach_zero_lora(base_model: nn.Module, rank: int) -> PeftModel:
    targets = collect_target_module_names(base_model)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, cfg)
    zero_lora_weights(peft_model)
    return peft_model


def zero_lora_weights(peft_model: PeftModel) -> None:
    for name, module in peft_model.named_modules():
        if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
            nn.init.zeros_(module.weight)


def reset_standard_lora_weights(peft_model: PeftModel, seed: int = 42) -> None:
    """Initialize lora_A with Kaiming uniform and lora_B with zeros (zero deltaW, non-zero grad)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not hasattr(module, "weight"):
                continue
            if "lora_A" in name:
                w = torch.empty(module.weight.shape, dtype=torch.float32)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=gen)
                module.weight.copy_(w.to(device=module.weight.device, dtype=module.weight.dtype))
            elif "lora_B" in name:
                nn.init.zeros_(module.weight)


def _svd_project_delta_to_lora(delta_w: Tensor, rank: int, seed: int) -> tuple[Tensor, Tensor]:
    """Project a dense delta matrix ``delta_w`` (d_out, d_in) to ``(lora_A, lora_B)`` of rank ``r``."""
    d_out, d_in = delta_w.shape
    eff_r = min(rank, d_out, d_in)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    U, S, V = torch.svd_lowrank(delta_w.float(), q=min(min(d_out, d_in), 2 * eff_r + 4), niter=6)
    U = U[:, :eff_r]
    S = S[:eff_r]
    Vh = V[:, :eff_r].T
    sqrt_s = torch.sqrt(S.clamp(min=0.0))
    B_part = U * sqrt_s.unsqueeze(0)
    A_part = sqrt_s.unsqueeze(1) * Vh
    if eff_r < rank:
        B_pad = torch.zeros(d_out, rank, dtype=B_part.dtype, device=B_part.device)
        A_pad = torch.zeros(rank, d_in, dtype=A_part.dtype, device=A_part.device)
        B_pad[:, :eff_r] = B_part
        A_pad[:eff_r, :] = A_part
        return A_pad, B_pad
    return A_part, B_part


def fit_ara_lora(
    *,
    peft_model: PeftModel,
    tokenizer: Any,
    feats: dict[str, Any],
    heldout_pairs: Sequence[dict[str, Any]],
    ref_eval: dict[str, Any],
    heretic_root: Path,
    rank: int,
    seed: int = 42,
    n_opt_steps: int = 5,
    max_iter: int = 20,
    preserve_row_magnitudes: bool = True,
) -> dict[str, Any]:
    """Run heretic's ARA L-BFGS optimization across candidate configs, pick best on heldout."""
    heretic = load_heretic_ara_primitives(heretic_root)
    ara_loss = heretic["ara_loss"]
    Params = heretic["Parameters"]
    n_layers = feats["n_layers"]
    n_prompts = feats["good_res"].shape[0]
    k_nn = min(5, max(1, n_prompts))

    mid = n_layers // 2
    candidates = [
        Params(
            start_layer_index=mid,
            end_layer_index=n_layers,
            preserve_good_behavior_weight=0.7,
            steer_bad_behavior_weight=0.2,
            overcorrect_relative_weight=0.5,
            neighbor_count=k_nn,
        ),
        Params(
            start_layer_index=max(0, n_layers // 3),
            end_layer_index=n_layers,
            preserve_good_behavior_weight=0.5,
            steer_bad_behavior_weight=0.4,
            overcorrect_relative_weight=0.8,
            neighbor_count=k_nn,
        ),
    ]

    layers = get_transformer_layers(peft_model)
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None

    for c_idx, params in enumerate(candidates):
        zero_lora_weights(peft_model)
        gen = torch.Generator().manual_seed(seed + c_idx)

        for l_idx in range(params.start_layer_index, params.end_layer_index):
            for comp, mods in get_layer_target_modules(layers[l_idx]).items():
                for m_idx, mod in enumerate(mods):
                    W_base = mod.base_layer.weight.detach().float()
                    W_row_norms = LA.vector_norm(W_base, dim=1, keepdim=True).detach()
                    lora_A = mod.lora_A["default"].weight
                    lora_B = mod.lora_B["default"].weight

                    init_A = torch.empty(lora_A.shape)
                    nn.init.kaiming_uniform_(init_A, a=math.sqrt(5), generator=gen)
                    with torch.no_grad():
                        lora_A.copy_(init_A.to(device=lora_A.device, dtype=lora_A.dtype))
                        lora_B.zero_()

                    # Optimize in float32 parameters, then copy into adapter
                    opt_A = lora_A.detach().float().clone().requires_grad_(True)
                    opt_B = lora_B.detach().float().clone().requires_grad_(True)

                    g_in, g_out = feats["good_io"][l_idx][comp][m_idx]
                    b_in, b_out = feats["bad_io"][l_idx][comp][m_idx]
                    g_in = g_in.to(device=W_base.device)
                    g_out = g_out.to(device=W_base.device)
                    b_in = b_in.to(device=W_base.device)
                    b_out = b_out.to(device=W_base.device)

                    g_prod = g_in @ W_base.T
                    b_prod = b_in @ W_base.T
                    bias = getattr(mod.base_layer, "bias", None)
                    if bias is not None:
                        bias = bias.detach().float().to(W_base.device)

                    def objective() -> Tensor:
                        ng = g_prod + (g_in @ opt_A.T) @ opt_B.T
                        nb = b_prod + (b_in @ opt_A.T) @ opt_B.T
                        if preserve_row_magnitudes:
                            eff_norms = LA.vector_norm(W_base + (opt_B @ opt_A), dim=1).clamp(min=1e-8)
                            scale = W_row_norms.squeeze(1) / eff_norms
                            ng = ng * scale
                            nb = nb * scale
                        if bias is not None:
                            ng = ng + bias
                            nb = nb + bias
                        return ara_loss(g_out, b_out, ng, nb, params)

                    optimizer = torch.optim.LBFGS(
                        [opt_A, opt_B],
                        lr=1.0,
                        max_iter=max_iter,
                        history_size=10,
                        line_search_fn="strong_wolfe",
                    )

                    def closure() -> Tensor:
                        optimizer.zero_grad()
                        loss = objective()
                        loss.backward()
                        return loss

                    for _ in range(n_opt_steps):
                        optimizer.step(closure)

                    with torch.no_grad():
                        if preserve_row_magnitudes:
                            W_eff = W_base + (opt_B @ opt_A)
                            W_eff = F.normalize(W_eff, p=2, dim=1) * W_row_norms
                            delta_w = W_eff - W_base
                            proj_A, proj_B = _svd_project_delta_to_lora(delta_w, rank=rank, seed=seed)
                        else:
                            proj_A, proj_B = opt_A.detach(), opt_B.detach()
                        lora_A.copy_(proj_A.to(device=lora_A.device, dtype=lora_A.dtype))
                        lora_B.copy_(proj_B.to(device=lora_B.device, dtype=lora_B.dtype))

        peft_model.enable_adapter_layers()
        ev = evaluate_heldout_contrastive_objective(
            peft_model, tokenizer, heldout_pairs, ref_good_lp=ref_eval["good_lps"]
        )
        if best_metrics is None or ev["objective"] > best_metrics["objective"]:
            best_metrics = {k: v for k, v in ev.items() if k != "good_lps"}
            best_params = params.to_dict()
            best_state = {
                n: p.detach().cpu().clone()
                for n, p in peft_model.named_parameters()
                if "lora_" in n
            }

    assert best_state is not None
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(device=p.device, dtype=p.dtype))
    return {"heldout_metrics": best_metrics, "selected_params": best_params, "heretic_ara_path": heretic["ara_path"]}


def fit_mean_diff_ablation_lora(
    *,
    peft_model: PeftModel,
    tokenizer: Any,
    feats: dict[str, Any],
    heldout_pairs: Sequence[dict[str, Any]],
    ref_eval: dict[str, Any],
    rank_grid: Sequence[int] = (1, 4, 8),
    strength_grid: Sequence[float] = (0.5, 1.0),
) -> dict[str, Any]:
    """Mean-difference orthogonalized directional ablation (rank 1..k) -> PEFT LoRA."""
    n_layers = feats["n_layers"]
    good_res = feats["good_res"]  # (N, L+1, D)
    bad_res = feats["bad_res"]
    good_mean = good_res.mean(dim=0)  # (L+1, D)
    diff_res = bad_res - good_mean.unsqueeze(0)  # (N, L+1, D)
    layers = get_transformer_layers(peft_model)

    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_cfg: dict[str, Any] | None = None

    start_l = n_layers // 2
    max_rank = peft_model.peft_config["default"].r

    for k in rank_grid:
        eff_k = min(k, max_rank)
        for lam in strength_grid:
            zero_lora_weights(peft_model)
            for l_idx in range(start_l, n_layers):
                # Orthogonalize difference matrix against good_mean[l_idx + 1]
                g_dir = F.normalize(good_mean[l_idx + 1], p=2, dim=0)
                M = diff_res[:, l_idx + 1, :]  # (N, D)
                M_orth = M - (M @ g_dir).unsqueeze(1) * g_dir.unsqueeze(0)
                if eff_k == 1:
                    v = F.normalize(M_orth.mean(dim=0), p=2, dim=0).unsqueeze(1)  # (D, 1)
                else:
                    _, _, V = torch.linalg.svd(M_orth, full_matrices=False)
                    v = V[:eff_k].T  # (D, actual_k)

                for _, mods in get_layer_target_modules(layers[l_idx]).items():
                    for mod in mods:
                        W_base = mod.base_layer.weight.detach().float()
                        v_dev = v.to(W_base.device)
                        if v_dev.shape[0] != W_base.shape[0]:
                            continue
                        actual_k = v_dev.shape[1]
                        # delta W = -lam * Q @ (Q^T @ W)
                        lora_B_part = -lam * v_dev  # (d_out, actual_k)
                        lora_A_part = v_dev.T @ W_base  # (actual_k, d_in)
                        lA = mod.lora_A["default"].weight
                        lB = mod.lora_B["default"].weight
                        with torch.no_grad():
                            lA.zero_()
                            lB.zero_()
                            lA[:actual_k, :].copy_(lora_A_part.to(device=lA.device, dtype=lA.dtype))
                            lB[:, :actual_k].copy_(lora_B_part.to(device=lB.device, dtype=lB.dtype))

            peft_model.enable_adapter_layers()
            ev = evaluate_heldout_contrastive_objective(
                peft_model, tokenizer, heldout_pairs, ref_good_lp=ref_eval["good_lps"]
            )
            if best_metrics is None or ev["objective"] > best_metrics["objective"]:
                best_metrics = {k2: v2 for k2, v2 in ev.items() if k2 != "good_lps"}
                best_cfg = {"k": eff_k, "lambda": lam, "start_layer": start_l, "end_layer": n_layers}
                best_state = {
                    n: p.detach().cpu().clone()
                    for n, p in peft_model.named_parameters()
                    if "lora_" in n
                }

    assert best_state is not None
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(device=p.device, dtype=p.dtype))
    return {"heldout_metrics": best_metrics, "selected_params": best_cfg}


def fit_random_ablation_lora(
    *,
    peft_model: PeftModel,
    tokenizer: Any,
    heldout_pairs: Sequence[dict[str, Any]],
    ref_eval: dict[str, Any],
    rank: int,
    reference_norms: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Random orthonormal subspace ablation of matched rank & Frobenius norm."""
    n_layers = len(get_transformer_layers(peft_model))
    layers = get_transformer_layers(peft_model)
    start_l = n_layers // 2
    zero_lora_weights(peft_model)
    gen = torch.Generator().manual_seed(seed + 1009)

    for l_idx in range(start_l, n_layers):
        for comp, mods in get_layer_target_modules(layers[l_idx]).items():
            for m_idx, mod in enumerate(mods):
                W_base = mod.base_layer.weight.detach().float()
                d_out, d_in = W_base.shape
                eff_r = min(rank, d_out)
                rand_mat = torch.randn(d_out, eff_r, generator=gen)
                Q, _ = torch.linalg.qr(rand_mat, mode="reduced")
                Q = Q.to(W_base.device)
                # Standard directional ablation delta = -scale * Q @ (Q^T @ W)
                raw_delta = -(Q @ (Q.T @ W_base))
                key = f"{l_idx}:{comp}:{m_idx}"
                if reference_norms and key in reference_norms:
                    cur_norm = LA.matrix_norm(raw_delta).clamp(min=1e-8)
                    raw_delta = raw_delta * (reference_norms[key] / cur_norm)
                else:
                    # Gentle default scaling to keep random ablation comparable
                    raw_delta = raw_delta * 0.25
                proj_A, proj_B = _svd_project_delta_to_lora(raw_delta, rank=rank, seed=seed + l_idx)
                lA = mod.lora_A["default"].weight
                lB = mod.lora_B["default"].weight
                with torch.no_grad():
                    lA.copy_(proj_A.to(device=lA.device, dtype=lA.dtype))
                    lB.copy_(proj_B.to(device=lB.device, dtype=lB.dtype))

    peft_model.enable_adapter_layers()
    ev = evaluate_heldout_contrastive_objective(
        peft_model, tokenizer, heldout_pairs, ref_good_lp=ref_eval["good_lps"]
    )
    return {
        "heldout_metrics": {k: v for k, v in ev.items() if k != "good_lps"},
        "selected_params": {"rank": rank, "start_layer": start_l, "end_layer": n_layers, "seed": seed + 1009},
    }


def collect_lora_module_norms(peft_model: PeftModel) -> dict[str, float]:
    layers = get_transformer_layers(peft_model)
    norms: dict[str, float] = {}
    for l_idx, layer in enumerate(layers):
        for comp, mods in get_layer_target_modules(layer).items():
            for m_idx, mod in enumerate(mods):
                lA = mod.lora_A["default"].weight.detach().float()
                lB = mod.lora_B["default"].weight.detach().float()
                delta = lB @ lA
                norms[f"{l_idx}:{comp}:{m_idx}"] = float(LA.matrix_norm(delta).item())
    return norms


def fit_dpo_lora(
    *,
    peft_model: PeftModel,
    tokenizer: Any,
    fit_pairs: Sequence[dict[str, Any]],
    heldout_pairs: Sequence[dict[str, Any]],
    ref_eval: dict[str, Any],
    beta: float = 0.1,
    lr: float = 5e-5,
    epochs: int = 2,
    max_train_pairs: int = 600,
    seed: int = 42,
) -> dict[str, Any]:
    """Train r=32 DPO-LoRA on exact same contrastive fit split."""
    torch.manual_seed(seed)
    reset_standard_lora_weights(peft_model, seed=seed)
    device = next(peft_model.parameters()).device
    train_subset = list(fit_pairs[:max_train_pairs])

    # Precompute frozen reference logprobs with adapter disabled (no 2nd model copy needed)
    encoded_pairs: list[tuple[Tensor, int, Tensor, int, float, float]] = []
    peft_model.disable_adapter_layers()
    peft_model.eval()
    with torch.no_grad():
        for p in train_subset:
            g_ids, g_start = encode_context_and_reply(tokenizer, p["context_messages"], p["good_reply"])
            b_ids, b_start = encode_context_and_reply(tokenizer, p["context_messages"], p["bad_reply"])
            g_t = torch.tensor([g_ids], dtype=torch.long, device=device)
            b_t = torch.tensor([b_ids], dtype=torch.long, device=device)
            ref_g = float(compute_reply_logprob_sum(peft_model, g_t, g_start).item())
            ref_b = float(compute_reply_logprob_sum(peft_model, b_t, b_start).item())
            encoded_pairs.append((g_t, g_start, b_t, b_start, ref_g, ref_b))

    peft_model.enable_adapter_layers()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0

    for ep in range(1, epochs + 1):
        peft_model.train()
        for idx, (g_t, g_start, b_t, b_start, ref_g, ref_b) in enumerate(encoded_pairs):
            pol_g = compute_reply_logprob_sum(peft_model, g_t, g_start)[0]
            pol_b = compute_reply_logprob_sum(peft_model, b_t, b_start)[0]
            logits = beta * ((pol_g - ref_g) - (pol_b - ref_b))
            loss = -F.logsigmoid(logits)
            loss.backward()
            if (idx + 1) % 8 == 0 or (idx + 1) == len(encoded_pairs):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        peft_model.eval()
        ev = evaluate_heldout_contrastive_objective(
            peft_model, tokenizer, heldout_pairs, ref_good_lp=ref_eval["good_lps"]
        )
        if best_metrics is None or ev["objective"] > best_metrics["objective"]:
            best_metrics = {k: v for k, v in ev.items() if k != "good_lps"}
            best_epoch = ep
            best_state = {
                n: p.detach().cpu().clone()
                for n, p in peft_model.named_parameters()
                if "lora_" in n
            }

    assert best_state is not None
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(device=p.device, dtype=p.dtype))
    peft_model.eval()
    return {
        "heldout_metrics": best_metrics,
        "selected_params": {"rank": peft_model.peft_config["default"].r, "beta": beta, "lr": lr, "best_epoch": best_epoch},
    }


def fit_steering_addition(
    *,
    base_model: nn.Module,
    tokenizer: Any,
    feats: dict[str, Any],
    heldout_pairs: Sequence[dict[str, Any]],
    ref_eval: dict[str, Any],
    alpha_grid: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """PIVOT-style normalized residual vector addition via toggleable forward hooks."""
    n_layers = feats["n_layers"]
    good_mean = feats["good_res"].mean(dim=0)  # (L+1, D)
    bad_mean = feats["bad_res"].mean(dim=0)
    raw_vecs = good_mean[1:] - bad_mean[1:]  # (L, D)
    unit_vecs = F.normalize(raw_vecs, p=2, dim=1)

    start_l = n_layers // 2
    end_l = n_layers
    best_alpha = 1.0
    best_metrics: dict[str, Any] | None = None

    for alpha in alpha_grid:
        ctrl = ModificationController(
            model=base_model,
            tokenizer=tokenizer,
            kind="steering_hook",
            steering_vectors=unit_vecs,
            steering_layers=list(range(start_l, end_l)),
            steering_alpha=alpha,
        )
        ctrl.set_enabled(True)
        try:
            ev = evaluate_heldout_contrastive_objective(
                base_model, tokenizer, heldout_pairs, ref_good_lp=ref_eval["good_lps"]
            )
        finally:
            ctrl.set_enabled(False)
        if best_metrics is None or ev["objective"] > best_metrics["objective"]:
            best_metrics = {k: v for k, v in ev.items() if k != "good_lps"}
            best_alpha = alpha

    return {
        "unit_vectors": unit_vecs,
        "layers": list(range(start_l, end_l)),
        "alpha": best_alpha,
        "heldout_metrics": best_metrics,
    }


# ---------------------------------------------------------------------------
# Unified toggleable ModificationController & 20-prompt exact-unplug check
# ---------------------------------------------------------------------------


@dataclass
class ModificationController:
    """Reversible controller supporting plug/unplug for all 6 Route E0 methods."""

    model: nn.Module
    tokenizer: Any
    kind: str  # "none" | "peft_lora" | "steering_hook" | "prompt_only"
    steering_vectors: Tensor | None = None
    steering_layers: list[int] | None = None
    steering_alpha: float = 0.0
    prompt_suffix: str = ANTI_LEAKAGE_PROMPT_SENTENCE
    enabled: bool = False
    _hooks: list[Any] | None = None

    def set_enabled(self, flag: bool) -> None:
        if self.kind == "peft_lora":
            assert isinstance(self.model, PeftModel)
            if flag:
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()
        elif self.kind == "steering_hook":
            self._remove_hooks()
            if flag:
                self._install_steering_hooks()
        elif self.kind == "prompt_only":
            pass
        elif self.kind == "none":
            pass
        else:
            raise ValueError(f"Unknown modification kind: {self.kind}")
        self.enabled = flag

    def _remove_hooks(self) -> None:
        if self._hooks:
            for h in self._hooks:
                h.remove()
        self._hooks = []

    def _install_steering_hooks(self) -> None:
        assert self.steering_vectors is not None and self.steering_layers is not None
        layers = get_transformer_layers(self.model)
        self._hooks = []
        alpha = float(self.steering_alpha)
        for l_idx in self.steering_layers:
            vec = self.steering_vectors[l_idx].detach()

            def _make_hook(v_cpu: Tensor):
                def hook(_mod: nn.Module, _inp: tuple[Any, ...], output: Any):
                    if isinstance(output, tuple):
                        hs = output[0]
                        delta = (alpha * v_cpu).to(device=hs.device, dtype=hs.dtype)
                        return (hs + delta, *output[1:])
                    delta = (alpha * v_cpu).to(device=output.device, dtype=output.dtype)
                    return output + delta

                return hook

            self._hooks.append(layers[l_idx].register_forward_hook(_make_hook(vec)))

    @contextmanager
    def plugged(self):
        prev = self.enabled
        self.set_enabled(True)
        try:
            yield self
        finally:
            self.set_enabled(prev)

    @contextmanager
    def unplugged(self):
        prev = self.enabled
        self.set_enabled(False)
        try:
            yield self
        finally:
            self.set_enabled(prev)

    def transform_messages(self, messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        out = [dict(m) for m in messages]
        if self.kind == "prompt_only" and self.enabled:
            if out and out[0]["role"] == "system":
                content = out[0]["content"]
                if self.prompt_suffix not in content:
                    if "\nProblem:" in content:
                        head, tail = content.split("\nProblem:", 1)
                        out[0]["content"] = f"{head}\n{self.prompt_suffix}\nProblem:{tail}"
                    else:
                        out[0]["content"] = f"{content}\n{self.prompt_suffix}"
            else:
                out.insert(0, {"role": "system", "content": self.prompt_suffix})
        return out

    def _encode_messages(self, msgs: Sequence[dict[str, str]]) -> dict[str, Tensor]:
        device = next(self.model.parameters()).device
        t_msgs = self.transform_messages(msgs)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None
        ):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    t_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    t_msgs, tokenize=False, add_generation_prompt=True
                )
        else:
            rendered = "\n".join(f"{m['role']}: {m['content']}" for m in t_msgs) + "\nassistant: "
        enc = self.tokenizer(rendered, return_tensors="pt", return_token_type_ids=False)
        out = {"input_ids": enc["input_ids"].to(device)}
        if enc.get("attention_mask") is not None:
            out["attention_mask"] = enc["attention_mask"].to(device)
        return out

    @torch.no_grad()
    def first_token_logits(self, prompts_messages: Sequence[Sequence[dict[str, str]]]) -> list[Tensor]:
        self.model.eval()
        out_logits: list[Tensor] = []
        for msgs in prompts_messages:
            enc = self._encode_messages(msgs)
            logits = self.model(**enc, use_cache=False).logits[0, -1, :].detach().float().cpu()
            out_logits.append(logits)
        return out_logits

    @torch.no_grad()
    def greedy_generate_ids(
        self,
        prompts_messages: Sequence[Sequence[dict[str, str]]],
        *,
        max_new_tokens: int = 32,
    ) -> list[list[int]]:
        self.model.eval()
        all_gen_ids: list[list[int]] = []
        for msgs in prompts_messages:
            enc = self._encode_messages(msgs)
            input_ids = enc["input_ids"]
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
                use_cache=True,
            )
            new_tokens = gen[0, input_ids.shape[1] :].detach().cpu().tolist()
            all_gen_ids.append(new_tokens)
        return all_gen_ids

    @torch.no_grad()
    def sample_generate_ids(
        self,
        prompts_messages: Sequence[Sequence[dict[str, str]]],
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> list[list[int]]:
        """Sampled generation matching PedagogicalRL/EduardoRL rollout temperature/top_p."""
        self.model.eval()
        torch.manual_seed(seed)
        all_gen_ids: list[list[int]] = []
        do_sample = temperature > 0.0
        for msgs in prompts_messages:
            enc = self._encode_messages(msgs)
            input_ids = enc["input_ids"]
            gen_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": enc.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", 0) or 0,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            gen = self.model.generate(**gen_kwargs)
            new_tokens = gen[0, input_ids.shape[1] :].detach().cpu().tolist()
            all_gen_ids.append(new_tokens)
        return all_gen_ids


def verify_unplug_greedy_exact_match(
    *,
    unmodified_token_ids: Sequence[Sequence[int]],
    controller: ModificationController,
    prompts_messages: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int = 32,
    require_plugged_effect: bool = True,
) -> dict[str, Any]:
    """Verify: (1) unplugged matches unmodified base 100% token-for-token; (2) plugged takes effect."""
    if len(unmodified_token_ids) != len(prompts_messages):
        raise ValueError("Prompt count mismatch")
    with controller.unplugged():
        unplugged_ids = controller.greedy_generate_ids(prompts_messages, max_new_tokens=max_new_tokens)
        unplugged_logits = controller.first_token_logits(prompts_messages[:4])
    with controller.plugged():
        plugged_ids = controller.greedy_generate_ids(prompts_messages, max_new_tokens=max_new_tokens)
        plugged_logits = controller.first_token_logits(prompts_messages[:4])

    mismatches = sum(1 for a, b in zip(unmodified_token_ids, unplugged_ids) if list(a) != list(b))
    plugged_changed = sum(1 for a, b in zip(unmodified_token_ids, plugged_ids) if list(a) != list(b))
    max_logit_diff = max(
        float((pl - ul).abs().max().item()) for pl, ul in zip(plugged_logits, unplugged_logits)
    ) if plugged_logits else 0.0

    if mismatches != 0:
        raise AssertionError(
            f"Unplugged token-by-token match FAILED: {mismatches}/{len(prompts_messages)} prompts differ"
        )
    if require_plugged_effect and controller.kind != "none" and plugged_changed == 0 and max_logit_diff <= 1e-6:
        raise AssertionError(
            f"Plugged modification ({controller.kind}) had zero effect across {len(prompts_messages)} prompts"
        )
    return {
        "n_prompts": len(prompts_messages),
        "unplug_exact_matches": len(prompts_messages) - mismatches,
        "unplug_mismatches": mismatches,
        "plugged_changed_prompts": plugged_changed,
        "plugged_max_logit_diff": max_logit_diff,
    }


def export_merged_bf16_for_vllm(
    *,
    base_model_path: Path,
    artifact_dir: Path,
    merged_out_dir: Path,
) -> Path:
    """Temporarily merge a PEFT LoRA artifact in bfloat16 for high-speed vLLM rollout."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig

    manifest = json.loads((Path(artifact_dir) / "modification_manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "peft_lora":
        raise ValueError(f"Only peft_lora can be merged for vLLM, got {manifest['kind']}")

    configs = PretrainedConfig.get_config_dict(str(base_model_path))
    cls = AutoModelForImageTextToText if any("vision_config" in c for c in configs) else AutoModelForCausalLM
    base = cls.from_pretrained(str(base_model_path), dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    peft_m = PeftModel.from_pretrained(base, str(artifact_dir / "adapter"))
    merged = peft_m.merge_and_unload()
    merged_out_dir = Path(merged_out_dir)
    if merged_out_dir.exists():
        shutil.rmtree(merged_out_dir)
    merged_out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_out_dir))
    tok = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    tok.save_pretrained(str(merged_out_dir))
    return merged_out_dir


def fit_all_e0_methods_for_base(
    *,
    base_label: str,
    base_model_path: str | Path | None = None,
    loaded_model: nn.Module | None = None,
    loaded_tokenizer: Any | None = None,
    fit_pairs: Sequence[dict[str, Any]],
    heldout_pairs: Sequence[dict[str, Any]],
    heretic_root: Path,
    out_dir: Path,
    methods: Sequence[str] = ALL_E0_METHODS,
    ara_small_rank: int = 16,
    unplug_check_prompts: Sequence[Sequence[dict[str, str]]] | None = None,
    seed: int = 42,
    dpo_epochs: int = 2,
    max_feature_pairs: int = 400,
    ara_opt_steps: int = 5,
    ara_max_iter: int = 20,
) -> dict[str, Any]:
    """Fit and export all E0 modifications for one base model + run 20-prompt unplug self-check."""
    out_dir = Path(out_dir) / base_label
    out_dir.mkdir(parents=True, exist_ok=True)

    if loaded_model is None or loaded_tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig

        assert base_model_path is not None
        configs = PretrainedConfig.get_config_dict(str(base_model_path))
        cls = AutoModelForImageTextToText if any("vision_config" in c for c in configs) else AutoModelForCausalLM
        loaded_tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        loaded_model = cls.from_pretrained(
            str(base_model_path),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    loaded_model.eval()
    ref_eval = evaluate_heldout_contrastive_objective(loaded_model, loaded_tokenizer, heldout_pairs)
    feats = extract_contrastive_activations(
        loaded_model, loaded_tokenizer, fit_pairs, max_pairs=max_feature_pairs
    )

    if unplug_check_prompts is None:
        unplug_check_prompts = [p["context_messages"] for p in heldout_pairs[:20]]
    base_ctrl = ModificationController(model=loaded_model, tokenizer=loaded_tokenizer, kind="none")
    base_greedy_ids = base_ctrl.greedy_generate_ids(unplug_check_prompts, max_new_tokens=24)

    artifacts: dict[str, Any] = {}
    ara50_norms: dict[str, float] | None = None

    # Helper to wrap fresh PEFT adapter, fit, verify unplug, save, and unwrap back to pristine base
    def _run_peft_method(method_name: str, rank: int, fit_fn) -> dict[str, Any]:
        peft_m = _attach_zero_lora(loaded_model, rank=rank)
        try:
            meta = fit_fn(peft_m)
            ctrl = ModificationController(model=peft_m, tokenizer=loaded_tokenizer, kind="peft_lora")
            chk = verify_unplug_greedy_exact_match(
                unmodified_token_ids=base_greedy_ids,
                controller=ctrl,
                prompts_messages=unplug_check_prompts,
                max_new_tokens=24,
            )
            m_dir = out_dir / method_name
            m_dir.mkdir(parents=True, exist_ok=True)
            peft_m.save_pretrained(str(m_dir / "adapter"))
            manifest = {
                "base_label": base_label,
                "method": method_name,
                "kind": "peft_lora",
                "rank": rank,
                "unplug_check": chk,
                **meta,
            }
            atomic_write_json(m_dir / "modification_manifest.json", manifest)
            return manifest
        finally:
            if method_name == "ara_r50":
                nonlocal ara50_norms
                ara50_norms = collect_lora_module_norms(peft_m)
            peft_m.unload()

    for mname in methods:
        if mname == "ara_r50":
            artifacts[mname] = _run_peft_method(
                "ara_r50",
                50,
                lambda pm: fit_ara_lora(
                    peft_model=pm,
                    tokenizer=loaded_tokenizer,
                    feats=feats,
                    heldout_pairs=heldout_pairs,
                    ref_eval=ref_eval,
                    heretic_root=heretic_root,
                    rank=50,
                    seed=seed,
                    n_opt_steps=ara_opt_steps,
                    max_iter=ara_max_iter,
                ),
            )
        elif mname == "ara_r16":
            artifacts[mname] = _run_peft_method(
                f"ara_r{ara_small_rank}",
                ara_small_rank,
                lambda pm: fit_ara_lora(
                    peft_model=pm,
                    tokenizer=loaded_tokenizer,
                    feats=feats,
                    heldout_pairs=heldout_pairs,
                    ref_eval=ref_eval,
                    heretic_root=heretic_root,
                    rank=ara_small_rank,
                    seed=seed,
                    n_opt_steps=ara_opt_steps,
                    max_iter=ara_max_iter,
                ),
            )
        elif mname == "dir_ablation":
            artifacts[mname] = _run_peft_method(
                "dir_ablation",
                8,
                lambda pm: fit_mean_diff_ablation_lora(
                    peft_model=pm,
                    tokenizer=loaded_tokenizer,
                    feats=feats,
                    heldout_pairs=heldout_pairs,
                    ref_eval=ref_eval,
                ),
            )
        elif mname == "dpo_lora_r32":
            artifacts[mname] = _run_peft_method(
                "dpo_lora_r32",
                32,
                lambda pm: fit_dpo_lora(
                    peft_model=pm,
                    tokenizer=loaded_tokenizer,
                    fit_pairs=fit_pairs,
                    heldout_pairs=heldout_pairs,
                    ref_eval=ref_eval,
                    epochs=dpo_epochs,
                    seed=seed,
                ),
            )
        elif mname == "random_ablation":
            artifacts[mname] = _run_peft_method(
                "random_ablation",
                50,
                lambda pm: fit_random_ablation_lora(
                    peft_model=pm,
                    tokenizer=loaded_tokenizer,
                    heldout_pairs=heldout_pairs,
                    ref_eval=ref_eval,
                    rank=50,
                    reference_norms=ara50_norms,
                    seed=seed,
                ),
            )
        elif mname == "dir_addition":
            s_res = fit_steering_addition(
                base_model=loaded_model,
                tokenizer=loaded_tokenizer,
                feats=feats,
                heldout_pairs=heldout_pairs,
                ref_eval=ref_eval,
            )
            ctrl = ModificationController(
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                kind="steering_hook",
                steering_vectors=s_res["unit_vectors"],
                steering_layers=s_res["layers"],
                steering_alpha=s_res["alpha"],
            )
            chk = verify_unplug_greedy_exact_match(
                unmodified_token_ids=base_greedy_ids,
                controller=ctrl,
                prompts_messages=unplug_check_prompts,
                max_new_tokens=24,
            )
            m_dir = out_dir / "dir_addition"
            m_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"unit_vectors": s_res["unit_vectors"], "layers": s_res["layers"], "alpha": s_res["alpha"]},
                m_dir / "steering_vectors.pt",
            )
            manifest = {
                "base_label": base_label,
                "method": "dir_addition",
                "kind": "steering_hook",
                "layers": s_res["layers"],
                "alpha": s_res["alpha"],
                "heldout_metrics": s_res["heldout_metrics"],
                "unplug_check": chk,
            }
            atomic_write_json(m_dir / "modification_manifest.json", manifest)
            artifacts[mname] = manifest
        elif mname == "prompt_only":
            ctrl = ModificationController(
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                kind="prompt_only",
            )
            chk = verify_unplug_greedy_exact_match(
                unmodified_token_ids=base_greedy_ids,
                controller=ctrl,
                prompts_messages=unplug_check_prompts,
                max_new_tokens=24,
            )
            m_dir = out_dir / "prompt_only"
            m_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "base_label": base_label,
                "method": "prompt_only",
                "kind": "prompt_only",
                "prompt_suffix": ANTI_LEAKAGE_PROMPT_SENTENCE,
                "unplug_check": chk,
            }
            atomic_write_json(m_dir / "modification_manifest.json", manifest)
            artifacts[mname] = manifest
        else:
            raise ValueError(f"Unknown method: {mname}")

    atomic_write_json(out_dir / "fit_summary.json", {"base_label": base_label, "methods": artifacts})
    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 E0: fit & export 6 pluggable methods on both bases")
    ap.add_argument("--base-label", required=True, choices=["base_4b", "a_group_step100"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--fit-jsonl", required=True)
    ap.add_argument("--heldout-jsonl", required=True)
    ap.add_argument("--heretic-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ara-small-rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with Path(args.fit_jsonl).open("r", encoding="utf-8") as f:
        fit_pairs = [json.loads(line) for line in f if line.strip()]
    with Path(args.heldout_jsonl).open("r", encoding="utf-8") as f:
        heldout_pairs = [json.loads(line) for line in f if line.strip()]

    artifacts = fit_all_e0_methods_for_base(
        base_label=args.base_label,
        base_model_path=args.base_model,
        fit_pairs=fit_pairs,
        heldout_pairs=heldout_pairs,
        heretic_root=Path(args.heretic_root),
        out_dir=Path(args.out_dir),
        ara_small_rank=args.ara_small_rank,
        seed=args.seed,
    )
    all_unplug_ok = all(a["unplug_check"]["unplug_mismatches"] == 0 for a in artifacts.values())
    print_machine_summary(
        {
            "status": "P4_FIT_DONE",
            "rc": 0,
            "calls": len(artifacts),
            "failures": 0 if all_unplug_ok else 1,
            "base_label": args.base_label,
            "methods_fitted": len(artifacts),
            "unplug_all_exact": all_unplug_ok,
        }
    )


if __name__ == "__main__":
    main()
