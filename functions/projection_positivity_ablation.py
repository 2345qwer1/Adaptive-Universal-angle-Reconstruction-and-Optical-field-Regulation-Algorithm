from __future__ import annotations

"""
projection_positivity_ablation.py

用途：
    为“投影矩阵是否必须非负”这一消融实验提供独立函数。

设计原则：
    1. 不修改原有 generate_initial_projection_matrix / quantize_projection_matrix /
       optimize_projection_matrix 的行为。
    2. 原流程仍然代表“投影强制非负”的物理约束方案。
    3. 本文件中的 signed 函数代表“允许投影为负”的实验方案。

注意：
    允许负投影是数值消融实验，不等同于真实光强可以为负。
    它用于研究 Radon 滤波后负值被截断这一约束对最终 print out 的影响。
"""

from scipy.fft import fft, ifft
from skimage.transform import radon
import numpy as np

from .config import VAMConfig
from .initial_projection import _get_projection_precompute
from .optimization import compute_gradient, compute_loss_and_grad_wrt_dose
from .physics_printing import (
    calculate_iou,
    forward_dose_simulation,
    solve_optimal_global_scale,
)


def generate_initial_projection_matrix_signed(sinogram: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    """
    生成允许为负的 raw projection。

    与原函数 functions.initial_projection.generate_initial_projection_matrix 的区别：
        原函数最后执行 np.clip(filtered, 0.0, None)，会删除滤波后的负值；
        本函数保留 filtered 的原始正负号，用于消融实验。

    输入：
        sinogram:
            已由 Radon target projection 归一化得到的初始 sinogram。
        cfg:
            VAM 参数配置。

    输出：
        filtered:
            经过树脂深度补偿和频域滤波后的连续投影矩阵，允许出现负值。
    """
    filter_kernel, depth = _get_projection_precompute(cfg)
    sinogram_f64 = np.asarray(sinogram, dtype=np.float64)

    compensated = sinogram_f64 * np.exp(float(cfg.resin_alpha) * depth)[:, None]
    filtered = ifft(fft(compensated, axis=0) * filter_kernel[:, None], axis=0).real
    return np.asarray(filtered, dtype=np.float64)


def quantize_projection_matrix_signed(
    projection_matrix: np.ndarray,
    cfg: VAMConfig,
    *,
    reference_abs_value: float | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool]]:
    """
    对 signed projection 做对称离散化。

    原量化逻辑：
        先裁剪到非负，再离散到 0, T, 2T, ..., K*T。

    本实验量化逻辑：
        不裁剪负值，而是按最大绝对值离散到 -K*T, ..., -T, 0, T, ..., K*T。

    这样正负方案使用相同的 K = projection_quantization_max_index，
    区别只在于 signed 方案允许负等级存在。
    """
    proj = np.asarray(projection_matrix, dtype=np.float64)
    enabled = bool(cfg.projection_quantization_enabled)
    max_index = int(cfg.projection_quantization_max_index)
    if max_index < 0:
        raise ValueError("projection_quantization_max_index 必须 >= 0。")

    if reference_abs_value is None:
        reference_abs_value = float(np.max(np.abs(proj))) if proj.size > 0 else 0.0
    reference_abs_value = max(float(reference_abs_value), 0.0)

    if (not enabled) or max_index == 0 or reference_abs_value <= 0.0:
        quantized = proj.copy() if not enabled else np.zeros_like(proj)
        metadata = {
            "enabled": bool(enabled),
            "signed": True,
            "max_index": int(max_index),
            "reference_abs_value": float(reference_abs_value),
            "effective_step": 0.0 if max_index == 0 else float(reference_abs_value) / float(max_index) if reference_abs_value > 0 else 0.0,
            "level_min": -int(max_index),
            "level_max": int(max_index),
        }
        return (quantized, metadata) if return_metadata else quantized

    step = float(reference_abs_value) / float(max_index)
    level_index = np.rint(proj / step)
    level_index = np.clip(level_index, -max_index, max_index)
    quantized = level_index.astype(np.float64) * step

    metadata = {
        "enabled": bool(enabled),
        "signed": True,
        "max_index": int(max_index),
        "reference_abs_value": float(reference_abs_value),
        "effective_step": float(step),
        "level_min": -int(max_index),
        "level_max": int(max_index),
    }
    return (quantized, metadata) if return_metadata else quantized


def compensate_and_print_initial_projection_signed(
    raw_projection: np.ndarray,
    target_mask: np.ndarray,
    cfg: VAMConfig,
) -> dict[str, np.ndarray | float | dict[str, float | int | bool]]:
    """
    对允许负值的 raw projection 执行 proto 阶段补偿和打印仿真。

    与原 compensate_and_print_initial_projection 的共同点：
        1. 使用同样的 target Radon projection 生成 weight map。
        2. 使用同样的全局缩放搜索 solve_optimal_global_scale。
        3. 使用同样的 forward_dose_simulation 得到 dose/cured/IoU。

    与原函数的区别：
        所有 projection 量化都调用 signed 量化函数，不把负投影裁剪为 0。
    """
    target_proj = radon(target_mask, theta=cfg.angles, circle=True, preserve_range=True)
    target_proj = target_proj.astype(np.float64)
    target_proj /= (np.max(target_proj) + 1e-12)

    weight = cfg.min_weight + (1.0 - cfg.min_weight) * target_proj ** cfg.weight_power

    base_projection_continuous = np.asarray(raw_projection, dtype=np.float64) * weight
    base_projection, base_quant_meta = quantize_projection_matrix_signed(
        base_projection_continuous,
        cfg,
        return_metadata=True,
    )

    base_dose = forward_dose_simulation(base_projection, cfg)
    global_k = solve_optimal_global_scale(base_dose, target_mask, cfg.resin_Ec, cfg)

    final_projection_continuous = base_projection * global_k
    final_projection, quant_meta = quantize_projection_matrix_signed(
        final_projection_continuous,
        cfg,
        return_metadata=True,
    )
    final_dose = forward_dose_simulation(final_projection, cfg)
    cured = (final_dose >= cfg.resin_Ec).astype(np.float64)
    iou = calculate_iou(target_mask, cured, cfg=cfg)

    return {
        "projection_matrix": final_projection,
        "projection_matrix_continuous": final_projection_continuous,
        "dose_field": final_dose,
        "cured_mask": cured,
        "iou": float(iou),
        "global_k": float(global_k),
        "weight_map": weight,
        "base_quantization": base_quant_meta,
        "quantization_reference_abs_value": float(quant_meta["reference_abs_value"]),
        "quantization_effective_step": float(quant_meta["effective_step"]),
    }


def _is_better_state(candidate_iou: float, candidate_loss: float, best_iou: float, best_loss: float) -> bool:
    """按照原优化器的规则判断候选状态是否优于历史 best。"""
    if candidate_iou > best_iou:
        return True
    if np.isclose(candidate_iou, best_iou, rtol=0.0, atol=1e-12) and candidate_loss < best_loss:
        return True
    return False


def optimize_projection_matrix_signed(
    initial_projection: np.ndarray,
    target_mask: np.ndarray,
    cfg: VAMConfig,
) -> dict[str, np.ndarray | float | dict[str, object]]:
    """
    对 signed projection 执行离散迭代优化。

    与原 optimize_projection_matrix 的关键区别：
        原优化器把初始投影裁剪到 >= 0，并把离散 level 限制在 [0, K]；
        本函数不做非负裁剪，并把离散 level 限制在 [-K, K]。

    其它部分尽量保持一致：
        - loss 函数一致；
        - 梯度计算一致；
        - greedy_stop / explore_best 模式一致；
        - 每次迭代仍然按梯度符号移动 1 个离散 level。
    """
    optimization_mode = str(getattr(cfg, "optimization_mode", "greedy_stop")).strip()
    if optimization_mode not in {"greedy_stop", "explore_best"}:
        raise ValueError("optimization_mode 只能为 'greedy_stop' 或 'explore_best'。")

    proj_input = np.asarray(initial_projection, dtype=np.float64)
    K = int(cfg.projection_quantization_max_index)
    if K < 0:
        raise ValueError("projection_quantization_max_index 必须 >= 0。")

    reference_abs_value = float(np.max(np.abs(proj_input))) if proj_input.size > 0 else 0.0
    if reference_abs_value <= 0.0 or K == 0:
        proj_best = np.zeros_like(proj_input, dtype=np.float64)
        dose_best = forward_dose_simulation(proj_best, cfg)
        cured_best = (dose_best >= cfg.resin_Ec).astype(np.float64)
        iou_best = calculate_iou(target_mask, cured_best, cfg=cfg)
        loss_best, _ = compute_loss_and_grad_wrt_dose(dose_best, target_mask, cfg)
        history = {
            "mode": optimization_mode,
            "signed_levels": True,
            "ious": [float(iou_best)],
            "losses": [float(loss_best)],
            "applied_level_step": [0],
            "changed_entries": [0],
            "accepted": [True],
            "best_ious": [float(iou_best)],
            "best_losses": [float(loss_best)],
            "best_iou_discrete": float(iou_best),
            "best_loss_at_best_iou": float(loss_best),
            "final_loss": float(loss_best),
        }
        return {
            "projection_matrix": proj_best,
            "dose_field": dose_best,
            "cured_mask": cured_best,
            "iou": float(iou_best),
            "history": history,
            "quantization_reference_abs_value": float(reference_abs_value),
            "quantization_effective_step": 0.0,
        }

    effective_step = reference_abs_value / float(K)
    level_current = np.rint(proj_input / effective_step).astype(np.int64)
    level_current = np.clip(level_current, -K, K)
    proj_current = level_current.astype(np.float64) * effective_step

    dose_current = forward_dose_simulation(proj_current, cfg)
    loss_current, grad_wrt_dose_current = compute_loss_and_grad_wrt_dose(dose_current, target_mask, cfg)
    cured_current = (dose_current >= cfg.resin_Ec).astype(np.float64)
    iou_current = calculate_iou(target_mask, cured_current, cfg=cfg)

    proj_best = proj_current.copy()
    dose_best = dose_current.copy()
    cured_best = cured_current.copy()
    iou_best = float(iou_current)
    loss_best = float(loss_current)

    history = {
        "mode": optimization_mode,
        "signed_levels": True,
        "ious": [float(iou_current)],
        "losses": [float(loss_current)],
        "applied_level_step": [0],
        "changed_entries": [0],
        "accepted": [True],
        "best_ious": [float(iou_best)],
        "best_losses": [float(loss_best)],
    }

    print(
        f"[signed] 初始离散投影 IoU = {iou_current:.4f} | "
        f"levels = {-K} ... {K} | effective step = {effective_step:.6g}"
    )

    for it in range(int(cfg.num_iterations)):
        grad = compute_gradient(proj_current, grad_wrt_dose_current, cfg)
        grad_sign = np.sign(grad).astype(np.int8)
        level_candidate = np.clip(level_current - grad_sign.astype(np.int64), -K, K)
        changed_entries = int(np.count_nonzero(level_candidate != level_current))

        if changed_entries == 0:
            history["ious"].append(float(iou_current))
            history["losses"].append(float(loss_current))
            history["applied_level_step"].append(0)
            history["changed_entries"].append(0)
            history["accepted"].append(False)
            history["best_ious"].append(float(iou_best))
            history["best_losses"].append(float(loss_best))
            if optimization_mode == "greedy_stop":
                print(f"[signed] Iter {it + 1:4d}/{cfg.num_iterations} | 无可更新元素，提前停止。")
                break
            continue

        proj_candidate = level_candidate.astype(np.float64) * effective_step
        dose_candidate = forward_dose_simulation(proj_candidate, cfg)
        loss_candidate, grad_wrt_dose_candidate = compute_loss_and_grad_wrt_dose(dose_candidate, target_mask, cfg)
        cured_candidate = (dose_candidate >= cfg.resin_Ec).astype(np.float64)
        iou_candidate = calculate_iou(target_mask, cured_candidate, cfg=cfg)

        if optimization_mode == "greedy_stop":
            accept = (
                (loss_candidate < loss_current)
                or (
                    np.isclose(loss_candidate, loss_current, rtol=0.0, atol=1e-12)
                    and iou_candidate > iou_current
                )
            )
            if not accept:
                history["ious"].append(float(iou_current))
                history["losses"].append(float(loss_current))
                history["applied_level_step"].append(0)
                history["changed_entries"].append(changed_entries)
                history["accepted"].append(False)
                history["best_ious"].append(float(iou_best))
                history["best_losses"].append(float(loss_best))
                print(
                    f"[signed] Iter {it + 1:4d}/{cfg.num_iterations} | "
                    f"候选未接受：loss {loss_current:.2e} -> {loss_candidate:.2e}, "
                    f"IoU {iou_current:.4f} -> {iou_candidate:.4f} | changed = {changed_entries}"
                )
                break
        else:
            accept = True

        level_current = level_candidate
        proj_current = proj_candidate
        dose_current = dose_candidate
        grad_wrt_dose_current = grad_wrt_dose_candidate
        loss_current = float(loss_candidate)
        cured_current = cured_candidate
        iou_current = float(iou_candidate)

        if _is_better_state(iou_current, loss_current, iou_best, loss_best):
            proj_best = proj_current.copy()
            dose_best = dose_current.copy()
            cured_best = cured_current.copy()
            iou_best = float(iou_current)
            loss_best = float(loss_current)

        history["ious"].append(float(iou_current))
        history["losses"].append(float(loss_current))
        history["applied_level_step"].append(1)
        history["changed_entries"].append(changed_entries)
        history["accepted"].append(bool(accept))
        history["best_ious"].append(float(iou_best))
        history["best_losses"].append(float(loss_best))

        if (it + 1) % int(cfg.print_interval) == 0 or it == int(cfg.num_iterations) - 1:
            print(
                f"[signed] Iter {it + 1:4d}/{cfg.num_iterations} | "
                f"IoU(current) = {iou_current:.4f} | loss = {loss_current:.2e} | "
                f"best IoU = {iou_best:.4f} | changed = {changed_entries}"
            )

    history["best_iou_discrete"] = float(iou_best)
    history["best_loss_at_best_iou"] = float(loss_best)
    history["final_loss"] = float(loss_best)

    return {
        "projection_matrix": proj_best,
        "dose_field": dose_best,
        "cured_mask": cured_best,
        "iou": float(iou_best),
        "history": history,
        "quantization_reference_abs_value": float(reference_abs_value),
        "quantization_effective_step": float(effective_step),
    }
