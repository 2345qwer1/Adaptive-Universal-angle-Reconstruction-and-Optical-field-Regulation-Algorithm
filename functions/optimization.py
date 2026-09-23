from __future__ import annotations

import numpy as np

from .config import VAMConfig
from .physics_printing import (
    calculate_iou,
    forward_dose_simulation,
    get_cached_angle_tables,
    get_cached_pixel_geometry,
)

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    def prange(*args):
        return range(*args)


@njit(parallel=True, fastmath=True)
def _compute_gradient_numba(angles, voxel_size, container_radius,
                            resin_alpha, exposure_time_ms, grad_wrt_dose, N):
    num_angles = angles.shape[0]
    grad = np.zeros((N, num_angles), dtype=np.float64)
    center = N // 2
    exposure_time_s = exposure_time_ms / 1000.0
    angles_rad = np.deg2rad(angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)

    x_1d = (np.arange(N) - center) * voxel_size
    y_1d = (center - np.arange(N)) * voxel_size

    for a in prange(num_angles):
        cos_t = cos_table[a]
        sin_t = sin_table[a]
        angle_rad = angles_rad[a]
        grad_col = grad[:, a]

        for i in range(N):
            y = y_1d[i]
            for j in range(N):
                x = x_1d[j]
                r2 = x * x + y * y
                if r2 > container_radius * container_radius:
                    continue

                r = np.sqrt(r2)
                base_ang = np.arctan2(y, x)
                s_phys = x * cos_t + y * sin_t
                s_pixel = s_phys / voxel_size + center
                if s_pixel < 0.0 or s_pixel > N - 1:
                    continue

                s_floor = int(np.floor(s_pixel))
                s_ceil = min(s_floor + 1, N - 1)
                w_ceil = s_pixel - s_floor
                w_floor = 1.0 - w_ceil

                ang = base_ang - angle_rad
                depth = container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * container_radius)

                atten = np.exp(-resin_alpha * depth) * exposure_time_s
                contrib = grad_wrt_dose[i, j] * atten

                grad_col[s_floor] += contrib * w_floor
                if s_ceil != s_floor:
                    grad_col[s_ceil] += contrib * w_ceil

    return grad


@njit(parallel=True, fastmath=True, cache=True)
def _compute_gradient_geometry_numba(
    angles_rad,
    cos_table,
    sin_table,
    rows,
    cols,
    x_values,
    y_values,
    radius_values,
    base_angle_values,
    detector_voxel_size,
    container_radius,
    resin_alpha,
    exposure_time_ms,
    grad_wrt_dose,
    N,
):
    num_angles = angles_rad.shape[0]
    center = N // 2
    exposure_time_s = exposure_time_ms / 1000.0
    grad = np.zeros((N, num_angles), dtype=np.float64)

    for a in prange(num_angles):
        cos_t = cos_table[a]
        sin_t = sin_table[a]
        angle_rad = angles_rad[a]
        grad_col = grad[:, a]

        for p in range(rows.shape[0]):
            x = x_values[p]
            y = y_values[p]
            s_pixel = (x * cos_t + y * sin_t) / detector_voxel_size + center
            if s_pixel < 0.0 or s_pixel > N - 1:
                continue

            s_floor = int(np.floor(s_pixel))
            s_ceil = min(s_floor + 1, N - 1)
            w_ceil = s_pixel - s_floor
            w_floor = 1.0 - w_ceil

            depth = container_radius - radius_values[p] * np.cos(base_angle_values[p] - angle_rad)
            depth = min(max(depth, 0.0), 2.0 * container_radius)

            atten = np.exp(-resin_alpha * depth) * exposure_time_s
            contrib = grad_wrt_dose[rows[p], cols[p]] * atten
            grad_col[s_floor] += contrib * w_floor
            if s_ceil != s_floor:
                grad_col[s_ceil] += contrib * w_ceil

    return grad


def _resolve_loss_thresholds(cfg: VAMConfig) -> tuple[float, float]:
    bg_over_safe_threshold = (
        cfg.resin_Ec
        if cfg.loss_background_over_safe_threshold is None
        else float(cfg.loss_background_over_safe_threshold)
    )
    target_under_safe_threshold = (
        cfg.resin_Ec
        if cfg.loss_target_under_safe_threshold is None
        else float(cfg.loss_target_under_safe_threshold)
    )
    return float(bg_over_safe_threshold), float(target_under_safe_threshold)


def compute_loss_and_grad_wrt_dose(dose: np.ndarray, target_mask: np.ndarray, cfg: VAMConfig):
    pos_mask = target_mask == 1
    neg_mask = target_mask == 0

    bg_over_safe_threshold, target_under_safe_threshold = _resolve_loss_thresholds(cfg)
    background_overdose = np.maximum(0.0, dose - bg_over_safe_threshold)
    target_underdose = np.maximum(0.0, target_under_safe_threshold - dose)

    loss_bg = (
        float(cfg.loss_background_over_weight) * np.sum(background_overdose[neg_mask] ** 2)
        if np.any(neg_mask) else 0.0
    )
    loss_target = (
        float(cfg.loss_target_under_weight) * np.sum(target_underdose[pos_mask] ** 2)
        if np.any(pos_mask) else 0.0
    )
    loss = float(loss_bg + loss_target)

    grad_wrt_dose = np.zeros_like(dose, dtype=np.float64)
    if np.any(neg_mask):
        grad_wrt_dose[neg_mask] = 2.0 * float(cfg.loss_background_over_weight) * background_overdose[neg_mask]
    if np.any(pos_mask):
        grad_wrt_dose[pos_mask] = -2.0 * float(cfg.loss_target_under_weight) * target_underdose[pos_mask]

    return loss, grad_wrt_dose


def compute_gradient(proj_matrix: np.ndarray, grad_wrt_dose: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    N = proj_matrix.shape[0]

    if cfg.use_numba and NUMBA_AVAILABLE:
        angles_rad, cos_table, sin_table = get_cached_angle_tables(cfg.angles)
        rows, cols, x_values, y_values, radius_values, base_angle_values = get_cached_pixel_geometry(
            int(N),
            float(cfg.voxel_size_xy),
            float(cfg.container_radius),
        )
        return _compute_gradient_geometry_numba(
            angles_rad,
            cos_table,
            sin_table,
            rows,
            cols,
            x_values,
            y_values,
            radius_values,
            base_angle_values,
            float(cfg.voxel_size_xy),
            float(cfg.container_radius),
            float(cfg.resin_alpha),
            float(cfg.exposure_time_per_angle_ms),
            np.asarray(grad_wrt_dose, dtype=np.float64),
            int(N),
        )

    num_angles = len(cfg.angles)
    grad = np.zeros((N, num_angles), dtype=np.float64)
    center = N // 2
    exposure_time_s = cfg.exposure_time_per_angle_ms / 1000.0

    angles_rad = np.deg2rad(cfg.angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)
    x_1d = (np.arange(N) - center) * cfg.voxel_size_xy
    y_1d = (center - np.arange(N)) * cfg.voxel_size_xy

    for a in range(num_angles):
        for i in range(N):
            y = y_1d[i]
            for j in range(N):
                x = x_1d[j]
                r2 = x * x + y * y
                if r2 > cfg.container_radius ** 2:
                    continue

                r = np.sqrt(r2)
                s_phys = x * cos_table[a] + y * sin_table[a]
                s_pixel = s_phys / cfg.voxel_size_xy + center
                if s_pixel < 0.0 or s_pixel > N - 1:
                    continue

                s_floor = int(np.floor(s_pixel))
                s_ceil = min(s_floor + 1, N - 1)
                w_ceil = s_pixel - s_floor
                w_floor = 1.0 - w_ceil

                base_ang = np.arctan2(y, x)
                ang = base_ang - angles_rad[a]
                depth = cfg.container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * cfg.container_radius)

                atten = np.exp(-cfg.resin_alpha * depth) * exposure_time_s
                contrib = grad_wrt_dose[i, j] * atten
                grad[s_floor, a] += contrib * w_floor
                if s_ceil != s_floor:
                    grad[s_ceil, a] += contrib * w_ceil

    return grad


def _is_better_state(candidate_iou: float, candidate_loss: float,
                     best_iou: float, best_loss: float) -> bool:
    if candidate_iou > best_iou:
        return True
    if np.isclose(candidate_iou, best_iou, rtol=0.0, atol=1e-12) and candidate_loss < best_loss:
        return True
    return False


def optimize_projection_matrix(initial_projection: np.ndarray, target_mask: np.ndarray, cfg: VAMConfig):
    """离散投影矩阵优化主函数。"""
    optimization_mode = str(getattr(cfg, "optimization_mode", "greedy_stop")).strip()
    if optimization_mode not in {"greedy_stop", "explore_best"}:
        raise ValueError(
            f"不支持的 optimization_mode：{optimization_mode}。"
            f"可选值为 'greedy_stop' 或 'explore_best'。"
        )

    proj_input = np.clip(np.asarray(initial_projection, dtype=np.float64), 0.0, None)
    K = int(cfg.projection_quantization_max_index)
    if K < 0:
        raise ValueError("projection_quantization_max_index 必须 >= 0。")

    reference_max_value = float(np.max(proj_input)) if proj_input.size > 0 else 0.0
    if reference_max_value <= 0.0 or K == 0:
        proj_best = np.zeros_like(proj_input, dtype=np.float64)
        dose_best = forward_dose_simulation(proj_best, cfg)
        cured_best = (dose_best >= cfg.resin_Ec).astype(np.float64)
        iou_best = calculate_iou(target_mask, cured_best, cfg=cfg)
        loss_best, _ = compute_loss_and_grad_wrt_dose(dose_best, target_mask, cfg)

        history = {
            "mode": optimization_mode,
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
            "quantization_reference_max_value": float(reference_max_value),
            "quantization_effective_step": 0.0,
        }

    effective_step = reference_max_value / float(K)
    level_current = np.rint(proj_input / effective_step).astype(np.int64)
    level_current = np.clip(level_current, 0, K)
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
        "ious": [float(iou_current)],
        "losses": [float(loss_current)],
        "applied_level_step": [0],
        "changed_entries": [0],
        "accepted": [True],
        "best_ious": [float(iou_best)],
        "best_losses": [float(loss_best)],
    }

    bg_safe_threshold, target_safe_threshold = _resolve_loss_thresholds(cfg)
    print(f"初始离散投影容器内 IoU = {iou_current:.4f}")
    print(
        f"离散迭代模式：{optimization_mode} | "
        f"levels = 0 ... {K} | "
        f"effective step = {effective_step:.6g}"
    )
    print(
        "当前 loss 参数："
        f"background_over_weight = {float(cfg.loss_background_over_weight):.6g}, "
        f"background_over_safe_threshold = {bg_safe_threshold:.6g}, "
        f"target_under_weight = {float(cfg.loss_target_under_weight):.6g}, "
        f"target_under_safe_threshold = {target_safe_threshold:.6g}"
    )

    for it in range(cfg.num_iterations):
        grad = compute_gradient(proj_current, grad_wrt_dose_current, cfg)
        grad_sign = np.sign(grad).astype(np.int8)
        level_candidate = np.clip(level_current - grad_sign.astype(np.int64), 0, K)
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
                print(
                    f"Iter {it + 1:4d}/{cfg.num_iterations} | "
                    "无可更新元素（全部受边界或零梯度限制），提前停止。"
                )
                break

            if (it + 1) % cfg.print_interval == 0 or it == cfg.num_iterations - 1:
                print(
                    f"Iter {it + 1:4d}/{cfg.num_iterations} | "
                    f"IoU(current) = {iou_current:.4f} | "
                    f"loss(current) = {loss_current:.2e} | "
                    f"best IoU = {iou_best:.4f} | "
                    f"best loss@bestIoU = {loss_best:.2e} | "
                    "changed = 0"
                )
            continue

        proj_candidate = level_candidate.astype(np.float64) * effective_step
        dose_candidate = forward_dose_simulation(proj_candidate, cfg)
        loss_candidate, grad_wrt_dose_candidate = compute_loss_and_grad_wrt_dose(dose_candidate, target_mask, cfg)
        cured_candidate = (dose_candidate >= cfg.resin_Ec).astype(np.float64)
        iou_candidate = calculate_iou(target_mask, cured_candidate, cfg=cfg)

        if optimization_mode == "greedy_stop":
            accept = (
                (loss_candidate < loss_current) or
                (
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
                    f"Iter {it + 1:4d}/{cfg.num_iterations} | "
                    "候选离散更新未被接受："
                    f"loss {loss_current:.2e} -> {loss_candidate:.2e}, "
                    f"IoU {iou_current:.4f} -> {iou_candidate:.4f} | "
                    f"changed = {changed_entries} | 提前停止。"
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

        if (it + 1) % cfg.print_interval == 0 or it == cfg.num_iterations - 1:
            if optimization_mode == "greedy_stop":
                print(
                    f"Iter {it + 1:4d}/{cfg.num_iterations} | "
                    f"IoU(discrete) = {iou_current:.4f} | "
                    f"best(discrete) = {iou_best:.4f} | "
                    f"loss = {loss_current:.2e} | "
                    "step = 1 level | "
                    f"changed = {changed_entries}"
                )
            else:
                print(
                    f"Iter {it + 1:4d}/{cfg.num_iterations} | "
                    f"IoU(current) = {iou_current:.4f} | "
                    f"loss(current) = {loss_current:.2e} | "
                    f"best IoU = {iou_best:.4f} | "
                    f"best loss@bestIoU = {loss_best:.2e} | "
                    f"changed = {changed_entries}"
                )

    history["best_iou_discrete"] = float(iou_best)
    history["best_loss_at_best_iou"] = float(loss_best)
    history["final_loss"] = float(loss_best)

    print(f"优化完成，返回结果模式 = {optimization_mode}")
    print(f"返回投影矩阵对应的 IoU = {iou_best:.4f}")
    print(f"返回投影矩阵对应的 loss = {loss_best:.2e}")

    return {
        "projection_matrix": proj_best,
        "dose_field": dose_best,
        "cured_mask": cured_best,
        "iou": float(iou_best),
        "history": history,
        "quantization_reference_max_value": float(reference_max_value),
        "quantization_effective_step": float(effective_step),
    }
