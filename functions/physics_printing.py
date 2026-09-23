from __future__ import annotations

import numpy as np
from skimage.transform import radon

from .config import VAMConfig

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


_ANGLE_TABLE_CACHE: dict[tuple[int, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PIXEL_GEOMETRY_CACHE: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def get_cached_angle_tables(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        raise ValueError("angles must not be empty.")
    key = (
        int(angles.size),
        float(angles[0]),
        float(angles[-1]),
        float(np.sum(angles)),
    )
    cached = _ANGLE_TABLE_CACHE.get(key)
    if cached is None:
        angles_rad = np.deg2rad(angles).astype(np.float64)
        cached = (
            angles_rad,
            np.cos(angles_rad).astype(np.float64),
            np.sin(angles_rad).astype(np.float64),
        )
        _ANGLE_TABLE_CACHE[key] = cached
    return cached


def get_cached_pixel_geometry(
    grid_size: int,
    voxel_size: float,
    container_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_size = int(grid_size)
    voxel_size = float(voxel_size)
    container_radius = float(container_radius)
    key = (grid_size, voxel_size, container_radius)
    cached = _PIXEL_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached

    center = grid_size // 2
    rows_2d, cols_2d = np.indices((grid_size, grid_size), dtype=np.int64)
    x_2d = (cols_2d.astype(np.float64) - float(center)) * voxel_size
    y_2d = (float(center) - rows_2d.astype(np.float64)) * voxel_size
    r_2d = np.sqrt(x_2d * x_2d + y_2d * y_2d)
    mask = r_2d <= container_radius

    rows = np.ascontiguousarray(rows_2d[mask].astype(np.int64))
    cols = np.ascontiguousarray(cols_2d[mask].astype(np.int64))
    x = np.ascontiguousarray(x_2d[mask].astype(np.float64))
    y = np.ascontiguousarray(y_2d[mask].astype(np.float64))
    radius = np.ascontiguousarray(r_2d[mask].astype(np.float64))
    base_angle = np.ascontiguousarray(np.arctan2(y, x).astype(np.float64))

    cached = (rows, cols, x, y, radius, base_angle)
    _PIXEL_GEOMETRY_CACHE[key] = cached
    return cached


def build_container_mask(shape: tuple[int, int], cfg: VAMConfig) -> np.ndarray:
    """生成与当前容器半径一致的圆形 mask。"""
    if len(shape) != 2:
        raise ValueError("shape 必须为二维。")

    rows, cols = int(shape[0]), int(shape[1])
    center_row = rows // 2
    center_col = cols // 2

    yy = (np.arange(rows) - center_row) * cfg.voxel_size_xy
    xx = (np.arange(cols) - center_col) * cfg.voxel_size_xy
    x_grid, y_grid = np.meshgrid(xx, -yy)
    mask = (x_grid ** 2 + y_grid ** 2) <= (cfg.container_radius ** 2)
    return mask


def quantize_projection_matrix(
    projection_matrix: np.ndarray,
    cfg: VAMConfig | None = None,
    *,
    max_index: int | None = None,
    enabled: bool | None = None,
    reference_max_value: float | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool]]:
    """将投影矩阵按当前动态范围离散到 0, T, 2T, ..., KT。

    这里的 T 不再由用户预先定义，而是由当前投影矩阵的参考最大值
    `reference_max_value` 自动决定：T = reference_max_value / K。
    若未显式提供 `reference_max_value`，默认使用当前投影矩阵的最大值。
    这样离散化只控制级数，不额外限制投影矩阵范围。
    """
    proj = np.asarray(projection_matrix, dtype=np.float64)

    if cfg is not None:
        if enabled is None:
            enabled = cfg.projection_quantization_enabled
        if max_index is None:
            max_index = cfg.projection_quantization_max_index

    if enabled is None:
        enabled = True
    if max_index is None:
        raise ValueError("必须提供 max_index 或 cfg。")
    if max_index < 0:
        raise ValueError("projection_quantization_max_index 必须 >= 0。")

    proj = np.clip(proj, 0.0, None)

    if reference_max_value is None:
        reference_max_value = float(np.max(proj)) if proj.size > 0 else 0.0
    reference_max_value = max(float(reference_max_value), 0.0)

    if (not enabled) or max_index == 0 or reference_max_value <= 0.0:
        quantized = proj.copy() if enabled is False else np.zeros_like(proj) if reference_max_value <= 0.0 or max_index == 0 else proj.copy()
        metadata = {
            "enabled": bool(enabled),
            "max_index": int(max_index),
            "reference_max_value": float(reference_max_value),
            "effective_step": 0.0 if max_index == 0 else float(reference_max_value) / float(max_index) if reference_max_value > 0 else 0.0,
        }
        return (quantized, metadata) if return_metadata else quantized

    step = float(reference_max_value) / float(max_index)
    level_index = np.rint(proj / step)
    level_index = np.clip(level_index, 0, max_index)
    quantized = level_index.astype(np.float64) * step

    metadata = {
        "enabled": bool(enabled),
        "max_index": int(max_index),
        "reference_max_value": float(reference_max_value),
        "effective_step": float(step),
    }
    return (quantized, metadata) if return_metadata else quantized


def calculate_iou(
    target: np.ndarray,
    cured: np.ndarray,
    cfg: VAMConfig | None = None,
    container_mask: np.ndarray | None = None,
) -> float:
    """只在容器 mask 内计算目标图与固化结果的 IoU。"""
    target_bool = np.asarray(target).astype(bool)
    cured_bool = np.asarray(cured).astype(bool)

    if container_mask is None and cfg is not None:
        container_mask = build_container_mask(target_bool.shape, cfg)
    if container_mask is not None:
        mask_bool = np.asarray(container_mask).astype(bool)
        target_bool = np.logical_and(target_bool, mask_bool)
        cured_bool = np.logical_and(cured_bool, mask_bool)

    intersection = np.logical_and(target_bool, cured_bool)
    union = np.logical_or(target_bool, cured_bool)
    union_sum = np.sum(union)
    return float(np.sum(intersection) / union_sum) if union_sum > 0 else 0.0


@njit(parallel=True, fastmath=True, cache=True)
def _forward_dose_simulation_numba(projection_matrix, angles, voxel_size, container_radius,
                                   resin_alpha, exposure_time_ms):
    N = projection_matrix.shape[0]
    center = N // 2
    dose_field = np.zeros((N, N), dtype=np.float64)
    exposure_time_s = exposure_time_ms / 1000.0
    angles_rad = np.deg2rad(angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)

    for yy in prange(N):
        y = (center - yy) * voxel_size
        for xx in range(N):
            x = (xx - center) * voxel_size
            r = np.sqrt(x * x + y * y)
            if r > container_radius:
                continue
            dose_val = 0.0
            base_ang = np.arctan2(y, x)
            for i in range(angles.shape[0]):
                cos_t = cos_table[i]
                sin_t = sin_table[i]
                s = (x * cos_t + y * sin_t) / voxel_size + center
                s = min(max(s, 0.0), N - 1.0)
                f = int(np.floor(s))
                c = min(f + 1, N - 1)
                w = s - f
                I = (1.0 - w) * projection_matrix[f, i] + w * projection_matrix[c, i]
                ang = base_ang - angles_rad[i]
                depth = container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * container_radius)
                dose_val += I * np.exp(-resin_alpha * depth)
            dose_field[yy, xx] = dose_val * exposure_time_s
    return dose_field


@njit(parallel=True, fastmath=True, cache=True)
def _forward_dose_simulation_geometry_numba(
    projection_matrix,
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
    output_grid_size,
):
    N_det = projection_matrix.shape[0]
    center_det = N_det // 2
    dose_field = np.zeros((output_grid_size, output_grid_size), dtype=np.float64)
    exposure_time_s = exposure_time_ms / 1000.0

    for p in prange(rows.shape[0]):
        x = x_values[p]
        y = y_values[p]
        radius = radius_values[p]
        base_angle = base_angle_values[p]
        dose_val = 0.0

        for i in range(angles_rad.shape[0]):
            s = (x * cos_table[i] + y * sin_table[i]) / detector_voxel_size + center_det
            if s < 0.0:
                s = 0.0
            elif s > N_det - 1.0:
                s = N_det - 1.0

            f = int(np.floor(s))
            c = min(f + 1, N_det - 1)
            w = s - f
            intensity = (1.0 - w) * projection_matrix[f, i] + w * projection_matrix[c, i]

            angle = base_angle - angles_rad[i]
            depth = container_radius - radius * np.cos(angle)
            depth = min(max(depth, 0.0), 2.0 * container_radius)
            dose_val += intensity * np.exp(-resin_alpha * depth)

        dose_field[rows[p], cols[p]] = dose_val * exposure_time_s

    return dose_field


def _forward_dose_simulation_numpy(projection_matrix, angles, voxel_size, container_radius,
                                   resin_alpha, exposure_time_ms):
    N = projection_matrix.shape[0]
    center = N // 2
    dose_field = np.zeros((N, N), dtype=np.float64)
    exposure_time_s = exposure_time_ms / 1000.0
    angles_rad = np.deg2rad(angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)

    for yy in range(N):
        y = (center - yy) * voxel_size
        for xx in range(N):
            x = (xx - center) * voxel_size
            r = np.sqrt(x * x + y * y)
            if r > container_radius:
                continue
            dose_val = 0.0
            base_ang = np.arctan2(y, x)
            for i in range(len(angles)):
                s = (x * cos_table[i] + y * sin_table[i]) / voxel_size + center
                s = min(max(s, 0.0), N - 1.0)
                f = int(np.floor(s))
                c = min(f + 1, N - 1)
                w = s - f
                I = (1.0 - w) * projection_matrix[f, i] + w * projection_matrix[c, i]
                ang = base_ang - angles_rad[i]
                depth = container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * container_radius)
                dose_val += I * np.exp(-resin_alpha * depth)
            dose_field[yy, xx] = dose_val * exposure_time_s
    return dose_field


def forward_dose_simulation(projection_matrix: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    """真实物理打印的前向剂量仿真函数。"""
    if cfg.use_numba and NUMBA_AVAILABLE:
        angles_rad, cos_table, sin_table = get_cached_angle_tables(cfg.angles)
        rows, cols, x_values, y_values, radius_values, base_angle_values = get_cached_pixel_geometry(
            int(projection_matrix.shape[0]),
            float(cfg.voxel_size_xy),
            float(cfg.container_radius),
        )
        return _forward_dose_simulation_geometry_numba(
            np.asarray(projection_matrix, dtype=np.float64),
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
            int(projection_matrix.shape[0]),
        )
    return _forward_dose_simulation_numpy(
        projection_matrix,
        cfg.angles.astype(np.float64),
        cfg.voxel_size_xy,
        cfg.container_radius,
        cfg.resin_alpha,
        cfg.exposure_time_per_angle_ms,
    )


@njit(parallel=True, fastmath=True, cache=True)
def _forward_dose_simulation_resampled_numba(projection_matrix, angles, detector_voxel_size,
                                             vis_voxel_size, container_radius,
                                             resin_alpha, exposure_time_ms,
                                             output_grid_size):
    N_det = projection_matrix.shape[0]
    center_det = N_det // 2
    M = int(output_grid_size)
    center_vis = M // 2
    dose_field = np.zeros((M, M), dtype=np.float64)
    exposure_time_s = exposure_time_ms / 1000.0
    angles_rad = np.deg2rad(angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)

    for yy in prange(M):
        y = (center_vis - yy) * vis_voxel_size
        for xx in range(M):
            x = (xx - center_vis) * vis_voxel_size
            r = np.sqrt(x * x + y * y)
            if r > container_radius:
                continue
            dose_val = 0.0
            base_ang = np.arctan2(y, x)
            for i in range(angles.shape[0]):
                cos_t = cos_table[i]
                sin_t = sin_table[i]
                s = (x * cos_t + y * sin_t) / detector_voxel_size + center_det
                s = min(max(s, 0.0), N_det - 1.0)
                f = int(np.floor(s))
                c = min(f + 1, N_det - 1)
                w = s - f
                I = (1.0 - w) * projection_matrix[f, i] + w * projection_matrix[c, i]
                ang = base_ang - angles_rad[i]
                depth = container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * container_radius)
                dose_val += I * np.exp(-resin_alpha * depth)
            dose_field[yy, xx] = dose_val * exposure_time_s
    return dose_field


def _forward_dose_simulation_resampled_numpy(projection_matrix, angles, detector_voxel_size,
                                             vis_voxel_size, container_radius,
                                             resin_alpha, exposure_time_ms,
                                             output_grid_size):
    N_det = projection_matrix.shape[0]
    center_det = N_det // 2
    M = int(output_grid_size)
    center_vis = M // 2
    dose_field = np.zeros((M, M), dtype=np.float64)
    exposure_time_s = exposure_time_ms / 1000.0
    angles_rad = np.deg2rad(angles)
    cos_table = np.cos(angles_rad)
    sin_table = np.sin(angles_rad)

    for yy in range(M):
        y = (center_vis - yy) * vis_voxel_size
        for xx in range(M):
            x = (xx - center_vis) * vis_voxel_size
            r = np.sqrt(x * x + y * y)
            if r > container_radius:
                continue
            dose_val = 0.0
            base_ang = np.arctan2(y, x)
            for i in range(len(angles)):
                s = (x * cos_table[i] + y * sin_table[i]) / detector_voxel_size + center_det
                s = min(max(s, 0.0), N_det - 1.0)
                f = int(np.floor(s))
                c = min(f + 1, N_det - 1)
                w = s - f
                I = (1.0 - w) * projection_matrix[f, i] + w * projection_matrix[c, i]
                ang = base_ang - angles_rad[i]
                depth = container_radius - r * np.cos(ang)
                depth = min(max(depth, 0.0), 2.0 * container_radius)
                dose_val += I * np.exp(-resin_alpha * depth)
            dose_field[yy, xx] = dose_val * exposure_time_s
    return dose_field


def forward_dose_simulation_visualization(
    projection_matrix: np.ndarray,
    cfg: VAMConfig,
    output_grid_size: int | None = None,
) -> np.ndarray:
    """仅用于可视化的高分辨率前向剂量仿真。

    与 ``forward_dose_simulation`` 的区别是：
    - 投影矩阵仍保持原 detector 采样与物理尺寸；
    - 仅将输出空间网格细化为 ``output_grid_size × output_grid_size``；
    - 物理视场大小保持不变，因此空间采样步长会按网格倍率缩小；
    - 不参与优化、IoU、cured、loss 或梯度，只用于 dose 可视化与 profile 分析。
    """
    if output_grid_size is None:
        return forward_dose_simulation(projection_matrix, cfg)

    output_grid_size = int(output_grid_size)
    if output_grid_size <= 0:
        raise ValueError('output_grid_size 必须为正整数。')

    detector_grid_size = int(projection_matrix.shape[0])
    if output_grid_size == detector_grid_size:
        return forward_dose_simulation(projection_matrix, cfg)

    vis_voxel_size = cfg.voxel_size_xy * float(detector_grid_size) / float(output_grid_size)

    if cfg.use_numba and NUMBA_AVAILABLE:
        return _forward_dose_simulation_resampled_numba(
            np.asarray(projection_matrix, dtype=np.float64),
            cfg.angles.astype(np.float64),
            cfg.voxel_size_xy,
            vis_voxel_size,
            cfg.container_radius,
            cfg.resin_alpha,
            cfg.exposure_time_per_angle_ms,
            output_grid_size,
        )
    return _forward_dose_simulation_resampled_numpy(
        np.asarray(projection_matrix, dtype=np.float64),
        cfg.angles.astype(np.float64),
        cfg.voxel_size_xy,
        vis_voxel_size,
        cfg.container_radius,
        cfg.resin_alpha,
        cfg.exposure_time_per_angle_ms,
        output_grid_size,
    )


def solve_optimal_global_scale(base_dose: np.ndarray, target_mask: np.ndarray, resin_Ec: float, cfg: VAMConfig) -> float:
    """通过最大化容器内 IoU 找到最优全局缩放系数 k。"""
    if np.sum(target_mask) == 0:
        raise RuntimeError("目标掩膜为空，无法求解 k。")

    target_vals = base_dose[target_mask > 0]
    if target_vals.size == 0:
        print("警告：目标区域内无剂量值，使用默认 k = 1")
        return 1.0

    max_dose = np.max(target_vals)
    min_dose = np.min(target_vals)
    positive_vals = target_vals[target_vals > 0]
    min_positive = np.min(positive_vals) if positive_vals.size > 0 else None

    print(
        "目标区域剂量统计: "
        f"最小值={min_dose:.3e}, 最大值={max_dose:.3e}, "
        f"非零最小值={min_positive if min_positive is not None else '无'}"
    )

    if max_dose <= 0:
        print("警告：目标区域内剂量全为 0，无法通过缩放固化，返回 k = 1")
        return 1.0

    k_low = resin_Ec / max_dose
    if min_positive is not None and min_positive > 0:
        k_high = resin_Ec / min_positive
    else:
        k_high = resin_Ec / max_dose * 100.0

    k_low = max(1e-6, k_low * 0.1)
    k_high = min(1e6, k_high * 10.0)
    if k_low >= k_high:
        k_low, k_high = 1e-3, 1e5

    print(f"搜索范围: [{k_low:.3e}, {k_high:.3e}]")

    log_k_vals = np.linspace(np.log10(k_low), np.log10(k_high), 200)
    k_vals = 10 ** log_k_vals

    best_iou = -1.0
    best_k = 1.0
    for k in k_vals:
        cured = (base_dose * k >= resin_Ec).astype(np.float64)
        iou = calculate_iou(target_mask, cured, cfg=cfg)
        if iou > best_iou:
            best_iou = iou
            best_k = k

    print(f"最佳 k = {best_k:.3e}，对应容器内 IoU = {best_iou:.4f}")
    return float(best_k)


def compensate_and_print_initial_projection(raw_projection: np.ndarray, target_mask: np.ndarray,
                                           cfg: VAMConfig):
    """对初始投影做经验补偿，并保证 proto 阶段前后都保持离散强度。"""
    target_proj = radon(target_mask, theta=cfg.angles, circle=True, preserve_range=True)
    target_proj = target_proj.astype(np.float64)
    target_proj /= (np.max(target_proj) + 1e-12)

    weight = cfg.min_weight + (1.0 - cfg.min_weight) * target_proj ** cfg.weight_power

    # 严格实现 proto 阶段离散：
    # 1) raw_projection 先经过权重补偿形成连续的 base_projection_continuous；
    # 2) 在进入第一次前向剂量仿真之前，先对 base_projection 做一次统一量化；
    # 3) 再基于这个离散 base_projection 求全局缩放系数 global_k；
    # 4) 缩放后得到的 final_projection_continuous 再做一次统一量化，
    #    使最终 proto_projection 也保持在 0, T, 2T, ..., KT 的离散等级上。
    #
    # 这样可保证：
    # raw -> proto 的补偿链路中，真正参与打印求解与最终输出的矩阵都经过离散化，
    # 而不是只在最后一步离散。
    base_projection_continuous = raw_projection * weight
    base_projection, _ = quantize_projection_matrix(
        base_projection_continuous, cfg, return_metadata=True
    )

    base_dose = forward_dose_simulation(base_projection, cfg)
    global_k = solve_optimal_global_scale(base_dose, target_mask, cfg.resin_Ec, cfg)

    final_projection_continuous = base_projection * global_k
    final_projection, quant_meta = quantize_projection_matrix(
        final_projection_continuous, cfg, return_metadata=True
    )
    final_dose = forward_dose_simulation(final_projection, cfg)
    cured = (final_dose >= cfg.resin_Ec).astype(np.float64)
    iou = calculate_iou(target_mask, cured, cfg=cfg)

    return {
        "projection_matrix": final_projection,
        "projection_matrix_continuous": final_projection_continuous,
        "dose_field": final_dose,
        "cured_mask": cured,
        "iou": iou,
        "global_k": global_k,
        "weight_map": weight,
        "quantization_reference_max_value": quant_meta["reference_max_value"],
        "quantization_effective_step": quant_meta["effective_step"],
    }


def print_any_projection(
    projection_matrix: np.ndarray,
    target_mask: np.ndarray,
    cfg: VAMConfig,
    quantization_reference_max_value: float | None = None,
):
    """对任意一个投影矩阵执行最终量化后的真实打印仿真。"""
    quantized_projection, quant_meta = quantize_projection_matrix(
        projection_matrix, cfg,
        reference_max_value=quantization_reference_max_value,
        return_metadata=True,
    )
    dose = forward_dose_simulation(quantized_projection, cfg)
    cured = (dose >= cfg.resin_Ec).astype(np.float64)
    iou = calculate_iou(target_mask, cured, cfg=cfg)
    return {
        "projection_matrix": quantized_projection,
        "dose_field": dose,
        "cured_mask": cured,
        "iou": iou,
        "quantization_reference_max_value": quant_meta["reference_max_value"],
        "quantization_effective_step": quant_meta["effective_step"],
    }
