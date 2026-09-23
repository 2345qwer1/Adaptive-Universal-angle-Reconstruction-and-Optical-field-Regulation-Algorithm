
from __future__ import annotations

"""
位置分析脚本（支持模式1 / 模式2）。

模式1：
- 每个位置单独生成一个 target
- 每个位置分别打印一次
- 生成每个位置的 figure1 / figure2
- 再汇总生成 summary_fig1 / 2 / 3 / 6

模式2：
- 所有位置的同类结构先合并成一个总 target
- 只打印一次 proto / optimized
- 生成一次总的 figure1 / figure2
- 但 summary_fig1 / 2 / 3 / 6 仍按“每个位置对应一个目标结构”分别统计

重要约定：
1. figure2 中：
   - 白圈 / 白点：target 轮廓与几何重心
   - 绿圈 / 绿点：dose > 0.8 * Ec 的匹配连通域轮廓与几何重心
   - 绿点不是 dose 加权中心，而是绿圈区域的几何中心
2. 模式1和模式2统一按“最近重心”匹配白圈与绿圈
3. figure2 中绿色只能是细绿线，不允许绿色填充
4. rect 的白圈改为基于几何参数直接绘制，避免轮廓出现边中心突出 1 px 的显示问题
5. dose80 中的阈值比例由 DOSE80_THRESHOLD_RATIO 控制，可随时调节
"""

from pathlib import Path
import sys
import copy
import csv
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from skimage.measure import find_contours, label, regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions import (  # noqa: E402
    VAMConfig,
    get_default_config,
    ensure_output_dirs,
    generate_initial_sinogram,
    generate_initial_projection_matrix,
    compensate_and_print_initial_projection,
    optimize_projection_matrix,
    build_container_mask,
    calculate_iou,
    save_projection_npz,
)
from functions.physics_printing import forward_dose_simulation_visualization  # noqa: E402


# ============================================================
# 脚本层可修改参数
# ============================================================
ANALYSIS_NAME = "position_analysis"
ANALYSIS_OUTPUT_ROOT = "F:/USTC/项目/VAM/411test"
# = None 时使用 Path(cfg.output_dir) / "position_analysis" / ANALYSIS_NAME
# = 字符串路径时直接保存到该目录

ANALYSIS_MODE = "mode1"
# "mode1" : 每个位置分别打印
# "mode2" : 所有位置合并为一个总 target，只打印一次

RUN_MODE = "both"                  # "proto" / "optimized" / "both"
STRUCTURE_TYPE = "disk"           # "point" / "disk" / "rect"
SCAN_MODE = "radial_scan"          # "radial_scan" / "angular_scan"

POINT_RADIUS_PX = 0
DISK_RADIUS_PX = 50
RECT_WIDTH_PX = 21
RECT_HEIGHT_PX = 21

PHI_DEG_FIXED = 0
R_RATIO_LIST = [0.0, 0.9]

R_RATIO_FIXED = 0.5
PHI_DEG_LIST = [0, 45, 90, 135]

USE_FINE_DOSE_VIS = False
FINE_DOSE_GRID_SIZE = 2160

SHOW_FIGURES_OVERRIDE = None
SAVE_FIGURES = True
SAVE_RESULTS = False
# False 时只生成 figure，不保存 csv / json / npy / npz / config_snapshot 等非图像输出

DOSE80_THRESHOLD_RATIO = 0.8
# 绿色轮廓区域定义为：dose > DOSE80_THRESHOLD_RATIO * Ec
# 例如 0.8 表示使用固化阈值的 80%

SHOW_WHITE_CENTROIDS = False
# 是否显示白点（target 几何重心）
# True  : 在 figure2 / figure6 中显示白点
# False : 不显示白点

SHOW_GREEN_CENTROIDS = False
# 是否显示绿点（dose > 0.8Ec 匹配区域的几何重心）
# True  : 在 figure2 / figure6 中显示绿点
# False : 不显示绿点

CFG_N_SLICE = None
CFG_ANGLE_NUM = None
CFG_RESIN_ALPHA = None
CFG_RESIN_EC = None
CFG_NUM_ITERATIONS = None
CFG_USE_NUMBA = None


# ============================================================
# 基础工具函数
# ============================================================

def _ensure_binary(arr: np.ndarray) -> np.ndarray:
    return (np.asarray(arr) > 0).astype(np.uint8)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _format_r_ratio(r_ratio: float) -> str:
    return f"{float(r_ratio):.1f}"


def _format_phi_deg(phi_deg: float) -> str:
    phi_float = float(phi_deg)
    if np.isclose(phi_float, round(phi_float)):
        return str(int(round(phi_float)))
    return f"{phi_float:g}"


def _position_id(r_ratio: float, phi_deg: float) -> str:
    return f"position_r{_format_r_ratio(r_ratio)}_phi{_format_phi_deg(phi_deg)}"


def _compute_centroid(binary_img: np.ndarray) -> Tuple[float, float]:
    coords = np.argwhere(_ensure_binary(binary_img) > 0)
    if coords.size == 0:
        return np.nan, np.nan
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _nearest_active_index(binary_profile: np.ndarray, center_idx: int) -> Optional[int]:
    active = np.flatnonzero(binary_profile > 0)
    if active.size == 0:
        return None
    return int(active[np.argmin(np.abs(active - center_idx))])


def _contiguous_binary_width(profile: np.ndarray, center_idx: int) -> float:
    p = _ensure_binary(profile).astype(bool).ravel()
    n = p.size
    if n == 0 or not p.any():
        return np.nan

    center_idx = int(np.clip(center_idx, 0, n - 1))
    if not p[center_idx]:
        nearest = _nearest_active_index(p.astype(np.uint8), center_idx)
        if nearest is None:
            return np.nan
        center_idx = nearest

    left = center_idx
    while left - 1 >= 0 and p[left - 1]:
        left -= 1
    right = center_idx
    while right + 1 < n and p[right + 1]:
        right += 1
    return float(right - left + 1)


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    k = max(1, int(np.ceil(vals.size * float(fraction))))
    sorted_vals = np.sort(vals)
    return float(np.mean(sorted_vals[-k:]))


def _bottom_fraction_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    k = max(1, int(np.ceil(vals.size * float(fraction))))
    sorted_vals = np.sort(vals)
    return float(np.mean(sorted_vals[:k]))


def _extract_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
    binary = _ensure_binary(binary_mask)
    if np.sum(binary) == 0:
        return []
    return find_contours(binary.astype(float), level=0.5)


def _scale_rc(
    rc: Tuple[float, float],
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> Tuple[float, float]:
    from_h, from_w = int(from_shape[0]), int(from_shape[1])
    to_h, to_w = int(to_shape[0]), int(to_shape[1])
    return float(rc[0]) * to_h / from_h, float(rc[1]) * to_w / from_w


def _scale_contours(
    contours: Sequence[np.ndarray],
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> List[np.ndarray]:
    from_h, from_w = int(from_shape[0]), int(from_shape[1])
    to_h, to_w = int(to_shape[0]), int(to_shape[1])
    row_scale = float(to_h) / float(from_h)
    col_scale = float(to_w) / float(from_w)
    out: List[np.ndarray] = []
    for contour in contours:
        contour = np.asarray(contour, dtype=float)
        scaled = contour.copy()
        scaled[:, 0] *= row_scale
        scaled[:, 1] *= col_scale
        out.append(scaled)
    return out


def _finite_max(*groups: Sequence[float], default: float = 1.0) -> float:
    vals: List[float] = []
    for group in groups:
        arr = np.asarray(group, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.extend(arr.tolist())
    if not vals:
        return float(default)
    vmax = max(vals)
    return float(vmax if vmax > 0 else default)


def _finite_min(*groups: Sequence[float], default: float = 0.0) -> float:
    vals: List[float] = []
    for group in groups:
        arr = np.asarray(group, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.extend(arr.tolist())
    if not vals:
        return float(default)
    return float(min(vals))


def _format_value_for_annotation(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.3g}"


def _annotate_bar_values(ax, bars) -> None:
    ymin, ymax = ax.get_ylim()
    offset = 0.015 * (ymax - ymin if ymax > ymin else 1.0)
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            _format_value_for_annotation(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _branch_has_any_finite(summary_rows: List[Dict[str, Any]], keys: Sequence[str]) -> bool:
    for row in summary_rows:
        for key in keys:
            value = row.get(key, np.nan)
            if np.isfinite(float(value)):
                return True
    return False


def _short_position_label(position_id: str) -> str:
    return str(position_id).replace("position_", "")


def _configure_pretty_axes(ax) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_placeholder_panel(ax, title: str, message: str) -> None:
    _configure_pretty_axes(ax)
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")


def _imshow_binary_panel(ax, binary_img: np.ndarray, title: str) -> None:
    _configure_pretty_axes(ax)
    ax.imshow(_ensure_binary(binary_img), cmap="gray", origin="upper", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)


def _imshow_error_panel(ax, error_map: np.ndarray, title: str) -> None:
    _configure_pretty_axes(ax)
    ax.imshow(error_map, cmap=ERROR_CMAP, origin="upper", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title(title)


def _draw_contours(ax, contours: Sequence[np.ndarray], color: str, linewidth: float = 0.25) -> None:
    for contour in contours:
        contour = np.asarray(contour, dtype=float)
        if contour.size == 0:
            continue
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def _annotate_metric_text(ax, text: str):
    ax.text(0.5, -0.10, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _container_radius_px(cfg: VAMConfig) -> float:
    return float(cfg.container_radius / cfg.voxel_size_xy)


def polar_to_center_rc(cfg: VAMConfig, r_ratio: float, phi_deg: float) -> Tuple[int, int]:
    radius_px = _container_radius_px(cfg)
    r_px = float(r_ratio) * radius_px
    phi_rad = np.deg2rad(float(phi_deg))
    center_row = cfg.n_slice // 2
    center_col = cfg.n_slice // 2
    row = int(round(center_row - r_px * np.sin(phi_rad)))
    col = int(round(center_col + r_px * np.cos(phi_rad)))
    return row, col


def _validate_runtime_parameters() -> None:
    global RUN_MODE, STRUCTURE_TYPE, SCAN_MODE, ANALYSIS_MODE
    RUN_MODE = str(RUN_MODE).strip().lower()
    STRUCTURE_TYPE = str(STRUCTURE_TYPE).strip().lower()
    SCAN_MODE = str(SCAN_MODE).strip().lower()
    ANALYSIS_MODE = str(ANALYSIS_MODE).strip().lower()

    if ANALYSIS_MODE not in {"mode1", "mode2"}:
        raise ValueError("ANALYSIS_MODE must be one of {'mode1', 'mode2'}.")
    if RUN_MODE not in {"proto", "optimized", "both"}:
        raise ValueError("RUN_MODE must be one of {'proto', 'optimized', 'both'}.")
    if STRUCTURE_TYPE not in {"point", "disk", "rect"}:
        raise ValueError("STRUCTURE_TYPE must be one of {'point', 'disk', 'rect'}.")
    if SCAN_MODE not in {"radial_scan", "angular_scan"}:
        raise ValueError("SCAN_MODE must be one of {'radial_scan', 'angular_scan'}.")

    if SCAN_MODE == "radial_scan":
        if not R_RATIO_LIST:
            raise ValueError("R_RATIO_LIST 不能为空。")
        for r_ratio in R_RATIO_LIST:
            if float(r_ratio) < 0.0 or float(r_ratio) >= 1.0:
                raise ValueError(f"radial_scan 中 r_ratio 必须满足 0 <= r_ratio < 1，当前值={r_ratio}")
    else:
        if float(R_RATIO_FIXED) < 0.0 or float(R_RATIO_FIXED) >= 1.0:
            raise ValueError(f"R_RATIO_FIXED 必须满足 0 <= r_ratio < 1，当前值={R_RATIO_FIXED}")
        if not PHI_DEG_LIST:
            raise ValueError("PHI_DEG_LIST 不能为空。")

    if POINT_RADIUS_PX < 0:
        raise ValueError("POINT_RADIUS_PX 必须 >= 0。")
    if STRUCTURE_TYPE == "disk" and DISK_RADIUS_PX <= 0:
        raise ValueError("当 STRUCTURE_TYPE='disk' 时，DISK_RADIUS_PX 必须 > 0。")
    if STRUCTURE_TYPE == "rect" and (RECT_WIDTH_PX <= 0 or RECT_HEIGHT_PX <= 0):
        raise ValueError("当 STRUCTURE_TYPE='rect' 时，RECT_WIDTH_PX 和 RECT_HEIGHT_PX 必须 > 0。")

    if USE_FINE_DOSE_VIS and int(FINE_DOSE_GRID_SIZE) <= 0:
        raise ValueError("FINE_DOSE_GRID_SIZE 必须为正整数。")
    if not (0.0 < float(DOSE80_THRESHOLD_RATIO) <= 1.0):
        raise ValueError("DOSE80_THRESHOLD_RATIO 必须满足 0 < ratio <= 1。")


# ============================================================
# target 构造与几何轮廓
# ============================================================

def _check_target_extent_in_image(shape: Tuple[int, int], center_rc: Tuple[int, int], structure_type: str) -> None:
    h, w = int(shape[0]), int(shape[1])
    row_c, col_c = int(center_rc[0]), int(center_rc[1])
    structure_type = str(structure_type).strip().lower()

    if structure_type == "point":
        radius = int(POINT_RADIUS_PX)
        row_min = row_c - radius
        row_max = row_c + radius
        col_min = col_c - radius
        col_max = col_c + radius
    elif structure_type == "disk":
        radius = int(DISK_RADIUS_PX)
        row_min = row_c - radius
        row_max = row_c + radius
        col_min = col_c - radius
        col_max = col_c + radius
    elif structure_type == "rect":
        half_h_low = RECT_HEIGHT_PX // 2
        half_h_high = RECT_HEIGHT_PX - half_h_low
        half_w_low = RECT_WIDTH_PX // 2
        half_w_high = RECT_WIDTH_PX - half_w_low
        row_min = row_c - half_h_low
        row_max = row_c + half_h_high - 1
        col_min = col_c - half_w_low
        col_max = col_c + half_w_high - 1
    else:
        raise ValueError(f"Unsupported STRUCTURE_TYPE: {structure_type}")

    if row_min < 0 or row_max >= h or col_min < 0 or col_max >= w:
        raise ValueError(
            f"Target exceeds image boundary: center_rc={center_rc}, structure_type={structure_type}, "
            f"bbox_rows=[{row_min}, {row_max}], bbox_cols=[{col_min}, {col_max}], image_shape={shape}"
        )


def build_point_target(shape: Tuple[int, int], center_rc: Tuple[int, int], point_radius_px: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    row_c, col_c = int(center_rc[0]), int(center_rc[1])
    target = np.zeros((h, w), dtype=np.uint8)
    if point_radius_px == 0:
        target[row_c, col_c] = 1
        return target
    rr, cc = np.ogrid[:h, :w]
    mask = (rr - row_c) ** 2 + (cc - col_c) ** 2 <= int(point_radius_px) ** 2
    target[mask] = 1
    return target


def build_disk_target(shape: Tuple[int, int], center_rc: Tuple[int, int], disk_radius_px: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    rr, cc = np.ogrid[:h, :w]
    mask = (rr - int(center_rc[0])) ** 2 + (cc - int(center_rc[1])) ** 2 <= int(disk_radius_px) ** 2
    target = np.zeros((h, w), dtype=np.uint8)
    target[mask] = 1
    return target


def build_rect_target(shape: Tuple[int, int], center_rc: Tuple[int, int], rect_width_px: int, rect_height_px: int) -> np.ndarray:
    """
    注意：
    这里直接采用标准轴对齐矩形填充，不再用任何会引入“边中心多出 1 px”的近似轮廓构造。
    """
    h, w = int(shape[0]), int(shape[1])
    row_c, col_c = int(center_rc[0]), int(center_rc[1])

    half_h_low = rect_height_px // 2
    half_h_high = rect_height_px - half_h_low
    half_w_low = rect_width_px // 2
    half_w_high = rect_width_px - half_w_low

    row_start = row_c - half_h_low
    row_end = row_c + half_h_high
    col_start = col_c - half_w_low
    col_end = col_c + half_w_high

    if row_start < 0 or row_end > h or col_start < 0 or col_end > w:
        raise ValueError("Rect target exceeds image boundary before container check.")

    target = np.zeros((h, w), dtype=np.uint8)
    target[row_start:row_end, col_start:col_end] = 1
    return target


def _circle_contour(center_rc: Tuple[float, float], radius: float, n: int = 181) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    row = float(center_rc[0]) - radius * np.sin(theta)
    col = float(center_rc[1]) + radius * np.cos(theta)
    return np.stack([row, col], axis=1)


def _square_pixel_contour(center_rc: Tuple[float, float]) -> np.ndarray:
    r = float(center_rc[0])
    c = float(center_rc[1])
    return np.array(
        [
            [r - 0.5, c - 0.5],
            [r - 0.5, c + 0.5],
            [r + 0.5, c + 0.5],
            [r + 0.5, c - 0.5],
            [r - 0.5, c - 0.5],
        ],
        dtype=float,
    )


def _rect_contour(center_rc: Tuple[float, float], rect_width_px: int, rect_height_px: int) -> np.ndarray:
    """
    基于几何参数直接生成矩形轮廓，避免 find_contours 在某些奇偶宽高下产生轮廓视觉伪突起。
    """
    row_c, col_c = float(center_rc[0]), float(center_rc[1])

    half_h_low = rect_height_px // 2
    half_h_high = rect_height_px - half_h_low
    half_w_low = rect_width_px // 2
    half_w_high = rect_width_px - half_w_low

    top = row_c - half_h_low - 0.5
    bottom = row_c + half_h_high - 0.5
    left = col_c - half_w_low - 0.5
    right = col_c + half_w_high - 0.5

    return np.array(
        [
            [top, left],
            [top, right],
            [bottom, right],
            [bottom, left],
            [top, left],
        ],
        dtype=float,
    )


def _build_target_geometry_contours(structure_type: str, center_rc: Tuple[int, int]) -> List[np.ndarray]:
    structure_type = str(structure_type).strip().lower()
    if structure_type == "point":
        if int(POINT_RADIUS_PX) == 0:
            return [_square_pixel_contour(center_rc)]
        return [_circle_contour(center_rc, float(POINT_RADIUS_PX) + 0.5)]
    if structure_type == "disk":
        return [_circle_contour(center_rc, float(DISK_RADIUS_PX) + 0.5)]
    if structure_type == "rect":
        return [_rect_contour(center_rc, int(RECT_WIDTH_PX), int(RECT_HEIGHT_PX))]
    raise ValueError(f"Unsupported STRUCTURE_TYPE: {structure_type}")


def build_single_target_item(
    cfg: VAMConfig,
    structure_type: str,
    center_rc: Tuple[int, int],
    container_mask: np.ndarray,
    position_info: Dict[str, Any],
) -> Dict[str, Any]:
    shape = (cfg.n_slice, cfg.n_slice)
    _check_target_extent_in_image(shape, center_rc, structure_type)

    if structure_type == "point":
        target_mask = build_point_target(shape, center_rc, POINT_RADIUS_PX)
    elif structure_type == "disk":
        target_mask = build_disk_target(shape, center_rc, DISK_RADIUS_PX)
    elif structure_type == "rect":
        target_mask = build_rect_target(shape, center_rc, RECT_WIDTH_PX, RECT_HEIGHT_PX)
    else:
        raise ValueError(f"Unsupported STRUCTURE_TYPE: {structure_type}")

    if np.any(np.logical_and(target_mask > 0, np.logical_not(container_mask.astype(bool)))):
        raise ValueError(
            f"Target exceeds container mask: center_rc={center_rc}, structure_type={structure_type}, "
            f"position_id={position_info['position_id']}"
        )

    target_mask = np.logical_and(target_mask > 0, container_mask.astype(bool)).astype(np.uint8)
    if np.sum(target_mask) == 0:
        raise ValueError(f"Generated empty target at center_rc={center_rc}")

    return {
        "position_id": position_info["position_id"],
        "position_info": copy.deepcopy(position_info),
        "structure_type": structure_type,
        "center_row": int(center_rc[0]),
        "center_col": int(center_rc[1]),
        "target_mask": target_mask,
        "target_centroid_row": float(center_rc[0]),
        "target_centroid_col": float(center_rc[1]),
        "target_contours": _build_target_geometry_contours(structure_type, center_rc),
    }


def generate_scan_positions(cfg: VAMConfig) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    if SCAN_MODE == "radial_scan":
        for r_ratio in R_RATIO_LIST:
            phi_deg = float(PHI_DEG_FIXED)
            center_rc = polar_to_center_rc(cfg, r_ratio=float(r_ratio), phi_deg=phi_deg)
            positions.append(
                {
                    "position_id": _position_id(float(r_ratio), phi_deg),
                    "scan_mode": SCAN_MODE,
                    "r_ratio": float(r_ratio),
                    "phi_deg": phi_deg,
                    "center_row": int(center_rc[0]),
                    "center_col": int(center_rc[1]),
                }
            )
    else:
        for phi_deg in PHI_DEG_LIST:
            r_ratio = float(R_RATIO_FIXED)
            center_rc = polar_to_center_rc(cfg, r_ratio=r_ratio, phi_deg=float(phi_deg))
            positions.append(
                {
                    "position_id": _position_id(r_ratio, float(phi_deg)),
                    "scan_mode": SCAN_MODE,
                    "r_ratio": r_ratio,
                    "phi_deg": float(phi_deg),
                    "center_row": int(center_rc[0]),
                    "center_col": int(center_rc[1]),
                }
            )
    return positions


def build_target_items(cfg: VAMConfig, container_mask: np.ndarray, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for position_info in positions:
        center_rc = (int(position_info["center_row"]), int(position_info["center_col"]))
        item = build_single_target_item(cfg, STRUCTURE_TYPE, center_rc, container_mask, position_info)
        items.append(item)

    if ANALYSIS_MODE == "mode2":
        union = np.zeros((cfg.n_slice, cfg.n_slice), dtype=np.uint8)
        for item in items:
            overlap = np.logical_and(union > 0, item["target_mask"] > 0)
            if np.any(overlap):
                raise ValueError(
                    f"Mode2 下目标结构彼此重叠，当前不允许重叠。出错位置：{item['position_id']}"
                )
            union = np.maximum(union, item["target_mask"])
    return items


def build_combined_target_mask(target_items: List[Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    target = np.zeros(shape, dtype=np.uint8)
    for item in target_items:
        target = np.maximum(target, _ensure_binary(item["target_mask"]))
    return target


# ============================================================
# dose80 连通域配对与分目标指标
# ============================================================

def _build_connected_components(binary_mask: np.ndarray) -> List[Dict[str, Any]]:
    binary = _ensure_binary(binary_mask).astype(bool)
    if not np.any(binary):
        return []
    labeled = label(binary, connectivity=2)
    props = regionprops(labeled)
    components: List[Dict[str, Any]] = []
    for prop in props:
        component_mask = (labeled == prop.label).astype(np.uint8)
        components.append(
            {
                "label_id": int(prop.label),
                "mask": component_mask,
                "centroid_row": float(prop.centroid[0]),
                "centroid_col": float(prop.centroid[1]),
                "area_px": int(prop.area),
                "contours": _extract_contours(component_mask),
            }
        )
    return components


def _match_components_greedy(
    target_items: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
) -> List[Optional[int]]:
    """
    先做唯一 greedy 最近匹配；
    若绿色连通域数量少于目标数，则剩余目标允许复用最近绿色连通域。
    """
    n_targets = len(target_items)
    n_comps = len(components)
    if n_targets == 0:
        return []
    if n_comps == 0:
        return [None] * n_targets

    dist = np.full((n_targets, n_comps), np.inf, dtype=float)
    for i, item in enumerate(target_items):
        tr = float(item["target_centroid_row"])
        tc = float(item["target_centroid_col"])
        for j, comp in enumerate(components):
            dist[i, j] = float(np.hypot(comp["centroid_row"] - tr, comp["centroid_col"] - tc))

    matched: List[Optional[int]] = [None] * n_targets
    remaining_targets = set(range(n_targets))
    remaining_comps = set(range(n_comps))

    while remaining_targets and remaining_comps:
        best_pair = None
        best_dist = np.inf
        for i in remaining_targets:
            for j in remaining_comps:
                if dist[i, j] < best_dist:
                    best_dist = dist[i, j]
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        matched[i] = j
        remaining_targets.remove(i)
        remaining_comps.remove(j)

    for i in range(n_targets):
        if matched[i] is None:
            j = int(np.argmin(dist[i, :]))
            matched[i] = j

    return matched


def _build_local_background_mask_mode2(
    item: Dict[str, Any],
    all_target_mask: np.ndarray,
    container_mask: np.ndarray,
) -> np.ndarray:
    """
    模式2下，为每个目标构造局部 background：
    - 取该目标 bbox 外扩一个与目标尺寸相关的边界
    - 在该 ROI 内、容器内、且不属于任何 target 的区域，作为该目标的 local background
    """
    target = _ensure_binary(item["target_mask"]).astype(bool)
    coords = np.argwhere(target)
    if coords.size == 0:
        return np.zeros_like(target, dtype=np.uint8)

    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    h = int(rmax - rmin + 1)
    w = int(cmax - cmin + 1)
    pad = max(8, 2 * max(h, w))

    H, W = target.shape
    rr0 = max(0, int(rmin) - pad)
    rr1 = min(H, int(rmax) + pad + 1)
    cc0 = max(0, int(cmin) - pad)
    cc1 = min(W, int(cmax) + pad + 1)

    roi = np.zeros_like(target, dtype=bool)
    roi[rr0:rr1, cc0:cc1] = True

    bg = np.logical_and(roi, _ensure_binary(container_mask).astype(bool))
    bg = np.logical_and(bg, np.logical_not(_ensure_binary(all_target_mask).astype(bool)))
    return bg.astype(np.uint8)


def _compute_single_item_metrics(
    item: Dict[str, Any],
    dose_field: np.ndarray,
    cured_mask: np.ndarray,
    cfg: VAMConfig,
    container_mask: np.ndarray,
    all_target_mask: Optional[np.ndarray] = None,
    mode2_local_background: bool = False,
) -> Dict[str, Any]:
    target_bin = _ensure_binary(item["target_mask"]).astype(bool)
    cured_bin = _ensure_binary(cured_mask).astype(bool)
    container_bool = _ensure_binary(container_mask).astype(bool)

    target_in_container = np.logical_and(target_bin, container_bool)
    cured_in_container = np.logical_and(cured_bin, container_bool)

    overcure = np.logical_and(cured_in_container, np.logical_not(target_in_container))
    undercure = np.logical_and(target_in_container, np.logical_not(cured_in_container))

    target_vals = np.asarray(dose_field, dtype=float)[target_in_container]

    if mode2_local_background and all_target_mask is not None:
        background_mask = _build_local_background_mask_mode2(item, all_target_mask, container_mask).astype(bool)
    else:
        background_mask = np.logical_and(container_bool, np.logical_not(target_in_container))

    background_vals = np.asarray(dose_field, dtype=float)[background_mask]

    dose80_binary = np.logical_and(np.asarray(dose_field, dtype=float) > (float(DOSE80_THRESHOLD_RATIO) * float(cfg.resin_Ec)), container_bool)
    components = _build_connected_components(dose80_binary)

    # 单目标指标时依然按最近重心配对，而不是简单取最大连通域
    matched_idx = _match_components_greedy([item], components)[0]
    matched_component = None if matched_idx is None else components[matched_idx]

    if matched_component is None:
        dose80_mask = np.zeros_like(target_bin, dtype=np.uint8)
        dose80_contours: List[np.ndarray] = []
        dose80_centroid = (np.nan, np.nan)
        dose80_area_px = 0
    else:
        dose80_mask = matched_component["mask"]
        dose80_contours = matched_component["contours"]
        dose80_centroid = (matched_component["centroid_row"], matched_component["centroid_col"])
        dose80_area_px = int(matched_component["area_px"])

    target_centroid = (float(item["target_centroid_row"]), float(item["target_centroid_col"]))

    if np.any(np.isnan(target_centroid)) or np.any(np.isnan(dose80_centroid)):
        centroid_distance_px = np.nan
    else:
        centroid_distance_px = float(
            np.hypot(dose80_centroid[0] - target_centroid[0], dose80_centroid[1] - target_centroid[1])
        )

    if np.any(np.isnan(dose80_centroid)) or dose80_area_px <= 0:
        fwhm_x = np.nan
        fwhm_y = np.nan
    else:
        centroid_r = int(np.clip(round(dose80_centroid[0]), 0, dose80_mask.shape[0] - 1))
        centroid_c = int(np.clip(round(dose80_centroid[1]), 0, dose80_mask.shape[1] - 1))
        fwhm_x = _contiguous_binary_width(dose80_mask[centroid_r, :], center_idx=centroid_c)
        fwhm_y = _contiguous_binary_width(dose80_mask[:, centroid_c], center_idx=centroid_r)

    if (
        not np.isfinite(fwhm_x)
        or not np.isfinite(fwhm_y)
        or float(fwhm_x) <= 0
        or float(fwhm_y) <= 0
    ):
        anisotropy_ratio = np.nan
    else:
        anisotropy_ratio = float(max(fwhm_x, fwhm_y) / min(fwhm_x, fwhm_y))

    metrics = {
        "iou": float(calculate_iou(target_in_container, cured_in_container, cfg=None, container_mask=container_bool)),
        "overcure_area_px": int(np.sum(overcure)),
        "undercure_area_px": int(np.sum(undercure)),
        "target_dose_mean": float(np.mean(target_vals)) if target_vals.size > 0 else np.nan,
        "target_dose_min10_mean": _bottom_fraction_mean(target_vals, fraction=0.1),
        "background_dose_mean": float(np.mean(background_vals)) if background_vals.size > 0 else np.nan,
        "background_dose_top10_mean": _top_fraction_mean(background_vals, fraction=0.1),
        "target_centroid_row": float(target_centroid[0]),
        "target_centroid_col": float(target_centroid[1]),
        "dose80_centroid_row": float(dose80_centroid[0]) if np.isfinite(dose80_centroid[0]) else np.nan,
        "dose80_centroid_col": float(dose80_centroid[1]) if np.isfinite(dose80_centroid[1]) else np.nan,
        "dose80_area_px": int(dose80_area_px),
        "centroid_distance_px": float(centroid_distance_px) if np.isfinite(centroid_distance_px) else np.nan,
        "FWHMx": float(fwhm_x) if np.isfinite(fwhm_x) else np.nan,
        "FWHMy": float(fwhm_y) if np.isfinite(fwhm_y) else np.nan,
        "anisotropy_ratio": float(anisotropy_ratio) if np.isfinite(anisotropy_ratio) else np.nan,
        "target_contours": item["target_contours"],
        "dose80_component": dose80_mask.astype(np.uint8),
        "dose80_contours": dose80_contours,
        "overcure_mask": overcure.astype(np.uint8),
        "undercure_mask": undercure.astype(np.uint8),
    }
    return metrics


def compute_display_matches_for_figure2(
    target_items: List[Dict[str, Any]],
    dose_field: np.ndarray,
    cfg: VAMConfig,
    container_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    用于 figure2 / summary_fig6_overlay 的多白圈多绿圈显示。
    - 白圈/白点：每个 target 自身的几何轮廓与几何重心
    - 绿圈/绿点：dose > 0.8Ec 连通域中，按最近重心匹配到该 target 的区域及其几何重心
    """
    dose80_binary = np.logical_and(np.asarray(dose_field, dtype=float) > (float(DOSE80_THRESHOLD_RATIO) * float(cfg.resin_Ec)), _ensure_binary(container_mask).astype(bool))
    components = _build_connected_components(dose80_binary)
    matched_idx = _match_components_greedy(target_items, components)

    matches: List[Dict[str, Any]] = []
    for item, comp_idx in zip(target_items, matched_idx):
        if comp_idx is None:
            comp = None
        else:
            comp = components[comp_idx]
        matches.append(
            {
                "position_id": item["position_id"],
                "target_contours": item["target_contours"],
                "target_centroid_row": float(item["target_centroid_row"]),
                "target_centroid_col": float(item["target_centroid_col"]),
                "dose80_component": np.zeros_like(item["target_mask"], dtype=np.uint8) if comp is None else comp["mask"],
                "dose80_contours": [] if comp is None else comp["contours"],
                "dose80_centroid_row": np.nan if comp is None else float(comp["centroid_row"]),
                "dose80_centroid_col": np.nan if comp is None else float(comp["centroid_col"]),
                "dose80_area_px": 0 if comp is None else int(comp["area_px"]),
                "centroid_distance_px": np.nan if comp is None else float(np.hypot(
                    float(comp["centroid_row"]) - float(item["target_centroid_row"]),
                    float(comp["centroid_col"]) - float(item["target_centroid_col"])
                )),
            }
        )
    return matches


# ============================================================
# 图像绘制
# ============================================================

ERROR_CMAP = ListedColormap(["red", "black", "green"])


def make_error_map(target_mask: np.ndarray, cured_mask: np.ndarray, container_mask: np.ndarray) -> np.ndarray:
    target_bin = np.logical_and(_ensure_binary(target_mask).astype(bool), _ensure_binary(container_mask).astype(bool))
    cured_bin = np.logical_and(_ensure_binary(cured_mask).astype(bool), _ensure_binary(container_mask).astype(bool))
    error_map = np.zeros_like(target_bin, dtype=int)
    error_map[np.logical_and(target_bin, np.logical_not(cured_bin))] = -1
    error_map[np.logical_and(cured_bin, np.logical_not(target_bin))] = 1
    return error_map


def plot_figure1_cure_compare(
    output_path: Path,
    title_prefix: str,
    target_mask: np.ndarray,
    proto_cured: np.ndarray,
    proto_iou_metrics: Dict[str, Any],
    optimized_cured: Optional[np.ndarray],
    optimized_iou_metrics: Optional[Dict[str, Any]],
    container_mask: np.ndarray,
    save_figures: bool,
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.3))

    _imshow_binary_panel(axes[0, 0], target_mask, "Target")
    _imshow_binary_panel(axes[0, 1], proto_cured, "Proto cured")
    if optimized_cured is not None:
        _imshow_binary_panel(axes[0, 2], optimized_cured, "Optimized cured")
    else:
        _plot_placeholder_panel(axes[0, 2], "Optimized cured", "optimized not run")

    _imshow_binary_panel(axes[1, 0], target_mask, "Target reference")

    proto_error = make_error_map(target_mask, proto_cured, container_mask)
    _imshow_error_panel(axes[1, 1], proto_error, "Proto error map")
    _annotate_metric_text(
        axes[1, 1],
        f"IoU = {proto_iou_metrics['iou']:.4f}\novercure = {proto_iou_metrics['overcure_area_px']} px\nundercure = {proto_iou_metrics['undercure_area_px']} px",
    )

    if optimized_cured is not None and optimized_iou_metrics is not None:
        optimized_error = make_error_map(target_mask, optimized_cured, container_mask)
        _imshow_error_panel(axes[1, 2], optimized_error, "Optimized error map")
        _annotate_metric_text(
            axes[1, 2],
            f"IoU = {optimized_iou_metrics['iou']:.4f}\novercure = {optimized_iou_metrics['overcure_area_px']} px\nundercure = {optimized_iou_metrics['undercure_area_px']} px",
        )
    else:
        _plot_placeholder_panel(axes[1, 2], "Optimized error map", "optimized not run")

    fig.suptitle(f"Figure 1 - Cure comparison and error maps\n{title_prefix}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_figures:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


def _get_display_dose_for_fig2(projection: np.ndarray, dose_field: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    if not USE_FINE_DOSE_VIS:
        return np.asarray(dose_field, dtype=float)
    return np.asarray(
        forward_dose_simulation_visualization(
            projection_matrix=projection,
            cfg=cfg,
            output_grid_size=int(FINE_DOSE_GRID_SIZE),
        ),
        dtype=float,
    )


def _scale_display_matches_for_fine_grid(
    matches: List[Dict[str, Any]],
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches:
        out.append(
            {
                "position_id": m["position_id"],
                "target_contours": _scale_contours(m["target_contours"], from_shape, to_shape),
                "target_centroid_row": _scale_rc((m["target_centroid_row"], m["target_centroid_col"]), from_shape, to_shape)[0],
                "target_centroid_col": _scale_rc((m["target_centroid_row"], m["target_centroid_col"]), from_shape, to_shape)[1],
                "dose80_contours": _scale_contours(m["dose80_contours"], from_shape, to_shape),
                "dose80_centroid_row": _scale_rc((m["dose80_centroid_row"], m["dose80_centroid_col"]), from_shape, to_shape)[0]
                    if np.isfinite(m["dose80_centroid_row"]) and np.isfinite(m["dose80_centroid_col"]) else np.nan,
                "dose80_centroid_col": _scale_rc((m["dose80_centroid_row"], m["dose80_centroid_col"]), from_shape, to_shape)[1]
                    if np.isfinite(m["dose80_centroid_row"]) and np.isfinite(m["dose80_centroid_col"]) else np.nan,
                "centroid_distance_px": m["centroid_distance_px"] * float(to_shape[0]) / float(from_shape[0]),
            }
        )
    return out


def _draw_multi_matches(
    ax,
    matches: List[Dict[str, Any]],
    *,
    show_white_centroids: bool = True,
    show_green_centroids: bool = True,
) -> None:
    for m in matches:
        _draw_contours(ax, m["target_contours"], color="white", linewidth=0.25)
        _draw_contours(ax, m["dose80_contours"], color="lime", linewidth=0.25)
        if show_white_centroids and np.isfinite(m["target_centroid_row"]) and np.isfinite(m["target_centroid_col"]):
            ax.plot(m["target_centroid_col"], m["target_centroid_row"], marker=",", linestyle="None", color="white")
        if show_green_centroids and np.isfinite(m["dose80_centroid_row"]) and np.isfinite(m["dose80_centroid_col"]):
            ax.plot(m["dose80_centroid_col"], m["dose80_centroid_row"], marker=",", linestyle="None", color="lime")


def plot_figure2_dose_compare(
    output_path: Path,
    title_prefix: str,
    proto_projection: np.ndarray,
    proto_dose: np.ndarray,
    proto_matches: List[Dict[str, Any]],
    optimized_projection: Optional[np.ndarray],
    optimized_dose: Optional[np.ndarray],
    optimized_matches: Optional[List[Dict[str, Any]]],
    cfg: VAMConfig,
    save_figures: bool,
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7))

    proto_display_dose = _get_display_dose_for_fig2(proto_projection, proto_dose, cfg)
    proto_display_matches = proto_matches
    if USE_FINE_DOSE_VIS:
        proto_display_matches = _scale_display_matches_for_fine_grid(proto_matches, (cfg.n_slice, cfg.n_slice), proto_display_dose.shape)

    optimized_display_dose = None
    optimized_display_matches = None
    if optimized_projection is not None and optimized_dose is not None and optimized_matches is not None:
        optimized_display_dose = _get_display_dose_for_fig2(optimized_projection, optimized_dose, cfg)
        optimized_display_matches = optimized_matches
        if USE_FINE_DOSE_VIS:
            optimized_display_matches = _scale_display_matches_for_fine_grid(optimized_matches, (cfg.n_slice, cfg.n_slice), optimized_display_dose.shape)

    common_vmin = 0.0
    common_vmax = _finite_max(
        np.asarray(proto_display_dose).ravel(),
        [] if optimized_display_dose is None else np.asarray(optimized_display_dose).ravel(),
        default=max(float(cfg.resin_Ec), 1.0),
    )

    for ax in axes:
        _configure_pretty_axes(ax)

    im0 = axes[0].imshow(proto_display_dose, origin="upper", cmap="plasma", interpolation="nearest", vmin=common_vmin, vmax=common_vmax)
    axes[0].set_title("Proto dose map")
    _draw_multi_matches(
        axes[0],
        proto_display_matches,
        show_white_centroids=SHOW_WHITE_CENTROIDS,
        show_green_centroids=SHOW_GREEN_CENTROIDS,
    )
    proto_dist_text = ", ".join([f"{m['position_id']}:{m['centroid_distance_px']:.2f}" if np.isfinite(m["centroid_distance_px"]) else f"{m['position_id']}:nan" for m in proto_display_matches])
    _annotate_metric_text(axes[0], f"centroid_distance_px = {proto_dist_text}" if len(proto_display_matches) <= 4 else "see summary table / summary_fig6")

    if optimized_display_dose is not None and optimized_display_matches is not None:
        im1 = axes[1].imshow(optimized_display_dose, origin="upper", cmap="plasma", interpolation="nearest", vmin=common_vmin, vmax=common_vmax)
        axes[1].set_title("Optimized dose map")
        _draw_multi_matches(
            axes[1],
            optimized_display_matches,
            show_white_centroids=SHOW_WHITE_CENTROIDS,
            show_green_centroids=SHOW_GREEN_CENTROIDS,
        )
        opt_dist_text = ", ".join([f"{m['position_id']}:{m['centroid_distance_px']:.2f}" if np.isfinite(m["centroid_distance_px"]) else f"{m['position_id']}:nan" for m in optimized_display_matches])
        _annotate_metric_text(axes[1], f"centroid_distance_px = {opt_dist_text}" if len(optimized_display_matches) <= 4 else "see summary table / summary_fig6")
    else:
        _plot_placeholder_panel(axes[1], "Optimized dose map", "optimized not run")
        im1 = None

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Dose (mJ/cm²)")
    if im1 is not None:
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Dose (mJ/cm²)")

    fig.suptitle(f"Figure 2 - Dose map comparison\n{title_prefix}", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_figures:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


# ============================================================
# 结果保存
# ============================================================

def _write_single_row_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({k: _to_jsonable(v) for k, v in row.items()})


def _write_summary_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_jsonable(v) for k, v in row.items()})


def save_single_result_bundle(
    output_dir: Path,
    tag: str,
    target_mask: np.ndarray,
    proto_result: Dict[str, Any],
    optimized_result: Optional[Dict[str, Any]],
    cfg: VAMConfig,
    metadata: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "target_mask.npy", _ensure_binary(target_mask).astype(np.uint8))
    np.save(output_dir / "proto_dose.npy", np.asarray(proto_result["dose_field"], dtype=np.float64))
    np.save(output_dir / "proto_cured.npy", _ensure_binary(proto_result["cured_mask"]).astype(np.uint8))
    save_projection_npz(
        output_dir / "proto_projection.npz",
        np.asarray(proto_result["projection_matrix"], dtype=np.float64),
        cfg,
        target_mask=_ensure_binary(target_mask).astype(np.uint8),
        extra_metadata={"analysis_type": np.array([f"{tag}_proto"], dtype=object), "bundle_metadata": np.array([json.dumps(_to_jsonable(metadata), ensure_ascii=False)], dtype=object)},
    )

    if optimized_result is not None:
        np.save(output_dir / "optimized_dose.npy", np.asarray(optimized_result["dose_field"], dtype=np.float64))
        np.save(output_dir / "optimized_cured.npy", _ensure_binary(optimized_result["cured_mask"]).astype(np.uint8))
        save_projection_npz(
            output_dir / "optimized_projection.npz",
            np.asarray(optimized_result["projection_matrix"], dtype=np.float64),
            cfg,
            target_mask=_ensure_binary(target_mask).astype(np.uint8),
            extra_metadata={"analysis_type": np.array([f"{tag}_optimized"], dtype=object), "bundle_metadata": np.array([json.dumps(_to_jsonable(metadata), ensure_ascii=False)], dtype=object)},
        )


# ============================================================
# summary 数据、图和 overlay 表格
# ============================================================

def build_summary_row(
    position_info: Dict[str, Any],
    structure_type: str,
    proto_metrics: Dict[str, Any],
    optimized_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "position_id": position_info["position_id"],
        "scan_mode": position_info["scan_mode"],
        "structure_type": structure_type,
        "r_ratio": position_info["r_ratio"],
        "phi_deg": position_info["phi_deg"],
        "center_row": position_info["center_row"],
        "center_col": position_info["center_col"],
    }

    keys = [
        "iou",
        "overcure_area_px",
        "undercure_area_px",
        "target_dose_mean",
        "target_dose_min10_mean",
        "background_dose_mean",
        "background_dose_top10_mean",
        "target_centroid_row",
        "target_centroid_col",
        "dose80_centroid_row",
        "dose80_centroid_col",
        "dose80_area_px",
        "centroid_distance_px",
        "FWHMx",
        "FWHMy",
        "anisotropy_ratio",
    ]
    for key in keys:
        row[f"proto_{key}"] = proto_metrics.get(key, np.nan)
    if optimized_metrics is None:
        for key in keys:
            row[f"optimized_{key}"] = np.nan
    else:
        for key in keys:
            row[f"optimized_{key}"] = optimized_metrics.get(key, np.nan)

    return row


def _barplot_two_metrics(
    ax,
    labels: Sequence[str],
    values_a: Sequence[float],
    values_b: Sequence[float],
    title: str,
    label_a: str,
    label_b: str,
    *,
    ylim_top: Optional[float] = None,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    bars_a = ax.bar(x - width / 2.0, values_a, width=width, label=label_a, color="#4C78A8")
    bars_b = ax.bar(x + width / 2.0, values_b, width=width, label=label_b, color="#F58518")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    if ylim_top is not None and np.isfinite(ylim_top):
        ax.set_ylim(0.0, float(ylim_top))
    _annotate_bar_values(ax, bars_a)
    _annotate_bar_values(ax, bars_b)


def plot_summary_target_dose(summary_rows: List[Dict[str, Any]], output_root: Path, save_figures: bool, show_figures: bool) -> None:
    labels = [_short_position_label(row["position_id"]) for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.2), sharey=True)

    proto_mean = [row["proto_target_dose_mean"] for row in summary_rows]
    proto_min10 = [row["proto_target_dose_min10_mean"] for row in summary_rows]
    opt_mean = [row["optimized_target_dose_mean"] for row in summary_rows]
    opt_min10 = [row["optimized_target_dose_min10_mean"] for row in summary_rows]
    y_top = 1.18 * _finite_max(proto_mean, proto_min10, opt_mean, opt_min10, default=1.0)

    _barplot_two_metrics(axes[0], labels, proto_mean, proto_min10, "Proto target dose", "target_dose_mean", "target_dose_min10_mean", ylim_top=y_top)
    axes[0].set_ylabel("Dose (mJ/cm²)")

    if _branch_has_any_finite(summary_rows, ["optimized_target_dose_mean", "optimized_target_dose_min10_mean"]):
        _barplot_two_metrics(axes[1], labels, opt_mean, opt_min10, "Optimized target dose", "target_dose_mean", "target_dose_min10_mean", ylim_top=y_top)
    else:
        _plot_placeholder_panel(axes[1], "Optimized target dose", "optimized not run")

    fig.suptitle("Figure 3 - Summary target region dose statistics", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    if save_figures:
        fig.savefig(_summary_figure_path(output_root, 3, "summary_fig1_target_dose.png"), dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


def plot_summary_background_dose(summary_rows: List[Dict[str, Any]], output_root: Path, save_figures: bool, show_figures: bool) -> None:
    labels = [_short_position_label(row["position_id"]) for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.2), sharey=True)

    proto_mean = [row["proto_background_dose_mean"] for row in summary_rows]
    proto_top10 = [row["proto_background_dose_top10_mean"] for row in summary_rows]
    opt_mean = [row["optimized_background_dose_mean"] for row in summary_rows]
    opt_top10 = [row["optimized_background_dose_top10_mean"] for row in summary_rows]
    y_top = 1.18 * _finite_max(proto_mean, proto_top10, opt_mean, opt_top10, default=1.0)

    _barplot_two_metrics(axes[0], labels, proto_mean, proto_top10, "Proto background dose", "background_dose_mean", "background_dose_top10_mean", ylim_top=y_top)
    axes[0].set_ylabel("Dose (mJ/cm²)")

    if _branch_has_any_finite(summary_rows, ["optimized_background_dose_mean", "optimized_background_dose_top10_mean"]):
        _barplot_two_metrics(axes[1], labels, opt_mean, opt_top10, "Optimized background dose", "background_dose_mean", "background_dose_top10_mean", ylim_top=y_top)
    else:
        _plot_placeholder_panel(axes[1], "Optimized background dose", "optimized not run")

    fig.suptitle("Figure 4 - Summary background region dose statistics", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    if save_figures:
        fig.savefig(_summary_figure_path(output_root, 4, "summary_fig2_background_dose.png"), dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


def _add_fwhm_table(ax, labels: Sequence[str], fwhmx_vals: Sequence[float], fwhmy_vals: Sequence[float]) -> None:
    cell_text = [
        [_format_value_for_annotation(v) for v in fwhmx_vals],
        [_format_value_for_annotation(v) for v in fwhmy_vals],
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=["FWHMx", "FWHMy"],
        colLabels=list(labels),
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, -0.47, 1.0, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)


def plot_summary_anisotropy(summary_rows: List[Dict[str, Any]], output_root: Path, save_figures: bool, show_figures: bool) -> None:
    labels = [_short_position_label(row["position_id"]) for row in summary_rows]
    x = np.arange(len(labels), dtype=float)

    proto_ratios = [row["proto_anisotropy_ratio"] for row in summary_rows]
    opt_ratios = [row["optimized_anisotropy_ratio"] for row in summary_rows]
    y_min = _finite_min(proto_ratios, opt_ratios, default=1.0)
    y_max = _finite_max(proto_ratios, opt_ratios, default=1.0)
    if y_max <= y_min:
        y_min = max(0.0, 0.95 * y_min)
        y_max = 1.10 * y_max if y_max > 0 else 1.0
    else:
        margin = 0.10 * (y_max - y_min)
        y_min = max(0.0, y_min - margin)
        y_max = y_max + margin

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 8.0), sharey=True)
    for ax, prefix, title in [
        (axes[0], "proto", "Proto anisotropy ratio"),
        (axes[1], "optimized", "Optimized anisotropy ratio"),
    ]:
        ratios = [row[f"{prefix}_anisotropy_ratio"] for row in summary_rows]
        finite_mask = np.isfinite(np.asarray(ratios, dtype=float))
        if not np.any(finite_mask):
            _plot_placeholder_panel(ax, title, f"{prefix} branch has no finite anisotropy data")
            continue

        ax.plot(x, ratios, marker="o", linewidth=2.0, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Anisotropy ratio")
        ax.set_xlabel("position_id")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)

        for idx, ratio in enumerate(ratios):
            if np.isfinite(ratio):
                ax.annotate(_format_value_for_annotation(ratio), (idx, ratio), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        fwhmx_vals = [row[f"{prefix}_FWHMx"] for row in summary_rows]
        fwhmy_vals = [row[f"{prefix}_FWHMy"] for row in summary_rows]
        _add_fwhm_table(ax, labels, fwhmx_vals, fwhmy_vals)

    fig.suptitle("Figure 5 - Summary anisotropy ratio with FWHMx / FWHMy", fontsize=14)
    fig.subplots_adjust(bottom=0.34, wspace=0.22, top=0.88)
    if save_figures:
        fig.savefig(_summary_figure_path(output_root, 5, "summary_fig3_anisotropy.png"), dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


def _build_overlay_table_rows(overlay_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return overlay_rows


def _format_table_value(value: Any) -> str:
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "nan"
        if abs(float(value) - round(float(value))) < 1e-9:
            return str(int(round(float(value))))
        return f"{float(value):.3f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def write_summary_overlay_table(overlay_rows: List[Dict[str, Any]], output_root: Path) -> None:
    _write_summary_csv(output_root / "summary_fig6_overlay_table.csv", overlay_rows)


def plot_summary_overlay(
    cfg: VAMConfig,
    target_union: np.ndarray,
    proto_matches: List[Dict[str, Any]],
    optimized_matches: Optional[List[Dict[str, Any]]],
    overlay_rows: List[Dict[str, Any]],
    output_root: Path,
    save_figures: bool,
    show_figures: bool,
) -> None:
    has_optimized = optimized_matches is not None

    fig_height = max(8.2, 5.8 + 0.42 * max(len(overlay_rows), 1))
    fig = plt.figure(figsize=(14.5, fig_height))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3.2, max(1.6, 0.42 * max(len(overlay_rows), 1))], hspace=0.15, wspace=0.12)

    ax_proto = fig.add_subplot(gs[0, 0])
    ax_optimized = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    black_cmap = ListedColormap(["black", "white"])

    for ax, branch, title, matches in [
        (ax_proto, "proto", "Proto", proto_matches),
        (ax_optimized, "optimized", "Optimized", optimized_matches),
    ]:
        _configure_pretty_axes(ax)
        if branch == "optimized" and not has_optimized:
            _plot_placeholder_panel(ax, title, "optimized not run")
            continue

        ax.imshow(target_union, origin="upper", cmap=black_cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title)
        if matches is not None:
            for m in matches:
                _draw_contours(ax, m["dose80_contours"], color="lime", linewidth=0.25)
                if SHOW_WHITE_CENTROIDS and np.isfinite(m["target_centroid_row"]) and np.isfinite(m["target_centroid_col"]):
                    ax.plot(m["target_centroid_col"], m["target_centroid_row"], marker=",", linestyle="None", color="white")
                if SHOW_GREEN_CENTROIDS and np.isfinite(m["dose80_centroid_row"]) and np.isfinite(m["dose80_centroid_col"]):
                    ax.plot(m["dose80_centroid_col"], m["dose80_centroid_row"], marker=",", linestyle="None", color="lime")

    ax_table.axis("off")
    col_labels = [
        "position_id",
        "proto_area_px",
        "proto_dist_px",
        "optimized_area_px",
        "optimized_dist_px",
    ]
    cell_text = [
        [
            _format_table_value(row["position_id"]),
            _format_table_value(row["proto_dose80_area_px"]),
            _format_table_value(row["proto_centroid_distance_px"]),
            _format_table_value(row["optimized_dose80_area_px"]),
            _format_table_value(row["optimized_centroid_distance_px"]),
        ]
        for row in overlay_rows
    ]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.0, 0.96, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.18)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
        cell.set_linewidth(0.6)

    fig.suptitle("Figure 6 - All targets, dose80 contours, centroids, and overlay table", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if save_figures:
        fig.savefig(_summary_figure_path(output_root, 6, "summary_fig6_overlay.png"), dpi=220, bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


# ============================================================
# 主运行辅助
# ============================================================

def _summary_figure_path(output_root: Path, figure_index: int, legacy_filename: str) -> Path:
    """
    模式2下，将 summary_fig1 / 2 / 3 / 6 的文件名改为 figure3 / 4 / 5 / 6。
    模式1保持原有 summary 命名，避免打乱既有批处理目录结构。
    """
    if ANALYSIS_MODE == "mode2":
        mapping = {
            3: "figure3_target_dose.png",
            4: "figure4_background_dose.png",
            5: "figure5_anisotropy.png",
            6: "figure6_overlay.png",
        }
        return output_root / mapping[figure_index]
    return output_root / legacy_filename

def build_runtime_config() -> VAMConfig:
    cfg = get_default_config()
    if CFG_N_SLICE is not None:
        cfg.n_slice = int(CFG_N_SLICE)
    if CFG_ANGLE_NUM is not None:
        cfg.angle_num = int(CFG_ANGLE_NUM)
    if CFG_RESIN_ALPHA is not None:
        cfg.resin_alpha = float(CFG_RESIN_ALPHA)
    if CFG_RESIN_EC is not None:
        cfg.resin_Ec = float(CFG_RESIN_EC)
    if CFG_NUM_ITERATIONS is not None:
        cfg.num_iterations = int(CFG_NUM_ITERATIONS)
    if CFG_USE_NUMBA is not None:
        cfg.use_numba = bool(CFG_USE_NUMBA)
    if SHOW_FIGURES_OVERRIDE is not None:
        cfg.show_figures = bool(SHOW_FIGURES_OVERRIDE)
    return cfg


def save_config_snapshot(output_root: Path, cfg: VAMConfig) -> None:
    payload = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_output_root": ANALYSIS_OUTPUT_ROOT,
        "analysis_mode": ANALYSIS_MODE,
        "run_mode": RUN_MODE,
        "structure_type": STRUCTURE_TYPE,
        "scan_mode": SCAN_MODE,
        "point_radius_px": POINT_RADIUS_PX,
        "disk_radius_px": DISK_RADIUS_PX,
        "rect_width_px": RECT_WIDTH_PX,
        "rect_height_px": RECT_HEIGHT_PX,
        "phi_deg_fixed": PHI_DEG_FIXED,
        "r_ratio_list": R_RATIO_LIST,
        "r_ratio_fixed": R_RATIO_FIXED,
        "phi_deg_list": PHI_DEG_LIST,
        "use_fine_dose_vis": USE_FINE_DOSE_VIS,
        "fine_dose_grid_size": FINE_DOSE_GRID_SIZE,
        "save_figures": SAVE_FIGURES,
        "save_results": SAVE_RESULTS,
        "dose80_threshold_ratio": DOSE80_THRESHOLD_RATIO,
        "show_white_centroids": SHOW_WHITE_CENTROIDS,
        "show_green_centroids": SHOW_GREEN_CENTROIDS,
        "cfg": cfg.to_metadata(),
    }
    with open(output_root / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def _build_output_root(cfg: VAMConfig) -> Path:
    if ANALYSIS_OUTPUT_ROOT is None:
        return Path(cfg.output_dir) / "position_analysis" / ANALYSIS_NAME
    return Path(ANALYSIS_OUTPUT_ROOT)


def _compute_overall_iou_metrics(target_mask: np.ndarray, dose_field: np.ndarray, cured_mask: np.ndarray, cfg: VAMConfig, container_mask: np.ndarray) -> Dict[str, Any]:
    """
    用于 figure1 的整体 IoU / 过固化 / 欠固化统计。
    这里保持对整个 target mask 的全局统计。
    """
    target_bin = np.logical_and(_ensure_binary(target_mask).astype(bool), _ensure_binary(container_mask).astype(bool))
    cured_bin = np.logical_and(_ensure_binary(cured_mask).astype(bool), _ensure_binary(container_mask).astype(bool))
    overcure = np.logical_and(cured_bin, np.logical_not(target_bin))
    undercure = np.logical_and(target_bin, np.logical_not(cured_bin))
    return {
        "iou": float(calculate_iou(target_bin, cured_bin, cfg=None, container_mask=_ensure_binary(container_mask).astype(bool))),
        "overcure_area_px": int(np.sum(overcure)),
        "undercure_area_px": int(np.sum(undercure)),
    }


def _build_summary_and_overlay_from_results(
    target_items: List[Dict[str, Any]],
    target_union: np.ndarray,
    proto_result: Dict[str, Any],
    optimized_result: Optional[Dict[str, Any]],
    cfg: VAMConfig,
    container_mask: np.ndarray,
    mode2: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    summary_rows: List[Dict[str, Any]] = []
    overlay_rows: List[Dict[str, Any]] = []

    proto_matches = compute_display_matches_for_figure2(target_items, proto_result["dose_field"], cfg, container_mask)
    optimized_matches = None if optimized_result is None else compute_display_matches_for_figure2(target_items, optimized_result["dose_field"], cfg, container_mask)

    for idx, item in enumerate(target_items):
        proto_metrics = _compute_single_item_metrics(
            item=item,
            dose_field=np.asarray(proto_result["dose_field"], dtype=float),
            cured_mask=_ensure_binary(proto_result["cured_mask"]),
            cfg=cfg,
            container_mask=container_mask,
            all_target_mask=target_union,
            mode2_local_background=mode2,
        )
        if optimized_result is None:
            optimized_metrics = None
        else:
            optimized_metrics = _compute_single_item_metrics(
                item=item,
                dose_field=np.asarray(optimized_result["dose_field"], dtype=float),
                cured_mask=_ensure_binary(optimized_result["cured_mask"]),
                cfg=cfg,
                container_mask=container_mask,
                all_target_mask=target_union,
                mode2_local_background=mode2,
            )

        summary_rows.append(
            build_summary_row(
                position_info=item["position_info"],
                structure_type=item["structure_type"],
                proto_metrics=proto_metrics,
                optimized_metrics=optimized_metrics,
            )
        )

        proto_match = proto_matches[idx]
        opt_match = None if optimized_matches is None else optimized_matches[idx]
        overlay_rows.append(
            {
                "position_id": item["position_id"],
                "scan_mode": item["position_info"]["scan_mode"],
                "structure_type": item["structure_type"],
                "r_ratio": item["position_info"]["r_ratio"],
                "phi_deg": item["position_info"]["phi_deg"],
                "center_row": item["position_info"]["center_row"],
                "center_col": item["position_info"]["center_col"],
                "proto_dose80_area_px": proto_match["dose80_area_px"],
                "proto_centroid_distance_px": proto_match["centroid_distance_px"],
                "optimized_dose80_area_px": np.nan if opt_match is None else opt_match["dose80_area_px"],
                "optimized_centroid_distance_px": np.nan if opt_match is None else opt_match["centroid_distance_px"],
            }
        )

    return summary_rows, overlay_rows, proto_matches, optimized_matches


def _save_mode2_item_metrics(output_root: Path, summary_rows: List[Dict[str, Any]]) -> None:
    for row in summary_rows:
        position_dir = output_root / row["position_id"]
        position_dir.mkdir(parents=True, exist_ok=True)
        _write_single_row_csv(position_dir / "metrics.csv", row)
        with open(position_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(row), f, ensure_ascii=False, indent=2)


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    _validate_runtime_parameters()

    cfg = build_runtime_config()
    ensure_output_dirs(cfg)

    output_root = _build_output_root(cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    if SAVE_RESULTS:
        save_config_snapshot(output_root, cfg)

    container_mask = build_container_mask((cfg.n_slice, cfg.n_slice), cfg)
    positions = generate_scan_positions(cfg)
    target_items = build_target_items(cfg, container_mask, positions)

    run_optimized = RUN_MODE in {"optimized", "both"}
    show_figures = bool(cfg.show_figures)

    if ANALYSIS_MODE == "mode1":
        summary_rows_all: List[Dict[str, Any]] = []
        overlay_rows_all: List[Dict[str, Any]] = []
        proto_matches_all: List[Dict[str, Any]] = []
        optimized_matches_all: List[Dict[str, Any]] = []
        target_union = build_combined_target_mask(target_items, (cfg.n_slice, cfg.n_slice))

        for index, item in enumerate(target_items, start=1):
            position_info = item["position_info"]
            position_id = item["position_id"]
            position_dir = output_root / position_id
            if SAVE_FIGURES:
                position_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{index}/{len(target_items)}] Running {position_id} ...")

            target_mask = item["target_mask"].astype(np.uint8)

            try:
                initial_sinogram = generate_initial_sinogram(target_mask, cfg)
                raw_projection = generate_initial_projection_matrix(initial_sinogram, cfg)
                proto_result = compensate_and_print_initial_projection(raw_projection, target_mask, cfg)
                optimized_result = optimize_projection_matrix(proto_result["projection_matrix"], target_mask, cfg) if run_optimized else None

                # 单位置 summary 与 overlay
                summary_rows, overlay_rows, proto_matches, optimized_matches = _build_summary_and_overlay_from_results(
                    target_items=[item],
                    target_union=target_mask,
                    proto_result=proto_result,
                    optimized_result=optimized_result,
                    cfg=cfg,
                    container_mask=container_mask,
                    mode2=False,
                )
                summary_row = summary_rows[0]
                overlay_row = overlay_rows[0]
                proto_match = proto_matches[0]
                optimized_match = None if optimized_matches is None else optimized_matches[0]

                summary_rows_all.append(summary_row)
                overlay_rows_all.append(overlay_row)
                proto_matches_all.append(proto_match)
                if optimized_match is not None:
                    optimized_matches_all.append(optimized_match)

                if SAVE_RESULTS:
                    save_single_result_bundle(
                        output_dir=position_dir,
                        tag="position_analysis",
                        target_mask=target_mask,
                        proto_result=proto_result,
                        optimized_result=optimized_result,
                        cfg=cfg,
                        metadata={
                            "position_id": position_id,
                            "analysis_mode": ANALYSIS_MODE,
                            "run_mode": RUN_MODE,
                            "structure_type": STRUCTURE_TYPE,
                        },
                    )
                    _write_single_row_csv(position_dir / "metrics.csv", summary_row)
                    with open(position_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(_to_jsonable(summary_row), f, ensure_ascii=False, indent=2)

                proto_iou_metrics = _compute_overall_iou_metrics(target_mask, proto_result["dose_field"], proto_result["cured_mask"], cfg, container_mask)
                optimized_iou_metrics = None if optimized_result is None else _compute_overall_iou_metrics(target_mask, optimized_result["dose_field"], optimized_result["cured_mask"], cfg, container_mask)

                if SAVE_FIGURES:
                    plot_figure1_cure_compare(
                        output_path=position_dir / "fig1_cure_compare.png",
                        title_prefix=f"{position_id} | center=({position_info['center_row']}, {position_info['center_col']})",
                        target_mask=target_mask,
                        proto_cured=_ensure_binary(proto_result["cured_mask"]),
                        proto_iou_metrics=proto_iou_metrics,
                        optimized_cured=None if optimized_result is None else _ensure_binary(optimized_result["cured_mask"]),
                        optimized_iou_metrics=optimized_iou_metrics,
                        container_mask=container_mask,
                        save_figures=True,
                        show_figures=show_figures,
                    )
                    plot_figure2_dose_compare(
                        output_path=position_dir / "fig2_dose_compare.png",
                        title_prefix=position_id,
                        proto_projection=np.asarray(proto_result["projection_matrix"], dtype=float),
                        proto_dose=np.asarray(proto_result["dose_field"], dtype=float),
                        proto_matches=[proto_match],
                        optimized_projection=None if optimized_result is None else np.asarray(optimized_result["projection_matrix"], dtype=float),
                        optimized_dose=None if optimized_result is None else np.asarray(optimized_result["dose_field"], dtype=float),
                        optimized_matches=None if optimized_match is None else [optimized_match],
                        cfg=cfg,
                        save_figures=True,
                        show_figures=show_figures,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Position analysis failed at {position_id} | center=({position_info['center_row']}, {position_info['center_col']})"
                ) from exc

        if SAVE_RESULTS:
            _write_summary_csv(output_root / "summary_metrics.csv", summary_rows_all)
            write_summary_overlay_table(overlay_rows_all, output_root)

        if SAVE_FIGURES and summary_rows_all:
            plot_summary_target_dose(summary_rows_all, output_root, SAVE_FIGURES, show_figures)
            plot_summary_background_dose(summary_rows_all, output_root, SAVE_FIGURES, show_figures)
            plot_summary_anisotropy(summary_rows_all, output_root, SAVE_FIGURES, show_figures)
            plot_summary_overlay(
                cfg=cfg,
                target_union=target_union,
                proto_matches=proto_matches_all,
                optimized_matches=None if not optimized_matches_all else optimized_matches_all,
                overlay_rows=overlay_rows_all,
                output_root=output_root,
                save_figures=SAVE_FIGURES,
                show_figures=show_figures,
            )

    else:
        print(f"[mode2] Running combined target with {len(target_items)} structures ...")
        target_union = build_combined_target_mask(target_items, (cfg.n_slice, cfg.n_slice))

        initial_sinogram = generate_initial_sinogram(target_union, cfg)
        raw_projection = generate_initial_projection_matrix(initial_sinogram, cfg)
        proto_result = compensate_and_print_initial_projection(raw_projection, target_union, cfg)
        optimized_result = optimize_projection_matrix(proto_result["projection_matrix"], target_union, cfg) if run_optimized else None

        summary_rows, overlay_rows, proto_matches, optimized_matches = _build_summary_and_overlay_from_results(
            target_items=target_items,
            target_union=target_union,
            proto_result=proto_result,
            optimized_result=optimized_result,
            cfg=cfg,
            container_mask=container_mask,
            mode2=True,
        )

        if SAVE_RESULTS:
            _write_summary_csv(output_root / "summary_metrics.csv", summary_rows)
            write_summary_overlay_table(overlay_rows, output_root)
            save_single_result_bundle(
                output_dir=output_root,
                tag="combined_analysis",
                target_mask=target_union,
                proto_result=proto_result,
                optimized_result=optimized_result,
                cfg=cfg,
                metadata={
                    "analysis_mode": ANALYSIS_MODE,
                    "run_mode": RUN_MODE,
                    "structure_type": STRUCTURE_TYPE,
                    "n_targets": len(target_items),
                },
            )
            _save_mode2_item_metrics(output_root, summary_rows)

        proto_iou_metrics = _compute_overall_iou_metrics(target_union, proto_result["dose_field"], proto_result["cured_mask"], cfg, container_mask)
        optimized_iou_metrics = None if optimized_result is None else _compute_overall_iou_metrics(target_union, optimized_result["dose_field"], optimized_result["cured_mask"], cfg, container_mask)

        if SAVE_FIGURES:
            plot_figure1_cure_compare(
                output_path=output_root / "fig1_cure_compare.png",
                title_prefix=f"mode2 | total_targets={len(target_items)}",
                target_mask=target_union,
                proto_cured=_ensure_binary(proto_result["cured_mask"]),
                proto_iou_metrics=proto_iou_metrics,
                optimized_cured=None if optimized_result is None else _ensure_binary(optimized_result["cured_mask"]),
                optimized_iou_metrics=optimized_iou_metrics,
                container_mask=container_mask,
                save_figures=True,
                show_figures=show_figures,
            )
            plot_figure2_dose_compare(
                output_path=output_root / "fig2_dose_compare.png",
                title_prefix=f"mode2 | total_targets={len(target_items)}",
                proto_projection=np.asarray(proto_result["projection_matrix"], dtype=float),
                proto_dose=np.asarray(proto_result["dose_field"], dtype=float),
                proto_matches=proto_matches,
                optimized_projection=None if optimized_result is None else np.asarray(optimized_result["projection_matrix"], dtype=float),
                optimized_dose=None if optimized_result is None else np.asarray(optimized_result["dose_field"], dtype=float),
                optimized_matches=optimized_matches,
                cfg=cfg,
                save_figures=True,
                show_figures=show_figures,
            )
            plot_summary_target_dose(summary_rows, output_root, SAVE_FIGURES, show_figures)
            plot_summary_background_dose(summary_rows, output_root, SAVE_FIGURES, show_figures)
            plot_summary_anisotropy(summary_rows, output_root, SAVE_FIGURES, show_figures)
            plot_summary_overlay(
                cfg=cfg,
                target_union=target_union,
                proto_matches=proto_matches,
                optimized_matches=optimized_matches,
                overlay_rows=overlay_rows,
                output_root=output_root,
                save_figures=SAVE_FIGURES,
                show_figures=show_figures,
            )

    print(f"Analysis finished. Output root: {output_root}")


if __name__ == "__main__":
    main()
