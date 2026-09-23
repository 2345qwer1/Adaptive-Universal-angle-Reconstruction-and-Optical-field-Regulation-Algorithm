from __future__ import annotations

"""
run_point_psf_error_analysis.py

用途：
    构造一个点状 target，分别分析 proto / optimized 打印结果的 PSF 扩散与误差。

核心分析内容：
    1. 点 target 的几何中心、固化区域中心、IoU、过固化/欠固化像素。
    2. dose 峰值位置、x/y 方向 FWHM。
    3. cured 区域在 x/y 方向的连续宽度。
    4. 可选使用更高分辨率 dose 可视化网格观察细节。

典型用途：
    用单点结构评估系统点扩散行为，判断 proto 和 optimized 是否产生明显扩散、
    偏移或各向异性。
"""

from pathlib import Path
import sys
import json
from typing import Any, Dict, Optional, Tuple, Union

# 允许从 scripts/ 目录直接运行脚本时，仍能 import 项目根目录下的 functions 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from functions import (
    VAMConfig,
    get_default_config,
    ensure_output_dirs,
    generate_initial_sinogram,
    generate_initial_projection_matrix,
    compensate_and_print_initial_projection,
    calculate_iou,
    build_container_mask,
    optimize_projection_matrix,
    save_projection_npz,
)
from functions.physics_printing import forward_dose_simulation_visualization

SizeType = Union[int, Tuple[int, int]]

# ============================================================
# 脚本层可修改参数（统一放在最前面）
# ============================================================
# 运行分支：
#     "proto"     只分析初始补偿投影
#     "optimized" 只分析优化投影
#     "both"      同时分析两者
RUN_MODE = "both"                        # "proto" / "optimized" / "both"

# 点 target 的半径和中心位置。POINT_CENTER_RC 使用 (row, col) 顺序。
POINT_RADIUS_PX = 1
POINT_CENTER_RC = (100, 540)

# 分析图像尺寸。None 表示使用 cfg.n_slice；也可以写成 1080 或 (1080, 1080)。
ANALYSIS_SIZE = None                      # None / 1080 / (1080,1080)

# 输出文件名前缀。
OUTPUT_NAME = "point_psf"

# 是否额外生成高分辨率 dose 可视化，用于更细致观察 PSF 形状。
USE_FINE_DOSE_VIS = True
FINE_DOSE_GRID_SIZE = 2160

# PSF 结果输出根目录，以及是否按 (radius,row) 建立子目录。
PSF_OUTPUT_PARENT_DIR = Path(r"F:\USTC\项目\VAM\PSF")
PSF_OUTPUT_USE_RADIUS_ROW_FOLDER = True   # True -> 文件夹命名为 (radius,row)

# 图像和结果保存开关。
SHOW_FIGURES_OVERRIDE = None
SAVE_FIGURES = True
SAVE_RESULTS = True

# 以下 CFG_* 为临时覆盖参数；None 表示沿用 get_default_config() 的默认值。
CFG_N_SLICE = None
CFG_ANGLE_NUM = None
CFG_RESIN_ALPHA = None
CFG_RESIN_EC = None
CFG_NUM_ITERATIONS = None
CFG_USE_NUMBA = None


# ============================================================
# 基础工具
# ============================================================
def _normalize_shape(size: SizeType) -> Tuple[int, int]:
    """把 int 或 (H, W) 形式的尺寸统一转换为合法的二维 shape。"""
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive.")
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        h, w = int(size[0]), int(size[1])
        if h <= 0 or w <= 0:
            raise ValueError("shape dimensions must be positive.")
        return h, w
    raise TypeError("size must be int or (H, W).")


def _ensure_binary(arr: np.ndarray) -> np.ndarray:
    """把任意数值数组转换成 0/1 二值 mask。"""
    return (np.asarray(arr) > 0).astype(np.uint8)


def _compute_centroid(binary_img: np.ndarray) -> Tuple[float, float]:
    """计算二值区域的几何重心；空区域返回 NaN。"""
    binary = _ensure_binary(binary_img)
    coords = np.argwhere(binary > 0)
    if coords.size == 0:
        return np.nan, np.nan
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _nearest_active_index(binary_profile: np.ndarray, center_idx: int) -> Optional[int]:
    """在一维二值 profile 中寻找距离 center_idx 最近的有效像素。"""
    active = np.flatnonzero(binary_profile > 0)
    if active.size == 0:
        return None
    return int(active[np.argmin(np.abs(active - center_idx))])


def _contiguous_binary_width(profile: np.ndarray, center_idx: int) -> float:
    """计算一维二值 profile 中经过 center_idx 的连续固化宽度。"""
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


def _half_max_width(profile: np.ndarray, peak_idx: Optional[int] = None) -> float:
    """计算一维 dose profile 的半高宽 FWHM。"""
    p = np.asarray(profile, dtype=float).ravel()
    n = p.size
    if n < 3 or not np.isfinite(p).any():
        return np.nan

    if peak_idx is None:
        peak_idx = int(np.nanargmax(p))
    else:
        peak_idx = int(np.clip(peak_idx, 0, n - 1))

    peak_val = p[peak_idx]
    if not np.isfinite(peak_val) or peak_val <= 0:
        if np.nanmax(p) <= 0:
            return np.nan
        peak_idx = int(np.nanargmax(p))
        peak_val = p[peak_idx]
        if not np.isfinite(peak_val) or peak_val <= 0:
            return np.nan

    half_val = peak_val / 2.0
    left_idx = peak_idx
    while left_idx > 0 and p[left_idx] >= half_val:
        left_idx -= 1
    if left_idx == 0 and p[left_idx] >= half_val:
        x_left = 0.0
    else:
        x1, y1 = left_idx, p[left_idx]
        x2, y2 = left_idx + 1, p[left_idx + 1]
        x_left = float(left_idx) + 0.5 if (not (np.isfinite(y1) and np.isfinite(y2)) or y2 == y1) else x1 + (half_val - y1) / (y2 - y1)

    right_idx = peak_idx
    while right_idx < n - 1 and p[right_idx] >= half_val:
        right_idx += 1
    if right_idx == n - 1 and p[right_idx] >= half_val:
        x_right = float(n - 1)
    else:
        x1, y1 = right_idx - 1, p[right_idx - 1]
        x2, y2 = right_idx, p[right_idx]
        x_right = float(right_idx) - 0.5 if (not (np.isfinite(y1) and np.isfinite(y2)) or y2 == y1) else x1 + (half_val - y1) / (y2 - y1)

    width = x_right - x_left
    return float(width) if width >= 0 else np.nan


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _format_scalar_for_folder(value: Any) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):g}"
    return str(value).replace(" ", "")


def _build_psf_output_dir(parent_dir: Path, center_rc: Tuple[int, int], radius_px: int) -> Path:
    if PSF_OUTPUT_USE_RADIUS_ROW_FOLDER:
        folder_name = f"({_format_scalar_for_folder(radius_px)},{_format_scalar_for_folder(center_rc[0])})"
    else:
        folder_name = OUTPUT_NAME
    return Path(parent_dir) / folder_name


def _scale_center_rc(center_rc: Tuple[int, int], from_shape: Tuple[int, int], to_shape: Tuple[int, int]) -> Tuple[int, int]:
    from_h, from_w = int(from_shape[0]), int(from_shape[1])
    to_h, to_w = int(to_shape[0]), int(to_shape[1])
    if from_h <= 0 or from_w <= 0 or to_h <= 0 or to_w <= 0:
        raise ValueError("shape dimensions must be positive.")

    row = int(np.clip(round(float(center_rc[0]) * float(to_h) / float(from_h)), 0, to_h - 1))
    col = int(np.clip(round(float(center_rc[1]) * float(to_w) / float(from_w)), 0, to_w - 1))
    return row, col


def build_point_target(size: SizeType, center_rc: Optional[Tuple[int, int]] = None, radius_px: int = 2) -> np.ndarray:
    h, w = _normalize_shape(size)
    if center_rc is None:
        center_rc = (h // 2, w // 2)

    row_c, col_c = int(center_rc[0]), int(center_rc[1])
    if not (0 <= row_c < h and 0 <= col_c < w):
        raise ValueError("center_rc is out of image bounds.")
    if radius_px < 0:
        raise ValueError("radius_px must be >= 0.")

    rr, cc = np.ogrid[:h, :w]
    mask = (rr - row_c) ** 2 + (cc - col_c) ** 2 <= radius_px ** 2
    target = np.zeros((h, w), dtype=np.uint8)
    target[mask] = 1

    center = h // 2
    circle_mask = (cc - center) ** 2 + (rr - center) ** 2 <= center ** 2
    return (target * circle_mask).astype(np.uint8)


def compute_geometric_metrics(
    target: np.ndarray,
    cured: np.ndarray,
    cfg: Optional[VAMConfig] = None,
    container_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    t = _ensure_binary(target).astype(bool)
    c = _ensure_binary(cured).astype(bool)

    if container_mask is None and cfg is not None:
        container_mask = build_container_mask(t.shape, cfg)
    if container_mask is not None:
        mask = np.asarray(container_mask).astype(bool)
        t = np.logical_and(t, mask)
        c = np.logical_and(c, mask)

    iou = calculate_iou(t, c, cfg=None, container_mask=container_mask)
    overcure = np.logical_and(c, np.logical_not(t))
    undercure = np.logical_and(t, np.logical_not(c))
    target_centroid = _compute_centroid(t.astype(np.uint8))
    cured_centroid = _compute_centroid(c.astype(np.uint8))

    if np.any(np.isnan(target_centroid)) or np.any(np.isnan(cured_centroid)):
        centroid_shift_px = np.nan
    else:
        centroid_shift_px = float(np.hypot(cured_centroid[0] - target_centroid[0], cured_centroid[1] - target_centroid[1]))

    return {
        "iou": iou,
        "overcure_area_px": int(overcure.sum()),
        "undercure_area_px": int(undercure.sum()),
        "centroid_shift_px": centroid_shift_px,
        "target_centroid_rc": target_centroid,
        "cured_centroid_rc": cured_centroid,
    }


def measure_spread_metrics(dose: np.ndarray, cured: np.ndarray, fallback_center_rc: Tuple[int, int]) -> Dict[str, Any]:
    dose = np.asarray(dose, dtype=float)
    cured_bin = _ensure_binary(cured)

    dose_h, dose_w = dose.shape
    cured_h, cured_w = cured_bin.shape
    if np.isfinite(dose).any():
        peak_r, peak_c = map(int, np.unravel_index(np.nanargmax(dose), dose.shape))
    else:
        peak_r, peak_c = _scale_center_rc(fallback_center_rc, from_shape=(cured_h, cured_w), to_shape=(dose_h, dose_w))

    dose_x_profile = dose[peak_r, :]
    dose_y_profile = dose[:, peak_c]
    dose_fwhm_x_px = _half_max_width(dose_x_profile, peak_idx=peak_c)
    dose_fwhm_y_px = _half_max_width(dose_y_profile, peak_idx=peak_r)

    cured_centroid = _compute_centroid(cured_bin)
    if np.any(np.isnan(cured_centroid)):
        cured_r, cured_c = fallback_center_rc
    else:
        cured_r = int(np.clip(round(cured_centroid[0]), 0, cured_h - 1))
        cured_c = int(np.clip(round(cured_centroid[1]), 0, cured_w - 1))

    cured_x_profile = cured_bin[cured_r, :]
    cured_y_profile = cured_bin[:, cured_c]
    cured_width_x_px = _contiguous_binary_width(cured_x_profile, center_idx=cured_c)
    cured_width_y_px = _contiguous_binary_width(cured_y_profile, center_idx=cured_r)

    return {
        "dose_grid_shape": (dose_h, dose_w),
        "cured_grid_shape": (cured_h, cured_w),
        "dose_peak_rc": (peak_r, peak_c),
        "dose_fwhm_x_px": dose_fwhm_x_px,
        "dose_fwhm_y_px": dose_fwhm_y_px,
        "cured_analysis_rc": (cured_r, cured_c),
        "cured_width_x_px": cured_width_x_px,
        "cured_width_y_px": cured_width_y_px,
        "dose_x_profile": dose_x_profile,
        "dose_y_profile": dose_y_profile,
        "cured_x_profile": cured_x_profile,
        "cured_y_profile": cured_y_profile,
    }


def run_point_psf_error_analysis(
    cfg: VAMConfig,
    size: Optional[SizeType] = None,
    center_rc: Optional[Tuple[int, int]] = None,
    radius_px: int = 2,
    run_mode: str = "both",
    show_figures: Optional[bool] = None,
    save_figures: bool = False,
    save_results: bool = True,
    output_name: str = "point_psf",
    fine_dose_grid_size: Optional[int] = None,
    output_parent_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    执行一次点 PSF 误差分析。

    流程：
        1. 构造点状 target。
        2. 生成 proto projection 并模拟 dose/cured。
        3. 按 run_mode 可选执行 optimized 分支。
        4. 计算几何误差、FWHM、固化宽度等指标。
        5. 按开关保存 npz/json/figure 结果。
    """
    if run_mode not in {"proto", "optimized", "both"}:
        raise ValueError("run_mode must be one of {'proto', 'optimized', 'both'}.")

    if size is None:
        size = int(cfg.n_slice)
    h, w = _normalize_shape(size)
    if h != w:
        raise ValueError("Current v1.0 workflow assumes square 2D slice. Please use N or (N, N).")
    if center_rc is None:
        center_rc = (h // 2, w // 2)
    if show_figures is None:
        show_figures = bool(cfg.show_figures)

    # 准备输出目录和点状 target。
    ensure_output_dirs(cfg)
    parent_dir = Path(output_parent_dir) if output_parent_dir is not None else PSF_OUTPUT_PARENT_DIR
    psf_dir = _build_psf_output_dir(parent_dir, center_rc=center_rc, radius_px=radius_px)
    psf_dir.mkdir(parents=True, exist_ok=True)

    # target -> sinogram -> raw projection -> proto projection/dose。
    target = build_point_target(size=(h, w), center_rc=center_rc, radius_px=radius_px).astype(np.float64)
    initial_sinogram = generate_initial_sinogram(target, cfg)
    raw_projection = generate_initial_projection_matrix(initial_sinogram, cfg)
    proto_result = compensate_and_print_initial_projection(raw_projection, target, cfg)

    results: Dict[str, Any] = {
        "run_mode": run_mode,
        "center_rc": center_rc,
        "radius_px": int(radius_px),
        "target": target,
        "initial_sinogram": initial_sinogram,
        "raw_projection": raw_projection,
        "proto_projection": proto_result["projection_matrix"],
        "fine_dose_grid_size": None if fine_dose_grid_size is None else int(fine_dose_grid_size),
    }

    if run_mode in {"proto", "both"}:
        results["proto_dose"] = proto_result["dose_field"]
        results["proto_cured"] = _ensure_binary(proto_result["cured_mask"])
        results["proto_global_k"] = float(proto_result["global_k"])
        results["proto_metrics"] = compute_geometric_metrics(target, results["proto_cured"], cfg=cfg)
        results["proto_spread_metrics"] = measure_spread_metrics(results["proto_dose"], results["proto_cured"], center_rc)
        if fine_dose_grid_size is not None:
            results["proto_dose_vis"] = forward_dose_simulation_visualization(results["proto_projection"], cfg, output_grid_size=fine_dose_grid_size)
            results["proto_spread_metrics_vis"] = measure_spread_metrics(results["proto_dose_vis"], results["proto_cured"], center_rc)

    if run_mode in {"optimized", "both"}:
        opt_result = optimize_projection_matrix(results["proto_projection"], target, cfg)
        results["optimized_projection"] = opt_result["projection_matrix"]
        results["optimized_dose"] = opt_result["dose_field"]
        results["optimized_cured"] = _ensure_binary(opt_result["cured_mask"])
        results["optimized_history"] = opt_result["history"]
        results["optimized_metrics"] = compute_geometric_metrics(target, results["optimized_cured"], cfg=cfg)
        results["optimized_spread_metrics"] = measure_spread_metrics(results["optimized_dose"], results["optimized_cured"], center_rc)
        if fine_dose_grid_size is not None:
            results["optimized_dose_vis"] = forward_dose_simulation_visualization(results["optimized_projection"], cfg, output_grid_size=fine_dose_grid_size)
            results["optimized_spread_metrics_vis"] = measure_spread_metrics(results["optimized_dose_vis"], results["optimized_cured"], center_rc)

    if save_results:
        save_projection_npz(
            psf_dir / f"{output_name}_raw_projection.npz",
            raw_projection,
            cfg,
            target_mask=target,
            extra_metadata={
                "analysis_type": np.array(["point_psf_raw"], dtype=object),
                "point_center_rc": np.array(center_rc),
                "point_radius_px": np.array(radius_px),
            },
        )
        save_projection_npz(
            psf_dir / f"{output_name}_proto_projection.npz",
            results["proto_projection"],
            cfg,
            target_mask=target,
            extra_metadata={
                "analysis_type": np.array(["point_psf_proto"], dtype=object),
                "point_center_rc": np.array(center_rc),
                "point_radius_px": np.array(radius_px),
            },
        )
        if "optimized_projection" in results:
            save_projection_npz(
                psf_dir / f"{output_name}_optimized_projection.npz",
                results["optimized_projection"],
                cfg,
                target_mask=target,
                extra_metadata={
                    "analysis_type": np.array(["point_psf_optimized"], dtype=object),
                    "point_center_rc": np.array(center_rc),
                    "point_radius_px": np.array(radius_px),
                },
            )

        summary = {
            "run_mode": run_mode,
            "center_rc": center_rc,
            "radius_px": int(radius_px),
            "fine_dose_grid_size": None if fine_dose_grid_size is None else int(fine_dose_grid_size),
        }
        if "proto_metrics" in results:
            summary["proto_metrics"] = results["proto_metrics"]
            summary["proto_spread_metrics"] = {k: v for k, v in results["proto_spread_metrics"].items() if not k.endswith("_profile")}
            if "proto_spread_metrics_vis" in results:
                summary["proto_spread_metrics_vis"] = {k: v for k, v in results["proto_spread_metrics_vis"].items() if not k.endswith("_profile")}
            summary["proto_global_k"] = results["proto_global_k"]
        if "optimized_metrics" in results:
            summary["optimized_metrics"] = results["optimized_metrics"]
            summary["optimized_spread_metrics"] = {k: v for k, v in results["optimized_spread_metrics"].items() if not k.endswith("_profile")}
            if "optimized_spread_metrics_vis" in results:
                summary["optimized_spread_metrics_vis"] = {k: v for k, v in results["optimized_spread_metrics_vis"].items() if not k.endswith("_profile")}
            summary["optimized_history"] = results["optimized_history"]

        with open(psf_dir / f"{output_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2)

    results["figures"] = plot_point_psf_analysis(
        results=results,
        cfg=cfg,
        center_rc=center_rc,
        show_figures=show_figures,
        save_figures=save_figures,
        save_dir=psf_dir if save_figures else None,
        figure_prefix=output_name,
    )
    results["output_dir"] = psf_dir
    return results


def _branch_error_maps(target: np.ndarray, cured: np.ndarray, container_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_bin = _ensure_binary(target).astype(bool)
    cured_bin = _ensure_binary(cured).astype(bool)
    if container_mask is not None:
        mask = np.asarray(container_mask).astype(bool)
        target_bin = np.logical_and(target_bin, mask)
        cured_bin = np.logical_and(cured_bin, mask)

    overcure = np.logical_and(cured_bin, np.logical_not(target_bin)).astype(np.uint8)
    undercure = np.logical_and(target_bin, np.logical_not(cured_bin)).astype(np.uint8)
    signed = np.zeros_like(target_bin, dtype=int)
    signed[undercure == 1] = -1
    signed[overcure == 1] = 1
    return overcure, undercure, signed


def _imshow_matrix(ax, img: np.ndarray, title: str, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    return im


def _projection_quantization_summary(proj: np.ndarray, cfg: VAMConfig) -> Dict[str, Any]:
    proj = np.asarray(proj, dtype=float)
    proj_max = float(np.nanmax(proj)) if proj.size > 0 and np.isfinite(proj).any() else 0.0
    unique_vals = np.unique(np.round(proj, decimals=12)) if proj.size > 0 else np.array([], dtype=float)

    summary: Dict[str, Any] = {
        "proj_max": proj_max,
        "is_quantized": False,
        "configured_steps": None,
        "configured_levels": None,
        "effective_step": None,
        "used_level_count": int(unique_vals.size),
        "used_step_count": max(int(unique_vals.size) - 1, 0),
    }
    if cfg.projection_quantization_enabled:
        K = int(cfg.projection_quantization_max_index)
        summary["configured_steps"] = K
        summary["configured_levels"] = K + 1
        if K > 0 and proj_max > 0:
            eff_step = proj_max / float(K)
            level_idx = np.rint(np.clip(proj, 0.0, None) / eff_step)
            level_idx = np.clip(level_idx, 0, K).astype(int)
            used_indices = np.unique(level_idx)
            reconstructed = level_idx.astype(float) * eff_step
            summary["effective_step"] = float(eff_step)
            summary["used_level_count"] = int(used_indices.size)
            summary["used_step_count"] = max(int(used_indices.size) - 1, 0)
            summary["is_quantized"] = np.allclose(np.clip(proj, 0.0, None), reconstructed, rtol=1e-7, atol=max(1e-12, eff_step * 1e-7))
        else:
            summary["effective_step"] = 0.0
            summary["is_quantized"] = True
    return summary


def _format_projection_title(base_title: str, proj: np.ndarray, cfg: VAMConfig) -> str:
    info = _projection_quantization_summary(proj, cfg)
    lines = [base_title]
    if cfg.projection_quantization_enabled:
        status = "quantized" if info["is_quantized"] else "continuous / pre-quantization"
        lines.append(f"{status} | configured steps={info['configured_steps']}, levels={info['configured_levels']}")
        if info["effective_step"] is not None:
            lines.append(f"effective step={info['effective_step']:.6g} | used steps={info['used_step_count']}")
    else:
        lines.append(f"continuous | unique values={info['used_level_count']}")
    return "\n".join(lines)


def _compute_focus_roi(target: np.ndarray, cured: Optional[np.ndarray] = None,
                       center_rc: Optional[Tuple[int, int]] = None,
                       min_half_width: int = 24, pad: int = 12) -> Tuple[slice, slice]:
    target_bin = _ensure_binary(target).astype(bool)
    active = target_bin.copy()
    if cured is not None:
        active |= _ensure_binary(cured).astype(bool)

    coords = np.argwhere(active)
    h, w = target_bin.shape
    if coords.size == 0:
        if center_rc is None:
            center_rc = (h // 2, w // 2)
        r0 = int(np.clip(center_rc[0], 0, h - 1))
        c0 = int(np.clip(center_rc[1], 0, w - 1))
        return (
            slice(max(0, r0 - min_half_width), min(h, r0 + min_half_width + 1)),
            slice(max(0, c0 - min_half_width), min(w, c0 + min_half_width + 1)),
        )

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    r_min = max(0, int(r_min) - pad)
    r_max = min(h, int(r_max) + pad + 1)
    c_min = max(0, int(c_min) - pad)
    c_max = min(w, int(c_max) + pad + 1)

    if (r_max - r_min) < 2 * min_half_width + 1:
        r_center = (r_min + r_max - 1) // 2
        r_min = max(0, r_center - min_half_width)
        r_max = min(h, r_center + min_half_width + 1)
    if (c_max - c_min) < 2 * min_half_width + 1:
        c_center = (c_min + c_max - 1) // 2
        c_min = max(0, c_center - min_half_width)
        c_max = min(w, c_center + min_half_width + 1)
    return slice(r_min, r_max), slice(c_min, c_max)


def _imshow_matrix_roi(ax, img: np.ndarray, title: str, roi: Tuple[slice, slice], cmap=None, vmin=None, vmax=None):
    sub = np.asarray(img)[roi[0], roi[1]]
    im = ax.imshow(sub, cmap=cmap, interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(f"col {roi[1].start}:{roi[1].stop - 1}")
    ax.set_ylabel(f"row {roi[0].start}:{roi[0].stop - 1}")
    return im


def _imshow_projection_heatmap(ax, proj: np.ndarray, title: str, cfg: VAMConfig):
    proj = np.asarray(proj, dtype=float)
    im = ax.imshow(proj, cmap="viridis", interpolation="nearest", origin="lower", aspect="auto")
    ax.set_title(_format_projection_title(title, proj, cfg), fontsize=10)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Detector pixel index")
    return im


def _select_dose_for_visualization(results: Dict[str, Any], branch_name: str) -> tuple[np.ndarray, str, Dict[str, Any]]:
    vis_key = f"{branch_name}_dose_vis"
    vis_sm_key = f"{branch_name}_spread_metrics_vis"
    base_key = f"{branch_name}_dose"
    base_sm_key = f"{branch_name}_spread_metrics"
    if vis_key in results and vis_sm_key in results:
        dose_vis = np.asarray(results[vis_key], dtype=float)
        grid_h, grid_w = dose_vis.shape
        return dose_vis, f"{branch_name.capitalize()} dose (fine grid {grid_h}×{grid_w})", results[vis_sm_key]
    dose = np.asarray(results[base_key], dtype=float)
    grid_h, grid_w = dose.shape
    return dose, f"{branch_name.capitalize()} dose ({grid_h}×{grid_w})", results[base_sm_key]


def plot_point_psf_analysis(
    results: Dict[str, Any],
    cfg: VAMConfig,
    center_rc: Tuple[int, int],
    show_figures: bool = True,
    save_figures: bool = False,
    save_dir: Optional[Path] = None,
    figure_prefix: str = "point_psf",
) -> Dict[str, plt.Figure]:
    figs: Dict[str, plt.Figure] = {}
    target = results["target"]
    container_mask = build_container_mask(target.shape, cfg)
    active = []
    if "proto_cured" in results:
        active.append("proto")
    if "optimized_cured" in results:
        active.append("optimized")

    if save_figures and save_dir is None:
        raise ValueError("save_dir must be provided when save_figures=True.")
    if save_figures:
        save_dir.mkdir(parents=True, exist_ok=True)

    if len(active) == 2:
        fig1, axes = plt.subplots(1, 4, figsize=(15, 4.2))
        _imshow_matrix(axes[0], target, "Point Target", cmap="gray")
        _imshow_matrix(axes[1], results["proto_cured"], "Proto cured", cmap="gray")
        _imshow_matrix(axes[2], results["optimized_cured"], "Optimized cured", cmap="gray")
        axes[3].axis("off")
        axes[3].text(
            0.02, 0.95,
            f"Center (row, col): {center_rc}\n"
            f"Radius: {results['radius_px']} px\n\n"
            f"Proto IoU: {results['proto_metrics']['iou']:.4f}\n"
            f"Optimized IoU: {results['optimized_metrics']['iou']:.4f}",
            va="top", ha="left", fontsize=10,
        )
        fig1.suptitle("Figure 1 - Structure comparison", fontsize=14)
        fig1.tight_layout()
    else:
        branch = active[0]
        fig1, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        _imshow_matrix(axes[0], target, "Point Target", cmap="gray")
        _imshow_matrix(axes[1], results[f"{branch}_cured"], f"{branch.capitalize()} cured", cmap="gray")
        fig1.suptitle("Figure 1 - Structure comparison", fontsize=14)
        fig1.tight_layout()
    figs["structure"] = fig1

    heatmap_items = [("raw_projection", "Raw projection")]
    if "proto_projection" in results:
        heatmap_items.append(("proto_projection", "Proto projection"))
    if "optimized_projection" in results:
        heatmap_items.append(("optimized_projection", "Optimized projection"))

    figp, axes = plt.subplots(1, len(heatmap_items), figsize=(5.8 * len(heatmap_items), 5.2), squeeze=False)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, heatmap_items):
        im = _imshow_projection_heatmap(ax, results[key], title, cfg)
        figp.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figp.suptitle("Figure 2 - Projection matrix heatmaps (with discretization summary)", fontsize=14)
    figp.tight_layout()
    figs["projection_heatmap"] = figp

    if len(active) == 2:
        proto_dose_vis, proto_dose_title, _ = _select_dose_for_visualization(results, "proto")
        optimized_dose_vis, optimized_dose_title, _ = _select_dose_for_visualization(results, "optimized")
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = _imshow_matrix(axes[0], proto_dose_vis, proto_dose_title)
        fig2.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = _imshow_matrix(axes[1], optimized_dose_vis, optimized_dose_title)
        fig2.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig2.suptitle("Figure 3 - Dose maps", fontsize=14)
        fig2.tight_layout()
    else:
        branch = active[0]
        branch_dose_vis, branch_dose_title, _ = _select_dose_for_visualization(results, branch)
        fig2, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        im = _imshow_matrix(ax, branch_dose_vis, branch_dose_title)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig2.suptitle("Figure 3 - Dose map", fontsize=14)
        fig2.tight_layout()
    figs["dose"] = fig2

    cmap_signed = ListedColormap(["red", "black", "green"])
    cmap_over = ListedColormap(["black", "green"])
    cmap_under = ListedColormap(["black", "red"])
    if len(active) == 2:
        fig3, axes = plt.subplots(2, 3, figsize=(12, 8))
        p_over, p_under, p_signed = _branch_error_maps(target, results["proto_cured"], container_mask=container_mask)
        o_over, o_under, o_signed = _branch_error_maps(target, results["optimized_cured"], container_mask=container_mask)
        p_roi = _compute_focus_roi(target, results["proto_cured"], center_rc=center_rc)
        o_roi = _compute_focus_roi(target, results["optimized_cured"], center_rc=center_rc)
        _imshow_matrix_roi(axes[0, 0], p_over, f"Proto overcure\narea={int(p_over.sum())} px", roi=p_roi, cmap=cmap_over, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[0, 1], p_under, f"Proto undercure\narea={int(p_under.sum())} px", roi=p_roi, cmap=cmap_under, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[0, 2], p_signed + 1, "Proto signed error\nzoomed ROI", roi=p_roi, cmap=cmap_signed, vmin=0, vmax=2)
        _imshow_matrix_roi(axes[1, 0], o_over, f"Optimized overcure\narea={int(o_over.sum())} px", roi=o_roi, cmap=cmap_over, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[1, 1], o_under, f"Optimized undercure\narea={int(o_under.sum())} px", roi=o_roi, cmap=cmap_under, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[1, 2], o_signed + 1, "Optimized signed error\nzoomed ROI", roi=o_roi, cmap=cmap_signed, vmin=0, vmax=2)
        fig3.suptitle("Figure 4 - Error decomposition (zoomed ROI within container)", fontsize=14)
        fig3.tight_layout()
    else:
        branch = active[0]
        over, under, signed = _branch_error_maps(target, results[f"{branch}_cured"], container_mask=container_mask)
        roi = _compute_focus_roi(target, results[f"{branch}_cured"], center_rc=center_rc)
        fig3, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        _imshow_matrix_roi(axes[0], over, f"{branch.capitalize()} overcure\narea={int(over.sum())} px", roi=roi, cmap=cmap_over, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[1], under, f"{branch.capitalize()} undercure\narea={int(under.sum())} px", roi=roi, cmap=cmap_under, vmin=0, vmax=1)
        _imshow_matrix_roi(axes[2], signed + 1, f"{branch.capitalize()} signed error\nzoomed ROI", roi=roi, cmap=cmap_signed, vmin=0, vmax=2)
        fig3.suptitle("Figure 4 - Error decomposition (zoomed ROI within container)", fontsize=14)
        fig3.tight_layout()
    figs["error"] = fig3

    def _plot_dose_profile_pair(ax_x, ax_y, branch_name: str) -> None:
        _, _, sm = _select_dose_for_visualization(results, branch_name)
        dx = sm["dose_x_profile"]
        dy = sm["dose_y_profile"]
        peak_r, peak_c = sm["dose_peak_rc"]
        dose_h, dose_w = sm["dose_grid_shape"]
        ax_x.plot(dx)
        ax_x.axvline(peak_c, linestyle="--")
        ax_x.set_title(f"{branch_name.capitalize()} dose X-profile ({dose_w} px)\nFWHM_x = {sm['dose_fwhm_x_px']:.3f} px")
        ax_x.set_xlabel("Column index")
        ax_x.set_ylabel("Dose")
        ax_y.plot(dy)
        ax_y.axvline(peak_r, linestyle="--")
        ax_y.set_title(f"{branch_name.capitalize()} dose Y-profile ({dose_h} px)\nFWHM_y = {sm['dose_fwhm_y_px']:.3f} px")
        ax_y.set_xlabel("Row index")
        ax_y.set_ylabel("Dose")

    def _plot_cured_profile_pair(ax_x, ax_y, branch_name: str) -> None:
        sm = results[f"{branch_name}_spread_metrics"]
        cx = np.asarray(sm["cured_x_profile"], dtype=float)
        cy = np.asarray(sm["cured_y_profile"], dtype=float)
        cured_r, cured_c = sm["cured_analysis_rc"]
        ax_x.step(np.arange(cx.size), cx, where="mid")
        ax_x.axvline(cured_c, linestyle="--")
        ax_x.set_ylim(-0.05, 1.05)
        ax_x.set_title(f"{branch_name.capitalize()} cured X-profile\nwidth_x = {sm['cured_width_x_px']:.3f} px")
        ax_x.set_xlabel("Column index")
        ax_x.set_ylabel("Cured")
        ax_y.step(np.arange(cy.size), cy, where="mid")
        ax_y.axvline(cured_r, linestyle="--")
        ax_y.set_ylim(-0.05, 1.05)
        ax_y.set_title(f"{branch_name.capitalize()} cured Y-profile\nwidth_y = {sm['cured_width_y_px']:.3f} px")
        ax_y.set_xlabel("Row index")
        ax_y.set_ylabel("Cured")

    if len(active) == 2:
        fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
        _plot_dose_profile_pair(axes[0, 0], axes[0, 1], "proto")
        _plot_dose_profile_pair(axes[1, 0], axes[1, 1], "optimized")
        fig4.suptitle("Figure 5 - Dose profiles and spread metrics", fontsize=14)
        fig4.tight_layout()
    else:
        branch = active[0]
        fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
        _plot_dose_profile_pair(axes[0, 0], axes[0, 1], branch)
        _plot_cured_profile_pair(axes[1, 0], axes[1, 1], branch)
        fig4.suptitle("Figure 5 - Dose and cured profile analysis", fontsize=14)
        fig4.tight_layout()
    figs["profiles"] = fig4

    if save_figures:
        for key, fig in figs.items():
            fig.savefig(save_dir / f"{figure_prefix}_{key}.png", dpi=300, bbox_inches="tight")

    if show_figures:
        plt.show()
    else:
        plt.close("all")
    return figs


if __name__ == "__main__":
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

    show_figures = cfg.show_figures if SHOW_FIGURES_OVERRIDE is None else SHOW_FIGURES_OVERRIDE
    results = run_point_psf_error_analysis(
        cfg=cfg,
        size=ANALYSIS_SIZE,
        center_rc=POINT_CENTER_RC,
        radius_px=POINT_RADIUS_PX,
        run_mode=RUN_MODE,
        show_figures=show_figures,
        save_figures=SAVE_FIGURES,
        save_results=SAVE_RESULTS,
        output_name=OUTPUT_NAME,
        fine_dose_grid_size=FINE_DOSE_GRID_SIZE if USE_FINE_DOSE_VIS else None,
        output_parent_dir=PSF_OUTPUT_PARENT_DIR,
    )

    print("=" * 60)
    print(f"误差分析完成，输出目录：{results['output_dir']}")
    print(f"run_mode = {RUN_MODE}")
    print(f"point center = {results['center_rc']}, radius = {results['radius_px']} px")
    if "proto_metrics" in results:
        print("\n[Proto metrics]")
        print(results["proto_metrics"])
        print(results["proto_spread_metrics"])
    if "optimized_metrics" in results:
        print("\n[Optimized metrics]")
        print(results["optimized_metrics"])
        print(results["optimized_spread_metrics"])
