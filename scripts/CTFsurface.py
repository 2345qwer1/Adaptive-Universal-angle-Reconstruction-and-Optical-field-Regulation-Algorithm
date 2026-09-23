"""从 a1/a2 CTF detail CSV 生成平滑三维 CTF surface 图。

曲面高度与常规颜色均表示 ``CTF_dose_mean``。原始数据中
``all_over_Ec > ALL_OVER_EC_THRESHOLD`` 的区域以红色表面显示，
但不改变其 CTF 高度。输入 CSV 不会被修改。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# 用户可调参数：输入、输出与绘图风格
# =============================================================================

# 八个输入 CSV；列表顺序与 OUTPUT_PNG_PATHS 一一对应。
INPUT_CSV_PATHS = [
    Path(r"F:\USTC\项目\VAM\repair_a2\a1_00.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a1_25.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a1_50.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a1_75.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a2_00.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a2_25.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a2_50.csv"),
    Path(r"F:\USTC\项目\VAM\repair_a2\a2_75.csv"),
]

# PNG 输出；SVG 自动使用相同文件名和 .svg 后缀。
OUTPUT_PNG_PATHS = [
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a1_00_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a1_25_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a1_50_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a1_75_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a2_00_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a2_25_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a2_50_surface.png"),
    Path(r"F:\USTC\项目\VAM\repair_a2\figure\a2_75_surface.png"),
]

# 保持与 heatmap_a2.py 一致的 CTF 色标和 all_over_Ec 判定。
COLOR_MIN = 0.0
COLOR_MAX = 0.8
COLORMAP_NAME = "viridis"
ALL_OVER_EC_THRESHOLD = 0.15
ALL_OVER_EC_COLOR = "#D62728"

# m=1 的空 CTF 只在绘图阶段视为 0，不写回 CSV。
M1_EMPTY_CTF_AS_ZERO = True

# 曲面平滑参数。每个原始网格间隔细分为该值，GAUSSIAN_SIGMA 以原始网格为单位。
SURFACE_REFINEMENT = 4
GAUSSIAN_SIGMA = 0.75

# 输出与视角参数。
FIGURE_WIDTH_INCH = 7.0
FIGURE_HEIGHT_INCH = 5.6
OUTPUT_DPI = 220
VIEW_ELEVATION_DEG = 28.0
VIEW_AZIMUTH_DEG = -58.0


REQUIRED_FIELDS = {"dose_type", "m", "r_over_R", "CTF_dose_mean", "all_over_Ec"}
ALL_OVER_EC_FIELD = "all_over_Ec"


def _read_ctf_csv(csv_path: Path) -> list[dict[str, Any]]:
    """读取并验证 heatmap_a2.py 所用的 CTF detail CSV 字段。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在：{csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_fields = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f"{csv_path} 缺少必要字段：{', '.join(missing_fields)}")
        rows: list[dict[str, Any]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            try:
                m_value = float(raw_row["m"])
                ctf_text = str(raw_row["CTF_dose_mean"]).strip()
                ctf_value = float(ctf_text) if ctf_text else float("nan")
                if not np.isfinite(ctf_value) and M1_EMPTY_CTF_AS_ZERO and m_value == 1.0:
                    ctf_value = 0.0
                all_over_ec_text = str(raw_row[ALL_OVER_EC_FIELD]).strip()
                all_over_ec = float(all_over_ec_text) if all_over_ec_text else float("nan")
                rows.append({
                    "dose_type": str(raw_row["dose_type"]).strip(),
                    "m": m_value,
                    "r_over_R": float(raw_row["r_over_R"]),
                    "CTF_dose_mean": ctf_value,
                    ALL_OVER_EC_FIELD: all_over_ec,
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path} 第 {line_number} 行包含无效数据：{exc}") from exc
    if not rows:
        raise ValueError(f"输入 CSV 没有数据行：{csv_path}")
    return rows


def _build_matrices(
    rows: list[dict[str, Any]], dose_type: str, m_values: list[float], r_values: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """构造 (m, r/R) 矩阵，保留原始无效 CTF 为 NaN。"""
    ctf = np.full((len(m_values), len(r_values)), np.nan, dtype=float)
    all_over_ec = np.full_like(ctf, np.nan)
    m_index = {value: index for index, value in enumerate(m_values)}
    r_index = {value: index for index, value in enumerate(r_values)}
    occupied: set[tuple[int, int]] = set()
    for row in rows:
        if row["dose_type"] != dose_type:
            continue
        cell = (m_index[row["m"]], r_index[row["r_over_R"]])
        if cell in occupied:
            raise ValueError(f"重复网格：dose_type={dose_type}, m={row['m']}, r/R={row['r_over_R']}")
        occupied.add(cell)
        ctf[cell] = row["CTF_dose_mean"]
        all_over_ec[cell] = row[ALL_OVER_EC_FIELD]
    return ctf, all_over_ec


def _normalized_gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """仅用有限值参与平滑，未被数据支持的位置继续保留 NaN。"""
    valid = np.isfinite(values)
    weighted_values = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    result = np.full(values.shape, np.nan, dtype=float)
    supported = weights > 1e-8
    result[supported] = weighted_values[supported] / weights[supported]
    return result


def _refine_surface(
    ctf: np.ndarray, all_over_ec: np.ndarray, m_values: list[float], r_values: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """平滑 CTF 后线性细分；红色掩膜以原始判定的最近邻扩展。"""
    if len(m_values) < 2 or len(r_values) < 2:
        raise ValueError("三维曲面至少需要两个 m 和两个 r/R 网格点。")
    smooth_ctf = _normalized_gaussian_smooth(ctf, GAUSSIAN_SIGMA)
    refined_m = np.linspace(m_values[0], m_values[-1], (len(m_values) - 1) * SURFACE_REFINEMENT + 1)
    refined_r = np.linspace(r_values[0], r_values[-1], (len(r_values) - 1) * SURFACE_REFINEMENT + 1)
    m_grid, r_grid = np.meshgrid(refined_m, refined_r, indexing="ij")
    query_points = np.column_stack((m_grid.ravel(), r_grid.ravel()))
    z_interpolator = RegularGridInterpolator((m_values, r_values), smooth_ctf, bounds_error=False, fill_value=np.nan)
    z_grid = z_interpolator(query_points).reshape(m_grid.shape)
    red_mask = np.isfinite(all_over_ec) & (all_over_ec > ALL_OVER_EC_THRESHOLD)
    red_interpolator = RegularGridInterpolator((m_values, r_values), red_mask.astype(float), method="nearest", bounds_error=False, fill_value=0.0)
    red_grid = red_interpolator(query_points).reshape(m_grid.shape) > 0.5
    return r_grid, m_grid, z_grid, red_grid


def generate_surface(csv_path: Path, output_png: Path) -> tuple[Path, int]:
    """从一个 CSV 生成带红色 all_over_Ec 区域的 PNG 与 SVG。"""
    rows = _read_ctf_csv(csv_path)
    dose_types = sorted({row["dose_type"] for row in rows})
    m_values = sorted({float(row["m"]) for row in rows})
    r_values = sorted({float(row["r_over_R"]) for row in rows})
    if not dose_types:
        raise ValueError(f"{csv_path} 不包含 dose_type。")

    fig = plt.figure(figsize=(FIGURE_WIDTH_INCH * len(dose_types), FIGURE_HEIGHT_INCH), constrained_layout=True)
    cmap = plt.get_cmap(COLORMAP_NAME)
    norm = colors.Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX, clip=True)
    red_rgba = colors.to_rgba(ALL_OVER_EC_COLOR)
    red_cell_count = 0
    axes = []
    for index, dose_type in enumerate(dose_types, start=1):
        ax = fig.add_subplot(1, len(dose_types), index, projection="3d")
        axes.append(ax)
        ctf, all_over_ec = _build_matrices(rows, dose_type, m_values, r_values)
        r_grid, m_grid, z_grid, red_grid = _refine_surface(ctf, all_over_ec, m_values, r_values)
        facecolors = cmap(norm(np.nan_to_num(z_grid, nan=COLOR_MIN)))
        facecolors[red_grid] = red_rgba
        facecolors[~np.isfinite(z_grid), 3] = 0.0
        ax.plot_surface(r_grid, m_grid, z_grid, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
        ax.set(title=f"{csv_path.stem} | {dose_type}", xlabel="r/R", ylabel="m / px", zlabel="CTF_dose_mean")
        ax.set_zlim(COLOR_MIN, COLOR_MAX)
        ax.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)
        ax.set_box_aspect((1.25, 1.35, 0.65))
        red_cell_count += int(np.count_nonzero(np.isfinite(all_over_ec) & (all_over_ec > ALL_OVER_EC_THRESHOLD)))

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=axes, fraction=0.035, pad=0.06)
    colorbar.set_label("CTF_dose_mean (red: all_over_Ec > 0.15)")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight")
    output_svg = output_png.with_suffix(".svg")
    fig.savefig(output_svg, format="svg", dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_svg, red_cell_count


def main() -> None:
    if len(INPUT_CSV_PATHS) != len(OUTPUT_PNG_PATHS):
        raise ValueError("INPUT_CSV_PATHS 与 OUTPUT_PNG_PATHS 的数量必须相等。")
    if SURFACE_REFINEMENT < 1 or GAUSSIAN_SIGMA < 0:
        raise ValueError("SURFACE_REFINEMENT 必须不小于 1，GAUSSIAN_SIGMA 必须不小于 0。")
    for csv_path, output_png in zip(INPUT_CSV_PATHS, OUTPUT_PNG_PATHS):
        output_svg, red_count = generate_surface(csv_path, output_png)
        print(f"[surface] input={csv_path} | png={output_png} | svg={output_svg} | red_original_cells={red_count}")


if __name__ == "__main__":
    main()
