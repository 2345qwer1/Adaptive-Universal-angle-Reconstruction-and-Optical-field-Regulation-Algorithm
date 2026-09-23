"""
从一个或多个 CTF detail CSV 生成对应的 CTF_dose heatmap PNG。

直接使用脚本顶部绝对路径：
    python scripts/heatmap.py

使用命令行临时覆盖：
    python scripts/heatmap.py \
        --input-csv "25 CTF.csv" "50 CTF.csv" "75 CTF.csv" \
        --output-png "25 CTF.png" "50 CTF.png" "75 CTF.png"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text


# =============================================================================
# 用户可调参数
# =============================================================================

# 输入 CTF CSV 与输出 PNG 的绝对路径。
# 两个列表按下标一一对应；命令行参数可以临时覆盖这些默认路径。
INPUT_CSV_PATHS = [
    Path(
        r"F:\USTC\项目\VAM\repair_a2\a1_00.csv"
    ),
    Path(
       r"F:\USTC\项目\VAM\repair_a2\a1_25.csv"
    ),
    Path(
       r"F:\USTC\项目\VAM\repair_a2\a1_50.csv"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\a1_75.csv"
    ),
]

OUTPUT_PNG_PATHS = [
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A200.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A225.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A250.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A275.png"
    ),
]

# CTF_dose heatmap 的固定色值范围。
COLOR_MIN = 0.0

# 是否在输出 PNG 的同时保存同名 SVG。
SAVE_SVG = True
SVG_HIDE_VISIBLE_NUMBERS = True

# CTF heatmap 的 SVG 主纵轴仅保留这些 m 位置的刻度线，不显示数字标签。
SVG_HEATMAP_M_TICKS = (27, 54, 81, 108)

SAVE_W_CURVE_PNG = True

# 仅在绘图和 W 计算阶段把 m=1 的空 CTF 当作 0；不会回写输入 CSV。
M1_EMPTY_CTF_AS_ZERO = True

# 四个衰减率 detail CSV 位于同一目录时，不应共同覆盖一个 summary CSV。
UPDATE_RESOLUTION_W_SUMMARY = False

# 四个衰减率合并为同一张 W 曲线；后续主流程不再生成四张单独 W 图。
OUTPUT_COMBINED_W_CURVE_PNG = Path(
    r"F:\USTC\项目\VAM\repair_a2\data\a2_w_vs_r.png"
)
ATTENUATION_LABELS = ("0%", "25%", "50%", "75%")
W_CURVE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
W_CURVE_LINE_STYLES = ("--", "--", "--", "--")
W_CURVE_MARKERS = ("o", "s", "^", "D")

OUTPUT_W_CURVE_PNG_PATHS = [
        Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A200w.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A225w.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A250w.png"
    ),
    Path(
        r"F:\USTC\项目\VAM\repair_a2\data\A275w.png"
    ),
]

COLOR_MAX = 0.8

# 沿用原始 heatmap 的 viridis 配色；缺失值显示为灰色。
COLORMAP_NAME = "viridis"
MISSING_VALUE_COLOR = "#d9d9d9"

# all_over_Ec 严格超过阈值时，为对应格子添加简单纯色覆盖。
SHOW_ALL_OVER_EC_OVERLAY = True

# 覆盖风格：
#     "dark"：暗黑色覆盖
#     "dark_red"：暗红色覆盖
BACKGROUND_OVERLAY_STYLE = "dark_red" 

# all_over_Ec 严格大于该值时添加覆盖；等于该值时不覆盖。
# w 命中所要求的 CTF_dose_mean 阈值；必须严格大于该值
ALL_OVER_EC_THRESHOLD = 0.15
W_CURVE_CTF_THRESHOLD = 0.2

# 两种覆盖风格的颜色与统一透明度。
BACKGROUND_OVERLAY_DARK_COLOR = "#202020"
BACKGROUND_OVERLAY_DARK_RED_COLOR = "#7A1F1F"
BACKGROUND_OVERLAY_ALPHA = 1

# PNG 输出参数。
FIGURE_WIDTH_INCH = 7.0
FIGURE_HEIGHT_INCH = 4.8
OUTPUT_DPI = 200
W_CURVE_WIDTH_INCH = 3.5
W_CURVE_HEIGHT_INCH = 2.85
W_CURVE_DPI = 300
W_CURVE_LINE_WIDTH = 1.5
W_CURVE_MARKER_SIZE = 4.0
MISSING_W_MARKER = "x"
MISSING_W_MARKER_COLOR = "#7A1F1F"
MISSING_W_MARKER_SIZE = 8.0
MISSING_W_MARKER_EDGE_WIDTH = 2.0


# =============================================================================
# CSV 字段和数据读取
# =============================================================================

REQUIRED_FIELDS = {
    "dose_type",
    "m",
    "r_over_R",
    "CTF_dose_mean",
    "all_over_Ec",
}
ALL_OVER_EC_FIELD = "all_over_Ec"
SUMMARY_FIELDS = [
    "structure_type",
    "r_over_R",
    "r_px",
    "w_proto_px",
    "w_optimized_px",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取一个或多个 CTF detail CSV，并使用固定 0–0.5 色尺生成对应 PNG heatmap。"
        )
    )
    parser.add_argument(
        "--input-csv",
        nargs="+",
        metavar="CSV",
        help="一个或多个输入 CTF CSV 路径；与 --output-png 同时提供时覆盖脚本顶部配置。",
    )
    parser.add_argument(
        "--output-png",
        nargs="+",
        metavar="PNG",
        help="与输入 CSV 按顺序一一对应的输出 PNG 路径；与 --input-csv 同时提供。",
    )
    parser.add_argument(
        "--no-overlay",
        "--no-red-overlay",
        dest="show_overlay",
        action="store_false",
        default=SHOW_ALL_OVER_EC_OVERLAY,
        help="不绘制 all_over_Ec 超过阈值格子的覆盖效果。",
    )
    args = parser.parse_args()

    input_was_provided = args.input_csv is not None
    output_was_provided = args.output_png is not None
    if input_was_provided != output_was_provided:
        parser.error("--input-csv 与 --output-png 必须同时提供，或同时省略。")

    if input_was_provided:
        args.input_csv = [Path(value) for value in args.input_csv]
        args.output_png = [Path(value) for value in args.output_png]
    else:
        args.input_csv = list(INPUT_CSV_PATHS)
        args.output_png = list(OUTPUT_PNG_PATHS)
        non_absolute_paths = [
            path
            for path in [*args.input_csv, *args.output_png]
            if not path.is_absolute()
        ]
        if non_absolute_paths:
            parser.error(
                "脚本顶部 INPUT_CSV_PATHS 和 OUTPUT_PNG_PATHS 必须使用绝对路径："
                + ", ".join(str(path) for path in non_absolute_paths)
            )

    if not args.input_csv:
        parser.error("至少需要配置一组输入 CSV 和输出 PNG 路径。")
    if len(args.input_csv) != len(args.output_png):
        parser.error(
            "--input-csv 与 --output-png 的路径数量必须相等；"
            f"当前分别为 {len(args.input_csv)} 和 {len(args.output_png)}。"
        )
    return args


def _resolve_overlay_color(style: str) -> str:
    colors = {
        "dark": BACKGROUND_OVERLAY_DARK_COLOR,
        "dark_red": BACKGROUND_OVERLAY_DARK_RED_COLOR,
    }
    normalized_style = str(style).strip().lower()
    if normalized_style not in colors:
        raise ValueError(
            "BACKGROUND_OVERLAY_STYLE 只能是 'dark' 或 'dark_red'，"
            f"当前值为 {style!r}。"
        )
    return colors[normalized_style]


def _validate_overlay_config() -> None:
    _resolve_overlay_color(BACKGROUND_OVERLAY_STYLE)
    threshold = float(ALL_OVER_EC_THRESHOLD)
    w_curve_ctf_threshold = float(W_CURVE_CTF_THRESHOLD)
    alpha = float(BACKGROUND_OVERLAY_ALPHA)
    if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError(
            "ALL_OVER_EC_THRESHOLD 必须是 [0, 1] 内的有限数值，"
            f"当前值为 {ALL_OVER_EC_THRESHOLD!r}。"
        )
    if not np.isfinite(w_curve_ctf_threshold) or w_curve_ctf_threshold < 0.0:
        raise ValueError(
            "W_CURVE_CTF_THRESHOLD 必须是不小于 0 的有限数值，"
            f"当前值为 {W_CURVE_CTF_THRESHOLD!r}。"
        )
    if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError(
            "BACKGROUND_OVERLAY_ALPHA 必须是 [0, 1] 内的有限数值，"
            f"当前值为 {BACKGROUND_OVERLAY_ALPHA!r}。"
        )


def _save_figure_png_and_svg(
    fig,
    output_png: Path,
    *,
    dpi: int,
    svg_y_ticks_by_axis: list[tuple[plt.Axes, list[float]]] | None = None,
) -> Path | None:
    """保存 PNG，并按配置从同一个 Figure 同步保存同名 SVG。"""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    if not SAVE_SVG:
        return None
    output_svg = output_png.with_suffix(".svg")
    hidden_texts: list[tuple[Text, bool]] = []
    original_y_axis_state: list[tuple[plt.Axes, Any, Any]] = []
    for axis, svg_y_ticks in svg_y_ticks_by_axis or []:
        original_y_axis_state.append(
            (
                axis,
                axis.yaxis.get_major_locator(),
                axis.yaxis.get_major_formatter(),
            )
        )
        axis.set_yticks(svg_y_ticks)
        axis.set_yticklabels([""] * len(svg_y_ticks))
    if SVG_HIDE_VISIBLE_NUMBERS:
        for text_artist in fig.findobj(match=Text):
            hidden_texts.append((text_artist, bool(text_artist.get_visible())))
            text_artist.set_visible(False)
    try:
        fig.savefig(output_svg, format="svg", dpi=dpi, bbox_inches="tight")
    finally:
        for text_artist, was_visible in hidden_texts:
            text_artist.set_visible(was_visible)
        for axis, original_locator, original_formatter in original_y_axis_state:
            axis.yaxis.set_major_locator(original_locator)
            axis.yaxis.set_major_formatter(original_formatter)
    return output_svg


def _read_ctf_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在：{csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_fields = sorted(REQUIRED_FIELDS - fieldnames)
        if missing_fields:
            raise ValueError(
                f"{csv_path} 缺少必要字段：{', '.join(missing_fields)}"
            )

        rows: list[dict[str, Any]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            try:
                m_value = float(raw_row["m"])
                ctf_text = str(raw_row["CTF_dose_mean"]).strip()
                all_over_ec_text = str(raw_row[ALL_OVER_EC_FIELD]).strip()
                all_over_ec = (
                    float(all_over_ec_text)
                    if all_over_ec_text
                    else float("nan")
                )
                if np.isfinite(all_over_ec) and not (0.0 <= all_over_ec <= 1.0):
                    raise ValueError(
                        "all_over_Ec 必须是 [0, 1] 内的有限数值，"
                        f"当前值为 {raw_row[ALL_OVER_EC_FIELD]!r}"
                    )
                if ctf_text:
                    ctf_value = float(ctf_text)
                elif M1_EMPTY_CTF_AS_ZERO and m_value == 1.0:
                    ctf_value = 0.0
                else:
                    ctf_value = float("nan")
                rows.append(
                    {
                        "dose_type": str(raw_row["dose_type"]).strip(),
                        "m": m_value,
                        "r_over_R": float(raw_row["r_over_R"]),
                        "CTF_dose_mean": ctf_value,
                        ALL_OVER_EC_FIELD: all_over_ec,
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path} 第 {line_number} 行包含无效数据：{exc}"
                ) from exc

    if not rows:
        raise ValueError(f"输入 CSV 没有数据行：{csv_path}")
    if any(not row["dose_type"] for row in rows):
        raise ValueError(f"{csv_path} 中存在空的 dose_type。")
    return rows


def _build_matrices(
    rows: list[dict[str, Any]],
    dose_type: str,
    m_values: list[float],
    r_over_r_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    ctf_matrix = np.full(
        (len(r_over_r_values), len(m_values)),
        np.nan,
        dtype=np.float64,
    )
    all_over_ec_matrix = np.full(
        (len(r_over_r_values), len(m_values)),
        np.nan,
        dtype=np.float64,
    )
    m_to_index = {value: index for index, value in enumerate(m_values)}
    r_to_index = {value: index for index, value in enumerate(r_over_r_values)}
    occupied_cells: set[tuple[int, int]] = set()

    for row in rows:
        if row["dose_type"] != dose_type:
            continue
        m_index = m_to_index[row["m"]]
        r_index = r_to_index[row["r_over_R"]]
        cell = (r_index, m_index)
        if cell in occupied_cells:
            raise ValueError(
                "CSV 中存在重复格子："
                f"dose_type={dose_type}, m={row['m']:g}, "
                f"r/R={row['r_over_R']:g}"
            )
        occupied_cells.add(cell)
        ctf_matrix[cell] = row["CTF_dose_mean"]
        all_over_ec_matrix[cell] = row[ALL_OVER_EC_FIELD]

    return ctf_matrix, all_over_ec_matrix


# =============================================================================
# Heatmap 绘制
# =============================================================================

def _plot_all_over_ec_overlay(
    ax: plt.Axes,
    all_over_ec_matrix: np.ndarray,
    r_over_r_values: list[float],
    r_step: float,
    *,
    threshold: float,
    overlay_color: str,
) -> None:
    for r_index, r_over_r in enumerate(r_over_r_values):
        for m_index in range(all_over_ec_matrix.shape[1]):
            all_over_ec = float(
                all_over_ec_matrix[r_index, m_index]
            )
            if not np.isfinite(all_over_ec) or not (
                all_over_ec > float(threshold)
            ):
                continue
            ax.add_patch(
                Rectangle(
                    (float(r_over_r) - 0.5 * r_step, float(m_index) - 0.5),
                    r_step,
                    1.0,
                    fill=True,
                    facecolor=overlay_color,
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=BACKGROUND_OVERLAY_ALPHA,
                    clip_on=False,
                    zorder=10,
                )
            )


def generate_heatmap(
    csv_path: Path,
    output_png: Path,
    *,
    show_overlay: bool = SHOW_ALL_OVER_EC_OVERLAY,
) -> None:
    _validate_overlay_config()
    overlay_color = _resolve_overlay_color(BACKGROUND_OVERLAY_STYLE)
    rows = _read_ctf_csv(csv_path)
    dose_types = sorted({str(row["dose_type"]) for row in rows})
    m_values = sorted({float(row["m"]) for row in rows})
    r_over_r_values = sorted({float(row["r_over_R"]) for row in rows})

    finite_ctf_values = np.asarray(
        [
            float(row["CTF_dose_mean"])
            for row in rows
            if np.isfinite(float(row["CTF_dose_mean"]))
        ],
        dtype=np.float64,
    )
    if finite_ctf_values.size == 0:
        raise ValueError(f"{csv_path} 中没有有限的 CTF_dose_mean 数值。")

    over_max_count = int(np.count_nonzero(finite_ctf_values > COLOR_MAX))
    below_min_count = int(np.count_nonzero(finite_ctf_values < COLOR_MIN))
    overlay_count = int(
        sum(
            float(row[ALL_OVER_EC_FIELD])
            > float(ALL_OVER_EC_THRESHOLD)
            for row in rows
        )
    )

    figure_width = FIGURE_WIDTH_INCH * len(dose_types)
    fig, axes = plt.subplots(
        1,
        len(dose_types),
        figsize=(figure_width, FIGURE_HEIGHT_INCH),
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = list(axes.ravel())
    cmap = plt.get_cmap(COLORMAP_NAME).copy()
    cmap.set_bad(color=MISSING_VALUE_COLOR)
    r_step = (
        0.1
        if len(r_over_r_values) == 1
        else float(np.median(np.diff(r_over_r_values)))
    )

    images = []
    for ax, dose_type in zip(axes_list, dose_types):
        ctf_matrix, all_over_ec_matrix = _build_matrices(
            rows,
            dose_type,
            m_values,
            r_over_r_values,
        )
        image = ax.imshow(
            ctf_matrix.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=COLOR_MIN,
            vmax=COLOR_MAX,
            extent=[
                min(r_over_r_values) - 0.5 * r_step,
                max(r_over_r_values) + 0.5 * r_step,
                -0.5,
                len(m_values) - 0.5,
            ],
        )
        images.append(image)

        if show_overlay:
            _plot_all_over_ec_overlay(
                ax,
                all_over_ec_matrix,
                r_over_r_values,
                r_step,
                threshold=ALL_OVER_EC_THRESHOLD,
                overlay_color=overlay_color,
            )

        for spine in ax.spines.values():
            spine.set_zorder(20)
        ax.set_title(f"{csv_path.stem} | {dose_type}")
        ax.set_xlabel("r/R")
        ax.set_xticks(r_over_r_values)
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"{value:g}" for value in m_values])
        ax.set_ylabel("m / px")

    colorbar = fig.colorbar(
        images[0],
        ax=axes_list,
        fraction=0.046,
        pad=0.04,
        ticks=np.linspace(COLOR_MIN, COLOR_MAX, 6),
    )
    colorbar.set_label("CTF_dose_mean")
    m_to_index = {float(value): index for index, value in enumerate(m_values)}
    missing_svg_m_ticks = [
        value for value in SVG_HEATMAP_M_TICKS if float(value) not in m_to_index
    ]
    if missing_svg_m_ticks:
        raise ValueError(
            f"{csv_path} 缺少 SVG 纵轴刻度所需的 m：{missing_svg_m_ticks}"
        )
    svg_m_tick_positions = [
        float(m_to_index[float(value)]) for value in SVG_HEATMAP_M_TICKS
    ]
    output_svg = _save_figure_png_and_svg(
        fig,
        output_png,
        dpi=OUTPUT_DPI,
        svg_y_ticks_by_axis=[
            (axis, svg_m_tick_positions) for axis in axes_list
        ],
    )
    plt.close(fig)

    print(
        f"[heatmap] input={csv_path} | output_png={output_png} | "
        f"output_svg={output_svg} | "
        f"rows={len(rows)} | dose_types={dose_types} | "
        f"CTF_range={finite_ctf_values.min():.12g}.."
        f"{finite_ctf_values.max():.12g} | "
        f"all_over_Ec_threshold={ALL_OVER_EC_THRESHOLD:g} | "
        f"overlay_cells={overlay_count} | "
        f"overlay_style={BACKGROUND_OVERLAY_STYLE} | "
        f"overlay={'on' if show_overlay else 'off'}"
    )
    if over_max_count or below_min_count:
        print(
            f"[heatmap] warning: {over_max_count} value(s) > {COLOR_MAX:g}, "
            f"{below_min_count} value(s) < {COLOR_MIN:g}; "
            "这些格子将显示为色尺端点颜色，CSV 原值不变。"
        )


def _compute_w_curve_rows(
    rows: list[dict[str, Any]],
    dose_type: str,
    r_over_r_values: list[float],
) -> list[tuple[float, float]]:
    w_rows: list[tuple[float, float]] = []
    for r_over_r in r_over_r_values:
        eligible_m_values = sorted(
            {
                float(row["m"])
                for row in rows
                if row["dose_type"] == dose_type
                and float(row["r_over_R"]) == float(r_over_r)
                and np.isfinite(float(row["CTF_dose_mean"]))
                and float(row["CTF_dose_mean"])
                > float(W_CURVE_CTF_THRESHOLD)
                and float(row[ALL_OVER_EC_FIELD])
                <= float(ALL_OVER_EC_THRESHOLD)
            }
        )
        hit_w = eligible_m_values[0] if eligible_m_values else float("nan")
        w_rows.append((float(r_over_r), hit_w))
    return w_rows


def _update_resolution_w_summary(
    detail_csv_path: Path,
    optimized_w_rows: list[tuple[float, float]],
) -> Path:
    """将 W 曲线使用的 optimized W 同步写回同目录的汇总 CSV。"""
    summary_csv_path = detail_csv_path.with_name("resolution_w_summary.csv")
    if not summary_csv_path.is_file():
        raise FileNotFoundError(f"对应的 resolution_w_summary.csv 不存在：{summary_csv_path}")

    w_by_r_over_r = {
        float(r_over_r): float(w)
        for r_over_r, w in optimized_w_rows
    }
    with summary_csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError(
                f"{summary_csv_path} 的列必须严格为 {SUMMARY_FIELDS}，"
                f"当前为 {reader.fieldnames}"
            )
        summary_rows = list(reader)

    for line_number, summary_row in enumerate(summary_rows, start=2):
        try:
            r_over_r = float(summary_row["r_over_R"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{summary_csv_path} 第 {line_number} 行的 r_over_R 无效："
                f"{summary_row['r_over_R']!r}"
            ) from exc
        if r_over_r not in w_by_r_over_r:
            raise ValueError(
                f"{summary_csv_path} 的 r_over_R={r_over_r:g} 在 {detail_csv_path} 中没有 W 曲线结果"
            )
        w = w_by_r_over_r[r_over_r]
        summary_row["w_optimized_px"] = "" if not np.isfinite(w) else str(w)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_csv_path


def generate_w_curve(
    csv_path: Path,
    output_png: Path,
    *,
    update_summary_csv: bool = False,
) -> None:
    _validate_overlay_config()
    rows = _read_ctf_csv(csv_path)
    dose_types = sorted({str(row["dose_type"]) for row in rows})
    r_over_r_values = sorted({float(row["r_over_R"]) for row in rows})

    fig, ax = plt.subplots(figsize=(W_CURVE_WIDTH_INCH, W_CURVE_HEIGHT_INCH))
    hit_counts: dict[str, int] = {}
    all_finite_w_values: list[float] = []
    missing_points_by_dose_type: dict[str, np.ndarray] = {}
    w_rows_by_dose_type: dict[str, list[tuple[float, float]]] = {}
    for dose_type in dose_types:
        w_rows = _compute_w_curve_rows(rows, dose_type, r_over_r_values)
        w_rows_by_dose_type[dose_type] = w_rows
        x = np.asarray([item[0] for item in w_rows], dtype=np.float64)
        y = np.asarray([item[1] for item in w_rows], dtype=np.float64)
        finite_mask = np.isfinite(y)
        hit_counts[dose_type] = int(np.count_nonzero(np.isfinite(y)))
        all_finite_w_values.extend(float(value) for value in y[finite_mask])
        missing_points_by_dose_type[dose_type] = x[~finite_mask]
        ax.plot(x, y, marker="o", linewidth=2.0, label=dose_type)

    finite_w = np.asarray(all_finite_w_values, dtype=np.float64)
    if finite_w.size:
        y_min = float(np.min(finite_w))
        y_max = float(np.max(finite_w))
        y_span = max(y_max - y_min, 1.0)
        missing_marker_y = y_min - 0.08 * y_span
        ax.set_ylim(missing_marker_y - 0.08 * y_span, y_max + 0.12 * y_span)
    else:
        missing_marker_y = 0.0
        ax.set_ylim(-1.0, 1.0)

    for dose_type, missing_x in missing_points_by_dose_type.items():
        if missing_x.size == 0:
            continue
        ax.plot(
            missing_x,
            np.full(missing_x.shape, missing_marker_y, dtype=np.float64),
            linestyle="None",
            marker=MISSING_W_MARKER,
            markersize=MISSING_W_MARKER_SIZE,
            markeredgewidth=MISSING_W_MARKER_EDGE_WIDTH,
            color=MISSING_W_MARKER_COLOR,
            label=f"{dose_type} missing w",
        )

    ax.set_xlabel("r/R")
    ax.set_ylabel("w / px")
    ax.set_title(f"{csv_path.stem} optimized resolution w vs r/R")
    ax.set_xticks(r_over_r_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_svg = _save_figure_png_and_svg(fig, output_png, dpi=W_CURVE_DPI)
    plt.close(fig)
    updated_summary_csv: Path | None = None
    if update_summary_csv and output_svg is not None:
        if "optimized" not in w_rows_by_dose_type:
            raise ValueError(
                f"{csv_path} 不包含 dose_type='optimized'，无法更新 w_optimized_px"
            )
        updated_summary_csv = _update_resolution_w_summary(
            csv_path,
            w_rows_by_dose_type["optimized"],
        )
    print(
        f"[w-curve] input={csv_path} | output_png={output_png} | "
        f"output_svg={output_svg} | "
        f"updated_summary_csv={updated_summary_csv} | "
        "source_fields=m,CTF_dose_mean,all_over_Ec | "
        f"CTF_threshold=>{W_CURVE_CTF_THRESHOLD:g} | "
        f"all_over_Ec_threshold=<={ALL_OVER_EC_THRESHOLD:g} | "
        f"hit_counts={hit_counts}"
    )


def generate_combined_w_curve(csv_paths: list[Path], output_png: Path) -> None:
    """将四个衰减率的 optimized W 曲线绘制到同一张图中。"""
    if len(csv_paths) != len(ATTENUATION_LABELS):
        raise ValueError(
            "合并 W 曲线需要与 ATTENUATION_LABELS 数量一致的输入 CSV。"
        )

    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for csv_path, attenuation_label in zip(csv_paths, ATTENUATION_LABELS):
        rows = _read_ctf_csv(csv_path)
        dose_types = {str(row["dose_type"]) for row in rows}
        if "optimized" not in dose_types:
            raise ValueError(f"{csv_path} 不包含 dose_type='optimized'。")
        r_over_r_values = sorted({float(row["r_over_R"]) for row in rows})
        w_rows = _compute_w_curve_rows(rows, "optimized", r_over_r_values)
        curves.append(
            (
                attenuation_label,
                np.asarray([item[0] for item in w_rows], dtype=np.float64),
                np.asarray([item[1] for item in w_rows], dtype=np.float64),
            )
        )

    finite_w = np.concatenate([w[np.isfinite(w)] for _, _, w in curves])
    if finite_w.size:
        y_min, y_max = float(finite_w.min()), float(finite_w.max())
        y_span = max(y_max - y_min, 1.0)
        missing_marker_y = y_min - 0.08 * y_span
        y_limits = (
            missing_marker_y - 0.07 * y_span,
            y_max + 0.12 * y_span,
        )
    else:
        missing_marker_y, y_limits = 0.0, (-1.0, 1.0)

    fig, ax = plt.subplots(figsize=(W_CURVE_WIDTH_INCH, W_CURVE_HEIGHT_INCH))
    missing_count = 0
    for color, line_style, marker, (attenuation_label, r_over_r, w) in zip(
        W_CURVE_COLORS,
        W_CURVE_LINE_STYLES,
        W_CURVE_MARKERS,
        curves,
    ):
        ax.plot(
            r_over_r,
            w,
            color=color,
            linestyle=line_style,
            linewidth=W_CURVE_LINE_WIDTH,
            marker=marker,
            markersize=W_CURVE_MARKER_SIZE,
            label=attenuation_label,
            zorder=3,
        )
        missing_mask = ~np.isfinite(w)
        missing_count += int(np.count_nonzero(missing_mask))
        if np.any(missing_mask):
            ax.plot(
                r_over_r[missing_mask],
                np.full(np.count_nonzero(missing_mask), missing_marker_y),
                linestyle="None",
                marker=MISSING_W_MARKER,
                markersize=MISSING_W_MARKER_SIZE,
                markeredgewidth=MISSING_W_MARKER_EDGE_WIDTH,
                color=MISSING_W_MARKER_COLOR,
                zorder=4,
            )

    x_values = np.unique(np.concatenate([r for _, r, _ in curves]))
    ax.set(xlabel="r/R", ylabel="w / px", ylim=y_limits)
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{value:.1f}" for value in x_values])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3.0)
    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.tight_layout(pad=0.35)
    output_svg = _save_figure_png_and_svg(fig, output_png, dpi=W_CURVE_DPI)
    plt.close(fig)

    print(
        f"[combined-w-curve] curves={len(curves)} | "
        f"finite_w={finite_w.size} | missing_w={missing_count} | "
        f"output_png={output_png} | output_svg={output_svg}"
    )


def main() -> None:
    args = _parse_args()
    use_default_paths = (
        list(args.input_csv) == list(INPUT_CSV_PATHS)
        and list(args.output_png) == list(OUTPUT_PNG_PATHS)
    )
    for input_csv, output_png in zip(args.input_csv, args.output_png):
        generate_heatmap(
            input_csv,
            output_png,
            show_overlay=args.show_overlay,
        )
    if SAVE_W_CURVE_PNG and use_default_paths:
        generate_combined_w_curve(
            list(args.input_csv),
            OUTPUT_COMBINED_W_CURVE_PNG,
        )


if __name__ == "__main__":
    main()
