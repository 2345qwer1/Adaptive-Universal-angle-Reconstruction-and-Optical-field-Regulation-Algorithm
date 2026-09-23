from __future__ import annotations

"""绘制 a1/a2 在四种衰减条件下的局部 W 曲线。"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 用户可调参数
# =============================================================================

# 8 个 W 汇总 CSV 的绝对输入目录；文件名须为 a1_00.csv 至 a2_75.csv。
INPUT_DIR = Path(r"F:\USTC\项目\VAM\w数据\w")

# 两张曲线图的 PNG 与 SVG 绝对输出目录。
OUTPUT_DIR = Path(r"F:\USTC\项目\VAM\w数据\w曲线")

# 结构、衰减标签与输入文件名。每项按顺序对应一条曲线。
ATTENUATIONS = (("00", "0%"), ("25", "25%"), ("50", "50%"), ("75", "75%"))
STRUCTURES = ("a1", "a2")

# SCI 单栏曲线图样式参数。
FIGURE_SIZE_INCH = (3.5, 2.85)
PNG_DPI = 300
LINE_WIDTH_PT = 1.5
MARKER_SIZE_PT = 4.0
MISSING_MARKER = "x"
MISSING_MARKER_COLOR = "#C00000"
MISSING_MARKER_SIZE_PT = 6.5
MISSING_MARKER_EDGE_WIDTH_PT = 1.4
# 四种曲线颜色与衰减条件按顺序一一对应；不同标记保证灰度打印仍可区分。
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
LINE_STYLES = ("--", "--", "--", "--")
MARKERS = ("o", "s", "^", "D")


REQUIRED_FIELDS = ("r_over_R", "w_optimized_px")


def read_w_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一条 W 曲线；空的 w_optimized_px 保留为 NaN。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在：{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_fields = [field for field in REQUIRED_FIELDS if field not in fieldnames]
        if missing_fields:
            raise ValueError(f"{csv_path} 缺少必要字段：{', '.join(missing_fields)}")

        points: list[tuple[float, float]] = []
        seen_r_over_r: set[float] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                r_over_r = float(str(row["r_over_R"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path} 第 {line_number} 行的 r_over_R 无效：{row['r_over_R']!r}"
                ) from exc
            if not math.isfinite(r_over_r):
                raise ValueError(f"{csv_path} 第 {line_number} 行的 r_over_R 必须为有限数")
            if r_over_r in seen_r_over_r:
                raise ValueError(f"{csv_path} 存在重复 r_over_R={r_over_r:g}")
            seen_r_over_r.add(r_over_r)

            w_text = str(row["w_optimized_px"] or "").strip()
            if not w_text:
                w = float("nan")
            else:
                try:
                    w = float(w_text)
                except ValueError as exc:
                    raise ValueError(
                        f"{csv_path} 第 {line_number} 行的 w_optimized_px 无效：{w_text!r}"
                    ) from exc
                if not math.isfinite(w):
                    raise ValueError(
                        f"{csv_path} 第 {line_number} 行的 w_optimized_px 必须为有限数或空值"
                    )
            points.append((r_over_r, w))

    if not points:
        raise ValueError(f"输入 CSV 没有数据行：{csv_path}")
    points.sort(key=lambda point: point[0])
    return (
        np.asarray([point[0] for point in points], dtype=float),
        np.asarray([point[1] for point in points], dtype=float),
    )


def plot_structure(structure: str) -> tuple[Path, Path]:
    """绘制一个结构在 0%、25%、50%、75% 衰减下的四条 W 曲线。"""
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for attenuation_code, attenuation_label in ATTENUATIONS:
        csv_path = INPUT_DIR / f"{structure}_{attenuation_code}.csv"
        r_over_r, w = read_w_curve(csv_path)
        curves.append((attenuation_label, r_over_r, w))

    finite_w = np.concatenate([w[np.isfinite(w)] for _, _, w in curves])
    if finite_w.size:
        y_min, y_max = float(finite_w.min()), float(finite_w.max())
        y_span = max(y_max - y_min, 1.0)
        missing_marker_y = y_min - 0.08 * y_span
        y_limits = (missing_marker_y - 0.07 * y_span, y_max + 0.12 * y_span)
    else:
        missing_marker_y, y_limits = 0.0, (-1.0, 1.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCH)
    for color, line_style, marker, (attenuation_label, r_over_r, w) in zip(
        COLORS,
        LINE_STYLES,
        MARKERS,
        curves,
    ):
        # 直接传入 NaN，Matplotlib 会在空 W 处断开折线，不会跨缺失点连接。
        ax.plot(
            r_over_r,
            w,
            color=color,
            linestyle=line_style,
            linewidth=LINE_WIDTH_PT,
            marker=marker,
            markersize=MARKER_SIZE_PT,
            label=attenuation_label,
            zorder=3,
        )
        missing_mask = ~np.isfinite(w)
        if np.any(missing_mask):
            ax.plot(
                r_over_r[missing_mask],
                np.full(np.count_nonzero(missing_mask), missing_marker_y),
                linestyle="None",
                marker=MISSING_MARKER,
                markersize=MISSING_MARKER_SIZE_PT,
                markeredgewidth=MISSING_MARKER_EDGE_WIDTH_PT,
                color=MISSING_MARKER_COLOR,
                zorder=4,
            )

    x_values = np.unique(np.concatenate([r_over_r for _, r_over_r, _ in curves]))
    ax.set(
        xlabel=r"Normalized radial position, $r/R$",
        ylabel=r"Resolution width, $w$ (px)",
        ylim=y_limits,
    )
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_DIR / f"{structure}_w_vs_r.png"
    output_svg = OUTPUT_DIR / f"{structure}_w_vs_r.svg"
    fig.savefig(output_png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    missing_count = sum(int(np.count_nonzero(~np.isfinite(w))) for _, _, w in curves)
    print(
        f"[www] structure={structure} | curves={len(curves)} | "
        f"finite_w={finite_w.size} | missing_w={missing_count} | "
        f"output_png={output_png} | output_svg={output_svg}"
    )
    return output_png, output_svg


def main() -> None:
    for structure in STRUCTURES:
        plot_structure(structure)


if __name__ == "__main__":
    main()
