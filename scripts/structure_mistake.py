from __future__ import annotations

"""
Plot the relationship between local curing correctness and CTF_dose_mean.

Usage:
    1. Edit only the user input parameter section below.
    2. Run from the CAL 1.2 directory:
       python scripts\\structure_mistake.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# User input parameters: edit this section
# =============================================================================

# Input data source note. This script uses the manual lists below.
INPUT_DATA_SCRIPT_PATH = Path(r"F:\VsCodeProjects\VAM\CAL 1.2\scripts\structure_mistake.py")
INPUT_DATA_DESCRIPTION = "Manual 9-point local CTF_dose_mean and all_over_Ec data"

# Nine local CTF_dose_mean values. Order must match ALL_OVER_EC_VALUES.
CTF_DOSE_MEAN_VALUES = [
0.32239695403152618,
0.28726302215540533,
0.36525883112777868,
0.42428821374913755,
0.54469778461732987,
0.5360370107801648,
0.4158960388123833,
0.5474001075367062,
0.46025209690387681,





    # 0.5065882459719654,
    # 0.39801637648407395,
    # 0.3539217194049401,
    # 0.30029234568316854,
    # 0.2800496582944526,
    # 0.26294361579645065,
    # 0.21663166998761946,
    # 0.17115814748924835,
    # 0.2000796821683836,
]

# Nine local all_over_Ec values. Correct rate is computed as 1 - all_over_Ec.
ALL_OVER_EC_VALUES = [
0.55752212389380529,
0.48453608247422686,
0.50884086444007859,
0.081061164333087743,
0.011792452830188704,
0.034872298624754383,
0.014315789473684171,
0.017317612380250536,
0.030779305828421699,


    # 0.037037037037037035,
    # 0.012345679012345678,
    # 0.012345679012345678,
    # 0.024691358024691357,
    # 0.06172839506172839,
    # 0.037037037037037035,
    # 0.09876543209876543,
    # 0.12345679012345678,
    # 0.12345679012345678,
]

# Optional x-axis labels. Leave empty to use 1..9.
STRUCTURE_LABELS: list[str] = []

# Output PNG and SVG paths.
OUTPUT_PNG_PATH = Path(r"F:\USTC\项目\VAM\论文图\heatmap\相关性.png")
OUTPUT_SVG_PATH = Path(r"F:\USTC\项目\VAM\论文图\heatmap\相关性.svg")
# Pearson r text position in figure coordinates.
PEARSON_TEXT_X = 0.92
PEARSON_TEXT_Y = 0.92
PEARSON_TEXT_FONT_SIZE = 11

# Journal-style figure parameters.
FIGURE_WIDTH_INCH = 7.0
FIGURE_HEIGHT_INCH = 4.8
OUTPUT_DPI = 300
BAR_COLOR = "#B8C4D6"
BAR_EDGE_COLOR = "#202020"
BAR_EDGE_WIDTH = 0.8
BAR_WIDTH = 0.68
LINE_COLOR = "#8B1E1E"
LINE_MARKER = "o"
LINE_WIDTH = 1.8
LINE_MARKER_SIZE = 5.0
LEFT_Y_LABEL = "Correct rate"
RIGHT_Y_LABEL = "CTF_dose_mean"
X_AXIS_LABEL = "Local structure index"
LEFT_Y_LIMITS = (0.0, 1.0)
AUTO_LEFT_Y_LIMITS = True
LEFT_Y_PADDING_FRACTION = 0.12
LEFT_Y_MIN_SPAN = 0.05
LEFT_Y_HARD_BOUNDS = (0.0, 1.0)
RIGHT_Y_LIMITS = (0.0, 0.8)
AXIS_LINE_WIDTH = 1.0
TICK_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 11
GRID_ALPHA = 0.18


# =============================================================================
# Validation and correlation
# =============================================================================


def _as_finite_float_array(values: list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list.")
    if array.size != 9:
        raise ValueError(f"{name} must contain exactly 9 values; got {array.size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_labels(labels: list[str], count: int) -> list[str]:
    if not labels:
        return [str(index) for index in range(1, count + 1)]
    if len(labels) != count:
        raise ValueError(f"STRUCTURE_LABELS must contain {count} labels when provided.")
    return [str(label) for label in labels]


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def _validate_style_parameters() -> None:
    if LEFT_Y_LIMITS[1] <= LEFT_Y_LIMITS[0]:
        raise ValueError("LEFT_Y_LIMITS must be increasing.")
    if LEFT_Y_HARD_BOUNDS[1] <= LEFT_Y_HARD_BOUNDS[0]:
        raise ValueError("LEFT_Y_HARD_BOUNDS must be increasing.")
    if LEFT_Y_PADDING_FRACTION < 0.0:
        raise ValueError("LEFT_Y_PADDING_FRACTION must be non-negative.")
    if LEFT_Y_MIN_SPAN <= 0.0:
        raise ValueError("LEFT_Y_MIN_SPAN must be positive.")
    if RIGHT_Y_LIMITS[1] <= RIGHT_Y_LIMITS[0]:
        raise ValueError("RIGHT_Y_LIMITS must be increasing.")
    if FIGURE_WIDTH_INCH <= 0.0 or FIGURE_HEIGHT_INCH <= 0.0:
        raise ValueError("Figure size must be positive.")
    if OUTPUT_DPI <= 0:
        raise ValueError("OUTPUT_DPI must be positive.")


def _resolve_left_y_limits(correct_rate_values: np.ndarray) -> tuple[float, float]:
    if not AUTO_LEFT_Y_LIMITS:
        return float(LEFT_Y_LIMITS[0]), float(LEFT_Y_LIMITS[1])

    values = np.asarray(correct_rate_values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float(LEFT_Y_LIMITS[0]), float(LEFT_Y_LIMITS[1])

    hard_low, hard_high = (float(value) for value in LEFT_Y_HARD_BOUNDS)
    data_low = float(np.min(finite_values))
    data_high = float(np.max(finite_values))
    data_center = 0.5 * (data_low + data_high)
    span = max(data_high - data_low, float(LEFT_Y_MIN_SPAN))
    span = span * (1.0 + 2.0 * float(LEFT_Y_PADDING_FRACTION))

    low = data_center - 0.5 * span
    high = data_center + 0.5 * span

    if low < hard_low:
        high = min(hard_high, high + (hard_low - low))
        low = hard_low
    if high > hard_high:
        low = max(hard_low, low - (high - hard_high))
        high = hard_high

    if high <= low:
        return hard_low, hard_high
    return float(low), float(high)


# =============================================================================
# Plotting
# =============================================================================


def plot_structure_mistake() -> Path:
    ctf_values = _as_finite_float_array(CTF_DOSE_MEAN_VALUES, name="CTF_DOSE_MEAN_VALUES")
    all_over_ec_values = _as_finite_float_array(ALL_OVER_EC_VALUES, name="ALL_OVER_EC_VALUES")
    _validate_style_parameters()

    correct_rate_values = 1.0 - all_over_ec_values
    labels = _validate_labels(STRUCTURE_LABELS, ctf_values.size)
    pearson = _pearson_r(ctf_values, correct_rate_values)

    x = np.arange(ctf_values.size)
    fig, ax_left = plt.subplots(figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH))
    ax_right = ax_left.twinx()

    ax_left.bar(
        x,
        correct_rate_values,
        width=BAR_WIDTH,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label=LEFT_Y_LABEL,
    )
    ax_right.plot(
        x,
        ctf_values,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        marker=LINE_MARKER,
        markersize=LINE_MARKER_SIZE,
        label=RIGHT_Y_LABEL,
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax_left.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
    ax_left.set_ylabel(LEFT_Y_LABEL, fontsize=AXIS_LABEL_SIZE)
    ax_right.set_ylabel(RIGHT_Y_LABEL, fontsize=AXIS_LABEL_SIZE)
    left_y_low, left_y_high = _resolve_left_y_limits(correct_rate_values)
    ax_left.set_ylim(left_y_low, left_y_high)
    ax_right.set_ylim(float(RIGHT_Y_LIMITS[0]), float(RIGHT_Y_LIMITS[1]))
    ax_left.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)
    ax_left.set_axisbelow(True)
    ax_left.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH)
    ax_right.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH, colors=LINE_COLOR)
    ax_right.yaxis.label.set_color(LINE_COLOR)

    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    for side in ("bottom", "left"):
        ax_left.spines[side].set_linewidth(AXIS_LINE_WIDTH)
    ax_right.spines["right"].set_linewidth(AXIS_LINE_WIDTH)
    ax_right.spines["right"].set_color(LINE_COLOR)

    fig.text(
        PEARSON_TEXT_X,
        PEARSON_TEXT_Y,
        f"Pearson r = {pearson:.3f}",
        ha="right",
        va="top",
        fontsize=PEARSON_TEXT_FONT_SIZE,
    )

    output_path = Path(OUTPUT_PNG_PATH)
    svg_output_path = Path(OUTPUT_SVG_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.14, top=0.88)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    fig.savefig(svg_output_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"output_png={output_path}")
    print(f"output_svg={svg_output_path}")
    print(f"pearson_r={pearson:.12g}")
    return output_path


if __name__ == "__main__":
    plot_structure_mistake()
