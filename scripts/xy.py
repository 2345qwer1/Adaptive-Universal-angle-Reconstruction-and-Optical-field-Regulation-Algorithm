from __future__ import annotations

"""
xy.py

Purpose:
    Plot manually entered 2D target/background error-pixel counts.

Usage:
    1. Edit only the "User input parameters" section below.
    2. Run:
       python "F:\\VsCodeProjects\\VAM\\CAL 1.2\\scripts\\xy.py"

Output:
    One PNG figure for a1 and one PNG figure for a2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# User input parameters: edit only this section
# =============================================================================

# Output directory for generated PNG files.
OUTPUT_DIR = Path(r"F:\USTC\项目\VAM\论文图\heatmap\xoy")

# Output PNG file names by structure.
OUTPUT_FILE_BY_STRUCTURE = {
    "a1": "xy_a1_error_px.png",
    "a2": "xy_a2_error_px.png",
}

# X-axis values: structure feature width m, unit is percent.
M_VALUES_PERCENT = [2.0, 5.0, 12.5]

# Four attenuation curves to draw in every figure.
ATTENUATION_LABELS = ["0%", "25%", "50%", "75%"]

# Manually entered y-axis values:
# whole 2D target/background-region error-pixel counts.
#
# For each structure and attenuation label, enter exactly one value for each
# m in M_VALUES_PERCENT, in the same order.
ERROR_PX_BY_STRUCTURE = {
    "a1": {
        "0%": [22035,12974,8072],
        "25%": [46086,26148,10739],
        "50%": [94840,73808,9377],
        "75%": [119335,88584,11045],
    },
    "a2": {
        "0%": [57834,64282,54490],
        "25%": [61396,85096,51956],
        "50%": [66570,145361,64321],
        "75%": [170522,102202,68330],
    },
}

# Figure style parameters.
FIGURE_DPI = 300
FIGURE_SIZE = (7.0, 4.8)
X_AXIS_LABEL = "m / %"
Y_AXIS_LABEL = "Error pixels"
TITLE_BY_STRUCTURE = {
    "a1": "a1 error pixels vs m",
    "a2": "a2 error pixels vs m",
}

# Axis control. Use None for automatic y-axis scaling.
Y_AXIS_LIMITS = None  # Example: (0, 100000)

# Line and marker style.
LINE_WIDTH = 2.0
MARKER_SIZE = 5.0


# =============================================================================
# Validation and plotting logic
# =============================================================================


def _validate_inputs() -> None:
    if not M_VALUES_PERCENT:
        raise ValueError("M_VALUES_PERCENT cannot be empty.")

    m_values = np.asarray(M_VALUES_PERCENT, dtype=np.float64)
    if not np.all(np.isfinite(m_values)):
        raise ValueError("M_VALUES_PERCENT must contain only finite numbers.")
    if np.any(m_values <= 0.0):
        raise ValueError("M_VALUES_PERCENT must contain only positive values.")

    if not ATTENUATION_LABELS:
        raise ValueError("ATTENUATION_LABELS cannot be empty.")
    if len(set(ATTENUATION_LABELS)) != len(ATTENUATION_LABELS):
        raise ValueError("ATTENUATION_LABELS cannot contain duplicate labels.")

    expected_structures = set(OUTPUT_FILE_BY_STRUCTURE)
    provided_structures = set(ERROR_PX_BY_STRUCTURE)
    if expected_structures != provided_structures:
        raise ValueError(
            "ERROR_PX_BY_STRUCTURE keys must match OUTPUT_FILE_BY_STRUCTURE keys: "
            f"expected {sorted(expected_structures)}, got {sorted(provided_structures)}"
        )

    for structure, series_by_attenuation in ERROR_PX_BY_STRUCTURE.items():
        labels = set(series_by_attenuation)
        expected_labels = set(ATTENUATION_LABELS)
        if labels != expected_labels:
            raise ValueError(
                f"{structure} attenuation labels must be {ATTENUATION_LABELS}, "
                f"got {list(series_by_attenuation)}"
            )

        for attenuation_label in ATTENUATION_LABELS:
            values = series_by_attenuation[attenuation_label]
            if len(values) != len(M_VALUES_PERCENT):
                raise ValueError(
                    f"{structure} {attenuation_label} has {len(values)} y-values, "
                    f"but M_VALUES_PERCENT has {len(M_VALUES_PERCENT)} values."
                )
            values_array = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(values_array)):
                raise ValueError(f"{structure} {attenuation_label} contains non-finite values.")
            if np.any(values_array < 0.0):
                raise ValueError(f"{structure} {attenuation_label} contains negative error-pixel counts.")


def _plot_structure(structure: str) -> Path:
    output_path = OUTPUT_DIR / OUTPUT_FILE_BY_STRUCTURE[structure]
    x = np.asarray(M_VALUES_PERCENT, dtype=np.float64)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for attenuation_label in ATTENUATION_LABELS:
        y = np.asarray(ERROR_PX_BY_STRUCTURE[structure][attenuation_label], dtype=np.float64)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=attenuation_label,
        )

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_title(TITLE_BY_STRUCTURE.get(structure, structure))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in x])
    if Y_AXIS_LIMITS is not None:
        ax.set_ylim(float(Y_AXIS_LIMITS[0]), float(Y_AXIS_LIMITS[1]))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Attenuation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    _validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for structure in OUTPUT_FILE_BY_STRUCTURE:
        output_path = _plot_structure(structure)
        print(f"[xy] saved {output_path}")


if __name__ == "__main__":
    main()
