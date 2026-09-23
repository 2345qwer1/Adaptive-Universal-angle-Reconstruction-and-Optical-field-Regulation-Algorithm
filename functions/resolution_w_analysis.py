from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


STRUCTURE_RADIAL_CONCENTRIC_RING = "radial_concentric_ring"
STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER = "angular_blocks_single_layer"
STRUCTURE_ANGULAR_BLOCKS_MULTI_LAYER = "angular_blocks_multi_layer"
ANGULAR_MULTI_LAYER_AXIS_THETA = 0.5 * np.pi


_PIXEL_GEOMETRY_CACHE: dict[
    tuple[int, float, float],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
] = {}


def _round_positive_px(value_px: float, *, name: str = "m") -> int:
    value = float(value_px)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return max(1, int(np.floor(value + 0.5)))


def _validate_positive_percent(value_percent: float, *, name: str = "m_percent") -> float:
    value = float(value_percent)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive percentage.")
    return value


def radial_m_percent_to_px(m_percent: float, container_r_px: float) -> float:
    """Convert a1 ring-width percentage of container radius R to pixels."""
    percent = _validate_positive_percent(m_percent)
    radius = float(container_r_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("container_r_px must be positive.")
    return radius * percent / 100.0


def angular_m_percent_to_radians(m_percent: float) -> float:
    """Convert a2 target-sector arc percentage of a circumference to radians."""
    percent = _validate_positive_percent(m_percent)
    return 2.0 * np.pi * percent / 100.0


def angular_m_percent_to_px(m_percent: float, analyzed_r_px: float) -> float:
    """Convert a2 target-sector arc percentage to arc length at analysis radius r."""
    radius = float(analyzed_r_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("analyzed_r_px must be positive.")
    return radius * angular_m_percent_to_radians(m_percent)


def _normalize_center(center: tuple[float, float] | float | int, n: int) -> tuple[float, float]:
    if isinstance(center, tuple):
        return float(center[0]), float(center[1])
    center_value = float(center)
    return center_value, center_value


def _pixel_geometry(
    n: int,
    center: tuple[float, float] | float | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    center_x, center_y = _normalize_center(center, n)
    key = (int(n), float(center_x), float(center_y))
    cached = _PIXEL_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached

    rows, cols = np.indices((n, n), dtype=np.float64)
    x = cols - center_x
    y = center_y - rows
    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x) + 2.0 * np.pi) % (2.0 * np.pi)
    cached = (x, y, r, theta, (center_x, center_y))
    _PIXEL_GEOMETRY_CACHE[key] = cached
    return cached


def _build_container_mask_px(
    n: int,
    r_px: float,
    center: tuple[float, float] | float | int,
) -> np.ndarray:
    _, _, radius, _, _ = _pixel_geometry(n, center)
    return radius <= float(r_px)


def generate_radial_concentric_rings(
    n: int,
    r_px: float,
    m: float,
    center: tuple[float, float] | float | int,
) -> np.ndarray:
    """
    Generate the a1 radial concentric-ring target.

    m is a percentage of the container radius R. Target-ring width and
    background-gap width are both R * m / 100 pixels.
    """
    m_px = radial_m_percent_to_px(m, r_px)

    _, _, radius, _, _ = _pixel_geometry(n, center)
    container_mask = _build_container_mask_px(n, r_px, center)
    radial_band_index = np.floor(radius / float(m_px)).astype(np.int64)
    target = (radial_band_index % 2) == 0
    return np.logical_and(target, container_mask).astype(np.uint8)


def generate_angular_blocks_single_layer(
    n: int,
    r_px: float,
    m: float,
    center: tuple[float, float] | float | int,
    m_reference_r_px: float,
) -> np.ndarray:
    """
    Generate the a2 full-radius alternating angular-sector target.

    m is the target-sector arc length as a percentage of the circumference.
    Therefore delta_theta = 2*pi*m/100 at every radius. Each background sector
    has the same angular width. Any incomplete final period is left as
    background and is not stretched to fill the circle.
    """
    if m_reference_r_px <= 0:
        raise ValueError("m_reference_r_px must be positive.")

    _, _, radius, theta, _ = _pixel_geometry(n, center)
    container_mask = _build_container_mask_px(n, r_px, center)
    target_width, period_width, _, full_period_count = _angular_single_layer_layout(
        m,
        float(m_reference_r_px),
    )
    full_span = float(full_period_count) * period_width
    in_full_periods = theta < full_span
    theta_local = theta - np.floor(theta / period_width) * period_width
    target = np.logical_and(in_full_periods, theta_local < target_width)
    return np.logical_and(target, container_mask).astype(np.uint8)


def _angular_single_layer_layout(
    m: float,
    m_reference_r_px: float,
    *,
    axis_theta: float = ANGULAR_MULTI_LAYER_AXIS_THETA,
) -> tuple[float, float, float, int]:
    """
    Return exact target width, period width, zero phase offset, and full-period count.

    For a2, m is a percentage of the circumference. The requested angular
    width is never changed to force an integer number of periods.
    """
    del axis_theta
    if m_reference_r_px <= 0:
        raise ValueError("m_reference_r_px must be positive.")

    target_width = angular_m_percent_to_radians(m)
    period_width = 2.0 * target_width
    full_period_count = int(np.floor((2.0 * np.pi + 1e-12) / period_width))
    return float(target_width), float(period_width), 0.0, int(full_period_count)


def _angular_multi_layer_layout(
    analyzed_r_px: float,
    m: float,
    *,
    axis_theta: float = ANGULAR_MULTI_LAYER_AXIS_THETA,
) -> tuple[float, float, float, int]:
    """
    Return target-sector width, period width, center offset, and block count.

    One period contains one target block and one background gap. If the block
    count is odd, one target block is centered on +Y. If it is even, target
    blocks are split evenly on both sides of +Y.
    """
    del axis_theta
    if analyzed_r_px <= 0:
        raise ValueError("analyzed_r_px must be positive.")
    m_px = _round_positive_px(m)

    block_count = max(1, int(round(np.pi * float(analyzed_r_px) / float(m_px))))
    period_width = 2.0 * np.pi / float(block_count)
    target_width = 0.5 * period_width

    center_offset = 0.0 if (block_count % 2 == 1) else 0.5 * period_width
    return float(target_width), float(period_width), float(center_offset), int(block_count)


def generate_angular_blocks_multi_layer(
    n: int,
    r_px: float,
    m: float | dict[float, float],
    center: tuple[float, float] | float | int,
    r_over_r_list: list[float],
    axis_theta: float = ANGULAR_MULTI_LAYER_AXIS_THETA,
) -> np.ndarray:
    """
    Generate 9 annular-layer angular-block structures.

    Each layer corresponds to one radial band:
        0.05R~0.15R
        0.15R~0.25R
        ...
        0.85R~0.95R

    For each layer centered at r_i, m defines the desired target-sector arc
    length. Odd block counts place one target block centered on +Y; even block
    counts place equal target blocks on both sides of +Y.
    """
    if isinstance(m, dict):
        m_px_by_r_over_r = {
            float(r_over_r): _round_positive_px(value, name="m")
            for r_over_r, value in m.items()
        }
        default_m_px = None
    else:
        m_px_by_r_over_r = {}
        default_m_px = _round_positive_px(m)

    _, _, radius, theta, _ = _pixel_geometry(n, center)
    container_mask = _build_container_mask_px(n, r_px, center)
    target = np.zeros((n, n), dtype=bool)

    for r_over_r in r_over_r_list:
        analyzed_r_px = float(r_over_r) * float(r_px)
        if analyzed_r_px <= 0:
            continue
        layer_m_px = m_px_by_r_over_r.get(float(r_over_r), default_m_px)
        if layer_m_px is None:
            raise ValueError(f"Missing m for r_over_R={float(r_over_r)}.")
        r_inner = float(r_over_r - 0.05) * float(r_px)
        r_outer = float(r_over_r + 0.05) * float(r_px)
        ring_mask = np.logical_and(radius >= r_inner, radius < r_outer)
        target_width, period_width, center_offset, _ = _angular_multi_layer_layout(
            analyzed_r_px,
            layer_m_px,
            axis_theta=float(axis_theta),
        )
        theta_local = ((theta - float(axis_theta) - center_offset + 0.5 * period_width) % period_width) - (
            0.5 * period_width
        )
        target |= np.logical_and(ring_mask, np.abs(theta_local) <= 0.5 * target_width)

    return np.logical_and(target, container_mask).astype(np.uint8)


def _sample_bilinear(image: np.ndarray, row: float, col: float) -> float:
    n_rows, n_cols = image.shape
    row = float(np.clip(row, 0.0, n_rows - 1.0))
    col = float(np.clip(col, 0.0, n_cols - 1.0))
    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, n_rows - 1)
    c1 = min(c0 + 1, n_cols - 1)
    dr = row - r0
    dc = col - c0
    top = (1.0 - dc) * image[r0, c0] + dc * image[r0, c1]
    bottom = (1.0 - dc) * image[r1, c0] + dc * image[r1, c1]
    return float((1.0 - dr) * top + dr * bottom)


def _sample_bilinear_many(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sampler for many profile points."""
    image_f64 = np.asarray(image, dtype=np.float64)
    n_rows, n_cols = image_f64.shape
    rows = np.clip(np.asarray(rows, dtype=np.float64), 0.0, n_rows - 1.0)
    cols = np.clip(np.asarray(cols, dtype=np.float64), 0.0, n_cols - 1.0)

    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = np.minimum(r0 + 1, n_rows - 1)
    c1 = np.minimum(c0 + 1, n_cols - 1)
    dr = rows - r0
    dc = cols - c0

    top = (1.0 - dc) * image_f64[r0, c0] + dc * image_f64[r0, c1]
    bottom = (1.0 - dc) * image_f64[r1, c0] + dc * image_f64[r1, c1]
    return (1.0 - dr) * top + dr * bottom


def sample_radial_profiles_four_axes(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Sample 1D radial dose profiles along:
        +x, -x, +y, -y
    """
    center_x, center_y = _normalize_center(center, dose.shape[0])
    distances = np.arange(0.0, float(r_px) + 1.0, 1.0, dtype=np.float64)
    rows_cols = {
        "+x": (np.full_like(distances, center_y), center_x + distances),
        "-x": (np.full_like(distances, center_y), center_x - distances),
        "+y": (center_y - distances, np.full_like(distances, center_x)),
        "-y": (center_y + distances, np.full_like(distances, center_x)),
    }

    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for direction, (rows, cols) in rows_cols.items():
        values = _sample_bilinear_many(dose, rows, cols)
        profiles[direction] = (distances.copy(), values)
    return profiles


def sample_angular_profile(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    num_theta_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample dose along a circle with radius r_px.
    Return theta and dose profile.
    """
    if num_theta_samples <= 0:
        raise ValueError("num_theta_samples must be positive.")

    center_x, center_y = _normalize_center(center, dose.shape[0])
    theta = np.linspace(0.0, 2.0 * np.pi, int(num_theta_samples), endpoint=False, dtype=np.float64)
    rows = center_y - float(r_px) * np.sin(theta)
    cols = center_x + float(r_px) * np.cos(theta)
    values = _sample_bilinear_many(dose, rows, cols)
    return theta, values


def get_angular_num_samples_for_arc_step(r_px: float, arc_step_px: float = 1.0) -> int:
    """Approximate one sample per arc_step_px along the circle."""
    if r_px <= 0.0:
        raise ValueError("r_px must be positive.")
    if arc_step_px <= 0.0:
        raise ValueError("arc_step_px must be positive.")
    circumference_px = 2.0 * np.pi * float(r_px)
    return max(16, int(np.ceil(circumference_px / float(arc_step_px))))


def build_radial_profile_rows(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    *,
    direction: str = "+x",
    target_or_background_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, float | int | str]]:
    profiles = sample_radial_profiles_four_axes(dose, center, r_px)
    if direction not in profiles:
        raise ValueError(f"Unsupported direction: {direction}")
    center_x, center_y = _normalize_center(center, dose.shape[0])
    distances, values = profiles[direction]

    if direction == "+x":
        rows = np.full_like(distances, center_y)
        cols = center_x + distances
    elif direction == "-x":
        rows = np.full_like(distances, center_y)
        cols = center_x - distances
    elif direction == "+y":
        rows = center_y - distances
        cols = np.full_like(distances, center_x)
    else:
        rows = center_y + distances
        cols = np.full_like(distances, center_x)

    if target_or_background_func is None:
        target_or_background = np.zeros_like(distances, dtype=np.int64)
    else:
        target_or_background = np.asarray(target_or_background_func(distances), dtype=np.int64)
        if target_or_background.shape != distances.shape:
            raise ValueError("target_or_background_func must return one value per radial sample.")

    rows_out: list[dict[str, float | int | str]] = []
    for sample_index, (distance_px, row, col, dose_value, target_value) in enumerate(
        zip(distances, rows, cols, values, target_or_background)
    ):
        rows_out.append(
            {
                "sample_index": int(sample_index),
                "distance_px": float(distance_px),
                "row": float(row),
                "col": float(col),
                "dose": float(dose_value),
                "direction": direction,
                "target_or_background": int(target_value),
            }
        )
    return rows_out


def build_angular_profile_rows(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    *,
    arc_step_px: float = 1.0,
    target_or_background_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, float | int]]:
    center_x, center_y = _normalize_center(center, dose.shape[0])
    num_theta_samples = get_angular_num_samples_for_arc_step(r_px, arc_step_px=arc_step_px)
    theta, values = sample_angular_profile(dose, center, r_px, num_theta_samples)
    rows = center_y - float(r_px) * np.sin(theta)
    cols = center_x + float(r_px) * np.cos(theta)
    arc_length_px = theta * float(r_px)
    if target_or_background_func is None:
        target_or_background = np.zeros_like(theta, dtype=np.int64)
    else:
        target_or_background = np.asarray(target_or_background_func(theta), dtype=np.int64)
        if target_or_background.shape != theta.shape:
            raise ValueError("target_or_background_func must return one value per theta sample.")

    rows_out: list[dict[str, float | int]] = []
    for sample_index, (theta_rad, length_px, row, col, dose_value, target_value) in enumerate(
        zip(theta, arc_length_px, rows, cols, values, target_or_background)
    ):
        rows_out.append(
            {
                "sample_index": int(sample_index),
                "theta_deg": float(np.degrees(theta_rad)),
                "arc_length_px": float(length_px),
                "row": float(row),
                "col": float(col),
                "dose": float(dose_value),
                "target_or_background": int(target_value),
            }
        )
    return rows_out


def _resolve_adaptive_profile_y_layout(
    y: np.ndarray,
    *,
    y_limits: tuple[float, float] | None,
    adaptive_y_max_levels: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, float]:
    """Resolve profile y limits and a proportional physical axis-height scale."""
    if adaptive_y_max_levels is None:
        return y_limits, 1.0

    low_y_max, high_y_max = (float(value) for value in adaptive_y_max_levels)
    if low_y_max <= 0.0 or high_y_max <= low_y_max:
        raise ValueError("adaptive_y_max_levels must contain two increasing positive values.")

    y_f64 = np.asarray(y, dtype=np.float64)
    finite_y = y_f64[np.isfinite(y_f64)]
    profile_max = float(np.max(finite_y)) if finite_y.size else 0.0
    resolved_y_max = low_y_max if profile_max <= low_y_max else high_y_max
    return (0.0, resolved_y_max), float(resolved_y_max / low_y_max)


def export_radial_profile_figure_and_csv(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    figure_path: str | Path,
    csv_path: str | Path,
    *,
    title: str,
    direction: str = "+x",
    save_figure: bool = True,
    save_csv: bool = True,
    y_limits: tuple[float, float] | None = None,
    adaptive_y_max_levels: tuple[float, float] | None = None,
    target_or_background_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    figure_path = Path(figure_path)
    csv_path = Path(csv_path)
    profile_rows = build_radial_profile_rows(
        dose,
        center,
        r_px,
        direction=direction,
        target_or_background_func=target_or_background_func,
    )
    if save_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_rows(
            csv_path,
            ["sample_index", "distance_px", "row", "col", "dose", "direction", "target_or_background"],
            profile_rows,
            significant_digits=4,
        )

    if save_figure:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        x = np.array([float(row["distance_px"]) for row in profile_rows], dtype=np.float64)
        y = np.array([float(row["dose"]) for row in profile_rows], dtype=np.float64)
        resolved_y_limits, figure_height_scale = _resolve_adaptive_profile_y_layout(
            y,
            y_limits=y_limits,
            adaptive_y_max_levels=adaptive_y_max_levels,
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.8 * figure_height_scale))
        ax.plot(
            x,
            y,
            linewidth=0.7,
            marker="o",
            markersize=2.2,
            markeredgewidth=0.0,
            color="#0b5fff",
        )
        ax.set_xlabel("distance / px")
        ax.set_ylabel("dose")
        ax.set_title(title)
        if resolved_y_limits is not None:
            ax.set_ylim(float(resolved_y_limits[0]), float(resolved_y_limits[1]))
        ax.grid(True, alpha=0.3)
        if adaptive_y_max_levels is None:
            fig.tight_layout()
        else:
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.88)
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


def export_angular_profile_figure_and_csv(
    dose: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    figure_path: str | Path,
    csv_path: str | Path,
    *,
    title: str,
    arc_step_px: float = 1.0,
    save_figure: bool = True,
    save_csv: bool = True,
    y_limits: tuple[float, float] | None = None,
    adaptive_y_max_levels: tuple[float, float] | None = None,
    target_or_background_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    figure_path = Path(figure_path)
    csv_path = Path(csv_path)
    profile_rows = build_angular_profile_rows(
        dose,
        center,
        r_px,
        arc_step_px=arc_step_px,
        target_or_background_func=target_or_background_func,
    )
    if save_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_rows(
            csv_path,
            ["sample_index", "theta_deg", "arc_length_px", "row", "col", "dose", "target_or_background"],
            profile_rows,
            significant_digits=4,
        )

    if save_figure:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        x = np.array([float(row["arc_length_px"]) for row in profile_rows], dtype=np.float64)
        y = np.array([float(row["dose"]) for row in profile_rows], dtype=np.float64)
        resolved_y_limits, figure_height_scale = _resolve_adaptive_profile_y_layout(
            y,
            y_limits=y_limits,
            adaptive_y_max_levels=adaptive_y_max_levels,
        )
        fig, ax = plt.subplots(figsize=(9.0, 4.8 * figure_height_scale))
        ax.plot(
            x,
            y,
            linewidth=0.7,
            marker="o",
            markersize=2.2,
            markeredgewidth=0.0,
            color="#0b5fff",
        )
        ax.set_xlabel("arc length / px")
        ax.set_ylabel("dose")
        ax.set_title(title)
        if resolved_y_limits is not None:
            ax.set_ylim(float(resolved_y_limits[0]), float(resolved_y_limits[1]))
        ax.grid(True, alpha=0.3)
        if adaptive_y_max_levels is None:
            fig.tight_layout()
        else:
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.88)
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


def export_target_cured_error_figure(
    target: np.ndarray,
    dose: np.ndarray,
    resin_ec: float,
    save_path: str | Path,
    *,
    title: str,
) -> None:
    """
    Export a 1x3 summary figure:
    1. target structure
    2. cured structure
    3. signed error map

    Signed error colors:
    - red: undercure
    - green: overcure
    - black: correct / empty
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    target_bool = np.asarray(target).astype(bool)
    cured_bool = np.asarray(dose, dtype=np.float64) >= float(resin_ec)
    undercure = np.logical_and(target_bool, np.logical_not(cured_bool))
    overcure = np.logical_and(cured_bool, np.logical_not(target_bool))

    error_rgb = np.zeros(target_bool.shape + (3,), dtype=np.float64)
    error_rgb[undercure] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    error_rgb[overcure] = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    axes[0].imshow(target_bool.astype(np.uint8), cmap="gray", interpolation="nearest", origin="upper")
    axes[0].set_title("Target")
    axes[1].imshow(cured_bool.astype(np.uint8), cmap="gray", interpolation="nearest", origin="upper")
    axes[1].set_title(f"Cured | Ec={float(resin_ec):.3g}")
    axes[2].imshow(error_rgb, interpolation="nearest", origin="upper")
    axes[2].set_title(
        f"Error | under={int(np.count_nonzero(undercure))} px | over={int(np.count_nonzero(overcure))} px"
    )

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def export_dose_distribution_figure(
    dose: np.ndarray,
    resin_ec: float,
    save_path: str | Path,
    *,
    title: str,
    dose_vmax: float | None = None,
) -> None:
    """Export one 2D dose distribution figure with the project dose colormap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dose_f64 = np.asarray(dose, dtype=np.float64)
    finite_dose = dose_f64[np.isfinite(dose_f64)]
    if dose_vmax is None:
        resolved_dose_vmax = float(max(np.max(finite_dose), float(resin_ec))) if finite_dose.size else float(resin_ec)
    else:
        resolved_dose_vmax = float(max(float(dose_vmax), float(resin_ec)))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(
        dose_f64,
        cmap="plasma",
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=resolved_dose_vmax,
    )
    ax.set_title(f"Dose | Ec={float(resin_ec):.3g}\n{title}")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dose")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def export_resolved_structure_image_set(
    target: np.ndarray,
    dose: np.ndarray,
    resin_ec: float,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str,
    figure_suffix: str = ".svg",
    dose_vmax: float | None = None,
) -> None:
    """
    Export separate hit-w structure figures:
    dose distribution, target, cured, and signed error.

    Error colors:
    - red: undercure
    - green: overcure
    - black: correct / empty
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = str(figure_suffix)
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    target_bool = np.asarray(target).astype(bool)
    dose_f64 = np.asarray(dose, dtype=np.float64)
    cured_bool = dose_f64 >= float(resin_ec)
    undercure = np.logical_and(target_bool, np.logical_not(cured_bool))
    overcure = np.logical_and(cured_bool, np.logical_not(target_bool))

    error_rgb = np.zeros(target_bool.shape + (3,), dtype=np.float64)
    error_rgb[undercure] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    error_rgb[overcure] = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    dose_vmin = 0.0
    finite_dose = dose_f64[np.isfinite(dose_f64)]
    if dose_vmax is None:
        resolved_dose_vmax = float(max(np.max(finite_dose), float(resin_ec))) if finite_dose.size else float(resin_ec)
    else:
        resolved_dose_vmax = float(max(float(dose_vmax), float(resin_ec)))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(
        dose_f64,
        cmap="plasma",
        interpolation="nearest",
        origin="upper",
        vmin=dose_vmin,
        vmax=resolved_dose_vmax,
    )
    ax.set_title(f"Dose | Ec={float(resin_ec):.3g}\n{title}")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dose")
    fig.tight_layout()
    fig.savefig(output_dir / f"{base_name}_dose{suffix}", bbox_inches="tight")
    plt.close(fig)

    binary_figures = (
        ("target", target_bool.astype(np.uint8), "Target"),
        ("cured", cured_bool.astype(np.uint8), f"Cured | Ec={float(resin_ec):.3g}"),
    )
    for suffix_name, image, image_title in binary_figures:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.imshow(image, cmap="gray", interpolation="nearest", origin="upper")
        ax.set_title(f"{image_title}\n{title}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / f"{base_name}_{suffix_name}{suffix}", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(error_rgb, interpolation="nearest", origin="upper")
    ax.set_title(
        f"Error | under={int(np.count_nonzero(undercure))} px | "
        f"over={int(np.count_nonzero(overcure))} px\n{title}"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / f"{base_name}_error{suffix}", bbox_inches="tight")
    plt.close(fig)


def compute_ctf_dose(d_peak: float, d_valley: float) -> float:
    """
    Compute:
        CTF_dose = (D_peak - D_valley) / (D_peak + D_valley)
    Safely handle a non-positive denominator.
    """
    d_peak = float(d_peak)
    d_valley = float(d_valley)
    denominator = d_peak + d_valley
    if denominator <= 0.0:
        return float("nan")
    return float((d_peak - d_valley) / denominator)


def _build_radial_target_intervals(m: float, r_px: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start = 0.0
    while start < float(r_px):
        end = min(start + float(m), float(r_px))
        intervals.append((start, end))
        start += 2.0 * float(m)
    return intervals


def _nearest_radial_target_intervals(
    r_px: float,
    m: float,
    container_r_px: float,
    num_nearest_rings: int,
) -> list[tuple[float, float]]:
    intervals = _build_radial_target_intervals(m, container_r_px)
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: abs((item[0] + item[1]) * 0.5 - float(r_px)))
    return ordered[: int(num_nearest_rings)]


def compute_radial_ring_ctf_dose_mean(
    dose: np.ndarray,
    target: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    analyzed_r_px: float,
    m: float,
    num_nearest_rings: int = 3,
) -> tuple[float, float]:
    """
    For radial concentric rings:
    - use +x, -x, +y, -y
    - find nearest 3 target rings around r_px
    - for each local ring:
        D_peak = max dose in target ring
        D_valley = min dose in adjacent background gap
    - average over rings and axes
    """
    del target
    profiles = sample_radial_profiles_four_axes(dose, center, r_px)
    nearest_intervals = _nearest_radial_target_intervals(analyzed_r_px, m, r_px, num_nearest_rings)
    local_ctf_dose_values: list[float] = []

    for _, (distances, values) in profiles.items():
        for inner, outer in nearest_intervals:
            ring_mask = np.logical_and(distances >= inner, distances < outer)
            if not np.any(ring_mask):
                continue

            adjacent_masks = []
            if inner > 0.0:
                adjacent_masks.append(np.logical_and(distances >= max(0.0, inner - float(m)), distances < inner))
            if outer < float(r_px):
                adjacent_masks.append(np.logical_and(distances >= outer, distances < min(float(r_px), outer + float(m))))

            valley_mask = np.zeros_like(ring_mask, dtype=bool)
            for candidate in adjacent_masks:
                valley_mask |= candidate
            if not np.any(valley_mask):
                continue

            d_peak = float(np.max(values[ring_mask]))
            d_valley = float(np.min(values[valley_mask]))
            ctf_dose_value = compute_ctf_dose(d_peak, d_valley)
            if np.isfinite(ctf_dose_value):
                local_ctf_dose_values.append(ctf_dose_value)

    if not local_ctf_dose_values:
        return float("nan"), float("nan")

    ctf_dose_array = np.asarray(local_ctf_dose_values, dtype=np.float64)
    return float(np.mean(ctf_dose_array)), float(np.std(ctf_dose_array))


def compute_angular_block_ctf_dose_mean(
    dose: np.ndarray,
    target: np.ndarray,
    center: tuple[float, float] | float | int,
    r_px: float,
    analyzed_r_px: float,
    m: float,
    num_theta_samples: int = 2048,
    multi_layer_axis_theta: float | None = None,
    m_reference_r_px: float | None = None,
) -> tuple[float, float]:
    """
    For angular-block structures:
    - sample dose on the circle at r_px
    - for each angular period:
        D_peak = mean dose in target angular block
        D_valley = mean dose in adjacent background angular block
    - average over all angular periods
    """
    del target
    del r_px
    theta, dose_profile = sample_angular_profile(dose, center, analyzed_r_px, num_theta_samples)

    if multi_layer_axis_theta is None and m_reference_r_px is not None:
        delta_theta, period_width, center_offset, full_period_count = _angular_single_layer_layout(
            float(m),
            float(m_reference_r_px),
        )
        del center_offset
        full_span = float(full_period_count) * period_width
        in_full_periods = theta < full_span
        period_index = np.floor(theta / period_width).astype(np.int64)
        theta_local = theta - period_index.astype(np.float64) * period_width
        target_mask = np.logical_and(in_full_periods, theta_local < delta_theta)
        background_mask = np.logical_and(
            in_full_periods,
            np.logical_and(theta_local >= delta_theta, theta_local < period_width),
        )
    elif multi_layer_axis_theta is None:
        delta_theta = angular_m_percent_to_radians(m)
        period_width = 2.0 * delta_theta
        full_period_count = int(np.floor((2.0 * np.pi + 1e-12) / period_width))
        full_span = float(full_period_count) * period_width
        in_full_periods = theta < full_span
        period_index = np.floor(theta / period_width).astype(np.int64)
        theta_local = theta - period_index.astype(np.float64) * period_width
        target_mask = np.logical_and(in_full_periods, theta_local < delta_theta)
        background_mask = np.logical_and(
            in_full_periods,
            np.logical_and(theta_local >= delta_theta, theta_local < period_width),
        )
    else:
        delta_theta, period_width, center_offset, _ = _angular_multi_layer_layout(
            float(analyzed_r_px),
            int(m),
            axis_theta=float(multi_layer_axis_theta),
        )
        theta_shifted = (theta - float(multi_layer_axis_theta) - center_offset + 0.5 * period_width) % (2.0 * np.pi)
        period_index = np.floor(theta_shifted / period_width).astype(np.int64)
        theta_local = (theta_shifted % period_width) - 0.5 * period_width
        target_mask = np.abs(theta_local) <= 0.5 * delta_theta
        background_mask = np.abs(theta_local) > 0.5 * delta_theta

    period_count = int(period_index.max()) + 1 if period_index.size > 0 else 0

    if period_count <= 0:
        return float("nan"), float("nan")

    target_counts = np.bincount(period_index[target_mask], minlength=period_count)
    background_counts = np.bincount(period_index[background_mask], minlength=period_count)
    valid = np.logical_and(target_counts > 0, background_counts > 0)
    if not np.any(valid):
        return float("nan"), float("nan")

    target_values = dose_profile[target_mask]
    background_values = dose_profile[background_mask]
    target_period_index = period_index[target_mask]
    background_period_index = period_index[background_mask]
    local_ctf_dose_values: list[float] = []
    for period_idx in np.flatnonzero(valid):
        target_period_values = target_values[target_period_index == period_idx]
        background_period_values = background_values[background_period_index == period_idx]
        if target_period_values.size == 0 or background_period_values.size == 0:
            continue
        d_peak = float(np.max(target_period_values))
        d_valley = float(np.min(background_period_values))
        ctf_dose_value = compute_ctf_dose(d_peak, d_valley)
        if np.isfinite(ctf_dose_value):
            local_ctf_dose_values.append(ctf_dose_value)

    ctf_dose_array = np.asarray(local_ctf_dose_values, dtype=np.float64)
    if ctf_dose_array.size == 0:
        return float("nan"), float("nan")

    return float(np.mean(ctf_dose_array)), float(np.std(ctf_dose_array))


def find_resolution_w_for_structure(
    structure_type: str,
    m_values: range | list[float],
    r_over_r_list: list[float],
    ctf_dose_threshold: float,
    dose_types: list[str],
    run_simulation_func: Callable[[np.ndarray], dict[str, np.ndarray]],
    output_dir: Path,
    *,
    n: int,
    r_px: float,
    center: tuple[float, float] | float | int,
    radial_num_nearest_rings: int = 3,
    angular_num_theta_samples: int = 2048,
    on_resolved_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], float]]:
    """
    Scan m from small to large.
    For each structure_type / dose_type / r_over_R,
    record the first m satisfying CTF_dose_mean >= CTF_dose_threshold.

    Stop automatically once all w values are found.
    """
    del output_dir
    resolved_w = {
        (dose_type, float(r_over_r)): float("nan")
        for dose_type in dose_types
        for r_over_r in r_over_r_list
    }
    detail_rows: list[dict[str, Any]] = []
    center_xy = _normalize_center(center, n)

    for m in m_values:
        if structure_type == STRUCTURE_RADIAL_CONCENTRIC_RING:
            m_percent = float(m)
            m_px = radial_m_percent_to_px(m_percent, r_px)
            target_mask = generate_radial_concentric_rings(n, r_px, m_percent, center_xy)
            dose_fields = run_simulation_func(target_mask)
            for r_over_r in r_over_r_list:
                analyzed_r_px = float(r_over_r) * float(r_px)
                for dose_type in dose_types:
                    previous_w = float(resolved_w[(dose_type, float(r_over_r))])
                    ctf_dose_mean, ctf_dose_std = compute_radial_ring_ctf_dose_mean(
                        dose_fields[dose_type],
                        target_mask,
                        center_xy,
                        r_px,
                        analyzed_r_px,
                        m_px,
                        num_nearest_rings=radial_num_nearest_rings,
                    )
                    resolved = bool(
                        np.isfinite(ctf_dose_mean)
                        and ctf_dose_mean >= float(ctf_dose_threshold)
                    )
                    if np.isnan(previous_w) and resolved:
                        resolved_w[(dose_type, float(r_over_r))] = float(m)
                        if on_resolved_callback is not None:
                            on_resolved_callback(
                                {
                                    "structure_type": structure_type,
                                    "dose_type": dose_type,
                                    "m": m_percent,
                                    "r_over_R": float(r_over_r),
                                    "r_px": float(analyzed_r_px),
                                    "target_mask": target_mask,
                                    "dose_field": dose_fields[dose_type],
                                    "center": center_xy,
                                    "container_r_px": float(r_px),
                                }
                            )
                    w_percent = float(resolved_w[(dose_type, float(r_over_r))])
                    w_px = radial_m_percent_to_px(w_percent, r_px) if np.isfinite(w_percent) else float("nan")
                    detail_rows.append(
                        {
                            "structure_type": structure_type,
                            "dose_type": dose_type,
                            "m": m_percent,
                            "m_percent": m_percent,
                            "m_px": m_px,
                            "r_over_R": float(r_over_r),
                            "r_px": float(analyzed_r_px),
                            "CTF_dose_mean": float(ctf_dose_mean),
                            "CTF_dose_std": float(ctf_dose_std),
                            "resolved": resolved,
                            "w_percent": w_percent,
                            "w_px": w_px,
                        }
                    )
            continue

        if structure_type == STRUCTURE_ANGULAR_BLOCKS_MULTI_LAYER:
            target_mask = generate_angular_blocks_multi_layer(
                n,
                r_px,
                int(m),
                center_xy,
                list(r_over_r_list),
                axis_theta=ANGULAR_MULTI_LAYER_AXIS_THETA,
            )
            dose_fields = run_simulation_func(target_mask)
            for r_over_r in r_over_r_list:
                analyzed_r_px = float(r_over_r) * float(r_px)
                for dose_type in dose_types:
                    previous_w = float(resolved_w[(dose_type, float(r_over_r))])
                    ctf_dose_mean, ctf_dose_std = compute_angular_block_ctf_dose_mean(
                        dose_fields[dose_type],
                        target_mask,
                        center_xy,
                        r_px,
                        analyzed_r_px,
                        int(m),
                        num_theta_samples=angular_num_theta_samples,
                        multi_layer_axis_theta=ANGULAR_MULTI_LAYER_AXIS_THETA,
                    )
                    resolved = bool(
                        np.isfinite(ctf_dose_mean)
                        and ctf_dose_mean >= float(ctf_dose_threshold)
                    )
                    if np.isnan(previous_w) and resolved:
                        resolved_w[(dose_type, float(r_over_r))] = float(m)
                        if on_resolved_callback is not None:
                            on_resolved_callback(
                                {
                                    "structure_type": structure_type,
                                    "dose_type": dose_type,
                                    "m": int(m),
                                    "r_over_R": float(r_over_r),
                                    "r_px": float(analyzed_r_px),
                                    "target_mask": target_mask,
                                    "dose_field": dose_fields[dose_type],
                                    "center": center_xy,
                                    "container_r_px": float(r_px),
                                }
                            )
                    w_px = float(resolved_w[(dose_type, float(r_over_r))])
                    detail_rows.append(
                        {
                            "structure_type": structure_type,
                            "dose_type": dose_type,
                            "m": int(m),
                            "r_over_R": float(r_over_r),
                            "r_px": float(analyzed_r_px),
                            "CTF_dose_mean": float(ctf_dose_mean),
                            "CTF_dose_std": float(ctf_dose_std),
                            "resolved": resolved,
                            "w_px": w_px,
                        }
                    )
            continue

        if structure_type == STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER:
            for r_over_r in r_over_r_list:
                m_percent = float(m)
                analyzed_r_px = float(r_over_r) * float(r_px)
                target_mask = generate_angular_blocks_single_layer(
                    n,
                    r_px,
                    m_percent,
                    center_xy,
                    analyzed_r_px,
                )
                dose_fields = run_simulation_func(target_mask)
                for dose_type in dose_types:
                    previous_w = float(resolved_w[(dose_type, float(r_over_r))])
                    ctf_dose_mean, ctf_dose_std = compute_angular_block_ctf_dose_mean(
                        dose_fields[dose_type],
                        target_mask,
                        center_xy,
                        r_px,
                        analyzed_r_px,
                        m_percent,
                        num_theta_samples=angular_num_theta_samples,
                    )
                    resolved = bool(
                        np.isfinite(ctf_dose_mean)
                        and ctf_dose_mean >= float(ctf_dose_threshold)
                    )
                    if np.isnan(previous_w) and resolved:
                        resolved_w[(dose_type, float(r_over_r))] = float(m)
                        if on_resolved_callback is not None:
                            on_resolved_callback(
                                {
                                    "structure_type": structure_type,
                                    "dose_type": dose_type,
                                    "m": m_percent,
                                    "r_over_R": float(r_over_r),
                                    "r_px": float(analyzed_r_px),
                                    "target_mask": target_mask,
                                    "dose_field": dose_fields[dose_type],
                                    "center": center_xy,
                                    "container_r_px": float(r_px),
                                }
                            )
                    w_percent = float(resolved_w[(dose_type, float(r_over_r))])
                    m_px = angular_m_percent_to_px(m_percent, analyzed_r_px)
                    w_px = (
                        angular_m_percent_to_px(w_percent, analyzed_r_px)
                        if np.isfinite(w_percent)
                        else float("nan")
                    )
                    detail_rows.append(
                        {
                            "structure_type": structure_type,
                            "dose_type": dose_type,
                            "m": m_percent,
                            "m_percent": m_percent,
                            "m_px": m_px,
                            "r_over_R": float(r_over_r),
                            "r_px": float(analyzed_r_px),
                            "CTF_dose_mean": float(ctf_dose_mean),
                            "CTF_dose_std": float(ctf_dose_std),
                            "resolved": resolved,
                            "w_percent": w_percent,
                            "w_px": w_px,
                        }
                    )
            continue

        raise ValueError(f"Unsupported structure_type: {structure_type}")

    return detail_rows, resolved_w


def build_resolution_summary_rows(
    structure_type: str,
    r_over_r_list: list[float],
    r_px: float,
    resolved_w: dict[tuple[str, float], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r_over_r in r_over_r_list:
        analyzed_r_px = float(r_over_r) * float(r_px)
        proto_value = float(resolved_w.get(("proto", float(r_over_r)), float("nan")))
        optimized_value = float(resolved_w.get(("optimized", float(r_over_r)), float("nan")))
        is_percent_structure = structure_type in (
            STRUCTURE_RADIAL_CONCENTRIC_RING,
            STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER,
        )

        def resolved_percent_to_px(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            if structure_type == STRUCTURE_RADIAL_CONCENTRIC_RING:
                return radial_m_percent_to_px(value, r_px)
            if structure_type == STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER:
                return angular_m_percent_to_px(value, analyzed_r_px)
            return float(value)

        rows.append(
            {
                "structure_type": structure_type,
                "r_over_R": float(r_over_r),
                "r_px": analyzed_r_px,
                "w_proto_percent": proto_value if is_percent_structure else float("nan"),
                "w_proto_px": resolved_percent_to_px(proto_value),
                "w_optimized_percent": optimized_value if is_percent_structure else float("nan"),
                "w_optimized_px": resolved_percent_to_px(optimized_value),
            }
        )
    return rows


def _format_csv_value(value: Any, significant_digits: int | None = None) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    if significant_digits is not None and isinstance(value, (float, np.floating)):
        value_float = float(value)
        if not np.isfinite(value_float):
            return ""
        return f"{value_float:.{int(significant_digits)}g}"
    return value


def write_csv_rows(
    csv_path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    *,
    significant_digits: int | None = None,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key), significant_digits) for key in fieldnames})


def build_ctf_dose_heatmap_matrix(
    detail_rows: list[dict[str, Any]],
    dose_type: str,
    m_values: range | list[float],
    r_over_r_list: list[float],
) -> np.ndarray:
    m_values_list = [float(m) for m in m_values]
    matrix = np.full((len(r_over_r_list), len(m_values_list)), np.nan, dtype=np.float64)
    m_to_index = {float(m): idx for idx, m in enumerate(m_values_list)}
    r_to_index = {float(r_over_r): idx for idx, r_over_r in enumerate(r_over_r_list)}

    for row in detail_rows:
        if row["dose_type"] != dose_type:
            continue
        m_idx = m_to_index[float(row["m"])]
        r_idx = r_to_index[float(row["r_over_R"])]
        matrix[r_idx, m_idx] = float(row["CTF_dose_mean"])
    return matrix


def build_all_mistake_matrix(
    detail_rows: list[dict[str, Any]],
    dose_type: str,
    m_values: range | list[float],
    r_over_r_list: list[float],
) -> np.ndarray:
    m_values_list = [float(m) for m in m_values]
    matrix = np.zeros((len(r_over_r_list), len(m_values_list)), dtype=bool)
    m_to_index = {float(m): idx for idx, m in enumerate(m_values_list)}
    r_to_index = {float(r_over_r): idx for idx, r_over_r in enumerate(r_over_r_list)}

    for row in detail_rows:
        if row["dose_type"] != dose_type:
            continue
        m_idx = m_to_index[float(row["m"])]
        r_idx = r_to_index[float(row["r_over_R"])]
        matrix[r_idx, m_idx] = bool(row.get("all_mistake", False))
    return matrix


def plot_resolution_w_vs_r(
    summary_rows: list[dict[str, Any]],
    structure_type: str,
    dose_type: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([float(row["r_over_R"]) for row in summary_rows], dtype=np.float64)
    is_percent_structure = structure_type in (
        STRUCTURE_RADIAL_CONCENTRIC_RING,
        STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER,
    )
    field_name = f"w_{dose_type}_{'percent' if is_percent_structure else 'px'}"
    y = np.array([float(row[field_name]) for row in summary_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, y, marker="o", linewidth=2.0, label=dose_type)
    ax.set_xlabel("r/R")
    ax.set_ylabel("w / %" if is_percent_structure else "w / px")
    ax.set_title(f"{structure_type} {dose_type} resolution w vs r/R")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ctf_dose_heatmap(
    detail_rows: list[dict[str, Any]],
    structure_type: str,
    dose_type: str,
    m_values: range | list[float],
    r_over_r_list: list[float],
    save_path: str | Path,
    *,
    color_min: float = 0.0,
    color_max: float = 1.0,
    colorbar_ticks: list[float] | None = None,
    m_unit: str = "px",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    m_values_list = [float(m) for m in m_values]
    r_over_r_values = list(r_over_r_list)
    resolved_color_min = float(color_min)
    resolved_color_max = float(color_max)
    if not np.isfinite(resolved_color_min) or not np.isfinite(resolved_color_max):
        raise ValueError("CTF_dose heatmap color limits must be finite.")
    if resolved_color_max <= resolved_color_min:
        raise ValueError("CTF_dose heatmap color_max must be greater than color_min.")

    resolved_colorbar_ticks = (
        None
        if colorbar_ticks is None
        else [float(value) for value in colorbar_ticks]
    )
    if resolved_colorbar_ticks is not None and not all(
        np.isfinite(value) for value in resolved_colorbar_ticks
    ):
        raise ValueError("CTF_dose heatmap colorbar ticks must be finite.")

    if not m_values_list or not r_over_r_values:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.text(
            0.5,
            0.5,
            "No heatmap data\nempty m_values or r/R list",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"{structure_type} {dose_type} CTF_dose_mean heatmap")
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")

    matrix = build_ctf_dose_heatmap_matrix(detail_rows, dose_type, m_values_list, r_over_r_values)
    all_mistake_matrix = build_all_mistake_matrix(detail_rows, dose_type, m_values_list, r_over_r_values)
    r_step = 0.1 if len(r_over_r_values) == 1 else float(np.median(np.diff(sorted(r_over_r_values))))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=resolved_color_min,
        vmax=resolved_color_max,
        extent=[
            -0.5,
            len(m_values_list) - 0.5,
            min(r_over_r_values) - 0.5 * r_step,
            max(r_over_r_values) + 0.5 * r_step,
        ],
    )
    for spine in ax.spines.values():
        spine.set_zorder(2)
    for r_idx, r_over_r in enumerate(r_over_r_values):
        for m_idx, m in enumerate(m_values_list):
            if not all_mistake_matrix[r_idx, m_idx]:
                continue
            ax.add_patch(
                Rectangle(
                    (float(m_idx) - 0.5, float(r_over_r) - 0.5 * r_step),
                    1.0,
                    r_step,
                    fill=True,
                    facecolor="#d62728",
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.35,
                    clip_on=False,
                    zorder=10,
                )
            )
    ax.set_title(dose_type)
    ax.set_xlabel(f"m / {m_unit}")
    ax.set_xticks(range(len(m_values_list)))
    ax.set_xticklabels([f"{m:g}" for m in m_values_list], rotation=45, ha="right")
    ax.set_yticks(r_over_r_values)
    ax.set_ylabel("r/R")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=resolved_colorbar_ticks,
        label="CTF_dose_mean",
    )

    fig.suptitle(f"{structure_type} {dose_type} CTF_dose_mean heatmap")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    finite_values = matrix[np.isfinite(matrix)]
    over_max_count = int(np.count_nonzero(finite_values > resolved_color_max))
    below_min_count = int(np.count_nonzero(finite_values < resolved_color_min))
    if over_max_count or below_min_count:
        print(
            f"[CTF heatmap] warning: save_path={save_path} | "
            f"{over_max_count} value(s) > {resolved_color_max:g}, "
            f"{below_min_count} value(s) < {resolved_color_min:g}; "
            "these cells use endpoint colors and the original values are unchanged."
        )
