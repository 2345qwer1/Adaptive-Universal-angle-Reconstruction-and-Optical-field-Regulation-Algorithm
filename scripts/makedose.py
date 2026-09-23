from __future__ import annotations

"""从 NPZ 文件读取 dose 二维数组，并生成使用分段色尺的 SVG 热图。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, ListedColormap
import numpy as np


# ============================== 用户参数区 ==============================
# 输入 NPZ 文件的绝对路径；默认文件中的 dose 数组键名为 dose_field。
INPUT_NPZ_PATH = Path(
   r"F:\VsCodeProjects\VAM\CAL 1.2\_codex_tmp\819data\819data\attenuation_0p50\a2\m_iterations\m_050\rR_0.500\optimized_data.npz"
    
)

# 输出 SVG 文件的绝对路径。
OUTPUT_SVG_PATH = Path(
    r"F:\USTC\项目\VAM\论文图\heatmap\dose\dose.svg"

)

# 按顺序尝试从 NPZ 中读取这些键。
DOSE_KEY_CANDIDATES = ("dose_field", "dose")

# 固定的剂量分界值。
THRESHOLD_DOSE = 15.0

# 15 在最终色标中的固定位置，也是从原生 plasma 采样颜色的位置。
THRESHOLD_POSITION = 0.5

# 15 至最大 dose 段的幂律指数；小于 1 时增强靠近 15 的颜色变化。
HIGH_DOSE_POWER1 = 0.2

# 0 至 15 dose 段的反向幂律指数；小于 1 时增强靠近 15 的颜色变化。
HIGH_DOSE_POWER2 = 0.6
# 含负值时使用的固定显示下限，以及 0 在色标中的固定位置。
MIN_SIGNED_DOSE = -50.0
ZERO_COLORBAR_POSITION = 0.25

# 图形与导出参数。
FIGURE_SIZE_INCH = (5.4, 5.0)
SVG_RASTER_DPI = 300
COLORMAP_SAMPLES = 4096
# ======================================================================


def load_dose(npz_path: Path) -> tuple[np.ndarray, str]:
    """读取二维 dose 数组，并返回数组及实际使用的键名。"""
    if not npz_path.is_file():
        raise FileNotFoundError(f"未找到输入 NPZ 文件：{npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        dose_key = next((key for key in DOSE_KEY_CANDIDATES if key in data.files), None)
        if dose_key is None:
            raise KeyError(
                f"NPZ 中未找到 dose 数组；已尝试 {DOSE_KEY_CANDIDATES}，"
                f"实际键为 {tuple(data.files)}"
            )
        dose = np.asarray(data[dose_key], dtype=np.float64)

    if dose.ndim != 2:
        raise ValueError(f"dose 必须是二维数组，实际形状为 {dose.shape}")
    if not np.any(np.isfinite(dose)):
        raise ValueError("dose 中没有有限数值。")
    return dose, dose_key


def build_dose_colormap() -> ListedColormap:
    """构建在 15 处连续连接的双段 plasma 色图。"""
    if not 0.0 < THRESHOLD_POSITION < 1.0:
        raise ValueError("THRESHOLD_POSITION 必须位于 (0, 1)。")

    plasma = plt.get_cmap("plasma")
    colorbar_positions = np.arange(COLORMAP_SAMPLES, dtype=np.float64)
    colorbar_positions /= COLORMAP_SAMPLES
    threshold_index = min(
        int(THRESHOLD_POSITION * COLORMAP_SAMPLES),
        COLORMAP_SAMPLES - 1,
    )
    colors = np.empty((COLORMAP_SAMPLES, 4), dtype=np.float64)

    lower = np.arange(COLORMAP_SAMPLES) < threshold_index
    lower_positions = colorbar_positions[lower]
    colors[lower, :3] = plasma(lower_positions)[:, :3]

    upper = np.arange(COLORMAP_SAMPLES) >= threshold_index
    upper_fraction = np.clip(
        (colorbar_positions[upper] - THRESHOLD_POSITION)
        / (1.0 - THRESHOLD_POSITION),
        0.0,
        1.0,
    )
    upper_positions = THRESHOLD_POSITION + upper_fraction * (
        1.0 - THRESHOLD_POSITION
    )
    colors[upper, :3] = plasma(upper_positions)[:, :3]

    colors[threshold_index, :3] = plasma(THRESHOLD_POSITION)[:3]
    colors[-1, :3] = plasma(1.0)[:3]
    colors[:, 3] = 1.0
    cmap = ListedColormap(colors, name="dose_piecewise_plasma")
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


def build_dose_norm(vmax: float, threshold: float = THRESHOLD_DOSE) -> FuncNorm:
    """将 0、threshold、vmax 映射到色标的固定三段位置。"""
    if not np.isfinite(vmax) or vmax <= threshold:
        raise ValueError(f"vmax 必须大于 threshold，实际为 {vmax}")
    if not 0.0 < HIGH_DOSE_POWER1 <= 1.0:
        raise ValueError("HIGH_DOSE_POWER1 必须位于 (0, 1]。")
    if not 0.0 < HIGH_DOSE_POWER2 <= 1.0:
        raise ValueError("HIGH_DOSE_POWER2 必须位于 (0, 1]。")

    def forward(values):
        values = np.ma.asarray(values, dtype=np.float64)
        return np.ma.where(
            values <= threshold,
            THRESHOLD_POSITION
            * (
                1.0
                - np.clip(1.0 - values / threshold, 0.0, 1.0)
                ** HIGH_DOSE_POWER2
            ),
            THRESHOLD_POSITION
            + (1.0 - THRESHOLD_POSITION)
            * np.clip(
                (values - threshold) / (vmax - threshold), 0.0, 1.0
            )
            ** HIGH_DOSE_POWER1,
        )

    def inverse(positions):
        positions = np.ma.asarray(positions, dtype=np.float64)
        return np.ma.where(
            positions <= THRESHOLD_POSITION,
            threshold
            * (
                1.0
                - np.clip(1.0 - positions / THRESHOLD_POSITION, 0.0, 1.0)
                ** (1.0 / HIGH_DOSE_POWER2)
            ),
            threshold
            + (vmax - threshold)
            * (
                np.clip(
                    (positions - THRESHOLD_POSITION)
                    / (1.0 - THRESHOLD_POSITION),
                    0.0,
                    1.0,
                )
            )
            ** (1.0 / HIGH_DOSE_POWER1),
        )

    return FuncNorm((forward, inverse), vmin=0.0, vmax=vmax, clip=True)


def build_signed_dose_norm(
    vmax: float,
    threshold: float = THRESHOLD_DOSE,
    vmin: float = MIN_SIGNED_DOSE,
) -> FuncNorm:
    """将 vmin、0、threshold、vmax 映射到固定色标位置。"""
    if not np.isfinite(vmin) or vmin >= 0.0:
        raise ValueError(f"vmin 必须为有限负数，实际为 {vmin}")
    if not np.isfinite(vmax) or vmax <= threshold:
        raise ValueError(f"vmax 必须大于 threshold，实际为 {vmax}")
    if not 0.0 < HIGH_DOSE_POWER1 <= 1.0:
        raise ValueError("HIGH_DOSE_POWER1 必须位于 (0, 1]。")
    if not 0.0 < HIGH_DOSE_POWER2 <= 1.0:
        raise ValueError("HIGH_DOSE_POWER2 必须位于 (0, 1]。")
    if not 0.0 < ZERO_COLORBAR_POSITION < THRESHOLD_POSITION < 1.0:
        raise ValueError("色标位置必须满足 0 < zero < threshold < 1。")

    def forward(values):
        values = np.ma.asarray(values, dtype=np.float64)
        return np.ma.where(
            values <= 0.0,
            ZERO_COLORBAR_POSITION * (values - vmin) / (0.0 - vmin),
            np.ma.where(
                values <= threshold,
                ZERO_COLORBAR_POSITION
                + (THRESHOLD_POSITION - ZERO_COLORBAR_POSITION)
                * (
                    1.0
                    - np.clip(1.0 - values / threshold, 0.0, 1.0)
                    ** HIGH_DOSE_POWER2
                ),
                THRESHOLD_POSITION
                + (1.0 - THRESHOLD_POSITION)
                * np.clip(
                    (values - threshold) / (vmax - threshold), 0.0, 1.0
                )
                ** HIGH_DOSE_POWER1,
            ),
        )

    def inverse(positions):
        positions = np.ma.asarray(positions, dtype=np.float64)
        return np.ma.where(
            positions <= ZERO_COLORBAR_POSITION,
            vmin + (0.0 - vmin) * positions / ZERO_COLORBAR_POSITION,
            np.ma.where(
                positions <= THRESHOLD_POSITION,
                threshold
                * (
                    1.0
                    - np.clip(
                        1.0
                        - (positions - ZERO_COLORBAR_POSITION)
                        / (THRESHOLD_POSITION - ZERO_COLORBAR_POSITION),
                        0.0,
                        1.0,
                    )
                    ** (1.0 / HIGH_DOSE_POWER2)
                ),
                threshold
                + (vmax - threshold)
                * (
                    np.clip(
                        (positions - THRESHOLD_POSITION)
                        / (1.0 - THRESHOLD_POSITION),
                        0.0,
                        1.0,
                    )
                )
                ** (1.0 / HIGH_DOSE_POWER1),
            ),
        )

    return FuncNorm((forward, inverse), vmin=vmin, vmax=vmax, clip=True)


def colorbar_ticks(
    vmax: float,
    threshold: float,
    has_negative: bool = False,
) -> list[float]:
    """按 dose 是否含负值返回三个或四个关键刻度。"""
    if has_negative:
        return [MIN_SIGNED_DOSE, 0.0, float(threshold), float(vmax)]
    return [0.0, float(threshold), float(vmax)]


def save_dose_svg(dose: np.ndarray, output_path: Path) -> tuple[float, float]:
    """使用真实线性 dose 范围绘图并保存 SVG。"""
    finite_values = dose[np.isfinite(dose)]
    dose_min = float(np.min(finite_values))
    dose_max = float(np.max(finite_values))
    has_negative = dose_min < 0.0
    cmap = build_dose_colormap()
    cmap.set_bad((0.0, 0.0, 0.0, 1.0))
    if has_negative:
        norm = build_signed_dose_norm(dose_max)
    else:
        norm = build_dose_norm(dose_max)

    # 将图像中心内切圆以外的区域作为背景，不依据 dose 数值判断。
    height, width = dose.shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(height, width) / 2.0
    y_coordinates, x_coordinates = np.ogrid[:height, :width]
    outside_circle = (
        (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2
        > radius**2
    )
    masked_dose = np.ma.masked_invalid(dose)
    masked_dose = np.ma.masked_where(outside_circle, masked_dose)

    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCH)
    ax.set_facecolor("black")
    image = ax.imshow(
        masked_dose,
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_axis_off()
    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=colorbar_ticks(dose_max, THRESHOLD_DOSE, has_negative),
    )
    colorbar.ax.tick_params(
        axis="y",
        which="both",
        labelleft=False,
        labelright=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output_path,
        format="svg",
        dpi=SVG_RASTER_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    return dose_min, dose_max


def main() -> None:
    dose, dose_key = load_dose(INPUT_NPZ_PATH)
    dose_min, dose_max = save_dose_svg(dose, OUTPUT_SVG_PATH)
    print(f"输入文件：{INPUT_NPZ_PATH}")
    print(f"dose 键名：{dose_key}")
    print(f"dose 尺寸：{dose.shape[1]} × {dose.shape[0]}")
    print(f"dose 范围：{dose_min:.12g} ～ {dose_max:.12g}")
    print(f"dose 最大值：{dose_max:.12g}")
    print(f"SVG 已生成：{OUTPUT_SVG_PATH}")


if __name__ == "__main__":
    main()
