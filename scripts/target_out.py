from __future__ import annotations

"""
target_out.py

用途：
    独立生成 run_resolution_w_study_2.0.py 中 a1 / a2 / a3 三种结构的 target PNG。

结构对应关系：
    a1：径向同心环结构 radial_concentric_ring。
    a2：单层角向块结构 angular_blocks_single_layer。
    a3：多层角向块结构 angular_blocks_multi_layer。

输出：
    F:\\USTC\\项目\\VAM\\Target Image\\a1\\Target.png
    F:\\USTC\\项目\\VAM\\Target Image\\a2\\Target.png
    F:\\USTC\\项目\\VAM\\Target Image\\a3\\Target.png
"""

from pathlib import Path
import sys

import numpy as np
from skimage.io import imsave

# 允许从 scripts/ 目录直接运行本脚本时，仍能导入项目根目录下的 functions 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.resolution_w_analysis import (
    generate_angular_blocks_multi_layer,
    generate_angular_blocks_single_layer,
    generate_radial_concentric_rings,
)


# =============================================================================
# 用户可修改参数
# =============================================================================

# 需要生成的结构列表。固定支持 "a1"、"a2"、"a3"。
STRUCTURE_ALIASES = ["a1", "a2"]

# 所有 target PNG 的输出根目录。
# 每种结构会写入单独子目录，避免三个 Target.png 互相覆盖。
OUTPUT_DIR = r"F:\USTC\项目\VAM\Target Image"

# 每种结构子目录下的统一输出文件名。
OUTPUT_FILENAME = "Target.png"

# target 图像尺寸。1080 表示输出 1080 x 1080 的二值 PNG。
TARGET_IMAGE_SIZE_PX = 1080

# 容器半径，单位为像素。默认取图像半宽，与 run_resolution_w_study_2.0.py 保持一致。
TARGET_CONTAINER_RADIUS_PX = TARGET_IMAGE_SIZE_PX / 2.0

# 图像中心坐标，格式为 (center_x, center_y)。
TARGET_CENTER_XY = (TARGET_IMAGE_SIZE_PX / 2.0, TARGET_IMAGE_SIZE_PX / 2.0)

# a1 径向同心环结构参数。
# target ring 宽度 = A1_RING_WIDTH_PX，background gap 宽度也等于 A1_RING_WIDTH_PX。
A1_RING_WIDTH_PX = 20

# a2 单层角向块结构参数。
# A2_ARC_WIDTH_PX 表示在 A2_ANALYSIS_R_OVER_R 对应半径处的目标弧长宽度。
A2_ARC_WIDTH_PX = 20
A2_ANALYSIS_R_OVER_R = 0.5

# a3 多层角向块结构参数。
# A3_ARC_WIDTH_PX 表示每一层上的目标弧长宽度。
# A3_LAYER_R_OVER_R_LIST 表示生成哪些半径层。
A3_ARC_WIDTH_PX = 2
A3_LAYER_R_OVER_R_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# a3 多层角向块结构的对称轴。0.5*pi 对应图像中的 +Y 方向。
A3_AXIS_THETA = 0.5 * np.pi


def build_target_mask(structure_alias: str) -> np.ndarray:
    """
    根据结构别名生成二值 target mask。

    返回：
        uint8 数组，target 区域为 1，background 为 0。
    """
    alias = str(structure_alias).strip().lower()
    n = int(TARGET_IMAGE_SIZE_PX)
    container_r_px = float(TARGET_CONTAINER_RADIUS_PX)
    center = (float(TARGET_CENTER_XY[0]), float(TARGET_CENTER_XY[1]))

    if alias == "a1":
        return generate_radial_concentric_rings(
            n,
            container_r_px,
            int(A1_RING_WIDTH_PX),
            center,
        )

    if alias == "a2":
        analyzed_r_px = float(A2_ANALYSIS_R_OVER_R) * container_r_px
        return generate_angular_blocks_single_layer(
            n,
            container_r_px,
            int(A2_ARC_WIDTH_PX),
            center,
            analyzed_r_px,
        )

    if alias == "a3":
        return generate_angular_blocks_multi_layer(
            n,
            container_r_px,
            int(A3_ARC_WIDTH_PX),
            center,
            list(A3_LAYER_R_OVER_R_LIST),
            axis_theta=float(A3_AXIS_THETA),
        )

    raise ValueError(f"不支持的结构别名：{structure_alias}。可选值为 'a1', 'a2', 'a3'。")


def save_target_png(mask: np.ndarray, output_path: Path) -> None:
    """把二值 target mask 保存为黑白 PNG。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    imsave(str(output_path), image, check_contrast=False)


def main() -> None:
    """按 STRUCTURE_ALIASES 批量生成 target PNG。"""
    output_root = Path(OUTPUT_DIR)
    for alias in STRUCTURE_ALIASES:
        target_mask = build_target_mask(alias)
        output_path = output_root / str(alias) / OUTPUT_FILENAME
        save_target_png(target_mask, output_path)
        print(f"[target_out] saved {alias}: {output_path}")


if __name__ == "__main__":
    main()
