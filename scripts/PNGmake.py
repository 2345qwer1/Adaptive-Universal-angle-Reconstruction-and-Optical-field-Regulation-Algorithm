from __future__ import annotations

"""生成含黑色圆孔的 1080 × 1080 圆形容器 Target.png。

运行方式（在 Anaconda Prompt 或终端中）：
    python "F:/VsCodeProjects/VAM/CAL 1.2/scripts/PNGmake.py"
"""

from pathlib import Path
import math

import numpy as np
from PIL import Image


# ============================== 用户参数区 ==============================
# 输出 PNG 的绝对路径。
OUTPUT_PNG_PATH = Path(r"F:\USTC\项目\VAM\Limits_Imposed_by_Projection_Positivity\Target.png")

# 图像与圆形容器尺寸，单位均为 px；容器直径固定为图像边长。
OUTPUT_SVG_PATH = OUTPUT_PNG_PATH.with_suffix(".svg")

IMAGE_SIZE_PX = 1080
CONTAINER_DIAMETER_PX = 1080

# 黑色圆孔参数列表：每项为 (r_px, a_px, b_deg)。
# r_px：圆孔直径；a_px：圆孔中心到容器中心的距离；
# b_deg：从 X 正半轴逆时针测得的角度，0° 在右侧，90° 在图像上方。
# 示例：HOLES = [(20, 100, 0), (30, 200, 120), (15, 350, 270)]
HOLES = [(108, 0, 0), (108, 243, 360/8*0),(108, 243, 360/8*1),(108, 243, 360/8*2),(108, 243, 360/8*3),(108, 243, 360/8*4),
         (108, 243, 360/8*5),(108, 243, 360/8*6),(108, 243, 360/8*7),(108, 486, 360/16*0),(108, 486, 360/16*1),(108, 486, 360/16*2)
         ,(108, 486, 360/16*3),(108, 486, 360/16*4),(108, 486, 360/16*5),(108, 486, 360/16*6),(108, 486, 360/16*7),(108, 486, 360/16*8)
         ,(108, 486, 360/16*9),(108, 486, 360/16*10),(108, 486, 360/16*11),(108, 486, 360/16*12),(108, 486, 360/16*13),(108, 486, 360/16*14)
         ,(108, 486, 360/16*15)]
# ======================================================================


def validate_parameters() -> float:
    """验证容器和圆孔参数，并返回容器半径。"""
    if IMAGE_SIZE_PX <= 0 or IMAGE_SIZE_PX % 2 != 0:
        raise ValueError("IMAGE_SIZE_PX 必须是正偶数。")
    if CONTAINER_DIAMETER_PX <= 0 or CONTAINER_DIAMETER_PX > IMAGE_SIZE_PX:
        raise ValueError("CONTAINER_DIAMETER_PX 必须为正，且不大于 IMAGE_SIZE_PX。")

    container_radius = CONTAINER_DIAMETER_PX / 2.0
    for index, (diameter_px, distance_px, angle_deg) in enumerate(HOLES, start=1):
        if diameter_px <= 0.0:
            raise ValueError(f"第 {index} 个圆孔的 r_px 必须大于 0。")
        if not 0.0 <= distance_px <= container_radius:
            raise ValueError(f"第 {index} 个圆孔的 a_px 必须位于 [0, {container_radius:g}]。")
        if not 0.0 <= angle_deg <= 360.0:
            raise ValueError(f"第 {index} 个圆孔的 b_deg 必须位于 [0, 360]。")
        if distance_px + diameter_px / 2.0 > container_radius:
            raise ValueError(
                f"第 {index} 个圆孔超出容器：a_px + r_px/2 必须不大于 {container_radius:g}。"
            )
    return container_radius


def generate_target() -> np.ndarray:
    """返回容器内白色、圆孔及容器外黑色的灰度图。"""
    container_radius = validate_parameters()
    center = (IMAGE_SIZE_PX - 1) / 2.0
    y, x = np.ogrid[:IMAGE_SIZE_PX, :IMAGE_SIZE_PX]
    container_mask = (x - center) ** 2 + (y - center) ** 2 <= container_radius**2

    image = np.zeros((IMAGE_SIZE_PX, IMAGE_SIZE_PX), dtype=np.uint8)
    image[container_mask] = 255

    for diameter_px, distance_px, angle_deg in HOLES:
        angle_rad = math.radians(angle_deg)
        hole_center_x = center + distance_px * math.cos(angle_rad)
        hole_center_y = center - distance_px * math.sin(angle_rad)
        hole_radius = diameter_px / 2.0
        hole_mask = (x - hole_center_x) ** 2 + (y - hole_center_y) ** 2 <= hole_radius**2
        image[hole_mask & container_mask] = 0

    return image


def verify_output(image: np.ndarray) -> None:
    """验证尺寸与容器外黑色像素约束。"""
    expected_shape = (IMAGE_SIZE_PX, IMAGE_SIZE_PX)
    if image.shape != expected_shape:
        raise RuntimeError(f"输出尺寸错误：实际 {image.shape}，预期 {expected_shape}。")

    center = (IMAGE_SIZE_PX - 1) / 2.0
    container_radius = CONTAINER_DIAMETER_PX / 2.0
    y, x = np.ogrid[:IMAGE_SIZE_PX, :IMAGE_SIZE_PX]
    outside_container = (x - center) ** 2 + (y - center) ** 2 > container_radius**2
    if np.any(image[outside_container] != 0):
        raise RuntimeError("容器外存在非黑色像素。")


def save_svg(output_path: Path) -> None:
    """Write a vector SVG that matches the current target parameters."""
    center = (IMAGE_SIZE_PX - 1) / 2.0
    container_radius = CONTAINER_DIAMETER_PX / 2.0
    circles = [
        f'  <circle cx="{center:g}" cy="{center:g}" r="{container_radius:g}" fill="white" />'
    ]

    for diameter_px, distance_px, angle_deg in HOLES:
        angle_rad = math.radians(angle_deg)
        hole_center_x = center + distance_px * math.cos(angle_rad)
        hole_center_y = center - distance_px * math.sin(angle_rad)
        hole_radius = diameter_px / 2.0
        circles.append(
            f'  <circle cx="{hole_center_x:g}" cy="{hole_center_y:g}" r="{hole_radius:g}" fill="black" />'
        )

    svg_content = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{IMAGE_SIZE_PX}" '
                f'height="{IMAGE_SIZE_PX}" viewBox="0 0 {IMAGE_SIZE_PX} {IMAGE_SIZE_PX}">'
            ),
            f'  <rect width="{IMAGE_SIZE_PX}" height="{IMAGE_SIZE_PX}" fill="black" />',
            *circles,
            '</svg>',
            '',
        ]
    )
    output_path.write_text(svg_content, encoding="utf-8")


def main() -> None:
    image = generate_target()
    verify_output(image)
    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(OUTPUT_PNG_PATH, format="PNG")
    save_svg(OUTPUT_SVG_PATH)
    print(f"Target 已生成：{OUTPUT_PNG_PATH}")
    print(f"SVG 已生成：{OUTPUT_SVG_PATH}")
    print(f"图像尺寸：{IMAGE_SIZE_PX} × {IMAGE_SIZE_PX} px；圆孔数量：{len(HOLES)}")


if __name__ == "__main__":
    main()
