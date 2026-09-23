from __future__ import annotations

"""
run_initial_projection_and_print.py

用途：
    从 target 图像生成初始投影，执行经验补偿、量化和全局缩放，
    得到 proto projection，并模拟一次打印结果。

核心输出：
    1. proto_projection.npz：未经补偿缩放的初始投影量化版本。
    2. projection.npz：补偿、量化、全局缩放后的 proto 投影。
    3. initial_compare.png 及分图：target / dose / cured / error 对比。

注意：
    这是整个流程的第一步；后续 optimized 脚本通常读取这里生成的 projection.npz。
"""

from pathlib import Path
import sys

# 允许从 scripts/ 目录直接运行脚本时，仍能 import 项目根目录下的 functions 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions import (
    get_default_config,
    ensure_output_dirs,
    preprocess_target_image,
    generate_initial_sinogram,
    generate_initial_projection_matrix,
    compensate_and_print_initial_projection,
    visualize_and_save_results,
    save_projection_npz,
    quantize_projection_matrix,
)

# =========================
# 脚本层可修改参数
# =========================
# 原始初始投影的保存文件名，用于保留未经过补偿缩放的 baseline。
PROTO_PROJECTION_FILENAME = "proto_projection.npz"

# 补偿、量化、全局缩放后的 proto 投影文件名；optimized 脚本默认读取该文件。
COMPENSATED_PROJECTION_FILENAME = "projection.npz"

# target / dose / cured / error 总览图文件名。
COMPARE_IMAGE_FILENAME = "initial_compare.png"

# 分图输出文件名前缀。
FIGURE_PREFIX_NAME = "initial"

# 以下 CFG_* 为临时覆盖参数；None 表示沿用 get_default_config() 中的默认值。
CFG_N_SLICE = None
CFG_ANGLE_NUM = None
CFG_RESIN_ALPHA = None
CFG_RESIN_EC = None
CFG_USE_NUMBA = None


if __name__ == "__main__":
    # 1. 构建配置，并按脚本顶部的 CFG_* 覆盖默认配置。
    cfg = get_default_config()
    if CFG_N_SLICE is not None:
        cfg.n_slice = int(CFG_N_SLICE)
    if CFG_ANGLE_NUM is not None:
        cfg.angle_num = int(CFG_ANGLE_NUM)
    if CFG_RESIN_ALPHA is not None:
        cfg.resin_alpha = float(CFG_RESIN_ALPHA)
    if CFG_RESIN_EC is not None:
        cfg.resin_Ec = float(CFG_RESIN_EC)
    if CFG_USE_NUMBA is not None:
        cfg.use_numba = bool(CFG_USE_NUMBA)

    # 2. 准备输出目录和 target mask。
    dirs = ensure_output_dirs(cfg)
    target_mask = preprocess_target_image(cfg)

    # 3. target -> sinogram -> raw projection，生成未补偿的初始投影。
    initial_sinogram = generate_initial_sinogram(target_mask, cfg)
    raw_projection = generate_initial_projection_matrix(initial_sinogram, cfg)

    # 4. 保存 raw projection 的量化版本，便于后续比较或复现实验。
    raw_projection_quantized, raw_quant_meta = quantize_projection_matrix(raw_projection, cfg, return_metadata=True)

    # 5. 执行补偿、量化、全局缩放，并模拟 proto dose。
    result = compensate_and_print_initial_projection(raw_projection, target_mask, cfg)

    # 6. 统一构造输出路径。
    proto_path = dirs["projection"] / PROTO_PROJECTION_FILENAME
    weighted_path = dirs["projection"] / COMPENSATED_PROJECTION_FILENAME
    compare_path = dirs["compare"] / COMPARE_IMAGE_FILENAME
    figure_prefix = dirs["figure"] / FIGURE_PREFIX_NAME

    # 7. 保存未补偿初始投影及其元数据。
    save_projection_npz(
        proto_path,
        raw_projection_quantized,
        cfg,
        target_mask=target_mask,
        extra_metadata={
            "initial_sinogram": initial_sinogram,
            "projection_matrix_continuous": raw_projection,
            "quantization_reference_max_value": raw_quant_meta["reference_max_value"],
            "quantization_effective_step": raw_quant_meta["effective_step"],
        },
    )

    # 8. 保存补偿后的 proto 投影及其元数据。
    save_projection_npz(
        weighted_path,
        result["projection_matrix"],
        cfg,
        target_mask=target_mask,
        extra_metadata={
            "global_k": result["global_k"],
            "projection_matrix_continuous": result["projection_matrix_continuous"],
            "quantization_reference_max_value": result["quantization_reference_max_value"],
            "quantization_effective_step": result["quantization_effective_step"],
        },
    )

    print(f"初次打印容器内 IoU = {result['iou']:.4f}")
    print(f"全局缩放系数 k = {result['global_k']:.6f}")
    print(
        f"投影离散化: levels = 0 ... {cfg.projection_quantization_max_index}, "
        f"effective step = {result['quantization_effective_step']:.6g}"
    )

    # 9. 保存可视化结果。
    visualize_and_save_results(
        target_mask,
        result["dose_field"],
        result["cured_mask"],
        result["projection_matrix"],
        cfg,
        compare_save_path=compare_path,
        figure_prefix=figure_prefix,
    )

    print(f"量化后的原始初始投影已保存：{proto_path}")
    print(f"量化后的补偿投影已保存：{weighted_path}")
    print(f"compare 图已保存：{compare_path}")
