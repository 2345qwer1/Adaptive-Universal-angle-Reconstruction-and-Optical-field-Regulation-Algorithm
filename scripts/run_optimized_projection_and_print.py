from __future__ import annotations

"""
run_optimized_projection_and_print.py

用途：
    读取 run_initial_projection_and_print.py 生成的 proto 投影，
    在其基础上执行迭代优化，保存 optimized projection，并输出优化前后对比图。

核心输出：
    1. optimized_projection.npz：优化后的投影矩阵及元数据。
    2. optimized_compare.png 及分图：优化前后 cured / error / dose 对比。

注意：
    本脚本依赖 INPUT_PROJECTION_FILENAME 指向的投影文件存在；
    默认读取 projection.npz。
"""

from pathlib import Path
import sys
import numpy as np

# 允许从 scripts/ 目录直接运行脚本时，仍能 import 项目根目录下的 functions 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions import (
    get_default_config,
    ensure_output_dirs,
    preprocess_target_image,
    load_projection_npz,
    build_config_from_metadata,
    optimize_projection_matrix,
    print_any_projection,
    visualize_and_save_results,
    save_projection_npz,
)

# =========================
# 脚本层可修改参数
# =========================
# 输入投影文件名，默认读取初始打印脚本生成的 projection.npz。
INPUT_PROJECTION_FILENAME = "projection.npz"

# 优化后投影保存文件名。
OUTPUT_PROJECTION_FILENAME = "optimized_projection.npz"

# 优化前后对比图文件名。
COMPARE_IMAGE_FILENAME = "optimized_compare.png"

# 分图输出文件名前缀。
FIGURE_PREFIX_NAME = "optimized"

# error map 对比布局模式，由 visualize_and_save_results 内部使用。
ERROR_MAP_LAYOUT_MODE = "compare_2x3"

# 以下 CFG_* 为临时覆盖参数；None 表示沿用默认配置或投影元数据中的配置。
CFG_NUM_ITERATIONS = None
CFG_PRINT_INTERVAL = None
CFG_USE_NUMBA = None


if __name__ == "__main__":
    # 1. 构建默认配置，并按脚本顶部参数覆盖优化相关设置。
    cfg = get_default_config()
    if CFG_NUM_ITERATIONS is not None:
        cfg.num_iterations = int(CFG_NUM_ITERATIONS)
    if CFG_PRINT_INTERVAL is not None:
        cfg.print_interval = int(CFG_PRINT_INTERVAL)
    if CFG_USE_NUMBA is not None:
        cfg.use_numba = bool(CFG_USE_NUMBA)

    # 2. 准备输出目录、输入投影路径和 target mask。
    dirs = ensure_output_dirs(cfg)
    input_projection_path = dirs["projection"] / INPUT_PROJECTION_FILENAME
    target_mask = preprocess_target_image(cfg)

    # 3. 读取 proto 投影，并用保存时的元数据回填配置。
    initial_projection, metadata = load_projection_npz(input_projection_path)
    cfg = build_config_from_metadata(cfg, metadata)
    cfg.error_map_layout_mode = ERROR_MAP_LAYOUT_MODE

    # 4. 先打印未优化投影，作为优化前 baseline。
    quant_ref = float(metadata["quantization_reference_max_value"]) if "quantization_reference_max_value" in metadata else None
    baseline_result = print_any_projection(
        initial_projection,
        target_mask,
        cfg,
        quantization_reference_max_value=quant_ref,
    )

    # 5. 执行投影优化，得到 optimized projection 和对应 dose/cured。
    result = optimize_projection_matrix(initial_projection, target_mask, cfg)

    # 6. 构造输出路径。
    optimized_path = dirs["projection"] / OUTPUT_PROJECTION_FILENAME
    compare_path = dirs["compare"] / COMPARE_IMAGE_FILENAME
    figure_prefix = dirs["figure"] / FIGURE_PREFIX_NAME

    # 7. 保存优化后投影和优化历史摘要。
    save_projection_npz(
        optimized_path,
        result["projection_matrix"],
        cfg,
        target_mask=target_mask,
        extra_metadata={
            "history": np.array([str(result["history"])], dtype=object),
            "quantization_reference_max_value": result["quantization_reference_max_value"],
            "quantization_effective_step": result["quantization_effective_step"],
        },
    )

    print(f"优化前打印容器内 IoU = {baseline_result['iou']:.4f}")
    print(f"优化后最终量化输出 IoU = {result['iou']:.4f}")
    print(f"优化后离散有效步长 = {result['quantization_effective_step']:.6g}")

    # 8. 输出优化前后对比图。
    visualize_and_save_results(
        target_mask,
        result["dose_field"],
        result["cured_mask"],
        result["projection_matrix"],
        cfg,
        compare_save_path=compare_path,
        figure_prefix=figure_prefix,
        reference_cured=baseline_result["cured_mask"],
        reference_label="Unoptimized",
        current_label="Optimized",
    )

    print(f"优化投影已保存：{optimized_path}")
    print(f"compare 图已保存：{compare_path}")
