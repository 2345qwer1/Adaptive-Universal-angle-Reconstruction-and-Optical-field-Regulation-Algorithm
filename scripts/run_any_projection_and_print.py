from __future__ import annotations

"""
run_any_projection_and_print.py

用途：
    读取已经保存的投影矩阵 npz 文件，重新执行一次 dose forward simulation，
    并输出 target / dose / cured / error 的对比图。

典型使用场景：
    1. 已经有 projection.npz、proto_projection.npz 或 optimized_projection.npz。
    2. 想在不重新生成投影的情况下，检查某个投影文件的打印效果和 IoU。

注意：
    本脚本不会优化投影，也不会生成新的投影矩阵；它只负责“加载已有投影并打印验证”。
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
    load_projection_npz,
    build_config_from_metadata,
    print_any_projection,
    visualize_and_save_results,
)

# =========================
# 脚本层可修改参数
# =========================
# 要测试的投影文件名。文件会从 cfg.output_dir / "projection_result" 下读取。
PROJECTION_FILENAME = "projection.npz"

# compare 图文件名后缀。最终文件名会使用投影文件 stem + 该后缀。
COMPARE_NAME_SUFFIX = "_compare.png"

# figure 输出前缀。None 表示自动使用投影文件 stem。
FIGURE_PREFIX_NAME = None


if __name__ == "__main__":
    # 1. 读取默认配置，并定位待测试投影文件。
    cfg = get_default_config()
    projection_to_test = Path(cfg.output_dir) / "projection_result" / PROJECTION_FILENAME

    # 2. 准备输出目录、target mask，并从 npz 中读取投影矩阵及保存时的元数据。
    dirs = ensure_output_dirs(cfg)
    target_mask = preprocess_target_image(cfg)
    projection_matrix, metadata = load_projection_npz(projection_to_test)

    # 3. 用投影文件中的元数据回填配置，避免打印参数与投影生成时不一致。
    cfg = build_config_from_metadata(cfg, metadata)

    # 4. 如果投影保存了量化参考值，则沿用同一参考值执行打印验证。
    quant_ref = float(metadata["quantization_reference_max_value"]) if "quantization_reference_max_value" in metadata else None
    result = print_any_projection(projection_matrix, target_mask, cfg, quantization_reference_max_value=quant_ref)

    # 5. 构造输出文件名。compare 是总览图，figure_prefix 控制分图输出前缀。
    compare_name = Path(projection_to_test).stem + COMPARE_NAME_SUFFIX
    compare_path = dirs["compare"] / compare_name
    prefix_name = FIGURE_PREFIX_NAME if FIGURE_PREFIX_NAME is not None else Path(projection_to_test).stem
    figure_prefix = dirs["figure"] / prefix_name

    print(f"待测投影文件：{projection_to_test}")
    print(f"打印容器内 IoU = {result['iou']:.4f}")
    print(f"打印使用离散有效步长 = {result['quantization_effective_step']:.6g}")

    # 6. 保存 target、dose、cured、error、projection 等可视化结果。
    visualize_and_save_results(
        target_mask,
        result["dose_field"],
        result["cured_mask"],
        result["projection_matrix"],
        cfg,
        compare_save_path=compare_path,
        figure_prefix=figure_prefix,
    )

    print(f"compare 图已保存：{compare_path}")
