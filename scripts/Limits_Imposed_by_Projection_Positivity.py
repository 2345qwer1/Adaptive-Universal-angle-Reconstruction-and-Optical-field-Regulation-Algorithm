from __future__ import annotations

"""
Limits_Imposed_by_Projection_Positivity.py

用途：
    对同一个 target 执行“投影非负约束”的消融实验。

实验问题：
    Radon 变换后的频域滤波可能产生负值。
    原 VAM 流程会把负投影裁剪为 0，这相当于强制投影能量非负。
    本脚本比较：
        A. no_negative：原流程，滤波后负值被裁剪，优化等级为 0 ... K。
        B. allowed_negative：消融流程，滤波后负值保留，优化等级为 -K ... K。

核心输出：
    同一个 target、同一套 cfg、相同 num_iterations 下，
    两种方案的最终 projection / dose / cured / error / IoU 对比。
"""

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

# 允许从 scripts/ 目录直接运行本脚本时，仍能导入项目根目录下的 functions 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.config import get_default_config
from functions.initial_projection import generate_initial_projection_matrix
from functions.io_utils import save_projection_npz
from functions.optimization import compute_gradient, compute_loss_and_grad_wrt_dose
from functions.physics_printing import calculate_iou, compensate_and_print_initial_projection, forward_dose_simulation
from functions.sinogram import generate_initial_sinogram, preprocess_target_image
from functions.visualization import build_compare_map, visualize_and_save_results
from functions.projection_positivity_ablation import (
    compensate_and_print_initial_projection_signed,
    generate_initial_projection_matrix_signed,
)


# =============================================================================
# 脚本层可修改参数
# =============================================================================

# 本次消融实验的绝对输出根目录。
# 所有 no_negative / allowed_negative / summary / comparison 输出都会写入该目录。
EXPERIMENT_OUTPUT_DIR = r"F:\USTC\项目\VAM\Limits_Imposed_by_Projection_Positivity"

# 是否运行优化阶段。
# True：比较两种方案在相同 num_iterations 后的最终 optimized print out。
# False：只比较 proto 阶段 print out。
RUN_OPTIMIZATION = True

# 是否保存两种方案各自的 projection / dose / cured / error 分图。
SAVE_INDIVIDUAL_FIGURES = True

# 是否保存两种方案并排比较图。
SAVE_COMPARISON_FIGURE = True

# 保存并排比较图时，是否同时输出同名 SVG 文件。
SAVE_COMPARISON_SVG = True

# 是否保存 projection npz 文件，便于后续复现实验或单独打印。
SAVE_PROJECTION_NPZ = True

# 是否保存每种方案在各记录迭代点的最终 dose npz 文件。
SAVE_DOSE_NPZ = True

# 是否弹出 matplotlib 窗口。批量运行建议保持 False。
SHOW_FIGURES = False

# 以下 CFG_* 参数为本脚本临时覆盖项；None 表示沿用 get_default_config() 的默认值。
# 本实验使用的 target 图像路径。
TARGET_IMAGE_PATH = r"F:\USTC\项目\VAM\Limits_Imposed_by_Projection_Positivity\Target.png"

# 本实验需要记录输出的优化迭代次数。
# 脚本只会优化到最大值，并在这些迭代次数处保存快照，避免重复运行。
RECORD_ITERATIONS = [5,10]

# 优化过程的终端进度打印间隔。
# 例如设置为 5 时，每 5 次迭代打印一次；设置为 0 或负数时关闭固定间隔打印。
PROGRESS_PRINT_INTERVAL = 5

# 本实验两种方案使用相同的优化模式。
# 可选值：
#   "greedy_stop"：候选更新不改善时提前停止。
#   "explore_best"：持续迭代到 max(RECORD_ITERATIONS)，并在 RECORD_ITERATIONS 处记录历史 best。
OPTIMIZATION_MODE = "greedy_stop"

# 以下 CFG_* 参数为高级临时覆盖项；None 表示沿用 get_default_config() 的默认值。
CFG_OUTPUT_DIR = None
CFG_N_SLICE = None
CFG_ANGLE_NUM = None
CFG_FILTER_TYPE = None
CFG_PRINT_INTERVAL = None
CFG_PROJECTION_QUANTIZATION_ENABLED = None
CFG_PROJECTION_QUANTIZATION_MAX_INDEX = None
CFG_USE_NUMBA = None


def apply_script_overrides(cfg):
    """
    应用脚本顶部的 CFG_* 覆盖参数。

    这样所有用户常改参数都集中在文件开头，
    主流程只负责执行实验，不再散落配置。
    """
    cfg.target_image_path = str(TARGET_IMAGE_PATH)
    cfg.optimization_mode = str(OPTIMIZATION_MODE)

    if CFG_OUTPUT_DIR is not None:
        cfg.output_dir = str(CFG_OUTPUT_DIR)
    if CFG_N_SLICE is not None:
        cfg.n_slice = int(CFG_N_SLICE)
    if CFG_ANGLE_NUM is not None:
        cfg.angle_num = int(CFG_ANGLE_NUM)
    if CFG_FILTER_TYPE is not None:
        cfg.filter_type = str(CFG_FILTER_TYPE)
    if CFG_PRINT_INTERVAL is not None:
        cfg.print_interval = int(CFG_PRINT_INTERVAL)
    if CFG_PROJECTION_QUANTIZATION_ENABLED is not None:
        cfg.projection_quantization_enabled = bool(CFG_PROJECTION_QUANTIZATION_ENABLED)
    if CFG_PROJECTION_QUANTIZATION_MAX_INDEX is not None:
        cfg.projection_quantization_max_index = int(CFG_PROJECTION_QUANTIZATION_MAX_INDEX)
    if CFG_USE_NUMBA is not None:
        cfg.use_numba = bool(CFG_USE_NUMBA)
    cfg.show_figures = bool(SHOW_FIGURES)
    return cfg


def projection_summary(projection: np.ndarray) -> dict[str, float]:
    """统计投影矩阵中的正负值分布，专门用于本消融实验的 summary.csv。"""
    proj = np.asarray(projection, dtype=np.float64)
    total = int(proj.size)
    negative_count = int(np.count_nonzero(proj < 0.0))
    positive_count = int(np.count_nonzero(proj > 0.0))
    return {
        "projection_min": float(np.min(proj)) if total > 0 else 0.0,
        "projection_max": float(np.max(proj)) if total > 0 else 0.0,
        "projection_mean": float(np.mean(proj)) if total > 0 else 0.0,
        "projection_abs_max": float(np.max(np.abs(proj))) if total > 0 else 0.0,
        "projection_negative_fraction": float(negative_count / total) if total > 0 else 0.0,
        "projection_positive_fraction": float(positive_count / total) if total > 0 else 0.0,
    }


def result_summary(label: str, stage: str, result: dict, target_mask: np.ndarray, cfg) -> dict[str, float | str]:
    """把一次 print out 结果整理为 CSV 的一行。"""
    projection = np.asarray(result["projection_matrix"], dtype=np.float64)
    dose = np.asarray(result["dose_field"], dtype=np.float64)
    cured = np.asarray(result["cured_mask"], dtype=np.float64)
    row: dict[str, float | str] = {
        "scheme": label,
        "stage": stage,
        "iou": float(result.get("iou", calculate_iou(target_mask, cured, cfg=cfg))),
        "dose_min": float(np.min(dose)),
        "dose_max": float(np.max(dose)),
        "dose_mean": float(np.mean(dose)),
        "cured_area_px": int(np.count_nonzero(cured > 0.5)),
        "target_area_px": int(np.count_nonzero(target_mask > 0.5)),
        "num_iterations": int(cfg.num_iterations),
        "optimization_mode": str(cfg.optimization_mode),
    }
    row.update(projection_summary(projection))
    return row


def write_summary_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    """保存两种方案的指标总表。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_comparison_figure(
    fig_path: Path,
    target_mask: np.ndarray,
    positive_result: dict,
    signed_result: dict,
    cfg,
) -> None:
    """
    保存两种最终 print out 的并排比较图。

    图像布局：
        第一行：no_negative。
        第二行：allowed_negative。

    每行列含义：
        1. target
        2. dose
        3. print out，也就是最终 cured mask
        4. error 对比图，并在图下方标明 undercure / overcure / IoU。
    """
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    positive_dose = np.asarray(positive_result["dose_field"], dtype=np.float64)
    signed_dose = np.asarray(signed_result["dose_field"], dtype=np.float64)
    positive_cured = np.asarray(positive_result["cured_mask"], dtype=np.float64)
    signed_cured = np.asarray(signed_result["cured_mask"], dtype=np.float64)

    positive_compare = build_compare_map(target_mask, positive_cured)
    signed_compare = build_compare_map(target_mask, signed_cured)

    error_cmap = plt.matplotlib.colors.ListedColormap(["white", "red", "green"])

    def error_metrics(cured: np.ndarray, result: dict) -> tuple[int, int, float]:
        target_bool = np.asarray(target_mask).astype(bool)
        cured_bool = np.asarray(cured).astype(bool)
        undercure_px = int(np.count_nonzero(np.logical_and(target_bool, np.logical_not(cured_bool))))
        overcure_px = int(np.count_nonzero(np.logical_and(np.logical_not(target_bool), cured_bool)))
        return undercure_px, overcure_px, float(result["iou"])

    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.4))
    rows = [
        ("no_negative", positive_dose, positive_cured, positive_compare, positive_result),
        ("allowed_negative", signed_dose, signed_cured, signed_compare, signed_result),
    ]

    for row_idx, (scheme_name, dose, cured, compare_map, result) in enumerate(rows):
        row_axes = axes[row_idx]
        row_axes[0].imshow(target_mask, cmap="gray", origin="lower")
        row_axes[0].set_title(f"{scheme_name}\ntarget", fontsize=9)

        im_dose = row_axes[1].imshow(dose, cmap="plasma", origin="lower")
        row_axes[1].set_title("dose", fontsize=9)
        fig.colorbar(im_dose, ax=row_axes[1], fraction=0.046, pad=0.04)

        row_axes[2].imshow(cured, cmap="gray", origin="lower")
        row_axes[2].set_title("print out", fontsize=9)

        error_map = np.zeros_like(compare_map)
        error_map[compare_map == 2] = 1  # undercure
        error_map[compare_map == 3] = 2  # overcure
        row_axes[3].imshow(error_map, cmap=error_cmap, vmin=0, vmax=2, origin="lower")
        row_axes[3].set_title("error", fontsize=9)
        undercure_px, overcure_px, iou = error_metrics(cured, result)
        row_axes[3].set_xlabel(
            f"undercure={undercure_px} px\n"
            f"overcure={overcure_px} px | IoU={iou:.4f}",
            fontsize=8,
        )

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Limits Imposed by Projection Positivity\n"
        f"iterations={cfg.num_iterations}, mode={cfg.optimization_mode}, "
        f"levels K={cfg.projection_quantization_max_index}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[positivity-ablation] comparison PNG saved = {fig_path}")
    if SAVE_COMPARISON_SVG:
        svg_path = fig_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", dpi=200, bbox_inches="tight")
        print(f"[positivity-ablation] comparison SVG saved = {svg_path}")
    if cfg.show_figures:
        plt.show()
    else:
        plt.close(fig)


def save_scheme_outputs(
    scheme_dir: Path,
    scheme_name: str,
    cfg,
    target_mask: np.ndarray,
    proto_result: dict,
    final_result: dict,
) -> None:
    """保存单个方案的 projection、dose npz 和可视化结果。"""
    scheme_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_DOSE_NPZ:
        dose_path = scheme_dir / "dose.npz"
        dose_field = np.asarray(final_result["dose_field"], dtype=np.float64)
        np.savez_compressed(dose_path, dose_field=dose_field)
        print(f"[positivity-ablation] dose NPZ saved = {dose_path}")

    if SAVE_PROJECTION_NPZ:
        save_projection_npz(
            scheme_dir / "proto_projection.npz",
            np.asarray(proto_result["projection_matrix"], dtype=np.float64),
            cfg,
            target_mask=target_mask,
            extra_metadata={
                "scheme": scheme_name,
                "stage": "proto",
                "iou": float(proto_result["iou"]),
            },
        )
        save_projection_npz(
            scheme_dir / "final_projection.npz",
            np.asarray(final_result["projection_matrix"], dtype=np.float64),
            cfg,
            target_mask=target_mask,
            extra_metadata={
                "scheme": scheme_name,
                "stage": "optimized" if RUN_OPTIMIZATION else "proto",
                "iou": float(final_result["iou"]),
                "history": np.array([str(final_result.get("history", {}))], dtype=object),
            },
        )

    if SAVE_INDIVIDUAL_FIGURES:
        visualize_and_save_results(
            target_mask,
            np.asarray(final_result["dose_field"], dtype=np.float64),
            np.asarray(final_result["cured_mask"], dtype=np.float64),
            np.asarray(final_result["projection_matrix"], dtype=np.float64),
            cfg,
            compare_save_path=scheme_dir / "final_compare.png",
            figure_prefix=scheme_dir / "final",
            reference_cured=np.asarray(proto_result["cured_mask"], dtype=np.float64),
            reference_label=f"{scheme_name} proto",
            current_label=f"{scheme_name} final",
        )


def normalize_record_iterations(record_iterations: list[int]) -> list[int]:
    """Return sorted positive iteration checkpoints."""
    values = sorted({int(value) for value in record_iterations if int(value) > 0})
    if not values:
        raise ValueError("RECORD_ITERATIONS must contain at least one positive integer.")
    return values


def is_better_state(candidate_iou: float, candidate_loss: float, best_iou: float, best_loss: float) -> bool:
    """Use the same best-state rule as the optimization module."""
    if candidate_iou > best_iou:
        return True
    if np.isclose(candidate_iou, best_iou, rtol=0.0, atol=1e-12) and candidate_loss < best_loss:
        return True
    return False


def build_snapshot_result(
    *,
    projection: np.ndarray,
    dose: np.ndarray,
    cured: np.ndarray,
    iou: float,
    history: dict[str, object],
    quantization_key: str,
    quantization_reference_value: float,
    quantization_effective_step: float,
) -> dict[str, object]:
    """Build an optimizer-like result dictionary for one recorded iteration."""
    return {
        "projection_matrix": np.asarray(projection, dtype=np.float64).copy(),
        "dose_field": np.asarray(dose, dtype=np.float64).copy(),
        "cured_mask": np.asarray(cured, dtype=np.float64).copy(),
        "iou": float(iou),
        "history": {key: list(value) if isinstance(value, list) else value for key, value in history.items()},
        quantization_key: float(quantization_reference_value),
        "quantization_effective_step": float(quantization_effective_step),
    }


def optimize_projection_with_snapshots(
    initial_projection: np.ndarray,
    target_mask: np.ndarray,
    cfg,
    record_iterations: list[int],
    *,
    signed: bool,
    scheme_name: str,
) -> dict[int, dict[str, object]]:
    """Optimize once to max(record_iterations) and store best-state snapshots."""
    record_iterations = normalize_record_iterations(record_iterations)
    max_iterations = max(record_iterations)
    record_set = set(record_iterations)
    optimization_mode = str(getattr(cfg, "optimization_mode", "greedy_stop")).strip()
    if optimization_mode not in {"greedy_stop", "explore_best"}:
        raise ValueError("optimization_mode must be 'greedy_stop' or 'explore_best'.")

    proj_input = np.asarray(initial_projection, dtype=np.float64)
    if not signed:
        proj_input = np.clip(proj_input, 0.0, None)
    K = int(cfg.projection_quantization_max_index)
    if K < 0:
        raise ValueError("projection_quantization_max_index must be >= 0.")

    reference_value = float(np.max(np.abs(proj_input))) if signed and proj_input.size > 0 else (
        float(np.max(proj_input)) if proj_input.size > 0 else 0.0
    )
    level_min = -K if signed else 0
    level_max = K
    quantization_key = "quantization_reference_abs_value" if signed else "quantization_reference_max_value"

    def make_result() -> dict[str, object]:
        return build_snapshot_result(
            projection=proj_best,
            dose=dose_best,
            cured=cured_best,
            iou=iou_best,
            history=history,
            quantization_key=quantization_key,
            quantization_reference_value=reference_value,
            quantization_effective_step=effective_step,
        )

    snapshots: dict[int, dict[str, object]] = {}
    if reference_value <= 0.0 or K == 0:
        effective_step = 0.0
        proj_best = np.zeros_like(proj_input, dtype=np.float64)
        dose_best = forward_dose_simulation(proj_best, cfg)
        cured_best = (dose_best >= cfg.resin_Ec).astype(np.float64)
        iou_best = calculate_iou(target_mask, cured_best, cfg=cfg)
        loss_best, _ = compute_loss_and_grad_wrt_dose(dose_best, target_mask, cfg)
        history = {
            "mode": optimization_mode,
            "snapshot": True,
            "signed_levels": bool(signed),
            "ious": [float(iou_best)],
            "losses": [float(loss_best)],
            "best_ious": [float(iou_best)],
            "best_losses": [float(loss_best)],
        }
        result = make_result()
        return {iteration: result for iteration in record_iterations}

    effective_step = reference_value / float(K)
    level_current = np.rint(proj_input / effective_step).astype(np.int64)
    level_current = np.clip(level_current, level_min, level_max)
    proj_current = level_current.astype(np.float64) * effective_step

    dose_current = forward_dose_simulation(proj_current, cfg)
    loss_current, grad_wrt_dose_current = compute_loss_and_grad_wrt_dose(dose_current, target_mask, cfg)
    cured_current = (dose_current >= cfg.resin_Ec).astype(np.float64)
    iou_current = calculate_iou(target_mask, cured_current, cfg=cfg)

    proj_best = proj_current.copy()
    dose_best = dose_current.copy()
    cured_best = cured_current.copy()
    iou_best = float(iou_current)
    loss_best = float(loss_current)
    history = {
        "mode": optimization_mode,
        "snapshot": True,
        "signed_levels": bool(signed),
        "ious": [float(iou_current)],
        "losses": [float(loss_current)],
        "best_ious": [float(iou_best)],
        "best_losses": [float(loss_best)],
        "changed_entries": [0],
        "accepted": [True],
    }

    stopped = False
    for it in range(max_iterations):
        iteration = it + 1
        grad = compute_gradient(proj_current, grad_wrt_dose_current, cfg)
        grad_sign = np.sign(grad).astype(np.int8)
        level_candidate = np.clip(level_current - grad_sign.astype(np.int64), level_min, level_max)
        changed_entries = int(np.count_nonzero(level_candidate != level_current))

        if changed_entries == 0:
            accept = False
            if optimization_mode == "greedy_stop":
                stopped = True
        else:
            proj_candidate = level_candidate.astype(np.float64) * effective_step
            dose_candidate = forward_dose_simulation(proj_candidate, cfg)
            loss_candidate, grad_wrt_dose_candidate = compute_loss_and_grad_wrt_dose(dose_candidate, target_mask, cfg)
            cured_candidate = (dose_candidate >= cfg.resin_Ec).astype(np.float64)
            iou_candidate = calculate_iou(target_mask, cured_candidate, cfg=cfg)

            if optimization_mode == "greedy_stop":
                accept = (
                    (loss_candidate < loss_current)
                    or (
                        np.isclose(loss_candidate, loss_current, rtol=0.0, atol=1e-12)
                        and iou_candidate > iou_current
                    )
                )
                if not accept:
                    stopped = True
            else:
                accept = True

            if accept:
                level_current = level_candidate
                proj_current = proj_candidate
                dose_current = dose_candidate
                loss_current = float(loss_candidate)
                cured_current = cured_candidate
                iou_current = float(iou_candidate)

                if is_better_state(iou_current, loss_current, iou_best, loss_best):
                    proj_best = proj_current.copy()
                    dose_best = dose_current.copy()
                    cured_best = cured_current.copy()
                    iou_best = float(iou_current)
                    loss_best = float(loss_current)

        history["ious"].append(float(iou_current))
        history["losses"].append(float(loss_current))
        history["best_ious"].append(float(iou_best))
        history["best_losses"].append(float(loss_best))
        history["changed_entries"].append(changed_entries)
        history["accepted"].append(bool(accept))

        print_interval = int(PROGRESS_PRINT_INTERVAL)
        should_print_progress = (
            (print_interval > 0 and iteration % print_interval == 0)
            or iteration in record_set
            or iteration == max_iterations
            or stopped
        )
        if should_print_progress:
            snapshot_note = " | snapshot" if iteration in record_set else ""
            stop_note = " | stopped" if stopped else ""
            print(
                f"[{scheme_name}] iter {iteration}/{max_iterations}{snapshot_note}{stop_note} | "
                f"current IoU={float(iou_current):.4f} | "
                f"best IoU={float(iou_best):.4f} | "
                f"loss={float(loss_current):.6e} | "
                f"changed={changed_entries} | "
                f"accepted={bool(accept)}"
            )

        if iteration in record_set:
            snapshots[iteration] = make_result()
        if stopped:
            result = make_result()
            for remaining in record_iterations:
                snapshots.setdefault(remaining, result)
            break

    if snapshots:
        last_result = snapshots[max(snapshots)]
        for iteration in record_iterations:
            snapshots.setdefault(iteration, last_result)
    return snapshots


def save_iteration_snapshot_outputs(
    *,
    cfg,
    base_output_dir: Path,
    target_mask: np.ndarray,
    iteration: int,
    positive_proto: dict,
    positive_final: dict,
    signed_proto: dict,
    signed_final: dict,
) -> list[dict[str, float | str]]:
    """Save one recorded iteration snapshot using the existing output layout."""
    cfg.num_iterations = int(iteration)
    experiment_dir = base_output_dir / f"iter_{int(iteration):03d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    save_scheme_outputs(
        experiment_dir / "no_negative",
        "no_negative",
        cfg,
        target_mask,
        positive_proto,
        positive_final,
    )
    save_scheme_outputs(
        experiment_dir / "allowed_negative",
        "allowed_negative",
        cfg,
        target_mask,
        signed_proto,
        signed_final,
    )

    stage_name = "optimized" if RUN_OPTIMIZATION else "proto"
    rows = [
        result_summary("no_negative", stage_name, positive_final, target_mask, cfg),
        result_summary("allowed_negative", stage_name, signed_final, target_mask, cfg),
    ]
    for row in rows:
        row["output_dir"] = str(experiment_dir)
    write_summary_csv(experiment_dir / "summary.csv", rows)

    if SAVE_COMPARISON_FIGURE:
        save_ablation_comparison_figure(
            experiment_dir / "positivity_ablation_comparison.png",
            target_mask,
            positive_final,
            signed_final,
            cfg,
        )

    print(f"[positivity-ablation] snapshot saved iteration={int(iteration)} -> {experiment_dir}")
    return rows


def main() -> None:
    """Run all configured projection-positivity ablation experiments."""
    cfg = apply_script_overrides(get_default_config())
    base_output_dir = Path(EXPERIMENT_OUTPUT_DIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    record_iterations = normalize_record_iterations(list(RECORD_ITERATIONS))
    max_iterations = max(record_iterations)

    print("[positivity-ablation] start")
    print(f"[positivity-ablation] base_output_dir = {base_output_dir}")
    print(f"[positivity-ablation] record_iterations = {record_iterations}")
    print(f"[positivity-ablation] max_iterations = {max_iterations}")
    print(f"[positivity-ablation] optimization_mode = {cfg.optimization_mode}")

    target_mask = preprocess_target_image(cfg)
    initial_sinogram = generate_initial_sinogram(target_mask, cfg)

    positive_raw_projection = generate_initial_projection_matrix(initial_sinogram, cfg)
    positive_proto = compensate_and_print_initial_projection(positive_raw_projection, target_mask, cfg)

    signed_raw_projection = generate_initial_projection_matrix_signed(initial_sinogram, cfg)
    signed_proto = compensate_and_print_initial_projection_signed(signed_raw_projection, target_mask, cfg)

    cfg.num_iterations = int(max_iterations)
    if RUN_OPTIMIZATION:
        positive_snapshots = optimize_projection_with_snapshots(
            np.asarray(positive_proto["projection_matrix"], dtype=np.float64),
            target_mask,
            cfg,
            record_iterations,
            signed=False,
            scheme_name="no_negative",
        )
        signed_snapshots = optimize_projection_with_snapshots(
            np.asarray(signed_proto["projection_matrix"], dtype=np.float64),
            target_mask,
            cfg,
            record_iterations,
            signed=True,
            scheme_name="allowed_negative",
        )
    else:
        positive_snapshots = {iteration: positive_proto for iteration in record_iterations}
        signed_snapshots = {iteration: signed_proto for iteration in record_iterations}

    all_rows: list[dict[str, float | str]] = []
    for iteration in record_iterations:
        rows = save_iteration_snapshot_outputs(
            cfg=cfg,
            base_output_dir=base_output_dir,
            target_mask=target_mask,
            iteration=int(iteration),
            positive_proto=positive_proto,
            positive_final=positive_snapshots[int(iteration)],
            signed_proto=signed_proto,
            signed_final=signed_snapshots[int(iteration)],
        )
        all_rows.extend(rows)

    all_summary_path = base_output_dir / "all_iterations_summary.csv"
    write_summary_csv(all_summary_path, all_rows)
    print("[positivity-ablation] done")
    print(f"[positivity-ablation] all summary saved = {all_summary_path}")


if __name__ == "__main__":
    main()
