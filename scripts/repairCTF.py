from __future__ import annotations

"""从已保存的 a2 optimized_data.npz 独立重算 CTF 与 resolution-detail CSV。

本脚本不会重新运行投影优化，也不会修改任何 NPZ、JSON 或原始 CSV。
`m_iterations` 在这里表示不同 m 的扫描结果，不表示优化器内部的逐次迭代。
"""

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


# ============================== 用户参数区 ==============================
# 另一台电脑上的数据根目录。脚本会从四个 attenuation 子目录读取 NPZ。
INPUT_ROOT = Path(r"D:\zhngzhu\20260902_optimized_only")

# 在每个 attenuation_xxx\a2 目录中新建此 CSV；不会覆盖原始 detail CSV。
OUTPUT_FILENAME = "angular_blocks_single_layer_resolution_detail_repaired.csv"

# 若输出文件已经存在，False 会停止并保护旧结果；确认需要重算时可改为 True。
OVERWRITE_EXISTING_OUTPUT = False

ATTENUATION_FOLDERS = (
    "attenuation_0p00",
    "attenuation_0p25",
    "attenuation_0p50",
    "attenuation_0p75",
)
M_VALUES_PX = range(1, 109)
R_OVER_R = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# 与本批 makeW2.0 设置保持一致。
RESIN_EC = 15.0
CTF_THRESHOLD = 0.2
CORRECT_RATE_THRESHOLD = 0.85

# 每个圆周至少取 4096 点；同时保证圆周采样间距不大于 0.25 px。
MIN_ANGULAR_SAMPLES = 4096
MAX_ARC_STEP_PX = 0.25

DETAIL_FIELDS = [
    "structure_type",
    "dose_type",
    "m",
    "target_size_px",
    "background_size_px",
    "pair_count",
    "r_over_R",
    "r_px",
    "CTF_dose_mean",
    "CTF_dose_std",
    "all_over_Ec",
    "correct_rate",
    "resolved",
    "w_px",
]


def a2_layout(reference_r_px: float, target_size_px: int) -> tuple[int, float]:
    """复现 makeW2.0 的 a2 完整圆周布局。"""
    if reference_r_px <= 0.0:
        raise ValueError("分析半径必须大于 0。")
    circumference = 2.0 * math.pi * reference_r_px
    candidates: list[tuple[int, float]] = []
    for pair_count in range(1, int(math.floor(circumference / target_size_px)) + 1):
        background_size_px = (
            circumference - pair_count * target_size_px
        ) / pair_count
        if background_size_px > 0.0:
            candidates.append((pair_count, background_size_px))
    if not candidates:
        raise ValueError(
            f"m={target_size_px} px 无法在 r={reference_r_px:g} px 形成完整 a2 周期。"
        )
    return min(candidates, key=lambda item: (abs(item[1] - target_size_px), -item[0]))


def bilinear_circle_profile(dose: np.ndarray, radius_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在偶数网格真实中心的圆周上双线性采样剂量。"""
    circumference = 2.0 * math.pi * radius_px
    sample_count = max(
        MIN_ANGULAR_SAMPLES,
        int(math.ceil(circumference / MAX_ARC_STEP_PX)),
    )
    theta = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    center_row = (dose.shape[0] - 1.0) / 2.0
    center_col = (dose.shape[1] - 1.0) / 2.0
    rows = center_row - radius_px * np.sin(theta)
    cols = center_col + radius_px * np.cos(theta)

    if rows.min() < 0.0 or cols.min() < 0.0:
        raise ValueError("采样圆周超出剂量矩阵的上边界或左边界。")
    if rows.max() > dose.shape[0] - 1 or cols.max() > dose.shape[1] - 1:
        raise ValueError("采样圆周超出剂量矩阵的下边界或右边界。")

    row0 = np.floor(rows).astype(np.int64)
    col0 = np.floor(cols).astype(np.int64)
    row1 = np.minimum(row0 + 1, dose.shape[0] - 1)
    col1 = np.minimum(col0 + 1, dose.shape[1] - 1)
    dr = rows - row0
    dc = cols - col0

    profile = (
        (1.0 - dr)
        * ((1.0 - dc) * dose[row0, col0] + dc * dose[row0, col1])
        + dr * ((1.0 - dc) * dose[row1, col0] + dc * dose[row1, col1])
    )
    return profile, rows, cols


def circular_runs(labels: np.ndarray) -> list[tuple[bool, np.ndarray]]:
    """把闭合圆周二值标签拆分为按顺序排列的连续区段。"""
    transitions = np.flatnonzero(labels != np.roll(labels, 1))
    if transitions.size == 0:
        return []
    start = int(transitions[0])
    rotated = np.roll(labels, -start)
    cuts = np.r_[
        0,
        np.flatnonzero(rotated[1:] != rotated[:-1]) + 1,
        labels.size,
    ]
    return [
        (
            bool(rotated[cuts[index]]),
            (np.arange(cuts[index], cuts[index + 1]) + start) % labels.size,
        )
        for index in range(len(cuts) - 1)
    ]


def calculate_a2_metrics(
    target_mask: np.ndarray,
    dose_field: np.ndarray,
    analyzed_r_px: float,
    expected_pair_count: int,
    resin_ec: float,
) -> tuple[float, float, float, float, int]:
    """由真实 mask 与最终剂量计算 peak-valley CTF 和局部固化正确率。"""
    if target_mask.ndim != 2 or dose_field.ndim != 2:
        raise ValueError("target_mask 和 dose_field 必须都是二维数组。")
    if target_mask.shape != dose_field.shape:
        raise ValueError(
            f"target_mask{target_mask.shape} 与 dose_field{dose_field.shape} 形状不同。"
        )
    if target_mask.shape[0] != target_mask.shape[1]:
        raise ValueError(f"当前仅支持方形数据，实际形状为 {target_mask.shape}。")
    if not np.all(np.isfinite(dose_field)):
        raise ValueError("dose_field 含有 NaN 或无穷值。")
    unique_mask_values = np.unique(target_mask)
    if not np.all(np.isin(unique_mask_values, (0, 1, False, True))):
        raise ValueError(f"target_mask 不是二值数组：{unique_mask_values!r}")

    dose = np.asarray(dose_field, dtype=np.float64)
    profile, rows, cols = bilinear_circle_profile(dose, analyzed_r_px)
    nearest_rows = np.rint(rows).astype(np.int64)
    nearest_cols = np.rint(cols).astype(np.int64)
    target_labels = np.asarray(
        target_mask[nearest_rows, nearest_cols], dtype=bool
    )
    runs = circular_runs(target_labels)
    if not runs:
        raise ValueError("指定圆周上未检测到 Target/Background 转换。")

    ctf_values: list[float] = []
    target_indices: list[np.ndarray] = []
    background_indices: list[np.ndarray] = []
    for run_index, (is_target, target_run) in enumerate(runs):
        if not is_target:
            continue
        next_is_target, background_run = runs[(run_index + 1) % len(runs)]
        if next_is_target or target_run.size == 0 or background_run.size == 0:
            raise ValueError("检测到无法配对的 Target/Background 圆周区段。")
        target_peak = float(np.max(profile[target_run]))
        background_valley = float(np.min(profile[background_run]))
        denominator = target_peak + background_valley
        if denominator <= 0.0:
            raise ValueError("CTF 分母不大于 0。")
        ctf_values.append((target_peak - background_valley) / denominator)
        target_indices.append(target_run)
        background_indices.append(background_run)

    observed_pair_count = len(ctf_values)
    if observed_pair_count != expected_pair_count:
        raise ValueError(
            f"周期数不一致：expected={expected_pair_count}, "
            f"observed={observed_pair_count}。"
        )

    target_values = profile[np.concatenate(target_indices)]
    background_values = profile[np.concatenate(background_indices)]
    total = int(target_values.size + background_values.size)
    if total == 0:
        raise ValueError("没有可用于局部正确率计算的采样点。")
    correct_count = int(np.count_nonzero(target_values >= resin_ec))
    correct_count += int(np.count_nonzero(background_values < resin_ec))
    correct_rate = correct_count / total

    finite_ctf = np.asarray(ctf_values, dtype=np.float64)
    return (
        float(np.mean(finite_ctf)),
        float(np.std(finite_ctf)),
        float(1.0 - correct_rate),
        float(correct_rate),
        observed_pair_count,
    )


def load_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """只读取重算所需的两个真实数组。"""
    with np.load(npz_path, allow_pickle=False) as data:
        missing = {"target_mask", "dose_field"}.difference(data.files)
        if missing:
            raise KeyError(f"NPZ 缺少字段：{sorted(missing)}")
        return data["target_mask"].copy(), data["dose_field"].copy()


def blank_if_nan(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return value


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and not OVERWRITE_EXISTING_OUTPUT:
        raise FileExistsError(
            f"输出已存在：{path}\n"
            "如确认需要覆盖，请把 OVERWRITE_EXISTING_OUTPUT 改为 True。"
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: blank_if_nan(row.get(field, "")) for field in DETAIL_FIELDS}
            )


def repair_one_attenuation(attenuation_folder: str) -> tuple[Path, int, list[str]]:
    a2_root = INPUT_ROOT / attenuation_folder / "a2"
    iterations_root = a2_root / "m_iterations"
    if not iterations_root.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{iterations_root}")

    output_path = a2_root / OUTPUT_FILENAME
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    found_w = {r_over_r: float("nan") for r_over_r in R_OVER_R}
    earlier_missing = {r_over_r: False for r_over_r in R_OVER_R}

    for m_px in M_VALUES_PX:
        for r_over_r in R_OVER_R:
            npz_path = (
                iterations_root
                / f"m_{m_px:03d}"
                / f"rR_{r_over_r:.3f}"
                / "optimized_data.npz"
            )
            ctf_mean = ctf_std = all_over_ec = correct_rate = float("nan")
            resolved = False
            # 默认 1080×1080 数据的 R=540；成功读取后由实际矩阵尺寸更新。
            container_r_px = 540.0
            analyzed_r_px = r_over_r * container_r_px
            pair_count, background_size_px = a2_layout(analyzed_r_px, m_px)

            try:
                if not npz_path.is_file():
                    raise FileNotFoundError(f"缺少 {npz_path}")
                target_mask, dose_field = load_npz(npz_path)
                container_r_px = target_mask.shape[0] / 2.0
                analyzed_r_px = r_over_r * container_r_px
                pair_count, background_size_px = a2_layout(analyzed_r_px, m_px)
                (
                    ctf_mean,
                    ctf_std,
                    all_over_ec,
                    correct_rate,
                    _observed_pair_count,
                ) = calculate_a2_metrics(
                    target_mask,
                    dose_field,
                    analyzed_r_px,
                    pair_count,
                    RESIN_EC,
                )
                resolved = bool(
                    ctf_mean >= CTF_THRESHOLD
                    and correct_rate >= CORRECT_RATE_THRESHOLD
                )
            except Exception as exc:
                earlier_missing[r_over_r] = True
                errors.append(
                    f"[{attenuation_folder}] m={m_px}, r/R={r_over_r:.3f}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if (
                resolved
                and math.isnan(found_w[r_over_r])
                and not earlier_missing[r_over_r]
            ):
                found_w[r_over_r] = float(m_px)

            rows.append(
                {
                    "structure_type": "angular_blocks_single_layer",
                    "dose_type": "optimized",
                    "m": int(m_px),
                    "target_size_px": int(m_px),
                    "background_size_px": float(background_size_px),
                    "pair_count": int(pair_count),
                    "r_over_R": float(r_over_r),
                    "r_px": float(analyzed_r_px),
                    "CTF_dose_mean": ctf_mean,
                    "CTF_dose_std": ctf_std,
                    "all_over_Ec": all_over_ec,
                    "correct_rate": correct_rate,
                    "resolved": resolved,
                    "w_px": found_w[r_over_r],
                }
            )

    write_detail_csv(output_path, rows)
    return output_path, len(rows), errors


def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"输入根目录不存在：{INPUT_ROOT}\n"
            "请确认另一台电脑上的绝对路径，或修改脚本顶部 INPUT_ROOT。"
        )

    all_errors: list[str] = []
    outputs: list[Path] = []
    for attenuation_folder in ATTENUATION_FOLDERS:
        output_path, row_count, errors = repair_one_attenuation(
            attenuation_folder
        )
        outputs.append(output_path)
        all_errors.extend(errors)
        print(
            f"完成 {attenuation_folder}: {row_count} 行，"
            f"异常单元 {len(errors)}，输出 {output_path}"
        )

    print(f"\n完成：共生成 {len(outputs)} 个 CSV。")
    if all_errors:
        print(f"共有 {len(all_errors)} 个单元未能计算，相关数值已留空：")
        for message in all_errors:
            print(message)
    else:
        print("全部单元均通过 NPZ、周期数和数值有效性检查。")


if __name__ == "__main__":
    main()
