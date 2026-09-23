from __future__ import annotations

"""独立的 a1/a2、多衰减率 resolution-w 批处理脚本（m 的单位固定为 px）。

本文件不依赖 run_resolution_w_study_2.0.py 或 resolution_w_analysis.py。
输出目录的组织仅参考前者：汇总 CSV/图、m_iterations 与 hit_w_structure。
"""

import csv
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import generate_initial_projection_matrix, get_default_config, optimize_projection_matrix
from functions.physics_printing import forward_dose_simulation, quantize_projection_matrix, solve_optimal_global_scale


# ============================== 用户参数区 ==============================
# 所有输出写入此绝对路径；每个衰减率、结构各占一个独立子目录。
OUTPUT_ROOT = Path(r"F:\USTC\VAM\makeW2_output")
STRUCTURES = ["a1", "a2"]
ATTENUATIONS = [0.0, 0.5]  # edge-to-center attenuation；0.5 表示边缘到中心衰减 50%。
M_VALUES_PX =[5,50,100]  # 待扫描结构宽度，单位始终为绝对 px。
R_OVER_R = {"a1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "a2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
CTF_THRESHOLD = 0.15
CORRECT_RATE_THRESHOLD = 0.7  # 当前 r/R 局部 Target+Background 的二值结构正确率下限。
N_SLICE = 1080
ANGLES = 720
EC = 15.0
# 投影矩阵优化参数：greedy_stop 可提前停止；explore_best 固定迭代到上限并返回历史最佳解。
OPT_ITERS = 40
OPTIMIZATION_MODE = "greedy_stop"  # 可选："greedy_stop"、"explore_best"（本研究中称为退火模式）。
OPT_PRINT_INTERVAL = 5  # 每隔多少次投影优化迭代打印一次进度；最后一次和提前停止始终会打印。
NUM_THETA = 2048
KEEP_SCANNING_AFTER_RESOLVED = True

# 输出开关。所有导出实现均在本文件内，删除 study 2.0 后不受影响。
SAVE_DETAIL_CSV = True
SAVE_SUMMARY_CSV = True
SAVE_W_CURVE_FIGURES = True
SAVE_CTF_DOSE_HEATMAP_FIGURES = True
SAVE_M_ITERATION_EVIDENCE = True
SAVE_HIT_W_EVIDENCE = True
SAVE_PROFILE_CSV = True
DRY_RUN = False

DETAIL_FIELDS = ["structure_type", "dose_type", "m", "target_size_px", "background_size_px", "pair_count", "r_over_R", "r_px", "CTF_dose_mean",
                 "CTF_dose_std", "all_over_Ec", "correct_rate", "resolved", "w_px"]
SUMMARY_FIELDS = ["structure_type", "r_over_R", "r_px", "w_proto_px", "w_optimized_px"]
STRUCTURE_TYPE = {"a1": "radial_concentric_ring", "a2": "angular_blocks_single_layer"}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(row.get(key), float) and math.isnan(row[key]) else row.get(key, "") for key in fields})


def config_for(attenuation: float):
    if not 0.0 <= attenuation < 1.0:
        raise ValueError("ATTENUATIONS 中每项必须位于 [0, 1)。")
    cfg = get_default_config()
    if int(OPT_ITERS) <= 0:
        raise ValueError("OPT_ITERS 必须为正整数。")
    if int(OPT_PRINT_INTERVAL) <= 0:
        raise ValueError("OPT_PRINT_INTERVAL 必须为正整数。")
    if OPTIMIZATION_MODE not in {"greedy_stop", "explore_best"}:
        raise ValueError("OPTIMIZATION_MODE 只能为 'greedy_stop' 或 'explore_best'。")
    cfg.n_slice, cfg.angle_num = N_SLICE, ANGLES
    cfg.resin_Ec, cfg.num_iterations, cfg.show_figures = EC, OPT_ITERS, False
    cfg.optimization_mode = OPTIMIZATION_MODE
    cfg.print_interval = OPT_PRINT_INTERVAL
    cfg.resin_alpha = 0.0 if attenuation == 0.0 else -math.log(1.0 - attenuation) / cfg.container_radius
    return cfg


def geometry(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y, x = np.indices((n, n), dtype=float)
    center = n // 2
    dx, dy = x - center, center - y
    return np.hypot(dx, dy), (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi), x, float(center)


def a1_layout(radius: float, target_size_px: int) -> tuple[int, float]:
    """返回 a1 的 Target 环数量和统一 Background 径向宽度。"""
    candidates = []
    for pair_count in range(2, int(math.floor(radius / target_size_px)) + 1):
        background_size = (radius - pair_count * target_size_px) / (pair_count - 1)
        if background_size > 0.0:
            candidates.append((pair_count, background_size))
    if not candidates:
        raise ValueError(f"a1 需要 0 < m < R/2 才能形成 Target/Background 交替结构；当前 m={target_size_px}, R={radius:g}。")
    return min(candidates, key=lambda item: (abs(item[1] - target_size_px), -item[0]))


def a1_target(n: int, radius: float, target_size_px: int, background_size_px: float) -> np.ndarray:
    """中心 Target 起始、最外 Target 环贴合容器边缘的 a1 结构。"""
    r, _, _, _ = geometry(n)
    period = float(target_size_px) + background_size_px
    radial_phase = np.mod(r, period)
    target = (radial_phase < float(target_size_px)) | np.isclose(r, radius)
    return (target & (r <= radius)).astype(np.uint8)


def a2_layout(reference_r_px: float, target_size_px: int) -> tuple[int, float]:
    """返回 a2 的完整圆周对数和统一 Background 弧长。"""
    if reference_r_px <= 0.0:
        raise ValueError("a2 的分析半径必须大于 0。")
    circumference = 2.0 * math.pi * reference_r_px
    candidates = []
    for pair_count in range(1, int(math.floor(circumference / target_size_px)) + 1):
        background_size = (circumference - pair_count * target_size_px) / pair_count
        if background_size > 0.0:
            candidates.append((pair_count, background_size))
    if not candidates:
        raise ValueError(f"a2 需要 m 小于当前圆周长；当前 m={target_size_px}, circumference={circumference:g}。")
    return min(candidates, key=lambda item: (abs(item[1] - target_size_px), -item[0]))


def a2_target(n: int, radius: float, target_size_px: int, reference_r_px: float, background_size_px: float) -> np.ndarray:
    """构造完整闭合、关于图像竖直中心线左右对称的 a2 结构。"""
    # a1 保持原有 geometry；a2 单独使用偶数图像的真实中心，保证 mask 像素级左右镜像。
    y, x = np.indices((n, n), dtype=float)
    center = (n - 1) / 2.0
    dx, dy = x - center, center - y
    r = np.hypot(dx, dy)
    theta = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    target_width = float(target_size_px) / reference_r_px
    period = (float(target_size_px) + background_size_px) / reference_r_px
    # 将 Target 条带中心对准图像竖直中心线（theta=pi/2）。
    # 对左右镜像 theta -> pi-theta，centered_phase 仅改变正负号，故 mask 严格左右对称。
    centered_phase = np.mod(theta - math.pi / 2.0 + period / 2.0, period) - period / 2.0
    return ((np.abs(centered_phase) < target_width / 2.0) & (r <= radius)).astype(np.uint8)


def dose_for_target(target: np.ndarray, cfg) -> np.ndarray:
    sino = radon(target, theta=cfg.angles, circle=True, preserve_range=True)
    low, high = sino.min(axis=0), sino.max(axis=0)
    normalized = np.divide(sino - low, high - low, out=np.zeros_like(sino), where=high > low)
    raw = generate_initial_projection_matrix(normalized, cfg)
    raw, _ = quantize_projection_matrix(raw, cfg, return_metadata=True)
    base_dose = forward_dose_simulation(raw, cfg)
    scale = solve_optimal_global_scale(base_dose, target, cfg.resin_Ec, cfg)
    raw, _ = quantize_projection_matrix(raw * scale, cfg, return_metadata=True)
    return optimize_projection_matrix(raw, target, cfg)["dose_field"]


def ctf(peak: float, valley: float) -> float:
    denom = peak + valley
    return float((peak - valley) / denom) if denom > 0.0 else float("nan")


def radial_ctf(dose: np.ndarray, radius: float, analyzed_r: float, target_size_px: int, background_size_px: float, pair_count: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """a1：四方向上最近 target 环的 peak/valley CTF。"""
    period = target_size_px + background_size_px
    band = min(range(pair_count), key=lambda i: abs(i * period + 0.5 * target_size_px - analyzed_r))
    inner, outer = band * period, band * period + target_size_px
    center = dose.shape[0] // 2
    values: list[float] = []
    targets: list[np.ndarray] = []
    backgrounds: list[np.ndarray] = []
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        distances = np.arange(0, int(math.floor(radius)) + 1, dtype=float)
        rows = np.clip(center + dr * distances.astype(int), 0, dose.shape[0] - 1)
        cols = np.clip(center + dc * distances.astype(int), 0, dose.shape[1] - 1)
        profile = dose[rows, cols]
        peak_mask = (distances >= inner) & (distances < outer)
        valley_mask = ((distances >= max(0.0, inner - background_size_px)) & (distances < inner)) | ((distances >= outer) & (distances < min(radius, outer + background_size_px)))
        if peak_mask.any() and valley_mask.any():
            values.append(ctf(float(profile[peak_mask].max()), float(profile[valley_mask].min())))
            targets.append(profile[peak_mask])
            backgrounds.append(profile[valley_mask])
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return (float(finite.mean()) if finite.size else float("nan"), float(finite.std()) if finite.size else float("nan"),
            np.concatenate(targets) if targets else np.empty(0), np.concatenate(backgrounds) if backgrounds else np.empty(0))


def angular_ctf(dose: np.ndarray, target_mask: np.ndarray, analyzed_r: float, target_size_px: int, background_size_px: float, pair_count: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """a2：按真实中心和相位，对每个 Target/Background 周期计算 peak/valley CTF。"""
    circumference = 2.0 * np.pi * analyzed_r
    sample_count = max(NUM_THETA, int(math.ceil(circumference / 0.25)))
    theta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    center = (dose.shape[0] - 1.0) / 2.0
    rows_float = center - analyzed_r * np.sin(theta)
    cols_float = center + analyzed_r * np.cos(theta)
    row0 = np.floor(rows_float).astype(int)
    col0 = np.floor(cols_float).astype(int)
    row1 = np.minimum(row0 + 1, dose.shape[0] - 1)
    col1 = np.minimum(col0 + 1, dose.shape[1] - 1)
    dr, dc = rows_float - row0, cols_float - col0
    profile = ((1.0 - dr) * ((1.0 - dc) * dose[row0, col0] + dc * dose[row0, col1])
               + dr * ((1.0 - dc) * dose[row1, col0] + dc * dose[row1, col1]))

    width = float(target_size_px) / analyzed_r
    period = (float(target_size_px) + background_size_px) / analyzed_r
    centered_phase = np.mod(theta - np.pi / 2.0 + period / 2.0, period) - period / 2.0
    analytic_target = np.abs(centered_phase) < width / 2.0
    nearest_rows = np.clip(np.rint(rows_float).astype(int), 0, target_mask.shape[0] - 1)
    nearest_cols = np.clip(np.rint(cols_float).astype(int), 0, target_mask.shape[1] - 1)
    target = np.asarray(target_mask[nearest_rows, nearest_cols], dtype=bool)
    # 保留解析标签计算，明确实际 mask 与理论相位的对应关系。
    if float(np.mean(target == analytic_target)) < 0.5:
        return float("nan"), float("nan"), np.empty(0), np.empty(0)
    transitions = np.flatnonzero(target != np.roll(target, 1))
    if transitions.size == 0:
        return float("nan"), float("nan"), np.empty(0), np.empty(0)
    start = int(transitions[0])
    rotated_target = np.roll(target, -start)
    cuts = np.r_[0, np.flatnonzero(rotated_target[1:] != rotated_target[:-1]) + 1, sample_count]
    runs = [(bool(rotated_target[cuts[i]]), (np.arange(cuts[i], cuts[i + 1]) + start) % sample_count)
            for i in range(len(cuts) - 1)]

    values: list[float] = []
    target_indices: list[np.ndarray] = []
    background_indices: list[np.ndarray] = []
    for run_index, (is_target, p_indices) in enumerate(runs):
        if not is_target:
            continue
        next_is_target, v_indices = runs[(run_index + 1) % len(runs)]
        if next_is_target or not p_indices.size or not v_indices.size:
            continue
        values.append(ctf(float(profile[p_indices].max()), float(profile[v_indices].min())))
        target_indices.append(p_indices)
        background_indices.append(v_indices)
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size != pair_count:
        return float("nan"), float("nan"), np.empty(0), np.empty(0)
    target_values = profile[np.concatenate(target_indices)] if target_indices else np.empty(0)
    background_values = profile[np.concatenate(background_indices)] if background_indices else np.empty(0)
    return float(finite.mean()), float(finite.std()), target_values, background_values


def local_binary_structure_metrics(target_values: np.ndarray, background_values: np.ndarray, ec: float) -> tuple[float, float]:
    """返回局部二值结构错误率 all_over_Ec 及正确率。"""
    total = int(target_values.size + background_values.size)
    if total == 0:
        return float("nan"), float("nan")
    correct = int(np.count_nonzero(target_values >= ec) + np.count_nonzero(background_values < ec)) / total
    return float(1.0 - correct), float(correct)


def save_dose_figure(dose: np.ndarray, ec: float, path: Path, title: str, save_svg: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0)); im = ax.imshow(dose, cmap="plasma", vmin=0.0, vmax=max(float(np.nanmax(dose)), ec))
    ax.set_title(f"Dose | Ec={ec:g}\n{title}"); ax.set_axis_off(); fig.colorbar(im, ax=ax, label="Dose")
    fig.tight_layout(); fig.savefig(path, dpi=180)
    if save_svg: fig.savefig(path.with_suffix(".svg"), format="svg")
    plt.close(fig)


def save_target_cured_error(target: np.ndarray, dose: np.ndarray, ec: float, path: Path, title: str, save_svg: bool = False) -> None:
    cured = dose >= ec; under, over = (target.astype(bool) & ~cured), (cured & ~target.astype(bool))
    under_count, over_count = int(np.count_nonzero(under)), int(np.count_nonzero(over))
    error = np.zeros(target.shape + (3,), dtype=float); error[under] = (1, 0, 0); error[over] = (0, 1, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    labels = ("Target", "Cured", f"Error: red=under, green=over\\n欠固化: {under_count} px | 过固化: {over_count} px")
    for ax, image, label in zip(axes, (target, cured, error), labels):
        ax.imshow(image, cmap=None if image.ndim == 3 else "gray"); ax.set_title(label); ax.set_axis_off()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=180)
    if save_svg: fig.savefig(path.with_suffix(".svg"), format="svg")
    plt.close(fig)


def save_profile(dose: np.ndarray, structure: str, analyzed_r: float, path_base: Path, title: str, save_svg: bool = False) -> None:
    center = dose.shape[0] // 2
    if structure == "a1":
        # 偶数网格的中心右侧最后一个有效列是 n-1，故不能多取一个不存在的 +R 点。
        max_distance = min(center, dose.shape[1] - 1 - center)
        x = np.arange(0, max_distance + 1, dtype=float)
        y = dose[center, center:center + len(x)]
        x_label = "distance / px"
    else:
        count = max(64, int(round(2.0 * np.pi * analyzed_r))); theta = np.linspace(0, 2.0 * np.pi, count, endpoint=False)
        rows = np.clip(np.rint(center - analyzed_r * np.sin(theta)).astype(int), 0, dose.shape[0] - 1)
        cols = np.clip(np.rint(center + analyzed_r * np.cos(theta)).astype(int), 0, dose.shape[1] - 1)
        x, y, x_label = theta * analyzed_r, dose[rows, cols], "arc length / px"
    if SAVE_PROFILE_CSV:
        write_csv(path_base.with_suffix(".csv"), ["sample_index", "position_px", "dose"], [{"sample_index": i, "position_px": float(a), "dose": float(b)} for i, (a, b) in enumerate(zip(x, y))])
    fig, ax = plt.subplots(figsize=(8.0, 4.8)); ax.plot(x, y, linewidth=.8); ax.set(xlabel=x_label, ylabel="dose", title=title); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path_base.with_suffix(".png"), dpi=180)
    if save_svg: fig.savefig(path_base.with_suffix(".svg"), format="svg")
    plt.close(fig)


def save_evidence(folder: Path, target: np.ndarray, dose: np.ndarray, cfg, meta: dict[str, Any], structure: str, analyzed_r: float, save_svg: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "optimized"
    title = f"{meta['structure_type']} | optimized | m={meta['m']} | r/R={meta['r_over_R']:.3f}"
    save_dose_figure(dose, cfg.resin_Ec, folder / f"{prefix}_dose.png", title, save_svg=save_svg)
    save_target_cured_error(target, dose, cfg.resin_Ec, folder / f"{prefix}_target_cured_error.png", title, save_svg=save_svg)
    save_profile(dose, structure, analyzed_r, folder / f"{prefix}_dose_distribution", title, save_svg=save_svg)
    np.savez_compressed(folder / f"{prefix}_data.npz", target_mask=target, dose_field=dose, cured_mask=dose >= cfg.resin_Ec)
    meta = dict(meta); meta.update({"resin_Ec": float(cfg.resin_Ec), "dose_figure": f"{prefix}_dose.png", "structure_figure": f"{prefix}_target_cured_error.png", "profile_figure": f"{prefix}_dose_distribution.png", "profile_csv": f"{prefix}_dose_distribution.csv"})
    (folder / f"{prefix}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_summary_figures(out: Path, alias: str, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    if SAVE_W_CURVE_FIGURES:
        x = [r["r_over_R"] for r in summary]; y = [r["w_optimized_px"] for r in summary]
        fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(x, y, marker="o"); ax.set(xlabel="r/R", ylabel="w / px", title=f"{STRUCTURE_TYPE[alias]} optimized resolution w vs r/R"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / f"fig_{STRUCTURE_TYPE[alias]}_w_vs_r_optimized.png", dpi=200); plt.close(fig)
    if SAVE_CTF_DOSE_HEATMAP_FIGURES:
        ms, radii = list(M_VALUES_PX), R_OVER_R[alias]; matrix = np.full((len(radii), len(ms)), np.nan)
        for row in rows: matrix[radii.index(row["r_over_R"]), ms.index(row["m"])] = row["CTF_dose_mean"]
        fig, ax = plt.subplots(figsize=(7, 4.8)); im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set(xticks=range(len(ms)), xticklabels=ms, yticks=range(len(radii)), yticklabels=radii, xlabel="m / px", ylabel="r/R", title=f"{STRUCTURE_TYPE[alias]} optimized CTF_dose_mean")
        fig.colorbar(im, ax=ax, label="CTF_dose_mean"); fig.tight_layout(); fig.savefig(out / f"fig_{STRUCTURE_TYPE[alias]}_CTF_dose_heatmap_optimized.png", dpi=200); plt.close(fig)


def run_one(alias: str, attenuation: float) -> dict[str, Any]:
    cfg = config_for(attenuation); radius, out = cfg.n_slice / 2.0, OUTPUT_ROOT / f"attenuation_{attenuation:.2f}".replace(".", "p") / alias
    out.mkdir(parents=True, exist_ok=True); rows: list[dict[str, Any]] = []; found = {q: float("nan") for q in R_OVER_R[alias]}
    for m in M_VALUES_PX:
        if alias == "a1":
            # a1 的一个 m 对应唯一同心环 target；同一 dose 场供所有分析半径复用。
            pair_count, background_size = a1_layout(radius, m)
            target = a1_target(cfg.n_slice, radius, m, background_size)
            shared_dose = dose_for_target(target, cfg)
            jobs = [(q, target, shared_dose, pair_count, background_size) for q in R_OVER_R[alias]]
        else:
            # a2 的每个 r/R 都有独立的角宽和 target，不能共享 dose 场。
            jobs = []
            for q in R_OVER_R[alias]:
                pair_count, background_size = a2_layout(q * radius, m)
                target = a2_target(cfg.n_slice, radius, m, q * radius, background_size)
                jobs.append((q, target, dose_for_target(target, cfg), pair_count, background_size))
        for q, target, dose, pair_count, background_size in jobs:
            analyzed_r = q * radius
            mean, std, target_values, background_values = radial_ctf(dose, radius, analyzed_r, m, background_size, pair_count) if alias == "a1" else angular_ctf(dose, target, analyzed_r, m, background_size, pair_count)
            all_over_ec, correct_rate = local_binary_structure_metrics(target_values, background_values, cfg.resin_Ec)
            resolved = bool(np.isfinite(mean) and mean >= CTF_THRESHOLD and np.isfinite(correct_rate) and correct_rate >= CORRECT_RATE_THRESHOLD)
            if resolved and math.isnan(found[q]): found[q] = float(m)
            row = {"structure_type": STRUCTURE_TYPE[alias], "dose_type": "optimized", "m": int(m), "target_size_px": int(m), "background_size_px": background_size, "pair_count": pair_count, "r_over_R": float(q), "r_px": float(analyzed_r), "CTF_dose_mean": mean, "CTF_dose_std": std, "all_over_Ec": all_over_ec, "correct_rate": correct_rate, "resolved": resolved, "w_px": found[q]}
            rows.append(row)
            meta = {"output_kind": "m_iteration", "structure_type": STRUCTURE_TYPE[alias], "dose_type": "optimized", "m": int(m), "target_size_px": int(m), "background_size_px": background_size, "pair_count": pair_count, "r_over_R": float(q), "r_px": float(analyzed_r), "container_r_px": radius, "attenuation": attenuation, "resin_alpha": float(cfg.resin_alpha), "optimization_mode": cfg.optimization_mode, "max_optimization_iterations": cfg.num_iterations, "optimization_print_interval": cfg.print_interval, "CTF_dose_mean": mean, "CTF_dose_std": std, "all_over_Ec": all_over_ec, "correct_rate": correct_rate, "resolved": resolved, "w_px": found[q]}
            if SAVE_M_ITERATION_EVIDENCE: save_evidence(out / "m_iterations" / f"m_{m:03d}" / f"rR_{q:.3f}", target, dose, cfg, meta, alias, analyzed_r, save_svg=True)
            if SAVE_HIT_W_EVIDENCE and resolved and found[q] == float(m):
                meta["output_kind"] = "resolved_w"; save_evidence(out / "hit_w_structure" / f"rR_{q:.3f}", target, dose, cfg, meta, alias, analyzed_r)
    summary = [{"structure_type": STRUCTURE_TYPE[alias], "r_over_R": q, "r_px": q * radius, "w_proto_px": float("nan"), "w_optimized_px": found[q]} for q in R_OVER_R[alias]]
    if SAVE_DETAIL_CSV: write_csv(out / f"{STRUCTURE_TYPE[alias]}_resolution_detail.csv", DETAIL_FIELDS, rows)
    if SAVE_SUMMARY_CSV: write_csv(out / "resolution_w_summary.csv", SUMMARY_FIELDS, summary)
    save_summary_figures(out, alias, rows, summary)
    return {"output_dir": str(out), "structure_type": STRUCTURE_TYPE[alias], "m_count": len(M_VALUES_PX), "r_over_R_list": json.dumps(R_OVER_R[alias]), "CTF_threshold": CTF_THRESHOLD, "correct_rate_threshold": CORRECT_RATE_THRESHOLD, "optimization_mode": cfg.optimization_mode, "max_optimization_iterations": cfg.num_iterations, "optimization_print_interval": cfg.print_interval, "resin_alpha": cfg.resin_alpha, "optimized_found": sum(not math.isnan(v) for v in found.values()), "total_positions": len(found)}


def main() -> None:
    manifest: list[dict[str, Any]] = []
    for attenuation in ATTENUATIONS:
        for alias in STRUCTURES:
            row: dict[str, Any] = {"attenuation": attenuation, "structure": alias, "m_unit": "px", "m_values_px": json.dumps(list(M_VALUES_PX)), "started_at": datetime.now().isoformat(), "status": "running"}
            try:
                if DRY_RUN: row["status"] = "dry_run"
                else: row.update(run_one(alias, attenuation)); row["status"] = "success"
            except Exception:
                row["status"], row["error_message"] = "failed", traceback.format_exc(); print(row["error_message"])
            row["finished_at"] = datetime.now().isoformat(); manifest.append(row)
    fields = sorted({key for row in manifest for key in row})
    write_csv(OUTPUT_ROOT / "batch_run_manifest.csv", fields, manifest)


if __name__ == "__main__":
    main()
