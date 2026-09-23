from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 用户参数区
# =============================================================================
# 用途：
# 1. 读取 resolution_detail.csv 中不同 m、不同 r/R、不同 dose_type 对应的 CTF_dose_mean。
# 2. 对每个 dose_type + r/R，寻找“第一个满足 CTF_dose_mean >= 设定阈值”的 m。
# 3. 将这个首个命中的 m 视为该位置的 w。
# 4. 输出：
#    - 一份汇总 CSV
#    - 一张 w 随 r/R 变化的折线图
#
# 你平时只需要修改这一段，不必进入下面函数内部。


# -------------------------
# 输入 / 输出路径
# -------------------------
# 输入的明细 CSV。
# 该文件通常来自 run_resolution_w_study_2.0.py 的 detail 输出。
USER_CSV_PATH = Path(
    r"F:\USTC\项目\VAM\5.20-多层角向块\resolution_w_study_2_0\angular_blocks_multi_layer_resolution_detail.csv"
)

# 输出目录。
# 汇总 CSV 和折线图 PNG 都会保存到这里。
USER_OUTPUT_DIR = Path(__file__).resolve().parent / "try_CTF_dose_output"

# 输出文件名前缀。
# 实际文件名后面会自动拼接 CTF_dose 值，例如：
# - try_CTF_dose_summary_CTF_dose_0_17647058823529413.csv
# - try_CTF_dose_w_vs_r_CTF_dose_0_17647058823529413.png
USER_SUMMARY_BASENAME = "try_CTF_dose_summary"
USER_FIGURE_BASENAME = "try_CTF_dose_w_vs_r"


# -------------------------
# 核心判定参数
# -------------------------
# CTF_dose 阈值。
# 本脚本会把“第一个满足 CTF_dose_mean >= USER_CTF_DOSE_THRESHOLD 的 m”
# 视为该 dose_type、该 r/R 下命中的 w。
USER_CTF_DOSE_THRESHOLD = 0.17647058823529413

# 可选：只保留部分 dose_type。
# None 表示保留 CSV 里的全部 dose_type。
# 例如只看优化后结果：
# USER_DOSE_TYPES = ["optimized"]
# 例如同时看 proto 和 optimized：
# USER_DOSE_TYPES = ["proto", "optimized"]
USER_DOSE_TYPES: list[str] | None = None


# -------------------------
# 输出开关
# -------------------------
# 是否保存汇总 CSV。
USER_SAVE_SUMMARY_CSV = True

# 是否保存折线图 PNG。
USER_SAVE_FIGURE_PNG = True

# 是否在终端打印每个 dose_type、每个 r/R 的命中结果。
USER_PRINT_CONSOLE_ROWS = True


# -------------------------
# 图像样式参数
# -------------------------
# 折线图宽度，单位英寸。
USER_FIGURE_WIDTH_INCH = 8.0

# 折线图高度，单位英寸。
USER_FIGURE_HEIGHT_INCH = 5.0

# 输出图像分辨率。
USER_FIGURE_DPI = 150

# 折线宽度。
USER_LINE_WIDTH = 1.8

# 折线点标记样式。
USER_MARKER = "o"

# 网格透明度。
USER_GRID_ALPHA = 0.3


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    设计原则：
    - 默认使用文件开头“用户参数区”的设置。
    - 如果命令行传入参数，则命令行优先，用于临时覆盖。

    例如：
        python try_k.py --ctf-dose-threshold 0.17647058823529413
        python try_k.py --dose-types optimized
    """
    parser = argparse.ArgumentParser(
        description="给定 CTF_dose 阈值，寻找各个 r/R 位置首次命中的 w，并绘制 w-r/R 折线图。"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=USER_CSV_PATH,
        help="输入的 resolution detail CSV 路径。",
    )
    parser.add_argument(
        "--ctf-dose-threshold",
        type=float,
        default=USER_CTF_DOSE_THRESHOLD,
        help="CTF_dose 判定阈值。第一个满足 CTF_dose_mean >= threshold 的 m 被视为 w。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=USER_OUTPUT_DIR,
        help="输出目录，用于保存汇总 CSV 和折线图。",
    )
    parser.add_argument(
        "--dose-types",
        nargs="*",
        default=USER_DOSE_TYPES,
        help="可选的 dose_type 子集，例如：--dose-types proto optimized",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    """
    将 CSV 字段安全转换为 float。

    对于空字符串，返回 NaN。
    这样可以兼容 detail CSV 中可能为空的 w_px 字段。
    """
    text = (value or "").strip()
    if not text:
        return float("nan")
    return float(text)


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    """
    读取明细 CSV，并将常用字段转成便于后续处理的数值类型。

    输入 CSV 预期包含这些列：
    - structure_type
    - dose_type
    - m
    - r_over_R
    - r_px
    - CTF_dose_mean
    - CTF_dose_std
    - resolved
    - w_px
    """
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "structure_type": row["structure_type"],
                    "dose_type": row["dose_type"],
                    "m": int(row["m"]),
                    "r_over_R": float(row["r_over_R"]),
                    "r_px": float(row["r_px"]),
                    "CTF_dose_mean": float(row["CTF_dose_mean"]),
                    "CTF_dose_std": float(row["CTF_dose_std"]),
                    "resolved": str(row["resolved"]).strip().lower() == "true",
                    "w_px": _to_float(row.get("w_px", "")),
                }
            )

    if not rows:
        raise ValueError(f"CSV 为空：{csv_path}")
    return rows


def filter_rows(rows: list[dict[str, object]], dose_types: list[str] | None) -> list[dict[str, object]]:
    """
    按 dose_type 过滤数据。

    - 当 dose_types 为 None 时，不过滤，保留全部。
    - 当 dose_types 为列表时，只保留指定的 dose_type。
    """
    if not dose_types:
        return rows

    keep = set(dose_types)
    filtered = [row for row in rows if str(row["dose_type"]) in keep]
    if not filtered:
        raise ValueError(f"没有匹配到指定的 dose_type：{sorted(keep)}")
    return filtered


def find_first_hit_w(rows: list[dict[str, object]], ctf_dose_threshold: float) -> list[dict[str, object]]:
    """
    对每个 dose_type + r/R，寻找首个命中的 m，并将其视为 w。

    判定逻辑：
    1. 按 (dose_type, r_over_R) 分组。
    2. 在每组内部按 m 从小到大排序。
    3. 找到第一个满足 CTF_dose_mean >= CTF_dose_threshold 的条目。
    4. 将该条目的 m 记为该位置的 w。

    若某组始终没有命中，则：
    - w_px 记为 NaN
    - hit_m 留空
    - hit_CTF_dose_mean / hit_CTF_dose_std 记为 NaN
    """
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dose_type"]), float(row["r_over_R"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for (dose_type, r_over_R), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        items_sorted = sorted(items, key=lambda item: int(item["m"]))
        hit_row = next(
            (
                item
                for item in items_sorted
                if float(item["CTF_dose_mean"]) >= ctf_dose_threshold
            ),
            None,
        )
        summary_rows.append(
            {
                "dose_type": dose_type,
                "r_over_R": r_over_R,
                "w_px": float(hit_row["m"]) if hit_row is not None else float("nan"),
                "hit_m": int(hit_row["m"]) if hit_row is not None else "",
                "hit_CTF_dose_mean": float(hit_row["CTF_dose_mean"]) if hit_row is not None else float("nan"),
                "hit_CTF_dose_std": float(hit_row["CTF_dose_std"]) if hit_row is not None else float("nan"),
                "r_px": float(items_sorted[0]["r_px"]),
            }
        )
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    """
    将首命中结果写成汇总 CSV。

    输出列含义：
    - dose_type: 剂量类型，例如 proto / optimized
    - r_over_R: 分析半径相对容器半径的比例
    - r_px: 对应像素半径
    - w_px: 首次命中的 w，等于首次命中的 m
    - hit_m: 与 w_px 相同，单独保留便于阅读
    - hit_CTF_dose_mean: 命中时的 CTF_dose_mean
    - hit_CTF_dose_std: 命中时的 CTF_dose_std
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dose_type",
                "r_over_R",
                "r_px",
                "w_px",
                "hit_m",
                "hit_CTF_dose_mean",
                "hit_CTF_dose_std",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_w_vs_r(summary_rows: list[dict[str, object]], ctf_dose_threshold: float, output_path: Path) -> None:
    """
    绘制 w-r/R 折线图。

    图的含义：
    - 横轴：r/R
    - 纵轴：w / px
    - 每一条线：一个 dose_type
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(USER_FIGURE_WIDTH_INCH, USER_FIGURE_HEIGHT_INCH),
        dpi=USER_FIGURE_DPI,
    )

    dose_types = sorted({str(row["dose_type"]) for row in summary_rows})
    for dose_type in dose_types:
        rows = [row for row in summary_rows if str(row["dose_type"]) == dose_type]
        rows.sort(key=lambda item: float(item["r_over_R"]))
        x = np.array([float(row["r_over_R"]) for row in rows], dtype=np.float64)
        y = np.array([float(row["w_px"]) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker=USER_MARKER, linewidth=USER_LINE_WIDTH, label=dose_type)

    ax.set_xlabel("r/R")
    ax.set_ylabel("w / px")
    ax.set_title(f"First-hit w vs r/R | CTF_dose_threshold = {ctf_dose_threshold:.3f}")
    ax.grid(True, alpha=USER_GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    主流程：
    1. 读取命令行参数或使用文件顶部默认参数
    2. 读取 CSV
    3. 过滤 dose_type
    4. 计算各组首命中的 w
    5. 输出汇总 CSV 和折线图
    6. 在终端打印结果摘要
    """
    args = parse_args()
    rows = load_rows(args.csv)
    rows = filter_rows(rows, args.dose_types)
    summary_rows = find_first_hit_w(rows, args.ctf_dose_threshold)

    # 将小数点替换为下划线，便于输出文件名统一管理。
    safe_threshold = str(args.ctf_dose_threshold).replace(".", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.output_dir / f"{USER_SUMMARY_BASENAME}_CTF_dose_{safe_threshold}.csv"
    figure_path = args.output_dir / f"{USER_FIGURE_BASENAME}_CTF_dose_{safe_threshold}.png"

    if USER_SAVE_SUMMARY_CSV:
        write_summary_csv(summary_rows, summary_csv_path)
    if USER_SAVE_FIGURE_PNG:
        plot_w_vs_r(summary_rows, args.ctf_dose_threshold, figure_path)

    print(f"CTF_dose_threshold = {args.ctf_dose_threshold}")
    print(f"input_csv = {args.csv}")
    print(f"output_dir = {args.output_dir}")
    if USER_SAVE_SUMMARY_CSV:
        print(f"summary_csv = {summary_csv_path}")
    if USER_SAVE_FIGURE_PNG:
        print(f"figure_png = {figure_path}")

    if USER_PRINT_CONSOLE_ROWS:
        for row in summary_rows:
            w_text = "NaN" if not np.isfinite(float(row["w_px"])) else f"{float(row['w_px']):.0f}"
            print(
                f"dose_type={row['dose_type']}, r/R={float(row['r_over_R']):.3f}, "
                f"w={w_text}, hit_CTF_dose={float(row['hit_CTF_dose_mean']):.4f}"
            )


if __name__ == "__main__":
    main()
