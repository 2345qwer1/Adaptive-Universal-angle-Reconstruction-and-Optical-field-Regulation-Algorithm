from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import numpy as np


@dataclass
class VAMConfig:
    target_image_path: str = "F:/USTC/项目/VAM/Target Image/Target-circle2.png"
    output_dir: str = "F:/USTC/项目/VAM/output"

    n_slice: int = 1080
    voxel_size_xy: float = 50e-6
    angle_num: int = 720
    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    angles_override: np.ndarray | None = None

    filter_type: str = "hanning"

    resin_alpha: float = 26.0
    resin_Ec: float = 15.0
    exposure_time_per_angle_ms: float = 100.0
    weight_power: float = 1.0
    min_weight: float = 0.1

    num_iterations: int = 50
    print_interval: int = 1
    optimization_mode: str = "greedy_stop"         # 优化模式："greedy_stop"（原先贪心提前停止）/"explore_best"（固定迭代到上限并记录 best）

    loss_background_over_weight: float = 1.0
    loss_background_over_safe_threshold: float | None = None
    loss_target_under_weight: float = 1.0
    loss_target_under_safe_threshold: float | None = None

    projection_quantization_enabled: bool = True
    projection_quantization_max_index: int = 255

    error_map_layout_mode: str = "compare_2x3"
    error_map_zoom_to_roi: bool = True
    error_map_roi_pad_px: int = 12
    error_map_min_half_width_px: int = 24

    use_numba: bool = True
    show_figures: bool = True
    save_compare_without_title: bool = True
    preprocess_flipud_for_display_consistency: bool = True

    @property
    def container_radius(self) -> float:
        return self.n_slice * self.voxel_size_xy / 2.0

    @property
    def angles(self) -> np.ndarray:
        if self.angles_override is not None:
            return np.asarray(self.angles_override, dtype=np.float64)
        return np.linspace(self.angle_start_deg, self.angle_end_deg, self.angle_num, endpoint=False)

    @property
    def projection_quantization_levels(self) -> int:
        return int(self.projection_quantization_max_index) + 1

    def to_metadata(self) -> Dict[str, Any]:
        meta = asdict(self)
        meta["container_radius"] = self.container_radius
        meta["angles"] = self.angles
        meta["projection_quantization_levels"] = self.projection_quantization_levels
        return meta


def get_default_config() -> VAMConfig:
    return VAMConfig()


def ensure_output_dirs(cfg: VAMConfig) -> Dict[str, Path]:
    root = Path(cfg.output_dir)
    subdirs = {
        "root": root,
        "projection": root / "projection_result",
        "compare": root / "compare_image",
        "figure": root / "figures",
        "dose": root / "dose_result",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
