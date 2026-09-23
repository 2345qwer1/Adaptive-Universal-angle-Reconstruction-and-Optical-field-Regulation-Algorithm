from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from .config import VAMConfig


def save_projection_npz(
    save_path: str | Path,
    projection_matrix: np.ndarray,
    cfg: VAMConfig,
    target_mask: np.ndarray | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "projection_matrix": projection_matrix,
        "angles": cfg.angles,
        "N": cfg.n_slice,
        "voxel_size_xy": cfg.voxel_size_xy,
        "container_radius": cfg.container_radius,
        "resin_alpha": cfg.resin_alpha,
        "resin_Ec": cfg.resin_Ec,
        "exposure_time_per_angle_ms": cfg.exposure_time_per_angle_ms,
        "projection_quantization_enabled": np.array(cfg.projection_quantization_enabled),
        "projection_quantization_max_index": np.array(cfg.projection_quantization_max_index),
    }
    if target_mask is not None:
        payload["target_mask"] = target_mask.astype(np.float64)
    if extra_metadata:
        payload.update(extra_metadata)
    np.savez(save_path, **payload)


def load_projection_npz(npz_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"投影矩阵文件不存在：{npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    projection_matrix = data["projection_matrix"].astype(np.float64)
    metadata = {key: data[key] for key in data.files if key != "projection_matrix"}
    return projection_matrix, metadata


def build_config_from_metadata(cfg: VAMConfig, metadata: Dict[str, Any]) -> VAMConfig:
    if "N" in metadata:
        cfg.n_slice = int(metadata["N"])
    if "voxel_size_xy" in metadata:
        cfg.voxel_size_xy = float(metadata["voxel_size_xy"])
    if "resin_alpha" in metadata:
        cfg.resin_alpha = float(metadata["resin_alpha"])
    if "resin_Ec" in metadata:
        cfg.resin_Ec = float(metadata["resin_Ec"])
    if "exposure_time_per_angle_ms" in metadata:
        cfg.exposure_time_per_angle_ms = float(metadata["exposure_time_per_angle_ms"])
    if "projection_quantization_enabled" in metadata:
        cfg.projection_quantization_enabled = bool(metadata["projection_quantization_enabled"])
    if "projection_quantization_max_index" in metadata:
        cfg.projection_quantization_max_index = int(metadata["projection_quantization_max_index"])

    if "angles" in metadata:
        angles = np.asarray(metadata["angles"], dtype=np.float64)
        cfg.angles_override = angles
        cfg.angle_num = int(len(angles))
        if len(angles) >= 1:
            cfg.angle_start_deg = float(angles[0])
        if len(angles) >= 2:
            step = float(angles[1] - angles[0])
            cfg.angle_end_deg = float(angles[-1] + step)
    else:
        cfg.angles_override = None
    return cfg
