from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft

from .config import VAMConfig
from .sinogram import create_radon_filter


_PROJECTION_PRECOMPUTE_CACHE: dict[tuple[int, float, float, str], tuple[np.ndarray, np.ndarray]] = {}


def _get_projection_precompute(cfg: VAMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Cache filter/depth terms that are identical for every target in one scan."""
    key = (
        int(cfg.n_slice),
        float(cfg.voxel_size_xy),
        float(cfg.container_radius),
        str(cfg.filter_type).lower(),
    )
    cached = _PROJECTION_PRECOMPUTE_CACHE.get(key)
    if cached is not None:
        return cached

    filter_kernel = create_radon_filter(cfg.n_slice, cfg.filter_type).astype(np.float64)
    center = cfg.n_slice // 2
    s_coords = np.arange(cfg.n_slice, dtype=np.float64)
    s_phys = (s_coords - float(center)) * float(cfg.voxel_size_xy)
    depth = 2.0 * np.sqrt(np.clip(float(cfg.container_radius) ** 2 - s_phys ** 2, 0.0, None))

    cached = (filter_kernel, depth)
    _PROJECTION_PRECOMPUTE_CACHE[key] = cached
    return cached


def generate_initial_projection_matrix(sinogram: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    """由初始正弦图生成连续的原始初始投影矩阵。

    本函数只负责反向补偿、滤波和非负裁剪。
    投影矩阵离散化统一交由 physics_printing.quantize_projection_matrix 处理。
    """
    filter_kernel, depth = _get_projection_precompute(cfg)
    sinogram_f64 = np.asarray(sinogram, dtype=np.float64)

    compensated = sinogram_f64 * np.exp(float(cfg.resin_alpha) * depth)[:, None]
    filtered = ifft(fft(compensated, axis=0) * filter_kernel[:, None], axis=0).real
    return np.clip(filtered, 0.0, None)
