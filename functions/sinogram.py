from __future__ import annotations

import numpy as np
from skimage.transform import radon, resize
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.fft import fftfreq, fftshift

from .config import VAMConfig



def preprocess_target_image(cfg: VAMConfig) -> np.ndarray:
    """读取并预处理目标图像。

    输出为圆形容器内的二值目标掩膜。
    """
    img = imread(cfg.target_image_path)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])

    img = resize(img, (cfg.n_slice, cfg.n_slice), anti_aliasing=True, preserve_range=True)
    img = (img > 0.5).astype(np.float64)

    center = cfg.n_slice // 2
    y, x = np.ogrid[:cfg.n_slice, :cfg.n_slice]
    circle_mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2
    img = img * circle_mask

    # 这里保留你原始代码的显示方向约定。
    if cfg.preprocess_flipud_for_display_consistency:
        img = np.flipud(img)
    return img



def generate_initial_sinogram(target_mask: np.ndarray, cfg: VAMConfig) -> np.ndarray:
    """由目标掩膜生成初始正弦图，并做逐角度直接归一化。

    保持函数签名、输入输出形状和调用方式不变。
    与旧版本不同之处仅在于：
    - 不再做逐角度 5%/95% percentile 裁剪
    - 每个角度的投影直接按本列最小值/最大值归一化到 [0, 1]

    这样可以避免极小、极稀疏目标在 percentile 裁剪后整列被压成 0。
    """
    sinogram_raw = radon(target_mask, theta=cfg.angles, circle=True, preserve_range=True)
    sinogram_processed = np.zeros_like(sinogram_raw, dtype=np.float64)

    for i in range(sinogram_raw.shape[1]):
        proj = sinogram_raw[:, i].astype(np.float64)
        proj_min = np.min(proj)
        proj_max = np.max(proj)

        if proj_max > proj_min:
            sinogram_processed[:, i] = (proj - proj_min) / (proj_max - proj_min)
        else:
            sinogram_processed[:, i] = 0.0

    return sinogram_processed



def create_radon_filter(proj_len: int, filter_type: str = "hanning") -> np.ndarray:
    """生成频域滤波核。"""
    freq = fftshift(fftfreq(proj_len))
    ramp = np.abs(freq)
    if filter_type.lower() == "hanning":
        window = 0.5 + 0.5 * np.cos(2.0 * np.pi * np.linspace(-0.5, 0.5, proj_len))
    elif filter_type.lower() == "none":
        window = np.ones(proj_len, dtype=np.float64)
    else:
        raise ValueError(f"不支持的滤波类型：{filter_type}")
    return fftshift(ramp * window)
