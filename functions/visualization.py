from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .config import VAMConfig
from .physics_printing import calculate_iou, build_container_mask


def build_compare_map(target: np.ndarray, cured: np.ndarray, container_mask: np.ndarray | None = None) -> np.ndarray:
    """生成 target 与 cured 的对比标签图。"""
    target_bool = np.asarray(target).astype(bool)
    cured_bool = np.asarray(cured).astype(bool)

    if container_mask is not None:
        mask_bool = np.asarray(container_mask).astype(bool)
        target_bool = np.logical_and(target_bool, mask_bool)
        cured_bool = np.logical_and(cured_bool, mask_bool)

    img = np.zeros_like(target_bool, dtype=int)
    img[np.logical_and(target_bool, cured_bool)] = 1
    img[np.logical_and(target_bool, np.logical_not(cured_bool))] = 2
    img[np.logical_and(np.logical_not(target_bool), cured_bool)] = 3
    return img


def _branch_error_maps(
    target: np.ndarray,
    cured: np.ndarray,
    container_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_bool = np.asarray(target).astype(bool)
    cured_bool = np.asarray(cured).astype(bool)

    if container_mask is not None:
        mask_bool = np.asarray(container_mask).astype(bool)
        target_bool = np.logical_and(target_bool, mask_bool)
        cured_bool = np.logical_and(cured_bool, mask_bool)

    overcure = np.logical_and(cured_bool, np.logical_not(target_bool)).astype(np.uint8)
    undercure = np.logical_and(target_bool, np.logical_not(cured_bool)).astype(np.uint8)
    signed = np.zeros_like(target_bool, dtype=np.int8)
    signed[undercure == 1] = -1
    signed[overcure == 1] = 1
    return overcure, undercure, signed


def _compute_focus_roi(
    target: np.ndarray,
    *extra_arrays: np.ndarray | None,
    min_half_width: int = 24,
    pad: int = 12,
) -> tuple[slice, slice]:
    active = np.asarray(target).astype(bool).copy()
    for arr in extra_arrays:
        if arr is not None:
            active |= np.asarray(arr).astype(bool)

    coords = np.argwhere(active)
    h, w = active.shape
    if coords.size == 0:
        r0, c0 = h // 2, w // 2
        return (
            slice(max(0, r0 - min_half_width), min(h, r0 + min_half_width + 1)),
            slice(max(0, c0 - min_half_width), min(w, c0 + min_half_width + 1)),
        )

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    r_min = max(0, int(r_min) - pad)
    r_max = min(h, int(r_max) + pad + 1)
    c_min = max(0, int(c_min) - pad)
    c_max = min(w, int(c_max) + pad + 1)

    if (r_max - r_min) < 2 * min_half_width + 1:
        r_center = (r_min + r_max - 1) // 2
        r_min = max(0, r_center - min_half_width)
        r_max = min(h, r_center + min_half_width + 1)
    if (c_max - c_min) < 2 * min_half_width + 1:
        c_center = (c_min + c_max - 1) // 2
        c_min = max(0, c_center - min_half_width)
        c_max = min(w, c_center + min_half_width + 1)
    return slice(r_min, r_max), slice(c_min, c_max)


def _imshow_error_map(ax, img: np.ndarray, title: str, *, cmap, vmin: int, vmax: int,
                      roi: tuple[slice, slice] | None = None):
    if roi is not None:
        disp = np.asarray(img)[roi[0], roi[1]]
        im = ax.imshow(disp, cmap=cmap, interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_xlabel(f"col {roi[1].start}:{roi[1].stop - 1}")
        ax.set_ylabel(f"row {roi[0].start}:{roi[0].stop - 1}")
    else:
        im = ax.imshow(img, cmap=cmap, interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_axis_off()
    ax.set_title(title)
    return im


def _plot_scalar_image(data: np.ndarray, *, title: str, cmap: str, extent: list[float],
                       origin: str = "lower", colorbar_label: str | None = None):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap=cmap, extent=extent, origin=origin, aspect="equal")
    if colorbar_label is not None:
        plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.axis("off")
    return fig


def _plot_projection_heatmap(proj: np.ndarray, cfg: VAMConfig):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(proj, cmap="viridis", origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Projection intensity")

    title = "Projection Matrix Heatmap"
    if cfg.projection_quantization_enabled:
        proj_max = float(np.max(np.asarray(proj, dtype=np.float64))) if np.size(proj) > 0 else 0.0
        if cfg.projection_quantization_max_index > 0 and proj_max > 0:
            eff_step = proj_max / float(cfg.projection_quantization_max_index)
            title += (
                f"\nlevels = 0 ... {cfg.projection_quantization_max_index}, "
                f"effective step = {eff_step:.6g}"
            )
        else:
            title += f"\nlevels = 0 ... {cfg.projection_quantization_max_index}"

    ax.set_title(title)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Detector pixel index")
    return fig


def _plot_error_decomposition_figure(
    target: np.ndarray,
    cured: np.ndarray,
    cfg: VAMConfig,
    *,
    container_mask: np.ndarray | None,
    reference_cured: np.ndarray | None = None,
    reference_label: str = "Unoptimized",
    current_label: str = "Optimized",
):
    cmap_signed = ListedColormap(["red", "black", "green"])
    cmap_over = ListedColormap(["black", "green"])
    cmap_under = ListedColormap(["black", "red"])

    roi = None
    if bool(getattr(cfg, "error_map_zoom_to_roi", True)):
        roi = _compute_focus_roi(
            target,
            reference_cured,
            cured,
            min_half_width=int(getattr(cfg, "error_map_min_half_width_px", 24)),
            pad=int(getattr(cfg, "error_map_roi_pad_px", 12)),
        )

    layout_mode = getattr(cfg, "error_map_layout_mode", "single_row")
    if layout_mode == "compare_2x3":
        if reference_cured is None:
            raise ValueError("compare_2x3 模式要求提供 reference_cured。")

        ref_over, ref_under, ref_signed = _branch_error_maps(target, reference_cured, container_mask=container_mask)
        cur_over, cur_under, cur_signed = _branch_error_maps(target, cured, container_mask=container_mask)
        ref_iou = calculate_iou(target, reference_cured, cfg=cfg, container_mask=container_mask)
        cur_iou = calculate_iou(target, cured, cfg=cfg, container_mask=container_mask)

        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
        _imshow_error_map(axes[0, 0], ref_over, f"{reference_label} overcure\narea={int(ref_over.sum())} px", cmap=cmap_over, vmin=0, vmax=1, roi=roi)
        _imshow_error_map(axes[0, 1], ref_under, f"{reference_label} undercure\narea={int(ref_under.sum())} px", cmap=cmap_under, vmin=0, vmax=1, roi=roi)
        _imshow_error_map(axes[0, 2], ref_signed + 1, f"{reference_label} signed error", cmap=cmap_signed, vmin=0, vmax=2, roi=roi)
        _imshow_error_map(axes[1, 0], cur_over, f"{current_label} overcure\narea={int(cur_over.sum())} px", cmap=cmap_over, vmin=0, vmax=1, roi=roi)
        _imshow_error_map(axes[1, 1], cur_under, f"{current_label} undercure\narea={int(cur_under.sum())} px", cmap=cmap_under, vmin=0, vmax=1, roi=roi)
        _imshow_error_map(axes[1, 2], cur_signed + 1, f"{current_label} signed error", cmap=cmap_signed, vmin=0, vmax=2, roi=roi)

        roi_note = " (zoomed ROI within container)" if roi is not None else ""
        fig.suptitle(
            f"Error Decomposition{roi_note}\n"
            f"{reference_label} IoU(container) = {ref_iou:.4f} | "
            f"{current_label} IoU(container) = {cur_iou:.4f}",
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    over, under, signed = _branch_error_maps(target, cured, container_mask=container_mask)
    iou = calculate_iou(target, cured, cfg=cfg, container_mask=container_mask)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    _imshow_error_map(axes[0], over, f"Overcure\narea={int(over.sum())} px", cmap=cmap_over, vmin=0, vmax=1, roi=roi)
    _imshow_error_map(axes[1], under, f"Undercure\narea={int(under.sum())} px", cmap=cmap_under, vmin=0, vmax=1, roi=roi)
    _imshow_error_map(axes[2], signed + 1, "Signed error", cmap=cmap_signed, vmin=0, vmax=2, roi=roi)
    roi_note = " (zoomed ROI within container)" if roi is not None else ""
    fig.suptitle(f"Error Decomposition{roi_note} | IoU(container) = {iou:.4f}", fontsize=13)
    fig.tight_layout()
    return fig


def visualize_and_save_results(
    target: np.ndarray,
    dose: np.ndarray,
    cured: np.ndarray,
    proj: np.ndarray,
    cfg: VAMConfig,
    compare_save_path: str | Path | None = None,
    figure_prefix: str | Path | None = None,
    show: bool | None = None,
    reference_cured: np.ndarray | None = None,
    reference_label: str = "Unoptimized",
    current_label: str = "Optimized",
) -> None:
    if show is None:
        show = cfg.show_figures

    N = target.shape[0]
    voxel = cfg.voxel_size_xy
    extent = [0, N * voxel, 0, N * voxel]

    if figure_prefix is not None:
        figure_prefix = Path(figure_prefix)
        figure_prefix.parent.mkdir(parents=True, exist_ok=True)

    container_mask = build_container_mask(target.shape, cfg)
    compare_img = build_compare_map(target, cured, container_mask=container_mask)
    compare_cmap = ListedColormap(["black", "white", "red", "green"])

    figs = [
        (_plot_scalar_image(target, title="Target", cmap="gray", extent=extent), "target"),
        (_plot_projection_heatmap(proj, cfg), "projection_heatmap"),
        (_plot_scalar_image(dose, title=f"Dose Distribution | Ec = {cfg.resin_Ec}", cmap="plasma", extent=extent, colorbar_label="Dose (mJ/cm²)"), "dose"),
        (_plot_scalar_image(cured, title="Cured Structure", cmap="gray", extent=extent), "cured"),
        (_plot_error_decomposition_figure(
            target,
            cured,
            cfg,
            container_mask=container_mask,
            reference_cured=reference_cured,
            reference_label=reference_label,
            current_label=current_label,
        ), "compare"),
    ]

    if figure_prefix is not None:
        for fig_obj, suffix in figs:
            fig_obj.savefig(
                str(figure_prefix.with_name(f"{figure_prefix.name}_{suffix}.png")),
                dpi=200,
                bbox_inches="tight",
            )

    if compare_save_path is not None:
        compare_save_path = Path(compare_save_path)
        compare_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img = np.flipud(compare_img) if cfg.save_compare_without_title else compare_img
        plt.imsave(compare_save_path, save_img, cmap=compare_cmap, vmin=0, vmax=3, format="png")

    if show:
        plt.show()
    else:
        plt.close("all")
