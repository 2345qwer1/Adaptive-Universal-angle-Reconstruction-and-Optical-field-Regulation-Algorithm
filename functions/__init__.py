from .config import VAMConfig, get_default_config, ensure_output_dirs
from .sinogram import preprocess_target_image, generate_initial_sinogram
from .initial_projection import generate_initial_projection_matrix
from .physics_printing import (
    compensate_and_print_initial_projection,
    print_any_projection,
    calculate_iou,
    build_container_mask,
    quantize_projection_matrix,
)
from .optimization import optimize_projection_matrix
from .visualization import visualize_and_save_results
from .io_utils import save_projection_npz, load_projection_npz, build_config_from_metadata
from .resolution_w_analysis import (
    STRUCTURE_RADIAL_CONCENTRIC_RING,
    STRUCTURE_ANGULAR_BLOCKS_SINGLE_LAYER,
    STRUCTURE_ANGULAR_BLOCKS_MULTI_LAYER,
    ANGULAR_MULTI_LAYER_AXIS_THETA,
    generate_radial_concentric_rings,
    generate_angular_blocks_single_layer,
    generate_angular_blocks_multi_layer,
    sample_radial_profiles_four_axes,
    sample_angular_profile,
    compute_ctf_dose,
    compute_radial_ring_ctf_dose_mean,
    compute_angular_block_ctf_dose_mean,
    find_resolution_w_for_structure,
    build_resolution_summary_rows,
    write_csv_rows,
    plot_resolution_w_vs_r,
    plot_ctf_dose_heatmap,
)
