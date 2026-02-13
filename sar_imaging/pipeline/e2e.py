"""E2E Pipeline for Urban SAR Simulator/Imager.

Implements the integration design from 04_integration_design_urban_sar.md.
Stages: Config → Scene → Platform/Raw → Imaging → Metrics → Visual.
"""
import numpy as np
import os
import json
import time
import logging
import yaml
from typing import Dict, Any, Optional

from ..config import load_config, validate_config, _get_nested
from ..data_contracts import RadarMeta, PlatformMeta, Scatterers, BPGrid
from ..scene.synthetic_dem import (
    make_flat_dem, make_slope_dem, make_box_dem, make_wall_dem,
    dem_to_triangles, compute_triangle_normals_and_areas,
)
from ..scene.scatterers import facet_brdf_scatterers, centroid_scatterers, add_point_targets
from ..scene.platform import generate_stripmap_trajectory
from ..uf.uf20_shadow import compute_shadow_mask, apply_shadow_mask
from ..uf.uf21_layover import compute_layover_flag, apply_layover_flag
from ..uf.uf22_enu import create_bp_grid_from_dem
from ..uf.uf25_power_norm import apply_power_normalization
from ..uf.uf28_raw_echo import generate_raw_echo
from ..uf.uf27_streaming_bp import backproject_full, backproject_tiled
from ..metrics.uf17_metrics import compute_all_metrics
from ..metrics.uf29_visual import run_visual_verification

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, level: str = 'INFO'):
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'log.txt')

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    root.addHandler(ch)


def run_pipeline(config_path: str = None, config_override: dict = None) -> Dict[str, Any]:
    """Run the full E2E SAR pipeline.

    Args:
        config_path: Path to YAML config file.
        config_override: Dict to override config values.

    Returns:
        Dict with results: metrics, paths, timing.
    """
    t_pipeline_start = time.time()

    # --- Stage 0: Config Validation ---
    cfg = load_config(config_path)
    if config_override:
        _deep_update(cfg, config_override)
    validate_config(cfg)

    seed = cfg.get('seed', 42)
    np.random.seed(seed)

    run_id = _get_nested(cfg, 'output.run_id') or 'run_001'
    root_dir = _get_nested(cfg, 'output.root_dir') or 'results'
    output_dir = os.path.join(root_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    log_level = _get_nested(cfg, 'logging.level') or 'INFO'
    setup_logging(output_dir, log_level)
    logger.info(f"=== Pipeline start: {run_id} ===")
    logger.info(f"Config: seed={seed}, method={_get_nested(cfg, 'imaging.method')}")

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save meta
    meta = {
        'seed': seed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_id': run_id,
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    results = {'run_id': run_id, 'output_dir': output_dir}

    # --- Stage 1: Scene Pre-stage ---
    logger.info("--- Stage 1: Scene Pre-stage ---")
    t_scene_start = time.time()

    scene_cfg = cfg.get('scene', {})
    texture_path = scene_cfg.get('texture_path', None)

    texture_gray = None  # Will be set if real texture is loaded

    if texture_path and os.path.exists(texture_path):
        # *** Real urban data path: load texture TIF ***
        from ..scene.texture_loader import (
            load_texture_tif, texture_to_rcs, texture_to_flat_dem,
        )
        target_spacing = scene_cfg.get('grid_spacing_m', 1.0)
        x_axis, y_axis, texture_gray, dem_spacing = load_texture_tif(
            texture_path, target_spacing_m=target_spacing)

        # Pseudo-DEM from texture brightness
        building_cfg = scene_cfg.get('building_detection', {})
        dem = texture_to_flat_dem(
            x_axis, y_axis, texture_gray,
            base_height=building_cfg.get('base_height_m', 0.0),
            building_height=building_cfg.get('building_height_m', 25.0),
            building_threshold=building_cfg.get('threshold', 0.65),
        )

        # Texture-based RCS map
        rcs_map = texture_to_rcs(texture_gray,
                                  rcs_min=scene_cfg.get('rcs_min', 0.01),
                                  rcs_max=scene_cfg.get('rcs_max', 5.0))

        logger.info(f"Real texture loaded: {texture_path}")
        logger.info(f"  Grid: {len(x_axis)}x{len(y_axis)}, spacing={dem_spacing:.2f}m")
        logger.info(f"  ENU x=[{x_axis[0]:.1f},{x_axis[-1]:.1f}], "
                     f"y=[{y_axis[0]:.1f},{y_axis[-1]:.1f}]")
    else:
        # *** Synthetic DEM path ***
        rcs_map = None
        synthetic_cfg = scene_cfg.get('synthetic', {})
        dem_type = synthetic_cfg.get('type', 'flat')
        dem_size = tuple(synthetic_cfg.get('size', [128, 128]))
        dem_spacing = synthetic_cfg.get('spacing_m', 1.0)

        if dem_type == 'flat':
            x_axis, y_axis, dem = make_flat_dem(size=dem_size, spacing=dem_spacing)
        elif dem_type == 'slope':
            slope_deg = synthetic_cfg.get('slope_deg', 15.0)
            x_axis, y_axis, dem = make_slope_dem(size=dem_size, spacing=dem_spacing,
                                                 slope_deg=slope_deg)
        elif dem_type == 'box':
            box_h = synthetic_cfg.get('box_height_m', 30.0)
            box_w = synthetic_cfg.get('box_width_m', 50.0)
            x_axis, y_axis, dem = make_box_dem(size=dem_size, spacing=dem_spacing,
                                               box_height=box_h, box_width=box_w)
        elif dem_type == 'wall':
            wall_h = synthetic_cfg.get('wall_height_m', 30.0)
            x_axis, y_axis, dem = make_wall_dem(size=dem_size, spacing=dem_spacing,
                                                wall_height=wall_h)
        else:
            raise ValueError(f"Unknown DEM type: {dem_type}")

        logger.info(f"DEM created: {dem_type}, size={dem.shape}, spacing={dem_spacing}m")

    # BP grid from DEM
    bp_grid = create_bp_grid_from_dem(x_axis, y_axis)

    # Generate scatterers
    scatter_cfg = cfg.get('scatter', {})
    model = scatter_cfg.get('model', 'facet_brdf')

    # We need a representative platform position for scattering
    plat_cfg = cfg.get('platform', {})
    altitude = plat_cfg.get('altitude_m', 5000.0)
    look_angle = plat_cfg.get('look_angle_deg', 45.0)
    ground_range = altitude * np.tan(np.radians(look_angle))
    # Approx platform position (side-looking from negative x)
    repr_platform_pos = np.array([-ground_range, 0.0, altitude])

    if model == 'facet_brdf':
        vertices, triangles, centroids = dem_to_triangles(x_axis, y_axis, dem)
        normals, areas = compute_triangle_normals_and_areas(vertices, triangles)
        brdf_cfg = scatter_cfg.get('brdf', {})
        scatterers = facet_brdf_scatterers(
            vertices, triangles, centroids, normals, areas,
            platform_pos=repr_platform_pos,
            alpha=brdf_cfg.get('alpha', 0.7),
            beta=brdf_cfg.get('beta', 0.3),
            p=brdf_cfg.get('p', 10.0),
        )
    else:
        scatterers = centroid_scatterers(x_axis, y_axis, dem)

    # Apply texture-based RCS modulation if rcs_map is available
    if rcs_map is not None:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (y_axis, x_axis), rcs_map.astype(np.float64),
            method='nearest', bounds_error=False, fill_value=0.01)
        pts = np.column_stack([scatterers.sy.astype(np.float64),
                                scatterers.sx.astype(np.float64)])
        rcs_values = interp(pts).astype(np.float32)
        scatterers.s_rcs = scatterers.s_rcs * rcs_values
        logger.info(f"Texture RCS applied: rcs range=[{scatterers.s_rcs.min():.4f}, "
                     f"{scatterers.s_rcs.max():.4f}]")

    # Add point targets
    point_targets = scene_cfg.get('point_targets', [])
    if point_targets:
        scatterers = add_point_targets(scatterers, point_targets)
        logger.info(f"Added {len(point_targets)} point target(s)")

    logger.info(f"Scatterers: N={scatterers.N}, model={model}")

    # Shadow mask
    if scatter_cfg.get('enable_shadow', True):
        vis_mask = compute_shadow_mask(
            scatterers, repr_platform_pos, x_axis, y_axis, dem)
        scatterers = apply_shadow_mask(scatterers, vis_mask)

    # Layover flag
    if scatter_cfg.get('enable_layover', True) and scatterers.snx is not None:
        layover_flag = compute_layover_flag(scatterers, repr_platform_pos)
        scatterers = apply_layover_flag(scatterers, layover_flag)

    # Power normalization
    if scatter_cfg.get('enable_power_norm', True):
        scatterers = apply_power_normalization(scatterers, repr_platform_pos)

    scatterers.validate()
    t_scene_end = time.time()
    results['scene_time_s'] = t_scene_end - t_scene_start

    # --- Stage 2: Platform & Raw Generation ---
    logger.info("--- Stage 2: Platform & Raw Generation ---")
    t_raw_start = time.time()

    radar_cfg = cfg.get('radar', {})
    radar = RadarMeta(
        fc=radar_cfg.get('fc_hz', 9.65e9),
        bandwidth=radar_cfg.get('bandwidth_hz', 300e6),
        prf=radar_cfg.get('prf_hz', 500.0),
        tp=radar_cfg.get('tp_s', 10e-6),
    )
    radar.validate()

    platform = generate_stripmap_trajectory(
        altitude=plat_cfg.get('altitude_m', 5000.0),
        velocity=plat_cfg.get('velocity_m_s', 100.0),
        look_angle_deg=plat_cfg.get('look_angle_deg', 45.0),
        azimuth_deg=plat_cfg.get('azimuth_deg', 0.0),
        num_pulses=plat_cfg.get('num_pulses', 256),
        prf=radar.prf,
    )

    raw, tau_axis = generate_raw_echo(scatterers, platform, radar)
    t_raw_end = time.time()
    results['raw_time_s'] = t_raw_end - t_raw_start

    # --- Stage 3: Imaging (BP) ---
    logger.info("--- Stage 3: Imaging (BP) ---")
    t_bp_start = time.time()

    bp_cfg = cfg.get('bp', {})
    if bp_cfg.get('enable_streaming', False):
        tile_size = tuple(bp_cfg.get('tile_size', [128, 128]))
        k_batch = bp_cfg.get('k_batch', 64)
        overlap = bp_cfg.get('overlap', 16)
        stitch = bp_cfg.get('stitch_method', 'linear_blend')
        image = backproject_tiled(raw, tau_axis, platform, radar, bp_grid,
                                  tile_size=tile_size, k_batch=k_batch,
                                  overlap=overlap, stitch_method=stitch)
    else:
        image = backproject_full(raw, tau_axis, platform, radar, bp_grid)

    t_bp_end = time.time()
    results['bp_time_s'] = t_bp_end - t_bp_start
    logger.info(f"BP image: shape={image.shape}, dtype={image.dtype}")

    # --- Stage 3.5: Speckle Noise (optional) ---
    speckle_cfg = cfg.get('speckle', {})
    if speckle_cfg.get('enable', False):
        logger.info("--- Stage 3.5: Speckle Noise ---")
        from ..uf.uf31_speckle import add_speckle
        enl = speckle_cfg.get('enl', 1.0)
        speckle_seed = seed + 1000  # derived seed for reproducibility
        # Save clean image first
        np.save(os.path.join(output_dir, 'slc_clean.npy'), image)
        image = add_speckle(image, enl=enl, seed=speckle_seed)

    # --- Stage 4: Metrics & Product ---
    logger.info("--- Stage 4: Metrics & Product ---")
    visual_cfg = cfg.get('visual', {})
    scene_type = visual_cfg.get('scene_type', 'point_target')
    metrics = compute_all_metrics(image, scene_type=scene_type)
    results['metrics'] = metrics

    # Stage timings (UF-32)
    stage_timings = {
        'scene_time_s': results.get('scene_time_s', 0),
        'raw_time_s': results.get('raw_time_s', 0),
        'bp_time_s': results.get('bp_time_s', 0),
    }
    metrics['stage_timings'] = stage_timings

    # Save products
    np.save(os.path.join(output_dir, 'slc.npy'), image)

    # Save intensity PNG
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    peak_db = np.max(intensity_db)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(intensity_db - peak_db, cmap='gray', vmin=-40, vmax=0,
              aspect='equal', origin='lower')
    ax.set_title(f'SAR Intensity - {run_id}')
    fig.savefig(os.path.join(output_dir, 'intensity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # --- Stage 5: Visual Verification ---
    if visual_cfg.get('enable', True):
        logger.info("--- Stage 5: Visual Verification ---")
        vis_report_dir = os.path.join(output_dir, 'visual_report')

        # Grid scatterer-level masks onto BP grid for visualization
        vis_mask_2d = _grid_scatterer_mask(
            scatterers, scatterers.vis_mask, bp_grid, default_val=1.0)
        layover_2d = _grid_scatterer_mask(
            scatterers, scatterers.layover_flag, bp_grid, default_val=0.0)

        visual_metrics = run_visual_verification(
            image, vis_report_dir,
            scene_type=scene_type,
            vis_mask=vis_mask_2d.ravel() if vis_mask_2d is not None else None,
            layover_flag=layover_2d.ravel() if layover_2d is not None else None,
            bp_grid_shape=(bp_grid.Ny, bp_grid.Nx),
            texture_gray=texture_gray,
        )
        results['visual_metrics'] = visual_metrics

    t_pipeline_end = time.time()
    results['total_time_s'] = t_pipeline_end - t_pipeline_start

    logger.info(f"=== Pipeline complete: {results['total_time_s']:.2f}s ===")
    logger.info(f"Results: {output_dir}")

    return results


def _grid_scatterer_mask(scatterers: Scatterers, mask_1d,
                         bp_grid: BPGrid, default_val: float = 1.0):
    """Grid a per-scatterer mask onto the BP grid using nearest-neighbor.

    Scatterers from facet BRDF model have centroids that don't align 1:1
    with BP grid pixels. This function bins scatterer values onto the grid.

    Returns:
        mask_2d: [Ny, Nx] gridded mask, or None if mask_1d is None.
    """
    if mask_1d is None:
        return None

    Ny, Nx = bp_grid.Ny, bp_grid.Nx
    x_axis = bp_grid.x_axis
    y_axis = bp_grid.y_axis

    # Grid with accumulation
    mask_sum = np.full((Ny, Nx), 0.0, dtype=np.float64)
    mask_count = np.zeros((Ny, Nx), dtype=np.int32)

    dx = x_axis[1] - x_axis[0] if Nx > 1 else 1.0
    dy = y_axis[1] - y_axis[0] if Ny > 1 else 1.0

    # Map scatterer positions to grid indices
    ix = np.round((scatterers.sx - x_axis[0]) / dx).astype(np.int64)
    iy = np.round((scatterers.sy - y_axis[0]) / dy).astype(np.int64)

    valid = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
    v = np.where(valid)[0]

    np.add.at(mask_sum, (iy[v], ix[v]), mask_1d[v].astype(np.float64))
    np.add.at(mask_count, (iy[v], ix[v]), 1)

    # Average where we have data, default elsewhere
    result = np.full((Ny, Nx), default_val, dtype=np.float32)
    has_data = mask_count > 0
    result[has_data] = (mask_sum[has_data] / mask_count[has_data]).astype(np.float32)

    return result


def _deep_update(d: dict, u: dict):
    """Recursively update dict d with dict u."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
