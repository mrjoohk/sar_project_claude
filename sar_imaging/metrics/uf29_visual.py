"""UF-29: Visual BP Verification Module.

Generates diagnostic plots for visual inspection of BP results.
Fulfills SR-07 and enables agent-based visual quality verification.
"""
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info(f"Saved: {path}")


def generate_intensity_db_plot(image: np.ndarray, output_path: str,
                               vmin: float = -40, vmax: float = 0):
    """Generate dB-scale intensity image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    # Normalize relative to peak
    peak_db = np.max(intensity_db)
    intensity_db -= peak_db

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(intensity_db, cmap='gray', vmin=vmin, vmax=vmax,
                   aspect='equal', origin='lower')
    ax.set_title('SAR Intensity (dB, rel. to peak)')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Azimuth [pixel]')
    plt.colorbar(im, ax=ax, label='dB')
    _save_figure(fig, output_path)


def generate_phase_plot(image: np.ndarray, output_path: str):
    """Generate phase image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phase = np.angle(image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                   aspect='equal', origin='lower')
    ax.set_title('SAR Phase')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Azimuth [pixel]')
    plt.colorbar(im, ax=ax, label='radians')
    _save_figure(fig, output_path)


def generate_histogram_plot(image: np.ndarray, output_path: str):
    """Generate intensity histogram with dynamic range annotation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    peak_db = np.max(intensity_db)
    intensity_db_rel = intensity_db - peak_db

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(intensity_db_rel.ravel(), bins=100, color='steelblue', alpha=0.7)
    ax.set_xlabel('Intensity (dB, rel. to peak)')
    ax.set_ylabel('Count')
    ax.set_title('Intensity Histogram')

    # Annotate dynamic range
    noise_floor = np.percentile(intensity_db_rel[intensity_db_rel > -200], 5)
    ax.axvline(noise_floor, color='red', linestyle='--', label=f'5th pct: {noise_floor:.1f} dB')
    ax.axvline(0, color='green', linestyle='--', label='Peak: 0 dB')
    ax.legend()
    _save_figure(fig, output_path)


def generate_irf_plots(image: np.ndarray, output_dir: str,
                       peak_yx: Optional[Tuple[int, int]] = None,
                       crop_radius: int = 32):
    """Generate IRF diagnostic plots for point target.

    Generates: irf_2d_contour.png, range_cut.png, azimuth_cut.png.

    Returns:
        psf_metrics: dict with aspect_ratio, 3dB widths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    if peak_yx is None:
        idx = np.argmax(intensity)
        peak_iy, peak_ix = np.unravel_index(idx, intensity.shape)
    else:
        peak_iy, peak_ix = peak_yx

    Ny, Nx = image.shape

    # Crop around peak
    y0 = max(peak_iy - crop_radius, 0)
    y1 = min(peak_iy + crop_radius + 1, Ny)
    x0 = max(peak_ix - crop_radius, 0)
    x1 = min(peak_ix + crop_radius + 1, Nx)
    crop = image[y0:y1, x0:x1]
    crop_int = np.abs(crop) ** 2
    crop_db = 10.0 * np.log10(np.maximum(crop_int, 1e-30))
    peak_db = np.max(crop_db)
    crop_db_rel = crop_db - peak_db

    # 2D contour plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    levels = np.arange(-30, 1, 3)
    ax.contourf(crop_db_rel, levels=levels, cmap='jet')
    ax.contour(crop_db_rel, levels=levels, colors='k', linewidths=0.3)
    ax.set_title('IRF 2D Contour (dB)')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Azimuth [pixel]')
    ax.set_aspect('equal')
    _save_figure(fig, os.path.join(output_dir, 'irf_2d_contour.png'))

    # Range cut (horizontal through peak)
    local_py = peak_iy - y0
    local_px = peak_ix - x0
    range_cut = crop_db_rel[local_py, :]
    az_cut = crop_db_rel[:, local_px]

    # Range cut plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range_cut, 'b-', linewidth=1.5)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax.axhline(-13, color='g', linestyle='--', alpha=0.5, label='-13 dB (PSLR target)')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title('Range Cut through Peak')
    ax.set_ylim(-40, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, os.path.join(output_dir, 'range_cut.png'))

    # Azimuth cut plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(az_cut, 'b-', linewidth=1.5)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax.axhline(-13, color='g', linestyle='--', alpha=0.5, label='-13 dB (PSLR target)')
    ax.set_xlabel('Azimuth [pixel]')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title('Azimuth Cut through Peak')
    ax.set_ylim(-40, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, os.path.join(output_dir, 'azimuth_cut.png'))

    # Compute PSF metrics
    # 3dB width in range and azimuth
    range_3db = _compute_3db_width(range_cut)
    az_3db = _compute_3db_width(az_cut)
    aspect_ratio = max(range_3db, az_3db) / max(min(range_3db, az_3db), 0.5)

    psf_metrics = {
        'range_3db_pixels': float(range_3db),
        'azimuth_3db_pixels': float(az_3db),
        'psf_aspect_ratio': float(aspect_ratio),
        'peak_iy': int(peak_iy),
        'peak_ix': int(peak_ix),
    }
    return psf_metrics


def _compute_3db_width(cut_db: np.ndarray) -> float:
    """Compute -3dB width from a 1D dB cut (0 = peak)."""
    above_3db = cut_db >= -3.0
    if not np.any(above_3db):
        return 1.0
    indices = np.where(above_3db)[0]
    return float(indices[-1] - indices[0] + 1)


def generate_shadow_overlay(image: np.ndarray, vis_mask_2d: np.ndarray,
                            output_path: str):
    """Generate shadow overlay plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    peak_db = np.max(intensity_db)
    intensity_db_rel = intensity_db - peak_db

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(intensity_db_rel, cmap='gray', vmin=-40, vmax=0,
              aspect='equal', origin='lower')
    # Overlay shadow mask in red
    shadow = vis_mask_2d < 0.5
    overlay = np.zeros((*shadow.shape, 4))
    overlay[shadow] = [1, 0, 0, 0.3]  # red with alpha
    ax.imshow(overlay, aspect='equal', origin='lower')
    ax.set_title('Shadow Overlay (red = shadow)')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Azimuth [pixel]')
    _save_figure(fig, output_path)


def generate_layover_overlay(image: np.ndarray, layover_flag_2d: np.ndarray,
                             output_path: str):
    """Generate layover overlay plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    peak_db = np.max(intensity_db)
    intensity_db_rel = intensity_db - peak_db

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(intensity_db_rel, cmap='gray', vmin=-40, vmax=0,
              aspect='equal', origin='lower')
    # Overlay layover in blue
    layover = layover_flag_2d > 0.5
    overlay = np.zeros((*layover.shape, 4))
    overlay[layover] = [0, 0, 1, 0.3]  # blue with alpha
    ax.imshow(overlay, aspect='equal', origin='lower')
    ax.set_title('Layover Overlay (blue = layover)')
    ax.set_xlabel('Range [pixel]')
    ax.set_ylabel('Azimuth [pixel]')
    _save_figure(fig, output_path)


def generate_texture_comparison(image: np.ndarray, texture_gray: np.ndarray,
                                output_path: str):
    """Generate side-by-side comparison of SAR image and input texture.

    This is critical for verifying coordinate alignment: features in the
    texture (buildings, roads) should appear at the same spatial locations
    in the SAR image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intensity = np.abs(image) ** 2
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-30))
    peak_db = np.max(intensity_db)
    intensity_db_rel = intensity_db - peak_db

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Left: Input texture (ENU-aligned)
    axes[0].imshow(texture_gray, cmap='gray', origin='lower',
                   aspect='equal', vmin=0, vmax=1)
    axes[0].set_title('Input Texture (ENU)')
    axes[0].set_xlabel('X [pixel]')
    axes[0].set_ylabel('Y [pixel]')

    # Middle: SAR intensity
    axes[1].imshow(intensity_db_rel, cmap='gray', vmin=-40, vmax=0,
                   origin='lower', aspect='equal')
    axes[1].set_title('SAR Intensity (dB)')
    axes[1].set_xlabel('Range [pixel]')
    axes[1].set_ylabel('Azimuth [pixel]')

    # Right: Overlay (texture in green, SAR in magenta)
    Ny_sar, Nx_sar = image.shape
    Ny_tex, Nx_tex = texture_gray.shape
    # Resize texture to match SAR grid if sizes differ
    if (Ny_tex != Ny_sar) or (Nx_tex != Nx_sar):
        from scipy.ndimage import zoom
        zy = Ny_sar / Ny_tex
        zx = Nx_sar / Nx_tex
        tex_resized = zoom(texture_gray, (zy, zx), order=1)
    else:
        tex_resized = texture_gray

    # Normalize SAR for overlay
    sar_norm = np.clip((intensity_db_rel + 40) / 40, 0, 1)
    tex_norm = np.clip(tex_resized, 0, 1)

    overlay = np.stack([sar_norm, tex_norm, sar_norm * 0.3], axis=-1)
    axes[2].imshow(overlay, origin='lower', aspect='equal')
    axes[2].set_title('Overlay (M=SAR, G=Texture)')
    axes[2].set_xlabel('Pixel')

    plt.tight_layout()
    _save_figure(fig, output_path)


def run_visual_verification(image: np.ndarray,
                            output_dir: str,
                            scene_type: str = 'point_target',
                            vis_mask: Optional[np.ndarray] = None,
                            layover_flag: Optional[np.ndarray] = None,
                            bp_grid_shape: Optional[Tuple[int, int]] = None,
                            texture_gray: Optional[np.ndarray] = None
                            ) -> Dict[str, Any]:
    """Run full visual verification suite.

    Args:
        image: [Ny, Nx] complex SLC.
        output_dir: Directory to save plots.
        scene_type: 'point_target' or 'urban'.
        vis_mask: [N] visibility mask (will be reshaped to image shape if possible).
        layover_flag: [N] layover flags.
        bp_grid_shape: (Ny, Nx) for reshaping masks.
        texture_gray: [Ny_tex, Nx_tex] optional input texture for comparison.

    Returns:
        visual_metrics: Dict of visual quality metrics.
    """
    _ensure_dir(output_dir)
    visual_metrics = {}

    # Basic plots
    generate_intensity_db_plot(image, os.path.join(output_dir, 'intensity_db.png'))
    generate_phase_plot(image, os.path.join(output_dir, 'phase.png'))
    generate_histogram_plot(image, os.path.join(output_dir, 'histogram.png'))

    # Dynamic range
    from .uf17_metrics import compute_dynamic_range
    dr = compute_dynamic_range(image)
    visual_metrics['dynamic_range_db'] = dr

    if scene_type == 'point_target':
        psf_metrics = generate_irf_plots(image, output_dir)
        visual_metrics.update(psf_metrics)
    elif scene_type == 'urban':
        Ny, Nx = image.shape
        if vis_mask is not None:
            if vis_mask.shape[0] == Ny * Nx:
                vm_2d = vis_mask.reshape(Ny, Nx)
            elif bp_grid_shape and vis_mask.shape[0] >= bp_grid_shape[0] * bp_grid_shape[1]:
                vm_2d = vis_mask[:Ny*Nx].reshape(Ny, Nx)
            else:
                vm_2d = np.ones((Ny, Nx), dtype=np.float32)
            generate_shadow_overlay(image, vm_2d,
                                    os.path.join(output_dir, 'shadow_overlay.png'))

            # Shadow contrast metric
            shadow_region = np.abs(image[vm_2d < 0.5]) ** 2
            nonshadow_region = np.abs(image[vm_2d >= 0.5]) ** 2
            if len(shadow_region) > 0 and len(nonshadow_region) > 0:
                mean_shadow = np.mean(shadow_region)
                mean_nonshadow = np.mean(nonshadow_region)
                if mean_nonshadow > 1e-30:
                    contrast_db = 10.0 * np.log10(max(mean_shadow, 1e-30) / mean_nonshadow)
                    visual_metrics['shadow_contrast_db'] = float(contrast_db)

        if layover_flag is not None:
            if layover_flag.shape[0] == Ny * Nx:
                lf_2d = layover_flag.reshape(Ny, Nx)
            elif bp_grid_shape and layover_flag.shape[0] >= bp_grid_shape[0] * bp_grid_shape[1]:
                lf_2d = layover_flag[:Ny*Nx].reshape(Ny, Nx)
            else:
                lf_2d = np.zeros((Ny, Nx), dtype=np.float32)
            generate_layover_overlay(image, lf_2d,
                                     os.path.join(output_dir, 'layover_overlay.png'))

    # Texture comparison (for real data coordinate alignment verification)
    if texture_gray is not None:
        generate_texture_comparison(image, texture_gray,
                                    os.path.join(output_dir, 'texture_comparison.png'))
        logger.info("Generated texture comparison plot for coordinate alignment check")

    # Save visual metrics
    metrics_path = os.path.join(output_dir, '..', 'visual_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(visual_metrics, f, indent=2)
    logger.info(f"Visual metrics saved: {metrics_path}")

    return visual_metrics
