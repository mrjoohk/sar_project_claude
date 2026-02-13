"""Texture loader for real urban imagery (GeoTIFF or plain TIF).

Handles coordinate alignment between TIF pixel space and ENU coordinates.

KEY COORDINATE ALIGNMENT:
  TIF convention:  row 0 = top of image,  row increases downward
  ENU convention:  y=0 = south edge,     y increases northward

  Therefore: TIF must be flipped vertically when converting to ENU.
  TIF row 0 → ENU y_max (north edge)
  TIF row H → ENU y_min (south edge)
"""
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_texture_tif(path: str,
                     target_spacing_m: float = 1.0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a TIF texture and convert to ENU-aligned grayscale.

    Args:
        path: Path to TIF file.
        target_spacing_m: Target grid spacing for SAR simulation [m].
            The texture will be downsampled to this spacing.

    Returns:
        x_axis: [Nx] ENU x-coordinates (East) [m]
        y_axis: [Ny] ENU y-coordinates (North) [m]
        texture_gray: [Ny, Nx] grayscale intensity [0..1], ENU-aligned
            (row 0 = south/y_min, row -1 = north/y_max)
        pixel_spacing: actual pixel spacing [m]
    """
    import rasterio

    with rasterio.open(path) as src:
        bands = src.count
        H, W = src.height, src.width
        transform = src.transform

        # Read all bands
        data = src.read()  # [bands, H, W]

        # Determine pixel spacing
        if transform and transform.a != 1.0:
            # Has real geotransform
            pixel_spacing_x = abs(transform.a)
            pixel_spacing_y = abs(transform.e)
            pixel_spacing = (pixel_spacing_x + pixel_spacing_y) / 2
            bounds = src.bounds
            extent_x = bounds.right - bounds.left
            extent_y = bounds.top - bounds.bottom
            logger.info(f"Georeferenced TIF: {pixel_spacing:.4f}m/pixel, "
                        f"extent={extent_x:.1f}x{extent_y:.1f}m")
        else:
            # No geotransform — estimate from typical aerial imagery
            pixel_spacing = 0.25  # 25cm default (Korean orthophoto)
            extent_x = W * pixel_spacing
            extent_y = H * pixel_spacing
            logger.warning(f"No geotransform. Assuming {pixel_spacing}m/pixel, "
                           f"extent={extent_x:.1f}x{extent_y:.1f}m")

    # Convert to grayscale [0..1]
    if bands >= 3:
        # Luminance: 0.299*R + 0.587*G + 0.114*B
        gray = (0.299 * data[0].astype(np.float64) +
                0.587 * data[1].astype(np.float64) +
                0.114 * data[2].astype(np.float64)) / 255.0
    elif bands == 1:
        gray = data[0].astype(np.float64) / 255.0
    else:
        gray = np.mean(data.astype(np.float64), axis=0) / 255.0

    # *** CRITICAL: Flip vertically for ENU alignment ***
    # TIF row 0 = north (top), but ENU row 0 = south (bottom)
    gray = gray[::-1, :]

    # Downsample if needed
    downsample = max(1, int(round(target_spacing_m / pixel_spacing)))
    if downsample > 1:
        Hnew = H // downsample
        Wnew = W // downsample
        gray = gray[:Hnew * downsample, :Wnew * downsample]
        gray = gray.reshape(Hnew, downsample, Wnew, downsample).mean(axis=(1, 3))
        actual_spacing = pixel_spacing * downsample
        logger.info(f"Downsampled {downsample}x: {W}x{H} → {Wnew}x{Hnew}, "
                     f"spacing={actual_spacing:.2f}m")
    else:
        actual_spacing = pixel_spacing
        Hnew, Wnew = H, W

    # Create ENU axes centered on the scene
    extent_x = Wnew * actual_spacing
    extent_y = Hnew * actual_spacing
    x_axis = np.arange(Wnew, dtype=np.float64) * actual_spacing - extent_x / 2
    y_axis = np.arange(Hnew, dtype=np.float64) * actual_spacing - extent_y / 2

    logger.info(f"Texture loaded: {Wnew}x{Hnew}, spacing={actual_spacing:.2f}m, "
                f"ENU extent=[{x_axis[0]:.1f},{x_axis[-1]:.1f}]x"
                f"[{y_axis[0]:.1f},{y_axis[-1]:.1f}]m")

    return x_axis, y_axis, gray, actual_spacing


def texture_to_rcs(texture_gray: np.ndarray,
                   rcs_min: float = 0.01,
                   rcs_max: float = 5.0,
                   edge_boost: float = 2.0) -> np.ndarray:
    """Convert grayscale texture to RCS values.

    Bright pixels → higher RCS (buildings, concrete).
    Dark pixels → lower RCS (vegetation, shadow).
    Edges → boosted RCS (building edges, structural features).

    Args:
        texture_gray: [Ny, Nx] grayscale [0..1]
        rcs_min: Minimum RCS value.
        rcs_max: Maximum RCS value.
        edge_boost: Multiplier for edge regions.

    Returns:
        rcs_map: [Ny, Nx] RCS values.
    """
    from scipy.ndimage import sobel

    # Base RCS from intensity (linear map)
    rcs = rcs_min + (rcs_max - rcs_min) * texture_gray

    # Edge detection for structural features
    edge_x = sobel(texture_gray, axis=1)
    edge_y = sobel(texture_gray, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_mag /= max(edge_mag.max(), 1e-6)

    # Boost edges
    rcs += edge_boost * edge_mag

    rcs = np.clip(rcs, rcs_min, rcs_max * edge_boost).astype(np.float32)

    logger.info(f"Texture→RCS: min={rcs.min():.3f}, max={rcs.max():.3f}, "
                f"mean={rcs.mean():.3f}")
    return rcs


def texture_to_flat_dem(x_axis: np.ndarray, y_axis: np.ndarray,
                        texture_gray: np.ndarray,
                        base_height: float = 0.0,
                        building_height: float = 30.0,
                        building_threshold: float = 0.65
                        ) -> np.ndarray:
    """Create a pseudo-DEM from texture brightness.

    Bright regions (buildings) → elevated height.
    Dark regions (vegetation/roads) → base height.

    Args:
        texture_gray: [Ny, Nx] grayscale [0..1]
        base_height: Ground level [m].
        building_height: Max building height [m].
        building_threshold: Brightness threshold for building detection.

    Returns:
        dem: [Ny, Nx] height map [m].
    """
    from scipy.ndimage import median_filter, binary_opening

    # Simple building detection: bright + spatially coherent regions
    bright = texture_gray > building_threshold

    # Morphological cleanup to remove noise
    structure = np.ones((5, 5))
    bright_clean = binary_opening(bright, structure=structure)

    # Smooth height transitions
    height_map = bright_clean.astype(np.float64) * building_height + base_height
    height_map = median_filter(height_map, size=3)

    n_building = np.sum(bright_clean)
    total = bright_clean.size
    logger.info(f"Pseudo-DEM: {n_building}/{total} pixels elevated "
                f"({100*n_building/total:.1f}%), max_h={building_height}m")

    return height_map.astype(np.float64)
