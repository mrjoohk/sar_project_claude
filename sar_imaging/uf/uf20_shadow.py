"""UF-20: Shadow Masking (LOS approximation).

Implements height-profile test for shadow detection.
Uses ground-projected LOS sampling for correct near-field behavior.
"""
import numpy as np
from ..data_contracts import Scatterers
import logging

logger = logging.getLogger(__name__)


def compute_shadow_mask(scatterers: Scatterers,
                        platform_pos: np.ndarray,
                        dem_x: np.ndarray,
                        dem_y: np.ndarray,
                        dem: np.ndarray,
                        num_steps: int = 100) -> np.ndarray:
    """Compute shadow visibility mask using ground-projected height-profile test.

    Algorithm (improved):
    For each scatterer, sample points along the ground-projected LOS direction
    (from scatterer toward radar), stepping by DEM grid spacing. At each step,
    compute the expected LOS height using the elevation angle from scatterer
    to platform. If the DEM height exceeds the LOS height, the scatterer
    is in shadow.

    This correctly handles cases where the platform is far away but
    obstructions are close to the scatterer.

    Args:
        scatterers: Scatterer data.
        platform_pos: [3] representative platform position (ENU).
        dem_x: [Nx] x-axis of DEM.
        dem_y: [Ny] y-axis of DEM.
        dem: [Ny, Nx] DEM heights.
        num_steps: Max number of DEM cells to check along the LOS.

    Returns:
        vis_mask: [N] array, 1.0 = visible, 0.0 = shadowed.
    """
    N = scatterers.N
    vis_mask = np.ones(N, dtype=np.float32)
    platform_pos = np.asarray(platform_pos, dtype=np.float64)

    scat_pos = np.column_stack([
        scatterers.sx.astype(np.float64),
        scatterers.sy.astype(np.float64),
        scatterers.sz.astype(np.float64),
    ])  # [N, 3]

    # DEM grid bounds and spacing
    x_min, x_max = dem_x[0], dem_x[-1]
    y_min, y_max = dem_y[0], dem_y[-1]
    dx = dem_x[1] - dem_x[0] if len(dem_x) > 1 else 1.0
    dy = dem_y[1] - dem_y[0] if len(dem_y) > 1 else 1.0
    step_size = min(abs(dx), abs(dy))

    # Direction from scatterer to platform (3D)
    dir_to_plat = platform_pos[np.newaxis, :] - scat_pos  # [N, 3]

    # Ground-projected direction (2D, xy only)
    dir_2d = dir_to_plat[:, :2]  # [N, 2]
    dist_2d = np.linalg.norm(dir_2d, axis=1, keepdims=True)
    dist_2d = np.maximum(dist_2d, 1e-6)
    dir_2d_unit = dir_2d / dist_2d  # [N, 2] unit vector toward radar on ground

    # Elevation angle from each scatterer toward platform
    # tan(el) = (plat_z - scat_z) / ground_distance
    dz = platform_pos[2] - scat_pos[:, 2]  # [N]
    tan_el = dz / dist_2d.ravel()  # [N]

    # Maximum ground distance to check (limited by DEM extent)
    dem_diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    max_ground_dist = min(dem_diag, dist_2d.max())

    # Determine actual number of steps
    actual_steps = min(num_steps, int(max_ground_dist / step_size) + 1)
    actual_steps = max(actual_steps, 10)

    # Sample along the ground path from each scatterer toward radar
    for step_i in range(1, actual_steps):
        ground_dist = step_i * step_size

        # Ground position along LOS toward radar
        sample_xy = scat_pos[:, :2] + dir_2d_unit * ground_dist  # [N, 2]
        sample_x = sample_xy[:, 0]
        sample_y = sample_xy[:, 1]

        # Expected LOS height at this ground position
        # z_los = scat_z + ground_dist * tan(elevation_angle)
        z_los = scat_pos[:, 2] + ground_dist * tan_el  # [N]

        # Check if sample is within DEM bounds
        in_bounds = ((sample_x >= x_min) & (sample_x <= x_max) &
                     (sample_y >= y_min) & (sample_y <= y_max))

        if not np.any(in_bounds):
            continue

        # Convert to DEM grid indices (only for in-bounds points)
        ix = np.clip(((sample_x - x_min) / dx).astype(int), 0, len(dem_x) - 1)
        iy = np.clip(((sample_y - y_min) / dy).astype(int), 0, len(dem_y) - 1)

        # DEM height at sampled positions
        dem_height = dem[iy, ix]

        # If DEM height exceeds LOS height AND point is in bounds, scatterer is blocked
        # Also exclude self-obstruction (where the scatterer is ON the elevated surface)
        blocked = in_bounds & (dem_height > z_los + 0.5)
        vis_mask[blocked] = 0.0

    n_shadowed = np.sum(vis_mask < 0.5)
    logger.info(f"Shadow mask: {n_shadowed}/{N} scatterers shadowed "
                f"({100*n_shadowed/max(N,1):.1f}%)")
    return vis_mask


def apply_shadow_mask(scatterers: Scatterers, vis_mask: np.ndarray) -> Scatterers:
    """Apply shadow mask to scatterers (modifies vis_mask field)."""
    scatterers.vis_mask = vis_mask
    return scatterers
