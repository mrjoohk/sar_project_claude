"""UF-21: Layover Approximation.

Computes layover flags based on local slope vs incidence angle.
"""
import numpy as np
from ..data_contracts import Scatterers
import logging

logger = logging.getLogger(__name__)


def compute_layover_flag(scatterers: Scatterers,
                         platform_pos: np.ndarray) -> np.ndarray:
    """Compute layover flags for scatterers.

    Layover condition: local slope angle > incidence angle
    (surface tilted toward radar more than the incidence angle).

    Args:
        scatterers: Must have normals (snx, sny, snz).
        platform_pos: [3] representative platform position (ENU).

    Returns:
        layover_flag: [N] array, 1.0 = layover, 0.0 = normal.
    """
    N = scatterers.N
    if scatterers.snx is None:
        logger.warning("No normals available, skipping layover computation.")
        return np.zeros(N, dtype=np.float32)

    platform_pos = np.asarray(platform_pos, dtype=np.float64)

    normals = np.column_stack([
        scatterers.snx.astype(np.float64),
        scatterers.sny.astype(np.float64),
        scatterers.snz.astype(np.float64),
    ])  # [N, 3]

    scat_pos = np.column_stack([
        scatterers.sx.astype(np.float64),
        scatterers.sy.astype(np.float64),
        scatterers.sz.astype(np.float64),
    ])  # [N, 3]

    # Incidence vector (scatterer to platform)
    inc_vec = platform_pos[np.newaxis, :] - scat_pos
    inc_dist = np.linalg.norm(inc_vec, axis=1, keepdims=True)
    inc_dist = np.maximum(inc_dist, 1e-6)
    inc_hat = inc_vec / inc_dist

    # Incidence angle: angle between normal and incidence direction
    cos_inc = np.sum(normals * inc_hat, axis=1)
    cos_inc = np.clip(cos_inc, -1.0, 1.0)
    incidence_angle = np.arccos(np.abs(cos_inc))  # [0, pi/2]

    # Slope angle: angle between normal and vertical (z-axis)
    cos_slope = np.abs(normals[:, 2])
    cos_slope = np.clip(cos_slope, -1.0, 1.0)
    slope_angle = np.arccos(cos_slope)  # 0 = flat, pi/2 = vertical

    # Layover: slope_angle > incidence_angle
    layover_flag = (slope_angle > incidence_angle).astype(np.float32)

    n_layover = np.sum(layover_flag > 0.5)
    logger.info(f"Layover: {n_layover}/{N} scatterers flagged "
                f"({100*n_layover/max(N,1):.1f}%)")

    return layover_flag


def apply_layover_flag(scatterers: Scatterers,
                       layover_flag: np.ndarray) -> Scatterers:
    """Attach layover flag to scatterers."""
    scatterers.layover_flag = layover_flag
    return scatterers
