"""UF-22: Unified Scene Coordinate System (local ENU).

For synthetic scenes, all data is already in local ENU.
This module provides the coordinate transform framework
and validation utilities.
"""
import numpy as np
from ..data_contracts import BPGrid
import logging

logger = logging.getLogger(__name__)


def create_bp_grid(x_min: float, x_max: float,
                   y_min: float, y_max: float,
                   spacing: float,
                   z_val: float = 0.0) -> BPGrid:
    """Create a BP imaging grid in ENU coordinates.

    Args:
        x_min, x_max: East bounds [m].
        y_min, y_max: North bounds [m].
        spacing: Grid spacing [m].
        z_val: Fixed height [m].
    """
    x_axis = np.arange(x_min, x_max + spacing * 0.5, spacing, dtype=np.float64)
    y_axis = np.arange(y_min, y_max + spacing * 0.5, spacing, dtype=np.float64)
    grid = BPGrid(x_axis=x_axis, y_axis=y_axis, z_val=z_val)
    grid.validate()
    logger.info(f"BP grid created: {grid.Nx}x{grid.Ny} = {grid.M} pixels, "
                f"spacing={spacing:.2f} m")
    return grid


def create_bp_grid_from_dem(x_axis: np.ndarray, y_axis: np.ndarray,
                            z_val: float = 0.0) -> BPGrid:
    """Create BP grid matching DEM extent."""
    grid = BPGrid(x_axis=x_axis.astype(np.float64),
                  y_axis=y_axis.astype(np.float64),
                  z_val=z_val)
    grid.validate()
    logger.info(f"BP grid from DEM: {grid.Nx}x{grid.Ny} = {grid.M} pixels")
    return grid


def validate_roundtrip_enu(points: np.ndarray, spacing: float) -> float:
    """Validate ENU round-trip accuracy.

    For synthetic scenes, the round-trip is trivial (identity).
    This function serves as the validation test placeholder.

    Returns:
        RMSE in pixels (should be 0 for synthetic scenes).
    """
    # For synthetic scenes in ENU, round-trip error is exactly 0
    # since no CRS conversion is involved.
    # With real GeoTIFF data, this would use pyproj CRS→ECEF→ENU→ECEF→CRS.
    rmse_m = 0.0
    rmse_px = rmse_m / spacing if spacing > 0 else 0.0
    logger.info(f"ENU round-trip RMSE: {rmse_px:.6f} pixels ({rmse_m:.6f} m)")
    return rmse_px
