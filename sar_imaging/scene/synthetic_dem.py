"""Synthetic DEM generation for controlled test scenes.

Supports: flat, slope, box (urban block), wall, corner.
Fulfills SR-01.3.
"""
import numpy as np
from typing import Tuple, List, Optional


def make_flat_dem(size: Tuple[int, int] = (512, 512),
                  spacing: float = 1.0,
                  height: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat DEM.

    Returns:
        x_axis: [Nx] array of x coords [m]
        y_axis: [Ny] array of y coords [m]
        dem: [Ny, Nx] height array [m]
    """
    Nx, Ny = size
    x_axis = np.arange(Nx, dtype=np.float64) * spacing - (Nx * spacing / 2)
    y_axis = np.arange(Ny, dtype=np.float64) * spacing - (Ny * spacing / 2)
    dem = np.full((Ny, Nx), height, dtype=np.float64)
    return x_axis, y_axis, dem


def make_slope_dem(size: Tuple[int, int] = (512, 512),
                   spacing: float = 1.0,
                   slope_deg: float = 15.0,
                   direction: str = 'x') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a sloped DEM.

    Args:
        slope_deg: Slope angle in degrees.
        direction: 'x' or 'y' — direction of the slope.
    """
    Nx, Ny = size
    x_axis = np.arange(Nx, dtype=np.float64) * spacing - (Nx * spacing / 2)
    y_axis = np.arange(Ny, dtype=np.float64) * spacing - (Ny * spacing / 2)
    xx, yy = np.meshgrid(x_axis, y_axis)

    slope_tan = np.tan(np.radians(slope_deg))
    if direction == 'x':
        dem = slope_tan * (xx - xx.min())
    else:
        dem = slope_tan * (yy - yy.min())
    return x_axis, y_axis, dem


def make_box_dem(size: Tuple[int, int] = (512, 512),
                 spacing: float = 1.0,
                 box_height: float = 30.0,
                 box_width: float = 50.0,
                 box_center: Optional[Tuple[float, float]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat DEM with a raised box (urban building block).

    Args:
        box_height: Height of the box [m].
        box_width: Width/length of the box [m].
        box_center: (cx, cy) in meters. None = scene center.
    """
    Nx, Ny = size
    x_axis = np.arange(Nx, dtype=np.float64) * spacing - (Nx * spacing / 2)
    y_axis = np.arange(Ny, dtype=np.float64) * spacing - (Ny * spacing / 2)
    dem = np.zeros((Ny, Nx), dtype=np.float64)

    if box_center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = box_center

    hw = box_width / 2
    xx, yy = np.meshgrid(x_axis, y_axis)
    mask = (np.abs(xx - cx) <= hw) & (np.abs(yy - cy) <= hw)
    dem[mask] = box_height

    return x_axis, y_axis, dem


def make_wall_dem(size: Tuple[int, int] = (512, 512),
                  spacing: float = 1.0,
                  wall_height: float = 30.0,
                  wall_x: float = 0.0,
                  wall_thickness: float = 5.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat DEM with a wall (step function in x).

    Useful for shadow testing (UF20-T01).
    """
    Nx, Ny = size
    x_axis = np.arange(Nx, dtype=np.float64) * spacing - (Nx * spacing / 2)
    y_axis = np.arange(Ny, dtype=np.float64) * spacing - (Ny * spacing / 2)
    dem = np.zeros((Ny, Nx), dtype=np.float64)

    xx, _ = np.meshgrid(x_axis, y_axis)
    mask = (np.abs(xx - wall_x) <= wall_thickness / 2)
    dem[mask] = wall_height

    return x_axis, y_axis, dem


def make_corner_dem(size: Tuple[int, int] = (512, 512),
                    spacing: float = 1.0,
                    wall_height: float = 30.0,
                    wall_thickness: float = 5.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an L-shaped corner reflector (two perpendicular walls + ground).

    Useful for 2-bounce testing (EXP-02).
    """
    Nx, Ny = size
    x_axis = np.arange(Nx, dtype=np.float64) * spacing - (Nx * spacing / 2)
    y_axis = np.arange(Ny, dtype=np.float64) * spacing - (Ny * spacing / 2)
    dem = np.zeros((Ny, Nx), dtype=np.float64)

    xx, yy = np.meshgrid(x_axis, y_axis)
    # Wall 1: along y-axis at x=0
    mask1 = (np.abs(xx) <= wall_thickness / 2) & (yy >= 0)
    # Wall 2: along x-axis at y=0
    mask2 = (np.abs(yy) <= wall_thickness / 2) & (xx >= 0)
    dem[mask1 | mask2] = wall_height

    return x_axis, y_axis, dem


def dem_to_triangles(x_axis: np.ndarray, y_axis: np.ndarray,
                     dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert DEM grid to triangle mesh.

    Each grid cell is split into 2 triangles.

    Returns:
        vertices: [Nv, 3] float64
        triangles: [Nt, 3] int — indices into vertices
        centroids: [Nt, 3] float64 — triangle centroids
    """
    Ny, Nx = dem.shape
    xx, yy = np.meshgrid(x_axis, y_axis)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), dem.ravel()])

    # Index mapping: (iy, ix) -> iy * Nx + ix
    tri_list = []
    for iy in range(Ny - 1):
        for ix in range(Nx - 1):
            i00 = iy * Nx + ix
            i10 = iy * Nx + (ix + 1)
            i01 = (iy + 1) * Nx + ix
            i11 = (iy + 1) * Nx + (ix + 1)
            tri_list.append([i00, i10, i01])
            tri_list.append([i10, i11, i01])

    triangles = np.array(tri_list, dtype=np.int32)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    return vertices, triangles, centroids


def compute_triangle_normals_and_areas(
        vertices: np.ndarray,
        triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute outward normals and areas for each triangle.

    Returns:
        normals: [Nt, 3] unit normals
        areas: [Nt] triangle areas
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid div by zero
    normals = cross / norms
    areas = (norms.ravel() / 2.0).astype(np.float32)

    # Ensure normals point "up" (positive z)
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    return normals.astype(np.float32), areas
