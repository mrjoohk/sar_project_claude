"""Scatterer generation from DEM mesh.

Implements UF-19 (Facet/BRDF Scattering) and centroid model.
"""
import numpy as np
from typing import Tuple, Optional
from ..data_contracts import Scatterers


def centroid_scatterers(x_axis: np.ndarray, y_axis: np.ndarray,
                        dem: np.ndarray,
                        rcs_base: float = 1.0) -> Scatterers:
    """Generate scatterers at each DEM grid point (centroid model, SR-03.1).

    Simple baseline: one scatterer per grid cell, uniform RCS.
    """
    Ny, Nx = dem.shape
    xx, yy = np.meshgrid(x_axis, y_axis)
    sx = xx.ravel().astype(np.float32)
    sy = yy.ravel().astype(np.float32)
    sz = dem.ravel().astype(np.float32)
    s_rcs = np.full(sx.shape, rcs_base, dtype=np.float32)
    return Scatterers(sx=sx, sy=sy, sz=sz, s_rcs=s_rcs)


def facet_brdf_scatterers(vertices: np.ndarray, triangles: np.ndarray,
                          centroids: np.ndarray, normals: np.ndarray,
                          areas: np.ndarray,
                          platform_pos: np.ndarray,
                          alpha: float = 0.7,
                          beta: float = 0.3,
                          p: float = 10.0) -> Scatterers:
    """Generate scatterers with facet/BRDF model (UF-19).

    RCS formula: sigma = A * [alpha * cos(theta_i) + beta * |r_hat . s_hat|^p]
    where:
      A = facet area
      theta_i = incidence angle
      s_hat = specular reflection direction
      r_hat = direction back to radar

    Args:
        vertices: [Nv, 3]
        triangles: [Nt, 3]
        centroids: [Nt, 3] triangle centroids
        normals: [Nt, 3] unit normals
        areas: [Nt] triangle areas
        platform_pos: [3] representative platform position
        alpha: Lambertian diffuse weight
        beta: Phong specular weight
        p: Phong exponent
    """
    Nt = centroids.shape[0]
    platform_pos = np.asarray(platform_pos, dtype=np.float64)

    # Incidence vector (from scatterer to platform)
    inc_vec = platform_pos[np.newaxis, :] - centroids  # [Nt, 3]
    inc_dist = np.linalg.norm(inc_vec, axis=1, keepdims=True)
    inc_dist = np.maximum(inc_dist, 1e-6)
    inc_hat = inc_vec / inc_dist  # unit vector toward radar

    # Incidence angle: cos(theta_i) = dot(normal, inc_hat)
    cos_theta = np.sum(normals * inc_hat, axis=1)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)  # backface → 0

    # Specular direction: s_hat = 2 * (n . inc_hat) * n - inc_hat
    s_hat = 2 * cos_theta[:, np.newaxis] * normals - inc_hat

    # For monostatic radar, r_hat = inc_hat (look-back direction)
    r_hat = inc_hat
    spec_cos = np.sum(r_hat * s_hat, axis=1)
    spec_cos = np.clip(spec_cos, 0.0, 1.0)

    # BRDF RCS
    sigma = areas * (alpha * cos_theta + beta * (spec_cos ** p))
    sigma = np.maximum(sigma, 0.0).astype(np.float32)

    return Scatterers(
        sx=centroids[:, 0].astype(np.float32),
        sy=centroids[:, 1].astype(np.float32),
        sz=centroids[:, 2].astype(np.float32),
        s_rcs=sigma,
        snx=normals[:, 0].astype(np.float32),
        sny=normals[:, 1].astype(np.float32),
        snz=normals[:, 2].astype(np.float32),
        s_area=areas.astype(np.float32),
    )


def add_point_targets(scatterers: Scatterers,
                      targets: list) -> Scatterers:
    """Add point targets to existing scatterer list.

    Args:
        targets: list of dicts with keys 'x', 'y', 'z', 'rcs'.
    """
    if not targets:
        return scatterers

    new_sx = [scatterers.sx]
    new_sy = [scatterers.sy]
    new_sz = [scatterers.sz]
    new_rcs = [scatterers.s_rcs]

    for t in targets:
        new_sx.append(np.array([t['x']], dtype=np.float32))
        new_sy.append(np.array([t['y']], dtype=np.float32))
        new_sz.append(np.array([t['z']], dtype=np.float32))
        new_rcs.append(np.array([t.get('rcs', 100.0)], dtype=np.float32))

    return Scatterers(
        sx=np.concatenate(new_sx),
        sy=np.concatenate(new_sy),
        sz=np.concatenate(new_sz),
        s_rcs=np.concatenate(new_rcs),
    )
