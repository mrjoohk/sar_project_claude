"""Unit tests for UF-19 Facet/BRDF Scattering.

Tests: UF19-T01 (flat ground BRDF sanity), UF19-T02 (slope incidence dependence).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.scene.synthetic_dem import (
    make_flat_dem, make_slope_dem, dem_to_triangles,
    compute_triangle_normals_and_areas,
)
from sar_imaging.scene.scatterers import facet_brdf_scatterers


def test_uf19_t01_flat_ground_brdf_sanity():
    """UF19-T01: Flat ground should have near-vertical normals and uniform RCS."""
    x, y, dem = make_flat_dem(size=(64, 64), spacing=1.0, height=0.0)
    verts, tris, cents = dem_to_triangles(x, y, dem)
    normals, areas = compute_triangle_normals_and_areas(verts, tris)

    # Check normals are nearly vertical
    mean_nz = np.mean(np.abs(normals[:, 2]))
    assert mean_nz > 0.99, f"Normal z-component mean {mean_nz} < 0.99 (not flat)"

    # Generate scatterers with BRDF
    platform_pos = np.array([-5000.0, 0.0, 5000.0])
    scat = facet_brdf_scatterers(verts, tris, cents, normals, areas,
                                  platform_pos=platform_pos)
    scat.validate()

    # Check RCS uniformity: CV < 0.1
    rcs = scat.s_rcs
    rcs_nonzero = rcs[rcs > 0]
    cv = np.std(rcs_nonzero) / np.mean(rcs_nonzero)
    assert cv < 0.1, f"RCS CV={cv:.4f} >= 0.1 (not uniform for flat ground)"
    print(f"  PASS: UF19-T01 nz_mean={mean_nz:.6f}, rcs_cv={cv:.4f}")


def test_uf19_t02_slope_incidence_dependence():
    """UF19-T02: Mean RCS should decrease monotonically with incidence angle."""
    platform_altitude = 5000.0
    look_angles = [20, 30, 40, 50, 60]  # degrees from nadir
    mean_rcs_list = []

    for look_deg in look_angles:
        ground_range = platform_altitude * np.tan(np.radians(look_deg))
        platform_pos = np.array([-ground_range, 0.0, platform_altitude])

        x, y, dem = make_flat_dem(size=(32, 32), spacing=1.0)
        verts, tris, cents = dem_to_triangles(x, y, dem)
        normals, areas = compute_triangle_normals_and_areas(verts, tris)

        scat = facet_brdf_scatterers(verts, tris, cents, normals, areas,
                                      platform_pos=platform_pos)
        mean_rcs = np.mean(scat.s_rcs[scat.s_rcs > 0])
        mean_rcs_list.append(mean_rcs)

    # Check monotonic decrease
    mean_rcs_db = [10 * np.log10(r) for r in mean_rcs_list]
    for i in range(len(mean_rcs_db) - 1):
        diff = mean_rcs_db[i] - mean_rcs_db[i + 1]
        assert diff > 0.3, (
            f"RCS did not decrease by > 0.3 dB between "
            f"{look_angles[i]}° ({mean_rcs_db[i]:.1f} dB) and "
            f"{look_angles[i+1]}° ({mean_rcs_db[i+1]:.1f} dB), diff={diff:.2f} dB"
        )

    print(f"  PASS: UF19-T02 RCS(dB) by look angle: "
          f"{list(zip(look_angles, [f'{r:.1f}' for r in mean_rcs_db]))}")


if __name__ == '__main__':
    test_uf19_t01_flat_ground_brdf_sanity()
    test_uf19_t02_slope_incidence_dependence()
    print("All UF-19 tests passed.")
