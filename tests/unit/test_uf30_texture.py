"""UF-30 Texture Loader unit tests.

UF30-T01: TIF→ENU coordinate flip verification
UF30-T02: Texture-to-RCS distribution
UF30-T03: Pseudo-DEM building detection range
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_uf30_t01_coordinate_flip():
    """UF30-T01: Verify bottom-left preservation after vertical flip."""
    from sar_imaging.scene.texture_loader import load_texture_tif

    tif_path = 'dataset/gangseo_crop_500m.tif'
    if not os.path.exists(tif_path):
        print("  SKIP: UF30-T01 (gangseo_crop_500m.tif not found)")
        return

    x_axis, y_axis, texture_gray, spacing = load_texture_tif(tif_path, target_spacing_m=5.0)

    # Basic shape checks
    Ny, Nx = texture_gray.shape
    assert Ny > 0 and Nx > 0, "Empty texture"
    assert len(x_axis) == Nx, "x_axis size mismatch"
    assert len(y_axis) == Ny, "y_axis size mismatch"

    # ENU axes should be centered
    assert x_axis[0] < 0 and x_axis[-1] > 0, "x_axis not centered"
    assert y_axis[0] < 0 and y_axis[-1] > 0, "y_axis not centered"

    # After vertical flip, row 0 = south (y_min)
    # The texture should be valid grayscale
    assert 0.0 <= texture_gray.min(), "texture below 0"
    assert texture_gray.max() <= 1.0, "texture above 1"

    print(f"  PASS: UF30-T01 texture={Nx}x{Ny}, spacing={spacing:.2f}m, "
          f"range=[{texture_gray.min():.3f}, {texture_gray.max():.3f}]")


def test_uf30_t02_rcs_distribution():
    """UF30-T02: RCS distribution from texture."""
    from sar_imaging.scene.texture_loader import texture_to_rcs

    # Create synthetic texture with known properties
    Ny, Nx = 100, 100
    texture = np.zeros((Ny, Nx), dtype=np.float64)
    texture[40:60, 40:60] = 0.9  # bright building block
    texture[0:20, :] = 0.1  # dark vegetation strip

    rcs = texture_to_rcs(texture, rcs_min=0.01, rcs_max=5.0, edge_boost=2.0)

    assert rcs.shape == (Ny, Nx), "RCS shape mismatch"
    assert rcs.min() >= 0.01, f"RCS min {rcs.min()} below rcs_min"
    assert rcs.max() <= 5.0 * 2.0, f"RCS max {rcs.max()} above rcs_max*edge_boost"

    # Building edge pixels should have higher RCS than interior
    building_interior = rcs[45:55, 45:55]
    building_edge = np.concatenate([rcs[40, 40:60], rcs[59, 40:60],
                                     rcs[40:60, 40], rcs[40:60, 59]])
    assert building_edge.mean() > building_interior.mean(), \
        f"Edge RCS ({building_edge.mean():.3f}) not > interior ({building_interior.mean():.3f})"

    print(f"  PASS: UF30-T02 rcs=[{rcs.min():.3f}, {rcs.max():.3f}], "
          f"edge_mean={building_edge.mean():.3f} > interior_mean={building_interior.mean():.3f}")


def test_uf30_t03_pseudo_dem():
    """UF30-T03: Pseudo-DEM building detection."""
    from sar_imaging.scene.texture_loader import texture_to_flat_dem

    Ny, Nx = 100, 100
    x_axis = np.arange(Nx, dtype=np.float64) - Nx / 2
    y_axis = np.arange(Ny, dtype=np.float64) - Ny / 2

    # Texture with 30% bright (buildings)
    texture = np.random.RandomState(42).uniform(0.0, 0.5, (Ny, Nx))
    texture[20:40, 20:40] = 0.8  # building block 1
    texture[60:80, 50:70] = 0.85  # building block 2

    dem = texture_to_flat_dem(x_axis, y_axis, texture,
                              base_height=0.0, building_height=25.0,
                              building_threshold=0.65)

    assert dem.shape == (Ny, Nx), "DEM shape mismatch"

    # Building regions should be elevated
    building_region = dem[25:35, 25:35]
    ground_region = dem[0:10, 0:10]
    assert building_region.mean() > 20.0, \
        f"Building height {building_region.mean():.1f} < 20m"
    assert ground_region.mean() < 5.0, \
        f"Ground height {ground_region.mean():.1f} > 5m"

    # Elevated pixel fraction should be reasonable (10-50%)
    elevated = np.sum(dem > 10.0)
    frac = elevated / dem.size
    assert 0.05 < frac < 0.60, f"Elevated fraction {frac:.2f} out of range"

    print(f"  PASS: UF30-T03 building_h={building_region.mean():.1f}m, "
          f"ground_h={ground_region.mean():.1f}m, elevated={100*frac:.1f}%")


if __name__ == '__main__':
    test_uf30_t01_coordinate_flip()
    test_uf30_t02_rcs_distribution()
    test_uf30_t03_pseudo_dem()
    print("All UF-30 tests passed.")
