"""Unit tests for UF-27 Streaming BP.

Tests: UF27-T01 (tiling equivalence), UF27-T03 (linear scaling).
"""
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.scene.synthetic_dem import make_flat_dem
from sar_imaging.scene.scatterers import centroid_scatterers, add_point_targets
from sar_imaging.scene.platform import generate_stripmap_trajectory
from sar_imaging.data_contracts import RadarMeta
from sar_imaging.uf.uf22_enu import create_bp_grid_from_dem
from sar_imaging.uf.uf28_raw_echo import generate_raw_echo
from sar_imaging.uf.uf27_streaming_bp import backproject_full, backproject_tiled


def _setup_point_target_scene(grid_size=64, num_pulses=128):
    """Helper: create a simple point target scene."""
    x, y, dem = make_flat_dem(size=(grid_size, grid_size), spacing=1.0)
    scat = centroid_scatterers(x, y, dem, rcs_base=0.01)
    scat = add_point_targets(scat, [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'rcs': 1000.0}])

    radar = RadarMeta(fc=9.65e9, bandwidth=300e6, prf=500.0, tp=10e-6)
    platform = generate_stripmap_trajectory(
        altitude=5000.0, velocity=100.0, look_angle_deg=45.0,
        num_pulses=num_pulses, prf=radar.prf)

    raw, tau_axis = generate_raw_echo(scat, platform, radar)
    bp_grid = create_bp_grid_from_dem(x, y)
    return raw, tau_axis, platform, radar, bp_grid


def test_uf27_t01_tiling_equivalence():
    """UF27-T01: Tiled BP should match full BP (peak-normalized RMS < 1e-3)."""
    raw, tau_axis, platform, radar, bp_grid = _setup_point_target_scene(
        grid_size=64, num_pulses=128)

    # Full BP
    img_full = backproject_full(raw, tau_axis, platform, radar, bp_grid)

    # Tiled BP
    img_tiled = backproject_tiled(raw, tau_axis, platform, radar, bp_grid,
                                   tile_size=(32, 32), k_batch=64, overlap=8)

    # Peak-normalized RMS difference
    peak = np.max(np.abs(img_full))
    if peak < 1e-20:
        raise ValueError("Full BP peak is near zero")

    diff = np.abs(img_full - img_tiled)
    rms = np.sqrt(np.mean(diff ** 2)) / peak
    assert rms < 1e-3, f"Tiling RMS diff = {rms:.6f} >= 1e-3"
    print(f"  PASS: UF27-T01 peak-normalized RMS = {rms:.2e}")


def test_uf27_t03_linear_scaling():
    """UF27-T03: Runtime should scale linearly with pixel count (±20%)."""
    radar = RadarMeta(fc=9.65e9, bandwidth=300e6, prf=500.0, tp=10e-6)
    K = 64
    runtimes = {}

    for grid_size in [32, 64]:
        x, y, dem = make_flat_dem(size=(grid_size, grid_size), spacing=1.0)
        scat = centroid_scatterers(x, y, dem, rcs_base=0.01)
        scat = add_point_targets(scat, [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'rcs': 100.0}])

        platform = generate_stripmap_trajectory(
            altitude=5000.0, velocity=100.0, look_angle_deg=45.0,
            num_pulses=K, prf=radar.prf)
        raw, tau_axis = generate_raw_echo(scat, platform, radar)
        bp_grid = create_bp_grid_from_dem(x, y)

        t0 = time.time()
        _ = backproject_full(raw, tau_axis, platform, radar, bp_grid)
        runtimes[grid_size] = time.time() - t0

    # Check scaling: 64^2/32^2 = 4x pixels, so runtime ratio should be ~4x (±20%)
    ratio = runtimes[64] / max(runtimes[32], 1e-6)
    expected_ratio = (64 / 32) ** 2  # = 4
    deviation = abs(ratio - expected_ratio) / expected_ratio

    assert deviation <= 0.3, (
        f"Scaling deviation {deviation*100:.0f}% > 30%: "
        f"32²={runtimes[32]:.3f}s, 64²={runtimes[64]:.3f}s, ratio={ratio:.2f}"
    )
    print(f"  PASS: UF27-T03 32²={runtimes[32]:.3f}s, 64²={runtimes[64]:.3f}s, "
          f"ratio={ratio:.2f} (expected ~{expected_ratio:.0f})")


if __name__ == '__main__':
    test_uf27_t01_tiling_equivalence()
    test_uf27_t03_linear_scaling()
    print("All UF-27 tests passed.")
