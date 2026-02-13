"""Unit tests for UF-20 Shadow Masking.

Tests: UF20-T01 (single wall shadow), UF20-T02 (multi-wall shadow comparison).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.scene.synthetic_dem import make_wall_dem
from sar_imaging.scene.scatterers import centroid_scatterers
from sar_imaging.uf.uf20_shadow import compute_shadow_mask


def test_uf20_t01_single_wall_shadow():
    """UF20-T01: Wall should create shadow behind it (far-range side)."""
    x, y, dem = make_wall_dem(size=(128, 128), spacing=1.0,
                               wall_height=30.0, wall_x=0.0, wall_thickness=3.0)
    scat = centroid_scatterers(x, y, dem)

    # Platform at -5000, 0, 5000 (side-looking from negative x)
    platform_pos = np.array([-5000.0, 0.0, 5000.0])
    vis_mask = compute_shadow_mask(scat, platform_pos, x, y, dem)

    # "Behind" the wall = positive x side, close to wall (within shadow length)
    # Shadow length ≈ wall_height / tan(elevation) = 30/1 = 30m at 45° elevation
    # Check region right behind wall (x: 5 to 25m) which should be fully shadowed
    behind_mask = (scat.sx > 5.0) & (scat.sx < 25.0) & (scat.sz < 1.0)
    front_mask = (scat.sx < -5.0) & (scat.sz < 1.0)  # ground in front of wall

    mean_vis_behind = np.mean(vis_mask[behind_mask]) if np.any(behind_mask) else 1.0
    mean_vis_front = np.mean(vis_mask[front_mask]) if np.any(front_mask) else 0.0

    assert mean_vis_behind <= 0.3, f"Behind-wall vis_mask mean {mean_vis_behind:.3f} > 0.3"
    assert mean_vis_front >= 0.9, f"Front-of-wall vis_mask mean {mean_vis_front:.3f} < 0.9"
    print(f"  PASS: UF20-T01 vis_behind={mean_vis_behind:.3f}, vis_front={mean_vis_front:.3f}")


def test_uf20_t02_multi_wall_shadow():
    """UF20-T02: Taller wall should cast longer shadow."""
    platform_pos = np.array([-5000.0, 0.0, 5000.0])

    # Short wall (15m)
    x1, y1, dem1 = make_wall_dem(size=(128, 64), spacing=1.0,
                                  wall_height=15.0, wall_x=-20.0)
    scat1 = centroid_scatterers(x1, y1, dem1)
    vis1 = compute_shadow_mask(scat1, platform_pos, x1, y1, dem1)
    n_shadow_short = np.sum(vis1 < 0.5)

    # Tall wall (30m)
    x2, y2, dem2 = make_wall_dem(size=(128, 64), spacing=1.0,
                                  wall_height=30.0, wall_x=-20.0)
    scat2 = centroid_scatterers(x2, y2, dem2)
    vis2 = compute_shadow_mask(scat2, platform_pos, x2, y2, dem2)
    n_shadow_tall = np.sum(vis2 < 0.5)

    ratio = n_shadow_tall / max(n_shadow_short, 1)
    assert ratio >= 1.5, (
        f"Tall wall shadow ({n_shadow_tall}) / short wall shadow ({n_shadow_short}) "
        f"= {ratio:.2f} < 1.5"
    )
    print(f"  PASS: UF20-T02 short={n_shadow_short}, tall={n_shadow_tall}, ratio={ratio:.2f}")


if __name__ == '__main__':
    test_uf20_t01_single_wall_shadow()
    test_uf20_t02_multi_wall_shadow()
    print("All UF-20 tests passed.")
