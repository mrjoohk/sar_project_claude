"""UF-31 Speckle Noise unit tests.

UF31-T01: Single-look speckle statistics (ENL~1)
UF31-T02: ENL adjustment (ENL=4 should yield measured ENL≥3)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_uf31_t01_single_look():
    """UF31-T01: Single-look speckle (ENL=1) intensity CV≈1."""
    from sar_imaging.uf.uf31_speckle import add_speckle

    # Create uniform-intensity complex image
    Ny, Nx = 200, 200
    image = np.ones((Ny, Nx), dtype=np.complex64) * 10.0  # uniform amplitude=10

    speckled = add_speckle(image, enl=1.0, seed=42)

    intensity = np.abs(speckled) ** 2
    mean_i = np.mean(intensity)
    var_i = np.var(intensity)
    cv = np.sqrt(var_i) / mean_i  # coefficient of variation

    # For single-look speckle, CV should be ~1.0
    assert 0.5 < cv < 1.5, f"CV={cv:.3f}, expected ~1.0 for single-look"

    # Measured ENL = mean²/var
    measured_enl = mean_i ** 2 / var_i
    assert 0.5 < measured_enl < 2.0, \
        f"measured_ENL={measured_enl:.2f}, expected ~1.0"

    print(f"  PASS: UF31-T01 CV={cv:.3f}, measured_ENL={measured_enl:.2f}")


def test_uf31_t02_enl_adjustment():
    """UF31-T02: ENL=4 should give smoother speckle (measured ENL≥3)."""
    from sar_imaging.uf.uf31_speckle import add_speckle

    Ny, Nx = 200, 200
    image = np.ones((Ny, Nx), dtype=np.complex64) * 10.0

    # ENL=1
    speckled_1 = add_speckle(image, enl=1.0, seed=42)
    int_1 = np.abs(speckled_1) ** 2
    enl_1 = np.mean(int_1) ** 2 / np.var(int_1)

    # ENL=4
    speckled_4 = add_speckle(image, enl=4.0, seed=42)
    int_4 = np.abs(speckled_4) ** 2
    enl_4 = np.mean(int_4) ** 2 / np.var(int_4)

    assert enl_4 > enl_1, \
        f"ENL=4 ({enl_4:.2f}) not > ENL=1 ({enl_1:.2f})"
    assert enl_4 >= 2.5, \
        f"ENL=4 measured {enl_4:.2f} < 2.5 (too far from target)"

    print(f"  PASS: UF31-T02 ENL=1→{enl_1:.2f}, ENL=4→{enl_4:.2f}")


def test_uf31_reproducibility():
    """Speckle with same seed should produce identical results."""
    from sar_imaging.uf.uf31_speckle import add_speckle

    Ny, Nx = 64, 64
    image = np.ones((Ny, Nx), dtype=np.complex64) * 5.0

    s1 = add_speckle(image, enl=1.0, seed=123)
    s2 = add_speckle(image, enl=1.0, seed=123)

    assert np.allclose(s1, s2), "Same seed produces different results"

    print("  PASS: Speckle reproducibility (same seed = same output)")


if __name__ == '__main__':
    test_uf31_t01_single_look()
    test_uf31_t02_enl_adjustment()
    test_uf31_reproducibility()
    print("All UF-31 tests passed.")
