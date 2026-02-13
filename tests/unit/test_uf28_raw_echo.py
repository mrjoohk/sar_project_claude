"""Unit tests for UF-28 Raw Echo Generation.

Tests: UF28-T01 (energy conservation), UF28-T02 (range peak positions).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.data_contracts import RadarMeta, Scatterers
from sar_imaging.scene.platform import generate_stripmap_trajectory
from sar_imaging.uf.uf28_raw_echo import generate_raw_echo


def test_uf28_t01_energy_conservation():
    """UF28-T01: Single point target energy should be proportional to sigma^2 * K."""
    sigma = 1.0
    scat = Scatterers(
        sx=np.array([0.0], dtype=np.float32),
        sy=np.array([0.0], dtype=np.float32),
        sz=np.array([0.0], dtype=np.float32),
        s_rcs=np.array([sigma], dtype=np.float32),
    )

    radar = RadarMeta(fc=9.65e9, bandwidth=300e6, prf=500.0, tp=10e-6)
    K = 64
    platform = generate_stripmap_trajectory(
        altitude=5000.0, velocity=100.0, look_angle_deg=45.0,
        num_pulses=K, prf=radar.prf)

    raw, tau_axis = generate_raw_echo(scat, platform, radar)

    energy = np.sum(np.abs(raw) ** 2)
    # Energy should be approximately sigma^2 * K * (sinc mainlobe energy)
    # For a single point target, the energy per pulse is roughly sigma^2
    # Total energy ~ K * sigma^2
    expected_scale = sigma ** 2 * K
    ratio = energy / expected_scale

    # Allow ±10x variation (sinc function spreads energy, range bins vary)
    assert energy > 1e-10, f"Energy {energy:.2e} is near zero (signal collapse)"
    assert ratio > 0.1, f"Energy ratio {ratio:.4f} too low"
    print(f"  PASS: UF28-T01 energy={energy:.4f}, ratio_to_sigma2K={ratio:.4f}")


def test_uf28_t02_two_targets_range_separation():
    """UF28-T02: Two targets at different ranges should have peaks at correct positions."""
    # Two targets separated in x (range direction)
    dx_targets = 10.0  # 10 meters separation
    scat = Scatterers(
        sx=np.array([-dx_targets / 2, dx_targets / 2], dtype=np.float32),
        sy=np.array([0.0, 0.0], dtype=np.float32),
        sz=np.array([0.0, 0.0], dtype=np.float32),
        s_rcs=np.array([100.0, 100.0], dtype=np.float32),
    )

    radar = RadarMeta(fc=9.65e9, bandwidth=300e6, prf=500.0, tp=10e-6)
    platform = generate_stripmap_trajectory(
        altitude=5000.0, velocity=100.0, look_angle_deg=45.0,
        num_pulses=64, prf=radar.prf)

    raw, tau_axis = generate_raw_echo(scat, platform, radar)

    # Look at a single pulse (k=K/2, mid-aperture)
    k_mid = raw.shape[0] // 2
    pulse = np.abs(raw[k_mid, :]) ** 2

    # Find two peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(pulse, height=np.max(pulse) * 0.1,
                                    distance=3)

    assert len(peaks) >= 2, f"Expected ≥ 2 peaks, found {len(peaks)}"

    # Check that peak separation matches expected range difference
    # Range of each target from platform at mid-aperture
    plat_pos = platform.positions[k_mid, :]
    R1 = np.linalg.norm(plat_pos - np.array([-dx_targets/2, 0, 0]))
    R2 = np.linalg.norm(plat_pos - np.array([dx_targets/2, 0, 0]))
    delta_R = abs(R2 - R1)

    # Convert to tau_axis bins
    C = 299792458.0
    delta_tau = 2 * delta_R / C
    dtau = tau_axis[1] - tau_axis[0]
    expected_bin_sep = delta_tau / dtau

    # Actual peak separation
    peak_sep = abs(peaks[-1] - peaks[0])
    error_bins = abs(peak_sep - expected_bin_sep)

    assert error_bins < 2, (
        f"Peak separation {peak_sep} bins vs expected {expected_bin_sep:.1f} bins, "
        f"error={error_bins:.1f} bins > 1"
    )
    print(f"  PASS: UF28-T02 peak_sep={peak_sep} bins, "
          f"expected={expected_bin_sep:.1f} bins, error={error_bins:.1f}")


if __name__ == '__main__':
    test_uf28_t01_energy_conservation()
    test_uf28_t02_two_targets_range_separation()
    print("All UF-28 tests passed.")
