"""UF-28: Raw Echo Generation.

Generates range-compressed echo data from scatterer list + platform trajectory.
Implements SR-04.

Optimized with sparse range-bin accumulation instead of dense [N, Nr] matrices.
"""
import numpy as np
from ..data_contracts import Scatterers, PlatformMeta, RadarMeta
import logging

logger = logging.getLogger(__name__)

C = 299792458.0  # speed of light


def generate_raw_echo(scatterers: Scatterers,
                      platform: PlatformMeta,
                      radar: RadarMeta,
                      Nr: int = None,
                      range_compressed: bool = True,
                      sinc_halfwidth: int = 8) -> np.ndarray:
    """Generate raw echo data (range-compressed by default).

    Range-compressed echo:
      s_rc(k, n) = sum_i sigma_i * sinc(B * (tau_n - 2*R_i(k)/c))
                   * exp(-j * 4*pi*fc * R_i(k) / c)

    Optimized: only accumulates sinc contributions within ±sinc_halfwidth
    bins of each scatterer's range bin (sparse accumulation).

    Args:
        scatterers: Scatterer data.
        platform: Platform state vectors.
        radar: Radar parameters.
        Nr: Number of range bins. If None, auto-computed.
        range_compressed: If True, generate range-compressed echo.
        sinc_halfwidth: Half-width of sinc window in range bins.

    Returns:
        raw: [K, Nr] complex64 array.
        tau_axis: [Nr] range time axis.
    """
    K = platform.K
    effective_rcs = scatterers.get_effective_rcs().astype(np.float64)
    scat_pos = scatterers.positions.astype(np.float64)  # [N, 3]
    N = scatterers.N

    logger.info(f"Generating raw echo: K={K} pulses, N={N} scatterers, "
                f"range_compressed={range_compressed}")

    # Compute slant ranges for all pulses: [K, N]
    plat_pos = platform.positions.astype(np.float64)  # [K, 3]

    # Vectorized range computation using broadcasting
    # plat_pos: [K, 1, 3], scat_pos: [1, N, 3] -> diff: [K, N, 3]
    # For large K*N, compute per-pulse to save memory
    R_min_global = np.inf
    R_max_global = -np.inf

    # First pass: find global range extent
    for k in range(K):
        diff = scat_pos - plat_pos[k, :]  # [N, 3]
        R = np.sqrt(np.sum(diff ** 2, axis=1))  # [N]
        R_min_global = min(R_min_global, R.min())
        R_max_global = max(R_max_global, R.max())

    tau_min_global = 2.0 * R_min_global / C
    tau_max_global = 2.0 * R_max_global / C

    # Range resolution
    range_res = C / (2.0 * radar.bandwidth)

    if Nr is None:
        range_extent = R_max_global - R_min_global
        Nr = max(int(range_extent / range_res * 2) + 64, 128)
        Nr = min(Nr, 4096)

    # Range time axis with margin
    tau_axis = np.linspace(tau_min_global - 2 * range_res / C,
                           tau_max_global + 2 * range_res / C, Nr)
    dtau = tau_axis[1] - tau_axis[0] if Nr > 1 else 1.0

    logger.info(f"Range: R=[{R_min_global:.1f}, {R_max_global:.1f}] m, Nr={Nr}, "
                f"range_res={range_res:.3f} m")

    fc = radar.fc
    B = radar.bandwidth
    raw = np.zeros((K, Nr), dtype=np.complex128)

    # Precompute sinc kernel offsets
    hw = sinc_halfwidth
    offsets = np.arange(-hw, hw + 1)  # [-hw, ..., +hw]

    for k in range(K):
        diff = scat_pos - plat_pos[k, :]  # [N, 3]
        R = np.sqrt(np.sum(diff ** 2, axis=1))  # [N]
        tau = 2.0 * R / C

        # Phase term
        phase = -4.0 * np.pi * fc * R / C
        amp = effective_rcs * np.exp(1j * phase)  # [N]

        if range_compressed:
            # Compute fractional range bin index for each scatterer
            idx_float = (tau - tau_axis[0]) / dtau  # [N]
            idx_center = np.round(idx_float).astype(np.int64)  # [N]

            # For each offset in the sinc window
            for off in offsets:
                idx = idx_center + off
                valid = (idx >= 0) & (idx < Nr)
                v = np.where(valid)[0]
                if len(v) == 0:
                    continue

                # Sinc value at this offset
                tau_bin = tau_axis[0] + idx[v] * dtau
                sinc_val = np.sinc(B * (tau_bin - tau[v]))

                # Accumulate using np.add.at for scatter-add
                np.add.at(raw[k, :], idx[v], amp[v] * sinc_val)
        else:
            # Phase-history only
            idx_float = (tau - tau_axis[0]) / dtau
            idx0 = np.floor(idx_float).astype(np.int64)
            frac = idx_float - idx0
            valid = (idx0 >= 0) & (idx0 < Nr - 1)
            v = np.where(valid)[0]
            np.add.at(raw[k, :], idx0[v], amp[v] * (1 - frac[v]))
            np.add.at(raw[k, :], idx0[v] + 1, amp[v] * frac[v])

    raw = raw.astype(np.complex64)

    energy = np.sum(np.abs(raw) ** 2)
    logger.info(f"Raw echo generated: shape={raw.shape}, energy={energy:.2e}")

    if energy < 1e-30:
        logger.warning("Raw echo energy is near zero! Possible signal collapse.")

    return raw, tau_axis
