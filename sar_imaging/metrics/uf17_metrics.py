"""UF-17 / UF-26: Quality Metrics.

Computes SNR, PSLR, ISLR for point targets, and extended metrics (ENL).
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_snr(image: np.ndarray, signal_mask: Optional[np.ndarray] = None) -> float:
    """Compute signal-to-noise ratio in dB.

    If signal_mask is None, use peak region vs border region.
    """
    intensity = np.abs(image) ** 2
    if signal_mask is not None:
        signal = intensity[signal_mask]
        noise = intensity[~signal_mask]
    else:
        # Use center 25% as signal, outer 25% as noise
        Ny, Nx = intensity.shape
        cy, cx = Ny // 2, Nx // 2
        qy, qx = Ny // 4, Nx // 4
        signal = intensity[cy-qy:cy+qy, cx-qx:cx+qx]
        noise_region = np.concatenate([
            intensity[:qy, :].ravel(),
            intensity[-qy:, :].ravel(),
        ])
        noise = noise_region if len(noise_region) > 0 else np.array([1e-30])

    sig_power = np.mean(signal) if len(signal) > 0 else 1e-30
    noise_power = np.mean(noise) if len(noise) > 0 else 1e-30
    noise_power = max(noise_power, 1e-30)

    snr_db = 10.0 * np.log10(sig_power / noise_power)
    return float(snr_db)


def find_peak(image: np.ndarray) -> Tuple[int, int]:
    """Find the peak pixel location in the intensity image."""
    intensity = np.abs(image) ** 2
    idx = np.argmax(intensity)
    iy, ix = np.unravel_index(idx, intensity.shape)
    return int(iy), int(ix)


def compute_pslr_islr(image: np.ndarray,
                      peak_yx: Optional[Tuple[int, int]] = None,
                      cut_radius: int = 64,
                      mainlobe_null_search: int = 20) -> Dict[str, float]:
    """Compute PSLR and ISLR from a point target response.

    Uses 1D range and azimuth cuts through the peak.

    Args:
        image: [Ny, Nx] complex SLC.
        peak_yx: (iy, ix) peak location. None = auto-detect.
        cut_radius: Half-width of the 1D cut.
        mainlobe_null_search: Search radius for first null.

    Returns:
        Dict with pslr_range_db, pslr_az_db, islr_range_db, islr_az_db.
    """
    if peak_yx is None:
        peak_yx = find_peak(image)
    iy, ix = peak_yx

    Ny, Nx = image.shape
    results = {}

    for axis_name, axis_idx, center_idx, size in [
        ('range', 1, ix, Nx),
        ('azimuth', 0, iy, Ny),
    ]:
        # Extract 1D cut
        if axis_name == 'range':
            start = max(center_idx - cut_radius, 0)
            end = min(center_idx + cut_radius + 1, Nx)
            cut = np.abs(image[iy, start:end]) ** 2
            peak_local = center_idx - start
        else:
            start = max(center_idx - cut_radius, 0)
            end = min(center_idx + cut_radius + 1, Ny)
            cut = np.abs(image[start:end, ix]) ** 2
            peak_local = center_idx - start

        if len(cut) < 3:
            results[f'pslr_{axis_name}_db'] = 0.0
            results[f'islr_{axis_name}_db'] = 0.0
            continue

        peak_val = cut[peak_local]
        if peak_val < 1e-30:
            results[f'pslr_{axis_name}_db'] = 0.0
            results[f'islr_{axis_name}_db'] = 0.0
            continue

        # Find mainlobe extent (first nulls on each side)
        null_left = peak_local
        for i in range(peak_local - 1, max(peak_local - mainlobe_null_search, 0) - 1, -1):
            if cut[i] > cut[i + 1]:
                null_left = i + 1
                break
            null_left = i

        null_right = peak_local
        for i in range(peak_local + 1, min(peak_local + mainlobe_null_search, len(cut) - 1)):
            if cut[i] > cut[i - 1]:
                null_right = i - 1
                break
            null_right = i

        # Mainlobe and sidelobe masks
        mainlobe_mask = np.zeros(len(cut), dtype=bool)
        mainlobe_mask[null_left:null_right + 1] = True
        sidelobe_mask = ~mainlobe_mask

        mainlobe_energy = np.sum(cut[mainlobe_mask])
        sidelobe_energy = np.sum(cut[sidelobe_mask])

        # PSLR: peak sidelobe / main peak
        if np.any(sidelobe_mask):
            peak_sidelobe = np.max(cut[sidelobe_mask])
            pslr_db = 10.0 * np.log10(max(peak_sidelobe, 1e-30) / peak_val)
        else:
            pslr_db = -50.0

        # ISLR: sidelobe energy / mainlobe energy
        if mainlobe_energy > 1e-30:
            islr_db = 10.0 * np.log10(max(sidelobe_energy, 1e-30) / mainlobe_energy)
        else:
            islr_db = 0.0

        results[f'pslr_{axis_name}_db'] = float(pslr_db)
        results[f'islr_{axis_name}_db'] = float(islr_db)

    # Overall PSLR/ISLR (worst case of range and azimuth)
    results['pslr_db'] = max(results.get('pslr_range_db', -50),
                             results.get('pslr_az_db', -50))
    results['islr_db'] = max(results.get('islr_range_db', -50),
                             results.get('islr_az_db', -50))

    return results


def compute_enl(intensity: np.ndarray,
                roi: Optional[np.ndarray] = None) -> float:
    """Compute Equivalent Number of Looks (ENL).

    ENL = (mean^2) / variance for homogeneous regions.
    """
    if roi is not None:
        data = intensity[roi]
    else:
        data = intensity.ravel()

    data = data[data > 0]
    if len(data) < 10:
        return 1.0

    mean_val = np.mean(data)
    var_val = np.var(data)
    if var_val < 1e-30:
        return float('inf')

    return float(mean_val ** 2 / var_val)


def compute_dynamic_range(image: np.ndarray) -> float:
    """Compute dynamic range in dB."""
    intensity = np.abs(image) ** 2
    intensity = intensity[intensity > 0]
    if len(intensity) < 2:
        return 0.0
    max_val = np.max(intensity)
    # Use 5th percentile as noise floor (avoid exact zeros)
    noise_floor = np.percentile(intensity, 5)
    if noise_floor < 1e-30:
        noise_floor = np.min(intensity[intensity > 0])
    if noise_floor < 1e-30:
        return 0.0
    return float(10.0 * np.log10(max_val / noise_floor))


def compute_all_metrics(image: np.ndarray,
                        scene_type: str = 'point_target') -> Dict[str, Any]:
    """Compute all quality metrics.

    Args:
        image: [Ny, Nx] complex SLC.
        scene_type: 'point_target' or 'urban'.

    Returns:
        Dict of metrics.
    """
    metrics = {}
    metrics['snr_db'] = compute_snr(image)
    metrics['dynamic_range_db'] = compute_dynamic_range(image)

    intensity = np.abs(image) ** 2
    metrics['peak_intensity'] = float(np.max(intensity))
    metrics['mean_intensity'] = float(np.mean(intensity))

    if scene_type == 'point_target':
        irf = compute_pslr_islr(image)
        metrics.update(irf)

    metrics['enl'] = compute_enl(intensity)

    logger.info(f"Metrics: SNR={metrics['snr_db']:.1f} dB, "
                f"DR={metrics['dynamic_range_db']:.1f} dB")
    if 'pslr_db' in metrics:
        logger.info(f"  PSLR={metrics['pslr_db']:.1f} dB, "
                    f"ISLR={metrics['islr_db']:.1f} dB")

    return metrics
