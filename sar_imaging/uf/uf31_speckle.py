"""UF-31: Speckle Noise Module.

Adds realistic multiplicative speckle noise to SAR images.
Implements SR-09.

Single-look speckle: intensity follows exponential distribution (ENL=1).
Multi-look equivalent: Gamma(L, 1/L) multiplicative factor.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_speckle(image: np.ndarray,
                enl: float = 1.0,
                seed: int = None) -> np.ndarray:
    """Add multiplicative speckle noise to a complex SAR image.

    For single-look (ENL=1):
        I_speckle = I_clean * exponential(1)
    In complex domain:
        amplitude *= sqrt(speckle_factor)
        phase += uniform(0, 2*pi)

    For multi-look (ENL=L):
        speckle_factor ~ Gamma(L, 1/L)  (mean=1, var=1/L)

    Args:
        image: [Ny, Nx] complex SLC image.
        enl: Equivalent number of looks. 1=single-look (default).
        seed: Random seed for reproducibility.

    Returns:
        speckled: [Ny, Nx] complex SLC with speckle noise added.
    """
    rng = np.random.RandomState(seed)

    Ny, Nx = image.shape

    if enl <= 0:
        logger.warning(f"Invalid ENL={enl}, returning original image.")
        return image.copy()

    # Generate speckle intensity factor (multiplicative)
    # Gamma(shape=enl, scale=1/enl) has mean=1, variance=1/enl
    speckle_intensity = rng.gamma(shape=enl, scale=1.0 / enl, size=(Ny, Nx))

    # Apply in complex domain:
    # amplitude *= sqrt(speckle_factor)
    # phase += random uniform phase
    amplitude_factor = np.sqrt(speckle_intensity).astype(np.float32)
    phase_noise = rng.uniform(0, 2 * np.pi, size=(Ny, Nx)).astype(np.float32)

    speckled = image * amplitude_factor * np.exp(1j * phase_noise)

    # Compute measured ENL for logging
    intensity = np.abs(speckled) ** 2
    nonzero = intensity > 0
    if np.any(nonzero):
        mean_i = np.mean(intensity[nonzero])
        var_i = np.var(intensity[nonzero])
        measured_enl = (mean_i ** 2) / var_i if var_i > 0 else float('inf')
    else:
        measured_enl = 0.0

    logger.info(f"Speckle added: target_ENL={enl:.1f}, measured_ENL={measured_enl:.2f}")

    return speckled.astype(np.complex64)
