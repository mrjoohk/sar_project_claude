"""UF-25: Beam/Power Normalization.

Corrects range-dependent amplitude falloff to prevent signal collapse.
"""
import numpy as np
from ..data_contracts import Scatterers
import logging

logger = logging.getLogger(__name__)


def apply_power_normalization(scatterers: Scatterers,
                              platform_pos: np.ndarray,
                              exponent: float = 2.0) -> Scatterers:
    """Apply range-dependent power normalization to scatterer RCS.

    gain = (R / R_ref)^exponent
    where R_ref = median range to scene center.

    Args:
        scatterers: Scatterer data with s_rcs.
        platform_pos: [3] representative platform position (ENU).
        exponent: Power law exponent (2 = two-way propagation).

    Returns:
        Modified scatterers with normalized s_rcs.
    """
    scat_pos = scatterers.positions.astype(np.float64)
    platform_pos = np.asarray(platform_pos, dtype=np.float64)

    ranges = np.linalg.norm(scat_pos - platform_pos[np.newaxis, :], axis=1)
    R_ref = np.median(ranges)

    if R_ref < 1e-3:
        logger.warning("R_ref near zero, skipping power normalization")
        return scatterers

    gain = (ranges / R_ref) ** exponent
    gain = np.maximum(gain, 1e-6).astype(np.float32)

    scatterers.s_rcs = scatterers.s_rcs * gain

    logger.info(f"Power normalization applied: R_ref={R_ref:.1f} m, "
                f"gain range=[{gain.min():.3f}, {gain.max():.3f}]")
    return scatterers
